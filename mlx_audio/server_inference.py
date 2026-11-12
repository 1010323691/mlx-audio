from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import mlx.core as mx


@dataclass
class InferenceResultChunk:
    kind: str
    payload: Any = None
    error: BaseException | None = None


@dataclass
class InferenceContext:
    request_id: str
    endpoint_kind: str
    model_name: str
    queued_at: float
    batch_key: Any = None


@dataclass
class InferenceRequest:
    endpoint_kind: str
    model_name: str
    payload: Any
    normalized_kwargs: dict[str, Any] = field(default_factory=dict)
    stream: bool = False
    batch_key: Any = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    queued_at: float = field(default_factory=time.time)
    result_queue: "queue.Queue[InferenceResultChunk]" = field(
        default_factory=queue.Queue
    )
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def emit_data(self, payload: Any) -> None:
        self.result_queue.put(InferenceResultChunk(kind="data", payload=payload))

    def emit_error(self, error: BaseException) -> None:
        self.result_queue.put(InferenceResultChunk(kind="error", error=error))

    def emit_done(self) -> None:
        self.result_queue.put(InferenceResultChunk(kind="done"))


@dataclass
class InferenceHandle:
    context: InferenceContext
    result_queue: "queue.Queue[InferenceResultChunk]"
    cancel_event: threading.Event

    def cancel(self) -> None:
        self.cancel_event.set()


class ModelExecutionAdapter(Protocol):
    max_batch_size: int
    # Fixed-window batching: after the first request of a batch, keep collecting
    # compatible requests while new ones arrive at most ``batch_gather_s`` apart
    # (bounded by ``batch_gather_max_s``) or until the batch is full, so a client
    # that sends a whole batch at once has it executed as one batch. 0 = off.
    batch_gather_s: float
    batch_gather_max_s: float

    def supports_batch(self, request: InferenceRequest) -> bool: ...

    def batch_key(self, request: InferenceRequest) -> Any: ...

    def is_batch_compatible(
        self, request: InferenceRequest, candidate: InferenceRequest
    ) -> bool: ...

    def run_serial(self, request: InferenceRequest) -> None: ...

    def run_batch(self, requests: list[InferenceRequest]) -> None: ...

    def supports_continuous_batch(self, request: InferenceRequest) -> bool: ...

    def continuous_batch_key(self, request: InferenceRequest) -> Any: ...

    def create_continuous_batch_session(
        self, request: InferenceRequest
    ) -> "ContinuousBatchSession": ...


class ContinuousBatchSession(Protocol):
    @property
    def idle(self) -> bool: ...

    def submit(self, request: InferenceRequest) -> None: ...

    def step(self) -> None: ...

    def fail(self, error: BaseException) -> None: ...


class BaseModelExecutionAdapter:
    max_batch_size = 1
    batch_gather_s = 0.0
    batch_gather_max_s = 0.0
    # Quick jobs that other requests wait on before they are submitted (a model-load
    # preflight): a batch gather window runs them instead of making them wait.
    run_while_gathering = False

    def supports_batch(self, request: InferenceRequest) -> bool:
        del request
        return False

    def batch_key(self, request: InferenceRequest) -> Any:
        del request
        return None

    def is_batch_compatible(
        self, request: InferenceRequest, candidate: InferenceRequest
    ) -> bool:
        return candidate.batch_key == request.batch_key

    def run_serial(self, request: InferenceRequest) -> None:
        raise NotImplementedError

    def run_batch(self, requests: list[InferenceRequest]) -> None:
        if len(requests) != 1:
            raise NotImplementedError
        self.run_serial(requests[0])

    def supports_continuous_batch(self, request: InferenceRequest) -> bool:
        del request
        return False

    def continuous_batch_key(self, request: InferenceRequest) -> Any:
        return self.batch_key(request)

    def create_continuous_batch_session(
        self, request: InferenceRequest
    ) -> ContinuousBatchSession:
        del request
        raise NotImplementedError


class InferenceBroker:
    def __init__(
        self,
        *,
        idle_poll_s: float = 0.1,
    ):
        self.idle_poll_s = idle_poll_s
        self._requests: "queue.Queue[Optional[InferenceRequest]]" = queue.Queue()
        self._adapters: dict[str, ModelExecutionAdapter] = {}
        self._continuous_sessions: dict[Any, ContinuousBatchSession] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def register_adapter(
        self, endpoint_kind: str, adapter: ModelExecutionAdapter
    ) -> None:
        self._adapters[endpoint_kind] = adapter

    def stop_and_join(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._requests.put(None)
        self._thread.join(timeout=timeout)
        for adapter in self._adapters.values():
            shutdown = getattr(adapter, "shutdown", None)
            if callable(shutdown):
                shutdown()

    def submit(
        self,
        *,
        endpoint_kind: str,
        model_name: str,
        payload: Any,
        normalized_kwargs: Optional[dict[str, Any]] = None,
        stream: bool = False,
        batch_key: Any = None,
    ) -> InferenceHandle:
        adapter = self._adapters.get(endpoint_kind)
        if adapter is None:
            raise ValueError(f"No inference adapter registered for {endpoint_kind!r}")

        request = InferenceRequest(
            endpoint_kind=endpoint_kind,
            model_name=model_name,
            payload=payload,
            normalized_kwargs=normalized_kwargs or {},
            stream=stream,
            batch_key=batch_key,
        )
        if request.batch_key is None:
            request.batch_key = adapter.batch_key(request)

        self._requests.put(request)
        return InferenceHandle(
            context=InferenceContext(
                request_id=request.request_id,
                endpoint_kind=request.endpoint_kind,
                model_name=request.model_name,
                queued_at=request.queued_at,
                batch_key=request.batch_key,
            ),
            result_queue=request.result_queue,
            cancel_event=request.cancel_event,
        )

    def _run(self) -> None:
        # MLX default streams are thread-local. The inference broker owns all
        # model execution on this worker, so create its CPU/GPU streams here
        # rather than inheriting streams from the ASGI or model-load threads.
        worker_streams = [mx.new_stream(mx.cpu), mx.new_stream(mx.gpu)]
        for stream in worker_streams:
            mx.set_default_stream(stream)

        pending: list[InferenceRequest] = []
        try:
            while not self._stop.is_set():
                has_continuous_work = bool(self._continuous_sessions)
                self._fill_pending(
                    pending, block=not pending and not has_continuous_work
                )
                pending = [
                    request for request in pending if not request.cancel_event.is_set()
                ]

                self._route_continuous_requests(pending)
                self._step_continuous_sessions()

                # Keep all GPU work serialized. Whole-request serial or fixed-window
                # batch calls run only after active continuous sessions drain.
                if self._continuous_sessions:
                    continue

                if not pending:
                    continue

                request = pending.pop(0)
                adapter = self._adapters.get(request.endpoint_kind)
                if adapter is None:
                    request.emit_error(
                        ValueError(
                            f"No inference adapter registered for "
                            f"{request.endpoint_kind!r}"
                        )
                    )
                    request.emit_done()
                    continue

                requests = [request]
                if adapter.supports_batch(request) and adapter.max_batch_size > 1:
                    requests.extend(self._gather_batch(request, adapter, pending))

                try:
                    if len(requests) > 1:
                        adapter.run_batch(requests)
                    else:
                        adapter.run_serial(request)
                except Exception as exc:  # pragma: no cover - defensive broker guard
                    traceback.print_exc()
                    for failed_request in requests:
                        failed_request.emit_error(exc)
                        failed_request.emit_done()
                finally:
                    mx.clear_cache()
        finally:
            for session in list(self._continuous_sessions.values()):
                session.fail(RuntimeError("Inference broker stopped."))
            self._continuous_sessions.clear()

    def _fill_pending(self, pending: list[InferenceRequest], *, block: bool) -> None:
        try:
            if block:
                item = self._requests.get(timeout=self.idle_poll_s)
            else:
                item = self._requests.get_nowait()
        except queue.Empty:
            return

        if item is None:
            self._stop.set()
            return
        pending.append(item)

        while True:
            try:
                item = self._requests.get_nowait()
            except queue.Empty:
                return
            if item is None:
                self._stop.set()
                return
            pending.append(item)

    def _select_batch_candidates(
        self,
        request: InferenceRequest,
        adapter: ModelExecutionAdapter,
        pending: list[InferenceRequest],
        compatible: Optional[list[InferenceRequest]] = None,
    ) -> list[InferenceRequest]:
        """Move compatible requests from ``pending`` into ``compatible`` (up to the
        batch size, counting ``request`` itself) and return it."""
        compatible = [] if compatible is None else compatible
        remaining: list[InferenceRequest] = []
        for candidate in pending:
            if len(compatible) >= adapter.max_batch_size - 1:
                remaining.append(candidate)
                continue
            if self._is_batch_compatible(request, candidate, adapter):
                compatible.append(candidate)
            else:
                remaining.append(candidate)
        pending[:] = remaining
        return compatible

    def _gather_batch(
        self,
        request: InferenceRequest,
        adapter: ModelExecutionAdapter,
        pending: list[InferenceRequest],
    ) -> list[InferenceRequest]:
        """Compatible requests to run with ``request``. With a gather window, wait
        for more while they keep arriving (each within ``batch_gather_s`` of the
        last, all within ``batch_gather_max_s``) until the batch is full: a client
        that sends a whole batch at once otherwise has its first arrivals start a
        small batch while the rest wait for it to finish."""
        compatible = self._select_batch_candidates(request, adapter, pending)
        gap = float(getattr(adapter, "batch_gather_s", 0.0) or 0.0)
        if gap <= 0:
            return compatible
        # A request that waits for a quick broker job first (the HTTP layer's model-load
        # preflight) only arrives once that job has run: run such jobs right here.
        pending[:] = self._run_inline_jobs(pending)
        limit = max(gap, float(getattr(adapter, "batch_gather_max_s", 0.0) or 0.0))
        deadline = time.monotonic() + limit
        while (
            len(compatible) < adapter.max_batch_size - 1
            and not self._stop.is_set()
            and not request.cancel_event.is_set()
        ):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            arrived: list[InferenceRequest] = []
            try:
                item = self._requests.get(timeout=min(gap, remaining))
            except queue.Empty:
                break
            while True:
                if item is None:
                    self._stop.set()
                    break
                arrived.append(item)
                try:
                    item = self._requests.get_nowait()
                except queue.Empty:
                    break
            self._select_batch_candidates(request, adapter, arrived, compatible)
            pending.extend(self._run_inline_jobs(arrived))
        return [c for c in compatible if not c.cancel_event.is_set()]

    def _run_inline_jobs(
        self, requests: list[InferenceRequest]
    ) -> list[InferenceRequest]:
        """Run the requests whose adapter sets ``run_while_gathering`` (quick jobs
        other requests wait on) and return the rest, in order."""
        rest: list[InferenceRequest] = []
        for job in requests:
            adapter = self._adapters.get(job.endpoint_kind)
            if adapter is None or not getattr(adapter, "run_while_gathering", False):
                rest.append(job)
                continue
            if job.cancel_event.is_set():
                continue
            try:
                adapter.run_serial(job)
            except Exception as exc:  # pragma: no cover - defensive broker guard
                traceback.print_exc()
                job.emit_error(exc)
                job.emit_done()
        return rest

    def _is_batch_compatible(
        self,
        request: InferenceRequest,
        candidate: InferenceRequest,
        adapter: ModelExecutionAdapter,
    ) -> bool:
        return (
            not candidate.cancel_event.is_set()
            and candidate.endpoint_kind == request.endpoint_kind
            and candidate.model_name == request.model_name
            and _adapter_batch_compatible(adapter, request, candidate)
            and adapter.supports_batch(candidate)
        )

    def _route_continuous_requests(self, pending: list[InferenceRequest]) -> None:
        if not pending:
            return

        remaining: list[InferenceRequest] = []
        for request in pending:
            adapter = self._adapters.get(request.endpoint_kind)
            if adapter is None:
                remaining.append(request)
                continue
            if not adapter.supports_continuous_batch(request):
                remaining.append(request)
                continue

            session_key = (
                request.endpoint_kind,
                request.model_name,
                adapter.continuous_batch_key(request),
            )
            session = self._continuous_sessions.get(session_key)
            try:
                if session is None or session.idle:
                    session = adapter.create_continuous_batch_session(request)
                    self._continuous_sessions[session_key] = session
                session.submit(request)
            except Exception as exc:
                traceback.print_exc()
                request.emit_error(exc)
                request.emit_done()

        pending[:] = remaining

    def _step_continuous_sessions(self) -> None:
        for session_key, session in list(self._continuous_sessions.items()):
            try:
                session.step()
            except Exception as exc:
                traceback.print_exc()
                session.fail(exc)
                self._continuous_sessions.pop(session_key, None)
                continue

            if session.idle:
                self._continuous_sessions.pop(session_key, None)
                mx.clear_cache()


def _adapter_batch_compatible(
    adapter: ModelExecutionAdapter,
    request: InferenceRequest,
    candidate: InferenceRequest,
) -> bool:
    check = getattr(adapter, "is_batch_compatible", None)
    if callable(check):
        return bool(check(request, candidate))
    return candidate.batch_key == request.batch_key
