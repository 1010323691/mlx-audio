import threading
import time

import mlx.core as mx

from mlx_audio.server_inference import (
    BaseModelExecutionAdapter,
    InferenceBroker,
    InferenceRequest,
)


def _collect(handle, timeout=2.0):
    deadline = time.time() + timeout
    chunks = []
    while time.time() < deadline:
        chunk = handle.result_queue.get(timeout=timeout)
        chunks.append(chunk)
        if chunk.kind == "done":
            return chunks
    raise TimeoutError("timed out waiting for broker results")


class SerializedAdapter(BaseModelExecutionAdapter):
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def run_serial(self, request: InferenceRequest) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

        time.sleep(0.05)
        request.emit_data(request.payload)

        with self.lock:
            self.active -= 1

        request.emit_done()


class ContinuousSession:
    def __init__(self):
        self.requests = []
        self.submitted_payloads = []
        self.failed = None

    @property
    def idle(self):
        return not self.requests

    def submit(self, request: InferenceRequest) -> None:
        self.submitted_payloads.append(request.payload)
        self.requests.append(request)

    def step(self) -> None:
        if not self.requests:
            return
        request = self.requests.pop(0)
        request.emit_data(("continuous", request.payload))
        request.emit_done()

    def fail(self, error: BaseException) -> None:
        self.failed = error
        for request in self.requests:
            request.emit_error(error)
            request.emit_done()
        self.requests.clear()


class ContinuousAdapter(BaseModelExecutionAdapter):
    max_batch_size = 4

    def __init__(self):
        self.sessions = []
        self.serial_payloads = []

    def supports_continuous_batch(self, request: InferenceRequest) -> bool:
        del request
        return True

    def continuous_batch_key(self, request: InferenceRequest):
        del request
        return "continuous"

    def create_continuous_batch_session(
        self,
        request: InferenceRequest,
    ) -> ContinuousSession:
        del request
        session = ContinuousSession()
        self.sessions.append(session)
        return session

    def run_serial(self, request: InferenceRequest) -> None:
        self.serial_payloads.append(request.payload)
        request.emit_data(("serial", request.payload))
        request.emit_done()


class MlxWorkerAdapter(BaseModelExecutionAdapter):
    def run_serial(self, request: InferenceRequest) -> None:
        value = int(mx.array([request.payload])[0].item())
        request.emit_data(value)
        request.emit_done()


def test_inference_broker_serializes_requests():
    broker = InferenceBroker()
    adapter = SerializedAdapter()
    broker.register_adapter("serial", adapter)

    try:
        first = broker.submit(
            endpoint_kind="serial",
            model_name="model-a",
            payload="first",
        )
        second = broker.submit(
            endpoint_kind="serial",
            model_name="model-a",
            payload="second",
        )

        assert _collect(first)[0].payload == "first"
        assert _collect(second)[0].payload == "second"
        assert adapter.max_active == 1
    finally:
        broker.stop_and_join()


def test_inference_broker_initializes_thread_local_mlx_streams():
    broker = InferenceBroker()
    broker.register_adapter("mlx", MlxWorkerAdapter())

    try:
        handle = broker.submit(
            endpoint_kind="mlx",
            model_name="model-a",
            payload=42,
        )

        chunks = _collect(handle)
        assert chunks[0].kind == "data"
        assert chunks[0].payload == 42
    finally:
        broker.stop_and_join()


def test_inference_broker_routes_continuous_batch_sessions():
    broker = InferenceBroker(idle_poll_s=0.01)
    adapter = ContinuousAdapter()
    broker.register_adapter("continuous", adapter)

    try:
        first = broker.submit(
            endpoint_kind="continuous",
            model_name="model-a",
            payload="first",
        )
        second = broker.submit(
            endpoint_kind="continuous",
            model_name="model-a",
            payload="second",
        )

        assert _collect(first)[0].payload == ("continuous", "first")
        assert _collect(second)[0].payload == ("continuous", "second")
        assert adapter.serial_payloads == []
        assert [
            payload
            for session in adapter.sessions
            for payload in session.submitted_payloads
        ] == [
            "first",
            "second",
        ]
    finally:
        broker.stop_and_join()


class BatchingAdapter(BaseModelExecutionAdapter):
    def __init__(self, max_batch_size=8, gather_s=0.0, gather_max_s=0.0):
        self.max_batch_size = max_batch_size
        self.batch_gather_s = gather_s
        self.batch_gather_max_s = gather_max_s
        self.batches = []

    def supports_batch(self, request: InferenceRequest) -> bool:
        del request
        return True

    def batch_key(self, request: InferenceRequest):
        return request.payload[0]

    def run_serial(self, request: InferenceRequest) -> None:
        self.run_batch([request])

    def run_batch(self, requests) -> None:
        self.batches.append([request.payload for request in requests])
        time.sleep(0.05)
        for request in requests:
            request.emit_data(request.payload)
            request.emit_done()


def _submit_staggered(broker, payloads, delay):
    handles = []
    for payload in payloads:
        handles.append(
            broker.submit(endpoint_kind="batch", model_name="m", payload=payload)
        )
        time.sleep(delay)
    return handles


def test_inference_broker_gathers_a_staggered_batch_into_one_call():
    broker = InferenceBroker(idle_poll_s=0.01)
    adapter = BatchingAdapter(max_batch_size=8, gather_s=0.2, gather_max_s=2.0)
    broker.register_adapter("batch", adapter)
    try:
        handles = _submit_staggered(broker, [("a", i) for i in range(6)], 0.03)
        for handle in handles:
            _collect(handle)
        assert adapter.batches == [[("a", i) for i in range(6)]]
    finally:
        broker.stop_and_join()


def test_inference_broker_without_gather_window_starts_on_first_arrival():
    broker = InferenceBroker(idle_poll_s=0.01)
    adapter = BatchingAdapter(max_batch_size=8)
    broker.register_adapter("batch", adapter)
    try:
        handles = _submit_staggered(broker, [("a", i) for i in range(4)], 0.03)
        for handle in handles:
            _collect(handle)
        assert len(adapter.batches) > 1
    finally:
        broker.stop_and_join()


def test_inference_broker_gather_stops_when_full_and_keeps_incompatible_requests():
    broker = InferenceBroker(idle_poll_s=0.01)
    # A long window: a full batch must start without waiting for it to run out.
    adapter = BatchingAdapter(max_batch_size=3, gather_s=5.0, gather_max_s=5.0)
    broker.register_adapter("batch", adapter)
    try:
        started = time.monotonic()
        handles = _submit_staggered(
            broker, [("a", 0), ("b", 0), ("a", 1), ("a", 2), ("a", 3)], 0.01
        )
        first = handles[0]
        _collect(first)
        assert time.monotonic() - started < 2.0
        assert adapter.batches[0] == [("a", 0), ("a", 1), ("a", 2)]
        for handle in handles[1:]:
            _collect(handle, timeout=12.0)
        assert sorted(sum(adapter.batches, [])) == sorted(
            [("a", 0), ("b", 0), ("a", 1), ("a", 2), ("a", 3)]
        )
    finally:
        broker.stop_and_join()


def test_inference_broker_gather_window_has_an_overall_limit():
    broker = InferenceBroker(idle_poll_s=0.01)
    adapter = BatchingAdapter(max_batch_size=100, gather_s=0.2, gather_max_s=0.3)
    broker.register_adapter("batch", adapter)
    try:
        handles = _submit_staggered(broker, [("a", i) for i in range(10)], 0.1)
        for handle in handles:
            _collect(handle)
        assert len(adapter.batches) >= 2
        assert sum(len(batch) for batch in adapter.batches) == 10
    finally:
        broker.stop_and_join()


class PreflightAdapter(BaseModelExecutionAdapter):
    run_while_gathering = True

    def run_serial(self, request: InferenceRequest) -> None:
        request.emit_done()


def test_inference_broker_runs_preflight_jobs_while_gathering():
    # Like the HTTP layer: each request first waits for a preflight broker job, then
    # submits the real request. The gather window must run those preflights, or only
    # the first request makes it into the batch.
    broker = InferenceBroker(idle_poll_s=0.01)
    adapter = BatchingAdapter(max_batch_size=8, gather_s=0.2, gather_max_s=2.0)
    broker.register_adapter("batch", adapter)
    broker.register_adapter("preflight", PreflightAdapter())
    handles = [None] * 5

    def client(i):
        _collect(broker.submit(endpoint_kind="preflight", model_name="m", payload=None))
        handles[i] = broker.submit(
            endpoint_kind="batch", model_name="m", payload=("a", i)
        )

    try:
        threads = []
        for i in range(5):
            thread = threading.Thread(target=client, args=(i,))
            thread.start()
            threads.append(thread)
            time.sleep(0.03)
        for thread in threads:
            thread.join(timeout=5)
        for handle in handles:
            _collect(handle)
        assert adapter.batches == [[("a", i) for i in range(5)]]
    finally:
        broker.stop_and_join()


def test_inference_broker_clears_cache_when_continuous_session_drains(monkeypatch):
    broker = InferenceBroker(idle_poll_s=0.01)
    adapter = ContinuousAdapter()
    broker.register_adapter("continuous", adapter)
    cleared = threading.Event()

    def clear_cache():
        if adapter.sessions and adapter.sessions[-1].idle:
            cleared.set()

    monkeypatch.setattr(mx, "clear_cache", clear_cache)

    try:
        handle = broker.submit(
            endpoint_kind="continuous",
            model_name="model-a",
            payload="first",
        )

        assert _collect(handle)[0].payload == ("continuous", "first")
        assert cleared.wait(timeout=2)
    finally:
        broker.stop_and_join()
