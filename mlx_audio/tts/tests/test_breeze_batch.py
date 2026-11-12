"""Batched generation tests for Breeze TTS 2.

The light tests run a tiny random-weight model on the CPU device.  On Metal,
fp32 matmuls for one row (GEMV) and several rows (GEMM) do not round alike,
so greedy batch/single parity is only exact where reductions are exact; the
CPU keeps the comparison deterministic.  The ``requires_weights`` test uses the
real ``mlx-community/Breeze-TTS-2-mlx`` checkpoint and is skipped when it is
not in the Hugging Face cache (or ``BREEZE_TTS_MLX_PATH`` is not set).
"""

import math
import os
from pathlib import Path

import pytest

try:
    import mlx.core as mx
except (ImportError, RuntimeError) as exc:  # pragma: no cover - no Metal host
    pytest.skip(f"MLX device unavailable: {exc}", allow_module_level=True)

from mlx_audio.tts.models.base import BatchGenerationResult
from mlx_audio.tts.models.breeze_tts.breeze_tts import Model
from mlx_audio.tts.models.breeze_tts.config import ModelConfig

REPO_ID = "mlx-community/Breeze-TTS-2-mlx"


def _tiny_config():
    return ModelConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_codebooks=4,
        vocab_size=8,
        text_vocab_size=32,
        text_encoder_config={
            "hidden_size": 12,
            "num_hidden_layers": 1,
            "intermediate_size": 24,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 6,
            "rms_norm_eps": 1e-6,
            "vocab_size": 32,
            "layer_types": ["full_attention"],
        },
        depth_decoder_config={
            "hidden_size": 12,
            "num_hidden_layers": 1,
            "intermediate_size": 24,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 6,
            "rms_norm_eps": 1e-5,
            "num_codebooks": 4,
            "vocab_size": 8,
            "audio_embed_size": 16,
        },
    )


class _Tokenizer:
    def __call__(self, text, add_special_tokens=True):
        del add_special_tokens
        return {"input_ids": [1] + [(ord(c) * 7) % 30 + 2 for c in text]}


class _AudioTokenizer:
    """Records decoded code sequences; the reference "codec" is deterministic."""

    decode_upsample_rate = 4

    def __init__(self):
        self.decoded = []
        self.encode_calls = 0

    def encode(self, audio):
        self.encode_calls += 1
        frames = max(1, audio.shape[-1] // 4)
        return (mx.arange(frames * 4).reshape(1, 4, frames) % 6).astype(mx.int32)

    def decode(self, codes):
        self.decoded.append(codes[0].tolist())
        frames = codes.shape[1]
        return mx.zeros((1, frames * 4)), mx.array([frames * 4])


@pytest.fixture
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _model(seed: int) -> Model:
    mx.random.seed(seed)
    model = Model(_tiny_config())
    head = model.depth_decoder.codebooks_head
    head.weight = mx.random.normal(head.weight.shape)
    embed = model.text_encoder.embed_tokens
    embed.weight = mx.random.normal(embed.weight.shape)
    # Scale the EOS column so rows end at different steps (exercises the
    # per-row stop and batch compaction) without ending immediately.
    weight = mx.random.normal(model.lm_head.weight.shape)
    weight[-1] = weight[-1] * 0.6
    model.lm_head.weight = weight
    model.tokenizer = _Tokenizer()
    model.audio_tokenizer = _AudioTokenizer()
    mx.eval(model.parameters())
    return model


# (text, instruct, has reference audio)
ROWS = [
    ("abc", None, False),
    ("hello there friend", "happy", False),
    ("x", None, True),
    ("longer text here ok", "sad", True),
    ("mid size", "calm", False),
]


def _single_codes(model, rows, ref, **kwargs):
    codes = []
    for text, instruct, has_ref in rows:
        model.audio_tokenizer.decoded = []
        list(
            model.generate(
                text,
                instruct=instruct,
                ref_audio=ref if has_ref else None,
                ref_text="ref" if has_ref else None,
                split_pattern=None,
                **kwargs,
            )
        )
        codes.append(model.audio_tokenizer.decoded[-1])
    return codes


def _batch(model, rows, ref, **kwargs):
    model.audio_tokenizer.decoded = []
    results = list(
        model.batch_generate(
            [row[0] for row in rows],
            instructs=[row[1] for row in rows],
            ref_audios=[ref if row[2] else None for row in rows],
            ref_texts=["ref" if row[2] else None for row in rows],
            **kwargs,
        )
    )
    return results, list(model.audio_tokenizer.decoded)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("repetition_penalty", [1.0, 1.3])
def test_greedy_batch_matches_single_rows(cpu, seed, repetition_penalty):
    model = _model(seed)
    ref = mx.ones((20,))
    kwargs = dict(
        cfg_scale=3.0,
        temperature=0,
        max_tokens=30,
        repetition_penalty=repetition_penalty,
    )
    expected = _single_codes(model, ROWS, ref, **kwargs)
    results, decoded = _batch(model, ROWS, ref, **kwargs)

    assert [r.sequence_idx for r in results] == list(range(len(ROWS)))
    assert all(isinstance(r, BatchGenerationResult) for r in results)
    assert decoded == expected
    for result, codes in zip(results, expected):
        assert result.sample_rate == model.sample_rate
        assert result.samples == result.audio.shape[0]
        assert result.token_count in (0, len(codes))


@pytest.mark.parametrize("seed", range(3))
def test_greedy_batch_with_a_different_reference_clip_per_row(cpu, seed):
    # The server batches requests whose reference clips differ (one speaker per
    # row, as in audiobook synthesis): each row must still see only its own clip.
    model = _model(seed)
    rows = [
        ("abc", "calm", mx.ones((20,)), "first"),
        ("hello there friend", "happy", mx.ones((36,)), "second speaker"),
        ("x", None, None, None),
        ("longer text here ok", "sad", mx.ones((12,)), "third"),
    ]
    kwargs = dict(cfg_scale=3.0, temperature=0, max_tokens=30)
    expected = []
    for text, instruct, ref_audio, ref_text in rows:
        model.audio_tokenizer.decoded = []
        list(
            model.generate(
                text,
                instruct=instruct,
                ref_audio=ref_audio,
                ref_text=ref_text,
                split_pattern=None,
                **kwargs,
            )
        )
        expected.append(model.audio_tokenizer.decoded[-1])
    model.audio_tokenizer.decoded = []
    list(
        model.batch_generate(
            [row[0] for row in rows],
            instructs=[row[1] for row in rows],
            ref_audios=[row[2] for row in rows],
            ref_texts=[row[3] for row in rows],
            **kwargs,
        )
    )
    assert model.audio_tokenizer.decoded == expected


def test_rows_stop_independently_and_cfg_rows_compact(cpu):
    # Seed 0 has rows ending at 1, 20+ and 30 frames, including a CFG row that
    # finishes while other rows (and other CFG twins) keep decoding.
    model = _model(0)
    ref = mx.ones((20,))
    kwargs = dict(cfg_scale=3.0, temperature=0, max_tokens=30)
    single_counts = []
    for text, instruct, has_ref in ROWS:
        (result,) = model.generate(
            text,
            instruct=instruct,
            ref_audio=ref if has_ref else None,
            ref_text="ref" if has_ref else None,
            split_pattern=None,
            **kwargs,
        )
        single_counts.append(result.token_count)
    results, _ = _batch(model, ROWS, ref, **kwargs)
    counts = [r.token_count for r in results]
    assert counts == single_counts
    assert len(set(counts)) > 2
    assert max(counts) == 30


def test_batch_caches_grow_in_small_steps(cpu):
    from mlx_audio.tts.models.breeze_tts.breeze_tts import BACKBONE_CACHE_STEP

    model = _model(0)
    ref = mx.ones((400,)) * 0.3
    hidden, cache, _ = model._batch_prefill(
        ["x", "a longer target text"],
        [None, None],
        [None, None],
        [ref, ref],
        ["a long reference transcript here", "a long reference transcript here"],
        [False, False],
    )
    assert BACKBONE_CACHE_STEP == 64
    assert all(c.step == BACKBONE_CACHE_STEP for c in cache)
    assert cache[0].keys.shape[2] % BACKBONE_CACHE_STEP == 0
    assert cache[0].keys.shape[2] - cache[0]._idx < BACKBONE_CACHE_STEP


def test_compaction_waits_until_it_pays_off():
    compact = Model._should_compact
    assert not compact(96, 95)  # one dead row: not worth a cache copy
    assert not compact(96, 73)  # 23 dead (< 1/4), still 65-128 rows
    assert compact(96, 72)  # a quarter dead
    assert compact(96, 64)  # drops into the 9-64 band
    assert compact(200, 128) and not compact(200, 151)
    assert not compact(8, 7) and compact(8, 6)


def test_finished_rows_are_kept_until_compaction(cpu, monkeypatch):
    """Rows that finish early stay in the batch (outputs ignored) until the
    policy says to compact, and the results are the same either way."""
    model = _model(0)
    ref = mx.ones((20,))
    kwargs = dict(cfg_scale=3.0, temperature=0, max_tokens=30)
    outputs = {}
    for name, policy in (("always", True), ("never", False)):
        monkeypatch.setattr(
            Model, "_should_compact", staticmethod(lambda rows, live: policy)
        )
        results, codes = _batch(model, ROWS, ref, **kwargs)
        outputs[name] = ([r.token_count for r in results], codes)
    assert outputs["never"] == outputs["always"]
    assert len(set(outputs["never"][0])) > 2


@pytest.mark.parametrize("sync_steps", [1, 3, 64])
def test_host_sync_window_does_not_change_results(cpu, sync_steps):
    model = _model(0)
    ref = mx.ones((20,))
    kwargs = dict(cfg_scale=3.0, temperature=0, max_tokens=30)
    expected = _single_codes(model, ROWS, ref, **kwargs)
    results, decoded = _batch(model, ROWS, ref, sync_steps=sync_steps, **kwargs)
    assert decoded == expected


def test_per_row_max_tokens(cpu):
    model = _model(8)
    caps = [3, 11, 7]
    texts = ["a", "bb", "ccc"]
    expected_counts, expected_codes = [], []
    for text, cap in zip(texts, caps):
        model.audio_tokenizer.decoded = []
        (result,) = model.generate(
            text, temperature=0, max_tokens=cap, split_pattern=None
        )
        expected_counts.append(result.token_count)
        expected_codes.append(model.audio_tokenizer.decoded[-1])
    # At least two rows are cut by their own (different) cap.
    assert sum(c == cap for c, cap in zip(expected_counts, caps)) >= 2
    model.audio_tokenizer.decoded = []
    results = list(
        model.batch_generate(texts, temperature=0, max_tokens=caps, sync_steps=4)
    )
    assert [r.token_count for r in results] == expected_counts
    assert model.audio_tokenizer.decoded == expected_codes


def test_cfg_scale_one_or_missing_instruct_skips_unconditional_rows(cpu, monkeypatch):
    model = _model(1)
    widths = []
    original = model.backbone_model.make_batch_cache

    def record(left_padding):
        widths.append(len(left_padding))
        return original(left_padding)

    monkeypatch.setattr(model.backbone_model, "make_batch_cache", record)
    for scale in (1.0, None):
        list(
            model.batch_generate(
                ["a", "b"], instructs=["x", None], cfg_scale=scale, max_tokens=2
            )
        )
    list(
        model.batch_generate(
            ["a", "b"], instructs=["x", None], cfg_scale=2.0, max_tokens=2
        )
    )
    assert widths == [2, 2, 3]


def test_shared_reference_is_encoded_once(cpu):
    model = _model(2)
    ref = mx.ones((20,))
    list(
        model.batch_generate(
            ["a", "bb", "ccc"],
            instructs=[None, "x", "y"],
            ref_audios=ref,
            ref_texts="ref",
            cfg_scale=2.0,
            max_tokens=3,
        )
    )
    assert model.audio_tokenizer.encode_calls == 1


def test_reference_clip_files_are_encoded_once_across_calls(cpu, tmp_path):
    import numpy as np
    import soundfile as sf

    model = _model(2)
    clip = tmp_path / "voice.wav"
    sf.write(clip, np.full(2400, 0.1, dtype=np.float32), model.sample_rate)
    kwargs = dict(ref_audios=str(clip), ref_texts="ref", max_tokens=2)
    list(model.batch_generate(["a", "bb"], **kwargs))
    list(model.batch_generate(["ccc"], **kwargs))
    list(model.generate("dd", ref_audio=str(clip), ref_text="ref", max_tokens=2))
    assert model.audio_tokenizer.encode_calls == 1

    # A replaced file is encoded again.
    sf.write(clip, np.full(3200, 0.2, dtype=np.float32), model.sample_rate)
    list(model.batch_generate(["a"], **kwargs))
    assert model.audio_tokenizer.encode_calls == 2


def test_sampled_batch_respects_max_tokens_and_seed(cpu):
    model = _model(3)
    kwargs = dict(temperature=0.9, top_k=5, top_p=0.9, max_tokens=7, seed=11)
    texts = ["one", "two words", "three words here"]
    first = [r.token_count for r in model.batch_generate(texts, **kwargs)]
    decoded = list(model.audio_tokenizer.decoded)
    model.audio_tokenizer.decoded = []
    second = [r.token_count for r in model.batch_generate(texts, **kwargs)]
    assert first == second
    assert model.audio_tokenizer.decoded == decoded
    assert all(0 <= count <= 7 for count in first)


def test_batch_argument_validation(cpu):
    model = _model(0)
    with pytest.raises(ValueError, match="streaming"):
        list(model.batch_generate(["a"], stream=True))
    with pytest.raises(ValueError, match="instructs"):
        list(model.batch_generate(["a", "b", "c"], instructs=["x", "y"]))
    with pytest.raises(ValueError, match="ref_text"):
        list(model.batch_generate(["a"], ref_audios=[mx.ones((8,))]))
    assert list(model.batch_generate([])) == []


def test_supports_tts_batch():
    model = Model(_tiny_config())
    assert model.supports_tts_batch()
    assert model.supports_tts_batch(
        voice="S1", instruct="calm", ref_audio="a.wav", ref_text="hi", cfg_scale=4
    )
    assert model.supports_tts_batch(speed=None, pitch=None)
    assert not model.supports_tts_batch(stream=True)
    assert not model.supports_tts_batch(speed=1.2)
    assert not model.supports_tts_batch(pitch=0.9)


def test_server_routes_breeze_requests_to_batch_generate():
    """The HTTP server must pick ``batch_generate`` (not serial ``generate``) for
    Breeze clone / directed / design requests, with cfg_scale passed through."""
    from mlx_audio.server import SpeechRequest, SpeechTaskPayload, TTSExecutionAdapter
    from mlx_audio.server_inference import InferenceRequest

    model = Model(_tiny_config())
    adapter = TTSExecutionAdapter()

    def request(**fields):
        item = InferenceRequest(
            endpoint_kind="tts",
            model_name="breeze",
            payload=SpeechTaskPayload(
                request=SpeechRequest(model="breeze", input="hello", **fields)
            ),
        )
        setattr(item, TTSExecutionAdapter._REQUEST_MODEL_ATTR, model)
        return item

    for fields in (
        {"ref_audio": "ref.wav", "ref_text": "ref", "cfg_scale": 4.0},
        {
            "ref_audio": "ref.wav",
            "ref_text": "ref",
            "instruct": "calm",
            "cfg_scale": 4.0,
        },
        {"instruct": "a warm voice", "cfg_scale": 4.0},
    ):
        assert adapter.supports_batch(request(**fields)), fields
        assert not adapter.supports_continuous_batch(request(**fields))
    assert not adapter.supports_batch(request(stream=True))
    # Directed and plain rows with the same cfg_scale share one batch.
    plain = request(ref_audio="ref.wav", ref_text="ref", cfg_scale=4.0)
    directed = request(
        ref_audio="ref.wav", ref_text="ref", instruct="calm", cfg_scale=4.0
    )
    assert adapter.batch_key(plain) == adapter.batch_key(directed)


def _full_prompt_prefill(model, rows, cfg_scale):
    """Reference: the whole prompt of every row in one left-padded prefill."""
    prompts = []
    for text, instruct, ref_audio, ref_text in rows:
        prompts.append(
            model._prompt_embeddings(
                text,
                voice=None,
                instruct=instruct,
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
        )
    for text, instruct, ref_audio, ref_text in rows:
        if instruct and cfg_scale not in (None, 1.0):
            prompts.append(
                model._prompt_embeddings(
                    text,
                    voice=None,
                    instruct=None,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )
            )
    width = max(int(p.shape[1]) for p in prompts)
    pad = [width - int(p.shape[1]) for p in prompts]
    embeddings = mx.concatenate(
        [mx.pad(p, [(0, 0), (n, 0), (0, 0)]) for p, n in zip(prompts, pad)]
    )
    cache = model.backbone_model.make_batch_cache(pad)
    hidden = model.backbone_model(
        input_embeddings=embeddings, cache=cache, mask=model._prefill_mask(pad, width)
    )[:, -1, :]
    return hidden, cache


def test_shared_prefix_prefill_matches_full_prompt_prefill(cpu, monkeypatch):
    """Rows reuse their voice's prefix keys/values; nothing else changes.

    Two voices with different clip and transcript lengths, rows without a
    reference, targets shorter and longer than the block, and CFG rows
    (which share the prefix of their conditional row).
    """
    model = _model(5)
    long_clip, short_clip = mx.ones((400,)) * 0.3, mx.ones((240,)) * -0.2
    rows = [
        ("x", None, long_clip, "a long reference transcript here"),
        ("short one", "warm", long_clip, "a long reference transcript here"),
        ("a much longer target sentence than the rest", None, short_clip, "ref b"),
        ("tiny", None, None, None),
        ("another target of medium size", "sad", short_clip, "ref b"),
        ("no reference but quite a long target text", None, None, None),
    ]
    cfg_scale = 2.0
    instructs = [row[1] for row in rows]
    use_cfg = [bool(ins) for ins in instructs]
    widths = []
    backbone_call = type(model.backbone_model).__call__

    def record(self, *args, input_embeddings=None, **kwargs):
        if input_embeddings is not None:
            widths.append(tuple(input_embeddings.shape[:2]))
        return backbone_call(self, *args, input_embeddings=input_embeddings, **kwargs)

    monkeypatch.setattr(type(model.backbone_model), "__call__", record)
    hidden, cache, owner = model._batch_prefill(
        [row[0] for row in rows],
        [None] * len(rows),
        instructs,
        [row[2] for row in rows],
        [row[3] for row in rows],
        use_cfg,
    )
    assert owner == [1, 4]
    # One prefill per distinct prefix (2 voices), then the batched block of
    # all 8 cache rows, which is much narrower than the whole prompts.
    assert len(widths) == 2 and widths[0][0] == 2 and widths[1][0] == 8
    expected_hidden, expected_cache = _full_prompt_prefill(model, rows, cfg_scale)
    assert widths[1][1] < widths[2][1] / 2
    assert cache[0].offset.tolist() == expected_cache[0].offset.tolist()
    assert mx.allclose(hidden, expected_hidden, atol=1e-5).item()

    # The caches must also agree for everything that comes after the prompt.
    frame = mx.array(
        [[[(r + c) % 8 for c in range(model.num_codebooks)]] for r in range(8)],
        dtype=mx.int32,
    )
    for _ in range(3):
        step = model.backbone_model(input_ids=frame, cache=cache)[:, -1, :]
        expected = model.backbone_model(input_ids=frame, cache=expected_cache)[:, -1, :]
        assert mx.allclose(step, expected, atol=1e-5).item()


def test_prefill_mask_hides_left_padding_without_empty_rows():
    mask = Model._prefill_mask([0, 2], 4)
    assert mask.shape == (2, 1, 4, 4)
    rows = mask[:, 0].tolist()
    assert rows[0] == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]
    assert rows[1] == [
        [True, False, False, False],
        [False, True, False, False],
        [False, False, True, False],
        [False, False, True, True],
    ]


# ----------------------------------------------------------------------------
# Real checkpoint
# ----------------------------------------------------------------------------


def _real_model_path():
    path = os.environ.get("BREEZE_TTS_MLX_PATH")
    if path:
        return Path(path) if Path(path).is_dir() else None
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover
        return None
    index = try_to_load_from_cache(REPO_ID, "model.safetensors.index.json")
    if not isinstance(index, str):
        return None
    snapshot = Path(index).parent
    if not (snapshot / "audio_tokenizer").is_dir():
        return None
    if not list(snapshot.glob("model-*.safetensors")):
        return None
    return snapshot


@pytest.fixture(scope="module")
def real_model():
    path = _real_model_path()
    if path is None:
        pytest.skip(f"{REPO_ID} weights are not available locally")
    from mlx_audio.tts.utils import load_model

    model = load_model(path)
    yield model
    del model
    mx.clear_cache()


def _record_codes(model, monkeypatch):
    captured = []
    original = model._decode_codes

    def record(codes):
        captured.append(codes[0].tolist())
        return original(codes)

    monkeypatch.setattr(model, "_decode_codes", record)
    return captured


def teacher_forced_agreement(model, rows, expected, *, cfg_scale, max_tokens):
    """Greedy agreement of the batched path with single-row generations.

    ``rows`` are ``(text, instruct, ref_audio, ref_text)`` and ``expected``
    the frames each row produced alone.  The batched prefill/decode is fed
    every row's own frames (teacher forcing), so one rounding flip cannot
    derail the rest of the sequence; every first-codebook choice (including
    EOS) and every depth-decoder choice is compared with the single run.
    """
    count = len(rows)
    instructs = [row[1] for row in rows]
    use_cfg = [bool(ins) and cfg_scale not in (None, 1.0) for ins in instructs]
    hidden, cache, owner = model._batch_prefill(
        [row[0] for row in rows],
        [None] * count,
        instructs,
        [row[2] for row in rows],
        [row[3] for row in rows],
        use_cfg,
    )
    uncond = list(range(count))
    for j, row in enumerate(owner):
        uncond[row] = count + j
    uncond_index = mx.array(uncond) if owner else None
    expand_index = mx.array(list(range(count)) + owner) if owner else None
    scale = mx.array([[cfg_scale if use_cfg[r] else 1.0] for r in range(count)])
    depth = model.depth_decoder
    eos = model.vocab_size
    stats = {"first": [0, 0], "depth": [0, 0]}
    for step in range(max(len(f) for f in expected) + 1):
        logits = model._combine_cfg(model.lm_head(hidden), count, uncond_index, scale)
        chosen = model._sample_array(
            model._mask_reserved_codec_logits(logits),
            temperature=0,
            top_p=1.0,
            top_k=0,
            allow_eos=True,
        ).tolist()
        frames = []
        for row in range(count):
            want = expected[row]
            if step < len(want):
                target = want[step][0]
            elif step == len(want) and len(want) < max_tokens:
                target = eos
            else:
                target = None
            if target is not None:
                stats["first"][0] += int(chosen[row] == target)
                stats["first"][1] += 1
            frames.append(want[step] if step < len(want) else [0] * model.num_codebooks)
        frames = mx.array(frames, dtype=mx.int32)
        depth_cache = depth.model.make_cache()
        depth.start_frame(hidden, depth_cache)
        for head in range(model.num_codebooks - 1):
            tokens = frames[:, head]
            if expand_index is not None:
                tokens = tokens[expand_index]
            logits = depth.step_logits(depth_cache, head_idx=head, token_id=tokens)
            logits = model._combine_cfg(logits, count, uncond_index, scale)
            chosen = model._sample_array(
                model._mask_reserved_codec_logits(logits),
                temperature=0,
                top_p=1.0,
                top_k=0,
            ).tolist()
            for row in range(count):
                if step < len(expected[row]):
                    stats["depth"][0] += int(
                        chosen[row] == expected[row][step][head + 1]
                    )
                    stats["depth"][1] += 1
        inputs = frames if expand_index is None else frames[expand_index]
        hidden = model.backbone_model(input_ids=inputs[:, None, :], cache=cache)[
            :, -1, :
        ]
    return {key: hits / max(1, total) for key, (hits, total) in stats.items()}


def single_greedy_frames(model, rows, *, cfg_scale, max_tokens):
    captured = []
    original = model._decode_codes

    def record(codes):
        captured.append(codes[0].tolist())
        return original(codes)

    model._decode_codes = record
    try:
        frames = []
        for text, instruct, ref_audio, ref_text in rows:
            captured.clear()
            list(
                model.generate(
                    text,
                    instruct=instruct,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    cfg_scale=cfg_scale,
                    temperature=0,
                    max_tokens=max_tokens,
                    split_pattern=None,
                )
            )
            # An immediate EOS decodes a dummy frame; record no frames then.
            frames.append(captured[-1] if captured else [])
        return frames
    finally:
        del model._decode_codes


@pytest.mark.requires_weights
def test_real_greedy_batch_matches_single_generation(real_model, monkeypatch):
    """Batch rows reproduce single-row greedy generation on the real model.

    A one-row batch without CFG is bit-identical to ``generate``.  Larger
    batches (and CFG, which always batches two rows) round differently in
    bf16 GEMM vs GEMV kernels (about one bf16 ulp in the hidden state), so a
    free-running greedy sequence eventually flips a low-margin depth token
    and diverges.  With teacher forcing the batched path must make the same
    choice as the single run almost everywhere.
    """
    model = real_model
    sr = model.sample_rate
    t = mx.arange(sr * 2) / sr
    ref = (0.1 * mx.sin(2 * math.pi * 220 * t)).astype(mx.float32)
    rows = [
        ("你好。", None, None, None),
        ("今天的天气很好，我们一起去公园散步吧。", "语气温柔", None, None),
        ("我回来了。", None, ref, "嗯。"),
        ("船离岸的时候，他回头望了一眼。", "语气平静克制", ref, "嗯。"),
    ]
    max_tokens = 40
    expected = single_greedy_frames(model, rows, cfg_scale=4.0, max_tokens=max_tokens)

    captured = _record_codes(model, monkeypatch)
    single = list(
        model.batch_generate(
            [rows[0][0]], temperature=0, max_tokens=max_tokens, cfg_scale=4.0
        )
    )
    assert captured[-1] == expected[0]
    assert single[0].token_count == len(expected[0])

    agreement = teacher_forced_agreement(
        model, rows, expected, cfg_scale=4.0, max_tokens=max_tokens
    )
    assert agreement["first"] >= 0.97, agreement
    assert agreement["depth"] >= 0.9, agreement

    captured.clear()
    results = list(
        model.batch_generate(
            [row[0] for row in rows],
            instructs=[row[1] for row in rows],
            ref_audios=[row[2] for row in rows],
            ref_texts=[row[3] for row in rows],
            cfg_scale=4.0,
            temperature=0,
            max_tokens=max_tokens,
        )
    )
    assert [r.sequence_idx for r in results] == [0, 1, 2, 3]
    for result in results:
        assert result.sample_rate == sr
        assert result.samples > 0
