"""MLX inference implementation for Breeze TTS 2.

Breeze uses an autoregressive Qwen3-style backbone for codebook zero and a
small depth decoder for the remaining Qwen3-TTS codec codebooks.  The text
encoder and codec bundled by Breeze are loaded from the original checkpoint;
no PyTorch or upstream Breeze runtime is required at inference time.
"""

import json
import math
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_audio.lm.models import llama, qwen3
from mlx_audio.lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_audio.lm.models.cache import BatchKVCache, KVCache
from mlx_audio.lm.models.rope_utils import initialize_rope
from mlx_audio.lm.sample_utils import make_sampler
from mlx_audio.tts.models.base import BatchGenerationResult, GenerationResult
from mlx_audio.utils import load_audio

from .config import ModelConfig

CODEC_DTYPE_ENV = "MLX_AUDIO_BREEZE_CODEC_DTYPE"
# Growth step (positions) of the batched backbone KV cache.  BatchKVCache's
# default of 256 rounds every row up to 256 positions and copies the whole
# cache when it grows; a short-sentence batch (prompt ~250 + ~40 frames) then
# held 512 positions per row.  64 keeps the waste small and changes no output.
BACKBONE_CACHE_STEP = 64
_CODEC_DTYPES = {"float16": mx.float16, "float32": mx.float32}


def _split_long_segment(text: str, max_chars: int = 600) -> list[str]:
    """Split text into non-empty chunks no longer than ``max_chars``."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")
    if len(text) <= max_chars:
        return [text]

    # Keep sentence-ending punctuation attached to the preceding sentence.
    # Full-width terminators cover Chinese and Japanese text, where sentence
    # boundaries are commonly not followed by whitespace.
    sentences = [s for s in re.split(r"(?<=[.!?。！？])", text) if s]
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        while sentence:
            if current and len(current) + len(sentence) > max_chars:
                chunks.append(current.strip())
                current = ""
                sentence = sentence.lstrip()
                continue

            available = max_chars - len(current)
            if len(sentence) <= available:
                current += sentence
                break

            # A sentence can itself exceed the limit. Prefer a nearby word
            # boundary, then fall back to a hard split for unspaced/CJK text.
            split_at = available
            whitespace = list(re.finditer(r"\s+", sentence[: available + 1]))
            if whitespace and whitespace[-1].start() >= available // 2:
                split_at = whitespace[-1].start()
            chunk = (current + sentence[:split_at]).strip()
            if chunk:
                chunks.append(chunk)
            current = ""
            sentence = sentence[split_at:].lstrip()

    if current.strip():
        chunks.append(current.strip())
    return chunks


class _AudioEmbedding(nn.Module):
    """Sum wrapper-level audio codebook embeddings.

    The nested Qwen config describes the text backbone and therefore has a
    different vocabulary from Breeze's audio wrapper.  Keep the wrapper
    vocabulary/codebook count explicit here.  ``audio_embeds_projector`` is
    present only for checkpoints whose embedding width differs from the
    backbone width, matching ``BreezeBackboneModelEmbeddings`` upstream.
    """

    def __init__(
        self,
        num_codebooks: int,
        vocab_size: int,
        hidden_size: int,
        audio_embed_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_codebooks = int(num_codebooks)
        self.vocab_size = int(vocab_size)
        self.audio_embed_size = int(audio_embed_size or hidden_size)
        self.embed_audio_tokens = nn.Embedding(
            self.num_codebooks * self.vocab_size, self.audio_embed_size
        )
        if self.audio_embed_size != hidden_size:
            self.audio_embeds_projector = nn.Linear(
                self.audio_embed_size, hidden_size, bias=False
            )
        else:
            self.audio_embeds_projector = None

    def __call__(self, codebooks: mx.array) -> mx.array:
        if codebooks.ndim != 3 or codebooks.shape[-1] != self.num_codebooks:
            raise ValueError(
                "Breeze audio codebooks must have shape "
                f"[batch, time, {self.num_codebooks}], got {codebooks.shape}."
            )
        offsets = (
            mx.arange(self.num_codebooks, dtype=codebooks.dtype)[None, None, :]
            * self.vocab_size
        )
        hidden = self.embed_audio_tokens(codebooks + offsets)
        if self.audio_embeds_projector is not None:
            hidden = self.audio_embeds_projector(hidden)
        return mx.sum(hidden, axis=-2)


class _Backbone(nn.Module):
    """Qwen3 transformer whose names match ``backbone_model.*`` weights."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        wrapper_num_codebooks = config.wrapper_num_codebooks
        wrapper_vocab_size = config.wrapper_vocab_size
        hidden_size = config.backbone_hidden_size
        args = qwen3.ModelArgs(
            model_type="qwen3",
            hidden_size=hidden_size,
            num_hidden_layers=config.backbone_value(
                "num_hidden_layers", config.num_hidden_layers
            ),
            intermediate_size=config.backbone_value(
                "intermediate_size", config.intermediate_size
            ),
            num_attention_heads=config.backbone_value(
                "num_attention_heads", config.num_attention_heads
            ),
            num_key_value_heads=config.backbone_value(
                "num_key_value_heads", config.num_key_value_heads
            ),
            head_dim=config.backbone_value("head_dim", config.head_dim),
            rms_norm_eps=config.backbone_value("rms_norm_eps", config.rms_norm_eps),
            # This is metadata for the Qwen args.  The actual embedding below
            # deliberately uses the wrapper audio vocabulary.
            vocab_size=config.backbone_value("vocab_size", wrapper_vocab_size),
            max_position_embeddings=config.backbone_value(
                "max_position_embeddings", config.max_position_embeddings
            ),
            rope_theta=config.backbone_value("rope_theta", config.rope_theta),
            rope_scaling=config.backbone_value("rope_scaling", config.rope_scaling),
            tie_word_embeddings=config.backbone_value("tie_word_embeddings", False),
        )
        self.args = args
        self.embed_tokens = _AudioEmbedding(
            wrapper_num_codebooks,
            wrapper_vocab_size,
            args.hidden_size,
            config.wrapper_audio_embed_size,
        )
        self.layers = [
            qwen3.TransformerBlock(args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        *,
        input_embeddings: Optional[mx.array] = None,
        cache: Optional[list[KVCache]] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        if (input_ids is None) == (input_embeddings is None):
            raise ValueError("Pass exactly one of input_ids or input_embeddings.")
        hidden = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(input_ids)
        )
        if cache is None:
            cache = [None] * len(self.layers)
        if mask is None:
            mask = create_attention_mask(hidden, cache[0])
        for layer, layer_cache in zip(self.layers, cache):
            hidden = layer(hidden, mask, layer_cache)
        return self.norm(hidden)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]

    def make_batch_cache(self, left_padding: list[int]) -> list[BatchKVCache]:
        """Per-row offset caches for left-padded batch prompts."""
        caches = [BatchKVCache(left_padding) for _ in self.layers]
        for cache in caches:
            cache.step = BACKBONE_CACHE_STEP
        return caches


class _TextEmbedding(nn.Module):
    """T5Gemma2 embedding with Breeze's separate end-of-input embedding."""

    def __init__(self, vocab_size: int, hidden_size: int, eoi_token_index: int):
        super().__init__()
        self.weight = mx.zeros((vocab_size, hidden_size))
        self.eoi_embedding = mx.zeros((hidden_size,))
        self.eoi_token_index = eoi_token_index

    def __call__(self, input_ids: mx.array) -> mx.array:
        embeds = self.weight[input_ids] * (self.weight.shape[-1] ** 0.5)
        return mx.where(
            (input_ids == self.eoi_token_index)[..., None], self.eoi_embedding, embeds
        )


class _T5Gemma2RMSNorm(nn.Module):
    """T5Gemma2's zero-initialized, offset RMS norm."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


def _t5gemma2_attention_mask(
    length: int,
    layer_type: str,
    sliding_window: Optional[int],
    attention_mask: Optional[mx.array] = None,
) -> Optional[mx.array]:
    """Build the official non-causal T5Gemma2 attention pattern.

    Full-attention layers are bidirectional.  Sliding layers use the
    asymmetric-looking but symmetric-around-the-centre window specified by
    T5Gemma2: ``(window + 1) // 2`` tokens left and ``window // 2 + 1`` right
    (both include the query position).
    """
    if length < 0:
        raise ValueError(f"Attention length must be non-negative, got {length}")
    if layer_type not in {"full_attention", "sliding_attention"}:
        raise ValueError(f"Unsupported T5Gemma2 attention layer: {layer_type}")
    if layer_type == "sliding_attention" and (
        sliding_window is None or sliding_window <= 0
    ):
        raise ValueError(
            "T5Gemma2 sliding_attention requires a positive sliding_window"
        )

    valid_keys = None
    if attention_mask is not None:
        valid_keys = mx.array(attention_mask).astype(mx.bool_)
        if valid_keys.ndim == 1:
            if valid_keys.shape[0] != length:
                raise ValueError(
                    "T5Gemma2 attention_mask length does not match input: "
                    f"{valid_keys.shape[0]} != {length}."
                )
            valid_keys = valid_keys[None, :]
        elif valid_keys.ndim == 2:
            if valid_keys.shape[1] != length:
                raise ValueError(
                    "T5Gemma2 attention_mask length does not match input: "
                    f"{valid_keys.shape[1]} != {length}."
                )
        else:
            raise ValueError(
                "T5Gemma2 attention_mask must have shape [length] or "
                f"[batch, length], got {valid_keys.shape}."
            )

    if layer_type == "full_attention":
        # ``None`` is intentional: MLX interprets this as unconstrained,
        # bidirectional attention.  A padding mask is broadcast over query
        # rows while masking invalid key/value positions, matching the
        # upstream T5Gemma2 eager implementation.
        if valid_keys is None:
            return None
        return valid_keys[:, None, None, :]

    query = mx.arange(length)[:, None]
    key = mx.arange(length)[None, :]
    distance = query - key
    left = (sliding_window + 1) // 2
    right = sliding_window // 2 + 1
    local = ((distance >= 0) & (distance < left)) | (
        (distance < 0) & (-distance < right)
    )
    if valid_keys is None:
        return local
    return local[None, None, :, :] & valid_keys[:, None, None, :]


class _TextAttention(nn.Module):
    """Bidirectional attention matching the T5Gemma2 text encoder."""

    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        self.n_heads = int(config["num_attention_heads"])
        self.n_kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.scale = config.get("query_pre_attn_scalar", 256) ** -0.5
        layer_types = (
            config.get("layer_types")
            or ["sliding_attention"] * config["num_hidden_layers"]
        )
        if layer_idx >= len(layer_types):
            raise ValueError(
                "T5Gemma2 layer_types has fewer entries than num_hidden_layers"
            )
        self.layer_type = layer_types[layer_idx]
        self.sliding_window = (
            config.get("sliding_window", 512)
            if self.layer_type == "sliding_attention"
            else None
        )
        dim = int(config["hidden_size"])
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        eps = config["rms_norm_eps"]
        self.q_norm = _T5Gemma2RMSNorm(self.head_dim, eps)
        self.k_norm = _T5Gemma2RMSNorm(self.head_dim, eps)
        rope_parameters = config.get("rope_parameters") or {}
        rope = rope_parameters.get(self.layer_type) or {}
        self.rope = initialize_rope(
            self.head_dim,
            base=rope.get(
                "rope_theta",
                10_000.0 if self.layer_type == "sliding_attention" else 1_000_000.0,
            ),
            traditional=False,
            scaling_config=rope if rope.get("rope_type") == "linear" else None,
            max_position_embeddings=config.get("max_position_embeddings", 32768),
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array]) -> mx.array:
        batch, length, _ = x.shape
        queries = self.q_norm(
            self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        values = (
            self.v_proj(x)
            .reshape(batch, length, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        output = scaled_dot_product_attention(
            self.rope(queries), self.rope(keys), values, None, self.scale, mask
        )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class _TextMLP(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        dim, intermediate = config["hidden_size"], config["intermediate_size"]
        self.gate_proj = nn.Linear(dim, intermediate, bias=False)
        self.up_proj = nn.Linear(dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class _TextLayer(nn.Module):
    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        self.self_attn = _TextAttention(config, layer_idx)
        eps, dim = config["rms_norm_eps"], config["hidden_size"]
        self.pre_self_attn_layernorm = _T5Gemma2RMSNorm(dim, eps)
        self.post_self_attn_layernorm = _T5Gemma2RMSNorm(dim, eps)
        self.mlp = _TextMLP(config)
        self.pre_feedforward_layernorm = _T5Gemma2RMSNorm(dim, eps)
        self.post_feedforward_layernorm = _T5Gemma2RMSNorm(dim, eps)
        self.layer_type = self.self_attn.layer_type

    def __call__(self, x: mx.array, mask: Optional[mx.array]) -> mx.array:
        hidden = self.self_attn(self.pre_self_attn_layernorm(x), mask)
        hidden = x + self.post_self_attn_layernorm(hidden)
        return hidden + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(hidden))
        )


class _TextEncoder(nn.Module):
    """MLX implementation of Breeze's bidirectional T5Gemma2 text encoder."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        config = self._normalize_config(config)
        self.config = config
        self.embed_tokens = _TextEmbedding(
            config["vocab_size"],
            config["hidden_size"],
            config.get("eoi_token_index", 256000),
        )
        self.layers = [
            _TextLayer(config, i) for i in range(config["num_hidden_layers"])
        ]
        self.norm = _T5Gemma2RMSNorm(config["hidden_size"], config["rms_norm_eps"])

    @staticmethod
    def _normalize_config(config: Optional[dict]) -> dict:
        """Fill only the T5Gemma2 defaults needed by direct MLX callers."""
        config = dict(ModelConfig._as_mapping(config))
        defaults = {
            "vocab_size": 262208,
            "hidden_size": 2304,
            "intermediate_size": 9216,
            "num_hidden_layers": 26,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "rms_norm_eps": 1e-6,
            "query_pre_attn_scalar": 256,
            "max_position_embeddings": 131072,
            "sliding_window": 4096,
            "eoi_token_index": 256000,
            "dropout_rate": 0.0,
            "attention_bias": False,
        }
        for key, value in defaults.items():
            config.setdefault(key, value)
        if not config.get("layer_types"):
            pattern = config.pop(
                "_sliding_window_pattern",
                config.pop("sliding_window_pattern", 6),
            )
            if not isinstance(pattern, int) or pattern <= 0:
                raise ValueError(
                    "T5Gemma2 sliding_window_pattern must be a positive integer"
                )
            config["layer_types"] = [
                "sliding_attention" if (idx + 1) % pattern else "full_attention"
                for idx in range(config["num_hidden_layers"])
            ]
        # Keep the upstream defaults for the two attention-specific RoPE
        # bases, while allowing an explicitly supplied mapping to override
        # either layer type.
        rope_parameters = {
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
            },
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 1_000_000.0,
            },
        }
        provided_rope_parameters = config.get("rope_parameters") or {}
        for key, value in provided_rope_parameters.items():
            rope_parameters[key] = dict(value or {})
        # Older T5Gemma2 exports use the generic ``rope_scaling`` field for
        # the full-attention stream.  Keep accepting that form while giving
        # the per-layer ``rope_parameters`` mapping precedence when present.
        if config.get("rope_scaling") and (
            "full_attention" not in provided_rope_parameters
        ):
            rope_parameters["full_attention"].update(config["rope_scaling"])
        config["rope_parameters"] = rope_parameters
        return config

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        hidden = self.embed_tokens(input_ids)
        masks = {}
        for layer in self.layers:
            if layer.layer_type in masks:
                continue
            masks[layer.layer_type] = _t5gemma2_attention_mask(
                hidden.shape[1],
                layer.layer_type,
                layer.self_attn.sliding_window,
                attention_mask,
            )
        for layer in self.layers:
            hidden = layer(hidden, masks[layer.layer_type])
        return self.norm(hidden)


class _DepthKVCache:
    """Per-frame depth cache: grows by concatenation, holds exactly the frame.

    A depth frame is at most 16 positions.  ``KVCache`` reserves 256 slots
    per row and layer (zero-filled for every frame) and hands attention a
    strided view of them; returning the contiguous concatenation instead is
    5-12% faster per frame from 8 rows up, with identical tokens.  There is
    deliberately no ``make_mask``: positions are added one at a time, so
    ``create_attention_mask`` returns ``None`` (no mask) as it should.
    """

    def __init__(self):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        self.offset += keys.shape[2]
        return self.keys, self.values


class _DepthModel(nn.Module):
    """Breeze's codebook-autoregressive depth decoder."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        depth = ModelConfig._as_mapping(config.depth_decoder_config)
        self.vocab_size = int(depth.get("vocab_size", config.wrapper_vocab_size))
        self.num_codebooks = int(
            depth.get("num_codebooks", config.wrapper_num_codebooks)
        )
        self.audio_embed_size = int(
            depth.get("audio_embed_size", config.wrapper_audio_embed_size)
        )
        self.backbone_hidden_size = int(
            depth.get("backbone_hidden_size", config.backbone_hidden_size)
        )
        self.hidden_size = int(depth.get("hidden_size", 1024))
        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=self.hidden_size,
            num_hidden_layers=depth.get("num_hidden_layers", 12),
            intermediate_size=depth.get("intermediate_size", 8192),
            num_attention_heads=depth.get("num_attention_heads", 8),
            num_key_value_heads=depth.get("num_key_value_heads", 2),
            head_dim=depth.get("head_dim", 128),
            rms_norm_eps=depth.get("rms_norm_eps", 1e-5),
            vocab_size=self.num_codebooks * self.vocab_size,
            max_position_embeddings=depth.get("max_position_embeddings", 33),
            rope_theta=depth.get("rope_theta", 500000.0),
            rope_scaling=depth.get("rope_scaling"),
            tie_word_embeddings=False,
            rope_traditional=False,
        )
        self.embed_tokens = nn.Embedding(args.vocab_size, self.audio_embed_size)
        if self.backbone_hidden_size != self.audio_embed_size:
            self.backbone_hidden_state_projector = nn.Linear(
                self.backbone_hidden_size, self.audio_embed_size, bias=False
            )
        else:
            self.backbone_hidden_state_projector = None
        self.inputs_embeds_projector = nn.Linear(
            self.audio_embed_size, self.hidden_size, bias=False
        )
        self.layers = [
            llama.TransformerBlock(args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(self.hidden_size, eps=args.rms_norm_eps)

    def make_cache(self) -> list[_DepthKVCache]:
        """A fresh KV cache for one frame of depth decoding."""
        return [_DepthKVCache() for _ in self.layers]

    def store_linear_weights_transposed(self) -> None:
        """Keep each Linear weight as a transposed view of an [in, out] buffer.

        ``nn.Linear`` computes ``x @ weight.T``.  With the checkpoint's
        row-major ``[out, in]`` weights, MLX's matmul for more than 64 rows
        streams the transposed operand once per 64-row tile; stored this
        way, ``weight.T`` is row-major and the 16 depth passes per frame run
        9-17% faster at 72-128 rows, unchanged at 64 rows or fewer.  The
        values (and the parameter shapes) are the same; only the memory
        layout changes, so it costs no extra memory.
        """
        for _, module in self.named_modules():
            if isinstance(module, nn.Linear):
                module.weight = mx.contiguous(module.weight.T).T

    def embed_codebook_token(
        self, token_id: Union[int, mx.array], codebook_idx: int
    ) -> mx.array:
        """Embed codebook ids as ``[batch, 1, embed]``.

        ``token_id`` is an int, a one-element array (batch 1) or a ``[batch]``
        device array (one token per row).
        """
        if not isinstance(token_id, mx.array):
            token_id = mx.array(token_id)
        ids = token_id.astype(mx.int32).reshape(-1, 1) + codebook_idx * self.vocab_size
        return self.embed_tokens(ids)

    def step(self, embeds: mx.array, cache: list[KVCache]) -> mx.array:
        """Run one position through the stack, extending ``cache`` in place."""
        hidden = self.inputs_embeds_projector(embeds)
        mask = create_attention_mask(hidden, cache[0])
        for layer, layer_cache in zip(self.layers, cache):
            hidden = layer(hidden, mask, layer_cache)
        return self.norm(hidden)

    def __call__(
        self, token_ids: mx.array, backbone_hidden_state: mx.array
    ) -> mx.array:
        if token_ids.ndim != 2:
            raise ValueError(
                f"Depth decoder token_ids must have shape [batch, time], got {token_ids.shape}."
            )
        if backbone_hidden_state.ndim != 2:
            raise ValueError(
                "Depth decoder backbone_hidden_state must have shape "
                f"[batch, hidden], got {backbone_hidden_state.shape}."
            )
        if backbone_hidden_state.shape[0] != token_ids.shape[0]:
            raise ValueError(
                "Depth decoder token and hidden-state batch sizes differ: "
                f"{token_ids.shape[0]} != {backbone_hidden_state.shape[0]}."
            )
        # token_ids starts with a placeholder. Position zero is replaced by the
        # backbone state, while later positions carry codebook-specific offsets.
        positions = mx.maximum(mx.arange(token_ids.shape[1]) - 1, 0)
        embeds = self.embed_tokens(token_ids + positions[None, :] * self.vocab_size)
        if self.backbone_hidden_state_projector is not None:
            backbone_hidden_state = self.backbone_hidden_state_projector(
                backbone_hidden_state
            )
        embeds = mx.concatenate(
            [backbone_hidden_state[:, None, :], embeds[:, 1:, :]], axis=1
        )
        hidden = self.inputs_embeds_projector(embeds)
        mask = create_attention_mask(hidden, None)
        for layer in self.layers:
            hidden = layer(hidden, mask, None)
        return self.norm(hidden)


class _DepthDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model = _DepthModel(config)
        self.codebooks_head = _CodebooksHead(
            self.model.num_codebooks - 1,
            self.model.hidden_size,
            self.model.vocab_size,
        )

    def next_logits(
        self, token_ids: mx.array, backbone_hidden_state: mx.array
    ) -> mx.array:
        # The most recent input token is codebook N-1, and predicts codebook N.
        head_idx = token_ids.shape[1] - 2
        if not 0 <= head_idx < self.codebooks_head.weight.shape[0]:
            raise ValueError(
                "Depth decoder input must contain the placeholder and at least "
                f"one codebook token; got sequence length {token_ids.shape[1]}."
            )
        hidden = self.model(token_ids, backbone_hidden_state)[:, -1, :]
        return hidden @ self.codebooks_head.weight[head_idx]

    def start_frame(
        self, backbone_hidden_state: mx.array, cache: list[KVCache]
    ) -> None:
        """Seed a frame's cache with position zero: the backbone hidden state."""
        model = self.model
        if model.backbone_hidden_state_projector is not None:
            backbone_hidden_state = model.backbone_hidden_state_projector(
                backbone_hidden_state
            )
        model.step(backbone_hidden_state[:, None, :], cache)

    def step_logits(
        self, cache: list[KVCache], *, head_idx: int, token_id: Union[int, mx.array]
    ) -> mx.array:
        """Predict codebook ``head_idx + 1`` from the token at ``head_idx``."""
        embeds = self.model.embed_codebook_token(token_id, head_idx)
        hidden = self.model.step(embeds, cache)[:, -1, :]
        return hidden @ self.codebooks_head.weight[head_idx]


class _CodebooksHead(nn.Module):
    """Position-specific output projections for depth-decoded codebooks."""

    def __init__(self, num_heads: int, hidden_size: int, vocab_size: int):
        super().__init__()
        self.weight = mx.zeros((num_heads, hidden_size, vocab_size))


class Model(nn.Module):
    """Breeze TTS 2 text-to-speech model."""

    preserve_ref_audio_path = True

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_codebooks = config.wrapper_num_codebooks
        self.vocab_size = config.wrapper_vocab_size
        self.lm_head = nn.Linear(
            config.backbone_hidden_size, self.vocab_size + 1, bias=False
        )
        self.embed_text_tokens = nn.Embedding(
            config.text_vocab_size, config.backbone_hidden_size
        )
        self.backbone_model = _Backbone(config)
        self.depth_decoder = _DepthDecoder(config)
        text_config = _TextEncoder._normalize_config(config.text_encoder_config)
        self.text_encoder = _TextEncoder(text_config)
        self.text_encoder_proj = nn.Linear(
            text_config["hidden_size"],
            config.backbone_hidden_size,
            bias=False,
        )
        self.tokenizer = None
        self.audio_tokenizer = None

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def model_type(self) -> str:
        return "breeze"

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Keep only main-model tensors; the bundled codec is loaded separately."""
        sanitized = {}
        for key, value in weights.items():
            if (
                key.startswith("codec_model.")
                or key.endswith(".initialized")
                or "rotary_emb.inv_freq" in key
            ):
                continue
            sanitized[key] = value
        # The official checkpoint ties these embeddings and stores only the
        # depth-decoder tensor in safetensors. MLX modules retain independent
        # parameters, so materialize the matching backbone entry at load time.
        tied = "depth_decoder.model.embed_tokens.weight"
        backbone_tied = "backbone_model.embed_tokens.embed_audio_tokens.weight"
        if tied in sanitized:
            # Always derive the backbone copy from the canonical depth tensor.
            # This handles exports that contain a stale duplicate while
            # preserving exact value/shape parity for strict MLX loading.
            sanitized[backbone_tied] = sanitized[tied]
        return sanitized

    @staticmethod
    def _validate_codec_weight_keys(expected: set[str], provided: set[str]) -> None:
        """Reject codec snapshots except for generated quantizer flags.

        Qwen3-TTS computes the 32 ``codebook.initialized`` flags at runtime;
        they are intentionally not serialized.  Every real parameter must be
        present exactly once so a changed codec architecture cannot be loaded
        with a misleading ``strict=False`` fallback.
        """
        allowed_missing = {
            name
            for name in expected
            if name.startswith("encoder_model.quantizer.")
            and name.endswith(".codebook.initialized")
        }
        missing = expected - provided - allowed_missing
        unexpected = provided - expected
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing: {', '.join(sorted(missing))}")
            if unexpected:
                details.append(f"unexpected: {', '.join(sorted(unexpected))}")
            raise ValueError(
                "Incompatible Breeze audio tokenizer weights ("
                + "; ".join(details)
                + ")"
            )

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        from transformers import PreTrainedTokenizerFast

        # AutoTokenizer attempts to instantiate the unknown upstream ``breeze``
        # config before loading the otherwise standard fast tokenizer. Loading
        # tokenizer.json directly avoids that PyTorch-runtime-only registration.
        tokenizer_config = json.loads(
            (model_path / "tokenizer_config.json").read_text()
        )
        for key in (
            "tokenizer_class",
            "model_max_length",
            "clean_up_tokenization_spaces",
            "added_tokens_decoder",
        ):
            tokenizer_config.pop(key, None)
        model.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(model_path / "tokenizer.json"), **tokenizer_config
        )
        codec_path = model_path / "audio_tokenizer"
        if not codec_path.is_dir():
            raise FileNotFoundError(
                f"Breeze checkpoint has no audio_tokenizer at {codec_path}"
            )
        model.audio_tokenizer = cls._load_audio_tokenizer(codec_path)
        model.depth_decoder.model.store_linear_weights_transposed()
        mx.eval(model.depth_decoder.parameters())
        codec_dtype = cls._codec_decode_dtype()
        if codec_dtype != mx.float32:
            model.audio_tokenizer.decoder.set_compute_dtype(codec_dtype)
        return model

    @staticmethod
    def _codec_decode_dtype() -> mx.Dtype:
        """Precision of the codec's waveform decoder (not its encoder).

        Half precision is the default: on an Apple M6 it decodes about 2.3x
        faster than float32 with half the memory; against the float32
        waveform the difference is typically 43-45 dB below the signal and
        30-36 dB on the worst clips measured.  Set ``MLX_AUDIO_BREEZE_CODEC_DTYPE=float32`` to keep the
        checkpoint's precision.
        """
        raw = os.environ.get(CODEC_DTYPE_ENV, "").strip().lower() or "float16"
        try:
            return _CODEC_DTYPES[raw]
        except KeyError:
            raise ValueError(
                f"{CODEC_DTYPE_ENV} must be 'float16' or 'float32', got {raw!r}."
            ) from None

    @staticmethod
    def _load_audio_tokenizer(codec_path: Path):
        from mlx_audio.tts.models.qwen3_tts.config import (
            Qwen3TTSTokenizerConfig,
            Qwen3TTSTokenizerDecoderConfig,
            Qwen3TTSTokenizerEncoderConfig,
            filter_dict_for_dataclass,
        )
        from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import (
            Qwen3TTSSpeechTokenizer,
        )

        config_data = json.loads((codec_path / "config.json").read_text())
        decoder = Qwen3TTSTokenizerDecoderConfig(
            **filter_dict_for_dataclass(
                Qwen3TTSTokenizerDecoderConfig, config_data["decoder_config"]
            )
        )
        encoder = Qwen3TTSTokenizerEncoderConfig(
            **filter_dict_for_dataclass(
                Qwen3TTSTokenizerEncoderConfig, config_data["encoder_config"]
            )
        )
        codec_config = Qwen3TTSTokenizerConfig(
            decoder_config=decoder, encoder_config=encoder
        )
        for key, value in config_data.items():
            if key not in {"decoder_config", "encoder_config"} and hasattr(
                codec_config, key
            ):
                setattr(codec_config, key, value)
        codec = Qwen3TTSSpeechTokenizer(codec_config)
        weights = {}
        for weight_file in codec_path.glob("*.safetensors"):
            weights.update(mx.load(str(weight_file)))
        sanitized = Qwen3TTSSpeechTokenizer.sanitize(weights)
        Model._validate_codec_weight_keys(
            {name for name, _ in tree_flatten(codec.parameters())}, set(sanitized)
        )
        # The 32 quantizer ``initialized`` flags are generated locally by
        # ``update_in_place`` below.  All serialized keys were validated above.
        # ``initialized`` is an MLX-only runtime flag, so the model loader must
        # permit precisely those known missing keys.  The complete set of
        # real keys was validated above; this is intentionally not a blanket
        # strict=False load.
        codec.load_weights(list(sanitized.items()), strict=False)
        if codec.encoder_model is not None:
            quantizer = codec.encoder_model.quantizer
            for layer in quantizer.rvq_first.vq.layers + quantizer.rvq_rest.vq.layers:
                layer.codebook.update_in_place()
        mx.eval(codec.parameters())
        codec.eval()
        return codec

    def _speaker(self, voice: Optional[str]) -> str:
        # mlx-audio's generic default is a Kokoro voice; Breeze's documented
        # default speaker tag is S0.
        speaker = "S0" if voice in (None, "", "af_heart") else voice
        return speaker if speaker.startswith("[") else f"[{speaker}]"

    def _text_ids(self, text: str) -> mx.array:
        if self.tokenizer is None:
            raise RuntimeError("Breeze tokenizer was not loaded")
        return mx.array(self.tokenizer(text, add_special_tokens=True)["input_ids"])

    # Reference clips encoded by path, kept across calls (a few KiB each).
    REFERENCE_CACHE_SIZE = 64

    def _encode_reference(self, ref_audio: Union[str, Path, mx.array]) -> mx.array:
        """Codec codes ``[1, frames, codebooks]`` of a reference clip.

        Clips given by path are cached across calls, keyed by the resolved
        path, modification time, size and codec, so a replaced file (or a
        swapped codec) is re-encoded.
        """
        if self.audio_tokenizer is None:
            raise RuntimeError("Breeze audio tokenizer was not loaded")
        key = None
        if isinstance(ref_audio, (str, Path)):
            try:
                stat = os.stat(ref_audio)
                key = (
                    os.path.realpath(ref_audio),
                    stat.st_mtime_ns,
                    stat.st_size,
                    id(self.audio_tokenizer),
                )
            except OSError:
                key = None
            # A plain attribute: nn.Module would treat a dict as parameters.
            cache = self.__dict__.setdefault("_reference_codes", OrderedDict())
            if key is not None and key in cache:
                cache.move_to_end(key)
                return cache[key]
            codes = self._encode_reference_audio(ref_audio)
            if key is not None:
                cache[key] = codes
                while len(cache) > self.REFERENCE_CACHE_SIZE:
                    cache.popitem(last=False)
            return codes
        return self._encode_reference_audio(ref_audio)

    def _encode_reference_audio(
        self, ref_audio: Union[str, Path, mx.array]
    ) -> mx.array:
        if isinstance(ref_audio, (str, Path)):
            audio = load_audio(str(ref_audio), sample_rate=self.sample_rate)
        else:
            audio = ref_audio
        audio = mx.array(audio)
        if audio.ndim == 1:
            audio = audio[None, None, :]
        elif audio.ndim == 2:
            audio = audio[:, None, :]
        elif audio.ndim != 3:
            raise ValueError(
                "ref_audio must be a path, [samples], [batch, samples], or [batch, channels, samples]."
            )
        if audio.shape[0] != 1:
            raise ValueError(
                "Breeze supports exactly one reference audio item per generation; "
                f"received batch size {audio.shape[0]}."
            )
        return mx.transpose(self.audio_tokenizer.encode(audio), (0, 2, 1))

    def _prompt_embeddings(
        self,
        text: str,
        *,
        voice: Optional[str],
        instruct: Optional[str],
        ref_audio: Optional[Union[str, Path, mx.array]],
        ref_text: Optional[str],
        reference_codes: Optional[mx.array] = None,
    ) -> mx.array:
        """Build one row's ``[1, length, hidden]`` prompt embeddings.

        ``reference_codes`` lets batch callers reuse an already encoded
        ``ref_audio`` (``[1, frames, codebooks]``) instead of re-running the
        codec encoder for every row that shares a reference clip.
        """
        return self._embed_segments(
            self._prompt_segments(
                text,
                voice=voice,
                instruct=instruct,
                ref_audio=ref_audio,
                ref_text=ref_text,
                reference_codes=reference_codes,
            )
        )

    def _prompt_segments(
        self,
        text: str,
        *,
        voice: Optional[str],
        instruct: Optional[str],
        ref_audio: Optional[Union[str, Path, mx.array]],
        ref_text: Optional[str],
        reference_codes: Optional[mx.array] = None,
    ) -> list[tuple[str, Any]]:
        """A prompt as ``("text", string)`` / ``("audio", codes)`` segments."""
        if isinstance(ref_audio, (list, tuple)):
            if len(ref_audio) != 1:
                raise ValueError(
                    "Breeze supports exactly one reference audio item per generation."
                )
            ref_audio = ref_audio[0]
        if isinstance(ref_text, (list, tuple)):
            if len(ref_text) != 1:
                raise ValueError(
                    "Breeze supports exactly one reference transcript per generation."
                )
            ref_text = ref_text[0]
        if ref_audio is not None and not ref_text:
            raise ValueError("Breeze voice cloning requires ref_text with ref_audio.")
        speaker = self._speaker(voice)
        segments: list[tuple[str, Any]] = []
        if ref_audio is not None:
            segments.append(("text", f"{speaker}{ref_text}"))
            segments.append(
                (
                    "audio",
                    (
                        reference_codes
                        if reference_codes is not None
                        else self._encode_reference(ref_audio)
                    ),
                )
            )
        target = f"{speaker}{text}"
        if instruct:
            target = f"{speaker}<ins_bos>{instruct}<ins_eos>{text}"
        segments.append(("text", target))
        return segments

    def _embed_segments(
        self,
        segments: list[tuple[str, Any]],
        text_embeddings: Optional[dict[str, mx.array]] = None,
    ) -> mx.array:
        """Concatenate segment embeddings into ``[1, length, hidden]``.

        ``text_embeddings`` maps each text segment to its already encoded
        ``[1, tokens, hidden]`` embedding (see ``_encode_texts``); without
        it every text segment is encoded on its own.
        """
        embeddings: list[mx.array] = []
        for kind, values in segments:
            if kind == "text":
                if text_embeddings is not None:
                    embeddings.append(text_embeddings[values])
                    continue
                hidden = self.text_encoder(self._text_ids(values)[None, :])
                embeddings.append(self.text_encoder_proj(hidden))
            else:
                embeddings.append(self.backbone_model.embed_tokens(values))
                eos_codes = mx.full(
                    (1, 1, self.num_codebooks), self.config.codebook_eos_token_id
                )
                embeddings.append(self.backbone_model.embed_tokens(eos_codes))
        return mx.concatenate(embeddings, axis=1)

    def _encode_texts(self, strings: list[str], chunk: int = 64) -> dict[str, mx.array]:
        """Encode each distinct string once, in right-padded batches.

        Returns ``{string: [1, tokens, hidden]}``, the projected text-encoder
        output that ``_embed_segments`` concatenates.  The encoder is
        bidirectional, so padding is hidden with its attention mask; rows
        are grouped by token count to keep the padding small.
        """
        ids = {text: self._text_ids(text) for text in dict.fromkeys(strings)}
        order = sorted(ids, key=lambda text: int(ids[text].shape[0]))
        encoded: dict[str, mx.array] = {}
        for start in range(0, len(order), chunk):
            part = order[start : start + chunk]
            lengths = [int(ids[text].shape[0]) for text in part]
            width = max(lengths)
            tokens = mx.stack(
                [mx.pad(ids[text], (0, width - n)) for text, n in zip(part, lengths)]
            )
            mask = (mx.arange(width)[None, :] < mx.array(lengths)[:, None]).astype(
                mx.int32
            )
            hidden = self.text_encoder_proj(
                self.text_encoder(tokens, attention_mask=mask)
            )
            for row, (text, n) in enumerate(zip(part, lengths)):
                encoded[text] = hidden[row : row + 1, :n]
        return encoded

    def _sample(
        self,
        logits: mx.array,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        allow_eos: bool = False,
    ) -> int:
        token = self._sample_array(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            allow_eos=allow_eos,
        )
        return int(token.item())

    def _sample_array(
        self,
        logits: mx.array,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        allow_eos: bool = False,
    ) -> mx.array:
        """Sample token ids from ``[batch, vocab]`` logits, leaving them on device."""
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 <= top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")

        valid = self.vocab_size + 1 if allow_eos else self.vocab_size
        if logits.shape[-1] < valid:
            raise ValueError(
                "Breeze logits do not contain the expected codebook vocabulary: "
                f"{logits.shape[-1]} < {valid}."
            )

        # Keep EOS as the one extra class used by the backbone while reserving
        # padding/control ids in every regular codebook distribution.  Clamp
        # top-k to this distribution: tiny deterministic fixtures often use a
        # vocab smaller than the user-facing default of 50.
        logits = self._mask_reserved_codec_logits(logits)[..., :valid]
        effective_top_k = min(top_k, valid) if top_k else 0
        if effective_top_k == valid:
            effective_top_k = 0
        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=effective_top_k)
        return sampler(nn.log_softmax(logits, axis=-1))

    def _mask_reserved_codec_logits(self, logits: mx.array) -> mx.array:
        """Mask ids outside the codec codebook in a logits tensor.

        Breeze's wrapper vocabulary contains codec entries followed by
        padding/control ids (and one separate backbone EOS class).  The same
        mask is used for the first codebook and for every depth-decoder step;
        callers may still retain the EOS column by slicing after this helper.
        """
        if logits.shape[-1] < self.vocab_size:
            raise ValueError(
                "Breeze logits do not contain the wrapper vocabulary: "
                f"{logits.shape[-1]} < {self.vocab_size}."
            )
        start = self.config.codec_vocab_size
        if start >= self.vocab_size:
            return logits
        return logits.at[..., start : self.vocab_size].add(-mx.inf)

    @staticmethod
    def _apply_repetition_penalty(
        logits: mx.array, generated_tokens: list[int], penalty: float
    ) -> mx.array:
        """Apply the standard sign-aware repetition penalty to prior tokens."""
        if penalty == 1.0 or not generated_tokens:
            return logits
        token_ids = sorted(
            {token for token in generated_tokens if 0 <= token < logits.shape[-1]}
        )
        if not token_ids:
            return logits
        indices = mx.array(token_ids, dtype=mx.int32)
        selected = mx.take(logits, indices, axis=-1)
        penalized = mx.where(selected < 0, selected * penalty, selected / penalty)
        return mx.put_along_axis(logits, indices[None, :], penalized, axis=-1)

    def _depth_tokens(
        self,
        first_codebook: int,
        conditional_hidden: mx.array,
        *,
        unconditional_hidden: Optional[mx.array],
        cfg_scale: float,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> list[int]:
        """Complete one frame using fresh caches and a single host readback."""
        depth = self.depth_decoder
        cond_cache = depth.model.make_cache()
        depth.start_frame(conditional_hidden, cond_cache)
        uncond_cache = None
        if unconditional_hidden is not None:
            uncond_cache = depth.model.make_cache()
            depth.start_frame(unconditional_hidden, uncond_cache)
        token: Union[int, mx.array] = first_codebook
        sampled: list[mx.array] = []
        for head_idx in range(self.num_codebooks - 1):
            logits = depth.step_logits(cond_cache, head_idx=head_idx, token_id=token)
            if uncond_cache is not None:
                unconditional_logits = depth.step_logits(
                    uncond_cache, head_idx=head_idx, token_id=token
                )
                logits = unconditional_logits + cfg_scale * (
                    logits - unconditional_logits
                )
            # Custom samplers must also receive masked logits.
            logits = self._mask_reserved_codec_logits(logits)
            token = self._sample_array(
                logits, temperature=temperature, top_p=top_p, top_k=top_k
            ).reshape(1)
            # Start this step on the GPU while the next one is being built.
            mx.async_eval(token)
            sampled.append(token)
        if not sampled:
            return [first_codebook]
        return [first_codebook, *mx.concatenate(sampled).tolist()]

    @staticmethod
    def _audio_vector(audio: Any) -> mx.array:
        """Normalize codec output to the one-dimensional public waveform."""
        if hasattr(audio, "audio_values"):
            audio = audio.audio_values
        while isinstance(audio, (list, tuple)):
            audio = audio[0]
        audio = mx.array(audio)
        while audio.ndim > 1:
            audio = audio[0]
        return audio

    def _decode_codes(self, codes: mx.array) -> mx.array:
        """Decode one unpadded sequence into its full waveform.

        The codec also returns a "valid length", counted as the frames whose
        first code is above 0, for batches right-padded with zeros.  Breeze
        never pads: a row ends at the EOS class, and 0 is an ordinary code
        that turns up inside speech (about 1 frame in 700), so that length
        cut 80 ms off the end of the row for every such frame.
        """
        decoded = self.audio_tokenizer.decode(codes)
        if isinstance(decoded, (list, tuple)):
            if not decoded:
                return mx.zeros((0,), dtype=mx.float32)
            decoded = decoded[0]
        return self._audio_vector(decoded)

    def _empty_audio(self) -> mx.array:
        """Return the official silent fallback when EOS precedes all frames."""
        # The upstream implementation decodes a one-frame dummy code rather
        # than raising.  Keep that behavior when a codec is available, while
        # retaining a zero-length CPU-safe fallback for light test doubles.
        try:
            dummy = mx.ones((1, 1, self.num_codebooks), dtype=mx.int32)
            return self._decode_codes(dummy)
        except Exception:  # pragma: no cover - only used by partial fakes
            return mx.zeros((0,), dtype=mx.float32)

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        instruct: Optional[str] = None,
        ref_audio: Optional[Union[str, Path, mx.array]] = None,
        ref_text: Optional[str] = None,
        cfg_scale: Optional[float] = None,
        max_tokens: int = 750,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
        stream: bool = False,
        streaming_interval: float = 2.0,
        split_pattern: Optional[str] = "\n",
        **_: object,
    ) -> Generator[GenerationResult, None, None]:
        """Generate Breeze audio for voice design, cloning, or direction."""
        if seed is not None:
            mx.random.seed(seed)
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if top_k < 0:
            raise ValueError("top_k must be non-negative.")
        if temperature < 0:
            raise ValueError("temperature must be non-negative.")
        if not 0 <= top_p <= 1:
            raise ValueError("top_p must be between 0 and 1.")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive.")
        if cfg_scale is not None and not math.isfinite(cfg_scale):
            raise ValueError("cfg_scale must be finite.")
        if streaming_interval <= 0:
            raise ValueError("streaming_interval must be positive.")

        if split_pattern:
            try:
                raw_segments = [
                    s.strip() for s in re.split(split_pattern, text) if s.strip()
                ]
            except Exception:
                raw_segments = [
                    s.strip() for s in text.split(split_pattern) if s.strip()
                ]
            segments: list[str] = []
            for segment in raw_segments:
                segments.extend(_split_long_segment(segment))
        else:
            segments = [text.strip()] if text.strip() else []
        if not segments:
            segments = [text]

        decode_rate = getattr(self.audio_tokenizer, "decode_upsample_rate", None)
        if decode_rate is None:
            decoder = getattr(self.audio_tokenizer, "decoder", None)
            decode_rate = getattr(decoder, "decode_upsample_rate", None)
        if decode_rate is None or decode_rate <= 0:
            raise ValueError("Breeze audio tokenizer has no valid decode rate")
        chunk_frames = max(1, int(streaming_interval * self.sample_rate / decode_rate))

        total_segments = len(segments)
        for segment_idx, segment_text in enumerate(segments):
            is_last_segment = segment_idx == total_segments - 1
            started = time.perf_counter()
            cond = self._prompt_embeddings(
                segment_text,
                voice=voice,
                instruct=instruct,
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
            use_cfg = bool(instruct) and cfg_scale not in (None, 1.0)
            scale = 1.0 if cfg_scale is None else cfg_scale
            if use_cfg:
                uncond = self._prompt_embeddings(
                    segment_text,
                    voice=voice,
                    instruct=None,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                )

            cond_cache = self.backbone_model.make_cache()
            cond_hidden = self.backbone_model(input_embeddings=cond, cache=cond_cache)[
                :, -1, :
            ]
            if use_cfg:
                uncond_cache = self.backbone_model.make_cache()
                uncond_hidden = self.backbone_model(
                    input_embeddings=uncond, cache=uncond_cache
                )[:, -1, :]

            frames: list[list[int]] = []
            pending_frames: list[list[int]] = []
            if stream:
                self.audio_tokenizer.decoder.reset_streaming_state()

            def stream_result(
                audio: mx.array, token_count: int, final: bool, s_idx: int = segment_idx
            ):
                mx.eval(audio)
                samples = audio.shape[0]
                elapsed = time.perf_counter() - started
                tokens_per_sec = token_count / elapsed if elapsed > 0 else 0.0
                samples_per_sec = samples / elapsed if elapsed > 0 else 0.0
                return GenerationResult(
                    audio=audio,
                    samples=samples,
                    sample_rate=self.sample_rate,
                    segment_idx=s_idx,
                    token_count=token_count,
                    audio_duration=f"00:00:{samples / self.sample_rate:06.3f}",
                    real_time_factor=elapsed / max(samples / self.sample_rate, 1e-6),
                    prompt={
                        "tokens": token_count,
                        "tokens-per-sec": tokens_per_sec,
                    },
                    audio_samples={
                        "samples": samples,
                        "samples-per-sec": samples_per_sec,
                    },
                    processing_time_seconds=elapsed,
                    peak_memory_usage=mx.get_peak_memory() / 1e9,
                    is_streaming_chunk=True,
                    is_final_chunk=final and is_last_segment,
                )

            for _ in range(max_tokens):
                cond_logits = self.lm_head(cond_hidden)
                if use_cfg:
                    uncond_logits = self.lm_head(uncond_hidden)
                    logits = uncond_logits + scale * (cond_logits - uncond_logits)
                else:
                    logits = cond_logits
                logits = self._apply_repetition_penalty(
                    logits, [frame[0] for frame in frames], repetition_penalty
                )
                logits = self._mask_reserved_codec_logits(logits)
                first = self._sample(
                    logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    allow_eos=True,
                )
                if first == self.vocab_size:
                    break
                frame = self._depth_tokens(
                    first,
                    cond_hidden,
                    unconditional_hidden=uncond_hidden if use_cfg else None,
                    cfg_scale=scale,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                frames.append(frame)
                pending_frames.append(frame)
                if stream and len(pending_frames) >= chunk_frames:
                    chunk = pending_frames[:chunk_frames]
                    del pending_frames[:chunk_frames]
                    pending_codes = mx.array(chunk, dtype=mx.int32)[None, :, :]
                    stream_audio = self.audio_tokenizer.decoder.streaming_step(
                        mx.transpose(pending_codes, (0, 2, 1))
                    )
                    yield stream_result(
                        self._audio_vector(stream_audio), len(chunk), final=False
                    )
                codebooks = mx.array(frame, dtype=mx.int32)[None, None, :]
                cond_hidden = self.backbone_model(
                    input_ids=codebooks, cache=cond_cache
                )[:, -1, :]
                if use_cfg:
                    uncond_hidden = self.backbone_model(
                        input_ids=codebooks, cache=uncond_cache
                    )[:, -1, :]

            if not frames:
                empty_audio = self._empty_audio()
                if stream:
                    self.audio_tokenizer.decoder.reset_streaming_state()
                mx.eval(empty_audio)
                elapsed = time.perf_counter() - started
                samples = empty_audio.shape[0]
                samples_per_sec = samples / elapsed if elapsed > 0 else 0.0
                yield GenerationResult(
                    audio=empty_audio,
                    samples=samples,
                    sample_rate=self.sample_rate,
                    segment_idx=segment_idx,
                    token_count=0,
                    audio_duration=f"00:00:{samples / self.sample_rate:06.3f}",
                    real_time_factor=0.0,
                    prompt={"tokens": 0, "tokens-per-sec": 0.0},
                    audio_samples={
                        "samples": samples,
                        "samples-per-sec": samples_per_sec,
                    },
                    processing_time_seconds=elapsed,
                    peak_memory_usage=mx.get_peak_memory() / 1e9,
                    is_streaming_chunk=stream,
                    is_final_chunk=is_last_segment,
                )
                continue
            if stream:
                if pending_frames:
                    pending_codes = mx.array(pending_frames, dtype=mx.int32)[None, :, :]
                    stream_audio = self.audio_tokenizer.decoder.streaming_step(
                        mx.transpose(pending_codes, (0, 2, 1))
                    )
                    self.audio_tokenizer.decoder.reset_streaming_state()
                    yield stream_result(
                        self._audio_vector(stream_audio),
                        len(pending_frames),
                        final=True,
                    )
                else:
                    self.audio_tokenizer.decoder.reset_streaming_state()
                continue
            codes = mx.array(frames, dtype=mx.int32)[None, :, :]
            audio = self._decode_codes(codes)
            samples = audio.shape[0]
            mx.eval(audio)
            elapsed = time.perf_counter() - started
            token_count = len(frames)
            tokens_per_sec = token_count / elapsed if elapsed > 0 else 0.0
            samples_per_sec = samples / elapsed if elapsed > 0 else 0.0
            yield GenerationResult(
                audio=audio,
                samples=samples,
                sample_rate=self.sample_rate,
                segment_idx=segment_idx,
                token_count=token_count,
                audio_duration=f"00:00:{samples / self.sample_rate:06.3f}",
                real_time_factor=elapsed / max(samples / self.sample_rate, 1e-6),
                prompt={
                    "tokens": token_count,
                    "tokens-per-sec": tokens_per_sec,
                },
                audio_samples={
                    "samples": samples,
                    "samples-per-sec": samples_per_sec,
                },
                processing_time_seconds=elapsed,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
                is_final_chunk=is_last_segment,
            )

    # ------------------------------------------------------------------
    # Batched generation
    # ------------------------------------------------------------------

    # batch_generate takes a reference clip (and transcript) per row, so the server
    # may batch requests whose clips differ.
    batch_generate_per_row_refs = True

    def supports_tts_batch(
        self,
        *,
        stream: bool = False,
        speed: Optional[float] = 1.0,
        pitch: Optional[float] = 1.0,
        **kwargs,
    ) -> bool:
        """Whether a request can be served by :meth:`batch_generate`.

        Voice tags, instructions, reference audio/text and CFG are all handled
        per row; only streaming and post-processing knobs are unsupported.
        """
        del kwargs
        if stream:
            return False
        return speed in (None, 1.0) and pitch in (None, 1.0)

    @staticmethod
    def _per_row(values: Any, count: int, name: str) -> list:
        """Broadcast ``None``/a scalar to every row, or validate a per-row list."""
        if values is None:
            return [None] * count
        if isinstance(values, (str, Path)):
            return [values] * count
        if isinstance(values, (list, tuple)):
            if len(values) == count:
                return list(values)
            if len(values) == 1:
                return [values[0]] * count
            raise ValueError(
                f"Breeze batch_generate got {len(values)} {name} for {count} texts."
            )
        # A single array-like reference clip shared by every row.
        return [values] * count

    @staticmethod
    def _prefill_mask(
        left_padding: list[int], length: int, offset: int = 0
    ) -> mx.array:
        """Causal mask that hides each row's left padding.

        The ``length`` queries sit after ``offset`` positions already in the
        cache.  Padded query positions may still attend to themselves so
        that no softmax row is fully masked (which would put NaNs into the
        padded key/value slots of later layers); real positions never see
        padding.
        """
        query = mx.arange(offset, offset + length)[:, None]
        key = mx.arange(offset + length)[None, :]
        pad = mx.array(left_padding, dtype=mx.int32)[:, None, None, None]
        visible = (query >= key) & (key >= pad)
        return visible | (query == key)

    @staticmethod
    def _combine_cfg(
        logits: mx.array,
        rows: int,
        uncond_index: Optional[mx.array],
        scale: Optional[mx.array],
    ) -> mx.array:
        """``uncond + scale * (cond - uncond)`` per row.

        ``logits`` holds the ``rows`` conditional rows followed by the extra
        unconditional rows.  Rows without CFG point ``uncond_index`` at
        themselves with scale 1, which returns their logits unchanged.
        """
        if uncond_index is None:
            return logits
        cond = logits[:rows]
        uncond = logits[uncond_index]
        return uncond + scale.astype(cond.dtype) * (cond - uncond)

    @staticmethod
    def _should_compact(rows: int, live: int) -> bool:
        """Whether to remove the finished rows from a ``rows``-row batch now.

        ``BatchKVCache.filter`` copies every layer's cache and costs about
        one decode step at ~96 rows, while a decode step costs about the
        same for any row count within 1-8, 9-64, 65-128 and 129+ rows.
        So compact when the live rows drop into a cheaper band, or when a
        quarter of the batch is dead weight.
        """

        def band(n: int) -> int:
            return 0 if n <= 8 else 1 if n <= 64 else 2 if n <= 128 else 3

        return band(live) < band(rows) or rows - live >= rows / 4

    def _depth_tokens_batch(
        self,
        first_codebook: mx.array,
        hidden: mx.array,
        *,
        rows: int,
        uncond_index: Optional[mx.array],
        expand_index: Optional[mx.array],
        scale: Optional[mx.array],
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> mx.array:
        """Complete one frame for every row; returns ``[rows, codebooks - 1]``.

        ``hidden`` contains the conditional rows and then the unconditional
        CFG rows; ``expand_index`` maps each of those back to the conditional
        row whose sampled token it consumes (the same token is fed to the
        conditional and unconditional depth passes, as in ``_depth_tokens``).
        """
        depth = self.depth_decoder
        cache = depth.model.make_cache()
        depth.start_frame(hidden, cache)
        token = first_codebook
        sampled: list[mx.array] = []
        for head_idx in range(self.num_codebooks - 1):
            inputs = token if expand_index is None else token[expand_index]
            logits = depth.step_logits(cache, head_idx=head_idx, token_id=inputs)
            logits = self._combine_cfg(logits, rows, uncond_index, scale)
            logits = self._mask_reserved_codec_logits(logits)
            token = self._sample_array(
                logits, temperature=temperature, top_p=top_p, top_k=top_k
            ).reshape(rows)
            sampled.append(token)
        if not sampled:
            return mx.zeros((rows, 0), dtype=mx.int32)
        return mx.stack(sampled, axis=1).astype(mx.int32)

    def _batch_prefill(
        self,
        texts: list[str],
        voices: list,
        instructs: list,
        ref_audios: list,
        ref_texts: list,
        use_cfg: list[bool],
    ) -> tuple[mx.array, list[BatchKVCache], list[int]]:
        """Left-pad every row's prompt and run one batched backbone prefill.

        Cache rows are the conditional rows in input order followed by one
        unconditional row per CFG row (``uncond_owner`` gives its input row).
        Returns the last hidden state of every cache row and the cache.

        A prompt is ``[reference prefix | target text]`` and every row of one
        voice has the same prefix, so the backbone runs each distinct prefix
        once and copies its keys/values into the rows.  Only the last
        ``block`` positions of every prompt (``block`` = longest target; a
        shorter target is topped up with the end of its prefix) go through
        the batched prefill on top of that.  The backbone is causal, so the
        copied keys/values are the ones the row would have computed itself.
        """
        # Encode each distinct reference clip once per call.
        encoded_refs: dict[Any, mx.array] = {}

        def reference_codes(ref_audio):
            if ref_audio is None:
                return None
            if isinstance(ref_audio, (list, tuple)):
                if len(ref_audio) != 1:
                    raise ValueError(
                        "Breeze supports exactly one reference audio item per row."
                    )
                ref_audio = ref_audio[0]
            key = (
                ("path", str(ref_audio))
                if isinstance(ref_audio, (str, Path))
                else ("object", id(ref_audio))
            )
            if key not in encoded_refs:
                encoded_refs[key] = self._encode_reference(ref_audio)
            return encoded_refs[key]

        # Cache rows are [conditional rows..., unconditional CFG rows...].
        cond_prompts: list[list[tuple[str, Any]]] = []
        uncond_prompts: list[list[tuple[str, Any]]] = []
        uncond_owner: list[int] = []
        for row, text in enumerate(texts):
            text = text.strip() or text
            if ref_audios[row] is not None and not ref_texts[row]:
                raise ValueError(
                    "Breeze voice cloning requires ref_text with ref_audio."
                )
            codes = reference_codes(ref_audios[row])
            prompt_kwargs = dict(
                voice=voices[row],
                ref_audio=ref_audios[row],
                ref_text=ref_texts[row],
                reference_codes=codes,
            )
            cond_prompts.append(
                self._prompt_segments(text, instruct=instructs[row], **prompt_kwargs)
            )
            if use_cfg[row]:
                uncond_prompts.append(
                    self._prompt_segments(text, instruct=None, **prompt_kwargs)
                )
                uncond_owner.append(row)
        # Encode every distinct text segment once, batched: rows that share
        # a reference transcript reuse its encoding.
        segments = cond_prompts + uncond_prompts
        text_embeddings = self._encode_texts(
            [value for prompt in segments for kind, value in prompt if kind == "text"]
        )
        # Split every prompt into its (shared) prefix and its target text.
        prefix_index: dict[tuple, int] = {}
        prefixes: list[mx.array] = []
        row_prefix: list[int] = []  # index into ``prefixes``; -1 = no prefix
        targets: list[mx.array] = []
        for prompt in segments:
            *head, (_, target) = prompt
            targets.append(text_embeddings[target])
            if not head:
                row_prefix.append(-1)
                continue
            key = tuple(
                value if kind == "text" else ("audio", id(value))
                for kind, value in head
            )
            if key not in prefix_index:
                prefix_index[key] = len(prefixes)
                prefixes.append(self._embed_segments(head, text_embeddings))
            row_prefix.append(prefix_index[key])
        del text_embeddings, cond_prompts, uncond_prompts, segments

        prefix_len = [int(p.shape[1]) for p in prefixes]
        totals = [
            (prefix_len[u] if u >= 0 else 0) + int(t.shape[1])
            for u, t in zip(row_prefix, targets)
        ]
        block = max(int(t.shape[1]) for t in targets)
        # ``kept``: prefix positions copied from the shared prefix cache;
        # ``block_pad``: left padding inside the block (prompt < block).
        kept = [max(0, total - block) for total in totals]
        block_pad = [max(0, block - total) for total in totals]
        shared = max(kept)
        left_padding = [shared - k + pad for k, pad in zip(kept, block_pad)]
        cache = self._shared_prefix_cache(prefixes, row_prefix, kept, block_pad, block)

        pieces = []
        for u, k, pad, target in zip(row_prefix, kept, block_pad, targets):
            parts = []
            if pad:
                parts.append(mx.zeros((1, pad, target.shape[-1]), dtype=target.dtype))
            if u >= 0 and prefix_len[u] > k:
                parts.append(prefixes[u][:, k:])
            parts.append(target)
            pieces.append(mx.concatenate(parts, axis=1))
        embeddings = mx.concatenate(pieces, axis=0)
        del pieces, prefixes, targets

        hidden = self.backbone_model(
            input_embeddings=embeddings,
            cache=cache,
            mask=self._prefill_mask(left_padding, block, offset=shared),
        )[:, -1, :]
        return hidden, cache, uncond_owner

    def _shared_prefix_cache(
        self,
        prefixes: list[mx.array],
        row_prefix: list[int],
        kept: list[int],
        block_pad: list[int],
        block: int,
    ) -> list[BatchKVCache]:
        """Backbone cache holding each row's first ``kept`` prefix positions.

        Each distinct prefix runs through the backbone once (only as far as
        its rows need), then its keys/values are gathered into the rows,
        right-aligned at ``shared = max(kept)`` with the rest left-padded.
        The buffers are sized for the following ``block`` positions, as a
        prefill of the whole prompt would have allocated them.
        """
        rows = len(row_prefix)
        shared = max(kept)
        left_padding = [shared - k + pad for k, pad in zip(kept, block_pad)]
        if shared == 0:
            return self.backbone_model.make_batch_cache(left_padding)

        need = [0] * len(prefixes)
        for u, k in zip(row_prefix, kept):
            if u >= 0:
                need[u] = max(need[u], k)
        used = [u for u in range(len(prefixes)) if need[u]]
        slot = {u: j for j, u in enumerate(used)}
        width = max(need)
        pad = [width - need[u] for u in used]
        prefix_cache = self.backbone_model.make_batch_cache(pad)
        self.backbone_model(
            input_embeddings=mx.concatenate(
                [
                    mx.pad(prefixes[u][:, : need[u]], [(0, 0), (p, 0), (0, 0)])
                    for u, p in zip(used, pad)
                ],
                axis=0,
            ),
            cache=prefix_cache,
            mask=self._prefill_mask(pad, width),
        )

        # Row r, slot t < shared holds prefix position t - (shared - kept[r])
        # of its voice; earlier slots are padding (masked; any finite value).
        position = mx.arange(shared)[None, :]
        start = mx.array([shared - k for k in kept])[:, None]
        source_row = mx.array(
            [slot.get(u, 0) if k else 0 for u, k in zip(row_prefix, kept)]
        )[:, None]
        source_start = mx.array(
            [width - need[u] if k else 0 for u, k in zip(row_prefix, kept)]
        )[:, None]
        flat = mx.where(
            position >= start,
            source_row * width + source_start + position - start,
            0,
        ).reshape(-1)
        step = BACKBONE_CACHE_STEP
        capacity = -(-(shared + block) // step) * step

        def gather(states: mx.array) -> mx.array:
            # [prefixes, heads, width, dim] -> [rows, heads, capacity, dim]
            _, heads, _, dim = states.shape
            states = states.transpose(0, 2, 1, 3).reshape(-1, heads, dim)
            states = states[flat].reshape(rows, shared, heads, dim)
            states = states.transpose(0, 2, 1, 3)
            return mx.pad(states, [(0, 0), (0, 0), (0, capacity - shared), (0, 0)])

        cache = []
        for layer_cache in prefix_cache:
            keys, values, _, _ = layer_cache.state
            row_cache = BatchKVCache(left_padding)
            row_cache.step = step
            row_cache.keys = gather(keys)
            row_cache.values = gather(values)
            row_cache.offset = mx.array([k - p for k, p in zip(kept, block_pad)])
            row_cache._idx = shared
            cache.append(row_cache)
        return cache

    def batch_generate(
        self,
        texts: list[str],
        voices: Optional[Union[str, list[Optional[str]]]] = None,
        instructs: Optional[Union[str, list[Optional[str]]]] = None,
        ref_audios: Optional[Any] = None,
        ref_texts: Optional[Union[str, list[Optional[str]]]] = None,
        cfg_scale: Optional[float] = None,
        max_tokens: Union[int, list[int]] = 750,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
        stream: bool = False,
        verbose: bool = False,
        sync_steps: int = 4,
        **_: object,
    ) -> Generator[BatchGenerationResult, None, None]:
        """Generate several independent utterances in one batched decode.

        Every row is the same as one ``generate`` call with
        ``split_pattern=None`` (the text is only stripped): no newline or
        600-character splitting is done here, so callers should send one
        line per row.  Rows may differ in text, voice tag, instruction and
        reference clip.  A row uses classifier-free guidance iff it has an
        instruction and ``cfg_scale`` is not ``None``/``1.0``; only those
        rows get an extra unconditional backbone/depth row.

        Prompts are left-padded to a common length.  A per-row causal mask
        hides the padding and ``BatchKVCache`` gives each row its own RoPE
        positions, so a padded row sees exactly what it would see alone
        (results can still differ in the last bits because batched kernels
        reduce in a different order, which greedy decoding can amplify).
        Each row stops at its own EOS or after ``max_tokens`` frames
        (an int, or one cap per row).  Finished rows stop producing frames
        and are removed from the batch only when that pays off (see
        ``_should_compact``).  Sampled frames are read back to the host every
        ``sync_steps`` steps so graph construction overlaps GPU work; rows
        that finished inside a window are trimmed, so the window only costs
        a few wasted steps, never different output.  Results are yielded in
        row order after the whole batch has finished.  With the same
        ``seed`` a batch is reproducible, but sampled rows do not reproduce
        the random stream of separate ``generate`` calls.
        """
        if stream:
            raise ValueError("Breeze batch_generate does not support streaming.")
        if top_k < 0:
            raise ValueError("top_k must be non-negative.")
        if temperature < 0:
            raise ValueError("temperature must be non-negative.")
        if not 0 <= top_p <= 1:
            raise ValueError("top_p must be between 0 and 1.")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive.")
        if cfg_scale is not None and not math.isfinite(cfg_scale):
            raise ValueError("cfg_scale must be finite.")
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        count = len(texts)
        if count == 0:
            return
        if seed is not None:
            mx.random.seed(seed)

        if isinstance(max_tokens, (list, tuple)):
            token_caps = [
                int(cap) for cap in self._per_row(max_tokens, count, "max_tokens")
            ]
        else:
            token_caps = [int(max_tokens)] * count
        if min(token_caps) <= 0:
            raise ValueError("max_tokens must be positive.")
        voices = self._per_row(voices, count, "voices")
        instructs = self._per_row(instructs, count, "instructs")
        ref_texts = self._per_row(ref_texts, count, "ref_texts")
        ref_audios = self._per_row(ref_audios, count, "ref_audios")

        started = time.perf_counter()
        use_cfg = [bool(ins) and cfg_scale not in (None, 1.0) for ins in instructs]
        scale_value = 1.0 if cfg_scale is None else float(cfg_scale)

        hidden, cache, uncond_owner = self._batch_prefill(
            texts, voices, instructs, ref_audios, ref_texts, use_cfg
        )

        # ``active`` lists the original row of each conditional cache row;
        # ``active_uncond`` the original row of each unconditional cache row.
        active = list(range(count))
        active_uncond = list(uncond_owner)
        frames: list[list[list[int]]] = [[] for _ in range(count)]
        valid = self.vocab_size + 1
        penalize = repetition_penalty != 1.0
        seen = mx.zeros((count, valid), dtype=mx.bool_) if penalize else None
        vocab_ids = mx.arange(valid)[None, :]

        def cfg_indices():
            if not active_uncond:
                return None, None, None
            rows = len(active)
            position = {row: i for i, row in enumerate(active)}
            uncond_index = list(range(rows))
            expand_index = list(range(rows))
            for j, row in enumerate(active_uncond):
                uncond_index[position[row]] = rows + j
                expand_index.append(position[row])
            scale = [scale_value if use_cfg[row] else 1.0 for row in active]
            return (
                mx.array(uncond_index, dtype=mx.int32),
                mx.array(expand_index, dtype=mx.int32),
                mx.array(scale)[:, None],
            )

        uncond_index, expand_index, scale = cfg_indices()
        done = [False] * count  # per batch position of ``active``
        sync_steps = max(1, int(sync_steps))
        pending_frames: list[mx.array] = []
        pending_eos: list[mx.array] = []
        steps = 0
        step_limit = max(token_caps)
        while active:
            rows = len(active)
            logits = self._combine_cfg(self.lm_head(hidden), rows, uncond_index, scale)
            if penalize:
                penalized = mx.where(
                    logits < 0,
                    logits * repetition_penalty,
                    logits / repetition_penalty,
                )
                logits = mx.where(seen, penalized, logits)
            logits = self._mask_reserved_codec_logits(logits)
            first = self._sample_array(
                logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                allow_eos=True,
            ).reshape(rows)
            is_eos = first == self.vocab_size
            first_safe = mx.where(is_eos, 0, first).astype(mx.int32)
            rest = self._depth_tokens_batch(
                first_safe,
                hidden,
                rows=rows,
                uncond_index=uncond_index,
                expand_index=expand_index,
                scale=scale,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            frame = mx.concatenate([first_safe[:, None], rest], axis=1)
            if penalize:
                seen = seen | (vocab_ids == first[:, None])
            inputs = frame if expand_index is None else frame[expand_index]
            hidden = self.backbone_model(input_ids=inputs[:, None, :], cache=cache)[
                :, -1, :
            ]
            mx.async_eval(hidden, frame, is_eos, *([seen] if penalize else []))
            pending_frames.append(frame)
            pending_eos.append(is_eos)
            steps += 1
            # Read frames back only every ``sync_steps`` steps: without a
            # blocking host sync, Python builds the next step's graph while
            # the GPU is still running this one.  A row that hit EOS inside
            # the window keeps decoding (ignored) until the next sync.
            if len(pending_frames) < sync_steps and steps < step_limit:
                continue
            block_frames = mx.stack(pending_frames, axis=1).tolist()
            block_eos = mx.stack(pending_eos, axis=1).tolist()
            pending_frames, pending_eos = [], []

            for i, row in enumerate(active):
                if done[i]:
                    continue
                for frame_row, eos in zip(block_frames[i], block_eos[i]):
                    if eos:
                        done[i] = True
                        break
                    frames[row].append(frame_row)
                    if len(frames[row]) >= token_caps[row]:
                        done[i] = True
                        break
            keep = [i for i in range(rows) if not done[i]]
            if not keep:
                break
            if len(keep) == rows:
                continue
            step_limit = max(token_caps[active[i]] for i in keep)
            # Finished rows keep decoding (ignored) until removing them pays
            # for the cache copy; then drop them and their unconditional
            # twins so they no longer cost backbone or depth compute.
            if not self._should_compact(rows, len(keep)):
                continue
            kept_rows = {active[i] for i in keep}
            keep_uncond = [
                rows + j for j, row in enumerate(active_uncond) if row in kept_rows
            ]
            cache_keep = mx.array(keep + keep_uncond, dtype=mx.int32)
            for layer_cache in cache:
                layer_cache.filter(cache_keep)
            hidden = hidden[cache_keep]
            if penalize:
                seen = seen[mx.array(keep, dtype=mx.int32)]
            active = [active[i] for i in keep]
            active_uncond = [row for row in active_uncond if row in kept_rows]
            done = [False] * len(active)
            uncond_index, expand_index, scale = cfg_indices()
            if verbose:
                print(f"[breeze batch] step {steps}: {len(active)}/{count} rows active")

        del cache, hidden, seen
        elapsed = time.perf_counter() - started
        if verbose:
            total = sum(len(f) for f in frames)
            print(
                f"[breeze batch] {count} rows, {steps} steps, {total} frames "
                f"in {elapsed:.2f}s"
            )
        for row in range(count):
            if frames[row]:
                codes = mx.array(frames[row], dtype=mx.int32)[None, :, :]
                audio = self._decode_codes(codes)
            else:
                audio = self._empty_audio()
            mx.eval(audio)
            samples = int(audio.shape[0])
            token_count = len(frames[row])
            frames[row] = []
            yield BatchGenerationResult(
                audio=audio,
                sequence_idx=row,
                samples=samples,
                sample_rate=self.sample_rate,
                token_count=token_count,
                audio_duration=f"00:00:{samples / self.sample_rate:06.3f}",
                processing_time_seconds=time.perf_counter() - started,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
            )
        mx.clear_cache()
