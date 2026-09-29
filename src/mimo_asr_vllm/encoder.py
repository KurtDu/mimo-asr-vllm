"""MiMo-V2.5-ASR audio path: waveform -> RVQ codes -> LLM input embeddings.

This is a compact re-implementation of the parts of the official code that are
needed for ASR (the MiMo-Audio-Tokenizer *encoder* and the MiMo
``input_local_transformer``). It is written for batched inference:

* All 30 s chunks of all audios in a batch are packed into one encoder call
  (variable-length flash attention).
* Padding is handled so that every chunk is encoded exactly as if it were
  processed alone, which is what the official implementation does.

The numerics (bf16 activations, fp32 RVQ distances, bf16 RoPE tables, bf16
accumulation order of the speech embeddings) follow the reference code.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from . import audio as A

logger = logging.getLogger("vllm.plugins.mimo_asr")  # inherits vLLM log handlers


# --------------------------------------------------------------------------
# Attention helpers
# --------------------------------------------------------------------------

@functools.cache
def _flash_attn_varlen(head_size: int, heads: int, dtype: torch.dtype, device: torch.device):
    """vLLM's bundled FlashAttention if it works on this GPU, else ``None``.

    The kernel is probed once: FlashAttention needs sm80+ and fp16/bf16, and a
    driver that is too old for the wheel's CUDA toolkit fails at launch time.
    """
    if device.type != "cuda" or dtype not in (torch.float16, torch.bfloat16):
        return None
    if os.environ.get("MIMO_ASR_DISABLE_FLASH_ATTN", "0") not in ("", "0"):
        return None  # troubleshooting switch: force the PyTorch SDPA path
    if torch.cuda.get_device_capability(device) < (8, 0):
        return None
    try:
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        version = get_flash_attn_version(head_size=head_size)
        if version is None:
            return None
        fn = functools.partial(flash_attn_varlen_func, fa_version=version)
        x = torch.randn(8, heads, head_size, device=device, dtype=dtype)
        cu = torch.tensor([0, 3, 8], device=device, dtype=torch.int32)
        fn(x, x, x, max_seqlen_q=5, cu_seqlens_q=cu, max_seqlen_k=5, cu_seqlens_k=cu, causal=False)
        torch.cuda.synchronize(device)
        return fn
    except Exception as exc:  # ImportError on non-CUDA builds, launch errors, ...
        logger.warning("FlashAttention unavailable for the MiMo audio encoder (%s); "
                       "falling back to PyTorch SDPA.", str(exc).splitlines()[0][:200])
        return None


def _varlen_attention(q, k, v, cu_seqlens, max_seqlen, lengths):
    """Bidirectional attention over packed sequences. q/k/v: [T, H, D]."""
    fa = _flash_attn_varlen(q.shape[-1], q.shape[1], q.dtype, q.device)
    if fa is not None:
        return fa(q, k, v, max_seqlen_q=max_seqlen, cu_seqlens_q=cu_seqlens,
                  max_seqlen_k=max_seqlen, cu_seqlens_k=cu_seqlens, causal=False)
    out = torch.empty_like(q)
    start = 0
    for n in lengths:
        sl = slice(start, start + n)
        o = F.scaled_dot_product_attention(
            q[sl].transpose(0, 1), k[sl].transpose(0, 1), v[sl].transpose(0, 1))
        out[sl] = o.transpose(0, 1)
        start += n
    return out


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(inv_freq, positions, dtype):
    # fp32 table, cast to the activation dtype (HF / MiMo convention).
    freqs = positions[:, None].float() * inv_freq[None, :].float()
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _inv_freq(base: float, dim: int):
    # Computed on CPU in fp32 like the reference; moved with the module later.
    arange = torch.arange(0, dim, 2, dtype=torch.int64, device="cpu").float()
    return 1.0 / (base ** (arange / dim))


# --------------------------------------------------------------------------
# MiMo-Audio-Tokenizer encoder (+ RVQ)
# --------------------------------------------------------------------------

class _EncoderAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, cos, sin, cu_seqlens, max_seqlen, lengths):
        t = x.shape[0]
        q = self.q_proj(x).view(t, self.heads, self.head_dim)
        k = self.k_proj(x).view(t, self.heads, self.head_dim)
        v = self.v_proj(x).view(t, self.heads, self.head_dim)
        cos, sin = cos[:, None], sin[:, None]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        o = _varlen_attention(q, k, v, cu_seqlens, max_seqlen, lengths)
        return self.out_proj(o.reshape(t, -1))


class _EncoderLayer(nn.Module):
    def __init__(self, dim, heads, ffn):
        super().__init__()
        self.self_attn = _EncoderAttention(dim, heads)
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffn)
        self.fc2 = nn.Linear(ffn, dim)
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(self, x, *attn_args):
        x = x + self.self_attn(self.self_attn_layer_norm(x), *attn_args)
        x = x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))
        # The reference clamps +-inf only when present; an unconditional clamp is
        # a no-op for finite values and avoids a host sync per layer.
        limit = torch.finfo(x.dtype).max - 1000
        return x.clamp(min=-limit, max=limit)


class _Codebook(nn.Module):
    def __init__(self, size, dim):
        super().__init__()
        # Stored in fp32: the reference computes RVQ distances in fp32.
        self.register_buffer("embed", torch.empty(size, dim, dtype=torch.float32))


class MiMoAudioEncoder(nn.Module):
    """MiMo-Audio-Tokenizer encoder: log-mel -> first ``n_q`` RVQ code streams."""

    EXPECTED = {  # front-end constants hard-coded in audio.py
        "sampling_rate": A.SAMPLE_RATE, "nfft": A.N_FFT, "hop_length": A.HOP_LENGTH,
        "window_size": A.WIN_LENGTH, "n_mels": A.N_MELS, "kernel_size": 3,
        "stride_size": 2, "avg_pooler": 2, "encoder_causal": False,
        "position_embedding_type": "rope", "ln_type": "LayerNorm",
        "activation_function": "gelu", "scale_embedding": False,
    }

    def __init__(self, cfg: dict, n_q: int):
        super().__init__()
        for key, value in self.EXPECTED.items():
            if cfg.get(key, value) != value:
                raise ValueError(f"Unsupported MiMo-Audio-Tokenizer config: {key}={cfg.get(key)!r}")
        if list(cfg.get("encoder_attn_window_size", [-1, -1])) != [-1, -1]:
            raise ValueError("Unsupported MiMo-Audio-Tokenizer config: windowed encoder attention")
        d, heads = cfg["d_model"], cfg["encoder_attention_heads"]
        self.d_model, self.heads, self.n_q = d, heads, n_q
        self.skip_layer_idx = cfg.get("encoder_skip_layer_id")
        self.conv1 = nn.Conv1d(A.N_MELS, d, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1)
        self.layers = nn.ModuleList(
            [_EncoderLayer(d, heads, cfg["encoder_ffn_dim"]) for _ in range(cfg["encoder_layers"])])
        self.layer_norm = nn.LayerNorm(d)
        self.down_sample_layer = nn.Sequential(nn.Conv1d(d, d, 2, 2, bias=False), nn.GELU())
        self.down_sample_norm = nn.LayerNorm(d)
        sizes = cfg["codebook_size"]
        sizes = sizes if isinstance(sizes, list) else [sizes] * cfg["num_quantizers"]
        self.codebooks = nn.ModuleList([_Codebook(sizes[i], d) for i in range(n_q)])
        # The reference casts the whole tokenizer with `.bfloat16()`, which also
        # rounds this buffer to bf16 before it is used in fp32.
        inv_freq = _inv_freq(cfg.get("rope_theta", 10000), d // heads).bfloat16().float()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def load_config(path: Path) -> dict:
        return json.loads((Path(path) / "config.json").read_text())

    def checkpoint_mapping(self) -> dict[str, str]:
        """checkpoint key -> module key."""
        mapping = {}
        for key in self.state_dict():
            if key.startswith("codebooks."):
                i = key.split(".")[1]
                mapping[f"encoder.quantizer.vq.layers.{i}._codebook.embed"] = key
            else:
                mapping["encoder." + key] = key
        return mapping

    @torch.no_grad()
    def forward(self, mels: list[torch.Tensor]) -> list[torch.Tensor]:
        """mels: list of [frames, n_mels] -> list of int codes [codec_frames, n_q]."""
        dtype = self.conv1.weight.dtype
        device = self.conv1.weight.device
        mel_lens = [m.shape[0] for m in mels]
        enc_lens = [(n - 1) // 2 + 1 for n in mel_lens]
        batch = len(mels)
        x = torch.zeros(batch, A.N_MELS, max(mel_lens), device=device, dtype=dtype)
        for i, m in enumerate(mels):
            x[i, :, : m.shape[0]] = m.T.to(dtype)
        x = F.gelu(self.conv1(x))
        # Zero conv1 outputs past each sequence end so conv2 sees the same zero
        # padding it would see if the chunk were processed alone.
        if batch > 1:
            pos = torch.arange(x.shape[-1], device=device)
            x = x * (pos[None, :] < torch.tensor(mel_lens, device=device)[:, None])[:, None, :]
        x = F.gelu(self.conv2(x)).transpose(1, 2)  # [B, T_enc, D]
        valid = torch.arange(x.shape[1], device=device)[None, :] < torch.tensor(enc_lens, device=device)[:, None]
        h = x[valid]  # packed [sum(enc_lens), D]

        positions = torch.cat([torch.arange(n, device=device) for n in enc_lens])
        cos, sin = _rope_cos_sin(self.inv_freq, positions, dtype)
        cu = F.pad(torch.tensor(enc_lens, device=device).cumsum(0), (1, 0)).to(torch.int32)
        attn_args = (cos, sin, cu, max(enc_lens), enc_lens)
        skip = None
        for idx, layer in enumerate(self.layers):
            h = layer(h, *attn_args)
            if self.skip_layer_idx is not None and idx == self.skip_layer_idx - 1:
                skip = h.clone()
        if skip is not None:
            h = h + skip
        h = self.layer_norm(h)

        # Average-pool by 2 (strided conv) with zero padding of odd lengths.
        pooled_lens = [(n + 1) // 2 for n in enc_lens]
        t_pad = 2 * max(pooled_lens)
        grid = h.new_zeros(batch, t_pad, self.d_model)
        grid[F.pad(valid, (0, t_pad - valid.shape[1]))] = h
        y = self.down_sample_layer(grid.transpose(1, 2)).transpose(1, 2)  # [B, T/2, D]
        pvalid = torch.arange(y.shape[1], device=device)[None, :] < torch.tensor(pooled_lens, device=device)[:, None]
        y = self.down_sample_norm(y[pvalid])

        codes = self._quantize(y.float())
        return list(codes.split(pooled_lens))

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        prev = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")  # the reference uses exact fp32
        try:
            residual, out = x, []
            for cb in self.codebooks:
                embed = cb.embed
                dist = -(residual.pow(2).sum(1, keepdim=True) - 2 * residual @ embed.t()
                         + embed.t().pow(2).sum(0, keepdim=True))
                idx = dist.max(dim=-1).indices
                residual = residual - F.embedding(idx, embed)
                out.append(idx)
            return torch.stack(out, dim=1)
        finally:
            torch.set_float32_matmul_precision(prev)


# --------------------------------------------------------------------------
# MiMo input local transformer (codes -> one LLM embedding per 4-frame group)
# --------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class _LocalAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        shape = (b, t, self.heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        # Full (bidirectional) attention inside each 4-frame group. For such tiny
        # sequences an explicit fp32 softmax is ~4x faster than SDPA kernels.
        scores = (q.float() @ k.float().transpose(-1, -2)) * self.head_dim ** -0.5
        o = (scores.softmax(dim=-1) @ v.float()).to(v.dtype)
        return self.o_proj(o.transpose(1, 2).reshape(b, t, -1))


class _LocalMLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _LocalLayer(nn.Module):
    def __init__(self, dim, heads, hidden, eps):
        super().__init__()
        self.self_attn = _LocalAttention(dim, heads)
        self.mlp = _LocalMLP(dim, hidden)
        self.input_layernorm = _RMSNorm(dim, eps)
        self.post_attention_layernorm = _RMSNorm(dim, eps)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class _InputLocalTransformer(nn.Module):
    def __init__(self, dim, layers, heads, hidden, eps, rope_theta):
        super().__init__()
        self.layers = nn.ModuleList([_LocalLayer(dim, heads, hidden, eps) for _ in range(layers)])
        self.norm = _RMSNorm(dim, eps)
        self.register_buffer("inv_freq", _inv_freq(rope_theta, dim // heads), persistent=False)

    def forward(self, x):
        cos, sin = _rope_cos_sin(self.inv_freq, torch.arange(x.shape[1], device=x.device), x.dtype)
        cos, sin = cos[None, None], sin[None, None]
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)


def _parse_list(value, n):
    if isinstance(value, str) and "-" in value:
        return [int(v) for v in value.split("-")]
    return [int(value)] * n


class MiMoAudioAdapter(nn.Module):
    """RVQ codes -> speech embeddings -> input local transformer -> LLM space."""

    def __init__(self, config):
        super().__init__()
        self.group_size = config.group_size
        self.channels = config.audio_channels
        if self.group_size != A.GROUP_SIZE:
            raise ValueError(f"Unsupported group_size={self.group_size}")
        if not getattr(config, "input_full_attention", True):
            raise ValueError("Only input_full_attention=True checkpoints are supported")
        dim = config.input_local_dim
        heads = config.local_attn_heads
        sizes = _parse_list(config.speech_vocab_size, self.channels)
        self.empty_ids = _parse_list(config.speech_zeroemb_idx, self.channels)
        self.speech_embeddings = nn.ModuleList(
            [nn.Embedding(sizes[i], dim, padding_idx=self.empty_ids[i]) for i in range(self.channels)])
        self.input_local_transformer = _InputLocalTransformer(
            dim, config.input_local_layers, heads, dim * 4, config.rms_norm_eps,
            _rope_theta(config))
        self.speech_group_downcast = nn.Linear(dim * self.group_size, config.hidden_size, bias=False)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: [groups, group_size, channels] -> [groups, hidden_size]."""
        dtype = self.speech_group_downcast.weight.dtype
        emb = torch.zeros(*codes.shape[:2], self.speech_embeddings[0].embedding_dim,
                          dtype=dtype, device=codes.device)
        for i, table in enumerate(self.speech_embeddings):  # same order/dtype as reference
            ids = codes[:, :, i]
            cur = table(ids)
            cur = cur.masked_fill((ids == self.empty_ids[i]).unsqueeze(-1), 0.0)
            emb = emb + cur
        out = self.input_local_transformer(emb)
        return self.speech_group_downcast(out.reshape(out.shape[0], -1))


def _rope_theta(config) -> float:
    params = getattr(config, "rope_parameters", None) or {}
    return float(params.get("rope_theta", getattr(config, "rope_theta", 640000)))


def group_codes(codes: torch.Tensor, group_size: int = A.GROUP_SIZE) -> torch.Tensor:
    """[frames, channels] -> [groups, group_size, channels]; pad by repeating the last frame."""
    pad = (-codes.shape[0]) % group_size
    if pad:
        codes = torch.cat([codes, codes[-1:].expand(pad, -1)])
    return codes.reshape(-1, group_size, codes.shape[-1])


__all__ = ["MiMoAudioEncoder", "MiMoAudioAdapter", "group_codes"]
