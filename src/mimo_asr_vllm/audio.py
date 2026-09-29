"""Audio front-end shared by the vLLM processor, the model and the Python API.

Everything here mirrors the official MiMo-V2.5-ASR implementation
(``MimoAudio.preprocess_input``) so that the audio codes fed to the LLM are the
same as with the reference HuggingFace code.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# MiMo-Audio-Tokenizer (v1) front-end constants. They are validated against the
# tokenizer config when the weights are loaded (see encoder.py).
SAMPLE_RATE = 24_000
N_FFT = 960
HOP_LENGTH = 240
WIN_LENGTH = 960
N_MELS = 128
CHUNK_SECONDS = 30  # waveform is tokenized in independent 30 s chunks
GROUP_SIZE = 4  # 4 codec frames (25 Hz) -> 1 LLM position (6.25 Hz)
# Longer inputs are split at low-energy points and decoded as parallel clips
# (override with MIMO_ASR_MAX_CLIP_S; "none" keeps whole files when they fit).
DEFAULT_MAX_CLIP_S = 60.0


def max_clip_seconds() -> float | None:
    """``MIMO_ASR_MAX_CLIP_S`` (``none``/``0`` disables splitting) or the default."""
    import os

    value = os.environ.get("MIMO_ASR_MAX_CLIP_S", "").strip().lower()
    if value == "":
        return DEFAULT_MAX_CLIP_S
    return None if value in ("none", "0", "off") else float(value)


def chunk_bounds(num_samples: int) -> list[tuple[int, int]]:
    """Split a waveform into 30 s chunks exactly like the official code.

    A trailing piece shorter than ``N_FFT`` is merged into the previous chunk
    (it would break reflect padding of the STFT).
    """
    chunk = CHUNK_SECONDS * SAMPLE_RATE
    bounds, start = [], 0
    while start < num_samples:
        end = min(start + chunk, num_samples)
        if 0 < num_samples - end < N_FFT:
            end = num_samples
        bounds.append((start, end))
        start = end
    return bounds


def num_mel_frames(num_samples: int) -> int:
    # Very short inputs are zero padded to N_FFT samples (center=True STFT).
    return max(num_samples, N_FFT) // HOP_LENGTH + 1


def num_codec_frames(num_mel: int) -> int:
    enc = (num_mel - 1) // 2 + 1  # conv2: kernel 3, stride 2, padding 1
    return (enc + 1) // 2  # avg pooler 2, last odd frame padded


def num_audio_groups(num_samples: int) -> int:
    """Number of LLM placeholder positions produced for a 24 kHz waveform."""
    if num_samples <= 0:
        raise ValueError("audio must contain at least one sample")
    frames = sum(num_codec_frames(num_mel_frames(e - s)) for s, e in chunk_bounds(num_samples))
    return math.ceil(frames / GROUP_SIZE)


def max_samples_for_groups(num_groups: int) -> int:
    """Largest waveform length (24 kHz samples) that yields <= num_groups."""
    # 6.25 groups per second; search down from the analytic upper bound.
    n = int(num_groups * GROUP_SIZE * 2 * 2 * HOP_LENGTH)
    while n > 0 and num_audio_groups(n) > num_groups:
        n -= HOP_LENGTH
    return n


def split_on_pauses(wav: np.ndarray, sample_rate: int, max_clip_s: float) -> list[np.ndarray]:
    """Split long audio into clips of at most ``max_clip_s`` at the quietest pause.

    For every clip the cut is placed at the lowest-energy 200 ms window in the
    second half of the allowed span, so cuts land between sentences instead of
    inside words (vLLM's default only searches the last second). The last clip
    is at least 1 s long.
    """
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    max_len = int(max_clip_s * sample_rate)
    if max_len <= 0 or len(wav) <= max_len:
        return [wav]
    hop = max(1, sample_rate // 50)  # 20 ms frames
    n = len(wav) // hop
    energy = np.square(wav[: n * hop].astype(np.float64)).reshape(n, hop).mean(axis=1)
    win = 10  # 200 ms smoothing
    csum = np.concatenate([[0.0], np.cumsum(energy)])
    smooth = (csum[win:] - csum[:-win]) / win  # smooth[i] = mean(energy[i:i+win])
    clips, start = [], 0
    while len(wav) - start > max_len:
        lo = (start + max_len // 2) // hop
        hi = (min(start + max_len, len(wav) - sample_rate) // hop) - win
        if hi <= lo:
            cut = start + max_len
        else:
            seg = smooth[lo:hi + 1]
            best = lo + (len(seg) - 1 - int(np.argmin(seg[::-1])))  # latest minimum
            cut = (best + win // 2) * hop
        clips.append(wav[start:cut])
        start = cut
    clips.append(wav[start:])
    return clips


def to_mono_float32(wav: np.ndarray | torch.Tensor, channels_first: bool) -> torch.Tensor:
    """Float32 mono waveform; integer PCM is scaled to [-1, 1]."""
    if not isinstance(wav, torch.Tensor):
        wav = np.asarray(wav)
        if np.issubdtype(wav.dtype, np.integer):
            wav = wav.astype(np.float32) / float(np.iinfo(wav.dtype).max + 1)
        wav = torch.from_numpy(np.ascontiguousarray(wav))
    elif not wav.is_floating_point():
        wav = wav.to(torch.float32) / float(torch.iinfo(wav.dtype).max + 1)
    wav = wav.to(torch.float32)
    if wav.ndim == 2:
        wav = wav.mean(dim=0 if channels_first else 1)
    if wav.ndim != 1:
        raise ValueError(f"expected mono or 2-D audio, got shape {tuple(wav.shape)}")
    return wav


def resample(wav: torch.Tensor, orig_sr: int) -> torch.Tensor:
    """Resample to 24 kHz with torchaudio's sinc resampler (as the official code)."""
    if int(orig_sr) == SAMPLE_RATE:
        return wav
    import torchaudio

    return torchaudio.functional.resample(wav, int(orig_sr), SAMPLE_RATE)


def load_audio(source, sample_rate: int | None = None) -> np.ndarray:
    """Load a file path / file-like object / array into a 24 kHz mono float32 array.

    * path or file-like: decoded with soundfile (falls back to PyAV via vLLM for
      containers soundfile cannot read, e.g. mp3/m4a/webm).
    * ``(array, sample_rate)`` tuple or ``array`` + ``sample_rate``: arrays are
      ``[samples]`` or ``[samples, channels]`` (soundfile layout).
    """
    if isinstance(source, tuple):
        source, sample_rate = source
    if isinstance(source, (np.ndarray, torch.Tensor, list)):
        if sample_rate is None:
            raise ValueError("sample_rate is required when passing a raw waveform")
        wav = to_mono_float32(source, channels_first=False)
        sr = int(sample_rate)
    else:
        wav, sr = _decode_file(source)
    if wav.numel() == 0:
        raise ValueError("audio is empty")
    if not torch.isfinite(wav).all():
        raise ValueError("audio contains NaN or Inf")
    return resample(wav, sr).numpy()


def _decode_file(source) -> tuple[torch.Tensor, int]:
    import soundfile as sf
    from pathlib import Path

    if isinstance(source, (str, Path)) and not Path(source).is_file():
        raise FileNotFoundError(f"No such audio file: {source}")
    try:
        data, sr = sf.read(source, dtype="float32", always_2d=True)  # [samples, channels]
        return to_mono_float32(data, channels_first=False), int(sr)
    except sf.LibsndfileError:
        if hasattr(source, "seek"):
            source.seek(0)
        # Formats libsndfile cannot decode (mp3 on old libsndfile, m4a, webm...).
        from vllm.multimodal.media.audio import load_audio as vllm_load_audio

        data, sr = vllm_load_audio(source, sr=None, mono=True)
        return to_mono_float32(data, channels_first=True), int(sr)


class LogMelSpectrogram(torch.nn.Module):
    """``log(clamp(MelSpectrogram(power=1), 1e-7))`` as in the official code."""

    def __init__(self):
        super().__init__()
        import torchaudio

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH, f_min=0, f_max=None, n_mels=N_MELS,
            power=1.0, center=True,
        )

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """[samples] float32 -> [frames, n_mels] float32."""
        if wav.numel() < N_FFT:
            wav = torch.nn.functional.pad(wav, (0, N_FFT - wav.numel()))
        spec = self.mel(wav[None].float())
        return torch.log(torch.clip(spec, min=1e-7)).squeeze(0).transpose(0, 1)
