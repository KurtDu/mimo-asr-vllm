"""CPU-only tests of the audio front-end and prompt helpers."""

import numpy as np
import pytest
import torch

from mimo_asr_vllm import audio as A
from mimo_asr_vllm.prompt import clean_transcript, normalize_language, split_language_tag

# (24 kHz samples, audio groups) measured with the official implementation.
OFFICIAL_GROUPS = [(223488, 59), (138880, 37), (59768, 16), (76568, 20), (73561, 20),
                   (40125, 11), (2146696, 560), (2124411, 554)]


@pytest.mark.parametrize("samples,groups", OFFICIAL_GROUPS)
def test_num_groups_matches_official(samples, groups):
    assert A.num_audio_groups(samples) == groups


@pytest.mark.parametrize("n", [1, 100, 959, 960, 961, 1199, 1200, 23_999, 24_000, 240_241])
def test_mel_frames_formula(n):
    mel = A.LogMelSpectrogram()(torch.randn(n))
    assert mel.shape == (A.num_mel_frames(n), A.N_MELS)
    assert torch.isfinite(mel).all()


def test_chunking_rules():
    chunk = A.CHUNK_SECONDS * A.SAMPLE_RATE
    assert A.chunk_bounds(chunk) == [(0, chunk)]
    # A tail shorter than n_fft is merged into the previous chunk.
    assert A.chunk_bounds(chunk + A.N_FFT - 1) == [(0, chunk + A.N_FFT - 1)]
    assert A.chunk_bounds(chunk + A.N_FFT) == [(0, chunk), (chunk, chunk + A.N_FFT)]
    bounds = A.chunk_bounds(3 * chunk + 12345)
    assert bounds[0] == (0, chunk) and bounds[-1][1] == 3 * chunk + 12345
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))


@pytest.mark.parametrize("groups", [1, 7, 100, 4000])
def test_max_samples_for_groups(groups):
    n = A.max_samples_for_groups(groups)
    assert A.num_audio_groups(n) <= groups
    assert A.num_audio_groups(n + 4 * A.GROUP_SIZE * A.HOP_LENGTH) > groups


def test_load_audio_array_resamples_and_downmixes():
    sr = 16_000
    stereo = np.random.default_rng(0).standard_normal((sr, 2)).astype(np.float32)
    wav = A.load_audio((stereo, sr))
    assert wav.ndim == 1 and wav.dtype == np.float32
    assert len(wav) == A.SAMPLE_RATE
    with pytest.raises(ValueError):
        A.load_audio(np.zeros(10, np.float32))  # sample rate missing
    with pytest.raises(ValueError):
        A.load_audio((np.array([np.nan], np.float32), sr))


def test_load_audio_file(tmp_path):
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "x.flac"
    sf.write(path, np.zeros((44_100, 2), np.float32), 44_100)
    wav = A.load_audio(str(path))
    assert len(wav) == A.SAMPLE_RATE


def test_transcript_cleanup():
    assert split_language_tag("<chinese> 你好。") == ("你好。", "zh")
    assert split_language_tag(" <english> Hello there.") == ("Hello there.", "en")
    assert split_language_tag("plain text") == ("plain text", None)
    assert clean_transcript("<chinese> 好<|empty|>的<|eot|>") == "好的"


@pytest.mark.parametrize("value,expected", [
    (None, None), ("auto", None), ("", None), ("zh", "zh"), ("Chinese", "zh"),
    ("yue", None), ("zh-TW", "zh"), ("en_US", "en"), ("en", "en"), ("English", "en"), ("fr", None)])
def test_normalize_language(value, expected):
    assert normalize_language(value) == expected


@pytest.mark.parametrize("deltas,expected", [
    (["<", "ch", "inese", ">", " 你", "好", "。"], "你好。"),
    ([" <english>", " Hello", " world."], "Hello world."),
    (["Hello", " world"], "Hello world"),
    (["<", "b>", " x"], "<b> x"),
    (["<ch"], "<ch"),
])
def test_streaming_tag_stripping(deltas, expected):
    pytest.importorskip("vllm")
    from mimo_asr_vllm.model import _StripLanguageTag

    p = _StripLanguageTag()
    out = "".join(p.process_delta(d, i == len(deltas) - 1) for i, d in enumerate(deltas))
    assert out == expected


def test_integer_pcm_is_normalized():
    pcm = (np.sin(np.linspace(0, 100, 24_000)) * 16384).astype(np.int16)
    wav = A.load_audio((pcm, 24_000))
    assert np.abs(wav).max() == pytest.approx(0.5, abs=1e-3)
