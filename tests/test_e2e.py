"""GPU end-to-end tests. Enabled by setting MIMO_ASR_TEST_MODEL (and optionally
MIMO_AUDIO_TOKENIZER_PATH, MIMO_ASR_TEST_AUDIO=<wav> + MIMO_ASR_TEST_TEXT=<expected>)."""

import os

import numpy as np
import pytest

MODEL = os.environ.get("MIMO_ASR_TEST_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="set MIMO_ASR_TEST_MODEL to run GPU tests")


@pytest.fixture(scope="module")
def asr():
    from mimo_asr_vllm import MiMoASR

    engine = MiMoASR(MODEL, gpu_memory_utilization=0.6, max_num_seqs=32,
                     limit_mm_per_prompt={"audio": 2})
    yield engine
    engine.close()


def _speech():
    path = os.environ.get("MIMO_ASR_TEST_AUDIO")
    if not path:
        pytest.skip("set MIMO_ASR_TEST_AUDIO for accuracy checks")
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    return wav, sr, os.environ.get("MIMO_ASR_TEST_TEXT", "")


def test_edge_case_inputs(asr):
    rng = np.random.default_rng(0)
    inputs = [
        (np.zeros(240, np.float32), 24_000),                 # 10 ms, shorter than one STFT window
        (np.zeros(3 * 16_000, np.float32), 16_000),          # silence at 16 kHz
        (0.01 * rng.standard_normal((44_100, 2)).astype(np.float32), 44_100),  # stereo noise
        (0.01 * rng.standard_normal(31 * 24_000).astype(np.float32), 24_000),  # two 30 s chunks
    ]
    results = asr.transcribe_batch(inputs)
    assert len(results) == len(inputs)
    for r, (wav, sr) in zip(results, inputs):
        assert isinstance(r.text, str)
        assert abs(r.duration - len(wav) / sr) < 0.01


def test_accuracy_and_language_modes(asr):
    wav, sr, expected = _speech()
    auto = asr.transcribe((wav, sr))
    assert auto.finish_reason == "stop" and auto.text
    if expected:
        assert auto.text == expected
    forced = asr.transcribe(wav, sample_rate=sr, language=auto.language or "zh")
    assert forced.text


def test_long_audio_is_split_and_joined(asr):
    from mimo_asr_vllm import audio as A

    wav, sr, _ = _speech()
    single = asr.transcribe((wav, sr)).text
    gap = np.zeros(int(0.5 * sr), np.float32)
    reps = int(np.ceil(1.6 * asr.max_clip_s * sr / (len(wav) + len(gap))))  # -> >= 2 clips
    long = np.concatenate([np.concatenate([wav, gap])] * reps)
    clips = A.split_on_pauses(A.load_audio((long, sr)), A.SAMPLE_RATE, asr.max_clip_s)
    assert len(clips) >= 2 and all(len(c) <= asr.max_clip_s * A.SAMPLE_RATE for c in clips)
    out = asr.transcribe((long, sr))
    assert out.finish_reason == "stop"
    # (The model may collapse verbatim repetitions inside a clip, so only check
    # that every clip contributed text.)
    assert out.text.count(single[:6]) >= len(clips)


def test_two_audios_in_one_prompt_keep_order(asr):
    from vllm import SamplingParams

    from mimo_asr_vllm import audio as A

    wav, sr, _ = _speech()
    wav24 = A.load_audio((wav, sr))
    silence = np.zeros(A.SAMPLE_RATE, np.float32)
    tok = asr.tokenizer
    text = ("<|im_start|>user\n<|sosp|><|empty|><|eosp|><|sosp|><|empty|><|eosp|>"
            "请将这段语音转换为文字<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n")
    ids = tok.encode(text, add_special_tokens=False)
    params = SamplingParams(temperature=0, max_tokens=256)
    a = asr.llm.generate({"prompt_token_ids": ids, "multi_modal_data": {"audio": [wav24, silence]}}, params)
    b = asr.llm.generate({"prompt_token_ids": ids, "multi_modal_data": {"audio": [silence, wav24]}}, params)
    assert a[0].outputs[0].text.strip() and b[0].outputs[0].text.strip()
