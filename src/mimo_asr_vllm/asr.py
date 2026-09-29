"""High-level batch transcription API on top of ``vllm.LLM``."""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import audio as A
from .prompt import ENDOFTEXT, IM_END, build_prompt_ids, normalize_language, split_language_tag

DEFAULT_MODEL = "XiaomiMiMo/MiMo-V2.5-ASR"


@dataclass
class Transcription:
    text: str
    language: str | None
    """Language requested, or the one tagged by the model in automatic mode."""
    duration: float
    """Audio duration in seconds."""
    num_tokens: int
    finish_reason: str
    """``stop``; ``length`` (hit ``max_tokens``, text truncated); ``error``."""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MiMoASR:
    """Batch speech recognition with MiMo-V2.5-ASR on vLLM.

    Example::

        asr = MiMoASR()                       # downloads the model on first use
        print(asr.transcribe("audio.wav").text)
        results = asr.transcribe_batch(["a.wav", "b.mp3"], language="zh")

    Args:
        model: HF repo id or local directory of MiMo-V2.5-ASR.
        audio_tokenizer: HF repo id or local directory of MiMo-Audio-Tokenizer
            (default: ``$MIMO_AUDIO_TOKENIZER_PATH``, ``<model>/audio_tokenizer``
            or ``XiaomiMiMo/MiMo-Audio-Tokenizer``).
        max_clip_s: audio longer than this (seconds) is split at pauses into clips
            decoded in parallel; ``None`` decodes whole files when they fit into
            the context (quality degrades beyond ~2 minutes). Default 60, or
            ``$MIMO_ASR_MAX_CLIP_S``.
        num_io_threads: threads used to decode / resample audio files.
        **llm_kwargs: forwarded to ``vllm.LLM`` (``gpu_memory_utilization``,
            ``tensor_parallel_size``, ``max_num_seqs``, ``quantization``, ...).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        audio_tokenizer: str | None = None,
        *,
        max_clip_s: float | None | str = "default",
        num_io_threads: int = 8,
        **llm_kwargs,
    ):
        from . import register

        register()  # also sets environment-dependent defaults
        from vllm import LLM

        kwargs: dict[str, Any] = dict(
            max_model_len=8192,
            limit_mm_per_prompt={"audio": 1},
            gpu_memory_utilization=0.9,
            seed=0,
        )
        if audio_tokenizer is not None:
            path = Path(audio_tokenizer).expanduser()
            overrides = dict(llm_kwargs.pop("hf_overrides", None) or {})
            overrides["audio_tokenizer_path"] = str(path.resolve()) if path.is_dir() else audio_tokenizer
            kwargs["hf_overrides"] = overrides
        kwargs.update(llm_kwargs)
        self.llm = LLM(model=model, **kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.max_model_len = self.llm.llm_engine.model_config.max_model_len
        self.max_clip_s = A.max_clip_seconds() if max_clip_s == "default" else max_clip_s
        self._stop_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in (IM_END, ENDOFTEXT)]
        self._prompts = {lang: build_prompt_ids(self.tokenizer, lang) for lang in (None, "zh", "en")}
        self._pool = ThreadPoolExecutor(max_workers=max(1, num_io_threads))

    # ------------------------------------------------------------------ API
    def transcribe(self, audio, *, language: str | None = None, sample_rate: int | None = None,
                   max_tokens: int | None = None) -> Transcription:
        """Transcribe one input: path, bytes, file object, ``(array, sr)``, or
        array + ``sample_rate``. Raises ``ValueError`` if the audio cannot be read."""
        if sample_rate is not None:
            audio = (audio, sample_rate)
        return self.transcribe_batch([audio], language=language, max_tokens=max_tokens,
                                     on_error="raise")[0]

    def transcribe_batch(
        self,
        audios: Sequence,
        *,
        language: str | None | Sequence[str | None] = None,
        max_tokens: int | None = None,
        on_error: str = "raise",
        use_tqdm: bool = False,
    ) -> list[Transcription]:
        """Transcribe many inputs at once (continuous batching inside vLLM).

        Args:
            language: ``None``/``"auto"`` (detect), ``"zh"``, ``"en"``, or one
                value per input.
            max_tokens: cap on generated tokens per clip (default: context limit).
            on_error: ``"raise"`` or ``"skip"`` (unreadable inputs return a
                result with ``finish_reason="error"``).
        """
        from vllm import SamplingParams

        if on_error not in ("raise", "skip"):
            raise ValueError("on_error must be 'raise' or 'skip'")
        audios = list(audios)
        if not audios:
            return []
        langs = [language] * len(audios) if language is None or isinstance(language, str) else list(language)
        if len(langs) != len(audios):
            raise ValueError("language must be a string or have one entry per audio")
        langs = [normalize_language(lang) for lang in langs]

        loaded = list(self._pool.map(_safe_load, audios))
        requests, owners = [], []
        for i, ((wav, err), lang) in enumerate(zip(loaded, langs)):
            if err is not None:
                if on_error == "raise":
                    raise ValueError(f"Cannot read audio #{i} ({_describe(audios[i])}): {err}")
                continue
            for clip in self._split(wav):
                requests.append({"prompt_token_ids": self._prompts[lang],
                                 "multi_modal_data": {"audio": clip}})
                owners.append(i)

        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop_token_ids=self._stop_ids,
                                skip_special_tokens=False, detokenize=False)
        outputs = self.llm.generate(requests, params, use_tqdm=use_tqdm) if requests else []

        parts: list[list] = [[] for _ in audios]
        for owner, out in zip(owners, outputs):
            parts[owner].append(out.outputs[0])
        results = []
        for (wav, err), lang, clips in zip(loaded, langs, parts):
            if err is not None:
                results.append(Transcription("", lang, 0.0, 0, "error", err))
                continue
            texts, detected, tokens, reason = [], None, 0, "stop"
            for c in clips:
                ids = list(c.token_ids)
                if ids and ids[-1] in self._stop_ids:
                    ids = ids[:-1]
                text, tag = split_language_tag(self.tokenizer.decode(ids, skip_special_tokens=False))
                texts.append(text)
                detected = detected or tag
                tokens += len(c.token_ids)
                if c.finish_reason == "length":
                    reason = "length"
            out_lang = lang or detected
            sep = "" if out_lang == "zh" else " "
            results.append(Transcription(sep.join(t for t in texts if t), out_lang,
                                         len(wav) / A.SAMPLE_RATE, tokens, reason))
        return results

    # ------------------------------------------------------------- helpers
    def _split(self, wav: np.ndarray) -> list[np.ndarray]:
        # A single request must also fit into the context window (audio + transcript).
        fit = A.max_samples_for_groups(max(1, (self.max_model_len - 64) // 2)) / A.SAMPLE_RATE
        max_len = fit if self.max_clip_s is None else min(self.max_clip_s, fit)
        return A.split_on_pauses(wav, A.SAMPLE_RATE, max_len)

    def close(self) -> None:
        self._pool.shutdown(wait=False)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _safe_load(source):
    try:
        if isinstance(source, (bytes, bytearray, memoryview)):
            source = io.BytesIO(bytes(source))
        return A.load_audio(source), None
    except Exception as exc:  # reported per input
        return None, f"{type(exc).__name__}: {exc}"


def _describe(source) -> str:
    if isinstance(source, (str, Path)):
        return str(source)
    if isinstance(source, tuple):
        return "array"
    return type(source).__name__
