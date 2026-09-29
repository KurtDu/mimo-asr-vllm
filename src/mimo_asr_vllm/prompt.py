"""Prompt construction and output clean-up, matching the official ``asr_sft``."""

from __future__ import annotations

SOSP, EMPTY, EOSP = "<|sosp|>", "<|empty|>", "<|eosp|>"
IM_END = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"
# One audio item in a (chat) prompt; expanded to one <|empty|> per audio group.
AUDIO_PLACEHOLDER = SOSP + EMPTY + EOSP

LANGUAGE_TAGS = {None: "", "auto": "", "zh": "<chinese>", "en": "<english>"}
LANGUAGE_TAG_TEXTS = ("<chinese>", "<english>")
# Speech-generation markers that may appear in raw decoded text.
SPECIAL_MARKERS = (EMPTY, "<|eot|>", "<|eostm|>", IM_END, ENDOFTEXT)
# The official code samples a random instruction; we pin the first zh / en one.
INSTRUCTIONS = {"zh": "请将这段语音转换为文字", "en": "Please transcribe this audio file"}


def normalize_language(language: str | None) -> str | None:
    """Map user input to ``"zh"``, ``"en"`` or ``None`` (automatic detection)."""
    if language is None:
        return None
    lang = language.strip().lower().replace("_", "-")
    if lang in ("zh", "chinese", "mandarin", "cmn", "zh-cn", "zh-hans", "zh-tw", "zh-hant", "zh-hk"):
        return "zh"
    if lang in ("en", "english", "en-us", "en-gb"):
        return "en"
    return None  # auto / dialects (yue, wuu, ...) / other languages: let the model detect


def build_prompt_ids(tokenizer, language: str | None = None, instruction: str | None = None) -> list[int]:
    """Token ids of the ASR prompt with a single audio placeholder.

    Segments are tokenized separately, exactly as ``MimoAudio.get_asr_sft_prompt``.
    The placeholder ``<|sosp|><|empty|><|eosp|>`` is expanded by the vLLM
    multimodal processor to one ``<|empty|>`` per audio group.
    """
    lang = normalize_language(language)
    if instruction is None:
        instruction = INSTRUCTIONS["en" if lang == "en" else "zh"]

    def enc(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    ids = enc("<|im_start|>user\n")
    ids += [tokenizer.convert_tokens_to_ids(t) for t in (SOSP, EMPTY, EOSP)]
    ids += enc(instruction)
    ids += enc("<|im_end|>\n")
    ids += enc("<|im_start|>assistant\n")
    ids += enc("<think>\n\n</think>\n" + LANGUAGE_TAGS[lang])
    return ids


def split_language_tag(text: str) -> tuple[str, str | None]:
    """Remove special markers and the ``<chinese>``/``<english>`` tag.

    Returns ``(transcript, detected_language)``; mirrors the official
    post-processing in ``MimoAudio.forward`` / ``asr_sft``.
    """
    text = text.strip()
    for marker in SPECIAL_MARKERS:
        text = text.replace(marker, "")
    detected = "zh" if "<chinese>" in text else "en" if "<english>" in text else None
    if detected is not None:
        text = text.replace("<chinese>", "").replace("<english>", "").strip()
    return text, detected


def clean_transcript(text: str) -> str:
    return split_language_tag(text)[0]
