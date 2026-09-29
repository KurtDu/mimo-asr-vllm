"""vLLM implementation of ``MiMoV2ASRForCausalLM`` (XiaomiMiMo/MiMo-V2.5-ASR).

The model is registered out-of-tree through the ``vllm.general_plugins`` entry
point, so ``vllm serve XiaomiMiMo/MiMo-V2.5-ASR`` works once this package is
installed. Audio is encoded inside the vLLM worker (batched across requests)
and scattered into the Qwen2 decoder at the ``<|empty|>`` placeholder
positions; decoding uses the standard vLLM Qwen2 implementation (paged KV
cache, continuous batching, CUDA graphs, tensor parallelism).
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import torch
from torch import nn
from transformers import BatchFeature

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.inputs import MultiModalDataDict, PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    StreamingTranscriptionPostProcessor,
    SupportsMultiModal,
    SupportsTranscription,
)
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config

from . import audio as A
from .encoder import MiMoAudioAdapter, MiMoAudioEncoder, group_codes
from .prompt import (
    AUDIO_PLACEHOLDER,
    EMPTY,
    EOSP,
    LANGUAGE_TAG_TEXTS,
    SOSP,
    SPECIAL_MARKERS,
    build_prompt_ids,
    clean_transcript,
    normalize_language,
)

logger = init_logger("vllm.plugins.mimo_asr")

DEFAULT_AUDIO_TOKENIZER = "XiaomiMiMo/MiMo-Audio-Tokenizer"
# Prompt tokens around the audio (chat markup, instruction, language tag).
_PROMPT_OVERHEAD = 64
# Upper bound of mel frames per audio-encoder call (keeps activation memory flat).
_MAX_MEL_FRAMES_PER_CALL = 128_000


# --------------------------------------------------------------------------
# Multimodal processing (runs in the API / frontend process)
# --------------------------------------------------------------------------

class MiMoASRProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}

    def get_data_parser(self) -> MultiModalDataParser:
        # Resample with torchaudio's sinc resampler, like the reference code.
        return MultiModalDataParser(
            target_sr=A.SAMPLE_RATE,
            target_channels=1,
            audio_resample_method="torchaudio",
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
        )

    def get_max_audio_groups(self, seq_len: int) -> int:
        return max(1, seq_len - _PROMPT_OVERHEAD)

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int]):
        # One placeholder per 4 codec frames (+ <|sosp|>/<|eosp|> markers).
        return {"audio": self.get_max_audio_groups(seq_len) + 2}


class MiMoASRDummyInputsBuilder(BaseDummyInputsBuilder[MiMoASRProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return AUDIO_PLACEHOLDER * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        length = A.max_samples_for_groups(self.info.get_max_audio_groups(seq_len))
        return {
            "audio": self._get_dummy_audios(
                length=length,
                num_audios=mm_counts.get("audio", 0),
                overrides=mm_options.get("audio"),
            )
        }


class MiMoASRMultiModalProcessor(BaseMultiModalProcessor[MiMoASRProcessingInfo]):
    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        wavs, groups = [], []
        if "audio" in mm_items:
            audios = mm_items.get_items("audio", AudioProcessorItems)
            for i in range(audios.get_count()):
                wav = A.to_mono_float32(audios.get(i), channels_first=True)
                if wav.numel() == 0:
                    raise ValueError("Audio input is empty")
                if not torch.isfinite(wav).all():
                    raise ValueError("Audio input contains NaN or Inf values")
                wavs.append(wav)
                groups.append(A.num_audio_groups(wav.numel()))
        return BatchFeature({
            "audio_wavs": wavs,
            "audio_num_groups": torch.tensor(groups, dtype=torch.long),
        })

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "audio_wavs": MultiModalFieldConfig.batched("audio"),
            "audio_num_groups": MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        vocab = self.info.get_tokenizer().get_vocab()
        sosp, empty, eosp = vocab[SOSP], vocab[EMPTY], vocab[EOSP]

        def replacement(item_idx: int) -> PromptUpdateDetails:
            n = int(out_mm_kwargs["audio"][item_idx]["audio_num_groups"].data)
            return PromptUpdateDetails.select_token_id(
                [sosp] + [empty] * n + [eosp], embed_token_id=empty)

        return [PromptReplacement(modality="audio", target=[sosp, empty, eosp],
                                  replacement=replacement)]


# --------------------------------------------------------------------------
# Model (runs in the vLLM worker)
# --------------------------------------------------------------------------

_SERVING_LIMITS = {
    "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": "512",
    "VLLM_MAX_AUDIO_DECODE_DURATION_S": str(3 * 3600),
    "VLLM_MAX_AUDIO_DECODE_BYTES": str(4 * 1024**3),
}


class _StripLanguageTag(StreamingTranscriptionPostProcessor):
    """Streaming counterpart of ``clean_transcript``: hides the leading
    ``<chinese>``/``<english>`` tag the model emits in automatic mode."""

    def __init__(self) -> None:
        self.raw = ""
        self.emitted = 0

    def process_delta(self, text_delta: str, finished: bool) -> str:
        self.raw += text_delta
        text = self.raw.lstrip()
        if not finished and any(tag.startswith(text) for tag in LANGUAGE_TAG_TEXTS):
            return ""  # could still become a tag
        for tag in LANGUAGE_TAG_TEXTS:
            if text.startswith(tag):
                text = text[len(tag):].lstrip()
                break
        for marker in SPECIAL_MARKERS:
            text = text.replace(marker, "")
        if finished:
            text = text.rstrip()
        new, self.emitted = text[self.emitted:], max(self.emitted, len(text))
        return new


def _patch_transcription_serving() -> None:
    """Make ``/v1/audio/transcriptions`` preprocess audio like the reference code.

    * vLLM decodes uploads with a fixed resampler (libswresample); MiMo was
      run with torchaudio's sinc resampler, and the mismatch measurably changes
      the audio codes for non-24 kHz input. Decode at the native rate and
      resample with torchaudio instead.
    * Long uploads are split at the quietest pause (see ``split_on_pauses``)
      instead of within the last second before the clip limit.

    Installed only in processes that serve this model.
    """
    try:
        from vllm.entrypoints.speech_to_text.base import serving
        from vllm.multimodal.audio import resample_audio_torchaudio
    except ImportError:
        logger.warning("Could not patch vLLM transcription preprocessing; using defaults")
        return
    if getattr(serving, "_mimo_asr_patched", False):
        return
    # vLLM's defaults (25 MB uploads, 10 min decode) are tuned for short clips;
    # allow typical long recordings unless the user configured the limits.
    for name, value in _SERVING_LIMITS.items():
        os.environ.setdefault(name, value)
    original_load = serving.load_audio

    def load_audio(path, *, sr=None, **kwargs):
        wav, native_sr = original_load(path, sr=None, **kwargs)
        if sr is None or int(native_sr) == int(sr):
            return wav, native_sr
        return resample_audio_torchaudio(wav, orig_sr=native_sr, target_sr=sr), int(sr)

    def split_audio(audio_data, sample_rate, max_clip_duration_s, overlap_duration_s,
                    min_energy_window_size):
        return A.split_on_pauses(audio_data, int(sample_rate), max_clip_duration_s)

    serving.load_audio = load_audio
    serving.split_audio = split_audio
    serving._mimo_asr_patched = True


def resolve_audio_tokenizer(model_config: ModelConfig) -> Path:
    """Locate (or download) MiMo-Audio-Tokenizer.

    Priority: ``MIMO_AUDIO_TOKENIZER_PATH`` env var > ``audio_tokenizer_path`` in
    ``--hf-overrides`` > ``<model_dir>/audio_tokenizer`` > HF/ModelScope repo
    ``XiaomiMiMo/MiMo-Audio-Tokenizer``.
    """
    name = (os.environ.get("MIMO_AUDIO_TOKENIZER_PATH")
            or getattr(model_config.hf_config, "audio_tokenizer_path", None))
    if not name:
        local = Path(model_config.model) / "audio_tokenizer"
        name = str(local) if local.is_dir() else DEFAULT_AUDIO_TOKENIZER
    path = Path(name).expanduser()
    if path.is_dir():
        return path
    if path.is_absolute() or name.startswith((".", "~")):
        raise FileNotFoundError(f"MiMo audio tokenizer directory not found: {path}")
    patterns = ["config.json", "*.safetensors"]
    from vllm import envs

    try:
        if envs.VLLM_USE_MODELSCOPE:
            from modelscope import snapshot_download as ms_download

            return Path(ms_download(name, allow_patterns=patterns))
        from huggingface_hub import snapshot_download

        logger.info("Using MiMo audio tokenizer %s (downloaded on first use)", name)
        return Path(snapshot_download(name, allow_patterns=patterns))
    except Exception as exc:
        raise RuntimeError(
            f"Could not fetch the MiMo audio tokenizer '{name}' ({type(exc).__name__}: {exc}). "
            "Download it once (e.g. `hf download XiaomiMiMo/MiMo-Audio-Tokenizer --local-dir DIR`) "
            "and set MIMO_AUDIO_TOKENIZER_PATH=DIR, or use HF_ENDPOINT / VLLM_USE_MODELSCOPE=True."
        ) from exc


@MULTIMODAL_REGISTRY.register_processor(
    MiMoASRMultiModalProcessor,
    info=MiMoASRProcessingInfo,
    dummy_inputs=MiMoASRDummyInputsBuilder,
)
class MiMoV2ASRForCausalLM(nn.Module, SupportsMultiModal, SupportsTranscription):
    supported_languages = {"zh": "Chinese", "en": "English"}

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "model.": "language_model.model.",
        "lm_head.": "language_model.lm_head.",
        "speech_embeddings.": "audio_adapter.speech_embeddings.",
        "input_local_transformer.": "audio_adapter.input_local_transformer.",
        "speech_group_downcast.": "audio_adapter.speech_group_downcast.",
        # Speech-generation (TTS) heads are not used for ASR.
        "local_transformer.": None,
        "local_transformer_lm_heads.": None,
        "hidden_states_downcast.": None,
    })

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return AUDIO_PLACEHOLDER
        raise ValueError("Only audio input is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.tokenizer_path = resolve_audio_tokenizer(vllm_config.model_config)
        tok_cfg = MiMoAudioEncoder.load_config(self.tokenizer_path)

        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_encoder = MiMoAudioEncoder(tok_cfg, n_q=config.audio_channels)
            self.audio_adapter = MiMoAudioAdapter(config)
        # Front-end buffers are built in fp32 on CPU (as in the reference) and
        # moved to the device after loading.
        with torch.device("cpu"):
            default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)
            try:
                self.mel = A.LogMelSpectrogram()
            finally:
                torch.set_default_dtype(default_dtype)

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen2ForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model"))
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors
        # <|empty|> switches the original model to speech generation (TTS head),
        # which is out of scope for ASR, so it is never sampled as a text token.
        try:
            tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
            self._empty_token_id = tokenizer.convert_tokens_to_ids(EMPTY)
        except Exception:  # e.g. --skip-tokenizer-init
            self._empty_token_id = 151667

    # ---------------------------------------------------------------- audio
    def _device(self) -> torch.device:
        return self.audio_adapter.speech_group_downcast.weight.device

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        wavs = kwargs.get("audio_wavs")
        if wavs is None:
            return []
        if isinstance(wavs, torch.Tensor):
            wavs = list(wavs.unbind(0)) if wavs.ndim == 2 else [wavs]
        device = self._device()
        expected = kwargs.get("audio_num_groups")
        expected = [int(x) for x in torch.as_tensor(expected).reshape(-1)] if expected is not None else None

        mels, owner = [], []
        for i, wav in enumerate(wavs):
            wav = wav.to(device=device, dtype=torch.float32).reshape(-1)
            for s, e in A.chunk_bounds(wav.numel()):
                mels.append(self.mel(wav[s:e]))
                owner.append(i)

        codes = self._encode_mels(mels)
        per_item: list[list[torch.Tensor]] = [[] for _ in wavs]
        for i, c in zip(owner, codes):
            per_item[i].append(c)
        grouped = [group_codes(torch.cat(parts)) for parts in per_item]
        sizes = [g.shape[0] for g in grouped]
        if expected is not None and sizes != expected:
            raise RuntimeError(f"Audio token count mismatch: {sizes} vs {expected}")
        embeds = self.audio_adapter(torch.cat(grouped))
        return list(embeds.split(sizes))

    def _encode_mels(self, mels: list[torch.Tensor]) -> list[torch.Tensor]:
        out, batch, frames = [], [], 0
        for mel in mels:
            if batch and frames + mel.shape[0] > _MAX_MEL_FRAMES_PER_CALL:
                out += self.audio_encoder(batch)
                batch, frames = [], 0
            batch.append(mel)
            frames += mel.shape[0]
        if batch:
            out += self.audio_encoder(batch)
        return out

    # ------------------------------------------------------------- decoder
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model.model(input_ids, positions, intermediate_tensors,
                                         inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.language_model.compute_logits(hidden_states)
        if logits is not None:
            logits[..., self._empty_token_id] = float("-inf")
        return logits

    # ------------------------------------------------------------- weights
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        loaded |= self._load_audio_tokenizer()
        device = self._device()
        self.audio_encoder.to(device)
        self.audio_adapter.to(device)  # moves the fp32 RoPE buffers
        self.mel.to(device)
        return loaded

    def _load_audio_tokenizer(self) -> set[str]:
        import json

        from safetensors import safe_open

        root = self.tokenizer_path
        index = root / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(root / "model.safetensors", framework="pt") as f:
                weight_map = {k: "model.safetensors" for k in f.keys()}
        mapping = self.audio_encoder.checkpoint_mapping()
        missing = sorted(set(mapping) - set(weight_map))
        if missing:
            raise ValueError(f"MiMo audio tokenizer at {root} is missing weights: {missing[:5]}")
        state = dict(self.audio_encoder.named_parameters())
        state.update(self.audio_encoder.named_buffers())
        by_file: dict[str, list[str]] = {}
        for ckpt_key in mapping:
            by_file.setdefault(weight_map[ckpt_key], []).append(ckpt_key)
        loaded = set()
        with torch.no_grad():
            for file, keys in by_file.items():
                with safe_open(root / file, framework="pt", device="cpu") as f:
                    for ckpt_key in keys:
                        target = state[mapping[ckpt_key]]
                        value = f.get_tensor(ckpt_key)
                        if value.shape != target.shape:
                            raise ValueError(f"Shape mismatch for {ckpt_key}: {tuple(value.shape)}"
                                             f" vs {tuple(target.shape)}")
                        target.copy_(value)
                        loaded.add("audio_encoder." + mapping[ckpt_key])
        logger.info("Loaded MiMo audio tokenizer encoder from %s", root)
        return loaded

    # ------------------------------------------------------- transcription
    @classmethod
    def get_speech_to_text_config(cls, model_config: ModelConfig, task_type: str) -> SpeechToTextConfig:
        _patch_transcription_serving()
        # Long uploads are split at low-energy points into clips of at most
        # MIMO_ASR_MAX_CLIP_S seconds that are transcribed in parallel.
        max_clip = A.max_clip_seconds()
        return SpeechToTextConfig(
            sample_rate=A.SAMPLE_RATE,
            max_audio_clip_s=None if max_clip is None else int(max_clip),
            overlap_chunk_second=1,
            min_energy_split_window_size=A.SAMPLE_RATE // 10,
        )

    @classmethod
    def get_generation_prompt(cls, stt_params: SpeechToTextParams) -> PromptType:
        if stt_params.task_type != "transcribe":
            raise ValueError("MiMo-V2.5-ASR only supports transcription")
        tokenizer = cached_tokenizer_from_config(stt_params.model_config)
        return TokensPrompt(
            prompt_token_ids=build_prompt_ids(tokenizer, stt_params.language),
            multi_modal_data={"audio": stt_params.audio},
        )

    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        """Accept ``auto``/aliases/any code instead of rejecting the request.

        ``zh``/``en`` add the official language tag to the prompt; anything else
        (including ``None``) lets the model detect the language.
        """
        if language is None:
            return None
        normalized = normalize_language(language)
        if normalized is not None:
            return normalized
        if language.strip().lower() not in ("", "auto"):
            logger.warning_once("language=%r is not a MiMo language tag (zh/en); using automatic "
                                "language detection.", language)
        return None

    @classmethod
    def get_streaming_post_processor_cls(cls) -> type[StreamingTranscriptionPostProcessor]:
        return _StripLanguageTag

    @classmethod
    def get_num_audio_tokens(cls, audio_duration_s: float, stt_config: SpeechToTextConfig,
                             model_config: ModelConfig) -> int | None:
        return A.num_audio_groups(max(1, int(audio_duration_s * A.SAMPLE_RATE)))

    @classmethod
    def post_process_output(cls, text: str) -> str:
        return clean_transcript(text)
