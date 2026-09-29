"""MiMo-V2.5-ASR on vLLM.

Installing this package registers ``MiMoV2ASRForCausalLM`` with vLLM through the
``vllm.general_plugins`` entry point, so ``vllm serve XiaomiMiMo/MiMo-V2.5-ASR``
and ``vllm.LLM`` work directly. ``MiMoASR`` is a convenience batch API.
"""

import logging
import os

__version__ = "0.1.0"

ARCHITECTURE = "MiMoV2ASRForCausalLM"
SUPPORTED_VLLM = ((0, 30), (0, 31))  # [min, max)

logger = logging.getLogger("vllm.plugins.mimo_asr")


def register() -> None:
    """vLLM plugin hook, executed in every vLLM process (API server, engine, workers)."""
    from vllm import ModelRegistry

    _check_vllm_version()
    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        # Lazy "module:Class" string: importing the model must not initialize CUDA here.
        ModelRegistry.register_model(ARCHITECTURE, "mimo_asr_vllm.model:MiMoV2ASRForCausalLM")
    _maybe_disable_flashinfer_sampler()
    _greedy_by_default()


def _check_vllm_version() -> None:
    import vllm

    try:
        version = tuple(int(x) for x in vllm.__version__.split("+")[0].split(".")[:2])
    except ValueError:  # dev builds
        return
    if not SUPPORTED_VLLM[0] <= version < SUPPORTED_VLLM[1]:
        logger.warning("mimo-asr-vllm %s is tested with vLLM %d.%d.x, found %s; if loading fails, "
                       "install a matching version (pip install 'vllm==%d.%d.*').", __version__,
                       *SUPPORTED_VLLM[0], vllm.__version__, *SUPPORTED_VLLM[0])


def _maybe_disable_flashinfer_sampler() -> None:
    """FlashInfer's JIT sampling kernels fail with "device kernel image is invalid"
    when the NVIDIA driver is older than the CUDA runtime of the PyTorch wheel.
    Fall back to vLLM's PyTorch sampler in that case (unless the user decided)."""
    if "VLLM_USE_FLASHINFER_SAMPLER" in os.environ:
        return
    try:
        import torch
        from vllm.third_party import pynvml

        if not torch.version.cuda:
            return
        major, minor = (int(x) for x in torch.version.cuda.split(".")[:2])
        pynvml.nvmlInit()
        try:
            driver = pynvml.nvmlSystemGetCudaDriverVersion()
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return
    if driver < major * 1000 + minor * 10:
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        logger.warning("NVIDIA driver supports CUDA %d.%d but PyTorch uses CUDA %d.%d: disabling the "
                       "FlashInfer sampler (VLLM_USE_FLASHINFER_SAMPLER=0). Upgrading the driver is "
                       "recommended.", driver // 1000, driver % 1000 // 10, major, minor)


def _greedy_by_default() -> None:
    """The official ASR decodes greedily, but the checkpoint's generation_config.json
    (temperature=0.6) would make chat/completions requests sample. Use greedy
    defaults for this architecture unless the user overrides them."""
    try:
        from vllm.config import ModelConfig
    except ImportError:
        return
    original = ModelConfig.get_diff_sampling_param
    if getattr(original, "_mimo_asr", False):
        return

    def get_diff_sampling_param(self):
        params = original(self)
        overrides = getattr(self, "override_generation_config", None) or {}
        if (ARCHITECTURE in (getattr(self, "architectures", None) or [])
                and getattr(self, "generation_config", "auto") == "auto"
                and "temperature" not in overrides):
            params = {k: v for k, v in params.items() if k not in ("temperature", "top_p", "top_k")}
            params["temperature"] = 0.0
        return params

    get_diff_sampling_param._mimo_asr = True
    ModelConfig.get_diff_sampling_param = get_diff_sampling_param


def __getattr__(name):
    if name in ("MiMoASR", "Transcription"):
        from . import asr

        return getattr(asr, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MiMoASR", "Transcription", "register", "__version__"]
