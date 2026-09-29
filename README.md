# mimo-asr-vllm

Fast inference and OpenAI-compatible serving of
[**XiaomiMiMo/MiMo-V2.5-ASR**](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-ASR) with
[vLLM](https://github.com/vllm-project/vllm).

* **Drop-in vLLM plugin** – `pip install` and `vllm serve XiaomiMiMo/MiMo-V2.5-ASR` just works
  (no fork of vLLM, no weight conversion, uses the original checkpoints).
* **Fast** – continuous batching, paged KV cache, CUDA graphs and a batched audio encoder
  inside the vLLM worker: **~30× the throughput** of the official implementation on one GPU and
  **3.5× lower latency** for a single request.
* **Same results as the official code** – same audio front-end, prompt, greedy decoding and
  post-processing; CER/WER match the official implementation (see [Accuracy](#accuracy)).
* **Every interface you need** – `/v1/audio/transcriptions` (incl. streaming),
  `/v1/chat/completions` with audio, a Python batch API and a CLI.
* **Long audio** – recordings of any length are split at pauses and decoded in parallel
  (quality is *better* than the official single pass on >2 min audio, see [Long audio](#long-audio)).

## Install

Requires Linux, Python ≥ 3.10 and an NVIDIA GPU with ≥ 24 GB of memory (bf16; with
`--quantization fp8` about 18 GB suffice). A fresh virtual environment is recommended
(vLLM pins its own PyTorch).

```bash
# NVIDIA driver >= 580 (CUDA 13):
pip install "git+https://github.com/KurtDu/mimo-asr-vllm.git"

# NVIDIA driver 525-575 (CUDA 12.x): install vLLM's CUDA 12.9 build first
pip install "https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu129
pip install "git+https://github.com/KurtDu/mimo-asr-vllm.git"
```

(`nvidia-smi` shows your driver version. From a local checkout: `pip install .`)

The model (`XiaomiMiMo/MiMo-V2.5-ASR`) and the audio tokenizer (`XiaomiMiMo/MiMo-Audio-Tokenizer`)
are downloaded from Hugging Face on first use. Set `HF_ENDPOINT` for a mirror, or
`VLLM_USE_MODELSCOPE=True` (with `pip install modelscope`) to download from ModelScope.
Already downloaded? Pass local directories, see [Configuration](#configuration).

## Quick start

### OpenAI-compatible server

```bash
vllm serve XiaomiMiMo/MiMo-V2.5-ASR
# or, equivalently, with the short model alias "mimo-asr":
mimo-asr serve
```

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F model=XiaomiMiMo/MiMo-V2.5-ASR -F file=@audio.wav
# {"text":"简单地说，这相当于惠普把消费领域市场拱手相让了。", ...}
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
with open("audio.mp3", "rb") as f:
    print(client.audio.transcriptions.create(model="XiaomiMiMo/MiMo-V2.5-ASR", file=f).text)

# optional: language="zh" / "en" (default: automatic), stream=True
```

Any `vllm serve` option can be added, e.g. `--tensor-parallel-size 2`, `--quantization fp8`,
`--gpu-memory-utilization 0.6`, `--port 8080`, `--api-key ...`.

<details>
<summary>Chat completions with audio input</summary>

```python
import base64
audio = base64.b64encode(open("audio.wav", "rb").read()).decode()
resp = client.chat.completions.create(
    model="XiaomiMiMo/MiMo-V2.5-ASR",
    messages=[{"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}},
        {"type": "text", "text": "请将这段语音转换为文字"},
    ]}],
)
print(resp.choices[0].message.content)   # "<chinese> 简单地说，……"
```

The raw model output starts with a language tag (`<chinese>` / `<english>`); the transcription
endpoint, the Python API and the CLI remove it for you.
</details>

### Python

```python
from mimo_asr_vllm import MiMoASR

asr = MiMoASR()                                   # or MiMoASR("/path/MiMo-V2.5-ASR", "/path/MiMo-Audio-Tokenizer")
print(asr.transcribe("audio.wav").text)

# batch: files (wav/flac/mp3/m4a/ogg/...), bytes, file objects or (array, sample_rate)
results = asr.transcribe_batch(["a.wav", "b.mp3", (samples, 16000)], language="auto")
for r in results:
    print(r.text, r.language, r.duration)
```

`MiMoASR(...)` forwards extra keyword arguments to `vllm.LLM`
(`tensor_parallel_size`, `gpu_memory_utilization`, `quantization="fp8"`, `max_num_seqs`, ...).
You can also use `vllm.LLM` directly – the model is registered automatically.

### Command line

```bash
mimo-asr transcribe audio_dir/ more.wav -o results.jsonl     # directories are searched recursively
mimo-asr transcribe -i file_list.txt --language zh -o results.jsonl
```

Each JSONL line contains `audio, text, language, duration, num_tokens, finish_reason, error`.
Unreadable files are reported (`finish_reason="error"`) without stopping the run.

## Performance

1× NVIDIA H20 (96 GB), bf16, 3,632 utterances / 5.25 h of audio
(Seed-TTS-eval zh + en, AISHELL-3, 24 synthetic 35–99 s recordings).
RTFx = seconds of audio transcribed per second of wall-clock time (higher is better).

| System | Mode | RTFx | Latency / request |
|---|---|---:|---:|
| Official implementation (HF Transformers) | one request at a time (its only mode) | 9.6 | 0.54 s |
| **mimo-asr-vllm**, Python API | one request at a time | 33 | 0.16 s |
| **mimo-asr-vllm**, Python API | whole set at once | **327** | – |
| **mimo-asr-vllm**, Python API, `quantization="fp8"` | whole set at once | 397 | – |
| **mimo-asr-vllm**, Python API, 2 GPUs (`tensor_parallel_size=2`) | whole set at once | 431 | – |
| **mimo-asr-vllm**, server | 1 concurrent request | 38 | 0.15 s |
| **mimo-asr-vllm**, server | 32 concurrent requests | 247 | 0.67 s |
| **mimo-asr-vllm**, server | 128 concurrent requests | 339 | 1.9 s |

Server numbers include HTTP upload and audio decoding; latency is the mean per request. The audio
encoder (a 32-layer transformer) runs in the same batch as the decoder, so on faster GPUs than the
compute-limited H20 the numbers scale up further.

## Accuracy

Greedy decoding, automatic language detection, same data as above.
CER for Chinese, WER for English (punctuation removed, case-folded).

| Subset | # | Official | mimo-asr-vllm (Python) | mimo-asr-vllm (server) |
|---|---:|---:|---:|---:|
| Seed-TTS-eval zh | 2020 | 0.951 | 0.956 | 0.958 |
| Seed-TTS-eval en | 1088 | 1.650 | 1.675 | 1.683 |
| AISHELL-3 (44.1 kHz) | 500 | 3.432 | 3.400 | 3.335 |
| 35–99 s recordings | 24 | 1.252 | 1.210 | 1.836 ¹ |
| **all Chinese (CER %)** | 2532 | **1.105** | **1.105** | 1.163 |
| **all English (WER %)** | 1100 | **1.680** | **1.695** | 1.709 |

¹ The server uses the default 60 s clip limit (the Python column decodes each file in one pass,
like the official code); see [Long audio](#long-audio).

The transcripts are character-for-character identical to the official ones for 97% of the utterances. The rest differ
by a punctuation mark or a word where two tokens are almost tied: bf16 kernels are not
bit-identical across libraries. For reference, merely switching the official code's attention
kernel (PyTorch SDPA → math kernel) changes 1% of its transcripts, and switching it to eager
attention changes 23% (and increases its CER/WER by ~1 point); the vLLM port is far closer to the
official default than those variants.

## Long audio

MiMo-V2.5-ASR degrades when a single pass covers more than ~2 minutes of speech (the official
code transcribes a whole file in one pass). This package therefore splits audio longer than
**60 s** at the quietest pause (never inside a word) and transcribes the clips in parallel:

| 12 recordings of 2.3–5 min (mixed zh/en) | CER/WER |
|---|---:|
| Official implementation, single pass | 22.9 % |
| mimo-asr-vllm, single pass (`max_clip_s=None`) | 26.9 % |
| **mimo-asr-vllm, default (split at pauses, ≤ 60 s clips)** | **1.5 %** |

On 40 further recordings (2–7 min, 3 h in total) split into the same 60 s clips, the official
implementation reaches 2.23 % CER / 1.70 % WER and mimo-asr-vllm 2.38 % CER / 1.68 % WER.
The splitter cuts at the quietest 200 ms in the second half of each clip; clips are decoded
in parallel, so long files also finish much faster.

Change the limit with `MIMO_ASR_MAX_CLIP_S=<seconds>` (server and Python), `MiMoASR(max_clip_s=...)`
or `mimo-asr transcribe --max-clip-s ...`; `none` disables splitting.
The server accepts uploads up to 512 MB / 3 h by default
(`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB`, `VLLM_MAX_AUDIO_DECODE_DURATION_S`, `VLLM_MAX_AUDIO_DECODE_BYTES`).

## Configuration

| Setting | Purpose |
|---|---|
| `MIMO_AUDIO_TOKENIZER_PATH=/path/or/repo` | location of MiMo-Audio-Tokenizer (default: `<model>/audio_tokenizer` if present, else `XiaomiMiMo/MiMo-Audio-Tokenizer`). Also `--hf-overrides '{"audio_tokenizer_path": "..."}'` or `MiMoASR(audio_tokenizer=...)`. |
| `MIMO_ASR_MAX_CLIP_S=60` | long-audio clip length, `none` to disable splitting |
| `MIMO_ASR_DISABLE_FLASH_ATTN=1` | use PyTorch SDPA instead of FlashAttention in the audio encoder (automatic on GPUs without FlashAttention) |
| `language` | `auto` (default; the model detects Chinese/English/dialects/code-switching), `zh` or `en` add the official language tag. Other values fall back to `auto`. |
| decoding | greedy by default everywhere (as the official ASR); pass `temperature` to override |

Local model directories work everywhere a repo id does:

```bash
MIMO_AUDIO_TOKENIZER_PATH=/models/MiMo-Audio-Tokenizer vllm serve /models/MiMo-V2.5-ASR
```

## How it works

MiMo-V2.5-ASR is a Qwen2-7B decoder that reads audio as 8-channel RVQ codes from
MiMo-Audio-Tokenizer; every 4 codec frames (160 ms) are fused by a small "input local
transformer" into one LLM input embedding. The plugin implements `MiMoV2ASRForCausalLM` for vLLM:

1. **Front-end (API process)** – decode, down-mix and resample to 24 kHz with torchaudio's sinc
   resampler (as the official code); the prompt gets one `<|empty|>` placeholder per audio group.
2. **Audio encoder (vLLM worker)** – log-mel (30 s chunks, as the official code) → 32-layer
   tokenizer encoder with variable-length FlashAttention → fp32 RVQ (8 codebooks) → input local
   transformer → LLM embeddings. All chunks of all requests in a scheduler step are encoded in one
   batch; padding is handled so that every chunk is encoded exactly as if it were alone.
3. **Decoder** – vLLM's Qwen2 implementation (paged attention, CUDA graphs, TP, quantization).

The speech-generation (TTS) heads of the checkpoint are not loaded; `<|empty|>` (the switch to speech
output) is never sampled.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA error: the provided PTX was compiled with an unsupported toolchain` / `device kernel image is invalid` | The NVIDIA driver is older than the CUDA of your PyTorch wheel. Install the matching build (see [Install](#install)) or upgrade the driver. The plugin already disables FlashInfer sampling and probes FlashAttention in this case. |
| `Free memory on device ... is less than desired GPU memory utilization` | Other processes use the GPU: lower `--gpu-memory-utilization` (e.g. `0.6`). |
| Out of memory on 24 GB GPUs | `--quantization fp8` (Ada/Hopper+), `--max-model-len 4096`, or `--max-num-seqs 64`. |
| `413 Request Entity Too Large` / decode limit errors | raise `VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` / `VLLM_MAX_AUDIO_DECODE_DURATION_S`. |
| Audio tokenizer download is slow / blocked | download `XiaomiMiMo/MiMo-Audio-Tokenizer` once and set `MIMO_AUDIO_TOKENIZER_PATH`. |
| `Model architectures ['MiMoV2ASRForCausalLM'] are not supported` | the plugin is not installed in the Python environment that runs vLLM (`pip show mimo-asr-vllm`). |
| `An attempt has been made to start a new process before the current process has finished its bootstrapping phase` | your script initialised CUDA before creating `MiMoASR`/`LLM`, so vLLM uses `spawn`: put the code under `if __name__ == "__main__":`. |
| Any other kernel error in the audio encoder | `MIMO_ASR_DISABLE_FLASH_ATTN=1` switches the encoder to PyTorch SDPA (slower). |

## Limitations

* Text output only (the model's TTS/speech-generation path is not supported).
* The OpenAI `prompt` field is ignored (MiMo-V2.5-ASR has no context prompt); no word timestamps
  (`verbose_json`/`srt`/`vtt` are not available).
* Requires vLLM 0.30.x. Verified on NVIDIA H20 (Hopper) with the CUDA 12.9 build of vLLM 0.30.0:
  bf16 / fp16 / fp8, 1–2 GPUs (tensor parallel), FlashAttention and SDPA encoder paths.
  Pipeline parallelism is not supported.

## Reproduce the benchmarks

See [`benchmarks/`](benchmarks/README.md).

## License

Apache-2.0. MiMo-V2.5-ASR and MiMo-Audio-Tokenizer are released by Xiaomi under their own licenses.
