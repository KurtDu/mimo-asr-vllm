# Benchmarks

Scripts used for the numbers in the main README.

| Script | Purpose |
|---|---|
| `make_manifest.py` | build the evaluation manifests (Seed-TTS-eval zh/en, AISHELL-3, long-form sets) |
| `run_official.py` | run the **unmodified** official implementation (its own environment) |
| `run_vllm_offline.py` | run `mimo_asr_vllm.MiMoASR` (batch or `--sequential`) |
| `bench_server.py` | load-test `/v1/audio/transcriptions` of a running server |
| `score.py` | CER (zh) / WER (en) per subset + exact-match rate between systems |

```bash
# 0) data (Seed-TTS-eval as packaged on HF by sglang-omni, AISHELL-3 test)
python make_manifest.py /data/eval_data manifests/

# 1) official implementation (separate env: torch 2.6, transformers 4.49, flash-attn 2.7.4)
git clone https://github.com/XiaomiMiMo/MiMo-V2.5-ASR official
python run_official.py --upstream official --model MODEL_DIR --audio-tokenizer TOKENIZER_DIR \
    --manifest manifests/{seed_zh,seed_en,aishell3,long}.jsonl --output results/official.jsonl

# 2) this package, whole set at once / one request at a time
python run_vllm_offline.py --model MODEL_DIR --audio-tokenizer TOKENIZER_DIR \
    --manifest manifests/{seed_zh,seed_en,aishell3,long}.jsonl --output results/vllm.jsonl
python run_vllm_offline.py ... --sequential --output results/vllm_seq.jsonl

# 3) server
MIMO_AUDIO_TOKENIZER_PATH=TOKENIZER_DIR vllm serve MODEL_DIR --served-model-name mimo-asr &
python bench_server.py --model mimo-asr --concurrency 32 \
    --manifest manifests/{seed_zh,seed_en,aishell3,long}.jsonl --output results/server_c32.jsonl

# 4) score
python score.py official=results/official.jsonl vllm=results/vllm.jsonl server=results/server_c32.jsonl
```

Notes
* The official code picks a random instruction for every request; `run_official.py` pins it to the
  first Chinese/English template, which is also what this package uses.
* Restart the server between load tests: vLLM's prefix/encoder caches make repeated runs on the
  same files unrealistically fast.
* `run_vllm_offline.py` defaults to `--max-clip-s none` (single pass, like the official code);
  the server and `MiMoASR()` split audio longer than 60 s at pauses by default.
