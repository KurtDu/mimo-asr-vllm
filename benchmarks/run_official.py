"""Run the unmodified official MiMo-V2.5-ASR implementation (HF transformers 4.49).

Requires a separate environment with the official requirements
(torch 2.6, transformers 4.49, flash-attn 2.7.4) and a clone of
https://github.com/XiaomiMiMo/MiMo-V2.5-ASR passed as --upstream.

Only the random prompt-template choice is pinned (to the first zh/en template) so
that results are reproducible and comparable with the vLLM adapter.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio-tokenizer", required=True)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--language", choices=["auto", "labeled"], default="auto")
    ap.add_argument("--attn-impl", default=None, help="HF attn_implementation override")
    ap.add_argument("--sdpa-math", action="store_true", help="force the SDPA math kernel (noise floor)")
    args = ap.parse_args()

    sys.path.insert(0, args.upstream)
    from src.mimo_audio import mimo_audio as official

    rows = [json.loads(l) for m in args.manifest for l in Path(m).read_text().splitlines()]
    rows = rows[args.shard::args.num_shards][: args.limit]
    if args.sdpa_math:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
    if args.attn_impl:
        orig_fp = official.MiMoAudioForCausalLM.from_pretrained
        official.MiMoAudioForCausalLM.from_pretrained = (
            lambda *a, **k: orig_fp(*a, attn_implementation=args.attn_impl, **k))
    model = official.MimoAudio(args.model, args.audio_tokenizer)

    capture = {}
    orig_generate = model.model.generate
    orig_pre = model.preprocess_input

    def generate(inputs, generation_config, **kw):
        out = orig_generate(inputs, generation_config, **kw)
        capture["tokens"] = out[:, inputs.shape[1]:].reshape(-1, 9)[::4, 0].tolist()
        return out

    def preprocess(x):
        codes = orig_pre(x)
        capture["codes"] = codes.reshape(-1, model.audio_channels).to(torch.int16).clone()
        return codes

    model.model.generate = generate
    model.preprocess_input = preprocess

    def run(row):
        lang = row["language"] if args.language == "labeled" else "auto"
        tpl = "Please transcribe this audio file" if lang == "en" else "请将这段语音转换为文字"
        official.asr_zh_templates[:] = [tpl]
        official.asr_en_templates[:] = [tpl]
        tag = {"auto": "", "zh": "<chinese>", "en": "<english>"}[lang]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        text = model.asr_sft(row["audio"], tag)
        torch.cuda.synchronize()
        return text, time.perf_counter() - t0

    run(rows[0])  # warm-up (excluded)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    codes = {}
    with out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            text, sec = run(row)
            codes[row["id"]] = capture["codes"]
            f.write(json.dumps({**row, "text": text, "seconds": sec, "tokens": capture["tokens"]},
                               ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"[shard {args.shard}] {i + 1}/{len(rows)} {sec:.3f}s {text[:40]}", flush=True)
    torch.save(codes, out.with_suffix(".codes.pt"))


if __name__ == "__main__":
    main()
