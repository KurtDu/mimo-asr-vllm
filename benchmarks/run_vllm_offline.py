"""Offline throughput / accuracy run of mimo_asr_vllm.MiMoASR on manifests."""
import argparse, json, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio-tokenizer", required=True)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--language", choices=["auto", "labeled"], default="auto")
    ap.add_argument("--max-clip-s", default="none")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--sequential", action="store_true", help="one request at a time (latency mode)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()
    from mimo_asr_vllm import MiMoASR

    rows = [json.loads(l) for m in args.manifest for l in Path(m).read_text().splitlines()][: args.limit]
    clip = None if args.max_clip_s.lower() == "none" else float(args.max_clip_s)
    t0 = time.perf_counter()
    extra = {"quantization": args.quantization} if args.quantization else {}
    asr = MiMoASR(args.model, args.audio_tokenizer, max_clip_s=clip, max_num_seqs=args.max_num_seqs,
                  enforce_eager=args.enforce_eager, tensor_parallel_size=args.tp, dtype=args.dtype, **extra)
    load_s = time.perf_counter() - t0
    langs = [r["language"] if args.language == "labeled" else None for r in rows]
    asr.transcribe_batch([rows[0]["audio"]])  # warm-up
    t0 = time.perf_counter()
    if args.sequential:
        res, secs = [], []
        for r, lang in zip(rows, langs):
            t = time.perf_counter()
            res.append(asr.transcribe(r["audio"], language=lang))
            secs.append(time.perf_counter() - t)
    else:
        res = asr.transcribe_batch([r["audio"] for r in rows], language=langs, use_tqdm=True)
        secs = [None] * len(rows)
    wall = time.perf_counter() - t0
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r, o, s in zip(rows, res, secs):
            f.write(json.dumps({**r, "text": o.text, "detected": o.language, "num_tokens": o.num_tokens,
                                "finish_reason": o.finish_reason, "seconds": s}, ensure_ascii=False) + "\n")
    audio = sum(r["duration"] for r in rows)
    summary = {"n": len(rows), "audio_s": audio, "wall_s": wall, "rtfx": audio / wall, "load_s": load_s,
               "mode": "sequential" if args.sequential else "batch", "max_num_seqs": args.max_num_seqs}
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=1))
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
