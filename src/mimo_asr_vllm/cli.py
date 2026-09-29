"""mimo-asr: MiMo-V2.5-ASR accelerated by vLLM.

  mimo-asr transcribe a.wav b.mp3 ...         batch transcription (JSONL output)
  mimo-asr serve [vllm serve options]         OpenAI-compatible server
                                              (same as `vllm serve XiaomiMiMo/MiMo-V2.5-ASR`)
"""

from __future__ import annotations

import argparse
import os
import json
import sys
import time
from pathlib import Path

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".webm", ".wma", ".mp4"}


def _collect_inputs(args) -> list[str]:
    files: list[str] = []
    for item in args.audio:
        p = Path(item)
        if p.is_dir():  # a directory: all audio files below it
            files += sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXTS)
        else:
            files.append(item)
    if args.input_list:
        text = Path(args.input_list).read_text(encoding="utf-8")
        files += [line.strip() for line in text.splitlines() if line.strip()]
    if not files:
        raise SystemExit("mimo-asr: no input audio (pass files/directories or --input-list)")
    return files


def transcribe_main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="mimo-asr transcribe",
                                description="Batch transcription with MiMo-V2.5-ASR on vLLM")
    p.add_argument("audio", nargs="*", help="audio files or directories")
    p.add_argument("-i", "--input-list", help="text file with one audio path per line")
    p.add_argument("-o", "--output", help="JSONL output file (default: stdout)")
    p.add_argument("--language", default="auto", help="auto (default), zh or en")
    p.add_argument("--model", default="XiaomiMiMo/MiMo-V2.5-ASR")
    p.add_argument("--audio-tokenizer", default=None,
                   help="MiMo-Audio-Tokenizer dir or repo (default: $MIMO_AUDIO_TOKENIZER_PATH "
                        "or XiaomiMiMo/MiMo-Audio-Tokenizer)")
    p.add_argument("--max-clip-s", default="default",
                   help="split longer audio at pauses into clips of this many seconds "
                        "(default 60; 'none' = whole files)")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=2048,
                   help="files submitted to vLLM at once (bounds host memory)")
    p.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--quantization", default=None, help="e.g. fp8 (weights of the LLM decoder)")
    p.add_argument("--enforce-eager", action="store_true")
    args = p.parse_args(argv)

    files = _collect_inputs(args)
    # vLLM logs to stdout by default, which would corrupt the JSONL output;
    # must be set before vLLM is imported (also inherited by engine processes).
    os.environ.setdefault("VLLM_LOGGING_STREAM", "ext://sys.stderr")
    from .asr import MiMoASR

    max_clip = args.max_clip_s
    if max_clip != "default":
        max_clip = None if max_clip.lower() in ("none", "0", "off") else float(max_clip)
    extra = {"quantization": args.quantization} if args.quantization else {}
    asr = MiMoASR(args.model, args.audio_tokenizer, max_clip_s=max_clip,
                  tensor_parallel_size=args.tensor_parallel_size,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
                  enforce_eager=args.enforce_eager, **extra)
    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    start, audio_s, failed, truncated = time.perf_counter(), 0.0, 0, 0
    try:
        for i in range(0, len(files), args.batch_size):
            batch = files[i:i + args.batch_size]
            results = asr.transcribe_batch(batch, language=args.language, max_tokens=args.max_tokens,
                                           on_error="skip", use_tqdm=len(files) > 1)
            for f, r in zip(batch, results):
                out.write(json.dumps({"audio": f, **r.to_dict()}, ensure_ascii=False) + "\n")
                audio_s += r.duration
                failed += r.finish_reason == "error"
                truncated += r.finish_reason == "length"
            out.flush()
    finally:
        if out is not sys.stdout:
            out.close()
        asr.close()
    elapsed = time.perf_counter() - start
    print(f"[mimo-asr] {len(files)} files, {audio_s:.1f}s audio in {elapsed:.1f}s "
          f"(RTFx {audio_s / max(elapsed, 1e-9):.0f})", file=sys.stderr)
    if failed:
        print(f"[mimo-asr] {failed} file(s) could not be read (see 'error' field)", file=sys.stderr)
    if truncated:
        print(f"[mimo-asr] {truncated} output(s) hit --max-tokens and are truncated", file=sys.stderr)
    if failed == len(files):
        raise SystemExit(1)


def serve_main(argv: list[str]) -> None:
    """``vllm serve`` with MiMo defaults; any vLLM option can be appended."""
    model = "XiaomiMiMo/MiMo-V2.5-ASR"
    if argv and not argv[0].startswith("-"):
        model, argv = argv[0], argv[1:]
    if not any(a == "--served-model-name" or a.startswith("--served-model-name=") for a in argv):
        # Clients can use the repo id or the short alias "mimo-asr".
        argv = ["--served-model-name", model, "mimo-asr", *argv]
    from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv = ["vllm", "serve", model, *argv]
    vllm_main()


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    if argv[0] == "serve":
        return serve_main(argv[1:])
    if argv[0] == "transcribe":
        return transcribe_main(argv[1:])
    # `mimo-asr a.wav` is a shortcut for `mimo-asr transcribe a.wav`
    return transcribe_main(argv)


if __name__ == "__main__":
    main()
