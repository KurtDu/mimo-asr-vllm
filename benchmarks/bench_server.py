"""Concurrent load test of an OpenAI-compatible /v1/audio/transcriptions endpoint."""
import argparse, asyncio, json, statistics, time
from pathlib import Path
import httpx


async def one(client, url, model, row, language, sem, results):
    async with sem:
        data = {"model": model, "response_format": "json", "temperature": "0"}
        if language:
            data["language"] = language
        audio = Path(row["audio"]).read_bytes()
        t = time.perf_counter()
        try:
            r = await client.post(url, data=data, files={"file": (Path(row["audio"]).name, audio)})
            r.raise_for_status()
        except Exception as e:  # record, do not abort the whole run
            results.append({**row, "error": f"{type(e).__name__}: {e}"[:300]})
            return
        dt = time.perf_counter() - t
        results.append({**row, "text": r.json()["text"], "seconds": dt})


async def run(args, rows):
    sem = asyncio.Semaphore(args.concurrency)
    results = []
    url = args.base_url.rstrip("/") + "/v1/audio/transcriptions"
    async with httpx.AsyncClient(timeout=3600, limits=httpx.Limits(max_connections=args.concurrency + 8)) as c:
        await one(c, url, args.model, rows[0], None, asyncio.Semaphore(1), [])  # warm-up
        t = time.perf_counter()
        await asyncio.gather(*[one(c, url, args.model, r, r["language"] if args.language == "labeled" else None,
                                   sem, results) for r in rows])
        wall = time.perf_counter() - t
    return results, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--language", choices=["auto", "labeled"], default="auto")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for m in args.manifest for l in Path(m).read_text().splitlines()][: args.limit]
    results, wall = asyncio.run(run(args, rows))
    errors = [r for r in results if "error" in r]
    for e in errors[:3]:
        print("ERROR", e["id"], e["error"])
    results = [r for r in results if "error" not in r]
    lat = sorted(r["seconds"] for r in results)
    audio = sum(r["duration"] for r in results)
    summary = {"concurrency": args.concurrency, "n": len(results), "errors": len(errors), "audio_s": round(audio, 1),
               "wall_s": round(wall, 2), "rtfx": round(audio / wall, 1), "req_per_s": round(len(results) / wall, 2),
               "lat_mean": round(statistics.mean(lat), 3), "lat_p50": round(lat[len(lat) // 2], 3),
               "lat_p95": round(lat[int(len(lat) * 0.95) - 1], 3), "lat_p99": round(lat[int(len(lat) * 0.99) - 1], 3)}
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path(args.output + ".summary.json").write_text(json.dumps(summary, indent=1))
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
