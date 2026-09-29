"""Build the evaluation manifests used in the README (deterministic).

usage: python make_manifest.py <eval_data_root> <out_dir>

<eval_data_root> must contain seed-tts-eval/{zh,en} (as distributed on HF by
the sglang-omni team) and AISHELL-3/test. Writes JSONL files with
{id, audio, reference, language, duration}:
  seed_zh / seed_en / aishell3 (500) : short utterances
  long   : 24 recordings of 35-99 s (concatenated utterances)
  long2  : 12 recordings of 2-5 min
  long3  : 40 recordings of 2-7 min
"""
import hashlib, json, random, sys
from pathlib import Path

import numpy as np
import soundfile as sf


def seed(root, lang):
    rows = []
    for line in (root / "seed-tts-eval" / lang / "meta.lst").read_text().splitlines():
        utt, _, _, text = line.split("|")[:4]
        p = root / "seed-tts-eval" / lang / "wavs" / f"{utt}.wav"
        if p.exists():
            rows.append({"id": f"seed-{lang}/{utt}", "audio": str(p), "reference": text, "language": lang})
    return rows


def aishell(root, n):
    base = root / "AISHELL-3" / "test"
    rows = []
    for line in (base / "content.txt").read_text().splitlines():
        name, words = line.split("\t", 1)
        p = base / "wav" / name[:7] / name
        if p.exists():
            rows.append({"id": "aishell3/" + name, "audio": str(p),
                         "reference": "".join(words.split()[::2]), "language": "zh"})
    return sorted(rows, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[:n]


def concat(name, pools, n, lo, hi, gap, seed_, out):
    """Concatenate random utterances (resampled to 24 kHz) into long recordings."""
    import torch
    import torchaudio.functional as AF

    rng = random.Random(seed_)
    wav_dir = out / f"{name}_wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        lang = "zh" if i % 2 == 0 else "en"
        target, parts, texts, dur = rng.uniform(lo, hi), [], [], 0.0
        while dur < target:
            r = rng.choice(pools[lang])
            w, sr = sf.read(r["audio"], dtype="float32")
            w = w.mean(1) if w.ndim > 1 else w
            if sr != 24000:
                w = AF.resample(torch.from_numpy(w), sr, 24000).numpy()
            g = gap[0] if gap[0] == gap[1] else rng.uniform(*gap)
            parts += [w, np.zeros(int(g * 24000), np.float32)]
            texts.append(r["reference"])
            dur += len(w) / 24000 + g
        p = wav_dir / f"{name}_{i:03d}.wav"
        sf.write(p, np.concatenate(parts), 24000)
        rows.append({"id": f"{name}/{i:03d}", "audio": str(p), "language": lang,
                     "reference": ("" if lang == "zh" else " ").join(texts)})
    return rows


def main():
    root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    zh, en = seed(root, "zh"), seed(root, "en")
    ai = aishell(root, 500)
    sets = {
        "seed_zh": zh, "seed_en": en, "aishell3": ai,
        "long": concat("long", {"zh": zh, "en": en}, 24, 35, 95, (0.3, 0.3), 0, out),
        "long2": concat("long2", {"zh": zh, "en": en}, 12, 120, 300, (0.2, 0.8), 7, out),
        "long3": concat("long3", {"zh": zh + ai, "en": en}, 40, 120, 420, (0.15, 1.0), 11, out),
    }
    for name, rows in sets.items():
        for r in rows:
            r["duration"] = sf.info(r["audio"]).duration
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{name}: {len(rows)} files, {sum(r['duration'] for r in rows) / 3600:.2f} h")


if __name__ == "__main__":
    main()
