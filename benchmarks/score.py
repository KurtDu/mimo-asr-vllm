"""Score ASR outputs (CER for zh, WER for en) and compare systems.

usage: score.py name1=results1.jsonl [name2=results2.jsonl ...]
The first system is the reference for the exact-match comparison.
"""
import json, re, string, sys, unicodedata
from collections import defaultdict
import jiwer
from zhon.hanzi import punctuation as zh_punct

PUNCT = set(string.punctuation + zh_punct + "“”‘’、。，！？；：（）《》【】…—·")


def norm_zh(t):
    t = unicodedata.normalize("NFKC", t).lower()
    return "".join(c for c in t if c not in PUNCT and not c.isspace())


def norm_en(t):
    t = unicodedata.normalize("NFKC", t).lower().replace("-", " ")
    t = "".join(" " if (c in PUNCT and c != "'") else c for c in t)
    return re.sub(r"\s+", " ", t).strip()


def subset(row):
    return row["id"].split("/")[0]


def score(rows):
    by = defaultdict(lambda: {"ref": [], "hyp": [], "lang": None})
    for r in rows:
        for key in (subset(r), "ALL-" + r["language"]):
            d = by[key]; d["lang"] = r["language"]
            if r["language"] == "zh":
                d["ref"].append(" ".join(norm_zh(r["reference"]))); d["hyp"].append(" ".join(norm_zh(r["text"])))
            else:
                d["ref"].append(norm_en(r["reference"])); d["hyp"].append(norm_en(r["text"]))
    out = {}
    for k, d in sorted(by.items()):
        hyp = [h if h else "<empty>" for h in d["hyp"]]
        out[k] = (jiwer.wer(d["ref"], hyp), len(d["ref"]), "CER" if d["lang"] == "zh" else "WER")
    return out


def main():
    systems = [a.split("=", 1) for a in sys.argv[1:]]
    data = {n: {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8")} for n, p in systems}
    common = set.intersection(*[set(d) for d in data.values()])
    print(f"{len(common)} utterances common to all systems")
    scores = {n: score([d[i] for i in common]) for n, d in data.items()}
    keys = sorted(next(iter(scores.values())))
    print(f"{'subset':14s} {'n':>5s} " + " ".join(f"{n:>16s}" for n, _ in systems))
    for k in keys:
        _, cnt, metric = scores[systems[0][0]][k]
        print(f"{k:14s} {cnt:5d} " + " ".join(f"{metric} {100 * scores[n][k][0]:8.3f}%" for n, _ in systems))
    ref_name = systems[0][0]
    for n, _ in systems[1:]:
        same = sum(data[n][i]["text"] == data[ref_name][i]["text"] for i in common)
        print(f"exact transcript match {n} vs {ref_name}: {same}/{len(common)} = {100 * same / len(common):.2f}%")


if __name__ == "__main__":
    main()
