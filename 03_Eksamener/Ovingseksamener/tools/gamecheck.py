#!/usr/bin/env python3
"""Tester om et øvingssett kan «games»: bruker typiske gjettestrategier som ikke krever
fagkunnskap og regner ut forventet antall riktige (av 40). Tilfeldig gjetting gir 10.

Bruk: python3 gamecheck.py src/eksamen_01.txt [...]
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import parse, plain, ABSOLUTE  # noqa: E402

HEDGE = re.compile(r"\b(kan|typisk|ofte|vanligvis|gjerne|som regel|i hovedsak|tilnærmet|omtrent|sjelden)\b", re.I)
WORD = re.compile(r"[A-Za-zÆØÅæøå0-9]{3,}")
NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def words(t):
    return set(w.lower() for w in WORD.findall(plain(t)))


def pick(scores, maximize=True):
    """Returnerer liste over indekser med beste score (uavgjort -> deling)."""
    best = max(scores) if maximize else min(scores)
    return [i for i, s in enumerate(scores) if s == best]


def expected_hit(winners, correct=0):
    return (1.0 / len(winners)) if correct in winners else 0.0


def strategies(q):
    opts = [q["right"]] + q["wrong"]  # indeks 0 er riktig
    P = [plain(o) for o in opts]
    stem = words("\n".join(q["stem"]))
    W = [words(o) for o in opts]
    L = [len(p) for p in P]
    res = {}
    res["lengst"] = pick(L)
    res["kortest"] = pick(L, False)
    mean = sum(L) / 4
    res["lengde-avviker mest"] = pick([abs(x - mean) for x in L])
    A = [len(ABSOLUTE.findall(p)) for p in P]
    res["færrest absolutte ord"] = pick(A, False)
    H = [len(HEDGE.findall(p)) for p in P]
    res["flest forbehold (kan/typisk)"] = pick(H)
    ov = [len(w & stem) for w in W]
    res["mest ordlikhet med stammen"] = pick(ov)
    # konvergens: alternativet som deler flest ord med de andre alternativene
    conv = []
    for i in range(4):
        conv.append(sum(len(W[i] & W[j]) / max(1, len(W[i] | W[j])) for j in range(4) if j != i))
    res["konvergens (likest de andre)"] = pick(conv)
    res["minst lik de andre"] = pick(conv, False)
    # numerisk midtverdi
    nums = []
    for p in P:
        m = NUM.findall(p)
        nums.append(float(m[0].replace(",", ".")) if len(m) == 1 or (m and len(p) < 40) else None)
    if all(n is not None for n in nums):
        srt = sorted(nums)
        mids = [srt[1], srt[2]]
        res["tall: en av de to midterste"] = [i for i, n in enumerate(nums) if n in mids]
    return res


def main(files):
    for f in files:
        parts = parse(f)
        tot = {}
        n_num = 0
        for p in parts:
            for q in p:
                for name, win in strategies(q).items():
                    tot.setdefault(name, [0.0, 0])
                    tot[name][0] += expected_hit(win)
                    tot[name][1] += 1
        print(f"== {f}")
        if "--tall" in sys.argv:
            for pi, p in enumerate(parts, 1):
                for qi, q in enumerate(p, 1):
                    st = strategies(q)
                    if "tall: en av de to midterste" in st:
                        print(f"   tall {pi}.{qi}: riktig blant de to midterste = {0 in st['tall: en av de to midterste']}")
        for name, (hit, n) in tot.items():
            flag = "  <-- FOR HØY" if hit / n > 0.375 and n >= 8 else ""
            print(f"  {name:32s} {hit:5.1f} / {n:2d}  ({100*hit/n:4.0f} %){flag}")


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")])
