#!/usr/bin/env python3
"""Lager en «blind» tekstversjon av et sett (samme stokking/bokstaver som PDF-en, uten fasit),
og en separat fasitfil. Brukes til uavhengig review.

Bruk: python3 blind.py <utmappe> src/eksamen_01.txt [...]
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import parse, positions_for, LETTERS  # noqa: E402


def export(path, outdir):
    nr = int(re.search(r"(\d+)", Path(path).stem).group(1))
    parts = parse(path)
    nq = sum(len(p) for p in parts)
    pos, rng = positions_for(nr, nq)  # identisk med build.to_tex
    lines, key = [f"ØVINGSSETT {nr}\n"], []
    k = 0
    for pi, p in enumerate(parts, 1):
        lines.append(f"\n########## Part {pi}\n")
        for qi, q in enumerate(p, 1):
            wrong = q["wrong"][:]
            rng.shuffle(wrong)
            opts = wrong[:]
            opts.insert(pos[k], q["right"])
            key.append(f"{pi}.{qi} {LETTERS[pos[k]]}")
            k += 1
            stem = "\n".join(l for l in q["stem"] if not l.strip().startswith("%%%")).strip()
            lines.append(f"\n--- Spørsmål {pi}.{qi}\n{stem}\n")
            for L, o in zip(LETTERS, opts):
                lines.append(f"  ({L}) {o}\n")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"blind_{nr:02d}.txt").write_text("".join(lines), encoding="utf-8")
    (out / f"fasit_{nr:02d}.txt").write_text("\n".join(key) + "\n", encoding="utf-8")


if __name__ == "__main__":
    for f in sys.argv[2:]:
        export(f, sys.argv[1])
