#!/usr/bin/env python3
"""Skriver en kort oversikt over alle spørsmål i alle sett (stamme + riktig svar), for å unngå duplikater.
Bruk: python3 tools/indeks.py [--part N]"""
import glob, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import parse, plain
only = int(sys.argv[sys.argv.index("--part")+1]) if "--part" in sys.argv else None
for f in sorted(glob.glob(str(Path(__file__).resolve().parent.parent / "src/eksamen_*.txt"))):
    nr = int(re.search(r"(\d+)", Path(f).stem).group(1))
    for pi, p in enumerate(parse(f), 1):
        if only and pi != only: continue
        for qi, q in enumerate(p, 1):
            stem = plain(" ".join(l for l in q["stem"] if not l.strip().startswith("%%%")))
            print(f"S{nr} {pi}.{qi}: {stem[:230]}  ==> {plain(q['right'])[:110]}")
