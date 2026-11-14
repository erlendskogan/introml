#!/usr/bin/env python3
"""Validerer kodekurset ved å faktisk kjøre koden.

Sjekker for hver leksjon:
  * hver ```python-blokk (kjøres i nettleseren) kjører uten feil, raskt, og bruker bare
    pakker som finnes i Pyodide (ikke torch/shap/gymnasium/keras/tensorflow/lime)
  * hver oppgave: løsning + test består, og startkode + test består IKKE
  * hver ```python statisk-blokk (uten ikkekjor) kjører uten feil lokalt (torch/shap osv.)
  * hver quiz har nøyaktig ett riktig svar (gjøres av build.py)

Bruk:
  <venv>/bin/python verktoy/valider.py              # alt
  <venv>/bin/python verktoy/valider.py 03-pandas    # bare filer som matcher
  <venv>/bin/python verktoy/valider.py --rask       # hopp over statiske blokker
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("PYTHONWARNINGS", "ignore")

HER = Path(__file__).resolve().parent
sys.path.insert(0, str(HER))
import build  # noqa: E402
import runner  # noqa: E402

IKKE_I_PYODIDE = re.compile(r"^\s*(import|from)\s+(torch|shap|gymnasium|gym|keras|tensorflow|lime|torchvision)\b", re.M)
FORBUDT = re.compile(r"\binput\s*\(|time\.sleep\(")
MAKS_SEK = 4.0


def kjor(src, test=None):
    t0 = time.time()
    res = json.loads(runner.kk_run(src, test))
    return res, time.time() - t0


def tekst_ut(res):
    return "".join(e[1] for e in res["events"] if e[0] in ("text", "warn"))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rask = "--rask" in sys.argv
    vis = "--vis" in sys.argv
    try:
        if args:
            deler = [build.parse_fil(p) for p in sorted(build.KILDE.glob("*.md")) if any(a in p.name for a in args)]
        else:
            deler = build.les_kurs()
    except build.Feil as e:
        print("PARSEFEIL:", e)
        sys.exit(1)

    arbeidsmappe = Path(tempfile.mkdtemp(prefix="kk_valider_"))
    for navn, sti in build.DATASETT.items():
        shutil.copy(sti, arbeidsmappe / navn)
    os.chdir(arbeidsmappe)

    feil, advarsler, antall = [], [], 0
    for d in deler:
        if args and not any(a in d["fil"] for a in args):
            continue
        for l in d["leksjoner"]:
            sted = f"{d['fil']} {l['nr']} «{l['tittel']}»"
            # kode inni merk-bokser tas med på sin plass i rekkefølgen
            blokker = []
            for b in l["blokker"]:
                blokker.extend(b["blokker"] if b["t"] == "merk" else [b])
            statisk_hittil = []
            for b in blokker:
                if b["t"] == "kode":
                    antall += 1
                    if IKKE_I_PYODIDE.search(b["kode"]):
                        feil.append(f"{sted} linje {b['linje']}: kjørbar blokk importerer pakke som ikke finnes i nettleseren – bruk ```python statisk")
                        continue
                    if FORBUDT.search(b["kode"]):
                        feil.append(f"{sted} linje {b['linje']}: input()/sleep er ikke lov i kjørbare blokker")
                    res, sek = kjor(b["kode"])
                    if b.get("feil"):
                        if res["status"] != "feil":
                            feil.append(f"{sted} linje {b['linje']}: ```python feil-blokken skulle gitt en feilmelding, men kjørte fint")
                        elif isinstance(b["feil"], str) and b["feil"] + ":" not in (res["feil"] or ""):
                            feil.append(f"{sted} linje {b['linje']}: ventet {b['feil']}, fikk:\n      {res['feil']}")
                    elif res["status"] != "ok":
                        feil.append(f"{sted} linje {b['linje']}: eksempelet krasjer:\n      {res['feil']}")
                    elif sek > MAKS_SEK:
                        advarsler.append(f"{sted} linje {b['linje']}: tok {sek:.1f} s (treg i nettleseren)")
                    if vis:
                        print(f"--- {sted} linje {b['linje']}\n{tekst_ut(res)}")
                elif b["t"] == "oppgave":
                    antall += 1
                    for k in ("start", "losning", "test"):
                        if IKKE_I_PYODIDE.search(b[k]):
                            feil.append(f"{sted} oppgave «{b['tittel']}»: ```{k} importerer pakke som ikke finnes i nettleseren")
                    res, sek = kjor(b["losning"], b["test"])
                    if res["status"] != "bestatt":
                        feil.append(f"{sted} oppgave «{b['tittel']}» (linje {b['linje']}): løsningen består ikke testen: {res['status']}\n      {res['feil']}")
                    elif sek > MAKS_SEK:
                        advarsler.append(f"{sted} oppgave «{b['tittel']}»: løsningen tok {sek:.1f} s")
                    res2, _ = kjor(b["start"], b["test"])
                    if res2["status"] == "bestatt":
                        feil.append(f"{sted} oppgave «{b['tittel']}» (linje {b['linje']}): startkoden består testen allerede – testen er for svak")
                    if res2["status"] == "feil" and "SyntaxError" in (res2["feil"] or ""):
                        advarsler.append(f"{sted} oppgave «{b['tittel']}»: startkoden har syntaksfeil (ok hvis det er meningen):\n      {res2['feil']}")
                elif b["t"] == "statisk" and b["kjor"] and not rask:
                    antall += 1
                    kode = "\n".join(statisk_hittil + [b["kode"]]) if b["fortsett"] else b["kode"]
                    statisk_hittil = (statisk_hittil if b["fortsett"] else []) + [b["kode"]]
                    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=arbeidsmappe) as f:
                        f.write("import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nplt.show = lambda *a, **k: plt.close('all')\n")
                        f.write(kode)
                        sti = f.name
                    try:
                        p = subprocess.run([sys.executable, sti], capture_output=True, text=True, timeout=300)
                        if p.returncode != 0:
                            siste = "\n      ".join(p.stderr.strip().splitlines()[-4:])
                            feil.append(f"{sted} linje {b['linje']}: statisk blokk krasjer lokalt:\n      {siste}")
                        elif vis:
                            print(f"--- STATISK {sted} linje {b['linje']}\n{p.stdout}")
                    except subprocess.TimeoutExpired:
                        advarsler.append(f"{sted} linje {b['linje']}: statisk blokk brukte over 300 s")
    print(f"Kjørte {antall} blokker.")
    for a in advarsler:
        print("ADVARSEL:", a)
    for f in feil:
        print("FEIL:", f)
    print(f"{len(feil)} feil, {len(advarsler)} advarsler.")
    shutil.rmtree(arbeidsmappe, ignore_errors=True)
    sys.exit(1 if feil else 0)


if __name__ == "__main__":
    main()
