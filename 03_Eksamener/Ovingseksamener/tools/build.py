#!/usr/bin/env python3
"""Bygger øvingseksamener for TDT4172 fra en enkel markup-fil.

Markup (src/eksamen_XX.txt):

    @@PART
    @@Q
    <stamme i LaTeX, flere linjer>
    @INSTR Velg ett alternativ        (valgfri, default "Velg ett alternativ:")
    @+ riktig alternativ (én linje LaTeX)
    @- feil alternativ
    @- feil alternativ
    @- feil alternativ

Byggeskriptet stokker alternativene slik at riktig svar havner nøyaktig like
ofte på A, B, C og D i hvert sett, og genererer fasiten automatisk.

Bruk:
    python3 build.py check src/eksamen_01.txt [...]   # statistikk + flagg
    python3 build.py build src/eksamen_01.txt [...]   # lag .tex og .pdf
"""
import os
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
LETTERS = "ABCD"

ABSOLUTE = re.compile(
    r"\b(alltid|aldri|garanter\w*|kun|bare|umulig|ingen|alle|fullstendig|nøyaktig|"
    r"utelukkende|helt|må|nødvendigvis)\b", re.I)


# ---------------------------------------------------------------- parsing
def parse(path):
    parts, part, q = [], None, None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\n")
        s = line.strip()
        if s.startswith("%%%"):  # forfatterkommentar, ignoreres
            continue
        if s == "@@PART":
            part = []
            parts.append(part)
            q = None
        elif s == "@@Q":
            if part is None:
                raise SystemExit(f"{path}: @@Q før @@PART")
            q = {"stem": [], "right": None, "wrong": [], "instr": None}
            part.append(q)
        elif s.startswith("@+"):
            if q["right"] is not None:
                raise SystemExit(f"{path}: to riktige alternativer i et spørsmål")
            q["right"] = s[2:].strip()
        elif s.startswith("@-"):
            q["wrong"].append(s[2:].strip())
        elif s.startswith("@INSTR"):
            q["instr"] = s[len("@INSTR"):].strip()
        else:
            if q is None:
                if s:
                    raise SystemExit(f"{path}: tekst utenfor spørsmål: {s[:60]}")
                continue
            if q["right"] is not None or q["wrong"]:
                if s:
                    raise SystemExit(
                        f"{path}: stammetekst etter alternativer (alternativer må være én linje): {s[:60]}")
                continue
            q["stem"].append(line)
    # validering
    errs = []
    if len(parts) != 10:
        errs.append(f"forventet 10 parts, fant {len(parts)}")
    for pi, p in enumerate(parts, 1):
        if len(p) != 4:
            errs.append(f"Part {pi}: forventet 4 spørsmål, fant {len(p)}")
        for qi, q in enumerate(p, 1):
            if q["right"] is None:
                errs.append(f"{pi}.{qi}: mangler riktig alternativ (@+)")
            if len(q["wrong"]) != 3:
                errs.append(f"{pi}.{qi}: forventet 3 feil alternativer, fant {len(q['wrong'])}")
    if errs:
        raise SystemExit(f"{path}:\n  " + "\n  ".join(errs))
    return parts


# ---------------------------------------------------------------- analyse
def plain(tex):
    t = re.sub(r"\\(texttt|textbf|emph|textit|mathrm|text|mathbf)\{", "{", tex)
    t = re.sub(r"\\[a-zA-Z]+", "x", t)  # makroer teller som ett tegn
    t = re.sub(r"[{}$\\^_]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def check(path, verbose=True):
    parts = parse(path)
    n = 0
    longest = 0
    shortest = 0
    ratios = []
    flags = []
    abs_right = abs_wrong = 0
    for pi, p in enumerate(parts, 1):
        for qi, q in enumerate(p, 1):
            n += 1
            r = len(plain(q["right"]))
            ws = [len(plain(w)) for w in q["wrong"]]
            mean_w = sum(ws) / 3
            ratios.append(r / mean_w if mean_w else 1)
            if r > max(ws):
                longest += 1
                if r > 1.2 * max(ws) and r > 25:
                    flags.append(f"{pi}.{qi}: riktig alternativ er klart lengst ({r} vs maks feil {max(ws)})")
            if r < min(ws):
                shortest += 1
                if r < 0.8 * min(ws) and min(ws) > 25:
                    flags.append(f"{pi}.{qi}: riktig alternativ er klart kortest ({r} vs min feil {min(ws)})")
            allv = [r] + ws
            if max(allv) > 2.2 * max(1, min(allv)) and max(allv) > 30:
                flags.append(f"{pi}.{qi}: stor lengdespredning i alternativene {allv}")
            abs_right += len(ABSOLUTE.findall(plain(q["right"])))
            abs_wrong += sum(len(ABSOLUTE.findall(plain(w))) for w in q["wrong"])
            # ord som bare finnes i stammen og i riktig alternativ
            stem_words = set(w.lower() for w in re.findall(r"[A-Za-zÆØÅæøå]{6,}", plain("\n".join(q["stem"]))))
            rw = set(w.lower() for w in re.findall(r"[A-Za-zÆØÅæøå]{6,}", plain(q["right"])))
            ww = set()
            for w in q["wrong"]:
                ww |= set(x.lower() for x in re.findall(r"[A-Za-zÆØÅæøå]{6,}", plain(w)))
            echo = (rw & stem_words) - ww
            if len(echo) >= 2:
                flags.append(f"{pi}.{qi}: riktig alternativ gjentar stamme-ord som ingen distraktorer har: {sorted(echo)}")
    if verbose:
        print(f"== {path}")
        print(f"  spørsmål: {n}")
        print(f"  riktig er lengst: {longest}/{n} ({100*longest/n:.0f} %, tilfeldig ~25 %)")
        print(f"  riktig er kortest: {shortest}/{n} ({100*shortest/n:.0f} %, tilfeldig ~25 %)")
        print(f"  snitt lengde riktig / snitt feil: {sum(ratios)/n:.2f} (mål ~1.00)")
        print(f"  absolutte ord: per riktig alt. {abs_right/n:.2f}, per feil alt. {abs_wrong/(3*n):.2f}")
        for f in flags:
            print("  FLAGG", f)
    return parts


# ---------------------------------------------------------------- LaTeX
PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[norsk]{babel}
\usepackage{tgtermes}
\usepackage{tgheros}
\renewcommand{\ttdefault}{txtt}
\usepackage{amsmath,amssymb}
\usepackage[a4paper,left=2.45cm,right=2.6cm,top=2.3cm,bottom=2.2cm,headsep=0.9cm]{geometry}
\usepackage{fancyhdr}
\usepackage{lastpage}
\usepackage{xcolor}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{booktabs}
\usepackage{array}
\usepackage{enumitem}
\usepackage{listings}
\usepackage{needspace}
\usepackage{multirow}
\usetikzlibrary{positioning,arrows.meta,calc}
\definecolor{insgrey}{gray}{0.42}
\definecolor{radiogrey}{gray}{0.70}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.45em}
\setlist[itemize]{itemsep=0pt,topsep=2pt,label=\textbullet,font=\sffamily}
\lstset{basicstyle=\ttfamily\small,columns=fullflexible,keepspaces=true,
  showstringspaces=false,upquote=true,aboveskip=0.6em,belowskip=0.6em,
  literate={æ}{{\ae}}1 {ø}{{\o}}1 {å}{{\aa}}1 {Æ}{{\AE}}1 {Ø}{{\O}}1 {Å}{{\AA}}1}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyhead[R]{\fontsize{7.5}{9}\selectfont\sffamily\color{insgrey}TDT4172 - Introduksjon til maskinlæring - Øvingssett @@NR@@}
\fancyfoot[R]{\fontsize{7}{9}\selectfont\sffamily\color{insgrey}\thepage/\pageref{LastPage}}
\newcommand{\delstart}[1]{\clearpage
  \noindent\hspace*{-1.55em}\makebox[1.55em][l]{\sffamily\bfseries\fontsize{12}{14}\selectfont\raisebox{0.35em}{#1}}%
  {\sffamily\bfseries\fontsize{18}{22}\selectfont Part #1}\par\vspace{1.0em}}
\newcommand{\sporsmal}[1]{\par\Needspace{9\baselineskip}\vspace{1.9em}%
  \noindent{\bfseries Spørsmål #1}\par\vspace{-0.45em}%
  \noindent{\bfseries -{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-}\par\vspace{-0.15em}}
\newcommand{\velg}[1]{\par\Needspace{6\baselineskip}\noindent{\bfseries #1}\par\vspace{0.3em}}
\newcommand{\radio}[1]{\tikz[baseline=-0.62ex]{\draw[radiogrey,line width=0.45pt,fill=gray!5] (0,0) circle (0.56em);
  \node[inner sep=0pt,font=\sffamily\fontsize{5.5}{6}\selectfont,text=insgrey] at (0,0) {#1};}}
\newenvironment{alternativer}{\begin{list}{}{\setlength{\leftmargin}{2.35em}\setlength{\labelwidth}{1.4em}%
  \setlength{\labelsep}{0.6em}\setlength{\itemsep}{0.85em}\setlength{\topsep}{0.45em}\setlength{\parsep}{0pt}}%
  \sffamily\fontsize{10}{12.6}\selectfont}{\end{list}}
\newcommand{\alt}[2]{\item[\radio{#1}]#2}
\newcommand{\makspoeng}{\par\vspace{2.2em}\hfill{\sffamily\fontsize{8.5}{10}\selectfont Maks poeng: 10}\hspace*{0.5em}\par}
\begin{document}
"""

FRONT = r"""
\noindent\hspace*{-1.55em}\makebox[1.55em][l]{\raisebox{0.3em}{\sffamily\bfseries\fontsize{11}{12}\selectfont i}}%
{\sffamily\bfseries\fontsize{18}{22}\selectfont Frontpage}\par\vspace{1.1em}

{\bfseries Eksamensoppgave i TDT4172 - Introduksjon til maskinlæring}\\
{\bfseries Øvingssett:} @@NR@@ av 10 (uoffisielt øvingssett)\\
{\bfseries Tid:} 4 timer

\vspace{0.6em}
{\bfseries Faglig kontakt:} --- (øvingssett, ikke en offisiell eksamen)\\
{\bfseries Møter i eksamenslokalet:} Nei

\vspace{0.6em}
{\bfseries Hjelpemiddelkode/Tillatte hjelpemidler:} D / Ingen trykte eller håndskrevne hjelpemidler tillatt.
Bestemt, enkel kalkulator tillatt.

\vspace{0.6em}
{\bfseries\underline{ANNEN INFORMASJON:}}

\vspace{0.4em}
{\bfseries Les oppgavene nøye og gjør dine egne antagelser.} Presiser i besvarelsen hvilke forutsetninger du
har lagt til grunn i tolkning/avgrensing av oppgaven.

Dette øvingssettet er laget for egentrening, med samme format og vanskelighetsgrad som de
siste ordinære eksamenene i emnet. Løsningsforslaget står på siste side.

\vspace{1.6em}
{\bfseries\underline{FAGSPESIFIKK INFORMASJON}}

\vspace{0.4em}
{\bfseries Ingen papirhåndtegninger:}\\
Denne eksamenen tillater ikke bruk av håndtegninger.

\vspace{1.4em}
{\bfseries Vekting av oppgavene}:\\
Denne eksamenen inneholder flervalgsoppgaver organisert i ti grupper på fire spørsmål hver. Hver
gruppe vurderes separat etter følgende regler:

\vspace{0.6em}
For hver av de fire spørsmålene i gruppen får du følgende poeng
\begin{itemize}[leftmargin=2.2em,label=\textperiodcentered,itemsep=0pt]
\item {\bfseries Riktig svar:} +2,5 poeng
\item {\bfseries Feil svar (et alternativ er valgt, men ikke korrekt):} $-$0,83 poeng
\item {\bfseries Ikke besvart:} 0 poeng
\end{itemize}
{\bfseries Merk} at dersom du først har valgt et svaralternativ, kan du {\bfseries ikke velge det bort} igjen. Da må du
svare på det spørsmålet. Med andre ord mister du muligheten til å la et spørsmål stå blankt så fort
du har trykket på et alternativ.

\vspace{0.6em}
Poengsummen for hver gruppe er {\bfseries summen} av poengene for de fire enkeltspørsmålene.\\
For {\bfseries alle spørsmålene} gjelder en {\bfseries null-grense}: dersom summen for denne gruppen blir negativ,
settes verdien til {\bfseries 0}. Den totale poengsummen på eksamen er summen av poengene fra alle
gruppene.
"""


def positions_for(nr, nq):
    rng = random.Random(4172 * 1000 + nr)
    base = [i % 4 for i in range(nq)]
    while True:
        rng.shuffle(base)
        run, ok = 1, True
        for a, b in zip(base, base[1:]):
            run = run + 1 if a == b else 1
            if run > 2:
                ok = False
                break
        if ok:
            return base, rng


def to_tex(parts, nr):
    nq = sum(len(p) for p in parts)
    pos, rng = positions_for(nr, nq)
    out = [PREAMBLE.replace("@@NR@@", str(nr)), FRONT.replace("@@NR@@", str(nr))]
    key = []
    k = 0
    for pi, p in enumerate(parts, 1):
        out.append(f"\n\\delstart{{{pi}}}\n")
        row = []
        for qi, q in enumerate(p, 1):
            wrong = q["wrong"][:]
            rng.shuffle(wrong)
            opts = wrong[:]
            opts.insert(pos[k], q["right"])
            row.append(LETTERS[pos[k]])
            k += 1
            out.append(f"\\sporsmal{{{qi}}}\n")
            out.append("\n".join(q["stem"]).strip() + "\n")
            instr = q["instr"] or "Velg ett alternativ:"
            out.append(f"\\velg{{{instr}}}\n\\begin{{alternativer}}\n")
            for L, o in zip(LETTERS, opts):
                out.append(f"\\alt{{{L}}}{{{o}}}\n")
            out.append("\\end{alternativer}\n")
        out.append("\\makspoeng\n")
        key.append(row)
    # fasit
    out.append(r"""
\clearpage
{\sffamily\bfseries\fontsize{18}{22}\selectfont Løsningsforslag}\par\vspace{0.3em}
\noindent\rule{\textwidth}{0.4pt}\par\vspace{1.2em}
Riktig alternativ per spørsmål (bokstaven står i rundingen foran hvert alternativ).\par\vspace{1em}
\begin{center}\sffamily\renewcommand{\arraystretch}{1.35}
\begin{tabular}{l*{4}{>{\centering\arraybackslash}p{1.6cm}}}
\toprule
& \textbf{Sp.\ 1} & \textbf{Sp.\ 2} & \textbf{Sp.\ 3} & \textbf{Sp.\ 4}\\
\midrule
""")
    for pi, row in enumerate(key, 1):
        out.append(f"Part {pi} & " + " & ".join(row) + r"\\" + "\n")
    out.append(r"""\bottomrule
\end{tabular}
\end{center}
\end{document}
""")
    return "".join(out), key


def build(path):
    nr = int(re.search(r"(\d+)", Path(path).stem).group(1))
    parts = parse(path)
    tex, key = to_tex(parts, nr)
    BUILD.mkdir(exist_ok=True)
    name = f"Ovingseksamen_{nr:02d}"
    tfile = BUILD / f"{name}.tex"
    tfile.write_text(tex, encoding="utf-8")
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tfile.name],
                           cwd=BUILD, capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            log = r.stdout[-3000:]
            raise SystemExit(f"LaTeX-feil i {path}:\n{log}")
    log = (BUILD / f"{name}.log").read_text(encoding="latin-1")
    over = re.findall(r"Overfull \\hbox \((\d+\.\d+)pt", log)
    big = [float(x) for x in over if float(x) > 3]
    print(f"OK {name}.pdf  fasit: " + " | ".join("".join(r) for r in key)
          + (f"  ({len(big)} overfull hbox > 3pt)" if big else ""))
    return BUILD / f"{name}.pdf"


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("check", "build"):
        print(__doc__)
        sys.exit(1)
    for f in sys.argv[2:]:
        if sys.argv[1] == "check":
            check(f)
        else:
            check(f)
            build(f)
