#!/usr/bin/env python3
"""Bygger kodekurset.

Leser kilde/*.md (formatet er beskrevet i SKRIVEGUIDE.md) og skriver
  assets/kurs.js  – alle deler og leksjoner som JSON + kjøremotoren (runner.py)
  assets/data.js  – datasettene fra prosjektene, så pd.read_csv("problem2.csv") virker i nettleseren

Bruk:  python3 verktoy/build.py
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

ROT = Path(__file__).resolve().parent.parent          # Kurs/kodekurs
REPO = ROT.parent.parent                              # introml
KILDE = ROT / "kilde"
ASSETS = ROT / "assets"

DATASETT = {
    "regression.csv": REPO / "02_Prosjekter/Prosjekt_1/regression.csv",
    "classification.csv": REPO / "02_Prosjekter/Prosjekt_1/classification.csv",
    "problem1.csv": REPO / "02_Prosjekter/Prosjekt_2/problem1.csv",
    "problem2.csv": REPO / "02_Prosjekter/Prosjekt_2/problem2.csv",
    "classification_data.csv": REPO / "02_Prosjekter/Prosjekt_3/classification_data.csv",
    "regression_data.csv": REPO / "02_Prosjekter/Prosjekt_3/regression_data.csv",
}

MERK_TYPER = {"tips", "husk", "advarsel", "eksamen", "prosjekt", "hvorfor"}


class Feil(Exception):
    pass


# --------------------------------------------------------------------------
# Mini-markdown
# --------------------------------------------------------------------------

def _inline(tekst: str) -> str:
    """Inline-markdown. Matte ($...$) og `kode` beskyttes først."""
    vern: list[str] = []

    def beskytt(s: str) -> str:
        vern.append(s)
        return f"\x00{len(vern) - 1}\x00"

    # `kode`
    tekst = re.sub(r"`([^`]+)`", lambda m: beskytt(f"<code>{html.escape(m.group(1))}</code>"), tekst)
    # $$...$$ og $...$ (sendes urørt til KaTeX, men HTML-escapes)
    tekst = re.sub(r"\$\$(.+?)\$\$", lambda m: beskytt("$$" + html.escape(m.group(1), quote=False) + "$$"), tekst)
    tekst = re.sub(r"(?<![\\\w])\$([^\s$](?:[^$]*?[^\s$])?)\$", lambda m: beskytt("$" + html.escape(m.group(1), quote=False) + "$"), tekst)
    # rå HTML-tagger slipper gjennom
    tekst = re.sub(r"</?[a-zA-Z][^<>]*>", lambda m: beskytt(m.group(0)), tekst)
    tekst = html.escape(tekst, quote=False)
    tekst = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", tekst)
    tekst = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", tekst)
    tekst = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', tekst)
    while "\x00" in tekst:
        tekst = re.sub(r"\x00(\d+)\x00", lambda m: vern[int(m.group(1))], tekst)
    return tekst


def markdown(tekst: str) -> str:
    linjer = tekst.strip("\n").split("\n")
    ut: list[str] = []
    i = 0
    while i < len(linjer):
        l = linjer[i]
        s = l.strip()
        if not s:
            i += 1
            continue
        if s.startswith("$$") and not (s.endswith("$$") and len(s) > 4):
            # matteblokk over flere linjer
            blokk = [s]
            i += 1
            while i < len(linjer) and not linjer[i].strip().endswith("$$"):
                blokk.append(linjer[i])
                i += 1
            if i < len(linjer):
                blokk.append(linjer[i].strip())
                i += 1
            ut.append('<div class="math-block">' + html.escape("\n".join(blokk), quote=False) + "</div>")
            continue
        if s.startswith("$$") and s.endswith("$$"):
            ut.append('<div class="math-block">' + html.escape(s, quote=False) + "</div>")
            i += 1
            continue
        m = re.match(r"^(#{2,4})\s+(.*)$", s)
        if m:
            niva = min(len(m.group(1)) + 1, 4)  # ## -> h3
            ut.append(f"<h{niva}>{_inline(m.group(2))}</h{niva}>")
            i += 1
            continue
        if s.startswith("|"):
            rader = []
            while i < len(linjer) and linjer[i].strip().startswith("|"):
                rader.append(linjer[i].strip())
                i += 1
            celler = [[c.strip() for c in r.strip("|").split("|")] for r in rader]
            har_hode = len(celler) > 1 and all(re.fullmatch(r":?-{2,}:?", c) for c in celler[1])
            h = ['<div class="tabell-wrap"><table>']
            if har_hode:
                h.append("<thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in celler[0]) + "</tr></thead>")
                celler = celler[2:]
            h.append("<tbody>" + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in celler) + "</tbody></table></div>")
            ut.append("".join(h))
            continue
        if re.match(r"^[-*]\s+", s) or re.match(r"^\d+[.)]\s+", s):
            ordnet = bool(re.match(r"^\d+[.)]\s+", s))
            punkter: list[str] = []
            while i < len(linjer):
                t = linjer[i]
                ts = t.strip()
                if re.match(r"^[-*]\s+", ts) or re.match(r"^\d+[.)]\s+", ts):
                    punkter.append(re.sub(r"^([-*]|\d+[.)])\s+", "", ts))
                elif ts and t.startswith((" ", "\t")) and punkter:
                    punkter[-1] += " " + ts
                else:
                    break
                i += 1
            tag = "ol" if ordnet else "ul"
            ut.append(f"<{tag}>" + "".join(f"<li>{_inline(p)}</li>" for p in punkter) + f"</{tag}>")
            continue
        if s.startswith("<") and not s.startswith("<code") and re.match(r"^</?(div|table|details|figure|svg|p|ul|ol|pre|blockquote|section)\b", s):
            blokk = []
            while i < len(linjer) and linjer[i].strip():
                blokk.append(linjer[i])
                i += 1
            ut.append("\n".join(blokk))
            continue
        avsnitt = []
        while i < len(linjer):
            ts = linjer[i].strip()
            if not ts or re.match(r"^(#{2,4}\s|\||[-*]\s|\d+[.)]\s|\$\$)", ts):
                break
            avsnitt.append(ts)
            i += 1
        ut.append("<p>" + _inline(" ".join(avsnitt)) + "</p>")
    return "\n".join(ut)


# --------------------------------------------------------------------------
# Parser for kildeformatet
# --------------------------------------------------------------------------

FENCE = re.compile(r"^```(\S*)\s*(.*)$")


def _les_fence(linjer, i, fil):
    """linjer[i] er en åpnende ``` – returner (info, attrs, innhold, ny_i)."""
    m = FENCE.match(linjer[i].strip())
    info, attrs = m.group(1), m.group(2).split()
    start = i + 1
    i += 1
    while i < len(linjer) and linjer[i].strip() != "```":
        i += 1
    if i >= len(linjer):
        raise Feil(f"{fil}:{start}: kodeblokk ``` uten avslutning")
    innhold = "\n".join(linjer[start:i])
    return info, attrs, innhold.rstrip("\n"), i + 1


def _parse_blokker(linjer, fil, linjenr0, inni=None):
    """Parser innholdet i en leksjon (eller inni en :::-boks). Returnerer liste med blokker."""
    blokker = []
    tekst: list[str] = []

    def tøm():
        if any(t.strip() for t in tekst):
            blokker.append({"t": "tekst", "html": markdown("\n".join(tekst))})
        tekst.clear()

    i = 0
    while i < len(linjer):
        l = linjer[i]
        s = l.strip()
        nr = linjenr0 + i
        if s.startswith("```"):
            tøm()
            info, attrs, innhold, i = _les_fence(linjer, i, fil)
            if info == "python":
                if "statisk" in attrs:
                    blokker.append({"t": "statisk", "kode": innhold, "kjor": "ikkekjor" not in attrs,
                                    "fortsett": "fortsett" in attrs, "linje": nr})
                else:
                    b = {"t": "kode", "kode": innhold, "linje": nr}
                    for a in attrs:  # «feil» eller «feil=TypeError»: eksempelet skal gi en feilmelding
                        if a == "feil" or a.startswith("feil="):
                            b["feil"] = a.partition("=")[2] or True
                    blokker.append(b)
            elif info == "output":
                if not blokker or blokker[-1]["t"] not in ("statisk", "kode", "quizkode"):
                    raise Feil(f"{fil}:{nr}: ```output må komme rett etter en kodeblokk")
                blokker[-1]["output"] = innhold
            elif info in ("start", "losning", "test", "kode") and inni:
                blokker.append({"t": "_" + info, "kode": innhold, "linje": nr})
            else:
                raise Feil(f"{fil}:{nr}: ukjent kodeblokk ```{info} {' '.join(attrs)}")
            continue
        if s.startswith(":::") and s != ":::":
            if inni:
                raise Feil(f"{fil}:{nr}: ::: inni ::: er ikke lov")
            tøm()
            hode = s[3:].strip()
            type_, _, tittel = hode.partition(" ")
            j = i + 1
            i_fence = False
            while j < len(linjer):
                ls = linjer[j].strip()
                if not i_fence and ls.startswith("```"):
                    i_fence = True
                elif i_fence and ls == "```":
                    i_fence = False
                elif ls == ":::" and not i_fence:
                    break
                j += 1
            if j >= len(linjer):
                raise Feil(f"{fil}:{nr}: ::: {type_} uten avsluttende :::")
            indre = linjer[i + 1:j]
            blokker.append(_parse_boks(type_, tittel.strip(), indre, fil, nr + 1))
            i = j + 1
            continue
        tekst.append(l)
        i += 1
    tøm()
    return blokker


def _parse_boks(type_, tittel, linjer, fil, nr):
    if type_ in MERK_TYPER:
        indre = _parse_blokker(linjer, fil, nr, inni=type_)
        for b in indre:
            if b["t"] not in ("tekst", "kode", "statisk"):
                raise Feil(f"{fil}:{nr}: ::: {type_} kan bare ha tekst og kode")
        return {"t": "merk", "type": type_, "tittel": tittel, "blokker": indre}
    if type_ == "oppsummering":
        return {"t": "oppsummering", "html": markdown("\n".join(linjer))}
    if type_ == "oppgave":
        indre = _parse_blokker(linjer, fil, nr, inni="oppgave")
        o = {"t": "oppgave", "tittel": tittel or "Oppgave", "html": "", "linje": nr}
        tekster = []
        for b in indre:
            if b["t"] == "tekst":
                tekster.append(b["html"])
            elif b["t"] in ("_start", "_losning", "_test"):
                o[b["t"][1:]] = b["kode"]
            elif b["t"] in ("kode", "statisk"):
                tekster.append(f'<pre class="inline-kode"><code>{html.escape(b["kode"])}</code></pre>')
            else:
                raise Feil(f"{fil}:{nr}: ukjent innhold i oppgave: {b['t']}")
        for k in ("start", "losning", "test"):
            if k not in o:
                raise Feil(f"{fil}:{nr}: oppgaven «{o['tittel']}» mangler ```{k}")
        o["html"] = "\n".join(tekster)
        return o
    if type_ == "quiz":
        q = {"t": "quiz", "html": "", "kode": None, "alternativer": [], "riktig": None, "forklaring": "", "linje": nr}
        sporsmal, forklaring = [], []
        i = 0
        while i < len(linjer):
            s = linjer[i].strip()
            if s.startswith("```"):
                info, attrs, innhold, i = _les_fence(linjer, i, fil)
                if info not in ("kode", "python"):
                    raise Feil(f"{fil}:{nr + i}: i quiz er bare ```kode lov")
                q["kode"] = innhold
                continue
            m = re.match(r"^- \[([ xX])\]\s+(.*)$", s)
            if m:
                if m.group(1).lower() == "x":
                    if q["riktig"] is not None:
                        raise Feil(f"{fil}:{nr + i}: quiz har flere riktige svar")
                    q["riktig"] = len(q["alternativer"])
                q["alternativer"].append(_inline(m.group(2)))
            elif s.startswith("?"):
                forklaring.append(s[1:].strip())
            elif not q["alternativer"]:
                sporsmal.append(linjer[i])
            elif s:
                raise Feil(f"{fil}:{nr + i}: uventet linje i quiz etter alternativene: {s[:60]}")
            i += 1
        if q["riktig"] is None or len(q["alternativer"]) < 2:
            raise Feil(f"{fil}:{nr}: quiz må ha minst to alternativer og nøyaktig ett [x]")
        q["html"] = markdown("\n".join(sporsmal))
        q["forklaring"] = markdown("\n".join(forklaring))
        return q
    raise Feil(f"{fil}:{nr}: ukjent boks ::: {type_}")


def parse_fil(sti: Path) -> dict:
    fil = sti.name
    linjer = sti.read_text(encoding="utf-8").split("\n")
    del_ = None
    leksjon = None
    buffer: list[str] = []
    buffer_start = 0

    def avslutt():
        nonlocal leksjon
        if leksjon is not None:
            leksjon["blokker"] = _parse_blokker(buffer, fil, buffer_start)
            del_["leksjoner"].append(leksjon)
        elif del_ is not None:
            del_["ingress"] = markdown("\n".join(buffer))
        leksjon = None

    for n, l in enumerate(linjer, start=1):
        if l.startswith("=== DEL"):
            if del_ is not None:
                raise Feil(f"{fil}:{n}: bare én DEL per fil")
            deler = [d.strip() for d in l[len("=== DEL"):].split("|")]
            if len(deler) != 3:
                raise Feil(f"{fil}:{n}: skal være  === DEL <nr> | <id> | <tittel>")
            del_ = {"nr": deler[0], "id": deler[1], "tittel": deler[2], "ingress": "", "leksjoner": [], "fil": fil}
            buffer, buffer_start = [], n + 1
            continue
        if l.startswith("--- LEKSJON"):
            if del_ is None:
                raise Feil(f"{fil}:{n}: LEKSJON før DEL")
            avslutt()
            deler = [d.strip() for d in l[len("--- LEKSJON"):].split("|")]
            if len(deler) != 3:
                raise Feil(f"{fil}:{n}: skal være  --- LEKSJON <nr> | <id> | <tittel>")
            leksjon = {"nr": deler[0], "id": deler[1], "tittel": deler[2], "linje": n}
            buffer, buffer_start = [], n + 1
            continue
        buffer.append(l)
    if del_ is None:
        raise Feil(f"{fil}: mangler  === DEL")
    avslutt()
    return del_


def les_kurs() -> list[dict]:
    deler = [parse_fil(p) for p in sorted(KILDE.glob("*.md"))]
    sett = set()
    for d in deler:
        for l in d["leksjoner"]:
            if l["nr"] in sett:
                raise Feil(f"{d['fil']}: leksjonsnummer {l['nr']} er brukt to ganger")
            sett.add(l["nr"])
    return deler


def _rens(o):
    """Fjern interne felt (linjenummer) før publisering."""
    if isinstance(o, dict):
        return {k: _rens(v) for k, v in o.items() if k not in ("linje", "fil")}
    if isinstance(o, list):
        return [_rens(v) for v in o]
    return o


def main():
    try:
        deler = les_kurs()
    except Feil as e:
        print("FEIL:", e, file=sys.stderr)
        sys.exit(1)
    runner = (ROT / "verktoy/runner.py").read_text(encoding="utf-8")
    ASSETS.mkdir(exist_ok=True)
    js = (
        "// Generert av verktoy/build.py – ikke rediger for hånd. Rediger kilde/*.md.\n"
        "window.KK_KURS = " + json.dumps(_rens(deler), ensure_ascii=False) + ";\n"
        "window.KK_RUNNER = " + json.dumps(runner, ensure_ascii=False) + ";\n"
    )
    (ASSETS / "kurs.js").write_text(js, encoding="utf-8")
    data = {navn: sti.read_text(encoding="utf-8") for navn, sti in DATASETT.items()}
    (ASSETS / "data.js").write_text(
        "// Generert av verktoy/build.py: datasettene fra prosjekt 1–3.\nwindow.KK_DATA = " + json.dumps(data) + ";\n",
        encoding="utf-8",
    )
    # cache-busting: ?v=<tidsstempel> på egne filer i index.html
    import time
    idx = ROT / "index.html"
    v = str(int(time.time()))
    h = idx.read_text(encoding="utf-8")
    h = re.sub(r'(assets/(?:kurs|data|app)\.js|assets/kodekurs\.css)(\?v=\d+)?"', lambda m: f'{m.group(1)}?v={v}"', h)
    idx.write_text(h, encoding="utf-8")
    n = sum(len(d["leksjoner"]) for d in deler)
    print(f"Bygget {len(deler)} deler, {n} leksjoner -> assets/kurs.js ({len(js) // 1024} kB)")


if __name__ == "__main__":
    main()
