/* TDT4172 · Kodekurs – app
   Kursinnholdet ligger i kurs.js (generert fra kilde/*.md av verktoy/build.py).
   Python kjøres i nettleseren med Pyodide, som lastes første gang du trenger det. */
(() => {
  "use strict";

  const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/";
  const KURS = window.KK_KURS || [];
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // ---------------------------------------------------------------- lagring
  const store = {
    get(k, d) { try { const v = localStorage.getItem("kk:" + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem("kk:" + k, JSON.stringify(v)); } catch (e) { /* privat modus o.l. */ } },
  };
  let ferdig = store.get("ferdig", {});
  let quizSvar = store.get("quiz", {});
  let oppgLost = store.get("oppg", {});

  // ---------------------------------------------------------------- indeks
  const LEKSJONER = [];
  KURS.forEach(d => d.leksjoner.forEach(l => { l.del = d; LEKSJONER.push(l); }));
  const finnLeksjon = nr => LEKSJONER.find(l => l.nr === nr);
  const finnDel = id => KURS.find(d => d.id === id);
  const delFerdig = d => d.leksjoner.filter(l => ferdig[l.nr]).length;

  // ---------------------------------------------------------------- Python (Pyodide)
  let pyLover = null;
  const pyStatus = (s, tekst) => {
    const el = $("#py-status");
    if (!el) return;
    el.dataset.s = s;
    $(".tekst", el).textContent = tekst;
  };
  function lastSkript(src) {
    return new Promise((ok, feil) => {
      const s = document.createElement("script");
      s.src = src; s.onload = ok; s.onerror = () => feil(new Error("Kunne ikke laste " + src));
      document.head.appendChild(s);
    });
  }
  function startPython() {
    if (pyLover) return pyLover;
    pyStatus("laster", "Python: laster …");
    pyLover = (async () => {
      if (!window.loadPyodide) await lastSkript(PYODIDE_URL + "pyodide.js");
      const py = await window.loadPyodide({ indexURL: PYODIDE_URL });
      const data = window.KK_DATA || {};
      for (const [navn, innhold] of Object.entries(data)) py.FS.writeFile(navn, innhold);
      py.runPython('import os\nos.environ["MPLBACKEND"] = "Agg"');
      py.runPython(window.KK_RUNNER);
      pyStatus("klar", "Python: klar");
      return py;
    })().catch(e => {
      pyLover = null;
      pyStatus("feil", "Python: feil ved lasting");
      throw e;
    });
    return pyLover;
  }
  const lastetPakker = new Set();
  async function kjorPython(src, test, statusCb) {
    const py = await startPython();
    const alt = src + "\n" + (test || "");
    const nye = ["numpy", "pandas", "sklearn", "matplotlib", "scipy"].filter(p => new RegExp("\\b(import|from)\\s+" + p + "\\b").test(alt) && !lastetPakker.has(p));
    if (nye.length) {
      statusCb && statusCb("Laster " + nye.map(p => p === "sklearn" ? "scikit-learn" : p).join(", ") + " (bare første gang) …");
      await py.loadPackagesFromImports(alt);
      nye.forEach(p => lastetPakker.add(p));
    }
    statusCb && statusCb("Kjører …");
    await new Promise(r => setTimeout(r, 30));
    const fn = py.globals.get("kk_run");
    try {
      return JSON.parse(fn(src, test || null));
    } finally {
      fn.destroy();
    }
  }

  // ---------------------------------------------------------------- editor
  const nyeEditorer = [];  // CodeMirror må tegnes på nytt når boksen er satt inn i siden
  function lagEditor(vert, kode, { lesbar = false, onRun = null } = {}) {
    const ed = lagEditorIntern(vert, kode, { lesbar, onRun });
    nyeEditorer.push(ed);
    if (document.body.contains(vert)) ed.refresh();
    return ed;
  }
  const tegnEditorer = () => requestAnimationFrame(() => nyeEditorer.splice(0).forEach(e => e.refresh()));
  function lagEditorIntern(vert, kode, { lesbar = false, onRun = null } = {}) {
    if (window.CodeMirror) {
      const cm = window.CodeMirror(vert, {
        value: kode, mode: "python", lineNumbers: true, indentUnit: 4, tabSize: 4, matchBrackets: true,
        viewportMargin: Infinity, readOnly: lesbar ? "nocursor" : false, lineWrapping: false,
        extraKeys: {
          "Ctrl-Enter": () => onRun && onRun(), "Cmd-Enter": () => onRun && onRun(), "Shift-Enter": () => onRun && onRun(),
          Tab: c => c.somethingSelected() ? c.indentSelection("add") : c.replaceSelection("    ", "end"),
          "Shift-Tab": c => c.indentSelection("subtract"),
        },
      });
      if (lesbar) cm.getWrapperElement().classList.add("readonly");
      return { getValue: () => cm.getValue(), setValue: v => cm.setValue(v), refresh: () => cm.refresh() };
    }
    const ta = document.createElement("textarea");
    ta.className = "reserve"; ta.value = kode; ta.spellcheck = false; ta.readOnly = lesbar;
    ta.rows = Math.max(3, kode.split("\n").length + 1);
    ta.addEventListener("keydown", e => {
      if ((e.ctrlKey || e.metaKey || e.shiftKey) && e.key === "Enter") { e.preventDefault(); onRun && onRun(); }
      if (e.key === "Tab" && !lesbar) { e.preventDefault(); ta.setRangeText("    ", ta.selectionStart, ta.selectionEnd, "end"); }
    });
    vert.appendChild(ta);
    return { getValue: () => ta.value, setValue: v => { ta.value = v; }, refresh() {} };
  }

  // ---------------------------------------------------------------- visualisering av variabler
  // Fargeskala: sekvensiell (hvit → blå) eller divergerende (rød ← hvit → blå) når data har begge fortegn.
  function farge(v, min, maks) {
    if (v === null || v === undefined || min === undefined) return null;
    const lerp = (a, b, t) => a.map((x, i) => Math.round(x + (b[i] - x) * t));
    const hvit = [248, 249, 252];
    if (min < 0 && maks > 0) {
      const m = Math.max(-min, maks);
      const t = Math.min(1, Math.abs(v) / m);
      return { rgb: lerp(hvit, v < 0 ? [205, 80, 60] : [37, 88, 180], t), mork: t > 0.55 };
    }
    const t = maks > min ? (v - min) / (maks - min) : 0.5;
    return { rgb: lerp(hvit, [30, 62, 150], t), mork: t > 0.55 };
  }
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; };
  const fargeCelle = (td, f) => { if (f) { td.style.background = `rgb(${f.rgb})`; td.style.color = f.mork ? "#fff" : "#1d1f23"; } };

  function visTabell(hode, rader, { indeks = null, mangler = null, tall = null, min, maks } = {}) {
    const wrap = el("div", "vis-tabell");
    const t = el("table");
    if (hode) {
      const tr = el("tr");
      if (indeks) tr.appendChild(el("th", "hj"));
      hode.forEach(h => tr.appendChild(el("th", "", h)));
      const th = el("thead"); th.appendChild(tr); t.appendChild(th);
    }
    const tb = el("tbody");
    rader.forEach((r, i) => {
      const tr = el("tr");
      if (indeks) tr.appendChild(el("th", "idx", esc(indeks[i])));
      r.forEach((c, j) => {
        const td = el("td");
        td.textContent = c;
        if (mangler && mangler[i][j]) td.className = "mangler";
        else if (tall) fargeCelle(td, farge(tall[i][j], min, maks));
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    wrap.appendChild(t);
    return wrap;
  }

  function visVerdi(d) {
    const kort = el("div", "vis-kort");
    const hode = el("div", "vis-hode");
    hode.innerHTML = (d.navn ? `<code class="vis-navn">${esc(d.navn)}</code>` : "") + `<span class="vis-type">${esc(d.kort || d.type)}</span>`;
    const merker = [];
    if (d.art === "df") {
      if (d.nan_totalt) merker.push(`<span class="vis-merke rod">${d.nan_totalt} manglende verdier</span>`);
      else if (d.nan_totalt === 0) merker.push(`<span class="vis-merke gronn">ingen manglende verdier</span>`);
    }
    if (d.art === "arr") {
      merker.push(`<span class="vis-merke">dtype ${esc(d.dtype)}</span>`);
      if (d.min !== undefined) merker.push(`<span class="vis-merke">min ${+d.min.toPrecision(4)} · maks ${+d.maks.toPrecision(4)}</span>`);
      if (d.nan) merker.push(`<span class="vis-merke rod">${d.nan} NaN</span>`);
    }
    hode.insertAdjacentHTML("beforeend", merker.join(""));
    kort.appendChild(hode);
    const kropp = el("div", "vis-kropp");
    const fot = [];
    switch (d.art) {
      case "df": {
        const hodeCeller = d.kolonner.map((k, i) => `${esc(k)}<small>${esc(d.dtyper[i])}${d.nan[i] ? ` · <b class="rod">${d.nan[i]} NaN</b>` : ""}</small>`);
        kropp.appendChild(visTabell(hodeCeller, d.rader, { indeks: d.indeks, mangler: d.mangler }));
        if (d.flere_rader) fot.push(`… og ${d.flere_rader} rader til`);
        if (d.flere_kol) fot.push(`… og ${d.flere_kol} kolonner til`);
        break;
      }
      case "arr": {
        if (d.heatmap) {
          const r = d.tall.length, k = d.tall[0].length;
          const c = document.createElement("canvas");
          c.width = k; c.height = r; c.className = "vis-heatmap";
          const ctx = c.getContext("2d"), img = ctx.createImageData(k, r);
          d.tall.forEach((rad, i) => rad.forEach((v, j) => {
            const f = farge(v, d.min, d.maks) || { rgb: [255, 0, 0] };
            img.data.set([...f.rgb, 255], (i * k + j) * 4);
          }));
          ctx.putImageData(img, 0, 0);
          const skala = Math.max(3, Math.floor(260 / Math.max(r, k)));
          c.style.width = k * skala + "px"; c.style.height = r * skala + "px";
          kropp.appendChild(c);
        } else if (d.bool) {
          const t = visTabell(d.en_d ? d.verdier[0].map((_, i) => i) : null, d.verdier, { indeks: d.en_d ? null : d.verdier.map((_, i) => i) });
          t.querySelectorAll("td").forEach(td => td.classList.add(td.textContent === "True" ? "sann" : "usann"));
          kropp.appendChild(t);
        } else {
          kropp.appendChild(visTabell(d.en_d ? d.verdier[0].map((_, i) => i) : d.verdier[0].map((_, i) => i), d.verdier,
            { indeks: d.en_d ? null : d.verdier.map((_, i) => i), tall: d.tall, min: d.min, maks: d.maks }));
        }
        if (d.merk) fot.push(d.merk);
        if (d.flere_rader) fot.push(`… og ${d.flere_rader} rader til`);
        if (d.flere_kol) fot.push(`… og ${d.flere_kol} kolonner til`);
        break;
      }
      case "skalar":
        kropp.appendChild(el("pre", "vis-skalar")).textContent = d.verdi;
        break;
      case "liste": {
        const rad = el("div", "vis-liste");
        d.verdier.forEach((v, i) => {
          const b = el("div", "vis-boks");
          if (d.indekser) b.appendChild(el("span", "i", String(i)));
          b.appendChild(el("span", "v")).textContent = v;
          rad.appendChild(b);
        });
        if (!d.verdier.length) rad.appendChild(el("span", "vis-tom", "(tom)"));
        kropp.appendChild(rad);
        if (d.flere) fot.push(`… og ${d.flere} elementer til`);
        break;
      }
      case "nostet": {
        const alle = d.tall.flat().filter(x => x !== null);
        const min = alle.length ? Math.min(...alle) : undefined, maks = alle.length ? Math.max(...alle) : undefined;
        kropp.appendChild(visTabell(null, d.verdier, { indeks: d.verdier.map((_, i) => i), tall: d.tall, min, maks }));
        if (d.flere) fot.push(`… og ${d.flere} rader til`);
        break;
      }
      case "dict":
        kropp.appendChild(visTabell(["nøkkel", "verdi"], d.par)).classList.add("nokkel");
        if (d.flere) fot.push(`… og ${d.flere} par til`);
        break;
      case "modell":
        if (d.laert.length) {
          kropp.appendChild(el("div", "vis-under", d.type === "Pipeline" ? "Steg i pipelinen" : "Lært under <code>fit</code>"));
          kropp.appendChild(visTabell(null, d.laert)).classList.add("nokkel");
        } else kropp.appendChild(el("div", "vis-under", "Ikke trent ennå: kall <code>fit</code> først."));
        if (d.params.length) {
          const det = el("details", "vis-params");
          det.appendChild(el("summary", "", "Hyperparametre"));
          det.appendChild(visTabell(null, d.params)).classList.add("nokkel");
          kropp.appendChild(det);
        }
        break;
      case "objekt":
        kropp.appendChild(el("div", "vis-under", "Attributter (<code>self.…</code>)"));
        kropp.appendChild(d.attrs.length ? visTabell(null, d.attrs) : el("span", "vis-tom", "(ingen attributter)")).classList.add("nokkel");
        break;
    }
    kort.appendChild(kropp);
    if (fot.length) kort.appendChild(el("div", "vis-fot", fot.map(esc).join(" · ")));
    return kort;
  }

  function visVariabler(vert, variabler, autoApne = true) {
    vert.innerHTML = "";
    if (!variabler || !variabler.length) { vert.classList.remove("vis"); return; }
    vert.classList.add("vis");
    vert.appendChild(el("div", "tittel", `Variabler etter kjøring <span>klikk for å se innholdet</span>`));
    const brikker = el("div", "var-brikker"), flate = el("div", "var-flate");
    let aktiv = null;
    const velg = (d, b) => {
      flate.innerHTML = "";
      brikker.querySelectorAll(".var-brikke").forEach(x => x.classList.remove("aktiv"));
      if (aktiv === d) { aktiv = null; return; }
      aktiv = d; b.classList.add("aktiv");
      flate.appendChild(visVerdi(d));
    };
    variabler.forEach(d => {
      const b = el("button", "var-brikke " + d.art, `<code>${esc(d.navn)}</code><span>${esc(d.kort)}</span>`);
      b.addEventListener("click", () => velg(d, b));
      brikker.appendChild(b);
    });
    vert.appendChild(brikker);
    vert.appendChild(flate);
    // åpne den siste datastrukturen automatisk (tabell/array), det er som regel den koden jobbet med
    const data = variabler.filter(d => d.art === "df" || d.art === "arr");
    const auto = data[data.length - 1];
    if (auto && autoApne) velg(auto, brikker.children[variabler.indexOf(auto)]);
  }

  // ---------------------------------------------------------------- utdata
  function visUtdata(boks, res, { test = false } = {}) {
    boks.innerHTML = "";
    boks.classList.add("vis");
    const tittel = document.createElement("div");
    tittel.className = "tittel"; tittel.textContent = "Resultat";
    boks.appendChild(tittel);
    let noe = false, lengde = 0;
    for (const [type, innhold] of res.events || []) {
      if (type === "img") {
        const img = new Image(); img.src = "data:image/png;base64," + innhold; img.alt = "Figur fra koden";
        boks.appendChild(img); noe = true;
      } else if (type === "vis") {
        const d = el("div", "vis-inline");
        d.appendChild(visVerdi(innhold));
        boks.appendChild(d); noe = true;
      } else if (type === "html") {
        const d = document.createElement("div");
        d.className = "df-wrap"; d.innerHTML = innhold;
        boks.appendChild(d); noe = true;
      } else {
        if (lengde > 40000) continue;
        let t = innhold;
        if (lengde + t.length > 40000) t = t.slice(0, 40000 - lengde) + "\n… (utdata kuttet)";
        lengde += t.length;
        const pre = document.createElement("pre");
        if (type === "warn") pre.className = "warn";
        pre.textContent = t.replace(/\n$/, "");
        if (pre.textContent) { boks.appendChild(pre); noe = true; }
      }
    }
    if (res.status === "feil") {
      const pre = document.createElement("pre");
      pre.className = "feil"; pre.textContent = res.feil;
      boks.appendChild(pre); noe = true;
    }
    if (test) {
      const r = document.createElement("div");
      if (res.status === "bestatt") { r.className = "resultat ok"; r.textContent = "✔ Riktig! Bra jobbet."; }
      else if (res.status === "ikke_bestatt") { r.className = "resultat nei"; r.textContent = "✘ Ikke helt ennå: " + res.feil; }
      else if (res.status === "feil") { r.className = "resultat nei"; r.textContent = "✘ Koden stoppet med en feil før den kunne sjekkes. Les feilmeldingen over."; }
      if (r.className) { boks.appendChild(r); noe = true; }
    }
    if (!noe) {
      const d = document.createElement("div");
      d.className = "tom"; d.textContent = "(Koden kjørte, men skrev ikke ut noe. Bruk print() for å se verdier.)";
      boks.appendChild(d);
    }
  }
  function visLaster(boks, tekst) {
    boks.classList.add("vis");
    boks.innerHTML = `<div class="tittel">Resultat</div><div class="tom">${esc(tekst)}</div>`;
  }
  function visLastefeil(boks, e) {
    boks.classList.add("vis");
    boks.innerHTML = `<div class="tittel">Resultat</div><pre class="feil">Kunne ikke starte Python i nettleseren (${esc(e.message || e)}).\nPyodide lastes fra internett første gang – sjekk nettilkoblingen og prøv igjen.</pre>`;
  }

  // ---------------------------------------------------------------- blokker
  let blokkTeller = 0;
  const etiketter = { tips: "Tips", husk: "Husk", advarsel: "Vanlig feil", eksamen: "Eksamen", prosjekt: "Fra prosjektet", hvorfor: "Hvorfor?" };

  function renderBlokker(blokker, l, forelder) {
    for (const b of blokker) forelder.appendChild(renderBlokk(b, l));
  }

  function renderBlokk(b, l) {
    const id = l.nr + "#" + (blokkTeller++);
    const div = document.createElement("div");
    switch (b.t) {
      case "tekst":
        div.className = "prose"; div.innerHTML = b.html; return div;
      case "merk": {
        div.className = "merk " + b.type;
        div.innerHTML = `<span class="etikett">${esc(b.tittel || etiketter[b.type] || b.type)}</span>`;
        renderBlokker(b.blokker, l, div);
        return div;
      }
      case "oppsummering":
        div.className = "oppsummering";
        div.innerHTML = `<span class="etikett">Dette har du lært</span><div class="prose">${b.html}</div>`;
        return div;
      case "kode": return kodeBoks(b, l);
      case "statisk": return statiskBoks(b);
      case "oppgave": return oppgaveBoks(b, l, id);
      case "quiz": return quizBoks(b, l, id);
      default:
        div.textContent = "(ukjent blokk)"; return div;
    }
  }

  function kodeBoks(b) {
    const boks = document.createElement("div");
    boks.className = "kodeboks eksempel";
    boks.innerHTML = `<div class="hode">Eksempel <span class="merkelapp">Endre koden og trykk Kjør</span></div>
      <div class="editor"></div>
      <div class="verktoy"><button class="kjor-knapp">▶ Kjør</button><button class="liten-knapp tilbake">↺ Tilbakestill</button><span class="hint">Ctrl/Shift + Enter kjører</span></div>
      <div class="utdata"></div><div class="var-panel"></div>`;
    const ut = $(".utdata", boks), knapp = $(".kjor-knapp", boks);
    const kjor = async () => {
      knapp.disabled = true;
      visLaster(ut, pyLover ? "Kjører …" : "Starter Python i nettleseren (tar 5–15 s første gang) …");
      try {
        const res = await kjorPython(ed.getValue(), null, t => visLaster(ut, t));
        visUtdata(ut, res);
        visVariabler($(".var-panel", boks), res.variabler, !(res.events || []).some(e => e[0] === "vis"));
      }
      catch (e) { visLastefeil(ut, e); }
      finally { knapp.disabled = false; }
    };
    const ed = lagEditor($(".editor", boks), b.kode, { onRun: kjor });
    knapp.addEventListener("click", kjor);
    $(".tilbake", boks).addEventListener("click", () => { ed.setValue(b.kode); ut.classList.remove("vis"); $(".var-panel", boks).classList.remove("vis"); });
    return boks;
  }

  function statiskBoks(b) {
    const boks = document.createElement("div");
    boks.className = "kodeboks statisk";
    const hvorfor = "Denne koden bruker biblioteker (PyTorch, SHAP, gymnasium, Keras …) som ikke kan kjøres i nettleseren. Kjør den i Jupyter lokalt.";
    boks.innerHTML = `<div class="hode">Kode for Jupyter <span class="merkelapp" title="${esc(hvorfor)}">kjøres lokalt, ikke i nettleseren</span></div><div class="editor"></div>`;
    lagEditor($(".editor", boks), b.kode, { lesbar: true });
    if (b.output) {
      const f = document.createElement("div");
      f.className = "forventet";
      f.innerHTML = `<div class="tittel">Utdata når koden kjøres</div><pre></pre>`;
      $("pre", f).textContent = b.output;
      boks.appendChild(f);
    }
    return boks;
  }

  function oppgaveBoks(b, l, id) {
    const nokkel = l.nr + ":" + b.tittel;
    const boks = document.createElement("div");
    boks.className = "kodeboks oppgave" + (oppgLost[nokkel] ? " lost" : "");
    boks.innerHTML = `<div class="hode">Oppgave <span class="merkelapp">${esc(b.tittel)}</span></div>
      <div class="oppgavetekst prose">${b.html}</div>
      <div class="editor"></div>
      <div class="verktoy"><button class="kjor-knapp">▶ Kjør og sjekk</button><button class="liten-knapp vis-losning">Vis løsning</button><button class="liten-knapp tilbake">↺ Start på nytt</button><span class="hint">Ctrl/Shift + Enter</span></div>
      <div class="utdata"></div><div class="var-panel"></div>
      <div class="losning"><div class="tittel">Løsningsforslag <button class="liten-knapp bruk">Kopier inn i editoren</button></div><div class="editor"></div></div>`;
    const ut = $(".utdata", boks), knapp = $(".kjor-knapp", boks);
    const lagret = store.get("kode:" + nokkel, null);
    const kjor = async () => {
      knapp.disabled = true;
      store.set("kode:" + nokkel, ed.getValue());
      visLaster(ut, pyLover ? "Kjører …" : "Starter Python i nettleseren (tar 5–15 s første gang) …");
      try {
        const res = await kjorPython(ed.getValue(), b.test, t => visLaster(ut, t));
        visUtdata(ut, res, { test: true });
        visVariabler($(".var-panel", boks), res.variabler, false);
        if (res.status === "bestatt") {
          boks.classList.add("lost");
          oppgLost[nokkel] = true; store.set("oppg", oppgLost);
        }
      } catch (e) { visLastefeil(ut, e); }
      finally { knapp.disabled = false; }
    };
    const ed = lagEditor($(".editor", boks), lagret ?? b.start, { onRun: kjor });
    knapp.addEventListener("click", kjor);
    let losEd = null;
    $(".vis-losning", boks).addEventListener("click", e => {
      const los = $(".losning", boks);
      const vis = !los.classList.contains("vis");
      los.classList.toggle("vis", vis);
      e.target.textContent = vis ? "Skjul løsning" : "Vis løsning";
      if (vis && !losEd) losEd = lagEditor($(".losning .editor", boks), b.losning, { lesbar: true });
      if (losEd) losEd.refresh();
    });
    $(".bruk", boks).addEventListener("click", () => ed.setValue(b.losning));
    $(".tilbake", boks).addEventListener("click", () => { ed.setValue(b.start); store.set("kode:" + nokkel, null); ut.classList.remove("vis"); $(".var-panel", boks).classList.remove("vis"); });
    return boks;
  }

  function quizBoks(b, l, id) {
    const nokkel = l.nr + ":" + b.html.replace(/<[^>]+>/g, "").slice(0, 80);
    const boks = document.createElement("div");
    boks.className = "kodeboks quiz";
    boks.innerHTML = `<div class="hode">Test deg selv <span class="merkelapp">eksamensstil</span></div><div class="sporsmal prose">${b.html}</div>`;
    if (b.kode) {
      const k = document.createElement("div");
      k.className = "kode-vis";
      boks.appendChild(k);
      lagEditor(k, b.kode, { lesbar: true });
    }
    const alt = document.createElement("div");
    alt.className = "alternativer";
    b.alternativer.forEach((a, i) => {
      const btn = document.createElement("button");
      btn.className = "alt";
      btn.innerHTML = `<span class="bokstav">${"ABCDEFGH"[i]}</span><span>${a}</span>`;
      btn.addEventListener("click", () => svar(i));
      alt.appendChild(btn);
    });
    boks.appendChild(alt);
    const fork = document.createElement("div");
    fork.className = "forklaring";
    boks.appendChild(fork);
    function svar(i, stille = false) {
      if (boks.classList.contains("besvart") && !stille) return;
      boks.classList.add("besvart");
      const knapper = $$(".alt", alt);
      knapper[b.riktig].classList.add("riktig");
      if (i !== b.riktig) knapper[i].classList.add("galt");
      const rett = i === b.riktig;
      fork.innerHTML = `<span class="dom ${rett ? "ok" : "nei"}">${rett ? "✔ Riktig!" : "✘ Feil – riktig svar er " + "ABCDEFGH"[b.riktig] + "."}</span>${b.forklaring || ""}
        <p><button class="liten-knapp prov-igjen">Prøv igjen</button></p>`;
      $(".prov-igjen", fork).addEventListener("click", () => {
        boks.classList.remove("besvart");
        knapper.forEach(k => k.classList.remove("riktig", "galt"));
        delete quizSvar[nokkel]; store.set("quiz", quizSvar);
      });
      if (!stille) { quizSvar[nokkel] = i; store.set("quiz", quizSvar); }
      renderMatte(fork);
    }
    if (quizSvar[nokkel] !== undefined && quizSvar[nokkel] < b.alternativer.length) svar(quizSvar[nokkel], true);
    return boks;
  }

  // ---------------------------------------------------------------- matte
  function renderMatte(el) {
    if (window.renderMathInElement) {
      try {
        window.renderMathInElement(el, {
          delimiters: [{ left: "$$", right: "$$", display: true }, { left: "$", right: "$", display: false }],
          ignoredClasses: ["CodeMirror", "editor", "utdata"], throwOnError: false,
        });
      } catch (e) { /* ignorer */ }
    }
  }

  // ---------------------------------------------------------------- sidemeny
  function byggSidemeny() {
    const vert = $("#sb-deler");
    vert.innerHTML = "";
    for (const d of KURS) {
      const el = document.createElement("div");
      el.className = "sb-del"; el.dataset.del = d.id;
      const pst = Math.round(100 * delFerdig(d) / Math.max(1, d.leksjoner.length));
      el.innerHTML = `<button aria-expanded="false"><span class="nr">${esc(d.nr)}</span><span>${esc(d.tittel)}</span><span class="pil">▶</span></button>
        <span class="mini"><i style="width:${pst}%"></i></span>
        <ul>${d.leksjoner.map(l => `<li data-sok="${esc((l.nr + " " + l.tittel + " " + d.tittel).toLowerCase())}"><a href="#${esc(l.nr)}" data-nr="${esc(l.nr)}"><span class="nr">${esc(l.nr)}</span><span>${esc(l.tittel)}</span>${ferdig[l.nr] ? '<span class="ferdig">✓</span>' : ""}</a></li>`).join("")}</ul>`;
      $("button", el).addEventListener("click", () => {
        const apen = !el.classList.contains("apen");
        el.classList.toggle("apen", apen);
        $("button", el).setAttribute("aria-expanded", String(apen));
      });
      vert.appendChild(el);
    }
  }
  function oppdaterSidemeny(aktivNr) {
    $$(".sb-del a").forEach(a => a.classList.toggle("aktiv", a.dataset.nr === aktivNr));
    const l = aktivNr && finnLeksjon(aktivNr);
    if (l) {
      const el = $(`.sb-del[data-del="${CSS.escape(l.del.id)}"]`);
      if (el && !el.classList.contains("apen")) { el.classList.add("apen"); $("button", el).setAttribute("aria-expanded", "true"); }
      const a = $(`.sb-del a[data-nr="${CSS.escape(aktivNr)}"]`);
      if (a) {
        const sb = $("#sidebar"), r = a.getBoundingClientRect(), rs = sb.getBoundingClientRect();
        if (r.top < rs.top + 40 || r.bottom > rs.bottom - 20) a.scrollIntoView({ block: "center" });
      }
    }
  }
  function oppdaterFremdrift() {
    const n = LEKSJONER.length, f = LEKSJONER.filter(l => ferdig[l.nr]).length;
    const pst = n ? Math.round(100 * f / n) : 0;
    const el = $("#total-fremdrift");
    $("i", el).style.width = pst + "%";
    $(".tall", el).textContent = `${f}/${n} leksjoner`;
    for (const d of KURS) {
      const el2 = $(`.sb-del[data-del="${CSS.escape(d.id)}"] .mini i`);
      if (el2) el2.style.width = Math.round(100 * delFerdig(d) / Math.max(1, d.leksjoner.length)) + "%";
    }
  }
  function markerFerdig(nr, verdi = true) {
    if (verdi) ferdig[nr] = true; else delete ferdig[nr];
    store.set("ferdig", ferdig);
    const a = $(`.sb-del a[data-nr="${CSS.escape(nr)}"]`);
    if (a) {
      const f = $(".ferdig", a);
      if (verdi && !f) a.insertAdjacentHTML("beforeend", '<span class="ferdig">✓</span>');
      if (!verdi && f) f.remove();
    }
    oppdaterFremdrift();
  }

  // ---------------------------------------------------------------- sider
  const hoved = () => $("#hoved");

  function navRad(l, bunn) {
    const i = LEKSJONER.indexOf(l);
    const forrige = LEKSJONER[i - 1], neste = LEKSJONER[i + 1];
    const rad = document.createElement("div");
    rad.className = "navrad" + (bunn ? " bunn" : "");
    rad.innerHTML = `<a class="navknapp sekundar" ${forrige ? `href="#${esc(forrige.nr)}"` : 'aria-disabled="true"'}>❮ Forrige</a>` +
      (bunn
        ? (neste ? `<a class="navknapp neste-ferdig" href="#${esc(neste.nr)}">✓ Fullført – neste ❯</a>` : `<a class="navknapp neste-ferdig" href="#">✓ Fullført – til forsiden</a>`)
        : `<a class="navknapp" ${neste ? `href="#${esc(neste.nr)}"` : 'aria-disabled="true"'}>Neste ❯</a>`);
    const nf = $(".neste-ferdig", rad);
    if (nf) nf.addEventListener("click", () => markerFerdig(l.nr, true));
    return rad;
  }

  function visLeksjon(l) {
    blokkTeller = 0;
    const m = hoved();
    m.innerHTML = "";
    const art = document.createElement("article");
    art.className = "leksjon";
    art.innerHTML = `<div class="brodsmule"><a href="#">Kodekurs</a> › <a href="#del-${esc(l.del.id)}">Del ${esc(l.del.nr)}: ${esc(l.del.tittel)}</a></div>
      <h1>${esc(l.tittel)}</h1>`;
    art.appendChild(navRad(l, false));
    renderBlokker(l.blokker, l, art);
    const fr = document.createElement("p");
    fr.className = "laster-tekst";
    fr.innerHTML = ferdig[l.nr]
      ? `✓ Du har fullført denne leksjonen. <button class="liten-knapp angre">Marker som ikke fullført</button>`
      : "";
    art.appendChild(fr);
    const angre = $(".angre", fr);
    if (angre) angre.addEventListener("click", () => { markerFerdig(l.nr, false); fr.innerHTML = ""; });
    art.appendChild(navRad(l, true));
    m.appendChild(art);
    tegnEditorer();
    renderMatte(art);
    document.title = `${l.nr} ${l.tittel} · Kodekurs`;
    store.set("sist", l.nr);
    oppdaterSidemeny(l.nr);
    if (!pyLover && l.blokker.some(b => b.t === "kode" || b.t === "oppgave")) {
      // start Python i bakgrunnen, så første «Kjør» går raskt
      setTimeout(() => startPython().catch(() => {}), 1200);
    }
  }

  function visDel(d) {
    const m = hoved();
    const forste = d.leksjoner[0];
    m.innerHTML = `<article class="leksjon">
      <div class="brodsmule"><a href="#">Kodekurs</a> › Del ${esc(d.nr)}</div>
      <h1>Del ${esc(d.nr)}: ${esc(d.tittel)}</h1>
      <div class="prose">${d.ingress}</div>
      <ul class="leksjonsliste">${d.leksjoner.map(l => `<li><a href="#${esc(l.nr)}"><span class="nr">${esc(l.nr)}</span><span>${esc(l.tittel)}</span>${ferdig[l.nr] ? '<span class="ferdig">✓</span>' : ""}</a></li>`).join("")}</ul>
      ${forste ? `<a class="navknapp" href="#${esc(forste.nr)}">Start del ${esc(d.nr)} ❯</a>` : ""}
    </article>`;
    renderMatte(m);
    document.title = `Del ${d.nr}: ${d.tittel} · Kodekurs`;
    oppdaterSidemeny(null);
    const el = $(`.sb-del[data-del="${CSS.escape(d.id)}"]`);
    if (el) el.classList.add("apen");
  }

  function visForside() {
    const m = hoved();
    const sist = finnLeksjon(store.get("sist", null));
    const forste = LEKSJONER[0];
    const nesteUferdig = LEKSJONER.find(l => !ferdig[l.nr]);
    m.innerHTML = `<article class="leksjon" style="max-width:1000px">
      <section class="hero">
        <div class="kicker">TDT4172 · Kodepensum fra bunnen av</div>
        <h1>Lær all koden i IntroML, én ting om gangen</h1>
        <p>Et kodekurs der du lærer ved å skrive kode selv. Du begynner med <code>print("Hei")</code> og ender med å trene nevrale nettverk, Q-læringsagenter og SHAP-forklaringer. Hver del har maks et par setninger forklaring, og resten er oppgaver du løser direkte i nettleseren, med automatisk sjekk.</p>
        <div class="knapperad">
          ${sist ? `<a class="navknapp" href="#${esc(sist.nr)}">Fortsett: ${esc(sist.nr)} ${esc(sist.tittel)} ❯</a>` : (forste ? `<a class="navknapp" href="#${esc(forste.nr)}">Start kurset ❯</a>` : "")}
          ${sist && nesteUferdig && nesteUferdig !== sist ? `<a class="navknapp sekundar" href="#${esc(nesteUferdig.nr)}">Første ufullførte: ${esc(nesteUferdig.nr)}</a>` : ""}
        </div>
      </section>
      <div class="info-grid">
        <div class="info-kort"><b>▶ Kjør koden selv</b>Alle grønne «Kjør»-knapper kjører ekte Python (numpy, pandas, scikit-learn, matplotlib) i nettleseren. Første gang tar det noen sekunder å starte.</div>
        <div class="info-kort"><b>✎ Oppgaver med fasit</b>Skriv koden, trykk «Kjør og sjekk», så får du beskjed om svaret er riktig. Løsningsforslag finnes, men prøv selv først.</div>
        <div class="info-kort"><b>? Eksamensstil</b>Eksamen 2025 hadde egne spørsmål om kodelesing. Del 13 er ren kodelesingstrening med flervalgsspørsmål i samme stil.</div>
        <div class="info-kort"><b>⚙ PyTorch, SHAP og gymnasium</b>Disse finnes ikke i nettleseren. Syntaksen vises i korte kodeblokker som du kan kjøre i Jupyter, og selve ideene øver du på i numpy-oppgaver her.</div>
      </div>
      <h2 style="font-size:1.35rem;margin:10px 0 12px">Delene i kurset</h2>
      <div class="del-grid">${KURS.map(d => {
        const f = delFerdig(d), n = d.leksjoner.length;
        const ingress = (d.ingress || "").replace(/<[^>]+>/g, "").slice(0, 150);
        return `<a class="del-kort" href="#del-${esc(d.id)}"><span class="nr">DEL ${esc(d.nr)}</span><h3>${esc(d.tittel)}</h3>
          <span style="font-size:13.5px;color:var(--text-soft)">${esc(ingress)}${ingress.length >= 150 ? " …" : ""}</span>
          <span class="meta"><span>${f}/${n}</span><span class="bar"><i style="width:${Math.round(100 * f / Math.max(1, n))}%"></i></span></span></a>`;
      }).join("")}</div>
      <p class="laster-tekst" style="margin-top:30px">Fremdriften din, svarene og koden i oppgavene lagres bare i denne nettleseren. <button class="liten-knapp" id="nullstill">Nullstill fremdrift</button></p>
    </article>`;
    renderMatte(m);
    document.title = "TDT4172 · Kodekurs";
    oppdaterSidemeny(null);
    $("#nullstill").addEventListener("click", () => {
      if (!window.confirm || window.confirm("Nullstille all fremdrift, quizsvar og lagret kode?")) {
        try { Object.keys(localStorage).filter(k => k.startsWith("kk:") && k !== "kk:tema").forEach(k => localStorage.removeItem(k)); } catch (e) {}
        ferdig = {}; quizSvar = {}; oppgLost = {};
        byggSidemeny(); oppdaterFremdrift(); visForside();
      }
    });
  }

  function ruter() {
    const h = decodeURIComponent(location.hash.slice(1));
    document.body.classList.remove("meny-apen");
    $("#meny-knapp").setAttribute("aria-expanded", "false");
    if (h.startsWith("del-")) {
      const d = finnDel(h.slice(4));
      if (d) { visDel(d); window.scrollTo(0, 0); return; }
    }
    const l = finnLeksjon(h);
    if (l) visLeksjon(l); else visForside();
    window.scrollTo(0, 0);
  }

  // ---------------------------------------------------------------- oppstart
  function init() {
    if (!KURS.length) {
      hoved().innerHTML = '<div class="leksjon"><p>Fant ikke kursinnholdet (assets/kurs.js). Kjør <code>python3 verktoy/build.py</code>.</p></div>';
      return;
    }
    byggSidemeny();
    oppdaterFremdrift();
    $("#filter").addEventListener("input", e => {
      const q = e.target.value.trim().toLowerCase();
      const sb = $("#sidebar");
      sb.classList.toggle("filtrert", !!q);
      $$(".sb-del").forEach(d => {
        let treff = 0;
        $$("li", d).forEach(li => { const t = !q || li.dataset.sok.includes(q); li.classList.toggle("skjult", !t); treff += t; });
        d.classList.toggle("skjult", !!q && !treff);
      });
    });
    $("#meny-knapp").addEventListener("click", () => {
      const apen = document.body.classList.toggle("meny-apen");
      $("#meny-knapp").setAttribute("aria-expanded", String(apen));
    });
    $("#hoved").addEventListener("click", () => document.body.classList.remove("meny-apen"));
    $("#tema-knapp").addEventListener("click", () => {
      const rot = document.documentElement;
      const mork = rot.dataset.theme ? rot.dataset.theme === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
      rot.dataset.theme = mork ? "light" : "dark";
      store.set("tema", rot.dataset.theme);
      try { localStorage.setItem("kk:tema", rot.dataset.theme); } catch (e) {}
    });
    document.addEventListener("keydown", e => {
      if (e.altKey || e.ctrlKey || e.metaKey) return;
      const t = e.target;
      if (t.closest && (t.closest(".CodeMirror") || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
      const l = finnLeksjon(decodeURIComponent(location.hash.slice(1)));
      if (!l) return;
      const i = LEKSJONER.indexOf(l);
      if (e.key === "ArrowRight" && LEKSJONER[i + 1]) location.hash = LEKSJONER[i + 1].nr;
      if (e.key === "ArrowLeft" && LEKSJONER[i - 1]) location.hash = LEKSJONER[i - 1].nr;
    });
    window.addEventListener("hashchange", ruter);
    ruter();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
