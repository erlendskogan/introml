/* ==========================================================================
   TDT4172 – Interaktivt kurs · felles motor
   Navigasjon, progresjon, quiz, refleksjon, regneoppgaver, flashcards,
   KaTeX-rendering og en liten plotte-hjelper (CK.plot) for canvas-widgets.
   ========================================================================== */
(function () {
  "use strict";

  const CHAPTERS = [
    { id: "01", file: "01-grunnlag.html",      part: "Del I · Veiledet læring", title: "Hva er maskinlæring?",                 desc: "Data, features og targets, preprosessering, induktiv bias og den store arbeidsflyten." },
    { id: "02", file: "02-logistisk.html",     part: "Del I · Veiledet læring", title: "Logistisk regresjon og gradient descent", desc: "Sigmoid, likelihood, cross-entropy, derivasjon av gradienten, læringsrate og trening." },
    { id: "03", file: "03-evaluering.html",    part: "Del I · Veiledet læring", title: "Evaluering og ubalanserte data",        desc: "Confusion matrix, precision/recall, terskel, ROC og AUC, resampling og vektet tap." },
    { id: "04", file: "04-sannsynlighet.html", part: "Del I · Veiledet læring", title: "Entropi, Bayes og dimensjonsforbannelsen", desc: "Informasjon og entropi, Bayes' teorem, naiv Bayes og hvorfor høye dimensjoner er tomme." },
    { id: "05", file: "05-traer.html",         part: "Del I · Veiledet læring", title: "Beslutningstrær",                      desc: "Noder, Gini-urenhet, entropi og informasjonsgevinst, bygging og stoppkriterier." },
    { id: "06", file: "06-regresjon.html",     part: "Del I · Veiledet læring", title: "Lineær og polynomisk regresjon",        desc: "MSE, gradienter for β, feature engineering og polynomiske features." },
    { id: "07", file: "07-generalisering.html",part: "Del I · Veiledet læring", title: "Generalisering: validering og bias–varians", desc: "Trening/validering/test, læringskurver, early stopping, kryssvalidering, bias–varians og regresjonstrær." },
    { id: "08", file: "08-ensemble.html",      part: "Del I · Veiledet læring", title: "Ensemble-modeller",                    desc: "Bootstrap, bagging, random forest, AdaBoost og gradient boosting." },
    { id: "09", file: "09-nevrale-nett.html",  part: "Del II · Nevrale nettverk", title: "Fra perseptron til nevrale nettverk", desc: "Perseptronet, logiske porter og XOR, aktiveringsfunksjoner og arkitektur på matriseform." },
    { id: "10", file: "10-trening-nn.html",    part: "Del II · Nevrale nettverk", title: "Trening av nevrale nettverk",          desc: "Backpropagation, (mini-batch) SGD, epoker, og å bygge/trene modeller i Keras." },
    { id: "11", file: "11-clustering.html",    part: "Del III · Uveiledet læring", title: "Clustering",                         desc: "k-means og k-means++, klyngemetrikker, DBSCAN og avstand i høye dimensjoner." },
    { id: "12", file: "12-dimensjonsreduksjon.html", part: "Del III · Uveiledet læring", title: "Dimensjonsreduksjon: PCA og t-SNE", desc: "Kovarians og egenvektorer, PCA for hånd, SNE/t-SNE og KL-divergens." },
    { id: "13", file: "13-anomali.html",       part: "Del III · Uveiledet læring", title: "Anomalideteksjon",                   desc: "k-means-avstand, DBSCAN-støy og Isolation Forest." },
    { id: "14", file: "14-forsterkning.html",  part: "Del IV · Forsterkningslæring", title: "Forsterkningslæring", desc: "Agent, miljø, Markov-egenskapen, Q-learning og Bellman, Gym-API og Q-tabell, ε-greedy, DQN og catastrophic forgetting." },
    { id: "15", file: "15-xai-etikk.html",     part: "Del V · Forklarbar KI og etikk", title: "Forklarbar KI og etikk", desc: "Tolkbarhet og forklarbarhet, Shapley-verdier (taxieksempelet), SHAP med bakgrunnsdatasett, force plot og beeswarm, SHAP for bilder og tekst, LIME som lokal surrogatmodell, konseptdeteksjon (CAV og TCAV), de fem etiske prinsippene og AI Act (status i Norge, Digital Omnibus, høyrisikokrav og GPAI)." },
    { id: "16", file: "16-eksamen.html",       part: "Del VI · Eksamen", title: "Eksamenstrening",                               desc: "Tidligere eksamener og eksempelspørsmålene fra notatene som interaktive prøver, med forklaringer og eksamensstrategi." },
    { id: "17", file: "17-formelark.html",     part: "Del VI · Eksamen", title: "Formelark og begrepsliste",                     desc: "Alle formler, begreper og «knagger» samlet på ett sted for repetisjon." },
    { id: "18", file: "18-regneoppgaver.html", part: "Del VI · Eksamen", title: "Regneoppgaver steg for steg",                   desc: "Alle typene regneoppgaver fra eksamen og øvingssett, med oppskrift, gjennomregnet eksempel og en oppgave du regner selv." }
  ];

  const LS = "tdt4172:";
  const store = {
    get(k, d) { try { const v = localStorage.getItem(LS + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(LS + k, JSON.stringify(v)); } catch (e) { /* privat modus */ } },
    del(k) { try { localStorage.removeItem(LS + k); } catch (e) {} }
  };

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const el = (tag, attrs = {}, html = "") => { const e = document.createElement(tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (html) e.innerHTML = html; return e; };

  const inChapterDir = /\/kapitler\//.test(location.pathname);
  const root = inChapterDir ? "../" : "";
  const chapterId = document.body.dataset.chapter || null;
  // innebygd i alt.html (hele kurset på én side): kapittelet vises i en iframe
  const embedded = window.parent !== window && /[?&]embed\b/.test(location.search);
  if (embedded) document.documentElement.classList.add("embedded");

  /* ---------------- tema ---------------- */
  const urlTheme = (location.search.match(/[?&]theme=(light|dark)/) || [])[1];
  const savedTheme = (embedded && urlTheme) || store.get("theme", null);
  if (savedTheme) document.documentElement.setAttribute("data-theme", savedTheme);
  const themeListeners = [];
  function toggleTheme() {
    const cur = document.documentElement.getAttribute("data-theme") ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    store.set("theme", next);
    themeListeners.forEach(f => { try { f(); } catch (e) { console.error(e); } });
  }
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => themeListeners.forEach(f => f()));

  /* ---------------- topplinje + skuff ---------------- */
  function buildChrome() {
    const bar = el("header", { class: "topbar" });
    bar.innerHTML = `
      <button class="menu-btn" aria-label="Kapitler">☰ Kapitler</button>
      <a class="brand" href="${root}index.html">TDT4172 <small>Introduksjon til maskinlæring</small></a>
      <span class="spacer"></span>
      <button class="theme-btn" aria-label="Bytt lys/mørk modus">◐</button>
      <div class="progress-line"></div>`;
    document.body.prepend(bar);

    const drawer = el("nav", { class: "drawer", "aria-label": "Kapitler" });
    let html = `<h3>Kapitler</h3><ol>`;
    for (const c of CHAPTERS) {
      const s = store.get("ch:" + c.id, null);
      const badge = s && s.total ? `${s.correct}/${s.total}` : "";
      html += `<li><a href="${root}kapitler/${c.file}" class="${c.id === chapterId ? "current" : ""}"><span class="num">${c.id}</span><span>${c.title}${c.expected ? " <span class='muted'>(forventet)</span>" : ""}</span><span class="badge">${badge}</span></a></li>`;
    }
    html += `</ol><p style="margin:14px 10px 0"><a href="${root}kompendium.html">Kompendium med kapitteltester →</a></p><p style="margin:6px 10px 0"><a href="${root}alt.html${/^\d+$/.test(chapterId || "") ? "#k" + chapterId : ""}">Hele kurset på én side →</a></p>`;
    drawer.innerHTML = html;
    // innholdsfortegnelse for dette kapittelet
    const h2s = $$("main h2[id]");
    if (h2s.length) {
      const toc = el("div", { class: "toc" });
      toc.innerHTML = `<h3>I dette kapittelet</h3>` + h2s.map(h => `<a href="#${h.id}" style="display:block">${h.textContent}</a>`).join("");
      drawer.appendChild(toc);
    }
    const scrim = el("div", { class: "scrim" });
    document.body.append(drawer, scrim);
    const toggle = (open) => { drawer.classList.toggle("open", open); scrim.classList.toggle("open", open); };
    $(".menu-btn", bar).onclick = () => toggle(!drawer.classList.contains("open"));
    scrim.onclick = () => toggle(false);
    drawer.addEventListener("click", e => { if (e.target.closest("a")) toggle(false); });
    document.addEventListener("keydown", e => { if (e.key === "Escape") toggle(false); });
    $(".theme-btn", bar).onclick = toggleTheme;

    const line = $(".progress-line", bar);
    const onScroll = () => {
      const h = document.documentElement.scrollHeight - innerHeight;
      line.style.width = (h > 0 ? Math.min(100, scrollY / h * 100) : 0) + "%";
    };
    addEventListener("scroll", onScroll, { passive: true }); onScroll();
  }

  /* ---------------- forrige/neste ---------------- */
  function buildPager() {
    if (!chapterId) return;
    const i = CHAPTERS.findIndex(c => c.id === chapterId);
    if (i < 0) return;
    const main = $("main"); if (!main || $(".pager", main)) return;
    const p = el("nav", { class: "pager" });
    const prev = CHAPTERS[i - 1], next = CHAPTERS[i + 1];
    p.innerHTML =
      (prev ? `<a class="prev" href="${prev.file}"><small>← Forrige</small>${prev.id} · ${prev.title}</a>` : `<a class="prev" href="../index.html"><small>← Tilbake</small>Kursoversikt</a>`) +
      (next ? `<a class="next" href="${next.file}"><small>Neste →</small>${next.id} · ${next.title}</a>` : `<a class="next" href="../index.html"><small>Ferdig →</small>Kursoversikt</a>`);
    main.appendChild(p);
  }

  /* ---------------- quiz ---------------- */
  function quizKey(q, idx) { return "quiz:" + (q.dataset.id || (chapterId + "-auto-" + idx)); }

  function setupQuiz(q, idx) {
    const key = quizKey(q, idx);
    const items = $$(":scope > ul > li", q);
    const multi = q.hasAttribute("data-multi");
    const actions = el("div", { class: "q-actions" });
    const verdict = el("span", { class: "verdict" });
    const checkBtn = el("button", { class: "btn", type: "button" }, "Sjekk svar");
    const retryBtn = el("button", { class: "btn ghost", type: "button" }, "Prøv igjen");
    retryBtn.style.display = "none";
    if (multi) actions.append(checkBtn);
    actions.append(retryBtn, verdict);
    const explain = $(":scope > .explain", q);
    q.insertBefore(actions, explain || null);

    function grade(picks, save = true) {
      q.classList.add("done");
      let allRight = true;
      items.forEach((li, i) => {
        const correct = li.hasAttribute("data-correct");
        const picked = picks.includes(i);
        li.classList.toggle("picked", picked);
        if (correct) li.classList.add("right");
        if (picked && !correct) li.classList.add("wrong");
        if (correct !== picked) allRight = false;
      });
      verdict.textContent = allRight ? "Riktig!" : (multi ? "Ikke helt – se fasit." : "Feil – se forklaringen.");
      verdict.className = "verdict " + (allRight ? "ok" : "bad");
      checkBtn.style.display = "none"; retryBtn.style.display = "";
      if (save) { store.set(key, { picks, ok: allRight }); updateScore(); }
    }
    function reset() {
      q.classList.remove("done");
      items.forEach(li => li.classList.remove("picked", "right", "wrong"));
      verdict.textContent = ""; retryBtn.style.display = "none"; checkBtn.style.display = "";
      store.del(key); updateScore();
    }
    items.forEach((li, i) => li.addEventListener("click", () => {
      if (q.classList.contains("done")) return;
      if (multi) li.classList.toggle("picked");
      else grade([i]);
    }));
    checkBtn.onclick = () => {
      const picks = items.map((li, i) => li.classList.contains("picked") ? i : -1).filter(i => i >= 0);
      if (!picks.length) { verdict.textContent = "Velg minst ett alternativ."; verdict.className = "verdict"; return; }
      grade(picks);
    };
    retryBtn.onclick = reset;
    const saved = store.get(key, null);
    if (saved && Array.isArray(saved.picks)) grade(saved.picks, false);
  }

  /* ---------------- regneoppgave ---------------- */
  function parseNum(s) {
    s = String(s).trim().replace(/\s/g, "").replace(",", ".");
    if (/^-?\d*\.?\d+\/\d*\.?\d+$/.test(s)) { const [a, b] = s.split("/"); return parseFloat(a) / parseFloat(b); }
    if (s.endsWith("%")) return parseFloat(s) / 100;
    return parseFloat(s);
  }
  function setupCalc(c, idx) {
    const key = "calc:" + (c.dataset.id || (chapterId + "-calc-" + idx));
    const ans = parseFloat(c.dataset.answer);
    const tol = c.dataset.tol != null ? parseFloat(c.dataset.tol) : Math.max(1e-9, Math.abs(ans) * 0.01);
    const row = el("div", { class: "row" });
    const input = el("input", { type: "text", inputmode: "decimal", placeholder: "Ditt svar", "aria-label": "Ditt svar" });
    const btn = el("button", { class: "btn", type: "button" }, "Sjekk");
    const show = el("button", { class: "btn ghost", type: "button" }, "Vis løsning");
    const verdict = el("span", { class: "verdict" });
    row.append(input, btn, show, verdict);
    if (c.dataset.unit) row.insertBefore(el("span", { class: "muted" }, c.dataset.unit), btn);
    const sol = $(":scope > .solution", c);
    c.insertBefore(row, sol || null);
    function check(save = true) {
      const v = parseNum(input.value);
      if (isNaN(v)) { verdict.textContent = "Skriv inn et tall."; verdict.className = "verdict"; return; }
      const ok = Math.abs(v - ans) <= tol;
      verdict.textContent = ok ? "Riktig!" : "Ikke helt – prøv igjen eller se løsningen.";
      verdict.className = "verdict " + (ok ? "ok" : "bad");
      if (ok) c.classList.add("show");
      if (save) { store.set(key, { v: input.value, ok }); updateScore(); }
    }
    btn.onclick = () => check();
    input.addEventListener("keydown", e => { if (e.key === "Enter") check(); });
    show.onclick = () => c.classList.toggle("show");
    const saved = store.get(key, null);
    if (saved) { input.value = saved.v; check(false); }
  }

  /* ---------------- refleksjon ---------------- */
  function setupReflect(r, idx) {
    const key = "reflect:" + (r.dataset.id || (chapterId + "-r-" + idx));
    const ta = el("textarea", { placeholder: "Skriv tankene dine her før du ser forslaget – det er der læringen skjer. (Lagres i nettleseren.)" });
    const actions = el("div", { class: "r-actions" });
    const btn = el("button", { class: "btn ghost", type: "button" }, "Vis forslag til svar");
    const note = el("span", { class: "saved-note" });
    actions.append(btn, note);
    const ans = $(":scope > .answer", r);
    r.insertBefore(ta, ans || null);
    r.insertBefore(actions, ans || null);
    ta.value = store.get(key, "");
    let t;
    ta.addEventListener("input", () => { clearTimeout(t); t = setTimeout(() => { store.set(key, ta.value); note.textContent = "Lagret"; }, 400); });
    btn.onclick = () => { r.classList.toggle("show"); btn.textContent = r.classList.contains("show") ? "Skjul forslag" : "Vis forslag til svar"; };
    if (!ans) btn.style.display = "none";
  }

  /* ---------------- flashcards ---------------- */
  function setupCards() {
    $$(".cards .card").forEach(c => {
      if (!$(".front", c) && !$(".inner", c)) return;
      if (!$(".inner", c)) {
        const f = $(".front", c), b = $(".back", c);
        const inner = el("div", { class: "inner" }); inner.append(f, b); c.appendChild(inner);
      }
      c.tabIndex = 0;
      c.addEventListener("click", () => c.classList.toggle("flipped"));
      c.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); c.classList.toggle("flipped"); } });
    });
  }

  /* ---------------- poeng for kapittelet ---------------- */
  function updateScore() {
    if (!chapterId) return;
    const qs = $$(".quiz"), cs = $$(".calc[data-answer]");
    let total = qs.length + cs.length, correct = 0, answered = 0;
    qs.forEach((q, i) => { const s = store.get(quizKey(q, i), null); if (s) { answered++; if (s.ok) correct++; } });
    cs.forEach((c, i) => { const s = store.get("calc:" + (c.dataset.id || (chapterId + "-calc-" + i)), null); if (s) { answered++; if (s.ok) correct++; } });
    store.set("ch:" + chapterId, { total, correct, answered, t: Date.now() });
    $$(".score-panel").forEach(p => {
      p.innerHTML = `<div><div class="big">${correct} / ${total}</div><div class="muted small">riktige svar i dette kapittelet (${answered} besvart)</div></div>
        <span class="spacer" style="flex:1"></span><button class="btn ghost" type="button">Nullstill kapittelet</button>`;
      $("button", p).onclick = () => {
        if (!confirmReset()) return;
        qs.forEach((q, i) => store.del(quizKey(q, i)));
        cs.forEach((c, i) => store.del("calc:" + (c.dataset.id || (chapterId + "-calc-" + i))));
        location.reload();
      };
    });
  }
  function confirmReset() { return window.confirm ? window.confirm("Nullstille alle svar i dette kapittelet?") : true; }

  /* ---------------- KaTeX ---------------- */
  function renderMath() {
    if (window.renderMathInElement) {
      renderMathInElement(document.body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false },
          { left: "$", right: "$", display: false }
        ],
        ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "canvas"],
        throwOnError: false
      });
    }
  }

  /* ==========================================================================
     CK.plot – liten plotte-hjelper for canvas
     Bruk:
       const P = CK.plot(canvas, { xmin:-6, xmax:6, ymin:-0.1, ymax:1.1 });
       P.clear(); P.grid(); P.axes({xlabel:"z", ylabel:"σ(z)"});
       P.fn(z => 1/(1+Math.exp(-z)), { color: P.col("c1"), width: 2.5 });
       P.points([{x:1,y:.7}], { color: P.col("c2"), r: 4 });
     Canvas-elementets width/height-attributter er logisk størrelse; DPR håndteres.
     ========================================================================== */
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue("--" + name).trim() || "#888"; }

  function plot(canvas, view = {}) {
    const ctx = canvas.getContext("2d");
    const W = +canvas.getAttribute("width") || 600, H = +canvas.getAttribute("height") || 360;
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.aspectRatio = W + " / " + H;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const P = {
      ctx, W, H,
      pad: Object.assign({ l: 48, r: 16, t: 14, b: 40 }, view.pad || {}),
      xmin: view.xmin ?? 0, xmax: view.xmax ?? 1, ymin: view.ymin ?? 0, ymax: view.ymax ?? 1,
      col: cssVar,
      setView(v) { Object.assign(P, v); return P; },
      X(x) { return P.pad.l + (x - P.xmin) / (P.xmax - P.xmin) * (W - P.pad.l - P.pad.r); },
      Y(y) { return H - P.pad.b - (y - P.ymin) / (P.ymax - P.ymin) * (H - P.pad.t - P.pad.b); },
      invX(px) { return P.xmin + (px - P.pad.l) / (W - P.pad.l - P.pad.r) * (P.xmax - P.xmin); },
      invY(py) { return P.ymin + (H - P.pad.b - py) / (H - P.pad.t - P.pad.b) * (P.ymax - P.ymin); },
      /* museposisjon (logiske px) fra et event */
      mouse(e) { const r = canvas.getBoundingClientRect(); const t = e.touches ? e.touches[0] : e; return { px: (t.clientX - r.left) * W / r.width, py: (t.clientY - r.top) * H / r.height }; },
      clear() { ctx.clearRect(0, 0, W, H); ctx.fillStyle = cssVar("surface"); ctx.fillRect(0, 0, W, H); return P; },
      ticks(a, b, n = 6) {
        const span = b - a, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
        const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || raw;
        const out = []; for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
        return out;
      },
      grid(opt = {}) {
        ctx.save(); ctx.strokeStyle = cssVar("grid"); ctx.lineWidth = 1;
        const xt = opt.xticks || P.ticks(P.xmin, P.xmax, opt.nx || 6), yt = opt.yticks || P.ticks(P.ymin, P.ymax, opt.ny || 5);
        xt.forEach(x => { ctx.beginPath(); ctx.moveTo(P.X(x), P.pad.t); ctx.lineTo(P.X(x), H - P.pad.b); ctx.stroke(); });
        yt.forEach(y => { ctx.beginPath(); ctx.moveTo(P.pad.l, P.Y(y)); ctx.lineTo(W - P.pad.r, P.Y(y)); ctx.stroke(); });
        ctx.restore(); return P;
      },
      axes(opt = {}) {
        ctx.save(); ctx.strokeStyle = cssVar("axis"); ctx.fillStyle = cssVar("muted"); ctx.lineWidth = 1.2;
        ctx.font = "12px " + cssVar("font");
        const x0 = Math.min(Math.max(0, P.ymin), P.ymax), y0 = Math.min(Math.max(0, P.xmin), P.xmax);
        const ax = opt.frame ? P.ymin : x0, ay = opt.frame ? P.xmin : y0;
        ctx.beginPath(); ctx.moveTo(P.pad.l, P.Y(ax)); ctx.lineTo(W - P.pad.r, P.Y(ax)); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(P.X(ay), P.pad.t); ctx.lineTo(P.X(ay), H - P.pad.b); ctx.stroke();
        const fmt = opt.fmt || (v => Math.abs(v) >= 1000 ? v.toExponential(0) : String(+v.toFixed(3)));
        if (opt.xticks !== false) {
          ctx.textAlign = "center"; ctx.textBaseline = "top";
          (Array.isArray(opt.xticks) ? opt.xticks : P.ticks(P.xmin, P.xmax, opt.nx || 6)).forEach(x => ctx.fillText(fmt(x), P.X(x), H - P.pad.b + 6));
        }
        if (opt.yticks !== false) {
          ctx.textAlign = "right"; ctx.textBaseline = "middle";
          (Array.isArray(opt.yticks) ? opt.yticks : P.ticks(P.ymin, P.ymax, opt.ny || 5)).forEach(y => ctx.fillText(fmt(y), P.pad.l - 6, P.Y(y)));
        }
        ctx.fillStyle = cssVar("text-soft"); ctx.font = "13px " + cssVar("font");
        if (opt.xlabel) { ctx.textAlign = "right"; ctx.textBaseline = "bottom"; ctx.fillText(opt.xlabel, W - P.pad.r, H - 4); }
        if (opt.ylabel) { ctx.save(); ctx.translate(12, P.pad.t); ctx.rotate(-Math.PI / 2); ctx.textAlign = "right"; ctx.textBaseline = "top"; ctx.fillText(opt.ylabel, 0, -10); ctx.restore(); }
        ctx.restore(); return P;
      },
      fn(f, opt = {}) {
        ctx.save(); ctx.strokeStyle = opt.color || cssVar("c1"); ctx.lineWidth = opt.width || 2.2; if (opt.dash) ctx.setLineDash(opt.dash);
        ctx.beginPath(); let started = false; const n = opt.samples || 400;
        for (let i = 0; i <= n; i++) {
          const x = P.xmin + (P.xmax - P.xmin) * i / n, y = f(x);
          if (!isFinite(y)) { started = false; continue; }
          const py = Math.max(-1e4, Math.min(1e4, P.Y(y)));
          if (!started) { ctx.moveTo(P.X(x), py); started = true; } else ctx.lineTo(P.X(x), py);
        }
        ctx.stroke(); ctx.restore(); return P;
      },
      path(pts, opt = {}) {
        ctx.save(); ctx.strokeStyle = opt.color || cssVar("c1"); ctx.lineWidth = opt.width || 2; if (opt.dash) ctx.setLineDash(opt.dash);
        ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(P.X(p.x), P.Y(p.y)) : ctx.moveTo(P.X(p.x), P.Y(p.y)));
        if (opt.fill) { ctx.fillStyle = opt.fill; ctx.fill(); }
        ctx.stroke(); ctx.restore(); return P;
      },
      line(x1, y1, x2, y2, opt = {}) { return P.path([{ x: x1, y: y1 }, { x: x2, y: y2 }], opt); },
      points(pts, opt = {}) {
        ctx.save();
        pts.forEach(p => {
          ctx.beginPath(); ctx.arc(P.X(p.x), P.Y(p.y), p.r || opt.r || 4, 0, 2 * Math.PI);
          ctx.fillStyle = p.color || opt.color || cssVar("c1"); ctx.globalAlpha = opt.alpha ?? 0.9; ctx.fill();
          if (opt.stroke !== false) { ctx.globalAlpha = 1; ctx.strokeStyle = opt.strokeColor || cssVar("surface"); ctx.lineWidth = 1; ctx.stroke(); }
        });
        ctx.restore(); return P;
      },
      rect(x1, y1, x2, y2, opt = {}) {
        ctx.save(); const a = P.X(x1), b = P.Y(y2), w = P.X(x2) - a, h = P.Y(y1) - b;
        if (opt.fill) { ctx.fillStyle = opt.fill; ctx.globalAlpha = opt.alpha ?? 1; ctx.fillRect(a, b, w, h); ctx.globalAlpha = 1; }
        if (opt.color) { ctx.strokeStyle = opt.color; ctx.lineWidth = opt.width || 1.5; ctx.strokeRect(a, b, w, h); }
        ctx.restore(); return P;
      },
      text(s, x, y, opt = {}) {
        ctx.save(); ctx.fillStyle = opt.color || cssVar("text"); ctx.font = (opt.bold ? "600 " : "") + (opt.size || 13) + "px " + cssVar("font");
        ctx.textAlign = opt.align || "left"; ctx.textBaseline = opt.baseline || "middle";
        ctx.fillText(s, opt.px ? x : P.X(x), opt.px ? y : P.Y(y)); ctx.restore(); return P;
      },
      arrow(x1, y1, x2, y2, opt = {}) {
        const a = P.X(x1), b = P.Y(y1), c = P.X(x2), d = P.Y(y2), ang = Math.atan2(d - b, c - a), s = opt.head || 9;
        ctx.save(); ctx.strokeStyle = ctx.fillStyle = opt.color || cssVar("text"); ctx.lineWidth = opt.width || 2;
        ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(c, d); ctx.lineTo(c - s * Math.cos(ang - .4), d - s * Math.sin(ang - .4)); ctx.lineTo(c - s * Math.cos(ang + .4), d - s * Math.sin(ang + .4)); ctx.closePath(); ctx.fill();
        ctx.restore(); return P;
      },
      legend(items, opt = {}) {
        ctx.save(); ctx.font = "13px " + cssVar("font"); ctx.textBaseline = "middle";
        let x = opt.x ?? (P.pad.l + 10), y = opt.y ?? (P.pad.t + 12);
        items.forEach(it => {
          ctx.fillStyle = it.color; ctx.fillRect(x, y - 5, 14, 10); ctx.fillStyle = cssVar("text-soft");
          ctx.textAlign = "left"; ctx.fillText(it.label, x + 20, y); y += 18;
        });
        ctx.restore(); return P;
      }
    };
    return P;
  }

  /* deterministisk tilfeldighet (så figurer ser like ut hver gang) */
  function rng(seed = 42) {
    let s = seed >>> 0;
    const r = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    r.normal = (mu = 0, sd = 1) => { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return mu + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    return r;
  }

  /* kjør en tegnefunksjon nå, ved temabytte og ved resize */
  function live(draw) { draw(); themeListeners.push(draw); }

  /* ---------------- forside ---------------- */
  function buildIndex() {
    const host = $("#chapter-grid"); if (!host) return;
    let part = null, html = "";
    let tot = 0, cor = 0;
    for (const c of CHAPTERS) {
      if (c.part !== part) { if (part) html += `</div>`; part = c.part; html += `<h2 class="part-title">${part}</h2><div class="grid-cards">`; }
      const s = store.get("ch:" + c.id, null);
      const pct = s && s.total ? Math.round(100 * s.correct / s.total) : 0;
      if (s && s.total) { tot += s.total; cor += s.correct; }
      html += `<a class="ch-card${c.expected ? " expected" : ""}" href="kapitler/${c.file}">
        <span class="n">KAPITTEL ${c.id}${c.expected ? " · FORVENTET" : ""}</span>
        <span class="t">${c.title}</span><span class="d">${c.desc}</span>
        <span class="bar"><i style="width:${pct}%"></i></span>
        <span class="meta">${s && s.total ? `${s.correct}/${s.total} riktige · ${s.answered} besvart` : "Ikke startet"}</span></a>`;
    }
    html += `</div>`;
    host.innerHTML = html;
    const tp = $("#total-progress");
    if (tp) tp.textContent = tot ? `${cor} av ${tot} oppgaver riktig i kapitlene du har startet på.` : "Du har ikke svart på noen oppgaver ennå.";
  }

  /* ---------------- oppstart ---------------- */
  /* ---------------- innebygd modus (alt.html) ---------------- */
  function setupEmbed() {
    const post = msg => window.parent.postMessage(Object.assign({ ck: true, chapter: chapterId }, msg), "*");
    const sendHeight = () => post({ type: "height", h: Math.ceil(document.body.getBoundingClientRect().height) });
    if (window.ResizeObserver) new ResizeObserver(sendHeight).observe(document.body);
    addEventListener("load", sendHeight);
    sendHeight();
    post({ type: "toc", items: $$("main h2[id]").map(h => ({ id: h.id, text: h.textContent })) });

    const scrollToId = id => {
      let t = null;
      try { t = id ? document.getElementById(decodeURIComponent(id)) : null; } catch (e) {}
      post({ type: "scroll", y: t ? Math.max(0, t.getBoundingClientRect().top + scrollY - 8) : 0 });
    };
    const chapterOf = url => {
      if (!/\/kapitler\//.test(url.pathname)) return null;
      const f = url.pathname.split("/").pop();
      return CHAPTERS.find(c => c.file === f) || null;
    };
    // lenker til andre sider skal åpnes i toppvinduet, ikke inne i rammen
    $$("a[href]").forEach(a => {
      const h = a.getAttribute("href");
      if (h.startsWith("#") || a.target) return;
      try { if (!chapterOf(new URL(a.href))) a.target = "_top"; } catch (e) {}
    });
    document.addEventListener("click", e => {
      const a = e.target.closest && e.target.closest("a[href]");
      if (!a || e.defaultPrevented || e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey || a.target === "_blank") return;
      const h = a.getAttribute("href");
      if (h.startsWith("#")) { e.preventDefault(); scrollToId(h.slice(1)); return; }
      let url; try { url = new URL(a.href); } catch (err) { return; }
      const c = chapterOf(url);
      if (!c) return;
      e.preventDefault();
      if (c.id === chapterId) scrollToId(url.hash.slice(1));
      else post({ type: "goto", target: c.id, hash: url.hash.slice(1) });
    });
    addEventListener("message", e => {
      if (e.source !== window.parent || !e.data || !e.data.ck) return;
      if (e.data.type === "theme") {
        if (e.data.theme) document.documentElement.setAttribute("data-theme", e.data.theme);
        else document.documentElement.removeAttribute("data-theme");
        themeListeners.forEach(f => { try { f(); } catch (err) { console.error(err); } });
      } else if (e.data.type === "find") scrollToId(e.data.hash);
    });
  }

  function init() {
    if (embedded) setupEmbed(); else buildChrome();
    $$(".quiz").forEach(setupQuiz);
    $$(".calc[data-answer]").forEach(setupCalc);
    $$(".reflect").forEach(setupReflect);
    setupCards();
    if (!embedded) buildPager();
    updateScore();
    buildIndex();
    renderMath();
    if (chapterId && !embedded && CHAPTERS.some(c => c.id === chapterId)) store.set("last", chapterId);
    const cont = $("#continue-link");
    if (cont) {
      const last = store.get("last", null), c = CHAPTERS.find(x => x.id === last);
      if (c) { cont.href = "kapitler/" + c.file; cont.textContent = `Fortsett der du slapp: ${c.id} · ${c.title} →`; cont.style.display = ""; }
    }
  }

  window.CK = { CHAPTERS, store, plot, rng, live, cssVar, $, $$, onTheme: f => themeListeners.push(f) };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
