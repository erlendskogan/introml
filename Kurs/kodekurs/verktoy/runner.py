# Kjøremotor for kodekurset.
# Den samme koden brukes i nettleseren (Pyodide) og av valider.py lokalt,
# slik at det som testes lokalt oppfører seg likt som i kurset.
import ast
import base64
import io
import json
import sys
import traceback

_KK_FIGS_DPI = 90


def _kk_flush_figs(events):
    """Gjør alle åpne matplotlib-figurer om til PNG (som plt.show i Jupyter)."""
    if "matplotlib.pyplot" not in sys.modules:
        return
    plt = sys.modules["matplotlib.pyplot"]
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=_KK_FIGS_DPI, bbox_inches="tight")
        events.append(["img", base64.b64encode(buf.getvalue()).decode()])
    plt.close("all")


def _kk_patch_mpl(events):
    if "matplotlib.pyplot" not in sys.modules:
        return
    plt = sys.modules["matplotlib.pyplot"]
    plt.show = lambda *a, **k: _kk_flush_figs(events)


class _KKWriter:
    def __init__(self, events, kind):
        self.events, self.kind = events, kind

    def write(self, s):
        if not s:
            return 0
        if self.events and self.events[-1][0] == self.kind:
            self.events[-1][1] += s
        else:
            self.events.append([self.kind, s])
        return len(s)

    def flush(self):
        pass


def _kk_skip_repr(val):
    mod = type(val).__module__ or ""
    if mod.startswith("matplotlib"):
        return True
    if isinstance(val, (list, tuple)) and val and all((type(v).__module__ or "").startswith("matplotlib") for v in val):
        return True
    return False


def _kk_format_error(e, src, filename):
    lines = src.splitlines()
    if isinstance(e, SyntaxError) and e.filename == filename:
        nr = e.lineno or 0
        kode = lines[nr - 1].strip() if 0 < nr <= len(lines) else ""
        return f"Linje {nr}: {kode}\n{type(e).__name__}: {e.msg}"
    frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == filename]
    out = []
    if frames:
        f = frames[-1]
        kode = lines[f.lineno - 1].strip() if 0 < f.lineno <= len(lines) else ""
        out.append(f"Linje {f.lineno}: {kode}")
    out.append("".join(traceback.format_exception_only(type(e), e)).strip())
    return "\n".join(out)


def _kk_exec(src, ns, events, filename):
    """Kjør src i ns. Siste linje vises som i Jupyter hvis den er et uttrykk."""
    tree = ast.parse(src, filename=filename)
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    exec(compile(tree, filename, "exec"), ns)
    _kk_patch_mpl(events)
    if last is not None:
        val = eval(compile(last, filename, "eval"), ns)
        _kk_patch_mpl(events)
        if val is not None and not _kk_skip_repr(val):
            mod = type(val).__module__ or ""
            if mod.startswith("pandas") and type(val).__name__ == "DataFrame":
                sys.stdout.flush()
                events.append(["html", val._repr_html_()])
            else:
                print(repr(val))


# --------------------------------------------------------------------------
# Visualisering av variabler (variabelpanelet og display())
# Hver verdi beskrives som en liten JSON-ordbok som app.js tegner.
# --------------------------------------------------------------------------
_KK_MAKS_RADER = 25
_KK_MAKS_KOL = 14
_KK_MAKS_LISTE = 40
_KK_SKJUL = {"In", "Out", "exit", "quit", "get_ipython", "display"}


def _kk_kort(v, n=60):
    if (type(v).__module__ or "").startswith("numpy") and hasattr(v, "item") and getattr(v, "ndim", 1) == 0:
        v = v.item()  # np.float64(3.0) -> 3.0
    try:
        if isinstance(v, float):
            s = f"{v:.4g}"
        else:
            s = repr(v)
    except Exception:  # noqa: BLE001
        s = f"<{type(v).__name__}>"
    return s if len(s) <= n else s[: n - 1] + "…"


def _kk_tall(v):
    """Tall for fargeskala, eller None (NaN, inf, ikke-tall)."""
    try:
        f = float(v)
    except Exception:  # noqa: BLE001
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _kk_celle(v):
    """Formatert celleverdi + om den mangler (NaN/None)."""
    mangler = False
    try:
        import pandas as pd
        mangler = bool(pd.isna(v)) if not isinstance(v, (list, tuple, dict)) else False
    except Exception:  # noqa: BLE001
        mangler = v is None or (isinstance(v, float) and v != v)
    if mangler:
        return "NaN" if v is not None else "None", True
    if isinstance(v, float) or type(v).__name__.startswith("float"):
        return f"{float(v):.4g}", False
    if isinstance(v, str):
        return (v if len(v) <= 30 else v[:29] + "…"), False
    return _kk_kort(v, 30), False


def _kk_df(df, navn):
    import pandas as pd
    if isinstance(df, pd.Series):
        df = df.to_frame(name=df.name if df.name is not None else "verdi")
        art = "Series"
    else:
        art = "DataFrame"
    n, m = df.shape
    kol = list(df.columns[:_KK_MAKS_KOL])
    vis = df.iloc[:_KK_MAKS_RADER, :_KK_MAKS_KOL]
    rader, mangler = [], []
    for _, rad in vis.iterrows():
        cs = [_kk_celle(v) for v in rad.tolist()]
        rader.append([c[0] for c in cs])
        mangler.append([c[1] for c in cs])
    try:
        nan_per_kol = [int(df[c].isna().sum()) for c in kol]
    except Exception:  # noqa: BLE001
        nan_per_kol = [0] * len(kol)
    return {
        "art": "df", "type": art, "navn": navn, "shape": [n, m],
        "kolonner": [str(c) for c in kol], "dtyper": [str(df[c].dtype) for c in kol],
        "indeks": [_kk_kort(i, 20) for i in vis.index], "rader": rader, "mangler": mangler,
        "nan": nan_per_kol, "nan_totalt": int(df.isna().sum().sum()) if n * m < 5_000_000 else None,
        "flere_rader": max(0, n - _KK_MAKS_RADER), "flere_kol": max(0, m - _KK_MAKS_KOL),
    }


def _kk_arr(a, navn):
    import numpy as np
    a = np.asarray(a)
    d = {"art": "arr", "type": "ndarray", "navn": navn, "shape": list(a.shape), "dtype": str(a.dtype)}
    numerisk = a.dtype.kind in "biuf"
    if numerisk and a.size:
        endelig = a[np.isfinite(a)] if a.dtype.kind == "f" else a
        if endelig.size:
            d["min"], d["maks"] = float(endelig.min()), float(endelig.max())
        d["nan"] = int(np.isnan(a).sum()) if a.dtype.kind == "f" else 0
    d["bool"] = a.dtype.kind == "b"
    utsnitt = a
    if a.ndim > 2:
        d["merk"] = f"Viser [0, …] av {a.ndim}D-arrayen"
        while utsnitt.ndim > 2:
            utsnitt = utsnitt[0]
    if a.ndim == 0:
        d["verdier"] = [[_kk_celle(a.item())[0]]]
        d["tall"] = [[_kk_tall(a.item())]]
        return d
    if utsnitt.ndim == 1:
        utsnitt = utsnitt[None, :]
        d["en_d"] = True
    r, k = utsnitt.shape
    heatmap = numerisk and utsnitt.ndim == 2 and (r > 14 or k > 14) and r <= 64 and k <= 64 and not d.get("en_d")
    maks_r, maks_k = (64, 64) if heatmap else (14, 30 if d.get("en_d") else 14)
    del_ = utsnitt[:maks_r, :maks_k]
    d["heatmap"] = bool(heatmap)
    d["tall"] = [[_kk_tall(v) if numerisk else None for v in rad] for rad in del_.tolist()]
    if not heatmap:
        d["verdier"] = [[_kk_celle(v)[0] for v in rad] for rad in del_.tolist()]
    d["flere_rader"], d["flere_kol"] = max(0, r - maks_r), max(0, k - maks_k)
    return d


def _kk_er_estimator(v):
    return hasattr(v, "get_params") and hasattr(v, "fit")


def _kk_attr_sammendrag(v):
    """Kort tekst om en attributtverdi (brukt i modell- og objektvisning)."""
    import_np = sys.modules.get("numpy")
    if import_np is not None and isinstance(v, import_np.ndarray):
        if v.size <= 8:
            return "array(" + _kk_kort([round(float(x), 4) if v.dtype.kind == "f" else x for x in v.ravel().tolist()], 70) + ")" + (f" shape {v.shape}" if v.ndim > 1 else "")
        return f"ndarray {tuple(v.shape)} {v.dtype}"
    if type(v).__name__ in ("DataFrame", "Series"):
        return f"{type(v).__name__} {tuple(v.shape)}"
    if isinstance(v, (list, tuple)) and len(v) > 8:
        return f"{type(v).__name__} med {len(v)} elementer, siste: {_kk_kort(v[-1], 25)}"
    return _kk_kort(v, 80)


def _kk_beskriv(v, navn, dybde=0):
    t = type(v)
    mod = t.__module__ or ""
    try:
        if mod.startswith("pandas") and t.__name__ in ("DataFrame", "Series"):
            return _kk_df(v, navn)
        if mod.startswith("numpy") and t.__name__ == "ndarray":
            return _kk_arr(v, navn)
        if isinstance(v, (bool, int, float, complex, str, type(None))) or (mod.startswith("numpy") and hasattr(v, "dtype")):
            return {"art": "skalar", "type": t.__name__, "navn": navn, "verdi": _kk_kort(v, 200)}
        if isinstance(v, (list, tuple, set, range)):
            elementer = list(v)[:_KK_MAKS_LISTE]
            nostet = elementer and all(isinstance(e, (list, tuple)) for e in elementer) and dybde == 0
            if nostet and max(len(e) for e in elementer) <= 14:
                rader = [[_kk_celle(x)[0] for x in e] for e in elementer[:14]]
                tall = [[_kk_tall(x) if isinstance(x, (int, float)) else None for x in e] for e in elementer[:14]]
                return {"art": "nostet", "type": t.__name__, "navn": navn, "len": len(v), "verdier": rader, "tall": tall,
                        "flere": max(0, len(v) - 14)}
            return {"art": "liste", "type": t.__name__, "navn": navn, "len": len(v),
                    "verdier": [_kk_kort(e, 30) for e in elementer], "flere": max(0, len(v) - _KK_MAKS_LISTE),
                    "indekser": not isinstance(v, set)}
        if isinstance(v, dict):
            par = list(v.items())[:_KK_MAKS_LISTE]
            return {"art": "dict", "type": "dict", "navn": navn, "len": len(v),
                    "par": [[_kk_kort(k, 40), _kk_attr_sammendrag(x)] for k, x in par], "flere": max(0, len(v) - _KK_MAKS_LISTE)}
        if _kk_er_estimator(v):
            try:
                params = {k: _kk_kort(x, 40) for k, x in v.get_params(deep=False).items()}
            except Exception:  # noqa: BLE001
                params = {}
            laert = [[k, _kk_attr_sammendrag(x)] for k, x in vars(v).items() if k.endswith("_") and not k.startswith("_")]
            if laert:  # lærte størrelser som er properties og ikke ligger i vars()
                for k in ("feature_importances_", "explained_variance_ratio_"):
                    if k not in vars(v):
                        try:
                            laert.append([k, _kk_attr_sammendrag(getattr(v, k))])
                        except Exception:  # noqa: BLE001
                            pass
            trent = bool(laert)
            try:
                from sklearn.utils.validation import check_is_fitted
                check_is_fitted(v)
                trent = True
            except Exception:  # noqa: BLE001
                pass
            if hasattr(v, "steps") and isinstance(getattr(v, "steps"), list):  # Pipeline: vis stegene
                params = {}
                laert = []
                for stegnavn, steg in v.steps:
                    lært_steg = [k for k in vars(steg) if k.endswith("_") and not k.startswith("_")] if hasattr(steg, "__dict__") else []
                    laert.append([stegnavn, f"{type(steg).__name__}" + (f": lærte {', '.join(lært_steg[:4])}" if lært_steg else " (ikke trent)")])
            return {"art": "modell", "type": t.__name__, "navn": navn, "params": [[k, x] for k, x in params.items()],
                    "laert": laert, "trent": trent}
        if mod == "__main__" and hasattr(v, "__dict__") and not isinstance(v, type):
            attrs = [[k, _kk_attr_sammendrag(x)] for k, x in vars(v).items() if not k.startswith("__")]
            return {"art": "objekt", "type": t.__name__, "navn": navn, "attrs": attrs[:30]}
    except Exception as e:  # noqa: BLE001  – visningen skal aldri ødelegge kjøringen
        return {"art": "skalar", "type": t.__name__, "navn": navn, "verdi": f"(kunne ikke vises: {e})"}
    return None


def _kk_sammendrag(d):
    """Kort tekst til brikken i variabelpanelet."""
    a = d["art"]
    if a in ("df", "arr"):
        s = d["shape"]
        if a == "df" and d["type"] == "Series":
            return f"Series ({s[0]})"
        return f"{d['type']} {s[0]}×{s[1]}" if a == "df" else f"ndarray {tuple(s)}".replace(",)", ")")
    if a == "skalar":
        return f"{d['type']} = {d['verdi'][:24]}"
    if a in ("liste", "nostet"):
        return f"{d['type']} ({d['len']})"
    if a == "dict":
        return f"dict ({d['len']})"
    if a == "modell":
        return d["type"] + ("" if d["trent"] else " (ikke trent)")
    return d["type"]


def _kk_variabler(ns, start):
    import types
    ut = []
    for navn, v in ns.items():
        if navn in start or navn.startswith("_") or navn in _KK_SKJUL:
            continue
        if isinstance(v, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, type)) or callable(v) and not _kk_er_estimator(v) and not hasattr(v, "__dict__"):
            continue
        if (type(v).__module__ or "").startswith(("matplotlib", "typing")):
            continue
        d = _kk_beskriv(v, navn)
        if d is not None:
            d["kort"] = _kk_sammendrag(d)
            ut.append(d)
        if len(ut) >= 24:
            break
    return ut


def _kk_lag_display(events):
    def display(*objs, **_):
        """Som display() i Jupyter: viser verdien som tabell/rutenett midt i koden."""
        sys.stdout.flush()
        for o in objs:
            d = _kk_beskriv(o, "")
            if d is None:
                print(repr(o))
            else:
                d["kort"] = _kk_sammendrag(d)
                events.append(["vis", d])
    return display


def kk_run(src, test=None):
    """Returnerer JSON: {status, events, feil, variabler}.

    status: "ok" (koden kjørte), "feil" (unntak i koden),
            "bestatt" (testen gikk gjennom), "ikke_bestatt" (testen feilet).
    """
    events = []
    ns = {"__name__": "__main__", "display": _kk_lag_display(events)}
    start = set(ns)
    variabler = []
    old = sys.stdout, sys.stderr
    sys.stdout = _KKWriter(events, "text")
    sys.stderr = _KKWriter(events, "warn")
    status, feil = "ok", None
    try:
        try:
            if "matplotlib" in src or (test and "matplotlib" in test):
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot  # noqa: F401  (så plt.show kan kobles om før koden kjører)
            if "pandas" in src:
                import pandas as _pd
                _pd.set_option("display.width", 160)
                _pd.set_option("display.max_columns", 30)
            _kk_patch_mpl(events)
            _kk_exec(src, ns, events, "<kode>")
            if not test:
                _kk_flush_figs(events)
        except BaseException as e:  # noqa: BLE001 - vi vil vise alle feil til studenten
            if isinstance(e, KeyboardInterrupt):
                raise
            status, feil = "feil", _kk_format_error(e, src, "<kode>")
            _kk_flush_figs(events)
        try:
            variabler = _kk_variabler(ns, start)
        except Exception:  # noqa: BLE001
            variabler = []
        if status == "ok" and test:
            try:
                # testen kjøres FØR figurene vises/lukkes, så plt.gca()/plt.gcf() virker i testen
                exec(compile(test, "<test>", "exec"), ns)
                status = "bestatt"
            except AssertionError as e:
                status, feil = "ikke_bestatt", (str(e) or "Svaret er ikke riktig ennå.")
            except NameError as e:
                status, feil = "ikke_bestatt", f"Fant ikke navnet {getattr(e, 'name', '') or str(e)}. Har du definert det?"
            except Exception as e:  # noqa: BLE001
                status, feil = "ikke_bestatt", f"Sjekken krasjet: {type(e).__name__}: {e}"
        if test:
            _kk_flush_figs(events)
    finally:
        sys.stdout, sys.stderr = old
        if "matplotlib.pyplot" in sys.modules:
            sys.modules["matplotlib.pyplot"].close("all")
    return json.dumps({"status": status, "events": events, "feil": feil, "variabler": variabler}, allow_nan=False, default=str)
