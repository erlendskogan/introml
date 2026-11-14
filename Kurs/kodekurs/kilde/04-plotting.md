=== DEL 4 | plotting | Plotting med matplotlib
Et godt plott avslører mer enn en tabell med tall: om dataene kan skilles, om modellen overtilpasser, eller hvor agenten har lært noe. Her lærer du plottene som går igjen i prosjektene: linjeplott, scatter, histogram, stolpediagram, bilder, heatmaps og beslutningsgrenser.

--- LEKSJON 4.1 | linjeplott | Linjeplott
Plotting i Python gjøres nesten alltid med `matplotlib.pyplot`, importert som `plt`. `plt.plot(x, y)` tegner en linje gjennom punktene `(x[0], y[0])`, `(x[1], y[1])`, … og `plt.show()` viser figuren.

```python
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-6, 6, 100)
sigmoid = 1 / (1 + np.exp(-z))

plt.plot(z, sigmoid)
plt.show()
```

## Navn på aksene, tittel og rutenett
Et plott uten aksenavn er vanskelig å lese. `plt.xlabel`, `plt.ylabel` og `plt.title` setter tekstene, og `plt.grid(True)` legger på et rutenett. `plt.figure(figsize=(bredde, høyde))` lager en ny figur med valgt størrelse (i tommer). Skal du ha flere linjer, gir du hver en `label`, og `plt.legend()` viser forklaringen.

Den vanligste figuren i maskinlæring er **tapskurven**: tapet (loss) for hver epoke, for treningsdata og valideringsdata.

```python
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 61)
train_loss = 1.5 * np.exp(-epochs / 12) + 0.10
val_loss = 1.5 * np.exp(-epochs / 12) + 0.15 + 0.0002 * np.maximum(0, epochs - 25) ** 2

plt.figure(figsize=(6, 4))
plt.plot(epochs, train_loss, label="Train loss")
plt.plot(epochs, val_loss, label="Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Tapskurve")
plt.grid(True)
plt.legend()
plt.show()
```

Fra rundt epoke 40 begynner valideringstapet å stige, mens treningstapet fortsetter å synke. Det er tegnet på **overtilpasning** (del 7 og 9).

::: husk Bare y gir indekser på x-aksen
Skriver du `plt.plot(losses)` uten x-verdier, bruker matplotlib indeksene 0, 1, 2, … som x. Det er derfor `plt.plot(history["train"])` gir epokenummer (fra 0) langs x-aksen i prosjekt 3.
:::

## Utseende: farge, linjetype og markører
Valgfrie argumenter styrer utseendet: `color` (eller `c`), `linestyle`/`ls` (`"--"` stiplet, `":"` prikket), `marker` (`"o"`, `"x"`), `linewidth`/`lw` og `alpha` (gjennomsiktighet fra 0 til 1).

```python
import matplotlib.pyplot as plt

losses = [0.9, 0.6, 0.45, 0.38, 0.35, 0.33]
accuracies = [0.55, 0.70, 0.78, 0.82, 0.84, 0.85]

plt.plot(losses, c="red", marker="o", label="Loss")
plt.plot(accuracies, c="blue", ls="--", label="Accuracy")
plt.xlabel("Epoke")
plt.legend()
plt.show()
```

## Støyete kurver: glidende snitt og log-skala
I forsterkningslæring varierer antall steg per episode enormt. Prosjekt 4 plotter derfor rådataene svakt (`alpha=0.4`), legger et glidende snitt oppå (leksjon 2.11), og bruker logaritmisk y-akse med `plt.yscale("log")`.

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
episoder = np.arange(1000)
steps_per_episode = (300 * np.exp(-episoder / 150) + 20) * rng.lognormal(0, 0.5, 1000)

plt.figure(figsize=(8, 4))
plt.plot(steps_per_episode, alpha=0.4, label="Steps per episode")
plt.plot(np.convolve(steps_per_episode, np.ones(50) / 50, mode="valid"), label="Moving average (50 episodes)")
plt.yscale("log")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Steps per episode during training")
plt.legend()
plt.show()
```

::: prosjekt Prosjekt 1, 3 og 4
Forelesningsnotatene plotter `log_reg.losses` og `log_reg.train_accuracies` på denne måten for logistisk regresjon (prosjekt 1). Prosjekt 3 plotter train- og valideringstap for nettverket, og prosjekt 4 plotter steg per episode som over.
:::

::: oppgave Plott tapskurven
1. Finn epoken med lavest valideringstap og lagre den i `beste_epoke` (bruk `np.argmin` og `epochs`).
2. Plott `train_loss` og `val_loss` mot `epochs` i figuren `fig`, med `label="train"` og `label="validation"`. Ta med `plt.xlabel("Epoke")`, `plt.ylabel("Loss")` og `plt.legend()`.
```start
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 31)
train_loss = 1 / epochs + 0.1
val_loss = 1 / epochs + 0.15 + 0.004 * (epochs - 12) ** 2 / 10

beste_epoke = ...

fig = plt.figure(figsize=(6, 4))
# plott de to kurvene her

plt.show()
```
```losning
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 31)
train_loss = 1 / epochs + 0.1
val_loss = 1 / epochs + 0.15 + 0.004 * (epochs - 12) ** 2 / 10

beste_epoke = epochs[np.argmin(val_loss)]

fig = plt.figure(figsize=(6, 4))
plt.plot(epochs, train_loss, label="train")
plt.plot(epochs, val_loss, label="validation")
plt.xlabel("Epoke")
plt.ylabel("Loss")
plt.legend()
plt.show()
```
```test
import numpy as np
assert beste_epoke is not ... and int(beste_epoke) == 17, f"beste_epoke skal være epoken (ikke indeksen) med lavest val_loss. Fikk {beste_epoke}. Tips: epochs[np.argmin(val_loss)]."
assert len(fig.axes) == 1, "Figuren har ingen akser ennå. Har du kalt plt.plot etter fig = plt.figure(...)?"
ax = fig.axes[0]
assert len(ax.lines) == 2, f"Figuren skal ha 2 linjer, men har {len(ax.lines)}."
labels = sorted(l.get_label() for l in ax.lines)
assert labels == ["train", "validation"], f"Linjene skal ha label 'train' og 'validation', men har {labels}."
for l in ax.lines:
    ventet = train_loss if l.get_label() == "train" else val_loss
    assert np.allclose(l.get_xdata(), epochs) and np.allclose(l.get_ydata(), ventet), f"Linjen '{l.get_label()}' har feil data. Plott mot epochs."
assert ax.get_legend() is not None, "Husk plt.legend() så linjenavnene vises."
assert ax.get_xlabel() and ax.get_ylabel(), "Husk plt.xlabel(...) og plt.ylabel(...)."
```
:::

::: quiz
Hva er langs x-aksen etter denne koden?
```kode
losses = [0.9, 0.5, 0.3, 0.25]
plt.plot(losses)
plt.show()
```
- [ ] Verdiene 0.9, 0.5, 0.3, 0.25
- [x] Indeksene 0, 1, 2, 3
- [ ] Tallene 1, 2, 3, 4
- [ ] Ingenting, koden gir feilmelding fordi x mangler
? Med bare ett argument tolkes det som y-verdier, og x blir indeksene som starter på 0.
:::

::: quiz
Du har tegnet to linjer med `plt.plot(...)` og kalt `plt.legend()`, men forklaringen er tom og du får en advarsel. Hva mangler?
- [ ] `plt.show()` før `plt.legend()`
- [x] `label="..."` i hvert `plt.plot`-kall
- [ ] `plt.title(...)`
- [ ] `plt.grid(True)`
? `legend()` viser teksten fra `label`-argumentene. Uten labels har den ingenting å vise.
:::

::: oppsummering
- `import matplotlib.pyplot as plt`, så `plt.plot(x, y)` og `plt.show()`.
- `plt.plot(y)` alene bruker indeksene 0, 1, 2, … som x.
- `plt.xlabel`, `plt.ylabel`, `plt.title`, `plt.grid(True)`, `plt.figure(figsize=(b, h))`.
- `label="..."` i hvert plott pluss `plt.legend()` gir forklaring.
- Stil: `c`/`color`, `ls`/`linestyle`, `marker`, `alpha`. `plt.yscale("log")` for log-skala.
:::

--- LEKSJON 4.2 | scatter | Scatter: punkter med farge etter klasse
`plt.scatter(x, y)` tegner ett punkt per datapunkt, uten linjer mellom. Det er standardplottet for å se på data med to features. Slik starter prosjekt 1 med regresjonsdataene:

```python
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("regression.csv")

plt.figure(figsize=(6, 4))
plt.scatter(data["Net_Activity"], data["Energy"], c="blue", label="Data points")
plt.grid(True)
plt.xlabel("Network Activity", fontsize=14)
plt.ylabel("Energy", fontsize=14)
plt.title("Energy vs. Traffic", fontsize=16)
plt.legend()
plt.show()
```

## Farge etter klasse: c og cmap
For klassifisering vil vi se hvilke punkter som tilhører hvilken klasse. Gir du `c` en liste med tall (labelene), får hvert punkt en farge fra et **fargekart** (`cmap`). `"coolwarm"` gir blått for lave verdier og rødt for høye. `plt.colorbar()` viser hvilken farge som betyr hva.

```python
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]

plt.figure(figsize=(6, 4))
plt.scatter(train["x0"], train["x1"], c=train["y"], cmap="coolwarm")
plt.grid(True)
plt.xlabel("x0", fontsize=14)
plt.ylabel("x1", fontsize=14)
plt.title("not linearly separable", fontsize=16)
plt.colorbar()
plt.show()
```

::: prosjekt Prosjekt 1, 2 og 3
Denne cellen (med `c=train['y'], cmap='coolwarm'` og `plt.colorbar()`) står nesten likt i prosjekt 1, 2 og 3. Plottet over viser hvorfor en vanlig logistisk regresjon ikke klarer prosjekt 1: ingen rett linje skiller de røde fra de blå punktene.
:::

`c` virker også for kontinuerlige verdier. Da viser fargen størrelsen på `y`, slik prosjekt 3 gjør for regresjonsdataene:

```python
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("regression_data.csv")
train = data[data["split"] == "train"]

plt.scatter(train["x0"], train["x1"], c=train["y"], cmap="coolwarm")
plt.colorbar(label="y")
plt.xlabel("x0")
plt.ylabel("x1")
plt.title("Training samples")
plt.grid(True, alpha=0.3)
plt.show()
```

## Én scatter per klasse, med forklaring
Vil du ha en legend med klassenavn i stedet for en fargeskala, tegner du én scatter per klasse med en boolsk maske. Her med numpy-arrays fra `make_moons`, og med punktstørrelse `s`, `marker` og `alpha`:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=0.2, random_state=0)

for klasse, farge, markor in [(0, "tab:blue", "o"), (1, "tab:red", "x")]:
    maske = y == klasse
    plt.scatter(X[maske, 0], X[maske, 1], c=farge, marker=markor, s=25, alpha=0.7, label=f"Klasse {klasse}")
plt.legend()
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
```

::: advarsel c=… mot color=…
`c=train["y"]` med `cmap` farger etter verdi. `c="blue"` (eller `color="blue"`) gir alle punktene samme farge. Glemmer du `cmap`, brukes standardkartet `viridis` (lilla til gult), som også fungerer, men gir andre farger enn prosjektene.
:::

::: oppgave Plott testdataene
1. Les `classification.csv` og lagre radene der `split` er `"test"` i `test`.
2. Lag et scatterplott i `fig` av `x0` mot `x1` for testdataene, farget etter `y` med `cmap="coolwarm"`, og legg til `plt.colorbar()`.
```start
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("classification.csv")
test = ...

fig = plt.figure(figsize=(6, 4))
# scatterplott her

plt.show()
```
```losning
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("classification.csv")
test = data[data["split"] == "test"]

fig = plt.figure(figsize=(6, 4))
plt.scatter(test["x0"], test["x1"], c=test["y"], cmap="coolwarm")
plt.colorbar()
plt.show()
```
```test
import numpy as np
assert hasattr(test, "shape") and len(test) == 500, "test skal være de 500 radene der data['split'] == 'test'."
assert set(test["split"]) == {"test"}, "test skal bare inneholde rader med split == 'test'."
assert len(fig.axes) >= 1 and len(fig.axes[0].collections) == 1, "Fant ingen scatter i fig. Bruk plt.scatter(...) etter fig = plt.figure(...)."
pts = fig.axes[0].collections[0]
assert np.allclose(pts.get_offsets(), test[["x0", "x1"]].to_numpy()), "Punktene skal være test['x0'] (x) mot test['x1'] (y)."
farger = pts.get_array()
assert farger is not None and np.allclose(farger, test["y"]), "Punktene skal farges etter y: c=test['y']."
assert pts.get_cmap().name == "coolwarm", "Bruk cmap='coolwarm'."
assert len(fig.axes) == 2, "Husk plt.colorbar() (den lager en ekstra akse ved siden av plottet)."
```
:::

::: quiz
Hva gjør `c=train["y"], cmap="coolwarm"` i `plt.scatter(train["x0"], train["x1"], c=train["y"], cmap="coolwarm")`?
- [ ] Tegner `y` som en tredje akse
- [x] Farger hvert punkt etter labelen sin, med farger fra fargekartet coolwarm
- [ ] Bruker `y` som størrelse på punktene
- [ ] Tegner bare punktene der `y` er 1
? `c` gir fargeverdien per punkt, og `cmap` bestemmer hvilke farger verdiene oversettes til. Størrelse styres av `s`.
:::

::: oppsummering
- `plt.scatter(x, y)` tegner ett punkt per datapunkt.
- `c=labels, cmap="coolwarm"` farger etter klasse eller verdi. `plt.colorbar()` viser skalaen.
- `s` (størrelse), `marker`, `alpha` styrer utseendet.
- Én scatter per klasse med maske (`X[y == k, 0]`) og `label` gir en legend med klassenavn.
:::

--- LEKSJON 4.3 | hist-bar | Histogram og stolpediagram
Et **histogram** viser hvordan verdiene til én variabel er fordelt: tallinjen deles i intervaller (bins), og høyden er antall verdier i hvert. Et **stolpediagram** (bar chart) viser én verdi per kategori, for eksempel antall per klasse eller viktigheten til hver feature.

## plt.hist

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
hoyder = rng.normal(loc=175, scale=8, size=1000)

plt.hist(hoyder, bins=30)
plt.xlabel("Høyde (cm)")
plt.ylabel("Antall")
plt.show()
```

`bins` bestemmer antall intervaller. For få bins skjuler detaljer, for mange gir et hakkete plott. Prøv å endre tallet.

## Flere histogrammer oppå hverandre
For å sammenligne fordelinger (for eksempel én feature for to klasser) tegner du flere histogrammer i samme figur med `alpha=0.5`, så de skinner gjennom hverandre, og `label` for legend.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target          # 0 = ondartet, 1 = godartet
radius = X[:, 0]                       # feature 0: "mean radius"

plt.hist(radius[y == 0], bins=30, alpha=0.5, label="Ondartet")
plt.hist(radius[y == 1], bins=30, alpha=0.5, label="Godartet")
plt.xlabel(data.feature_names[0])
plt.ylabel("Antall")
plt.legend()
plt.title("En feature som skiller klassene godt")
plt.show()
```

::: prosjekt Prosjekt 3
Prosjekt 3 sammenligner to features slik: `plt.hist(features.iloc[:, 2], bins=50, alpha=0.5, label="Feature 2")` og tilsvarende for feature 13021, før `plt.legend()`.
:::

## plt.bar: én stolpe per kategori
`plt.bar(navn, verdier)` tegner én stolpe per element. Det er standard for å vise koeffisienter eller **feature importance**. Lange navn blir lesbare med `plt.xticks(rotation=45)`, og `plt.axhline(0)` tegner en horisontal linje ved 0 så positive og negative stolper skilles tydelig.

Under regner vi ut korrelasjonen mellom hver feature og målvariabelen i diabetes-datasettet, sorterer med `np.argsort` og plotter:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

data = load_diabetes()
X, y = data.data, data.target
navn = np.array(data.feature_names)

korrelasjon = np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])])
rekkefolge = np.argsort(korrelasjon)

plt.figure(figsize=(7, 4))
plt.bar(navn[rekkefolge], korrelasjon[rekkefolge], color="tab:green")
plt.axhline(0, color="black", lw=0.8)
plt.xticks(rotation=45)
plt.ylabel("Korrelasjon med y")
plt.title("Hvilke features henger sammen med y?")
plt.show()
```

Med mange features eller lange navn er `plt.barh` (liggende stolper) ofte enda mer lesbart:

```python
import matplotlib.pyplot as plt

features = ["sex_Female", "age", "armor_None", "prior_voyages", "provisions_kg"]
importance = [0.42, 0.21, 0.17, 0.12, 0.08]

plt.barh(features, importance)
plt.xlabel("Feature importance")
plt.gca().invert_yaxis()     # viktigste øverst
plt.show()
```

::: tips hist eller bar?
`plt.hist` tar **rådata** og teller selv. `plt.bar` tar **ferdige høyder**. Har du allerede tellingene (fra `np.bincount` eller `value_counts()`), bruker du `plt.bar`.
:::

::: oppgave Klassefordeling i digits
`y` inneholder sifferet (0–9) for hvert bilde i digits-datasettet.
1. Tell hvor mange bilder det er av hvert siffer med `np.bincount`, og lagre i `antall`.
2. Tegn et stolpediagram i `fig` med sifrene 0–9 på x-aksen og `antall` som høyde.
```start
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

y = load_digits().target
antall = ...

fig = plt.figure(figsize=(6, 3))
# stolpediagram her

plt.show()
```
```losning
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

y = load_digits().target
antall = np.bincount(y)

fig = plt.figure(figsize=(6, 3))
plt.bar(np.arange(10), antall)
plt.xlabel("Siffer")
plt.ylabel("Antall bilder")
plt.show()
```
```test
import numpy as np
assert isinstance(antall, np.ndarray) and len(antall) == 10, "antall skal ha én telling per siffer (10 tall). Bruk np.bincount(y)."
assert antall.sum() == len(y), "Tellingene skal summere til antall bilder."
assert len(fig.axes) >= 1, "Fant ingen plott i fig. Bruk plt.bar(...) etter fig = plt.figure(...)."
stolper = fig.axes[0].patches
assert len(stolper) == 10, f"Det skal være 10 stolper (en per siffer), men det er {len(stolper)}. Brukte du plt.hist i stedet for plt.bar?"
assert np.allclose([s.get_height() for s in stolper], antall), "Høyden på stolpene skal være antall."
```
:::

::: quiz
Du vil vise hvordan alderen til 1000 passasjerer er fordelt. Hvilken linje passer best?
- [x] `plt.hist(alder, bins=20)`
- [ ] `plt.bar(alder, bins=20)`
- [ ] `plt.plot(alder)`
- [ ] `plt.scatter(alder, alder)`
? Et histogram teller hvor mange verdier som havner i hvert intervall, og viser dermed fordelingen. `plt.bar` trenger ferdige høyder (og har ikke `bins`), og `plt.plot(alder)` viser bare verdiene i rekkefølgen de står i.
:::

::: oppsummering
- `plt.hist(x, bins=30)` viser fordelingen til én variabel.
- Flere histogrammer i samme figur: `alpha=0.5` og `label`, så `plt.legend()`.
- `plt.bar(navn, verdier)` / `plt.barh(...)` viser én verdi per kategori (koeffisienter, feature importance, tellinger).
- `plt.xticks(rotation=45)` for lange navn, `plt.axhline(0)` for en nullinje.
:::

--- LEKSJON 4.4 | subplots | Flere plott: subplots
Ofte vil du ha flere plott ved siden av hverandre, for eksempel tapskurven til venstre og dataene til høyre. `fig, ax = plt.subplots(rader, kolonner)` lager en figur (`fig`) med et rutenett av **akser** (`ax`). Hver akse er ett plott, og du tegner i den med metoder som `ax.plot(...)` og `ax.scatter(...)`.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))   # 1 rad, 2 kolonner
ax[0].plot(x, x ** 2)
ax[0].set_title("Kvadrat")
ax[0].set_xlabel("x")
ax[1].plot(x, np.tanh(x), color="tab:orange")
ax[1].set(title="tanh", xlabel="x", ylabel="tanh(x)")   # flere på én gang
plt.tight_layout()                                     # unngå at tekst overlapper
plt.show()
```

::: husk plt-funksjoner mot ax-metoder
Med `plt.` tegner du i den «aktive» aksen. Med `ax.` velger du selv hvilken akse. Navnene er nesten like, men innstillingene får `set_` foran:

| `plt.` | `ax.` |
|---|---|
| `plt.plot`, `plt.scatter`, `plt.hist`, `plt.bar` | `ax.plot`, `ax.scatter`, `ax.hist`, `ax.bar` |
| `plt.title("...")` | `ax.set_title("...")` |
| `plt.xlabel("...")`, `plt.ylabel("...")` | `ax.set_xlabel("...")`, `ax.set_ylabel("...")` |
| `plt.xlim(...)`, `plt.ylim(...)` | `ax.set_xlim(...)`, `ax.set_ylim(...)` |
| | `ax.set(title=..., xlabel=..., ylabel=..., ylim=...)` |
:::

`plt.subplots()` uten argumenter gir én enkelt akse. Det er en vanlig måte å starte et plott på, også når du bare skal ha ett.

## Vertikale og horisontale hjelpelinjer
`ax.axvline(x)` tegner en loddrett linje over hele plottet, og `ax.axhline(y)` en vannrett. Prosjekt 3 markerer epoken der early stopping lagret modellen, og støynivået i dataene, på denne måten:

```python
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(100)
train = 0.9 * np.exp(-epochs / 15) + 0.20
val = 0.9 * np.exp(-epochs / 15) + 0.25 + 0.00004 * np.maximum(0, epochs - 30) ** 2
best_epoch = np.argmin(val)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(train, label="train")
ax.plot(val, label="held-out")
ax.axvline(best_epoch, color="k", ls="--", label="early stopping checkpoint")
ax.axhline(0.2, color="gray", ls=":", label="noise level")
ax.set(title="Classification (BCE loss)", xlabel="epoch", ylabel="loss")
ax.legend()
plt.show()
```

## Rutenett av plott og axes.flat
Med flere rader og kolonner er `ax` en 2D-array av akser, og du indekserer med `axes[rad, kolonne]`. Vil du bare gå gjennom alle aksene i rekkefølge, bruk `axes.flat` (ofte sammen med `zip`). Dette eksempelet gjenskaper figuren om dimensjonalitetens forbannelse fra forelesningsnotatene: avstanden mellom tilfeldige punkter i enhetskuben, for økende dimensjon.

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
dims = [2, 3, 10, 50, 100, 500]

fig, axes = plt.subplots(2, 3, figsize=(10, 5))
print(axes.shape)
for ax, d in zip(axes.flat, dims):
    P1 = rng.uniform(size=(500, d))
    P2 = rng.uniform(size=(500, d))
    dists = np.linalg.norm(P1 - P2, axis=1)
    ax.hist(dists, bins=20)
    ax.set_title(f"d = {d}")
    ax.set_xlim(left=0)
fig.suptitle("Avstand mellom to tilfeldige punkter")
plt.tight_layout()
plt.show()
```

Avstandene blir både større og mer like hverandre når dimensjonen øker. Det er dimensjonalitetens forbannelse (del 10).

::: prosjekt Prosjekt 3
Prosjekt 3 viser 16 bilder med `fig, axes = plt.subplots(2, 8, figsize=(16, 4.5))` og `for ax, i in zip(axes.flat, ...)`, og to plott ved siden av hverandre med `fig, ax = plt.subplots(1, 2, figsize=(12, 4))`, der `ax[0]` og `ax[1]` fylles hver for seg.
:::

::: oppgave Én feature per plott
Lag `fig, axes = plt.subplots(1, 3, ...)` og tegn et histogram av hver av de tre første featurene i iris-datasettet, én per akse. Gi hver akse featurenavnet som tittel (fra `navn`). Bruk gjerne en løkke over `range(3)` eller `zip(axes.flat, ...)`.
```start
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
navn = iris.feature_names

fig, axes = ...
# ett histogram per akse her

plt.tight_layout()
plt.show()
```
```losning
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
navn = iris.feature_names

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for j, ax in enumerate(axes.flat):
    ax.hist(X[:, j], bins=15)
    ax.set_title(navn[j])

plt.tight_layout()
plt.show()
```
```test
assert hasattr(fig, "axes") and len(fig.axes) == 3, "fig skal ha 3 akser: fig, axes = plt.subplots(1, 3, ...)."
for j, ax in enumerate(fig.axes):
    assert len(ax.patches) > 0, f"Akse {j} er tom. Tegn ax.hist(X[:, {j}]) i hver akse."
    assert ax.get_title() == navn[j], f"Akse {j} skal ha tittelen '{navn[j]}', men har '{ax.get_title()}'. Bruk ax.set_title(...)."
```
:::

::: quiz
Hva skriver koden ut?
```kode
fig, axes = plt.subplots(2, 3)
print(axes.shape)
```
- [ ] `(3, 2)`
- [x] `(2, 3)`
- [ ] `(6,)`
- [ ] `6`
? `plt.subplots(2, 3)` gir 2 rader og 3 kolonner med akser, lagret i et 2D-array med shape `(2, 3)`. `axes[1, 0]` er nederst til venstre. `axes.flat` går gjennom alle seks.
:::

::: quiz
Hvilken linje setter tittelen på det venstre plottet etter `fig, ax = plt.subplots(1, 2)`?
- [ ] `ax[0].title("Tap")`
- [x] `ax[0].set_title("Tap")`
- [ ] `ax.set_title("Tap")`
- [ ] `plt.title(ax[0], "Tap")`
? På akser heter innstillingene `set_...`. `ax[0].title` er ikke en funksjon som tar tekst, og `ax` alene er et array av to akser, ikke én akse.
:::

::: oppsummering
- `fig, ax = plt.subplots(r, k, figsize=(b, h))` gir et rutenett av akser. Tegn med `ax[i].plot(...)` osv.
- Med én rad eller kolonne er `ax` 1D (`ax[0]`), med flere er den 2D (`axes[rad, kol]`).
- Innstillinger: `ax.set_title`, `ax.set_xlabel`, … eller alt på én gang med `ax.set(...)`.
- `axes.flat` går gjennom alle aksene. `plt.tight_layout()` rydder opp i avstandene.
- `ax.axvline(x)` / `ax.axhline(y)` tegner hjelpelinjer.
:::

--- LEKSJON 4.5 | imshow | imshow: bilder og heatmaps
`plt.imshow(A)` tegner et 2D-array som et bilde: hvert element blir en rute (piksel), og verdien bestemmer fargen. Det brukes både til ekte bilder og til **heatmaps** av matriser, for eksempel Q-verdiene til en agent.

## Bilder som rader i X
I maskinlæring lagres hvert bilde som **én rad** med pikselverdier, så det passer i `X`. Digits-datasettet har bilder på 8×8 piksler, altså 64 features per rad. For å vise ett bilde gjør du raden om til 8×8 med `reshape`:

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target
print(X.shape)                # 1797 bilder, 64 piksler hver

bilde = X[0].reshape(8, 8)
print("Bildet som array (før imshow):")
display(bilde)

plt.figure(figsize=(3, 3))
plt.imshow(bilde, cmap="gray")
plt.title(f"Label: {y[0]}")
plt.axis("off")               # skjul akser og tall
plt.show()
```

Allerede i `display(bilde)` ser du sifferet: rutene med høye verdier (0 til 16) danner en ring, altså en 0. `imshow` tegner nøyaktig de samme tallene, bare uten tekst. `cmap="gray"` gir gråtoner. Uten den brukes fargekartet `viridis`. `plt.axis("off")` fjerner aksene, som ikke betyr noe for et bilde.

Mange bilder vises best i et rutenett med `subplots` og `axes.flat`:

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

fig, axes = plt.subplots(2, 5, figsize=(8, 3.5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X[i].reshape(8, 8), cmap="gray_r")   # _r snur kartet: svart på hvitt
    ax.set_title(f"y = {y[i]}")
    ax.axis("off")
plt.tight_layout()
plt.show()
```

::: prosjekt Prosjekt 3
Prosjekt 3 gjør det samme med 128×128-bilder: `row.reshape(128, 128).T`, så normaliseres verdiene til [0, 1] med `(image - image.min()) / (image.max() - image.min())` før `plt.imshow(image, cmap="gray")` og `plt.axis("off")`. MNIST-bilder (28×28) vises med `reshape(28, 28)`.
:::

## Heatmap av en matrise
Et heatmap er det samme som et bilde, men med et fargekart som gjør verdiene lette å lese. `cmap="RdYlGn"` går fra rødt (lavt) via gult til grønt (høyt). Med `ax.text` kan du skrive tallet i hver rute.

Her er Q-tabellen til en agent i et 4×4-rutenett, med 16 tilstander og 4 handlinger. `q_table.max(axis=1)` gir beste verdi i hver tilstand (leksjon 2.6), og `reshape(4, 4)` legger den ut som kartet:

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
avstand = np.add.outer(np.arange(4)[::-1], np.arange(4)[::-1])   # avstand til målet nede til høyre
q_table = 10 * 0.8 ** avstand.reshape(16, 1) + rng.uniform(-0.5, 0.5, size=(16, 4))

q_grid = q_table.max(axis=1).reshape(4, 4)
vann = np.zeros((4, 4), dtype=bool)
vann[1, 1] = vann[2, 3] = True
q_grid = np.where(vann, np.nan, q_grid)     # nan-ruter blir blanke
display(q_grid)                             # tallene som skal tegnes

fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(q_grid, cmap="RdYlGn")
fig.colorbar(im, ax=ax)
for r in range(4):
    for c in range(4):
        if not np.isnan(q_grid[r, c]):
            ax.text(c, r, f"{q_grid[r, c]:.1f}", ha="center", va="center")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Q-Value Heatmap (max over actions)")
plt.show()
```

Sammenlign `display(q_grid)` med heatmapen: det er de samme 16 tallene på de samme plassene, og de to `nan`-rutene (vann) blir blanke i plottet.

::: advarsel Rad er y, kolonne er x
I `imshow` er rad 0 **øverst**, og ruten `A[r, c]` står på koordinatene `x = c`, `y = r`. Derfor skrives teksten med `ax.text(c, r, ...)`, med kolonnen først.
:::

::: prosjekt Prosjekt 4
`show_q_heatmap` i prosjekt 4 er bygget nøyaktig slik: `q_table.max(axis=1).reshape(nrows, ncols)`, `np.where(blank, np.nan, q_grid)`, `ax.imshow(q_grid, cmap='RdYlGn', interpolation='nearest')` og `ax.text(c, r, f'{q_grid[r, c]:.1f}', ha='center', va='center')`.
:::

::: oppgave Vis et siffer
1. Hent bilde nummer 42 fra `X` og gjør det om til et 8×8-array `bilde`. Lagre labelen i `label`.
2. Vis bildet i `ax` med `cmap="gray"`, sett tittelen til `f"Label: {label}"` og skru av aksene.
```start
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

bilde = ...
label = ...

fig, ax = plt.subplots(figsize=(3, 3))
# vis bildet her

plt.show()
```
```losning
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

bilde = X[42].reshape(8, 8)
label = y[42]

fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(bilde, cmap="gray")
ax.set_title(f"Label: {label}")
ax.axis("off")

plt.show()
```
```test
import numpy as np
assert hasattr(bilde, "shape") and bilde.shape == (8, 8), "bilde skal ha shape (8, 8). Bruk X[42].reshape(8, 8)."
assert np.allclose(bilde, digits.images[42]), "bilde skal være rad 42 i X."
assert label == y[42], "label skal være y[42]."
assert len(ax.images) == 1, "Vis bildet med ax.imshow(bilde, cmap='gray')."
assert np.allclose(ax.images[0].get_array(), bilde), "Det er ikke bilde som vises."
assert ax.images[0].get_cmap().name == "gray", "Bruk cmap='gray'."
assert ax.get_title() == f"Label: {label}", f"Tittelen skal være 'Label: {label}'."
assert not ax.axison, "Skru av aksene med ax.axis('off')."
```
:::

::: quiz
Hver rad i MNIST-datasettet har 784 pikselverdier. Hvilken linje viser bilde nummer `i` riktig?
- [ ] `plt.imshow(X[i])`
- [x] `plt.imshow(X[i].reshape(28, 28), cmap="gray")`
- [ ] `plt.imshow(X[:, i].reshape(28, 28), cmap="gray")`
- [ ] `plt.imshow(X[i].reshape(784, 1), cmap="gray")`
? Bildet er én rad (`X[i]`) som må gjøres om til 28×28 (28 · 28 = 784). `X[:, i]` er én piksel i alle bildene, og `imshow` krever 2D, så et 1D-array gir feilmelding. `(784, 1)` er 2D, men blir en tynn stripe i stedet for et bilde.
:::

::: oppsummering
- `plt.imshow(A, cmap="gray")` tegner et 2D-array som et bilde. `plt.axis("off")` skjuler aksene.
- Bilder lagres som rader. Vis med `X[i].reshape(8, 8)` (digits) eller `reshape(28, 28)` (MNIST).
- Heatmap: `im = ax.imshow(M, cmap="RdYlGn")`, `fig.colorbar(im, ax=ax)`, `ax.text(c, r, ...)` for tall i rutene.
- `np.nan`-ruter blir blanke. Rad 0 er øverst, og `A[r, c]` står på `x = c`, `y = r`.
:::

--- LEKSJON 4.6 | beslutningsgrense | Beslutningsgrenser med meshgrid og contourf
En klassifiseringsmodell deler planet i områder: her predikerer den klasse 0, der klasse 1. Grensen mellom områdene kalles **beslutningsgrensen**. For å se den lager vi et tett rutenett av punkter, lar modellen predikere på **alle** punktene, og farger planet etter svaret. Alt du trenger, har du allerede lært: `np.meshgrid` og `ravel` (del 2), en DataFrame (del 3) og `reshape`.

## Steg for steg på et lite rutenett

```python
import numpy as np
import pandas as pd

xx0, xx1 = np.meshgrid(np.arange(0, 3, 1.0),     # x0-verdier: 0, 1, 2
                       np.arange(0, 2, 1.0))     # x1-verdier: 0, 1
print(xx0.shape)                                 # (2, 3): (antall x1, antall x0)
display(xx0)
display(xx1)

grid = pd.DataFrame({"x0": xx0.ravel(), "x1": xx1.ravel()})   # ett punkt per rad
print(grid)

Z = (grid["x0"] + grid["x1"] > 1.5).astype(int).to_numpy()   # en «modell» som bare er en regel
print("Prediksjonene, én per rad i grid:")
display(Z)
Z = Z.reshape(xx0.shape)                                       # tilbake til rutenettets form
print("Etter reshape: samme form som rutenettet")
display(Z)
```

`display(xx0)` og `display(xx1)` viser rutenettet: rute `[i, j]` er punktet med x0 = `xx0[i, j]` og x1 = `xx1[i, j]`. Klikk på `grid` i variabelpanelet for å se de samme seks punktene som en tabell.

1. `np.meshgrid` gir to matriser med x0- og x1-koordinatene til alle punktene i rutenettet.
2. `ravel()` gjør dem om til lange kolonner, og `pd.DataFrame` setter dem sammen til en tabell med samme kolonnenavn som treningsdataene.
3. Modellen predikerer én klasse per rad.
4. `reshape(xx0.shape)` legger prediksjonene tilbake i rutenettets form, så de kan tegnes.

## Hele koden fra prosjekt 2
Nå med et ekte datasett og et beslutningstre. `plt.contourf(xx0, xx1, Z)` fyller områdene med farge etter `Z`, og `alpha=0.4` gjør dem svake, så treningspunktene oppå vises tydelig. (Selve modellen, `DecisionTreeClassifier`, forklares i del 8. Her trenger du bare å vite at `fit` trener den og `predict` gir en klasse per rad.)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("problem1.csv")
train = data[data["split"] == "train"]
X_train = train[["x0", "x1"]]
y_train = train["y"]

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

# visualize the decision boundary
x0_min, x0_max = train["x0"].min() - 1, train["x0"].max() + 1
x1_min, x1_max = train["x1"].min() - 1, train["x1"].max() + 1
xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, 0.05),
                       np.arange(x1_min, x1_max, 0.05))

grid = pd.DataFrame({"x0": xx0.ravel(), "x1": xx1.ravel()})
Z = clf.predict(grid)
Z = Z.reshape(xx0.shape)

plt.figure(figsize=(6, 4))
plt.contourf(xx0, xx1, Z, alpha=0.4, cmap="coolwarm")
plt.scatter(X_train["x0"], X_train["x1"], c=y_train, cmap="coolwarm", s=10)
plt.grid(True)
plt.xlabel("x0", fontsize=14)
plt.ylabel("x1", fontsize=14)
plt.title("Decision Tree predictions", fontsize=16)
plt.colorbar()
plt.show()
```

::: prosjekt Prosjekt 2
Koden er hentet fra prosjekt 2, men prosjektet bruker steg `0.01`. Det gir et rutenett som er 25 ganger så stort (over én million punkter), som tar merkbart lengre tid. Til utforsking holder `0.05` eller `0.1`. Legg merke til de skarpe, rektangulære områdene: et beslutningstre deler alltid langs aksene.
:::

::: hvorfor Hvorfor en DataFrame med kolonnenavn?
Modellen ble trent på en DataFrame med kolonnene `x0` og `x1`. Får `predict` data med andre (eller ingen) kolonnenavn, gir sklearn en advarsel. Derfor bygges `grid` med nøyaktig de samme navnene. Ble modellen trent på et numpy-array, kan du bruke `np.column_stack((xx0.ravel(), xx1.ravel()))` i stedet, slik `plot_decision_boundary` i prosjekt 3 gjør.
:::

## Sannsynligheter i stedet for klasser
Mange modeller kan også gi sannsynligheten for klasse 1 med `predict_proba(grid)[:, 1]`. Tegner du den med `contourf`, ser du hvor sikker modellen er. Her en logistisk regresjon (del 5 og 7) på dataene fra prosjekt 1: grensen blir en rett linje, og modellen er usikker nesten overalt, fordi dataene ikke kan skilles med en linje.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]

clf = LogisticRegression().fit(train[["x0", "x1"]], train["y"])

xx0, xx1 = np.meshgrid(np.arange(-4, 4, 0.05), np.arange(-4, 4, 0.05))
grid = pd.DataFrame({"x0": xx0.ravel(), "x1": xx1.ravel()})
P = clf.predict_proba(grid)[:, 1].reshape(xx0.shape)

fig, ax = plt.subplots(figsize=(6, 4))
cs = ax.contourf(xx0, xx1, P, levels=20, cmap="coolwarm", alpha=0.6)
fig.colorbar(cs, ax=ax, label="P(y = 1)")
ax.contour(xx0, xx1, P, levels=[0.5], colors="k")       # selve grensen, der P = 0.5
ax.scatter(train["x0"], train["x1"], c=train["y"], cmap="coolwarm", s=10)
ax.set(xlabel="x0", ylabel="x1", title="Logistisk regresjon: en rett grense")
plt.show()
```

::: oppgave Tegn beslutningsgrensen
`clf` er et beslutningstre trent på `train` fra prosjekt 1. Lag
1. `xx0, xx1` med `np.meshgrid` og `np.arange` fra -4 til 4 med steg 0.1 i begge retninger.
2. `grid`: en DataFrame med kolonnene `"x0"` og `"x1"` fra de utflatede rutenettene.
3. `Z`: prediksjonene til `clf` på `grid`, med samme shape som `xx0`.
4. Tegn `Z` i `fig` med `plt.contourf(..., alpha=0.4, cmap="coolwarm")`.
```start
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
clf = DecisionTreeClassifier(max_depth=4, random_state=0).fit(train[["x0", "x1"]], train["y"])

xx0, xx1 = ..., ...
grid = ...
Z = ...

fig = plt.figure(figsize=(6, 4))
# contourf her
plt.scatter(train["x0"], train["x1"], c=train["y"], cmap="coolwarm", s=10)
plt.show()
```
```losning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
clf = DecisionTreeClassifier(max_depth=4, random_state=0).fit(train[["x0", "x1"]], train["y"])

xx0, xx1 = np.meshgrid(np.arange(-4, 4, 0.1), np.arange(-4, 4, 0.1))
grid = pd.DataFrame({"x0": xx0.ravel(), "x1": xx1.ravel()})
Z = clf.predict(grid).reshape(xx0.shape)

fig = plt.figure(figsize=(6, 4))
plt.contourf(xx0, xx1, Z, alpha=0.4, cmap="coolwarm")
plt.scatter(train["x0"], train["x1"], c=train["y"], cmap="coolwarm", s=10)
plt.show()
```
```test
import numpy as np
import pandas as pd
assert isinstance(xx0, np.ndarray) and xx0.ndim == 2 and xx0.shape == xx1.shape, "xx0 og xx1 skal være 2D-arrays fra np.meshgrid(...)."
assert np.isclose(xx0.min(), -4) and np.isclose(xx1.min(), -4) and xx0.max() < 4 and xx0.max() > 3.8, "Rutenettet skal gå fra -4 til (nesten) 4: np.arange(-4, 4, 0.1)."
assert isinstance(grid, pd.DataFrame) and list(grid.columns) == ["x0", "x1"], "grid skal være en DataFrame med kolonnene 'x0' og 'x1'."
assert len(grid) == xx0.size, "grid skal ha én rad per punkt i rutenettet: bruk xx0.ravel() og xx1.ravel()."
assert isinstance(Z, np.ndarray) and Z.shape == xx0.shape, f"Z skal ha samme shape som xx0 {xx0.shape}. Husk .reshape(xx0.shape)."
fasit = clf.predict(pd.DataFrame({"x0": xx0.ravel(), "x1": xx1.ravel()})).reshape(xx0.shape)
assert np.array_equal(Z, fasit), "Z stemmer ikke med clf.predict(grid). Sjekk at x0 og x1 ikke er byttet om."
assert len(fig.axes) >= 1 and len(fig.axes[0].collections) >= 2, "Fant ikke contourf-plottet i fig. Bruk plt.contourf(xx0, xx1, Z, alpha=0.4, cmap='coolwarm')."
```
:::

::: quiz
Hva er shapen til `xx0` etter koden?
```kode
xx0, xx1 = np.meshgrid(np.arange(0, 3), np.arange(0, 5))
```
- [ ] `(3, 5)`
- [x] `(5, 3)`
- [ ] `(15,)`
- [ ] `(3,)`
? `meshgrid(xs, ys)` gir matriser med én **rad** per y-verdi og én **kolonne** per x-verdi: `(len(ys), len(xs))` = `(5, 3)`. Det passer med `imshow`/`contourf`, der rader er y-retningen.
:::

::: quiz
Hvorfor står det `Z = Z.reshape(xx0.shape)` etter `Z = clf.predict(grid)`?
- [ ] For å gjøre om sannsynligheter til klasser
- [x] `predict` gir ett 1D-array med én prediksjon per rad i `grid`, og `contourf` trenger dem ordnet i samme 2D-rutenett som `xx0` og `xx1`
- [ ] For å sortere prediksjonene
- [ ] Fordi modellen ble trent på 2D-data og må ha 2D-utdata
? `grid` ble laget med `ravel()`, så prediksjonene kommer som en lang rekke. `reshape(xx0.shape)` reverserer `ravel()` og legger hver prediksjon tilbake på sitt punkt i rutenettet.
:::

::: oppsummering
- Beslutningsgrense: prediker på et tett rutenett og farg etter svaret.
- `xx0, xx1 = np.meshgrid(np.arange(...), np.arange(...))` gir rutenettet. Shape er `(len(ys), len(xs))`.
- `grid = pd.DataFrame({"x0": xx0.ravel(), "x1": xx1.ravel()})`, med samme kolonnenavn som treningsdataene.
- `Z = clf.predict(grid).reshape(xx0.shape)`, så `plt.contourf(xx0, xx1, Z, alpha=0.4, cmap="coolwarm")`.
- `predict_proba(grid)[:, 1]` gir sannsynligheter. `contour(..., levels=[0.5])` tegner selve grensen.
- Mindre steg gir finere grense, men kvadratisk flere punkter: hold det rundt 0.05–0.1 når du utforsker.
:::
