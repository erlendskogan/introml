=== DEL 4 | plotting | Plotting med matplotlib
Her lager du plottene fra prosjektene: linjeplott, scatter, histogram, stolper, subplots, bilder, heatmaps og beslutningsgrenser.

--- LEKSJON 4.1 | linjeplott | Linjeplott
`plt.plot(x, y)` (fra `import matplotlib.pyplot as plt`) tegner en linje gjennom punktene, og `plt.plot(y)` alene bruker indeksene 0, 1, 2, … som x. Startkoden lager figuren med `fig, ax = plt.subplots()`, og `plt.`-kallene tegner i den.

::: oppgave Sigmoid-kurven
Lag `sigmoid = 1 / (1 + np.exp(-z))` og plott den mot `z`.
```start
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-6, 6, 100)
fig, ax = plt.subplots()
# lag sigmoid og plott den her
```
```losning
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-6, 6, 100)
fig, ax = plt.subplots()
sigmoid = 1 / (1 + np.exp(-z))
plt.plot(z, sigmoid)
```
```test
import numpy as np
assert len(ax.lines) == 1, f"Figuren skal ha 1 linje, men har {len(ax.lines)}. Bruk plt.plot(z, sigmoid)."
assert np.allclose(ax.lines[0].get_xdata(), z), "x-verdiene skal være z: plt.plot(z, sigmoid)."
assert np.allclose(ax.lines[0].get_ydata(), 1 / (1 + np.exp(-z))), "y-verdiene skal være 1 / (1 + np.exp(-z))."
```
:::

::: oppgave Bare y-verdier
Plott `losses` med bare ett argument, og lagre x-koordinaten til det siste punktet på linjen i `siste_x`.
```start
import matplotlib.pyplot as plt

losses = [0.9, 0.5, 0.3, 0.25]
fig, ax = plt.subplots()
# plott losses her

siste_x = ...
```
```losning
import matplotlib.pyplot as plt

losses = [0.9, 0.5, 0.3, 0.25]
fig, ax = plt.subplots()
plt.plot(losses)

siste_x = 3
```
```test
import numpy as np
assert len(ax.lines) == 1, "Plott losses med plt.plot(losses)."
assert np.allclose(ax.lines[0].get_ydata(), losses), "y-verdiene skal være losses. Gi bare ett argument: plt.plot(losses)."
assert np.allclose(ax.lines[0].get_xdata(), np.arange(4)), "Gi bare ett argument, så matplotlib velger x selv."
assert siste_x == 3, f"siste_x = {siste_x}. Med bare y bruker matplotlib indeksene 0, 1, 2, 3 som x, så siste punkt har x = 3."
```
:::

## Aksenavn, tittel og forklaring
`plt.xlabel`, `plt.ylabel`, `plt.title` og `plt.grid(True)` setter tekst og rutenett, og `label="..."` i hvert `plt.plot` pluss `plt.legend()` gir forklaring. Størrelsen velges med `plt.subplots(figsize=(bredde, høyde))`.

::: oppgave Tapskurve
Lag `fig, ax = plt.subplots(figsize=(8, 3))`, plott `train_loss` og `val_loss` mot `epochs` med `label="train"` og `label="validation"`, og legg til aksenavn, tittel, rutenett og forklaring.
```start
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 31)
train_loss = 1 / epochs + 0.1
val_loss = 1 / epochs + 0.15 + 0.004 * (epochs - 12) ** 2 / 10

fig, ax = None, None
# lag figuren og plott her
```
```losning
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 31)
train_loss = 1 / epochs + 0.1
val_loss = 1 / epochs + 0.15 + 0.004 * (epochs - 12) ** 2 / 10

fig, ax = plt.subplots(figsize=(8, 3))
plt.plot(epochs, train_loss, label="train")
plt.plot(epochs, val_loss, label="validation")
plt.xlabel("Epoke")
plt.ylabel("Loss")
plt.title("Tapskurve")
plt.grid(True)
plt.legend()
```
```test
import numpy as np
assert fig is not None and ax is not None, "Lag figuren med fig, ax = plt.subplots(figsize=(8, 3))."
assert np.allclose(fig.get_size_inches(), [8, 3]), f"Figuren skal ha figsize=(8, 3), men har {tuple(fig.get_size_inches())}."
assert len(ax.lines) == 2, f"Figuren skal ha 2 linjer, men har {len(ax.lines)}."
labels = sorted(l.get_label() for l in ax.lines)
assert labels == ["train", "validation"], f"Linjene skal ha label 'train' og 'validation', men har {labels}."
for l in ax.lines:
    ventet = train_loss if l.get_label() == "train" else val_loss
    assert np.allclose(l.get_xdata(), epochs) and np.allclose(l.get_ydata(), ventet), f"Linjen '{l.get_label()}' har feil data. Plott mot epochs."
assert ax.get_xlabel() and ax.get_ylabel(), "Husk plt.xlabel(...) og plt.ylabel(...)."
assert ax.get_title(), "Husk plt.title(...)."
assert any(g.get_visible() for g in ax.get_xgridlines()), "Husk plt.grid(True)."
assert ax.get_legend() is not None, "Husk plt.legend() så linjenavnene vises."
```
:::

## Utseende: farge, linjetype og markører
`color`/`c`, `linestyle`/`ls` (`"--"` stiplet, `":"` prikket), `marker` (`"o"`, `"x"`), `linewidth`/`lw` og `alpha` (0 til 1) styrer utseendet, f.eks. `plt.plot(y, c="red", ls="--", marker="o")`.

::: oppgave Stil på linjene
Plott `losses` rød med `marker="o"`, og `accuracies` blå og stiplet.
```start
import matplotlib.pyplot as plt

losses = [0.9, 0.6, 0.45, 0.38, 0.35, 0.33]
accuracies = [0.55, 0.70, 0.78, 0.82, 0.84, 0.85]
fig, ax = plt.subplots()
# to plott her
```
```losning
import matplotlib.pyplot as plt

losses = [0.9, 0.6, 0.45, 0.38, 0.35, 0.33]
accuracies = [0.55, 0.70, 0.78, 0.82, 0.84, 0.85]
fig, ax = plt.subplots()
plt.plot(losses, c="red", marker="o")
plt.plot(accuracies, c="blue", ls="--")
```
```test
import numpy as np
from matplotlib.colors import same_color
assert len(ax.lines) == 2, f"Figuren skal ha 2 linjer, men har {len(ax.lines)}."
l_loss = [l for l in ax.lines if np.allclose(l.get_ydata(), losses)]
l_acc = [l for l in ax.lines if np.allclose(l.get_ydata(), accuracies)]
assert l_loss and l_acc, "Plott både losses og accuracies."
assert same_color(l_loss[0].get_color(), "red"), "losses skal være rød: c='red'."
assert l_loss[0].get_marker() == "o", "losses skal ha marker='o'."
assert same_color(l_acc[0].get_color(), "blue"), "accuracies skal være blå: c='blue'."
assert l_acc[0].get_linestyle() == "--", "accuracies skal være stiplet: ls='--'."
```
:::

::: oppgave Marker beste epoke
Lagre epoken (ikke indeksen) med lavest `val_loss` i `beste_epoke`, plott `val_loss` mot `epochs`, og tegn det beste punktet med `plt.plot(beste_epoke, val_loss.min(), marker="o", color="red")`.
```start
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 31)
val_loss = 1 / epochs + 0.15 + 0.004 * (epochs - 12) ** 2 / 10
fig, ax = plt.subplots()

beste_epoke = ...
# to plott her
```
```losning
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(1, 31)
val_loss = 1 / epochs + 0.15 + 0.004 * (epochs - 12) ** 2 / 10
fig, ax = plt.subplots()

beste_epoke = epochs[np.argmin(val_loss)]
plt.plot(epochs, val_loss)
plt.plot(beste_epoke, val_loss.min(), marker="o", color="red")
```
```test
import numpy as np
assert beste_epoke is not ... and int(beste_epoke) == 17, f"beste_epoke skal være epoken (ikke indeksen) med lavest val_loss. Fikk {beste_epoke}. Tips: epochs[np.argmin(val_loss)]."
assert any(len(l.get_xdata()) == 30 and np.allclose(l.get_ydata(), val_loss) for l in ax.lines), "Plott val_loss mot epochs."
punkt = [l for l in ax.lines if len(np.atleast_1d(l.get_xdata())) == 1]
assert punkt, "Tegn det beste punktet med plt.plot(beste_epoke, val_loss.min(), marker='o', color='red')."
assert np.isclose(np.atleast_1d(punkt[0].get_xdata())[0], 17) and np.isclose(np.atleast_1d(punkt[0].get_ydata())[0], val_loss.min()), "Punktet skal ligge på (beste_epoke, val_loss.min())."
assert punkt[0].get_marker() == "o", "Punktet trenger marker='o', ellers syns det ikke."
```
:::

## Støyete kurver
`np.convolve(y, np.ones(k) / k, mode="valid")` gir glidende snitt over `k` punkter, og `plt.yscale("log")` gir logaritmisk y-akse.

::: oppgave Steg per episode
Lag `snitt`, det glidende snittet over 50 episoder, plott `steps_per_episode` med `alpha=0.4` og `snitt` oppå, og bruk log-skala på y-aksen.
```start
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
steps_per_episode = (300 * np.exp(-np.arange(1000) / 150) + 20) * rng.lognormal(0, 0.5, 1000)
fig, ax = plt.subplots(figsize=(8, 4))

snitt = ...
# plott her
```
```losning
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
steps_per_episode = (300 * np.exp(-np.arange(1000) / 150) + 20) * rng.lognormal(0, 0.5, 1000)
fig, ax = plt.subplots(figsize=(8, 4))

snitt = np.convolve(steps_per_episode, np.ones(50) / 50, mode="valid")
plt.plot(steps_per_episode, alpha=0.4, label="Steps per episode")
plt.plot(snitt, label="Moving average (50 episodes)")
plt.yscale("log")
```
```test
import numpy as np
assert isinstance(snitt, np.ndarray) and len(snitt) == 951, f"snitt skal ha lengde 1000 - 50 + 1 = 951. Bruk np.convolve(..., np.ones(50) / 50, mode='valid')."
assert np.allclose(snitt, np.convolve(steps_per_episode, np.ones(50) / 50, mode="valid")), "snitt har feil verdier. Del np.ones(50) på 50."
raa = [l for l in ax.lines if len(l.get_ydata()) == 1000 and np.allclose(l.get_ydata(), steps_per_episode)]
assert raa, "Plott rådataene: plt.plot(steps_per_episode, alpha=0.4)."
assert raa[0].get_alpha() == 0.4, "Rådataene skal ha alpha=0.4."
assert any(len(l.get_ydata()) == 951 and np.allclose(l.get_ydata(), snitt) for l in ax.lines), "Plott snitt oppå rådataene."
assert ax.get_yscale() == "log", "Bruk plt.yscale('log')."
```
:::

--- LEKSJON 4.2 | scatter | Scatter: punkter med farge etter klasse
`plt.scatter(x, y)` tegner ett punkt per datapunkt uten linjer, og `c="blue"` (eller `color="blue"`) gir alle punktene samme farge. `fontsize=14` i `plt.xlabel` gir større tekst.

::: oppgave Energi mot nettverkstrafikk
Lag et blått scatterplott av `data["Net_Activity"]` (x) mot `data["Energy"]` (y) med `label="Data points"` og forklaring, og bruk `plt.xlabel("Network Activity", fontsize=14)`.
```start
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("regression.csv")
fig, ax = plt.subplots(figsize=(6, 4))
# scatterplott her
```
```losning
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("regression.csv")
fig, ax = plt.subplots(figsize=(6, 4))
plt.scatter(data["Net_Activity"], data["Energy"], c="blue", label="Data points")
plt.xlabel("Network Activity", fontsize=14)
plt.ylabel("Energy", fontsize=14)
plt.legend()
```
```test
import numpy as np
from matplotlib.colors import same_color
assert len(ax.collections) == 1, "Fant ingen scatter. Bruk plt.scatter(...)."
pts = ax.collections[0]
assert np.allclose(pts.get_offsets(), data[["Net_Activity", "Energy"]].to_numpy()), "Punktene skal være Net_Activity (x) mot Energy (y)."
assert same_color(pts.get_facecolor()[0], "blue"), "Punktene skal være blå: c='blue'."
assert pts.get_label() == "Data points", "Gi punktene label='Data points'."
assert ax.get_legend() is not None, "Husk plt.legend()."
assert ax.get_xlabel() == "Network Activity" and ax.xaxis.label.get_fontsize() == 14, "Bruk plt.xlabel('Network Activity', fontsize=14)."
```
:::

## Farge etter klasse: c og cmap
`plt.scatter(x0, x1, c=train["y"], cmap="coolwarm")` farger hvert punkt etter verdien (uten `cmap` brukes `viridis`), og `plt.colorbar()` viser skalaen. Det virker både for klasser og kontinuerlige verdier.

::: oppgave Plott testdataene
Lagre radene der `split` er `"test"` i `test`, og lag et scatterplott av `x0` mot `x1` farget etter `y` med `cmap="coolwarm"` og `plt.colorbar()`.
```start
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("classification.csv")
fig, ax = plt.subplots(figsize=(6, 4))

test = ...
# scatterplott her
```
```losning
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("classification.csv")
fig, ax = plt.subplots(figsize=(6, 4))

test = data[data["split"] == "test"]
plt.scatter(test["x0"], test["x1"], c=test["y"], cmap="coolwarm")
plt.colorbar()
```
```test
import numpy as np
assert hasattr(test, "shape") and len(test) == 500, "test skal være de 500 radene der data['split'] == 'test'."
assert set(test["split"]) == {"test"}, "test skal bare inneholde rader med split == 'test'."
assert len(ax.collections) == 1, "Fant ingen scatter. Bruk plt.scatter(...)."
pts = ax.collections[0]
assert np.allclose(pts.get_offsets(), test[["x0", "x1"]].to_numpy()), "Punktene skal være test['x0'] (x) mot test['x1'] (y)."
farger = pts.get_array()
assert farger is not None and np.allclose(farger, test["y"]), "Punktene skal farges etter y: c=test['y']."
assert pts.get_cmap().name == "coolwarm", "Bruk cmap='coolwarm'."
assert len(fig.axes) == 2, "Husk plt.colorbar() (den lager en ekstra akse ved siden av plottet)."
```
:::

::: oppgave Farge etter verdi
Lag et scatterplott av `x0` mot `x1` for `train`, farget etter den kontinuerlige `y` med `cmap="coolwarm"`, og en fargeskala med `plt.colorbar(label="y")`.
```start
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("regression_data.csv")
train = data[data["split"] == "train"]
fig, ax = plt.subplots()
# scatterplott og colorbar her
```
```losning
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("regression_data.csv")
train = data[data["split"] == "train"]
fig, ax = plt.subplots()
plt.scatter(train["x0"], train["x1"], c=train["y"], cmap="coolwarm")
plt.colorbar(label="y")
```
```test
import numpy as np
assert len(ax.collections) == 1, "Fant ingen scatter. Bruk plt.scatter(...)."
pts = ax.collections[0]
assert np.allclose(pts.get_offsets(), train[["x0", "x1"]].to_numpy()), "Punktene skal være train['x0'] (x) mot train['x1'] (y)."
assert pts.get_array() is not None and np.allclose(pts.get_array(), train["y"]), "Farg etter verdien: c=train['y']."
assert pts.get_cmap().name == "coolwarm", "Bruk cmap='coolwarm'."
assert len(fig.axes) == 2, "Husk plt.colorbar(label='y')."
assert fig.axes[1].get_ylabel() == "y", "Fargeskalaen skal ha teksten 'y': plt.colorbar(label='y')."
```
:::

## Én scatter per klasse
`X[y == k, 0]` velger feature 0 for klasse `k`, så én `plt.scatter` per klasse med `label` gir forklaring med klassenavn. `s` (størrelse), `marker` og `alpha` styrer utseendet.

::: oppgave Klassene i make_moons
Tegn én scatter per klasse (0 og 1) med `label=f"Klasse {k}"`, `s=25` og `alpha=0.7`, gjerne med ulik `marker`, og legg til `plt.legend()`.
```start
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=0.2, random_state=0)
fig, ax = plt.subplots()
# én scatter per klasse her
```
```losning
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=0.2, random_state=0)
fig, ax = plt.subplots()
for k, markor in [(0, "o"), (1, "x")]:
    maske = y == k
    plt.scatter(X[maske, 0], X[maske, 1], marker=markor, s=25, alpha=0.7, label=f"Klasse {k}")
plt.legend()
```
```test
import numpy as np
assert len(ax.collections) == 2, f"Det skal være 2 scatter-kall (ett per klasse), men det er {len(ax.collections)}."
labels = sorted(c.get_label() for c in ax.collections)
assert labels == ["Klasse 0", "Klasse 1"], f"Labelene skal være 'Klasse 0' og 'Klasse 1', men er {labels}."
for c in ax.collections:
    k = int(c.get_label()[-1])
    assert np.allclose(c.get_offsets(), X[y == k]), f"Punktene i '{c.get_label()}' skal være X[y == {k}, 0] mot X[y == {k}, 1]."
    assert np.allclose(c.get_sizes(), 25), "Bruk s=25."
    assert c.get_alpha() == 0.7, "Bruk alpha=0.7."
assert ax.get_legend() is not None, "Husk plt.legend()."
```
:::

--- LEKSJON 4.3 | hist-bar | Histogram og stolpediagram
`plt.hist(x, bins=30)` deler rådataene i intervaller og teller selv, mens `plt.bar(navn, høyder)` tegner ferdige høyder, én stolpe per kategori.

## plt.hist
Flere histogrammer i samme figur blir lesbare med `alpha=0.5` og `label`, pluss `plt.legend()`.

::: oppgave Histogram av høyder
Tegn et histogram av `hoyder` med 25 bins og x-aksenavnet `"Høyde (cm)"`.
```start
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
hoyder = rng.normal(loc=175, scale=8, size=1000)
fig, ax = plt.subplots()
# histogram her
```
```losning
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
hoyder = rng.normal(loc=175, scale=8, size=1000)
fig, ax = plt.subplots()
plt.hist(hoyder, bins=25)
plt.xlabel("Høyde (cm)")
```
```test
import numpy as np
assert len(ax.patches) == 25, f"Histogrammet skal ha 25 stolper, men har {len(ax.patches)}. Bruk bins=25."
assert np.isclose(sum(p.get_height() for p in ax.patches), 1000), "Stolpene skal til sammen telle alle 1000 høydene."
assert ax.get_xlabel() == "Høyde (cm)", "Sett plt.xlabel('Høyde (cm)')."
```
:::

::: oppgave To fordelinger oppå hverandre
Tegn histogrammet av `radius` for klasse 0 (`label="Ondartet"`) og klasse 1 (`label="Godartet"`) i samme figur, begge med `bins=30` og `alpha=0.5`, og legg til forklaring.
```start
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target   # 0 = ondartet, 1 = godartet
radius = X[:, 0]
fig, ax = plt.subplots()
# to histogrammer her
```
```losning
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target   # 0 = ondartet, 1 = godartet
radius = X[:, 0]
fig, ax = plt.subplots()
plt.hist(radius[y == 0], bins=30, alpha=0.5, label="Ondartet")
plt.hist(radius[y == 1], bins=30, alpha=0.5, label="Godartet")
plt.legend()
```
```test
import numpy as np
assert len(ax.patches) == 60, f"Det skal være 2 histogrammer med 30 bins (60 stolper), men det er {len(ax.patches)}."
summer = sorted([sum(p.get_height() for p in ax.patches[:30]), sum(p.get_height() for p in ax.patches[30:])])
assert np.allclose(summer, sorted([np.sum(y == 0), np.sum(y == 1)])), "Hvert histogram skal telle radius for én klasse: radius[y == 0] og radius[y == 1]."
assert all(p.get_alpha() == 0.5 for p in ax.patches), "Bruk alpha=0.5 i begge, så de skinner gjennom hverandre."
leg = ax.get_legend()
assert leg is not None, "Husk plt.legend()."
assert sorted(t.get_text() for t in leg.get_texts()) == ["Godartet", "Ondartet"], "Labelene skal være 'Ondartet' og 'Godartet'."
```
:::

## plt.bar: én stolpe per kategori
`plt.bar(navn, verdier)` tegner stolper, `plt.barh` liggende stolper, `plt.xticks(rotation=45)` roterer lange navn og `plt.axhline(0)` tegner en nullinje. `plt.gca().invert_yaxis()` snur y-aksen.

::: oppgave Klassefordeling i digits
Tell hvor mange bilder det er av hvert siffer med `np.bincount`, lagre i `antall`, og tegn et stolpediagram med sifrene 0–9 på x-aksen.
```start
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

y = load_digits().target
fig, ax = plt.subplots(figsize=(6, 3))

antall = ...
# stolpediagram her
```
```losning
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

y = load_digits().target
fig, ax = plt.subplots(figsize=(6, 3))

antall = np.bincount(y)
plt.bar(np.arange(10), antall)
```
```test
import numpy as np
assert isinstance(antall, np.ndarray) and len(antall) == 10, "antall skal ha én telling per siffer (10 tall). Bruk np.bincount(y)."
assert antall.sum() == len(y), "Tellingene skal summere til antall bilder."
stolper = ax.patches
assert len(stolper) == 10, f"Det skal være 10 stolper (én per siffer), men det er {len(stolper)}. Brukte du plt.hist i stedet for plt.bar?"
assert np.allclose([s.get_height() for s in stolper], antall), "Høyden på stolpene skal være antall."
```
:::

::: oppgave Sortert korrelasjon
Lag `rekkefolge = np.argsort(korrelasjon)` og tegn stolper med `navn[rekkefolge]` og `korrelasjon[rekkefolge]`, med en nullinje (`plt.axhline(0)`) og navnene rotert 45 grader.
```start
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

data = load_diabetes()
X, y = data.data, data.target
navn = np.array(data.feature_names)
korrelasjon = np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])])
fig, ax = plt.subplots(figsize=(7, 4))

rekkefolge = ...
# stolpediagram her
```
```losning
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

data = load_diabetes()
X, y = data.data, data.target
navn = np.array(data.feature_names)
korrelasjon = np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])])
fig, ax = plt.subplots(figsize=(7, 4))

rekkefolge = np.argsort(korrelasjon)
plt.bar(navn[rekkefolge], korrelasjon[rekkefolge])
plt.axhline(0, color="black", lw=0.8)
plt.xticks(rotation=45)
```
```test
import numpy as np
assert len(ax.patches) == 10, f"Det skal være 10 stolper, men det er {len(ax.patches)}."
assert np.allclose([p.get_height() for p in ax.patches], np.sort(korrelasjon)), "Stolpene skal være sortert fra lavest til høyest: korrelasjon[rekkefolge]."
tekster = [t.get_text() for t in ax.get_xticklabels()]
assert tekster == list(navn[np.argsort(korrelasjon)]), "Navnene på x-aksen skal følge samme rekkefølge: navn[rekkefolge]."
assert any(len(l.get_ydata()) == 2 and np.allclose(l.get_ydata(), 0) for l in ax.lines), "Tegn nullinjen med plt.axhline(0)."
assert ax.get_xticklabels()[0].get_rotation() == 45, "Roter navnene med plt.xticks(rotation=45)."
```
:::

::: oppgave Liggende stolper
Tegn `importance` som liggende stolper med `features` som navn, og snu y-aksen så den viktigste står øverst.
```start
import matplotlib.pyplot as plt

features = ["sex_Female", "age", "armor_None", "prior_voyages", "provisions_kg"]
importance = [0.42, 0.21, 0.17, 0.12, 0.08]
fig, ax = plt.subplots()
# liggende stolper her
```
```losning
import matplotlib.pyplot as plt

features = ["sex_Female", "age", "armor_None", "prior_voyages", "provisions_kg"]
importance = [0.42, 0.21, 0.17, 0.12, 0.08]
fig, ax = plt.subplots()
plt.barh(features, importance)
plt.gca().invert_yaxis()
```
```test
import numpy as np
assert len(ax.patches) == 5, f"Det skal være 5 stolper, men det er {len(ax.patches)}. Bruk plt.barh(features, importance)."
assert np.allclose([p.get_width() for p in ax.patches], importance), "Stolpene skal ligge (plt.barh) med lengde importance."
assert ax.yaxis_inverted(), "Snu y-aksen med plt.gca().invert_yaxis(), så den første (viktigste) står øverst."
```
:::

--- LEKSJON 4.4 | subplots | Flere plott: subplots
`fig, ax = plt.subplots(1, 2, figsize=(10, 4))` gir to akser `ax[0]` og `ax[1]`, og du tegner med `ax[0].plot(...)` og setter tekst med `ax[0].set_title(...)`, `ax[0].set_xlabel(...)` eller alt på én gang med `ax[1].set(title=..., xlabel=...)`. `plt.tight_layout()` hindrer at tekst overlapper.

::: oppgave To plott ved siden av hverandre
Lag `fig, ax` med 1 rad og 2 kolonner, tegn `x ** 2` i `ax[0]` med tittelen `"Kvadrat"` og `np.tanh(x)` i `ax[1]` med tittelen `"tanh"`, og gi begge x-aksenavnet `"x"`.
```start
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
fig, ax = None, None
# lag subplots og tegn her
```
```losning
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(x, x ** 2)
ax[0].set_title("Kvadrat")
ax[0].set_xlabel("x")
ax[1].plot(x, np.tanh(x))
ax[1].set(title="tanh", xlabel="x")
plt.tight_layout()
```
```test
import numpy as np
assert fig is not None and len(fig.axes) == 2, "Lag to akser: fig, ax = plt.subplots(1, 2)."
a0, a1 = fig.axes
assert len(a0.lines) == 1 and np.allclose(a0.lines[0].get_ydata(), x ** 2), "ax[0] skal ha én linje med x ** 2."
assert len(a1.lines) == 1 and np.allclose(a1.lines[0].get_ydata(), np.tanh(x)), "ax[1] skal ha én linje med np.tanh(x)."
assert a0.get_title() == "Kvadrat" and a1.get_title() == "tanh", "Titlene skal være 'Kvadrat' og 'tanh'. På akser heter det ax[0].set_title(...)."
assert a0.get_xlabel() == "x" and a1.get_xlabel() == "x", "Begge aksene skal ha xlabel 'x': ax[i].set_xlabel('x') eller ax[i].set(xlabel='x')."
```
:::

## Hjelpelinjer
`ax.axvline(x)` tegner en loddrett linje over hele plottet og `ax.axhline(y)` en vannrett, begge med `color`, `ls` og `label` som vanlig.

::: oppgave Early stopping-linjen
Plott `train` og `val` i `ax` med labels, lagre indeksen med lavest `val` i `best_epoch`, marker den med `ax.axvline(..., ls="--")`, tegn støynivået 0.2 med `ax.axhline`, og legg til `ax.legend()`.
```start
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(100)
train = 0.9 * np.exp(-epochs / 15) + 0.20
val = 0.9 * np.exp(-epochs / 15) + 0.25 + 0.00004 * np.maximum(0, epochs - 30) ** 2
fig, ax = plt.subplots(figsize=(6, 4))

best_epoch = ...
# plott her
```
```losning
import numpy as np
import matplotlib.pyplot as plt

epochs = np.arange(100)
train = 0.9 * np.exp(-epochs / 15) + 0.20
val = 0.9 * np.exp(-epochs / 15) + 0.25 + 0.00004 * np.maximum(0, epochs - 30) ** 2
fig, ax = plt.subplots(figsize=(6, 4))

best_epoch = np.argmin(val)
ax.plot(train, label="train")
ax.plot(val, label="held-out")
ax.axvline(best_epoch, color="k", ls="--", label="early stopping checkpoint")
ax.axhline(0.2, color="gray", ls=":", label="noise level")
ax.legend()
```
```test
import numpy as np
assert best_epoch is not ... and int(best_epoch) == int(np.argmin(val)), f"best_epoch skal være np.argmin(val) = {np.argmin(val)}."
assert any(len(l.get_ydata()) == 100 and np.allclose(l.get_ydata(), train) for l in ax.lines), "Plott train i ax."
assert any(len(l.get_ydata()) == 100 and np.allclose(l.get_ydata(), val) for l in ax.lines), "Plott val i ax."
vert = [l for l in ax.lines if len(l.get_xdata()) == 2 and np.allclose(l.get_xdata(), np.argmin(val))]
assert vert, "Tegn en loddrett linje ved best_epoch med ax.axvline(best_epoch, ...)."
assert vert[0].get_linestyle() == "--", "Den loddrette linjen skal være stiplet: ls='--'."
assert any(len(l.get_ydata()) == 2 and np.allclose(l.get_ydata(), 0.2) for l in ax.lines), "Tegn støynivået med ax.axhline(0.2, ...)."
assert ax.get_legend() is not None, "Husk ax.legend()."
```
:::

## Rutenett av plott og axes.flat
Med flere rader er `axes` et 2D-array (`axes[rad, kol]`, shape `(rader, kolonner)`), og `for ax, d in zip(axes.flat, liste):` går gjennom alle aksene i rekkefølge. `fig.suptitle(...)` gir hele figuren en tittel.

::: oppgave Én feature per plott
Lag `fig, axes = plt.subplots(1, 3, ...)` og tegn et histogram av hver av de tre første featurene i `X`, én per akse, med featurenavnet fra `navn` som tittel.
```start
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
navn = iris.feature_names

fig, axes = None, None
# ett histogram per akse her
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
```
```test
assert fig is not None and len(fig.axes) == 3, "fig skal ha 3 akser: fig, axes = plt.subplots(1, 3, ...)."
for j, a in enumerate(fig.axes):
    assert len(a.patches) > 0, f"Akse {j} er tom. Tegn ax.hist(X[:, {j}]) i hver akse."
    assert a.get_title() == navn[j], f"Akse {j} skal ha tittelen '{navn[j]}', men har '{a.get_title()}'. Bruk ax.set_title(...)."
```
:::

::: oppgave Dimensjonalitetens forbannelse
Lag et 2×3-rutenett, tegn `ax.hist(avstander(d), bins=20)` med tittelen `f"d = {d}"` for hver `d` i `dims`, gi figuren tittelen `"Avstand mellom to tilfeldige punkter"`, og lagre aksen nederst til venstre i `nede_venstre`.
```start
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
dims = [2, 3, 10, 50, 100, 500]

def avstander(d):
    return np.linalg.norm(rng.uniform(size=(500, d)) - rng.uniform(size=(500, d)), axis=1)

fig, axes = None, None
nede_venstre = ...
```
```losning
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
dims = [2, 3, 10, 50, 100, 500]

def avstander(d):
    return np.linalg.norm(rng.uniform(size=(500, d)) - rng.uniform(size=(500, d)), axis=1)

fig, axes = plt.subplots(2, 3, figsize=(10, 5))
for ax, d in zip(axes.flat, dims):
    ax.hist(avstander(d), bins=20)
    ax.set_title(f"d = {d}")
fig.suptitle("Avstand mellom to tilfeldige punkter")
plt.tight_layout()
nede_venstre = axes[1, 0]
```
```test
assert fig is not None and len(fig.axes) == 6, "Lag 6 akser: fig, axes = plt.subplots(2, 3)."
for a, d in zip(fig.axes, dims):
    assert len(a.patches) == 20, f"Aksen for d = {d} skal ha et histogram med bins=20."
    assert a.get_title() == f"d = {d}", f"Aksen for d = {d} skal ha tittelen 'd = {d}', men har '{a.get_title()}'."
sup = getattr(fig, "_suptitle", None)
assert sup is not None and sup.get_text() == "Avstand mellom to tilfeldige punkter", "Gi figuren tittel med fig.suptitle('Avstand mellom to tilfeldige punkter')."
assert nede_venstre is fig.axes[3], "nede_venstre skal være axes[1, 0] (rad 1, kolonne 0)."
```
:::

--- LEKSJON 4.5 | imshow | imshow: bilder og heatmaps
`ax.imshow(A, cmap="gray")` tegner et 2D-array som et bilde der hver verdi blir én rute, og `ax.axis("off")` skjuler aksene. `"gray_r"` snur fargekartet (svart på hvitt).

## Bilder som rader i X
Hvert bilde er én rad i `X`, så `X[i].reshape(8, 8)` gjør en digits-rad om til et bilde (MNIST: `reshape(28, 28)`). Normaliser til [0, 1] med `(A - A.min()) / (A.max() - A.min())`.

::: oppgave Vis et siffer
Gjør bilde nummer 42 i `X` om til et 8×8-array `bilde`, lagre labelen i `label`, og vis bildet i `ax` med `cmap="gray"`, tittelen `f"Label: {label}"` og uten akser.
```start
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target
fig, ax = plt.subplots(figsize=(3, 3))

bilde = ...
label = ...
# vis bildet her
```
```losning
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target
fig, ax = plt.subplots(figsize=(3, 3))

bilde = X[42].reshape(8, 8)
label = y[42]
ax.imshow(bilde, cmap="gray")
ax.set_title(f"Label: {label}")
ax.axis("off")
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

::: oppgave Ti siffer i et rutenett
Lag `fig, axes = plt.subplots(2, 5)` og vis de ti første bildene i `X` med `cmap="gray_r"`, tittelen `f"y = {y[i]}"` og uten akser.
```start
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

fig, axes = None, None
# ett bilde per akse her
```
```losning
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

fig, axes = plt.subplots(2, 5, figsize=(8, 3.5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X[i].reshape(8, 8), cmap="gray_r")
    ax.set_title(f"y = {y[i]}")
    ax.axis("off")
plt.tight_layout()
```
```test
import numpy as np
assert fig is not None and len(fig.axes) == 10, "Lag 10 akser: fig, axes = plt.subplots(2, 5)."
for i, a in enumerate(fig.axes):
    assert len(a.images) == 1 and np.allclose(a.images[0].get_array(), X[i].reshape(8, 8)), f"Akse {i} skal vise X[{i}].reshape(8, 8)."
    assert a.images[0].get_cmap().name == "gray_r", "Bruk cmap='gray_r'."
    assert a.get_title() == f"y = {y[i]}", f"Akse {i} skal ha tittelen 'y = {y[i]}'."
    assert not a.axison, "Skru av aksene med ax.axis('off') i hver akse."
```
:::

::: oppgave Normaliser et MNIST-bilde
`rad` har 784 pikselverdier. Gjør den om til et 28×28-array `bilde`, lag `norm` med verdier fra 0 til 1, og vis `norm` med `cmap="gray"`.
```start
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
rad = rng.uniform(-3, 5, size=784)
fig, ax = plt.subplots(figsize=(3, 3))

bilde = ...
norm = ...
# vis norm her
```
```losning
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
rad = rng.uniform(-3, 5, size=784)
fig, ax = plt.subplots(figsize=(3, 3))

bilde = rad.reshape(28, 28)
norm = (bilde - bilde.min()) / (bilde.max() - bilde.min())
ax.imshow(norm, cmap="gray")
```
```test
import numpy as np
assert hasattr(bilde, "shape") and bilde.shape == (28, 28), "bilde skal ha shape (28, 28): rad.reshape(28, 28)."
assert hasattr(norm, "shape") and norm.shape == (28, 28), "norm skal ha samme shape som bilde."
assert np.isclose(norm.min(), 0) and np.isclose(norm.max(), 1), f"norm skal gå fra 0 til 1, men går fra {norm.min():.3f} til {norm.max():.3f}."
assert np.allclose(norm, (bilde - bilde.min()) / (bilde.max() - bilde.min())), "Bruk (bilde - bilde.min()) / (bilde.max() - bilde.min())."
assert len(ax.images) == 1 and np.allclose(ax.images[0].get_array(), norm), "Vis norm med ax.imshow(norm, cmap='gray')."
```
:::

## Heatmap av en matrise
`im = ax.imshow(M, cmap="RdYlGn")` og `fig.colorbar(im, ax=ax)` gir et heatmap, og `ax.text(c, r, tekst, ha="center", va="center")` skriver i ruten `M[r, c]` (kolonnen er x, så den kommer først). `np.nan`-ruter blir blanke.

::: oppgave Q-heatmap
Lag `q_grid`, den beste Q-verdien i hver av de 16 tilstandene lagt ut som 4×4, vis den med `cmap="RdYlGn"` og en fargeskala, og fjern tallene på aksene med `ax.set_xticks([])` og `ax.set_yticks([])`.
```start
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
q_table = rng.uniform(0, 10, size=(16, 4))   # 16 tilstander, 4 handlinger
fig, ax = plt.subplots(figsize=(5, 5))

q_grid = ...
# heatmap her
```
```losning
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
q_table = rng.uniform(0, 10, size=(16, 4))   # 16 tilstander, 4 handlinger
fig, ax = plt.subplots(figsize=(5, 5))

q_grid = q_table.max(axis=1).reshape(4, 4)
im = ax.imshow(q_grid, cmap="RdYlGn")
fig.colorbar(im, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
```
```test
import numpy as np
assert hasattr(q_grid, "shape") and q_grid.shape == (4, 4), "q_grid skal ha shape (4, 4): q_table.max(axis=1).reshape(4, 4)."
assert np.allclose(q_grid, q_table.max(axis=1).reshape(4, 4)), "q_grid skal være maks over handlingene (axis=1), ikke over tilstandene."
assert len(ax.images) == 1 and np.allclose(ax.images[0].get_array(), q_grid), "Vis q_grid med ax.imshow(q_grid, cmap='RdYlGn')."
assert ax.images[0].get_cmap().name == "RdYlGn", "Bruk cmap='RdYlGn'."
assert len(fig.axes) == 2, "Legg til fargeskala med fig.colorbar(im, ax=ax)."
assert len(ax.get_xticks()) == 0 and len(ax.get_yticks()) == 0, "Fjern tallene med ax.set_xticks([]) og ax.set_yticks([])."
```
:::

::: oppgave Tall i rutene
Skriv verdien `f"{q_grid[r, c]:.1f}"` midt i hver rute som ikke er `nan`, med `ax.text(c, r, ..., ha="center", va="center")`.
```start
import numpy as np
import matplotlib.pyplot as plt

q_grid = np.round(np.random.default_rng(2).uniform(0, 10, size=(4, 4)), 1)
q_grid[1, 1] = q_grid[2, 3] = np.nan          # vann
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(q_grid, cmap="RdYlGn")
# skriv tallene her
```
```losning
import numpy as np
import matplotlib.pyplot as plt

q_grid = np.round(np.random.default_rng(2).uniform(0, 10, size=(4, 4)), 1)
q_grid[1, 1] = q_grid[2, 3] = np.nan          # vann
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(q_grid, cmap="RdYlGn")
for r in range(4):
    for c in range(4):
        if not np.isnan(q_grid[r, c]):
            ax.text(c, r, f"{q_grid[r, c]:.1f}", ha="center", va="center")
```
```test
import numpy as np
assert len(ax.texts) == 14, f"Det skal være 14 tall (16 ruter minus 2 nan), men det er {len(ax.texts)}. Hopp over rutene der np.isnan(q_grid[r, c])."
for t in ax.texts:
    x, y = t.get_position()
    r, c = int(round(y)), int(round(x))
    assert t.get_text() == f"{q_grid[r, c]:.1f}", f"Teksten '{t.get_text()}' står på (x={x}, y={y}), men der står {q_grid[r, c]}. Husk ax.text(c, r, ...) med kolonnen først."
    assert t.get_ha() == "center" and t.get_va() == "center", "Bruk ha='center' og va='center'."
```
:::

--- LEKSJON 4.6 | beslutningsgrense | Beslutningsgrenser med meshgrid og contourf
Beslutningsgrensen tegnes ved å predikere på et tett rutenett: `xx0, xx1 = np.meshgrid(xs, ys)` gir to matriser med shape `(len(ys), len(xs))`, og `np.column_stack((xx0.ravel(), xx1.ravel()))` gir ett punkt per rad.

::: oppgave Rutenett med meshgrid
Lag `xx0, xx1` fra `xs` og `ys`, og `punkter` med ett (x0, x1)-punkt per rad.
```start
import numpy as np

xs = np.arange(0, 3)
ys = np.arange(0, 5)

xx0, xx1 = ..., ...
punkter = ...
```
```losning
import numpy as np

xs = np.arange(0, 3)
ys = np.arange(0, 5)

xx0, xx1 = np.meshgrid(xs, ys)
punkter = np.column_stack((xx0.ravel(), xx1.ravel()))
```
```test
import numpy as np
assert isinstance(xx0, np.ndarray) and xx0.shape == (5, 3), f"xx0 skal ha shape (len(ys), len(xs)) = (5, 3). Bruk np.meshgrid(xs, ys)."
assert np.array_equal(xx0[0], xs) and np.array_equal(xx1[:, 0], ys), "xx0 skal variere langs kolonnene (xs) og xx1 langs radene (ys)."
assert isinstance(punkter, np.ndarray) and punkter.shape == (15, 2), "punkter skal ha shape (15, 2): np.column_stack((xx0.ravel(), xx1.ravel()))."
assert np.array_equal(punkter[:, 0], xx0.ravel()) and np.array_equal(punkter[:, 1], xx1.ravel()), "Kolonne 0 skal være xx0.ravel() og kolonne 1 xx1.ravel()."
```
:::

::: oppgave Rutenett rundt dataene
Lag `xx0, xx1` med `np.arange` og steg 0.05 fra 1 under til 1 over minste og største verdi av `x0` og `x1` i `train`, og lagre antall punkter i rutenettet i `n_punkter`.
```start
import numpy as np
import pandas as pd

data = pd.read_csv("problem1.csv")
train = data[data["split"] == "train"]

xx0, xx1 = ..., ...
n_punkter = ...
```
```losning
import numpy as np
import pandas as pd

data = pd.read_csv("problem1.csv")
train = data[data["split"] == "train"]

x0_min, x0_max = train["x0"].min() - 1, train["x0"].max() + 1
x1_min, x1_max = train["x1"].min() - 1, train["x1"].max() + 1
xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, 0.05), np.arange(x1_min, x1_max, 0.05))
n_punkter = xx0.size
```
```test
import numpy as np
fx = np.arange(train["x0"].min() - 1, train["x0"].max() + 1, 0.05)
fy = np.arange(train["x1"].min() - 1, train["x1"].max() + 1, 0.05)
assert isinstance(xx0, np.ndarray) and xx0.shape == (len(fy), len(fx)), f"xx0 skal ha shape {(len(fy), len(fx))}, men har {np.shape(xx0)}. Sjekk grensene (min - 1, max + 1) og steget 0.05."
assert np.allclose(xx0[0], fx) and np.allclose(xx1[:, 0], fy), "xx0 skal komme fra x0-verdiene og xx1 fra x1-verdiene."
assert n_punkter == xx0.size, f"n_punkter skal være antall punkter i rutenettet: xx0.size = {xx0.size}."
```
:::

## Prediker på rutenettet
`grid = pd.DataFrame({"x0": xx0.ravel(), "x1": xx1.ravel()})` har samme kolonnenavn som treningsdataene, og `Z = clf.predict(grid).reshape(xx0.shape)` legger svarene tilbake i rutenettet. `plt.contourf(xx0, xx1, Z, alpha=0.4, cmap="coolwarm")` fargelegger områdene.

::: oppgave Tegn beslutningsgrensen
Lag `grid` og `Z` for beslutningstreet `clf`, og tegn `Z` med `plt.contourf(..., alpha=0.4, cmap="coolwarm")` under treningspunktene.
```start
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
clf = DecisionTreeClassifier(max_depth=4, random_state=0).fit(train[["x0", "x1"]], train["y"])
xx0, xx1 = np.meshgrid(np.arange(-4, 4, 0.1), np.arange(-4, 4, 0.1))
fig, ax = plt.subplots(figsize=(6, 4))

grid = ...
Z = ...
# contourf her
plt.scatter(train["x0"], train["x1"], c=train["y"], cmap="coolwarm", s=10)
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
fig, ax = plt.subplots(figsize=(6, 4))

grid = pd.DataFrame({"x0": xx0.ravel(), "x1": xx1.ravel()})
Z = clf.predict(grid).reshape(xx0.shape)
plt.contourf(xx0, xx1, Z, alpha=0.4, cmap="coolwarm")
plt.scatter(train["x0"], train["x1"], c=train["y"], cmap="coolwarm", s=10)
```
```test
import numpy as np
import pandas as pd
assert isinstance(grid, pd.DataFrame) and list(grid.columns) == ["x0", "x1"], "grid skal være en DataFrame med kolonnene 'x0' og 'x1'."
assert len(grid) == xx0.size, "grid skal ha én rad per punkt i rutenettet: bruk xx0.ravel() og xx1.ravel()."
assert isinstance(Z, np.ndarray) and Z.shape == xx0.shape, f"Z skal ha samme shape som xx0 {xx0.shape}. Husk .reshape(xx0.shape)."
fasit = clf.predict(pd.DataFrame({"x0": xx0.ravel(), "x1": xx1.ravel()})).reshape(xx0.shape)
assert np.array_equal(Z, fasit), "Z stemmer ikke med clf.predict(grid). Sjekk at x0 og x1 ikke er byttet om."
fylt = [c for c in ax.collections if getattr(c, "filled", False)]
assert fylt, "Fant ikke contourf-plottet. Bruk plt.contourf(xx0, xx1, Z, alpha=0.4, cmap='coolwarm')."
assert fylt[0].get_alpha() == 0.4, "Bruk alpha=0.4, så punktene oppå syns."
```
:::

## Sannsynligheter i stedet for klasser
`clf.predict_proba(grid)[:, 1]` gir P(y = 1) i hvert punkt, og `ax.contour(xx0, xx1, P, levels=[0.5], colors="k")` tegner selve grensen der P = 0.5.

::: oppgave Sannsynlighetskart
Lag `P`, sannsynligheten for klasse 1 med samme shape som `xx0`, tegn den med `ax.contourf(xx0, xx1, P, levels=20, cmap="coolwarm")`, og tegn grensen P = 0.5 i svart med `ax.contour`.
```start
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
clf = LogisticRegression().fit(train[["x0", "x1"]], train["y"])
xx0, xx1 = np.meshgrid(np.arange(-4, 4, 0.05), np.arange(-4, 4, 0.05))
grid = pd.DataFrame({"x0": xx0.ravel(), "x1": xx1.ravel()})
fig, ax = plt.subplots(figsize=(6, 4))

P = ...
# contourf og contour her
```
```losning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
clf = LogisticRegression().fit(train[["x0", "x1"]], train["y"])
xx0, xx1 = np.meshgrid(np.arange(-4, 4, 0.05), np.arange(-4, 4, 0.05))
grid = pd.DataFrame({"x0": xx0.ravel(), "x1": xx1.ravel()})
fig, ax = plt.subplots(figsize=(6, 4))

P = clf.predict_proba(grid)[:, 1].reshape(xx0.shape)
ax.contourf(xx0, xx1, P, levels=20, cmap="coolwarm")
ax.contour(xx0, xx1, P, levels=[0.5], colors="k")
```
```test
import numpy as np
assert isinstance(P, np.ndarray) and P.shape == xx0.shape, f"P skal ha shape {xx0.shape}. Husk .reshape(xx0.shape)."
assert np.allclose(P, clf.predict_proba(grid)[:, 1].reshape(xx0.shape)), "P skal være kolonne 1 av predict_proba: clf.predict_proba(grid)[:, 1]."
konturer = [c for c in ax.collections if hasattr(c, "filled")]
assert any(c.filled for c in konturer), "Tegn sannsynlighetene med ax.contourf(xx0, xx1, P, levels=20, cmap='coolwarm')."
linjer = [c for c in konturer if not c.filled]
assert linjer and np.allclose(linjer[0].levels, [0.5]), "Tegn grensen med ax.contour(xx0, xx1, P, levels=[0.5], colors='k')."
```
:::
