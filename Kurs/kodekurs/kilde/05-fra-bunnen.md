=== DEL 5 | fra-bunnen | Maskinlæring fra bunnen av (prosjekt 1)
Du bygger lineær og logistisk regresjon fra bunnen med numpy, steg for steg, til du har skrevet hele `linear_models.py` fra prosjekt 1 selv.

--- LEKSJON 5.1 | modell-prediksjon | Modell og prediksjon
En lineær modell predikerer $\hat y = w \cdot x + b$, der `x` er en feature og vekten `w` og bias `b` er **parametrene** som treningen skal finne.

::: oppgave Én prediksjon
Lag `w = 3.0`, `b = 5.0` og `x = 20.0`, og regn ut prediksjonen `y_hat`.
```start
w = ...
b = ...
x = ...
y_hat = ...
```
```losning
w = 3.0
b = 5.0
x = 20.0
y_hat = w * x + b
```
```test
assert y_hat is not ..., "Regn ut y_hat = w * x + b."
assert abs(w - 3) < 1e-9 and abs(b - 5) < 1e-9 and abs(x - 20) < 1e-9, "w, b og x skal være 3.0, 5.0 og 20.0."
assert abs(y_hat - 65) < 1e-9, f"y_hat skal være 3 · 20 + 5 = 65, men er {y_hat}."
```
:::

## Mange datapunkter på én gang
`w * x + b` på en numpy-array `x` gir én prediksjon per datapunkt.

::: oppgave Prediker strømforbruk
Hent `x` (kolonnen `Net_Activity`) og `y` (kolonnen `Energy`) som numpy-arrays fra `data`, og lag `y_hat` med `w = 2.0` og `b = 20.0`.
```start
import numpy as np
import pandas as pd

data = pd.read_csv("regression.csv")
w, b = 2.0, 20.0
x = ...
y = ...
y_hat = ...
```
```losning
import numpy as np
import pandas as pd

data = pd.read_csv("regression.csv")
w, b = 2.0, 20.0
x = data["Net_Activity"].values
y = data["Energy"].values
y_hat = w * x + b
```
```test
import numpy as np
assert x is not ... and y is not ... and y_hat is not ..., "Lag x, y og y_hat."
assert np.shape(x) == (1000,) and np.shape(y) == (1000,), f"x og y skal ha shape (1000,), men har {np.shape(x)} og {np.shape(y)}. Bruk data['kolonne'].values."
assert np.allclose(x, data["Net_Activity"].values), "x skal være kolonnen Net_Activity."
assert np.allclose(y, data["Energy"].values), "y skal være kolonnen Energy."
assert np.allclose(y_hat, 2.0 * x + 20.0), "y_hat skal være w * x + b for alle punktene."
```
:::

::: oppgave Tegn linjen over dataene
Tegn punktene med `ax.scatter(x, y, s=8)` og linjen med `ax.plot(x, y_hat, color="red")`. Prøv så å endre `w` og `b` til linjen passer bedre.
```start
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values
y = data["Energy"].values
w, b = 2.0, 20.0
y_hat = w * x + b

fig, ax = plt.subplots()
# tegn her
```
```losning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values
y = data["Energy"].values
w, b = 3.0, 5.0
y_hat = w * x + b

fig, ax = plt.subplots()
ax.scatter(x, y, s=8)
ax.plot(x, y_hat, color="red")
plt.show()
```
```test
assert len(ax.collections) >= 1, "Fant ingen punkter. Bruk ax.scatter(x, y, s=8)."
assert len(ax.lines) >= 1, "Fant ingen linje. Bruk ax.plot(x, y_hat, color='red')."
```
:::

## Flere features: matrisen X
Med `X` av shape `(m, n)` (én rad per datapunkt) og `w` av shape `(n,)` gir `X @ w + b` (samme som `np.matmul(X, w) + b`) én prediksjon per rad. `df[["kol"]].values` gir en 2D-matrise `(m, 1)`, mens `df["kol"].values` gir 1D `(m,)`.

::: oppgave Matrise ganger vektor
Regn ut `y_hat = X @ w + b`, og lagre shapen i `form`.
```start
import numpy as np

X = np.array([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0]])
w = np.array([0.5, -1.0])
b = 2.0

y_hat = ...
form = ...
```
```losning
import numpy as np

X = np.array([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0]])
w = np.array([0.5, -1.0])
b = 2.0

y_hat = X @ w + b
form = y_hat.shape
```
```test
import numpy as np
assert y_hat is not ..., "Regn ut y_hat = X @ w + b."
assert np.allclose(y_hat, [0.5, -0.5, -1.5]), f"y_hat = {y_hat} er feil. Første rad: 0.5·1 + (-1)·2 + 2 = 0.5."
assert tuple(form) == (3,), f"form skal være (3,): én prediksjon per rad. Fikk {form}."
```
:::

::: oppgave X som 2D-matrise
Lag `X` med shape `(1000, 1)` fra kolonnen `Net_Activity` (doble klammer) og `y` med shape `(1000,)` fra `Energy`.
```start
import pandas as pd

data = pd.read_csv("regression.csv")
X = ...
y = ...
```
```losning
import pandas as pd

data = pd.read_csv("regression.csv")
X = data[["Net_Activity"]].values
y = data["Energy"].values
```
```test
import numpy as np
assert X is not ... and y is not ..., "Lag både X og y."
assert np.shape(X) == (1000, 1), f"X skal ha shape (1000, 1), men har {np.shape(X)}. Bruk data[['Net_Activity']].values med doble klammer."
assert np.shape(y) == (1000,), f"y skal ha shape (1000,), men har {np.shape(y)}. Bruk data['Energy'].values med enkle klammer."
assert np.allclose(np.asarray(X)[:, 0], data["Net_Activity"].values), "X skal inneholde Net_Activity."
```
:::

::: oppgave Skriv predict-funksjonen
Lag funksjonen `predict(X, w, b)` som returnerer $\hat y = Xw + b$ for alle radene i `X`.
```start
import numpy as np

def predict(X, w, b):
    ...
```
```losning
import numpy as np

def predict(X, w, b):
    return X @ w + b
```
```test
import numpy as np
X_t = np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]])
w_t = np.array([2.0, -1.0])
res = predict(X_t, w_t, 1.0)
assert res is not None, "Funksjonen returnerer ingenting. Husk return."
res = np.asarray(res)
assert res.shape == (3,), f"Forventet én prediksjon per rad, altså shape (3,), men fikk {res.shape}."
assert np.allclose(res, [1.0, 3.0, 1.0]), f"Feil verdier: {res}. Regn ut X @ w + b."
```
:::

--- LEKSJON 5.2 | mse | Tapsfunksjon: MSE
MSE er $\frac{1}{n}\sum_i (\hat y_i - y_i)^2$, i numpy `np.mean((y_pred - y) ** 2)`, og kvadratet hindrer at positive og negative feil nuller hverandre ut.

::: oppgave Regn ut MSE
Lag `feil = y_pred - y` og `mse_verdi`, gjennomsnittet av de kvadrerte feilene.
```start
import numpy as np

y      = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = np.array([2.0, 2.0, 1.0, 4.0])

feil = ...
mse_verdi = ...
```
```losning
import numpy as np

y      = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = np.array([2.0, 2.0, 1.0, 4.0])

feil = y_pred - y
mse_verdi = np.mean(feil ** 2)
```
```test
import numpy as np
assert feil is not ... and mse_verdi is not ..., "Lag både feil og mse_verdi."
assert np.allclose(feil, [1.0, 0.0, -2.0, 0.0]), f"feil skal være y_pred - y = [1, 0, -2, 0], men er {feil}."
assert abs(mse_verdi - 5.0) > 1e-9, "Du fikk 5.0, som er summen. Del på antall punkter (bruk np.mean)."
assert abs(mse_verdi - (-0.25)) > 1e-9, "Du fikk -0.25. Husk å kvadrere feilene før du tar gjennomsnittet."
assert abs(mse_verdi - 1.25) < 1e-9, f"mse_verdi skal være 1.25, men er {mse_verdi}."
```
:::

::: oppgave Hvorfor kvadrere?
Modellen bommer med 10 på begge punktene. Lag `snitt_feil` (gjennomsnittet av `y_pred - y`, uten kvadrat) og `mse_verdi`, og se hvilken av dem som avslører feilen.
```start
import numpy as np

y      = np.array([10.0, 10.0])
y_pred = np.array([0.0, 20.0])

snitt_feil = ...
mse_verdi = ...
```
```losning
import numpy as np

y      = np.array([10.0, 10.0])
y_pred = np.array([0.0, 20.0])

snitt_feil = np.mean(y_pred - y)
mse_verdi = np.mean((y_pred - y) ** 2)
```
```test
assert snitt_feil is not ... and mse_verdi is not ..., "Lag både snitt_feil og mse_verdi."
assert abs(snitt_feil) < 1e-9, f"snitt_feil skal være 0 (feilene -10 og +10 nuller hverandre ut), men er {snitt_feil}."
assert abs(mse_verdi - 100) < 1e-9, f"mse_verdi skal være 100, men er {mse_verdi}."
```
:::

::: oppgave Skriv mse
Lag funksjonen `mse(y_true, y_pred)` som returnerer gjennomsnittlig kvadratisk feil mellom to numpy-arrays.
```start
import numpy as np

def mse(y_true, y_pred):
    ...
```
```losning
import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)
```
```test
import numpy as np
r = mse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 4.0, 0.0]))
assert r is not None, "Funksjonen returnerer ingenting. Husk return."
assert abs(r - 13 / 3) < 1e-9, f"Fikk {r}, forventet 13/3 ≈ 4.333. Kvadrer feilene og ta gjennomsnittet (ikke summen)."
assert abs(mse(np.array([5.0]), np.array([5.0]))) < 1e-12, "MSE skal være 0 når prediksjonen er perfekt."
```
:::

## Tapet som funksjon av parameteren
Å trene betyr å finne parametrene med lavest tap, og `np.argmin(tap)` gir indeksen til den laveste verdien i en liste.

::: oppgave Prøv mange verdier av w
Hold `b = 5` fast. Lag listen `tap` med MSE for hver `w` i `w_verdier`, og lagre den `w`-en som gir lavest tap i `beste_w`.
```start
import numpy as np
import pandas as pd

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values
y = data["Energy"].values
w_verdier = np.linspace(1, 5, 81)

tap = []
# fyll tap
beste_w = ...
```
```losning
import numpy as np
import pandas as pd

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values
y = data["Energy"].values
w_verdier = np.linspace(1, 5, 81)

tap = []
for w in w_verdier:
    tap.append(np.mean((w * x + 5 - y) ** 2))
beste_w = w_verdier[np.argmin(tap)]
```
```test
import numpy as np
assert len(tap) == 81, f"tap skal ha én verdi per w (81), men har {len(tap)}."
fasit = [np.mean((w_ * x + 5 - y) ** 2) for w_ in w_verdier]
assert np.allclose(tap, fasit), "Verdiene i tap er feil. Bruk MSE med prediksjonen w * x + 5."
assert beste_w is not ..., "Lag beste_w."
assert abs(beste_w - w_verdier[np.argmin(fasit)]) < 1e-9, f"beste_w = {beste_w} er feil. Bruk w_verdier[np.argmin(tap)], ikke selve indeksen."
```
:::

--- LEKSJON 5.3 | gradienter | Gradienter
For MSE er $\frac{\partial L}{\partial w} = \frac{2}{n}X^T(\hat y - y)$ og $\frac{\partial L}{\partial b} = \frac{2}{n}\sum_i(\hat y_i - y_i)$. Positiv gradient betyr at tapet øker når parameteren øker, så da skal parameteren **minke**.

::: oppgave Regn ut dw og db
Regn ut `y_pred`, `dw` og `db` for dataene under. `dw` skal ha samme shape som `w`, og `db` er ett tall.
```start
import numpy as np

X = np.array([[1.0], [2.0], [3.0]])
y = np.array([5.0, 7.0, 9.0])
w = np.array([0.0])
b = 0.0
n = len(y)

y_pred = ...
dw = ...
db = ...
```
```losning
import numpy as np

X = np.array([[1.0], [2.0], [3.0]])
y = np.array([5.0, 7.0, 9.0])
w = np.array([0.0])
b = 0.0
n = len(y)

y_pred = X @ w + b
dw = (2 / n) * X.T @ (y_pred - y)
db = (2 / n) * np.sum(y_pred - y)
```
```test
import numpy as np
assert dw is not ... and db is not ..., "Lag både dw og db."
assert np.shape(dw) == (1,), f"dw skal ha samme shape som w, (1,), men har {np.shape(dw)}. Bruk X.T @ (y_pred - y)."
assert np.allclose(dw, [-92 / 3]), f"dw = {dw} er feil, forventet ≈ -30.667. Sjekk fortegnet (y_pred - y) og faktoren 2/n."
assert abs(float(db) - (-14.0)) < 1e-9, f"db = {db} er feil, forventet -14. Det er (2/n) * summen av (y_pred - y)."
```
:::

::: oppgave Hvilken vei?
Lag den boolske arrayen `skal_minke` som er `True` for vektene som bør minkes (der gradienten er positiv).
```start
import numpy as np

dw = np.array([4.2, -1.5, 0.3, -0.01])
skal_minke = ...
```
```losning
import numpy as np

dw = np.array([4.2, -1.5, 0.3, -0.01])
skal_minke = dw > 0
```
```test
import numpy as np
assert skal_minke is not ..., "Lag skal_minke, f.eks. med en sammenligning på dw."
assert list(np.asarray(skal_minke, dtype=bool)) == [True, False, True, False], f"Fikk {skal_minke}. Positiv gradient betyr at tapet øker med vekten, så den vekten skal minke."
```
:::

::: oppgave Skriv gradients
Lag funksjonen `gradients(X, y, w, b)` som returnerer `dw, db` (i den rekkefølgen) for MSE-tapet til $\hat y = Xw + b$.
```start
import numpy as np

def gradients(X, y, w, b):
    n = len(y)
    y_pred = ...
    dw = ...
    db = ...
    return dw, db
```
```losning
import numpy as np

def gradients(X, y, w, b):
    n = len(y)
    y_pred = X @ w + b
    dw = (2 / n) * X.T @ (y_pred - y)
    db = (2 / n) * np.sum(y_pred - y)
    return dw, db
```
```test
import numpy as np
X_t = np.array([[1.0, 0.0], [2.0, 1.0], [0.0, 3.0], [1.0, 1.0]])
y_t = np.array([1.0, 2.0, 0.0, 4.0])
w_t = np.array([0.5, -0.5])
dw, db = gradients(X_t, y_t, w_t, 0.25)
assert dw is not ... and db is not ..., "Fyll inn y_pred, dw og db."
dw = np.asarray(dw, dtype=float)
assert dw.shape == (2,), f"dw skal ha samme shape som w, (2,), men har {dw.shape}. Bruk X.T @ (y_pred - y)."
assert np.allclose(dw, [-3.25, -4.375]), f"dw = {dw} er feil. Sjekk fortegnet (y_pred - y) og faktoren 2/n."
assert abs(float(db) - (-3.25)) < 1e-9, f"db = {db} er feil. Det skal være (2/n) * summen av (y_pred - y)."
```
:::

## Sjekk gradienten numerisk
Endelig differanse $\frac{L(w + h) - L(w - h)}{2h}$ med liten `h` (f.eks. `1e-5`) skal gi nesten samme tall som formelen.

::: oppgave Numerisk db
Regn ut den numeriske gradienten `db_num` for bias med funksjonen `tap(w, b)`, og sammenlign med den analytiske `db`.
```start
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(50, 2))
y = X @ np.array([1.5, -2.0]) + 0.7 + rng.normal(0, 0.1, 50)
w = np.array([0.3, 0.1])
b = 0.2

def tap(w, b):
    return np.mean((X @ w + b - y) ** 2)

db = (2 / len(y)) * np.sum(X @ w + b - y)
h = 1e-5
db_num = ...
```
```losning
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(50, 2))
y = X @ np.array([1.5, -2.0]) + 0.7 + rng.normal(0, 0.1, 50)
w = np.array([0.3, 0.1])
b = 0.2

def tap(w, b):
    return np.mean((X @ w + b - y) ** 2)

db = (2 / len(y)) * np.sum(X @ w + b - y)
h = 1e-5
db_num = (tap(w, b + h) - tap(w, b - h)) / (2 * h)
```
```test
assert db_num is not ..., "Regn ut db_num = (tap(w, b + h) - tap(w, b - h)) / (2 * h)."
assert abs(float(db_num) - float(db)) < 1e-6, f"db_num = {db_num} skulle vært nesten lik db = {db}. Husk å dele på 2 * h."
```
:::

::: oppgave Numerisk dw
Fyll `dw_num` med én numerisk gradient per vekt: flytt bare vekt `j` med `h` ved hjelp av `e = np.zeros(2)` og `e[j] = h`.
```start
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(50, 2))
y = X @ np.array([1.5, -2.0]) + 0.7 + rng.normal(0, 0.1, 50)
w = np.array([0.3, 0.1])
b = 0.2

def tap(w, b):
    return np.mean((X @ w + b - y) ** 2)

dw = (2 / len(y)) * X.T @ (X @ w + b - y)
h = 1e-5
dw_num = np.zeros(2)
for j in range(2):
    ...
```
```losning
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(50, 2))
y = X @ np.array([1.5, -2.0]) + 0.7 + rng.normal(0, 0.1, 50)
w = np.array([0.3, 0.1])
b = 0.2

def tap(w, b):
    return np.mean((X @ w + b - y) ** 2)

dw = (2 / len(y)) * X.T @ (X @ w + b - y)
h = 1e-5
dw_num = np.zeros(2)
for j in range(2):
    e = np.zeros(2)
    e[j] = h
    dw_num[j] = (tap(w + e, b) - tap(w - e, b)) / (2 * h)
```
```test
import numpy as np
assert np.shape(dw_num) == (2,), f"dw_num skal ha shape (2,), men har {np.shape(dw_num)}."
assert not np.allclose(dw_num, 0), "dw_num er fortsatt bare nuller. Fyll inn dw_num[j] i løkka."
assert np.allclose(dw_num, dw, atol=1e-6), f"dw_num = {dw_num} skulle vært nesten lik dw = {dw}. Flytt bare vekt j, og del på 2 * h."
```
:::

--- LEKSJON 5.4 | gradient-descent | Gradient descent-løkka
Gradient descent gjentar `w = w - lr * dw` og `b = b - lr * db` i `n_iterations` steg, der læringsraten `lr` og `n_iterations` er **hyperparametre**.

::: oppgave Ett steg
Ta ett gradient descent-steg: lag `w_ny` og `b_ny` fra `w`, `b`, gradientene og `lr`.
```start
import numpy as np

w = np.array([1.0, -2.0])
b = 0.5
dw = np.array([4.0, -3.0])
db = 2.0
lr = 0.1

w_ny = ...
b_ny = ...
```
```losning
import numpy as np

w = np.array([1.0, -2.0])
b = 0.5
dw = np.array([4.0, -3.0])
db = 2.0
lr = 0.1

w_ny = w - lr * dw
b_ny = b - lr * db
```
```test
import numpy as np
assert w_ny is not ... and b_ny is not ..., "Lag både w_ny og b_ny."
assert not np.allclose(w_ny, [1.4, -2.3]), "Du har gått i samme retning som gradienten. Det skal være minus: w - lr * dw."
assert np.allclose(w_ny, [0.6, -1.7]), f"w_ny = {w_ny} er feil, forventet [0.6, -1.7]."
assert abs(b_ny - 0.3) < 1e-9, f"b_ny = {b_ny} er feil, forventet 0.5 - 0.1 · 2 = 0.3."
```
:::

::: oppgave Skriv gradient descent-løkka
Fullfør løkka: legg MSE-tapet til i `loss_history` (før oppdateringen), regn ut `dw` og `db`, og oppdater `w` og `b`.
```start
import numpy as np

rng = np.random.default_rng(1)
X = rng.uniform(0, 2, size=(40, 1))
y = 2 * X[:, 0] - 1 + rng.normal(0, 0.1, 40)

w = np.zeros(1)
b = 0.0
lr = 0.1
n_iterations = 300
loss_history = []

for i in range(n_iterations):
    y_pred = X @ w + b
    # legg til tapet, regn ut dw og db, og oppdater w og b
    ...
```
```losning
import numpy as np

rng = np.random.default_rng(1)
X = rng.uniform(0, 2, size=(40, 1))
y = 2 * X[:, 0] - 1 + rng.normal(0, 0.1, 40)

w = np.zeros(1)
b = 0.0
lr = 0.1
n_iterations = 300
loss_history = []

for i in range(n_iterations):
    y_pred = X @ w + b
    loss_history.append(np.mean((y_pred - y) ** 2))
    dw = (2 / len(y)) * X.T @ (y_pred - y)
    db = (2 / len(y)) * np.sum(y_pred - y)
    w = w - lr * dw
    b = b - lr * db
```
```test
import numpy as np
assert len(loss_history) == 300, f"loss_history skal ha ett tap per iterasjon (300), men har {len(loss_history)}."
assert loss_history[-1] < loss_history[0], "Tapet øker. Har du riktig fortegn? Det skal være w = w - lr * dw."
assert abs(float(np.ravel(w)[0]) - 2) < 0.1, f"w = {w} skulle vært nær 2. Sjekk gradienten for w."
assert abs(float(b) - (-1)) < 0.15, f"b = {b} skulle vært nær -1. Sjekk gradienten for b."
```
:::

## Læringsraten: for liten, passe, for stor
For liten `lr` gir treg nedgang, og for stor `lr` får tapet til å oscillere og vokse (til slutt `inf` eller `nan`).

::: oppgave Sammenlign læringsrater
Lag dicten `historikk` med `train(lr)` for `lr` lik `0.01`, `0.1` og `0.5`, og lagre den læringsraten som gir lavest siste tap i `beste_lr`.
```start
import numpy as np

rng = np.random.default_rng(0)
X = rng.uniform(0, 2, size=(50, 1))
y = 3 * X[:, 0] + 5 + rng.normal(0, 0.5, 50)

def train(lr, n_iterations=60):
    w, b, history = np.zeros(1), 0.0, []
    for i in range(n_iterations):
        y_pred = X @ w + b
        history.append(np.mean((y_pred - y) ** 2))
        w = w - lr * (2 / len(y)) * X.T @ (y_pred - y)
        b = b - lr * (2 / len(y)) * np.sum(y_pred - y)
    return history

historikk = {}
# fyll historikk
beste_lr = ...
```
```losning
import numpy as np

rng = np.random.default_rng(0)
X = rng.uniform(0, 2, size=(50, 1))
y = 3 * X[:, 0] + 5 + rng.normal(0, 0.5, 50)

def train(lr, n_iterations=60):
    w, b, history = np.zeros(1), 0.0, []
    for i in range(n_iterations):
        y_pred = X @ w + b
        history.append(np.mean((y_pred - y) ** 2))
        w = w - lr * (2 / len(y)) * X.T @ (y_pred - y)
        b = b - lr * (2 / len(y)) * np.sum(y_pred - y)
    return history

historikk = {}
for lr in [0.01, 0.1, 0.5]:
    historikk[lr] = train(lr)
beste_lr = min(historikk, key=lambda lr: historikk[lr][-1])
```
```test
assert sorted(historikk) == [0.01, 0.1, 0.5], f"historikk skal ha nøklene 0.01, 0.1 og 0.5, men har {sorted(historikk)}."
assert all(len(h) == 60 for h in historikk.values()), "Hver verdi i historikk skal være listen fra train(lr)."
assert historikk[0.5][-1] > historikk[0.5][0], "Med lr = 0.5 skal tapet vokse. Har du brukt train(lr) med riktig lr?"
assert beste_lr == 0.1, f"beste_lr skal være læringsraten med lavest siste tap, altså 0.1, men er {beste_lr}."
```
:::

::: oppgave Plott tapskurvene
Tegn én kurve per læringsrate med `ax.plot(historikk[lr], label=...)`, og gjør y-aksen logaritmisk med `ax.set_yscale("log")`.
```start
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
X = rng.uniform(0, 2, size=(50, 1))
y = 3 * X[:, 0] + 5 + rng.normal(0, 0.5, 50)

def train(lr, n_iterations=60):
    w, b, history = np.zeros(1), 0.0, []
    for i in range(n_iterations):
        y_pred = X @ w + b
        history.append(np.mean((y_pred - y) ** 2))
        w = w - lr * (2 / len(y)) * X.T @ (y_pred - y)
        b = b - lr * (2 / len(y)) * np.sum(y_pred - y)
    return history

historikk = {lr: train(lr) for lr in [0.01, 0.1, 0.5]}

fig, ax = plt.subplots()
# tegn her
```
```losning
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
X = rng.uniform(0, 2, size=(50, 1))
y = 3 * X[:, 0] + 5 + rng.normal(0, 0.5, 50)

def train(lr, n_iterations=60):
    w, b, history = np.zeros(1), 0.0, []
    for i in range(n_iterations):
        y_pred = X @ w + b
        history.append(np.mean((y_pred - y) ** 2))
        w = w - lr * (2 / len(y)) * X.T @ (y_pred - y)
        b = b - lr * (2 / len(y)) * np.sum(y_pred - y)
    return history

historikk = {lr: train(lr) for lr in [0.01, 0.1, 0.5]}

fig, ax = plt.subplots()
for lr in historikk:
    ax.plot(historikk[lr], label=f"lr = {lr}")
ax.set_yscale("log")
ax.set_xlabel("Iterasjon")
ax.set_ylabel("MSE")
ax.legend()
plt.show()
```
```test
assert len(ax.lines) == 3, f"Det skal være tre kurver, én per læringsrate, men fant {len(ax.lines)}."
assert ax.get_yscale() == "log", "Y-aksen skal være logaritmisk: ax.set_yscale('log')."
```
:::

## Hva betyr en gradient nær null?
Gradient nær null betyr at vi står nær et minimum (eller et flatt område), så stegene `lr * dw` blir bitte små, og `np.sqrt(np.sum(dw ** 2) + db ** 2) < 1e-4` kan brukes som stoppkriterium med `break`.

::: oppgave Stopp når gradienten er nær null
Legg til en sjekk i løkka som stopper med `break` når gradientens lengde er under `1e-4`, og lagre iterasjonsnummeret der den stoppet i `stopp_iter`.
```start
import numpy as np

rng = np.random.default_rng(0)
X = rng.uniform(0, 2, size=(50, 1))
y = 3 * X[:, 0] + 5 + rng.normal(0, 0.5, 50)

w, b, lr = np.zeros(1), 0.0, 0.1
stopp_iter = None
for i in range(10000):
    y_pred = X @ w + b
    dw = (2 / len(y)) * X.T @ (y_pred - y)
    db = (2 / len(y)) * np.sum(y_pred - y)
    # sjekk gradienten her
    w = w - lr * dw
    b = b - lr * db
```
```losning
import numpy as np

rng = np.random.default_rng(0)
X = rng.uniform(0, 2, size=(50, 1))
y = 3 * X[:, 0] + 5 + rng.normal(0, 0.5, 50)

w, b, lr = np.zeros(1), 0.0, 0.1
stopp_iter = None
for i in range(10000):
    y_pred = X @ w + b
    dw = (2 / len(y)) * X.T @ (y_pred - y)
    db = (2 / len(y)) * np.sum(y_pred - y)
    if np.sqrt(np.sum(dw ** 2) + db ** 2) < 1e-4:
        stopp_iter = i
        break
    w = w - lr * dw
    b = b - lr * db
```
```test
import numpy as np
assert stopp_iter is not None, "stopp_iter er None. Sett stopp_iter = i rett før break."
assert 50 < stopp_iter < 2000, f"stopp_iter = {stopp_iter} virker feil. Stopp når np.sqrt(np.sum(dw ** 2) + db ** 2) < 1e-4."
dw_s = (2 / len(y)) * X.T @ (X @ w + b - y)
db_s = (2 / len(y)) * np.sum(X @ w + b - y)
assert np.sqrt(np.sum(dw_s ** 2) + db_s ** 2) < 1e-4, "Gradienten er ikke nær null der løkka stoppet. Sjekk betingelsen."
```
:::

--- LEKSJON 5.5 | linear-regression | LinearRegression-klassen
Klassen lagrer hyperparametrene i `__init__`, `fit(X, y)` trener og endrer `self.weights` og `self.bias` (uten å returnere noe), og `predict(X)` returnerer `X @ self.weights + self.bias`.

::: oppgave Skriv __init__
Lag `__init__` slik at den lagrer `lr` og `n_iterations` på `self`, setter `self.weights` og `self.bias` til `None`, og `self.loss_history` til en tom liste.
```start
class LinearRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        ...
```
```losning
class LinearRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights = None
        self.bias = None
        self.lr = lr
        self.n_iterations = n_iterations
        self.loss_history = []
```
```test
m_t = LinearRegression()
for navn in ["weights", "bias", "lr", "n_iterations", "loss_history"]:
    assert hasattr(m_t, navn), f"Mangler self.{navn}. Lag den i __init__."
assert m_t.weights is None and m_t.bias is None, "weights og bias skal være None før fit."
assert m_t.lr == 0.001 and m_t.n_iterations == 1000, "Standardverdiene skal være lr=0.001 og n_iterations=1000."
assert m_t.loss_history == [], "loss_history skal starte som en tom liste."
m2 = LinearRegression(lr=0.1, n_iterations=5)
assert m2.lr == 0.1 and m2.n_iterations == 5, "Bruk argumentene: self.lr = lr og self.n_iterations = n_iterations."
```
:::

::: oppgave Skriv predict
Fullfør `predict` slik at den bruker de lagrede `self.weights` og `self.bias`.
```start
import numpy as np

class LinearRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights = None
        self.bias = None
        self.lr = lr
        self.n_iterations = n_iterations
        self.loss_history = []

    def predict(self, X):
        ...
```
```losning
import numpy as np

class LinearRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights = None
        self.bias = None
        self.lr = lr
        self.n_iterations = n_iterations
        self.loss_history = []

    def predict(self, X):
        return X @ self.weights + self.bias
```
```test
import numpy as np
m_t = LinearRegression()
m_t.weights = np.array([2.0, -1.0])
m_t.bias = 0.5
p_t = m_t.predict(np.array([[1.0, 1.0], [0.0, 2.0], [3.0, 0.0]]))
assert p_t is not None, "predict returnerer ingenting. Husk return."
assert np.allclose(p_t, [1.5, -1.5, 6.5]), f"predict ga {p_t}, forventet [1.5, -1.5, 6.5]. Bruk X @ self.weights + self.bias."
```
:::

::: oppgave Fullfør fit og predict
Fyll inn de tre manglende linjene i `fit` og linjen i `predict`. Klassen skal virke for vilkårlig mange features.
```start
import numpy as np

class LinearRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights = None
        self.bias = None
        self.lr = lr
        self.n_iterations = n_iterations
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        for _ in range(self.n_iterations):
            y_pred = ...
            self.loss_history.append(np.mean((y_pred - y) ** 2))
            dw = ...
            db = ...
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        ...
```
```losning
import numpy as np

class LinearRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights = None
        self.bias = None
        self.lr = lr
        self.n_iterations = n_iterations
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        for _ in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias
            self.loss_history.append(np.mean((y_pred - y) ** 2))
            dw = (2 / n_samples) * X.T @ (y_pred - y)
            db = (2 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return X @ self.weights + self.bias
```
```test
import numpy as np
rng_t = np.random.default_rng(3)
X_t = rng_t.normal(size=(100, 3))
y_t = X_t @ np.array([1.0, -2.0, 0.5]) + 4.0
m_t = LinearRegression(lr=0.1, n_iterations=300)
m_t.fit(X_t, y_t)
assert len(m_t.loss_history) == 300, "loss_history skal få ett element per iterasjon."
assert m_t.loss_history[-1] < m_t.loss_history[0], "Tapet øker. Oppdateringen skal gå motsatt vei av gradienten (-=)."
assert np.shape(m_t.weights) == (3,), f"weights skal ha shape (3,), men har {np.shape(m_t.weights)}."
assert np.allclose(m_t.weights, [1.0, -2.0, 0.5], atol=0.01), f"weights = {m_t.weights} skulle vært [1, -2, 0.5]. Sjekk dw."
assert abs(m_t.bias - 4.0) < 0.01, f"bias = {m_t.bias} skulle vært 4. Sjekk db."
p_t = m_t.predict(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
assert p_t is not None, "predict returnerer ingenting. Husk return."
assert np.allclose(p_t, [4.0, 3.5], atol=0.02), f"predict ga {p_t}, forventet omtrent [4, 3.5]."
```
:::

## Prosjekt 1, problem 1: strømforbruk
`Net_Activity` ligger mellom 10 og 30, så uten skalering må `lr` være liten (rundt `0.002`) og `n_iterations` stor (rundt `20000`).

::: oppgave Tren på regression.csv
Lag `X` (2D, `Net_Activity`) og `y` (`Energy`), tren `model = LinearRegression(lr=0.002, n_iterations=20000)`, og lagre den lærte vekten i `w` (et tall) og bias i `b`.
```start
import numpy as np
import pandas as pd

class LinearRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights, self.bias = None, None
        self.lr, self.n_iterations = lr, n_iterations
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights, self.bias = np.zeros(n_features), 0.0
        for _ in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias
            self.loss_history.append(np.mean((y_pred - y) ** 2))
            self.weights -= self.lr * (2 / n_samples) * X.T @ (y_pred - y)
            self.bias -= self.lr * (2 / n_samples) * np.sum(y_pred - y)

    def predict(self, X):
        return X @ self.weights + self.bias

data = pd.read_csv("regression.csv")
X = ...
y = ...
model = ...
w = ...
b = ...
```
```losning
import numpy as np
import pandas as pd

class LinearRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights, self.bias = None, None
        self.lr, self.n_iterations = lr, n_iterations
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights, self.bias = np.zeros(n_features), 0.0
        for _ in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias
            self.loss_history.append(np.mean((y_pred - y) ** 2))
            self.weights -= self.lr * (2 / n_samples) * X.T @ (y_pred - y)
            self.bias -= self.lr * (2 / n_samples) * np.sum(y_pred - y)

    def predict(self, X):
        return X @ self.weights + self.bias

data = pd.read_csv("regression.csv")
X = data[["Net_Activity"]].values
y = data["Energy"].values
model = LinearRegression(lr=0.002, n_iterations=20000)
model.fit(X, y)
w = model.weights[0]
b = model.bias
```
```test
import numpy as np
assert np.shape(X) == (1000, 1), f"X skal ha shape (1000, 1), men har {np.shape(X)}. Bruk doble klammer."
assert isinstance(model, LinearRegression), "model skal være LinearRegression(lr=0.002, n_iterations=20000)."
assert model.weights is not None, "Modellen er ikke trent. Kall model.fit(X, y)."
assert w is not ... and b is not ..., "Lagre w = model.weights[0] og b = model.bias."
w_e, b_e = np.polyfit(data["Net_Activity"], data["Energy"], deg=1)
assert abs(float(np.ravel(w)[0]) - w_e) < 0.01, f"w = {w} skulle vært omtrent {w_e:.3f}."
assert abs(float(b) - b_e) < 0.05, f"b = {b} skulle vært omtrent {b_e:.3f}. Har du brukt n_iterations=20000?"
```
:::

::: oppgave Eksakt løsning med polyfit
`np.polyfit(x, y, deg=1)` gir den eksakte minste kvadraters-linjen som `[stigning, konstantledd]`. Lagre dem i `w_eksakt` og `b_eksakt`.
```start
import numpy as np
import pandas as pd

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values
y = data["Energy"].values
w_eksakt, b_eksakt = ..., ...
```
```losning
import numpy as np
import pandas as pd

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values
y = data["Energy"].values
w_eksakt, b_eksakt = np.polyfit(x, y, deg=1)
```
```test
import numpy as np
assert w_eksakt is not ... and b_eksakt is not ..., "Bruk w_eksakt, b_eksakt = np.polyfit(x, y, deg=1)."
assert abs(w_eksakt - 3.0087) < 1e-3 and abs(b_eksakt - 4.8917) < 1e-3, f"Fikk w = {w_eksakt}, b = {b_eksakt}. Rekkefølgen fra polyfit er stigning først, så konstantledd."
```
:::

## Residualer og forventningsrett estimator
Residualene er `y - y_hat`, og en forventningsrett (unbiased) modell har gjennomsnittlig residual nær 0 og ingen mønster når residualene plottes mot $x$.

::: oppgave Analyser residualene
Lag `residualer = y - y_hat` og `snitt` (gjennomsnittet av dem), og tegn fordelingen med `ax.hist(residualer, bins=30)`.
```start
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values
y = data["Energy"].values
w, b = np.polyfit(x, y, deg=1)
y_hat = w * x + b

residualer = ...
snitt = ...
fig, ax = plt.subplots()
# tegn histogrammet her
```
```losning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values
y = data["Energy"].values
w, b = np.polyfit(x, y, deg=1)
y_hat = w * x + b

residualer = y - y_hat
snitt = residualer.mean()
fig, ax = plt.subplots()
ax.hist(residualer, bins=30)
ax.set_xlabel("y - ŷ")
plt.show()
```
```test
import numpy as np
assert residualer is not ... and snitt is not ..., "Lag både residualer og snitt."
assert np.allclose(residualer, y - y_hat), "residualer skal være y - y_hat (i den rekkefølgen)."
assert abs(snitt) < 1e-6, f"snitt skal være gjennomsnittet av residualene (nær 0), men er {snitt}."
assert len(ax.patches) == 30, f"Histogrammet skal ha 30 søyler, men har {len(ax.patches)}. Bruk ax.hist(residualer, bins=30)."
```
:::

--- LEKSJON 5.6 | sigmoid | Sigmoid og logistisk regresjon
Logistisk regresjon gir sannsynligheten $p = \sigma(Xw + b)$ for klasse 1, der sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ klemmer alle tall inn i $(0, 1)$ og $\sigma(0) = 0.5$.

::: oppgave Sigmoid for hånd
Regn ut `p = 1 / (1 + np.exp(-z))` for arrayen `z`.
```start
import numpy as np

z = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
p = ...
```
```losning
import numpy as np

z = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
p = 1 / (1 + np.exp(-z))
```
```test
import numpy as np
assert p is not ..., "Regn ut p = 1 / (1 + np.exp(-z))."
assert np.allclose(p, [0.00669285, 0.26894142, 0.5, 0.73105858, 0.99330715]), f"Feil verdier: {p}. Husk minus: np.exp(-z)."
```
:::

## Numerisk stabil sigmoid
`np.exp(-z)` renner over for `z = -1000` og gir `RuntimeWarning`, så klipp først med `np.clip(z, -500, 500)`.

::: oppgave Stabil sigmoid
Lag funksjonen `sigmoid(z)` som virker på numpy-arrays og bruker `np.clip` slik at den ikke gir overflow-advarsler for verdier som `-1000`.
```start
import numpy as np

def sigmoid(z):
    ...
```
```losning
import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))
```
```test
import numpy as np
import warnings
r = sigmoid(np.array([-2.0, 0.0, 2.0]))
assert r is not None, "Funksjonen returnerer ingenting. Husk return."
assert np.allclose(r, [0.11920292, 0.5, 0.88079708]), f"Feil verdier: {r}. Formelen er 1 / (1 + np.exp(-z))."
with warnings.catch_warnings(record=True) as advarsler:
    warnings.simplefilter("always")
    r2 = sigmoid(np.array([-1000.0, 1000.0]))
assert not any(issubclass(a.category, RuntimeWarning) for a in advarsler), "Du får overflow-advarsel for z = -1000. Klipp z med np.clip(z, -500, 500) først."
assert np.allclose(r2, [0.0, 1.0]), f"sigmoid([-1000, 1000]) skal gi [0, 1], fikk {r2}."
```
:::

## Fra sannsynlighet til klasse
`(p > 0.5).astype(int)` gjør sannsynligheter om til klasser 0/1, og notatenes `[1 if _y > 0.5 else 0 for _y in p]` gir det samme som en liste.

::: oppgave Sannsynligheter og klasser
Regn ut `z = np.matmul(X, w) + b`, sannsynlighetene `y_pred = sigmoid(z)` og klassene `klasser` (numpy-array med 0/1, terskel 0.5).
```start
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

X = np.array([[1.0, 2.0], [-1.0, 0.5], [3.0, -1.0], [0.0, 0.0]])
w = np.array([0.8, -1.2])
b = 0.1

z = ...
y_pred = ...
klasser = ...
```
```losning
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

X = np.array([[1.0, 2.0], [-1.0, 0.5], [3.0, -1.0], [0.0, 0.0]])
w = np.array([0.8, -1.2])
b = 0.1

z = np.matmul(X, w) + b
y_pred = sigmoid(z)
klasser = (y_pred > 0.5).astype(int)
```
```test
import numpy as np
assert z is not ... and y_pred is not ... and klasser is not ..., "Lag z, y_pred og klasser."
assert np.allclose(z, [-1.5, -1.3, 3.7, 0.1]), f"z = {z} er feil. Bruk np.matmul(X, w) + b."
assert np.all((np.asarray(y_pred) > 0) & (np.asarray(y_pred) < 1)), "y_pred skal være sannsynligheter mellom 0 og 1, altså sigmoid(z), ikke z."
assert np.allclose(y_pred, 1 / (1 + np.exp(-z))), "y_pred skal være sigmoid(z)."
assert isinstance(klasser, np.ndarray), "klasser skal være en numpy-array. Bruk (y_pred > 0.5).astype(int)."
assert list(klasser) == [0, 0, 1, 1], f"klasser = {klasser}, forventet [0, 0, 1, 1]."
```
:::

::: oppgave Notatenes versjon
Lag den samme klasselisten som en vanlig Python-liste `pred` med en list comprehension, slik forelesningsnotatene gjør.
```start
import numpy as np

y_pred = np.array([0.18, 0.21, 0.98, 0.52])
pred = ...
```
```losning
import numpy as np

y_pred = np.array([0.18, 0.21, 0.98, 0.52])
pred = [1 if _y > 0.5 else 0 for _y in y_pred]
```
```test
assert isinstance(pred, list), "pred skal være en liste. Bruk [1 if _y > 0.5 else 0 for _y in y_pred]."
assert pred == [0, 0, 1, 1], f"pred = {pred}, forventet [0, 0, 1, 1]."
```
:::

--- LEKSJON 5.7 | cross-entropy | Cross-entropy og LogisticRegression-klassen
Klassifisering bruker binær cross-entropy, `-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))`, der `np.clip(p, 1e-15, 1 - 1e-15)` hindrer `log(0)`.

::: oppgave Skriv bce
Lag funksjonen `bce(y, p)` som klipper `p` med `np.clip(p, 1e-15, 1 - 1e-15)` og returnerer binær cross-entropy.
```start
import numpy as np

def bce(y, p):
    ...
```
```losning
import numpy as np

def bce(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
```
```test
import numpy as np
r = bce(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.2]))
assert r is not None, "Funksjonen returnerer ingenting. Husk return."
assert r > 0, f"Tapet skal være positivt, men er {r}. Husk minus foran np.mean."
assert abs(r - 0.164252) < 1e-5, f"Fikk {r}, forventet ≈ 0.1643. Sjekk formelen y·log(p) + (1 - y)·log(1 - p)."
r2 = bce(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
assert np.isfinite(r2), "bce gir inf når p er nøyaktig 0 eller 1. Klipp p med np.clip først."
```
:::

::: oppgave Selvsikre feil straffes hardt
Bruk `bce` til å lagre tapet for de tre modellene i `tap_god`, `tap_usikker` og `tap_feil`.
```start
import numpy as np

def bce(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

y = np.array([1, 0, 1, 0])
p_god     = np.array([0.9, 0.1, 0.8, 0.2])
p_usikker = np.array([0.5, 0.5, 0.5, 0.5])
p_feil    = np.array([0.01, 0.99, 0.01, 0.99])

tap_god = ...
tap_usikker = ...
tap_feil = ...
```
```losning
import numpy as np

def bce(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

y = np.array([1, 0, 1, 0])
p_god     = np.array([0.9, 0.1, 0.8, 0.2])
p_usikker = np.array([0.5, 0.5, 0.5, 0.5])
p_feil    = np.array([0.01, 0.99, 0.01, 0.99])

tap_god = bce(y, p_god)
tap_usikker = bce(y, p_usikker)
tap_feil = bce(y, p_feil)
```
```test
import numpy as np
assert tap_god is not ... and tap_usikker is not ... and tap_feil is not ..., "Lag alle tre tapene med bce(y, p_...)."
assert abs(tap_usikker - np.log(2)) < 1e-9, f"tap_usikker skal være ln 2 ≈ 0.693, men er {tap_usikker}."
assert abs(tap_feil - (-np.log(0.01))) < 1e-9, f"tap_feil skal være -ln 0.01 ≈ 4.605, men er {tap_feil}."
assert tap_god < tap_usikker < tap_feil, "Rekkefølgen skal være tap_god < tap_usikker < tap_feil."
```
:::

## Gradienten får samme form
Gjennom sigmoid forenkles gradienten til `dw = X.T @ (p - y) / n` og `db = np.mean(p - y)`: samme form som for MSE, men uten faktoren 2 og med `p` i stedet for `y_pred`.

::: oppgave Gradientene for cross-entropy
Regn ut `p`, `dw` og `db` for logistisk regresjon på dataene under.
```start
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

rng = np.random.default_rng(0)
X = rng.normal(size=(30, 2))
y = (X[:, 0] + X[:, 1] > 0).astype(float)
w, b = np.array([0.2, -0.4]), 0.1

p = ...
dw = ...
db = ...
```
```losning
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

rng = np.random.default_rng(0)
X = rng.normal(size=(30, 2))
y = (X[:, 0] + X[:, 1] > 0).astype(float)
w, b = np.array([0.2, -0.4]), 0.1

p = sigmoid(X @ w + b)
dw = X.T @ (p - y) / len(y)
db = np.mean(p - y)
```
```test
import numpy as np
assert p is not ... and dw is not ... and db is not ..., "Lag p, dw og db."
p_f = 1 / (1 + np.exp(-(X @ w + b)))
dw_f = X.T @ (p_f - y) / len(y)
assert np.allclose(p, p_f), "p skal være sigmoid(X @ w + b)."
assert np.shape(dw) == (2,), f"dw skal ha shape (2,), men har {np.shape(dw)}. Bruk X.T @ (p - y), ikke X @ (p - y)."
assert not np.allclose(dw, 2 * dw_f), "dw er dobbelt så stor som den skal. For cross-entropy er det ingen faktor 2."
assert np.allclose(dw, dw_f), f"dw = {dw} er feil. Det skal være X.T @ (p - y) / n."
assert abs(float(db) - np.mean(p_f - y)) < 1e-9, f"db = {db} er feil. Det skal være np.mean(p - y)."
```
:::

## Hele klassen
`LogisticRegression` er som `LinearRegression`, men med sigmoid rundt `X @ self.weights + self.bias`, cross-entropy i `loss_history`, `predict_proba` for sannsynligheter og `predict` for 0/1.

::: oppgave Fullfør LogisticRegression
Fyll inn cross-entropy-tapet (med `p_c`), `dw`, `db` og `predict` (terskel 0.5, returner 0/1).
```start
import numpy as np

class LogisticRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights, self.bias = None, None
        self.lr, self.n_iterations = lr, n_iterations
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights, self.bias = np.zeros(n_features), 0.0
        for _ in range(self.n_iterations):
            p = self.sigmoid(X @ self.weights + self.bias)
            p_c = np.clip(p, 1e-15, 1 - 1e-15)
            self.loss_history.append(...)
            dw = ...
            db = ...
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        return self.sigmoid(X @ self.weights + self.bias)

    def predict(self, X):
        ...
```
```losning
import numpy as np

class LogisticRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights, self.bias = None, None
        self.lr, self.n_iterations = lr, n_iterations
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights, self.bias = np.zeros(n_features), 0.0
        for _ in range(self.n_iterations):
            p = self.sigmoid(X @ self.weights + self.bias)
            p_c = np.clip(p, 1e-15, 1 - 1e-15)
            self.loss_history.append(-np.mean(y * np.log(p_c) + (1 - y) * np.log(1 - p_c)))
            dw = X.T @ (p - y) / n_samples
            db = np.mean(p - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        return self.sigmoid(X @ self.weights + self.bias)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)
```
```test
import numpy as np
rng_t = np.random.default_rng(5)
X_t = rng_t.normal(size=(200, 2))
y_t = (2 * X_t[:, 0] - X_t[:, 1] + 0.5 > 0).astype(float)
m_t = LogisticRegression(lr=0.5, n_iterations=300)
m_t.fit(X_t, y_t)
assert len(m_t.loss_history) == 300, "loss_history skal få ett tap per iterasjon."
assert all(isinstance(v, (float, np.floating)) for v in m_t.loss_history), "loss_history skal inneholde tall. Fyll inn cross-entropy med p_c."
assert abs(m_t.loss_history[0] - np.log(2)) < 1e-6, f"Første tap skal være ln 2 ≈ 0.693 (alle p = 0.5), men er {m_t.loss_history[0]}. Sjekk formelen og minustegnet."
assert m_t.loss_history[-1] < m_t.loss_history[0], "Tapet synker ikke. Sjekk dw og db."
assert np.shape(m_t.weights) == (2,), f"weights skal ha shape (2,), men har {np.shape(m_t.weights)}."
assert m_t.weights[0] > 0 and m_t.weights[1] < 0, f"weights = {m_t.weights} har feil fortegn. Gradienten er X.T @ (p - y) / n."
pr = m_t.predict(X_t)
assert pr is not None, "predict returnerer ingenting. Husk return."
pr = np.asarray(pr)
assert set(np.unique(pr)) <= {0, 1}, "predict skal returnere klasser 0 og 1, ikke sannsynligheter."
assert np.mean(pr == y_t) > 0.95, f"Accuracy er bare {np.mean(pr == y_t):.2f}. Sjekk dw, db og terskelen."
```
:::

## Notatenes versjon
Notatene skriver `np.matmul(self.weights, x.transpose())`, som gir det samme som `np.matmul(x, self.weights)` når `weights` er 1D. `np.mean(y_true == pred)` krever at minst én side er en numpy-array, for to lister sammenlignes som helhet og gir én `True`/`False`.

::: oppgave To skrivemåter
Regn ut den lineære delen på begge måtene: `a` med `np.matmul(weights, x.transpose()) + bias` og `c` med `np.matmul(x, weights) + bias`.
```start
import numpy as np

x = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.5]])
weights = np.array([0.4, -0.3])
bias = 0.5

a = ...
c = ...
```
```losning
import numpy as np

x = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.5]])
weights = np.array([0.4, -0.3])
bias = 0.5

a = np.matmul(weights, x.transpose()) + bias
c = np.matmul(x, weights) + bias
```
```test
import numpy as np
assert a is not ... and c is not ..., "Lag både a og c."
assert np.shape(a) == (3,) and np.shape(c) == (3,), f"Begge skal ha shape (3,), men har {np.shape(a)} og {np.shape(c)}."
assert np.allclose(a, [0.3, 2.0, 0.55]) and np.allclose(c, a), f"a = {a}, c = {c}. Begge skal være [0.3, 2.0, 0.55]."
```
:::

::: oppgave Rett accuracy-feilen
Koden gir accuracy 0.0 selv om tre av fire er riktige. Rett linjen slik at `acc` blir 0.75.
```start
import numpy as np

y_true = [1, 0, 1, 1]
pred   = [1, 0, 0, 1]
acc = np.mean(y_true == pred)
```
```losning
import numpy as np

y_true = [1, 0, 1, 1]
pred   = [1, 0, 0, 1]
acc = np.mean(np.array(y_true) == np.array(pred))
```
```test
assert abs(acc - 0.75) < 1e-9, f"acc = {acc}. To lister sammenlignes som helhet; gjør minst én av dem til np.array først."
```
:::

--- LEKSJON 5.8 | feature-engineering | Feature engineering (XOR)
En lineær modell gir en rett beslutningsgrense $w_0 x_0 + w_1 x_1 + b = 0$ (modellens **induktive bias**), så for et XOR-mønster må vi lage nye features (**feature engineering**).

::: oppgave Del i train og test
`classification.csv` har kolonnene `x0`, `x1`, `y` og `split`. Lag `train` og `test` med radene der `split` er `"train"` og `"test"`.
```start
import pandas as pd

data = pd.read_csv("classification.csv")
train = ...
test = ...
```
```losning
import pandas as pd

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
```
```test
assert train is not ... and test is not ..., "Lag både train og test med en boolsk maske på split."
assert len(train) == 500 and len(test) == 500, f"Forventet 500 rader i hver, fikk {len(train)} og {len(test)}."
assert set(train["split"]) == {"train"} and set(test["split"]) == {"test"}, "train skal bare ha split == 'train', og test bare 'test'."
```
:::

## Først uten feature engineering
Med bare `x0` og `x1` kan ingen rett linje skille klassene, så accuracy havner rundt 0.5.

::: oppgave Prøv uten nye features
Lag `X_train` og `X_test` med kolonnene `x0` og `x1`, tren modellen, og lagre test-accuracy i `acc_raa`.
```start
import numpy as np
import pandas as pd

class LogisticRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights, self.bias = None, None
        self.lr, self.n_iterations = lr, n_iterations
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    def fit(self, X, y):
        self.weights, self.bias = np.zeros(X.shape[1]), 0.0
        for _ in range(self.n_iterations):
            p = self.sigmoid(X @ self.weights + self.bias)
            self.weights -= self.lr * X.T @ (p - y) / len(y)
            self.bias -= self.lr * np.mean(p - y)
    def predict(self, X):
        return (self.sigmoid(X @ self.weights + self.bias) > 0.5).astype(int)

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
y_train, y_test = train["y"].values, test["y"].values

X_train = ...
X_test = ...
model = LogisticRegression(lr=0.1, n_iterations=1000)
acc_raa = ...
```
```losning
import numpy as np
import pandas as pd

class LogisticRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights, self.bias = None, None
        self.lr, self.n_iterations = lr, n_iterations
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    def fit(self, X, y):
        self.weights, self.bias = np.zeros(X.shape[1]), 0.0
        for _ in range(self.n_iterations):
            p = self.sigmoid(X @ self.weights + self.bias)
            self.weights -= self.lr * X.T @ (p - y) / len(y)
            self.bias -= self.lr * np.mean(p - y)
    def predict(self, X):
        return (self.sigmoid(X @ self.weights + self.bias) > 0.5).astype(int)

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
y_train, y_test = train["y"].values, test["y"].values

X_train = train[["x0", "x1"]].values
X_test = test[["x0", "x1"]].values
model = LogisticRegression(lr=0.1, n_iterations=1000)
model.fit(X_train, y_train)
acc_raa = np.mean(model.predict(X_test) == y_test)
```
```test
import numpy as np
assert np.shape(X_train) == (500, 2) and np.shape(X_test) == (500, 2), f"X_train og X_test skal ha shape (500, 2), men har {np.shape(X_train)} og {np.shape(X_test)}."
assert model.weights is not None, "Modellen er ikke trent. Kall model.fit(X_train, y_train)."
assert acc_raa is not ..., "Regn ut acc_raa = np.mean(model.predict(X_test) == y_test)."
assert 0.3 < acc_raa < 0.7, f"acc_raa = {acc_raa:.3f}. Med bare x0 og x1 skal den være rundt 0.5. Har du testet på X_test?"
```
:::

## Den nye featuren: x0 · x1
`np.column_stack([df["x0"], df["x1"], df["x0"] * df["x1"]])` legger til produktet, som er negativt i 2. og 4. kvadrant og positivt i 1. og 3.

::: oppgave Skriv lag_features
Lag funksjonen `lag_features(df)` som returnerer en numpy-matrise med kolonnene `x0`, `x1` og `x0 * x1`.
```start
import numpy as np
import pandas as pd

def lag_features(df):
    ...
```
```losning
import numpy as np
import pandas as pd

def lag_features(df):
    return np.column_stack([df["x0"], df["x1"], df["x0"] * df["x1"]])
```
```test
import numpy as np
import pandas as pd
df_t = pd.DataFrame({"x0": [1.0, -2.0], "x1": [3.0, 4.0], "y": [0, 1]})
r = lag_features(df_t)
assert r is not None, "Funksjonen returnerer ingenting. Husk return."
r = np.asarray(r)
assert r.shape == (2, 3), f"Resultatet skal ha shape (2, 3), én rad per datapunkt og tre kolonner, men har {r.shape}."
assert np.allclose(r, [[1, 3, 3], [-2, 4, -8]]), f"Fikk {r.tolist()}, forventet [[1, 3, 3], [-2, 4, -8]]."
```
:::

::: oppgave Nå 0.88
Lag `X_train_fe` og `X_test_fe` med de tre kolonnene `x0`, `x1` og `x0 * x1`, tren modellen, og lagre test-accuracy i `acc` (prosjektkravet er 0.88).
```start
import numpy as np
import pandas as pd

class LogisticRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights, self.bias = None, None
        self.lr, self.n_iterations = lr, n_iterations
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    def fit(self, X, y):
        self.weights, self.bias = np.zeros(X.shape[1]), 0.0
        for _ in range(self.n_iterations):
            p = self.sigmoid(X @ self.weights + self.bias)
            self.weights -= self.lr * X.T @ (p - y) / len(y)
            self.bias -= self.lr * np.mean(p - y)
    def predict(self, X):
        return (self.sigmoid(X @ self.weights + self.bias) > 0.5).astype(int)

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
y_train, y_test = train["y"].values, test["y"].values

X_train_fe = ...
X_test_fe = ...

model = LogisticRegression(lr=0.1, n_iterations=1000)
# tren og regn ut acc
acc = ...
```
```losning
import numpy as np
import pandas as pd

class LogisticRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights, self.bias = None, None
        self.lr, self.n_iterations = lr, n_iterations
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    def fit(self, X, y):
        self.weights, self.bias = np.zeros(X.shape[1]), 0.0
        for _ in range(self.n_iterations):
            p = self.sigmoid(X @ self.weights + self.bias)
            self.weights -= self.lr * X.T @ (p - y) / len(y)
            self.bias -= self.lr * np.mean(p - y)
    def predict(self, X):
        return (self.sigmoid(X @ self.weights + self.bias) > 0.5).astype(int)

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
y_train, y_test = train["y"].values, test["y"].values

X_train_fe = np.column_stack([train["x0"], train["x1"], train["x0"] * train["x1"]])
X_test_fe = np.column_stack([test["x0"], test["x1"], test["x0"] * test["x1"]])

model = LogisticRegression(lr=0.1, n_iterations=1000)
model.fit(X_train_fe, y_train)
acc = np.mean(model.predict(X_test_fe) == y_test)
```
```test
import numpy as np
assert np.shape(X_train_fe) == (500, 3), f"X_train_fe skal ha shape (500, 3), men har {np.shape(X_train_fe)}."
assert np.shape(X_test_fe) == (500, 3), f"X_test_fe skal ha shape (500, 3), men har {np.shape(X_test_fe)}."
assert np.allclose(np.asarray(X_test_fe)[:, 2], test["x0"].values * test["x1"].values), "Tredje kolonne i X_test_fe skal være x0 * x1 for testdataene."
assert isinstance(acc, (float, np.floating)), "acc skal være et tall, f.eks. np.mean(model.predict(X_test_fe) == y_test)."
assert acc >= 0.88, f"Accuracy er {acc:.3f}, kravet er 0.88. Har du trent på X_train_fe og testet på X_test_fe?"
```
:::

::: oppgave Tegn den krumme grensen
Lag et rutenett med `g0, g1 = np.meshgrid(akse, akse)`, gjør det om til `G` med de tre featurene, regn ut sannsynlighetene `P` med shape `(150, 150)`, og tegn med `ax.contourf(g0, g1, P, levels=20, cmap="coolwarm")`.
```start
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
X = np.column_stack([train["x0"], train["x1"], train["x0"] * train["x1"]])
y = train["y"].values
w, b = np.zeros(3), 0.0
for _ in range(1000):
    p = sigmoid(X @ w + b)
    w -= 0.1 * X.T @ (p - y) / len(y)
    b -= 0.1 * np.mean(p - y)

akse = np.linspace(-3.5, 3.5, 150)
g0, g1 = ...
G = ...
P = ...
fig, ax = plt.subplots()
# tegn her
ax.scatter(train["x0"], train["x1"], c=y, cmap="coolwarm", s=8)
```
```losning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
X = np.column_stack([train["x0"], train["x1"], train["x0"] * train["x1"]])
y = train["y"].values
w, b = np.zeros(3), 0.0
for _ in range(1000):
    p = sigmoid(X @ w + b)
    w -= 0.1 * X.T @ (p - y) / len(y)
    b -= 0.1 * np.mean(p - y)

akse = np.linspace(-3.5, 3.5, 150)
g0, g1 = np.meshgrid(akse, akse)
G = np.column_stack([g0.ravel(), g1.ravel(), g0.ravel() * g1.ravel()])
P = sigmoid(G @ w + b).reshape(g0.shape)
fig, ax = plt.subplots()
ax.contourf(g0, g1, P, levels=20, cmap="coolwarm", alpha=0.6)
ax.scatter(train["x0"], train["x1"], c=y, cmap="coolwarm", s=8)
plt.show()
```
```test
import numpy as np
assert np.shape(g0) == (150, 150), "g0 og g1 skal komme fra np.meshgrid(akse, akse)."
assert np.shape(G) == (22500, 3), f"G skal ha shape (22500, 3): ravel() av g0, g1 og g0 * g1 som kolonner. Har {np.shape(G)}."
assert np.shape(P) == (150, 150), f"P skal ha shape (150, 150). Bruk .reshape(g0.shape). Har {np.shape(P)}."
assert np.allclose(P, sigmoid(np.column_stack([g0.ravel(), g1.ravel(), g0.ravel() * g1.ravel()]) @ w + b).reshape(150, 150)), "P skal være sigmoid(G @ w + b) omformet til rutenettet."
assert len(ax.collections) >= 2, "Fant ikke konturplottet. Bruk ax.contourf(g0, g1, P, levels=20, cmap='coolwarm')."
```
:::

--- LEKSJON 5.9 | skalering | Skalering for gradient descent
Når en feature har store verdier (som `Net_Activity` rundt 20), må `lr` være liten for å unngå divergens, og da lærer `b` altfor sakte.

::: oppgave Se problemet
Fyll inn ett gradient descent-steg i løkka (`w -= lr * 2 * np.mean((y_pred - y) * x)` og tilsvarende for `b`), og se hva `resultat` viser for de tre læringsratene (fasit: `b` ≈ 4.89).
```start
import numpy as np
import pandas as pd

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values
y = data["Energy"].values

resultat = {}
for lr in [0.001, 0.002, 0.003]:
    w, b = 0.0, 0.0
    with np.errstate(all="ignore"):
        for _ in range(2000):
            y_pred = w * x + b
            ...
    resultat[lr] = (w, b)
```
```losning
import numpy as np
import pandas as pd

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values
y = data["Energy"].values

resultat = {}
for lr in [0.001, 0.002, 0.003]:
    w, b = 0.0, 0.0
    with np.errstate(all="ignore"):
        for _ in range(2000):
            y_pred = w * x + b
            w -= lr * 2 * np.mean((y_pred - y) * x)
            b -= lr * 2 * np.mean(y_pred - y)
    resultat[lr] = (w, b)
```
```test
import numpy as np
w1, b1 = resultat[0.001]
assert abs(w1 - 3.1706) < 0.01 and abs(b1 - 1.3855) < 0.01, f"Med lr = 0.001 skulle du fått w ≈ 3.17 og b ≈ 1.39, men fikk w = {w1}, b = {b1}. Sjekk gradientene."
assert not np.isfinite(resultat[0.003][0]), "Med lr = 0.003 skal w divergere til nan eller inf."
```
:::

## Standardisering
`x_s = (x - mu) / sigma` med `mu = x.mean()` og `sigma = x.std()` gir snitt 0 og standardavvik 1, og da konvergerer gradient descent på noen få iterasjoner.

::: oppgave Standardiser x
Lag `mu`, `sigma` og den standardiserte `x_s`.
```start
import numpy as np
import pandas as pd

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values

mu = ...
sigma = ...
x_s = ...
```
```losning
import numpy as np
import pandas as pd

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values

mu = x.mean()
sigma = x.std()
x_s = (x - mu) / sigma
```
```test
import numpy as np
assert x_s is not ..., "Lag x_s = (x - mu) / sigma."
assert abs(np.mean(x_s)) < 1e-9, f"x_s skal ha snitt 0, men har {np.mean(x_s)}."
assert abs(np.std(x_s) - 1) < 1e-9, f"x_s skal ha standardavvik 1, men har {np.std(x_s)}. Del på x.std()."
```
:::

::: oppgave 100 iterasjoner holder
Tren `w_s` og `b_s` på `x_s` i 100 iterasjoner med `lr = 0.1`, og regn om til originale enheter med `w_orig = w_s / sigma` og `b_orig = b_s - w_s * mu / sigma`.
```start
import numpy as np
import pandas as pd

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values
y = data["Energy"].values
mu, sigma = x.mean(), x.std()
x_s = (x - mu) / sigma

w_s, b_s = 0.0, 0.0
for _ in range(100):
    ...

w_orig = ...
b_orig = ...
```
```losning
import numpy as np
import pandas as pd

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values
y = data["Energy"].values
mu, sigma = x.mean(), x.std()
x_s = (x - mu) / sigma

w_s, b_s = 0.0, 0.0
for _ in range(100):
    y_pred = w_s * x_s + b_s
    w_s -= 0.1 * 2 * np.mean((y_pred - y) * x_s)
    b_s -= 0.1 * 2 * np.mean(y_pred - y)

w_orig = w_s / sigma
b_orig = b_s - w_s * mu / sigma
```
```test
import numpy as np
w_e, b_e = np.polyfit(x, y, deg=1)
assert w_s != 0, "w_s er fortsatt 0. Fyll inn gradient descent-steget i løkka."
assert abs(w_s - w_e * sigma) < 1e-3 and abs(b_s - y.mean()) < 1e-3, f"w_s = {w_s}, b_s = {b_s} har ikke konvergert. Bruk x_s (ikke x) i løkka."
assert w_orig is not ... and b_orig is not ..., "Regn om til w_orig og b_orig."
assert abs(w_orig - w_e) < 1e-3 and abs(b_orig - b_e) < 1e-3, f"w_orig = {w_orig}, b_orig = {b_orig}, forventet {w_e:.3f} og {b_e:.3f}. Sjekk omregningsformlene."
```
:::

## Med flere features og train/test
`mu = X_train.mean(axis=0)` og `sigma = X_train.std(axis=0)` gir statistikk per kolonne, og de **samme** tallene fra treningsdataene brukes på `X_test`. Min-maks-skalering `(X - X_min) / (X_max - X_min)` følger samme regel.

::: oppgave Standardiser med treningsstatistikk
Lag `X_train_s` og `X_test_s` ved å standardisere hver kolonne med gjennomsnitt og standardavvik fra **treningsdataene**.
```start
import numpy as np

rng = np.random.default_rng(7)
X_train = rng.normal([100, -3], [20, 0.5], size=(60, 2))
X_test = rng.normal([110, -3], [20, 0.5], size=(15, 2))

X_train_s = ...
X_test_s = ...
```
```losning
import numpy as np

rng = np.random.default_rng(7)
X_train = rng.normal([100, -3], [20, 0.5], size=(60, 2))
X_test = rng.normal([110, -3], [20, 0.5], size=(15, 2))

mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
X_train_s = (X_train - mu) / sigma
X_test_s = (X_test - mu) / sigma
```
```test
import numpy as np
assert np.shape(X_train_s) == (60, 2) and np.shape(X_test_s) == (15, 2), "Formene skal være uendret: (60, 2) og (15, 2)."
assert np.allclose(np.mean(X_train_s, axis=0), 0) and np.allclose(np.std(X_train_s, axis=0), 1), "Hver kolonne i X_train_s skal ha snitt 0 og std 1. Bruk axis=0."
fasit = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)
assert not np.allclose(np.mean(X_test_s, axis=0), 0), "X_test_s har snitt 0. Da har du brukt testdataenes egen statistikk. Bruk mu og sigma fra X_train."
assert np.allclose(X_test_s, fasit), "X_test_s skal være (X_test - mu) / sigma med mu og sigma fra X_train."
```
:::

::: oppgave Min-maks-skalering
Lag `X_train_mm` og `X_test_mm` med min-maks-skalering per kolonne, der `X_min` og `X_max` kommer fra **treningsdataene**.
```start
import numpy as np

rng = np.random.default_rng(7)
X_train = rng.normal([100, -3], [20, 0.5], size=(60, 2))
X_test = rng.normal([110, -3], [20, 0.5], size=(15, 2))

X_train_mm = ...
X_test_mm = ...
```
```losning
import numpy as np

rng = np.random.default_rng(7)
X_train = rng.normal([100, -3], [20, 0.5], size=(60, 2))
X_test = rng.normal([110, -3], [20, 0.5], size=(15, 2))

X_min = X_train.min(axis=0)
X_max = X_train.max(axis=0)
X_train_mm = (X_train - X_min) / (X_max - X_min)
X_test_mm = (X_test - X_min) / (X_max - X_min)
```
```test
import numpy as np
assert np.shape(X_train_mm) == (60, 2) and np.shape(X_test_mm) == (15, 2), "Formene skal være uendret: (60, 2) og (15, 2)."
assert np.allclose(np.min(X_train_mm, axis=0), 0) and np.allclose(np.max(X_train_mm, axis=0), 1), "Hver kolonne i X_train_mm skal gå fra 0 til 1. Bruk min og max med axis=0."
mn, mx = X_train.min(axis=0), X_train.max(axis=0)
assert np.allclose(X_test_mm, (X_test - mn) / (mx - mn)), "X_test_mm skal bruke X_min og X_max fra X_train, ikke fra X_test."
```
:::

--- LEKSJON 5.10 | sgd-minibatch | SGD og mini-batch
En **epoke** er én runde gjennom alle N treningspunktene: full batch gir 1 oppdatering per epoke, SGD (ett punkt per steg) gir N, og mini-batch med batch size B gir `ceil(N / B)`.

::: oppgave Oppdateringer per epoke
Med `N = 1000` og `B = 32`, lag `steg_full`, `steg_sgd` og `steg_mini` (antall oppdateringer per epoke, siste batch kan være mindre).
```start
import math

N = 1000
B = 32
steg_full = ...
steg_sgd = ...
steg_mini = ...
```
```losning
import math

N = 1000
B = 32
steg_full = 1
steg_sgd = N
steg_mini = math.ceil(N / B)
```
```test
assert steg_full == 1, "Full batch tar ett steg per epoke."
assert steg_sgd == 1000, "SGD tar ett steg per datapunkt, altså N."
assert steg_mini == 32, f"steg_mini = {steg_mini}. 1000 / 32 = 31.25, og den siste, mindre batchen gir et steg til: 32. Bruk math.ceil."
```
:::

## Stokke og dele opp
`perm = rng.permutation(n)` stokker indeksene på nytt hver epoke, og `perm[start:start + batch_size]` for `start in range(0, n, batch_size)` gir batchene.

::: oppgave Lag mini-batcher
Lag `minibatch_indekser(n, batch_size, seed)` som stokker indeksene med `np.random.default_rng(seed).permutation(n)` og returnerer en **liste** med index-arrays på maks `batch_size` elementer.
```start
import numpy as np

def minibatch_indekser(n, batch_size, seed):
    rng = np.random.default_rng(seed)
    perm = ...
    batcher = []
    # fyll batcher
    return batcher
```
```losning
import numpy as np

def minibatch_indekser(n, batch_size, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    batcher = []
    for start in range(0, n, batch_size):
        batcher.append(perm[start:start + batch_size])
    return batcher
```
```test
import numpy as np
b_t = minibatch_indekser(100, 32, 0)
assert isinstance(b_t, list), "Funksjonen skal returnere en liste med arrays."
assert len(b_t) == 4, f"100 punkter med batch_size 32 gir 4 batcher (32, 32, 32, 4), men du har {len(b_t)}."
assert [len(x) for x in b_t] == [32, 32, 32, 4], f"Batch-størrelsene er {[len(x) for x in b_t]}, forventet [32, 32, 32, 4]."
alle = np.concatenate(b_t)
assert np.array_equal(np.sort(alle), np.arange(100)), "Hver indeks 0..99 skal være med nøyaktig én gang."
assert not np.array_equal(alle, np.arange(100)), "Indeksene er ikke stokket. Bruk rng.permutation(n)."
```
:::

## Mini-batch gradient descent
Den indre løkka tar ett steg per batch med `Xb, yb = X[idx], y[idx]` og `len(yb)` i stedet for `n`. Batch-gradienten er støyete, så følg tapet på hele datasettet én gang per epoke.

::: oppgave Skriv minibatch_gd
Fullfør den indre løkka i `minibatch_gd`: hent batchen, regn ut gradientene på den og oppdater `w` og `b`.
```start
import numpy as np
import pandas as pd

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values
X = ((x - x.mean()) / x.std()).reshape(-1, 1)
y = data["Energy"].values

def minibatch_gd(X, y, batch_size, lr=0.05, n_epochs=20, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    w, b, history = np.zeros(X.shape[1]), 0.0, []
    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            ...
        history.append(np.mean((X @ w + b - y) ** 2))
    return w, b, history
```
```losning
import numpy as np
import pandas as pd

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values
X = ((x - x.mean()) / x.std()).reshape(-1, 1)
y = data["Energy"].values

def minibatch_gd(X, y, batch_size, lr=0.05, n_epochs=20, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    w, b, history = np.zeros(X.shape[1]), 0.0, []
    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            Xb, yb = X[idx], y[idx]
            y_pred = Xb @ w + b
            w = w - lr * (2 / len(yb)) * Xb.T @ (y_pred - yb)
            b = b - lr * (2 / len(yb)) * np.sum(y_pred - yb)
        history.append(np.mean((X @ w + b - y) ** 2))
    return w, b, history
```
```test
import numpy as np
w_t, b_t, h_t = minibatch_gd(X, y, 32)
assert len(h_t) == 20, f"history skal ha ett tap per epoke (20), men har {len(h_t)}."
w_e = np.polyfit(X[:, 0], y, deg=1)[0]
assert abs(float(np.ravel(w_t)[0]) - w_e) < 0.3, f"w = {w_t} skulle vært nær {w_e:.2f}. Bruk gradienten på batchen (Xb, yb)."
assert abs(b_t - y.mean()) < 0.3, f"b = {b_t} skulle vært nær {y.mean():.2f}. Sjekk oppdateringen av b."
X_k = np.array([[1.0], [2.0], [3.0], [4.0]])
y_k = np.array([2.0, 4.0, 6.0, 8.0])
w_k, b_k, _ = minibatch_gd(X_k, y_k, 2, lr=0.01, n_epochs=1)
w_full, b_full, _ = minibatch_gd(X_k, y_k, 4, lr=0.01, n_epochs=1)
assert not np.allclose(w_k, w_full), "Med batch_size 2 skal det tas to steg per epoke. Oppdaterer du w og b inne i den indre løkka?"
```
:::

::: oppgave Full batch mot mini-batch
Kjør `minibatch_gd` med `batch_size` 1000 (full batch) og 32, og lagre det siste tapet i `history` i `tap_full` og `tap_mini`.
```start
import numpy as np
import pandas as pd

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values
X = ((x - x.mean()) / x.std()).reshape(-1, 1)
y = data["Energy"].values

def minibatch_gd(X, y, batch_size, lr=0.05, n_epochs=20, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    w, b, history = np.zeros(X.shape[1]), 0.0, []
    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            Xb, yb = X[idx], y[idx]
            y_pred = Xb @ w + b
            w = w - lr * (2 / len(yb)) * Xb.T @ (y_pred - yb)
            b = b - lr * (2 / len(yb)) * np.sum(y_pred - yb)
        history.append(np.mean((X @ w + b - y) ** 2))
    return w, b, history

tap_full = ...
tap_mini = ...
```
```losning
import numpy as np
import pandas as pd

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values
X = ((x - x.mean()) / x.std()).reshape(-1, 1)
y = data["Energy"].values

def minibatch_gd(X, y, batch_size, lr=0.05, n_epochs=20, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    w, b, history = np.zeros(X.shape[1]), 0.0, []
    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            Xb, yb = X[idx], y[idx]
            y_pred = Xb @ w + b
            w = w - lr * (2 / len(yb)) * Xb.T @ (y_pred - yb)
            b = b - lr * (2 / len(yb)) * np.sum(y_pred - yb)
        history.append(np.mean((X @ w + b - y) ** 2))
    return w, b, history

tap_full = minibatch_gd(X, y, 1000)[2][-1]
tap_mini = minibatch_gd(X, y, 32)[2][-1]
```
```test
import numpy as np
assert tap_full is not ... and tap_mini is not ..., "Lag både tap_full og tap_mini."
assert abs(tap_full - minibatch_gd(X, y, 1000)[2][-1]) < 1e-9, "tap_full skal være siste element i history med batch_size=1000. history er tredje returverdi: [2][-1]."
assert abs(tap_mini - minibatch_gd(X, y, 32)[2][-1]) < 1e-9, "tap_mini skal være siste element i history med batch_size=32."
assert tap_mini < tap_full, "Mini-batch skal ha lavere tap etter 20 epoker, fordi den tar mange flere steg."
```
:::
