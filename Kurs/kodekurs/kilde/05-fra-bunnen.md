=== DEL 5 | fra-bunnen | Maskinlæring fra bunnen av (prosjekt 1)
Her bygger du lineær og logistisk regresjon helt fra bunnen med numpy: modell, tapsfunksjon, gradienter og gradient descent. Til slutt kan du skrive hele `linear_models.py` fra prosjekt 1 selv.

--- LEKSJON 5.1 | modell-prediksjon | Modell og prediksjon
En **modell** er en funksjon som gjør input om til en prediksjon. Den enkleste er en rett linje:

$$\hat y = w \cdot x + b$$

Her er $x$ en **feature** (det vi vet), $\hat y$ er **prediksjonen** (det modellen gjetter), og $w$ (vekt, weight) og $b$ (bias) er **parametrene**. Å «trene» modellen betyr å finne de $w$ og $b$ som gjør prediksjonene best mulig.

```python
w = 3.0
b = 5.0
x = 20.0

y_hat = w * x + b
print("Prediksjon:", y_hat)
```

## Mange datapunkter på én gang
Med numpy kan vi predikere for alle datapunktene samtidig. I prosjekt 1 skal du predikere strømforbruk (`Energy`) fra nettverksaktivitet (`Net_Activity`). Prøv å endre `w` og `b` til linjen passer bedre med punktene:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values
y = data["Energy"].values

w, b = 2.0, 20.0              # endre disse!
y_hat = w * x + b              # én prediksjon per datapunkt

plt.scatter(x, y, s=8, label="Data")
plt.plot(x, y_hat, color="red", label=f"y = {w}·x + {b}")
plt.xlabel("Net_Activity")
plt.ylabel("Energy")
plt.legend()
plt.show()
```

## Flere features: matrisen X
Som regel har vi flere features. Da samler vi dem i en matrise `X` med **én rad per datapunkt** og **én kolonne per feature**, altså shape `(m, n)`. Hver feature får sin egen vekt, så `w` har shape `(n,)`. Prediksjonen blir en vektet sum per rad:

$$\hat y = Xw + b$$

```python
import numpy as np

X = np.array([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0]])     # 3 datapunkter, 2 features
w = np.array([0.5, -1.0])      # én vekt per feature
b = 2.0

y_hat = X @ w + b              # samme som np.matmul(X, w) + b
print(X.shape, w.shape, "->", y_hat.shape)
print(y_hat)
print(np.matmul(X, w) + b)
```

Første rad gir $0.5 \cdot 1 + (-1) \cdot 2 + 2 = 0.5$. Bias `b` er ett tall som legges til alle radene (broadcasting). Se `X`, `w` og `y_hat` i variabelpanelet under resultatet: tre rader inn, tre prediksjoner ut.

::: advarsel X må være 2D
Klassene i prosjekt 1 forventer `X` med shape `(m, n)`, også når det bare er én feature. `data["Net_Activity"].values` gir shape `(1000,)`, mens `data[["Net_Activity"]].values` (doble klammer) gir `(1000, 1)`.

```python
import pandas as pd
data = pd.read_csv("regression.csv")
print(data["Net_Activity"].values.shape)
print(data[["Net_Activity"]].values.shape)
```
:::

::: oppgave Skriv predict-funksjonen
Lag funksjonen `predict(X, w, b)` som returnerer prediksjonene $\hat y = Xw + b$ for alle radene i `X`.
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

::: quiz
`X` har shape `(100, 3)` og `w` har shape `(3,)`. Hvilken shape har `y_hat`?
```kode
y_hat = np.matmul(X, w) + b
```
- [ ] `(3,)`
- [x] `(100,)`
- [ ] `(100, 3)`
- [ ] Koden gir feil fordi shapene ikke passer
? Matrise ganger vektor: `(100, 3) @ (3,)` gir én verdi per rad, altså `(100,)`. Én prediksjon per datapunkt.
:::

::: oppsummering
- En lineær modell er $\hat y = Xw + b$, der `w` og `b` er **parametrene**.
- `X` har shape `(m, n)`: m datapunkter (rader), n features (kolonner). `w` har shape `(n,)`.
- `X @ w + b` (eller `np.matmul(X, w) + b`) gir én prediksjon per rad.
- Bruk `df[["kolonne"]].values` for å få en 2D-matrise med én feature.
:::

--- LEKSJON 5.2 | mse | Tapsfunksjon: MSE
For å forbedre modellen må vi kunne **måle hvor feil den tar**. Et slikt mål kalles en **tapsfunksjon** (loss). For regresjon bruker vi nesten alltid gjennomsnittlig kvadratisk feil, **MSE** (mean squared error):

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} (\hat y_i - y_i)^2$$

Vi tar feilen for hvert punkt, kvadrerer den og tar gjennomsnittet. Jo lavere, jo bedre.

```python
import numpy as np
import pandas as pd

y      = np.array([3.0, 5.0, 7.0])
y_pred = np.array([2.5, 5.0, 9.0])

feil = y_pred - y
print("Feil:", feil)
print("Kvadrert:", feil ** 2)
display(pd.DataFrame({"y": y, "y_pred": y_pred, "feil": feil, "feil²": feil ** 2}))
print("MSE:", np.mean(feil ** 2))
```

Tabellen har én rad per datapunkt. MSE er gjennomsnittet av den siste kolonnen: det tredje punktet bommer med 2 og står for nesten hele tapet.

::: hvorfor Hvorfor kvadrere?
Uten kvadrering kan positive og negative feil nulle hverandre ut. Modellen under bommer kraftig på begge punktene, men gjennomsnittsfeilen er 0. Kvadratet gjør alle feil positive, straffer store feil ekstra, og gir en glatt funksjon som vi kan derivere (det trenger vi i neste leksjon).

```python
import numpy as np
y      = np.array([10.0, 10.0])
y_pred = np.array([0.0, 20.0])
print("Gjennomsnittlig feil:", np.mean(y_pred - y))
print("MSE:", np.mean((y_pred - y) ** 2))
```
:::

## Tapet som funksjon av parameteren
Tapet avhenger av parametrene. Her holder vi `b = 5` fast og prøver mange verdier av `w` på dataene fra prosjekt 1. Kurven er en parabel med ett bunnpunkt: den beste `w`. Å finne bunnpunktet er hele jobben til treningen.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values
y = data["Energy"].values

def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

w_verdier = np.linspace(1, 5, 81)
tap = [mse(y, w * x + 5) for w in w_verdier]

beste = w_verdier[np.argmin(tap)]
print(f"Lavest MSE for w = {beste:.2f}")

plt.plot(w_verdier, tap)
plt.axvline(beste, color="red", linestyle="--")
plt.xlabel("w")
plt.ylabel("MSE")
plt.show()
```

Å prøve alle verdier fungerer for én parameter, men ikke for tusenvis. Derfor trenger vi gradient descent.

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

::: quiz
Hva skrives ut?
```kode
y      = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = np.array([2.0, 2.0, 1.0, 4.0])
print(np.mean((y_pred - y) ** 2))
```
- [ ] `0.75`
- [x] `1.25`
- [ ] `5.0`
- [ ] `-0.25`
? Feilene er `[1, 0, -2, 0]`, kvadrert `[1, 0, 4, 0]`. Summen er 5, og gjennomsnittet over 4 punkter er 1.25. `5.0` er summen (glemt å dele), og `-0.25` er gjennomsnittet av feilene uten kvadrering.
:::

::: oppsummering
- Tapsfunksjonen måler hvor feil modellen tar. Lavere er bedre.
- MSE: `np.mean((y_pred - y) ** 2)`.
- Kvadratet hindrer at positive og negative feil nuller hverandre ut, og gjør tapet deriverbart.
- Å trene modellen betyr å finne parametrene som gir lavest tap.
:::

--- LEKSJON 5.3 | gradienter | Gradienter
**Gradienten** sier hvordan tapet endrer seg når vi endrer en parameter litt. Den deriverte $\partial L / \partial w$ er stigningstallet til tapskurven i punktet vi står i. Positiv gradient betyr at tapet øker når `w` øker, så da bør vi **minke** `w`.

For MSE med $\hat y = Xw + b$ gir kjerneregelen (den ytre deriverte $2(\hat y_i - y_i)$ ganger den indre $x_i$):

$$\frac{\partial L}{\partial w} = \frac{2}{n} X^T(\hat y - y), \qquad \frac{\partial L}{\partial b} = \frac{2}{n}\sum_{i=1}^n (\hat y_i - y_i)$$

I numpy er det to linjer:

```python
import numpy as np

X = np.array([[1.0], [2.0], [3.0]])   # 3 datapunkter, 1 feature
y = np.array([5.0, 7.0, 9.0])          # y = 2x + 3
w = np.array([0.0])
b = 0.0

n = len(y)
y_pred = X @ w + b
dw = (2 / n) * X.T @ (y_pred - y)
db = (2 / n) * np.sum(y_pred - y)
print("dw =", dw, " db =", db)
```

Begge gradientene er negative: tapet minker hvis vi øker `w` og `b`. Det stemmer, for de sanne verdiene er `w = 2` og `b = 3`.

::: husk Shapes
`X.T` har shape `(n_features, m)` og `y_pred - y` har shape `(m,)`. Produktet får shape `(n_features,)`: én gradient per vekt, akkurat som `w`. `db` er ett tall, akkurat som `b`.
:::

## Sjekk gradienten numerisk
Det er lett å gjøre fortegnsfeil eller glemme en faktor. En trygg sjekk er **endelig differanse**: flytt parameteren et lite stykke $h$ opp og ned, og se hvor mye tapet endrer seg:

$$\frac{\partial L}{\partial w} \approx \frac{L(w + h) - L(w - h)}{2h}$$

```python
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(50, 2))
y = X @ np.array([1.5, -2.0]) + 0.7 + rng.normal(0, 0.1, 50)
w = np.array([0.3, 0.1])
b = 0.2

def tap(w, b):
    return np.mean((X @ w + b - y) ** 2)

# analytisk gradient (formelen)
y_pred = X @ w + b
dw = (2 / len(y)) * X.T @ (y_pred - y)
db = (2 / len(y)) * np.sum(y_pred - y)

# numerisk gradient
h = 1e-5
dw_num = np.zeros(2)
for j in range(2):
    e = np.zeros(2)
    e[j] = h
    dw_num[j] = (tap(w + e, b) - tap(w - e, b)) / (2 * h)
db_num = (tap(w, b + h) - tap(w, b - h)) / (2 * h)

print("analytisk:", dw, db)
print("numerisk: ", dw_num, db_num)
```

Tallene er like med mange desimaler. Da vet vi at formelen er riktig implementert.

::: oppgave Skriv gradients
Lag funksjonen `gradients(X, y, w, b)` som returnerer `dw, db` (i den rekkefølgen) for MSE-tapet til modellen $\hat y = Xw + b$.
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
dw = np.asarray(dw, dtype=float)
assert dw.shape == (2,), f"dw skal ha samme shape som w, (2,), men har {dw.shape}. Bruk X.T @ (y_pred - y)."
assert np.allclose(dw, [-3.25, -4.375]), f"dw = {dw} er feil. Sjekk fortegnet (y_pred - y) og faktoren 2/n."
assert abs(float(db) - (-3.25)) < 1e-9, f"db = {db} er feil. Det skal være (2/n) * summen av (y_pred - y)."
```
:::

::: quiz
Gradienten `dw` for én vekt er `+4.2`. Hva bør gradient descent gjøre med denne vekten?
- [ ] Øke den, fordi gradienten er positiv
- [x] Minke den, fordi tapet øker når vekten øker
- [ ] Sette den til 0
- [ ] Ingenting, en positiv gradient betyr at vi er i minimum
? Gradienten peker i retningen der tapet **øker**. Vi vil ned, så vi går motsatt vei: minker vekten. I minimum er gradienten 0, ikke positiv.
:::

::: oppsummering
- Gradienten er den deriverte av tapet med hensyn på hver parameter, og peker «oppover».
- MSE: `dw = (2/n) * X.T @ (y_pred - y)` og `db = (2/n) * np.sum(y_pred - y)`.
- `dw` har samme shape som `w`, `db` er ett tall.
- Sjekk formelen med endelig differanse: `(L(w+h) - L(w-h)) / (2h)`.
:::

--- LEKSJON 5.4 | gradient-descent | Gradient descent-løkka
Nå setter vi det sammen. **Gradient descent** starter med tilfeldige (eller null-) parametre og tar mange små steg i motsatt retning av gradienten:

$$w \leftarrow w - \eta \frac{\partial L}{\partial w}, \qquad b \leftarrow b - \eta \frac{\partial L}{\partial b}$$

$\eta$ er **læringsraten** (learning rate, `lr`). Den bestemmer hvor lange stegene er. Læringsraten og antall iterasjoner er **hyperparametre**: de styrer læringen, men er ikke en del av modellen.

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
X = rng.uniform(0, 2, size=(50, 1))
y = 3 * X[:, 0] + 5 + rng.normal(0, 0.5, 50)      # sann sammenheng: w = 3, b = 5

w = np.zeros(1)
b = 0.0
lr = 0.1
n_iterations = 200
loss_history = []

for i in range(n_iterations):
    y_pred = X @ w + b
    loss_history.append(np.mean((y_pred - y) ** 2))
    dw = (2 / len(y)) * X.T @ (y_pred - y)
    db = (2 / len(y)) * np.sum(y_pred - y)
    w = w - lr * dw
    b = b - lr * db

print(f"w = {w[0]:.3f}, b = {b:.3f}, siste tap = {loss_history[-1]:.3f}")
plt.plot(loss_history)
plt.xlabel("Iterasjon")
plt.ylabel("MSE")
plt.show()
```

Tapet faller bratt i starten og flater så ut. Det er slik en vellykket trening ser ut. Se `loss_history` i variabelpanelet: en liste med 200 tall, ett per iterasjon, og `w` og `b` har havnet nær 3 og 5. Når kurven har flatet ut, er det lite mer å hente.

## Læringsraten: for liten, passe, for stor
Kjør samme løkke med tre ulike læringsrater. Y-aksen er logaritmisk så alle tre kurvene synes:

```python
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

for lr in [0.01, 0.1, 0.5]:
    plt.plot(train(lr), label=f"lr = {lr}")
plt.yscale("log")
plt.xlabel("Iterasjon")
plt.ylabel("MSE")
plt.legend()
plt.show()
```

- **For liten** (0.01): tapet synker, men så sakte at vi trenger veldig mange iterasjoner.
- **Passe** (0.1): rask og jevn nedgang.
- **For stor** (0.5): stegene hopper over bunnen, tapet **oscillerer og vokser**. Får det fortsette, blir tallene til slutt `inf` eller `nan`.

::: eksamen
Eksamen 2025 spurte: «Trenings-losset begynner å oscillere i stedet for å minke jevnt. Anta at implementasjonen er korrekt. Hva er den mest sannsynlige årsaken?» Svaret var at **læringsraten er for høy**.
:::

## Hva betyr en gradient nær null?
Når gradienten er nær null, blir stegene `lr * dw` nesten null, og parametrene slutter å endre seg. Det betyr at vi står (nesten) i et **minimum** av tapet, der kurven er flat. For MSE med lineær modell er tapet konvekst (én skål), så minimumet er det globale. For andre modeller kan en flat gradient også bety et lokalt minimum eller et flatt platå. Mange implementasjoner bruker dette som stoppkriterium:

```python
import numpy as np

rng = np.random.default_rng(0)
X = rng.uniform(0, 2, size=(50, 1))
y = 3 * X[:, 0] + 5 + rng.normal(0, 0.5, 50)

w, b, lr = np.zeros(1), 0.0, 0.1
for i in range(10000):
    y_pred = X @ w + b
    dw = (2 / len(y)) * X.T @ (y_pred - y)
    db = (2 / len(y)) * np.sum(y_pred - y)
    if np.sqrt(np.sum(dw ** 2) + db ** 2) < 1e-4:   # gradienten er (nesten) null
        print(f"Stoppet etter {i} iterasjoner")
        break
    w = w - lr * dw
    b = b - lr * db

print(f"w = {w[0]:.3f}, b = {b:.3f}")
```

::: prosjekt Prosjekt 1
Prosjektet spør: «Visualize the training loss curve. Is it stable?» og «What does it tell you when the gradient in SGD is close to zero?». Svarene finner du i plottene over: en stabil kurve synker og flater ut uten å hoppe, og en gradient nær null betyr at parametrene har sluttet å endre seg fordi vi er nær et minimum.
:::

::: oppgave Skriv gradient descent-løkka
Data `X` og `y` er gitt. Fullfør løkka slik at den oppdaterer `w` og `b` med gradient descent, og legger MSE-tapet for hver iterasjon til i `loss_history` (før oppdateringen).
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

::: quiz
Under trening oscillerer tapet og blir større for hver iterasjon. Implementasjonen er korrekt. Hva er den mest sannsynlige årsaken?
- [ ] Modellen har for mange parametre til at gradient descent kan brukes
- [ ] Datasettet er perfekt lineært separerbart
- [ ] Tapsfunksjonen er ikke deriverbar
- [x] Læringsraten er for høy
? Med for stor læringsrate tar vi så lange steg at vi hopper forbi bunnen og havner høyere opp på andre siden, igjen og igjen.
:::

::: quiz
Etter en del iterasjoner er gradienten nesten null. Hva forteller det?
- [x] Parametrene endrer seg nesten ikke lenger, fordi vi er nær et minimum (eller et flatt område) av tapet
- [ ] Modellen er perfekt og tapet er 0
- [ ] Læringsraten er for høy
- [ ] Modellen har overtilpasset seg treningsdataene
? En gradient på null betyr at tapskurven er flat der vi står. Tapet trenger ikke være 0 (data har støy), det er bare ikke mulig å komme lavere ved små endringer.
:::

::: oppsummering
- Oppdateringen er `w = w - lr * dw` og `b = b - lr * db`, gjentatt `n_iterations` ganger.
- Lagre tapet i en liste (`loss_history`) og plott det for å se om treningen fungerer.
- For liten læringsrate: treg. For stor: tapet oscillerer og eksploderer.
- Gradient nær null: parametrene står stille, vi er nær et minimum.
:::

--- LEKSJON 5.5 | linear-regression | LinearRegression-klassen
I prosjekt 1 skal koden ligge i en **klasse** med metodene `fit` (tren) og `predict` (prediker). Navnene `fit` og `predict` er standard i alle ML-biblioteker, så det lønner seg å bruke dem også når du skriver koden selv. Parametrene og hyperparametrene lagres som attributter på `self`.

```python
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


# test på enkle data: y = 2*x1 - 3*x2 + 1
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 2))
y = X @ np.array([2.0, -3.0]) + 1

model = LinearRegression(lr=0.1, n_iterations=200)
print("Før fit:")
display(model)
model.fit(X, y)
print("Etter fit:")
display(model)
print("weights:", model.weights.round(3))
print("bias:", round(model.bias, 3))
print("prediksjon for [1, 0]:", model.predict(np.array([[1.0, 0.0]])))
```

Legg merke til at `fit` ikke returnerer noe. Den endrer `self.weights` og `self.bias`, og etterpå bruker `predict` de lærte verdiene. Du ser det i de to visningene av `model`: før `fit` er `weights` og `bias` `None` og `loss_history` er tom, etter `fit` er `weights` en array nær `[2, -3]`, `bias` nær 1, og `loss_history` har 200 tall.

## Prosjekt 1, problem 1: strømforbruk
Nå trener vi på `regression.csv`. Fordi `Net_Activity` ligger mellom 10 og 30, må læringsraten være liten (større gir divergens), og da trengs mange iterasjoner. I leksjon 5.9 ser du hvordan skalering fikser dette.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
X = data[["Net_Activity"]].values      # shape (1000, 1)
y = data["Energy"].values              # shape (1000,)

model = LinearRegression(lr=0.002, n_iterations=20000)
model.fit(X, y)
w, b = model.weights[0], model.bias
print(f"Energy ≈ {w:.3f} · Net_Activity + {b:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(model.loss_history)
ax[0].set_yscale("log")
ax[0].set_xlabel("Iterasjon")
ax[0].set_ylabel("MSE")
ax[0].set_title("Tapskurve")
ax[1].scatter(X[:, 0], y, s=6, label="Data")
x_linje = np.linspace(10, 30, 2).reshape(-1, 1)
ax[1].plot(x_linje, model.predict(x_linje), color="red", label="Modell")
ax[1].set_xlabel("Net_Activity")
ax[1].set_ylabel("Energy")
ax[1].legend()
plt.show()
```

Vi kan sjekke svaret mot den eksakte minste kvadraters løsningen, som `np.polyfit` regner ut direkte:

```python
import numpy as np
import pandas as pd
data = pd.read_csv("regression.csv")
w, b = np.polyfit(data["Net_Activity"], data["Energy"], deg=1)
print(f"Eksakt: Energy ≈ {w:.3f} · Net_Activity + {b:.3f}")
```

## Residualer og forventningsrett estimator
**Residualene** (prediksjonsfeilene) er $y - \hat y$. Plott fordelingen deres. En god modell har residualer som er sentrert rundt 0 og ikke viser noe mønster når vi plotter dem mot $x$.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values
y = data["Energy"].values
w, b = np.polyfit(x, y, deg=1)          # samme svar som gradient descent gir
y_hat = w * x + b
residualer = y - y_hat
display(pd.DataFrame({"y": y, "ŷ": y_hat, "y - ŷ": residualer}).head(6).round(2))

print(f"Gjennomsnitt av residualene: {residualer.mean():.4f}")
print(f"Standardavvik: {residualer.std():.3f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(residualer, bins=30)
ax[0].set_xlabel("y - ŷ")
ax[0].set_title("Fordeling av feilen")
ax[1].scatter(x, residualer, s=6)
ax[1].axhline(0, color="red")
ax[1].set_xlabel("Net_Activity")
ax[1].set_ylabel("y - ŷ")
ax[1].set_title("Residualer mot x")
plt.show()
```

En **forventningsrett** (unbiased) estimator bommer ikke systematisk i én retning: i gjennomsnitt er feilen 0, $E[\hat y - y] = 0$. Her er residualene symmetriske rundt 0 (omtrent normalfordelt), så modellen overestimerer like ofte som den underestimerer. Hadde residualene hatt et mønster (for eksempel en bue), ville det vært et tegn på at en rett linje ikke passer dataene.

::: prosjekt Prosjekt 1, problem 1
Du skal skrive akkurat `fit` og `predict` i `LinearRegression` i `linear_models.py`, med attributtene `self.weights`, `self.bias`, `self.lr`, `self.n_iterations` og `self.loss_history`. Deretter: plott tapskurven, plott linjen over dataene, skriv opp formelen for Energy og analyser fordelingen av prediksjonsfeilen.
:::

::: oppgave Fullfør fit og predict
Fyll inn de tre manglende linjene i `fit` og linjen i `predict`. Klassen skal fungere for vilkårlig mange features.
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
assert np.shape(m_t.weights) == (3,), f"weights skal ha shape (3,), men har {np.shape(m_t.weights)}."
assert np.allclose(m_t.weights, [1.0, -2.0, 0.5], atol=0.01), f"weights = {m_t.weights} skulle vært [1, -2, 0.5]. Sjekk dw."
assert abs(m_t.bias - 4.0) < 0.01, f"bias = {m_t.bias} skulle vært 4. Sjekk db."
p_t = m_t.predict(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
assert p_t is not None, "predict returnerer ingenting. Husk return."
assert np.allclose(p_t, [4.0, 3.5], atol=0.02), f"predict ga {p_t}, forventet omtrent [4, 3.5]."
```
:::

::: quiz
Hva skjer når denne `fit`-metoden kjøres?
```kode
for _ in range(self.n_iterations):
    y_pred = X @ self.weights + self.bias
    dw = (2 / n_samples) * X.T @ (y_pred - y)
    db = (2 / n_samples) * np.sum(y_pred - y)
    self.weights += self.lr * dw
    self.bias += self.lr * db
```
- [ ] Modellen trenes som normalt
- [x] Tapet øker, fordi parametrene flyttes i samme retning som gradienten
- [ ] Koden gir `TypeError` fordi `+=` ikke virker på arrays
- [ ] Modellen konvergerer, men tregere
? Med `+=` går vi **oppover** i tapslandskapet (gradient ascent). Riktig er `-=`, altså motsatt retning av gradienten.
:::

::: oppsummering
- `__init__` lagrer hyperparametrene (`lr`, `n_iterations`) og setter `weights`/`bias` til `None`.
- `fit(X, y)` initialiserer `weights = np.zeros(n_features)`, `bias = 0` og kjører gradient descent-løkka.
- `predict(X)` returnerer `X @ self.weights + self.bias`.
- Residualene `y - y_pred` bør være sentrert rundt 0 uten mønster. En forventningsrett estimator har gjennomsnittlig feil 0.
:::

--- LEKSJON 5.6 | sigmoid | Sigmoid og logistisk regresjon
I **klassifisering** er target 0 eller 1, og vi vil at modellen skal gi en **sannsynlighet** for klasse 1. Den lineære delen $z = Xw + b$ kan bli et hvilket som helst tall, så vi sender den gjennom **sigmoid-funksjonen**, som klemmer alle tall inn i intervallet $(0, 1)$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Dette er **logistisk regresjon**: $p = \sigma(Xw + b)$.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print(sigmoid(np.array([-5, -1, 0, 1, 5])))

z = np.linspace(-8, 8, 200)
plt.plot(z, sigmoid(z))
plt.axhline(0.5, color="gray", linestyle="--")
plt.xlabel("z")
plt.ylabel("σ(z)")
plt.show()
```

Store negative $z$ gir nesten 0, store positive gir nesten 1, og $\sigma(0) = 0.5$.

## Numerisk stabil sigmoid
For veldig negative $z$ blir `np.exp(-z)` så stort at det renner over (`overflow`), og numpy gir en advarsel. Resultatet blir riktig nok 0, men advarslene fyller utskriften din under trening. Løsningen er å klippe $z$ med `np.clip` først. $\sigma(-500) \approx 10^{-218}$ og $\sigma(500) = 1$, så vi mister ingenting som betyr noe.

```python
import numpy as np

z = np.array([-1000.0, 0.0, 1000.0])
print("Uten clip:", 1 / (1 + np.exp(-z)))      # gir RuntimeWarning: overflow

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

print("Med clip: ", sigmoid(z))
```

## Fra sannsynlighet til klasse
Modellen gir sannsynligheter. For å få en klasse bruker vi en **terskel**, vanligvis 0.5. Forelesningsnotatene gjør det med en list comprehension, men en numpy-sammenligning gjør det samme:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

X = np.array([[1.0, 2.0], [-1.0, 0.5], [3.0, -1.0], [0.0, 0.0]])
w = np.array([0.8, -1.2])
b = 0.1

z = np.matmul(X, w) + b
y_pred = sigmoid(z)
print("z:", z)
print("sannsynligheter:", y_pred.round(3))

print([1 if _y > 0.5 else 0 for _y in y_pred])   # som i notatene (gir en liste)
print((y_pred > 0.5).astype(int))                # numpy-varianten (gir en array)
```

::: eksamen Eksamen 2025
Et kodelesingsspørsmål viste `z = np.matmul(X, w)` og `y_pred = 1 / (1 + np.exp(-z))` med `X` av shape `(n_samples, n_features)` og spurte hva `y_pred` inneholder. Svaret: **predikerte sannsynligheter i intervallet (0, 1) for hvert sample**. Ikke klasser 0/1 (da mangler terskelen), og ikke vilkårlige reelle tall (det er `z`).
:::

::: oppgave Stabil sigmoid
Lag funksjonen `sigmoid(z)` som virker på numpy-arrays og bruker `np.clip` slik at den ikke gir overflow-advarsler for store verdier som `-1000`.
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

::: quiz
`X` har shape `(n_samples, n_features)` og `w` har shape `(n_features,)`. Hva inneholder `y_pred`?
```kode
import numpy as np
z = np.matmul(X, w)
y_pred = 1 / (1 + np.exp(-z))
```
- [ ] For hvert sample en score som kan anta hvilken som helst reell verdi
- [ ] Logaritmen av de predikerte sannsynlighetene for hvert sample
- [x] Predikerte sannsynligheter i intervallet (0, 1) for hvert sample
- [ ] Klasse-labels 0 eller 1 for hvert sample
? Sigmoid gjør hver score `z` om til et tall mellom 0 og 1. For å få 0/1 trengs en terskel, for eksempel `y_pred > 0.5`.
:::

::: oppsummering
- Logistisk regresjon: `p = sigmoid(X @ w + b)`, en sannsynlighet for klasse 1.
- `sigmoid(z) = 1 / (1 + np.exp(-z))`, med `np.clip(z, -500, 500)` for å unngå overflow.
- Klasse fra sannsynlighet: `(p > 0.5).astype(int)` eller `[1 if _y > 0.5 else 0 for _y in p]`.
:::

--- LEKSJON 5.7 | cross-entropy | Cross-entropy og LogisticRegression-klassen
For klassifisering bruker vi ikke MSE, men **binær cross-entropy** (log loss):

$$L = -\frac{1}{n}\sum_{i=1}^n \Big[ y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \Big]$$

Når $y = 1$ er bare første ledd med, og tapet er $-\ln p$: lite når $p$ er nær 1, enormt når $p$ er nær 0. Selvsikre feil straffes hardt. Med sigmoid er cross-entropy **konveks**, så gradient descent finner det globale minimumet. MSE kombinert med sigmoid er ikke konveks.

```python
import numpy as np

def bce(y, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)          # unngå log(0)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

y = np.array([1, 0, 1, 0])
print("God modell:      ", bce(y, np.array([0.9, 0.1, 0.8, 0.2])))
print("Usikker modell:  ", bce(y, np.array([0.5, 0.5, 0.5, 0.5])))
print("Selvsikker, feil:", bce(y, np.array([0.01, 0.99, 0.01, 0.99])))
```

## Gradienten får samme form
Deriverer vi cross-entropy gjennom sigmoid, forkorter nesten alt seg bort (se forelesningsnotatene), og vi står igjen med:

$$\frac{\partial L}{\partial w} = \frac{1}{n} X^T(p - y), \qquad \frac{\partial L}{\partial b} = \frac{1}{n}\sum_{i=1}^n (p_i - y_i)$$

Det er nøyaktig samme form som for lineær regresjon (bare uten faktoren 2), med $p$ i stedet for $\hat y$. Vi sjekker numerisk:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def bce(y, p):
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

rng = np.random.default_rng(0)
X = rng.normal(size=(30, 2))
y = (X[:, 0] + X[:, 1] > 0).astype(float)
w, b = np.array([0.2, -0.4]), 0.1

p = sigmoid(X @ w + b)
dw = X.T @ (p - y) / len(y)
db = np.mean(p - y)

h = 1e-6
dw_num = np.array([(bce(y, sigmoid(X @ (w + e) + b)) - bce(y, sigmoid(X @ (w - e) + b))) / (2 * h)
                   for e in np.eye(2) * h])
print("analytisk:", dw, db)
print("numerisk: ", dw_num)
```

## Hele klassen
Strukturen er den samme som `LinearRegression`, med tre forskjeller: sigmoid rundt den lineære delen, cross-entropy som tap, og en egen `predict_proba` som gir sannsynligheter, mens `predict` gir klasser.

```python
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights = None
        self.bias = None
        self.lr = lr
        self.n_iterations = n_iterations
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
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

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)


# to punktskyer
rng = np.random.default_rng(0)
X = np.vstack([rng.normal(-1, 1, size=(100, 2)), rng.normal(1.5, 1, size=(100, 2))])
y = np.array([0] * 100 + [1] * 100)

model = LogisticRegression(lr=0.1, n_iterations=500)
display(model)                  # før fit: weights og bias er None
model.fit(X, y)
display(model)                  # etter fit: lærte tall
print("weights:", model.weights.round(3), " bias:", round(model.bias, 3))
print("accuracy på treningsdata:", np.mean(model.predict(X) == y))

plt.plot(model.loss_history)
plt.xlabel("Iterasjon")
plt.ylabel("Cross-entropy")
plt.show()
```

## Notatenes versjon
Forelesningsnotatene skriver den lineære delen på to måter: `np.matmul(self.weights, x.transpose())` i `fit` og `np.matmul(x, self.weights)` i `predict`. Med en 1D-vektor `w` gir de nøyaktig det samme, og `predict` returnerer en **liste**:

```python
import numpy as np

x = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.5]])
weights = np.array([0.4, -0.3])
bias = 0.5

a = np.matmul(weights, x.transpose()) + bias    # (2,) @ (2, 3) -> (3,)
c = np.matmul(x, weights) + bias                # (3, 2) @ (2,) -> (3,)
print(a, c, np.allclose(a, c))

y_pred = 1 / (1 + np.exp(-c))
pred = [1 if _y > 0.5 else 0 for _y in y_pred]
print(pred, type(pred))
```

::: advarsel Liste == liste er ikke elementvis
`accuracy` i notatene er `np.mean(true_values == predictions)`. Det virker når minst én av dem er en numpy-array. Er **begge** vanlige Python-lister, sammenligner `==` hele listene og gir én `True`/`False`, så «accuracy» blir 0.0 eller 1.0.

```python
import numpy as np
y_true = [1, 0, 1, 1]
pred   = [1, 0, 0, 1]
print(np.mean(y_true == pred))               # 0.0  (feil!)
print(np.mean(np.array(y_true) == pred))     # 0.75 (riktig)
```
:::

::: prosjekt Prosjekt 1, problem 2
I `linear_models.py` skal `LogisticRegression` ha `fit`, `predict_proba`, `predict` og `sigmoid`, med de samme attributtene som `LinearRegression`. `predict_proba` gir sannsynligheter (trengs for ROC-kurven, se leksjon 6.4), og `predict` gir 0/1.
:::

::: oppgave Gradientene i fit
Fullfør `dw` og `db` i `fit`, og `predict` (terskel 0.5, returner 0/1).
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
assert np.shape(m_t.weights) == (2,), f"weights skal ha shape (2,), men har {np.shape(m_t.weights)}."
assert m_t.weights[0] > 0 and m_t.weights[1] < 0, f"weights = {m_t.weights} har feil fortegn. Gradienten er X.T @ (p - y) / n."
pr = m_t.predict(X_t)
assert pr is not None, "predict returnerer ingenting. Husk return."
pr = np.asarray(pr)
assert set(np.unique(pr)) <= {0, 1}, "predict skal returnere klasser 0 og 1, ikke sannsynligheter."
assert np.mean(pr == y_t) > 0.95, f"Accuracy er bare {np.mean(pr == y_t):.2f}. Sjekk dw, db og terskelen."
```
:::

::: quiz
Hva er gradienten av binær cross-entropy med hensyn på vektene i logistisk regresjon? (`p` er sigmoid-utdataene.)
- [ ] `X.T @ (p - y) * p * (1 - p) / n`
- [x] `X.T @ (p - y) / n`
- [ ] `X @ (p - y) / n`
- [ ] `np.mean(p - y)`
? Faktoren $\sigma(1-\sigma)$ fra sigmoid forkorter seg mot nevneren fra logaritmen, så vi står igjen med `X.T @ (p - y) / n`. `X @ (p - y)` har feil shape, og `np.mean(p - y)` er gradienten for bias.
:::

::: oppsummering
- Binær cross-entropy: `-np.mean(y*np.log(p) + (1-y)*np.log(1-p))`, med `np.clip` på `p` for å unngå `log(0)`.
- Gradientene: `dw = X.T @ (p - y) / n` og `db = np.mean(p - y)`. Samme form som lineær regresjon.
- `predict_proba` gir sannsynligheter, `predict` gir klasser med terskel 0.5.
- `np.matmul(w, X.T)` og `np.matmul(X, w)` gir det samme for 1D `w`.
- Pass på at minst én side av `==` er en numpy-array når du regner accuracy.
:::

--- LEKSJON 5.8 | feature-engineering | Feature engineering (XOR)
En lineær modell kan bare lage en **rett** beslutningsgrense: $w_0 x_0 + w_1 x_1 + b = 0$ er en linje i planet. Det er modellens **induktive bias**, antakelsen den gjør om dataene. Er mønsteret ikke lineært, hjelper ikke mer trening. Da må vi gi modellen bedre features. Det kalles **feature engineering**.

Dataene i problem 2 av prosjekt 1 har kolonnene `x0`, `x1`, `y` og `split`:

```python
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
print(len(train), len(test))

plt.scatter(train["x0"], train["x1"], c=train["y"], cmap="coolwarm", s=10)
plt.axhline(0, color="gray")
plt.axvline(0, color="gray")
plt.xlabel("x0")
plt.ylabel("x1")
plt.title("not linearly separable")
plt.colorbar()
plt.show()
```

Klasse 1 ligger i 2. og 4. kvadrant, klasse 0 i 1. og 3. Det er et **XOR-mønster**: ingen rett linje kan skille dem.

## Først uten feature engineering

```python
import numpy as np
import pandas as pd

class LogisticRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights, self.bias = None, None
        self.lr, self.n_iterations = lr, n_iterations
        self.loss_history = []
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    def fit(self, X, y):
        self.weights, self.bias = np.zeros(X.shape[1]), 0.0
        for _ in range(self.n_iterations):
            p = self.sigmoid(X @ self.weights + self.bias)
            self.weights -= self.lr * X.T @ (p - y) / len(y)
            self.bias -= self.lr * np.mean(p - y)
    def predict_proba(self, X):
        return self.sigmoid(X @ self.weights + self.bias)
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]

X_train = train[["x0", "x1"]].values
X_test = test[["x0", "x1"]].values
y_train, y_test = train["y"].values, test["y"].values

model = LogisticRegression(lr=0.1, n_iterations=1000)
model.fit(X_train, y_train)
print("Test accuracy:", np.mean(model.predict(X_test) == y_test))
```

Rundt 0.5: like godt som å kaste mynt.

## Den nye featuren: x0 · x1
Produktet $x_0 \cdot x_1$ er **negativt** i 2. og 4. kvadrant og **positivt** i 1. og 3. Det er akkurat det som skiller klassene. Legger vi det til som en tredje kolonne, blir problemet lineært separerbart i det nye feature-rommet:

```python
import numpy as np
import pandas as pd

class LogisticRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights, self.bias = None, None
        self.lr, self.n_iterations = lr, n_iterations
        self.loss_history = []
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    def fit(self, X, y):
        self.weights, self.bias = np.zeros(X.shape[1]), 0.0
        for _ in range(self.n_iterations):
            p = self.sigmoid(X @ self.weights + self.bias)
            self.weights -= self.lr * X.T @ (p - y) / len(y)
            self.bias -= self.lr * np.mean(p - y)
    def predict_proba(self, X):
        return self.sigmoid(X @ self.weights + self.bias)
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

def lag_features(df):
    return np.column_stack([df["x0"], df["x1"], df["x0"] * df["x1"]])

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]

X_train, X_test = lag_features(train), lag_features(test)
y_train, y_test = train["y"].values, test["y"].values

print("Før: to features")
display(train[["x0", "x1", "y"]].head(6))
print("Etter: x0 * x1 som tredje feature")
display(pd.DataFrame(X_train[:6], columns=["x0", "x1", "x0*x1"]).assign(y=y_train[:6]))

model = LogisticRegression(lr=0.1, n_iterations=1000)
model.fit(X_train, y_train)
print("weights:", model.weights.round(2))
print("Test accuracy:", np.mean(model.predict(X_test) == y_test))
```

Sammenlign de to tabellene: i den nye kolonnen er `x0*x1` negativ nettopp i radene der `y = 1`. Vekten for `x0 * x1` er stor og negativ: når produktet er negativt, blir $z$ positiv og modellen sier klasse 1. Modellen er fortsatt lineær **i featurene**, men grensen er krum i det opprinnelige $(x_0, x_1)$-planet. Det ser vi med `meshgrid` og `contourf`:

```python
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

g0, g1 = np.meshgrid(np.linspace(-3.5, 3.5, 150), np.linspace(-3.5, 3.5, 150))
G = np.column_stack([g0.ravel(), g1.ravel(), g0.ravel() * g1.ravel()])
P = sigmoid(G @ w + b).reshape(g0.shape)

plt.contourf(g0, g1, P, levels=20, cmap="coolwarm", alpha=0.6)
plt.contour(g0, g1, P, levels=[0.5], colors="black")
plt.scatter(train["x0"], train["x1"], c=y, cmap="coolwarm", s=8, edgecolors="k", linewidths=0.2)
plt.xlabel("x0")
plt.ylabel("x1")
plt.show()
```

::: prosjekt Prosjekt 1, problem 2
Kravet var minst **0.88** accuracy på testsettet. Spørsmålene: «Explain poor initial performance and your improvements» (lineær grense kan ikke skille XOR, `x0 * x1` gjør det lineært separerbart) og «What is the model's inductive bias?» (antakelsen om at klassene kan skilles av et hyperplan i feature-rommet).
:::

::: oppgave Nå 0.88
Klassen og dataene er gitt. Lag `X_train_fe` og `X_test_fe` med tre kolonner: `x0`, `x1` og produktet `x0 * x1`. Tren modellen og lagre test-accuracy i `acc`.
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

::: quiz
Hvorfor gir logistisk regresjon på bare `x0` og `x1` rundt 50 % accuracy på XOR-dataene?
- [ ] Læringsraten er for lav, så modellen rekker ikke å lære
- [x] Beslutningsgrensen er en rett linje, og ingen rett linje kan skille klassene
- [ ] Sigmoid-funksjonen kan ikke gi sannsynligheter over 0.5
- [ ] Datasettet har for få punkter
? Logistisk regresjon er lineær i featurene, så grensen $w_0x_0 + w_1x_1 + b = 0$ er en linje. Mer trening hjelper ikke. En ny feature som `x0 * x1` gjør problemet lineært separerbart.
:::

::: oppsummering
- Logistisk regresjon gir en lineær beslutningsgrense. Det er den induktive biasen.
- **Feature engineering**: lag nye kolonner (her `x0 * x1`) så problemet blir lineært i de nye featurene.
- `np.column_stack([...])` setter sammen kolonner til en matrise.
- Lag de samme featurene for både trenings- og testdata.
:::

--- LEKSJON 5.9 | skalering | Skalering for gradient descent
I leksjon 5.5 trengte vi 20 000 iterasjoner på `regression.csv`. Årsaken er at `Net_Activity` har verdier rundt 20, mens bias-«featuren» i praksis er 1. Gradienten for `w` blir mye større enn for `b`, så en læringsrate som er trygg for `w` er altfor liten for `b`. Se hva som skjer med ulike læringsrater:

```python
import numpy as np
import pandas as pd

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values
y = data["Energy"].values

for lr in [0.001, 0.002, 0.003]:
    w, b = 0.0, 0.0
    with np.errstate(all="ignore"):              # skjul overflow-advarsler
        for _ in range(2000):
            y_pred = w * x + b
            w -= lr * 2 * np.mean((y_pred - y) * x)
            b -= lr * 2 * np.mean(y_pred - y)
    print(f"lr = {lr}: w = {w:.3f}, b = {b:.3f}")
```

Med 0.001 og 0.002 er `w` nær 3, men `b` er langt unna riktig verdi (rundt 4.9). Med 0.003 divergerer alt til `nan`.

## Standardisering
Løsningen er å **standardisere** featuren: trekk fra gjennomsnittet og del på standardavviket. Da får den gjennomsnitt 0 og standardavvik 1, og alle parametrene lærer i samme tempo.

$$x_{\text{skalert}} = \frac{x - \mu}{\sigma}$$

```python
import numpy as np
import pandas as pd

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values
y = data["Energy"].values

mu, sigma = x.mean(), x.std()
x_s = (x - mu) / sigma
print(f"Etter skalering: snitt = {x_s.mean():.3f}, std = {x_s.std():.3f}")

w, b = 0.0, 0.0
for _ in range(100):                       # bare 100 iterasjoner!
    y_pred = w * x_s + b
    w -= 0.1 * 2 * np.mean((y_pred - y) * x_s)
    b -= 0.1 * 2 * np.mean(y_pred - y)
print(f"I skalerte enheter: w_s = {w:.3f}, b_s = {b:.3f}")

# tilbake til originale enheter: w_s * (x - mu)/sigma + b_s = (w_s/sigma) * x + (b_s - w_s*mu/sigma)
w_orig = w / sigma
b_orig = b - w * mu / sigma
print(f"Energy ≈ {w_orig:.3f} · Net_Activity + {b_orig:.3f}")
```

Fra 20 000 til 100 iterasjoner, med samme svar. Skalering endrer ikke hvilken modell som er best, bare hvor lett gradient descent finner den.

## Med flere features og train/test
Med en matrise standardiserer vi hver kolonne for seg med `axis=0`. Gjennomsnitt og standardavvik regnes **bare fra treningsdataene**, og de samme tallene brukes på testdataene. Testdataene skal ikke påvirke noe i treningen.

```python
import numpy as np

rng = np.random.default_rng(0)
X_train = rng.normal([50, 0.01], [10, 0.002], size=(80, 2))   # to features med helt ulik skala
X_test = rng.normal([50, 0.01], [10, 0.002], size=(20, 2))

mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
X_train_s = (X_train - mu) / sigma
X_test_s = (X_test - mu) / sigma          # samme mu og sigma som for trening

print("Før (5 første rader):")
display(X_train[:5])
print("Etter:")
display(X_train_s[:5])

print("mu:", mu, " sigma:", sigma)
print("train snitt:", X_train_s.mean(axis=0).round(3), " std:", X_train_s.std(axis=0).round(3))
print("test snitt: ", X_test_s.mean(axis=0).round(3))
```

Før skaleringen er kolonnene helt ulike (rundt 50 og rundt 0.01). Etterpå ligger begge rundt 0 med verdier av samme størrelse, med negative tall i én farge og positive i en annen. Testdataene får ikke nøyaktig snitt 0, og det er helt riktig. En annen variant er **min-maks-skalering**, `(x - x_min) / (x_max - x_min)`, som gir verdier i $[0, 1]$. Samme regel gjelder: `x_min` og `x_max` fra treningsdataene. I del 7 gjør `StandardScaler` dette for oss.

::: eksamen
Både eksamen 2024 og 2025 hadde spørsmål om skalering: man tilpasser skaleringen (regner ut snitt/std eller min/maks) **på treningsdataene**, og bruker de samme tallene på testdataene. Å regne dem ut på nytt fra testdataene er feil.
:::

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

::: quiz
Du har standardisert treningsdataene med `mu = X_train.mean(axis=0)` og `sigma = X_train.std(axis=0)`. Hvordan skal testdataene skaleres?
- [ ] `(X_test - X_test.mean(axis=0)) / X_test.std(axis=0)`
- [x] `(X_test - mu) / sigma`
- [ ] Testdataene skal ikke skaleres
- [ ] `(X_test - mu) / X_test.std(axis=0)`
? Modellen er trent på data skalert med `mu` og `sigma`, så nye data må gjennom nøyaktig samme transformasjon. Ellers ser modellen tall i en annen «enhet» enn den lærte på.
:::

::: oppsummering
- Features med stor eller ulik skala gjør gradient descent tregt eller ustabilt.
- Standardisering: `(X - mu) / sigma` med `mu`, `sigma` fra treningsdataene (`axis=0` per kolonne).
- Bruk de samme `mu` og `sigma` på testdataene.
- Skalering endrer ikke den beste modellen, bare hvor raskt vi finner den.
:::

--- LEKSJON 5.10 | sgd-minibatch | SGD og mini-batch
Vanlig gradient descent regner gradienten på **alle** datapunktene før hvert steg. Med millioner av punkter blir hvert steg dyrt. **Stochastic gradient descent (SGD)** bruker et tilfeldig utvalg i stedet:

| Variant | Gradient regnet på | Oppdateringer per epoke (N punkter) |
|---|---|---|
| Gradient descent (full batch) | alle N punktene | 1 |
| SGD | ett tilfeldig punkt | N |
| Mini-batch SGD | B tilfeldige punkter | N / B |

En **epoke** (epoch) er alltid én hel runde gjennom treningsdataene. Gradienten fra en batch er et støyete, men forventningsrett estimat av den fulle gradienten. **Batch size** er en hyperparameter, ofte 32, 64 eller 128.

## Stokke og dele opp
Hver epoke stokker vi rekkefølgen med `rng.permutation`, og går gjennom indeksene i biter:

```python
import numpy as np

rng = np.random.default_rng(0)
n, batch_size = 10, 4

for epoch in range(2):
    perm = rng.permutation(n)
    print(f"Epoke {epoch}: rekkefølge {perm}")
    display(perm)
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        print("   batch:", idx)
```

Rutenettet viser `perm` for hver epoke: alle tallene 0 til 9 er med nøyaktig én gang, men i ny rekkefølge hver epoke, og batchene er bare påfølgende biter av den. Siste batch blir mindre (2 punkter) når `n` ikke er delelig med `batch_size`. Slicing utover slutten av en array er helt greit i Python.

## Mini-batch gradient descent
Samme løkke som før, men med en indre løkke over batcher. Vi lagrer tapet på hele datasettet én gang per epoke:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("regression.csv")
x = data["Net_Activity"].values
X = ((x - x.mean()) / x.std()).reshape(-1, 1)    # standardisert, shape (1000, 1)
y = data["Energy"].values
n = len(y)

def minibatch_gd(batch_size, lr=0.05, n_epochs=20, seed=0):
    rng = np.random.default_rng(seed)
    w, b, history = np.zeros(1), 0.0, []
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

for bs in [1000, 32, 1]:
    w, b, h = minibatch_gd(bs)
    print(f"batch_size = {bs:4d}: {int(np.ceil(n / bs))} oppdateringer per epoke, w = {w[0]:.2f}, b = {b:.2f}")
    plt.plot(h, label=f"batch_size = {bs}")
plt.yscale("log")
plt.xlabel("Epoke")
plt.ylabel("MSE (hele datasettet)")
plt.legend()
plt.show()
```

Full batch (1000) tar ett steg per epoke og trenger mange epoker. Mini-batch og SGD tar mange steg per epoke og kommer raskt ned, men tapet flater ut på et litt høyere og mer ustøtt nivå fordi gradienten er støyete.

::: tips Gradient nær null i SGD
I SGD varierer gradienten fra batch til batch. Én liten gradient betyr ikke at vi er ferdige. Se heller på tapet over hele datasettet per epoke. Når det slutter å synke og gradientene i snitt er nær null, er vi nær minimumet. Med konstant læringsrate fortsetter parametrene å «vibrere» litt rundt det. Derfor senker man ofte læringsraten underveis.
:::

::: oppgave Lag mini-batcher
Lag funksjonen `minibatch_indekser(n, batch_size, seed)` som stokker indeksene `0, ..., n-1` med `np.random.default_rng(seed).permutation(n)` og returnerer en **liste** med index-arrays, hver med maks `batch_size` elementer. Alle indeksene skal være med nøyaktig én gang.
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

::: quiz
Du har N = 1000 treningspunkter og bruker mini-batch SGD med batch size 100. Hvor mange ganger oppdateres parametrene i løpet av én epoke?
- [ ] 1
- [x] 10
- [ ] 100
- [ ] 1000
? Én epoke er én runde gjennom alle dataene. Med batcher på 100 blir det 1000 / 100 = 10 batcher, og én oppdatering per batch. Full batch ville gitt 1, ren SGD 1000.
:::

::: quiz
Hva er hovedgrunnen til å bruke mini-batch SGD i stedet for full-batch gradient descent?
- [ ] Mini-batch gir en eksakt gradient, mens full batch gir en støyete en
- [x] Hver oppdatering er mye billigere, og vi får mange oppdateringer per epoke
- [ ] Mini-batch trenger ingen læringsrate
- [ ] Full batch kan bare brukes på lineære modeller
? Gradienten på en batch er et støyete men forventningsrett estimat av den fulle gradienten. Den er billig å regne ut, så vi kan ta mange små steg for samme regnekostnad som ett stort.
:::

::: oppsummering
- **Epoke**: én runde gjennom alle treningsdataene.
- Full batch: 1 oppdatering per epoke. SGD: N. Mini-batch: N / B.
- Stokk hver epoke med `perm = rng.permutation(n)`, og bruk `idx = perm[start:start + batch_size]`.
- Mini-batch-gradienten er støyete men forventningsrett, og billig å regne ut.
:::
