=== DEL 9 | nevrale-nett | Nevrale nettverk (prosjekt 3)
Fra ett nevron i NumPy til fullverdige treningsløkker i PyTorch: lag som matriser, backprop, `MLPClassifier`, tapsfunksjoner med logits, mini-batcher, early stopping, cosine annealing og reproduserbarhet. Til slutt Keras-koden fra forelesningsnotatene. Etter denne delen kan du lese og skrive all koden i prosjekt 3, problem 1.

--- LEKSJON 9.1 | nevron-aktivering | Nevron og aktiveringsfunksjoner
Et **nevron** gjør to ting. Først regner det ut en vektet sum av input pluss en bias, $z = \mathbf{w} \cdot \mathbf{x} + b$. Så sender det $z$ gjennom en **aktiveringsfunksjon** $g$, og output er $a = g(z)$. Det er nøyaktig det samme som logistisk regresjon fra del 5, bare med flere mulige valg av $g$.

```python
import numpy as np

x = np.array([1.0, 2.0, -1.0])     # tre features
w = np.array([0.5, -0.25, 1.0])    # én vekt per feature
b = 0.1

z = np.dot(w, x) + b               # vektet sum: 0.5 - 0.5 - 1.0 + 0.1
print("z =", z)

a = 1 / (1 + np.exp(-z))           # sigmoid-aktivering
print("a =", a)
```

## Perseptronet: Heaviside som aktivering
Det opprinnelige **perseptronet** bruker Heaviside-funksjonen: output er 1 hvis $z \ge 0$, ellers 0. Notatene skriver det slik:

```python
import numpy as np

def heaviside(z):
    return np.where(z >= 0.0, 1.0, 0.0)

# AND-porten fra notatene: (w1, w2, b) = (1, 1, -1.5)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
w = np.array([1.0, 1.0])
b = -1.5

z = X @ w + b          # alle fire punktene på én gang
print(z)
print(heaviside(z))    # bare (1, 1) gir 1
```

Heaviside er flat overalt (derivert 0) og hopper i $z = 0$. Derfor kan den ikke trenes med gradient descent. Løsningen er å bytte den ut med en glatt funksjon.

## De vanlige aktiveringsfunksjonene
| Navn | Formel | Verdier | Brukes |
|---|---|---|---|
| sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ | $(0, 1)$ | output ved binær klassifisering |
| tanh | $\tanh(z)$ | $(-1, 1)$ | skjulte lag (prosjekt 3-klassifisereren) |
| ReLU | $\max(0, z)$ | $[0, \infty)$ | skjulte lag (standardvalget i notatene) |
| GELU | $z \cdot \Phi(z)$ | ca. $[-0.17, \infty)$ | glatt variant av ReLU |
| SiLU | $z \cdot \sigma(z)$ | ca. $[-0.28, \infty)$ | glatt variant av ReLU (prosjekt 3-regressoren) |

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

z = np.linspace(-4, 4, 200)

sigmoid = 1 / (1 + np.exp(-z))
tanh = np.tanh(z)
relu = np.maximum(0, z)
gelu = z * 0.5 * (1 + erf(z / np.sqrt(2)))   # Phi(z) = normalfordelingens CDF
silu = z * sigmoid

for navn, verdi in [("sigmoid", sigmoid), ("tanh", tanh), ("ReLU", relu), ("GELU", gelu), ("SiLU", silu)]:
    plt.plot(z, verdi, label=navn)
plt.axhline(0, color="gray", lw=0.5)
plt.xlabel("z")
plt.ylabel("g(z)")
plt.ylim(-1.5, 4)
plt.legend()
plt.show()
```

Legg merke til at `np.maximum(0, z)` (to argumenter, elementvis) er ReLU. `np.max(z)` er noe helt annet: det største elementet i hele arrayen.

## Softmax: én sannsynlighet per klasse
Ved flere klasser har nettet én output-node per klasse, og **softmax** gjør de $C$ tallene om til sannsynligheter som summerer til 1:

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Den naive koden krasjer på store tall, fordi `np.exp(1000)` er uendelig. Trikset er å trekke fra det største tallet først. Det endrer ikke svaret, fordi $e^{z_i - m} / \sum_j e^{z_j - m} = e^{z_i} / \sum_j e^{z_j}$.

```python
import numpy as np

def softmax_naiv(z):
    return np.exp(z) / np.sum(np.exp(z))

def softmax(z):
    e = np.exp(z - np.max(z))     # største verdi blir e^0 = 1
    return e / np.sum(e)

z = np.array([2.0, 1.0, 0.1])
print(softmax_naiv(z), softmax(z))     # likt

stor = np.array([1000.0, 1001.0, 1002.0])
print(softmax_naiv(stor))              # inf / inf = nan
print(softmax(stor))                   # riktig

# en liten batch: 3 datapunkter, 3 klasser
Z = np.array([[2.0, 1.0, 0.1],
              [0.0, 0.0, 5.0],
              [1.0, 1.0, 1.0]])
P = np.array([softmax(rad) for rad in Z])    # én rad om gangen
print("Logits Z:")
display(Z)
print("Sannsynligheter P:")
display(P)
print("Radsummer:", P.sum(axis=1))
```

Sammenlign `Z` og `P` i utdataene: den største logiten i hver rad får den største sannsynligheten, og like logits (siste rad) gir 1/3 til hver klasse. For en hel batch med shape `(N, C)` gjør du det samme uten løkke med `axis=1` og `keepdims=True`, slik at hver rad summerer til 1.

## Perseptron i scikit-learn
Notatene bruker `sklearn.linear_model.Perceptron`. Den har samme `fit`/`predict` som andre modeller, og vektene ligger i `coef_` og `intercept_`:

```python
from sklearn.datasets import make_blobs
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

X, y = make_blobs(n_samples=200, centers=2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

perceptron = Perceptron().fit(X_train, y_train)
y_pred = perceptron.predict(X_test)
print("Test-accuracy:", perceptron.score(X_test, y_test))

ws = perceptron.coef_
bs = perceptron.intercept_
print("Features:", X.shape[1])
print("Classes:", len(list(set(y))))
print("Weights:", ws.shape)
print("Biases:", bs.shape)
```

Med to klasser trengs bare én output-node, så `coef_` har shape `(1, 2)`. Med tre klasser får vi én node per klasse:

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron

X, y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, n_classes=3, random_state=0)
perceptron = Perceptron().fit(X, y)
print("Features:", X.shape[1])
print("Classes:", len(list(set(y))))
print("Weights:", perceptron.coef_.shape)       # (klasser, features)
print("Biases:", perceptron.intercept_.shape)   # (klasser,)
display(perceptron.coef_)                       # én rad per klasse, én kolonne per feature
```

To features og tre klasser gir $2 \cdot 3 = 6$ vekter og 3 bias-verdier, akkurat som notatene sier. Rutenettet viser vektene med fortegn (rødt og blått), og under `perceptron` i variabelpanelet finner du `coef_` og `intercept_` blant det modellen lærte.

::: eksamen
Eksamen 2025 spurte hva som er den viktigste konsekvensen av å bytte Heaviside med sigmoid. Svaret: nevronet blir **deriverbart** og kan trenes med gradient descent. Output blir ikke lenger bare 0 eller 1.
:::

::: oppgave ReLU og sigmoid
Skriv funksjonene `relu(z)` og `sigmoid(z)`. Begge skal virke elementvis på en hel NumPy-array (ingen løkker).
```start
import numpy as np

def relu(z):
    ...

def sigmoid(z):
    ...
```
```losning
import numpy as np

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```
```test
import numpy as np
z = np.array([-2.0, 0.0, 3.0])
r = relu(z)
assert r is not None, "relu returnerer ingenting. Husk return."
assert np.allclose(r, [0.0, 0.0, 3.0]), f"relu([-2, 0, 3]) skal være [0, 0, 3], fikk {r}. Bruk np.maximum(0, z), ikke np.max."
s = sigmoid(z)
assert s is not None, "sigmoid returnerer ingenting. Husk return."
assert np.allclose(s, [0.11920292, 0.5, 0.95257413]), f"sigmoid(z) skal være 1 / (1 + e^(-z)), fikk {s}"
assert np.shape(s) == (3,), "sigmoid skal returnere en array med samme shape som z"
```
:::

::: oppgave Softmax for en hel batch
Skriv `softmax(Z)` der `Z` har shape `(N, C)` (én rad per datapunkt, én kolonne per klasse). Hver rad i resultatet skal summere til 1, og funksjonen skal tåle store tall som `1000`. Trekk fra maks **per rad** med `axis=1, keepdims=True`.
```start
import numpy as np

def softmax(Z):
    ...
```
```losning
import numpy as np

def softmax(Z):
    E = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return E / np.sum(E, axis=1, keepdims=True)
```
```test
import numpy as np
Z = np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0], [0.0, 0.0, 50.0]])
P = softmax(Z)
assert P is not None, "softmax returnerer ingenting. Husk return."
P = np.asarray(P)
assert P.shape == (3, 3), f"Resultatet skal ha shape (3, 3), fikk {P.shape}. Bruk keepdims=True."
assert not np.isnan(P).any(), "Du får nan for store tall. Trekk fra np.max(Z, axis=1, keepdims=True) før np.exp."
assert np.allclose(P.sum(axis=1), 1), "Hver rad skal summere til 1. Summer langs axis=1."
assert np.allclose(P[0], [0.09003057, 0.24472847, 0.66524096]), f"Første rad er feil: {P[0]}"
assert np.allclose(P[1], [1/3, 1/3, 1/3]), "Tre like logits skal gi 1/3 til hver klasse"
```
:::

::: quiz
Et `Perceptron` trenes på data med 4 features og 3 klasser. Hva skriver koden ut?
```kode
perceptron = Perceptron().fit(X_train, y_train)
print(perceptron.coef_.shape, perceptron.intercept_.shape)
```
- [ ] `(4, 3) (3,)`
- [x] `(3, 4) (3,)`
- [ ] `(3, 4) (4,)`
- [ ] `(12,) (3,)`
? Det er én output-node per klasse, og hver node har én vekt per feature. scikit-learn lagrer det som `(klasser, features)` = `(3, 4)`, og én bias per klasse: `(3,)`.
:::

::: quiz
Hva skriver koden ut?
```kode
z = np.array([-1.0, 0.0, 2.0])
print(np.maximum(0, z).sum(), np.max(z))
```
- [ ] `2.0 [0. 0. 2.]`
- [ ] `1.0 2.0`
- [x] `2.0 2.0`
- [ ] `3.0 2.0`
? `np.maximum(0, z)` er ReLU elementvis: `[0, 0, 2]`, som summerer til 2. `np.max(z)` er det største elementet, også 2.
:::

::: oppsummering
- Et nevron regner ut $z = \mathbf{w} \cdot \mathbf{x} + b$ og så $a = g(z)$.
- Perseptronet bruker Heaviside, som ikke kan deriveres. Sigmoid, tanh, ReLU, GELU og SiLU kan det.
- ReLU er `np.maximum(0, z)`, sigmoid er `1 / (1 + np.exp(-z))`.
- Softmax gir sannsynligheter som summerer til 1. Trekk fra maks først for å unngå overflow.
- `Perceptron().fit(X, y)`: `coef_` har shape `(klasser, features)` (eller `(1, features)` ved to klasser), `intercept_` har shape `(klasser,)`.
:::

--- LEKSJON 9.2 | lag-matriser | Lag som matriser og parametertelling
Et **lag** er mange nevroner side om side som alle får samme input. I stedet for å regne ut hvert nevron for seg, samler vi vektene i en matrise $W$ og regner ut hele laget, for hele batchen, med én matrisemultiplikasjon:

$$H = g(XW + \mathbf{b})$$

| Størrelse | Shape |
|---|---|
| `X` (input, én rad per datapunkt) | `(N, inn)` |
| `W` (vekter) | `(inn, ut)` |
| `b` (bias, én per node) | `(ut,)` |
| `H = g(X @ W + b)` | `(N, ut)` |

Indre dimensjoner må passe: `(N, inn) @ (inn, ut)` gir `(N, ut)`. Biasen legges til hver rad med broadcasting (del 2).

```python
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(5, 2))          # 5 datapunkter, 2 features
W = rng.normal(size=(2, 3))          # 2 input -> 3 noder
b = np.zeros(3)

Z = X @ W + b
print("Før aktivering, Z = X @ W + b:")
display(Z)
H = np.tanh(Z)
print("Etter tanh, H:")
display(H)
print("X:", X.shape, " W:", W.shape, " H:", H.shape)
```

Hver rad er ett datapunkt og hver kolonne én node i laget. Legg merke til at tanh beholder fortegnet, men klemmer alle verdiene inn mellom -1 og 1.

Notatene skriver det samme for ett datapunkt som en kolonnevektor: $\mathbf{a} = g(W\mathbf{x} + \mathbf{b})$ med $W$ av shape `(3, 2)`. Det er samme regning, bare transponert. Det er derfor PyTorch lagrer vektene som `(ut, inn)` (leksjon 9.6) mens Keras og vår NumPy-kode bruker `(inn, ut)`.

## Forward pass gjennom et helt nett
Et nett er bare lag etter lag: output fra ett lag er input til neste. Her er arkitekturen til klassifisereren i prosjekt 3, 2 → 16 → 16 → 1, med tilfeldige vekter:

```python
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(8, 2))                          # batch med 8 punkter

W1, b1 = rng.normal(size=(2, 16)), np.zeros(16)
W2, b2 = rng.normal(size=(16, 16)), np.zeros(16)
W3, b3 = rng.normal(size=(16, 1)), np.zeros(1)

H1 = np.tanh(X @ W1 + b1)
H2 = np.tanh(H1 @ W2 + b2)
logit = H2 @ W3 + b3          # ingen aktivering i siste lag
print(H1.shape, H2.shape, logit.shape)
```

Åpne `H1`, `H2` og `logit` i variabelpanelet under resultatet. Batch-dimensjonen (8 rader) er den samme hele veien, mens antall kolonner følger antall noder: 16, 16 og til slutt 1. `W2` med shape `(16, 16)` vises som et heatmap.

Siste lag har **ingen** aktivering. Output kalles en **logit**, og sigmoid legges på senere (i tapsfunksjonen eller ved prediksjon, se leksjon 9.7). Ved regresjon skal output kunne være et hvilket som helst tall, og da skal siste lag også være lineært.

## Telle parametre
Et lag fra `inn` til `ut` noder har `inn · ut` vekter og `ut` bias-verdier, altså `inn · ut + ut` parametre.

```python
def antall_parametre(storrelser):
    total = 0
    for inn, ut in zip(storrelser[:-1], storrelser[1:]):
        print(f"  {inn:>3} -> {ut:<3}: {inn} * {ut} + {ut} = {inn * ut + ut}")
        total += inn * ut + ut
    return total

print("Klassifisereren i prosjekt 3:")
print("Totalt:", antall_parametre([2, 16, 16, 1]))
print("Regressoren i prosjekt 3 (per nett i ensemblet):")
print("Totalt:", antall_parametre([2, 128, 128, 128, 1]))
```

`zip(storrelser[:-1], storrelser[1:])` gir nabopar: `(2, 16)`, `(16, 16)`, `(16, 1)`. Det er et nyttig mønster å kjenne igjen.

::: prosjekt Prosjekt 3, problem 1
Prosjektet spør etter parametertallet for begge nettene. Klassifisereren har 337 parametre (48 + 272 + 17), og hver regressor har 33 537. I PyTorch får du det samme tallet med `sum(p.numel() for p in net.parameters())` (leksjon 9.6).
:::

## Hvorfor trenger vi ikke-lineær aktivering?
Uten aktivering er to lag bare $(XW_1)W_2 = X(W_1W_2)$, altså ett lineært lag. Uansett hvor mange lineære lag du stabler, kan nettet bare lære en lineær funksjon, og da kan det aldri løse XOR.

```python
import numpy as np

rng = np.random.default_rng(1)
X = rng.normal(size=(4, 2))
W1 = rng.normal(size=(2, 16))
W2 = rng.normal(size=(16, 1))

to_lag = (X @ W1) @ W2
ett_lag = X @ (W1 @ W2)               # W1 @ W2 har shape (2, 1)
print(np.allclose(to_lag, ett_lag))    # True: samme funksjon

med_relu = np.maximum(0, X @ W1) @ W2
print(np.allclose(med_relu, ett_lag))  # False: ReLU gjør nettet ikke-lineært
```

::: oppgave Tell parametrene
Skriv `antall_parametre(storrelser)` som tar en liste med lagstørrelser (input først, output sist) og returnerer totalt antall vekter og bias-verdier.
```start
def antall_parametre(storrelser):
    total = 0
    # gå gjennom nabopar (inn, ut)
    ...
    return total
```
```losning
def antall_parametre(storrelser):
    total = 0
    for inn, ut in zip(storrelser[:-1], storrelser[1:]):
        total += inn * ut + ut
    return total
```
```test
assert antall_parametre([3, 6]) == 24, f"Ett lag 3 -> 6 skal ha 3*6 + 6 = 24 parametre, fikk {antall_parametre([3, 6])}"
assert antall_parametre([2, 16, 16, 1]) == 337, f"2 -> 16 -> 16 -> 1 skal gi 337, fikk {antall_parametre([2, 16, 16, 1])}. Husk bias (+ ut) for hvert lag."
assert antall_parametre([2, 128, 128, 128, 1]) == 33537, "2 -> 128 -> 128 -> 128 -> 1 skal gi 33537"
assert antall_parametre([784, 128, 64, 10]) == 109386, "784 -> 128 -> 64 -> 10 (prosjekt 4) skal gi 109386"
```
:::

::: oppgave Forward pass
Skriv `forward(X, W1, b1, W2, b2)` for et nett med ett skjult lag: ReLU i det skjulte laget og **ingen** aktivering i output-laget. Returner output.
```start
import numpy as np

def forward(X, W1, b1, W2, b2):
    H = ...
    return ...
```
```losning
import numpy as np

def forward(X, W1, b1, W2, b2):
    H = np.maximum(0, X @ W1 + b1)
    return H @ W2 + b2
```
```test
import numpy as np
X = np.array([[1.0, 2.0], [-1.0, 0.5], [0.0, 0.0]])
W1 = np.array([[1.0, -1.0, 0.5], [0.5, 1.0, -1.0]])
b1 = np.array([0.0, 0.5, 0.1])
W2 = np.array([[1.0], [2.0], [-1.0]])
b2 = np.array([0.5])
ut = forward(X, W1, b1, W2, b2)
assert ut is not None, "forward returnerer ingenting"
ut = np.asarray(ut)
assert ut.shape == (3, 1), f"Output skal ha shape (3, 1), fikk {ut.shape}"
assert np.allclose(ut, [[5.5], [4.5], [1.4]]), f"Feil verdier: {ut.ravel()}. ReLU i skjult lag, lineært output-lag."
```
:::

::: quiz
En batch `X` har shape `(32, 10)`. Nettet har lagene 10 → 64 → 16 → 3 (vekter som `(inn, ut)`). Hva er shape på output fra det **andre** laget (16 noder)?
- [x] `(32, 16)`
- [ ] `(16,)`
- [ ] `(10, 16)`
- [ ] `(64, 16)`
? Hvert lag beholder batch-dimensjonen og bytter feature-dimensjonen med antall noder: `(32, 10) → (32, 64) → (32, 16) → (32, 3)`.
:::

::: quiz
Et nett har lagene 2 → 4 → 1 og **ingen** aktiveringsfunksjoner. Hva kan det lære?
- [ ] Alt et nett med ReLU kan lære, bare tregere
- [ ] XOR, fordi det har et skjult lag
- [x] Bare lineære funksjoner av input, akkurat som ett enkelt lineært lag
- [ ] Ingenting, fordi output alltid blir 0
? $(XW_1 + b_1)W_2 + b_2 = X(W_1W_2) + (b_1W_2 + b_2)$ er fortsatt lineært. Det er den ikke-lineære aktiveringen som gjør at flere lag gir mer uttrykkskraft.
:::

::: oppsummering
- Et lag er `H = g(X @ W + b)` med shapes `(N, inn) @ (inn, ut) → (N, ut)`.
- Et nett er lag etter lag. Siste lag har ingen aktivering (logit eller regresjonsverdi).
- Parametre per lag: `inn · ut + ut`. Prosjekt 3: 337 for 2 → 16 → 16 → 1, og 33 537 for 2 → 128 → 128 → 128 → 1.
- Uten ikke-lineær aktivering kollapser mange lag til ett lineært lag.
:::

--- LEKSJON 9.3 | backprop-numpy | Backpropagation i NumPy
For å trene nettet trenger vi gradienten av tapet med hensyn på **alle** vektene. **Backpropagation** regner den ut med kjerneregelen, bakover fra output: først gradienten for siste lag, så sendes den videre til laget før, og så videre. Her gjør vi det for hånd for et lite nett som løser XOR. I PyTorch skjer akkurat dette automatisk med `loss.backward()`.

Nettet er 2 → 4 → 1 med tanh i det skjulte laget og sigmoid + binær cross-entropy på output:

| Forward | Backward (kjerneregelen) |
|---|---|
| `Z1 = X @ W1 + b1` | `dW1 = X.T @ dZ1`, `db1 = dZ1.sum(axis=0)` |
| `H = np.tanh(Z1)` | `dZ1 = dH * (1 - H**2)` (fordi $\tanh'(z) = 1 - \tanh^2(z)$) |
| `z2 = H @ W2 + b2` | `dW2 = H.T @ dz2`, `db2 = dz2.sum(axis=0)`, `dH = dz2 @ W2.T` |
| `p = sigmoid(z2)`, `L = BCE(p, y)` | `dz2 = (p - y) / N` |

Det siste steget kjenner du fra logistisk regresjon i del 5: sigmoid og cross-entropy sammen gir den enkle gradienten $p - y$. Hver gradient har samme shape som parameteren den hører til.

```python
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)     # XOR
N = len(X)

rng = np.random.default_rng(1)
W1, b1 = rng.normal(size=(2, 4)), np.zeros(4)
W2, b2 = rng.normal(size=(4, 1)), np.zeros(1)
lr = 0.5
print("W1 før trening:")
display(W1)

for it in range(2001):
    # forward
    H = np.tanh(X @ W1 + b1)
    p = 1 / (1 + np.exp(-(H @ W2 + b2)))
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    # backward
    dz2 = (p - y) / N
    dW2, db2 = H.T @ dz2, dz2.sum(axis=0)
    dH = dz2 @ W2.T
    dZ1 = dH * (1 - H**2)
    dW1, db1 = X.T @ dZ1, dZ1.sum(axis=0)
    # oppdater
    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2
    if it % 500 == 0:
        print(f"iterasjon {it:4d}: tap {loss:.4f}")

print("W1 etter trening:")
display(W1)
print("Aktiveringer i det skjulte laget (én rad per XOR-punkt):")
display(H)
print("Prediksjoner:", p.ravel().round(3))
```

Sannsynlighetene ender nær `[0, 1, 1, 0]`: nettet har lært XOR, som én enkelt lineær modell ikke kan. Sammenlign `W1` før og etter: vektene har blitt mye større i absoluttverdi. I aktiveringene ser du hvordan de fire punktene får ulike mønstre i det skjulte laget, slik at output-laget kan skille dem med en rett linje.

::: tips
Bytt `default_rng(1)` til `default_rng(0)` og kjør igjen. Da blir tapet stående på ca. 0.35 og to av punktene får 0.5. Nettet har satt seg fast i et dårlig lokalt minimum. Tapsflaten til et nevralt nett er ikke konveks, så startvektene (og dermed seeden) betyr noe. Det er en grunn til at prosjekt 3 krever faste seeds.
:::

## Gradientsjekk
Hvordan vet vi at backward-koden er riktig? Vi kan sammenligne med en **numerisk** derivert: endre én vekt litt opp og litt ned og se hvor mye tapet endrer seg, $\frac{L(w+h) - L(w-h)}{2h}$. Det er for tregt til trening (ett ekstra forward pass per vekt), men perfekt som test.

```python
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)
rng = np.random.default_rng(1)
W1, b1 = rng.normal(size=(2, 4)), np.zeros(4)
W2, b2 = rng.normal(size=(4, 1)), np.zeros(1)

def tap(W1):
    H = np.tanh(X @ W1 + b1)
    p = 1 / (1 + np.exp(-(H @ W2 + b2)))
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

# analytisk gradient med backprop
H = np.tanh(X @ W1 + b1)
p = 1 / (1 + np.exp(-(H @ W2 + b2)))
dz2 = (p - y) / len(X)
dZ1 = (dz2 @ W2.T) * (1 - H**2)
dW1 = X.T @ dZ1

# numerisk gradient for vekten W1[0, 2]
h = 1e-5
W_pluss, W_minus = W1.copy(), W1.copy()
W_pluss[0, 2] += h
W_minus[0, 2] -= h
numerisk = (tap(W_pluss) - tap(W_minus)) / (2 * h)
print("backprop:", dW1[0, 2])
print("numerisk:", numerisk)
```

::: oppgave Kjerneregelen for et bitte lite nett
Et nett med én input, én skjult node og én output: $h = \tanh(w_1 x)$, $\hat{y} = w_2 h$, og tapet er $L = (\hat{y} - t)^2$.

Regn ut gradientene med kjerneregelen og lagre dem i `dw1` og `dw2`:
- $\frac{\partial L}{\partial w_2} = 2(\hat{y} - t) \cdot h$
- $\frac{\partial L}{\partial w_1} = 2(\hat{y} - t) \cdot w_2 \cdot (1 - h^2) \cdot x$
```start
import numpy as np

x, t = 1.5, 0.5
w1, w2 = 0.8, -1.2

# forward
h = np.tanh(w1 * x)
y_hat = w2 * h

# backward
dw2 = ...
dw1 = ...
```
```losning
import numpy as np

x, t = 1.5, 0.5
w1, w2 = 0.8, -1.2

h = np.tanh(w1 * x)
y_hat = w2 * h

dL_dyhat = 2 * (y_hat - t)
dw2 = dL_dyhat * h
dw1 = dL_dyhat * w2 * (1 - h**2) * x
```
```test
import numpy as np
def _L(a, b):
    return (b * np.tanh(a * 1.5) - 0.5) ** 2
_e = 1e-6
_num_w1 = (_L(0.8 + _e, -1.2) - _L(0.8 - _e, -1.2)) / (2 * _e)
_num_w2 = (_L(0.8, -1.2 + _e) - _L(0.8, -1.2 - _e)) / (2 * _e)
assert isinstance(dw2, (int, float, np.floating)), "dw2 skal være et tall"
assert isinstance(dw1, (int, float, np.floating)), "dw1 skal være et tall"
assert abs(dw2 - _num_w2) < 1e-5, f"dw2 er feil: fikk {dw2:.5f}, numerisk sjekk gir {_num_w2:.5f}"
assert abs(dw1 - _num_w1) < 1e-5, f"dw1 er feil: fikk {dw1:.5f}, numerisk sjekk gir {_num_w1:.5f}. Husk alle leddene: 2(ŷ - t), w2, (1 - h²) og x."
```
:::

::: quiz
I backward-koden over: hvilken linje sender gradienten fra output-laget **bakover** til det skjulte laget?
- [ ] `dW2 = H.T @ dz2`
- [x] `dH = dz2 @ W2.T`
- [ ] `dz2 = (p - y) / N`
- [ ] `W2 -= lr * dW2`
? `dW2` er gradienten for vektene i siste lag. `dH = dz2 @ W2.T` er hvor mye tapet endrer seg når aktiveringene i det skjulte laget endres. Den brukes videre til `dZ1` og `dW1`. Det er dette som er «propagering bakover».
:::

::: quiz
Hva er hovedformålet med backpropagation (eksamen 2025)?
- [x] Å beregne hvordan endringer i hver vekt påvirker tapet, med kjerneregelen bakover gjennom nettet
- [ ] Å beregne gradienten numerisk ved å endre hver vekt litt og måle endringen i tapet
- [ ] Å gjøre prediksjoner ved å sende data fra input til output
- [ ] Å fjerne skjulte lag som ikke forbedrer accuracy
? Backprop er en effektiv måte å regne ut gradienten for alle vektene på én gang. Den numeriske metoden er en gradientsjekk, ikke backprop. Å sende data fremover er forward pass.
:::

::: oppsummering
- Backprop = kjerneregelen bakover lag for lag. Hver gradient har samme shape som parameteren.
- For et lag `Z = A @ W + b`: `dW = A.T @ dZ`, `db = dZ.sum(axis=0)` og `dA = dZ @ W.T`.
- Deriverte av aktiveringer: tanh gir `1 - H**2`, sigmoid + cross-entropy gir `p - y`.
- Gradientsjekk: sammenlign med $\frac{L(w+h) - L(w-h)}{2h}$.
- Startvektene betyr noe: tapsflaten er ikke konveks.
:::

--- LEKSJON 9.4 | sklearn-mlp | MLPClassifier og MLPRegressor
Før vi går til PyTorch: scikit-learn har ferdige nevrale nett med vanlig `fit`/`predict`. `MLPClassifier` gjør alt fra forrige leksjon for deg (forward, backprop, Adam, mini-batcher og early stopping). Det er en fin måte å se hva arkitektur og hyperparametre betyr før du skriver treningsløkka selv.

```python
import pandas as pd
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("classification_data.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
X_train, y_train = train[["x0", "x1"]].values, train["y"].values
X_test, y_test = test[["x0", "x1"]].values, test["y"].values

mlp = MLPClassifier(
    hidden_layer_sizes=(16, 16),   # to skjulte lag: 2 -> 16 -> 16 -> 1
    activation="tanh",
    solver="adam",
    learning_rate_init=0.01,
    max_iter=1000,
    early_stopping=True,           # holder av 10 % av train til validering
    n_iter_no_change=20,           # patience
    random_state=0,
)
mlp.fit(X_train, y_train)

print("Test-accuracy:", mlp.score(X_test, y_test))
print("Epoker trent:", mlp.n_iter_)
print("Vekt-shapes:", [W.shape for W in mlp.coefs_])
print("Bias-shapes:", [b.shape for b in mlp.intercepts_])

print("Vektene i første lag, coefs_[0] (2 input × 16 noder):")
display(mlp.coefs_[0])
print("predict_proba for de 5 første testpunktene:")
display(mlp.predict_proba(X_test[:5]))
```

Merk at `coefs_` bruker `(inn, ut)`, akkurat som vår NumPy-kode. Teller du opp, får du 337 parametre også her. Under `mlp` i variabelpanelet ser du alt modellen lærte (`coefs_`, `intercepts_`, `loss_curve_` …). `predict_proba` gir én rad per punkt og én kolonne per klasse, og hver rad summerer til 1.

| Parameter | Betydning |
|---|---|
| `hidden_layer_sizes=(16, 16)` | bare de **skjulte** lagene. Input og output bestemmes av dataene |
| `activation` | `"relu"` (standard), `"tanh"`, `"logistic"` (sigmoid), `"identity"` |
| `solver` | `"adam"` (standard), `"sgd"`, `"lbfgs"` |
| `learning_rate_init` | læringsraten |
| `max_iter` | maks antall epoker |
| `early_stopping=True` | stopp når valideringsscore ikke bedres på `n_iter_no_change` epoker |
| `random_state` | seed for startvekter og mini-batcher |

## Tapskurve
`mlp.loss_curve_` er treningstapet etter hver epoke, og med `early_stopping=True` ligger accuracy på valideringsdelen i `mlp.validation_scores_`:

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("classification_data.csv")
train = data[data["split"] == "train"]
mlp = MLPClassifier(hidden_layer_sizes=(16, 16), activation="tanh", learning_rate_init=0.01,
                    max_iter=1000, early_stopping=True, n_iter_no_change=20, random_state=0)
mlp.fit(train[["x0", "x1"]].values, train["y"].values)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(mlp.loss_curve_)
ax[0].set(title="Treningstap (log loss)", xlabel="epoke")
ax[1].plot(mlp.validation_scores_)
ax[1].set(title="Validerings-accuracy", xlabel="epoke")
plt.show()
```

## Beslutningsgrensen
Med `meshgrid` og `contourf` fra leksjon 4.6 ser vi hva nettet har lært. XOR-mønsteret gir en grense langs de to aksene:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("classification_data.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
mlp = MLPClassifier(hidden_layer_sizes=(16, 16), activation="tanh", learning_rate_init=0.01,
                    max_iter=1000, early_stopping=True, n_iter_no_change=20, random_state=0)
mlp.fit(train[["x0", "x1"]].values, train["y"].values)

xx, yy = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-4, 4, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
p = mlp.predict_proba(grid)[:, 1].reshape(xx.shape)

plt.contourf(xx, yy, p, levels=20, cmap="coolwarm", alpha=0.6)
plt.contour(xx, yy, p, levels=[0.5], colors="k")
plt.scatter(test["x0"], test["x1"], c=test["y"], cmap="coolwarm", s=10, edgecolor="k", linewidth=0.3)
plt.title("MLPClassifier på prosjekt 3-dataene")
plt.show()
```

## Skalering
Nevrale nett trenes med gradient descent, så features på helt ulik skala gjør treningen vanskelig (del 5 og 7). Her ganger vi den ene featuren med 1000. Uten skalering blir nettet nesten ubrukelig, med `StandardScaler` i en `Pipeline` er alt som før:

```python
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("classification_data.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
skala = np.array([1, 1000])                      # x1 måles nå i «millimeter»
X_train = train[["x0", "x1"]].values * skala
X_test = test[["x0", "x1"]].values * skala

def ny_mlp():
    return MLPClassifier(hidden_layer_sizes=(16, 16), activation="tanh", learning_rate_init=0.01,
                         max_iter=1000, early_stopping=True, n_iter_no_change=20, random_state=0)

uten = ny_mlp().fit(X_train, train["y"])
med = make_pipeline(StandardScaler(), ny_mlp()).fit(X_train, train["y"])
print("Uten skalering:", uten.score(X_test, test["y"]))
print("Med skalering: ", med.score(X_test, test["y"]))
```

## MLPRegressor
For regresjon bruker du `MLPRegressor`. Output-laget er da lineært, og tapet er kvadratisk feil. Testradene i `regression_data.csv` har ingen `y`, men vi vet at dataene er laget fra $y = x_0^3 - 3x_0x_1^2$ pluss støy. Vi kan derfor regne RMSE mot den sanne funksjonen, slik `plot_complex_fit` gjør i prosjektet:

```python
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

data = pd.read_csv("regression_data.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
X_train, y_train = train[["x0", "x1"]].values, train["y"].values
X_test = test[["x0", "x1"]].values

reg = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu",
                   learning_rate_init=0.01, max_iter=2000, random_state=0)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
y_sann = X_test[:, 0]**3 - 3 * X_test[:, 0] * X_test[:, 1]**2   # uten støy
rmse = np.sqrt(np.mean((y_pred - y_sann) ** 2))
print("RMSE mot sann funksjon:", round(rmse, 3))
```

::: prosjekt Prosjekt 3, problem 1
Kravene var minst 90 % test-accuracy for klassifisering og maks 0.25 RMSE for regresjon, med Keras eller PyTorch. `MLPClassifier` når 90 % her, men regressoren over er langt unna 0.25. Løsningen i prosjektet trengte det PyTorch gir deg: SiLU-aktivering, cosine annealing, early stopping per modell og et ensemble av 5 nett (0.215). Det bygger vi opp mot i resten av denne delen.
:::

::: oppgave Ditt eget MLP
Tren en `MLPClassifier` på prosjekt 3-dataene (allerede lastet inn) med to skjulte lag på 16 noder og `random_state=0`. Velg aktivering og læringsrate selv. Lagre modellen i `mlp` og test-accuracy i `acc`. Målet er minst 0.85.
```start
import pandas as pd
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("classification_data.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
X_train, y_train = train[["x0", "x1"]].values, train["y"].values
X_test, y_test = test[["x0", "x1"]].values, test["y"].values

mlp = ...
acc = ...
```
```losning
import pandas as pd
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("classification_data.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
X_train, y_train = train[["x0", "x1"]].values, train["y"].values
X_test, y_test = test[["x0", "x1"]].values, test["y"].values

mlp = MLPClassifier(hidden_layer_sizes=(16, 16), activation="relu", learning_rate_init=0.01,
                    max_iter=1000, random_state=0)
mlp.fit(X_train, y_train)
acc = mlp.score(X_test, y_test)
```
```test
from sklearn.neural_network import MLPClassifier
assert isinstance(mlp, MLPClassifier), "mlp skal være en MLPClassifier"
assert hasattr(mlp, "coefs_"), "mlp er ikke trent. Kall mlp.fit(X_train, y_train)."
assert tuple(mlp.hidden_layer_sizes) == (16, 16), f"Bruk hidden_layer_sizes=(16, 16), du har {mlp.hidden_layer_sizes}"
assert isinstance(acc, float), "acc skal være et tall, f.eks. mlp.score(X_test, y_test)"
assert abs(acc - mlp.score(X_test, y_test)) < 1e-9, "acc skal være accuracy på TESTdataene"
assert acc >= 0.85, f"Test-accuracy er {acc:.3f}. Prøv en annen læringsrate eller aktivering."
```
:::

::: quiz
Du trener `MLPClassifier(hidden_layer_sizes=(8,))` på data med 5 features og 3 klasser. Hva gir `[W.shape for W in mlp.coefs_]`?
- [ ] `[(8, 5), (3, 8)]`
- [x] `[(5, 8), (8, 3)]`
- [ ] `[(5, 8)]`
- [ ] `[(5, 8), (8, 1)]`
? `hidden_layer_sizes` angir bare de skjulte lagene. Input (5) og output (3 klasser) kommer fra dataene, så nettet er 5 → 8 → 3, og scikit-learn lagrer vektene som `(inn, ut)`.
:::

::: quiz
Hva inneholder `mlp.loss_curve_` etter `fit`?
- [ ] Valideringstapet etter hver epoke
- [x] Treningstapet etter hver epoke (én verdi per iterasjon over treningsdataene)
- [ ] Test-accuracy etter hver epoke
- [ ] Tapet for hver mini-batch i siste epoke
? `loss_curve_` er treningstapet per epoke. Med `early_stopping=True` lagres accuracy på valideringsdelen i `validation_scores_`. Testdata ser modellen aldri under `fit`.
:::

::: oppsummering
- `MLPClassifier`/`MLPRegressor` er ferdige nevrale nett med `fit`/`predict`/`score`.
- `hidden_layer_sizes=(16, 16)` angir bare de skjulte lagene.
- `coefs_` og `intercepts_` er vektene, med shape `(inn, ut)`.
- `loss_curve_` er treningstapet per epoke, `early_stopping=True` holder av validering.
- Skaler input (`StandardScaler` i en `Pipeline`) før du trener nevrale nett.
:::

--- LEKSJON 9.5 | torch-tensorer | PyTorch: tensorer og autograd
Fra nå av bruker vi **PyTorch**, som prosjekt 3 og 4 gjør. PyTorch finnes ikke i nettleseren, så eksemplene vises med ferdige utdata. Kjør dem gjerne selv i Jupyter (`pip install torch`). Oppgavene løser du i NumPy, med samme idé.

En **tensor** er PyTorchs versjon av en NumPy-array. Det meste ser kjent ut: `.shape`, indeksering, `@`, `+`, `.mean()`. Forskjellen er at en tensor kan huske hvordan den ble regnet ut, slik at PyTorch kan derivere for deg.

```python statisk
import numpy as np
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
print(x)
print(x.shape, x.dtype)
print(x @ x.T)
print(x.mean(), x.mean().item())    # .item() gir et vanlig Python-tall
print(x.numpy())                     # tilbake til NumPy
```
```output
tensor([[1., 2.],
        [3., 4.]])
torch.Size([2, 2]) torch.float32
tensor([[ 5., 11.],
        [11., 25.]])
tensor(2.5000) 2.5
[[1. 2.]
 [3. 4.]]
```

## dtype: float32 og long
Nevrale nett regner i `torch.float32`. NumPy bruker `float64` som standard, og `torch.tensor` beholder den typen. Derfor står `dtype=torch.float32` overalt i prosjektkoden. Klasse-labels til `CrossEntropyLoss` skal derimot være heltall, `torch.long`:

```python statisk
import numpy as np
import pandas as pd
import torch

train = pd.read_csv("classification_data.csv").query("split == 'train'")

feil = torch.tensor(train[["x0", "x1"]].values)                       # float64
X = torch.tensor(train[["x0", "x1"]].values, dtype=torch.float32)      # som i prosjekt 3
y = torch.tensor(train["y"].values, dtype=torch.float32)               # BCE: float
y_klasse = torch.tensor(train["y"].values.astype(int), dtype=torch.long)  # CrossEntropy: long (= int64)

print(feil.dtype, X.dtype, y.dtype, y_klasse.dtype)
print(X.shape, y.shape)
print(y_klasse[:5])

a = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(a)    # deler minne med a (ingen kopi)
a[0] = 100.0
print(t)                   # endret seg også!
```
```output
torch.float64 torch.float32 torch.float32 torch.int64
torch.Size([500, 2]) torch.Size([500])
tensor([0, 1, 0, 1, 0])
tensor([100.,   2.,   3.], dtype=torch.float64)
```

## Endre shape: reshape, view, squeeze og unsqueeze
Nettet gir output med shape `(N, 1)`, mens targets ofte har shape `(N,)`. Da må du fjerne eller legge til en dimensjon av lengde 1. Det er en av de vanligste kildene til feil:

```python statisk
import torch

t = torch.arange(6.0)
print(t.shape)
print(t.reshape(2, 3).shape, t.view(3, 2).shape)
print(t.reshape(-1, 1).shape)     # -1: «regn ut selv»

out = torch.zeros(4, 1)            # som net(X) med ett output
print(out.squeeze(1).shape)        # fjern dimensjon 1 -> (4,)
y = torch.zeros(4)
print(y.unsqueeze(1).shape)        # legg til dimensjon 1 -> (4, 1)
```
```output
torch.Size([6])
torch.Size([2, 3]) torch.Size([3, 2])
torch.Size([6, 1])
torch.Size([4])
torch.Size([4, 1])
```

`view` krever at tensoren ligger sammenhengende i minnet, `reshape` virker alltid. I NumPy heter det samme `reshape`, `np.squeeze(a, axis=1)` og `a[:, None]` (eller `np.expand_dims`).

## Autograd: requires_grad, backward og grad
Sett `requires_grad=True`, så husker PyTorch alle regneoperasjonene. `.backward()` på et tall (tapet) regner ut den deriverte med hensyn på alle slike tensorer, og legger den i `.grad`:

```python statisk
import torch

x = torch.tensor(2.0, requires_grad=True)
f = x**2 + 3 * x          # f'(x) = 2x + 3
f.backward()
print(f.item(), x.grad)   # f'(2) = 7
```
```output
10.0 tensor(7.)
```

Det samme for en lineær modell. Sammenlign med gradientene du regnet ut for hånd i del 5:

```python statisk
import torch

X = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])
w = torch.tensor([[0.5]], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

loss = ((X @ w + b - y) ** 2).mean()   # MSE
loss.backward()
print("loss:", loss.item())
print("dL/dw:", w.grad)
print("dL/db:", b.grad)

# for hånd: dL/dw = 2/N * sum((wx + b - y) * x)
feil = X @ w.detach() + b.detach() - y
print("for hånd:", (2 * feil * X).mean().item(), (2 * feil).mean().item())
```
```output
loss: 10.5
dL/dw: tensor([[-14.]])
dL/db: tensor([-6.])
for hånd: -14.0 -6.0
```

## no_grad og detach
Når du bare skal predikere eller regne ut en metrikk, trenger du ikke gradienter. `with torch.no_grad():` slår av sporingen (raskere og mindre minne). `.detach()` gir en kopi av en tensor som er koblet fra regnegrafen. Den må du bruke før `.numpy()` på en tensor som krever gradient:

```python statisk
import torch

w = torch.tensor([1.0, 2.0], requires_grad=True)
z = w * 3
print(z.requires_grad)

with torch.no_grad():
    z2 = w * 3
print(z2.requires_grad)

try:
    z.numpy()
except RuntimeError as e:
    print("RuntimeError:", e)
print(z.detach().numpy())
```
```output
True
False
RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
[3. 6.]
```

::: advarsel Ikke gå via NumPy midt i beregningen
Notatene (deep Q-learning) sier det tydelig: konverterer du en tensor til NumPy underveis, mister PyTorch oversikten over den, og gradientene forsvinner. Tapet er da ikke lenger koblet til vektene, og `loss.backward()` feiler (eller verre: vektene får ingen gradient).
:::

```python statisk
import numpy as np
import torch

w = torch.tensor([1.0, 2.0], requires_grad=True)
pred = w * 3
pred_np = pred.detach().numpy()             # ut av PyTorch ...
loss = torch.tensor(np.sum(pred_np ** 2))   # ... og inn igjen
try:
    loss.backward()
except RuntimeError as e:
    print("RuntimeError:", e)

loss = torch.sum(pred ** 2)                 # hele veien i PyTorch
loss.backward()
print(w.grad)
```
```output
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
tensor([18., 36.])
```

::: oppgave Shapes som i PyTorch, bare i NumPy
`out` er output fra et nett med shape `(5, 1)`, og `y` er targets med shape `(5,)`. Lag
- `out_flat`: `out` med shape `(5,)` (som `out.squeeze(1)` i PyTorch)
- `y_kol`: `y` med shape `(5, 1)` (som `y.unsqueeze(1)`)
- `mse`: riktig MSE mellom `out` og `y`, regnet ut med like shapes.
```start
import numpy as np

out = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([1.0, 2.0, 3.0, 4.0, 6.0])

out_flat = ...
y_kol = ...
mse = ...
```
```losning
import numpy as np

out = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([1.0, 2.0, 3.0, 4.0, 6.0])

out_flat = np.squeeze(out, axis=1)
y_kol = y[:, None]
mse = np.mean((out_flat - y) ** 2)
```
```test
import numpy as np
assert np.shape(out_flat) == (5,), f"out_flat skal ha shape (5,), har {np.shape(out_flat)}. Bruk np.squeeze(out, axis=1) eller out[:, 0]."
assert np.shape(y_kol) == (5, 1), f"y_kol skal ha shape (5, 1), har {np.shape(y_kol)}. Bruk y[:, None] eller y.reshape(-1, 1)."
assert np.allclose(np.ravel(out_flat), [1, 2, 3, 4, 5]) and np.allclose(np.ravel(y_kol), y), "Verdiene skal ikke endres, bare shapen"
assert abs(mse - 0.2) < 1e-9, f"mse skal være 0.2, fikk {mse}. Med shapes (5, 1) og (5,) gir broadcasting en (5, 5)-matrise og feil svar."
```
:::

::: oppgave Numerisk derivert
Autograd regner ut eksakte deriverte. Skriv `num_grad(f, x, h=1e-5)` som tilnærmer $f'(x)$ med sentraldifferansen $\frac{f(x+h) - f(x-h)}{2h}$, så du kan sjekke autograd selv.
```start
def num_grad(f, x, h=1e-5):
    ...
```
```losning
def num_grad(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)
```
```test
import numpy as np
g = num_grad(lambda x: x**2 + 3 * x, 2.0)
assert g is not None, "num_grad returnerer ingenting"
assert abs(g - 7.0) < 1e-6, f"f(x) = x² + 3x har f'(2) = 7, fikk {g}"
assert abs(num_grad(np.tanh, 0.5) - (1 - np.tanh(0.5)**2)) < 1e-8, "Sjekk formelen: (f(x+h) - f(x-h)) / (2h)"
```
:::

::: quiz
Hva skriver koden ut?
```kode
a = np.array([0.5, 1.5])
t = torch.tensor(a)
print(t.dtype)
```
- [ ] `torch.float32`
- [x] `torch.float64`
- [ ] `torch.long`
- [ ] `torch.int64`
? NumPy bruker `float64` som standard, og `torch.tensor` beholder typen. Derfor skriver prosjektkoden `dtype=torch.float32`. Ellers får du feil når float64-data møter float32-vekter i nettet.
:::

::: quiz
Hva skriver koden ut?
```kode
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)
```
- [ ] `tensor(9.)`
- [x] `tensor(6.)`
- [ ] `tensor(3.)`
- [ ] `None`
? `y = x²` har derivert `2x`, som er 6 i `x = 3`. `backward()` legger den deriverte i `x.grad`.
:::

::: oppsummering
- `torch.tensor(data, dtype=torch.float32)` for features, `torch.long` for klasse-labels til `CrossEntropyLoss`.
- `.shape`, `.item()` (tall), `.numpy()` (array), `torch.from_numpy` (deler minne).
- `reshape`/`view` endrer shape, `squeeze(1)` fjerner og `unsqueeze(1)` legger til en dimensjon av lengde 1.
- `requires_grad=True` + `loss.backward()` fyller `.grad` med den deriverte.
- `torch.no_grad()` ved prediksjon, `.detach()` før `.numpy()`. Gå aldri via NumPy midt i en beregning du skal derivere.
:::

--- LEKSJON 9.6 | torch-moduler | nn.Linear, nn.Sequential og nn.Module
`torch.nn` har ferdige byggeklosser. Det viktigste er `nn.Linear(inn, ut)`, et fullt koblet lag med vekter og bias, og aktiveringer som `nn.Tanh()`, `nn.ReLU()`, `nn.GELU()` og `nn.SiLU()`.

```python statisk
import torch
import torch.nn as nn

torch.manual_seed(0)
lag = nn.Linear(2, 16)
print(lag)
print("weight:", lag.weight.shape)   # (ut, inn)!
print("bias:  ", lag.bias.shape)

X = torch.randn(5, 2)
H = lag(X)
print("output:", H.shape)
print(torch.allclose(H, X @ lag.weight.T + lag.bias))
```
```output
Linear(in_features=2, out_features=16, bias=True)
weight: torch.Size([16, 2])
bias:   torch.Size([16])
output: torch.Size([5, 16])
True
```

::: advarsel weight har shape (ut, inn)
`nn.Linear(2, 16).weight` har shape `(16, 2)`, og laget regner `X @ W.T + b`. Keras og `MLPClassifier.coefs_` lagrer vektene som `(inn, ut)`. Shapen på output er likevel den samme: `(N, 16)`.
:::

## nn.Sequential
`nn.Sequential` setter lag etter hverandre, slik at output fra ett lag er input til neste. Her er klassifisereren fra prosjekt 3:

```python statisk
import torch
import torch.nn as nn

torch.manual_seed(0)
net = nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    nn.Linear(16, 16),
    nn.Tanh(),
    nn.Linear(16, 1),      # ingen aktivering: output er en logit
)
print(net)
X = torch.randn(8, 2)
print(net(X).shape)
print(net[0])              # lagene kan indekseres som en liste
```
```output
Sequential(
  (0): Linear(in_features=2, out_features=16, bias=True)
  (1): Tanh()
  (2): Linear(in_features=16, out_features=16, bias=True)
  (3): Tanh()
  (4): Linear(in_features=16, out_features=1, bias=True)
)
torch.Size([8, 1])
Linear(in_features=2, out_features=16, bias=True)
```

## make_mlp fra prosjekt 3
I prosjektet ble nettet bygget av en funksjon, slik at bredde, dybde og aktivering kunne varieres i eksperimentene. Legg merke til `act` (en **klasse**, f.eks. `nn.Tanh`) som kalles med `act()` for å lage et nytt lag, og `*layers` som pakker ut listen til argumenter:

```python statisk
import torch.nn as nn

def make_mlp(n_in, width, depth, act):
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), act()]
        d = width
    return nn.Sequential(*layers, nn.Linear(d, 1))

def count_params(net):
    return sum(p.numel() for p in net.parameters())

cls_net = make_mlp(2, 16, 2, nn.Tanh)
reg_net = make_mlp(2, 128, 3, nn.SiLU)
print(cls_net)
print("Klassifiserer:", count_params(cls_net))
print("Regressor:    ", count_params(reg_net))
for navn, p in cls_net.named_parameters():
    print(f"  {navn:<9} {str(tuple(p.shape)):<9} {p.numel()}")
```
```output
Sequential(
  (0): Linear(in_features=2, out_features=16, bias=True)
  (1): Tanh()
  (2): Linear(in_features=16, out_features=16, bias=True)
  (3): Tanh()
  (4): Linear(in_features=16, out_features=1, bias=True)
)
Klassifiserer: 337
Regressor:     33537
  0.weight  (16, 2)   32
  0.bias    (16,)     16
  2.weight  (16, 16)  256
  2.bias    (16,)     16
  4.weight  (1, 16)   16
  4.bias    (1,)      1
```

`net.parameters()` gir alle vekt- og bias-tensorene, og `p.numel()` er antall tall i en tensor. Summen er de samme 337 og 33 537 som du regnet ut i leksjon 9.2.

## Egen klasse med nn.Module
Vil du ha mer kontroll, arver du fra `nn.Module` (arv og `super()` fra del 1). Lagene lages i `__init__`, og `forward` sier hvordan data flyter gjennom dem. Dette er nettet fra prosjekt 4 (MNIST-bilder med 28 · 28 = 784 piksler, 10 klasser):

```python statisk
import torch

torch.manual_seed(42)

class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()        # må kalles først
        self.fc1 = torch.nn.Linear(28*28, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    # Returns logits
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # Returns class prediction
    def forward_digit(self, x):
        return torch.argmax(torch.softmax(self(x), dim=1), dim=1)

MLP_model = SimpleNN()
bilder = torch.rand(3, 784)              # tre «bilder»
print(MLP_model(bilder).shape)           # net(x) kaller forward(x)
print(MLP_model.forward_digit(bilder))   # utrent: tilfeldige klasser
print(sum(p.numel() for p in MLP_model.parameters()))
```
```output
torch.Size([3, 10])
tensor([9, 4, 4])
109386
```

Du kaller nettet som en funksjon, `MLP_model(x)`, ikke `MLP_model.forward(x)`. `nn.Module` sørger for at `forward` kjøres. `super(SimpleNN, self).__init__()` er den gamle skrivemåten for `super().__init__()`, som notatene bruker. Glemmer du den, får du `AttributeError: cannot assign module before Module.__init__() call`.

## state_dict: vektene som en ordbok
`net.state_dict()` er en ordbok fra navn til vekt-tensorer. Den brukes til å lagre og laste vekter, og til early stopping (leksjon 9.10) og target-nettverket i deep Q-learning (del 11):

```python statisk
import torch
import torch.nn as nn

torch.manual_seed(0)
net = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
sd = net.state_dict()
for navn, verdi in sd.items():
    print(navn, tuple(verdi.shape))

kopi = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
kopi.load_state_dict(net.state_dict())    # samme vekter
X = torch.randn(3, 2)
print(torch.equal(net(X), kopi(X)))
torch.save(net.state_dict(), "modell.pth")   # lagre til fil
```
```output
0.weight (16, 2)
0.bias (16,)
2.weight (1, 16)
2.bias (1,)
True
```

Navnene `0.weight` og `2.weight` kommer fra posisjonen i `nn.Sequential` (lag 1 er `Tanh`, som ikke har parametre). I `SimpleNN` heter de `fc1.weight`, `fc2.weight` osv.

::: oppgave Lag din egen Linear-klasse
Skriv klassen `Linear` i NumPy som etterligner `nn.Linear`:
- `__init__(self, n_in, n_out)` lager `self.weight` med shape `(n_out, n_in)` (bruk `np.random.randn`) og `self.bias` som nuller med shape `(n_out,)`.
- `__call__(self, X)` returnerer `X @ self.weight.T + self.bias`.
- `n_params(self)` returnerer antall parametre.
```start
import numpy as np

class Linear:
    def __init__(self, n_in, n_out):
        ...

    def __call__(self, X):
        ...

    def n_params(self):
        ...
```
```losning
import numpy as np

class Linear:
    def __init__(self, n_in, n_out):
        self.weight = np.random.randn(n_out, n_in)
        self.bias = np.zeros(n_out)

    def __call__(self, X):
        return X @ self.weight.T + self.bias

    def n_params(self):
        return self.weight.size + self.bias.size
```
```test
import numpy as np
lag = Linear(2, 16)
assert hasattr(lag, "weight") and hasattr(lag, "bias"), "Lag self.weight og self.bias i __init__"
assert np.shape(lag.weight) == (16, 2), f"weight skal ha shape (n_out, n_in) = (16, 2), har {np.shape(lag.weight)}"
assert np.shape(lag.bias) == (16,), f"bias skal ha shape (16,), har {np.shape(lag.bias)}"
X = np.ones((5, 2))
H = lag(X)
assert H is not None and np.shape(H) == (5, 16), "lag(X) skal gi shape (5, 16). Bruk X @ self.weight.T + self.bias"
assert np.allclose(H, X @ lag.weight.T + lag.bias), "lag(X) skal være X @ self.weight.T + self.bias"
assert lag.n_params() == 48, f"Linear(2, 16) har 2*16 + 16 = 48 parametre, fikk {lag.n_params()}"
assert Linear(784, 128).n_params() == 100480, "Linear(784, 128) skal ha 100480 parametre"
```
:::

::: quiz
Hva skriver koden ut?
```kode
lag = nn.Linear(10, 3)
print(lag.weight.shape, lag(torch.zeros(4, 10)).shape)
```
- [ ] `torch.Size([10, 3]) torch.Size([4, 3])`
- [x] `torch.Size([3, 10]) torch.Size([4, 3])`
- [ ] `torch.Size([3, 10]) torch.Size([3, 4])`
- [ ] `torch.Size([10, 3]) torch.Size([4, 10])`
? PyTorch lagrer vektene som `(ut, inn)` = `(3, 10)` og regner `X @ W.T + b`. Output har fortsatt én rad per datapunkt og én kolonne per node: `(4, 3)`.
:::

::: quiz
Hvor mange parametre har `make_mlp(3, 8, 2, nn.ReLU)` (koden fra prosjekt 3 over)?
- [ ] 97
- [x] 113
- [ ] 105
- [ ] 89
? Nettet er 3 → 8 → 8 → 1: $(3 \cdot 8 + 8) + (8 \cdot 8 + 8) + (8 \cdot 1 + 1) = 32 + 72 + 9 = 113$. `ReLU` har ingen parametre.
:::

::: oppsummering
- `nn.Linear(inn, ut)`: `weight` har shape `(ut, inn)`, `bias` har `(ut,)`, og laget regner `X @ W.T + b`.
- `nn.Sequential(lag1, akt, lag2, ...)` stabler lag. `make_mlp` i prosjekt 3 bygger den med en løkke og `*layers`.
- Egen klasse: arv fra `nn.Module`, kall `super().__init__()`, lag lagene i `__init__` og skriv `forward`. Kall nettet med `net(x)`.
- `sum(p.numel() for p in net.parameters())` teller parametre.
- `net.state_dict()` / `net.load_state_dict(...)` henter og setter alle vektene.
:::

--- LEKSJON 9.7 | tapsfunksjoner | Tapsfunksjoner og logits
Tapsfunksjonen måler hvor feil nettet tar. I PyTorch lager du den som et objekt og kaller den med `loss_fn(prediksjon, target)`. Valget avhenger av oppgaven:

| Oppgave | Siste lag | Tapsfunksjon | Target |
|---|---|---|---|
| Regresjon | lineært, 1 node | `nn.MSELoss()` | float, shape `(N,)` som output |
| Binær klassifisering | lineært, 1 node (logit) | `nn.BCEWithLogitsLoss()` | float 0.0/1.0, samme shape som output |
| Binær, med sigmoid i nettet | `nn.Sigmoid()` | `nn.BCELoss()` | float 0.0/1.0 |
| $C$ klasser | lineært, $C$ noder (logits) | `nn.CrossEntropyLoss()` | **long**, shape `(N,)`, verdier `0 … C-1` |

En **logit** er output fra siste lag før sigmoid eller softmax, et hvilket som helst reelt tall. Både `BCEWithLogitsLoss` og `CrossEntropyLoss` tar logits og har sigmoid eller softmax innebygd. Det er numerisk mer stabilt enn å gjøre det selv.

## MSELoss

```python statisk
import torch
import torch.nn as nn

mse = nn.MSELoss()
pred = torch.tensor([2.5, 0.0, 2.0])
y = torch.tensor([3.0, -0.5, 2.0])
print(mse(pred, y))
print(((pred - y) ** 2).mean())    # samme
```
```output
tensor(0.1667)
tensor(0.1667)
```

## BCEWithLogitsLoss: sigmoid ligger inni
Klassifisereren i prosjekt 3 har ingen sigmoid i nettet. Den gir logits, og `BCEWithLogitsLoss` gjør sigmoid + binær cross-entropy i ett:

```python statisk
import torch
import torch.nn as nn

logits = torch.tensor([2.0, -1.0, 0.5, -3.0])
y = torch.tensor([1.0, 0.0, 0.0, 0.0])

print(nn.BCEWithLogitsLoss()(logits, y))
print(nn.BCELoss()(torch.sigmoid(logits), y))     # samme tall

p = torch.sigmoid(logits)
print(p)
print(logits > 0)        # beslutningsregel på logits
print(p > 0.5)           # gir alltid det samme
```
```output
tensor(0.3657)
tensor(0.3657)
tensor([0.8808, 0.2689, 0.6225, 0.0474])
tensor([ True, False,  True, False])
tensor([ True, False,  True, False])
```

Siden $\sigma(0) = 0.5$ og sigmoid er voksende, er `logit > 0` det samme som `sigmoid(logit) > 0.5`. Derfor bruker prosjekt 3 `(net(X).squeeze(1) > 0)` for å klassifisere, uten å regne ut sannsynligheten.

## CrossEntropyLoss: logits (N, C) og long-targets (N,)
For flere klasser gir nettet én logit per klasse. `CrossEntropyLoss` tar logits med shape `(N, C)` og **klasseindekser** (ikke one-hot) med shape `(N,)` og dtype `long`. Inni gjør den softmax, tar log av sannsynligheten for riktig klasse og snur fortegnet:

```python statisk
import torch
import torch.nn as nn

logits = torch.tensor([[2.0, 0.5, -1.0],
                       [0.1, 0.2, 3.0],
                       [1.0, 1.0, 1.0]])
y = torch.tensor([0, 2, 1])            # dtype torch.long (int64)

ce = nn.CrossEntropyLoss()
print(ce(logits, y))

p = torch.softmax(logits, dim=1)       # hver rad summerer til 1
print(p)
print(-torch.log(p[torch.arange(3), y]).mean())   # samme som ce
print(logits.argmax(dim=1))            # predikert klasse
```
```output
tensor(0.4832)
tensor([[0.7856, 0.1753, 0.0391],
        [0.0493, 0.0545, 0.8962],
        [0.3333, 0.3333, 0.3333]])
tensor(0.4832)
tensor([0, 2, 0])
```

## Vanlige feil
Disse feilene er lette å gjøre og typiske eksamensdistraktorer. Den første gir en feilmelding, de to andre gir **feil svar uten feilmelding** (bare en advarsel, eller ingenting):

```python statisk
import torch
import torch.nn as nn

# 1) Feil shape til BCEWithLogitsLoss: net(X) har shape (N, 1), y har (N,)
out = torch.zeros(4, 1)
y = torch.tensor([1.0, 0.0, 1.0, 0.0])
try:
    nn.BCEWithLogitsLoss()(out, y)
except ValueError as e:
    print("ValueError:", e)
print(nn.BCEWithLogitsLoss()(out.squeeze(1), y))    # riktig, som i prosjekt 3

# 2) Feil dtype til CrossEntropyLoss
try:
    nn.CrossEntropyLoss()(torch.randn(4, 3), torch.tensor([0.0, 2.0, 1.0, 1.0]))
except RuntimeError as e:
    print("RuntimeError:", e)
```
```output
ValueError: Target size (torch.Size([4])) must be the same as input size (torch.Size([4, 1]))
tensor(0.6931)
RuntimeError: expected target dtype to be Long or Byte, but got Float
```

```python statisk
import warnings
import torch
import torch.nn as nn

# 3) MSELoss med (N, 1) mot (N,): broadcasting til (N, N) gir feil tap
pred = torch.tensor([[1.0], [2.0], [3.0], [4.0]])   # perfekte prediksjoner
y = torch.tensor([1.0, 2.0, 3.0, 4.0])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print("feil:  ", nn.MSELoss()(pred, y))          # PyTorch gir bare en UserWarning
print("riktig:", nn.MSELoss()(pred.squeeze(1), y))

# 4) Sigmoid to ganger: nettet har nn.Sigmoid() til slutt OG du bruker BCEWithLogitsLoss
p = torch.tensor([0.01, 0.99])                       # nettet er nesten perfekt
y = torch.tensor([0.0, 1.0])
print("BCELoss:          ", nn.BCELoss()(p, y))
print("BCEWithLogitsLoss:", nn.BCEWithLogitsLoss()(p, y))   # sigmoid(p) ligger i (0.5, 0.73)
```
```output
feil:   tensor(2.5000)
riktig: tensor(0.)
BCELoss:           tensor(0.0101)
BCEWithLogitsLoss: tensor(0.5071)
```

I feil nr. 4 behandles sannsynlighetene som logits. $\sigma(p)$ for $p \in (0, 1)$ ligger alltid mellom 0.5 og 0.73, så tapet kan aldri bli lite, og nettet lærer dårlig. Regelen er: **enten** logits og `...WithLogitsLoss`/`CrossEntropyLoss`, **eller** sigmoid i nettet og `BCELoss`. Aldri begge.

::: eksamen
Kodelesingsspørsmål om tapsfunksjoner handler ofte om å matche siste lag med tapet. Spør deg selv: gir nettet logits eller sannsynligheter? Hvilken shape og dtype har target?
:::

::: oppgave Binær cross-entropy fra logits
Skriv `bce_with_logits(z, y)` i NumPy som gjør det samme som `nn.BCEWithLogitsLoss()`: gjør logits `z` om til sannsynligheter med sigmoid, og returner gjennomsnittet av $-[y \log p + (1-y)\log(1-p)]$.
```start
import numpy as np

def bce_with_logits(z, y):
    p = ...
    return ...
```
```losning
import numpy as np

def bce_with_logits(z, y):
    p = 1 / (1 + np.exp(-z))
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
```
```test
import numpy as np
z = np.array([2.0, -1.0, 0.5, -3.0])
y = np.array([1.0, 0.0, 0.0, 0.0])
v = bce_with_logits(z, y)
assert v is not None, "Funksjonen returnerer ingenting"
assert np.ndim(v) == 0, "Returner ett tall (gjennomsnittet), ikke en array"
assert abs(v - 0.3657) < 1e-3, f"Forventet ca. 0.3657 (samme som PyTorch), fikk {v}. Husk sigmoid først og minustegnet."
```
:::

::: oppgave Cross-entropy fra logits
Skriv `cross_entropy(logits, y)` i NumPy som gjør det samme som `nn.CrossEntropyLoss()`. `logits` har shape `(N, C)` og `y` inneholder klasseindekser med shape `(N,)`.
1. Softmax per rad (husk å trekke fra maks).
2. Plukk ut sannsynligheten for riktig klasse i hver rad: `P[np.arange(N), y]`.
3. Returner gjennomsnittet av $-\log$ av disse.
```start
import numpy as np

def cross_entropy(logits, y):
    ...
```
```losning
import numpy as np

def cross_entropy(logits, y):
    E = np.exp(logits - logits.max(axis=1, keepdims=True))
    P = E / E.sum(axis=1, keepdims=True)
    return -np.mean(np.log(P[np.arange(len(y)), y]))
```
```test
import numpy as np
logits = np.array([[2.0, 0.5, -1.0], [0.1, 0.2, 3.0], [1.0, 1.0, 1.0]])
y = np.array([0, 2, 1])
v = cross_entropy(logits, y)
assert v is not None, "Funksjonen returnerer ingenting"
assert np.ndim(v) == 0, "Returner ett tall (gjennomsnittet over datapunktene)"
assert abs(v - 0.4832) < 1e-3, f"Forventet ca. 0.4832 (samme som PyTorch), fikk {v}"
v2 = cross_entropy(np.array([[1000.0, 0.0]]), np.array([0]))
assert np.isfinite(v2) and abs(v2) < 1e-6, "Store logits gir nan/inf. Trekk fra maks per rad før np.exp."
```
:::

::: quiz
Nettet slutter med `nn.Linear(16, 1)` (ingen aktivering), og output squeezes til shape `(N,)`. Hvilken kombinasjon er riktig?
- [ ] `nn.BCELoss()` og prediksjon `net(X).squeeze(1) > 0.5`
- [x] `nn.BCEWithLogitsLoss()` og prediksjon `net(X).squeeze(1) > 0`
- [ ] `nn.CrossEntropyLoss()` og prediksjon `net(X).argmax(dim=1)`
- [ ] `nn.BCEWithLogitsLoss()` og prediksjon `net(X).squeeze(1) > 0.5`
? Output er en logit, så tapet må ha sigmoid innebygd (`BCEWithLogitsLoss`). `logit > 0` tilsvarer `sigmoid(logit) > 0.5`. Terskel 0.5 på logits ville tilsvart sannsynlighet 0.62. `BCELoss` krever sannsynligheter, og `CrossEntropyLoss` med én output-node gir ikke mening.
:::

::: quiz
`logits` har shape `(64, 10)`. Hvilket target fungerer med `nn.CrossEntropyLoss()(logits, target)`?
- [ ] `torch.tensor(labels, dtype=torch.float32)` med shape `(64,)`
- [ ] `torch.tensor(labels, dtype=torch.long)` med shape `(64, 1)`
- [x] `torch.tensor(labels, dtype=torch.long)` med shape `(64,)`
- [ ] `torch.tensor(labels, dtype=torch.long)` med shape `(10,)`
? Én klasseindeks (0–9) per datapunkt, som heltall (`long`), med shape `(N,)`. Prosjekt 4 gjør akkurat dette: `torch.tensor(y_train.astype(int), dtype=torch.long)`.
:::

::: oppsummering
- Regresjon: `nn.MSELoss()`. Pass på at prediksjon og target har lik shape (`squeeze(1)`).
- Binær: logits + `nn.BCEWithLogitsLoss()` (float targets), beslutning `logit > 0`. Eller sigmoid i nettet + `nn.BCELoss()`. Aldri sigmoid to ganger.
- Flere klasser: logits `(N, C)` + `nn.CrossEntropyLoss()` med `long`-targets `(N,)`. Prediksjon `argmax(dim=1)`.
- Feil shape i `MSELoss` gir feil tap uten feilmelding.
:::

--- LEKSJON 9.8 | treningslokka | Treningsløkka
All trening i PyTorch følger det samme mønsteret. For hver oppdatering:

1. `optimizer.zero_grad()` nullstiller gamle gradienter
2. `loss = loss_fn(net(X), y)` forward pass og tap
3. `loss.backward()` backprop: fyller `.grad` for alle parametre
4. `optimizer.step()` oppdaterer parametrene: $\theta \leftarrow \theta - \eta \nabla L$ (for SGD)

En **optimizer** får parametrene ved opprettelse, `torch.optim.SGD(net.parameters(), lr=0.1)`, og vet derfor hvilke tensorer den skal oppdatere.

## Først for hånd
Her er gradient descent for lineær regresjon (del 5), men med autograd i stedet for gradientformler. Oppdateringen må skje inne i `torch.no_grad()`, ellers prøver PyTorch å derivere selve oppdateringen:

```python statisk
import torch

torch.manual_seed(0)
X = torch.rand(100, 1) * 4
y = 2 * X + 1 + 0.1 * torch.randn(100, 1)      # sann modell: y = 2x + 1

w = torch.zeros(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.05
for epoch in range(300):
    loss = ((X @ w + b - y) ** 2).mean()     # forward + tap
    loss.backward()                           # gradienter
    with torch.no_grad():
        w -= lr * w.grad                      # oppdater
        b -= lr * b.grad
        w.grad.zero_()                        # nullstill
        b.grad.zero_()
print(w.item(), b.item())
```
```output
1.982932209968567 1.033018946647644
```

## Med nn.Linear og en optimizer
Det samme med byggeklossene. `optimizer.step()` gjør oppdateringen, og `optimizer.zero_grad()` nullstiller:

```python statisk
import torch
import torch.nn as nn

torch.manual_seed(0)
X = torch.rand(100, 1) * 4
y = 2 * X + 1 + 0.1 * torch.randn(100, 1)

net = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)

for epoch in range(300):
    optimizer.zero_grad()
    loss = loss_fn(net(X), y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"epoke {epoch}: tap {loss.item():.4f}")
print(net.weight.item(), net.bias.item())
```
```output
epoke 0: tap 58.9460
epoke 100: tap 0.0083
epoke 200: tap 0.0079
1.982999563217163 1.0328478813171387
```

## Hva skjer hvis du glemmer zero_grad?
`backward()` **legger til** i `.grad` i stedet for å overskrive. Uten `zero_grad()` blir gradienten summen av alle tidligere gradienter:

```python statisk
import torch

w = torch.tensor(1.0, requires_grad=True)
for i in range(3):
    loss = 3 * w           # dL/dw = 3
    loss.backward()
    print(w.grad)          # 3, 6, 9: akkumuleres!
```
```output
tensor(3.)
tensor(6.)
tensor(9.)
```

```python statisk
import torch
import torch.nn as nn

def tren(nullstill):
    torch.manual_seed(0)
    X = torch.rand(100, 1) * 4
    y = 2 * X + 1 + 0.1 * torch.randn(100, 1)
    net = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
    losses = []
    for epoch in range(200):
        if nullstill:
            optimizer.zero_grad()
        loss = nn.MSELoss()(net(X), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return [round(losses[i], 3) for i in [0, 10, 20, 50, 100, 199]]

print("epoke:          [0, 10, 20, 50, 100, 199]")
print("med zero_grad: ", tren(True))
print("uten zero_grad:", tren(False))
```
```output
epoke:          [0, 10, 20, 50, 100, 199]
med zero_grad:  [58.946, 0.035, 0.025, 0.012, 0.008, 0.008]
uten zero_grad: [58.946, 14.12, 50.921, 32.671, 13.271, 17.284]
```

Med `zero_grad()` synker tapet jevnt mot støynivået. Uten blir hvert steg summen av alle gradienter så langt. Parametrene skyter over, tapet hopper opp og ned, og modellen blir aldri god.

## Et nevralt nett på prosjekt 3-dataene
Samme fire steg, nå med klassifisereren 2 → 16 → 16 → 1. Hele treningsdatasettet brukes i hver oppdatering (full batch), så én epoke er én oppdatering. Vi lagrer tapet per epoke i en liste for å kunne plotte **tapskurven**:

```python statisk
import pandas as pd
import torch
import torch.nn as nn

torch.manual_seed(0)
data = pd.read_csv("classification_data.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
X_train = torch.tensor(train[["x0", "x1"]].values, dtype=torch.float32)
y_train = torch.tensor(train["y"].values, dtype=torch.float32)
X_test = torch.tensor(test[["x0", "x1"]].values, dtype=torch.float32)
y_test = torch.tensor(test["y"].values, dtype=torch.float32)

net = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1))
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

losses = []
for epoch in range(300):
    optimizer.zero_grad()
    loss = loss_fn(net(X_train).squeeze(1), y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())          # .item(): lagre et tall, ikke en tensor
    if epoch % 50 == 0:
        print(f"epoke {epoch:3d}: tap {loss.item():.4f}")

with torch.no_grad():
    acc = ((net(X_test).squeeze(1) > 0).float() == y_test).float().mean().item()
print("Test-accuracy:", round(acc, 3))
```
```output
epoke   0: tap 0.6869
epoke  50: tap 0.2764
epoke 100: tap 0.2261
epoke 150: tap 0.1991
epoke 200: tap 0.1825
epoke 250: tap 0.1721
Test-accuracy: 0.9
```

Tapskurven plotter du med `plt.plot(losses)`. Lagre `loss.item()` og ikke `loss`: en liste med tensorer holder på hele regnegrafen for hver epoke.

Her er det samme i NumPy, så du kan se kurven direkte. Lineær regresjon med de fire stegene, der steg 3 er gradientformelen fra del 5 i stedet for `backward()`:

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
X = rng.uniform(0, 4, size=(100, 1))
y = 2 * X[:, 0] + 1 + 0.1 * rng.normal(size=100)

w, b, lr = np.zeros(1), 0.0, 0.05
losses = []
for epoch in range(100):
    pred = X @ w + b                       # forward
    loss = np.mean((pred - y) ** 2)        # tap
    dw = 2 * X.T @ (pred - y) / len(y)     # gradient
    db = 2 * np.mean(pred - y)
    w -= lr * dw                           # oppdater
    b -= lr * db
    losses.append(loss)

plt.plot(losses)
plt.xlabel("epoke")
plt.ylabel("MSE")
plt.yscale("log")
plt.title(f"w = {w[0]:.2f}, b = {b:.2f}")
plt.show()
```

::: oppgave Treningsløkka i NumPy
Fullfør treningsløkka for lineær regresjon (én feature) med de fire stegene: forward, tap, gradient og oppdatering. Tren i 500 epoker med `lr = 0.05`. Lagre tapet for hver epoke i listen `losses`, og la `w` og `b` være de trente parametrene. Gradientene for MSE er $\frac{\partial L}{\partial w} = \frac{2}{N}\sum (\hat{y} - y)x$ og $\frac{\partial L}{\partial b} = \frac{2}{N}\sum (\hat{y} - y)$.
```start
import numpy as np

rng = np.random.default_rng(1)
x = rng.uniform(0, 4, size=100)
y = 2 * x + 1 + 0.1 * rng.normal(size=100)

w, b, lr = 0.0, 0.0, 0.05
losses = []
for epoch in range(500):
    # 1) forward
    # 2) tap (MSE), legg til i losses
    # 3) gradienter
    # 4) oppdater w og b
    ...
```
```losning
import numpy as np

rng = np.random.default_rng(1)
x = rng.uniform(0, 4, size=100)
y = 2 * x + 1 + 0.1 * rng.normal(size=100)

w, b, lr = 0.0, 0.0, 0.05
losses = []
for epoch in range(500):
    pred = w * x + b
    loss = np.mean((pred - y) ** 2)
    losses.append(loss)
    dw = 2 * np.mean((pred - y) * x)
    db = 2 * np.mean(pred - y)
    w -= lr * dw
    b -= lr * db
```
```test
import numpy as np
assert len(losses) == 500, f"losses skal ha én verdi per epoke (500), har {len(losses)}"
assert abs(losses[0] - np.mean(y ** 2)) < 1e-6, "Første tap skal være MSE med w = b = 0. Regn ut tapet FØR du oppdaterer."
assert losses[-1] < losses[0] / 100, "Tapet synker ikke nok. Sjekk gradientene og at du trekker fra (w -= lr * dw)."
assert abs(float(np.ravel(w)[0]) - 2) < 0.1 and abs(float(np.ravel(b)[0] if np.ndim(b) else b) - 1) < 0.2, f"w og b skal nærme seg 2 og 1, fikk w={w}, b={b}"
```
:::

::: quiz
Hva er galt med denne treningsløkka?
```kode
for epoch in range(100):
    loss = loss_fn(net(X), y)
    loss.backward()
    optimizer.step()
```
- [ ] Ingenting, `optimizer.step()` nullstiller gradientene automatisk
- [ ] `loss.backward()` må kalles etter `optimizer.step()`
- [x] Gradientene akkumuleres fra epoke til epoke fordi `optimizer.zero_grad()` mangler
- [ ] Nettet lærer ingenting fordi `net(X)` må kalles inne i `torch.no_grad()`
? `backward()` legger til i `.grad`. Uten `optimizer.zero_grad()` er gradienten i epoke $k$ summen av alle gradientene til nå, og stegene blir altfor store. `no_grad` rundt forward ville gjort at ingen gradienter blir beregnet i det hele tatt.
:::

::: quiz
Hvilken rekkefølge er riktig i én oppdatering?
- [ ] `loss.backward()`, `optimizer.zero_grad()`, `loss = loss_fn(net(X), y)`, `optimizer.step()`
- [ ] `optimizer.zero_grad()`, `loss = loss_fn(net(X), y)`, `optimizer.step()`, `loss.backward()`
- [x] `optimizer.zero_grad()`, `loss = loss_fn(net(X), y)`, `loss.backward()`, `optimizer.step()`
- [ ] `loss = loss_fn(net(X), y)`, `optimizer.step()`, `loss.backward()`, `optimizer.zero_grad()`
? Tapet må finnes før det kan deriveres, og gradientene må finnes før de kan brukes til et steg. `zero_grad()` kan stå først (som her) eller rett før `backward()` (som i prosjekt 3), bare den kommer før `backward()`.
:::

::: oppsummering
- De fire stegene: `optimizer.zero_grad()`, `loss = loss_fn(net(X), y)`, `loss.backward()`, `optimizer.step()`.
- `torch.optim.SGD(net.parameters(), lr=...)`: optimizeren får parametrene den skal oppdatere.
- `backward()` **akkumulerer** gradienter. Uten `zero_grad()` blir treningen ustabil.
- En epoke er én runde gjennom alle treningsdataene. Med full batch er det én oppdatering.
- Lagre `loss.item()` i en liste og plott tapskurven.
:::

--- LEKSJON 9.9 | minibatch-dataloader | Mini-batcher, DataLoader og prediksjon
Med **mini-batch** oppdaterer vi vektene etter en liten, tilfeldig bit av treningsdataene i stedet for hele settet. En epoke er fortsatt én runde gjennom alle dataene, men nå med $\lceil N / B \rceil$ oppdateringer, der $B$ er batch size. Gradienten blir litt støyete, men hver oppdatering er billig, og støyen hjelper ofte på generaliseringen.

## Mini-batcher med randperm (som prosjekt 3)
`torch.randperm(n)` gir tallene `0 … n-1` i tilfeldig rekkefølge. Vi stokker indeksene på nytt hver epoke og går gjennom dem i biter på `batch_size`:

```python statisk
import torch

torch.manual_seed(0)
X = torch.arange(10.0).reshape(10, 1)       # 10 datapunkter
batch_size = 4

for epoch in range(2):
    perm = torch.randperm(len(X))
    print(f"epoke {epoch}: perm = {perm.tolist()}")
    for b in range(0, len(perm), batch_size):
        j = perm[b:b + batch_size]
        print("   batch:", X[j].squeeze(1).tolist())
```
```output
epoke 0: perm = [4, 1, 7, 5, 3, 9, 0, 8, 6, 2]
   batch: [4.0, 1.0, 7.0, 5.0]
   batch: [3.0, 9.0, 0.0, 8.0]
   batch: [6.0, 2.0]
epoke 1: perm = [3, 9, 4, 2, 7, 8, 6, 0, 5, 1]
   batch: [3.0, 9.0, 4.0, 2.0]
   batch: [7.0, 8.0, 6.0, 0.0]
   batch: [5.0, 1.0]
```

Siste batch er mindre (10 = 4 + 4 + 2). Innerst i løkka kommer de fire stegene fra leksjon 9.8 med `X[j]` og `y[j]`. Det er akkurat slik `train_net` i prosjekt 3 gjør det:

```python statisk ikkekjor
for epoch in range(epochs):
    net.train()
    perm = train_idx[torch.randperm(len(train_idx), generator=g)]
    for b in range(0, len(perm), batch_size):
        j = perm[b:b + batch_size]
        loss = loss_fn(net(X[j]).squeeze(1), y[j])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## TensorDataset og DataLoader (som prosjekt 4)
PyTorch har ferdige verktøy for det samme. `TensorDataset(X, y)` parer features og targets, og `DataLoader` deler dem i batcher og stokker hver epoke når `shuffle=True`:

```python statisk
import torch
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
X = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))

train_dataset = TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

print(len(train_dataset), "datapunkter")
print(len(train_loader), "batcher per epoke")
X_batch, y_batch = next(iter(train_loader))     # første batch
print(X_batch.shape, y_batch.shape)
for X_batch, y_batch in train_loader:
    pass
print("siste batch:", X_batch.shape)
```
```output
1000 datapunkter
16 batcher per epoke
torch.Size([64, 784]) torch.Size([64])
siste batch: torch.Size([40, 784])
```

`len(train_loader)` er antall batcher per epoke: $\lceil 1000 / 64 \rceil = 16$, der den siste har $1000 - 15 \cdot 64 = 40$ punkter. Treningsløkka i prosjekt 4 blir:

```python statisk ikkekjor
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = MLP_model(X_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()
```

::: eksamen
Eksamen 2024: «Treningsdatasettet har 10 000 datapunkter og batch size er 500. Hvor mange iterasjoner er det i én epoke?» Svaret er 10 000 / 500 = 20. Full batch gir 1 oppdatering per epoke, ren SGD (batch size 1) gir $N$.
:::

## net.train() og net.eval()
Noen lag oppfører seg ulikt under trening og prediksjon. `nn.Dropout` setter tilfeldige aktiveringer til 0 under trening, men gjør ingenting under evaluering. `net.train()` og `net.eval()` skrur mellom de to modusene. Våre nett i prosjekt 3 har ingen slike lag, men det er god vane å bruke dem (slik `train_net` gjør):

```python statisk
import torch
import torch.nn as nn

torch.manual_seed(0)
net = nn.Sequential(nn.Linear(4, 4), nn.Dropout(p=0.5))
x = torch.ones(1, 4)

net.train()
print(net(x))
print(net(x))      # annerledes: nye tilfeldige nuller
net.eval()
print(net(x))
print(net(x))      # likt hver gang
print(net(x).requires_grad)   # eval() slår IKKE av gradienter
```
```output
tensor([[-0.6347,  1.2561, -1.0009, -0.0000]], grad_fn=<MulBackward0>)
tensor([[-0.6347,  1.2561, -1.0009, -2.4283]], grad_fn=<MulBackward0>)
tensor([[-0.3173,  0.6281, -0.5004, -1.2141]], grad_fn=<AddmmBackward0>)
tensor([[-0.3173,  0.6281, -0.5004, -1.2141]], grad_fn=<AddmmBackward0>)
True
```

Under trening ganges de gjenværende verdiene med $1/(1-p) = 2$, slik at gjennomsnittet blir det samme som i eval-modus. `net.eval()` og `torch.no_grad()` er to forskjellige ting. `eval()` endrer oppførselen til lag som Dropout, `no_grad()` slår av gradientsporingen. Ved prediksjon vil du som regel ha begge.

## Prediksjon og accuracy
Ved prediksjon pakker vi inn i `torch.no_grad()` og gjør logits om til klasser:

```python statisk
import torch

logits = torch.tensor([[1.2], [-0.3], [0.0], [2.5]])      # binær: shape (N, 1)
y = torch.tensor([1.0, 0.0, 1.0, 1.0])
with torch.no_grad():
    p = torch.sigmoid(logits).squeeze(1)
    pred = (p > 0.5).float()
    pred2 = (logits.squeeze(1) > 0).float()
print(p, pred, pred2)
print("accuracy:", (pred == y).float().mean().item())

logits = torch.tensor([[2.0, 0.5, -1.0],                   # flere klasser: (N, C)
                       [0.1, 0.2, 3.0]])
print(torch.softmax(logits, dim=1))       # sannsynligheter, hver rad summerer til 1
print(logits.argmax(dim=1))               # predikert klasse
print(torch.softmax(logits, dim=1).argmax(dim=1))   # samme: softmax endrer ikke rekkefølgen
```
```output
tensor([0.7685, 0.4256, 0.5000, 0.9241]) tensor([1., 0., 0., 1.]) tensor([1., 0., 0., 1.])
accuracy: 0.75
tensor([[0.7856, 0.1753, 0.0391],
        [0.0493, 0.0545, 0.8962]])
tensor([0, 2])
tensor([0, 2])
```

Legg merke til `logit = 0.0`: sigmoid gir nøyaktig 0.5, og både `p > 0.5` og `logit > 0` sier klasse 0. `forward_digit` i prosjekt 4 bruker `torch.argmax(torch.softmax(self(x), dim=1), dim=1)`. Softmax der er strengt tatt unødvendig, fordi den største logiten alltid gir den største sannsynligheten.

## Alt sammen: prosjekt 3-klassifisereren med DataLoader

```python statisk
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
data = pd.read_csv("classification_data.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
X_train = torch.tensor(train[["x0", "x1"]].values, dtype=torch.float32)
y_train = torch.tensor(train["y"].values, dtype=torch.float32)
X_test = torch.tensor(test[["x0", "x1"]].values, dtype=torch.float32)
y_test = torch.tensor(test["y"].values, dtype=torch.float32)

loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
net = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 1))
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)

for epoch in range(30):
    net.train()
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        loss = loss_fn(net(X_batch).squeeze(1), y_batch)
        loss.backward()
        optimizer.step()

net.eval()
with torch.no_grad():
    pred = (net(X_test).squeeze(1) > 0).float()
print("oppdateringer per epoke:", len(loader))
print("test-accuracy:", round((pred == y_test).float().mean().item(), 3))
```
```output
oppdateringer per epoke: 16
test-accuracy: 0.902
```

## Flere klasser: et lite «prosjekt 4» på sifferbilder
Prosjekt 4 trener `SimpleNN` på MNIST. Her gjør vi det samme på `load_digits` fra scikit-learn (8 × 8 = 64 piksler, 10 klasser), som ikke krever nedlasting. Legg merke til `torch.long` for targets, `CrossEntropyLoss` og `argmax(dim=1)`:

```python statisk
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
X_train = torch.tensor(X_train / 16.0, dtype=torch.float32)   # piksler 0-16 -> 0-1
X_test = torch.tensor(X_test / 16.0, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

torch.manual_seed(42)
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = SimpleNN()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

for epoch in range(10):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        acc = (model(X_test).argmax(dim=1) == y_test).float().mean().item()
    if epoch % 3 == 0 or epoch == 9:
        print(f"Epoch {epoch+1}/10, Test Accuracy: {acc:.4f}")
```
```output
Epoch 1/10, Test Accuracy: 0.5800
Epoch 4/10, Test Accuracy: 0.9156
Epoch 7/10, Test Accuracy: 0.9311
Epoch 10/10, Test Accuracy: 0.9511
```

::: oppgave Lag mini-batcher
Skriv `minibatcher(N, batch_size, rng)` som stokker indeksene `0 … N-1` med `rng.permutation(N)` og returnerer en **liste** med index-arrays, én per batch (den siste kan være mindre). Det er det `torch.randperm` + løkka i prosjekt 3 gjør.
```start
import numpy as np

def minibatcher(N, batch_size, rng):
    perm = ...
    batcher = []
    ...
    return batcher
```
```losning
import numpy as np

def minibatcher(N, batch_size, rng):
    perm = rng.permutation(N)
    batcher = []
    for b in range(0, N, batch_size):
        batcher.append(perm[b:b + batch_size])
    return batcher
```
```test
import numpy as np
B = minibatcher(10, 4, np.random.default_rng(0))
assert isinstance(B, list), "Returner en liste med index-arrays"
assert [len(b) for b in B] == [4, 4, 2], f"10 punkter med batch_size 4 skal gi batcher på 4, 4 og 2, fikk {[len(b) for b in B]}"
alle = np.concatenate(B)
assert sorted(alle.tolist()) == list(range(10)), "Hver indeks 0-9 skal komme nøyaktig én gang"
assert alle.tolist() != list(range(10)), "Indeksene skal stokkes med rng.permutation(N)"
assert len(minibatcher(1000, 64, np.random.default_rng(1))) == 16, "1000 punkter med batch_size 64 skal gi 16 batcher"
```
:::

::: oppgave Fra logits til accuracy
`logits_bin` er output fra et binært nett (shape `(N,)`), og `logits_multi` fra et nett med 3 klasser (shape `(N, 3)`). Lag
- `pred_bin`: 0/1-prediksjoner fra `logits_bin` (bruk logit-regelen)
- `pred_multi`: klasseprediksjoner fra `logits_multi`
- `acc_multi`: accuracy for `pred_multi` mot `y_multi`
```start
import numpy as np

logits_bin = np.array([1.5, -0.2, 0.0, 3.1, -2.0])
logits_multi = np.array([[2.0, 0.1, -1.0], [0.3, 0.2, 1.5], [0.0, 2.2, 0.1], [1.0, 1.2, 0.9]])
y_multi = np.array([0, 2, 1, 0])

pred_bin = ...
pred_multi = ...
acc_multi = ...
```
```losning
import numpy as np

logits_bin = np.array([1.5, -0.2, 0.0, 3.1, -2.0])
logits_multi = np.array([[2.0, 0.1, -1.0], [0.3, 0.2, 1.5], [0.0, 2.2, 0.1], [1.0, 1.2, 0.9]])
y_multi = np.array([0, 2, 1, 0])

pred_bin = (logits_bin > 0).astype(int)
pred_multi = np.argmax(logits_multi, axis=1)
acc_multi = np.mean(pred_multi == y_multi)
```
```test
import numpy as np
assert np.array_equal(np.asarray(pred_bin).astype(int), [1, 0, 0, 1, 0]), f"pred_bin skal være [1, 0, 0, 1, 0], fikk {pred_bin}. Bruk logits_bin > 0 (logit 0 gir p = 0.5, som ikke er > 0.5)."
assert np.array_equal(pred_multi, [0, 2, 1, 1]), f"pred_multi skal være [0, 2, 1, 1], fikk {pred_multi}. Bruk argmax langs axis=1."
assert abs(acc_multi - 0.75) < 1e-9, f"acc_multi skal være 0.75, fikk {acc_multi}"
```
:::

::: quiz
Du har 1000 treningspunkter og `DataLoader(dataset, batch_size=64, shuffle=True)`. Hvor mange ganger kalles `optimizer.step()` per epoke?
- [ ] 15
- [x] 16
- [ ] 64
- [ ] 1
? $1000 / 64 = 15.6$, så det blir 15 fulle batcher og én med 40 punkter: 16 oppdateringer. `len(train_loader)` gir det samme tallet.
:::

::: quiz
Hvilket utsagn om `net.eval()` er riktig?
- [ ] Den slår av gradientberegningen, så `torch.no_grad()` er unødvendig
- [ ] Den kjører nettet på testdataene og returnerer accuracy
- [x] Den setter lag som Dropout i evalueringsmodus, men gradienter spores fortsatt
- [ ] Den fryser vektene slik at `optimizer.step()` ikke endrer dem
? `eval()` endrer bare oppførselen til lag som Dropout og BatchNorm. Gradientsporing slås av med `torch.no_grad()`. Derfor bruker prosjektkoden begge ved prediksjon.
:::

::: oppsummering
- Mini-batch: stokk indeksene hver epoke (`torch.randperm`) og gå gjennom dem i biter. $\lceil N/B \rceil$ oppdateringer per epoke.
- `DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)` gjør det samme. `len(loader)` = antall batcher.
- `net.train()` før trening, `net.eval()` + `torch.no_grad()` ved prediksjon.
- Binær: `logits > 0` (= `torch.sigmoid(logits) > 0.5`). Flere klasser: `logits.argmax(dim=1)`, sannsynligheter med `torch.softmax(logits, dim=1)`.
- Accuracy: `(pred == y).float().mean().item()`.
:::

--- LEKSJON 9.10 | validering-early-stopping | Validering, early stopping, optimizere og scheduler
Treningstapet sier lite om hvor godt nettet generaliserer. Vi holder derfor av en **valideringsdel** av treningsdataene, regner ut tapet på den etter hver epoke, og bruker det til å velge når vi skal stoppe og hvilke hyperparametre vi skal bruke. Testdataene rører vi ikke før til slutt.

## Held-out validering (som prosjekt 3)
Prosjekt 3 holder av 100 av 500 treningsrader med en tilfeldig permutasjon av indeksene:

```python statisk
import pandas as pd
import torch

data = pd.read_csv("classification_data.csv")
train = data[data["split"] == "train"]
Xc_train = torch.tensor(train[["x0", "x1"]].values, dtype=torch.float32)
yc_train = torch.tensor(train["y"].values, dtype=torch.float32)

perm = torch.randperm(len(Xc_train), generator=torch.Generator().manual_seed(0))
c_val_idx, c_train_idx = perm[:100], perm[100:]
print(len(c_train_idx), "til trening,", len(c_val_idx), "til validering")
print(Xc_train[c_val_idx].shape)
```
```output
400 til trening, 100 til validering
torch.Size([100, 2])
```

## Kopi av vektene med .clone()
Early stopping må ta vare på vektene fra den beste epoken. Men `net.state_dict()` gir **referanser** til de levende vekt-tensorene, ikke en kopi. Uten `.clone()` endrer den «lagrede» versjonen seg hver gang optimizeren oppdaterer vektene:

```python statisk
import torch
import torch.nn as nn

torch.manual_seed(0)
net = nn.Linear(2, 1)
uten_kopi = net.state_dict()
med_kopi = {k: v.clone() for k, v in net.state_dict().items()}

with torch.no_grad():
    net.weight += 100.0          # «trening» endrer vektene

print("nå:      ", net.weight.data)
print("uten kopi", uten_kopi["weight"])     # endret seg også!
print("med kopi ", med_kopi["weight"])      # den gamle verdien
```
```output
nå:       tensor([[ 99.9947, 100.3793]])
uten kopi tensor([[ 99.9947, 100.3793]])
med kopi  tensor([[-0.0053,  0.3793]])
```

## Early stopping med beste state_dict
Etter hver epoke regner vi ut valideringstapet. Er det det beste til nå, tar vi en **kopi** av vektene. Har det ikke blitt bedre på `patience` epoker, stopper vi. Til slutt laster vi inn de beste vektene. Dette er kjernen i `train_net` fra prosjekt 3, litt forenklet:

```python statisk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

data = pd.read_csv("classification_data.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
X = torch.tensor(train[["x0", "x1"]].values, dtype=torch.float32)
y = torch.tensor(train["y"].values, dtype=torch.float32)
X_test = torch.tensor(test[["x0", "x1"]].values, dtype=torch.float32)
y_test = torch.tensor(test["y"].values, dtype=torch.float32)
perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))
val_idx, train_idx = perm[:100], perm[100:]
loss_fn = nn.BCEWithLogitsLoss()


def train_net(opt="adam", lr=3e-3, batch_size=32, epochs=300, patience=None, cosine=False, seed=0, wide=False):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    w = 64 if wide else 16
    net = nn.Sequential(nn.Linear(2, w), nn.Tanh(), nn.Linear(w, w), nn.Tanh(), nn.Linear(w, 1))
    optimizers = {
        "sgd": lambda p: torch.optim.SGD(p, lr=lr, momentum=0.9),
        "adam": lambda p: torch.optim.Adam(p, lr=lr),
        "adamw": lambda p: torch.optim.AdamW(p, lr=lr, weight_decay=0.1),
        "rmsprop": lambda p: torch.optim.RMSprop(p, lr=lr),
    }
    optimizer = optimizers[opt](net.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs) if cosine else None
    history = {"train": [], "val": []}
    best = (np.inf, None, 0)                       # (beste val-tap, vekter, epoke)
    for epoch in range(epochs):
        net.train()
        p = train_idx[torch.randperm(len(train_idx), generator=g)]
        for b in range(0, len(p), batch_size):
            j = p[b:b + batch_size]
            loss = loss_fn(net(X[j]).squeeze(1), y[j])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler:
            scheduler.step()                       # én gang per epoke
        net.eval()
        with torch.no_grad():
            history["train"].append(loss_fn(net(X[train_idx]).squeeze(1), y[train_idx]).item())
            history["val"].append(loss_fn(net(X[val_idx]).squeeze(1), y[val_idx]).item())
        if history["val"][-1] < best[0]:
            best = (history["val"][-1], {k: v.clone() for k, v in net.state_dict().items()}, epoch)
        elif patience is not None and epoch - best[2] > patience:
            break                                  # early stopping
    net.load_state_dict(best[1])                   # tilbake til beste epoke
    net.eval()
    history["best_epoch"] = best[2]
    return net, history


def test_acc(net):
    with torch.no_grad():
        return ((net(X_test).squeeze(1) > 0).float() == y_test).float().mean().item()


net, hist = train_net(patience=40, wide=True)
print("stoppet etter", len(hist["val"]), "epoker, beste epoke:", hist["best_epoch"])
for e in [0, hist["best_epoch"], len(hist["val"]) - 1]:
    print(f"  epoke {e:3d}: train {hist['train'][e]:.3f}  val {hist['val'][e]:.3f}")
print("test-accuracy med beste vekter:", round(test_acc(net), 3))
```
```output
stoppet etter 49 epoker, beste epoke: 7
  epoke   0: train 0.665  val 0.686
  epoke   7: train 0.337  val 0.368
  epoke  48: train 0.225  val 0.390
test-accuracy med beste vekter: 0.876
```

## Tolke tapskurvene
Plotter du `hist["train"]` og `hist["val"]` (som i prosjekt 3), ser du typisk dette:
- Begge synker i starten: nettet lærer det generelle mønsteret.
- Treningstapet fortsetter å synke, mens valideringstapet flater ut og så **stiger**: nettet har begynt å tilpasse seg støyen i treningsdataene (**overtilpasning**).
- Den beste epoken er der valideringstapet er lavest. Det er den early stopping tar vare på.

Tallene over viser nettopp det. Vi brukte et bredt nett (64 × 64) som overtilpasser raskt: treningstapet er lavere ved siste epoke enn ved den beste, men valideringstapet er høyere. Uten early stopping ville vi endt med de dårligere vektene fra siste epoke. Sluttverdiene er derfor ikke et godt mål på modellen. Vi bruker kurvene til å velge når vi skal stoppe, og måler modellen på testdataene.

## Optimizere
Alle optimizere tar `net.parameters()` og `lr`, og brukes med de samme fire stegene:

| Optimizer | Idé |
|---|---|
| `torch.optim.SGD(p, lr, momentum=0.9)` | gradient descent, momentum husker tidligere retninger og jevner ut støy |
| `torch.optim.RMSprop(p, lr)` | egen læringsrate per parameter, skalert med løpende snitt av kvadrerte gradienter |
| `torch.optim.Adam(p, lr)` | momentum + RMSprop. Godt standardvalg |
| `torch.optim.AdamW(p, lr, weight_decay=...)` | Adam med weight decay (L2-lignende regularisering) gjort riktig |

`weight_decay` drar vektene mot 0 i hvert steg og virker som regularisering. Prosjekt 4 bruker `torch.optim.Adam(..., weight_decay=0.01)`. Sammenligning på prosjekt 3-dataene (fortsettelse av koden over):

```python statisk fortsett
for opt, lr in [("sgd", 3e-2), ("rmsprop", 3e-3), ("adam", 3e-3), ("adamw", 3e-3)]:
    net, hist = train_net(opt=opt, lr=lr, patience=40)
    print(f"{opt:<8} lr={lr:<6} beste epoke {hist['best_epoch']:3d}  "
          f"val-tap {min(hist['val']):.3f}  test-acc {test_acc(net):.3f}")
```
```output
sgd      lr=0.03   beste epoke  43  val-tap 0.334  test-acc 0.906
rmsprop  lr=0.003  beste epoke  26  val-tap 0.345  test-acc 0.896
adam     lr=0.003  beste epoke  26  val-tap 0.349  test-acc 0.914
adamw    lr=0.003  beste epoke  26  val-tap 0.353  test-acc 0.916
```

Forskjellene er små på et så enkelt problem. Legg også merke til at lavest valideringstap ikke alltid gir høyest test-accuracy: valideringssettet har bare 100 punkter, og tapet måler noe annet enn accuracy.

## Full batch eller mini-batch?
`batch_size` lik antall treningspunkter gir full batch: én presis oppdatering per epoke. Mini-batch gir mange støyete oppdateringer per epoke:

```python statisk fortsett
for bs in [400, 32]:
    net, hist = train_net(batch_size=bs, lr=1e-2 if bs == 400 else 3e-3, patience=40)
    n_opp = -(-400 // bs)          # tak-divisjon: ceil(400 / bs)
    print(f"batch_size {bs:3d}: {n_opp:2d} oppdateringer/epoke, beste epoke {hist['best_epoch']:3d}, "
          f"test-acc {test_acc(net):.3f}")
```
```output
batch_size 400:  1 oppdateringer/epoke, beste epoke  50, test-acc 0.908
batch_size  32: 13 oppdateringer/epoke, beste epoke  26, test-acc 0.914
```

Full batch gjør bare én oppdatering per epoke. Den trenger derfor flere epoker (her nesten dobbelt så mange) og en høyere læringsrate (`1e-2` mot `3e-3`) for å komme like langt. Til gjengjeld er hver oppdatering presis. I prosjektet var mini-batch best for klassifisereren, mens full batch var best for regressoren.

## Læringsrate-scheduler: cosine annealing
En **scheduler** endrer læringsraten underveis. `CosineAnnealingLR(optimizer, T_max)` senker den langs en cosinuskurve fra startverdien $\eta_0$ ned mot $\eta_{min}$ (standard 0) over `T_max` kall til `scheduler.step()`:

$$\eta_t = \eta_{min} + \tfrac{1}{2}(\eta_0 - \eta_{min})\left(1 + \cos\left(\tfrac{\pi t}{T_{max}}\right)\right)$$

Store steg i starten for å komme raskt fremover, små steg til slutt for å finjustere. I prosjekt 3 er `T_max = epochs`, og `scheduler.step()` kalles én gang per epoke, **etter** batch-løkka:

```python statisk
import torch
import torch.nn as nn

net = nn.Linear(2, 1)
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    lr = optimizer.param_groups[0]["lr"]
    # ... batch-løkka med optimizer.step() her ...
    optimizer.step()
    scheduler.step()
    print(f"epoke {epoch}: lr = {lr:.4f}")
```
```output
epoke 0: lr = 0.1000
epoke 1: lr = 0.0976
epoke 2: lr = 0.0905
epoke 3: lr = 0.0794
epoke 4: lr = 0.0655
epoke 5: lr = 0.0500
epoke 6: lr = 0.0345
epoke 7: lr = 0.0206
epoke 8: lr = 0.0095
epoke 9: lr = 0.0024
```

::: prosjekt Prosjekt 3, problem 1
Prosjektet krevde minst tre teknikker fra listen: full batch vs mini-batch, scheduling (cosine annealing), hyperparametersøk, optimizere, aktiveringer og early stopping. Den endelige klassifisereren var 2 → 16 → 16 → 1 med tanh, Adam(W) med `lr=3e-3`, mini-batcher på 32, cosine annealing og early stopping med patience 40.
:::

::: oppgave Early stopping-logikken
Skriv `early_stopping(val_losses, patience)` som går gjennom valideringstapene epoke for epoke med samme regel som `train_net`:
- er tapet lavere enn det beste så langt, oppdateres beste tap og `best_epoch`
- ellers, hvis `epoch - best_epoch > patience`, stopper vi

Returner `(best_epoch, stop_epoch)`, der `stop_epoch` er epoken der vi stoppet, eller siste epoke hvis vi aldri stoppet.
```start
import numpy as np

def early_stopping(val_losses, patience):
    best_loss, best_epoch = np.inf, 0
    for epoch, loss in enumerate(val_losses):
        ...
    return best_epoch, epoch
```
```losning
import numpy as np

def early_stopping(val_losses, patience):
    best_loss, best_epoch = np.inf, 0
    for epoch, loss in enumerate(val_losses):
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
        elif epoch - best_epoch > patience:
            break
    return best_epoch, epoch
```
```test
v = [1.0, 0.8, 0.7, 0.72, 0.71, 0.75, 0.69, 0.8, 0.9, 0.95, 1.0, 1.1]
r = early_stopping(v, 2)
assert r == (2, 5), f"Med patience 2: beste epoke 2 (0.7), og vi stopper i epoke 5 (5 - 2 > 2). Fikk {r}"
r = early_stopping(v, 3)
assert r == (6, 10), f"Med patience 3 rekker vi å se 0.69 i epoke 6, og stopper i epoke 10. Fikk {r}"
r = early_stopping([3.0, 2.0, 1.0], 1)
assert r == (2, 2), f"Synker tapet hele tiden, er beste og siste epoke 2. Fikk {r}"
```
:::

::: oppgave Cosine annealing
Skriv `cosine_lr(lr0, t, T_max, lr_min=0.0)` som gir læringsraten i epoke `t` etter formelen over. Bruk `np.cos` og `np.pi`.
```start
import numpy as np

def cosine_lr(lr0, t, T_max, lr_min=0.0):
    ...
```
```losning
import numpy as np

def cosine_lr(lr0, t, T_max, lr_min=0.0):
    return lr_min + 0.5 * (lr0 - lr_min) * (1 + np.cos(np.pi * t / T_max))
```
```test
import numpy as np
assert cosine_lr(0.1, 0, 10) is not None, "Funksjonen returnerer ingenting"
assert abs(cosine_lr(0.1, 0, 10) - 0.1) < 1e-12, "Ved t = 0 skal læringsraten være lr0"
assert abs(cosine_lr(0.1, 5, 10) - 0.05) < 1e-12, "Halvveis (t = T_max/2) skal den være halvparten"
assert abs(cosine_lr(0.1, 10, 10)) < 1e-12, "Ved t = T_max skal den være lr_min (0)"
assert abs(cosine_lr(0.1, 2, 10) - 0.0904508497) < 1e-8, f"cosine_lr(0.1, 2, 10) skal være ca. 0.09045 (som PyTorch), fikk {cosine_lr(0.1, 2, 10)}"
assert abs(cosine_lr(0.1, 10, 10, lr_min=0.01) - 0.01) < 1e-12, "Husk lr_min"
```
:::

::: quiz
Hva er feil med denne early stopping-koden?
```kode
if val_loss < best_loss:
    best_loss = val_loss
    best_state = net.state_dict()
...
net.load_state_dict(best_state)
```
- [ ] `load_state_dict` kan bare laste vekter fra en fil
- [x] `state_dict()` gir referanser til de levende vektene, så `best_state` blir vektene fra siste epoke
- [ ] `state_dict()` inneholder bare biasene, ikke vektene
- [ ] Ingenting, koden laster vektene fra beste epoke
? Uten `.clone()` (eller `copy.deepcopy`) peker `best_state` på de samme tensorene som nettet bruker. Hver `optimizer.step()` endrer dem, og du «gjenoppretter» bare de siste vektene. Prosjekt 3 bruker `{k: v.clone() for k, v in net.state_dict().items()}`.
:::

::: quiz
Treningstapet synker hele veien, mens valideringstapet når et minimum i epoke 25 og så stiger jevnt. Hva er riktig tolkning?
- [ ] Læringsraten er for lav, så nettet bør trenes lenger
- [ ] Nettet underfitter, og trenger flere lag
- [x] Nettet overtilpasser seg treningsdataene etter epoke 25, og vektene fra epoke 25 bør brukes
- [ ] Valideringsdataene er feil, siden valideringstapet alltid skal synke når treningstapet synker
? Lavere treningstap og høyere valideringstap betyr at nettet lærer støy i treningsdataene. Early stopping med beste `state_dict` velger epoke 25.
:::

::: quiz
`scheduler = CosineAnnealingLR(optimizer, T_max=epochs)`. Hvor skal `scheduler.step()` stå?
- [ ] Inne i batch-løkka, rett etter `optimizer.step()`
- [x] Én gang per epoke, etter batch-løkka
- [ ] Før `optimizer.zero_grad()` i hver batch
- [ ] Bare én gang, etter all treningen
? Med `T_max=epochs` skal læringsraten gå fra start til minimum over alle epokene, altså ett `scheduler.step()` per epoke. Kalt per batch ville den nådd minimum etter `epochs` batcher (og så gått opp igjen).
:::

::: oppsummering
- Hold av en valideringsdel (`perm[:100]`). Regn ut `train`- og `val`-tap etter hver epoke under `torch.no_grad()`.
- Early stopping: lagre `{k: v.clone() for k, v in net.state_dict().items()}` når val-tapet er best, stopp etter `patience` epoker uten forbedring, og `load_state_dict` til slutt.
- Optimizere: `SGD(momentum=0.9)`, `RMSprop`, `Adam`, `AdamW(weight_decay=...)`.
- `CosineAnnealingLR(optimizer, T_max=epochs)` med `scheduler.step()` én gang per epoke.
- Full batch: 1 oppdatering per epoke. Mini-batch: $\lceil N/B \rceil$.
- Val-tap som stiger mens train-tap synker = overtilpasning.
:::

--- LEKSJON 9.11 | reproduserbarhet | Reproduserbarhet: seeds og determinisme
Prosjekt 3 krevde at notebooken gir **nøyaktig de samme tallene** hver gang den kjøres. Nevrale nett har mange kilder til tilfeldighet: startvektene, stokkingen av mini-batcher, train/val-splitten og Dropout. Alle må låses med en **seed**.

```python statisk
import torch

print(torch.rand(3))
print(torch.rand(3))         # nye tall hver gang
torch.manual_seed(0)
print(torch.rand(3))
torch.manual_seed(0)
print(torch.rand(3))         # samme seed gir samme tall
```
```output
tensor([0.8294, 0.3801, 0.4209])
tensor([0.1553, 0.7212, 0.3073])
tensor([0.4963, 0.7682, 0.0885])
tensor([0.4963, 0.7682, 0.0885])
```

## set_seed fra prosjekt 3
Python, NumPy og PyTorch har hver sin tilfeldighetsgenerator, så alle tre må seedes. I tillegg ber vi PyTorch bruke deterministiske algoritmer og én tråd. Med flere tråder kan summer regnes ut i ulik rekkefølge, og flyttall gir da bittesmå forskjeller som vokser under trening:

```python statisk
import random
import numpy as np
import torch
import torch.nn as nn

SEED = 0

def set_seed(seed=SEED):
    """Fix every source of randomness used in this notebook."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)  # single thread: identical floating point results on every run


def tren(seed):
    set_seed(seed)
    X = torch.randn(200, 2)
    y = (X[:, 0] * X[:, 1] < 0).float()           # XOR-mønster
    net = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    for epoch in range(100):
        optimizer.zero_grad()
        loss = nn.BCEWithLogitsLoss()(net(X).squeeze(1), y)
        loss.backward()
        optimizer.step()
    return loss.item()

print(tren(0))
print(tren(0))     # nøyaktig likt
print(tren(1))     # annen seed, annet resultat
```
```output
0.22152946889400482
0.22152946889400482
0.19518350064754486
```

## Egen Generator for stokking og splitting
`torch.manual_seed` styrer den **globale** generatoren, som brukes av alt: startvekter, `torch.randn`, `torch.randperm` og Dropout. Endrer du arkitekturen, trekkes et annet antall tall til startvektene, og da blir også stokkingen etterpå en annen. Prosjekt 3 bruker derfor en egen `torch.Generator` for stokking og splitting:

```python statisk
import torch
import torch.nn as nn

for bredde in [16, 64]:
    torch.manual_seed(0)
    net = nn.Linear(2, bredde)               # trekker 2*bredde + bredde tall
    global_perm = torch.randperm(8)
    g = torch.Generator().manual_seed(0)
    egen_perm = torch.randperm(8, generator=g)
    print(f"bredde {bredde:2d}: global {global_perm.tolist()}  egen generator {egen_perm.tolist()}")
```
```output
bredde 16: global [6, 0, 1, 7, 5, 2, 3, 4]  egen generator [4, 0, 7, 3, 2, 5, 1, 6]
bredde 64: global [2, 1, 4, 7, 0, 6, 5, 3]  egen generator [4, 0, 7, 3, 2, 5, 1, 6]
```

Med egen generator er stokkingen den samme uansett nettverksstørrelse, så sammenligningen av arkitekturer blir rettferdig. Det samme gjelder `DataLoader(..., shuffle=True, generator=torch.Generator().manual_seed(0))`.

::: prosjekt Prosjekt 3, problem 1
Spørsmålet «How did you ensure reproducibility?» besvares med koden over: `set_seed` (random, NumPy og PyTorch) kalles i starten av hver `train_net`, en egen `torch.Generator().manual_seed(seed)` brukes til `randperm`, og `torch.use_deterministic_algorithms(True)` og `torch.set_num_threads(1)` gjør flyttallsregningen lik fra kjøring til kjøring.
:::

::: oppgave set_seed
Skriv `set_seed(seed)` som seeder både Pythons `random`-modul og NumPy (`np.random.seed`), slik at koden i testen gir de samme tallene to ganger.
```start
import random
import numpy as np

def set_seed(seed):
    ...
```
```losning
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
```
```test
import random
import numpy as np
set_seed(3)
a = (random.random(), np.random.rand(3))
set_seed(3)
b = (random.random(), np.random.rand(3))
assert a[0] == b[0], "random.random() gir ulike tall. Husk random.seed(seed)."
assert np.array_equal(a[1], b[1]), "np.random.rand gir ulike tall. Husk np.random.seed(seed)."
set_seed(4)
assert random.random() != a[0], "Bruk seed-argumentet, ikke et fast tall"
```
:::

::: quiz
En student kaller bare `torch.manual_seed(0)` i starten, men splitter dataene med `np.random.permutation` og stokker med `random.shuffle`. Hva skjer når notebooken kjøres på nytt?
- [ ] Alt blir likt, fordi `torch.manual_seed` seeder alle generatorer
- [ ] Startvektene blir ulike, men splitten blir lik
- [x] Startvektene blir like, men splitten og stokkingen blir ulike
- [ ] Ingenting blir likt, fordi seeden må settes etter at modellen er laget
? `torch.manual_seed` styrer bare PyTorch. NumPy og `random` har egne generatorer som må seedes med `np.random.seed` og `random.seed`. Det er derfor `set_seed` i prosjekt 3 setter alle tre.
:::

::: quiz
Hvorfor bruker prosjekt 3 `torch.randperm(n, generator=g)` med `g = torch.Generator().manual_seed(seed)` i stedet for bare `torch.randperm(n)`?
- [ ] `torch.randperm(n)` er ikke tilfeldig uten en generator
- [x] Stokkingen blir uavhengig av hvor mange tall som ble trukket til startvektene, så den er lik for alle arkitekturer
- [ ] En egen generator gjør treningen raskere
- [ ] Det er påkrevd når `torch.use_deterministic_algorithms(True)` er satt
? Den globale generatoren deles av alt. Et bredere nett bruker flere tilfeldige tall til startvektene, og da endres stokkingen etterpå. En egen generator gir samme rekkefølge uansett.
:::

::: oppsummering
- Seed alle tre: `random.seed`, `np.random.seed` og `torch.manual_seed`.
- `torch.use_deterministic_algorithms(True)` og `torch.set_num_threads(1)` gir identiske flyttall.
- Egen `torch.Generator().manual_seed(seed)` for `randperm` og `DataLoader` gjør stokkingen uavhengig av modellen.
- Sjekk: kjør to ganger med samme seed og se at tallene er like.
:::

--- LEKSJON 9.12 | keras | Keras-versjonen fra notatene
Forelesningsnotatene bruker **Keras** (med TensorFlow), og prosjekt 3 lot deg velge Keras eller PyTorch. Ideene er de samme, men Keras skjuler treningsløkka i `model.fit`. Keras er ikke installert her, så koden vises uten å kjøres. Utdataene som står, er hentet fra notatene.

## Bygge en modell
Notatene lager en tensor med ettall og et nett med to skjulte lag (6 og 7 noder) og 2 output-noder med softmax:

```python statisk ikkekjor
import keras
from keras import ops
from keras.layers import Dense
from keras.models import Sequential

x = ops.ones((10, 3))           # 10 datapunkter, 3 features

model = Sequential()
model.add(Dense(6, activation="relu", name="layer1"))
model.add(Dense(7, activation="relu", name="layer2"))
model.add(Dense(2, activation="softmax", name="output"))
model(x)                        # shape (10, 2)
```

Det samme kan skrives med en liste, eller uten `Sequential` ved å kalle lagene etter hverandre:

```python statisk ikkekjor
model = Sequential([
    Dense(6, activation="relu", name="layer1"),
    Dense(7, activation="relu", name="layer2"),
    Dense(2, activation="softmax", name="output"),
])

layer1 = Dense(6, activation="relu", name="layer1")
layer2 = Dense(7, activation="relu", name="layer2")
layer3 = Dense(2, activation="softmax", name="output")
layer3(layer2(layer1(x)))       # ingen modell-objekt, bare lag kalt i rekkefølge
```

`Dense` er det samme som `nn.Linear` pluss en aktivering. `Dense(6)` sier bare hvor mange noder laget har **ut**. Antall inn finner Keras selv. Output-shape er alltid `(batch, antall noder i siste lag)`, her `(10, 2)`. Alle radene blir like, fordi alle radene i `x` er like, og verdiene er meningsløse fordi vektene er tilfeldige.

## Ingen vekter før modellen er bygget
Siden `Dense(6)` ikke vet hvor mange inputs den får, kan ikke vektmatrisen lages før modellen har sett data:

```python statisk ikkekjor
model = Sequential()
model.add(Dense(6, activation="relu", name="layer1"))
model.weights          # før modellen er kalt på data
```
```output
[]
```

`model.summary()` viser da `?` som output-shape. Kaller du `model(x)` én gang, eller oppgir input-shape med `keras.Input`, blir vektene laget:

```python statisk ikkekjor
model = keras.Sequential()
model.add(keras.Input(shape=(3,)))
model.add(Dense(6, activation="relu", name="layer1"))
model.summary()        # output-shape (None, 6), 24 parametre
model(x).shape
```
```output
TensorShape([10, 6])
```

24 parametre er $3 \cdot 6 + 6$, akkurat som i leksjon 9.2. `None` i shapen betyr «vilkårlig batch-størrelse». Keras lagrer vektene (kalt `kernel`) som `(inn, ut)` = `(3, 6)`, motsatt av `nn.Linear`.

## compile, fit og history
`model.compile` velger optimizer, tapsfunksjon og metrikker. Ingen vekter endres før `model.fit`, som kjører hele treningsløkka (zero_grad, forward, backward, step, mini-batcher og epoker) for deg:

```python statisk ikkekjor
model.compile(
    optimizer=keras.optimizers.SGD(),
    loss="binary_crossentropy",         # passer til Dense(1, activation="sigmoid")
    metrics=["accuracy"],
)
history = model.fit(x_train, y_train, batch_size=64,
                    epochs=10, validation_data=(x_val, y_val))

train_losses = history.history["loss"]
val_losses = history.history["val_loss"]
epochs = np.arange(1, len(train_losses) + 1)
plt.plot(epochs, train_losses, label="Train loss")
plt.plot(epochs, val_losses, label="Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

`history.history` er en ordbok med én liste per metrikk, med én verdi per epoke. Det er det samme som `history = {"train": [], "val": []}` i `train_net`.

## Early stopping, predict, evaluate, save og load
Early stopping er en **callback** som sendes til `fit`. `restore_best_weights=True` gjør det samme som å laste inn beste `state_dict`:

```python statisk ikkekjor
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                           restore_best_weights=True)
history = model.fit(x_train, y_train, batch_size=64, epochs=200,
                    validation_data=(x_val, y_val), callbacks=[early_stop])

y_prob = model.predict(x_test)                        # forward pass, ingen trening
test_loss, test_acc = model.evaluate(x_test, y_test)  # tap og metrikker
model.save("sted/for/lagring/navn.keras")
model = keras.models.load_model("sted/for/lagring/navn.keras")
```

## Prosjekt 3-klassifisereren i Keras
Til sammenligning med PyTorch-versjonen i leksjon 9.10:

```python statisk ikkekjor
keras.utils.set_random_seed(0)          # seeder Python, NumPy og backend
model = keras.Sequential([
    keras.Input(shape=(2,)),
    Dense(16, activation="tanh"),
    Dense(16, activation="tanh"),
    Dense(1),                            # logit, som i PyTorch-versjonen
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-3),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy(threshold=0.0)])
history = model.fit(X_train, y_train, batch_size=32, epochs=300,
                    validation_data=(X_val, y_val),
                    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=40,
                                                             restore_best_weights=True)])
print(model.count_params())              # 337
```

## Oversettelse Keras ↔ PyTorch

| Keras | PyTorch |
|---|---|
| `Dense(16, activation="tanh")` | `nn.Linear(inn, 16)`, `nn.Tanh()` |
| `Sequential([...])`, `model.add(lag)` | `nn.Sequential(...)` |
| `keras.Input(shape=(2,))` | inn-størrelsen i første `nn.Linear(2, ...)` |
| kernel med shape `(inn, ut)` | `weight` med shape `(ut, inn)` |
| `model.summary()`, `model.count_params()` | `print(net)`, `sum(p.numel() for p in net.parameters())` |
| `model.weights`, `model.get_weights()` | `list(net.parameters())`, `net.state_dict()` |
| `model.compile(optimizer, loss, metrics)` | lag `optimizer` og `loss_fn` selv |
| `model.fit(x, y, batch_size, epochs, validation_data)` | treningsløkka + `DataLoader` |
| `history.history["loss"]`, `["val_loss"]` | egne lister med `loss.item()` |
| `EarlyStopping(monitor="val_loss", patience, restore_best_weights=True)` | beste `state_dict` med `.clone()` + `patience` |
| `model.predict(x)` | `net.eval()`, `with torch.no_grad(): net(x)` |
| `model.evaluate(x, y)` | regn ut tap og accuracy selv |
| `model.save(...)`, `keras.models.load_model(...)` | `torch.save(net.state_dict(), ...)`, `net.load_state_dict(torch.load(...))` |
| `BinaryCrossentropy(from_logits=True)` | `nn.BCEWithLogitsLoss()` |
| `"binary_crossentropy"` + sigmoid i siste lag | `nn.BCELoss()` + `nn.Sigmoid()` |
| `"sparse_categorical_crossentropy"` + softmax | `nn.CrossEntropyLoss()` på logits |
| `keras.optimizers.SGD()`, `Adam(learning_rate=...)` | `torch.optim.SGD(...)`, `torch.optim.Adam(..., lr=...)` |
| `keras.utils.set_random_seed(0)` | `set_seed(0)` fra leksjon 9.11 |

::: oppgave Et Dense-lag som bygger seg selv
Lag klassen `Dense` i NumPy som oppfører seg som i Keras: vektene lages først når laget kalles på data.
- `__init__(self, units)` lagrer `self.units` og setter `self.kernel = None` og `self.bias = None`.
- `__call__(self, X)`: hvis `self.kernel` er `None`, lag `self.kernel` med shape `(X.shape[1], units)` (bruk `np.random.randn`) og `self.bias = np.zeros(units)`. Returner så `X @ self.kernel + self.bias`.
- `count_params(self)` returnerer antall parametre (0 før laget er bygget).
```start
import numpy as np

class Dense:
    def __init__(self, units):
        ...

    def __call__(self, X):
        ...

    def count_params(self):
        ...
```
```losning
import numpy as np

class Dense:
    def __init__(self, units):
        self.units = units
        self.kernel = None
        self.bias = None

    def __call__(self, X):
        if self.kernel is None:
            self.kernel = np.random.randn(X.shape[1], self.units)
            self.bias = np.zeros(self.units)
        return X @ self.kernel + self.bias

    def count_params(self):
        if self.kernel is None:
            return 0
        return self.kernel.size + self.bias.size
```
```test
import numpy as np
lag = Dense(6)
assert getattr(lag, "kernel", "mangler") is None, "self.kernel skal være None før laget er kalt"
assert lag.count_params() == 0, "Før laget er bygget skal count_params() gi 0"
x = np.ones((10, 3))
out = lag(x)
assert out is not None and np.shape(out) == (10, 6), f"Output skal ha shape (10, 6), fikk {np.shape(out)}"
assert np.shape(lag.kernel) == (3, 6), f"kernel skal ha shape (inn, ut) = (3, 6), har {np.shape(lag.kernel)}"
assert lag.count_params() == 24, f"Dense(6) med 3 input skal ha 24 parametre, fikk {lag.count_params()}"
k = lag.kernel.copy()
lag(x)
assert np.array_equal(k, lag.kernel), "Vektene skal bare lages første gang, ikke ved hvert kall"
```
:::

::: quiz
Hva skriver den siste linjen ut?
```kode
model = Sequential()
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
print(model.weights)
```
- [x] En tom liste, fordi modellen ikke kjenner input-dimensjonen og ikke har laget vektene
- [ ] Tilfeldige startvekter, fordi `Dense` lager vektene når laget legges til
- [ ] Vekter som er 0, fordi modellen ikke er trent med `model.fit`
- [ ] En feilmelding, fordi modellen må kompileres før vektene finnes
? Uten `keras.Input(shape=...)` eller et kall på data vet ikke det første laget hvor mange inputs det får, så vektmatrisen kan ikke lages ennå. Notatene: `model.weights` gir `[]`.
:::

::: quiz
Hvor mange parametre viser `model.summary()`?
```kode
model = keras.Sequential()
model.add(keras.Input(shape=(3,)))
model.add(Dense(6, activation="relu"))
model.add(Dense(7, activation="relu"))
model.add(Dense(2, activation="softmax"))
```
- [ ] 24
- [ ] 74
- [x] 89
- [ ] 15
? $(3 \cdot 6 + 6) + (6 \cdot 7 + 7) + (7 \cdot 2 + 2) = 24 + 49 + 16 = 89$. 24 er bare det første laget, 74 er uten bias-leddene, og 15 er antall noder.
:::

::: quiz
Hva skrives ut?
```kode
model = keras.Sequential([keras.Input(shape=(20,)),
                          Dense(32, activation="relu"),
                          Dense(5, activation="softmax")])
W, b = model.layers[0].get_weights()
print(W.shape, b.shape)
```
- [x] `(20, 32) (32,)`
- [ ] `(32, 20) (32,)`
- [ ] `(20, 32) (20,)`
- [ ] `(32, 5) (5,)`
? Keras lagrer kernel som `(inn, ut)` = `(20, 32)`, og én bias per node: `(32,)`. `(32, 20)` er PyTorch-konvensjonen for `nn.Linear(20, 32).weight`. `model.layers[0]` er det første `Dense`-laget, siden `Input` ikke regnes som et lag.
:::

::: oppsummering
- `Sequential` + `Dense(noder, activation=...)`. Output-shape = `(batch, noder i siste lag)`.
- Vektene finnes ikke før modellen er kalt på data eller har fått `keras.Input(shape=...)`. Før det gir `model.weights` en tom liste.
- `compile` velger optimizer, tap og metrikker. `fit` trener og returnerer `history` med `history.history["loss"]` og `["val_loss"]`.
- `EarlyStopping(monitor="val_loss", patience=..., restore_best_weights=True)` sendes som callback.
- `predict`, `evaluate`, `save` og `keras.models.load_model` for inferens, evaluering og lagring.
- Keras kernel er `(inn, ut)`, PyTorch weight er `(ut, inn)`.
:::
