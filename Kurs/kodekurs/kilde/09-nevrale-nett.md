=== DEL 9 | nevrale-nett | Nevrale nettverk (prosjekt 3)
Fra ett nevron i NumPy via `MLPClassifier` til treningsløkkene i PyTorch og Keras fra prosjekt 3 og 4.

--- LEKSJON 9.1 | nevron-aktivering | Nevron og aktiveringsfunksjoner
Et nevron regner ut en vektet sum `z = np.dot(w, x) + b` og sender den gjennom en aktiveringsfunksjon, f.eks. sigmoid `a = 1 / (1 + np.exp(-z))`.

::: oppgave Ett nevron
Regn ut den vektede summen `z` og aktiveringen `a` (sigmoid) for nevronet under.
```start
import numpy as np

x = np.array([1.0, 2.0, -1.0])
w = np.array([0.5, -0.25, 1.0])
b = 0.1

z = ...
a = ...
```
```losning
import numpy as np

x = np.array([1.0, 2.0, -1.0])
w = np.array([0.5, -0.25, 1.0])
b = 0.1

z = np.dot(w, x) + b
a = 1 / (1 + np.exp(-z))
```
```test
import numpy as np
assert np.ndim(z) == 0 and isinstance(z, (int, float, np.floating)), "z skal være ett tall: np.dot(w, x) + b"
assert abs(z - (-0.9)) < 1e-9, f"z skal være 0.5 - 0.5 - 1.0 + 0.1 = -0.9, fikk {z}. Husk biasen."
assert isinstance(a, (int, float, np.floating)) and abs(a - 1 / (1 + np.exp(0.9))) < 1e-9, f"a skal være sigmoid(z) = 1 / (1 + np.exp(-z)), fikk {a}"
```
:::

## Perseptronet
Perseptronet bruker Heaviside `np.where(z >= 0, 1.0, 0.0)` som aktivering. Den har derivert 0 overalt, så den kan ikke trenes med gradient descent, og det er derfor vi bytter til en glatt funksjon som sigmoid.

::: oppgave AND og OR med ett perseptron
Skriv `heaviside(z)`, og regn ut output for alle fire punktene i `X` på én gang med `X @ w + b`: `y_and` med `b = -1.5` og `y_or` med `b = -0.5`.
```start
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
w = np.array([1.0, 1.0])

def heaviside(z):
    ...

y_and = ...
y_or = ...
```
```losning
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
w = np.array([1.0, 1.0])

def heaviside(z):
    return np.where(z >= 0, 1.0, 0.0)

y_and = heaviside(X @ w - 1.5)
y_or = heaviside(X @ w - 0.5)
```
```test
import numpy as np
h = heaviside(np.array([-1.0, 0.0, 2.0]))
assert h is not None, "heaviside returnerer ingenting. Husk return."
assert np.allclose(h, [0, 1, 1]), f"heaviside([-1, 0, 2]) skal gi [0, 1, 1] (z >= 0 gir 1), fikk {h}"
assert np.shape(y_and) == (4,), "y_and skal ha én verdi per punkt i X. Bruk heaviside(X @ w + b)."
assert np.allclose(y_and, [0, 0, 0, 1]), f"AND skal gi [0, 0, 0, 1], fikk {y_and}. Bruk b = -1.5."
assert np.allclose(y_or, [0, 1, 1, 1]), f"OR skal gi [0, 1, 1, 1], fikk {y_or}. Bruk b = -0.5."
```
:::

## Aktiveringsfunksjoner
ReLU er `np.maximum(0, z)` (ikke `np.max(z)`, som gir det største elementet), tanh er `np.tanh(z)`, og SiLU er `z * sigmoid(z)`. GELU er en lignende glatt variant av ReLU, `z * Φ(z)` med normalfordelingens CDF.

::: oppgave ReLU, sigmoid og SiLU
Skriv `relu(z)`, `sigmoid(z)` og `silu(z)`, som alle virker elementvis på en hel array uten løkker.
```start
import numpy as np

def relu(z):
    ...

def sigmoid(z):
    ...

def silu(z):
    ...
```
```losning
import numpy as np

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def silu(z):
    return z * sigmoid(z)
```
```test
import numpy as np
z = np.array([-2.0, 0.0, 3.0])
r = relu(z)
assert r is not None, "relu returnerer ingenting. Husk return."
assert np.shape(r) == (3,) and np.allclose(r, [0.0, 0.0, 3.0]), f"relu([-2, 0, 3]) skal være [0, 0, 3], fikk {r}. Bruk np.maximum(0, z), ikke np.max."
s = sigmoid(z)
assert s is not None, "sigmoid returnerer ingenting. Husk return."
assert np.allclose(s, [0.11920292, 0.5, 0.95257413]), f"sigmoid(z) skal være 1 / (1 + e^(-z)), fikk {s}"
u = silu(z)
assert u is not None, "silu returnerer ingenting. Husk return."
assert np.allclose(u, z / (1 + np.exp(-z))), f"silu(z) skal være z * sigmoid(z), fikk {u}"
```
:::

## Softmax
Softmax `np.exp(z) / np.sum(np.exp(z))` gjør én logit per klasse om til sannsynligheter som summerer til 1. Trekk fra maks først, ellers blir `np.exp(1000)` uendelig og svaret `nan`.

::: oppgave Softmax for en hel batch
Skriv `softmax(Z)` der `Z` har shape `(N, C)`, slik at hver rad summerer til 1 og store tall som `1000` går fint. Bruk `axis=1, keepdims=True` både for maks og sum.
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

## Perseptron i scikit-learn
`Perceptron().fit(X, y)` lagrer vektene i `coef_` med shape `(klasser, features)` og biasene i `intercept_` med shape `(klasser,)`. Med bare to klasser er det én output-node, så `coef_` har shape `(1, features)`.

::: oppgave Vektene til et perseptron
Tren `perceptron = Perceptron().fit(X, y)` på dataene med 2 features og 3 klasser, og lagre antall vekter i `n_vekter` og antall bias-verdier i `n_bias`.
```start
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron

X, y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, n_classes=3, random_state=0)

perceptron = ...
n_vekter = ...
n_bias = ...
```
```losning
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron

X, y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, n_classes=3, random_state=0)

perceptron = Perceptron().fit(X, y)
n_vekter = perceptron.coef_.size
n_bias = perceptron.intercept_.size
```
```test
from sklearn.linear_model import Perceptron
assert isinstance(perceptron, Perceptron) and hasattr(perceptron, "coef_"), "perceptron skal være en trent Perceptron: Perceptron().fit(X, y)"
assert n_vekter == 6, f"2 features og 3 klasser gir 2 * 3 = 6 vekter (perceptron.coef_.size), fikk {n_vekter}"
assert n_bias == 3, f"Det er én bias per klasse, altså 3 (perceptron.intercept_.size), fikk {n_bias}"
```
:::

--- LEKSJON 9.2 | lag-matriser | Lag som matriser og parametertelling
Et helt lag for en hel batch er `H = g(X @ W + b)`, med shapes `(N, inn) @ (inn, ut)` som gir `(N, ut)`, og biasen legges på hver rad med broadcasting.

::: oppgave Ett lag med tanh
Regn ut laget `H` med tanh-aktivering for de 5 datapunktene i `X`, og lagre shapen i `form`.
```start
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(5, 2))
W = rng.normal(size=(2, 3))
b = np.array([0.1, 0.0, -0.1])

H = ...
form = ...
```
```losning
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(5, 2))
W = rng.normal(size=(2, 3))
b = np.array([0.1, 0.0, -0.1])

H = np.tanh(X @ W + b)
form = H.shape
```
```test
import numpy as np
assert np.shape(H) == (5, 3), f"H skal ha shape (5, 3): (5, 2) @ (2, 3). Fikk {np.shape(H)}"
assert np.allclose(H, np.tanh(X @ W + b)), "H skal være np.tanh(X @ W + b). Husk biasen."
assert tuple(form) == (5, 3), "form skal være H.shape"
```
:::

## Forward pass
Et nett er lag etter lag der output fra ett lag er input til neste. Siste lag har ingen aktivering, så output er en logit (eller en regresjonsverdi).

::: oppgave Forward pass
Skriv `forward(X, W1, b1, W2, b2)` for et nett med ett skjult lag: ReLU i det skjulte laget og ingen aktivering i output-laget.
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

::: oppgave Klassifisereren fra prosjekt 3
Lag tilfeldige vekter `W1`, `W2`, `W3` (med `rng.normal(size=...)`) og null-biaser `b1`, `b2`, `b3` for nettet 2 → 16 → 16 → 1, og regn ut `logit` for batchen `X` med tanh i de to skjulte lagene.
```start
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(8, 2))

W1, b1 = ..., ...
W2, b2 = ..., ...
W3, b3 = ..., ...
logit = ...
```
```losning
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(8, 2))

W1, b1 = rng.normal(size=(2, 16)), np.zeros(16)
W2, b2 = rng.normal(size=(16, 16)), np.zeros(16)
W3, b3 = rng.normal(size=(16, 1)), np.zeros(1)
H1 = np.tanh(X @ W1 + b1)
H2 = np.tanh(H1 @ W2 + b2)
logit = H2 @ W3 + b3
```
```test
import numpy as np
assert np.shape(W1) == (2, 16) and np.shape(W2) == (16, 16) and np.shape(W3) == (16, 1), f"Vektene skal ha shapes (2, 16), (16, 16) og (16, 1), fikk {np.shape(W1)}, {np.shape(W2)}, {np.shape(W3)}"
assert np.shape(b1) == (16,) and np.shape(b2) == (16,) and np.shape(b3) == (1,), "Biasene skal ha shapes (16,), (16,) og (1,)"
assert np.shape(logit) == (8, 1), f"logit skal ha shape (8, 1), fikk {np.shape(logit)}"
fasit = np.tanh(np.tanh(X @ W1 + b1) @ W2 + b2) @ W3 + b3
assert np.allclose(logit, fasit), "logit er feil. Bruk tanh i de to skjulte lagene og ingen aktivering i siste lag."
```
:::

## Telle parametre
Et lag fra `inn` til `ut` noder har `inn * ut + ut` parametre, og `zip(s[:-1], s[1:])` gir nabopar av lagstørrelser.

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
assert antall_parametre([2, 128, 128, 128, 1]) == 33537, "2 -> 128 -> 128 -> 128 -> 1 (regressoren i prosjekt 3) skal gi 33537"
assert antall_parametre([784, 128, 64, 10]) == 109386, "784 -> 128 -> 64 -> 10 (prosjekt 4) skal gi 109386"
```
:::

## Ikke-lineær aktivering
Uten aktivering er `(X @ W1) @ W2` det samme som `X @ (W1 @ W2)`, så mange lineære lag kollapser til ett lineært lag og kan aldri lære XOR.

::: oppgave To lineære lag er ett
Lag `W_samlet` slik at `X @ W_samlet` gir det samme som `(X @ W1) @ W2`, og regn ut `med_relu = np.maximum(0, X @ W1) @ W2`, som ikke kan skrives slik.
```start
import numpy as np

rng = np.random.default_rng(1)
X = rng.normal(size=(4, 2))
W1 = rng.normal(size=(2, 16))
W2 = rng.normal(size=(16, 1))

W_samlet = ...
med_relu = ...
```
```losning
import numpy as np

rng = np.random.default_rng(1)
X = rng.normal(size=(4, 2))
W1 = rng.normal(size=(2, 16))
W2 = rng.normal(size=(16, 1))

W_samlet = W1 @ W2
med_relu = np.maximum(0, X @ W1) @ W2
```
```test
import numpy as np
assert np.shape(W_samlet) == (2, 1), f"W_samlet skal ha shape (2, 1), fikk {np.shape(W_samlet)}. Gang sammen W1 og W2."
assert np.allclose(X @ W_samlet, (X @ W1) @ W2), "X @ W_samlet skal være lik (X @ W1) @ W2"
assert np.shape(med_relu) == (4, 1) and np.allclose(med_relu, np.maximum(0, X @ W1) @ W2), "med_relu skal være np.maximum(0, X @ W1) @ W2"
assert not np.allclose(med_relu, X @ W_samlet), "Med ReLU skal resultatet bli annerledes enn det lineære"
```
:::

--- LEKSJON 9.3 | backprop-numpy | Backpropagation i NumPy
Backprop er kjerneregelen bakover lag for lag, og hver gradient har samme shape som parameteren den hører til. I PyTorch gjør `loss.backward()` dette for deg.

::: oppgave Kjerneregelen for et bitte lite nett
Nettet er `h = tanh(w1 * x)`, `y_hat = w2 * h` med tap `L = (y_hat - t)**2`. Lagre `dw2 = 2(y_hat - t) * h` og `dw1 = 2(y_hat - t) * w2 * (1 - h**2) * x`.
```start
import numpy as np

x, t = 1.5, 0.5
w1, w2 = 0.8, -1.2

h = np.tanh(w1 * x)
y_hat = w2 * h

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

## Gradienter for ett lag
For et lag `Z = A @ W + b` er `dW = A.T @ dZ`, `db = dZ.sum(axis=0)`, og `dA = dZ @ W.T` sender gradienten videre bakover til laget før.

::: oppgave Tre gradienter
Gitt input `A`, vektene `W` og gradienten `dZ`, regn ut `dW`, `db` og `dA`.
```start
import numpy as np

rng = np.random.default_rng(0)
A = rng.normal(size=(3, 2))
W = rng.normal(size=(2, 4))
dZ = rng.normal(size=(3, 4))

dW = ...
db = ...
dA = ...
```
```losning
import numpy as np

rng = np.random.default_rng(0)
A = rng.normal(size=(3, 2))
W = rng.normal(size=(2, 4))
dZ = rng.normal(size=(3, 4))

dW = A.T @ dZ
db = dZ.sum(axis=0)
dA = dZ @ W.T
```
```test
import numpy as np
assert np.shape(dW) == (2, 4), f"dW skal ha samme shape som W, (2, 4). Fikk {np.shape(dW)}. Bruk A.T @ dZ."
assert np.allclose(dW, A.T @ dZ), "dW skal være A.T @ dZ"
assert np.shape(db) == (4,) and np.allclose(db, dZ.sum(axis=0)), "db skal være dZ.sum(axis=0), med shape (4,)"
assert np.shape(dA) == (3, 2), f"dA skal ha samme shape som A, (3, 2). Fikk {np.shape(dA)}. Bruk dZ @ W.T."
assert np.allclose(dA, dZ @ W.T), "dA skal være dZ @ W.T"
```
:::

## Backward for et helt nett
Med tanh i det skjulte laget er `dZ1 = dH * (1 - H**2)`, og sigmoid sammen med binær cross-entropy gir den enkle output-gradienten `dz2 = (p - y) / N`.

::: oppgave Backward for XOR-nettet
Fullfør `backward` for nettet 2 → 4 → 1 (tanh, så sigmoid og binær cross-entropy), og returner `dW1, db1, dW2, db2`.
```start
import numpy as np

def backward(X, y, W1, b1, W2, b2):
    N = len(X)
    H = np.tanh(X @ W1 + b1)
    p = 1 / (1 + np.exp(-(H @ W2 + b2)))
    dz2 = ...
    dW2 = ...
    db2 = ...
    dH = ...
    dZ1 = ...
    dW1 = ...
    db1 = ...
    return dW1, db1, dW2, db2
```
```losning
import numpy as np

def backward(X, y, W1, b1, W2, b2):
    N = len(X)
    H = np.tanh(X @ W1 + b1)
    p = 1 / (1 + np.exp(-(H @ W2 + b2)))
    dz2 = (p - y) / N
    dW2 = H.T @ dz2
    db2 = dz2.sum(axis=0)
    dH = dz2 @ W2.T
    dZ1 = dH * (1 - H**2)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)
    return dW1, db1, dW2, db2
```
```test
import numpy as np
_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
_y = np.array([[0], [1], [1], [0]], dtype=float)
_r = np.random.default_rng(1)
_P = [_r.normal(size=(2, 4)), _r.normal(size=4), _r.normal(size=(4, 1)), _r.normal(size=1)]
def _tap(P):
    H = np.tanh(_X @ P[0] + P[1])
    p = 1 / (1 + np.exp(-(H @ P[2] + P[3])))
    return -np.mean(_y * np.log(p) + (1 - _y) * np.log(1 - p))
_g = backward(_X, _y, *_P)
_navn = ["dW1", "db1", "dW2", "db2"]
for i in range(4):
    assert np.shape(_g[i]) == np.shape(_P[i]), f"{_navn[i]} skal ha shape {np.shape(_P[i])}, fikk {np.shape(_g[i])}"
    num = np.zeros_like(_P[i])
    for idx in np.ndindex(_P[i].shape):
        pl = [q.copy() for q in _P]; pl[i][idx] += 1e-6
        mi = [q.copy() for q in _P]; mi[i][idx] -= 1e-6
        num[idx] = (_tap(pl) - _tap(mi)) / 2e-6
    assert np.allclose(_g[i], num, atol=1e-6), f"{_navn[i]} stemmer ikke med numerisk gradient. Sjekk kjerneregelen for {_navn[i]}."
```
:::

## Gradientsjekk
Sjekk backprop med sentraldifferansen `(L(w + h) - L(w - h)) / (2 * h)`: endre én vekt litt opp og litt ned i hver sin kopi (`W.copy()`), og se hvor mye tapet endrer seg.

::: oppgave Sjekk én vekt
Lag kopiene `W_pluss` og `W_minus` der `W1[0, 2]` er endret med `+h` og `-h`, og regn ut den numeriske gradienten `numerisk`. Den skal være nesten lik `dW1[0, 2]` fra backprop, og `W1` skal ikke endres.
```start
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

H = np.tanh(X @ W1 + b1)
p = 1 / (1 + np.exp(-(H @ W2 + b2)))
dW1 = X.T @ (((p - y) / len(X)) @ W2.T * (1 - H**2))   # backprop

h = 1e-5
W_pluss = ...
W_minus = ...
numerisk = ...
```
```losning
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

H = np.tanh(X @ W1 + b1)
p = 1 / (1 + np.exp(-(H @ W2 + b2)))
dW1 = X.T @ (((p - y) / len(X)) @ W2.T * (1 - H**2))   # backprop

h = 1e-5
W_pluss = W1.copy()
W_pluss[0, 2] += h
W_minus = W1.copy()
W_minus[0, 2] -= h
numerisk = (tap(W_pluss) - tap(W_minus)) / (2 * h)
```
```test
import numpy as np
_W1 = np.random.default_rng(1).normal(size=(2, 4))
assert np.allclose(W1, _W1), "W1 er endret. Lag kopier med W1.copy() før du endrer én vekt."
assert np.shape(W_pluss) == (2, 4) and np.isclose(W_pluss[0, 2] - W1[0, 2], 1e-5), "W_pluss skal være en kopi av W1 der bare [0, 2] er økt med h"
assert np.shape(W_minus) == (2, 4) and np.isclose(W1[0, 2] - W_minus[0, 2], 1e-5), "W_minus skal være en kopi av W1 der bare [0, 2] er senket med h"
assert np.ndim(numerisk) == 0, "numerisk skal være ett tall"
assert abs(numerisk - dW1[0, 2]) < 1e-7, f"numerisk ({numerisk:.6f}) skal være nesten lik dW1[0, 2] ({dW1[0, 2]:.6f}). Bruk (tap(W_pluss) - tap(W_minus)) / (2 * h)."
```
:::

--- LEKSJON 9.4 | sklearn-mlp | MLPClassifier og MLPRegressor
`MLPClassifier(hidden_layer_sizes=(16, 16), activation="tanh", learning_rate_init=0.01, random_state=0)` er et ferdig nett med `fit`/`predict`/`score`, der `hidden_layer_sizes` bare angir de skjulte lagene. `activation` kan være `"relu"` (standard), `"tanh"`, `"logistic"` eller `"identity"`, og `solver` er `"adam"` som standard.

::: oppgave Ditt eget MLP
Tren `mlp` med to skjulte lag på 16 noder og `random_state=0` på prosjekt 3-dataene, og lagre test-accuracy i `acc`. Velg aktivering og læringsrate selv, målet er minst 0.85.
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

::: oppgave Vekt-shapes i coefs_
Tren `mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=300, random_state=0)` på dataene med 5 features og 3 klasser. Lagre `shapes = [W.shape for W in mlp.coefs_]` og totalt antall parametre (vekter i `coefs_` pluss biaser i `intercepts_`) i `n_params`.
```start
import warnings
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")

X, y = make_classification(n_samples=200, n_features=5, n_informative=3, n_classes=3, random_state=0)

mlp = ...
shapes = ...
n_params = ...
```
```losning
import warnings
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")

X, y = make_classification(n_samples=200, n_features=5, n_informative=3, n_classes=3, random_state=0)

mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=300, random_state=0).fit(X, y)
shapes = [W.shape for W in mlp.coefs_]
n_params = sum(W.size for W in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
```
```test
from sklearn.neural_network import MLPClassifier
assert isinstance(mlp, MLPClassifier) and hasattr(mlp, "coefs_"), "mlp skal være en trent MLPClassifier"
assert [tuple(s) for s in shapes] == [(5, 8), (8, 3)], f"Nettet er 5 -> 8 -> 3, og coefs_ lagres som (inn, ut). Forventet [(5, 8), (8, 3)], fikk {shapes}"
assert n_params == 75, f"(5*8 + 8) + (8*3 + 3) = 75, fikk {n_params}. Husk biasene i intercepts_."
```
:::

## Tapskurve og early stopping
`mlp.loss_curve_` er treningstapet per epoke. Med `early_stopping=True` holdes 10 % av treningsdataene av til validering, treningen stopper etter `n_iter_no_change` epoker uten forbedring, og valideringsaccuracy ligger i `mlp.validation_scores_`.

::: oppgave Plott tapskurven
Tren `mlp` med `early_stopping=True` og `n_iter_no_change=20` (resten er ferdig), og plott `mlp.loss_curve_` i `ax[0]` og `mlp.validation_scores_` i `ax[1]`.
```start
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("classification_data.csv")
train = data[data["split"] == "train"]
X_train, y_train = train[["x0", "x1"]].values, train["y"].values

mlp = MLPClassifier(hidden_layer_sizes=(16, 16), activation="tanh", learning_rate_init=0.01,
                    max_iter=1000, random_state=0)
mlp.fit(X_train, y_train)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
# plott her
```
```losning
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("classification_data.csv")
train = data[data["split"] == "train"]
X_train, y_train = train[["x0", "x1"]].values, train["y"].values

mlp = MLPClassifier(hidden_layer_sizes=(16, 16), activation="tanh", learning_rate_init=0.01,
                    max_iter=1000, early_stopping=True, n_iter_no_change=20, random_state=0)
mlp.fit(X_train, y_train)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(mlp.loss_curve_)
ax[1].plot(mlp.validation_scores_)
```
```test
import numpy as np
assert mlp.early_stopping, "Sett early_stopping=True i MLPClassifier(...)"
assert mlp.n_iter_no_change == 20, "Sett n_iter_no_change=20 (patience)"
assert len(ax[0].lines) >= 1, "Plott mlp.loss_curve_ i ax[0]"
assert np.allclose(ax[0].lines[0].get_ydata(), mlp.loss_curve_), "ax[0] skal vise mlp.loss_curve_"
assert len(ax[1].lines) >= 1, "Plott mlp.validation_scores_ i ax[1]"
assert np.allclose(ax[1].lines[0].get_ydata(), mlp.validation_scores_), "ax[1] skal vise mlp.validation_scores_"
```
:::

## Skalering
Nevrale nett trenes med gradient descent og sliter når features har helt ulik skala, så legg `StandardScaler` foran med `make_pipeline(StandardScaler(), MLPClassifier(...))`.

::: oppgave MLP med skalering
Featuren `x1` er ganget med 1000. Lag `modell` som en pipeline med `StandardScaler` og `ny_mlp()`, tren den på treningsdataene og lagre test-accuracy i `acc`.
```start
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("classification_data.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
skala = np.array([1, 1000])
X_train, y_train = train[["x0", "x1"]].values * skala, train["y"].values
X_test, y_test = test[["x0", "x1"]].values * skala, test["y"].values

def ny_mlp():
    return MLPClassifier(hidden_layer_sizes=(16, 16), activation="tanh", learning_rate_init=0.01,
                         max_iter=1000, early_stopping=True, n_iter_no_change=20, random_state=0)

modell = ...
acc = ...
```
```losning
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("classification_data.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
skala = np.array([1, 1000])
X_train, y_train = train[["x0", "x1"]].values * skala, train["y"].values
X_test, y_test = test[["x0", "x1"]].values * skala, test["y"].values

def ny_mlp():
    return MLPClassifier(hidden_layer_sizes=(16, 16), activation="tanh", learning_rate_init=0.01,
                         max_iter=1000, early_stopping=True, n_iter_no_change=20, random_state=0)

modell = make_pipeline(StandardScaler(), ny_mlp())
modell.fit(X_train, y_train)
acc = modell.score(X_test, y_test)
```
```test
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
assert isinstance(modell, Pipeline), "modell skal være en pipeline: make_pipeline(StandardScaler(), ny_mlp())"
assert isinstance(modell.steps[0][1], StandardScaler), "Første steg i pipelinen skal være StandardScaler()"
assert hasattr(modell.steps[-1][1], "coefs_"), "Pipelinen er ikke trent. Kall modell.fit(X_train, y_train)."
assert abs(acc - modell.score(X_test, y_test)) < 1e-9, "acc skal være modell.score(X_test, y_test)"
assert acc >= 0.85, f"Test-accuracy er bare {acc:.3f}. Er skaleringen med?"
```
:::

## MLPRegressor
`MLPRegressor` har lineært output-lag og kvadratisk tap, men ellers samme API som `MLPClassifier`.

::: oppgave Regresjon på prosjekt 3-dataene
Tren `reg = MLPRegressor(hidden_layer_sizes=(64, 64), learning_rate_init=0.01, max_iter=2000, random_state=0)` og regn ut `rmse` mellom `reg.predict(X_test)` og `y_sann`, den sanne funksjonen uten støy.
```start
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

data = pd.read_csv("regression_data.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
X_train, y_train = train[["x0", "x1"]].values, train["y"].values
X_test = test[["x0", "x1"]].values
y_sann = X_test[:, 0]**3 - 3 * X_test[:, 0] * X_test[:, 1]**2

reg = ...
rmse = ...
```
```losning
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

data = pd.read_csv("regression_data.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
X_train, y_train = train[["x0", "x1"]].values, train["y"].values
X_test = test[["x0", "x1"]].values
y_sann = X_test[:, 0]**3 - 3 * X_test[:, 0] * X_test[:, 1]**2

reg = MLPRegressor(hidden_layer_sizes=(64, 64), learning_rate_init=0.01, max_iter=2000, random_state=0)
reg.fit(X_train, y_train)
rmse = np.sqrt(np.mean((reg.predict(X_test) - y_sann) ** 2))
```
```test
import numpy as np
from sklearn.neural_network import MLPRegressor
assert isinstance(reg, MLPRegressor) and hasattr(reg, "coefs_"), "reg skal være en trent MLPRegressor"
fasit = np.sqrt(np.mean((reg.predict(X_test) - y_sann) ** 2))
assert np.ndim(rmse) == 0, "rmse skal være ett tall"
assert abs(rmse - fasit) < 1e-9, f"rmse skal være kvadratroten av gjennomsnittet av (prediksjon - y_sann)², forventet {fasit:.3f}"
assert rmse < 1.0, f"RMSE er {rmse:.3f}. Har du brukt hyperparametrene fra oppgaven?"
```
:::

--- LEKSJON 9.5 | torch-tensorer | PyTorch: tensorer og autograd
PyTorch kjører ikke i nettleseren, så torch-koden vises som statiske blokker og ideene øves i NumPy. `torch.tensor(data, dtype=torch.float32)` lager en tensor som fungerer som en array (`.shape`, `@`, `.mean()`), `.item()` gir et Python-tall og `.numpy()` en array.

```python statisk
import numpy as np
import torch

a = np.array([[1.0, 2.0], [3.0, 4.0]])
x = torch.tensor(a)                          # beholder float64 fra NumPy
X = torch.tensor(a, dtype=torch.float32)     # som i prosjektene
y = torch.tensor([0, 1], dtype=torch.long)   # klasse-labels til CrossEntropyLoss
print(x.dtype, X.dtype, y.dtype)
print(X.shape, (X @ X.T).shape, X.mean().item())
```
```output
torch.float64 torch.float32 torch.int64
torch.Size([2, 2]) torch.Size([2, 2]) 2.5
```

::: oppgave Riktige datatyper
Nettet regner i float32 og `CrossEntropyLoss` vil ha heltalls-labels. Lag `X32` som `X` konvertert med `.astype(np.float32)` og `y_klasse` som `y` konvertert med `.astype(int)`.
```start
import numpy as np

X = np.array([[0.5, 1.5], [2.0, -1.0], [0.0, 3.0]])   # float64
y = np.array([0.0, 2.0, 1.0])                          # float

X32 = ...
y_klasse = ...
```
```losning
import numpy as np

X = np.array([[0.5, 1.5], [2.0, -1.0], [0.0, 3.0]])
y = np.array([0.0, 2.0, 1.0])

X32 = X.astype(np.float32)
y_klasse = y.astype(int)
```
```test
import numpy as np
assert isinstance(X32, np.ndarray) and X32.dtype == np.float32, "X32 skal ha dtype float32: X.astype(np.float32)"
assert np.allclose(X32, X), "Verdiene i X32 skal være de samme som i X"
assert isinstance(y_klasse, np.ndarray) and np.issubdtype(y_klasse.dtype, np.integer), "y_klasse skal være heltall: y.astype(int)"
assert list(y_klasse) == [0, 2, 1], f"y_klasse skal være [0, 2, 1], fikk {y_klasse}"
```
:::

## Shapes: squeeze og unsqueeze
`out.squeeze(1)` gjør `(N, 1)` om til `(N,)` og `y.unsqueeze(1)` gjør motsatt, og i NumPy heter det `np.squeeze(out, axis=1)` og `y[:, None]` (eller `reshape(-1, 1)`). Blander du `(N, 1)` og `(N,)`, gir broadcasting en `(N, N)`-matrise og feil tap uten feilmelding.

::: oppgave Shapes som i PyTorch, bare i NumPy
Lag `out_flat` (`out` med shape `(5,)`), `y_kol` (`y` med shape `(5, 1)`) og `mse`, riktig MSE mellom `out` og `y` regnet ut med like shapes.
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

## Autograd
Med `requires_grad=True` husker PyTorch regnestykket, og `loss.backward()` legger den deriverte i `.grad`.

```python statisk
import torch

x = torch.tensor(2.0, requires_grad=True)
f = x**2 + 3 * x          # f'(x) = 2x + 3
f.backward()
print(x.grad)
```
```output
tensor(7.)
```

::: oppgave Numerisk derivert
Skriv `num_grad(f, x, h=1e-5)` som tilnærmer den deriverte med sentraldifferansen `(f(x + h) - f(x - h)) / (2 * h)`, så du kan sjekke autograd selv.
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

::: oppgave Det backward() ville gitt
Regn ut MSE-tapet `loss` og gradientene `dw` og `db` for modellen `X * w + b`, altså det `w.grad` og `b.grad` blir etter `loss.backward()`. Formlene er `dw = mean(2 * (pred - y) * X)` og `db = mean(2 * (pred - y))`.
```start
import numpy as np

X = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
w, b = 0.5, 0.0

loss = ...
dw = ...
db = ...
```
```losning
import numpy as np

X = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
w, b = 0.5, 0.0

pred = X * w + b
loss = np.mean((pred - y) ** 2)
dw = np.mean(2 * (pred - y) * X)
db = np.mean(2 * (pred - y))
```
```test
import numpy as np
assert np.ndim(loss) == 0 and abs(loss - 10.5) < 1e-9, f"loss skal være 10.5 (MSE), fikk {loss}"
assert np.ndim(dw) == 0 and abs(dw - (-14.0)) < 1e-9, f"dw skal være -14 (som w.grad i PyTorch), fikk {dw}. Husk faktoren 2 og X."
assert np.ndim(db) == 0 and abs(db - (-6.0)) < 1e-9, f"db skal være -6 (som b.grad), fikk {db}"
```
:::

## no_grad og detach
`with torch.no_grad():` slår av gradientsporingen ved prediksjon, og `.detach().numpy()` trengs for å få ut en array fra en tensor som krever gradient. Går du via NumPy midt i en beregning du skal derivere, mister PyTorch sporet og vektene får ingen gradient.

```python statisk
import torch

w = torch.tensor([1.0, 2.0], requires_grad=True)
with torch.no_grad():
    z = w * 3
print(z.requires_grad)
print((w * 3).detach().numpy())
```
```output
False
[3. 6.]
```

--- LEKSJON 9.6 | torch-moduler | nn.Linear, nn.Sequential og nn.Module
`nn.Linear(inn, ut)` lagrer `weight` med shape `(ut, inn)` og regner `X @ weight.T + bias`, mens Keras og `coefs_` bruker `(inn, ut)`. Aktiveringer er egne lag: `nn.Tanh()`, `nn.ReLU()`, `nn.GELU()` og `nn.SiLU()`.

```python statisk
import torch
import torch.nn as nn

lag = nn.Linear(2, 16)
print(lag.weight.shape, lag.bias.shape)
print(lag(torch.randn(5, 2)).shape)
```
```output
torch.Size([16, 2]) torch.Size([16])
torch.Size([5, 16])
```

::: oppgave Lag din egen Linear-klasse
Skriv klassen `Linear` som etterligner `nn.Linear`: `__init__(self, n_in, n_out)` lager `self.weight` med shape `(n_out, n_in)` (`np.random.randn`) og `self.bias` som nuller, `__call__(self, X)` returnerer `X @ self.weight.T + self.bias`, og `n_params(self)` gir antall parametre.
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

## nn.Sequential og make_mlp
`nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))` setter lag etter hverandre, og `make_mlp` i prosjekt 3 bygger listen med en løkke og pakker den ut med `*layers`. `sum(p.numel() for p in net.parameters())` teller parametrene.

```python statisk
import torch.nn as nn

def make_mlp(n_in, width, depth, act):
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), act()]
        d = width
    return nn.Sequential(*layers, nn.Linear(d, 1))

net = make_mlp(2, 16, 2, nn.Tanh)
print(net)
print(sum(p.numel() for p in net.parameters()))
```
```output
Sequential(
  (0): Linear(in_features=2, out_features=16, bias=True)
  (1): Tanh()
  (2): Linear(in_features=16, out_features=16, bias=True)
  (3): Tanh()
  (4): Linear(in_features=16, out_features=1, bias=True)
)
337
```

::: oppgave weight-shapes i make_mlp
Skriv `weight_shapes(n_in, width, depth)` som returnerer listen med `weight`-shapes `(ut, inn)` for alle `nn.Linear`-lagene som `make_mlp(n_in, width, depth, act)` lager.
```start
def weight_shapes(n_in, width, depth):
    shapes, d = [], n_in
    ...
    return shapes
```
```losning
def weight_shapes(n_in, width, depth):
    shapes, d = [], n_in
    for _ in range(depth):
        shapes.append((width, d))
        d = width
    shapes.append((1, d))
    return shapes
```
```test
r = weight_shapes(2, 16, 2)
assert r is not None, "Funksjonen returnerer ingenting"
r = [tuple(s) for s in r]
assert len(r) == 3, f"make_mlp(2, 16, 2, ...) har 3 Linear-lag, du returnerte {len(r)} shapes"
assert r == [(16, 2), (16, 16), (1, 16)], f"Forventet [(16, 2), (16, 16), (1, 16)] (ut, inn), fikk {r}"
assert [tuple(s) for s in weight_shapes(2, 128, 3)] == [(128, 2), (128, 128), (128, 128), (1, 128)], "weight_shapes(2, 128, 3) er feil"
```
:::

## Egen klasse med nn.Module
En egen modell arver fra `nn.Module`, kaller `super().__init__()` først, lager lagene i `__init__` og sier i `forward` hvordan data flyter. Du kaller nettet som `net(x)`, ikke `net.forward(x)`.

```python statisk
import torch

class SimpleNN(torch.nn.Module):          # nettet fra prosjekt 4
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):                 # returnerer logits
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def forward_digit(self, x):           # returnerer klassen
        return torch.argmax(torch.softmax(self(x), dim=1), dim=1)
```

::: oppgave Et MLP som klasse
Skriv klassen `MLP` med `__init__(self, n_in, n_hidden, n_out)` (lag `self.W1`, `self.b1`, `self.W2`, `self.b2` med `(inn, ut)`-shapes, `np.random.randn` for vekter og nuller for bias), `forward(self, X)` med ReLU i skjult lag og logits ut, `__call__(self, X)` som returnerer `self.forward(X)`, og `predict(self, X)` som gir klassen med `argmax`.
```start
import numpy as np

class MLP:
    def __init__(self, n_in, n_hidden, n_out):
        ...

    def forward(self, X):
        ...

    def __call__(self, X):
        ...

    def predict(self, X):
        ...
```
```losning
import numpy as np

class MLP:
    def __init__(self, n_in, n_hidden, n_out):
        self.W1 = np.random.randn(n_in, n_hidden)
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(n_hidden, n_out)
        self.b2 = np.zeros(n_out)

    def forward(self, X):
        H = np.maximum(0, X @ self.W1 + self.b1)
        return H @ self.W2 + self.b2

    def __call__(self, X):
        return self.forward(X)

    def predict(self, X):
        return np.argmax(self(X), axis=1)
```
```test
import numpy as np
net = MLP(4, 8, 3)
for navn, s in [("W1", (4, 8)), ("b1", (8,)), ("W2", (8, 3)), ("b2", (3,))]:
    assert np.shape(getattr(net, navn, None)) == s, f"self.{navn} skal ha shape {s}, har {np.shape(getattr(net, navn, None))}"
X = np.random.default_rng(0).normal(size=(5, 4))
fasit = np.maximum(0, X @ net.W1 + net.b1) @ net.W2 + net.b2
ut = net.forward(X)
assert ut is not None and np.shape(ut) == (5, 3), "forward(X) skal gi logits med shape (5, 3)"
assert np.allclose(ut, fasit), "forward er feil: ReLU i skjult lag, ingen aktivering til slutt"
assert net(X) is not None and np.allclose(net(X), fasit), "net(X) skal kalle forward. Skriv __call__."
p = net.predict(X)
assert p is not None and np.array_equal(p, fasit.argmax(axis=1)), "predict skal gi np.argmax(logits, axis=1)"
```
:::

## state_dict
`net.state_dict()` er en ordbok fra navn som `0.weight` eller `fc1.bias` til tensorer, `net.load_state_dict(sd)` setter vektene, og `torch.save(net.state_dict(), "modell.pth")` lagrer dem til fil.

::: oppgave Les en state_dict
`sd` er en state_dict for `nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))` med NumPy-arrays. Lag `n_params` (summen av `.size` for alle verdiene) og `vekt_navn`, listen med nøklene som slutter på `"weight"`.
```start
import numpy as np

sd = {"0.weight": np.zeros((16, 2)), "0.bias": np.zeros(16),
      "2.weight": np.zeros((1, 16)), "2.bias": np.zeros(1)}

n_params = ...
vekt_navn = ...
```
```losning
import numpy as np

sd = {"0.weight": np.zeros((16, 2)), "0.bias": np.zeros(16),
      "2.weight": np.zeros((1, 16)), "2.bias": np.zeros(1)}

n_params = sum(v.size for v in sd.values())
vekt_navn = [k for k in sd if k.endswith("weight")]
```
```test
assert n_params == 65, f"2*16 + 16 + 16*1 + 1 = 65 parametre, fikk {n_params}"
assert isinstance(vekt_navn, list) and sorted(vekt_navn) == ["0.weight", "2.weight"], f"vekt_navn skal være ['0.weight', '2.weight'], fikk {vekt_navn}. Bruk k.endswith('weight')."
```
:::

--- LEKSJON 9.7 | tapsfunksjoner | Tapsfunksjoner og logits
Regresjon bruker `nn.MSELoss()`, binær klassifisering `nn.BCEWithLogitsLoss()` på logits med float-targets 0.0/1.0, og flere klasser `nn.CrossEntropyLoss()` på logits `(N, C)` med `long`-targets `(N,)` (klasseindekser, ikke one-hot). Alle kalles som `loss_fn(prediksjon, target)`.

```python statisk
import torch
import torch.nn as nn

logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 0.2, 3.0], [1.0, 1.0, 1.0]])
y = torch.tensor([0, 2, 1])                          # long, ikke one-hot
print(nn.CrossEntropyLoss()(logits, y))
print(nn.BCEWithLogitsLoss()(torch.tensor([2.0, -1.0]), torch.tensor([1.0, 0.0])))
print(nn.MSELoss()(torch.tensor([2.5, 0.0]), torch.tensor([3.0, -0.5])))
```
```output
tensor(0.4832)
tensor(0.2201)
tensor(0.2500)
```

## BCEWithLogitsLoss
`BCEWithLogitsLoss` har sigmoid innebygd, og `logit > 0` gir samme beslutning som `sigmoid(logit) > 0.5`. Har nettet `nn.Sigmoid()` til slutt, skal du bruke `nn.BCELoss()`, aldri sigmoid to ganger.

::: oppgave Binær cross-entropy fra logits
Skriv `bce_with_logits(z, y)` som gjør som `nn.BCEWithLogitsLoss()`: sigmoid på logitene `z`, og så gjennomsnittet av `-(y * log(p) + (1 - y) * log(1 - p))`.
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
assert abs(bce_with_logits(np.array([2.0, -1.0]), np.array([1.0, 0.0])) - 0.2201) < 1e-3, "bce_with_logits([2, -1], [1, 0]) skal gi 0.2201 som i PyTorch-eksempelet"
```
:::

::: oppgave Sigmoid to ganger
Nettet har allerede sigmoid til slutt og gir de nesten perfekte sannsynlighetene `p`. Lagre `riktig = bce(p, y)` (som `BCELoss`) og `dobbel`, tapet du får når `p` behandles som logits og får sigmoid én gang til (som `BCEWithLogitsLoss`).
```start
import numpy as np

def bce(p, y):
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

p = np.array([0.01, 0.99])
y = np.array([0.0, 1.0])

riktig = ...
dobbel = ...
```
```losning
import numpy as np

def bce(p, y):
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

p = np.array([0.01, 0.99])
y = np.array([0.0, 1.0])

riktig = bce(p, y)
dobbel = bce(1 / (1 + np.exp(-p)), y)
```
```test
import numpy as np
assert np.ndim(riktig) == 0 and abs(riktig - 0.01005) < 1e-4, f"riktig skal være bce(p, y) ≈ 0.0101, fikk {riktig}"
assert np.ndim(dobbel) == 0 and abs(dobbel - 0.5071) < 1e-3, f"dobbel skal være bce(sigmoid(p), y) ≈ 0.5071, fikk {dobbel}. sigmoid(p) ligger mellom 0.5 og 0.73, så tapet kan aldri bli lite."
```
:::

## CrossEntropyLoss
`CrossEntropyLoss` gjør softmax inni og tar `-log` av sannsynligheten for riktig klasse, som du plukker ut med `P[np.arange(N), y]`. Nettet skal derfor gi rå logits, ikke softmax først.

::: oppgave Cross-entropy fra logits
Skriv `cross_entropy(logits, y)` som gjør som `nn.CrossEntropyLoss()`: softmax per rad (trekk fra maks), plukk ut sannsynligheten for riktig klasse i hver rad, og returner gjennomsnittet av `-log`.
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

::: oppgave Softmax før CrossEntropyLoss
`ce` gjør som `nn.CrossEntropyLoss()`. Lagre `ce_riktig = ce(logits, y)`, og `ce_feil`, tapet du får hvis nettet har softmax til slutt så `ce` får sannsynligheter i stedet for logits.
```start
import numpy as np
from scipy.special import log_softmax, softmax

def ce(logits, y):
    return -np.mean(log_softmax(logits, axis=1)[np.arange(len(y)), y])

logits = np.array([[4.0, 0.0, -2.0], [-1.0, 0.0, 5.0]])
y = np.array([0, 2])

ce_riktig = ...
ce_feil = ...
```
```losning
import numpy as np
from scipy.special import log_softmax, softmax

def ce(logits, y):
    return -np.mean(log_softmax(logits, axis=1)[np.arange(len(y)), y])

logits = np.array([[4.0, 0.0, -2.0], [-1.0, 0.0, 5.0]])
y = np.array([0, 2])

ce_riktig = ce(logits, y)
ce_feil = ce(softmax(logits, axis=1), y)
```
```test
import numpy as np
from scipy.special import log_softmax, softmax
_r = -np.mean(log_softmax(logits, axis=1)[[0, 1], [0, 2]])
_P = softmax(logits, axis=1)
_f = -np.mean(log_softmax(_P, axis=1)[[0, 1], [0, 2]])
assert np.ndim(ce_riktig) == 0 and abs(ce_riktig - _r) < 1e-9, f"ce_riktig skal være ce(logits, y) ≈ {_r:.4f}, fikk {ce_riktig}"
assert np.ndim(ce_feil) == 0 and abs(ce_feil - _f) < 1e-9, f"ce_feil skal være ce(softmax(logits, axis=1), y) ≈ {_f:.4f}, fikk {ce_feil}. Husk axis=1."
assert ce_feil > 10 * ce_riktig, "Softmax to ganger skal gi et mye større tap"
```
:::

--- LEKSJON 9.8 | treningslokka | Treningsløkka
Hver oppdatering er fire linjer: `optimizer.zero_grad()`, `loss = loss_fn(net(X), y)`, `loss.backward()` og `optimizer.step()`, der `optimizer = torch.optim.SGD(net.parameters(), lr=0.05)` vet hvilke parametre den skal oppdatere. Lagre `loss.item()` i en liste for å plotte tapskurven.

```python statisk
import torch
import torch.nn as nn

torch.manual_seed(0)
X = torch.rand(100, 1) * 4
y = 2 * X + 1 + 0.1 * torch.randn(100, 1)
net = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
losses = []
for epoch in range(300):
    optimizer.zero_grad()
    loss = loss_fn(net(X), y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())       # .item(): lagre et tall, ikke en tensor
print(net.weight.item(), net.bias.item())
```
```output
1.982999563217163 1.0328478813171387
```

::: oppgave Treningsløkka i NumPy
Tren lineær regresjon i 500 epoker med de fire stegene: forward, tap (legg det i `losses`), gradienter (`dw = mean(2 * (pred - y) * x)`, `db = mean(2 * (pred - y))`) og oppdatering med `lr`.
```start
import numpy as np

rng = np.random.default_rng(1)
x = rng.uniform(0, 4, size=100)
y = 2 * x + 1 + 0.1 * rng.normal(size=100)

w, b, lr = 0.0, 0.0, 0.05
losses = []
for epoch in range(500):
    # 1) forward  2) tap  3) gradienter  4) oppdater w og b
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
assert abs(float(np.ravel(w)[0]) - 2) < 0.1 and abs(float(np.ravel(b)[0]) - 1) < 0.2, f"w og b skal nærme seg 2 og 1, fikk w={w}, b={b}"
```
:::

## Glemt zero_grad
`loss.backward()` legger til i `.grad` i stedet for å overskrive, så uten `optimizer.zero_grad()` blir hvert steg summen av alle tidligere gradienter og treningen blir ustabil.

::: oppgave Finn feilen
Løkka under oppfører seg som PyTorch: gradientene `gw` og `gb` akkumuleres med `+=` (som `backward()`), men de nullstilles aldri. Legg til «zero_grad» på rett sted, så `w` og `b` nærmer seg 2 og 1.
```start
import numpy as np

rng = np.random.default_rng(1)
x = rng.uniform(0, 4, size=100)
y = 2 * x + 1 + 0.1 * rng.normal(size=100)

w, b, lr = 0.0, 0.0, 0.05
gw, gb = 0.0, 0.0
losses = []
for epoch in range(200):
    pred = w * x + b
    losses.append(np.mean((pred - y) ** 2))
    gw += 2 * np.mean((pred - y) * x)     # «backward»
    gb += 2 * np.mean(pred - y)
    w -= lr * gw                          # «step»
    b -= lr * gb
```
```losning
import numpy as np

rng = np.random.default_rng(1)
x = rng.uniform(0, 4, size=100)
y = 2 * x + 1 + 0.1 * rng.normal(size=100)

w, b, lr = 0.0, 0.0, 0.05
gw, gb = 0.0, 0.0
losses = []
for epoch in range(200):
    gw, gb = 0.0, 0.0                     # «zero_grad»
    pred = w * x + b
    losses.append(np.mean((pred - y) ** 2))
    gw += 2 * np.mean((pred - y) * x)
    gb += 2 * np.mean(pred - y)
    w -= lr * gw
    b -= lr * gb
```
```test
import numpy as np
assert len(losses) == 200, "Behold 200 epoker"
assert abs(w - 2) < 0.1 and abs(b - 1) < 0.2, f"w og b skal nærme seg 2 og 1, fikk w={w:.3f}, b={b:.3f}. Sett gw, gb = 0.0, 0.0 i starten av hver epoke."
assert losses[-1] < 0.05, f"Siste tap er {losses[-1]:.3f}. Nullstiller du gradientene i hver epoke?"
```
:::

::: oppgave Rett rekkefølge
Sorter strengene i `steg` i en rekkefølge som fungerer for én oppdatering.
```start
steg = ["optimizer.step()", "loss.backward()", "optimizer.zero_grad()", "loss = loss_fn(net(X), y)"]
```
```losning
steg = ["optimizer.zero_grad()", "loss = loss_fn(net(X), y)", "loss.backward()", "optimizer.step()"]
```
```test
_alle = {"optimizer.step()", "loss.backward()", "optimizer.zero_grad()", "loss = loss_fn(net(X), y)"}
assert isinstance(steg, list) and len(steg) == 4 and set(steg) == _alle, "steg skal inneholde de fire strengene nøyaktig én gang"
i = {s: steg.index(s) for s in steg}
assert i["loss = loss_fn(net(X), y)"] < i["loss.backward()"], "Tapet må regnes ut før det kan deriveres med backward()"
assert i["loss.backward()"] < i["optimizer.step()"], "Gradientene må finnes (backward) før step() kan bruke dem"
assert i["optimizer.zero_grad()"] < i["loss.backward()"], "zero_grad() må komme før backward(), ellers nullstilles gradientene du nettopp regnet ut"
```
:::

## Et nett på prosjekt 3-dataene
Med full batch er én epoke én oppdatering, og ved prediksjon gjør du logits om til klasser med `net(X_test).squeeze(1) > 0` inne i `with torch.no_grad():`.

::: oppgave Tren 2 → 16 → 1 med full batch
`forward_backward` gir tapet og gradientene. Tren i 1000 epoker med `lr = 0.5` (legg tapet i `losses` og oppdater alle fire parametrene), og lagre test-accuracy i `acc` med regelen `logit > 0`.
```start
import numpy as np
import pandas as pd

data = pd.read_csv("classification_data.csv")
train, test = data[data["split"] == "train"], data[data["split"] == "test"]
X, y = train[["x0", "x1"]].values, train["y"].values.astype(float)
X_test, y_test = test[["x0", "x1"]].values, test["y"].values

def forward_backward(W1, b1, W2, b2):
    H = np.tanh(X @ W1 + b1)
    p = 1 / (1 + np.exp(-(H @ W2 + b2)[:, 0]))
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    dz = ((p - y) / len(X))[:, None]
    dZ1 = (dz @ W2.T) * (1 - H**2)
    return loss, X.T @ dZ1, dZ1.sum(axis=0), H.T @ dz, dz.sum(axis=0)

rng = np.random.default_rng(0)
W1, b1 = rng.normal(size=(2, 16)), np.zeros(16)
W2, b2 = 0.5 * rng.normal(size=(16, 1)), np.zeros(1)
lr = 0.5
losses = []
# treningsløkka her

acc = ...
```
```losning
import numpy as np
import pandas as pd

data = pd.read_csv("classification_data.csv")
train, test = data[data["split"] == "train"], data[data["split"] == "test"]
X, y = train[["x0", "x1"]].values, train["y"].values.astype(float)
X_test, y_test = test[["x0", "x1"]].values, test["y"].values

def forward_backward(W1, b1, W2, b2):
    H = np.tanh(X @ W1 + b1)
    p = 1 / (1 + np.exp(-(H @ W2 + b2)[:, 0]))
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    dz = ((p - y) / len(X))[:, None]
    dZ1 = (dz @ W2.T) * (1 - H**2)
    return loss, X.T @ dZ1, dZ1.sum(axis=0), H.T @ dz, dz.sum(axis=0)

rng = np.random.default_rng(0)
W1, b1 = rng.normal(size=(2, 16)), np.zeros(16)
W2, b2 = 0.5 * rng.normal(size=(16, 1)), np.zeros(1)
lr = 0.5
losses = []
for epoch in range(1000):
    loss, dW1, db1, dW2, db2 = forward_backward(W1, b1, W2, b2)
    losses.append(loss)
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

logit = (np.tanh(X_test @ W1 + b1) @ W2 + b2)[:, 0]
acc = np.mean((logit > 0) == y_test)
```
```test
import numpy as np
assert len(losses) == 1000, f"losses skal ha én verdi per epoke (1000), har {len(losses)}"
assert losses[-1] < losses[0] - 0.2, "Tapet synker ikke. Oppdater alle fire parametrene med -= lr * gradient."
_logit = (np.tanh(X_test @ W1 + b1) @ W2 + b2)[:, 0]
_acc = np.mean((_logit > 0) == y_test)
assert np.ndim(acc) == 0 and abs(acc - _acc) < 1e-9, f"acc skal være andelen der (logit > 0) == y_test på TESTdataene, forventet {_acc:.3f}"
assert acc >= 0.85, f"Test-accuracy er {acc:.3f}, forventet minst 0.85"
```
:::

--- LEKSJON 9.9 | minibatch-dataloader | Mini-batcher, DataLoader og prediksjon
Med mini-batcher stokkes indeksene hver epoke (`perm = torch.randperm(n)`), og hver bit `j = perm[b:b + batch_size]` gir én oppdatering med `X[j]` og `y[j]`. Én epoke har da ⌈N / B⌉ oppdateringer, mens full batch har 1 og ren SGD (batch size 1) har N.

::: oppgave Lag mini-batcher
Skriv `minibatcher(N, batch_size, rng)` som stokker indeksene med `rng.permutation(N)` og returnerer en liste med index-arrays, én per batch (den siste kan være mindre).
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
    for s in range(0, N, batch_size):
        batcher.append(perm[s:s + batch_size])
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

::: oppgave Oppdateringer per epoke
Skriv `oppdateringer(N, batch_size)` som returnerer antall `optimizer.step()` per epoke, altså N / B rundet opp.
```start
def oppdateringer(N, batch_size):
    ...
```
```losning
import math

def oppdateringer(N, batch_size):
    return math.ceil(N / batch_size)
```
```test
assert oppdateringer(10000, 500) == 20, f"10 000 punkter og batch size 500 gir 20 oppdateringer, fikk {oppdateringer(10000, 500)}"
assert oppdateringer(1000, 64) == 16, f"1000 / 64 = 15.6, rundet opp blir det 16, fikk {oppdateringer(1000, 64)}"
assert oppdateringer(400, 400) == 1, "Full batch gir 1 oppdatering per epoke"
assert oppdateringer(400, 1) == 400, "Batch size 1 (ren SGD) gir N oppdateringer"
```
:::

::: oppgave Trening med mini-batcher
Tren lineær regresjon i 50 epoker med `batch_size = 16`: stokk med `rng.permutation` hver epoke, oppdater `w` og `b` på hver batch `x[j]`, `y[j]`, og tell oppdateringene i `n_steg`.
```start
import numpy as np

rng = np.random.default_rng(0)
x = rng.uniform(0, 4, size=100)
y = 2 * x + 1 + 0.1 * rng.normal(size=100)

w, b, lr, batch_size = 0.0, 0.0, 0.02, 16
n_steg = 0
for epoch in range(50):
    perm = ...
    # for s in range(0, len(x), batch_size): ...
```
```losning
import numpy as np

rng = np.random.default_rng(0)
x = rng.uniform(0, 4, size=100)
y = 2 * x + 1 + 0.1 * rng.normal(size=100)

w, b, lr, batch_size = 0.0, 0.0, 0.02, 16
n_steg = 0
for epoch in range(50):
    perm = rng.permutation(len(x))
    for s in range(0, len(x), batch_size):
        j = perm[s:s + batch_size]
        pred = w * x[j] + b
        w -= lr * 2 * np.mean((pred - y[j]) * x[j])
        b -= lr * 2 * np.mean(pred - y[j])
        n_steg += 1
```
```test
import numpy as np
assert n_steg == 350, f"100 punkter med batch_size 16 gir 7 batcher per epoke, altså 350 oppdateringer på 50 epoker. Fikk {n_steg}"
assert abs(w - 2) < 0.1 and abs(b - 1) < 0.25, f"w og b skal nærme seg 2 og 1, fikk w={w:.3f}, b={b:.3f}. Bruk x[j] og y[j] i gradientene."
```
:::

## DataLoader
`DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)` gjør det samme, og `len(loader)` er antall batcher per epoke.

```python statisk
import torch
from torch.utils.data import DataLoader, TensorDataset

X, y = torch.randn(1000, 784), torch.randint(0, 10, (1000,))
loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
print(len(loader))
for X_batch, y_batch in loader:
    pass                             # zero_grad, forward, backward, step
print(X_batch.shape, y_batch.shape)
```
```output
16
torch.Size([40, 784]) torch.Size([40])
```

## train() og eval()
`net.eval()` skrur av Dropout (som nuller tilfeldige aktiveringer under trening og skalerer resten med `1/(1-p)`), men slår ikke av gradientene, så ved prediksjon bruker du både `net.eval()` og `torch.no_grad()`, og `net.train()` før neste treningsepoke.

```python statisk
import torch
import torch.nn as nn

net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(16, 3))
X_test, y_test = torch.randn(8, 4), torch.randint(0, 3, (8,))
net.eval()                           # Dropout av
with torch.no_grad():                # gradientsporing av
    logits = net(X_test)
acc = (logits.argmax(dim=1) == y_test).float().mean().item()
net.train()                          # tilbake til treningsmodus
```

::: oppgave Dropout
Skriv `dropout(H, p, trening, rng)`: i trening settes hver verdi til 0 med sannsynlighet `p` (bruk `rng.random(H.shape) >= p` som maske) og resten ganges med `1 / (1 - p)`, og i eval-modus returneres `H` uendret.
```start
import numpy as np

def dropout(H, p, trening, rng):
    ...
```
```losning
import numpy as np

def dropout(H, p, trening, rng):
    if not trening:
        return H
    maske = rng.random(H.shape) >= p
    return H * maske / (1 - p)
```
```test
import numpy as np
H = np.ones((200, 50))
ev = dropout(H, 0.5, False, np.random.default_rng(0))
assert ev is not None and np.array_equal(ev, H), "I eval-modus (trening=False) skal H returneres uendret"
tr = dropout(H, 0.5, True, np.random.default_rng(0))
assert tr is not None and np.shape(tr) == H.shape, "I trening skal resultatet ha samme shape som H"
andel = np.mean(tr == 0)
assert 0.45 < andel < 0.55, f"Med p = 0.5 skal ca. halvparten bli 0, fikk andel {andel:.2f}"
assert np.allclose(tr[tr != 0], 2.0), "Verdiene som beholdes skal ganges med 1 / (1 - p) = 2"
assert abs(np.mean(dropout(H, 0.2, True, np.random.default_rng(1))) - 1) < 0.05, "Gjennomsnittet skal bli omtrent det samme som uten dropout"
```
:::

## Prediksjon og accuracy
Binært blir det `logits > 0`, og med flere klasser `logits.argmax(dim=1)` (softmax endrer ikke hvilken klasse som er størst). Accuracy er `(pred == y).float().mean().item()`.

::: oppgave Fra logits til accuracy
Lag `pred_bin` (0/1 fra `logits_bin` med logit-regelen), `pred_multi` (klassen fra `logits_multi`) og `acc_multi` (accuracy for `pred_multi` mot `y_multi`).
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

--- LEKSJON 9.10 | validering-early-stopping | Validering, early stopping, optimizere og scheduler
Hold av en valideringsdel av treningsdataene med `val_idx, train_idx = perm[:100], perm[100:]`, og regn ut tapet på den etter hver epoke. Synker treningstapet mens valideringstapet stiger, overtilpasser nettet, og testdataene rører du ikke før til slutt.

::: oppgave Held-out validering
Stokk indeksene til de 500 treningspunktene med `rng.permutation`, la de 100 første være `val_idx` og resten `train_idx`, og lag `X_tr` og `X_val`.
```start
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 2))

perm = ...
val_idx, train_idx = ..., ...
X_tr, X_val = ..., ...
```
```losning
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 2))

perm = rng.permutation(len(X))
val_idx, train_idx = perm[:100], perm[100:]
X_tr, X_val = X[train_idx], X[val_idx]
```
```test
import numpy as np
assert len(val_idx) == 100 and len(train_idx) == 400, f"val_idx skal ha 100 indekser og train_idx 400, fikk {len(val_idx)} og {len(train_idx)}"
assert sorted(np.concatenate([val_idx, train_idx]).tolist()) == list(range(500)), "Til sammen skal indeksene dekke 0-499 nøyaktig én gang"
assert list(val_idx) != list(range(100)), "Indeksene skal stokkes med rng.permutation før du deler"
assert np.shape(X_tr) == (400, 2) and np.allclose(X_tr, X[train_idx]), "X_tr skal være X[train_idx]"
assert np.shape(X_val) == (100, 2) and np.allclose(X_val, X[val_idx]), "X_val skal være X[val_idx]"
```
:::

## Early stopping
Når valideringstapet er det beste så langt, lagrer du en kopi av vektene med `{k: v.clone() for k, v in net.state_dict().items()}`, og etter `patience` epoker uten forbedring stopper du og kaller `net.load_state_dict(best_state)`. Uten `.clone()` peker kopien på de levende vektene og blir vektene fra siste epoke.

::: oppgave Early stopping-logikken
Skriv `early_stopping(val_losses, patience)` som går gjennom tapene: er tapet lavere enn det beste, oppdateres beste tap og `best_epoch`, ellers stopper vi hvis `epoch - best_epoch > patience`. Returner `(best_epoch, stop_epoch)`, der `stop_epoch` er epoken vi stoppet i (eller siste epoke).
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

::: oppgave Kopi av vektene
Lagre en ekte kopi av `params` i `beste` (som `.clone()` i PyTorch, her `.copy()`), slik at den ikke endres når «treningen» etterpå endrer `W` og `b` på stedet.
```start
import numpy as np

W = np.array([[1.0, 2.0]])
b = np.array([0.5])
params = {"weight": W, "bias": b}

beste = ...

W += 100.0    # «optimizer.step()» endrer vektene på stedet
b -= 1.0
```
```losning
import numpy as np

W = np.array([[1.0, 2.0]])
b = np.array([0.5])
params = {"weight": W, "bias": b}

beste = {k: v.copy() for k, v in params.items()}

W += 100.0
b -= 1.0
```
```test
import numpy as np
assert isinstance(beste, dict) and set(beste) == {"weight", "bias"}, "beste skal være en ordbok med de samme nøklene som params"
assert np.allclose(beste["weight"], [[1.0, 2.0]]) and np.allclose(beste["bias"], [0.5]), f"beste har endret seg med treningen: {beste}. dict(params) kopierer bare ordboka, ikke arrayene. Bruk {{k: v.copy() for k, v in params.items()}}."
```
:::

## Optimizere og scheduler
Optimizerne brukes likt: `torch.optim.SGD(p, lr, momentum=0.9)`, `RMSprop(p, lr)`, `Adam(p, lr)` og `AdamW(p, lr, weight_decay=0.1)`, der weight decay drar vektene mot 0. `CosineAnnealingLR(optimizer, T_max=epochs)` senker læringsraten langs en cosinuskurve, med `scheduler.step()` én gang per epoke etter batch-løkka.

```python statisk
import torch
import torch.nn as nn

net = nn.Linear(2, 1)
optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
for epoch in range(300):
    # for j in batcher: zero_grad, forward, backward, optimizer.step()
    optimizer.step()
    scheduler.step()                 # én gang per epoke, etter batch-løkka
```

::: oppgave SGD med momentum
Skriv `sgd_momentum(w, v, g, lr, mu=0.9)` som gjør ett steg slik PyTorch gjør det: først `v = mu * v + g`, så `w = w - lr * v`. Returner `(w, v)`.
```start
def sgd_momentum(w, v, g, lr, mu=0.9):
    ...
```
```losning
def sgd_momentum(w, v, g, lr, mu=0.9):
    v = mu * v + g
    w = w - lr * v
    return w, v
```
```test
r = sgd_momentum(1.0, 0.0, 2.0, 0.1)
assert r is not None and len(r) == 2, "Returner (w, v)"
w1, v1 = r
assert abs(v1 - 2.0) < 1e-12 and abs(w1 - 0.8) < 1e-12, f"Første steg: v = 0.9*0 + 2 = 2 og w = 1 - 0.1*2 = 0.8. Fikk w={w1}, v={v1}"
w2, v2 = sgd_momentum(w1, v1, 2.0, 0.1)
assert abs(v2 - 3.8) < 1e-12 and abs(w2 - 0.42) < 1e-12, f"Andre steg: v = 0.9*2 + 2 = 3.8 og w = 0.8 - 0.38 = 0.42. Fikk w={w2}, v={v2}"
```
:::

::: oppgave Cosine annealing
Skriv `cosine_lr(lr0, t, T_max, lr_min=0.0)` som gir læringsraten i epoke `t`: `lr_min + 0.5 * (lr0 - lr_min) * (1 + cos(pi * t / T_max))`.
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

--- LEKSJON 9.11 | reproduserbarhet | Reproduserbarhet: seeds og determinisme
Startvekter, stokking, train/val-split og Dropout er tilfeldige, og siden tapsflaten ikke er konveks kan en annen seed gi et helt annet resultat. `torch.manual_seed(0)` gjør at PyTorch trekker de samme tallene hver gang.

## set_seed
Python, NumPy og PyTorch har hver sin generator, så `set_seed` i prosjekt 3 kaller `random.seed(seed)`, `np.random.seed(seed)` og `torch.manual_seed(seed)`. I tillegg gir `torch.use_deterministic_algorithms(True)` og `torch.set_num_threads(1)` identiske flyttall fra kjøring til kjøring.

::: oppgave set_seed
Skriv `set_seed(seed)` som seeder både Pythons `random`-modul og NumPy, slik at koden i testen gir de samme tallene to ganger.
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

::: oppgave Reproduserbar trening
`tren` gir et nytt svar hver gang. Lag `rng = np.random.default_rng(seed)` i funksjonen og bruk den til både startvektene og stokkingen.
```start
import numpy as np

def tren(seed):
    W = np.random.normal(size=(2, 4))     # startvekter
    perm = np.random.permutation(10)      # stokking
    return W.sum() + perm[0]
```
```losning
import numpy as np

def tren(seed):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(2, 4))
    perm = rng.permutation(10)
    return W.sum() + perm[0]
```
```test
assert tren(0) == tren(0), "Samme seed skal gi nøyaktig samme resultat. Bruk rng = np.random.default_rng(seed)."
assert tren(0) != tren(1), "Ulike seeds skal gi ulike resultater. Bruk seed-argumentet."
```
:::

## Egen generator for stokking
`torch.randperm(n, generator=g)` med `g = torch.Generator().manual_seed(seed)` gir samme stokking uansett hvor mange tall startvektene brukte, så arkitekturer sammenlignes rettferdig (det samme gjelder `DataLoader(..., generator=g)`).

```python statisk
import torch
import torch.nn as nn

for bredde in [16, 64]:
    torch.manual_seed(0)
    net = nn.Linear(2, bredde)                   # trekker 3 * bredde tall
    g = torch.Generator().manual_seed(0)
    print(bredde, torch.randperm(6).tolist(), torch.randperm(6, generator=g).tolist())
```
```output
16 [2, 0, 1, 5, 3, 4] [2, 5, 3, 0, 1, 4]
64 [0, 1, 2, 4, 3, 5] [2, 5, 3, 0, 1, 4]
```

::: oppgave Stokking uavhengig av bredden
I `stokking` avhenger `perm` av hvor mange startvekter som ble trukket. Lag en egen `rng_stokk = np.random.default_rng(seed)` til `perm`, så resultatet blir likt for alle bredder.
```start
import numpy as np

def stokking(bredde, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(2, bredde))     # startvekter
    perm = rng.permutation(8)            # stokking
    return perm
```
```losning
import numpy as np

def stokking(bredde, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(2, bredde))
    rng_stokk = np.random.default_rng(seed)
    perm = rng_stokk.permutation(8)
    return perm
```
```test
import numpy as np
p16, p64 = stokking(16), stokking(64)
assert sorted(np.asarray(p16).tolist()) == list(range(8)), "perm skal være en stokking av 0-7"
assert np.array_equal(p16, p64), f"Stokkingen skal være lik for bredde 16 og 64, fikk {p16} og {p64}. Bruk en egen generator til permutation."
assert not np.array_equal(stokking(16, seed=0), stokking(16, seed=1)), "Ulike seeds skal gi ulik stokking"
```
:::

--- LEKSJON 9.12 | keras | Keras-versjonen fra notatene
Notatene bruker Keras, der `Dense(6, activation="relu")` er `nn.Linear` pluss aktivering og `Sequential([...])` stabler lagene (Keras er ikke installert her, så koden kjøres ikke). Output-shape er `(batch, noder i siste lag)`.

```python statisk ikkekjor
from keras import ops
from keras.layers import Dense
from keras.models import Sequential

x = ops.ones((10, 3))                   # 10 datapunkter, 3 features
model = Sequential([
    Dense(6, activation="relu", name="layer1"),
    Dense(7, activation="relu", name="layer2"),
    Dense(2, activation="softmax", name="output"),
])
model(x)                                # shape (10, 2)
```

## Vektene lages ved første kall
`Dense(6)` vet ikke hvor mange inputs den får, så `model.weights` er `[]` til modellen er kalt på data eller har fått `keras.Input(shape=(3,))`. Vektene (`kernel`) har shape `(inn, ut)`, motsatt av `nn.Linear`, og `model.summary()` viser parametrene.

::: oppgave Et Dense-lag som bygger seg selv
Skriv `Dense` der `__init__(self, units)` lagrer `self.units` og setter `self.kernel` og `self.bias` til `None`, og `__call__(self, X)` lager `kernel` med shape `(X.shape[1], units)` (`np.random.randn`) og null-bias første gang før den returnerer `X @ self.kernel + self.bias`. `count_params(self)` gir 0 før laget er bygget.
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

::: oppgave Kernel-shapes og parametre
Skriv `kernel_shapes(n_in, units)` som gir listen med kernel-shapes `(inn, ut)` for `keras.Input(shape=(n_in,))` fulgt av `Dense(u)` for hver `u` i `units`. Lagre parametertallet for `n_in = 3`, `units = [6, 7, 2]` i `n_params`.
```start
def kernel_shapes(n_in, units):
    ...

n_params = ...
```
```losning
def kernel_shapes(n_in, units):
    shapes, d = [], n_in
    for u in units:
        shapes.append((d, u))
        d = u
    return shapes

n_params = sum(inn * ut + ut for inn, ut in kernel_shapes(3, [6, 7, 2]))
```
```test
r = kernel_shapes(3, [6, 7, 2])
assert r is not None, "kernel_shapes returnerer ingenting"
assert [tuple(s) for s in r] == [(3, 6), (6, 7), (7, 2)], f"Forventet [(3, 6), (6, 7), (7, 2)], fikk {r}"
assert [tuple(s) for s in kernel_shapes(20, [32, 5])] == [(20, 32), (32, 5)], "kernel_shapes(20, [32, 5]) skal gi [(20, 32), (32, 5)]"
assert n_params == 89, f"(3*6 + 6) + (6*7 + 7) + (7*2 + 2) = 89, fikk {n_params}"
```
:::

## compile, fit og history
`model.compile(optimizer, loss, metrics)` velger optimizer og tap, og `history = model.fit(x, y, batch_size=64, epochs=10, validation_data=(x_val, y_val))` kjører hele treningsløkka. `history.history["loss"]` og `["val_loss"]` har én verdi per epoke, og `keras.utils.set_random_seed(0)` seeder alt.

```python statisk ikkekjor
model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-3),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=["accuracy"])
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=40,
                                           restore_best_weights=True)
history = model.fit(X_train, y_train, batch_size=32, epochs=300,
                    validation_data=(X_val, y_val), callbacks=[early_stop])
y_prob = model.predict(X_test)
test_loss, test_acc = model.evaluate(X_test, y_test)
model.save("modell.keras")              # keras.models.load_model("modell.keras")
```

::: oppgave Keras-tap i PyTorch
Fyll inn `til_pytorch` med PyTorch-tapsfunksjonen (f.eks. `"nn.MSELoss()"`) som tilsvarer hvert Keras-tap.
```start
til_pytorch = {
    "MeanSquaredError": ...,
    "BinaryCrossentropy(from_logits=True)": ...,
    "binary_crossentropy (sigmoid i siste lag)": ...,
    "sparse_categorical_crossentropy (logits)": ...,
}
```
```losning
til_pytorch = {
    "MeanSquaredError": "nn.MSELoss()",
    "BinaryCrossentropy(from_logits=True)": "nn.BCEWithLogitsLoss()",
    "binary_crossentropy (sigmoid i siste lag)": "nn.BCELoss()",
    "sparse_categorical_crossentropy (logits)": "nn.CrossEntropyLoss()",
}
```
```test
_v = {k: str(v) for k, v in til_pytorch.items()}
assert "MSELoss" in _v["MeanSquaredError"], "MeanSquaredError tilsvarer nn.MSELoss()"
assert "BCEWithLogitsLoss" in _v["BinaryCrossentropy(from_logits=True)"], "from_logits=True betyr sigmoid inni tapet: nn.BCEWithLogitsLoss()"
assert "BCELoss" in _v["binary_crossentropy (sigmoid i siste lag)"] and "Logits" not in _v["binary_crossentropy (sigmoid i siste lag)"], "Med sigmoid i nettet gir nettet sannsynligheter, så det er nn.BCELoss()"
assert "CrossEntropyLoss" in _v["sparse_categorical_crossentropy (logits)"], "Klasseindekser og logits gir nn.CrossEntropyLoss()"
```
:::

::: oppgave Plott history
`historikk` er som `history.history`. Plott `"loss"` og `"val_loss"` mot epoke 1, 2, 3, … i `ax` med `label`, lag `ax.legend()`, og lagre epoken (fra 1) med lavest valideringstap i `beste_epoke`, den `restore_best_weights=True` ville gitt.
```start
import numpy as np
import matplotlib.pyplot as plt

historikk = {"loss": [0.69, 0.52, 0.41, 0.35, 0.31, 0.29, 0.27],
             "val_loss": [0.70, 0.55, 0.46, 0.43, 0.44, 0.47, 0.50]}
fig, ax = plt.subplots()

beste_epoke = ...
```
```losning
import numpy as np
import matplotlib.pyplot as plt

historikk = {"loss": [0.69, 0.52, 0.41, 0.35, 0.31, 0.29, 0.27],
             "val_loss": [0.70, 0.55, 0.46, 0.43, 0.44, 0.47, 0.50]}
fig, ax = plt.subplots()

epoker = np.arange(1, len(historikk["loss"]) + 1)
ax.plot(epoker, historikk["loss"], label="Train loss")
ax.plot(epoker, historikk["val_loss"], label="Validation loss")
ax.legend()
beste_epoke = int(np.argmin(historikk["val_loss"])) + 1
```
```test
import numpy as np
assert len(ax.lines) == 2, f"Plott to kurver (loss og val_loss), fant {len(ax.lines)}"
ys = sorted([list(np.round(l.get_ydata(), 6)) for l in ax.lines])
assert ys == sorted([historikk["loss"], historikk["val_loss"]]), "Kurvene skal vise historikk['loss'] og historikk['val_loss']"
assert all(np.allclose(l.get_xdata(), np.arange(1, 8)) for l in ax.lines), "x-aksen skal være epoke 1, 2, ..., 7: np.arange(1, len(...) + 1)"
assert ax.get_legend() is not None, "Lag forklaring med ax.legend() (husk label=...)"
assert beste_epoke == 4, f"Lavest val_loss (0.43) er i epoke 4, fikk {beste_epoke}. np.argmin gir indeks 3, så legg til 1."
```
:::
