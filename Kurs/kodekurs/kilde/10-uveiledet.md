=== DEL 10 | uveiledet | Uveiledet læring (prosjekt 3)
Klynging, PCA, t-SNE og anomalideteksjon uten labels, med teknikkene fra prosjekt 3.

--- LEKSJON 10.1 | avstander | Avstander og dimensjonalitetens forbannelse
`np.linalg.norm(p - q)` gir den euklidske avstanden, og `np.linalg.norm(X - p, axis=1)` gir avstanden fra `p` til hver rad i `X`.

::: oppgave Avstand mellom to punkter
Regn ut den euklidske avstanden mellom `p` og `q`, og lagre den i `d`.
```start
import numpy as np

p = np.array([1.0, 2.0])
q = np.array([4.0, 6.0])
d = ...
```
```losning
import numpy as np

p = np.array([1.0, 2.0])
q = np.array([4.0, 6.0])
d = np.linalg.norm(p - q)
```
```test
assert d is not ..., "Sett d."
assert abs(float(d) - 5.0) < 1e-9, f"Avstanden skal være sqrt(3² + 4²) = 5, fikk {d}."
```
:::

::: oppgave Nærmeste punkt
Lagre avstanden fra `p` til hver rad i `X` i `dists` (shape `(5,)`), og indeksen til raden som ligger nærmest i `naermeste`.
```start
import numpy as np

p = np.array([1.0, 2.0])
X = np.array([[0, 0], [1, 1], [4, 6], [10, 0], [2, 3]])
dists = ...
naermeste = ...
```
```losning
import numpy as np

p = np.array([1.0, 2.0])
X = np.array([[0, 0], [1, 1], [4, 6], [10, 0], [2, 3]])
dists = np.linalg.norm(X - p, axis=1)
naermeste = np.argmin(dists)
```
```test
import numpy as np
assert np.asarray(dists).shape == (5,), f"dists skal ha én avstand per rad, shape (5,), fikk {np.asarray(dists).shape}. Bruk axis=1."
assert np.allclose(dists, [np.sqrt(5), 1, 5, np.sqrt(85), np.sqrt(2)]), "Feil avstander. Trekk p fra hver rad og ta normen langs axis=1."
assert naermeste == 1, f"Rad 1 ligger nærmest (avstand 1), fikk {naermeste}. Bruk np.argmin."
```
:::

::: oppgave Avstandsmatrise med broadcasting
Lag `D` med shape `(5, 3)` der `D[i, j]` er avstanden fra `X[i]` til `C[j]`, med `X[:, None, :] - C[None, :, :]` og normen langs siste akse (eller `cdist`).
```start
import numpy as np

X = np.array([[0, 0], [3, 4], [6, 8], [1, 0], [0, 2]], dtype=float)
C = np.array([[0, 0], [6, 8], [3, 0]], dtype=float)
D = ...
```
```losning
import numpy as np

X = np.array([[0, 0], [3, 4], [6, 8], [1, 0], [0, 2]], dtype=float)
C = np.array([[0, 0], [6, 8], [3, 0]], dtype=float)
D = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
```
```test
import numpy as np
from scipy.spatial.distance import cdist
assert np.asarray(D).shape == (5, 3), f"D skal ha shape (5, 3), fikk {np.asarray(D).shape}. Normen skal tas langs axis=2."
assert np.allclose(D, cdist(X, C)), "D[i, j] skal være avstanden fra X[i] til C[j]."
```
:::

## Avstander konsentrerer seg i høye dimensjoner
`d2.std() / d2.mean()` måler hvor ulike avstandene er, og i høye dimensjoner blir den nær 0, så «nær» og «fjern» nesten er like.

::: oppgave Relativ spredning
Bruk `get_cube_dists` (gitt) for `dim = 2` og `dim = 1000`, og lagre `std / mean` for hver i `rel_2` og `rel_1000`.
```start
import numpy as np

rng = np.random.default_rng(0)

def get_cube_dists(dim, n_iter=200):
    P1 = rng.uniform(size=(n_iter, dim))
    P2 = rng.uniform(size=(n_iter, dim))
    return np.linalg.norm(P1 - P2, axis=1)

rel_2 = ...
rel_1000 = ...
```
```losning
import numpy as np

rng = np.random.default_rng(0)

def get_cube_dists(dim, n_iter=200):
    P1 = rng.uniform(size=(n_iter, dim))
    P2 = rng.uniform(size=(n_iter, dim))
    return np.linalg.norm(P1 - P2, axis=1)

d2 = get_cube_dists(2)
d1000 = get_cube_dists(1000)
rel_2 = d2.std() / d2.mean()
rel_1000 = d1000.std() / d1000.mean()
```
```test
assert isinstance(rel_2, float) and isinstance(rel_1000, float), "rel_2 og rel_1000 skal være tall (std delt på snitt)."
assert 0.3 < rel_2 < 0.7, f"For d=2 bør std/snitt være rundt 0.5, fikk {rel_2:.3f}."
assert rel_1000 < 0.05, f"For d=1000 bør std/snitt være liten (rundt 0.02), fikk {rel_1000:.3f}."
```
:::

::: oppgave Skaler før du måler avstand
`X` har inntekt i kroner og alder i år. Finn indeksen til nærmeste nabo til rad 0 uten skalering (`nn_uten`) og etter `StandardScaler` (`nn_med`), og se at kronene bestemmer alt uten skalering.
```start
import numpy as np
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
X = np.column_stack([rng.normal(500000, 50000, 50), rng.uniform(20, 70, 50)])

nn_uten = ...
nn_med = ...
```
```losning
import numpy as np
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
X = np.column_stack([rng.normal(500000, 50000, 50), rng.uniform(20, 70, 50)])

d = np.linalg.norm(X - X[0], axis=1)
d[0] = np.inf
nn_uten = np.argmin(d)
Xs = StandardScaler().fit_transform(X)
ds = np.linalg.norm(Xs - Xs[0], axis=1)
ds[0] = np.inf
nn_med = np.argmin(ds)
```
```test
assert nn_uten is not ... and nn_med is not ..., "Sett både nn_uten og nn_med."
assert nn_uten != 0 and nn_med != 0, "Rad 0 er nærmest seg selv. Hopp over den, f.eks. ved å sette d[0] = np.inf."
assert nn_uten == 3, f"Uten skalering skal nærmeste nabo være rad 3, fikk {nn_uten}."
assert nn_med == 35, f"Med StandardScaler skal nærmeste nabo være rad 35, fikk {nn_med}. Skalerte du hele X før du målte avstand?"
```
:::

--- LEKSJON 10.2 | kmeans-for-hand | K-means for hånd
K-means gjentar to steg til sentroidene står stille: gi hvert punkt label til nærmeste sentroide (`np.argmin(D, axis=1)`), og flytt hver sentroide til snittet av punktene sine (`X[labels == j].mean(axis=0)`).

::: oppgave Tilordning
Skriv `tilordne(X, C)` som returnerer indeksen til nærmeste sentroide for hvert punkt (shape `(n,)`).
```start
import numpy as np

def tilordne(X, C):
    ...
```
```losning
import numpy as np

def tilordne(X, C):
    D = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
    return np.argmin(D, axis=1)
```
```test
import numpy as np
X_ = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0], [5.0, 0.0]])
C_ = np.array([[0.0, 0.0], [10.0, 10.0]])
lab = tilordne(X_, C_)
assert lab is not None, "tilordne må returnere noe."
assert np.asarray(lab).shape == (5,), f"tilordne skal returnere shape (5,), fikk {np.asarray(lab).shape}. Bruk argmin langs axis=1."
assert list(lab) == [0, 0, 1, 1, 0], f"Feil tilordning: {list(lab)}. Hvert punkt skal få indeksen (argmin, ikke min) til nærmeste sentroide."
```
:::

::: oppgave Oppdatering
Skriv `oppdater(X, labels, k)` som returnerer et array med shape `(k, d)` der rad `j` er snittet av punktene med label `j`.
```start
import numpy as np

def oppdater(X, labels, k):
    ...
```
```losning
import numpy as np

def oppdater(X, labels, k):
    return np.array([X[labels == j].mean(axis=0) for j in range(k)])
```
```test
import numpy as np
X_ = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0], [5.0, 0.0]])
C2 = oppdater(X_, np.array([0, 0, 1, 1, 0]), 2)
assert C2 is not None and np.asarray(C2).shape == (2, 2), "oppdater skal returnere shape (k, d) = (2, 2)."
assert np.allclose(C2, [[5/3, 1/3], [10.0, 10.5]]), "Hver rad skal være snittet av punktene i klyngen (mean med axis=0)."
```
:::

::: oppgave Inertia
`C[labels]` gir sentroiden til hvert punkt. Regn ut `inertia`, summen av kvadrerte avstander fra hvert punkt til sin egen sentroide.
```start
import numpy as np

X = np.array([[0.0, 0.0], [0.0, 2.0], [10.0, 10.0], [12.0, 10.0]])
C = np.array([[0.0, 1.0], [11.0, 10.0]])
labels = np.array([0, 0, 1, 1])
inertia = ...
```
```losning
import numpy as np

X = np.array([[0.0, 0.0], [0.0, 2.0], [10.0, 10.0], [12.0, 10.0]])
C = np.array([[0.0, 1.0], [11.0, 10.0]])
labels = np.array([0, 0, 1, 1])
inertia = np.sum((X - C[labels])**2)
```
```test
assert inertia is not ..., "Sett inertia."
assert abs(float(inertia) - 4.0) < 1e-9, f"Hvert punkt ligger 1 unna sin sentroide, så inertia skal være 4, fikk {inertia}. Husk å kvadrere."
```
:::

::: oppgave Hele algoritmen
Skriv `kmeans(X, k, seed=0, maks_iter=100)` som starter i `k` tilfeldige punkter (`rng.choice(len(X), size=k, replace=False)`), gjentar tilordne/oppdater til `np.allclose(C_ny, C)`, og returnerer `labels, C`.
```start
import numpy as np
from sklearn.datasets import make_blobs

def tilordne(X, C):
    return np.argmin(np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2), axis=1)

def oppdater(X, labels, k):
    return np.array([X[labels == j].mean(axis=0) for j in range(k)])

def kmeans(X, k, seed=0, maks_iter=100):
    rng = np.random.default_rng(seed)
    ...
```
```losning
import numpy as np
from sklearn.datasets import make_blobs

def tilordne(X, C):
    return np.argmin(np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2), axis=1)

def oppdater(X, labels, k):
    return np.array([X[labels == j].mean(axis=0) for j in range(k)])

def kmeans(X, k, seed=0, maks_iter=100):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=k, replace=False)]
    for _ in range(maks_iter):
        labels = tilordne(X, C)
        C_ny = oppdater(X, labels, k)
        if np.allclose(C_ny, C):
            break
        C = C_ny
    return labels, C
```
```test
import numpy as np
from sklearn.datasets import make_blobs
X_, _ = make_blobs(n_samples=300, centers=3, random_state=0)
res = kmeans(X_, 3)
assert res is not None and len(res) == 2, "kmeans skal returnere labels, C."
lab, C_ = res
assert np.asarray(lab).shape == (300,) and np.asarray(C_).shape == (3, 2), "labels skal ha shape (300,) og C shape (3, 2)."
assert np.allclose(C_, oppdater(X_, np.asarray(lab), 3)), "Sentroidene er ikke snittet av klyngene sine. Kjører løkka til C slutter å endre seg?"
assert np.array_equal(lab, tilordne(X_, C_)), "labels stemmer ikke med nærmeste sentroide i den endelige C."
```
:::

## Initialiseringen betyr noe
K-means finner bare et lokalt minimum, så sklearn kjører flere ganger (`n_init`) med k-means++-start og beholder lavest inertia.

::: oppgave Prøv flere seeds
Kjør `kmeans` (gitt) med `k=6` for `seed = 0, 1, ..., 7`, lagre inertiaen for hver i listen `inertias`, og seeden med lavest inertia i `beste_seed`.
```start
import numpy as np
from sklearn.datasets import make_blobs

def kmeans(X, k, seed=0, maks_iter=100):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=k, replace=False)]
    for _ in range(maks_iter):
        labels = np.argmin(np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2), axis=1)
        C_ny = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        if np.allclose(C_ny, C):
            break
        C = C_ny
    return labels, C

X, _ = make_blobs(n_samples=500, centers=6, cluster_std=0.6, random_state=4)
inertias = ...
beste_seed = ...
```
```losning
import numpy as np
from sklearn.datasets import make_blobs

def kmeans(X, k, seed=0, maks_iter=100):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=k, replace=False)]
    for _ in range(maks_iter):
        labels = np.argmin(np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2), axis=1)
        C_ny = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        if np.allclose(C_ny, C):
            break
        C = C_ny
    return labels, C

X, _ = make_blobs(n_samples=500, centers=6, cluster_std=0.6, random_state=4)
inertias = []
for seed in range(8):
    labels, C = kmeans(X, 6, seed=seed)
    inertias.append(np.sum((X - C[labels])**2))
beste_seed = int(np.argmin(inertias))
```
```test
import numpy as np
assert isinstance(inertias, list) and len(inertias) == 8, "inertias skal være en liste med 8 tall (seed 0 til 7)."
fasit = []
for s in range(8):
    l_, C_ = kmeans(X, 6, seed=s)
    fasit.append(np.sum((X - C_[l_])**2))
assert np.allclose(inertias, fasit), "Inertia for hver seed skal være np.sum((X - C[labels])**2)."
assert beste_seed == int(np.argmin(fasit)), f"beste_seed skal være seeden med lavest inertia, altså {int(np.argmin(fasit))}."
```
:::

--- LEKSJON 10.3 | kmeans-sklearn | KMeans i sklearn
`km = KMeans(n_clusters=3, n_init=10, random_state=0)` og `km.fit_predict(X)` gir labels, og etterpå finnes `km.labels_`, `km.cluster_centers_` og `km.inertia_`.

::: oppgave Tilpass KMeans
Tilpass `km` som over på `X`, og lagre labelene i `labels`, sentroidene i `sentroider` og inertiaen i `inertia`.
```start
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=300, centers=3, random_state=0)
km = ...
labels = ...
sentroider = ...
inertia = ...
```
```losning
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=300, centers=3, random_state=0)
km = KMeans(n_clusters=3, n_init=10, random_state=0)
labels = km.fit_predict(X)
sentroider = km.cluster_centers_
inertia = km.inertia_
```
```test
import numpy as np
f = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
assert labels is not ... and np.array_equal(labels, f.labels_), "labels skal komme fra km.fit_predict(X)."
assert np.asarray(sentroider).shape == (3, 2), "sentroider skal være km.cluster_centers_, shape (3, 2)."
assert np.allclose(sentroider, f.cluster_centers_), "sentroider stemmer ikke med cluster_centers_."
assert inertia is not ... and np.isclose(inertia, f.inertia_), "inertia skal være km.inertia_."
```
:::

::: oppgave predict og transform
`km.predict(nye)` gir nærmeste klynge og `km.transform(nye)` avstanden til alle sentroidene. Lagre dem i `klynger_nye` og `avst`, og avstanden til nærmeste sentroide i `naermeste_avst`.
```start
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=300, centers=3, random_state=0)
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
nye = np.array([[0, 4], [2, 0], [-1, 3]])

klynger_nye = ...
avst = ...
naermeste_avst = ...
```
```losning
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=300, centers=3, random_state=0)
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
nye = np.array([[0, 4], [2, 0], [-1, 3]])

klynger_nye = km.predict(nye)
avst = km.transform(nye)
naermeste_avst = avst.min(axis=1)
```
```test
import numpy as np
assert klynger_nye is not ... and np.array_equal(klynger_nye, km.predict(nye)), "klynger_nye skal være km.predict(nye)."
assert np.asarray(avst).shape == (3, 3), f"avst skal ha shape (antall punkter, antall sentroider) = (3, 3), fikk {np.asarray(avst).shape}."
assert np.allclose(avst, km.transform(nye)), "avst skal være km.transform(nye)."
assert np.asarray(naermeste_avst).shape == (3,) and np.allclose(naermeste_avst, avst.min(axis=1)), "naermeste_avst skal være minste avstand per rad: .min(axis=1)."
```
:::

::: oppgave MiniBatchKMeans
For store datasett har `MiniBatchKMeans` samme grensesnitt. Tilpass `mb = MiniBatchKMeans(n_clusters=3, n_init=3, random_state=0)` på `X` og lagre labelene i `labels_mb`.
```start
from sklearn.datasets import make_blobs
from sklearn.cluster import MiniBatchKMeans

X, _ = make_blobs(n_samples=300, centers=3, random_state=0)
mb = ...
labels_mb = ...
```
```losning
from sklearn.datasets import make_blobs
from sklearn.cluster import MiniBatchKMeans

X, _ = make_blobs(n_samples=300, centers=3, random_state=0)
mb = MiniBatchKMeans(n_clusters=3, n_init=3, random_state=0)
labels_mb = mb.fit_predict(X)
```
```test
import numpy as np
assert isinstance(mb, MiniBatchKMeans), "mb skal være en MiniBatchKMeans."
assert hasattr(mb, "cluster_centers_") and mb.cluster_centers_.shape == (3, 2), "Husk å tilpasse mb på X (fit eller fit_predict)."
assert labels_mb is not ... and len(labels_mb) == 300 and len(set(labels_mb)) == 3, "labels_mb skal ha én label per punkt, med 3 klynger."
```
:::

## Albuemetoden
Inertia synker alltid når `k` øker, så vi plotter `inertia_` mot `k` og ser etter knekken (albuen) i stedet for å ta lavest verdi.

::: oppgave Albuekurven
Lag listen `inertias` med `inertia_` for `KMeans(n_clusters=k, n_init=10, random_state=0)` for `k = 1, 2, ..., 8` på `X`, og plott den mot `k` på `ax`.
```start
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=400, centers=5, random_state=2)
fig, ax = plt.subplots()
inertias = ...
```
```losning
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=400, centers=5, random_state=2)
fig, ax = plt.subplots()
inertias = []
for k in range(1, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    inertias.append(km.inertia_)
ax.plot(range(1, 9), inertias, "o-")
```
```test
import numpy as np
assert isinstance(inertias, list), "inertias skal være en liste."
assert len(inertias) == 8, f"Forventet 8 verdier (k = 1 til 8), fikk {len(inertias)}."
assert all(a >= b for a, b in zip(inertias, inertias[1:])), "Inertia skal falle når k øker. Bruker du inertia_?"
fasit1 = KMeans(n_clusters=1, n_init=10, random_state=0).fit(X).inertia_
assert np.isclose(inertias[0], fasit1), "Første verdi skal være for k = 1. Start range på 1."
assert len(ax.lines) >= 1, "Plott inertias mot k med ax.plot."
```
:::

## Skaler først
Avstander domineres av featuren med størst tall, så kjør `StandardScaler().fit_transform(X)` før k-means når featurene har ulike enheter.

::: oppgave Med og uten skalering
Feature 0 i `X` er ganget med 100. Lagre KMeans-labelene (`n_clusters=2, n_init=10, random_state=0`) på rå `X` i `uten` og på skalert `X` i `med`.
```start
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X, y = make_blobs(n_samples=300, centers=[[0, 0], [0, 5]], random_state=0)
X[:, 0] = X[:, 0] * 100

uten = ...
med = ...
```
```losning
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X, y = make_blobs(n_samples=300, centers=[[0, 0], [0, 5]], random_state=0)
X[:, 0] = X[:, 0] * 100

uten = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
med = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(X))
```
```test
import numpy as np
def enighet(a, b):
    return max(np.mean(a == b), np.mean(a != b))
assert uten is not ... and len(uten) == 300, "uten skal være én label per punkt fra KMeans på rå X."
assert med is not ... and len(med) == 300, "med skal være én label per punkt fra KMeans på skalert X."
assert enighet(uten, y) < 0.7, "uten skal være KMeans på X uten skalering."
assert enighet(med, y) > 0.95, f"med bør finne de to klyngene ({enighet(med, y):.0%} enighet). Skalerte du X med StandardScaler først?"
```
:::

--- LEKSJON 10.4 | silhouette | Silhouette og valg av k
Silhouette for ett punkt er `s = (b - a) / max(a, b)`, der `a` er snittavstanden til egen klynge og `b` til nærmeste andre klynge. `silhouette_samples(X, labels)` gir `s` per punkt og `silhouette_score(X, labels)` snittet (nær 1 er bra, negativ betyr trolig feil klynge).

::: oppgave Silhouette for hånd
Regn ut `a`, `b` og `s` for punktet `X[2]` (verdien 2), der den andre klyngen er punktene 10 og 11.
```start
import numpy as np

X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
labels = np.array([0, 0, 0, 1, 1])

a = ...
b = ...
s = ...
```
```losning
import numpy as np

X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
labels = np.array([0, 0, 0, 1, 1])

a = np.mean([2 - 0, 2 - 1])
b = np.mean([10 - 2, 11 - 2])
s = (b - a) / max(a, b)
```
```test
import numpy as np
from sklearn.metrics import silhouette_samples
assert a is not ... and abs(float(a) - 1.5) < 1e-9, f"a er snittavstanden til de ANDRE punktene i egen klynge (0 og 1), altså 1.5. Fikk {a}."
assert b is not ... and abs(float(b) - 8.5) < 1e-9, f"b er snittavstanden til punktene i den andre klyngen (10 og 11), altså 8.5. Fikk {b}."
assert s is not ... and abs(float(s) - silhouette_samples(X, labels)[2]) < 1e-9, "s skal være (b - a) / max(a, b)."
```
:::

::: oppgave Finn feilplasserte punkter
Tre punkter i `labels` er lagt i feil klynge. Lagre `silhouette_samples` i `s_alle`, `silhouette_score` i `score`, og indeksene med negativ silhouette i `feil_plassert`.
```start
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

X, y = make_blobs(n_samples=60, centers=[[0, 0], [6, 0]], cluster_std=1.0, random_state=1)
labels = y.copy()
labels[[3, 10, 25]] = 1 - labels[[3, 10, 25]]

s_alle = ...
score = ...
feil_plassert = ...
```
```losning
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

X, y = make_blobs(n_samples=60, centers=[[0, 0], [6, 0]], cluster_std=1.0, random_state=1)
labels = y.copy()
labels[[3, 10, 25]] = 1 - labels[[3, 10, 25]]

s_alle = silhouette_samples(X, labels)
score = silhouette_score(X, labels)
feil_plassert = np.where(s_alle < 0)[0]
```
```test
import numpy as np
assert np.asarray(s_alle).shape == (60,), "s_alle skal ha én verdi per punkt (silhouette_samples)."
assert score is not ... and np.isclose(score, np.mean(s_alle)), "score skal være silhouette_score(X, labels), som er snittet av s_alle."
assert feil_plassert is not ... and sorted(np.asarray(feil_plassert).tolist()) == [3, 10, 25], f"feil_plassert skal være indeksene der s < 0 (np.where), fikk {feil_plassert}."
```
:::

## Velg k med en tabell
Prøv flere `k` og velg høyest `silhouette_score` eller `calinski_harabasz_score` (høy er bra), og sjekk klyngestørrelsene med `np.bincount(labels)`.

::: oppgave Finn beste k
Prøv `k = 2, ..., 7` med `KMeans(n_clusters=k, n_init=10, random_state=0)` på `X`, og lagre den `k`-en med høyest `silhouette_score` i `beste_k` (et heltall).
```start
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=400, centers=3, cluster_std=0.8, random_state=7)
beste_k = ...
```
```losning
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=400, centers=3, cluster_std=0.8, random_state=7)
scores = {}
for k in range(2, 8):
    lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    scores[k] = silhouette_score(X, lab)
beste_k = max(scores, key=scores.get)
```
```test
assert beste_k is not ..., "Sett beste_k."
assert int(beste_k) == beste_k, "beste_k skal være et heltall (selve k-en, ikke silhouette-verdien)."
assert beste_k == 3, f"Forventet at k = 3 gir høyest silhouette, fikk {beste_k}. Sammenligner du silhouette_score for hver k?"
```
:::

::: oppgave Tabell som i prosjekt 3
Lag DataFrame-en `tabell` med én rad per `k = 2, ..., 6` og kolonnene `k`, `silhouette`, `calinski_harabasz` og `storste_klynge` (antall punkter i største klynge, `np.bincount(lab).max()`).
```start
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

X, _ = make_blobs(n_samples=500, centers=4, random_state=3)
tabell = ...
```
```losning
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

X, _ = make_blobs(n_samples=500, centers=4, random_state=3)
rows = []
for k in range(2, 7):
    lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    rows.append({"k": k,
                 "silhouette": silhouette_score(X, lab),
                 "calinski_harabasz": calinski_harabasz_score(X, lab),
                 "storste_klynge": np.bincount(lab).max()})
tabell = pd.DataFrame(rows)
```
```test
import numpy as np
import pandas as pd
assert isinstance(tabell, pd.DataFrame), "tabell skal være en DataFrame, f.eks. pd.DataFrame(rows)."
for kol in ["k", "silhouette", "calinski_harabasz", "storste_klynge"]:
    assert kol in tabell.columns, f"Mangler kolonnen {kol}."
assert list(tabell["k"]) == [2, 3, 4, 5, 6], "k skal gå fra 2 til 6."
lab4 = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(X)
rad = tabell[tabell["k"] == 4].iloc[0]
assert np.isclose(rad["silhouette"], silhouette_score(X, lab4)), "Feil silhouette for k = 4."
assert np.isclose(rad["calinski_harabasz"], calinski_harabasz_score(X, lab4)), "Feil calinski_harabasz for k = 4."
assert rad["storste_klynge"] == np.bincount(lab4).max(), "storste_klynge skal være np.bincount(lab).max()."
```
:::

--- LEKSJON 10.5 | dbscan | DBSCAN
`DBSCAN(eps=0.2, min_samples=5).fit_predict(X)` lager klynger av tette områder (punkter med minst `min_samples` naboer innen `eps`), trenger ingen `k`, og finner vilkårlige former som k-means ikke klarer.

::: oppgave KMeans mot DBSCAN på halvmåner
Lagre labelene fra `KMeans(n_clusters=2, n_init=10, random_state=0)` i `lab_km` og fra `DBSCAN(eps=0.2, min_samples=5)` i `lab_db`, og plott `X` farget med `lab_db` på `ax`.
```start
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN

X, y = make_moons(n_samples=300, noise=0.05, random_state=0)
fig, ax = plt.subplots()
lab_km = ...
lab_db = ...
```
```losning
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN

X, y = make_moons(n_samples=300, noise=0.05, random_state=0)
fig, ax = plt.subplots()
lab_km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
lab_db = DBSCAN(eps=0.2, min_samples=5).fit_predict(X)
ax.scatter(X[:, 0], X[:, 1], c=lab_db, s=10)
```
```test
import numpy as np
def enighet(a, b):
    return max(np.mean(a == b), np.mean(a != b))
assert lab_km is not ... and len(lab_km) == 300, "lab_km skal være én label per punkt."
assert lab_db is not ... and len(lab_db) == 300, "lab_db skal være én label per punkt."
assert enighet(lab_km, y) < 0.9, "lab_km skal komme fra KMeans, som deler månene med en rett linje."
assert enighet(lab_db, y) > 0.99, "lab_db skal komme fra DBSCAN(eps=0.2, min_samples=5) og følge månene."
assert len(ax.collections) >= 1, "Plott X med ax.scatter og c=lab_db."
```
:::

## Labels, støy og kjernepunkter
Label `-1` er støy, `len(set(labels) - {-1})` gir antall klynger, og `db.core_sample_indices_` gir kjernepunktene; DBSCAN har ingen `predict`, så bruk `fit`/`fit_predict` på alle dataene.

::: oppgave Tell klynger, støy og kjernepunkter
Tilpass `db = DBSCAN(eps=0.3, min_samples=5).fit(X)`, og lagre `labels`, antall klynger uten støy i `n_klynger`, antall støypunkter i `n_stoy` og antall kjernepunkter i `n_kjerne`.
```start
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.3, random_state=5)
rng = np.random.default_rng(0)
X = np.vstack([X, rng.uniform(-12, 12, size=(15, 2))])

db = ...
labels = ...
n_klynger = ...
n_stoy = ...
n_kjerne = ...
```
```losning
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.3, random_state=5)
rng = np.random.default_rng(0)
X = np.vstack([X, rng.uniform(-12, 12, size=(15, 2))])

db = DBSCAN(eps=0.3, min_samples=5).fit(X)
labels = db.labels_
n_klynger = len(set(labels) - {-1})
n_stoy = int(np.sum(labels == -1))
n_kjerne = len(db.core_sample_indices_)
```
```test
import numpy as np
from sklearn.cluster import DBSCAN
f = DBSCAN(eps=0.3, min_samples=5).fit(X)
assert labels is not ... and np.array_equal(labels, f.labels_), "labels skal komme fra DBSCAN(eps=0.3, min_samples=5) på X."
assert n_klynger == len(set(f.labels_) - {-1}), f"n_klynger skal ikke telle støy (-1) som klynge. Fikk {n_klynger}."
assert n_stoy == np.sum(f.labels_ == -1), f"n_stoy skal være antall punkter med label -1. Fikk {n_stoy}."
assert n_kjerne == len(f.core_sample_indices_), "n_kjerne skal være len(db.core_sample_indices_)."
```
:::

## Velg eps med et k-distanse-plott
`NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)` gir avstander til naboene (første kolonne er punktet selv), og knekken i den sorterte siste kolonnen er et godt valg av `eps`.

::: oppgave k-distanse-plott
Lagre den sorterte avstanden til nabo nummer `min_samples` for hvert punkt i `k_dist` (shape `(300,)`), og plott den på `ax`.
```start
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors

X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)
min_samples = 5
fig, ax = plt.subplots()
k_dist = ...
```
```losning
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors

X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)
min_samples = 5
fig, ax = plt.subplots()
nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
distances, _ = nn.kneighbors(X)
k_dist = np.sort(distances[:, -1])
ax.plot(k_dist)
```
```test
import numpy as np
from sklearn.neighbors import NearestNeighbors
assert np.asarray(k_dist).shape == (300,), f"k_dist skal ha én verdi per punkt, shape (300,), fikk {np.asarray(k_dist).shape}. Ta siste kolonne: distances[:, -1]."
fasit = np.sort(NearestNeighbors(n_neighbors=5).fit(X).kneighbors(X)[0][:, -1])
assert np.allclose(k_dist, fasit), "k_dist skal være den sorterte siste kolonnen av avstandene fra kneighbors med n_neighbors=min_samples."
assert len(ax.lines) >= 1, "Plott k_dist med ax.plot."
```
:::

## Følsom for eps og dimensjon
For liten `eps` gir mye støy og for stor slår alt sammen, og med flere dimensjoner blir avstandene større så samme `eps` slutter å virke.

::: oppgave Prøv flere eps
Lag dicten `n_klynger` som for hver `eps` i `eps_liste` gir antall klynger (uten støy) fra `DBSCAN(eps=eps, min_samples=5)` på `X`.
```start
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)
eps_liste = [0.05, 0.1, 0.2, 0.5]
n_klynger = ...
```
```losning
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)
eps_liste = [0.05, 0.1, 0.2, 0.5]
n_klynger = {}
for eps in eps_liste:
    lab = DBSCAN(eps=eps, min_samples=5).fit_predict(X)
    n_klynger[eps] = len(set(lab) - {-1})
```
```test
assert isinstance(n_klynger, dict), "n_klynger skal være en dict med eps som nøkkel."
assert set(n_klynger) == {0.05, 0.1, 0.2, 0.5}, "Nøklene skal være verdiene i eps_liste."
assert n_klynger[0.2] == 2, f"eps=0.2 skal gi 2 klynger, fikk {n_klynger[0.2]}. Teller du -1 som en klynge?"
assert n_klynger[0.5] == 1, "eps=0.5 skal slå alt sammen til 1 klynge."
assert n_klynger[0.05] > 5, "eps=0.05 skal gi mange små klynger."
```
:::

::: oppgave Flere dimensjoner
Legg til 8 kolonner med støy (`Xe = np.hstack([X, ekstra])`), kjør `DBSCAN(eps=0.2, min_samples=5)` på `Xe`, og lagre antall støypunkter i `n_stoy_10d`.
```start
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)
rng = np.random.default_rng(0)
ekstra = rng.normal(0, 0.3, size=(300, 8))
n_stoy_10d = ...
```
```losning
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)
rng = np.random.default_rng(0)
ekstra = rng.normal(0, 0.3, size=(300, 8))
Xe = np.hstack([X, ekstra])
n_stoy_10d = int(np.sum(DBSCAN(eps=0.2, min_samples=5).fit_predict(Xe) == -1))
```
```test
assert n_stoy_10d is not ..., "Sett n_stoy_10d."
assert n_stoy_10d == 300, f"I 10 dimensjoner blir alle 300 punktene støy med eps=0.2, fikk {n_stoy_10d}. Kjørte du DBSCAN på Xe?"
```
:::

--- LEKSJON 10.6 | andre-klyngemetoder | Andre klyngemetoder
Klyngemetoder er sentroidebaserte (k-means), tetthetsbaserte (DBSCAN), fordelingsbaserte (Gaussian Mixture) eller hierarkibaserte (agglomerativ klynging).

::: oppgave Kategorier
Lag dicten `kategori` som gir kategorien (`"sentroide"`, `"tetthet"`, `"fordeling"` eller `"hierarki"`) for nøklene `"KMeans"`, `"DBSCAN"`, `"GaussianMixture"` og `"AgglomerativeClustering"`.
```start
kategori = ...
```
```losning
kategori = {"KMeans": "sentroide", "DBSCAN": "tetthet",
            "GaussianMixture": "fordeling", "AgglomerativeClustering": "hierarki"}
```
```test
assert isinstance(kategori, dict), "kategori skal være en dict."
fasit = {"KMeans": "sentroide", "DBSCAN": "tetthet", "GaussianMixture": "fordeling", "AgglomerativeClustering": "hierarki"}
for k, v in fasit.items():
    assert k in kategori, f"Mangler nøkkelen {k}."
    assert kategori[k] == v, f"{k} er ikke {kategori[k]!r}. Tenk på hva metoden bygger på."
```
:::

## Hierarkisk klynging og dendrogram
`Z = linkage(X, method="ward")` og `dendrogram(Z)` fra `scipy.cluster.hierarchy` viser hierarkiet, og `AgglomerativeClustering(n_clusters=2, linkage="single")` gir labels (`"single"` følger tynne former, `"ward"` lager runde klynger).

::: oppgave Dendrogram
Lag `Z` med `linkage(X, method="ward")` og tegn dendrogrammet med `d = dendrogram(Z)`.
```start
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import linkage, dendrogram

X, _ = make_blobs(n_samples=30, centers=3, random_state=0)
Z = ...
d = ...
```
```losning
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import linkage, dendrogram

X, _ = make_blobs(n_samples=30, centers=3, random_state=0)
Z = linkage(X, method="ward")
d = dendrogram(Z)
```
```test
import numpy as np
from scipy.cluster.hierarchy import linkage
assert np.asarray(Z).shape == (29, 4), "Z skal komme fra linkage(X, method='ward'): én rad per sammenslåing, shape (29, 4)."
assert np.allclose(Z, linkage(X, method="ward")), "Bruk method='ward'."
assert isinstance(d, dict) and "icoord" in d, "d skal være resultatet av dendrogram(Z)."
```
:::

::: oppgave Halvmåner med agglomerativ klynging
Bruk `AgglomerativeClustering` med 2 klynger på halvmånene, og velg den `linkage` som skiller dem perfekt. Lagre labelene i `lab`.
```start
from sklearn.datasets import make_moons
from sklearn.cluster import AgglomerativeClustering

X, y = make_moons(n_samples=300, noise=0.05, random_state=0)
lab = ...
```
```losning
from sklearn.datasets import make_moons
from sklearn.cluster import AgglomerativeClustering

X, y = make_moons(n_samples=300, noise=0.05, random_state=0)
lab = AgglomerativeClustering(n_clusters=2, linkage="single").fit_predict(X)
```
```test
import numpy as np
assert lab is not ... and len(lab) == 300, "lab skal være én label per punkt."
assert len(set(lab)) == 2, "Bruk n_clusters=2."
e = max(np.mean(lab == y), np.mean(lab != y))
assert e > 0.99, f"Bare {e:.0%} enighet med de sanne månene. Prøv en linkage som følger lange, tynne former."
```
:::

## Spektral klynging
`SpectralClustering(n_clusters=2, affinity="nearest_neighbors", n_neighbors=15, random_state=0)` kobler hvert punkt til naboene sine i en graf og deler grafen der den har færrest koblinger.

::: oppgave Enighet og spektral klynging
Skriv `enighet(a, b)` som returnerer `max(np.mean(a == b), np.mean(a != b))` (andel like labels uansett hva de to klyngene heter), og lagre spektrale labels for halvmånene i `sc`.
```start
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering

X, y = make_moons(n_samples=300, noise=0.05, random_state=0)

def enighet(a, b):
    ...

sc = ...
```
```losning
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering

X, y = make_moons(n_samples=300, noise=0.05, random_state=0)

def enighet(a, b):
    return max(np.mean(a == b), np.mean(a != b))

sc = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", n_neighbors=15, random_state=0).fit_predict(X)
```
```test
import numpy as np
e1 = enighet(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]))
assert e1 is not None and abs(e1 - 1.0) < 1e-9, "Omdøpte klynger ([0,0,1,1] mot [1,1,0,0]) skal gi enighet 1.0."
assert abs(enighet(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])) - 0.75) < 1e-9, "[0,0,1,1] mot [0,1,1,1] skal gi 0.75."
assert sc is not ... and len(sc) == 300, "sc skal være én label per punkt."
assert enighet(sc, y) > 0.99, "SpectralClustering med affinity='nearest_neighbors' skal skille månene."
```
:::

## Fordelingsbasert: Gaussian Mixture
`GaussianMixture(n_components=3, random_state=0).fit(X)` antar én normalfordeling per klynge, og `predict_proba(X)` gir en sannsynlighet per klynge (myk tilordning).

::: oppgave Myke tilordninger
Tilpass `gm` på `X`, og lagre `predict_proba(X)` i `proba` og `predict(X)` i `lab_gm`.
```start
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

X, _ = make_blobs(n_samples=300, centers=3, random_state=0)
gm = ...
proba = ...
lab_gm = ...
```
```losning
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

X, _ = make_blobs(n_samples=300, centers=3, random_state=0)
gm = GaussianMixture(n_components=3, random_state=0).fit(X)
proba = gm.predict_proba(X)
lab_gm = gm.predict(X)
```
```test
import numpy as np
assert np.asarray(proba).shape == (300, 3), f"proba skal ha én kolonne per komponent, shape (300, 3), fikk {np.asarray(proba).shape}."
assert np.allclose(np.sum(proba, axis=1), 1), "Hver rad i proba skal summere til 1."
assert lab_gm is not ... and np.array_equal(lab_gm, np.argmax(proba, axis=1)), "lab_gm skal være gm.predict(X), som er klyngen med høyest sannsynlighet."
```
:::

--- LEKSJON 10.7 | pca | PCA
Hovedkomponentene er egenvektorene til kovariansmatrisen sortert etter egenverdi (variansen langs hver), og PC1 er retningen med størst varians.

## For hånd
`np.cov(X, rowvar=False)` gir kovariansmatrisen og `np.linalg.eigh(S)` egenverdiene i **stigende** rekkefølge med egenvektorene som kolonner.

::: oppgave Kovarians og egenverdier
For notatenes data `X`, lagre kovariansmatrisen i `S` og egenverdiene (stigende, rett fra `eigh`) i `egenverdier`.
```start
import numpy as np

X = np.array([[1, 2], [2, 1], [3, 0]], dtype=float)
S = ...
egenverdier = ...
```
```losning
import numpy as np

X = np.array([[1, 2], [2, 1], [3, 0]], dtype=float)
S = np.cov(X, rowvar=False)
egenverdier, egenvektorer = np.linalg.eigh(S)
```
```test
import numpy as np
assert np.asarray(S).shape == (2, 2), f"S skal være 2 × 2 (én rad/kolonne per feature), fikk {np.asarray(S).shape}. Husk rowvar=False."
assert np.allclose(S, [[1, -1], [-1, 1]]), "S stemmer ikke. Bruk np.cov(X, rowvar=False)."
assert np.allclose(egenverdier, [0, 2]), f"Egenverdiene skal være 0 og 2, fikk {egenverdier}."
```
:::

::: oppgave PCA for hånd
Sentrer `X`, regn kovariansmatrisen, finn egenverdier og egenvektorer med `eigh`, og sorter synkende. Lagre egenverdiene (synkende) i `verdier`, egenvektorene som **kolonner** i samme rekkefølge i `W`, og projeksjonen på PC1 og PC2 i `Z` (shape `(n, 2)`).
```start
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 3)) @ np.array([[3, 1, 0], [0, 1, 0], [0, 0, 0.2]])

verdier = ...
W = ...
Z = ...
```
```losning
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 3)) @ np.array([[3, 1, 0], [0, 1, 0], [0, 0, 0.2]])

Xc = X - X.mean(axis=0)
S = np.cov(Xc, rowvar=False)
v, vec = np.linalg.eigh(S)
idx = np.argsort(v)[::-1]
verdier = v[idx]
W = vec[:, idx]
Z = Xc @ W[:, :2]
```
```test
import numpy as np
from sklearn.decomposition import PCA
p = PCA(n_components=3).fit(X)
assert verdier is not ... and np.allclose(verdier, p.explained_variance_), "verdier skal være egenverdiene sortert synkende. Husk at eigh gir stigende rekkefølge."
assert np.asarray(W).shape == (3, 3), "W skal ha egenvektorene som kolonner, shape (3, 3)."
assert np.allclose(np.abs(W.T), np.abs(p.components_), atol=1e-6), "Kolonnene i W skal være egenvektorene i samme rekkefølge som verdier."
assert np.asarray(Z).shape == (100, 2), f"Z skal ha shape (100, 2), fikk {np.asarray(Z).shape}."
assert np.allclose(np.abs(Z), np.abs(p.transform(X)[:, :2]), atol=1e-6), "Z stemmer ikke. Husk å sentrere X før du ganger med de to første kolonnene i W."
```
:::

## Med sklearn
`pca = PCA(n_components=2)` og `pca.fit_transform(X)` gir PC-koordinatene, og `explained_variance_ratio_`, `components_` (PC-ene som **rader**) og `mean_` er det modellen lærte.

::: oppgave PCA i sklearn
Kjør PCA med 2 komponenter på `X`, og lagre resultatet i `X_pca`, andelen forklart varians i `andel` og PC1 (en vektor med 2 tall) i `pc1`.
```start
import numpy as np
from sklearn.decomposition import PCA

x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([1.6, 4.2, 5.7, 8.4, 9.2])
X = np.vstack((x1, x2)).T

pca = PCA(n_components=2)
X_pca = ...
andel = ...
pc1 = ...
```
```losning
import numpy as np
from sklearn.decomposition import PCA

x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([1.6, 4.2, 5.7, 8.4, 9.2])
X = np.vstack((x1, x2)).T

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
andel = pca.explained_variance_ratio_
pc1 = pca.components_[0]
```
```test
import numpy as np
f = PCA(n_components=2).fit(X)
assert np.asarray(X_pca).shape == (5, 2), "X_pca skal ha shape (5, 2)."
assert np.allclose(np.abs(X_pca), np.abs(f.transform(X))), "X_pca skal være pca.fit_transform(X)."
assert np.allclose(andel, f.explained_variance_ratio_), "andel skal være explained_variance_ratio_."
assert andel[0] > 0.99, "PC1 skal forklare over 99 % av variansen."
assert np.asarray(pc1).shape == (2,) and np.allclose(np.abs(pc1), np.abs(f.components_[0])), "pc1 skal være første RAD i components_: pca.components_[0]."
```
:::

## Redusere til én dimensjon og tilbake
`PCA(n_components=1)` beholder bare PC1, og `pca.inverse_transform(X_pca)` går tilbake til det opprinnelige rommet; det som mangler er rekonstruksjonsfeilen.

::: oppgave Rekonstruksjon
Reduser `X` til én komponent, gå tilbake med `inverse_transform` (`X_tilbake`, shape `(5, 2)`), og lagre summen av kvadrerte avvik per punkt i `feil`.
```start
import numpy as np
from sklearn.decomposition import PCA

X = np.array([[1, 1.6], [2, 4.2], [3, 5.7], [4, 8.4], [5, 9.2]])
X_tilbake = ...
feil = ...
```
```losning
import numpy as np
from sklearn.decomposition import PCA

X = np.array([[1, 1.6], [2, 4.2], [3, 5.7], [4, 8.4], [5, 9.2]])
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)
X_tilbake = pca.inverse_transform(X_pca)
feil = np.sum((X - X_tilbake)**2, axis=1)
```
```test
import numpy as np
p = PCA(n_components=1).fit(X)
fasit = p.inverse_transform(p.transform(X))
assert np.asarray(X_tilbake).shape == (5, 2), f"X_tilbake skal ha samme shape som X, (5, 2), fikk {np.asarray(X_tilbake).shape}. Bruk inverse_transform."
assert np.allclose(X_tilbake, fasit), "X_tilbake stemmer ikke. Bruk PCA(n_components=1)."
assert np.asarray(feil).shape == (5,) and np.allclose(feil, np.sum((X - fasit)**2, axis=1)), "feil skal være summen av kvadrerte avvik per rad (axis=1)."
```
:::

## Hvor mange komponenter?
`np.cumsum(pca.explained_variance_ratio_)` gir kumulativ forklart varians, og `PCA(n_components=0.95)` velger selv nok komponenter til 95 %.

::: oppgave Komponenter for 95 %
Tilpass `PCA()` på sifrene, lagre den kumulative andelen i `kum` og antall komponenter som trengs for 95 % i `n_95`.
```start
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

X, y = load_digits(return_X_y=True)
kum = ...
n_95 = ...
```
```losning
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

X, y = load_digits(return_X_y=True)
pca = PCA().fit(X)
kum = np.cumsum(pca.explained_variance_ratio_)
n_95 = int(np.argmax(kum >= 0.95) + 1)
```
```test
import numpy as np
assert np.asarray(kum).shape == (64,), "kum skal ha én verdi per komponent (64). Bruk np.cumsum på explained_variance_ratio_."
assert np.isclose(kum[-1], 1.0), "Siste verdi i kum skal være 1."
fasit = PCA(n_components=0.95).fit(X).n_components_
assert n_95 == fasit, f"Forventet {fasit} komponenter, fikk {n_95}. Husk + 1: argmax gir indeksen, ikke antallet."
```
:::

## Skaler, og fit bare på trening
Som alle transformere skal `StandardScaler` og `PCA` bare `fit`-es på `X_train`, og så brukes med `transform` på `X_test`.

::: oppgave Skalering og PCA uten lekkasje
Skaler med `StandardScaler` og reduser til 2 komponenter med `PCA`, begge tilpasset på `X_train`, og lagre resultatene i `Z_train` og `Z_test`.
```start
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X, y = load_wine(return_X_y=True)
X_train, X_test = train_test_split(X, test_size=0.3, random_state=0)
Z_train = ...
Z_test = ...
```
```losning
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X, y = load_wine(return_X_y=True)
X_train, X_test = train_test_split(X, test_size=0.3, random_state=0)
scaler = StandardScaler().fit(X_train)
pca = PCA(n_components=2).fit(scaler.transform(X_train))
Z_train = pca.transform(scaler.transform(X_train))
Z_test = pca.transform(scaler.transform(X_test))
```
```test
import numpy as np
s_ = StandardScaler().fit(X_train)
p_ = PCA(n_components=2).fit(s_.transform(X_train))
assert np.asarray(Z_train).shape == (len(X_train), 2) and np.asarray(Z_test).shape == (len(X_test), 2), "Z_train og Z_test skal ha 2 kolonner."
assert np.allclose(np.abs(Z_train), np.abs(p_.transform(s_.transform(X_train))), atol=1e-6), "Z_train stemmer ikke. Skalerte du før PCA?"
assert np.allclose(np.abs(Z_test), np.abs(p_.transform(s_.transform(X_test))), atol=1e-6), "Z_test stemmer ikke. Bruk transform (ikke fit_transform) på X_test med scaler og PCA tilpasset på X_train."
```
:::

--- LEKSJON 10.8 | tsne | t-SNE
`TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X)` plasserer punktene i 2D slik at naboskap bevares (minimerer KL-divergens), og brukes bare til visualisering.

::: oppgave t-SNE av sifre
Kjør t-SNE som over på de 200 sifrene, lagre resultatet i `X_emb` og `tsne.kl_divergence_` i `kl`, og plott `X_emb` farget med `y` på `ax`.
```start
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

X, y = load_digits(return_X_y=True)
X, y = X[:200], y[:200]
fig, ax = plt.subplots()
X_emb = ...
kl = ...
```
```losning
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

X, y = load_digits(return_X_y=True)
X, y = X[:200], y[:200]
fig, ax = plt.subplots()
tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0)
X_emb = tsne.fit_transform(X)
kl = tsne.kl_divergence_
ax.scatter(X_emb[:, 0], X_emb[:, 1], c=y, cmap="tab10", s=10)
```
```test
import numpy as np
assert np.asarray(X_emb).shape == (200, 2), f"X_emb skal ha shape (200, 2), fikk {np.asarray(X_emb).shape}."
assert kl is not ... and float(kl) > 0, "kl skal være tsne.kl_divergence_ (et positivt tall)."
assert len(ax.collections) >= 1, "Plott X_emb med ax.scatter og c=y."
```
:::

## Perplexity
`perplexity` er omtrent hvor mange naboer hvert punkt tar hensyn til (typisk 5 til 50, og mindre enn antall punkter): lav gir små, oppsplittede grupper og høy mer global struktur.

::: oppgave Lav og høy perplexity
Lag `emb_lav` med `perplexity=5` og `emb_hoy` med `perplexity=40` (ellers `n_components=2, init="pca", random_state=0`) på `X`.
```start
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

X, y = load_digits(return_X_y=True)
X, y = X[:150], y[:150]
emb_lav = ...
emb_hoy = ...
```
```losning
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

X, y = load_digits(return_X_y=True)
X, y = X[:150], y[:150]
emb_lav = TSNE(n_components=2, perplexity=5, init="pca", random_state=0).fit_transform(X)
emb_hoy = TSNE(n_components=2, perplexity=40, init="pca", random_state=0).fit_transform(X)
```
```test
import numpy as np
assert np.asarray(emb_lav).shape == (150, 2) and np.asarray(emb_hoy).shape == (150, 2), "Begge embeddingene skal ha shape (150, 2)."
f5 = TSNE(n_components=2, perplexity=5, init="pca", random_state=0).fit_transform(X)
assert np.allclose(emb_lav, f5, atol=1e-3), "emb_lav skal bruke perplexity=5, init='pca' og random_state=0."
assert not np.allclose(emb_hoy, f5, atol=1e-3), "emb_hoy skal bruke perplexity=40."
```
:::

## t-SNE kan ikke brukes på nye punkter
t-SNE optimerer posisjonene til punktene den fikk og har ingen `transform`, så nye punkter krever at du kjører `fit_transform` på alt på nytt (PCA har `transform`).

::: oppgave Nye punkter
`PCA` er tilpasset på `X_gml`. Lagre PCA-koordinatene til `X_ny` i `pca_ny`, og t-SNE-koordinatene til de nye punktene i `tsne_ny` ved å kjøre t-SNE (`perplexity=20, init="pca", random_state=0`) på `np.vstack([X_gml, X_ny])` og ta de 10 siste radene.
```start
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

X, _ = load_digits(return_X_y=True)
X_gml, X_ny = X[:100], X[100:110]
pca = PCA(n_components=2).fit(X_gml)

pca_ny = ...
tsne_ny = ...
```
```losning
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

X, _ = load_digits(return_X_y=True)
X_gml, X_ny = X[:100], X[100:110]
pca = PCA(n_components=2).fit(X_gml)

pca_ny = pca.transform(X_ny)
alle = TSNE(n_components=2, perplexity=20, init="pca", random_state=0).fit_transform(np.vstack([X_gml, X_ny]))
tsne_ny = alle[-10:]
```
```test
import numpy as np
assert np.asarray(pca_ny).shape == (10, 2) and np.allclose(pca_ny, pca.transform(X_ny)), "pca_ny skal være pca.transform(X_ny), uten å tilpasse på nytt."
assert np.asarray(tsne_ny).shape == (10, 2), f"tsne_ny skal ha shape (10, 2), fikk {np.asarray(tsne_ny).shape}. Ta de 10 siste radene."
fasit = TSNE(n_components=2, perplexity=20, init="pca", random_state=0).fit_transform(np.vstack([X_gml, X_ny]))[-10:]
assert np.allclose(tsne_ny, fasit, atol=1e-3), "Kjør TSNE.fit_transform på np.vstack([X_gml, X_ny]) (gamle først) og ta [-10:]."
```
:::

## PCA mot t-SNE
Vanlig rørledning er PCA først (til 30 til 50 dimensjoner, raskt og fjerner støy) og så t-SNE til 2D.

::: oppgave PCA før t-SNE
Reduser `X` til 30 dimensjoner med `PCA(n_components=30, random_state=0)`, og kjør deretter `TSNE(n_components=2, perplexity=30, init="pca", random_state=0)` på resultatet. Lagre PCA-resultatet i `X_pca` og t-SNE-resultatet i `X_emb`.
```start
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

X, y = load_digits(return_X_y=True)
X = X[:300]

X_pca = ...
X_emb = ...
```
```losning
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

X, y = load_digits(return_X_y=True)
X = X[:300]

X_pca = PCA(n_components=30, random_state=0).fit_transform(X)
X_emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X_pca)
```
```test
import numpy as np
assert X_pca is not ... and np.asarray(X_pca).shape == (300, 30), "X_pca skal ha shape (300, 30)."
assert X_emb is not ... and np.asarray(X_emb).shape == (300, 2), "X_emb skal ha shape (300, 2). Kjørte du t-SNE på X_pca?"
```
:::

--- LEKSJON 10.9 | anomalideteksjon | Anomalideteksjon
Hver metode gir en score per punkt, og en terskel som `np.percentile(score, 95)` avgjør hvilke punkter som er anomalier.

## z-score (én feature)
`z = (x - x.mean()) / x.std()` sier hvor mange standardavvik et punkt er fra snittet, og `|z| > 3` er en vanlig grense.

::: oppgave z-score
Regn ut `z` for `x`, og lagre indeksene der `|z| > 3` i `avvik`.
```start
import numpy as np

rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(50, 5, 200), [95, 3]])
z = ...
avvik = ...
```
```losning
import numpy as np

rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(50, 5, 200), [95, 3]])
z = (x - x.mean()) / x.std()
avvik = np.where(np.abs(z) > 3)[0]
```
```test
import numpy as np
assert np.asarray(z).shape == (202,) and np.allclose(z, (x - x.mean()) / x.std()), "z skal være (x - x.mean()) / x.std()."
assert avvik is not ... and sorted(np.asarray(avvik).tolist()) == [200, 201], f"De innplantede verdiene på indeks 200 og 201 skal være avvik, fikk {avvik}. Husk np.abs."
```
:::

## Avstand til nærmeste sentroide (k-means)
`kmeans.transform(X_test).min(axis=1)` gir avstanden til nærmeste sentroide, og en 95-persentil merker alltid 5 % av punktene, også normale (falske positive).

::: oppgave k-means som anomalidetektor
Tilpass `KMeans(n_clusters=3, n_init=10, random_state=0)` på `X_train`, lagre avstanden fra hvert testpunkt til nærmeste sentroide i `distances`, 95-persentilen i `threshold` og den boolske masken `outliers`. Lagre også antall falske positive (merkede blant de 100 normale første radene) i `n_falske`.
```start
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X_train, _ = make_blobs(n_samples=400, centers=3, random_state=0)
X_test, _ = make_blobs(n_samples=100, centers=3, random_state=0)
X_test = np.vstack([X_test, [[8, 8], [-6, -6]]])   # rad 100 og 101 er anomalier

distances = ...
threshold = ...
outliers = ...
n_falske = ...
```
```losning
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X_train, _ = make_blobs(n_samples=400, centers=3, random_state=0)
X_test, _ = make_blobs(n_samples=100, centers=3, random_state=0)
X_test = np.vstack([X_test, [[8, 8], [-6, -6]]])   # rad 100 og 101 er anomalier

kmeans = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X_train)
distances = kmeans.transform(X_test).min(axis=1)
threshold = np.percentile(distances, 95)
outliers = distances > threshold
n_falske = int(np.sum(outliers[:100]))
```
```test
import numpy as np
km_ = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X_train)
d_ = km_.transform(X_test).min(axis=1)
assert np.asarray(distances).shape == (102,) and np.allclose(distances, d_), "distances skal være kmeans.transform(X_test).min(axis=1), med KMeans tilpasset på X_train."
assert threshold is not ... and np.isclose(threshold, np.percentile(d_, 95)), "threshold skal være np.percentile(distances, 95)."
assert np.asarray(outliers).dtype == bool and np.array_equal(outliers, d_ > threshold), "outliers skal være distances > threshold."
assert outliers[100] and outliers[101], "De to innplantede anomaliene skal være med."
assert n_falske == int(np.sum(d_[:100] > threshold)), "n_falske er antall True i outliers[:100]."
```
:::

## Isolation Forest, LOF og DBSCAN
`IsolationForest(contamination=0.01, random_state=0)` og `LocalOutlierFactor(n_neighbors=20)` gir `-1` for anomalier og `1` for normale fra `predict`/`fit_predict`, og `lof.negative_outlier_factor_` er mer negativ jo mer anomalt; i DBSCAN er støy (`labels_ == -1`) anomaliene.

::: oppgave Isolation Forest
Tilpass `iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=0)` på `X`, og lagre indeksene der `iso.predict(X)` er `-1` i `iso_anom`.
```start
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest

X, _ = make_blobs(n_samples=300, centers=2, cluster_std=0.8, random_state=1)
X = np.vstack([X, [[0, 10], [-12, 0], [5, -2]]])   # rad 300, 301, 302

iso = ...
iso_anom = ...
```
```losning
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest

X, _ = make_blobs(n_samples=300, centers=2, cluster_std=0.8, random_state=1)
X = np.vstack([X, [[0, 10], [-12, 0], [5, -2]]])   # rad 300, 301, 302

iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=0).fit(X)
iso_anom = np.where(iso.predict(X) == -1)[0]
```
```test
import numpy as np
assert isinstance(iso, IsolationForest) and hasattr(iso, "estimators_"), "iso skal være en tilpasset IsolationForest (husk .fit(X))."
assert iso_anom is not ..., "Lag iso_anom med np.where(iso.predict(X) == -1)[0]."
a = set(np.asarray(iso_anom).tolist())
assert {300, 301, 302} <= a, f"De innplantede radene 300, 301 og 302 skal være med, fikk {sorted(a)}."
assert len(a) <= 5, f"contamination=0.01 skal gi omtrent 1 % anomalier, fikk {len(a)}."
```
:::

::: oppgave LOF og DBSCAN
Lagre de 3 radene med mest negativ `negative_outlier_factor_` fra `LocalOutlierFactor(n_neighbors=20)` i `lof_topp3` (med `np.argsort`), og støypunktene fra `DBSCAN(eps=1.0, min_samples=5)` i `db_stoy`.
```start
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

X, _ = make_blobs(n_samples=300, centers=2, cluster_std=0.8, random_state=1)
X = np.vstack([X, [[0, 10], [-12, 0], [5, -2]]])

lof_topp3 = ...
db_stoy = ...
```
```losning
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

X, _ = make_blobs(n_samples=300, centers=2, cluster_std=0.8, random_state=1)
X = np.vstack([X, [[0, 10], [-12, 0], [5, -2]]])

lof = LocalOutlierFactor(n_neighbors=20)
lof.fit_predict(X)
lof_topp3 = np.argsort(lof.negative_outlier_factor_)[:3]
db_stoy = np.where(DBSCAN(eps=1.0, min_samples=5).fit(X).labels_ == -1)[0]
```
```test
import numpy as np
assert lof_topp3 is not ... and sorted(np.asarray(lof_topp3).tolist()) == [300, 301, 302], f"De tre mest negative skal være radene 300, 301 og 302, fikk {lof_topp3}. Sorter stigende og ta de 3 første."
fasit = np.where(DBSCAN(eps=1.0, min_samples=5).fit(X).labels_ == -1)[0]
assert db_stoy is not ... and sorted(np.asarray(db_stoy).tolist()) == fasit.tolist(), "db_stoy skal være indeksene der labels_ == -1."
```
:::

## Rekonstruksjonsfeil med PCA
`X_rec = pca.inverse_transform(pca.transform(X))` og `np.sum((X - X_rec)**2, axis=1)` gir stor feil for rader som ikke passer strukturen de andre deler.

::: oppgave Rekonstruksjonsfeil og persentil
Tilpass `PCA(n_components=2)` på `X`, regn ut `recon_error` (sum av kvadrerte avvik per rad), og lagre indeksene til radene med feil **over 99-persentilen** i `anomalier` (bruk `np.where`).
```start
import numpy as np
from sklearn.decomposition import PCA

rng = np.random.default_rng(0)
t = rng.normal(size=(500, 2))
X = t @ rng.normal(size=(2, 6)) + 0.05 * rng.normal(size=(500, 6))
X[[17, 230, 444]] += rng.normal(0, 3, size=(3, 6))

recon_error = ...
anomalier = ...
```
```losning
import numpy as np
from sklearn.decomposition import PCA

rng = np.random.default_rng(0)
t = rng.normal(size=(500, 2))
X = t @ rng.normal(size=(2, 6)) + 0.05 * rng.normal(size=(500, 6))
X[[17, 230, 444]] += rng.normal(0, 3, size=(3, 6))

pca = PCA(n_components=2).fit(X)
X_rec = pca.inverse_transform(pca.transform(X))
recon_error = np.sum((X - X_rec)**2, axis=1)
anomalier = np.where(recon_error > np.percentile(recon_error, 99))[0]
```
```test
import numpy as np
assert recon_error is not ... and np.asarray(recon_error).shape == (500,), "recon_error skal ha én verdi per rad, shape (500,). Summer langs axis=1."
assert anomalier is not ..., "Lag anomalier med np.where."
a = sorted(np.asarray(anomalier).tolist())
assert len(a) == 5, f"Over 99-persentilen av 500 rader skal gi 5 rader, fikk {len(a)}."
assert {17, 230, 444} <= set(a), f"De innplantede radene 17, 230 og 444 skal være med, fikk {a}."
```
:::

--- LEKSJON 10.10 | prosjekt-3 | Prosjekt 3-teknikker
Prosjekt 3 var en stor CSV uten labels med stokkede pikselkolonner, og her øver vi de samme teknikkene på `load_digits`.

## Klyngenøyaktighet som ikke bryr seg om navn
Klyngelabels er bare navn, så `c = contingency_matrix(true, pred)` og `r, k = linear_sum_assignment(-c)` finner beste kobling mellom klynger og klasser (minus fordi den minimerer), og `c[r, k].sum() / n` er nøyaktigheten.

::: oppgave Kontingenstabell
Lagre kontingenstabellen for `true_labels` og `cluster_labels` i `c`, og antall punkter fra klasse 0 som havnet i klynge 1 i `n_0_i_1`.
```start
import numpy as np
from sklearn.metrics.cluster import contingency_matrix

true_labels = np.array([0, 0, 0, 1, 1, 2, 2, 2])
cluster_labels = np.array([1, 1, 2, 0, 0, 2, 2, 1])
c = ...
n_0_i_1 = ...
```
```losning
import numpy as np
from sklearn.metrics.cluster import contingency_matrix

true_labels = np.array([0, 0, 0, 1, 1, 2, 2, 2])
cluster_labels = np.array([1, 1, 2, 0, 0, 2, 2, 1])
c = contingency_matrix(true_labels, cluster_labels)
n_0_i_1 = c[0, 1]
```
```test
import numpy as np
assert np.asarray(c).shape == (3, 3), "c skal være 3 × 3 (rad = sann klasse, kolonne = klynge)."
assert np.array_equal(c, contingency_matrix(true_labels, cluster_labels)), "Bruk contingency_matrix(true_labels, cluster_labels) i den rekkefølgen."
assert n_0_i_1 == 2, f"To punkter fra klasse 0 havnet i klynge 1 (c[0, 1]), fikk {n_0_i_1}."
```
:::

::: oppgave Skriv calculate_clustering_accuracy
Skriv `calculate_clustering_accuracy(true_labels, cluster_labels)` som returnerer andelen riktige etter beste omdøping av klyngene.
```start
import numpy as np
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment

def calculate_clustering_accuracy(true_labels, cluster_labels):
    ...
```
```losning
import numpy as np
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment

def calculate_clustering_accuracy(true_labels, cluster_labels):
    c = contingency_matrix(true_labels, cluster_labels)
    r, k = linear_sum_assignment(-c)
    return c[r, k].sum() / len(true_labels)
```
```test
import numpy as np
f = calculate_clustering_accuracy
assert f(np.array([0, 1, 2, 0, 1, 2]), np.array([1, 2, 0, 1, 2, 0])) is not None, "Funksjonen må returnere et tall."
assert abs(f(np.array([0, 1, 2, 0, 1, 2]), np.array([1, 2, 0, 1, 2, 0])) - 1.0) < 1e-9, "Omdøpte men like klynger skal gi 1.0."
assert abs(f(np.array([1, 1, 1, 2, 2, 2]), np.array([0, 0, 0, 0, 0, 0])) - 0.5) < 1e-9, "Alt i én klynge med to like store klasser skal gi 0.5."
assert abs(f(np.array([0, 0, 1, 1, 1]), np.array([5, 5, 5, 7, 7])) - 0.8) < 1e-9, "[0,0,1,1,1] mot [5,5,5,7,7] skal gi 0.8. Husk minus foran kontingenstabellen (vi maksimerer)."
```
:::

## Bilder lagret som rader
`X.reshape(-1, 8, 8)` gjør hver rad med 64 piksler om til et 8 × 8-bilde (`-1` betyr «regn ut selv»), og `ax.imshow(bilde, cmap="gray")` viser det.

::: oppgave Fra rader til bilder
Lag `images` med shape `(1797, 8, 8)` og vis `images[0]` med `ax.imshow`, og lagre sidelengden til et kvadratisk bilde med 16 384 piksler (som i prosjektet) i `side`.
```start
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
fig, ax = plt.subplots()
images = ...
side = ...
```
```losning
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
fig, ax = plt.subplots()
images = X.reshape(-1, 8, 8)
ax.imshow(images[0], cmap="gray")
side = int(np.sqrt(16384))
```
```test
import numpy as np
assert np.asarray(images).shape == (1797, 8, 8), f"images skal ha shape (1797, 8, 8), fikk {np.asarray(images).shape}. Bruk X.reshape(-1, 8, 8)."
assert np.array_equal(images[0], X[0].reshape(8, 8)), "images[0] skal være første rad omformet til 8 × 8."
assert len(ax.images) >= 1, "Vis images[0] med ax.imshow."
assert side == 128, f"16 384 = 128 · 128, så side skal være 128, fikk {side}."
```
:::

## Gjenopprette stokkede kolonner
Når kolonnenavnene er `col_<opprinnelig indeks>`, gir `order = np.argsort(indekser)` rekkefølgen som setter kolonnene tilbake, og `pixels = df.to_numpy()[:, order]` gjør det.

::: oppgave Sett pikslene på plass
Hent den opprinnelige indeksen fra hvert kolonnenavn i `df` (`int(c.split("_")[1])`), og lagre de gjenopprettede pikslene som numpy-array i `pixels`.
```start
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

X, _ = load_digits(return_X_y=True)
perm = np.random.default_rng(0).permutation(64)
df = pd.DataFrame(X[:, perm], columns=[f"col_{p}" for p in perm])

pixels = ...
```
```losning
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

X, _ = load_digits(return_X_y=True)
perm = np.random.default_rng(0).permutation(64)
df = pd.DataFrame(X[:, perm], columns=[f"col_{p}" for p in perm])

idx = [int(c.split("_")[1]) for c in df.columns]
order = np.argsort(idx)
pixels = df.to_numpy()[:, order]
```
```test
import numpy as np
assert np.asarray(pixels).shape == (1797, 64), f"pixels skal ha shape (1797, 64), fikk {np.asarray(pixels).shape}. Ordner du kolonnene (andre akse)?"
assert np.array_equal(pixels, X), "Kolonnene står ikke i riktig rekkefølge. Bruk np.argsort på indeksene (ikke np.sort)."
```
:::

## Rørledningen: PCA, klynging, t-SNE
Den typiske flyten er PCA først, så klynging i PCA-rommet, og t-SNE bare til å se på klyngene.

::: oppgave Hele rørledningen
Reduser sifrene til 30 dimensjoner med `PCA(n_components=30, random_state=0)` (`Z`), klyng med `KMeans(n_clusters=10, n_init=10, random_state=0)` (`km`), og lagre klyngenøyaktigheten i `acc`.
```start
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment

def calculate_clustering_accuracy(true_labels, cluster_labels):
    c = contingency_matrix(true_labels, cluster_labels)
    r, k = linear_sum_assignment(-c)
    return c[r, k].sum() / len(true_labels)

X, y = load_digits(return_X_y=True)
Z = ...
km = ...
acc = ...
```
```losning
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment

def calculate_clustering_accuracy(true_labels, cluster_labels):
    c = contingency_matrix(true_labels, cluster_labels)
    r, k = linear_sum_assignment(-c)
    return c[r, k].sum() / len(true_labels)

X, y = load_digits(return_X_y=True)
Z = PCA(n_components=30, random_state=0).fit_transform(X)
km = KMeans(n_clusters=10, n_init=10, random_state=0).fit_predict(Z)
acc = calculate_clustering_accuracy(y, km)
```
```test
import numpy as np
assert np.asarray(Z).shape == (1797, 30), "Z skal ha shape (1797, 30)."
assert km is not ... and np.asarray(km).shape == (1797,) and len(set(km)) == 10, "km skal være 10 klyngelabels per punkt fra KMeans på Z."
assert acc is not ... and np.isclose(acc, calculate_clustering_accuracy(y, km)), "acc skal være calculate_clustering_accuracy(y, km)."
assert acc > 0.6, f"Nøyaktigheten bør være over 0.6, fikk {acc:.3f}. Klynget du på Z?"
```
:::

## Lagre en innlevering
`submission = pd.DataFrame({"ID": ..., "cluster": ...})` og `submission.to_csv("submission.csv", index=False)` lager Kaggle-filen.

::: oppgave Anomalier som egen klynge
Lag `final_labels` der anomaliene får sin egen klynge `labels.max() + 1` (uten å endre `labels`), og `submission` med kolonnene `ID` (0, 1, 2, …) og `cluster`. Lagre den til `submission.csv` uten indeks.
```start
import numpy as np
import pandas as pd

labels = np.array([0, 2, 1, 1, 0, 2])
anomalies = np.array([3])
final_labels = ...
submission = ...
```
```losning
import numpy as np
import pandas as pd

labels = np.array([0, 2, 1, 1, 0, 2])
anomalies = np.array([3])
final_labels = labels.copy()
final_labels[anomalies] = labels.max() + 1
submission = pd.DataFrame({"ID": np.arange(len(labels)), "cluster": final_labels})
submission.to_csv("submission.csv", index=False)
```
```test
import numpy as np
import pandas as pd
assert list(labels) == [0, 2, 1, 1, 0, 2], "labels er endret. Bruk labels.copy() før du endrer."
assert final_labels is not ... and list(final_labels) == [0, 2, 1, 3, 0, 2], f"Rad 3 skal få klynge 3 (labels.max() + 1), fikk {list(final_labels)}."
assert isinstance(submission, pd.DataFrame) and list(submission.columns) == ["ID", "cluster"], "submission skal ha kolonnene ID og cluster."
assert list(submission["ID"]) == list(range(6)) and list(submission["cluster"]) == list(final_labels), "ID skal være 0 til 5 og cluster final_labels."
lest = pd.read_csv("submission.csv")
assert list(lest.columns) == ["ID", "cluster"], "Lagre med submission.to_csv('submission.csv', index=False)."
```
:::
