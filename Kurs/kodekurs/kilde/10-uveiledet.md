=== DEL 10 | uveiledet | Uveiledet læring (prosjekt 3)
Data uten labels: avstander i høye dimensjoner, k-means for hånd og i sklearn, silhouette, DBSCAN, PCA og t-SNE, anomalideteksjon, og teknikkene fra prosjekt 3 (klyngenøyaktighet, bilder som rader og hele rørledningen).

--- LEKSJON 10.1 | avstander | Avstander og dimensjonalitetens forbannelse
Nesten all uveiledet læring bygger på **avstand**: punkter som er nære hverandre, hører sammen. Den vanligste er den euklidske avstanden $d(p, q) = \sqrt{\sum_i (p_i - q_i)^2}$, som er `np.linalg.norm(p - q)`.

```python
import numpy as np

p = np.array([1.0, 2.0])
q = np.array([4.0, 6.0])
print(np.linalg.norm(p - q))            # sqrt(3² + 4²) = 5

# Avstand fra ett punkt til mange punkter på en gang
X = np.array([[0, 0], [1, 1], [4, 6], [10, 0]])
print(np.linalg.norm(X - p, axis=1))
```

Med broadcasting (leksjon 2.5) får vi **alle parvise avstander** i én linje. `X[:, None, :] - X[None, :, :]` har shape `(n, n, d)`, og normen langs siste akse gir en `(n, n)`-matrise. `scipy.spatial.distance.cdist` gjør det samme.

```python
import numpy as np
from scipy.spatial.distance import cdist

X = np.array([[0, 0], [3, 4], [6, 8]])
D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
display(D)                                # avstandsmatrisen som rutenett
print(np.allclose(D, cdist(X, X)))
```

`D[i, j]` er avstanden fra punkt `i` til punkt `j`. Diagonalen er 0 (avstand til seg selv), og matrisen er symmetrisk, fordi `D[i, j] == D[j, i]`.

## Avstander konsentrerer seg i høye dimensjoner
Notatene trekker to tilfeldige punkter i enhetskuben $[0, 1]^d$ og måler avstanden mellom dem, mange ganger, for økende $d$. Notatenes `get_cube_dist(dim)` gjør ett trekk om gangen med en Python-løkke. Med numpy trekker vi alle på en gang:

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

def get_cube_dists(dim, n_iter=200):
    P1 = rng.uniform(size=(n_iter, dim))
    P2 = rng.uniform(size=(n_iter, dim))
    return np.linalg.norm(P1 - P2, axis=1)

dims = [2, 3, 10, 100, 1000, 10000]
fig, axs = plt.subplots(2, 3, figsize=(11, 5))
for ax, dim in zip(axs.flat, dims):
    dists = get_cube_dists(dim)
    ax.hist(dists, bins=20, label=f"d={dim}")
    ax.set_xlim(left=0)
    ax.legend()
    print(f"d={dim:5d}: snitt {dists.mean():6.2f}, std {dists.std():.2f}, std/snitt {dists.std() / dists.mean():.3f}")
plt.tight_layout()
plt.show()
```

Snittet vokser (som $\sqrt{d/6}$), men spredningen gjør det ikke. **Relativt** sett blir alle avstander nesten like: i 10 000 dimensjoner er «nærmeste» og «fjerneste» punkt nesten like langt unna. Det kalles *concentration of distances*, og er grunnen til at k-means og DBSCAN sliter i høye dimensjoner. Flere datapunkter hjelper ikke. Løsningen er å redusere dimensjonen først (PCA, leksjon 10.7).

::: advarsel Skalering betyr noe for avstander
Trær bryr seg ikke om skala (leksjon 8.1), men avstander gjør det. Måles én feature i kroner og en annen i år, dominerer kronene avstanden fullstendig. Kjør derfor `StandardScaler` før k-means, DBSCAN og PCA når featurene har ulike enheter.
:::

::: oppgave Relativ spredning
Bruk funksjonen `get_cube_dists` (gitt) til å trekke avstander for `dim = 2` og `dim = 1000`. Lagre `std / mean` for hver i `rel_2` og `rel_1000`.
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

::: quiz
Hva skjer med avstanden mellom tilfeldige punkter i $[0,1]^d$ når $d$ blir stor?
- [ ] Både snitt og spredning går mot null
- [x] Snittet vokser, men spredningen relativt til snittet krymper, så alle avstander blir omtrent like
- [ ] Avstandene blir mer spredt, så klynger blir lettere å se
- [ ] Ingenting, avstand er uavhengig av dimensjon
? Snittet går som $\sqrt{d/6}$ mens standardavviket holder seg nesten konstant. Da blir forskjellen mellom nær og fjern liten, og avstandsbaserte metoder mister kraft.
:::

::: quiz
Hva er shape på `D`?
```kode
X = np.random.rand(50, 3)
C = np.random.rand(4, 3)
D = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
```
- [ ] `(50, 3)`
- [ ] `(4, 50)`
- [x] `(50, 4)`
- [ ] `(50, 4, 3)`
? Differansen har shape `(50, 4, 3)`. Normen langs `axis=2` fjerner den siste aksen, så vi får avstanden fra hvert av de 50 punktene til hvert av de 4 sentrene.
:::

::: oppsummering
- Euklidsk avstand: `np.linalg.norm(p - q)`. Mange på en gang: `np.linalg.norm(X - p, axis=1)`.
- Alle parvise avstander: `np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)` eller `cdist(X, Y)`.
- I høye dimensjoner blir alle avstander nesten like (concentration of distances). Flere data hjelper ikke, dimensjonsreduksjon gjør.
- Avstandsbaserte metoder krever skalerte features.
:::

--- LEKSJON 10.2 | kmeans-for-hand | K-means for hånd
**K-means** deler dataene i $k$ klynger. Hver klynge har et sentrum, en **sentroide**. Algoritmen fra notatene:

1. Velg $k$ og plasser $k$ sentroider (for eksempel på $k$ tilfeldige datapunkter).
2. **Tilordning:** gi hvert punkt label til nærmeste sentroide.
3. **Oppdatering:** flytt hver sentroide til snittet (`np.mean`) av punktene i klyngen.
4. Gjenta 2 og 3 til sentroidene ikke endrer seg.

Vi bygger det steg for steg med numpy. Først ett steg:

```python
import numpy as np
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=3, random_state=0)
k = 3
rng = np.random.default_rng(1)
C = X[rng.choice(len(X), size=k, replace=False)]      # steg 1: k tilfeldige punkter

D = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)  # shape (300, 3)
print("Avstand fra de 5 første punktene til hver sentroide:")
display(D[:5])
labels = np.argmin(D, axis=1)                               # steg 2: nærmeste sentroide
print("Klyngestørrelser:", np.bincount(labels, minlength=k))

C_ny = np.array([X[labels == j].mean(axis=0) for j in range(k)])  # steg 3: snitt per klynge
print("Gamle sentroider:\n", C.round(2))
print("Nye sentroider:\n", C_ny.round(2))

# steg 2 en gang til, nå med de nye sentroidene
labels_ny = np.argmin(np.linalg.norm(X[:, None, :] - C_ny[None, :, :], axis=2), axis=1)
print("Labels for de 20 første punktene, før og etter én iterasjon:")
display(labels[:20])
display(labels_ny[:20])
print("Punkter som byttet klynge:", np.sum(labels != labels_ny))
```

I hver rad av `D` er det den minste avstanden som bestemmer labelen. Sammenlign de to label-radene: når sentroidene flytter seg, bytter noen punkter klynge. K-means gjentar dette til ingen punkter bytter lenger.

Så hele algoritmen som en funksjon, med en løkke som stopper når sentroidene står stille:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def kmeans(X, k, seed=0, maks_iter=100):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=k, replace=False)]
    for it in range(maks_iter):
        D = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        labels = np.argmin(D, axis=1)
        C_ny = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        if np.allclose(C_ny, C):
            break
        C = C_ny
    inertia = np.sum((X - C[labels])**2)
    return labels, C, inertia, it + 1

X, _ = make_blobs(n_samples=300, centers=3, random_state=0)
labels, C, inertia, n_iter = kmeans(X, k=3)
print(f"Ferdig etter {n_iter} iterasjoner, inertia {inertia:.1f}")

plt.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap="viridis")
plt.scatter(C[:, 0], C[:, 1], c="red", marker="X", s=200)
plt.show()
```

`C[labels]` er et smart triks: det gir sentroiden til hvert punkt, med samme shape som `X`. Da er **inertia** (summen av kvadrerte avstander til egen sentroide) bare én linje. Det er denne størrelsen k-means minimerer.

## Initialiseringen betyr noe
K-means finner et *lokalt* minimum. Med uheldige startpunkter kan to sentroider havne i samme klynge. Prøv flere seeds:

```python
import numpy as np
from sklearn.datasets import make_blobs

def kmeans(X, k, seed=0, maks_iter=100):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=k, replace=False)]
    for it in range(maks_iter):
        labels = np.argmin(np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2), axis=1)
        C_ny = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        if np.allclose(C_ny, C):
            break
        C = C_ny
    return labels, C, np.sum((X - C[labels])**2)

X, _ = make_blobs(n_samples=500, centers=6, cluster_std=0.6, random_state=4)
for seed in range(8):
    _, _, inertia = kmeans(X, k=6, seed=seed)
    print(f"seed {seed}: inertia {inertia:8.1f}")
```

Noen seeds gir tydelig høyere inertia: de har satt seg fast i en dårlig løsning. Derfor kjører sklearn k-means flere ganger (`n_init`) og beholder den beste, og bruker den smartere startmetoden **k-means++**: neste sentroide trekkes med sannsynlighet proporsjonal med kvadrert avstand til nærmeste sentroide som allerede er valgt.

::: oppgave Tilordning og oppdatering
Skriv de to stegene i k-means som funksjoner:
- `tilordne(X, C)` returnerer et heltallsarray med indeksen til nærmeste sentroide for hvert punkt (shape `(n,)`).
- `oppdater(X, labels, k)` returnerer et array med shape `(k, d)` der rad `j` er snittet av punktene med label `j`.
```start
import numpy as np

def tilordne(X, C):
    ...

def oppdater(X, labels, k):
    ...
```
```losning
import numpy as np

def tilordne(X, C):
    D = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
    return np.argmin(D, axis=1)

def oppdater(X, labels, k):
    return np.array([X[labels == j].mean(axis=0) for j in range(k)])
```
```test
import numpy as np
X_ = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0], [5.0, 0.0]])
C_ = np.array([[0.0, 0.0], [10.0, 10.0]])
lab = tilordne(X_, C_)
assert lab is not None, "tilordne må returnere noe."
assert np.asarray(lab).shape == (5,), f"tilordne skal returnere shape (5,), fikk {np.asarray(lab).shape}. Bruk argmin langs riktig akse."
assert list(lab) == [0, 0, 1, 1, 0], f"Feil tilordning: {list(lab)}. Hvert punkt skal få indeksen til nærmeste sentroide."
C2 = oppdater(X_, np.array([0, 0, 1, 1, 0]), 2)
assert C2 is not None and np.asarray(C2).shape == (2, 2), "oppdater skal returnere shape (k, d) = (2, 2)."
assert np.allclose(C2, [[5/3, 1/3], [10.0, 10.5]]), "Hver rad skal være snittet av punktene i klyngen (mean med axis=0)."
```
:::

::: quiz
Hva gjør linjen under i k-means?
```kode
labels = np.argmin(D, axis=1)
```
der `D` har shape `(n, k)` og `D[i, j]` er avstanden fra punkt `i` til sentroide `j`.
- [ ] Finner punktet som ligger nærmest hver sentroide
- [x] Gir hvert punkt indeksen til sentroiden som ligger nærmest
- [ ] Finner den minste avstanden for hvert punkt
- [ ] Flytter sentroidene til snittet av klyngen
? `argmin` langs `axis=1` går over kolonnene (sentroidene) for hver rad (punkt), og returnerer *indeksen*, ikke selve avstanden (det ville vært `np.min`).
:::

::: oppsummering
- K-means: plasser sentroider, tilordne til nærmeste (`argmin` over avstander), oppdater til snittet, gjenta til ingen endring.
- `C[labels]` gir hvert punkts sentroide. Inertia = `np.sum((X - C[labels])**2)`, det k-means minimerer.
- Resultatet avhenger av startpunktene. Løsning: flere kjøringer (`n_init`) og k-means++.
:::

--- LEKSJON 10.3 | kmeans-sklearn | KMeans i sklearn
I praksis bruker vi `KMeans`. `n_init=10` kjører algoritmen ti ganger med ulike startpunkter og beholder den med lavest inertia. Standard initialisering er `"k-means++"`.

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=300, centers=3, random_state=0)
km = KMeans(n_clusters=3, n_init=10, random_state=0)
labels = km.fit_predict(X)

print("Første labels:", labels[:10])
print("Samme som labels_:", np.array_equal(labels, km.labels_))
print("Sentroider:\n", km.cluster_centers_.round(2))
print("Inertia:", round(km.inertia_, 1))

nye = np.array([[0, 4], [2, 0]])
print("Nye punkter havner i klynge:", km.predict(nye))
print("Avstand til hver sentroide:\n", km.transform(nye).round(2))
```

Under `km` i variabelpanelet ser du det modellen lærte: `labels_`, `cluster_centers_`, `inertia_` og `n_iter_`.

- `fit_predict(X)` = `fit(X)` og så `labels_`.
- `cluster_centers_`: sentroidene, shape `(k, d)`.
- `inertia_`: summen av kvadrerte avstander til egen sentroide.
- `predict(X_ny)`: nærmeste sentroide for nye punkter. `transform(X_ny)`: avstand til alle sentroidene (brukes til anomalideteksjon i leksjon 10.9).

::: husk Labels er bare navn
Klynge 0, 1 og 2 har ingen bestemt rekkefølge. Kjører du med en annen `random_state`, kan klyngen som het 0 nå hete 2. Derfor kan vi ikke sammenligne klyngelabels direkte med sanne labels (se leksjon 10.10).
:::

## Albuemetoden
Inertia faller **alltid** når $k$ øker (med $k = n$ er den 0). Vi kan derfor ikke velge $k$ med lavest inertia. I stedet ser vi etter en **albue**: punktet der kurven slutter å falle bratt.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=500, centers=4, random_state=3)
ks = range(1, 10)
inertias = [KMeans(n_clusters=k, n_init=10, random_state=0).fit(X).inertia_ for k in ks]

plt.plot(ks, inertias, "o-")
plt.xlabel("k")
plt.ylabel("inertia")
plt.title("Albuemetoden")
plt.show()
```

Albuen er ofte uklar. Silhouette (neste leksjon) gir et tall vi kan maksimere.

## Skaler først
Her har den ene featuren mye større skala. Uten skalering deler k-means bare langs den aksen:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X, y = make_blobs(n_samples=300, centers=[[0, 0], [0, 5]], random_state=0)
X[:, 0] = X[:, 0] * 100          # feature 0 i en "større enhet"

uten = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
med = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(StandardScaler().fit_transform(X))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(X[:, 0], X[:, 1], c=uten, s=10)
ax[0].set_title("Uten skalering")
ax[1].scatter(X[:, 0], X[:, 1], c=med, s=10)
ax[1].set_title("Med StandardScaler")
plt.show()
```

::: tips
For store datasett finnes `MiniBatchKMeans`, som notatene anbefaler. Den oppdaterer sentroidene fra små tilfeldige utvalg i stedet for alle punktene hver gang, akkurat som mini-batch gradient descent.
:::

::: oppgave Albuekurven
Lag listen `inertias` med `inertia_` for `KMeans(n_clusters=k, n_init=10, random_state=0)` for `k = 1, 2, ..., 8` på `X`.
```start
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=400, centers=5, random_state=2)
inertias = ...
```
```losning
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=400, centers=5, random_state=2)
inertias = []
for k in range(1, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    inertias.append(km.inertia_)
```
```test
import numpy as np
assert isinstance(inertias, list), "inertias skal være en liste."
assert len(inertias) == 8, f"Forventet 8 verdier (k = 1 til 8), fikk {len(inertias)}."
assert all(a >= b for a, b in zip(inertias, inertias[1:])), "Inertia skal falle når k øker. Bruker du inertia_ (ikke score)?"
fasit1 = KMeans(n_clusters=1, n_init=10, random_state=0).fit(X).inertia_
assert np.isclose(inertias[0], fasit1), "Første verdi skal være for k = 1. Start range på 1."
```
:::

::: quiz
Hvorfor kan vi ikke velge $k$ ved å ta den $k$-en som gir lavest `inertia_`?
- [ ] Fordi inertia bare kan regnes ut for $k = 2$
- [x] Fordi inertia alltid synker når $k$ øker, og blir 0 når hvert punkt er sin egen klynge
- [ ] Fordi inertia øker med $k$
- [ ] Fordi inertia avhenger av `random_state`
? Flere sentroider gir alltid kortere avstander. Derfor ser vi etter en albue, eller bruker silhouette.
:::

::: quiz
Hva returnerer `km.transform(X_test)` for en fitted `KMeans(n_clusters=4)` og `X_test` med 10 rader?
- [ ] Klyngelabelen til hvert punkt, shape `(10,)`
- [ ] Sentroidene, shape `(4, d)`
- [x] Avstanden fra hvert testpunkt til hver sentroide, shape `(10, 4)`
- [ ] Testdataene skalert, shape `(10, d)`
? `transform` gir avstander til alle sentroidene. `predict` gir labels. `.min(axis=1)` av `transform` gir avstanden til nærmeste sentroide.
:::

::: oppsummering
- `KMeans(n_clusters=k, n_init=10, random_state=0)`, så `fit_predict(X)`.
- Attributter: `labels_`, `cluster_centers_`, `inertia_`. Metoder: `predict` (label), `transform` (avstander).
- Klyngenumrene er vilkårlige.
- Albuemetoden: plott `inertia_` mot $k$ og se etter knekken. Inertia synker alltid.
- Skaler features før k-means.
:::

--- LEKSJON 10.4 | silhouette | Silhouette og valg av k
**Silhouette** måler for hvert punkt hvor godt det passer i sin klynge sammenlignet med nærmeste andre klynge:

$$s = \frac{b - a}{\max(a, b)}$$

- $a$: snittavstand til de andre punktene i **egen** klynge.
- $b$: snittavstand til punktene i **nærmeste andre** klynge.

$s$ nær 1 er bra (godt inne i egen klynge), 0 betyr på grensen, negativ betyr trolig feil klynge. `silhouette_score` er snittet over alle punktene. La oss regne ut én verdi for hånd først:

```python
import numpy as np
from sklearn.metrics import silhouette_samples

X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
labels = np.array([0, 0, 0, 1, 1])

i = 2                                      # punktet x = 2
egen = X[(labels == 0)]
a = np.mean(np.abs(egen[egen[:, 0] != X[i, 0]] - X[i]))   # snitt til 0 og 1
b = np.mean(np.abs(X[labels == 1] - X[i]))                # snitt til 10 og 11
s = (b - a) / max(a, b)
print(f"a = {a}, b = {b}, s = {s:.3f}")
print("sklearn:", silhouette_samples(X, labels)[i].round(3))
```

## Velg k med en tabell
Slik som i prosjekt 3 lager vi en tabell med $k$, silhouette og klyngestørrelser. `np.bincount(labels)` teller hvor mange punkter hver klynge har. Ujevne størrelser kan avsløre at en klynge er delt i to, eller at to er slått sammen.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

X, _ = make_blobs(n_samples=500, centers=4, random_state=3)

rows = []
for k in range(2, 9):
    lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    rows.append({"k": k,
                 "silhouette": silhouette_score(X, lab),
                 "calinski_harabasz": calinski_harabasz_score(X, lab),
                 "sizes": np.sort(np.bincount(lab))[::-1].tolist()})
pd.DataFrame(rows)
```

Silhouette er høyest for $k = 4$, som er riktig. Notatene nevner tre mål du bør kjenne: **inertia** (lav er bra, men synker alltid med $k$), **silhouette** (nær 1 er bra) og **Calinski-Harabasz** (høy er bra: stor avstand mellom klynger relativt til spredningen i dem).

::: advarsel
Silhouette bygger på avstander og favoriserer runde, kompakte klynger. For to halvmåner (neste leksjon) kan den feilaktig foretrekke en k-means-løsning som deler månene feil. Se alltid på dataene i 2D (PCA eller t-SNE) i tillegg.
:::

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

::: quiz
Et punkt har snittavstand $a = 2$ til sin egen klynge og $b = 8$ til nærmeste andre klynge. Hva er silhouette-verdien?
- [ ] 0.25
- [x] 0.75
- [ ] -0.75
- [ ] 6
? $s = (b - a)/\max(a, b) = (8 - 2)/8 = 0.75$. Punktet ligger godt inne i sin egen klynge.
:::

::: quiz
Hva betyr en silhouette-verdi nær $-1$ for et punkt?
- [ ] Punktet er en perfekt klyngesentroide
- [ ] Punktet ligger på grensen mellom to klynger
- [x] Punktet ligger nærmere en annen klynge enn sin egen, og er trolig i feil klynge
- [ ] Punktet er støy ifølge DBSCAN
? Negativ $s$ betyr $b < a$: nabo-klyngen er nærmere enn egen klynge. 0 er på grensen, nær 1 er godt plassert.
:::

::: oppsummering
- Silhouette $s = (b - a)/\max(a, b)$ per punkt. `silhouette_score(X, labels)` er snittet. Nær 1 bra, 0 grense, negativ feil.
- Velg $k$ med høyest silhouette, men sjekk også klyngestørrelser med `np.bincount(labels)` og se på et 2D-plott.
- `calinski_harabasz_score`: høy er bra. Inertia: lav er bra, men synker alltid med $k$.
:::

--- LEKSJON 10.5 | dbscan | DBSCAN
K-means antar runde klynger rundt et sentrum. **DBSCAN** definerer klynger som områder med høy **tetthet**, og kan derfor finne klynger med vilkårlig form. To hyperparametre:

- `eps` ($\epsilon$): radien som definerer naboskap.
- `min_samples`: hvor mange naboer (punktet selv inkludert) et punkt må ha innen `eps` for å være et **kjernepunkt**.

Punkter innen `eps` fra et kjernepunkt hører til klyngen (**kantpunkter**). Punkter som ikke når noe kjernepunkt er **støy** og får label **-1**. Du trenger ikke oppgi antall klynger.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN

X, y = make_moons(n_samples=300, noise=0.05, random_state=0)
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
db = DBSCAN(eps=0.2, min_samples=5).fit_predict(X)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(X[:, 0], X[:, 1], c=km, s=10)
ax[0].set_title("KMeans, k=2")
ax[1].scatter(X[:, 0], X[:, 1], c=db, s=10)
ax[1].set_title("DBSCAN, eps=0.2")
plt.show()
```

K-means deler med en rett linje, fordi hvert punkt går til nærmeste sentroide. DBSCAN følger tettheten langs hver måne.

## Labels, støy og kjernepunkter
```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

X, _ = make_moons(n_samples=300, noise=0.08, random_state=0)
X = np.vstack([X, [[2.5, 1.5], [-1.5, -1.0]]])      # to tydelige outliers

db = DBSCAN(eps=0.2, min_samples=5).fit(X)
labels = db.labels_
n_klynger = len(set(labels)) - (1 if -1 in labels else 0)
print("Klynger:", n_klynger)
print("Støypunkter:", np.sum(labels == -1), "på indeks", np.where(labels == -1)[0])
print("Antall kjernepunkter:", len(db.core_sample_indices_))
```

`set(labels)` gir de unike labelene. Vi trekker fra 1 hvis -1 er med, fordi støy ikke er en klynge. DBSCAN har **ingen** `predict`: den er ikke-parametrisk og må kjøres på alle dataene samtidig (`fit_predict`).

## Velg eps med et k-distanse-plott
For hvert punkt finner vi avstanden til den `min_samples`-te nærmeste naboen (punktet selv teller som den første), sorterer og plotter. Der kurven bøyer kraftig opp, går vi fra punkter inne i klynger til støy. Det er et godt valg av `eps`.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors

X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)
min_samples = 5
nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
distances, _ = nn.kneighbors(X)             # shape (300, 5), første kolonne er 0 (punktet selv)
k_dist = np.sort(distances[:, -1])

plt.plot(k_dist)
plt.xlabel("punkter sortert")
plt.ylabel(f"avstand til nabo nr. {min_samples}")
plt.title("k-distanse-plott")
plt.show()
```

## Følsom for eps og dimensjon
DBSCAN er svært følsom for `eps`. For liten gir mange små klynger og mye støy, for stor slår alt sammen:

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)
for eps in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
    lab = DBSCAN(eps=eps, min_samples=5).fit_predict(X)
    print(f"eps={eps:4}: {len(set(lab) - {-1})} klynger, {np.sum(lab == -1):3d} støypunkter")

# Samme data med ekstra støy-dimensjoner: avstandene vokser, og samme eps finner ingenting
rng = np.random.default_rng(0)
for ekstra in [0, 2, 8]:
    Xe = np.hstack([X, rng.normal(0, 0.3, size=(300, ekstra))])
    lab = DBSCAN(eps=0.2, min_samples=5).fit_predict(Xe)
    print(f"{2 + ekstra:2d} dimensjoner: {len(set(lab) - {-1})} klynger, {np.sum(lab == -1)} støypunkter")
```

Med flere dimensjoner blir alle avstander større og likere (leksjon 10.1), så `eps` må endres, og til slutt finnes det ingen god verdi. Derfor kjører vi ofte DBSCAN på noen få PCA-komponenter. Et annet svakt punkt: `eps` er **global**, så klynger med ulik tetthet er vanskelige.

::: prosjekt Prosjekt 3, problem 2
I prosjektet ble KMeans og DBSCAN sammenlignet. DBSCAN delte dataene i titalls klynger med hundrevis av støypunkter på 20 PCA-komponenter, men fungerte bedre på 5. Det er akkurat følsomheten for `eps` og dimensjon vist over.
:::

::: oppgave Tell klynger og støy
Kjør `DBSCAN(eps=0.3, min_samples=5)` på `X`. Lagre labelene i `labels`, antall klynger (uten støy) i `n_klynger` og antall støypunkter i `n_stoy`.
```start
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.3, random_state=5)
rng = np.random.default_rng(0)
X = np.vstack([X, rng.uniform(-12, 12, size=(15, 2))])

labels = ...
n_klynger = ...
n_stoy = ...
```
```losning
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.3, random_state=5)
rng = np.random.default_rng(0)
X = np.vstack([X, rng.uniform(-12, 12, size=(15, 2))])

labels = DBSCAN(eps=0.3, min_samples=5).fit_predict(X)
n_klynger = len(set(labels)) - (1 if -1 in labels else 0)
n_stoy = int(np.sum(labels == -1))
```
```test
import numpy as np
from sklearn.cluster import DBSCAN
fasit = DBSCAN(eps=0.3, min_samples=5).fit_predict(X)
assert labels is not ... and np.array_equal(labels, fasit), "labels skal komme fra DBSCAN(eps=0.3, min_samples=5) på X."
assert n_klynger == len(set(fasit) - {-1}), f"n_klynger skal ikke telle støy (-1) som klynge. Fikk {n_klynger}."
assert n_stoy == np.sum(fasit == -1), f"n_stoy skal være antall punkter med label -1. Fikk {n_stoy}."
```
:::

::: quiz
Ifølge DBSCAN, når havner to punkter i samme klynge?
- [ ] Når de har samme nærmeste sentroide
- [ ] Når avstanden mellom dem er mindre enn snittavstanden i datasettet
- [x] Når de er forbundet gjennom en kjede av kjernepunkter der hvert steg er kortere enn `eps`
- [ ] Når de har like mange naboer
? Klyngen vokser fra kjernepunkter til alle punkter som kan nås via naboskap innen `eps`. Det første alternativet beskriver k-means.
:::

::: quiz
Hva skriver koden ut?
```kode
labels = np.array([0, 0, 1, -1, 1, 2, -1])
print(len(set(labels)) - (1 if -1 in labels else 0), np.sum(labels == -1))
```
- [ ] `4 2`
- [x] `3 2`
- [ ] `3 0`
- [ ] `2 2`
? Unike labels er {-1, 0, 1, 2}, altså 4. Minus støy gir 3 klynger. To punkter har label -1.
:::

::: oppsummering
- `DBSCAN(eps, min_samples)`: tetthetsbasert, trenger ikke $k$, finner vilkårlige former.
- Label **-1** = støy. Antall klynger: `len(set(labels)) - (1 if -1 in labels else 0)`.
- Ingen `predict`: bruk `fit_predict` på alle dataene.
- Velg `eps` fra knekken i et k-distanse-plott (`NearestNeighbors(n_neighbors=min_samples)`, `kneighbors`, sorter siste kolonne).
- Svært følsom for `eps` og for antall dimensjoner. Global `eps` takler dårlig ulik tetthet.
:::

--- LEKSJON 10.6 | andre-klyngemetoder | Andre klyngemetoder
Notatene deler klyngemetoder i fire kategorier:

| Kategori | Idé | Eksempel |
|---|---|---|
| Sentroidebasert | Sentroider, tilordne punkter, tilpass til konvergens | k-means |
| Tetthetsbasert | Tette områder er klynger, isolerte punkter er støy | DBSCAN |
| Fordelingsbasert | Hver klynge er en sannsynlighetsfordeling (ofte Gauss) | Gaussian Mixture |
| Hierarkibasert | Bygg et hierarki fra ett punkt per klynge til én klynge | Agglomerativ klynging |

K-means og DBSCAN er pensum i detalj. De to andre kategoriene skal du ha hørt om, så vi ser kort på dem i kode.

## Hierarkisk klynging og dendrogram
**Agglomerativ** klynging starter med hvert punkt som egen klynge, og slår hele tiden sammen de to nærmeste klyngene. Et **dendrogram** viser hele hierarkiet. Kutter vi det i en viss høyde, får vi klyngene.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import linkage, dendrogram

X, _ = make_blobs(n_samples=30, centers=3, random_state=0)
Z = linkage(X, method="ward")
dendrogram(Z)
plt.ylabel("avstand ved sammenslåing")
plt.show()
```

De tre lange vertikale strekene øverst viser tre tydelige grupper. `linkage` bestemmer hvordan avstand mellom klynger måles: `"ward"` (minst økning i varians), `"single"` (nærmeste par), `"complete"` (fjerneste par), `"average"`. I sklearn:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import AgglomerativeClustering

X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, link in zip(ax, ["ward", "single"]):
    lab = AgglomerativeClustering(n_clusters=2, linkage=link).fit_predict(X)
    a.scatter(X[:, 0], X[:, 1], c=lab, s=10)
    a.set_title(f"linkage={link}")
plt.show()
```

`"single"` kan følge lange, tynne former (som DBSCAN), mens `"ward"` lager runde klynger (som k-means).

## Spektral klynging
`SpectralClustering(affinity="nearest_neighbors")` bygger en graf der hvert punkt kobles til sine nærmeste naboer, og deler grafen der den har færrest koblinger. Den finner sammenhengende former, og var den metoden som ga best resultat i løsningen av prosjekt 3.

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering, KMeans

X, y = make_moons(n_samples=300, noise=0.05, random_state=0)
sc = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", n_neighbors=15, random_state=0).fit_predict(X)
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)

def enighet(a, b):            # andel like labels, uavhengig av hva klyngene heter
    return max(np.mean(a == b), np.mean(a != b))

print("Spektral vs sann:", enighet(sc, y))
print("KMeans   vs sann:", enighet(km, y))
```

## Fordelingsbasert: Gaussian Mixture
`GaussianMixture` antar at hver klynge er en normalfordeling, og gir **myke** tilordninger: en sannsynlighet per klynge.

```python
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

X, _ = make_blobs(n_samples=300, centers=3, random_state=0)
gm = GaussianMixture(n_components=3, random_state=0).fit(X)
print("Labels:", gm.predict(X[:5]))
print("Sannsynligheter:\n", gm.predict_proba(X[:5]).round(3))
```

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

::: quiz
Hvilke kategorier hører k-means og DBSCAN til?
- [ ] Begge er tetthetsbaserte
- [x] K-means er sentroidebasert, DBSCAN er tetthetsbasert
- [ ] K-means er hierarkisk, DBSCAN er fordelingsbasert
- [ ] K-means er fordelingsbasert, DBSCAN er sentroidebasert
? K-means bygger på sentroider og avstand. DBSCAN bygger på tetthet (antall naboer innen `eps`).
:::

::: oppsummering
- Fire kategorier: sentroidebasert (k-means), tetthetsbasert (DBSCAN), fordelingsbasert (Gaussian Mixture), hierarkibasert (agglomerativ).
- `linkage` + `dendrogram` fra `scipy.cluster.hierarchy` viser hierarkiet. `AgglomerativeClustering(n_clusters, linkage=…)` i sklearn.
- `SpectralClustering(affinity="nearest_neighbors")` følger sammenhengende former via en nabograf.
- `GaussianMixture` gir myke tilordninger med `predict_proba`.
:::

--- LEKSJON 10.7 | pca | PCA
**PCA** (principal component analysis) finner nye, ortogonale akser der dataene varierer mest. Første hovedkomponent (PC1) er retningen med størst varians, PC2 er nest størst og står vinkelrett på PC1, og så videre. Formelt er hovedkomponentene **egenvektorene til kovariansmatrisen**, og egenverdiene er variansen langs hver.

## For hånd
Notatenes enkle eksempel er $x_1 = (1, 2, 3)$ og $x_2 = (2, 1, 0)$:

```python
import numpy as np

X = np.array([[1, 2], [2, 1], [3, 0]], dtype=float)   # rader = punkter, kolonner = features
S = np.cov(X, rowvar=False)                            # kovariansmatrisen, deler på N - 1
print("Kovariansmatrisen S:")
display(S)
egenverdier, egenvektorer = np.linalg.eigh(S)
print("Egenverdier:", egenverdier.round(6))
print("Egenvektorer (kolonner):\n", egenvektorer.round(3))
```

Diagonalen i `S` er variansen til hver feature (1 og 1), og de negative verdiene utenfor diagonalen sier at $x_2$ synker når $x_1$ øker. Egenverdiene er 0 og 2, som i notatene. Egenverdien 0 betyr at all variansen ligger langs én retning, $(1, -1)/\sqrt 2$: alle punktene ligger på linjen $x_1 + x_2 = 3$.

Nå hele oppskriften på notatenes tabell 9: **sentrer**, regn **kovariansmatrisen**, finn **egenvektorer**, **sorter** synkende, **projiser**.

```python
import numpy as np

x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([1.6, 4.2, 5.7, 8.4, 9.2])
X = np.vstack((x1, x2)).T                  # shape (5, 2)

Xc = X - X.mean(axis=0)                    # 1. sentrer
S = np.cov(Xc, rowvar=False)               # 2. kovarians
verdier, vektorer = np.linalg.eigh(S)      # 3. egenverdier/-vektorer (stigende!)
rekkefolge = np.argsort(verdier)[::-1]     # 4. sorter synkende
verdier, W = verdier[rekkefolge], vektorer[:, rekkefolge]
Z = Xc @ W                                 # 5. projiser: koordinater langs PC1 og PC2

print("S =\n", S.round(3))
print("Egenverdier:", verdier.round(4))
print("Andel varians:", (verdier / verdier.sum()).round(4))
print("PC1:", W[:, 0].round(4))
print("Data i PC-koordinater:\n", Z.round(3))
```

`np.linalg.eigh` (for symmetriske matriser) returnerer egenverdiene i **stigende** rekkefølge. Derfor må vi snu. PC1 forklarer over 99 % av variansen: dataene ligger nesten på en linje.

## Med sklearn
Notatenes kode, utvidet med attributtene du bør kjenne:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([1.6, 4.2, 5.7, 8.4, 9.2])
X = np.vstack((x1, x2)).T

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("Før transform, X:")
display(X)
print("Etter fit_transform, X_pca (kolonne 0 = PC1, kolonne 1 = PC2):")
display(X_pca)
print("explained_variance_:      ", pca.explained_variance_.round(4))
print("explained_variance_ratio_:", pca.explained_variance_ratio_.round(4))
print("components_ (rader = PC-er):\n", pca.components_.round(4))
print("mean_:", pca.mean_)

PC1, PC2 = X_pca[:, 0], X_pca[:, 1]
plt.scatter(x1, x2, color="blue", label="Original Data", marker="*")
plt.scatter(PC1, PC2, color="red", label="PCA Transformed Data")
plt.legend()
plt.axis("equal")
plt.show()
```

Samme tall som for hånd. I `X_pca` er nesten all variasjonen i første kolonne, mens PC2-kolonnen er nær 0. Under `pca` i variabelpanelet finner du `explained_variance_ratio_`, `components_` og `mean_`. `components_` har PC-ene som **rader** (vår `W` har dem som kolonner), og fortegnet kan være snudd. En egenvektor peker like godt begge veier.

## Redusere til én dimensjon og tilbake
Med `n_components=1` beholder vi bare PC1. `inverse_transform` går tilbake til det opprinnelige rommet. Punktene havner da på PC1-linjen, og det vi mistet er **rekonstruksjonsfeilen**:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

X = np.array([[1, 1.6], [2, 4.2], [3, 5.7], [4, 8.4], [5, 9.2]])
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)            # shape (5, 1)
X_tilbake = pca.inverse_transform(X_pca) # shape (5, 2), på linjen
feil = np.sum((X - X_tilbake)**2, axis=1)
print("Rekonstruksjonsfeil per punkt:", feil.round(3))

plt.scatter(X[:, 0], X[:, 1], label="original")
plt.scatter(X_tilbake[:, 0], X_tilbake[:, 1], marker="x", label="rekonstruert fra PC1")
plt.legend()
plt.axis("equal")
plt.show()
```

## Hvor mange komponenter?
Med ekte data plotter vi den **kumulative** andelen forklart varians og velger nok komponenter til å nå for eksempel 90 % eller 95 %. Gir du `PCA(n_components=0.95)`, velger sklearn selv. Her er de håndskrevne sifrene i `load_digits` (64 piksler per bilde):

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

X, y = load_digits(return_X_y=True)
pca = PCA().fit(X)
kum = np.cumsum(pca.explained_variance_ratio_)
print("Komponenter for 95 %:", np.argmax(kum >= 0.95) + 1)
print("PCA(0.95) velger:", PCA(n_components=0.95).fit(X).n_components_)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(range(1, 65), kum, "o-", ms=3)
ax[0].axhline(0.95, color="gray", ls="--")
ax[0].set_xlabel("antall komponenter")
ax[0].set_ylabel("kumulativ forklart varians")
Z = PCA(n_components=2).fit_transform(X)
sc = ax[1].scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=5)
ax[1].set_title("Sifrene projisert på PC1 og PC2")
plt.colorbar(sc, ax=ax[1])
plt.show()
```

To komponenter forklarer under 30 % av variansen, så sifrene overlapper mye i 2D. PCA er lineær, og det er grunnen til at notatene viser at MNIST i 2D ser rotete ut.

::: husk God praksis
- **Skaler** først med `StandardScaler` hvis featurene har ulike enheter. Ellers vinner featuren med størst tall.
- `fit` på treningsdata, `transform` på testdata, som alle andre transformere (leksjon 7.2). Notatene viser dette med `make_s_curve`.
:::

::: oppgave PCA for hånd
Gjør PCA for hånd på `X`: sentrer, regn kovariansmatrisen med `np.cov(..., rowvar=False)`, finn egenverdier og egenvektorer med `np.linalg.eigh`, og sorter synkende. Lagre egenverdiene (synkende) i `verdier`, egenvektorene som **kolonner** i samme rekkefølge i `W`, og projeksjonen på PC1 og PC2 i `Z` (shape `(n, 2)`).
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

::: quiz
Hva er hovedkomponentene i PCA?
- [ ] Featurene med høyest varians i de opprinnelige dataene
- [x] Egenvektorene til kovariansmatrisen, sortert etter egenverdi
- [ ] Klyngesentrene i dataene
- [ ] Radene i datamatrisen med størst norm
? PC-ene er nye retninger, lineærkombinasjoner av de gamle featurene. Egenverdien sier hvor mye varians som ligger langs hver.
:::

::: quiz
Hvilken kolonne i `vec` er PC1?
```kode
S = np.cov(Xc, rowvar=False)
val, vec = np.linalg.eigh(S)
```
- [ ] `vec[:, 0]`, fordi den kommer først
- [x] `vec[:, -1]`, fordi `eigh` sorterer egenverdiene stigende
- [ ] `vec[0, :]`, fordi egenvektorene er rader
- [ ] `vec[:, np.argmin(val)]`
? `eigh` returnerer stigende egenverdier og egenvektorene som kolonner. Størst varians er derfor siste kolonne.
:::

::: oppsummering
- For hånd: `Xc = X - X.mean(axis=0)`, `S = np.cov(Xc, rowvar=False)`, `np.linalg.eigh(S)`, sorter synkende, `Z = Xc @ W[:, :k]`.
- `eigh` gir **stigende** egenverdier og egenvektorer som **kolonner**.
- sklearn: `PCA(n_components)`, `fit_transform`, `transform`, `inverse_transform`.
- `explained_variance_ratio_` (andel varians per PC), `components_` (PC-ene som rader), `mean_`.
- Velg antall komponenter med `np.cumsum(explained_variance_ratio_)` eller `PCA(n_components=0.95)`.
- Skaler først, og `fit` bare på treningsdata.
:::

--- LEKSJON 10.8 | tsne | t-SNE
**t-SNE** er en ikke-lineær metode for å **visualisere** høydimensjonale data i 2D. Den regner ut hvor sannsynlig det er at to punkter er naboer i det høye rommet, plasserer punktene tilfeldig i 2D, og flytter dem med gradient descent slik at nabosannsynlighetene i 2D (en t-fordeling) ligner mest mulig. Tapet er KL-divergensen.

t-SNE er tregt for store datasett, så vi bruker 300 av sifrene:

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

X, y = load_digits(return_X_y=True)
X, y = X[:300], y[:300]

tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0)
X_emb = tsne.fit_transform(X)
print("Shape:", X_emb.shape, " KL-divergens:", round(tsne.kl_divergence_, 3))

sc = plt.scatter(X_emb[:, 0], X_emb[:, 1], c=y, cmap="tab10", s=12)
plt.colorbar(sc)
plt.title("t-SNE av 300 siffer")
plt.show()
```

Sammenlign med PCA-plottet i forrige leksjon: t-SNE skiller sifrene i tydelige grupper, fordi den bevarer **lokale naboskap** i stedet for global varians.

## Perplexity
`perplexity` er omtrent hvor mange naboer hvert punkt «bryr seg om». Lav verdi gir mange små, oppsplittede grupper. Høy verdi gir mer global struktur. Typiske verdier er 5 til 50, og den må være mindre enn antall punkter.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

X, y = load_digits(return_X_y=True)
X, y = X[:200], y[:200]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, perp in zip(ax, [3, 40]):
    emb = TSNE(n_components=2, perplexity=perp, init="pca", random_state=0).fit_transform(X)
    a.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=8)
    a.set_title(f"perplexity={perp}")
plt.show()
```

## t-SNE kan ikke brukes på nye punkter
t-SNE lærer ikke en funksjon fra høyt til lavt rom. Den optimerer **posisjonene** til akkurat de punktene den fikk. Derfor finnes bare `fit_transform`, ikke `transform`:

```python feil
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

X, _ = load_digits(return_X_y=True)
tsne = TSNE(n_components=2, perplexity=20, random_state=0)
tsne.fit(X[:100])
tsne.transform(X[100:110])
```

Kommer det nye punkter, må hele embeddingen regnes ut på nytt. Det er også grunnen til at vi ikke kan dele i trening og test slik vi gjør med PCA.

## PCA mot t-SNE
| | PCA | t-SNE |
|---|---|---|
| Mål | Bevare mest mulig **varians** | Bevare **naboskap** (minimere KL-divergens) |
| Lineær? | Ja | Nei |
| Nye punkter | `transform` virker | Nei, bare `fit_transform` |
| Nye features til en modell | Ja | Nei, bare til visualisering |
| Hastighet | Rask | Treg på store datasett |
| Tolkning av avstander | Meningsfulle | Avstander **mellom** klynger og klyngestørrelser betyr lite |

En vanlig rørledning er derfor PCA først (til 30 til 50 dimensjoner, raskt og fjerner støy), så t-SNE til 2D for plottet. `init="pca"` gir dessuten mer stabile resultater enn tilfeldig start.

::: eksamen
Eksamen 2024 spurte hva t-SNE vanligvis brukes til (visualisering i lav dimensjon, ikke klassifisering eller klynging). Eksamen 2025 spurte hvorfor t-SNE ikke kan brukes på nye punkter (den optimerer posisjonene til de opprinnelige punktene, og har ingen mapping).
:::

::: oppgave PCA før t-SNE
Reduser først `X` til 30 dimensjoner med `PCA(n_components=30, random_state=0)`, og kjør deretter `TSNE(n_components=2, perplexity=30, init="pca", random_state=0)` på resultatet. Lagre PCA-resultatet i `X_pca` og t-SNE-resultatet i `X_emb`.
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

::: quiz
To klynger ligger langt fra hverandre i et t-SNE-plott, og en tredje ligger nær den ene. Hva kan vi trygt konkludere?
- [ ] At den tredje klyngen er mest lik den den ligger nær i det opprinnelige rommet
- [ ] At den største klyngen i plottet har flest punkter
- [x] Lite om avstander mellom klyngene: t-SNE bevarer lokale naboskap, ikke globale avstander
- [ ] At dataene er lineært separerbare
? Avstander mellom klynger og klyngestørrelser i t-SNE avhenger av perplexity og tilfeldigheter. Det t-SNE viser pålitelig er hvilke punkter som er naboer.
:::

::: quiz
Hvilken metode kan du bruke til å lage features for nye testdata etter å ha tilpasset den på treningsdata?
- [x] `PCA`, med `pca.transform(X_test)`
- [ ] `TSNE`, med `tsne.transform(X_test)`
- [ ] Begge
- [ ] Ingen av dem
? PCA lærer en lineær projeksjon som kan brukes på nye punkter. t-SNE har ingen `transform`.
:::

::: oppsummering
- `TSNE(n_components=2, perplexity=…, init="pca", random_state=…).fit_transform(X)`. Bare for visualisering.
- `perplexity` ≈ antall naboer hvert punkt tar hensyn til (typisk 5 til 50).
- Ingen `transform`: t-SNE optimerer posisjonene til punktene den fikk, og kan ikke brukes på nye data.
- Avstander mellom klynger og klyngestørrelser i plottet betyr lite.
- Vanlig rørledning: PCA til 30 til 50 dimensjoner, så t-SNE til 2D. Bruk få punkter, t-SNE er treg.
:::

--- LEKSJON 10.9 | anomalideteksjon | Anomalideteksjon
I **anomalideteksjon** leter vi etter punktene som ikke ligner resten. Vi har mange normale eksempler og få eller ingen merkede anomalier, så det gjøres vanligvis uveiledet. Alle metodene under gir en **score** per punkt. Så velger vi en **terskel**, ofte en persentil.

## z-score (én feature)
Hvor mange standardavvik er punktet fra snittet? $|z| > 3$ er en vanlig tommelfingerregel.

```python
import numpy as np

rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(50, 5, 200), [95, 3]])   # to innplantede avvik
z = (x - x.mean()) / x.std()
print("Avvik (|z| > 3):", np.where(np.abs(z) > 3)[0], x[np.abs(z) > 3])
```

## Avstand til nærmeste sentroide (k-means)
Fra notatene: tilpass k-means på treningsdata, regn avstanden fra hvert testpunkt til nærmeste sentroide, og kall de 5 % fjerneste for anomalier.

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X_train, _ = make_blobs(n_samples=400, centers=3, random_state=0)
X_test, _ = make_blobs(n_samples=100, centers=3, random_state=0)
X_test = np.vstack([X_test, [[8, 8], [-6, -6]]])      # to anomalier på slutten

kmeans = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X_train)
distances = kmeans.transform(X_test).min(axis=1)
threshold = np.percentile(distances, 95)
outliers = distances > threshold
print("Terskel:", threshold.round(2))
print("Anomalier:", np.where(outliers)[0])
```

`np.percentile(distances, 95)` er verdien som 95 % av avstandene er mindre enn. Begge de innplantede punktene (indeks 100 og 101) blir funnet, men også fire normale punkter i utkanten av klyngene. Det er **falske positive**: en 95-persentil merker alltid 5 % av punktene, uansett hvor mange anomalier som faktisk finnes. Terskelen er et valg, og ingen verdi er riktig for alle data.

## DBSCAN, Isolation Forest og LOF
- **DBSCAN**: støypunktene (`labels_ == -1`) er anomaliene. Ingen terskel å velge, men `eps` må velges.
- **Isolation Forest**: bygger trær med *tilfeldige* splitter. Anomalier er få og ulike, så de isoleres etter få splitter, nær roten. Kort snittsti = anomali.
- **Local Outlier Factor (LOF)**: sammenligner tettheten rundt et punkt med tettheten rundt naboene. Mye lavere tetthet enn naboene = anomali. `negative_outlier_factor_` er mer negativ jo mer anomalt.

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

X, _ = make_blobs(n_samples=300, centers=2, cluster_std=0.8, random_state=1)
X = np.vstack([X, [[0, 10], [-12, 0], [5, -2]]])      # indeks 300, 301, 302

iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=0).fit(X)
print("IsolationForest:", np.where(iso.predict(X) == -1)[0])

lof = LocalOutlierFactor(n_neighbors=20)
lof_lab = lof.fit_predict(X)
score = lof.negative_outlier_factor_
print("LOF, 3 mest negative:", np.argsort(score)[:3])

db = DBSCAN(eps=1.0, min_samples=5).fit(X)
print("DBSCAN støy:", np.where(db.labels_ == -1)[0])
```

`predict` gir `1` for normale punkter og `-1` for anomalier, både for `IsolationForest` og `LocalOutlierFactor`. `contamination` er hvor stor andel du tror er anomalier, og bestemmer terskelen. Isolation Forest kan `fit` på trening og `predict` på test, slik notatene viser.

## Rekonstruksjonsfeil med PCA
PCA med få komponenter lærer strukturen de fleste punktene deler. Et punkt som ikke passer, kan ikke rekonstrueres godt. Slik fant løsningen av prosjekt 3 de tre radene som ikke var kartbilder. Her planter vi tre «bilder» som ikke er siffer blant de 1797 sifrene:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

X, _ = load_digits(return_X_y=True)
X = X.copy()
rng = np.random.default_rng(0)
X[100] = rng.uniform(0, 16, 64)          # støy
X[900] = np.tile([0, 16], 32)            # stripemønster
X[1500] = 8.0                            # helt grått

pca = PCA(n_components=20).fit(X)
X_rec = pca.inverse_transform(pca.transform(X))
recon_error = np.sum((X - X_rec)**2, axis=1)

topp = np.argsort(recon_error)[::-1][:5]
print("Største feil:", topp, recon_error[topp].round(0))
print("Median feil:", np.median(recon_error).round(0))

fig, ax = plt.subplots(1, 3, figsize=(7, 2.5))
for a, i in zip(ax, topp[:3]):
    a.imshow(X[i].reshape(8, 8), cmap="gray")
    a.set_title(f"rad {i}")
    a.axis("off")
plt.show()
```

De tre innplantede radene har mye større feil enn alle sifrene. Et histogram over `recon_error` (gjerne med `log=True`) viser et tydelig gap, og terskelen legges i gapet.

::: eksamen
Eksamen 2024: «Hva betyr det hvis en modell for anomalideteksjon gir mange falske positive?» Den merker mange **normale** punkter som anomalier, typisk fordi terskelen er for lav. Eksamen 2025 spurte hvorfor anomalier får høy score i Isolation Forest: de isoleres tidligere, med kortere sti fra roten.
:::

::: oppgave Rekonstruksjonsfeil og persentil
Tilpass `PCA(n_components=2)` på `X`, regn ut rekonstruksjonsfeilen `recon_error` (sum av kvadrerte avvik per rad), og lagre indeksene til radene med feil **over 99-persentilen** i `anomalier` (et array med indekser, bruk `np.where`).
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

::: quiz
Hva gjør koden?
```kode
distances = kmeans.transform(X_test).min(axis=1)
threshold = np.percentile(distances, 95)
outliers = distances > threshold
```
- [ ] Merker de 95 % fjerneste punktene som anomalier
- [x] Merker de omtrent 5 % testpunktene som ligger lengst fra sin nærmeste sentroide som anomalier
- [ ] Merker punktene som ligger nærmere enn 95 enheter fra en sentroide
- [ ] Fjerner 5 % av sentroidene
? `transform` gir avstand til alle sentroidene, `.min(axis=1)` avstanden til nærmeste. 95-persentilen er terskelen som 95 % ligger under.
:::

::: quiz
Hvorfor får anomalier kort snittsti i en Isolation Forest?
- [ ] Fordi trærne er trent på labels som sier hva som er anomalier
- [x] Fordi få og avvikende punkter blir isolert etter få tilfeldige splitter
- [ ] Fordi anomalier alltid havner i den største løvnoden
- [ ] Fordi trærne har `max_depth=1`
? Et punkt langt fra resten havner alene på én side av en tilfeldig terskel raskt. Normale punkter i tette områder trenger mange splitter.
:::

::: oppsummering
- z-score: `(x - x.mean()) / x.std()`, |z| > 3 er avvik.
- k-means: `kmeans.transform(X).min(axis=1)` og terskel `np.percentile(..., 95)`.
- DBSCAN: `labels_ == -1`. IsolationForest og LOF: `predict(...) == -1`, `contamination` styrer terskelen.
- LOF: `negative_outlier_factor_`, mer negativ = mer anomal.
- PCA-rekonstruksjonsfeil: `np.sum((X - pca.inverse_transform(pca.transform(X)))**2, axis=1)`, terskel i gapet.
- Mange falske positive = normale punkter merkes som anomalier (terskelen er for streng mot normale).
:::

--- LEKSJON 10.10 | prosjekt-3 | Prosjekt 3-teknikker
Prosjekt 3, problem 2 var en ekte uveiledet oppgave: en stor CSV uten labels, med stokkede kolonner, som viste seg å være bilder. Her øver vi på de samme teknikkene med `load_digits`.

## Klyngenøyaktighet som ikke bryr seg om navn
Klyngelabels er vilkårlige: `[1, 1, 0, 0]` er like riktig som `[0, 0, 1, 1]`. For å måle nøyaktighet lager vi en **kontingenstabell** (hvor mange punkter fra sann klasse $i$ havnet i klynge $j$), og finner den beste én-til-én-koblingen mellom klynger og klasser med den **ungarske algoritmen** (`linear_sum_assignment`). Dette er funksjonen fra prosjektet:

```python
import numpy as np
import scipy.optimize
import sklearn.metrics

def calculate_clustering_accuracy(true_labels, cluster_labels):
    contingency = sklearn.metrics.cluster.contingency_matrix(true_labels, cluster_labels)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-contingency)
    correct = contingency[row_ind, col_ind].sum()
    return correct / true_labels.size

true_labels = np.array([0, 1, 2, 0, 1, 2])
cluster_labels = np.array([1, 2, 0, 1, 2, 0])
print("Kontingenstabell (rad = sann klasse, kolonne = klynge):")
display(sklearn.metrics.cluster.contingency_matrix(true_labels, cluster_labels))
print("Omdøpte klynger:", calculate_clustering_accuracy(true_labels, cluster_labels))
print("Én sann klynge:  ", calculate_clustering_accuracy(np.zeros(6, int), cluster_labels))
print("Alt i én klynge: ", calculate_clustering_accuracy(np.array([1, 1, 1, 2, 2, 2]), np.zeros(6, int)))
```

I tabellen har hver rad nøyaktig én celle med 2: klasse 0 havnet i klynge 1, klasse 1 i klynge 2 og klasse 2 i klynge 0. `linear_sum_assignment` velger én celle per rad og kolonne slik at summen blir størst mulig. Den *minimerer* kostnaden, så vi gir den `-contingency` for å *maksimere* antall riktige.

## Bilder lagret som rader
Et gråtonebilde på $h \times w$ piksler kan lagres som én rad med $h \cdot w$ tall. `load_digits` har 64 kolonner = $8 \times 8$. Med `reshape(-1, 8, 8)` får vi bildene tilbake (`-1` betyr «regn ut selv»). I prosjektet var det 16 384 kolonner, altså $128 \times 128$.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
print("X:", X.shape)
print("Første siffer som én rad med 64 piksler:")
display(X[:1])
images = X.reshape(-1, 8, 8)
print("images:", images.shape)
print("Det samme sifferet etter reshape til 8 × 8:")
display(images[0])

fig, axes = plt.subplots(2, 8, figsize=(10, 3))
for ax, i in zip(axes.flat, range(16)):
    ax.imshow(images[i], cmap="gray")
    ax.set_title(str(y[i]))
    ax.axis("off")
plt.show()
```

Raden med 64 tall er vanskelig å lese, men etter `reshape` ser du sifferet (en 0) direkte i fargene i rutenettet. Det er nøyaktig de samme tallene, bare ordnet i 8 rader.

## Gjenopprette stokkede kolonner
I prosjektet var pikselkolonnene stokket, og en ekstra **header-rad** inneholdt den opprinnelige pikselindeksen til hver kolonne (`col_<indeks>`). `np.argsort` på disse indeksene gir rekkefølgen som setter kolonnene tilbake på plass:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

X, _ = load_digits(return_X_y=True)
rng = np.random.default_rng(0)
perm = rng.permutation(64)
df = pd.DataFrame(X[:, perm], columns=[f"col_{p}" for p in perm])   # "stokket CSV"
print(list(df.columns[:6]))

order = np.argsort([int(c.split("_")[1]) for c in df.columns])
pixels = df.to_numpy()[:, order]
print("Gjenopprettet riktig:", np.array_equal(pixels, X))
print("Før (stokket):")
display(df.iloc[:3, :8])
print("Etter (df.iloc[:, order]):")
display(df.iloc[:, order].iloc[:3, :8])

fig, ax = plt.subplots(1, 2, figsize=(5, 2.5))
ax[0].imshow(df.to_numpy()[0].reshape(8, 8), cmap="gray")
ax[0].set_title("stokket")
ax[1].imshow(pixels[0].reshape(8, 8), cmap="gray")
ax[1].set_title("gjenopprettet")
for a in ax:
    a.axis("off")
plt.show()
```

`np.argsort(idx)` svarer på: «hvilken kolonne skal stå på plass 0, 1, 2 …?». Merk at stokking av kolonner **ikke** endrer avstander eller PCA (det er bare en omnummerering av features). Den ødelegger bare muligheten til å se på bildene.

## Rørledningen: PCA, klynging, t-SNE
Den typiske flyten i prosjektet var: reduser dimensjonen med PCA, klyng i PCA-rommet, og bruk t-SNE bare til å *visualisere* klyngene.

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import sklearn.metrics
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import TSNE

def calculate_clustering_accuracy(true_labels, cluster_labels):
    c = sklearn.metrics.cluster.contingency_matrix(true_labels, cluster_labels)
    r, k = scipy.optimize.linear_sum_assignment(-c)
    return c[r, k].sum() / true_labels.size

X, y = load_digits(return_X_y=True)
Z = PCA(n_components=30, random_state=0).fit_transform(X)

km = KMeans(n_clusters=10, n_init=10, random_state=0).fit_predict(Z)
sc = SpectralClustering(n_clusters=10, affinity="nearest_neighbors", n_neighbors=10, random_state=0).fit_predict(Z)
print(f"KMeans:   {calculate_clustering_accuracy(y, km):.3f}")
print(f"Spectral: {calculate_clustering_accuracy(y, sc):.3f}")

idx = np.arange(0, len(X), 6)                # t-SNE på hvert 6. punkt for fart
T = TSNE(n_components=2, init="pca", random_state=0).fit_transform(Z[idx])
plt.scatter(T[:, 0], T[:, 1], c=sc[idx], cmap="tab10", s=8)
plt.title("Spektrale klynger vist med t-SNE")
plt.show()
```

I det virkelige prosjektet holdt ikke rå piksler: bildene var rotert og zoomet, så avstanden mellom to bilder av samme sted var dominert av rotasjonen. Løsningen var å lage rotasjonsinvariante features før PCA. Lærdommen er generell: klynging er aldri bedre enn avstandsmålet, og avstandsmålet er aldri bedre enn featurene.

## Lagre en innlevering
Kaggle ville ha en CSV med kolonnene `ID` og `cluster`. Anomaliene ble gitt en egen klynge:

```python
import numpy as np
import pandas as pd

labels = np.array([0, 2, 1, 1, 0, 2])
anomalies = np.array([3])
final_labels = labels.copy()
final_labels[anomalies] = labels.max() + 1          # egen klynge for anomaliene
submission = pd.DataFrame({"ID": np.arange(len(labels)), "cluster": final_labels})
submission.to_csv("submission.csv", index=False)
submission
```

::: oppgave Skriv calculate_clustering_accuracy
Skriv funksjonen `calculate_clustering_accuracy(true_labels, cluster_labels)` som returnerer andelen riktige etter beste omdøping av klyngene. Bruk `contingency_matrix` og `linear_sum_assignment` (importert).
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

::: quiz
Hva returnerer `calculate_clustering_accuracy(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]))`?
- [ ] 0.0, fordi ingen labels er like
- [ ] 0.5
- [x] 1.0, fordi klyngene er like etter omdøping
- [ ] En feil, fordi labelene er forskjellige
? Funksjonen finner den beste koblingen (klynge 1 = klasse 0, klynge 0 = klasse 1), og da er alle fire riktige.
:::

::: quiz
En CSV har 16 384 kolonner med pikselverdier i stokket rekkefølge, og header-raden er `col_<opprinnelig indeks>`. Hvilken linje setter kolonnene i riktig rekkefølge?
- [ ] `pixels = X[:, np.sort(idx)]`
- [x] `pixels = X[:, np.argsort(idx)]`
- [ ] `pixels = X[np.argsort(idx), :]`
- [ ] `pixels = X.reshape(128, 128)`
? `np.sort` gir verdiene sortert, ikke posisjonene. `np.argsort(idx)` gir hvilke kolonner som skal stå først, og vi indekserer kolonnene (andre akse).
:::

::: oppsummering
- Klyngenøyaktighet: `contingency_matrix` + `linear_sum_assignment(-c)`, så `c[r, k].sum() / n`. Uavhengig av klyngenavn.
- Bilder som rader: `X.reshape(-1, h, w)` og `imshow`. 64 = 8·8, 16 384 = 128·128.
- Stokkede kolonner: `order = np.argsort(opprinnelige_indekser)`, så `X[:, order]`. Stokking endrer ikke avstander.
- Rørledning: (skaler) → PCA → klynging (KMeans/DBSCAN/Spectral) → t-SNE bare for å se → anomalier med rekonstruksjonsfeil.
- Innlevering: `pd.DataFrame({"ID": …, "cluster": …}).to_csv("submission.csv", index=False)`.
:::
