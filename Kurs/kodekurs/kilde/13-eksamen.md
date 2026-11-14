=== DEL 13 | eksamen | Eksamenstrening: les og forstå kode
Flervalgsspørsmål i eksamensstil der du leser kode uten å kjøre den.

--- LEKSJON 13.1 | slik-ser-de-ut | Slik ser kodespørsmålene ut
Eksamen 2025 hadde fire korte kodespørsmål med fire alternativer hver, og du må «kjøre» koden i hodet.

## Strategi: fem spørsmål du stiller til koden

1. Hvilken shape har hver variabel?
2. Kalles `fit` bare på treningsdata?
3. Hva returneres: array, objekt eller `None`?
4. Finnes funksjonen i det hele tatt?
5. Krasjer, lekker eller gir hvert alternativ feil type?

## De fire spørsmålene fra eksamen 2025

::: quiz
**Eksamen 2025, spørsmål 1.** Du har trent en modell med koden under, og vil nå evaluere modellen på `X_test`. Hvilken kodelinje bruker **samme** skalering på testdataene?
```kode
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
```
- [ ] `X_test_scaled = StandardScaler().fit_transform(X_test)`
- [ ] `X_test_scaled = X_test / X_train.mean(axis=0)`
- [ ] `X_test_scaled = scaler.fit(X_test)`
- [x] `X_test_scaled = scaler.transform(X_test)`
? `scaler` har lært gjennomsnitt og standardavvik fra `X_train`, og `transform` bruker de samme tallene på testdataene (A og C tilpasser på nytt til testdata, B er ikke standardisering).
:::

::: quiz
**Eksamen 2025, spørsmål 2.** Anta at `X` er et 2D-array med shape `(n_samples, n_features)` og `w` er et 1D-array med shape `(n_features,)`. Hva inneholder `y_pred` etter koden?
```kode
import numpy as np
z = np.matmul(X, w)
y_pred = 1 / (1 + np.exp(-z))
```
- [ ] For hvert sample en score som kan anta hvilken som helst reell verdi
- [x] Predikerte sannsynligheter i intervallet (0, 1) for hvert sample
- [ ] Logaritmen av de predikerte sannsynlighetene for hvert sample
- [ ] Klasselabels 0 eller 1 for hvert sample
? `z` er logits, og sigmoid `1 / (1 + np.exp(-z))` klemmer hver verdi inn i (0, 1), så `y_pred` er sannsynligheter.
:::

::: quiz
**Eksamen 2025, spørsmål 3.** Du ønsker å splitte datasettet i en treningsdel og en testdel. Hvilken funksjon gjør dette?
- [x] `sklearn.model_selection.train_test_split(X, y)`
- [ ] `numpy.random.separate(X, y)`
- [ ] `pandas.Series.split_dataset(X, y)`
- [ ] `matplotlib.pyplot.partition(X, y)`
? `train_test_split` ligger i `sklearn.model_selection`, og de tre andre funksjonene finnes ikke.
:::

::: quiz
**Eksamen 2025, spørsmål 4.** Du har en replay-buffer lagret som en Python-liste med overganger (`buffer = [...]`). Hvilken Python-funksjon brukes for å trekke en tilfeldig minibatch på 32 overganger?
- [ ] `np.take(buffer, 32)`
- [ ] `buffer.sort()`
- [x] `random.sample(buffer, k=32)`
- [ ] `buffer[:32]`
? `random.sample(buffer, k=32)` trekker 32 ulike overganger tilfeldig, mens `np.take` gir én bestemt overgang, `sort()` returnerer `None` og `buffer[:32]` er de eldste.
:::

## Øv: riktig skalering

::: oppgave Skaler testdataene riktig
Rett siste linje slik at `X_test_scaled` får **samme** skalering som treningsdataene.
```start
import numpy as np
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
X_train = rng.normal(10, 2, size=(80, 3))
X_test = rng.normal(10, 2, size=(20, 3))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = StandardScaler().fit_transform(X_test)   # rett denne linjen
```
```losning
import numpy as np
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
X_train = rng.normal(10, 2, size=(80, 3))
X_test = rng.normal(10, 2, size=(20, 3))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
```test
assert isinstance(X_test_scaled, np.ndarray), "X_test_scaled skal være et array. Har du brukt fit i stedet for transform?"
assert X_test_scaled.shape == (20, 3), "X_test_scaled skal ha shape (20, 3)."
forventet = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)
assert np.allclose(X_test_scaled, forventet), "Testdataene skal skaleres med gjennomsnitt og standardavvik fra TRENINGSdataene. Bruk scaler-objektet som allerede er tilpasset X_train."
```
:::

## Varianter av de samme spørsmålene

::: quiz
Samme kode som i spørsmål 2, men nå har `X` shape `(100, 3)` og `w` shape `(3,)`. Hvilken shape har `y_pred`?
```kode
z = np.matmul(X, w)
y_pred = 1 / (1 + np.exp(-z))
```
- [ ] `(3,)`
- [ ] `(100, 3)`
- [x] `(100,)`
- [ ] `(1, 100)`
? `(100, 3)` ganger `(3,)` gir én verdi per rad, og sigmoid virker elementvis, så shape blir `(100,)`.
:::

::: quiz
En medstudent skriver `X_test_scaled = scaler.fit(X_test)` og får en merkelig feil i neste linje, `model.predict(X_test_scaled)`. Hva er `X_test_scaled`?
- [x] Selve `StandardScaler`-objektet, fordi `fit` returnerer objektet det ble kalt på
- [ ] Et array med testdataene skalert med test-gjennomsnittet
- [ ] `None`, fordi `fit` ikke returnerer noe
- [ ] Et array med testdataene skalert med trenings-gjennomsnittet
? `fit` returnerer alltid `self` i sklearn, så `X_test_scaled` blir scaler-objektet og ikke et array.
:::

::: quiz
Hvorfor trekker DQN en **tilfeldig** minibatch fra replay-bufferen i stedet for å trene på de 32 siste overgangene?
- [ ] Fordi `random.sample` er raskere enn slicing
- [x] Overganger som kommer rett etter hverandre er sterkt korrelerte. Tilfeldig trekning bryter korrelasjonen og gjør treningen mer stabil
- [ ] Fordi de siste overgangene alltid har reward 0
- [ ] Fordi target-nettverket bare kan brukes på tilfeldige data
? Påfølgende overganger er sterkt korrelerte, og tilfeldig trekning gir mer uavhengige treningseksempler og stabilere trening.
:::

--- LEKSJON 13.2 | hva-skrives-ut | Hva skrives ut?
Gå gjennom koden linje for linje og noter verdi og shape for hver variabel.

## Øv på shapes og NumPy

::: quiz
Hva skriver koden med `reshape` ut?
```kode
import numpy as np
a = np.arange(6)
b = a.reshape(-1, 1)
c = a.reshape(2, -1)
print(b.shape, c.shape)
```
- [x] `(6, 1) (2, 3)`
- [ ] `(1, 6) (2, 3)`
- [ ] `(6, 1) (3, 2)`
- [ ] `(6,) (2, 3)`
? `-1` betyr «regn ut selv»: `reshape(-1, 1)` gir `(6, 1)` og `reshape(2, -1)` gir `(2, 3)`.
:::

::: quiz
Et lag i et nevralt nett regnes ut med matrisemultiplikasjon. Hva skriver koden ut?
```kode
import numpy as np
X = np.ones((5, 3))
W = np.ones((3, 2))
b = np.ones(2)
out = X @ W + b
print(out.shape, out[0, 0])
```
- [ ] `(5, 2) 3.0`
- [x] `(5, 2) 4.0`
- [ ] `(3, 2) 4.0`
- [ ] `ValueError` fordi `b` har feil shape
? `(5, 3) @ (3, 2)` gir `(5, 2)` der hvert element er 1 + 1 + 1 = 3, og bias 1 broadcastes på alle rader, så 4.0.
:::

::: quiz
Hva skriver koden med `sum` langs akser ut?
```kode
import numpy as np
M = np.array([[1, 2, 3],
              [4, 5, 6]])
print(M.sum(axis=0), M.sum(axis=1))
```
- [ ] `[6 15]` og `[5 7 9]`
- [ ] `21` og `21`
- [x] `[5 7 9]` og `[6 15]`
- [ ] `[1 4]` og `[1 2 3]`
? `axis=0` summerer nedover (ett tall per kolonne), og `axis=1` summerer bortover (ett tall per rad).
:::

::: quiz
Modellen har gitt sannsynligheter for tre klasser til tre datapunkter. Hva skrives ut?
```kode
import numpy as np
probs = np.array([[0.1, 0.7, 0.2],
                  [0.3, 0.3, 0.4],
                  [0.8, 0.1, 0.1]])
print(np.argmax(probs, axis=1))
```
- [ ] `[2 0 1]`
- [ ] `[0.7 0.4 0.8]`
- [ ] `[1 0 0]`
- [x] `[1 2 0]`
? `argmax(axis=1)` gir indeksen til største verdi i hver rad, altså predikert klasse per datapunkt.
:::

::: quiz
En vanlig feil når man regner ut MSE. Hva skrives ut?
```kode
import numpy as np
y_test = np.array([3.0, 5.0, 4.0])          # shape (3,)
y_pred = np.array([[3.0], [5.0], [4.0]])    # shape (3, 1)
diff = y_test - y_pred
print(diff.shape, np.mean(diff ** 2) > 0)
```
- [x] `(3, 3) True`
- [ ] `(3,) False`
- [ ] `(3, 1) False`
- [ ] `ValueError`, fordi shapene ikke passer
? `(3,) - (3, 1)` broadcastes til `(3, 3)` uten feilmelding, så MSE blir større enn 0 selv med perfekte prediksjoner.
:::

::: quiz
Hva skriver koden med boolsk maske ut?
```kode
import numpy as np
a = np.array([3, -1, 4, -1, 5])
print(a[a > 0].sum(), (a < 0).mean())
```
- [ ] `12 2`
- [x] `12 0.4`
- [ ] `9 0.4`
- [ ] `3 0.4`
? `a[a > 0]` er `[3 4 5]` med sum 12, og snittet av masken `a < 0` er andelen `True`, 2/5 = 0.4.
:::

::: quiz
Hvor mange minibatcher blir det? Hva skriver koden ut?
```kode
n = 100
batch_size = 32
starts = list(range(0, n, batch_size))
print(starts, len(starts))
```
- [ ] `[0, 32, 64] 3`
- [ ] `[32, 64, 96] 3`
- [x] `[0, 32, 64, 96] 4`
- [ ] `[0, 32, 64, 96, 128] 5`
? `range(0, 100, 32)` starter på 0 og stopper før 100, så det blir 4 batcher der den siste bare har 4 punkter.
:::

## Lister, kopier og views

`b = a` kopierer aldri, et NumPy-slice er et view, og boolsk indeksering lager en kopi.

::: quiz
Hva skriver koden med `b = a` og `a.copy()` ut?
```kode
a = [1, 2, 3]
b = a
c = a.copy()
b.append(4)
print(len(a), len(c))
```
- [ ] `3 3`
- [x] `4 3`
- [ ] `3 4`
- [ ] `4 4`
? `b = a` er samme liste, så `append` via `b` endrer `a`, mens `c` er en ekte kopi.
:::

::: quiz
Slicing av et NumPy-array. Hva skrives ut?
```kode
import numpy as np
a = np.arange(5)
b = a[1:3]
b[0] = 99
print(a)
```
- [ ] `[0 1 2 3 4]`
- [ ] `[99 1 2 3 4]`
- [ ] `[0 99 99 3 4]`
- [x] `[0 99 2 3 4]`
? Et NumPy-slice er et view, så `b[0]` er det samme elementet som `a[1]`.
:::

::: quiz
Boolsk indeksering og vanlig tilordning. Hva skrives ut?
```kode
import numpy as np
a = np.array([1, 5, 2, 8])
b = a[a > 2]
b[0] = 0
c = a
c[0] = -1
print(a)
```
- [x] `[-1 5 2 8]`
- [ ] `[-1 0 2 8]`
- [ ] `[1 0 2 8]`
- [ ] `[1 5 2 8]`
? Boolsk indeksering lager en kopi, så `b[0] = 0` endrer ikke `a`, men `c = a` er samme array.
:::

::: quiz
En funksjon endrer en liste. Hva skrives ut?
```kode
def skaler(liste):
    for i in range(len(liste)):
        liste[i] = liste[i] / 10

v = [10, 20]
resultat = skaler(v)
print(resultat, v)
```
- [ ] `[1.0, 2.0] [10, 20]`
- [ ] `[1.0, 2.0] [1.0, 2.0]`
- [x] `None [1.0, 2.0]`
- [ ] `None [10, 20]`
? Uten `return` gir funksjonen `None`, men den endrer listen `v` på stedet.
:::

## pandas og formatering

::: quiz
Overlevelse per kjønn med `groupby`. Hva skrives ut?
```kode
import pandas as pd
df = pd.DataFrame({"kjonn": ["m", "k", "k", "m", "k"],
                   "overlevde": [0, 1, 1, 1, 0]})
rate = df.groupby("kjonn")["overlevde"].mean()
print(round(rate["k"], 2), round(rate["m"], 2))
```
- [x] `0.67 0.5`
- [ ] `0.5 0.67`
- [ ] `2 1`
- [ ] `0.6 0.4`
? Gruppen `"k"` har snitt 2/3 og `"m"` har snitt 1/2 (`2 1` ville vært `sum()`).
:::

::: quiz
Klassefordeling med `value_counts(normalize=True)`. Hva skrives ut?
```kode
import pandas as pd
y = pd.Series(["ja", "nei", "nei", "nei", "ja", "nei", "nei", "nei"])
andel = y.value_counts(normalize=True)
print(andel["nei"], andel.index[0])
```
- [ ] `6 nei`
- [ ] `0.75 ja`
- [x] `0.75 nei`
- [ ] `0.25 ja`
? `normalize=True` gir andeler (6/8 = 0.75) sortert fra størst til minst, så første indeks er `"nei"`.
:::

::: quiz
`loc` og `iloc` med en indeks som ikke starter på 0. Hva skrives ut?
```kode
import pandas as pd
df = pd.DataFrame({"x": [10, 20, 30, 40]}, index=[3, 2, 1, 0])
print(df.loc[1, "x"], df.iloc[1, 0])
```
- [ ] `20 20`
- [x] `30 20`
- [ ] `20 30`
- [ ] `30 30`
? `loc` bruker etiketten (etikett 1 har x = 30), mens `iloc` bruker posisjonen (posisjon 1 har x = 20).
:::

::: quiz
Hva skriver f-strengen ut?
```kode
acc = 0.87654
rate = 0.1234
print(f"Accuracy: {acc:.2f}, rate: {rate:.1%}")
```
- [ ] `Accuracy: 0.87, rate: 0.1%`
- [x] `Accuracy: 0.88, rate: 12.3%`
- [ ] `Accuracy: 0.88, rate: 0.12%`
- [ ] `Accuracy: 87.65, rate: 12.3%`
? `:.2f` runder til to desimaler, og `:.1%` ganger med 100, runder til én desimal og legger til prosenttegn.
:::

::: quiz
Sammenligning av to Python-lister. Hva skrives ut?
```kode
import numpy as np
y_true = [1, 0, 1, 1]
y_pred = [1, 1, 1, 1]
print(np.mean(y_true == y_pred), np.mean(np.array(y_true) == np.array(y_pred)))
```
- [x] `0.0 0.75`
- [ ] `0.75 0.75`
- [ ] `0.75 0.0`
- [ ] `False 0.75`
? `==` mellom to lister gir én `bool` (`False`, snitt 0.0), mens mellom to arrays er det elementvis (3/4 = 0.75).
:::

::: quiz
Telling med en dictionary, slik Naive Bayes trenger. Hva skrives ut?
```kode
teller = {}
for ord_ in ["ja", "nei", "ja", "ja"]:
    teller[ord_] = teller.get(ord_, 0) + 1
print(teller)
```
- [ ] `{'ja': 1, 'nei': 1}`
- [ ] `KeyError: 'ja'`
- [ ] `{'ja': 3, 'nei': 1, 'ja': 3}`
- [x] `{'ja': 3, 'nei': 1}`
? `teller.get(ord_, 0)` gir 0 første gang et ord dukker opp, så det blir ingen `KeyError`, og hver nøkkel finnes bare én gang.
:::

::: oppgave Rett MSE-feilen
`y_pred` har shape `(4, 1)` og `y_test` shape `(4,)`. Rett linjen slik at `mse` blir 0.125.
```start
import numpy as np
y_test = np.array([3.0, 5.0, 4.0, 6.0])
y_pred = np.array([[2.5], [5.0], [4.5], [6.0]])   # shape (4, 1)

mse = np.mean((y_test - y_pred) ** 2)
```
```losning
import numpy as np
y_test = np.array([3.0, 5.0, 4.0, 6.0])
y_pred = np.array([[2.5], [5.0], [4.5], [6.0]])   # shape (4, 1)

mse = np.mean((y_test - y_pred.ravel()) ** 2)
```
```test
assert np.ndim(mse) == 0, "mse skal være ett tall."
assert abs(mse - 0.125) < 1e-9, f"mse ble {float(mse):.4f}, men skal være 0.125. Sjekk shape på y_test - y_pred: blir den (4,) eller (4, 4)?"
```
:::

--- LEKSJON 13.3 | finn-feilen | Finn feilen
Koden kjører ofte fint, men lærer av feil data, evalueres på feil data eller oppdaterer i feil retning.

## Lekkasje du kan se

Alt som lærer fra data (scaler, imputer, feature-utvalg, hyperparametre) skal bare lære fra treningsdelen.

::: oppgave Rett lekkasjen
Koden skalerer før splitten. Skriv den om slik at `scaler` bare tilpasses `X_train`, og lag `X_train_scaled` og `X_test_scaled` (behold `random_state=0`).
```start
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(1)
X = rng.normal(5, 3, size=(100, 2))
y = rng.integers(0, 2, size=100)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # rett lekkasjen
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
```
```losning
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(1)
X = rng.normal(5, 3, size=(100, 2))
y = rng.integers(0, 2, size=100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
```test
_Xtr, _Xte, _, _ = train_test_split(X, y, test_size=0.2, random_state=0)
assert np.shape(X_train_scaled) == (80, 2) and np.shape(X_test_scaled) == (20, 2), "X_train_scaled skal ha shape (80, 2) og X_test_scaled (20, 2)."
_mu, _sd = _Xtr.mean(axis=0), _Xtr.std(axis=0)
assert np.allclose(X_train_scaled, (_Xtr - _mu) / _sd), "X_train_scaled skal skaleres med gjennomsnitt og standardavvik fra X_train alene. Splitt først (med random_state=0), så fit_transform på X_train."
assert np.allclose(X_test_scaled, (_Xte - _mu) / _sd), "X_test_scaled skal bruke scaleren som er tilpasset X_train: scaler.transform(X_test)."
```
:::

## Lekkasje og evaluering

::: quiz
Hva er problemet med denne koden?
```kode
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
model = LogisticRegression().fit(X_train, y_train)
print(model.score(X_test, y_test))
```
- [ ] `fit_transform` skal aldri brukes, bare `fit` og `transform` hver for seg
- [x] Scaleren har regnet gjennomsnitt og standardavvik med testradene, så testresultatet er påvirket av testdataene
- [ ] `train_test_split` må alltid ha `stratify=y`
- [ ] Logistisk regresjon tåler ikke skalerte data
? Skaleringen skjer før splitten, så testradene har vært med på å bestemme gjennomsnitt og standardavvik.
:::

::: quiz
Manglende verdier fylles inn med gjennomsnittet. Hva er feilen?
```kode
imp = SimpleImputer(strategy="mean")
X_train = imp.fit_transform(X_train)
X_test = imp.fit_transform(X_test)
```
- [ ] `SimpleImputer` kan bare brukes på treningsdata
- [ ] `strategy="mean"` er ulovlig, det må være `"median"`
- [ ] Ingenting, begge datasettene får fylt inn hull
- [x] Hullene i testdataene fylles med test-gjennomsnittet. Siste linje skal være `imp.transform(X_test)`
? Imputeren skal lære gjennomsnittet fra treningsdata, så siste linje skal bruke `imp.transform(X_test)`.
:::

::: quiz
Du har 500 features og vil bare bruke de 20 som korrelerer mest med målet. Hva er feilen?
```kode
korr = df.corr()["y"].abs().sort_values(ascending=False)
beste = korr.index[1:21]          # indeks 0 er y selv
X_train, X_test, y_train, y_test = train_test_split(df[beste], df["y"], test_size=0.2)
model = LinearRegression().fit(X_train, y_train)
print(model.score(X_test, y_test))
```
- [x] Featurene er valgt med `y` fra hele datasettet, også testradene, så testscoren blir for optimistisk
- [ ] `korr.index[1:21]` gir bare 19 features
- [ ] `df.corr()` kan ikke brukes på målvariabelen
- [ ] `LinearRegression` krever at featurene er skalert
? Feature-utvalget har brukt `y` for testradene, så features må velges på treningsdelen etter splitten (`[1:21]` gir 20 features).
:::

::: quiz
Hva er galt med denne evalueringen?
```kode
clf = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = clf.predict(X_train)
print("Forventet ytelse på nye data:", accuracy_score(y_train, y_pred))
```
- [ ] Argumentene til `accuracy_score` står i feil rekkefølge
- [ ] `DecisionTreeClassifier` må ha `max_depth` for å kunne predikere
- [x] Modellen evalueres på dataene den er trent på. Et tre uten dybdebegrensning får nesten 1.0 der, uansett hvor godt det generaliserer
- [ ] `predict` må erstattes med `predict_proba`
? Et tre uten dybdebegrensning kan pugge treningsdataene, så evalueringen må bruke `X_test` og `y_test`.
:::

::: quiz
Du velger `max_depth` slik. Hva er problemet?
```kode
beste = None
for d in [1, 2, 3, 5, 10]:
    acc = DecisionTreeClassifier(max_depth=d).fit(X_train, y_train).score(X_test, y_test)
    if beste is None or acc > beste[0]:
        beste = (acc, d)
print("Testnøyaktighet for valgt modell:", beste[0])
```
- [ ] `score` gir R², ikke accuracy, for klassifiseringstrær
- [ ] Løkka trener bare én modell
- [ ] `beste[0]` er dybden, ikke accuracy
- [x] Testdataene brukes til å velge hyperparameter, så den rapporterte testnøyaktigheten er for optimistisk
? Testsettet er brukt til å velge `max_depth` og er derfor ikke lenger usett (bruk et valideringssett eller `GridSearchCV`).
:::

## PyTorch

::: quiz
Treningsløkka under kjører, men tapet oppfører seg rart og eksploderer etter hvert. Hva mangler?
```kode
for epoch in range(100):
    for X_batch, y_batch in loader:
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
```
- [ ] `model.eval()` før løkka
- [x] `optimizer.zero_grad()`, så gradientene akkumuleres fra batch til batch
- [ ] `loss.item()` etter `backward`
- [ ] `torch.no_grad()` rundt `loss.backward()`
? `backward` legger nye gradienter til i `.grad`, så uten `optimizer.zero_grad()` hoper de seg opp fra batch til batch.
:::

::: quiz
Et binært klassifiseringsnett lærer dårlig. Hva er feilen?
```kode
model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(model(X).squeeze(1), y)
```
- [ ] `nn.ReLU` kan ikke brukes i skjulte lag
- [ ] Output-laget må ha 2 noder for to klasser
- [ ] `squeeze(1)` fjerner batch-dimensjonen
- [x] `BCEWithLogitsLoss` bruker sigmoid selv, så her brukes sigmoid to ganger
? `BCEWithLogitsLoss` forventer logits og tar sigmoid selv, så `nn.Sigmoid()` til slutt skal fjernes.
:::

::: quiz
Et nett for 10 klasser gir `RuntimeError: expected target dtype to be Long or Byte, but got Float`. Hva er feilen?
```kode
logits = model(X_batch)                                # shape (32, 10)
y_batch = torch.tensor(y_np, dtype=torch.float32)      # klasser 0-9, shape (32,)
loss = nn.CrossEntropyLoss()(logits, y_batch)
```
- [ ] Det mangler `torch.softmax(logits, dim=1)` før tapet
- [x] Klasse-labels til `CrossEntropyLoss` skal være heltall: `dtype=torch.long`
- [ ] `CrossEntropyLoss` kan bare brukes med to klasser
- [ ] `logits` må ha shape `(32,)`
? Labels med shape `(32,)` til `CrossEntropyLoss` er klasseindekser og må være `torch.long`, og softmax skal ikke brukes først.
:::

::: quiz
Nettet har dropout. Når du kjører evalueringen to ganger etter hverandre, får du forskjellig accuracy. Hvorfor?
```kode
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 2))
# ... trening ...
logits = model(X_test)
acc = (logits.argmax(dim=1) == y_test).float().mean()
```
- [x] Modellen står fortsatt i treningsmodus, så dropout slår av tilfeldige noder. Kall `model.eval()` (og bruk `torch.no_grad()`) før evaluering
- [ ] `argmax(dim=1)` velger tilfeldig når to logits er like
- [ ] `float().mean()` er ikke deterministisk på CPU
- [ ] `nn.Dropout` endrer vektene permanent hver gang modellen kalles
? Uten `model.eval()` står dropout fortsatt på og nuller tilfeldige noder ved hvert kall.
:::

## Reinforcement learning

::: quiz
Q-læringen under lærer nesten ingenting. Hva er feilen?
```kode
action = update_action(env, state, explore_rate, q_table)
_, reward, terminated, _ = env.step(action)
next_state = env.get_state_id()
target = reward + gamma * np.max(q_table[state])
q_table[state, action] += alpha * (target - q_table[state, action])
state = next_state
```
- [ ] `alpha` og `gamma` har byttet plass
- [ ] `np.max` skal være `np.argmax`
- [x] Fremtidig verdi skal hentes fra `q_table[next_state]`, ikke `q_table[state]`
- [ ] `state = next_state` skal stå før oppdateringen
? Fremtidig verdi skal hentes fra tilstanden agenten havnet i, `q_table[next_state]`, og `np.argmax` ville gitt en indeks i stedet for en verdi.
:::

::: quiz
ε starter på 1.0 og synker mot 0.01 i løpet av treningen. Agenten oppfører seg bra i starten og helt tilfeldig mot slutten. Hvorfor?
```kode
def velg_handling(q_table, state, epsilon):
    if np.random.random() > epsilon:
        return np.random.randint(4)
    return int(np.argmax(q_table[state]))
```
- [ ] `np.random.randint(4)` kan returnere 4
- [ ] `np.argmax` må ha `axis=1`
- [ ] ε skal øke, ikke synke, i løpet av treningen
- [x] Sammenligningen er snudd: med `>` utforsker agenten med sannsynlighet 1 - ε i stedet for ε
? Med `>` utforsker agenten med sannsynlighet 1 - ε, så riktig test er `if np.random.random() < epsilon:`.
:::

::: quiz
Belønning gis bare i målet (10). Etter trening er alle Q-verdier 0, bortsett fra for handlinger i ruta rett ved siden av målet. Hva er den mest sannsynlige årsaken?
```kode
GAMMA = 0.0
ALPHA = 0.1
q[s, a] += ALPHA * (r + GAMMA * np.max(q[s_next]) - q[s, a])
```
- [x] Med γ = 0 bryr agenten seg bare om umiddelbar belønning, så verdien av målet sprer seg aldri bakover
- [ ] α = 0.1 er for lavt til at noe læres
- [ ] `np.max(q[s_next])` skal være `np.min(q[s_next])`
- [ ] Belønningen 10 er for stor, den må være 1
? Det er leddet `GAMMA * np.max(q[s_next])` som sender verdi bakover fra målet, og med γ = 0 forsvinner det.
:::

## Skala og koding

::: quiz
Du klynger kunder etter inntekt (i kroner, rundt 300 000 til 900 000) og alder (20 til 80 år). Klyngene ser ut til å bare avhenge av inntekt. Hvorfor?
```kode
df = pd.DataFrame({"inntekt_kr": inntekt, "alder": alder})
labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(df)
```
- [ ] KMeans kan bare bruke én feature
- [ ] `n_init=10` gjør at bare første kolonne brukes
- [x] KMeans bruker euklidsk avstand, og inntekt har tusenvis av ganger større skala enn alder. Skaler først (for eksempel `StandardScaler`)
- [ ] Alder er en kategorisk variabel og ignoreres
? KMeans bruker euklidsk avstand, så inntekt dominerer alder fullstendig til featurene er skalert.
:::

::: quiz
Du predikerer boligpris med lineær regresjon. Hva er det metodiske problemet?
```kode
df["by"] = LabelEncoder().fit_transform(df["by"])   # Bergen=0, Oslo=1, Trondheim=2
model = LinearRegression().fit(df[["by", "areal"]], df["pris"])
```
- [ ] `LabelEncoder` kan ikke brukes på tekst
- [ ] Lineær regresjon krever at alle features er mellom 0 og 1
- [ ] `areal` må også label-encodes
- [x] By er nominell, men koden gjør den til et tall med rekkefølge og like avstander. Modellen tror Oslo ligger «midt mellom» Bergen og Trondheim. Bruk one-hot-koding
? Label encoding gir byene rekkefølge og like avstander, så en lineær modell trenger one-hot-koding (`pd.get_dummies` eller `OneHotEncoder`).
:::

::: quiz
Lineær regresjon med gradient descent. Tapet *øker* for hver iterasjon. Hvorfor?
```kode
for _ in range(1000):
    y_pred = X @ w + b
    grad_w = 2 * X.T @ (y_pred - y) / len(y)
    grad_b = 2 * np.mean(y_pred - y)
    w = w + lr * grad_w
    b = b + lr * grad_b
```
- [ ] Gradienten skal deles på `len(X.T)`
- [x] Parametrene skal oppdateres *mot* gradienten: `w = w - lr * grad_w` og `b = b - lr * grad_b`
- [ ] `X.T @ (y_pred - y)` skal være `X @ (y_pred - y)`
- [ ] Læringsraten må økes
? Gradienten peker dit tapet øker, så gradient descent må trekke fra: `w = w - lr * grad_w`.
:::

::: oppgave Rett Q-oppdateringen
Rett `q_oppdatering` slik at den bruker `q_table[next_state]` og bare `reward` når `terminated` er `True`, og returnerer tabellen.
```start
import numpy as np

def q_oppdatering(q_table, state, action, reward, next_state, terminated, alpha, gamma):
    target = reward + gamma * np.max(q_table[state])
    q_table[state, action] += alpha * (target - q_table[state, action])
    return q_table
```
```losning
import numpy as np

def q_oppdatering(q_table, state, action, reward, next_state, terminated, alpha, gamma):
    if terminated:
        target = reward
    else:
        target = reward + gamma * np.max(q_table[next_state])
    q_table[state, action] += alpha * (target - q_table[state, action])
    return q_table
```
```test
q = np.zeros((3, 2))
q[0] = [1.0, 0.0]
q[1] = [2.0, 4.0]
q = q_oppdatering(q, 0, 0, 1.0, 1, False, 0.5, 0.9)
assert q is not None, "Husk å returnere q_table."
assert abs(q[0, 0] - 2.8) < 1e-9, f"q[0, 0] ble {q[0, 0]:.3f}, men skal være 1 + 0.5 * (1 + 0.9 * 4 - 1) = 2.8. Bruker du q_table[next_state]?"
q2 = np.zeros((3, 2))
q2[2] = [5.0, 5.0]
q2 = q_oppdatering(q2, 1, 1, 10.0, 2, True, 1.0, 0.9)
assert abs(q2[1, 1] - 10.0) < 1e-9, f"Når terminated er True skal target være bare reward (10), men q ble {q2[1, 1]:.2f}."
```
:::

--- LEKSJON 13.4 | hva-gjor-funksjonen | Hva gjør denne funksjonen?
Les signaturen og `return` først, så løkkene, og sett til slutt inn konkrete tall i linjen spørsmålet gjelder.

## Logistisk regresjon fra notatene

::: quiz
I `fit` fra notatene har `x` shape `(200, 3)`. Hvilken shape har `lin_model`?
```kode
def fit(self, x, y):
    self.weights = np.zeros(x.shape[1])
    self.bias = 0
    for _ in range(self.epochs):
        lin_model = np.matmul(self.weights, x.transpose()) + self.bias
        y_pred = self._sigmoid(lin_model)
        grad_w, grad_b = self.compute_gradients(x, y, y_pred)
        self.update_parameters(grad_w, grad_b)
        loss = self._compute_loss(y, y_pred)
        pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred]
        self.train_accuracies.append(accuracy(y, pred_to_class))
        self.losses.append(loss)
```
- [ ] `(3,)`
- [ ] `(200, 3)`
- [x] `(200,)`
- [ ] `(3, 200)`
? `(3,) @ (3, 200)` gir én lineær score per datapunkt, altså shape `(200,)`.
:::

::: quiz
Samme `fit` som over kjøres med `epochs=30`. Hva inneholder `self.losses` etterpå?
```kode
for _ in range(self.epochs):
    lin_model = np.matmul(self.weights, x.transpose()) + self.bias
    y_pred = self._sigmoid(lin_model)
    grad_w, grad_b = self.compute_gradients(x, y, y_pred)
    self.update_parameters(grad_w, grad_b)
    loss = self._compute_loss(y, y_pred)
    self.losses.append(loss)
```
- [ ] Ett tall: tapet etter siste epoke
- [ ] 30 tall beregnet på testdataene
- [ ] Ett tall per datapunkt i `x`
- [x] 30 tall, ett per epoke, beregnet på treningsdataene
? `append` kalles én gang per epoke med treningsdataene som ble gitt til `fit`, så det blir 30 tall.
:::

::: quiz
`predict` fra notatene. Hva returnerer den?
```kode
def predict(self, x):
    lin_model = np.matmul(x, self.weights) + self.bias
    y_pred = self._sigmoid(lin_model)
    return [1 if _y > 0.5 else 0 for _y in y_pred]
```
- [x] En Python-liste med 0 og 1, én per datapunkt
- [ ] Et NumPy-array med sannsynligheter
- [ ] Ett tall: andelen datapunkter i klasse 1
- [ ] Et NumPy-array med logits
? List comprehension lager en Python-liste der hver sannsynlighet over 0.5 blir 1 og resten 0.
:::

## Q-læring fra prosjekt 4

::: quiz
`update_explore_rate` fra prosjekt 4. Hva skrives ut?
```kode
import numpy as np

def update_explore_rate(episode, epsilon_min, max_episodes, epsilon_max=1.0):
    return epsilon_min + (epsilon_max - epsilon_min) * np.exp(-5 * episode / max_episodes)

print(update_explore_rate(0, 0.01, 500), round(update_explore_rate(500, 0.01, 500), 3))
```
- [ ] `0.01 1.0`
- [x] `1.0 0.017`
- [ ] `1.0 0.0`
- [ ] `0.99 0.01`
? I episode 0 er `np.exp(0) = 1`, så ε = 1.0, og i siste episode er ε = 0.01 + 0.99 · `np.exp(-5)`, omtrent 0.017.
:::

::: quiz
`q_table[7]` er `[0.0, 2.5, 2.5, -1.0]`. Hva returnerer `update_action(env, 7, 0, q_table)`?
```kode
def update_action(env, state, explore_rate, q_table):
    if np.random.random() < explore_rate:
        return int(np.random.randint(env.action_space.n))
    return int(np.argmax(q_table[state]))
```
- [ ] 1 eller 2, tilfeldig
- [ ] 2, fordi `argmax` velger den siste av like verdier
- [x] Alltid 1
- [ ] En tilfeldig handling mellom 0 og 3
? Med `explore_rate = 0` utnytter agenten alltid, og `np.argmax` velger den første indeksen ved likhet, altså 1.
:::

::: quiz
Linjen under er fra `train_q_learning` i prosjekt 4. Hvorfor brukes bare `reward` når `terminated` er `True`?
```kode
target = reward if terminated else reward + gamma * np.max(q_table[next_state])
q_table[state, action] += learning_rate * (target - q_table[state, action])
```
- [ ] For at γ skal bli 0 resten av treningen
- [ ] Fordi `np.max` krasjer på en terminal tilstand
- [ ] For å straffe agenten for å avslutte episoden
- [x] Etter en terminal tilstand kommer ingen flere belønninger, så fremtidig verdi er 0
? Etter en terminal tilstand kommer ingen flere belønninger, så fremtidig verdi skal være 0.
:::

::: quiz
`get_state_id` fra miljøet i prosjekt 4. Kartet er 12 x 12, og agenten står i rad 2, kolonne 5. Hva returneres?
```kode
def get_state_id(self):
    rows, cols = self.observation.shape
    agent_pos = np.argwhere(self.observation == self.start_class)[0]
    agent_row, agent_col = agent_pos[0], agent_pos[1]
    grid_index = (agent_row * cols) + agent_col
    return grid_index
```
- [x] 29
- [ ] 17
- [ ] 60
- [ ] 7
? Rad 2 starter på 2 · 12 = 24, og kolonne 5 gir 24 + 5 = 29.
:::

## DQN fra notatene

::: quiz
I `learn` har `predicted_q` shape `(32, 4)` og `actions` shape `(32, 1)`. Hva er `y_pred = torch.gather(predicted_q, 1, actions)`?
```kode
predicted_q = self.q_network(states)
y_pred = torch.gather(predicted_q, 1, actions)
```
- [ ] Den høyeste Q-verdien i hver rad, shape `(32, 1)`
- [x] Q-verdien for handlingen som faktisk ble valgt i hver overgang, shape `(32, 1)`
- [ ] Indeksen til beste handling i hver rad, shape `(32,)`
- [ ] Alle Q-verdiene, shape `(32, 4)`
? `gather` langs dimensjon 1 plukker ut Q-verdien i kolonnen som `actions` peker på, én per rad.
:::

::: quiz
Hvorfor står target-beregningen i `learn` inne i `with torch.no_grad():`?
```kode
with torch.no_grad():
    next_target_q_value = self.target_network(next_states).max(dim=1, keepdim=True).values
    next_target_q_value = torch.where(dones, torch.zeros_like(next_target_q_value), next_target_q_value)
y_train = rewards + (self.discount * next_target_q_value)
```
- [x] Target skal behandles som en fast fasit. Gradienten skal bare gå gjennom `q_network`, ikke gjennom target-nettverket
- [ ] For at target-nettverket skal oppdateres oftere
- [ ] Fordi `torch.where` ikke virker med gradienter
- [ ] For å slå av dropout i Q-nettverket
? Med `no_grad` blir `y_train` en fast fasit, så gradienten går bare gjennom `q_network`.
:::

::: quiz
I samme `learn`: en overgang har `reward = 1.0`, `dones = True`, og target-nettverket gir 2.0 som beste Q-verdi for neste tilstand. `discount` er 0.9. Hva blir `y_train` for denne overgangen?
```kode
next_target_q_value = torch.where(dones, torch.zeros_like(next_target_q_value), next_target_q_value)
y_train = rewards + (self.discount * next_target_q_value)
```
- [ ] 2.8
- [ ] 2.0
- [x] 1.0
- [ ] 0.0
? `torch.where(dones, 0, ...)` setter fremtidig verdi til 0, så `y_train` blir 1.0 + 0.9 · 0 = 1.0.
:::

## Nevrale nett fra prosjekt 3

::: quiz
`make_mlp` fra prosjekt 3. Hva skrives ut?
```kode
def make_mlp(n_in, width, depth, act):
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), act()]
        d = width
    return nn.Sequential(*layers, nn.Linear(d, 1))

net = make_mlp(2, 16, 2, nn.Tanh)
print(len(net), sum(p.numel() for p in net.parameters()))
```
- [ ] `3 337`
- [ ] `5 305`
- [x] `5 337`
- [ ] `4 320`
? Løkka gir `Linear` + `Tanh` to ganger pluss et siste `Linear` (5 moduler), og parametrene er 48 + 272 + 17 = 337.
:::

::: quiz
Utdrag fra `train_net` i prosjekt 3. Hvilke vekter har `net` etter at funksjonen er ferdig?
```kode
best = (np.inf, None, 0)
for epoch in range(epochs):
    ...  # trening og utregning av history["val"]
    if history["val"][-1] < best[0]:
        best = (history["val"][-1], {k: v.clone() for k, v in net.state_dict().items()}, epoch)
    elif patience is not None and epoch - best[2] > patience:
        break
net.load_state_dict(best[1])
```
- [ ] Vektene fra siste epoke
- [ ] Vektene fra epoken med lavest *trenings*tap
- [ ] Vektene fra epoke 0
- [x] Vektene fra epoken med lavest valideringstap
? `best` lagrer en kopi av vektene bare når valideringstapet blir lavere, og de vektene lastes inn til slutt.
:::

::: quiz
Hvorfor brukes `v.clone()` når vektene lagres i `best` i `train_net`?
```kode
best = (history["val"][-1], {k: v.clone() for k, v in net.state_dict().items()}, epoch)
```
- [ ] For å gjøre vektene om til NumPy-arrays
- [x] Uten `clone` ville ordboka peke på de samme tensorene som fortsetter å endres under treningen, så «beste» vekter ville blitt overskrevet
- [ ] For å fjerne gradientene fra vektene
- [ ] Fordi `load_state_dict` bare godtar klonede tensorer
? `state_dict()` gir referanser til tensorer som fortsetter å endres, så `clone()` trengs for å få en ekte kopi.
:::

::: quiz
Fra `cls_run` i prosjekt 3. Nettet er trent med `BCEWithLogitsLoss`. Hvorfor sammenlignes output med 0 og ikke med 0.5?
```kode
val_acc = ((net(Xc_train[c_val_idx]).squeeze(1) > 0).numpy() == yc_train[c_val_idx].numpy()).mean()
```
- [ ] Fordi labels er -1 og 1
- [ ] Fordi `squeeze(1)` trekker fra 0.5
- [x] Nettet gir logits, og logit 0 tilsvarer sannsynlighet sigmoid(0) = 0.5
- [ ] Det er en feil, terskelen skal være 0.5
? Nettet gir logits, og `logit > 0` er det samme som `sigmoid(logit) > 0.5`.
:::

## Klynger, Bayes og beslutningsgrenser

::: quiz
`calculate_clustering_accuracy` fra prosjekt 3. Hva skrives ut?
```kode
def calculate_clustering_accuracy(true_labels, cluster_labels):
    contingency = sklearn.metrics.cluster.contingency_matrix(true_labels, cluster_labels)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-contingency)
    correct = contingency[row_ind, col_ind].sum()
    return correct / true_labels.size

print(calculate_clustering_accuracy(np.array([0, 1, 2, 0, 1, 2]),
                                    np.array([1, 2, 0, 1, 2, 0])))
```
- [ ] 0.0
- [ ] 0.333
- [ ] 0.5
- [x] 1.0
? Funksjonen finner den beste koblingen mellom klynger og klasser, og da stemmer alle 6 (naiv `==` ville gitt 0.0).
:::

::: quiz
I `calculate_clustering_accuracy` står det `linear_sum_assignment(-contingency)` med minus. Hvorfor?
```kode
row_ind, col_ind = scipy.optimize.linear_sum_assignment(-contingency)
```
- [ ] For å gjøre alle tellingene positive
- [ ] For å bytte om rader og kolonner
- [ ] Minus snur rekkefølgen på klyngene
- [x] `linear_sum_assignment` *minimerer* totalkostnad. Vi vil *maksimere* antall treff, og det er det samme som å minimere minus antall treff
? `linear_sum_assignment` minimerer kostnad, og å maksimere antall treff er det samme som å minimere minus antall treff.
:::

::: quiz
Naive Bayes for hånd, med telling. Hva skrives ut?
```kode
data = [("sol", "ja"), ("regn", "nei"), ("sol", "ja"),
        ("sol", "nei"), ("regn", "nei"), ("overskyet", "ja")]

def p_gitt_klasse(vaer, klasse):
    rader = [v for v, k in data if k == klasse]
    return rader.count(vaer) / len(rader)

def prior(klasse):
    return sum(1 for _, k in data if k == klasse) / len(data)

print(round(prior("ja") * p_gitt_klasse("sol", "ja"), 3))
```
- [x] 0.333
- [ ] 0.5
- [ ] 0.667
- [ ] 0.75
? `prior("ja")` er 3/6 og P(sol | ja) er 2/3, så produktet er 1/3.
:::

::: quiz
Kode for å tegne beslutningsgrensen (prosjekt 2). Hva er `grid.shape` og `Z.shape` til slutt?
```kode
xx0, xx1 = np.meshgrid(np.arange(-1, 1, 0.5), np.arange(0, 3, 1.0))
grid = pd.DataFrame({"x0": xx0.ravel(), "x1": xx1.ravel()})
Z = clf.predict(grid)
Z = Z.reshape(xx0.shape)
```
- [ ] `(12, 2)` og `(4, 3)`
- [x] `(12, 2)` og `(3, 4)`
- [ ] `(2, 12)` og `(3, 4)`
- [ ] `(7, 2)` og `(12,)`
? `meshgrid` gir shape `(3, 4)`, `ravel` gir 12 punkter i `grid`, og `reshape(xx0.shape)` gjør `Z` til `(3, 4)` igjen.
:::

::: quiz
Perceptron fra notatene, på data med 2 features og 3 klasser. Hva skrives ut?
```kode
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0,
                           n_classes=3, n_clusters_per_class=1, random_state=0)
perceptron = Perceptron().fit(X, y)
print(perceptron.coef_.shape, perceptron.intercept_.shape)
```
- [ ] `(2, 3) (3,)`
- [ ] `(2,) (1,)`
- [ ] `(3, 2) (1,)`
- [x] `(3, 2) (3,)`
? Med tre klasser har sklearn én lineær modell per klasse: 3 rader med 2 vekter og 3 bias-ledd.
:::

::: quiz
Fra prosjekt 4 (XAI). `shap_values.values` har shape `(4128, 8)`. Hva er `importance`?
```kode
importance = np.abs(shap_values.values).mean(axis=0)
```
- [x] Gjennomsnittlig absolutt SHAP-verdi per feature, shape `(8,)`: et globalt mål på hvor mye hver feature påvirker prediksjonene
- [ ] Gjennomsnittlig prediksjon per testpunkt, shape `(4128,)`
- [ ] Gjennomsnittlig SHAP-verdi per feature, der positive og negative bidrag kansellerer hverandre
- [ ] Baseverdien til explaineren
? `mean(axis=0)` gir ett tall per feature, og absoluttverdien hindrer at positive og negative bidrag kansellerer hverandre.
:::

::: oppgave Rett get_state_id
`get_state_id` gir feil numre når antall rader og kolonner er ulikt. Rett feilen slik at rutene nummereres rad for rad.
```start
import numpy as np

def get_state_id(observation, start_class=1):
    rows, cols = observation.shape
    agent_row, agent_col = np.argwhere(observation == start_class)[0]
    return agent_row * rows + agent_col
```
```losning
import numpy as np

def get_state_id(observation, start_class=1):
    rows, cols = observation.shape
    agent_row, agent_col = np.argwhere(observation == start_class)[0]
    return agent_row * cols + agent_col
```
```test
obs = np.zeros((3, 5), dtype=int)
obs[2, 1] = 1
svar = get_state_id(obs)
assert svar == 11, f"For rad 2, kolonne 1 i et 3 x 5-kart skal svaret være 2 * 5 + 1 = 11, men ble {svar}."
ider = set()
for r in range(3):
    for c in range(5):
        o = np.zeros((3, 5), dtype=int)
        o[r, c] = 1
        ider.add(int(get_state_id(o)))
assert ider == set(range(15)), "Alle 15 rutene skal få hvert sitt nummer fra 0 til 14."
```
:::

--- LEKSJON 13.5 | velg-riktig-linje | Velg riktig kodelinje
Se hva som finnes før hullet og hva som brukes etter: hullet må lage nøyaktig det neste linje trenger.

## ML fra bunnen

::: quiz
Hvilken linje mangler i `predict_proba`?
```kode
def predict_proba(self, X):
    z = X @ self.weights + self.bias
    # ??? (manglende linje)
```
- [ ] `return np.exp(-z) / (1 + np.exp(-z))`
- [ ] `return (z > 0).astype(int)`
- [x] `return 1 / (1 + np.exp(-z))`
- [ ] `return 1 / (1 - np.exp(z))`
? Sigmoid er `1 / (1 + np.exp(-z))`, mens A gir 1 - σ(z), B gir klasser og D har feil fortegn.
:::

::: quiz
Hvilken linje mangler i `predict` når terskelen er et argument?
```kode
def predict(self, X, threshold=0.5):
    p = self.predict_proba(X)
    # ??? (manglende linje)
```
- [x] `return (p >= threshold).astype(int)`
- [ ] `return p[p >= threshold]`
- [ ] `return np.argmax(p)`
- [ ] `return np.round(p, threshold)`
? `p >= threshold` gir én boolsk verdi per datapunkt, og `astype(int)` gjør dem om til 1 og 0.
:::

::: quiz
Gradient descent for lineær regresjon. Hvilken linje mangler?
```kode
for _ in range(self.n_iterations):
    y_pred = X @ self.weights + self.bias
    error = y_pred - y
    grad_w = 2 * X.T @ error / len(y)
    grad_b = 2 * error.mean()
    # ??? (manglende linje)
    self.bias -= self.lr * grad_b
```
- [x] `self.weights -= self.lr * grad_w`
- [ ] `self.weights += self.lr * grad_w`
- [ ] `self.weights = grad_w`
- [ ] `self.weights -= self.lr * grad_b`
? Vektene skal flyttes mot gradienten med `-=`, akkurat som bias-linjen under.
:::

## Evaluering og sklearn

::: quiz
Du skal regne ut precision. Hva skal stå i hullet?
```kode
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
precision = # ??? (manglende uttrykk)
```
- [ ] `tp / (tp + fn)`
- [ ] `(tp + tn) / (tn + fp + fn + tp)`
- [ ] `fp / (fp + tn)`
- [x] `tp / (tp + fp)`
? Precision er TP / (TP + FP), mens A er recall, B er accuracy og C er false positive rate.
:::

::: quiz
Du vil kryssvalidere en modell med skalering uten lekkasje. Hva skal stå i hullet?
```kode
modell = # ??? (manglende uttrykk)
scores = cross_val_score(modell, X_train, y_train, cv=5)
print(scores.mean())
```
- [ ] `make_pipeline(LogisticRegression(), StandardScaler())`
- [x] `make_pipeline(StandardScaler(), LogisticRegression())`
- [ ] `LogisticRegression().fit(StandardScaler().fit_transform(X_train), y_train)`
- [ ] `StandardScaler().fit(X_train)`
? En pipeline tilpasser scaleren på nytt i hver fold, og transformasjonene skal stå før modellen.
:::

::: quiz
Du vil one-hot-kode kolonnen `by`. Hvilket uttrykk passer i hullet, slik at utskriften blir `(3, 3)`?
```kode
df = pd.DataFrame({"by": ["Oslo", "Bergen", "Oslo"], "areal": [50, 70, 40]})
X = # ??? (manglende uttrykk)
print(X.shape)
```
- [ ] `LabelEncoder().fit_transform(df["by"])`
- [ ] `df["by"].value_counts()`
- [x] `pd.get_dummies(df, columns=["by"])`
- [ ] `pd.get_dummies(df["areal"])`
? `get_dummies` erstatter `by` med `by_Bergen` og `by_Oslo`, så sammen med `areal` blir det `(3, 3)`.
:::

::: quiz
Hvilken linje passer i hullet for å telle støypunktene fra DBSCAN?
```kode
labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_scaled)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = # ??? (manglende uttrykk)
```
- [ ] `labels.min()`
- [ ] `len(labels) - n_clusters`
- [ ] `(labels == 0).sum()`
- [x] `(labels == -1).sum()`
? DBSCAN gir label -1 til støy, og `sum()` teller antall `True` i det boolske arrayet.
:::

## PyTorch og Keras

::: quiz
Hvilken linje mangler i treningsløkka?
```kode
for X_batch, y_batch in loader:
    optimizer.zero_grad()
    loss = loss_fn(model(X_batch), y_batch)
    # ??? (manglende linje)
    optimizer.step()
```
- [ ] `optimizer.backward()`
- [ ] `loss.step()`
- [ ] `model.backward(loss)`
- [x] `loss.backward()`
? `loss.backward()` regner ut gradientene som `optimizer.step()` så bruker.
:::

::: quiz
Du vil evaluere en trent modell på testdata. Hvilken linje mangler?
```kode
model.eval()
# ??? (manglende linje)
    logits = model(X_test)
    y_pred = logits.argmax(dim=1)
```
- [x] `with torch.no_grad():`
- [ ] `model.train()`
- [ ] `optimizer.zero_grad()`
- [ ] `for epoch in range(epochs):`
? Linjene under er innrykket, så hullet må være blokken `with torch.no_grad():`.
:::

::: quiz
Et Keras-nett skal klassifisere MNIST-sifre (10 klasser, heltallslabels 0 til 9). Hvilket lag mangler?
```kode
model = keras.Sequential([
    keras.Input(shape=(784,)),
    layers.Dense(128, activation="relu"),
    # ??? (manglende lag)
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
```
- [ ] `layers.Dense(1, activation="sigmoid")`
- [ ] `layers.Dense(10, activation="relu")`
- [x] `layers.Dense(10, activation="softmax")`
- [ ] `layers.Dense(784, activation="softmax")`
? Output trenger én node per klasse med softmax, og `sparse_categorical_crossentropy` tar heltallslabels direkte.
:::

## RL og XAI

::: quiz
Hvilken linje mangler i Q-læringen?
```kode
action = update_action(env, state, explore_rate, q_table)
_, reward, terminated, _ = env.step(action)
next_state = env.get_state_id()
# ??? (manglende linje, anta at terminated er False)
state = next_state
```
- [ ] `q_table[next_state, action] = reward`
- [ ] `q_table[state, action] += alpha * (reward + gamma * np.max(q_table[state]) - q_table[state, action])`
- [ ] `q_table[state, action] = np.argmax(q_table[next_state])`
- [x] `q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])`
? Cellen `q_table[state, action]` skal oppdateres mot belønning pluss diskontert beste verdi i `next_state`.
:::

::: quiz
Replay-bufferen er en liste med overganger `(state, action, reward, next_state, done)`. Hvilken linje passer i hullet?
```kode
batch = random.sample(buffer, k=32)
# ??? (manglende linje)
print(len(states), len(rewards))   # 32 32
```
- [ ] `states, actions, rewards, next_states, dones = batch`
- [x] `states, actions, rewards, next_states, dones = zip(*batch)`
- [ ] `states, actions, rewards, next_states, dones = batch[0]`
- [ ] `states, actions, rewards, next_states, dones = np.take(batch, 5)`
? `zip(*batch)` transponerer 32 tupler med 5 elementer til 5 tupler med 32 elementer.
:::

::: quiz
Target-nettverket skal oppdateres med jevne mellomrom. Hvilken linje passer i hullet?
```kode
if total_steps % agent.target_update_frequency == 0:
    # ??? (manglende linje)
```
- [ ] `agent.target_network = agent.q_network`
- [ ] `agent.q_network.load_state_dict(agent.target_network.state_dict())`
- [x] `agent.target_network.load_state_dict(agent.q_network.state_dict())`
- [ ] `agent.target_network.train()`
? Vektene kopieres fra Q-nettverket til target-nettverket, og `=` ville bare gitt to navn på samme nett.
:::

::: quiz
Du skal forklare en `GradientBoostingRegressor` med SHAP. Hvilken linje passer i hullet?
```kode
model = GradientBoostingRegressor().fit(X_train, y_train)
explainer = # ??? (manglende uttrykk)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values)
```
- [x] `shap.TreeExplainer(model)`
- [ ] `shap.DeepExplainer(model, X_train)`
- [ ] `shap.TreeExplainer(X_test)`
- [ ] `shap.plots.waterfall(model)`
? `TreeExplainer` er laget for trebaserte modeller og tar modellen som argument.
:::

::: oppgave Fyll inn oppdateringene
Fyll inn de to linjene som oppdaterer `self.weights` og `self.bias`, slik at modellen lærer `y = 3x + 2`.
```start
import numpy as np

class LinearRegression:
    def __init__(self, lr=0.5, n_iterations=2000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights, self.bias = None, None

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        for _ in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y
            grad_w = 2 * X.T @ error / len(y)
            grad_b = 2 * error.mean()
            # linje 1: oppdater self.weights
            # linje 2: oppdater self.bias

    def predict(self, X):
        return X @ self.weights + self.bias

X = np.linspace(0, 1, 50).reshape(-1, 1)
y = 3 * X[:, 0] + 2
model = LinearRegression()
model.fit(X, y)
```
```losning
import numpy as np

class LinearRegression:
    def __init__(self, lr=0.5, n_iterations=2000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights, self.bias = None, None

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        for _ in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y
            grad_w = 2 * X.T @ error / len(y)
            grad_b = 2 * error.mean()
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

    def predict(self, X):
        return X @ self.weights + self.bias

X = np.linspace(0, 1, 50).reshape(-1, 1)
y = 3 * X[:, 0] + 2
model = LinearRegression()
model.fit(X, y)
```
```test
assert model.weights is not None and np.shape(model.weights) == (1,), "self.weights skal ha shape (1,)."
assert abs(model.weights[0] - 3) < 0.05, f"Vekten ble {model.weights[0]:.3f}, men skal være nær 3. Går du mot gradienten (minus)?"
assert abs(model.bias - 2) < 0.05, f"Bias ble {model.bias:.3f}, men skal være nær 2."
```
:::

--- LEKSJON 13.6 | blandet-prove | Blandet prøve
Spørsmål fra hele kurset i tilfeldig rekkefølge: svar uten å kjøre koden.

::: quiz
**1.** Forvirringsmatrise med `ravel`. Hva skrives ut?
```kode
from sklearn.metrics import confusion_matrix
y_true = [0, 0, 0, 1, 1, 1, 1]
y_pred = [0, 1, 1, 1, 0, 1, 1]
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(fp, fn)
```
- [ ] `1 2`
- [x] `2 1`
- [ ] `1 3`
- [ ] `2 3`
? FP er sanne 0-er predikert som 1 (to stykker), og FN er sanne 1-ere predikert som 0 (én).
:::

::: quiz
**2.** Et lineært lag i PyTorch. Hva skrives ut?
```kode
layer = nn.Linear(4, 3)
x = torch.randn(8, 4)
print(layer(x).shape, layer.weight.shape)
```
- [ ] `torch.Size([8, 3]) torch.Size([4, 3])`
- [ ] `torch.Size([8, 4]) torch.Size([3, 4])`
- [ ] `torch.Size([3, 8]) torch.Size([3, 4])`
- [x] `torch.Size([8, 3]) torch.Size([3, 4])`
? Output får shape `(8, 3)`, og PyTorch lagrer vektene som `(out_features, in_features)`, altså `(3, 4)`.
:::

::: quiz
**3.** Datasettet har 50 rader. Hva skrives ut?
```kode
X = np.arange(100).reshape(50, 2)
y = np.arange(50)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, y_test.shape)
```
- [x] `(40, 2) (10,)`
- [ ] `(40, 2) (40,)`
- [ ] `(10, 2) (10,)`
- [ ] `(40,) (10,)`
? 20 % av 50 er 10 testrader, og antall kolonner endres ikke.
:::

::: quiz
**4.** En gini-funksjon. Hva skrives ut?
```kode
def gini(counts):
    p = np.array(counts) / np.sum(counts)
    return 1 - np.sum(p ** 2)

print(gini([5, 5]), gini([10, 0]))
```
- [ ] `1.0 0.0`
- [ ] `0.0 0.5`
- [x] `0.5 0.0`
- [ ] `0.5 1.0`
? `[5, 5]` gir 1 - (0.25 + 0.25) = 0.5, og en ren node gir 1 - 1 = 0.
:::

::: quiz
**5.** ε-avtagning med en `while`-løkke. Hva skrives ut?
```kode
eps, eps_min, decay = 1.0, 0.1, 0.5
steg = 0
while eps > eps_min:
    eps *= decay
    steg += 1
print(steg, eps)
```
- [ ] `3 0.125`
- [x] `4 0.0625`
- [ ] `4 0.1`
- [ ] `5 0.03125`
? ε blir 0.5, 0.25, 0.125 og 0.0625, og først etter fjerde steg er den ikke lenger over 0.1.
:::

::: quiz
**6.** DBSCAN på seks punkter. Hva skrives ut?
```kode
X = np.array([[0, 0], [0, 0.1], [0.1, 0], [5, 5], [5, 5.1], [20, 20]])
labels = DBSCAN(eps=0.5, min_samples=2).fit_predict(X)
print(labels)
```
- [ ] `[0 0 0 1 1 2]`
- [ ] `[1 1 1 2 2 -1]`
- [ ] `[0 0 0 0 0 -1]`
- [x] `[0 0 0 1 1 -1]`
? De tre tette punktene blir klynge 0, de to ved (5, 5) blir klynge 1, og (20, 20) uten naboer blir støy, -1.
:::

::: quiz
**7.** Hvor mange hovedkomponenter trengs for å forklare minst 90 % av variansen?
```kode
r = np.array([0.55, 0.25, 0.12, 0.05, 0.03])   # pca.explained_variance_ratio_
k = np.argmax(np.cumsum(r) >= 0.90) + 1
print(k)
```
- [x] 3
- [ ] 2
- [ ] 4
- [ ] 1
? `np.cumsum(r)` er `[0.55, 0.80, 0.92, 0.97, 1.00]`, så indeks 2 er den første over 0.90, og `+ 1` gir 3.
:::

::: quiz
**8.** Softmax i PyTorch. Hva skrives ut?
```kode
z = torch.tensor([[2.0, 1.0, 0.1]])
p = torch.softmax(z, dim=1)
print(round(p.sum().item(), 4), p.argmax(dim=1).item())
```
- [ ] `3.1 0`
- [x] `1.0 0`
- [ ] `1.0 2`
- [ ] `1.0 2.0`
? Softmax gir sannsynligheter som summerer til 1 og bevarer rekkefølgen, så største logit (indeks 0) vinner.
:::

::: quiz
**9.** `PolynomialFeatures` med grad 2. Hva skrives ut?
```kode
X = np.array([[2, 3]])
print(PolynomialFeatures(degree=2).fit_transform(X))
```
- [ ] `[[2. 3. 4. 9.]]`
- [ ] `[[1. 2. 3. 4. 9.]]`
- [ ] `[[2. 3. 4. 6. 9.]]`
- [x] `[[1. 2. 3. 4. 6. 9.]]`
? Grad 2 med a = 2 og b = 3 gir kolonnene 1, a, b, a², ab, b², altså 1, 2, 3, 4, 6, 9.
:::

::: quiz
**10.** Hva skriver `LabelEncoder` ut?
```kode
le = LabelEncoder()
print(le.fit_transform(["katt", "hund", "fugl", "hund"]))
```
- [ ] `[0 1 2 1]`
- [x] `[2 1 0 1]`
- [ ] `[1 2 3 2]`
- [ ] `[2 1 0 2]`
? `LabelEncoder` sorterer klassene alfabetisk: fugl = 0, hund = 1, katt = 2.
:::

::: quiz
**11.** FrozenLake 4 x 4 uten glatt is. Handling 2 er «høyre». Hva skrives ut?
```kode
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
state, info = env.reset(seed=0)
next_state, reward, terminated, truncated, info = env.step(2)
print(state, next_state, terminated)
```
- [x] `0 1 False`
- [ ] `0 4 False`
- [ ] `1 2 False`
- [ ] `(0, {}) 1 False`
? Rutene nummereres rad for rad fra 0, så ett steg til høyre fra rute 0 gir rute 1.
:::

::: quiz
**12.** Diskontert retur. Hva skrives ut?
```kode
gamma = 0.9
rewards = [0, 0, 10]
G = sum(gamma ** t * r for t, r in enumerate(rewards))
print(round(G, 2))
```
- [ ] 10
- [ ] 9.0
- [x] 8.1
- [ ] 7.29
? Belønningen kommer i steg t = 2, så G = 0.9² · 10 = 8.1.
:::

::: quiz
**13.** `MinMaxScaler` på et testpunkt utenfor treningsområdet. Hva skrives ut?
```kode
X_train = np.array([[0.0], [5.0], [10.0]])
X_test = np.array([[20.0]])
s = MinMaxScaler().fit(X_train)
print(s.transform(X_test))
```
- [ ] `[[1.]]`
- [x] `[[2.]]`
- [ ] `[[0.]]`
- [ ] `ValueError`
? Scaleren lærte min 0 og maks 10 og klipper ikke, så (20 - 0) / (10 - 0) = 2.
:::

::: quiz
**14.** `SimpleImputer` på testdata. Hva skrives ut?
```kode
X_train = np.array([[1.0], [np.nan], [3.0]])
X_test = np.array([[np.nan]])
imp = SimpleImputer(strategy="mean").fit(X_train)
print(imp.transform(X_test))
```
- [ ] `[[nan]]`
- [ ] `[[1.33]]`
- [ ] `[[0.]]`
- [x] `[[2.]]`
? Gjennomsnittet av verdiene som finnes i treningsdataene er (1 + 3) / 2 = 2.
:::

::: quiz
**15.** Et `GridSearchCV`-oppsett. Hvor mange modeller trenes totalt, inkludert den endelige som trenes på alle treningsdataene (`refit=True` er standard)?
```kode
grid = GridSearchCV(DecisionTreeClassifier(random_state=0),
                    {"max_depth": [1, 2, 3], "criterion": ["gini", "entropy"]},
                    cv=5)
grid.fit(X_train, y_train)
```
- [ ] 6
- [ ] 30
- [x] 31
- [ ] 11
? 6 kombinasjoner ganger 5 folder gir 30 modeller, pluss én endelig refit på hele `X_train`: 31.
:::

::: quiz
**16.** `val_losses` er hentet fra `history.history["val_loss"]` etter fem epoker med Keras. Hvilken epoke var best? Hva skrives ut?
```kode
val_losses = [0.90, 0.62, 0.55, 0.58, 0.61]   # history.history["val_loss"]
best_epoch = np.argmin(val_losses) + 1
print(best_epoch)
```
- [ ] 2
- [x] 3
- [ ] 5
- [ ] 0.55
? `np.argmin` gir indeks 2, og `+ 1` gjør den om til epoke 3.
:::

::: quiz
**17.** Inntekt måles i millioner kroner i stedet for i kroner. Hva skrives ut?
```kode
model_kr = LinearRegression().fit(df[["inntekt_kr"]], y)
model_mill = LinearRegression().fit(df[["inntekt_kr"]] / 1_000_000, y)
print(round(model_mill.coef_[0] / model_kr.coef_[0]))
```
- [x] 1000000
- [ ] 1
- [ ] 0
- [ ] Det avhenger av dataene
? Når featuren deles på en million, må koeffisienten ganges med en million for å gi samme prediksjon.
:::

::: quiz
**18.** Hva måler `drop` i koden under?
```kode
base = accuracy_score(y_val, model.predict(X_val))
X_perm = X_val.copy()
X_perm[:, 2] = rng.permutation(X_perm[:, 2])
drop = base - accuracy_score(y_val, model.predict(X_perm))
```
- [ ] SHAP-verdien til feature 2 for ett datapunkt
- [ ] Koeffisienten til feature 2
- [x] Permutation importance for feature 2: hvor mye accuracy faller når sammenhengen mellom feature 2 og målet ødelegges
- [ ] Korrelasjonen mellom feature 2 og målet
? Å stokke kolonne 2 bryter koblingen til `y`, så fallet i accuracy er permutation importance for feature 2.
:::

::: quiz
**19.** Hvilke verdier kan `pred` inneholde?
```kode
iso = IsolationForest(random_state=0).fit(X_train)
pred = iso.predict(X_new)
```
- [ ] Bare 0 og 1, der 1 er anomali
- [x] Bare 1 og -1, der -1 er anomali
- [ ] Klyngenumre 0, 1, 2, ...
- [ ] Anomaliscorer mellom 0 og 1
? `IsolationForest.predict` gir 1 for normale punkter og -1 for avvik.
:::

::: quiz
**20.** En waterfall-plot for ett hus har baseverdi 2.0 og SHAP-verdiene under. Hva skrives ut?
```kode
base_value = 2.0
shap_vals = np.array([0.8, -0.3, 0.1])
print(round(base_value + shap_vals.sum(), 2))
```
- [ ] 2.0
- [ ] 3.2
- [x] 2.6
- [ ] 0.6
? SHAP-verdier er additive: 2.0 + 0.8 - 0.3 + 0.1 = 2.6.
:::

::: quiz
**21.** R² og MSE for en modell som alltid predikerer gjennomsnittet. Hva skrives ut?
```kode
y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([2.0, 2.0, 2.0])
print(r2_score(y_true, y_pred), round(mean_squared_error(y_true, y_pred), 3))
```
- [x] `0.0 0.667`
- [ ] `1.0 0.667`
- [ ] `0.0 0.0`
- [ ] `-1.0 2.0`
? Å alltid predikere gjennomsnittet gir R² = 0, og MSE = (1 + 0 + 1) / 3 ≈ 0.667.
:::

::: quiz
**22.** Et klassifiseringsnett gir output med shape `(32, 1)`, og `y` har shape `(32,)`. Hva skjer i siste linje?
```kode
loss_fn = nn.BCEWithLogitsLoss()
out = model(X_batch)          # shape (32, 1)
loss = loss_fn(out, y_batch)  # y_batch har shape (32,)
```
- [ ] Tapet regnes ut riktig, PyTorch fikser shape automatisk
- [ ] Tapet regnes ut via broadcasting til `(32, 32)`
- [x] `ValueError` om at target og input må ha samme størrelse
- [ ] `RuntimeError` om at labels må være `long`
? `BCEWithLogitsLoss` krever samme shape på input og target, og derfor bruker prosjekt 3 `squeeze(1)`.
:::

::: quiz
**23.** Hvor mange minibatcher gir `DataLoader`? Hva skrives ut?
```kode
ds = TensorDataset(torch.randn(100, 3), torch.randn(100))
loader = DataLoader(ds, batch_size=32, shuffle=True)
print(len(loader))
```
- [ ] 3
- [ ] 32
- [ ] 100
- [x] 4
? 100 punkter i batcher på 32 gir 32 + 32 + 32 + 4, altså 4 batcher.
:::

::: quiz
**24.** `np.random.seed` og reproduserbarhet. Hva skrives ut?
```kode
np.random.seed(0)
a = np.random.rand(2)
np.random.seed(0)
b = np.random.rand(2)
c = np.random.rand(2)
print((a == b).all(), (a == c).all())
```
- [x] `True False`
- [ ] `True True`
- [ ] `False False`
- [ ] `False True`
? Samme seed gir samme tall, mens `c` fortsetter sekvensen etter `b`.
:::

::: quiz
**25.** Du glemmer `super().__init__()` i et PyTorch-nett. Hva skjer når du kjører `Net()`?
```kode
class Net(nn.Module):
    def __init__(self):
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)
```
- [ ] Ingenting, nettet virker som normalt
- [ ] Nettet lages, men har 0 parametre
- [ ] `TypeError` fordi `forward` mangler `self`
- [x] `AttributeError`, fordi et lag ikke kan registreres før `nn.Module.__init__()` er kalt
? Uten `super().__init__()` krasjer allerede `self.fc = ...`, fordi `nn.Module` ikke er satt opp.
:::

::: quiz
**26.** `cross_val_score` med `cv=5`. Hva skrives ut?
```kode
scores = cross_val_score(DecisionTreeClassifier(random_state=0), X, y, cv=5)
print(type(scores).__name__, scores.shape)
```
- [ ] `float ()`
- [x] `ndarray (5,)`
- [ ] `list (5,)`
- [ ] `ndarray (5, 2)`
? `cross_val_score` gir ett tall per fold i et NumPy-array med shape `(5,)`.
:::
