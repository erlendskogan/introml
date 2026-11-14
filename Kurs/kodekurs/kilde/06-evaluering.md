=== DEL 6 | evaluering | Evaluering: hvor god er modellen?
Accuracy, train/test-split, forvirringsmatrise, precision og recall, ROC og AUC, og regresjonsmetrikker. Du regner alt for hånd med numpy først, og så med `sklearn.metrics`.

--- LEKSJON 6.1 | accuracy-split | Accuracy og train/test-split
**Accuracy** (treffsikkerhet) er andelen prediksjoner som er riktige. Med numpy-arrays er det én linje, akkurat som i forelesningsnotatene: `y_true == y_pred` gir en boolsk array, og gjennomsnittet av `True`/`False` er andelen `True`.

```python
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])

print(y_true == y_pred)
print("accuracy:", np.mean(y_true == y_pred))
```

## scikit-learn
**scikit-learn** (`sklearn`) er Python-biblioteket for klassisk maskinlæring. Del 7 går grundig gjennom det. Her trenger vi bare to ting: ferdige metrikker i `sklearn.metrics` og funksjonen som deler opp data. Metrikkene tar alltid **sanne verdier først**, så prediksjonene.

```python
import numpy as np
from sklearn.metrics import accuracy_score

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])
print(accuracy_score(y_true, y_pred))
```

## Hvorfor dele opp dataene?
Målet er at modellen skal fungere på **nye** data. Måler vi på de samme dataene som modellen ble trent på, belønner vi pugging. Derfor deler vi datasettet i en større **treningsdel** (til `fit`) og en mindre **testdel** som ikke røres før modellen er ferdig. Splitten må være **tilfeldig**, og `X` og `y` må stokkes med de **samme** indeksene så radene fortsatt hører sammen:

```python
import numpy as np

X = np.arange(20).reshape(10, 2)       # 10 datapunkter, 2 features
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

rng = np.random.default_rng(42)
perm = rng.permutation(len(y))
n_test = int(0.2 * len(y))
test_idx, train_idx = perm[:n_test], perm[n_test:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print("test-indekser:", test_idx)
print(X_test, y_test)
```

Se `perm`, `X_train` og `X_test` i variabelpanelet: radene i `X_test` er nøyaktig radene `test_idx` fra `X`, og `y_test` er hentet med de samme indeksene.

## train_test_split
I praksis bruker vi `train_test_split` fra `sklearn.model_selection`. Den returnerer **fire** arrays i fast rekkefølge: `X_train, X_test, y_train, y_test`.

- `test_size=0.2`: 20 % til test.
- `random_state=42`: samme tilfeldige splitt hver gang (reproduserbart).
- `stratify=y`: samme andel av hver klasse i trenings- og testdelen. Viktig når én klasse er sjelden.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, 1 - data.target      # snu så 1 = ondartet (positiv klasse)
print("X:", X.shape, " andel ondartet:", round(y.mean(), 3))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("train:", X_train.shape, " test:", X_test.shape)
display({"X_train": X_train.shape, "X_test": X_test.shape, "y_train": y_train.shape, "y_test": y_test.shape})
print("andel ondartet i train:", round(y_train.mean(), 3), " i test:", round(y_test.mean(), 3))
```

Tabellen viser de fire arrayene i samme rekkefølge som `train_test_split` returnerer dem. 455 + 114 = 569 rader, og `X_train` og `y_train` har like mange rader.

## Hele løpet
Nå kan vi evaluere en modell skikkelig. sklearn har sin egen `LogisticRegression` med de samme metodene som klassen du skrev i 5.7 (`fit`, `predict`, `predict_proba`). Del 7 forklarer den nærmere. Vi standardiserer med statistikk fra treningsdataene, som i 5.9:

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X, y = data.data, 1 - data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train, X_test = (X_train - mu) / sigma, (X_test - mu) / sigma

model = LogisticRegression()
model.fit(X_train, y_train)
print("train accuracy:", accuracy_score(y_train, model.predict(X_train)))
print("test accuracy: ", accuracy_score(y_test, model.predict(X_test)))
```

Test-accuracy er tallet vi rapporterer. Treningsaccuracy er nesten alltid litt høyere.

::: prosjekt
I prosjekt 1 var splitten allerede gjort med kolonnen `split` (`data[data["split"] == "train"]`). I prosjekt 2 (Titanic-delen) og prosjekt 4 (XAI) bruker du `train_test_split` selv.
:::

::: oppgave Splitt for hånd
`X` har 100 rader og `y` 100 labels. Lag `X_train, X_test, y_train, y_test` med 80 punkter til trening og 20 til test, tilfeldig valgt med `np.random.default_rng(0).permutation`. Radene i `X` og `y` må fortsatt høre sammen.
```start
import numpy as np

X = np.arange(200).reshape(100, 2)
y = np.arange(100) % 3

rng = np.random.default_rng(0)
# lag perm, train_idx og test_idx
X_train, X_test, y_train, y_test = ..., ..., ..., ...
```
```losning
import numpy as np

X = np.arange(200).reshape(100, 2)
y = np.arange(100) % 3

rng = np.random.default_rng(0)
perm = rng.permutation(len(y))
test_idx, train_idx = perm[:20], perm[20:]
X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
```
```test
import numpy as np
for navn, arr, form in [("X_train", X_train, (80, 2)), ("X_test", X_test, (20, 2)), ("y_train", y_train, (80,)), ("y_test", y_test, (20,))]:
    assert np.shape(arr) == form, f"{navn} skal ha shape {form}, men har {np.shape(arr)}."
rad_tr, rad_te = np.asarray(X_train)[:, 0] // 2, np.asarray(X_test)[:, 0] // 2
assert np.array_equal(np.asarray(y_train), rad_tr % 3) and np.array_equal(np.asarray(y_test), rad_te % 3), "Radene i X og y hører ikke sammen. Bruk de samme indeksene for X og y."
assert np.array_equal(np.sort(np.concatenate([rad_tr, rad_te])), np.arange(100)), "Hvert datapunkt skal være med nøyaktig én gang (i enten train eller test)."
assert not np.array_equal(rad_te, np.arange(20)), "Testdelen er bare de 20 første radene. Stokk med rng.permutation først."
```
:::

::: quiz
Du ønsker å splitte datasettet i en treningsdel og en testdel. Hvilken funksjon gjør dette?
- [ ] `numpy.random.separate(X, y)`
- [x] `sklearn.model_selection.train_test_split(X, y)`
- [ ] `pandas.Series.split_dataset(X, y)`
- [ ] `matplotlib.pyplot.partition(X, y)`
? `train_test_split` ligger i `sklearn.model_selection`. De tre andre funksjonene finnes ikke. Dette var et spørsmål på eksamen 2025.
:::

::: quiz
Hva er galt med denne linjen?
```kode
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)
```
- [ ] Ingenting, rekkefølgen spiller ingen rolle
- [x] `y_train` inneholder egentlig testdelen av `X`, fordi rekkefølgen skal være `X_train, X_test, y_train, y_test`
- [ ] `test_size` må være et heltall
- [ ] Funksjonen returnerer bare to verdier
? `train_test_split` returnerer alltid X-delene først, så y-delene. Med feil rekkefølge får variablene feil innhold uten at Python klager, og `fit` vil enten krasje eller trene på tull.
:::

::: oppsummering
- Accuracy: `np.mean(y_true == y_pred)` eller `accuracy_score(y_true, y_pred)`.
- Tren på treningsdata, rapporter på testdata som ikke er brukt i treningen.
- `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`.
- `random_state` gir reproduserbar splitt, `stratify=y` bevarer klassefordelingen.
:::

--- LEKSJON 6.2 | forvirringsmatrise | Forvirringsmatrise
Accuracy sier ikke **hvilke** feil modellen gjør. En **forvirringsmatrise** (confusion matrix) teller fire tilfeller for binær klassifisering (1 = positiv, 0 = negativ):

| | Predikert 0 | Predikert 1 |
|---|---|---|
| **Sann 0** | TN (sann negativ) | FP (falsk positiv) |
| **Sann 1** | FN (falsk negativ) | TP (sann positiv) |

Les navnene baklengs: siste ord er hva modellen **sa**, første ord om den hadde **rett**. En falsk positiv er altså «modellen sa positiv, og tok feil». Med numpy teller vi dem med boolske masker:

```python
import numpy as np
import pandas as pd

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1])

utfall = np.where(y_true == 1, np.where(y_pred == 1, "TP", "FN"), np.where(y_pred == 1, "FP", "TN"))
display(pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "utfall": utfall}))

tp = np.sum((y_true == 1) & (y_pred == 1))
tn = np.sum((y_true == 0) & (y_pred == 0))
fp = np.sum((y_true == 0) & (y_pred == 1))
fn = np.sum((y_true == 1) & (y_pred == 0))
print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}")
print("accuracy:", (tp + tn) / (tp + tn + fp + fn))
```

## confusion_matrix i sklearn
`confusion_matrix(y_true, y_pred)` gir en 2×2-matrise der **radene er sanne klasser** og **kolonnene er predikerte klasser**, sortert 0, 1. Med `.ravel()` flater vi den ut, og rekkefølgen blir `tn, fp, fn, tp`:

```python
import numpy as np
from sklearn.metrics import confusion_matrix

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1])

cm = confusion_matrix(y_true, y_pred)
print(cm)
display(cm)
tn, fp, fn, tp = cm.ravel()
print(f"tn={tn} fp={fp} fn={fn} tp={tp}")
```

Les rutenettet slik: øverst til venstre `cm[0, 0]` er TN, øverst til høyre `cm[0, 1]` er FP, nederst til venstre `cm[1, 0]` er FN og nederst til høyre `cm[1, 1]` er TP. Riktige svar ligger på diagonalen. `.ravel()` leser rad for rad, og derfor blir rekkefølgen `tn, fp, fn, tp`.

::: advarsel To klassiske feller
1. Rekkefølgen etter `.ravel()` er `tn, fp, fn, tp`, ikke `tp, fp, fn, tn`. Den følger matrisen rad for rad.
2. Byttes argumentene, `confusion_matrix(y_pred, y_true)`, blir matrisen transponert: FP og FN bytter plass uten noen feilmelding. Sanne verdier skal alltid først.
:::

## På ekte data
Modellen fra 6.1 på brystkreftdataene. `ConfusionMatrixDisplay` tegner matrisen med tall i hver celle:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data = load_breast_cancer()
X, y = data.data, 1 - data.target                  # 1 = ondartet
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train, X_test = (X_train - mu) / sigma, (X_test - mu) / sigma

model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["godartet", "ondartet"], cmap="Blues")
plt.show()
```

Diagonalen (TN og TP) er riktige svar. Her er det få feil, men de er ikke like alvorlige: en FN er en ondartet svulst som ble oversett.

::: oppgave Tell for hånd
Regn ut `tp`, `tn`, `fp` og `fn` for arrayene under, uten sklearn.
```start
import numpy as np

y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0])
y_pred = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0])

tp = ...
tn = ...
fp = ...
fn = ...
```
```losning
import numpy as np

y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0])
y_pred = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0])

tp = np.sum((y_true == 1) & (y_pred == 1))
tn = np.sum((y_true == 0) & (y_pred == 0))
fp = np.sum((y_true == 0) & (y_pred == 1))
fn = np.sum((y_true == 1) & (y_pred == 0))
```
```test
assert tp == 4, f"tp = {tp}, forventet 4. TP er sann 1 og predikert 1."
assert tn == 4, f"tn = {tn}, forventet 4. TN er sann 0 og predikert 0."
assert fp == 2, f"fp = {fp}, forventet 2. FP er sann 0, men predikert 1."
assert fn == 2, f"fn = {fn}, forventet 2. FN er sann 1, men predikert 0."
```
:::

::: quiz
`confusion_matrix(y_test, y_pred)` gir matrisen under. Hvor mange falske positive er det?
```kode
[[50 10]
 [ 5 35]]
```
- [ ] 5
- [x] 10
- [ ] 35
- [ ] 50
? Rader er sanne klasser, kolonner predikerte. Rad 0 (sann negativ), kolonne 1 (predikert positiv) er FP = 10. Tallet 5 er FN, 35 er TP og 50 er TN.
:::

::: quiz
Hva skrives ut?
```kode
y_true = np.array([1, 1, 1, 0, 0])
y_pred = np.array([1, 0, 0, 0, 1])
tn, fp, fn, tp = confusion_matrix(y_pred, y_true).ravel()
print(fn)
```
- [ ] 2
- [x] 1
- [ ] 0
- [ ] 3
? Argumentene er byttet, så matrisen er transponert og «fn» inneholder egentlig FP. Sann FP er 1 (punktet `0 -> 1`), sann FN er 2. Derfor skrives 1 ut. Riktig kall er `confusion_matrix(y_true, y_pred)`.
:::

::: oppsummering
- TP, TN, FP, FN: siste ord er hva modellen sa, første om den hadde rett.
- `confusion_matrix(y_true, y_pred)`: rader = sanne klasser, kolonner = predikerte.
- `tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()`.
- For hånd: `np.sum((y_true == 1) & (y_pred == 0))` osv.
:::

--- LEKSJON 6.3 | precision-recall | Precision, recall, FPR og F1
Fra de fire tallene i forvirringsmatrisen lager vi metrikker som svarer på konkrete spørsmål:

| Metrikk | Formel | Svarer på |
|---|---|---|
| Recall = TPR = sensitivitet | $\frac{TP}{TP + FN}$ | Av alle positive, hvor mange fant vi? |
| Spesifisitet = TNR | $\frac{TN}{TN + FP}$ | Av alle negative, hvor mange slapp fri? |
| FPR (falsk alarm-rate) | $\frac{FP}{FP + TN} = 1 - \text{spesifisitet}$ | Av alle negative, hvor mange fikk falsk alarm? |
| Precision = PPV | $\frac{TP}{TP + FP}$ | Av alle vi sa var positive, hvor mange var det? |
| F1 | $2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$ | Ett tall som balanserer precision og recall |

Recall, spesifisitet og FPR deler på en **rad** (en sann klasse). Precision deler på en **kolonne** (det modellen predikerte).

```python
import numpy as np
from sklearn.metrics import confusion_matrix

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1])
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

recall = tp / (tp + fn)
specificity = tn / (tn + fp)
fpr = fp / (fp + tn)
precision = tp / (tp + fp)
f1 = 2 * precision * recall / (precision + recall)

print(f"recall/TPR:  {recall:.3f}")
print(f"spesifisitet: {specificity:.3f}")
print(f"FPR:          {fpr:.3f}")
print(f"precision:    {precision:.3f}")
print(f"F1:           {f1:.3f}")
```

## Med sklearn
`precision_score`, `recall_score` og `f1_score` regner ut det samme. Det finnes ingen egen funksjon for FPR eller spesifisitet, så dem tar du fra forvirringsmatrisen. `classification_report` gir en hel oversikt per klasse:

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1])

print("precision:", precision_score(y_true, y_pred))
print("recall:   ", recall_score(y_true, y_pred))
print("F1:       ", f1_score(y_true, y_pred))
print(classification_report(y_true, y_pred))
```

I rapporten er raden for klasse `1` den vi har regnet på. Recall for klasse `0` er spesifisiteten.

::: tips Ingen positive prediksjoner
Predikerer modellen aldri 1, blir precision $\frac{0}{0}$. sklearn gir da en advarsel og returnerer 0. Med `precision_score(y_true, y_pred, zero_division=0)` slipper du advarselen.
:::

## Prosjekt 2: rapporter TPR og FPR
Prosjekt 2 ba om TPR og FPR med nøyaktig disse utskriftslinjene. Her med modellen på brystkreftdataene:

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, f1_score

data = load_breast_cancer()
X, y = data.data, 1 - data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train, X_test = (X_train - mu) / sigma, (X_test - mu) / sigma
y_pred = LogisticRegression().fit(X_train, y_train).predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
print(f"True positive rate: {tpr:.3f}")
print(f"False positive rate: {fpr:.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"F1: {f1_score(y_test, y_pred):.3f}")
```

::: prosjekt Prosjekt 2
Cellen i prosjektet sluttet med `print(f"True positive rate: {tpr:.3f}")` og `print(f"False positive rate: {fpr:.3f}")`, og ba deg rapportere andre relevante metrikker i tillegg. Mønsteret over (`ravel()` og to divisjoner) er alt som trengs.
:::

::: oppgave Regn ut metrikkene
Regn ut `tpr`, `fpr`, `precision` og `f1` for arrayene under (positiv klasse er 1). Du kan bruke numpy, `confusion_matrix` eller begge.
```start
import numpy as np
from sklearn.metrics import confusion_matrix

y_test = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
y_pred = np.array([1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1])

tpr = ...
fpr = ...
precision = ...
f1 = ...
```
```losning
import numpy as np
from sklearn.metrics import confusion_matrix

y_test = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
y_pred = np.array([1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1])

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
precision = tp / (tp + fp)
f1 = 2 * precision * tpr / (precision + tpr)
```
```test
assert abs(tpr - 0.6) < 1e-9, f"tpr = {tpr}, forventet 3/5 = 0.6. TPR = TP / (TP + FN)."
assert abs(fpr - 2 / 7) < 1e-9, f"fpr = {fpr}, forventet 2/7 ≈ 0.286. FPR = FP / (FP + TN)."
assert abs(precision - 0.6) < 1e-9, f"precision = {precision}, forventet 3/5 = 0.6. Precision = TP / (TP + FP)."
assert abs(f1 - 0.6) < 1e-9, f"f1 = {f1}, forventet 0.6. F1 = 2 * precision * recall / (precision + recall)."
```
:::

::: quiz
`tn, fp, fn, tp` er hentet fra forvirringsmatrisen. Hvilken metrikk regner denne linjen ut?
```kode
m = tp / (tp + fp)
```
- [ ] Recall
- [x] Precision
- [ ] FPR
- [ ] Spesifisitet
? Nevneren `tp + fp` er alle som ble **predikert** positive. Andelen av dem som faktisk var positive, er precision. Recall har `tp + fn` (alle som faktisk er positive) i nevneren.
:::

::: quiz
Du skal lage en modell som finner pasienter med en dødelig sykdom. Det er svært alvorlig å overse en syk pasient (falsk negativ). Hvilken metrikk bør modellen skåre høyt på?
- [ ] Precision
- [x] Recall (TPR)
- [ ] Spesifisitet
- [ ] FPR
? Recall = TP / (TP + FN) blir høy bare når FN er liten, altså når vi finner nesten alle syke. Dette er en oppgave fra forelesningsnotatene.
:::

::: oppsummering
- Recall/TPR = `tp / (tp + fn)`, FPR = `fp / (fp + tn)`, spesifisitet = `tn / (tn + fp)` = 1 − FPR.
- Precision = `tp / (tp + fp)`, F1 = harmonisk snitt av precision og recall.
- `precision_score`, `recall_score`, `f1_score`, `classification_report` (sanne verdier først).
- FPR og spesifisitet har ingen egen sklearn-funksjon: bruk forvirringsmatrisen.
:::

--- LEKSJON 6.4 | roc-auc | Terskler, ROC og AUC
`predict` bruker terskel 0.5, men terskelen er et valg. Lavere terskel gir flere positive prediksjoner: høyere TPR, men også høyere FPR. Vi ser det tydelig på en svakere modell som bare bruker to features:

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()
X, y = data.data[:, [1, 4]], 1 - data.target      # bare "mean texture" og "mean smoothness"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train, X_test = (X_train - mu) / sigma, (X_test - mu) / sigma

model = LogisticRegression().fit(X_train, y_train)
p = model.predict_proba(X_test)[:, 1]             # sannsynlighet for klasse 1

tabell = pd.DataFrame({"y_test": y_test[:8], "p": p[:8].round(3)})
for terskel in [0.8, 0.5, 0.3, 0.1]:
    tabell[f"y_pred ({terskel})"] = (p[:8] > terskel).astype(int)
display(tabell)

for terskel in [0.8, 0.5, 0.3, 0.1]:
    y_pred = (p > terskel).astype(int)
    tpr = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1)
    fpr = np.sum((y_pred == 1) & (y_test == 0)) / np.sum(y_test == 0)
    print(f"terskel {terskel}: TPR = {tpr:.2f}, FPR = {fpr:.2f}")
```

`predict_proba` gir én kolonne per klasse. Kolonne 1, `[:, 1]`, er sannsynligheten for positiv klasse. Tabellen viser de åtte første testpunktene: samme score `p` gir flere 1-ere jo lavere terskelen er. Hver 1-er for et punkt med `y_test = 1` er en TP, og hver 1-er for et punkt med `y_test = 0` er en FP.

## ROC-kurven for hånd
**ROC-kurven** viser TPR mot FPR for **alle** terskler fra 1 til 0. Terskel 1 gir ingen positive (punktet (0, 0)), terskel 0 gir bare positive (punktet (1, 1)). En god modell får høy TPR mens FPR fortsatt er lav, så kurven bøyer seg opp mot øvre venstre hjørne. Myntkast gir diagonalen.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()
X, y = data.data[:, [1, 4]], 1 - data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train, X_test = (X_train - mu) / sigma, (X_test - mu) / sigma
p = LogisticRegression().fit(X_train, y_train).predict_proba(X_test)[:, 1]

tpr_liste, fpr_liste = [], []
for terskel in np.linspace(1, 0, 101):
    y_pred = (p >= terskel).astype(int)
    tpr_liste.append(np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1))
    fpr_liste.append(np.sum((y_pred == 1) & (y_test == 0)) / np.sum(y_test == 0))

plt.plot(fpr_liste, tpr_liste, marker=".", label="Modell")
plt.plot([0, 1], [0, 1], "k--", label="Tilfeldig gjetning")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()
```

## roc_curve og roc_auc_score
sklearn gjør det samme med `roc_curve`, som prøver akkurat de tersklene der noe endrer seg. **AUC** (area under the curve) er arealet under ROC-kurven: 1 er perfekt, 0.5 er myntkast. `roc_auc_score` regner det direkte, og `auc(fpr, tpr)` regner arealet under en hvilken som helst kurve.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc
import pandas as pd

data = load_breast_cancer()
X, y = data.data, 1 - data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train, X_test = (X_train - mu) / sigma, (X_test - mu) / sigma

for navn, kolonner in [("alle 30 features", slice(None)), ("2 features", [1, 4])]:
    model = LogisticRegression().fit(X_train[:, kolonner], y_train)
    p = model.predict_proba(X_test[:, kolonner])[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, p)
    print(f"{navn}: roc_auc_score = {roc_auc_score(y_test, p):.3f}, auc = {auc(fpr, tpr):.3f}")
    plt.plot(fpr, tpr, label=f"{navn} (AUC = {roc_auc_score(y_test, p):.2f})")

print("De første radene fra roc_curve (modellen med 2 features):")
display(pd.DataFrame({"terskel": thresholds, "FPR": fpr, "TPR": tpr}).head(8))

plt.plot([0, 1], [0, 1], "k--", label="Tilfeldig (AUC = 0.5)")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend()
plt.show()
```

Tabellen viser hva `roc_curve` returnerer: én rad per terskel, sortert fra høy til lav. Den første terskelen er `inf`, som gir ingen positive og punktet (0, 0). For hver lavere terskel blir flere punkter positive, og TPR og FPR kan bare øke.

## Hva betyr AUC?
ROC AUC er **sannsynligheten for at modellen gir et tilfeldig valgt positivt punkt høyere score enn et tilfeldig valgt negativt punkt**. Vi kan sjekke det direkte ved å sammenligne alle par:

```python
import numpy as np
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
y = np.array([0] * 50 + [1] * 30)
p = np.where(y == 1, rng.normal(0.65, 0.15, 80), rng.normal(0.4, 0.15, 80))

pos, neg = p[y == 1], p[y == 0]
andel = np.mean(pos[:, None] > neg[None, :])        # alle 30 x 50 par
print("andel par der positiv > negativ:", round(andel, 4))
print("roc_auc_score:                   ", round(roc_auc_score(y, p), 4))
```

AUC avhenger bare av **rekkefølgen** på scorene, ikke av terskelen. Derfor sier den noe om hvor godt modellen skiller klassene over alle terskler.

::: advarsel ROC trenger sannsynligheter, ikke 0/1
Gir du `roc_curve` eller `roc_auc_score` ferdige klasser fra `predict`, finnes det bare én terskel. Kurven får tre punkter, og AUC blir feil (som regel for lav).

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

data = load_breast_cancer()
X, y = data.data[:, [1, 4]], 1 - data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression().fit(X_train, y_train)

print("med predict_proba:", round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 3))
print("med predict (0/1):", round(roc_auc_score(y_test, model.predict(X_test)), 3))
fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))
print("antall punkter på kurven:", len(fpr))
```
:::

::: tips Precision–recall-kurven
På samme måte kan vi plotte precision mot recall for alle terskler med `precision_recall_curve(y_test, p)`. Den er særlig nyttig når positiv klasse er sjelden (leksjon 6.6).
:::

::: prosjekt Prosjekt 1, problem 2
«Plot the ROC curve, what does the area under the curve measure?» Med din egen klasse: `fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_fe))`, så `plt.plot(fpr, tpr)` og `roc_auc_score(y_test, model.predict_proba(X_test_fe))`. Derfor trengte klassen `predict_proba`.
:::

::: oppgave Ett punkt på ROC-kurven
Lag funksjonen `roc_punkt(y_true, scores, terskel)` som predikerer 1 når `scores >= terskel` og returnerer `(fpr, tpr)` i den rekkefølgen.
```start
import numpy as np

def roc_punkt(y_true, scores, terskel):
    y_pred = ...
    fpr = ...
    tpr = ...
    return fpr, tpr
```
```losning
import numpy as np

def roc_punkt(y_true, scores, terskel):
    y_pred = (scores >= terskel).astype(int)
    fpr = np.sum((y_pred == 1) & (y_true == 0)) / np.sum(y_true == 0)
    tpr = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1)
    return fpr, tpr
```
```test
import numpy as np
yt = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1, 1])
sc = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.9, 0.3, 0.2, 0.6, 0.55])
r = roc_punkt(yt, sc, 0.5)
assert r is not None and len(r) == 2, "Funksjonen skal returnere (fpr, tpr)."
f, t = r
assert abs(t - 0.8) < 1e-9, f"tpr = {t} ved terskel 0.5, forventet 4/5 = 0.8."
assert abs(f - 0.2) < 1e-9, f"fpr = {f} ved terskel 0.5, forventet 1/5 = 0.2. Har du byttet om fpr og tpr?"
f0, t0 = roc_punkt(yt, sc, 0.0)
assert abs(f0 - 1) < 1e-9 and abs(t0 - 1) < 1e-9, "Terskel 0 skal gi punktet (1, 1). Bruk scores >= terskel."
```
:::

::: quiz
Hva representerer diagonalen fra (0, 0) til (1, 1) på en ROC-kurve?
- [x] En modell som ikke er bedre enn tilfeldig gjetning
- [ ] En modell som i gjennomsnitt predikerer 0.5 for alle dataene
- [ ] En klassifiseringsterskel på 0.5
- [ ] En middels god modell som er godt nok for de fleste problemer
? Langs diagonalen er TPR = FPR: modellen fanger like stor andel av de positive som av de negative, akkurat som et myntkast. AUC er da 0.5. Dette er et eksempelspørsmål fra forelesningsnotatene.
:::

::: quiz
Hvilket av følgende er **ikke** en korrekt tolkning av ROC AUC?
- [ ] En høy ROC AUC indikerer god evne til å skille mellom klassene over ulike terskler
- [ ] ROC AUC er sannsynligheten for at modellen rangerer et tilfeldig positivt eksempel høyere enn et tilfeldig negativt
- [x] En lav ROC AUC betyr at modellen har lav presisjon
- [ ] En ROC AUC på 0.5 betyr at modellen gjetter tilfeldig mellom klassene
? ROC-kurven er laget av TPR og FPR, ikke precision. Precision avhenger av terskelen og av klassefordelingen, så den kan ikke leses ut av AUC.
:::

::: oppsummering
- Lavere terskel gir flere positive: både TPR og FPR øker.
- ROC-kurven: TPR mot FPR for alle terskler. Diagonalen er tilfeldig gjetning.
- `fpr, tpr, thresholds = roc_curve(y_test, p)` og `roc_auc_score(y_test, p)`, der `p = model.predict_proba(X_test)[:, 1]`.
- AUC = sannsynligheten for at et tilfeldig positivt punkt får høyere score enn et tilfeldig negativt.
- Bruk sannsynligheter (scores), ikke 0/1-prediksjoner, til ROC.
:::

--- LEKSJON 6.5 | regresjonsmetrikker | Regresjonsmetrikker
For regresjon er det ingen «riktig» eller «feil», bare hvor langt unna prediksjonen er. De fire vanligste metrikkene:

| Metrikk | Formel | Tolkning |
|---|---|---|
| MSE | $\frac{1}{n}\sum (y_i - \hat y_i)^2$ | Kvadrert enhet, straffer store feil hardt |
| RMSE | $\sqrt{\text{MSE}}$ | Samme enhet som $y$: «typisk feil» |
| MAE | $\frac{1}{n}\sum \lvert y_i - \hat y_i\rvert$ | Gjennomsnittlig avvik, mindre følsom for uteliggere |
| $R^2$ | $1 - \frac{\sum (y_i - \hat y_i)^2}{\sum (y_i - \bar y)^2}$ | Andel av variasjonen i $y$ som modellen forklarer |

```python
import numpy as np

y_true = np.array([3.0, 5.0, 2.5, 7.0, 4.5])
y_pred = np.array([2.5, 5.0, 3.0, 8.0, 4.0])

mse = np.mean((y_true - y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_true - y_pred))
r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
print(f"MSE = {mse:.3f}, RMSE = {rmse:.3f}, MAE = {mae:.3f}, R² = {r2:.3f}")
```

## Med sklearn
`mean_squared_error`, `mean_absolute_error` og `r2_score` (sanne verdier først). RMSE er bare kvadratroten av MSE. Her på `regression.csv` fra prosjekt 1, med en linje tilpasset på treningsdelen:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv("regression.csv")
x, y = data["Net_Activity"].values, data["Energy"].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

w, b = np.polyfit(x_train, y_train, deg=1)     # samme linje som gradient descent finner
y_pred = w * x_test + b

mse = mean_squared_error(y_test, y_pred)
print(f"MSE  = {mse:.3f}")
print(f"RMSE = {np.sqrt(mse):.3f}")
print(f"MAE  = {mean_absolute_error(y_test, y_pred):.3f}")
print(f"R²   = {r2_score(y_test, y_pred):.4f}")
print(f"Standardavvik i y_test: {y_test.std():.2f}")
```

RMSE er rundt 2, mens `Energy` varierer med et standardavvik på rundt 17. Modellen bommer typisk med ±2, og forklarer nesten all variasjonen ($R^2 \approx 0.99$).

::: tips Er MSE = 960 høyt?
Det kommer an på skalaen til $y$. Forelesningsnotatene har et eksempel der MSE er 960 og targets er rundt 28: $\sqrt{960} \approx 31$, så den typiske feilen er like stor som verdiene selv. Det er dårlig. Sammenlign alltid RMSE med størrelsen og spredningen i $y$, eller bruk $R^2$, som ikke har enhet.
:::

## R² = 0 og negativ R²
En modell som alltid predikerer gjennomsnittet får $R^2 = 0$. Er modellen enda dårligere, blir $R^2$ negativ. Det er altså ikke et kvadrat av noe, og kan bli mindre enn 0:

```python
import numpy as np
from sklearn.metrics import r2_score

y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print("perfekt:         ", r2_score(y_true, y_true))
print("alltid snittet:  ", r2_score(y_true, np.full(5, y_true.mean())))
print("verre enn snittet:", r2_score(y_true, np.array([5.0, 4.0, 3.0, 2.0, 1.0])))
```

## Uteliggere: MSE mot MAE
Én stor feil dominerer MSE, fordi den kvadreres. MAE påvirkes mye mindre:

```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

y_true = np.array([10.0, 12.0, 11.0, 13.0, 12.0])
y_pred = np.array([10.5, 11.5, 11.0, 12.5, 12.0])
y_pred_uteligger = y_pred.copy()
y_pred_uteligger[0] = 30.0

for navn, yp in [("uten uteligger", y_pred), ("med uteligger", y_pred_uteligger)]:
    print(f"{navn}: MSE = {mean_squared_error(y_true, yp):6.2f}, MAE = {mean_absolute_error(y_true, yp):5.2f}")
```

::: prosjekt
Prosjekt 3 krevde **RMSE ≤ 0.25** på regresjonsoppgaven (`np.sqrt(mean_squared_error(y_test, y_pred))`). Prosjekt 4 (XAI) brukte `r2_score(y_test, y_pred)` for å sammenligne modellene før de ble forklart.
:::

::: oppgave RMSE, MAE og R² for hånd
Regn ut `rmse`, `mae` og `r2` for arrayene under med numpy (ikke sklearn).
```start
import numpy as np

y_true = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
y_pred = np.array([2.5, 3.5, 6.0, 9.0, 9.0, 12.5])

rmse = ...
mae = ...
r2 = ...
```
```losning
import numpy as np

y_true = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
y_pred = np.array([2.5, 3.5, 6.0, 9.0, 9.0, 12.5])

rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
mae = np.mean(np.abs(y_true - y_pred))
r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
```
```test
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
assert abs(rmse - np.sqrt(mean_squared_error(y_true, y_pred))) < 1e-9, f"rmse = {rmse} er feil. Ta kvadratroten av MSE."
assert abs(mae - mean_absolute_error(y_true, y_pred)) < 1e-9, f"mae = {mae} er feil. Bruk np.abs på feilene før gjennomsnittet."
assert abs(r2 - r2_score(y_true, y_pred)) < 1e-9, f"r2 = {r2} er feil. R² = 1 - SS_res / SS_tot, der SS_tot bruker gjennomsnittet av y_true."
```
:::

::: quiz
En regresjonsmodell predikerer alltid gjennomsnittet av `y_test`, uansett input. Hva blir `r2_score(y_test, y_pred)`?
- [ ] 1
- [x] 0
- [ ] -1
- [ ] Det kommer an på verdiene i `y_test`
? Når $\hat y_i = \bar y$ for alle $i$, er telleren og nevneren i brøken like, så $R^2 = 1 - 1 = 0$. Alle modeller som er dårligere enn dette, får negativ $R^2$.
:::

::: quiz
`y` er huspriser i kroner. Hvilken metrikk har samme enhet som `y` (kroner)?
- [ ] MSE
- [x] RMSE
- [ ] $R^2$
- [ ] Ingen av dem
? MSE har enhet kroner², og $R^2$ er uten enhet. RMSE = $\sqrt{\text{MSE}}$ er tilbake i kroner (det gjelder også MAE).
:::

::: oppsummering
- MSE = `np.mean((y_true - y_pred)**2)`, RMSE = `np.sqrt(mse)`, MAE = `np.mean(np.abs(y_true - y_pred))`.
- $R^2$ = `1 - SS_res / SS_tot`: 1 er perfekt, 0 er «alltid gjennomsnittet», negativ er verre.
- sklearn: `mean_squared_error`, `mean_absolute_error`, `r2_score` (sanne verdier først).
- RMSE og MAE har samme enhet som `y`. MSE er følsom for uteliggere.
:::

--- LEKSJON 6.6 | ubalanserte-klasser | Ubalanserte klasser
Forelesningsnotatene forteller om et jobbintervju: «Modellen vår får 99 % accuracy, men klarer nesten ikke å identifisere syke personer. Hva har skjedd?» Svaret er **ubalanserte klasser**. Er bare 1 % syke, får en modell som sier «frisk» til alle 99 % accuracy, men er helt ubrukelig.

## Majoritetsbaseline
Sammenlign alltid med den enkleste mulige modellen: predikér den vanligste klassen for alle. Det kan gjøres for hånd eller med `DummyClassifier`:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, recall_score

X, y = make_classification(n_samples=2000, n_features=5, n_informative=3,
                           weights=[0.95], class_sep=0.8, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print("andel positive:", y_test.mean())

# for hånd: alle får den vanligste klassen i treningsdataene
majoritet = np.bincount(y_train).argmax()
y_base = np.full(len(y_test), majoritet)
print("baseline accuracy:", accuracy_score(y_test, y_base), " recall:", recall_score(y_test, y_base))

dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print("DummyClassifier:  ", accuracy_score(y_test, dummy.predict(X_test)))
```

Accuracy 0.948 uten å se på dataene i det hele tatt. En ekte modell må slå **dette** tallet, ikke 0.5.

## Accuracy lurer, recall avslører
En vanlig logistisk regresjon på de samme dataene:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

X, y = make_classification(n_samples=2000, n_features=5, n_informative=3,
                           weights=[0.95], class_sep=0.8, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

def rapport(navn, y_pred):
    print(f"{navn:22s} accuracy={accuracy_score(y_test, y_pred):.3f}  "
          f"recall={recall_score(y_test, y_pred):.3f}  "
          f"precision={precision_score(y_test, y_pred, zero_division=0):.3f}")

model = LogisticRegression().fit(X_train, y_train)
p = model.predict_proba(X_test)[:, 1]
print("ROC AUC:", round(roc_auc_score(y_test, p), 3))
rapport("terskel 0.5", (p > 0.5).astype(int))
rapport("terskel 0.2", (p > 0.2).astype(int))

balansert = LogisticRegression(class_weight="balanced").fit(X_train, y_train)
rapport("class_weight=balanced", balansert.predict(X_test))
```

Med terskel 0.5 er accuracy høyere enn baseline, men recall er under 0.2: modellen finner bare én av fem positive. Både AUC og accuracy ser greie ut, så det er recall som avslører problemet.

## Hva kan vi gjøre?
- **Senke terskelen** (som notatene foreslår): flere positive prediksjoner, høyere recall, lavere precision. Modellen blir ikke bedre, vi endrer bare hvordan vi tolker den.
- **`class_weight="balanced"`**: sklearn vekter tapet slik at hver feil på den sjeldne klassen teller mer (omvendt proporsjonalt med klassestørrelsen). Recall øker kraftig, men precision faller.
- **Resampling**: undersampling (trekk like mange fra den store klassen som den lille har) eller oversampling (gjenta punkter fra den lille klassen). Gjøres bare på treningsdataene.
- **`stratify=y`** i `train_test_split`, så testsettet faktisk inneholder den sjeldne klassen.

## Hvilken metrikk når feilene koster ulikt?
| Situasjon | Dyreste feil | Metrikk å se på |
|---|---|---|
| Dødelig sykdom | FN (oversett syk) | Høy **recall** |
| Raketter mot passasjerfly (notatene) | FP (dyr rakett på tomt fly) | Lav **FPR**, høy **precision** |
| Ubalanserte data generelt | Den sjeldne klassen overses | Recall, precision, F1, ikke bare accuracy |

::: oppgave Slå baseline
`y_test` er gitt. Lag majoritetsbaselinen `y_base` som predikerer 0 for alle punktene, og regn ut `acc_base` (accuracy) og `recall_base` (recall for klasse 1) for den.
```start
import numpy as np

rng = np.random.default_rng(3)
y_test = (rng.random(400) < 0.03).astype(int)     # ca. 3 % positive

y_base = ...
acc_base = ...
recall_base = ...
```
```losning
import numpy as np

rng = np.random.default_rng(3)
y_test = (rng.random(400) < 0.03).astype(int)

y_base = np.zeros(len(y_test), dtype=int)
acc_base = np.mean(y_test == y_base)
recall_base = np.sum((y_base == 1) & (y_test == 1)) / np.sum(y_test == 1)
```
```test
import numpy as np
assert np.shape(y_base) == np.shape(y_test) and np.all(np.asarray(y_base) == 0), "y_base skal være 0 for alle punktene, med samme lengde som y_test."
assert abs(acc_base - np.mean(y_test == 0)) < 1e-9, f"acc_base = {acc_base}, men det skal være andelen nuller i y_test ({np.mean(y_test == 0):.4f})."
assert abs(recall_base) < 1e-9, f"recall_base = {recall_base}. En modell som aldri sier 1, finner ingen positive: recall skal være 0."
```
:::

::: quiz
Du skal levere en modell til klimaaktivister som vil skyte raketter på **tomme** passasjerfly. Rakettene er dyre, så det er alvorlig om modellen gir en falsk positiv. Hvilken metrikk bør modellen skåre **lavt** på?
- [ ] Recall (TPR)
- [x] FPR
- [ ] Spesifisitet
- [ ] Precision
? FPR = FP / (FP + TN) er andelen negative som feilaktig ble predikert positive. Den skal være lav når falske positive er dyre. Spesifisitet og precision skal være **høye**. Eksempelet er fra forelesningsnotatene.
:::

::: quiz
En modell for å finne en sjelden sykdom (1 % av pasientene) får 99 % accuracy på testsettet. Hva er den mest sannsynlige forklaringen?
- [ ] Modellen er svært god og klar til bruk
- [ ] Modellen er overtilpasset treningsdataene
- [x] Modellen predikerer «frisk» for nesten alle, og recall for syke er trolig svært lav
- [ ] Testsettet er for stort
? Med 99 % friske gir «alle er friske» 99 % accuracy. Sjekk recall og forvirringsmatrisen, og sammenlign med majoritetsbaselinen.
:::

::: oppsummering
- Med ubalanserte klasser kan accuracy være høy selv om modellen er ubrukelig.
- Sammenlign alltid med majoritetsbaselinen (`DummyClassifier(strategy="most_frequent")` eller for hånd).
- Se på recall, precision, F1 og forvirringsmatrisen.
- Tiltak: senke terskelen, `class_weight="balanced"`, resampling, og `stratify=y` i splitten.
- FN dyrest: høy recall. FP dyrest: lav FPR / høy precision.
:::
