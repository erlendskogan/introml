=== DEL 6 | evaluering | Evaluering: hvor god er modellen?
Her regner du ut accuracy, forvirringsmatrise, precision og recall, ROC og AUC og regresjonsmetrikker, først med numpy og så med `sklearn.metrics`.

--- LEKSJON 6.1 | accuracy-split | Accuracy og train/test-split
`np.mean(y_true == y_pred)` gir accuracy, fordi gjennomsnittet av `True`/`False` er andelen `True`. `accuracy_score(y_true, y_pred)` fra `sklearn.metrics` gjør det samme, og som alle metrikker tar den sanne verdier først.

::: oppgave Accuracy to ganger
Regn ut accuracy med numpy i `acc` og med `accuracy_score` i `acc_sk`.
```start
import numpy as np
from sklearn.metrics import accuracy_score

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])

acc = ...
acc_sk = ...
```
```losning
import numpy as np
from sklearn.metrics import accuracy_score

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])

acc = np.mean(y_true == y_pred)
acc_sk = accuracy_score(y_true, y_pred)
```
```test
assert acc is not ... and abs(acc - 0.75) < 1e-9, f"acc = {acc}, forventet 6/8 = 0.75. Bruk np.mean(y_true == y_pred)."
assert acc_sk is not ... and abs(acc_sk - 0.75) < 1e-9, f"acc_sk = {acc_sk}, forventet 0.75. Bruk accuracy_score(y_true, y_pred)."
```
:::

## Splitt for hånd
Modellen skal måles på data den ikke er trent på: `perm = rng.permutation(len(y))` gir en tilfeldig rekkefølge, og de samme indeksene brukes på både `X` og `y` så radene hører sammen.

::: oppgave Splitt for hånd
Lag `X_train, X_test, y_train, y_test` med 80 tilfeldige punkter til trening og 20 til test, med `rng.permutation`.
```start
import numpy as np

X = np.arange(200).reshape(100, 2)
y = np.arange(100) % 3
rng = np.random.default_rng(0)

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

## train_test_split
`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` fra `sklearn.model_selection` returnerer alltid X-delene først. `random_state` gir samme splitt hver gang, og `stratify=y` gir samme klasseandel i begge delene.

::: oppgave Stratifisert splitt
Del dataene med 20 % til test, `random_state=42` og stratifisering, og lagre andelen positive i treningsdelen i `andel_train` og i testdelen i `andel_test`.
```start
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, 1 - data.target      # 1 = ondartet

X_train, X_test, y_train, y_test = ..., ..., ..., ...
andel_train = ...
andel_test = ...
```
```losning
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, 1 - data.target      # 1 = ondartet

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
andel_train = y_train.mean()
andel_test = y_test.mean()
```
```test
import numpy as np
assert np.shape(X_train) == (455, 30) and np.shape(X_test) == (114, 30), f"X_train skal ha shape (455, 30) og X_test (114, 30), men har {np.shape(X_train)} og {np.shape(X_test)}. Rekkefølgen er X_train, X_test, y_train, y_test."
assert np.shape(y_train) == (455,) and np.shape(y_test) == (114,), "y_train skal ha 455 og y_test 114 elementer. Sjekk rekkefølgen: X_train, X_test, y_train, y_test."
f = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
assert np.array_equal(y_test, f[3]), "Bruk test_size=0.2, random_state=42 og stratify=y."
assert abs(andel_train - y_train.mean()) < 1e-9 and abs(andel_test - y_test.mean()) < 1e-9, "Andelen positive er gjennomsnittet av y-delen, f.eks. y_train.mean()."
```
:::

## Hele løpet
Standardiser med `mu, sigma` fra `X_train` (aldri fra testdataene), tren med `model.fit(X_train, y_train)` og rapporter accuracy på testdelen.

::: oppgave Train- og test-accuracy
Standardiser `X_train` og `X_test` med `mu` og `sigma` fra `X_train`, tren `model`, og lagre accuracy på treningsdelen i `train_acc` og på testdelen i `test_acc`.
```start
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X, y = data.data, 1 - data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression()

train_acc = ...
test_acc = ...
```
```losning
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X, y = data.data, 1 - data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = LogisticRegression()

mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train, X_test = (X_train - mu) / sigma, (X_test - mu) / sigma
model.fit(X_train, y_train)
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))
```
```test
import numpy as np
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
m, s = Xtr.mean(axis=0), Xtr.std(axis=0)
fasit = LogisticRegression().fit((Xtr - m) / s, ytr)
assert np.allclose(X_train.mean(axis=0), 0, atol=1e-6), "X_train skal være standardisert: (X_train - mu) / sigma."
assert np.allclose(X_test, (Xte - m) / s), "X_test skal standardiseres med mu og sigma fra X_train, ikke fra testdataene."
assert train_acc is not ... and abs(train_acc - fasit.score((Xtr - m) / s, ytr)) < 1e-9, "train_acc skal være accuracy_score(y_train, model.predict(X_train))."
assert test_acc is not ... and abs(test_acc - fasit.score((Xte - m) / s, yte)) < 1e-9, "test_acc skal være accuracy_score(y_test, model.predict(X_test))."
```
:::

--- LEKSJON 6.2 | forvirringsmatrise | Forvirringsmatrise
TP, TN, FP og FN telles med masker, f.eks. `fp = np.sum((y_true == 0) & (y_pred == 1))`. Siste ord er hva modellen sa (positiv/negativ), første ord om den hadde rett (sann/falsk).

::: oppgave Tell for hånd
Regn ut `tp`, `tn`, `fp` og `fn` uten sklearn.
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

## confusion_matrix
`confusion_matrix(y_true, y_pred)` har sanne klasser som rader og predikerte som kolonner, så `tn, fp, fn, tp = cm.ravel()`. Bytter du argumentene, blir matrisen transponert uten feilmelding.

::: oppgave Matrisen fra sklearn
Lag `cm` med `confusion_matrix`, og pakk den ut i `tn, fp, fn, tp`.
```start
import numpy as np
from sklearn.metrics import confusion_matrix

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1])

cm = ...
tn, fp, fn, tp = ..., ..., ..., ...
```
```losning
import numpy as np
from sklearn.metrics import confusion_matrix

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1])

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
```
```test
import numpy as np
assert isinstance(cm, np.ndarray) and cm.shape == (2, 2), "cm skal være en 2×2-matrise fra confusion_matrix(y_true, y_pred)."
assert np.array_equal(cm, [[3, 2], [1, 4]]), f"cm er feil:\n{cm}\nHar du byttet om y_true og y_pred? Sanne verdier skal først."
assert (tn, fp, fn, tp) == (3, 2, 1, 4), f"Fikk tn={tn}, fp={fp}, fn={fn}, tp={tp}. Rekkefølgen etter ravel() er tn, fp, fn, tp."
```
:::

::: oppgave Finn feilen
Koden under gir feil `fn`. Rett linjen så `tn, fp, fn, tp` blir riktige.
```start
import numpy as np
from sklearn.metrics import confusion_matrix

y_true = np.array([1, 1, 1, 0, 0])
y_pred = np.array([1, 0, 0, 0, 1])

tn, fp, fn, tp = confusion_matrix(y_pred, y_true).ravel()
```
```losning
import numpy as np
from sklearn.metrics import confusion_matrix

y_true = np.array([1, 1, 1, 0, 0])
y_pred = np.array([1, 0, 0, 0, 1])

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
```
```test
assert fn == 2, f"fn = {fn}, men to syke (y_true = 1) ble predikert 0. Sanne verdier skal først: confusion_matrix(y_true, y_pred)."
assert (tn, fp, fn, tp) == (1, 1, 2, 1), f"Fikk tn={tn}, fp={fp}, fn={fn}, tp={tp}."
```
:::

## ConfusionMatrixDisplay
`ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=[...], cmap="Blues")` tegner matrisen med tall i hver rute.

::: oppgave Tegn matrisen
Tegn forvirringsmatrisen for `y_test` og `y_pred` med navnene `"godartet"` og `"ondartet"` og `cmap="Blues"`, og lagre resultatet i `disp`.
```start
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay

data = load_breast_cancer()
X, y = data.data, 1 - data.target                  # 1 = ondartet
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
model = LogisticRegression().fit((X_train - mu) / sigma, y_train)
y_pred = model.predict((X_test - mu) / sigma)

disp = ...
```
```losning
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay

data = load_breast_cancer()
X, y = data.data, 1 - data.target                  # 1 = ondartet
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
model = LogisticRegression().fit((X_train - mu) / sigma, y_train)
y_pred = model.predict((X_test - mu) / sigma)

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["godartet", "ondartet"], cmap="Blues")
```
```test
import numpy as np
from sklearn.metrics import confusion_matrix
assert hasattr(disp, "confusion_matrix"), "disp skal være resultatet av ConfusionMatrixDisplay.from_predictions(...)."
assert np.array_equal(disp.confusion_matrix, confusion_matrix(y_test, y_pred)), "Matrisen er feil. Gi sanne verdier først: from_predictions(y_test, y_pred, ...)."
assert list(disp.display_labels) == ["godartet", "ondartet"], "Bruk display_labels=['godartet', 'ondartet']."
assert disp.im_.get_cmap().name == "Blues", "Bruk cmap='Blues'."
```
:::

--- LEKSJON 6.3 | precision-recall | Precision, recall, FPR og F1
Recall (TPR) = `tp / (tp + fn)`, spesifisitet = `tn / (tn + fp)`, FPR = `fp / (fp + tn)`, precision = `tp / (tp + fp)` og F1 = `2 * precision * recall / (precision + recall)`. Recall og FPR deler på en rad (en sann klasse), precision på en kolonne (det modellen sa).

::: oppgave Regn ut metrikkene
Regn ut `tpr`, `fpr`, `spesifisitet`, `precision` og `f1` (positiv klasse er 1).
```start
import numpy as np
from sklearn.metrics import confusion_matrix

y_test = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
y_pred = np.array([1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1])

tpr = ...
fpr = ...
spesifisitet = ...
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
spesifisitet = tn / (tn + fp)
precision = tp / (tp + fp)
f1 = 2 * precision * tpr / (precision + tpr)
```
```test
assert tpr is not ... and abs(tpr - 0.6) < 1e-9, f"tpr = {tpr}, forventet 3/5 = 0.6. TPR = TP / (TP + FN)."
assert fpr is not ... and abs(fpr - 2 / 7) < 1e-9, f"fpr = {fpr}, forventet 2/7 ≈ 0.286. FPR = FP / (FP + TN)."
assert spesifisitet is not ... and abs(spesifisitet - 5 / 7) < 1e-9, f"spesifisitet = {spesifisitet}, forventet 5/7 ≈ 0.714. Spesifisitet = TN / (TN + FP) = 1 - FPR."
assert precision is not ... and abs(precision - 0.6) < 1e-9, f"precision = {precision}, forventet 3/5 = 0.6. Precision = TP / (TP + FP)."
assert f1 is not ... and abs(f1 - 0.6) < 1e-9, f"f1 = {f1}, forventet 0.6. F1 = 2 * precision * recall / (precision + recall)."
```
:::

## Med sklearn
`precision_score`, `recall_score`, `f1_score` og `classification_report` tar sanne verdier først, og `recall_score(..., pos_label=0)` (recall for klasse 0) er spesifisiteten. FPR har ingen egen funksjon, og `zero_division=0` fjerner advarselen når modellen aldri predikerer 1.

::: oppgave Metrikker med sklearn
Regn ut `p`, `r` og `f` (precision, recall, F1) og `spes` (spesifisitet) med sklearn, og lagre `classification_report(y_true, y_pred)` i `rapport`.
```start
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1])

p = ...
r = ...
f = ...
spes = ...
rapport = ...
print(rapport)
```
```losning
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1])

p = precision_score(y_true, y_pred)
r = recall_score(y_true, y_pred)
f = f1_score(y_true, y_pred)
spes = recall_score(y_true, y_pred, pos_label=0)
rapport = classification_report(y_true, y_pred)
print(rapport)
```
```test
assert p is not ... and abs(p - 4 / 6) < 1e-9, f"p = {p}, forventet 4/6. Bruk precision_score(y_true, y_pred)."
assert r is not ... and abs(r - 0.8) < 1e-9, f"r = {r}, forventet 0.8. Bruk recall_score(y_true, y_pred)."
assert f is not ... and abs(f - f1_score(y_true, y_pred)) < 1e-9, "Bruk f1_score(y_true, y_pred)."
assert spes is not ... and abs(spes - 0.6) < 1e-9, f"spes = {spes}, forventet 3/5 = 0.6. Bruk recall_score(y_true, y_pred, pos_label=0)."
assert isinstance(rapport, str) and "precision" in rapport, "rapport skal være classification_report(y_true, y_pred)."
```
:::

::: oppgave Ingen positive prediksjoner
Lag funksjonen `trygg_precision(y_true, y_pred)` som returnerer precision og gir 0 uten advarsel når `y_pred` ikke har noen 1-ere.
```start
from sklearn.metrics import precision_score

def trygg_precision(y_true, y_pred):
    ...
```
```losning
from sklearn.metrics import precision_score

def trygg_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)
```
```test
import warnings
import numpy as np
yt = np.array([1, 0, 1, 0])
with warnings.catch_warnings():
    warnings.simplefilter("error")
    try:
        verdi = trygg_precision(yt, np.zeros(4, dtype=int))
    except Warning:
        raise AssertionError("Funksjonen gir en advarsel når y_pred bare har nuller. Bruk zero_division=0.")
assert verdi == 0, f"Uten positive prediksjoner skal precision være 0, men funksjonen ga {verdi}."
assert abs(trygg_precision(yt, np.array([1, 1, 0, 0])) - 0.5) < 1e-9, "Med y_pred = [1, 1, 0, 0] skal precision være 0.5."
```
:::

::: oppgave Rapporter TPR og FPR
Regn ut `tpr` og `fpr` for brystkreftmodellen fra `confusion_matrix`, og lag teksten `tekst = f"True positive rate: {tpr:.3f}"` som i prosjekt 2.
```start
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

data = load_breast_cancer()
X, y = data.data, 1 - data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
y_pred = LogisticRegression().fit((X_train - mu) / sigma, y_train).predict((X_test - mu) / sigma)

tpr = ...
fpr = ...
tekst = ...
print(tekst)
```
```losning
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

data = load_breast_cancer()
X, y = data.data, 1 - data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
y_pred = LogisticRegression().fit((X_train - mu) / sigma, y_train).predict((X_test - mu) / sigma)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
tekst = f"True positive rate: {tpr:.3f}"
print(tekst)
```
```test
from sklearn.metrics import confusion_matrix
tn_, fp_, fn_, tp_ = confusion_matrix(y_test, y_pred).ravel()
assert tpr is not ... and abs(tpr - tp_ / (tp_ + fn_)) < 1e-9, "tpr skal være tp / (tp + fn)."
assert fpr is not ... and abs(fpr - fp_ / (fp_ + tn_)) < 1e-9, "fpr skal være fp / (fp + tn)."
assert tekst == f"True positive rate: {tp_ / (tp_ + fn_):.3f}", f"tekst skal være f'True positive rate: {{tpr:.3f}}', men er {tekst!r}."
```
:::

--- LEKSJON 6.4 | roc-auc | Terskler, ROC og AUC
`p = model.predict_proba(X_test)[:, 1]` gir sannsynligheten for klasse 1, og `(p >= terskel).astype(int)` gir prediksjoner for en valgfri terskel (`predict` bruker 0.5). Lavere terskel gir flere positive, og dermed høyere TPR og høyere FPR.

::: oppgave To terskler
Lag `p`, prediksjonene `y_pred_05` og `y_pred_02` for tersklene 0.5 og 0.2, og TPR for hver i `tpr_05` og `tpr_02`.
```start
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()
X, y = data.data[:, [1, 4]], 1 - data.target      # bare to features: en svakere modell
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train, X_test = (X_train - mu) / sigma, (X_test - mu) / sigma
model = LogisticRegression().fit(X_train, y_train)

p = ...
y_pred_05 = ...
y_pred_02 = ...
tpr_05 = ...
tpr_02 = ...
```
```losning
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()
X, y = data.data[:, [1, 4]], 1 - data.target      # bare to features: en svakere modell
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train, X_test = (X_train - mu) / sigma, (X_test - mu) / sigma
model = LogisticRegression().fit(X_train, y_train)

p = model.predict_proba(X_test)[:, 1]
y_pred_05 = (p >= 0.5).astype(int)
y_pred_02 = (p >= 0.2).astype(int)
tpr_05 = np.sum((y_pred_05 == 1) & (y_test == 1)) / np.sum(y_test == 1)
tpr_02 = np.sum((y_pred_02 == 1) & (y_test == 1)) / np.sum(y_test == 1)
```
```test
import numpy as np
fasit_p = model.predict_proba(X_test)[:, 1]
assert isinstance(p, np.ndarray) and p.shape == (114,), "p skal ha én sannsynlighet per testpunkt: model.predict_proba(X_test)[:, 1]."
assert np.allclose(p, fasit_p), "p skal være kolonne 1 (sannsynligheten for klasse 1): predict_proba(X_test)[:, 1]."
assert np.array_equal(np.asarray(y_pred_05), (fasit_p >= 0.5).astype(int)), "y_pred_05 skal være (p >= 0.5).astype(int)."
assert np.array_equal(np.asarray(y_pred_02), (fasit_p >= 0.2).astype(int)), "y_pred_02 skal være (p >= 0.2).astype(int)."
pos = np.sum(y_test == 1)
assert abs(tpr_05 - np.sum((fasit_p >= 0.5) & (y_test == 1)) / pos) < 1e-9, "tpr_05 skal være TP / antall positive for terskel 0.5."
assert abs(tpr_02 - np.sum((fasit_p >= 0.2) & (y_test == 1)) / pos) < 1e-9, "tpr_02 skal være TP / antall positive for terskel 0.2."
```
:::

## ROC-kurven for hånd
ROC-kurven er TPR mot FPR for alle terskler fra 1 til 0, og diagonalen fra (0, 0) til (1, 1) er tilfeldig gjetning.

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

::: oppgave Hele kurven
Fyll `fpr_liste` og `tpr_liste` for hver terskel i `np.linspace(1, 0, 101)`, plott TPR mot FPR, og tegn diagonalen med `plt.plot([0, 1], [0, 1], "k--")`.
```start
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
y_test = np.array([0] * 60 + [1] * 40)
p = np.clip(np.where(y_test == 1, rng.normal(0.65, 0.2, 100), rng.normal(0.35, 0.2, 100)), 0, 0.999)
fig, ax = plt.subplots()

fpr_liste, tpr_liste = [], []
# løkke og plott her
```
```losning
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
y_test = np.array([0] * 60 + [1] * 40)
p = np.clip(np.where(y_test == 1, rng.normal(0.65, 0.2, 100), rng.normal(0.35, 0.2, 100)), 0, 0.999)
fig, ax = plt.subplots()

fpr_liste, tpr_liste = [], []
for terskel in np.linspace(1, 0, 101):
    y_pred = (p >= terskel).astype(int)
    fpr_liste.append(np.sum((y_pred == 1) & (y_test == 0)) / np.sum(y_test == 0))
    tpr_liste.append(np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1))
plt.plot(fpr_liste, tpr_liste, label="Modell")
plt.plot([0, 1], [0, 1], "k--", label="Tilfeldig gjetning")
plt.xlabel("FPR")
plt.ylabel("TPR")
```
```test
import numpy as np
assert len(fpr_liste) == 101 and len(tpr_liste) == 101, f"Det skal være 101 punkter (én per terskel), men det er {len(fpr_liste)}."
ts = np.linspace(1, 0, 101)
ff = [np.sum((p >= t) & (y_test == 0)) / 60 for t in ts]
ft = [np.sum((p >= t) & (y_test == 1)) / 40 for t in ts]
assert np.allclose(fpr_liste, ff) and np.allclose(tpr_liste, ft), "Verdiene stemmer ikke. Bruk p >= terskel, og del på antall negative (FPR) og positive (TPR)."
assert any(len(l.get_xdata()) == 101 and np.allclose(l.get_xdata(), ff) and np.allclose(l.get_ydata(), ft) for l in ax.lines), "Plott tpr_liste mot fpr_liste: plt.plot(fpr_liste, tpr_liste)."
assert any(len(l.get_xdata()) == 2 and np.allclose(l.get_xdata(), [0, 1]) and np.allclose(l.get_ydata(), [0, 1]) for l in ax.lines), "Tegn diagonalen med plt.plot([0, 1], [0, 1], 'k--')."
```
:::

## roc_curve og roc_auc_score
`fpr, tpr, thresholds = roc_curve(y_test, p)` gir kurven (første terskel er `inf`), `roc_auc_score(y_test, p)` gir arealet under den (1 er perfekt, 0.5 er myntkast), og `auc(fpr, tpr)` gir arealet under en hvilken som helst kurve. Bruk alltid sannsynligheter, ikke 0/1 fra `predict`.

::: oppgave Kurve og areal med sklearn
Lag `fpr, tpr, thresholds` med `roc_curve`, og regn ut AUC både med `roc_auc_score` (`auc_score`) og med `auc` (`auc_areal`).
```start
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc

data = load_breast_cancer()
X, y = data.data[:, [1, 4]], 1 - data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
model = LogisticRegression().fit((X_train - mu) / sigma, y_train)
p = model.predict_proba((X_test - mu) / sigma)[:, 1]

fpr, tpr, thresholds = ..., ..., ...
auc_score = ...
auc_areal = ...
```
```losning
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc

data = load_breast_cancer()
X, y = data.data[:, [1, 4]], 1 - data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
model = LogisticRegression().fit((X_train - mu) / sigma, y_train)
p = model.predict_proba((X_test - mu) / sigma)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, p)
auc_score = roc_auc_score(y_test, p)
auc_areal = auc(fpr, tpr)
```
```test
import numpy as np
ff, ft, fth = roc_curve(y_test, p)
assert np.shape(fpr) == np.shape(ff) and np.allclose(fpr, ff) and np.allclose(tpr, ft), "Bruk fpr, tpr, thresholds = roc_curve(y_test, p) (sanne verdier først, så sannsynlighetene)."
assert np.isinf(thresholds[0]), "thresholds skal komme fra roc_curve (første terskel er inf)."
assert auc_score is not ... and abs(auc_score - roc_auc_score(y_test, p)) < 1e-9, "auc_score skal være roc_auc_score(y_test, p)."
assert auc_areal is not ... and abs(auc_areal - auc(ff, ft)) < 1e-9, "auc_areal skal være auc(fpr, tpr)."
```
:::

::: oppgave Finn feilen i AUC
Koden gir `roc_auc_score` ferdige klasser fra `predict`. Rett den så `auc_verdi` bruker sannsynligheter.
```start
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

data = load_breast_cancer()
X, y = data.data[:, [1, 4]], 1 - data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train, X_test = (X_train - mu) / sigma, (X_test - mu) / sigma
model = LogisticRegression().fit(X_train, y_train)

auc_verdi = roc_auc_score(y_test, model.predict(X_test))
```
```losning
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

data = load_breast_cancer()
X, y = data.data[:, [1, 4]], 1 - data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train, X_test = (X_train - mu) / sigma, (X_test - mu) / sigma
model = LogisticRegression().fit(X_train, y_train)

auc_verdi = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
```
```test
riktig = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
assert abs(auc_verdi - riktig) < 1e-9, f"auc_verdi = {auc_verdi:.3f}, men med sannsynligheter blir den {riktig:.3f}. Bruk model.predict_proba(X_test)[:, 1]."
```
:::

## Hva betyr AUC?
ROC AUC er sannsynligheten for at et tilfeldig positivt punkt får høyere score enn et tilfeldig negativt, så den avhenger bare av rekkefølgen på scorene. `pos[:, None] > neg[None, :]` sammenligner alle par med broadcasting.

::: oppgave AUC som andel par
Lag `pos` og `neg` (scorene til de positive og negative punktene), og lagre andelen av alle par der positiv score er høyest i `andel`.
```start
import numpy as np
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
y = np.array([0] * 50 + [1] * 30)
p = np.where(y == 1, rng.normal(0.65, 0.15, 80), rng.normal(0.4, 0.15, 80))

pos = ...
neg = ...
andel = ...
```
```losning
import numpy as np
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
y = np.array([0] * 50 + [1] * 30)
p = np.where(y == 1, rng.normal(0.65, 0.15, 80), rng.normal(0.4, 0.15, 80))

pos = p[y == 1]
neg = p[y == 0]
andel = np.mean(pos[:, None] > neg[None, :])
```
```test
import numpy as np
assert np.shape(pos) == (30,) and np.shape(neg) == (50,), "pos skal være p[y == 1] (30 scorer) og neg p[y == 0] (50 scorer)."
assert andel is not ... and abs(andel - roc_auc_score(y, p)) < 1e-9, f"andel = {andel}, men roc_auc_score gir {roc_auc_score(y, p):.4f}. Sammenlign alle 30 × 50 par: np.mean(pos[:, None] > neg[None, :])."
```
:::

--- LEKSJON 6.5 | regresjonsmetrikker | Regresjonsmetrikker
`mse = np.mean((y_true - y_pred) ** 2)`, `rmse = np.sqrt(mse)` (samme enhet som `y`) og `mae = np.mean(np.abs(y_true - y_pred))`. $R^2 = 1 - \frac{\sum (y_i - \hat y_i)^2}{\sum (y_i - \bar y)^2}$ er andelen av variasjonen i $y$ som modellen forklarer.

::: oppgave MSE, RMSE, MAE og R² for hånd
Regn ut `mse`, `rmse`, `mae` og `r2` med numpy (ikke sklearn).
```start
import numpy as np

y_true = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
y_pred = np.array([2.5, 3.5, 6.0, 9.0, 9.0, 12.5])

mse = ...
rmse = ...
mae = ...
r2 = ...
```
```losning
import numpy as np

y_true = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
y_pred = np.array([2.5, 3.5, 6.0, 9.0, 9.0, 12.5])

mse = np.mean((y_true - y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_true - y_pred))
r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
```
```test
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
assert mse is not ... and abs(mse - mean_squared_error(y_true, y_pred)) < 1e-9, f"mse = {mse} er feil. Kvadrer feilene før gjennomsnittet."
assert rmse is not ... and abs(rmse - np.sqrt(mean_squared_error(y_true, y_pred))) < 1e-9, f"rmse = {rmse} er feil. Ta kvadratroten av MSE."
assert mae is not ... and abs(mae - mean_absolute_error(y_true, y_pred)) < 1e-9, f"mae = {mae} er feil. Bruk np.abs på feilene før gjennomsnittet."
assert r2 is not ... and abs(r2 - r2_score(y_true, y_pred)) < 1e-9, f"r2 = {r2} er feil. R² = 1 - SS_res / SS_tot, der SS_tot bruker gjennomsnittet av y_true."
```
:::

## Med sklearn
`mean_squared_error`, `mean_absolute_error` og `r2_score` fra `sklearn.metrics` tar sanne verdier først, og RMSE er `np.sqrt(mean_squared_error(y_test, y_pred))`. Om RMSE er stor eller liten, ser du ved å sammenligne med spredningen i `y`.

::: oppgave Linjen på prosjekt 1-dataene
Regn ut `rmse`, `mae` og `r2` for `y_pred` med sklearn, og lagre `rmse` delt på standardavviket til `y_test` i `forhold`.
```start
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv("regression.csv")
x, y = data["Net_Activity"].values, data["Energy"].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
w, b = np.polyfit(x_train, y_train, deg=1)
y_pred = w * x_test + b

rmse = ...
mae = ...
r2 = ...
forhold = ...
```
```losning
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv("regression.csv")
x, y = data["Net_Activity"].values, data["Energy"].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
w, b = np.polyfit(x_train, y_train, deg=1)
y_pred = w * x_test + b

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
forhold = rmse / y_test.std()
```
```test
import numpy as np
fr = np.sqrt(mean_squared_error(y_test, y_pred))
assert rmse is not ... and abs(rmse - fr) < 1e-9, f"rmse = {rmse}, forventet {fr:.4f}. Ta np.sqrt av mean_squared_error(y_test, y_pred)."
assert mae is not ... and abs(mae - mean_absolute_error(y_test, y_pred)) < 1e-9, "Bruk mean_absolute_error(y_test, y_pred)."
assert r2 is not ... and abs(r2 - r2_score(y_test, y_pred)) < 1e-9, "Bruk r2_score(y_test, y_pred). Sanne verdier først, ellers blir R² feil."
assert forhold is not ... and abs(forhold - fr / y_test.std()) < 1e-9, "forhold skal være rmse / y_test.std()."
```
:::

## R² = 0 og negativ R²
En modell som alltid predikerer gjennomsnittet får $R^2 = 0$, og en som er dårligere enn det får negativ $R^2$.

::: oppgave Verre enn gjennomsnittet
Lag `y_snitt` som predikerer gjennomsnittet av `y_true` for alle punktene, og en egen prediksjon `y_verre` (samme lengde) som gir negativ $R^2$. Lagre R² for dem i `r2_snitt` og `r2_verre`.
```start
import numpy as np
from sklearn.metrics import r2_score

y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

y_snitt = ...
y_verre = ...
r2_snitt = ...
r2_verre = ...
```
```losning
import numpy as np
from sklearn.metrics import r2_score

y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

y_snitt = np.full(len(y_true), y_true.mean())
y_verre = y_true[::-1]
r2_snitt = r2_score(y_true, y_snitt)
r2_verre = r2_score(y_true, y_verre)
```
```test
import numpy as np
assert np.shape(y_snitt) == (5,) and np.allclose(y_snitt, 3.0), "y_snitt skal være gjennomsnittet (3.0) for alle 5 punktene: np.full(len(y_true), y_true.mean())."
assert r2_snitt is not ... and abs(r2_snitt) < 1e-9, f"r2_snitt = {r2_snitt}, men alltid-gjennomsnittet gir R² = 0."
assert np.shape(y_verre) == (5,), "y_verre skal ha 5 prediksjoner."
assert r2_verre is not ... and abs(r2_verre - r2_score(y_true, y_verre)) < 1e-9, "r2_verre skal være r2_score(y_true, y_verre)."
assert r2_verre < 0, f"R² for y_verre er {r2_verre:.3f}. Velg prediksjoner som bommer mer enn gjennomsnittet, f.eks. y_true[::-1]."
```
:::

## Uteliggere: MSE mot MAE
Én stor feil dominerer MSE fordi den kvadreres, mens MAE påvirkes mye mindre. Bruk `.copy()` når du endrer en kopi av et array.

::: oppgave Én uteligger
Lag `y_pred_ut` som en kopi av `y_pred` med første verdi satt til 30.0, og lagre hvor mange ganger større MSE og MAE blir i `mse_faktor` og `mae_faktor`.
```start
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

y_true = np.array([10.0, 12.0, 11.0, 13.0, 12.0])
y_pred = np.array([10.5, 11.5, 11.0, 12.5, 12.0])

y_pred_ut = ...
mse_faktor = ...
mae_faktor = ...
```
```losning
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

y_true = np.array([10.0, 12.0, 11.0, 13.0, 12.0])
y_pred = np.array([10.5, 11.5, 11.0, 12.5, 12.0])

y_pred_ut = y_pred.copy()
y_pred_ut[0] = 30.0
mse_faktor = mean_squared_error(y_true, y_pred_ut) / mean_squared_error(y_true, y_pred)
mae_faktor = mean_absolute_error(y_true, y_pred_ut) / mean_absolute_error(y_true, y_pred)
```
```test
import numpy as np
assert np.allclose(y_pred, [10.5, 11.5, 11.0, 12.5, 12.0]), "y_pred er endret. Lag en kopi med y_pred.copy() før du endrer den."
assert np.allclose(y_pred_ut, [30.0, 11.5, 11.0, 12.5, 12.0]), "y_pred_ut skal være y_pred med første verdi satt til 30.0."
assert mse_faktor is not ... and abs(mse_faktor - 400.5 / 0.75) < 1e-6, f"mse_faktor = {mse_faktor}, forventet {400.5 / 0.75:.0f}. Del MSE med uteligger på MSE uten."
assert mae_faktor is not ... and abs(mae_faktor - 21 / 1.5) < 1e-6, f"mae_faktor = {mae_faktor}, forventet {21 / 1.5:.0f}. Del MAE med uteligger på MAE uten."
```
:::

--- LEKSJON 6.6 | ubalanserte-klasser | Ubalanserte klasser
Med 1 % syke får «alle er friske» 99 % accuracy, så sammenlign alltid med majoritetsbaselinen `np.bincount(y_train).argmax()` eller `DummyClassifier(strategy="most_frequent")` fra `sklearn.dummy`.

::: oppgave Majoritetsbaseline
Lag `y_base` som predikerer 0 for alle punktene, og regn ut `acc_base` (accuracy) og `recall_base` (recall for klasse 1) for den.
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
y_test = (rng.random(400) < 0.03).astype(int)     # ca. 3 % positive

y_base = np.zeros(len(y_test), dtype=int)
acc_base = np.mean(y_test == y_base)
recall_base = np.sum((y_base == 1) & (y_test == 1)) / np.sum(y_test == 1)
```
```test
import numpy as np
assert np.shape(y_base) == np.shape(y_test) and np.all(np.asarray(y_base) == 0), "y_base skal være 0 for alle punktene, med samme lengde som y_test."
assert acc_base is not ... and abs(acc_base - np.mean(y_test == 0)) < 1e-9, f"acc_base = {acc_base}, men det skal være andelen nuller i y_test ({np.mean(y_test == 0):.4f})."
assert recall_base is not ... and abs(recall_base) < 1e-9, f"recall_base = {recall_base}. En modell som aldri sier 1, finner ingen positive: recall skal være 0."
```
:::

::: oppgave DummyClassifier
Lagre den vanligste klassen i `y_train` i `majoritet` med `np.bincount`, tren `dummy = DummyClassifier(strategy="most_frequent")`, og lagre accuracy på testdelen i `acc_dummy`.
```start
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=2000, n_features=5, n_informative=3, weights=[0.95], class_sep=0.8, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

majoritet = ...
dummy = ...
acc_dummy = ...
```
```losning
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=2000, n_features=5, n_informative=3, weights=[0.95], class_sep=0.8, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

majoritet = np.bincount(y_train).argmax()
dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
acc_dummy = accuracy_score(y_test, dummy.predict(X_test))
```
```test
import numpy as np
assert majoritet is not ... and majoritet == 0, f"majoritet = {majoritet}. Den vanligste klassen er np.bincount(y_train).argmax()."
assert hasattr(dummy, "predict") and getattr(dummy, "strategy", None) == "most_frequent", "dummy skal være DummyClassifier(strategy='most_frequent')."
assert hasattr(dummy, "classes_"), "Husk å trene dummy med .fit(X_train, y_train)."
assert acc_dummy is not ... and abs(acc_dummy - np.mean(y_test == 0)) < 1e-9, f"acc_dummy skal være accuracy_score(y_test, dummy.predict(X_test)) = {np.mean(y_test == 0):.3f}."
```
:::

## Accuracy lurer, recall avslører
Senk terskelen (`(p >= 0.2).astype(int)`) eller bruk `LogisticRegression(class_weight="balanced")` for å finne flere positive, på bekostning av precision. `precision, recall, terskler = precision_recall_curve(y_test, p)` viser avveiningen for alle terskler.

::: oppgave Tre måter å finne de sjeldne
Regn ut recall på testdelen for `model` med terskel 0.5 (`recall_05`) og 0.2 (`recall_02`), og for en `LogisticRegression(class_weight="balanced")` trent på treningsdelen (`recall_bal`).
```start
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

X, y = make_classification(n_samples=2000, n_features=5, n_informative=3, weights=[0.95], class_sep=0.8, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model = LogisticRegression().fit(X_train, y_train)
p = model.predict_proba(X_test)[:, 1]

recall_05 = ...
recall_02 = ...
recall_bal = ...
```
```losning
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

X, y = make_classification(n_samples=2000, n_features=5, n_informative=3, weights=[0.95], class_sep=0.8, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model = LogisticRegression().fit(X_train, y_train)
p = model.predict_proba(X_test)[:, 1]

recall_05 = recall_score(y_test, (p >= 0.5).astype(int))
recall_02 = recall_score(y_test, (p >= 0.2).astype(int))
balansert = LogisticRegression(class_weight="balanced").fit(X_train, y_train)
recall_bal = recall_score(y_test, balansert.predict(X_test))
```
```test
import numpy as np
assert recall_05 is not ... and abs(recall_05 - recall_score(y_test, (p >= 0.5).astype(int))) < 1e-9, "recall_05 skal være recall_score(y_test, (p >= 0.5).astype(int))."
assert recall_02 is not ... and abs(recall_02 - recall_score(y_test, (p >= 0.2).astype(int))) < 1e-9, "recall_02 skal være recall for terskel 0.2."
fasit = recall_score(y_test, LogisticRegression(class_weight="balanced").fit(X_train, y_train).predict(X_test))
assert recall_bal is not ... and abs(recall_bal - fasit) < 1e-9, "recall_bal skal komme fra LogisticRegression(class_weight='balanced') trent på X_train, y_train og testet på X_test."
```
:::

::: oppgave Precision–recall-kurven
Lag `precision, recall, terskler` med `precision_recall_curve`, og lagre den høyeste recall der precision er minst 0.5 i `beste_recall`.
```start
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

X, y = make_classification(n_samples=2000, n_features=5, n_informative=3, weights=[0.95], class_sep=0.8, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
p = LogisticRegression().fit(X_train, y_train).predict_proba(X_test)[:, 1]

precision, recall, terskler = ..., ..., ...
beste_recall = ...
```
```losning
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

X, y = make_classification(n_samples=2000, n_features=5, n_informative=3, weights=[0.95], class_sep=0.8, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
p = LogisticRegression().fit(X_train, y_train).predict_proba(X_test)[:, 1]

precision, recall, terskler = precision_recall_curve(y_test, p)
beste_recall = recall[precision >= 0.5].max()
```
```test
import numpy as np
fp_, fr_, _ = precision_recall_curve(y_test, p)
assert np.shape(precision) == np.shape(fp_) and np.allclose(precision, fp_) and np.allclose(recall, fr_), "Bruk precision, recall, terskler = precision_recall_curve(y_test, p)."
assert beste_recall is not ... and abs(beste_recall - fr_[fp_ >= 0.5].max()) < 1e-9, "beste_recall skal være recall[precision >= 0.5].max()."
```
:::

## Resampling
Undersampling trekker like mange fra den store klassen som den lille har, og gjøres bare på treningsdataene (`stratify=y` i splitten sørger for at testdelen har den sjeldne klassen).

::: oppgave Undersampling
Lag `X_bal, y_bal` med alle positive punkter i treningsdelen pluss like mange tilfeldig valgte negative (`rng.choice(..., replace=False)`).
```start
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=2000, n_features=5, n_informative=3, weights=[0.95], class_sep=0.8, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
rng = np.random.default_rng(0)

X_bal, y_bal = ..., ...
```
```losning
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=2000, n_features=5, n_informative=3, weights=[0.95], class_sep=0.8, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
rng = np.random.default_rng(0)

pos_idx = np.where(y_train == 1)[0]
neg_idx = rng.choice(np.where(y_train == 0)[0], size=len(pos_idx), replace=False)
idx = np.concatenate([pos_idx, neg_idx])
X_bal, y_bal = X_train[idx], y_train[idx]
```
```test
import numpy as np
n_pos = int(np.sum(y_train == 1))
assert isinstance(y_bal, np.ndarray) and np.sum(y_bal == 1) == n_pos, f"y_bal skal ha alle {n_pos} positive fra y_train."
assert np.sum(y_bal == 0) == n_pos, f"y_bal skal ha like mange negative som positive ({n_pos}), men har {np.sum(y_bal == 0)}."
assert np.shape(X_bal) == (2 * n_pos, 5), f"X_bal skal ha shape {(2 * n_pos, 5)}."
oppslag = {tuple(rad): lab for rad, lab in zip(X_train, y_train)}
assert all(tuple(rad) in oppslag and oppslag[tuple(rad)] == lab for rad, lab in zip(X_bal, y_bal)), "Radene i X_bal skal komme fra X_train, med samme indekser for X og y (ikke fra testdelen)."
assert len({tuple(rad) for rad in X_bal}) == len(X_bal), "Ingen punkter skal være med to ganger. Bruk replace=False."
```
:::

## Hvilken metrikk når feilene koster ulikt?
Er FN dyrest (en oversett syk pasient), skal recall være høy. Er FP dyrest (en dyr rakett mot et tomt fly), skal FPR være lav og precision høy.

::: oppgave Velg modell
Regn ut `recall_a`, `recall_b`, `fpr_a` og `fpr_b`, og lagre `"a"` eller `"b"` i `valg_sykdom` (FN dyrest) og `valg_rakett` (FP dyrest).
```start
import numpy as np

y_true = np.array([1] * 10 + [0] * 10)
y_pred_a = np.array([1] * 9 + [0] * 1 + [1] * 4 + [0] * 6)
y_pred_b = np.array([1] * 6 + [0] * 4 + [1] * 1 + [0] * 9)

recall_a = ...
recall_b = ...
fpr_a = ...
fpr_b = ...
valg_sykdom = ...
valg_rakett = ...
```
```losning
import numpy as np

y_true = np.array([1] * 10 + [0] * 10)
y_pred_a = np.array([1] * 9 + [0] * 1 + [1] * 4 + [0] * 6)
y_pred_b = np.array([1] * 6 + [0] * 4 + [1] * 1 + [0] * 9)

recall_a = np.sum((y_pred_a == 1) & (y_true == 1)) / np.sum(y_true == 1)
recall_b = np.sum((y_pred_b == 1) & (y_true == 1)) / np.sum(y_true == 1)
fpr_a = np.sum((y_pred_a == 1) & (y_true == 0)) / np.sum(y_true == 0)
fpr_b = np.sum((y_pred_b == 1) & (y_true == 0)) / np.sum(y_true == 0)
valg_sykdom = "a"
valg_rakett = "b"
```
```test
assert recall_a is not ... and abs(recall_a - 0.9) < 1e-9 and abs(recall_b - 0.6) < 1e-9, f"Recall skal være 0.9 for a og 0.6 for b, men er {recall_a} og {recall_b}."
assert fpr_a is not ... and abs(fpr_a - 0.4) < 1e-9 and abs(fpr_b - 0.1) < 1e-9, f"FPR skal være 0.4 for a og 0.1 for b, men er {fpr_a} og {fpr_b}."
assert valg_sykdom == "a", "For sykdommen er FN dyrest, så velg modellen med høyest recall."
assert valg_rakett == "b", "For rakettene er FP dyrest, så velg modellen med lavest FPR."
```
:::
