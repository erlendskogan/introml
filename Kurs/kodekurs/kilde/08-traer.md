=== DEL 8 | traer | Trær, ensembler og Naive Bayes (prosjekt 2)
Du trener og tolker beslutningstrær, regner Gini og entropi for hånd, bygger bagging, random forest og boosting, regner ut Naive Bayes med pandas og løser hele prosjekt 2.

--- LEKSJON 8.1 | beslutningstre | DecisionTreeClassifier
`DecisionTreeClassifier(max_depth=3, random_state=0)` brukes med `fit`/`predict` som andre sklearn-modeller, og i `problem1.csv` sier kolonnen `split` om en rad er `"train"` eller `"test"`.

::: oppgave Del dataene og tell klassene
Lag `train` og `test` ved å filtrere på kolonnen `split`, og lagre `train["y"].value_counts()` i `antall_train`.
```start
import pandas as pd

df = pd.read_csv("problem1.csv")

train = ...
test = ...
antall_train = ...
```
```losning
import pandas as pd

df = pd.read_csv("problem1.csv")

train = df[df["split"] == "train"]
test = df[df["split"] == "test"]
antall_train = train["y"].value_counts()
```
```test
import pandas as pd
assert isinstance(train, pd.DataFrame) and isinstance(test, pd.DataFrame), 'train og test skal være DataFrames, f.eks. df[df["split"] == "train"].'
assert len(train) == 1006 and len(test) == 300, f"Forventet 1006 treningsrader og 300 testrader, fikk {len(train)} og {len(test)}."
assert isinstance(antall_train, pd.Series), 'antall_train skal være train["y"].value_counts().'
assert int(antall_train.loc[0]) == 930 and int(antall_train.loc[1]) == 76, "antall_train skal telle klassene i train (930 av klasse 0, 76 av klasse 1)."
```
:::

::: oppgave Tre uten balansering
Tren `clf = DecisionTreeClassifier(max_depth=3, random_state=0)` på `x0` og `x1` i treningsdataene, og lagre train- og test-accuracy i `train_acc` og `test_acc`.
```start
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("problem1.csv")
train, test = df[df["split"] == "train"], df[df["split"] == "test"]
X_train, y_train = train[["x0", "x1"]], train["y"]
X_test, y_test = test[["x0", "x1"]], test["y"]

clf = ...
train_acc = ...
test_acc = ...
```
```losning
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("problem1.csv")
train, test = df[df["split"] == "train"], df[df["split"] == "test"]
X_train, y_train = train[["x0", "x1"]], train["y"]
X_test, y_test = test[["x0", "x1"]], test["y"]

clf = DecisionTreeClassifier(max_depth=3, random_state=0)
clf.fit(X_train, y_train)
train_acc = accuracy_score(y_train, clf.predict(X_train))
test_acc = accuracy_score(y_test, clf.predict(X_test))
```
```test
from sklearn.tree import DecisionTreeClassifier
assert isinstance(clf, DecisionTreeClassifier), "clf skal være en DecisionTreeClassifier."
assert hasattr(clf, "tree_"), "Husk å kalle clf.fit(X_train, y_train)."
assert clf.max_depth == 3 and clf.class_weight is None, "Bruk max_depth=3 og ingen class_weight her."
assert abs(train_acc - clf.score(X_train, y_train)) < 1e-9, "train_acc skal være accuracy på treningsdataene."
assert abs(test_acc - clf.score(X_test, y_test)) < 1e-9, "test_acc skal være accuracy på testdataene."
```
:::

## Balansering
`class_weight="balanced"` gir minoritetsklassen større vekt, og `mino.sample(len(maj), replace=True, random_state=0)` oversampler den i stedet; `confusion_matrix(y_test, y_pred).ravel()` gir `tn, fp, fn, tp`.

::: oppgave Balansert tre med TPR og FPR
Tren `clf` med `criterion="entropy"`, `max_depth=2`, `class_weight="balanced"` og `random_state=0`, og lagre `acc`, `tpr = tp / (tp + fn)` og `fpr = fp / (fp + tn)` på testdataene.
```start
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("problem1.csv")
train, test = df[df["split"] == "train"], df[df["split"] == "test"]
X_train, y_train = train[["x0", "x1"]], train["y"]
X_test, y_test = test[["x0", "x1"]], test["y"]

clf = ...
acc = ...
tpr = ...
fpr = ...
```
```losning
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("problem1.csv")
train, test = df[df["split"] == "train"], df[df["split"] == "test"]
X_train, y_train = train[["x0", "x1"]], train["y"]
X_test, y_test = test[["x0", "x1"]], test["y"]

clf = DecisionTreeClassifier(criterion="entropy", max_depth=2, class_weight="balanced", random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
acc = accuracy_score(y_test, y_pred)
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
```
```test
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
assert isinstance(clf, DecisionTreeClassifier) and hasattr(clf, "tree_"), "clf skal være et trent DecisionTreeClassifier."
assert clf.criterion == "entropy", 'Bruk criterion="entropy".'
assert clf.get_depth() <= 2, "Bruk max_depth=2."
assert clf.class_weight == "balanced", 'Husk class_weight="balanced".'
_tn, _fp, _fn, _tp = confusion_matrix(y_test, clf.predict(X_test)).ravel()
assert abs(acc - (_tn + _tp) / len(y_test)) < 1e-9 and acc > 0.88, f"acc skal være test-accuracy (over 0.88), fikk {acc}."
assert abs(tpr - _tp / (_tp + _fn)) < 1e-9, "tpr skal være tp / (tp + fn). Rekkefølgen fra ravel() er tn, fp, fn, tp."
assert abs(fpr - _fp / (_fp + _tn)) < 1e-9, "fpr skal være fp / (fp + tn). Rekkefølgen fra ravel() er tn, fp, fn, tp."
```
:::

::: oppgave Oversampling med pandas
Lag `balansert` ved å trekke `len(maj)` rader fra `mino` med tilbakelegging og slå dem sammen med `maj` (`pd.concat`), tren `DecisionTreeClassifier(max_depth=3, random_state=0)` på den, og lagre test-accuracy i `acc_over`.
```start
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("problem1.csv")
train, test = df[df["split"] == "train"], df[df["split"] == "test"]
maj = train[train["y"] == 0]
mino = train[train["y"] == 1]

balansert = ...
acc_over = ...
```
```losning
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("problem1.csv")
train, test = df[df["split"] == "train"], df[df["split"] == "test"]
maj = train[train["y"] == 0]
mino = train[train["y"] == 1]

mino_opp = mino.sample(len(maj), replace=True, random_state=0)
balansert = pd.concat([maj, mino_opp])
clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(balansert[["x0", "x1"]], balansert["y"])
acc_over = clf.score(test[["x0", "x1"]], test["y"])
```
```test
import pandas as pd
assert isinstance(balansert, pd.DataFrame), "balansert skal være en DataFrame (pd.concat([maj, mino_opp]))."
_vc = balansert["y"].value_counts()
assert len(_vc) == 2 and int(_vc.loc[0]) == 930 and int(_vc.loc[1]) == 930, f"Begge klassene skal ha 930 rader i balansert, fikk {_vc.to_dict()}. Brukte du replace=True?"
assert isinstance(acc_over, float) and acc_over > 0.8, f"acc_over skal være test-accuracy for treet trent på balansert (over 0.8), fikk {acc_over}."
```
:::

## Beslutningsflate og skalering
`Z = clf.predict(grid).reshape(xx0.shape)` og `ax.contourf(xx0, xx1, Z, alpha=0.4)` tegner beslutningsflaten, og den har bare akseparallelle grenser fordi hver split er en terskel `x0 <= t` på én feature.

::: oppgave Tegn beslutningsflaten
Regn ut `Z` for rutenettet, tegn den med `ax.contourf`, og legg treningspunktene oppå med `ax.scatter(train["x0"], train["x1"], c=train["y"], s=8)`.
```start
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("problem1.csv")
train = df[df["split"] == "train"]
clf = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0).fit(train[["x0", "x1"]], train["y"])

xx0, xx1 = np.meshgrid(np.arange(-7, 6, 0.05), np.arange(-6, 6, 0.05))
grid = pd.DataFrame({"x0": xx0.ravel(), "x1": xx1.ravel()})
fig, ax = plt.subplots()

Z = ...
```
```losning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("problem1.csv")
train = df[df["split"] == "train"]
clf = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0).fit(train[["x0", "x1"]], train["y"])

xx0, xx1 = np.meshgrid(np.arange(-7, 6, 0.05), np.arange(-6, 6, 0.05))
grid = pd.DataFrame({"x0": xx0.ravel(), "x1": xx1.ravel()})
fig, ax = plt.subplots()

Z = clf.predict(grid).reshape(xx0.shape)
ax.contourf(xx0, xx1, Z, alpha=0.4, cmap="coolwarm")
ax.scatter(train["x0"], train["x1"], c=train["y"], cmap="coolwarm", s=8)
plt.show()
```
```test
import numpy as np
assert isinstance(Z, np.ndarray) and Z.shape == xx0.shape, "Z skal ha samme shape som xx0. Bruk clf.predict(grid).reshape(xx0.shape)."
assert np.array_equal(Z, clf.predict(grid).reshape(xx0.shape)), "Z skal være prediksjonene til clf på grid."
assert len(ax.collections) >= 2, "Tegn både ax.contourf(...) og ax.scatter(...) på ax."
```
:::

::: oppgave Trenger trær skalering?
Lag `X_skalert = X * [1000, 0.001]`, tren to like trær (`max_depth=4, random_state=0`) på `X` og `X_skalert`, lagre prediksjonene (på hvert sitt treningssett) i `a` og `b`, og lagre `np.array_equal(a, b)` i `like`.
```start
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("problem1.csv")
X, y = df[["x0", "x1"]], df["y"]

X_skalert = ...
a = ...
b = ...
like = ...
```
```losning
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("problem1.csv")
X, y = df[["x0", "x1"]], df["y"]

X_skalert = X * [1000, 0.001]
a = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y).predict(X)
b = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_skalert, y).predict(X_skalert)
like = np.array_equal(a, b)
```
```test
import numpy as np
assert X_skalert is not ... and np.allclose(np.asarray(X_skalert, dtype=float), np.asarray(X, dtype=float) * [1000, 0.001]), "X_skalert skal være X * [1000, 0.001]."
assert a is not ... and b is not ... and len(a) == len(y) and len(b) == len(y), "a og b skal være prediksjonene på hele X og hele X_skalert."
assert like is not ... and bool(like) == np.array_equal(a, b), "like skal være np.array_equal(a, b)."
assert bool(like), "Prediksjonene skal være like: en terskel på én feature påvirkes ikke av skalering. Brukte du samme max_depth og random_state?"
```
:::

--- LEKSJON 8.2 | gini-entropi | Gini og entropi for hånd
Gini er $G = 1 - \sum_k p_k^2$ og entropi er $H = -\sum_k p_k \log_2 p_k$, der andelene $p_k$ kommer fra `_, antall = np.unique(y, return_counts=True)`.

::: oppgave Skriv gini(y)
Skriv `gini(y)` som tar en liste eller et array med klasselabels og returnerer Gini-urenheten. Den skal virke for to eller flere klasser.
```start
import numpy as np

def gini(y):
    ...
```
```losning
import numpy as np

def gini(y):
    _, antall = np.unique(y, return_counts=True)
    p = antall / len(y)
    return 1 - np.sum(p**2)
```
```test
import numpy as np
assert gini([0, 0, 0, 0]) is not None, "gini må returnere et tall (husk return)."
assert abs(gini([0, 0, 0, 0]) - 0.0) < 1e-9, "En ren node skal ha Gini 0."
assert abs(gini([0, 1, 0, 1]) - 0.5) < 1e-9, "50/50 med to klasser skal gi 0.5."
assert abs(gini(np.array([1]*30 + [0]*10)) - 0.375) < 1e-9, "value = [10, 30] skal gi 1 - (0.75² + 0.25²) = 0.375."
assert abs(gini([0, 1, 2]) - 2/3) < 1e-9, "Tre like store klasser skal gi 2/3."
```
:::

::: oppgave Skriv entropi(y)
Skriv `entropi(y)` som returnerer $-\sum_k p_k \log_2 p_k$, og som gir 0 (ikke `nan`) for en ren node.
```start
import numpy as np

def entropi(y):
    ...
```
```losning
import numpy as np

def entropi(y):
    _, antall = np.unique(y, return_counts=True)
    p = antall / len(y)
    return -np.sum(p * np.log2(p))
```
```test
import numpy as np
assert entropi([1, 1, 1]) is not None, "entropi må returnere et tall (husk return)."
assert abs(entropi([1, 1, 1])) < 1e-9, "En ren node skal ha entropi 0. Bruk andelene fra np.unique, så slipper du log2(0)."
assert abs(entropi([0, 1, 0, 1]) - 1.0) < 1e-9, "50/50 med to klasser skal gi 1."
assert abs(entropi(np.array([1]*9 + [0]*5)) - 0.9403) < 1e-3, "9 mot 5 skal gi ca. 0.940. Bruk np.log2, ikke np.log."
assert abs(entropi([0, 1, 2]) - np.log2(3)) < 1e-9, "Tre like store klasser skal gi log2(3)."
```
:::

::: oppgave Gini og entropi som kurver
For andelen `p` av klasse 1, lag `g = 1 - p**2 - (1 - p)**2` og den tilsvarende entropien `h`, og plott begge på `ax` med `label`.
```start
import numpy as np
import matplotlib.pyplot as plt

p = np.linspace(0.001, 0.999, 200)
fig, ax = plt.subplots()

g = ...
h = ...
```
```losning
import numpy as np
import matplotlib.pyplot as plt

p = np.linspace(0.001, 0.999, 200)
fig, ax = plt.subplots()

g = 1 - p**2 - (1 - p)**2
h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
ax.plot(p, g, label="Gini")
ax.plot(p, h, label="Entropi")
ax.set_xlabel("andel av klasse 1")
ax.legend()
plt.show()
```
```test
import numpy as np
assert isinstance(g, np.ndarray) and np.allclose(g, 1 - p**2 - (1 - p)**2), "g skal være 1 - p**2 - (1 - p)**2 (et array)."
assert isinstance(h, np.ndarray) and np.allclose(h, -p * np.log2(p) - (1 - p) * np.log2(1 - p)), "h skal være -p·log2(p) - (1-p)·log2(1-p)."
assert len(ax.lines) >= 2, "Plott både g og h på ax."
```
:::

## Urenhet etter split og informasjonsgevinst
Urenheten etter en split er det vektede snittet $\frac{n_V}{n} I_V + \frac{n_H}{n} I_H$, og gevinsten er $I_{\text{før}}$ minus dette.

::: oppgave Informasjonsgevinst
Skriv `urenhet_etter(y_v, y_h, mal)` der `mal` er `gini` eller `entropi`, og lagre entropi-gevinsten når `y` deles i `y_v` og `y_h` i `gevinst`.
```start
import numpy as np

def gini(y):
    _, antall = np.unique(y, return_counts=True)
    p = antall / len(y)
    return 1 - np.sum(p**2)

def entropi(y):
    _, antall = np.unique(y, return_counts=True)
    p = antall / len(y)
    return -np.sum(p * np.log2(p))

def urenhet_etter(y_v, y_h, mal):
    ...

y = np.array([1]*9 + [0]*5)
y_v = np.array([1]*4 + [0]*5)
y_h = np.array([1]*5)
gevinst = ...
```
```losning
import numpy as np

def gini(y):
    _, antall = np.unique(y, return_counts=True)
    p = antall / len(y)
    return 1 - np.sum(p**2)

def entropi(y):
    _, antall = np.unique(y, return_counts=True)
    p = antall / len(y)
    return -np.sum(p * np.log2(p))

def urenhet_etter(y_v, y_h, mal):
    n = len(y_v) + len(y_h)
    return len(y_v) / n * mal(y_v) + len(y_h) / n * mal(y_h)

y = np.array([1]*9 + [0]*5)
y_v = np.array([1]*4 + [0]*5)
y_h = np.array([1]*5)
gevinst = entropi(y) - urenhet_etter(y_v, y_h, entropi)
```
```test
import numpy as np
_r = urenhet_etter(np.array([0, 0, 1, 1]), np.array([1, 1]), gini)
assert _r is not None, "urenhet_etter må returnere et tall (husk return)."
assert abs(_r - 4/6 * 0.5) < 1e-9, "urenhet_etter skal vekte barna med len(y_v)/n og len(y_h)/n, der n er antall punkter totalt."
assert abs(urenhet_etter(y_v, y_h, entropi) - 9/14 * entropi(y_v)) < 1e-9, "urenhet_etter skal bruke funksjonen mal på hvert barn."
assert gevinst is not ... and abs(gevinst - 0.3032) < 1e-3, f"gevinst skal være entropi(y) - urenhet_etter(y_v, y_h, entropi), ca. 0.303. Fikk {gevinst}."
```
:::

## Beste split for én feature
Treet prøver midtpunktene `(v[:-1] + v[1:]) / 2` mellom de sorterte unike verdiene `v = np.unique(x)`, sender `x <= t` til venstre og velger terskelen med lavest vektet urenhet.

::: oppgave Skriv beste_split(x, y)
Skriv `beste_split(x, y)` som returnerer `(t, g)`: terskelen med lavest vektet Gini og den vektede Gini-verdien. Testen sammenligner også med sklearn på `x0` i `problem1.csv`.
```start
import numpy as np

def gini(y):
    _, antall = np.unique(y, return_counts=True)
    p = antall / len(y)
    return 1 - np.sum(p**2)

def beste_split(x, y):
    ...

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
t, g = beste_split(x, y)
```
```losning
import numpy as np

def gini(y):
    _, antall = np.unique(y, return_counts=True)
    p = antall / len(y)
    return 1 - np.sum(p**2)

def beste_split(x, y):
    verdier = np.unique(x)
    terskler = (verdier[:-1] + verdier[1:]) / 2
    beste_t, beste_g = None, np.inf
    for t in terskler:
        venstre, hoyre = y[x <= t], y[x > t]
        g = (len(venstre) * gini(venstre) + len(hoyre) * gini(hoyre)) / len(y)
        if g < beste_g:
            beste_t, beste_g = t, g
    return beste_t, beste_g

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
t, g = beste_split(x, y)
```
```test
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
_res = beste_split(np.array([1, 2, 3, 4, 5, 6, 7, 8]), np.array([0, 0, 0, 1, 0, 1, 1, 1]))
assert isinstance(_res, tuple) and len(_res) == 2, "beste_split skal returnere (t, g)."
assert abs(_res[1] - 0.2) < 1e-9, f"Laveste vektede Gini på eksempelet er 0.2, fikk {_res[1]}. Vekter du med len(venstre) og len(hoyre)?"
assert _res[0] in (3.5, 5.5), f"Beste terskel skal være et midtpunkt (3.5 eller 5.5), fikk {_res[0]}."
_df = pd.read_csv("problem1.csv")
_x, _y = _df["x0"].to_numpy(), _df["y"].to_numpy()
_t, _g = beste_split(_x, _y)
_ts = DecisionTreeClassifier(max_depth=1).fit(_df[["x0"]], _y).tree_.threshold[0]
_gs = (np.sum(_x <= _ts) * gini(_y[_x <= _ts]) + np.sum(_x > _ts) * gini(_y[_x > _ts])) / len(_y)
assert _g <= _gs + 1e-9, f"På x0 fant sklearn en split med vektet Gini {_gs:.4f}, men beste_split fant {_g:.4f}. Prøver du alle midtpunktene?"
```
:::

--- LEKSJON 8.3 | tolke-treet | Visualisere og tolke treet
`plot_tree(clf, feature_names=list(X.columns), class_names=["died", "survived"], filled=True, ax=ax)` tegner treet, og hver node viser spørsmålet (sant går til venstre), `gini`, `samples`, `value` (antall per klasse) og `class` (flertallet).

::: oppgave Tegn treet
Tren `clf = DecisionTreeClassifier(max_depth=2, random_state=0)` på `X, y` og tegn det på `ax` med `plot_tree`, med feature-navn og klassenavnene `"died"` og `"survived"`.
```start
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv("problem2.csv")
X = pd.get_dummies(df[["sex", "role", "armor"]].fillna("None"), dtype=int)
y = df["survived"]
fig, ax = plt.subplots(figsize=(12, 5))

clf = ...
```
```losning
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv("problem2.csv")
X = pd.get_dummies(df[["sex", "role", "armor"]].fillna("None"), dtype=int)
y = df["survived"]
fig, ax = plt.subplots(figsize=(12, 5))

clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
plot_tree(clf, feature_names=list(X.columns), class_names=["died", "survived"], filled=True, ax=ax)
plt.show()
```
```test
from sklearn.tree import DecisionTreeClassifier
assert isinstance(clf, DecisionTreeClassifier) and hasattr(clf, "tree_"), "clf skal være et trent DecisionTreeClassifier."
assert clf.get_depth() == 2, "Bruk max_depth=2."
_tekster = [t.get_text() for t in ax.texts]
assert len(_tekster) >= clf.tree_.node_count, "Tegn treet på ax med plot_tree(clf, ..., ax=ax)."
assert any("armor" in t for t in _tekster), "Gi feature_names=list(X.columns), ellers står det bare x[3] i nodene."
assert any("survived" in t for t in _tekster), 'Gi class_names=["died", "survived"].'
```
:::

## Les en node
Fra `value = [278, 222]` er andelene $p = (278/500,\ 222/500)$, så $G = 1 - \sum_k p_k^2$, og `class` er klassen med flest punkter.

::: oppgave Gini fra value
Regn Gini for rotnoden `value_rot` og løvnoden `value_blad` med numpy, og lagre flertallsklassen i rotnoden (`"died"` eller `"survived"`) i `rot_klasse`.
```start
import numpy as np

klasser = ["died", "survived"]
value_rot = np.array([278, 222])
value_blad = np.array([15, 79])

rot_gini = ...
blad_gini = ...
rot_klasse = ...
```
```losning
import numpy as np

klasser = ["died", "survived"]
value_rot = np.array([278, 222])
value_blad = np.array([15, 79])

p = value_rot / value_rot.sum()
rot_gini = 1 - np.sum(p**2)
q = value_blad / value_blad.sum()
blad_gini = 1 - np.sum(q**2)
rot_klasse = klasser[np.argmax(value_rot)]
```
```test
assert rot_gini is not ... and abs(rot_gini - 0.4938) < 1e-3, f"rot_gini skal være ca. 0.494. Del value på summen før du kvadrerer. Fikk {rot_gini}."
assert blad_gini is not ... and abs(blad_gini - 0.2683) < 1e-3, f"blad_gini skal være ca. 0.268, fikk {blad_gini}."
assert rot_klasse == "died", 'rot_klasse skal være klassen med flest punkter, altså klasser[np.argmax(value_rot)].'
```
:::

## Treet som tekst og som tall
`export_text(clf, feature_names=list(X.columns))` gir treet som tekst, og `t = clf.tree_` har arrayene `t.feature`, `t.threshold`, `t.impurity`, `t.n_node_samples` og `t.children_left`, der løvnoder har `children_left == -1` og `feature == -2`.

::: oppgave Les roten med kode
Finn navnet på featuren rotnoden splitter på (`rot_feature`, via `clf.tree_.feature[0]` og `X.columns`), Gini i roten (`rot_gini`) og treet som tekst (`tekst`).
```start
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

df = pd.read_csv("problem2.csv")
X = pd.get_dummies(df[["sex", "role", "ship", "destination"]], dtype=int)
y = df["survived"]
clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)

rot_feature = ...
rot_gini = ...
tekst = ...
```
```losning
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

df = pd.read_csv("problem2.csv")
X = pd.get_dummies(df[["sex", "role", "ship", "destination"]], dtype=int)
y = df["survived"]
clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)

rot_feature = X.columns[clf.tree_.feature[0]]
rot_gini = clf.tree_.impurity[0]
tekst = export_text(clf, feature_names=list(X.columns))
print(tekst)
```
```test
assert isinstance(rot_feature, str), "rot_feature skal være et kolonnenavn (tekst). Slå opp i X.columns."
assert rot_feature == X.columns[clf.tree_.feature[0]], f"Rotnoden splitter ikke på {rot_feature}. Bruk clf.tree_.feature[0] som indeks i X.columns."
_p = y.mean()
assert rot_gini is not ... and abs(rot_gini - (1 - _p**2 - (1 - _p)**2)) < 1e-6, "rot_gini skal være Gini i rotnoden (clf.tree_.impurity[0])."
assert isinstance(tekst, str) and "|---" in tekst, "tekst skal være export_text(clf, ...)."
assert rot_feature in tekst, "Gi export_text feature_names=list(X.columns), så kolonnenavnene står i teksten."
```
:::

::: oppgave Løvnoder og indre noder
Med `t = clf.tree_`: lag den boolske masken `er_blad` (løvnoder har `t.children_left == -1`), lagre antall løvnoder i `n_blad` og tersklene til de indre nodene i `terskler`.
```start
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("problem2.csv")
X = pd.get_dummies(df[["sex", "role", "ship", "destination"]], dtype=int)
y = df["survived"]
clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
t = clf.tree_

er_blad = ...
n_blad = ...
terskler = ...
```
```losning
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("problem2.csv")
X = pd.get_dummies(df[["sex", "role", "ship", "destination"]], dtype=int)
y = df["survived"]
clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
t = clf.tree_

er_blad = t.children_left == -1
n_blad = er_blad.sum()
terskler = t.threshold[~er_blad]
```
```test
import numpy as np
assert isinstance(er_blad, np.ndarray) and er_blad.dtype == bool and len(er_blad) == t.node_count, "er_blad skal være en boolsk maske med én verdi per node: t.children_left == -1."
assert np.all(t.feature[er_blad] == -2), "er_blad skal velge løvnodene (de har feature == -2)."
assert n_blad == clf.get_n_leaves(), f"n_blad skal være antall løvnoder ({clf.get_n_leaves()}), fikk {n_blad}."
assert terskler is not ... and np.allclose(np.sort(terskler), np.sort(t.threshold[t.feature >= 0])), "terskler skal være t.threshold for de indre nodene: t.threshold[~er_blad]."
```
:::

## Feature importance
`clf.feature_importances_` summerer til 1 og viser hvor mye hver feature reduserte urenheten, altså hvor mye den ble brukt, ikke om den forårsaker noe.

::: oppgave Rangér featurene
Lag `viktighet = pd.Series(clf.feature_importances_, index=X.columns)` sortert synkende, og lagre navnet på den viktigste featuren i `viktigst`.
```start
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("problem2.csv")
X = pd.get_dummies(df[["sex", "role", "armor"]].fillna("None"), dtype=int)
y = df["survived"]
clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)

viktighet = ...
viktigst = ...
```
```losning
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("problem2.csv")
X = pd.get_dummies(df[["sex", "role", "armor"]].fillna("None"), dtype=int)
y = df["survived"]
clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)

viktighet = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
viktigst = viktighet.index[0]
```
```test
import numpy as np
import pandas as pd
assert isinstance(viktighet, pd.Series), "viktighet skal være en pandas Series med X.columns som indeks."
assert abs(viktighet.sum() - 1) < 1e-9, "viktighet skal inneholde clf.feature_importances_ (summerer til 1)."
assert list(viktighet.values) == sorted(viktighet.values, reverse=True), "Sorter synkende med .sort_values(ascending=False)."
assert viktigst == X.columns[np.argmax(clf.feature_importances_)], "viktigst skal være navnet på featuren med høyest importance."
```
:::

--- LEKSJON 8.4 | dybde-regresjonstre | Dybde, overtilpasning og regresjonstrær
Et tre med `max_depth=None` splitter til alle løvnoder er rene og overtilpasser; begrens det med `max_depth`, `min_samples_leaf` eller `min_samples_split`, og velg verdien med kryssvalidering, ikke ved å se på testdataene.

::: oppgave Accuracy mot dybde
For `d` i `range(1, 16)`: tren `DecisionTreeClassifier(max_depth=d, class_weight="balanced", random_state=0)`, legg train- og test-accuracy i `tr_acc` og `te_acc`, og plott begge kurvene på `ax`.
```start
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("problem1.csv")
train, test = df[df["split"] == "train"], df[df["split"] == "test"]
X_train, y_train = train[["x0", "x1"]], train["y"]
X_test, y_test = test[["x0", "x1"]], test["y"]
fig, ax = plt.subplots()

dybder = range(1, 16)
tr_acc, te_acc = [], []
```
```losning
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("problem1.csv")
train, test = df[df["split"] == "train"], df[df["split"] == "test"]
X_train, y_train = train[["x0", "x1"]], train["y"]
X_test, y_test = test[["x0", "x1"]], test["y"]
fig, ax = plt.subplots()

dybder = range(1, 16)
tr_acc, te_acc = [], []
for d in dybder:
    clf = DecisionTreeClassifier(max_depth=d, class_weight="balanced", random_state=0).fit(X_train, y_train)
    tr_acc.append(clf.score(X_train, y_train))
    te_acc.append(clf.score(X_test, y_test))
ax.plot(dybder, tr_acc, "o-", label="train")
ax.plot(dybder, te_acc, "o-", label="test")
ax.set_xlabel("max_depth")
ax.legend()
plt.show()
```
```test
import numpy as np
from sklearn.tree import DecisionTreeClassifier
assert len(tr_acc) == 15 and len(te_acc) == 15, f"Det skal være 15 verdier i hver liste (dybde 1 til 15), fikk {len(tr_acc)} og {len(te_acc)}."
_c = DecisionTreeClassifier(max_depth=15, class_weight="balanced", random_state=0).fit(X_train, y_train)
assert abs(tr_acc[-1] - _c.score(X_train, y_train)) < 1e-9 and abs(te_acc[-1] - _c.score(X_test, y_test)) < 1e-9, 'Verdiene stemmer ikke. Bruk class_weight="balanced" og random_state=0, og legg train i tr_acc og test i te_acc.'
assert tr_acc[-1] > tr_acc[0] and te_acc[-1] < max(te_acc), "Train skal stige med dybden mens test faller til slutt. Sjekk løkka."
assert len(ax.lines) >= 2, "Plott både tr_acc og te_acc på ax."
```
:::

::: oppgave Fullt tre mot min_samples_leaf
Tren `dypt = DecisionTreeClassifier(random_state=0)` og `begrenset = DecisionTreeClassifier(min_samples_leaf=20, random_state=0)`, og lagre antall løvnoder (`n_dypt`, `n_begr`) og test-accuracy (`te_dypt`, `te_begr`).
```start
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("problem1.csv")
train, test = df[df["split"] == "train"], df[df["split"] == "test"]
X_train, y_train = train[["x0", "x1"]], train["y"]
X_test, y_test = test[["x0", "x1"]], test["y"]

n_dypt, n_begr = ..., ...
te_dypt, te_begr = ..., ...
```
```losning
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("problem1.csv")
train, test = df[df["split"] == "train"], df[df["split"] == "test"]
X_train, y_train = train[["x0", "x1"]], train["y"]
X_test, y_test = test[["x0", "x1"]], test["y"]

dypt = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
begrenset = DecisionTreeClassifier(min_samples_leaf=20, random_state=0).fit(X_train, y_train)
n_dypt, n_begr = dypt.get_n_leaves(), begrenset.get_n_leaves()
te_dypt, te_begr = dypt.score(X_test, y_test), begrenset.score(X_test, y_test)
```
```test
from sklearn.tree import DecisionTreeClassifier
_d = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
_b = DecisionTreeClassifier(min_samples_leaf=20, random_state=0).fit(X_train, y_train)
assert n_dypt == _d.get_n_leaves(), f"n_dypt skal være dypt.get_n_leaves() ({_d.get_n_leaves()}), fikk {n_dypt}."
assert n_begr == _b.get_n_leaves(), f"n_begr skal være begrenset.get_n_leaves() ({_b.get_n_leaves()}), fikk {n_begr}."
assert abs(te_dypt - _d.score(X_test, y_test)) < 1e-9 and abs(te_begr - _b.score(X_test, y_test)) < 1e-9, "te_dypt og te_begr skal være accuracy på testdataene."
```
:::

## Regresjonstrær
`DecisionTreeRegressor` splitter for lavest MSE og predikerer snittet av `y` i løvnoden, så prediksjonen blir en trappefunksjon som tegnes med `plt.step(grid, reg.predict(grid), where="post")`.

::: oppgave Løvnoden er et snitt
Tren en stump `reg = DecisionTreeRegressor(max_depth=1)`, lagre terskelen i roten i `t` og gjennomsnittet av `y_train` for punktene med `X_train <= t` i `venstre_snitt`.
```start
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

np.random.seed(0)
X_data = np.linspace(0, 4, 100)
y_data = np.floor(X_data) + np.random.normal(0, 0.1, size=100)
X_train, X_test, y_train, y_test = train_test_split(X_data.reshape(-1, 1), y_data, random_state=0)

reg = ...
t = ...
venstre_snitt = ...
```
```losning
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

np.random.seed(0)
X_data = np.linspace(0, 4, 100)
y_data = np.floor(X_data) + np.random.normal(0, 0.1, size=100)
X_train, X_test, y_train, y_test = train_test_split(X_data.reshape(-1, 1), y_data, random_state=0)

reg = DecisionTreeRegressor(max_depth=1).fit(X_train, y_train)
t = reg.tree_.threshold[0]
venstre_snitt = y_train[X_train.ravel() <= t].mean()
```
```test
from sklearn.tree import DecisionTreeRegressor
assert isinstance(reg, DecisionTreeRegressor) and hasattr(reg, "tree_") and reg.get_depth() == 1, "reg skal være et trent DecisionTreeRegressor med max_depth=1."
assert t is not ... and abs(t - reg.tree_.threshold[0]) < 1e-9, "t skal være reg.tree_.threshold[0]."
assert venstre_snitt is not ... and abs(venstre_snitt - reg.predict([[t - 1e-3]])[0]) < 1e-9, "venstre_snitt skal være y_train[X_train.ravel() <= t].mean(), og det er nøyaktig det treet predikerer til venstre."
```
:::

::: oppgave Tersklene i trappen
Tren `reg = DecisionTreeRegressor(max_depth=2)`, lagre de sorterte tersklene til de indre nodene (`feature >= 0`) i `terskler`, og tegn `reg.predict(grid)` med `ax.step(..., where="post")`.
```start
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

np.random.seed(0)
X_data = np.linspace(0, 4, 100)
y_data = np.floor(X_data) + np.random.normal(0, 0.1, size=100)
X_train, X_test, y_train, y_test = train_test_split(X_data.reshape(-1, 1), y_data, random_state=0)
grid = np.linspace(0, 4, 500).reshape(-1, 1)
fig, ax = plt.subplots()
ax.scatter(X_train.ravel(), y_train, s=15)

reg = ...
terskler = ...
```
```losning
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

np.random.seed(0)
X_data = np.linspace(0, 4, 100)
y_data = np.floor(X_data) + np.random.normal(0, 0.1, size=100)
X_train, X_test, y_train, y_test = train_test_split(X_data.reshape(-1, 1), y_data, random_state=0)
grid = np.linspace(0, 4, 500).reshape(-1, 1)
fig, ax = plt.subplots()
ax.scatter(X_train.ravel(), y_train, s=15)

reg = DecisionTreeRegressor(max_depth=2).fit(X_train, y_train)
t = reg.tree_
terskler = np.sort(t.threshold[t.feature >= 0])
ax.step(grid.ravel(), reg.predict(grid), where="post", color="green")
plt.show()
```
```test
import numpy as np
from sklearn.tree import DecisionTreeRegressor
assert isinstance(reg, DecisionTreeRegressor) and hasattr(reg, "tree_") and reg.get_depth() == 2, "reg skal være et trent DecisionTreeRegressor med max_depth=2."
assert terskler is not ... and len(terskler) == 3, "Et tre med dybde 2 har tre indre noder. Bruk t.threshold[t.feature >= 0]."
assert np.allclose(terskler, [1, 2, 3], atol=0.1), f"Tersklene skal være sortert og ligge nær 1, 2 og 3, fikk {terskler}."
assert len(ax.lines) >= 1, 'Tegn prediksjonen med ax.step(grid.ravel(), reg.predict(grid), where="post").'
```
:::

::: oppgave Finn beste dybde for regresjonstreet
For `d` i `range(1, 9)`: tren `DecisionTreeRegressor(max_depth=d, random_state=0)`, legg test-MSE i `mse_liste`, og lagre dybden (ikke indeksen) med lavest MSE i `beste_dybde`.
```start
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(0)
X_data = np.linspace(0, 4, 100)
y_data = np.floor(X_data) + np.random.normal(0, 0.1, size=100)
X_train, X_test, y_train, y_test = train_test_split(X_data.reshape(-1, 1), y_data, random_state=0)

mse_liste = []
beste_dybde = ...
```
```losning
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(0)
X_data = np.linspace(0, 4, 100)
y_data = np.floor(X_data) + np.random.normal(0, 0.1, size=100)
X_train, X_test, y_train, y_test = train_test_split(X_data.reshape(-1, 1), y_data, random_state=0)

mse_liste = []
for d in range(1, 9):
    reg = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X_train, y_train)
    mse_liste.append(mean_squared_error(y_test, reg.predict(X_test)))
beste_dybde = int(np.argmin(mse_liste)) + 1
```
```test
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
_fasit = [mean_squared_error(y_test, DecisionTreeRegressor(max_depth=d, random_state=0).fit(X_train, y_train).predict(X_test)) for d in range(1, 9)]
assert len(mse_liste) == 8, f"mse_liste skal ha 8 verdier (dybde 1 til 8), har {len(mse_liste)}."
assert np.allclose(mse_liste, _fasit), "MSE-ene stemmer ikke. Regn test-MSE med mean_squared_error(y_test, reg.predict(X_test))."
assert beste_dybde == int(np.argmin(_fasit)) + 1, "beste_dybde skal være dybden (ikke indeksen) med lavest MSE. Husk at range starter på 1."
```
:::

--- LEKSJON 8.5 | bagging-random-forest | Bagging og random forest
Et bootstrap-utvalg `idx = rng.choice(n, size=n, replace=True)` trekker rader med tilbakelegging, så ca. 63 % av radene kommer med og resten er out-of-bag.

::: oppgave Bootstrap-utvalg
Trekk et bootstrap-utvalg `idx` av `n = 1000` rader, lagre andelen unike rader i `andel_unike`, og bruk `np.bincount(idx, minlength=n)` til å finne andelen rader som aldri ble trukket (`andel_oob`).
```start
import numpy as np

rng = np.random.default_rng(0)
n = 1000

idx = ...
andel_unike = ...
andel_oob = ...
```
```losning
import numpy as np

rng = np.random.default_rng(0)
n = 1000

idx = rng.choice(n, size=n, replace=True)
andel_unike = len(np.unique(idx)) / n
antall = np.bincount(idx, minlength=n)
andel_oob = np.mean(antall == 0)
```
```test
import numpy as np
assert isinstance(idx, np.ndarray) and len(idx) == 1000, "idx skal være 1000 indekser fra rng.choice(n, size=n, replace=True)."
assert idx.min() >= 0 and idx.max() < 1000, "Indeksene skal ligge mellom 0 og 999."
assert len(np.unique(idx)) < 1000, "Alle radene er unike, så du trakk uten tilbakelegging. Bruk replace=True."
assert abs(andel_unike - len(np.unique(idx)) / 1000) < 1e-9, "andel_unike skal være len(np.unique(idx)) / n."
assert 0.58 < andel_unike < 0.68, f"andel_unike skal være ca. 0.63, fikk {andel_unike}."
assert abs(andel_oob - (1 - andel_unike)) < 1e-9, "andel_oob skal være andelen rader som er trukket 0 ganger (np.bincount(idx, minlength=n) == 0)."
```
:::

## Bagging for hånd
Tren hvert tre på sitt eget bootstrap-utvalg og ta flertallsstemme med `(stemmer.mean(axis=0) > 0.5).astype(int)`; for regresjon tar man snittet.

::: oppgave 25 trær som stemmer
Tren 25 `DecisionTreeClassifier`-er på hvert sitt bootstrap-utvalg av treningsdataene, legg testprediksjonene i `stemmer` (shape `(25, len(X_test))`), og lagre flertallsstemmen i `y_pred` og accuracy i `acc_bag`.
```start
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_moons(n_samples=400, noise=0.3, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
acc_ett_tre = DecisionTreeClassifier(random_state=0).fit(X_train, y_train).score(X_test, y_test)
rng = np.random.default_rng(0)

stemmer = []
y_pred = ...
acc_bag = ...
```
```losning
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_moons(n_samples=400, noise=0.3, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
acc_ett_tre = DecisionTreeClassifier(random_state=0).fit(X_train, y_train).score(X_test, y_test)
rng = np.random.default_rng(0)

stemmer = []
for b in range(25):
    idx = rng.choice(len(X_train), size=len(X_train), replace=True)
    tre = DecisionTreeClassifier(random_state=b).fit(X_train[idx], y_train[idx])
    stemmer.append(tre.predict(X_test))
stemmer = np.array(stemmer)
y_pred = (stemmer.mean(axis=0) > 0.5).astype(int)
acc_bag = np.mean(y_pred == y_test)
```
```test
import numpy as np
_s = np.asarray(stemmer)
assert _s.shape == (25, len(X_test)), f"stemmer skal ha shape (25, {len(X_test)}), har {_s.shape}."
assert not all(np.array_equal(_s[0], r) for r in _s), "Alle trærne gir like prediksjoner. Tren hvert tre på X_train[idx], y_train[idx] med et nytt bootstrap-utvalg idx."
assert y_pred is not ... and np.array_equal(np.asarray(y_pred), (_s.mean(axis=0) > 0.5).astype(int)), "y_pred skal være flertallsstemmen: (stemmer.mean(axis=0) > 0.5).astype(int)."
assert acc_bag is not ... and abs(acc_bag - np.mean(np.asarray(y_pred) == y_test)) < 1e-9, "acc_bag skal være andelen riktige på testdataene."
```
:::

## BaggingClassifier og RandomForestClassifier
`BaggingClassifier(DecisionTreeClassifier(), n_estimators=20)` gjør det samme, og `RandomForestClassifier(max_features="sqrt", oob_score=True)` lar i tillegg hver split velge blant et tilfeldig utvalg features, så trærne blir mer ulike, og gir et gratis valideringsestimat i `oob_score_`.

::: oppgave BaggingClassifier
Tren `bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=20, random_state=0)` og ett `DecisionTreeClassifier(random_state=0)`, og lagre test-accuracy i `acc_bag` og `acc_tre`.
```start
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

bag = ...
acc_bag = ...
acc_tre = ...
```
```losning
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=20, random_state=0).fit(X_train, y_train)
acc_bag = bag.score(X_test, y_test)
acc_tre = DecisionTreeClassifier(random_state=0).fit(X_train, y_train).score(X_test, y_test)
```
```test
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
assert isinstance(bag, BaggingClassifier) and hasattr(bag, "estimators_"), "bag skal være en trent BaggingClassifier."
assert len(bag.estimators_) == 20, "Bruk n_estimators=20."
assert abs(acc_bag - bag.score(X_test, y_test)) < 1e-9, "acc_bag skal være test-accuracy for bag."
assert abs(acc_tre - DecisionTreeClassifier(random_state=0).fit(X_train, y_train).score(X_test, y_test)) < 1e-9, "acc_tre skal være test-accuracy for ett DecisionTreeClassifier(random_state=0)."
```
:::

::: oppgave Random forest med OOB
Tren `rf = RandomForestClassifier(n_estimators=50, max_features=5, oob_score=True, random_state=0)`, og lagre OOB-scoren i `oob`, test-accuracy i `test_acc` og antall trær (`len(rf.estimators_)`) i `antall_traer`.
```start
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

rf = ...
oob = ...
test_acc = ...
antall_traer = ...
```
```losning
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

rf = RandomForestClassifier(n_estimators=50, max_features=5, oob_score=True, random_state=0)
rf.fit(X_train, y_train)
oob = rf.oob_score_
test_acc = rf.score(X_test, y_test)
antall_traer = len(rf.estimators_)
```
```test
from sklearn.ensemble import RandomForestClassifier
assert isinstance(rf, RandomForestClassifier), "rf skal være en RandomForestClassifier."
assert rf.n_estimators == 50 and rf.max_features == 5, "Bruk n_estimators=50 og max_features=5."
assert hasattr(rf, "oob_score_"), "Husk oob_score=True og å kalle fit."
assert abs(oob - rf.oob_score_) < 1e-9, "oob skal være rf.oob_score_."
assert abs(test_acc - rf.score(X_test, y_test)) < 1e-9, "test_acc skal være accuracy på testdataene."
assert antall_traer == 50, "antall_traer skal være len(rf.estimators_)."
```
:::

--- LEKSJON 8.6 | boosting | Boosting: AdaBoost og gradient boosting
Boosting trener svake modeller (ofte stumper, `max_depth=1`) sekvensielt slik at hver retter feilene til de forrige, mens bagging trener dem parallelt og uavhengig.

## AdaBoostClassifier
`AdaBoostClassifier(n_estimators=50, learning_rate=1.0)` bruker stumper som standard, gir feilklassifiserte punkter høyere vekt i neste runde, og lagrer modellvektene $\alpha$ i `estimator_weights_`.

::: oppgave Én stump mot AdaBoost
Lagre test-accuracy for én `DecisionTreeClassifier(max_depth=1)` i `stump_acc`, tren `ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=0)`, og lagre `ada_acc` og de fem første $\alpha$-ene i `alfa`.
```start
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

stump_acc = ...
ada = ...
ada_acc = ...
alfa = ...
```
```losning
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

stump_acc = DecisionTreeClassifier(max_depth=1).fit(X_train, y_train).score(X_test, y_test)
ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=0).fit(X_train, y_train)
ada_acc = ada.score(X_test, y_test)
alfa = ada.estimator_weights_[:5]
```
```test
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
assert abs(stump_acc - DecisionTreeClassifier(max_depth=1).fit(X_train, y_train).score(X_test, y_test)) < 1e-9, "stump_acc skal være test-accuracy for ett tre med max_depth=1."
assert isinstance(ada, AdaBoostClassifier) and hasattr(ada, "estimators_"), "ada skal være en trent AdaBoostClassifier."
assert ada.n_estimators == 50, "Bruk n_estimators=50."
assert abs(ada_acc - ada.score(X_test, y_test)) < 1e-9, "ada_acc skal være test-accuracy for ada."
assert ada_acc > stump_acc, "AdaBoost skal slå én stump. Sjekk at du har trent ada."
assert alfa is not ... and np.allclose(alfa, ada.estimator_weights_[:5]), "alfa skal være ada.estimator_weights_[:5]."
```
:::

::: oppgave Dypere trær og kryssvalidering
Lag `ada2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=20, random_state=0)`, kjør `cross_val_score` med `cv=5` på treningsdataene, og lagre scorene i `cv` og teksten `f"{cv.mean():.3f} ± {cv.std():.3f}"` i `rapport`.
```start
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

ada2 = ...
cv = ...
rapport = ...
```
```losning
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

ada2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=20, random_state=0)
cv = cross_val_score(ada2, X_train, y_train, cv=5)
rapport = f"{cv.mean():.3f} ± {cv.std():.3f}"
```
```test
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
assert isinstance(ada2, AdaBoostClassifier), "ada2 skal være en AdaBoostClassifier."
assert ada2.estimator is not None and ada2.estimator.max_depth == 2, "Gi DecisionTreeClassifier(max_depth=2) som første argument."
assert ada2.n_estimators == 20, "Bruk n_estimators=20."
assert cv is not ... and len(cv) == 5, "cv skal være de fem scorene fra cross_val_score(..., cv=5)."
assert rapport == f"{np.mean(cv):.3f} ± {np.std(cv):.3f}", 'rapport skal være f"{cv.mean():.3f} ± {cv.std():.3f}".'
```
:::

## Gradient boosting for hånd
Start med `F = np.full(len(y), y.mean())`, tren hvert tre på pseudo-residualene `r = y - F`, og oppdater `F = F + lr * tre.predict(X)`.

::: oppgave Fullfør gradient boosting-løkka
Fullfør løkka slik at den kjører 30 runder med læringsrate 0.3 og trær med `max_depth=1`, og legg MSE-en etter hver runde i `mse_liste`.
```start
import numpy as np
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(1)
X = rng.uniform(-3, 3, size=(80, 1))
y = X.ravel()**2 + rng.normal(0, 0.3, 80)

lr = 0.3
F = np.full(len(y), y.mean())
mse_liste = []
for m in range(30):
    r = ...                 # pseudo-residualer
    tre = ...               # tren et tre med max_depth=1 på r
    F = ...                 # oppdater ensemblet
```
```losning
import numpy as np
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(1)
X = rng.uniform(-3, 3, size=(80, 1))
y = X.ravel()**2 + rng.normal(0, 0.3, 80)

lr = 0.3
F = np.full(len(y), y.mean())
mse_liste = []
for m in range(30):
    r = y - F
    tre = DecisionTreeRegressor(max_depth=1).fit(X, r)
    F = F + lr * tre.predict(X)
    mse_liste.append(np.mean((y - F)**2))
```
```test
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
assert isinstance(F, np.ndarray) and F.shape == y.shape, "F skal være et array med én prediksjon per punkt."
assert np.mean((y - F)**2) < np.var(y) * 0.5, "MSE har ikke falt nok. Er r = y - F, og legger du til lr * tre.predict(X)?"
_fasit = GradientBoostingRegressor(n_estimators=30, learning_rate=0.3, max_depth=1).fit(X, y).predict(X)
assert np.allclose(F, _fasit, atol=1e-6), "F avviker fra sklearn. Sjekk at treet trenes på r (ikke y) og har max_depth=1."
assert len(mse_liste) == 30 and abs(mse_liste[-1] - np.mean((y - F)**2)) < 1e-9, "Legg np.mean((y - F)**2) i mse_liste etter hver oppdatering (30 verdier)."
```
:::

## GradientBoostingClassifier og -Regressor
`GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)` og `GradientBoostingClassifier` gjør det samme, og lav `learning_rate` trenger flere trær.

::: oppgave GradientBoostingClassifier
Tren `gbc = GradientBoostingClassifier(n_estimators=50, random_state=0)` på treningsdataene og lagre test-accuracy i `gbc_acc`.
```start
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

gbc = ...
gbc_acc = ...
```
```losning
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

gbc = GradientBoostingClassifier(n_estimators=50, random_state=0).fit(X_train, y_train)
gbc_acc = gbc.score(X_test, y_test)
```
```test
from sklearn.ensemble import GradientBoostingClassifier
assert isinstance(gbc, GradientBoostingClassifier) and hasattr(gbc, "estimators_"), "gbc skal være en trent GradientBoostingClassifier."
assert gbc.n_estimators == 50, "Bruk n_estimators=50."
assert abs(gbc_acc - gbc.score(X_test, y_test)) < 1e-9, "gbc_acc skal være accuracy på testdataene."
```
:::

::: oppgave Læringsrate og antall trær
Tren to `GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=0)`, én med `learning_rate=0.01` og én med `0.1`, og lagre test-$R^2$ (`r2_score`) i `r2_lav` og `r2_hoy`.
```start
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

r2_lav = ...
r2_hoy = ...
```
```losning
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lav = GradientBoostingRegressor(n_estimators=50, learning_rate=0.01, max_depth=2, random_state=0).fit(X_train, y_train)
hoy = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=2, random_state=0).fit(X_train, y_train)
r2_lav = r2_score(y_test, lav.predict(X_test))
r2_hoy = r2_score(y_test, hoy.predict(X_test))
```
```test
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
def _r2(lr):
    m = GradientBoostingRegressor(n_estimators=50, learning_rate=lr, max_depth=2, random_state=0).fit(X_train, y_train)
    return r2_score(y_test, m.predict(X_test))
assert r2_lav is not ... and abs(r2_lav - _r2(0.01)) < 1e-6, "r2_lav skal være test-R² med learning_rate=0.01 (n_estimators=50, max_depth=2, random_state=0)."
assert r2_hoy is not ... and abs(r2_hoy - _r2(0.1)) < 1e-6, "r2_hoy skal være test-R² med learning_rate=0.1."
```
:::

--- LEKSJON 8.7 | naive-bayes | Naive Bayes for hånd
Naive Bayes antar at featurene er uavhengige gitt klassen, så $P(C \mid x) \propto P(C)\prod_i P(x_i \mid C)$, og evidensen $P(x)$ droppes fordi den er lik for alle klasser.

## Prior og likelihood
`df["survived"].value_counts(normalize=True)` gir prioren, og `pd.crosstab(df["role"], df["survived"], normalize="columns")` gir $P(\text{role} \mid \text{survived})$ med én kolonne per klasse.

::: oppgave Prior og likelihood-tabell
Lagre prioren i `prior`, likelihood-tabellen for `role` i `tab`, og $P(\text{role} = \text{Hirdman} \mid \text{survived} = 1)$ i `p_hird_1`.
```start
import pandas as pd

df = pd.read_csv("problem2.csv")

prior = ...
tab = ...
p_hird_1 = ...
```
```losning
import pandas as pd

df = pd.read_csv("problem2.csv")

prior = df["survived"].value_counts(normalize=True)
tab = pd.crosstab(df["role"], df["survived"], normalize="columns")
p_hird_1 = tab.loc["Hirdman", 1]
```
```test
import numpy as np
import pandas as pd
assert isinstance(prior, pd.Series) and abs(prior.loc[1] - df["survived"].mean()) < 1e-9, 'prior skal være df["survived"].value_counts(normalize=True).'
assert isinstance(tab, pd.DataFrame) and np.allclose(tab.sum(axis=0), 1), 'Hver kolonne i tab skal summere til 1. Bruk normalize="columns".'
assert "Hirdman" in tab.index, 'tab skal ha rollene som rader: pd.crosstab(df["role"], df["survived"], ...).'
_r = df[df["survived"] == 1]
assert p_hird_1 is not ... and abs(p_hird_1 - (_r["role"] == "Hirdman").mean()) < 1e-9, 'p_hird_1 skal være tab.loc["Hirdman", 1].'
```
:::

## Score for en ny viking
Scoren for klasse $C$ er `len(rader) / len(df)` ganget med `(rader[feature] == verdi).mean()` for hver feature, der `rader = df[df["survived"] == C]`, og med mange features summerer man heller logaritmene for å unngå underflow.

::: oppgave Naive Bayes-score for en kvinnelig karl
Regn ut den ikke-normaliserte scoren for begge klasser for en viking med `sex = "Female"`, `role = "Karl"` og `ship = "Sjøulven"`, lagre dem i `score0` og `score1`, og lagre `score1 > score0` i `overlever`.
```start
import pandas as pd

df = pd.read_csv("problem2.csv")
ny = {"sex": "Female", "role": "Karl", "ship": "Sjøulven"}

score0 = ...
score1 = ...
overlever = ...
```
```losning
import pandas as pd

df = pd.read_csv("problem2.csv")
ny = {"sex": "Female", "role": "Karl", "ship": "Sjøulven"}

def nb_score(klasse):
    rader = df[df["survived"] == klasse]
    s = len(rader) / len(df)
    for f, v in ny.items():
        s *= (rader[f] == v).mean()
    return s

score0 = nb_score(0)
score1 = nb_score(1)
overlever = score1 > score0
```
```test
def _fasit(k):
    r = df[df["survived"] == k]
    s = len(r) / len(df)
    for f, v in {"sex": "Female", "role": "Karl", "ship": "Sjøulven"}.items():
        s *= (r[f] == v).mean()
    return s
assert score0 is not ... and score1 is not ..., "Regn ut score0 og score1."
assert abs(score0 - _fasit(0)) < 1e-9, "score0 stemmer ikke. Husk prior len(rader)/len(df) ganget med andelen per feature blant radene med survived=0."
assert abs(score1 - _fasit(1)) < 1e-9, "score1 stemmer ikke. Bruk radene med survived=1."
assert bool(overlever) == (_fasit(1) > _fasit(0)), "overlever skal være score1 > score0."
```
:::

::: oppgave Log-score for Endre
Regn ut $\log P(C) + \sum_i \log P(x_i \mid C)$ med `np.log` for Endre for begge klasser (`log0`, `log1`), og lagre den normaliserte sannsynligheten `np.exp(log1) / (np.exp(log0) + np.exp(log1))` i `p_overlever`.
```start
import numpy as np
import pandas as pd

df = pd.read_csv("problem2.csv")
df["armor"] = df["armor"].fillna("None")
endre = {"sex": "Male", "role": "Hirdman", "ship": "Drakvinge",
         "destination": "Normandy", "clan_origin": "Nidaros", "armor": "Chainmail"}

log0 = ...
log1 = ...
p_overlever = ...
```
```losning
import numpy as np
import pandas as pd

df = pd.read_csv("problem2.csv")
df["armor"] = df["armor"].fillna("None")
endre = {"sex": "Male", "role": "Hirdman", "ship": "Drakvinge",
         "destination": "Normandy", "clan_origin": "Nidaros", "armor": "Chainmail"}

def log_score(klasse):
    rader = df[df["survived"] == klasse]
    s = np.log(len(rader) / len(df))
    for f, v in endre.items():
        s += np.log((rader[f] == v).mean())
    return s

log0 = log_score(0)
log1 = log_score(1)
p_overlever = np.exp(log1) / (np.exp(log0) + np.exp(log1))
```
```test
import numpy as np
def _ls(k):
    r = df[df["survived"] == k]
    s = np.log(len(r) / len(df))
    for f, v in endre.items():
        s += np.log((r[f] == v).mean())
    return s
assert log0 is not ... and log1 is not ..., "Regn ut log0 og log1."
assert log0 < 0 and log1 < 0, "Log-scorene skal være negative (logaritmen av sannsynligheter under 1). Brukte du np.log?"
assert abs(log0 - _ls(0)) < 1e-9 and abs(log1 - _ls(1)) < 1e-9, "Log-scorene stemmer ikke. Summer np.log av prioren og av hver likelihood."
assert p_overlever is not ... and abs(p_overlever - np.exp(_ls(1)) / (np.exp(_ls(0)) + np.exp(_ls(1)))) < 1e-9, "p_overlever skal være np.exp(log1) / (np.exp(log0) + np.exp(log1))."
```
:::

## Laplace-glatting
Med $P(x_i = v \mid C) = \frac{\text{antall}(v, C) + \alpha}{n_C + \alpha K}$ gjør ikke en verdi som aldri er sett (Endre har `prior_voyages = 12`, men maks i dataene er 9) hele produktet til 0.

::: oppgave Glattet likelihood
Skriv `likelihood(df, feature, verdi, klasse, alpha=1)` med Laplace-glatting, der `K = df[feature].nunique() + 1` (+1 fordi verdien kan være ny).
```start
import pandas as pd

df = pd.read_csv("problem2.csv")

def likelihood(df, feature, verdi, klasse, alpha=1):
    ...
```
```losning
import pandas as pd

df = pd.read_csv("problem2.csv")

def likelihood(df, feature, verdi, klasse, alpha=1):
    rader = df[df["survived"] == klasse]
    K = df[feature].nunique() + 1
    return ((rader[feature] == verdi).sum() + alpha) / (len(rader) + alpha * K)
```
```test
_r1 = df[df["survived"] == 1]
_K = df["prior_voyages"].nunique() + 1
_v = likelihood(df, "prior_voyages", 12, 1)
assert _v is not None, "likelihood må returnere et tall (husk return)."
assert _v > 0, "En ukjent verdi skal få en liten positiv sannsynlighet med alpha=1, ikke 0."
assert abs(_v - 1 / (len(_r1) + _K)) < 1e-9, "Feil verdi for prior_voyages=12. Formelen er (antall + alpha) / (n_C + alpha * K)."
assert abs(likelihood(df, "sex", "Male", 0, alpha=0) - (df[df["survived"] == 0]["sex"] == "Male").mean()) < 1e-9, "Med alpha=0 skal du få den vanlige andelen."
assert abs(likelihood(df, "sex", "Male", 1, alpha=2) - (((_r1["sex"] == "Male").sum() + 2) / (len(_r1) + 2 * 3))) < 1e-9, "Bruk alpha både i telleren og i nevneren (alpha * K)."
```
:::

## Med sklearn
`CategoricalNB(alpha=1)` gjør det samme med glatting og krever kategorier som heltall fra `OrdinalEncoder`, mens `GaussianNB` antar normalfordelte kontinuerlige features.

::: oppgave CategoricalNB for Endre
Kod `df[kat]` med `enc = OrdinalEncoder()`, tren `nb = CategoricalNB(alpha=1)` på `survived`, og lagre `nb.predict_proba(enc.transform(endre))` i `proba`.
```start
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB

df = pd.read_csv("problem2.csv")
df["armor"] = df["armor"].fillna("None")
df["clan_origin"] = df["clan_origin"].fillna("Unknown")
kat = ["sex", "role", "ship", "destination", "clan_origin", "armor"]
endre = pd.DataFrame([["Male", "Hirdman", "Drakvinge", "Normandy", "Nidaros", "Chainmail"]], columns=kat)

enc = ...
nb = ...
proba = ...
```
```losning
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB

df = pd.read_csv("problem2.csv")
df["armor"] = df["armor"].fillna("None")
df["clan_origin"] = df["clan_origin"].fillna("Unknown")
kat = ["sex", "role", "ship", "destination", "clan_origin", "armor"]
endre = pd.DataFrame([["Male", "Hirdman", "Drakvinge", "Normandy", "Nidaros", "Chainmail"]], columns=kat)

enc = OrdinalEncoder()
X = enc.fit_transform(df[kat])
nb = CategoricalNB(alpha=1).fit(X, df["survived"])
proba = nb.predict_proba(enc.transform(endre))
```
```test
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
assert isinstance(nb, CategoricalNB) and hasattr(nb, "classes_"), "nb skal være en trent CategoricalNB."
_e = OrdinalEncoder()
_fasit = CategoricalNB(alpha=1).fit(_e.fit_transform(df[kat]), df["survived"]).predict_proba(_e.transform(endre))
assert proba is not ... and np.shape(proba) == (1, 2), "proba skal ha shape (1, 2): én rad (Endre) og én kolonne per klasse."
assert np.allclose(proba, _fasit), "proba stemmer ikke. Tren på enc.fit_transform(df[kat]) og bruk enc.transform(endre) (ikke fit_transform) for Endre."
```
:::

::: oppgave GaussianNB på tallkolonnene
Tren `gnb = GaussianNB()` på `X_num` (der `age` er fylt med medianen) og lagre `gnb.predict_proba(endre_num)` i `proba_g`.
```start
import pandas as pd
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("problem2.csv")
num = ["age", "provisions_kg", "family_aboard", "prior_voyages"]
endre_num = pd.DataFrame([[25, 49, 2, 12]], columns=num)

X_num = ...
gnb = ...
proba_g = ...
```
```losning
import pandas as pd
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("problem2.csv")
num = ["age", "provisions_kg", "family_aboard", "prior_voyages"]
endre_num = pd.DataFrame([[25, 49, 2, 12]], columns=num)

X_num = df[num].fillna({"age": df["age"].median()})
gnb = GaussianNB().fit(X_num, df["survived"])
proba_g = gnb.predict_proba(endre_num)
```
```test
import numpy as np
from sklearn.naive_bayes import GaussianNB
assert X_num is not ... and X_num.isna().sum().sum() == 0, 'X_num skal være df[num] uten manglende verdier (fyll age med df["age"].median()).'
assert isinstance(gnb, GaussianNB) and hasattr(gnb, "theta_"), "gnb skal være en trent GaussianNB."
_fasit = GaussianNB().fit(df[num].fillna({"age": df["age"].median()}), df["survived"]).predict_proba(endre_num)
assert proba_g is not ... and np.allclose(proba_g, _fasit), "proba_g stemmer ikke. Fyll age med medianen og bruk gnb.predict_proba(endre_num)."
```
:::

--- LEKSJON 8.8 | prosjekt-2 | Hele prosjekt 2-løpet
Prosjektet på `problem2.csv` er analyse, preprosessering, tre med kryssvalidering og boosting, med mål om minst 0.80 test-accuracy.

## 1. Analyse
`df.groupby("role")["survived"].mean()` gir overlevelsesraten per gruppe, og en baseline er å alltid gjette flertallsklassen.

::: oppgave Analyse av vikingdataene
Lagre antall manglende verdier for kolonnene som mangler noe i `mangler`, overlevelsesraten for kvinner og menn i `rate_female` og `rate_male`, og accuracy ved å alltid gjette flertallsklassen i `baseline`.
```start
import pandas as pd

df = pd.read_csv("problem2.csv")

mangler = ...
rate_female = ...
rate_male = ...
baseline = ...
```
```losning
import pandas as pd

df = pd.read_csv("problem2.csv")

mangler = df.isna().sum()[df.isna().sum() > 0]
rate_female = df[df["sex"] == "Female"]["survived"].mean()
rate_male = df[df["sex"] == "Male"]["survived"].mean()
baseline = max(df["survived"].mean(), 1 - df["survived"].mean())
print(df.groupby("role")["survived"].agg(["mean", "count"]))
```
```test
import pandas as pd
assert isinstance(mangler, pd.Series) and set(mangler.index) == {"age", "clan_origin", "armor"}, "mangler skal være df.isna().sum() filtrert til kolonnene med flere enn 0 manglende."
assert int(mangler["armor"]) == 324, "mangler skal inneholde antallet manglende per kolonne."
assert rate_female is not ... and abs(rate_female - df.groupby("sex")["survived"].mean()["Female"]) < 1e-9, "rate_female skal være snittet av survived for kvinner."
assert rate_male is not ... and abs(rate_male - df.groupby("sex")["survived"].mean()["Male"]) < 1e-9, "rate_male skal være snittet av survived for menn."
assert baseline is not ... and abs(baseline - 0.556) < 1e-9, "baseline skal være andelen i flertallsklassen (1 - 0.444 = 0.556)."
```
:::

## 2. Preprosessering
Fjern `passenger_id` og `name` (unike per rad), gjør manglende `armor` og `clan_origin` til egne kategorier, bruk `pd.get_dummies(..., dtype=int)`, og fyll `age` med medianen fra **treningsdataene** etter splitten for å unngå lekkasje.

::: oppgave Preprosesseringsfunksjon
Skriv `forbered(df)` som returnerer `(X, y)`: `y` er `survived`, og `X` er uten `passenger_id`, `name` og `survived`, med `"None"` for manglende `armor`, `"Unknown"` for manglende `clan_origin`, og dummy-kolonner (`age` fylles ikke her). Den originale `df` skal ikke endres.
```start
import pandas as pd

def forbered(df):
    ...
    return X, y

df = pd.read_csv("problem2.csv")
X, y = forbered(df)
```
```losning
import pandas as pd

def forbered(df):
    d = df.drop(columns=["passenger_id", "name"])
    d["armor"] = d["armor"].fillna("None")
    d["clan_origin"] = d["clan_origin"].fillna("Unknown")
    y = d["survived"]
    X = pd.get_dummies(d.drop(columns="survived"), dtype=int)
    return X, y

df = pd.read_csv("problem2.csv")
X, y = forbered(df)
```
```test
import pandas as pd
assert isinstance(X, pd.DataFrame), "X skal være en DataFrame (fra pd.get_dummies)."
assert len(X) == 500 and len(y) == 500, "Ingen rader skal fjernes."
for _kol in ["passenger_id", "name", "survived"]:
    assert _kol not in X.columns, f"Kolonnen {_kol} skal ikke være med i X."
assert all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes), "Alle kolonner i X skal være tall. Bruk pd.get_dummies(..., dtype=int)."
assert "armor_None" in X.columns and "clan_origin_Unknown" in X.columns, 'Manglende armor og clan_origin skal bli kategoriene "None" og "Unknown".'
assert set(X.columns[X.isna().any()]) <= {"age"}, "Bare age skal ha manglende verdier igjen (den fylles etter splitten)."
assert df["armor"].isna().sum() == 324, "forbered skal ikke endre den originale df. Jobb på en kopi (drop lager en ny DataFrame)."
assert (y == df["survived"]).all(), "y skal være kolonnen survived."
```
:::

::: oppgave Split og median uten lekkasje
Del med `train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)`, og fyll `age` i både `X_train` og `X_test` med medianen fra `X_train` (lagre den i `med`).
```start
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("problem2.csv").drop(columns=["passenger_id", "name"])
df["armor"] = df["armor"].fillna("None")
df["clan_origin"] = df["clan_origin"].fillna("Unknown")
X = pd.get_dummies(df.drop(columns="survived"), dtype=int)
y = df["survived"]

X_train, X_test, y_train, y_test = ..., ..., ..., ...
med = ...
```
```losning
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("problem2.csv").drop(columns=["passenger_id", "name"])
df["armor"] = df["armor"].fillna("None")
df["clan_origin"] = df["clan_origin"].fillna("Unknown")
X = pd.get_dummies(df.drop(columns="survived"), dtype=int)
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
med = X_train["age"].median()
X_train = X_train.fillna({"age": med})
X_test = X_test.fillna({"age": med})
```
```test
import pandas as pd
assert isinstance(X_train, pd.DataFrame) and len(X_train) == 400 and len(X_test) == 100, "Bruk test_size=0.2, så får du 400 trenings- og 100 testrader."
assert abs(y_train.mean() - y_test.mean()) < 0.01, "Husk stratify=y, så klassefordelingen blir lik i train og test."
_med = X.loc[X_train.index, "age"].median()
assert med is not ... and abs(med - _med) < 1e-9, "med skal være medianen av age i X_train (ikke i hele X)."
assert X_train["age"].isna().sum() == 0 and X_test["age"].isna().sum() == 0, "Fyll age i både X_train og X_test."
_hull = X.loc[X_test.index, "age"].isna()
assert (X_test.loc[_hull, "age"] == _med).all(), "Manglende age i X_test skal fylles med medianen fra X_train."
```
:::

## 3. Tre med kryssvalidering
`cross_val_score(clf, X_train, y_train, cv=5).mean()` velger hyperparametre uten å se på testdataene, som brukes bare én gang til slutt (velg aldri `random_state` eller hyperparametre etter test-accuracy).

::: oppgave Velg kriterium og dybde med CV
Prøv alle kombinasjoner av `criterion` i `["gini", "entropy", "log_loss"]` og `max_depth` i `[2, 3, 4, 6]` (`random_state=0`), lagre den med høyest CV-snitt som tuppelen `beste = (criterion, depth)`, tren den på hele `X_train`, og lagre test-accuracy i `test_acc`.
```start
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("problem2.csv").drop(columns=["passenger_id", "name"])
df["armor"] = df["armor"].fillna("None")
df["clan_origin"] = df["clan_origin"].fillna("Unknown")
X = pd.get_dummies(df.drop(columns="survived"), dtype=int)
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
med = X_train["age"].median()
X_train, X_test = X_train.fillna({"age": med}), X_test.fillna({"age": med})

beste = ...
test_acc = ...
```
```losning
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("problem2.csv").drop(columns=["passenger_id", "name"])
df["armor"] = df["armor"].fillna("None")
df["clan_origin"] = df["clan_origin"].fillna("Unknown")
X = pd.get_dummies(df.drop(columns="survived"), dtype=int)
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
med = X_train["age"].median()
X_train, X_test = X_train.fillna({"age": med}), X_test.fillna({"age": med})

beste_cv = 0
for criterion in ["gini", "entropy", "log_loss"]:
    for depth in [2, 3, 4, 6]:
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=0)
        cv = cross_val_score(clf, X_train, y_train, cv=5).mean()
        if cv > beste_cv:
            beste_cv, beste = cv, (criterion, depth)

clf = DecisionTreeClassifier(criterion=beste[0], max_depth=beste[1], random_state=0).fit(X_train, y_train)
test_acc = clf.score(X_test, y_test)
```
```test
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
def _cv(c, d):
    return cross_val_score(DecisionTreeClassifier(criterion=c, max_depth=d, random_state=0), X_train, y_train, cv=5).mean()
assert isinstance(beste, tuple) and len(beste) == 2, "beste skal være en tuppel (criterion, depth)."
_alle = {(c, d): _cv(c, d) for c in ["gini", "entropy", "log_loss"] for d in [2, 3, 4, 6]}
assert beste in _alle, f"{beste} er ikke en av kombinasjonene som skulle prøves."
assert _alle[beste] >= max(_alle.values()) - 1e-12, f"{beste} har ikke høyest CV-snitt. Sammenlign cross_val_score(...).mean() på X_train."
_clf = DecisionTreeClassifier(criterion=beste[0], max_depth=beste[1], random_state=0).fit(X_train, y_train)
assert test_acc is not ... and abs(test_acc - _clf.score(X_test, y_test)) < 1e-9, "test_acc skal være test-accuracy for den valgte modellen trent på hele X_train."
```
:::

## 4. Boosting
`AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=0)` på de samme dataene rapporteres med CV-snitt ± standardavvik og test-accuracy.

::: oppgave AdaBoost på vikingdataene
Kjør 5-fold `cross_val_score` for `ada` på treningsdataene (`ada_cv`), tren `ada` på hele `X_train`, og lagre test-accuracy i `ada_acc` (målet er minst 0.80).
```start
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostClassifier

df = pd.read_csv("problem2.csv").drop(columns=["passenger_id", "name"])
df["armor"] = df["armor"].fillna("None")
df["clan_origin"] = df["clan_origin"].fillna("Unknown")
X = pd.get_dummies(df.drop(columns="survived"), dtype=int)
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
med = X_train["age"].median()
X_train, X_test = X_train.fillna({"age": med}), X_test.fillna({"age": med})

ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=0)
ada_cv = ...
ada_acc = ...
```
```losning
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostClassifier

df = pd.read_csv("problem2.csv").drop(columns=["passenger_id", "name"])
df["armor"] = df["armor"].fillna("None")
df["clan_origin"] = df["clan_origin"].fillna("Unknown")
X = pd.get_dummies(df.drop(columns="survived"), dtype=int)
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
med = X_train["age"].median()
X_train, X_test = X_train.fillna({"age": med}), X_test.fillna({"age": med})

ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=0)
ada_cv = cross_val_score(ada, X_train, y_train, cv=5)
ada.fit(X_train, y_train)
ada_acc = ada.score(X_test, y_test)
print(f"AdaBoost CV Accuracy: {ada_cv.mean():.3f} ± {ada_cv.std():.3f}")
print(f"AdaBoost test accuracy: {ada_acc:.4f}")
```
```test
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
assert ada_cv is not ... and len(ada_cv) == 5, "ada_cv skal være de fem scorene fra cross_val_score(ada, X_train, y_train, cv=5)."
assert np.allclose(ada_cv, cross_val_score(AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=0), X_train, y_train, cv=5)), "ada_cv stemmer ikke. Kjør kryssvalideringen på X_train og y_train."
assert hasattr(ada, "estimators_"), "Tren ada på hele X_train med ada.fit(X_train, y_train)."
assert ada_acc is not ... and abs(ada_acc - ada.score(X_test, y_test)) < 1e-9, "ada_acc skal være test-accuracy for ada."
assert ada_acc >= 0.8, f"Målet er minst 0.80 test-accuracy, fikk {ada_acc}."
```
:::
