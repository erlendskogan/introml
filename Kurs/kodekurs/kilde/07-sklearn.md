=== DEL 7 | sklearn | scikit-learn: arbeidsflyten
Ferdige modeller og forbehandlingssteg med `fit`, `predict` og `transform`, og hvordan du skalerer, koder, validerer og velger hyperparametre uten datalekkasje.

--- LEKSJON 7.1 | monsteret | fit, predict og transform
Alle modeller i scikit-learn følger mønsteret fra del 5: `model = LogisticRegression(max_iter=1000)`, `model.fit(X_train, y_train)` og `y_pred = model.predict(X_test)`. `model.score(X, y)` gir accuracy for klassifikatorer og $R^2$ for regresjonsmodeller.

::: oppgave Tren en klassifikator
Lag `model = LogisticRegression(max_iter=1000)`, tren den på treningsdataene, og lagre prediksjonene på testdataene i `y_pred` og test-accuracy i `acc`.
```start
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("problem2.csv")
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["age"] = df["age"].fillna(df["age"].median())
X = df[["sex", "age", "family_aboard", "prior_voyages", "provisions_kg"]]
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = ...
y_pred = ...
acc = ...
```
```losning
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("problem2.csv")
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["age"] = df["age"].fillna(df["age"].median())
X = df[["sex", "age", "family_aboard", "prior_voyages", "provisions_kg"]]
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = model.score(X_test, y_test)
```
```test
import numpy as np
from sklearn.linear_model import LogisticRegression
assert isinstance(model, LogisticRegression), "model skal være en LogisticRegression."
assert hasattr(model, "coef_"), "model er ikke trent. Kall model.fit(X_train, y_train)."
_f = LogisticRegression(max_iter=1000).fit(X_train, y_train)
assert np.allclose(model.coef_, _f.coef_, atol=1e-3), "Vektene stemmer ikke. Tren med max_iter=1000 på X_train og y_train (ikke testdataene)."
assert np.shape(y_pred) == (len(X_test),), f"y_pred skal ha én prediksjon per rad i X_test, men har shape {np.shape(y_pred)}."
assert np.array_equal(np.asarray(y_pred), model.predict(X_test)), "y_pred skal være model.predict(X_test)."
assert abs(float(acc) - model.score(X_test, y_test)) < 1e-9, "acc skal være model.score(X_test, y_test)."
```
:::

## predict_proba og lærte attributter
`model.predict_proba(X)` gir én kolonne per klasse (rekkefølgen står i `model.classes_`), og det modellen lærte ligger i attributter med understrek, som `coef_` og `intercept_`. `fit` returnerer modellen selv, så `LogisticRegression().fit(X, y)` går på én linje.

::: oppgave Sannsynligheter og vekter
Lagre `proba = model.predict_proba(X_test)`, sannsynligheten for klasse 1 i `p_overlever`, vektene i `w` (`coef_[0]`) og biasen i `b` (`intercept_[0]`).
```start
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("problem2.csv")
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["age"] = df["age"].fillna(df["age"].median())
X = df[["sex", "age", "family_aboard", "prior_voyages", "provisions_kg"]]
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

proba = ...
p_overlever = ...
w = ...
b = ...
```
```losning
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("problem2.csv")
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["age"] = df["age"].fillna(df["age"].median())
X = df[["sex", "age", "family_aboard", "prior_voyages", "provisions_kg"]]
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

proba = model.predict_proba(X_test)
p_overlever = proba[:, 1]
w = model.coef_[0]
b = model.intercept_[0]
```
```test
import numpy as np
assert np.shape(proba) == (len(X_test), 2), f"proba skal ha shape ({len(X_test)}, 2): én rad per testpunkt og én kolonne per klasse. Du har {np.shape(proba)}."
assert np.allclose(proba, model.predict_proba(X_test)), "proba skal være model.predict_proba(X_test)."
assert np.shape(p_overlever) == (len(X_test),) and np.allclose(p_overlever, proba[:, 1]), "p_overlever skal være kolonnen for klasse 1: proba[:, 1]."
assert np.shape(w) == (5,) and np.allclose(w, model.coef_[0]), "w skal være model.coef_[0] (fem vekter, én per feature)."
assert np.ndim(b) == 0 and abs(float(b) - model.intercept_[0]) < 1e-12, "b skal være ett tall: model.intercept_[0]."
```
:::

::: oppgave Regn ut sannsynligheten selv
Bruk sigmoid-formelen fra del 5 med `w` og `b` til å lage `p_selv` for alle rader i `X_test`, og lag `y_selv` som er 1 der `p_selv > 0.5` og ellers 0.
```start
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("problem2.csv")
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["age"] = df["age"].fillna(df["age"].median())
X = df[["sex", "age", "family_aboard", "prior_voyages", "provisions_kg"]]
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
w, b = model.coef_[0], model.intercept_[0]

p_selv = ...
y_selv = ...
```
```losning
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("problem2.csv")
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["age"] = df["age"].fillna(df["age"].median())
X = df[["sex", "age", "family_aboard", "prior_voyages", "provisions_kg"]]
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
w, b = model.coef_[0], model.intercept_[0]

p_selv = 1 / (1 + np.exp(-(X_test.values @ w + b)))
y_selv = (p_selv > 0.5).astype(int)
```
```test
import numpy as np
assert np.shape(p_selv) == (len(X_test),), f"p_selv skal ha én verdi per rad i X_test, men har shape {np.shape(p_selv)}."
assert np.allclose(np.asarray(p_selv, dtype=float), model.predict_proba(X_test)[:, 1]), "p_selv skal være 1 / (1 + np.exp(-(X_test @ w + b))), som er det samme som predict_proba(X_test)[:, 1]."
assert np.array_equal(np.asarray(y_selv).astype(int), model.predict(X_test)), "y_selv skal være 1 der p_selv > 0.5 og 0 ellers, og blir da lik model.predict(X_test)."
```
:::

## Regresjon
`LinearRegression().fit(X, y)` har det samme mønsteret, men `X` må være 2D også med én feature: `data[["Net_Activity"]]`, ikke `data["Net_Activity"]`.

::: oppgave Energiformelen med scikit-learn
Tren `reg = LinearRegression()` på `regression.csv` med `Net_Activity` som feature og `Energy` som target, og lagre stigningstallet i `a` (et tall), konstantleddet i `b` og $R^2$ på de samme dataene i `r2`.
```start
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("regression.csv")

reg = ...
a = ...
b = ...
r2 = ...
```
```losning
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("regression.csv")
X = data[["Net_Activity"]]
y = data["Energy"]
reg = LinearRegression().fit(X, y)
a = reg.coef_[0]
b = reg.intercept_
r2 = reg.score(X, y)
```
```test
import numpy as np
from sklearn.linear_model import LinearRegression
assert isinstance(reg, LinearRegression), "reg skal være en LinearRegression-modell."
assert hasattr(reg, "coef_"), "reg er ikke trent ennå. Kall reg.fit(X, y) med X = data[['Net_Activity']]."
assert np.size(a) == 1, "a skal være ett tall: reg.coef_[0]."
assert abs(float(np.ravel(a)[0]) - 3.00869655) < 1e-4, f"a skal være stigningstallet reg.coef_[0] (ca. 3.009), men er {a}"
assert abs(float(b) - 4.8917048) < 1e-3, f"b skal være reg.intercept_ (ca. 4.892), men er {b}"
assert abs(float(r2) - 0.98683800) < 1e-4, f"r2 skal være reg.score(X, y) (ca. 0.987), men er {r2}"
```
:::

## Transformere
En transformer som `StandardScaler()` har `fit` (lærer `mean_` og `scale_` per kolonne), `transform` (regner `(x - mean_) / scale_`) og `fit_transform` (begge). `fit` på en scaler regner altså ut statistikk uten `y`, mens `fit` på en modell trener den.

::: oppgave Din første scaler
Lag `scaler = StandardScaler()`, tilpass den på `X_train`, lagre `snitt = scaler.mean_` og de skalerte dataene i `X_s`.
```start
import numpy as np
from sklearn.preprocessing import StandardScaler

X_train = np.array([[1.0, 100.0],
                    [2.0, 200.0],
                    [3.0, 600.0]])

scaler = ...
snitt = ...
X_s = ...
```
```losning
import numpy as np
from sklearn.preprocessing import StandardScaler

X_train = np.array([[1.0, 100.0],
                    [2.0, 200.0],
                    [3.0, 600.0]])

scaler = StandardScaler()
scaler.fit(X_train)
snitt = scaler.mean_
X_s = scaler.transform(X_train)
```
```test
import numpy as np
from sklearn.preprocessing import StandardScaler
assert isinstance(scaler, StandardScaler), "scaler skal være en StandardScaler()."
assert hasattr(scaler, "mean_"), "scaler er ikke tilpasset. Kall scaler.fit(X_train)."
assert np.allclose(snitt, [2.0, 300.0]), f"snitt skal være scaler.mean_ ([2, 300]), men er {snitt}"
assert np.shape(X_s) == (3, 2), f"X_s skal ha shape (3, 2), men har {np.shape(X_s)}"
assert np.allclose(np.mean(X_s, axis=0), 0) and np.allclose(np.std(X_s, axis=0), 1), "Hver kolonne i X_s skal ha snitt 0 og std 1. Bruk scaler.transform(X_train) eller scaler.fit_transform(X_train)."
```
:::

::: oppgave Utrent modell
Kall `predict` på en utrent `LogisticRegression()` inni `try`, fang `NotFittedError` og lagre feilmeldingen `str(e)` i `melding`.
```start
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
melding = ""
```
```losning
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
melding = ""
try:
    model.predict([[1, 25, 0, 2, 10.0]])
except NotFittedError as e:
    melding = str(e)
```
```test
assert isinstance(melding, str) and "not fitted" in melding, "melding skal være teksten fra NotFittedError. Kall model.predict(...) i try og skriv except NotFittedError as e: melding = str(e)."
assert not hasattr(model, "coef_"), "model skal være utrent: ikke kall fit."
```
:::

--- LEKSJON 7.2 | skalering | Skalering uten datalekkasje
`StandardScaler` gir $(x - \mu)/\sigma$ (snitt 0 og std 1, ikke begrenset), og `MinMaxScaler` gir $(x - \min)/(\max - \min)$ (mellom 0 og 1 på dataene den ble tilpasset på). Gradient descent og avstandsbaserte metoder trenger skalering, beslutningstrær gjør det ikke.

::: oppgave To skalerere
Lag `std` med `StandardScaler().fit_transform(X)` og `mm` med `MinMaxScaler().fit_transform(X)`, og lagre største verdi i første kolonne av `std` i `std_maks`.
```start
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("problem2.csv")
X = df[["provisions_kg", "prior_voyages", "family_aboard"]]

std = ...
mm = ...
std_maks = ...
```
```losning
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("problem2.csv")
X = df[["provisions_kg", "prior_voyages", "family_aboard"]]

std = StandardScaler().fit_transform(X)
mm = MinMaxScaler().fit_transform(X)
std_maks = std[:, 0].max()
```
```test
import numpy as np
assert np.shape(std) == (500, 3) and np.shape(mm) == (500, 3), "std og mm skal begge ha shape (500, 3)."
assert np.allclose(np.mean(std, axis=0), 0) and np.allclose(np.std(std, axis=0), 1), "std skal ha snitt 0 og std 1 i hver kolonne (StandardScaler)."
assert np.allclose(np.min(mm, axis=0), 0) and np.allclose(np.max(mm, axis=0), 1), "mm skal ha min 0 og maks 1 i hver kolonne (MinMaxScaler)."
assert abs(float(std_maks) - np.asarray(std)[:, 0].max()) < 1e-9, "std_maks skal være std[:, 0].max()."
assert std_maks > 8, "std_maks skal være over 8: StandardScaler begrenser ikke verdiene."
```
:::

::: oppgave Skaler for hånd
Lag `std_hand` med formelen `(X - X.mean()) / X.std(ddof=0)` og `mm_hand` med `(X - X.min()) / (X.max() - X.min())`, og sjekk i variabelpanelet at de er like `std` og `mm`.
```start
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("problem2.csv")
X = df[["provisions_kg", "prior_voyages", "family_aboard"]]
std = StandardScaler().fit_transform(X)
mm = MinMaxScaler().fit_transform(X)

std_hand = ...
mm_hand = ...
```
```losning
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("problem2.csv")
X = df[["provisions_kg", "prior_voyages", "family_aboard"]]
std = StandardScaler().fit_transform(X)
mm = MinMaxScaler().fit_transform(X)

std_hand = (X - X.mean()) / X.std(ddof=0)
mm_hand = (X - X.min()) / (X.max() - X.min())
```
```test
import numpy as np
assert np.shape(std_hand) == (500, 3) and np.shape(mm_hand) == (500, 3), "std_hand og mm_hand skal ha shape (500, 3)."
assert np.allclose(np.asarray(std_hand, dtype=float), std), "std_hand skal være (X - X.mean()) / X.std(ddof=0). StandardScaler bruker ddof=0."
assert np.allclose(np.asarray(mm_hand, dtype=float), mm), "mm_hand skal være (X - X.min()) / (X.max() - X.min())."
```
:::

## Fit på train, transform på begge
`scaler.fit(X_train)` lærer tallene bare fra treningsdataene, og `scaler.transform(X_test)` bruker de samme tallene på testdataene.

::: oppgave Skaler vikingdataene riktig
Lag en `StandardScaler` som heter `scaler`, tilpass den på `X_train`, og lag `X_train_s` og `X_test_s` med den.
```start
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("problem2.csv")
X = df[["provisions_kg", "prior_voyages", "family_aboard"]]
X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)

scaler = ...
X_train_s = ...
X_test_s = ...
```
```losning
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("problem2.csv")
X = df[["provisions_kg", "prior_voyages", "family_aboard"]]
X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
```
```test
import numpy as np
from sklearn.preprocessing import StandardScaler
assert isinstance(scaler, StandardScaler), "scaler skal være en StandardScaler()."
assert hasattr(scaler, "mean_"), "scaler er ikke tilpasset. Kall scaler.fit(X_train)."
assert np.allclose(scaler.mean_, X_train.mean().values), "scaler skal tilpasses på X_train, ikke på X eller X_test."
_mu, _sd = X_train.mean().values, X_train.std(ddof=0).values
assert np.shape(X_train_s) == X_train.shape and np.allclose(X_train_s, (X_train.values - _mu) / _sd), "X_train_s skal være scaler.transform(X_train)."
assert np.shape(X_test_s) == X_test.shape, f"X_test_s skal ha shape {X_test.shape}"
assert np.allclose(X_test_s, (X_test.values - _mu) / _sd), "X_test_s skal skaleres med treningsdataenes snitt og std: scaler.transform(X_test). Ikke fit på testdataene."
```
:::

::: oppgave MinMaxScaler på nye verdier
Tilpass `mm = MinMaxScaler()` på `alder_train`, lagre de lærte tallene `mm.data_min_` og `mm.data_max_` i `lav` og `hoy`, og lag `alder_test_s` med `transform` (se at 70 havner utenfor [0, 1]).
```start
import numpy as np
from sklearn.preprocessing import MinMaxScaler

alder_train = np.array([[20.0], [35.0], [60.0]])
alder_test = np.array([[40.0], [70.0]])

mm = ...
lav = ...
hoy = ...
alder_test_s = ...
```
```losning
import numpy as np
from sklearn.preprocessing import MinMaxScaler

alder_train = np.array([[20.0], [35.0], [60.0]])
alder_test = np.array([[40.0], [70.0]])

mm = MinMaxScaler().fit(alder_train)
lav = mm.data_min_
hoy = mm.data_max_
alder_test_s = mm.transform(alder_test)
```
```test
import numpy as np
from sklearn.preprocessing import MinMaxScaler
assert isinstance(mm, MinMaxScaler) and hasattr(mm, "data_min_"), "mm skal være en MinMaxScaler tilpasset på alder_train."
assert np.allclose(lav, 20) and np.allclose(hoy, 60), f"lav og hoy skal være mm.data_min_ (20) og mm.data_max_ (60), men er {lav} og {hoy}. Tilpasset du på alder_train?"
assert np.allclose(np.ravel(alder_test_s), [0.5, 1.25]), f"alder_test_s skal være [0.5, 1.25]: (x - 20) / (60 - 20). Du har {np.ravel(alder_test_s)}."
```
:::

::: oppgave Hvilke tall bruker transform?
Alder har snitt 40 og std 8 i treningsdataene, og snitt 45 og std 10 i testdataene. Regn ut `z`: verdien `StandardScaler().fit(X_train)` og så `transform` gir en testperson på 52 år.
```start
snitt_train, std_train = 40, 8
snitt_test, std_test = 45, 10

z = ...
```
```losning
snitt_train, std_train = 40, 8
snitt_test, std_test = 45, 10

z = (52 - snitt_train) / std_train
```
```test
assert z is not ..., "Regn ut z."
assert abs(z - 0.7) > 1e-9, "Du brukte testdataenes tall. Scaleren er tilpasset på X_train, så den bruker snitt 40 og std 8."
assert abs(z - 1.5) < 1e-9, f"z skal være (52 - 40) / 8 = 1.5, men er {z}"
```
:::

## Datalekkasje
Datalekkasje betyr at testdataene påvirker treningen, for eksempel med `StandardScaler().fit(X)` på alle data før splitting, eller en ny scaler tilpasset på testdataene.

::: oppgave Ett nytt datapunkt
Lag `scaler_alle` tilpasset på hele `X` (feil) og `scaler_train` tilpasset på `X_train` (riktig). Skaler så `ny_viking` på to måter: `feil = StandardScaler().fit_transform(ny_viking)` og `riktig` med `scaler_train`.
```start
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("problem2.csv")
X = df[["provisions_kg", "prior_voyages", "family_aboard"]]
X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)
ny_viking = X_test.iloc[[0]]

scaler_alle = ...
scaler_train = ...
feil = ...
riktig = ...
```
```losning
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("problem2.csv")
X = df[["provisions_kg", "prior_voyages", "family_aboard"]]
X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)
ny_viking = X_test.iloc[[0]]

scaler_alle = StandardScaler().fit(X)
scaler_train = StandardScaler().fit(X_train)
feil = StandardScaler().fit_transform(ny_viking)
riktig = scaler_train.transform(ny_viking)
```
```test
import numpy as np
from sklearn.preprocessing import StandardScaler
assert isinstance(scaler_alle, StandardScaler) and hasattr(scaler_alle, "mean_"), "scaler_alle skal være StandardScaler().fit(X)."
assert np.allclose(scaler_alle.mean_, X.mean().values), "scaler_alle skal tilpasses på hele X."
assert isinstance(scaler_train, StandardScaler) and hasattr(scaler_train, "mean_"), "scaler_train skal være StandardScaler().fit(X_train)."
assert np.allclose(scaler_train.mean_, X_train.mean().values), "scaler_train skal tilpasses på X_train."
assert not np.allclose(scaler_alle.mean_, scaler_train.mean_), "De to scalerne skal ha ulike snitt."
assert np.shape(feil) == (1, 3) and np.allclose(feil, 0), "feil skal være StandardScaler().fit_transform(ny_viking): en scaler tilpasset på ett punkt gjør det til bare nuller."
assert np.shape(riktig) == (1, 3) and np.allclose(riktig, (ny_viking.values - X_train.mean().values) / X_train.std(ddof=0).values), "riktig skal være scaler_train.transform(ny_viking)."
```
:::

--- LEKSJON 7.3 | encoding | LabelEncoder og OneHotEncoder
Encoderne i `sklearn.preprocessing` husker kodingen fra `fit`, så train og test kodes likt, i motsetning til `map` og `pd.get_dummies` fra del 3.

## LabelEncoder
`le = LabelEncoder()` og `le.fit_transform(kolonne)` gir hver kategori et heltall i alfabetisk rekkefølge, `le.classes_` viser rekkefølgen, og `le.inverse_transform([0, 1])` går tilbake til tekst.

::: oppgave Kod kjønn
Lag `le = LabelEncoder()`, lagre `le.fit_transform(df["sex"])` i den nye kolonnen `df["sex_kode"]`, og gjør kodene `[0, 1, 1]` tilbake til tekst i `tilbake`.
```start
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("problem2.csv")

le = ...
tilbake = ...
```
```losning
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("problem2.csv")

le = LabelEncoder()
df["sex_kode"] = le.fit_transform(df["sex"])
tilbake = le.inverse_transform([0, 1, 1])
```
```test
from sklearn.preprocessing import LabelEncoder
assert isinstance(le, LabelEncoder) and hasattr(le, "classes_"), "le skal være en LabelEncoder tilpasset på df['sex']."
assert list(le.classes_) == ["Female", "Male"], f"le.classes_ skal være ['Female', 'Male'], men er {list(le.classes_)}."
assert "sex_kode" in df.columns, "Lag kolonnen df['sex_kode']."
assert (df["sex_kode"] == (df["sex"] == "Male").astype(int)).all(), "df['sex_kode'] skal være 0 for Female og 1 for Male (alfabetisk)."
assert list(tilbake) == ["Female", "Male", "Male"], f"tilbake skal være le.inverse_transform([0, 1, 1]), men er {list(tilbake)}."
```
:::

::: oppgave Gjett kodene
Lagre `LabelEncoder().fit_transform(["Thrall", "Jarl", "Karl", "Jarl"])` i `koder`, men skriv først ned hva du tror den gir.
```start
from sklearn.preprocessing import LabelEncoder

koder = ...
```
```losning
from sklearn.preprocessing import LabelEncoder

koder = LabelEncoder().fit_transform(["Thrall", "Jarl", "Karl", "Jarl"])
```
```test
import numpy as np
assert np.ndim(koder) == 1 and len(koder) == 4, "koder skal ha fire tall, ett per element i lista."
assert list(np.asarray(koder)) == [2, 0, 1, 0], f"koder skal være [2, 0, 1, 0]: alfabetisk er Jarl = 0, Karl = 1 og Thrall = 2. Du har {list(koder)}."
```
:::

## Falsk rekkefølge
Med flere kategorier lager LabelEncoder en rekkefølge som ikke finnes, og en lineær modell får én vekt for hele featuren. Bruk den til target `y` (og eventuelt for trær), ikke til nominelle features i lineære modeller eller nevrale nett.

::: oppgave Overlevelse per rollekode
Kod `role` med en ny `LabelEncoder` i `df["role_kode"]`, og lag `rate = df.groupby("role_kode")["survived"].mean()`. Lagre `monoton = rate.is_monotonic_increasing or rate.is_monotonic_decreasing`.
```start
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("problem2.csv")

rate = ...
monoton = ...
```
```losning
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("problem2.csv")

df["role_kode"] = LabelEncoder().fit_transform(df["role"])
rate = df.groupby("role_kode")["survived"].mean()
monoton = rate.is_monotonic_increasing or rate.is_monotonic_decreasing
```
```test
import numpy as np
assert "role_kode" in df.columns, "Lag kolonnen df['role_kode'] med LabelEncoder().fit_transform(df['role'])."
_k = {"Hirdman": 0, "Jarl": 1, "Karl": 2, "Thrall": 3}
assert (df["role_kode"] == df["role"].map(_k)).all(), "role_kode skal være alfabetisk: Hirdman 0, Jarl 1, Karl 2, Thrall 3."
assert len(rate) == 4, "rate skal ha én verdi per rollekode (4)."
assert np.allclose(np.asarray(rate), df.groupby(df["role"].map(_k))["survived"].mean().values), "rate skal være df.groupby('role_kode')['survived'].mean()."
assert monoton is False or monoton == False, "monoton skal bli False: overlevelsen går ikke jevnt opp eller ned med koden."
assert monoton is not ..., "Regn ut monoton."
```
:::

## OneHotEncoder
`OneHotEncoder(handle_unknown="ignore", sparse_output=False)` gir én 0/1-kolonne per kategori som et numpy-array, og `ohe.get_feature_names_out()` gir kolonnenavnene. Med `handle_unknown="ignore"` blir en ukjent kategori bare nuller, og `drop="first"` tilsvarer `drop_first=True` i `get_dummies`.

::: oppgave One-hot med fit på train
Lag `ohe` med `handle_unknown="ignore"` og `sparse_output=False`, tilpass den på kolonnene `role` og `ship` i `train`, og lag `X_train_oh` og `X_test_oh`.
```start
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("problem2.csv")
train, test = df.iloc[:400], df.iloc[400:]

ohe = ...
X_train_oh = ...
X_test_oh = ...
```
```losning
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("problem2.csv")
train, test = df.iloc[:400], df.iloc[400:]

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.fit(train[["role", "ship"]])
X_train_oh = ohe.transform(train[["role", "ship"]])
X_test_oh = ohe.transform(test[["role", "ship"]])
```
```test
import numpy as np
from sklearn.preprocessing import OneHotEncoder
assert isinstance(ohe, OneHotEncoder), "ohe skal være en OneHotEncoder."
assert hasattr(ohe, "categories_"), "ohe er ikke tilpasset. Kall ohe.fit(train[['role', 'ship']])."
assert ohe.handle_unknown == "ignore", 'Bruk handle_unknown="ignore".'
assert isinstance(X_train_oh, np.ndarray) and isinstance(X_test_oh, np.ndarray), "X_train_oh og X_test_oh skal være numpy-arrays. Bruk sparse_output=False."
assert X_train_oh.shape == (400, 9), f"X_train_oh skal ha shape (400, 9): 4 roller + 5 skip. Du har {X_train_oh.shape}"
assert X_test_oh.shape == (100, 9), f"X_test_oh skal ha shape (100, 9), men har {X_test_oh.shape}"
assert np.allclose(X_test_oh.sum(axis=1), 2), "Hver rad skal ha nøyaktig to enere (én for rolle og én for skip)."
```
:::

::: oppgave Et skip vi aldri har sett
Tilpass `ohe` (med `handle_unknown="ignore"` og `sparse_output=False`) på `df[["ship"]]`, lagre kolonnenavnene i `navn` og `ohe.transform(ny)` i `ny_oh`.
```start
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("problem2.csv")
ny = pd.DataFrame({"ship": ["Bison", "Nytt skip"]})

ohe = ...
navn = ...
ny_oh = ...
```
```losning
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("problem2.csv")
ny = pd.DataFrame({"ship": ["Bison", "Nytt skip"]})

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[["ship"]])
navn = ohe.get_feature_names_out()
ny_oh = ohe.transform(ny)
```
```test
import numpy as np
from sklearn.preprocessing import OneHotEncoder
assert isinstance(ohe, OneHotEncoder) and hasattr(ohe, "categories_"), "ohe skal være en OneHotEncoder tilpasset på df[['ship']]."
assert ohe.handle_unknown == "ignore", 'Bruk handle_unknown="ignore", ellers krasjer transform på «Nytt skip».'
assert len(navn) == 5 and "ship_Bison" in list(navn), f"navn skal være ohe.get_feature_names_out() med fem skip, som 'ship_Bison'. Du har {list(navn)}."
_ny = np.asarray(ny_oh.toarray() if hasattr(ny_oh, "toarray") else ny_oh)
assert _ny.shape == (2, 5), f"ny_oh skal ha shape (2, 5), men har {_ny.shape}."
assert _ny[0, list(navn).index("ship_Bison")] == 1 and _ny[0].sum() == 1, "Første rad (Bison) skal ha én ener i kolonnen ship_Bison."
assert _ny[1].sum() == 0, "Andre rad (Nytt skip) skal bare ha nuller."
```
:::

::: oppgave drop="first" og get_dummies
Kod `roller` med `OneHotEncoder(drop="first", sparse_output=False)` til `oh_drop`, og lag `dummies = pd.get_dummies(roller, drop_first=True, dtype=int)`.
```start
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

roller = pd.DataFrame({"role": ["Thrall", "Jarl", "Karl", "Hirdman", "Karl"]})

oh_drop = ...
dummies = ...
```
```losning
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

roller = pd.DataFrame({"role": ["Thrall", "Jarl", "Karl", "Hirdman", "Karl"]})

oh_drop = OneHotEncoder(drop="first", sparse_output=False).fit_transform(roller)
dummies = pd.get_dummies(roller, drop_first=True, dtype=int)
```
```test
import numpy as np
import pandas as pd
assert np.shape(oh_drop) == (5, 3), f"oh_drop skal ha shape (5, 3): fire roller minus den første. Du har {np.shape(oh_drop)}."
assert isinstance(dummies, pd.DataFrame) and dummies.shape == (5, 3), "dummies skal være pd.get_dummies(roller, drop_first=True, dtype=int) med tre kolonner."
assert np.allclose(np.asarray(oh_drop, dtype=float), dummies.values.astype(float)), "oh_drop og dummies skal inneholde de samme 0/1-verdiene."
```
:::

--- LEKSJON 7.4 | imputering | Imputering med SimpleImputer
`SimpleImputer(strategy="median")` fyller hull med medianen den lærte i `fit` (lagret i `statistics_`), så `fit` skjer på train og `transform` på begge.

::: oppgave Imputer alder og proviant
Lag `imp` med medianstrategi, tilpass den på `kolonner` i `train`, og lag `X_train_imp` og `X_test_imp`.
```start
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

df = pd.read_csv("problem2.csv")
train, test = train_test_split(df, test_size=0.2, random_state=0)
kolonner = ["age", "provisions_kg"]

imp = ...
X_train_imp = ...
X_test_imp = ...
```
```losning
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

df = pd.read_csv("problem2.csv")
train, test = train_test_split(df, test_size=0.2, random_state=0)
kolonner = ["age", "provisions_kg"]

imp = SimpleImputer(strategy="median")
imp.fit(train[kolonner])
X_train_imp = imp.transform(train[kolonner])
X_test_imp = imp.transform(test[kolonner])
```
```test
import numpy as np
from sklearn.impute import SimpleImputer
assert isinstance(imp, SimpleImputer), "imp skal være en SimpleImputer."
assert imp.strategy == "median", 'Bruk strategy="median".'
assert hasattr(imp, "statistics_"), "imp er ikke tilpasset. Kall imp.fit(train[kolonner])."
assert np.allclose(imp.statistics_, train[kolonner].median().values), "imp skal tilpasses på train, ikke på hele df eller test."
assert np.shape(X_train_imp) == (400, 2) and np.shape(X_test_imp) == (100, 2), "X_train_imp skal ha shape (400, 2) og X_test_imp (100, 2)."
assert not np.isnan(np.asarray(X_test_imp, dtype=float)).any(), "X_test_imp har fortsatt NaN."
_m = test["age"].isna().values
assert np.allclose(np.asarray(X_test_imp, dtype=float)[_m, 0], train["age"].median()), "Manglende alder i test skal fylles med medianen fra train."
```
:::

## Strategier for tekst
`strategy` kan være `"mean"` (standard), `"median"`, `"most_frequent"` eller `"constant"` med `fill_value=...`, og bare de to siste virker på tekstkolonner.

::: oppgave Median på tekst
Prøv `SimpleImputer(strategy="median").fit(df[["clan_origin"]])` inni `try`, og sett `median_feilet = True` i `except ValueError`.
```start
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv("problem2.csv")
median_feilet = False
```
```losning
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv("problem2.csv")
median_feilet = False
try:
    SimpleImputer(strategy="median").fit(df[["clan_origin"]])
except ValueError:
    median_feilet = True
```
```test
assert median_feilet is True, "median_feilet skal bli True: median kan ikke regnes ut for tekst, så fit kaster ValueError."
```
:::

::: oppgave Vanligste klan og ingen rustning
Lag `klan` med `strategy="most_frequent"` tilpasset på `df[["clan_origin"]]` og lagre den lærte verdien i `vanligste`. Fyll så `armor` med `strategy="constant", fill_value="None"` og lagre resultatet av `fit_transform` i `armor_fylt`.
```start
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv("problem2.csv")

klan = ...
vanligste = ...
armor_fylt = ...
```
```losning
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv("problem2.csv")

klan = SimpleImputer(strategy="most_frequent").fit(df[["clan_origin"]])
vanligste = klan.statistics_[0]
armor_fylt = SimpleImputer(strategy="constant", fill_value="None").fit_transform(df[["armor"]])
```
```test
import numpy as np
from sklearn.impute import SimpleImputer
assert isinstance(klan, SimpleImputer) and klan.strategy == "most_frequent", 'klan skal være SimpleImputer(strategy="most_frequent").'
assert hasattr(klan, "statistics_"), "klan er ikke tilpasset. Kall klan.fit(df[['clan_origin']])."
assert vanligste == "Nidaros", f"vanligste skal være klan.statistics_[0] (Nidaros), men er {vanligste!r}."
_a = np.asarray(armor_fylt).ravel()
assert len(_a) == 500, f"armor_fylt skal ha 500 verdier, men har {len(_a)}."
assert (_a == "None").sum() == 324, f"De 324 manglende rustningene skal fylles med 'None'. Du har {(_a == 'None').sum()}."
```
:::

## Manglende som informasjon
`SimpleImputer(strategy="median", add_indicator=True)` legger til en 0/1-kolonne som er 1 der verdien manglet.

::: oppgave Alder ukjent
Lag `imp` med medianstrategi og `add_indicator=True`, lagre `imp.fit_transform(df[["age"]])` i `ut` og kolonnenavnene i `navn`.
```start
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv("problem2.csv")

imp = ...
ut = ...
navn = ...
```
```losning
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv("problem2.csv")

imp = SimpleImputer(strategy="median", add_indicator=True)
ut = imp.fit_transform(df[["age"]])
navn = imp.get_feature_names_out()
```
```test
import numpy as np
from sklearn.impute import SimpleImputer
assert isinstance(imp, SimpleImputer) and imp.add_indicator, "imp skal være SimpleImputer(strategy='median', add_indicator=True)."
assert np.shape(ut) == (500, 2), f"ut skal ha shape (500, 2): alder og indikatorkolonnen. Du har {np.shape(ut)}."
assert np.allclose(np.asarray(ut)[:, 1], df["age"].isna().astype(float)), "Andre kolonne skal være 1 der alder manglet."
assert np.allclose(np.asarray(ut)[df["age"].isna().values, 0], df["age"].median()), "Manglende alder skal fylles med medianen."
assert list(navn) == ["age", "missingindicator_age"], f"navn skal være imp.get_feature_names_out(), men er {list(navn)}."
```
:::

--- LEKSJON 7.5 | kryssvalidering | Validering og kryssvalidering
Treningsdata brukes av `fit`, valideringsdata til å velge modell og hyperparametre, og testdata bare én gang helt til slutt. Med `train_test_split` to ganger får du tre deler: først 20 % til test, så `test_size=0.25` av resten til validering.

::: oppgave Tre deler
Del `X` og `y` i `X_train`, `X_val`, `X_test` (og tilsvarende `y_train`, `y_val`, `y_test`) med 300, 100 og 100 rader.
```start
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("problem2.csv")
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["age"] = df["age"].fillna(df["age"].median())
X = df[["sex", "age", "family_aboard", "prior_voyages", "provisions_kg"]]
y = df["survived"]

X_train = X_val = X_test = y_train = y_val = y_test = ...
```
```losning
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("problem2.csv")
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["age"] = df["age"].fillna(df["age"].median())
X = df[["sex", "age", "family_aboard", "prior_voyages", "provisions_kg"]]
y = df["survived"]

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=0)
```
```test
_d = {"X_train": X_train, "X_val": X_val, "X_test": X_test, "y_train": y_train, "y_val": y_val, "y_test": y_test}
for _n, _v in _d.items():
    assert hasattr(_v, "index"), f"{_n} skal være en DataFrame eller Series fra train_test_split."
assert (len(X_train), len(X_val), len(X_test)) == (300, 100, 100), f"Størrelsene skal være 300, 100 og 100, men er {len(X_train)}, {len(X_val)} og {len(X_test)}. Bruk test_size=0.2 og så test_size=0.25."
assert list(X_train.index) == list(y_train.index) and list(X_val.index) == list(y_val.index) and list(X_test.index) == list(y_test.index), "X og y må deles sammen, så radene hører sammen."
assert len(set(X_train.index) | set(X_val.index) | set(X_test.index)) == 500, "De tre delene skal ikke overlappe og til sammen dekke alle 500 radene."
```
:::

::: oppgave Ett valideringssett er usikkert
For hver `seed` i `range(5)`: del `X_temp` med `test_size=0.25, random_state=seed`, tren `LogisticRegression(max_iter=1000)` og legg validerings-accuracy i lista `val_acc`. Lagre `spredning = max(val_acc) - min(val_acc)`.
```start
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("problem2.csv")
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["age"] = df["age"].fillna(df["age"].median())
X = df[["sex", "age", "family_aboard", "prior_voyages", "provisions_kg"]]
y = df["survived"]
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

val_acc = []
spredning = ...
```
```losning
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("problem2.csv")
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["age"] = df["age"].fillna(df["age"].median())
X = df[["sex", "age", "family_aboard", "prior_voyages", "provisions_kg"]]
y = df["survived"]
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

val_acc = []
for seed in range(5):
    X_tr, X_va, y_tr, y_va = train_test_split(X_temp, y_temp, test_size=0.25, random_state=seed)
    model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    val_acc.append(model.score(X_va, y_va))
spredning = max(val_acc) - min(val_acc)
```
```test
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
assert len(val_acc) == 5, f"val_acc skal ha fem verdier (én per seed), men har {len(val_acc)}."
_f = []
for _s in range(5):
    _a, _b, _c, _d = train_test_split(X_temp, y_temp, test_size=0.25, random_state=_s)
    _f.append(LogisticRegression(max_iter=1000).fit(_a, _c).score(_b, _d))
assert np.allclose(val_acc, _f, atol=1e-6), "val_acc stemmer ikke. Del X_temp og y_temp med random_state=seed og mål score på valideringsdelen."
assert abs(spredning - (max(_f) - min(_f))) < 1e-6, "spredning skal være max(val_acc) - min(val_acc)."
```
:::

## Kryssvalidering
`cross_val_score(model, X_train, y_train, cv=5)` deler treningsdataene i 5 folds og trener 5 kopier, hver på 4 folds og validert på den femte. `model` selv er like utrent etterpå, så du må kalle `model.fit` før du tester.

::: oppgave Kryssvalider, tren og test
Lagre `cross_val_score(model, X_train, y_train, cv=5)` i `cv_scores`, snittet i `cv_mean` og standardavviket i `cv_std`. Tren så `model` på treningsdataene og lagre test-accuracy i `test_acc`.
```start
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

df = pd.read_csv("problem2.csv")
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["age"] = df["age"].fillna(df["age"].median())
X = df[["sex", "age", "family_aboard", "prior_voyages", "provisions_kg"]]
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)

cv_scores = ...
cv_mean = ...
cv_std = ...
test_acc = ...
print(f"CV Accuracy: {cv_mean:.3f} ± {cv_std:.3f}")
print(f"Test accuracy: {test_acc:.4f}")
```
```losning
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

df = pd.read_csv("problem2.csv")
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["age"] = df["age"].fillna(df["age"].median())
X = df[["sex", "age", "family_aboard", "prior_voyages", "provisions_kg"]]
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)

cv_scores = cross_val_score(model, X_train, y_train, cv=5)
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()
model.fit(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"CV Accuracy: {cv_mean:.3f} ± {cv_std:.3f}")
print(f"Test accuracy: {test_acc:.4f}")
```
```test
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
_fasit = cross_val_score(LogisticRegression(max_iter=1000), X_train, y_train, cv=5)
assert np.shape(cv_scores) == (5,), f"cv_scores skal ha 5 verdier (cv=5), men har shape {np.shape(cv_scores)}"
assert np.allclose(cv_scores, _fasit), "cv_scores stemmer ikke. Bruker du X_train og y_train (ikke hele X eller testdataene)?"
assert abs(cv_mean - _fasit.mean()) < 1e-9, "cv_mean skal være cv_scores.mean()."
assert abs(cv_std - _fasit.std()) < 1e-9, "cv_std skal være cv_scores.std()."
assert hasattr(model, "coef_"), "model er fortsatt utrent: cross_val_score trener bare kopier. Kall model.fit(X_train, y_train)."
assert abs(test_acc - LogisticRegression(max_iter=1000).fit(X_train, y_train).score(X_test, y_test)) < 1e-9, "test_acc skal være model.score(X_test, y_test) etter model.fit(X_train, y_train)."
```
:::

## KFold og StratifiedKFold
`for train_idx, val_idx in KFold(n_splits=5).split(X):` gir indeksene for hver fold i rekkefølge, og `StratifiedKFold(n_splits=5).split(X, y)` gir hver fold samme klassefordeling (standard i `cross_val_score` for klassifikatorer). `cv=KFold(n_splits=5, shuffle=True, random_state=0)` stokker først.

::: oppgave Klassene i hver fold
Lag lista `kf_klasser` med `y[val_idx]` for hver fold fra `KFold(n_splits=5)`, og lista `skf_klasser` på samme måte med `StratifiedKFold(n_splits=5)`.
```start
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

X = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

kf_klasser = []
skf_klasser = []
```
```losning
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

X = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

kf_klasser = []
for train_idx, val_idx in KFold(n_splits=5).split(X):
    kf_klasser.append(y[val_idx])
skf_klasser = []
for train_idx, val_idx in StratifiedKFold(n_splits=5).split(X, y):
    skf_klasser.append(y[val_idx])
```
```test
import numpy as np
assert len(kf_klasser) == 5 and len(skf_klasser) == 5, "Begge listene skal ha fem elementer, ett per fold."
assert [np.ravel(k).tolist() for k in kf_klasser] == [[0, 0], [0, 0], [0, 1], [1, 1], [1, 1]], "KFold uten stokking tar radene i rekkefølge, så fire av foldene har bare én klasse. Bruk KFold(n_splits=5).split(X) og y[val_idx]."
assert all(sorted(np.ravel(k).tolist()) == [0, 1] for k in skf_klasser), "Med StratifiedKFold skal hver valideringsfold ha én 0 og én 1. Husk .split(X, y)."
```
:::

::: oppgave Stokket KFold
Lag `cv = KFold(n_splits=4, shuffle=True, random_state=0)`, lagre antall trenings- og valideringsrader i første runde av `cv.split(X)` i `n_train` og `n_val`, og kryssvalider `LogisticRegression()` med `cv=cv` til `scores`.
```start
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 3))
y = (X[:, 0] + X[:, 1] > 0).astype(int)

cv = ...
n_train = ...
n_val = ...
scores = ...
```
```losning
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 3))
y = (X[:, 0] + X[:, 1] > 0).astype(int)

cv = KFold(n_splits=4, shuffle=True, random_state=0)
train_idx, val_idx = next(iter(cv.split(X)))
n_train = len(train_idx)
n_val = len(val_idx)
scores = cross_val_score(LogisticRegression(), X, y, cv=cv)
```
```test
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
assert isinstance(cv, KFold) and cv.n_splits == 4 and cv.shuffle, "cv skal være KFold(n_splits=4, shuffle=True, random_state=0)."
assert (n_train, n_val) == (75, 25), f"Med 4 folds og 100 rader trenes det på 75 og valideres på 25, men du har {n_train} og {n_val}."
assert np.shape(scores) == (4,), f"scores skal ha fire verdier (én per fold), men har shape {np.shape(scores)}."
assert np.allclose(scores, cross_val_score(LogisticRegression(), X, y, cv=KFold(n_splits=4, shuffle=True, random_state=0))), "scores skal være cross_val_score(LogisticRegression(), X, y, cv=cv)."
```
:::

## cross_validate
`cross_validate(model, X, y, cv=5, scoring=["accuracy", "f1"], return_train_score=True)` gir en dict med arrays som `res["test_accuracy"]`, `res["train_accuracy"]` og `res["test_f1"]`, der `test_` betyr valideringsfoldene. Stor avstand mellom train og validering tyder på overtilpasning.

::: oppgave Flere metrikker
Kjør `cross_validate` som over på `X` og `y` med `LogisticRegression(max_iter=1000)`, lagre dicten i `res` og snittene i `val_acc`, `train_acc` og `val_f1`.
```start
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

df = pd.read_csv("problem2.csv")
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["age"] = df["age"].fillna(df["age"].median())
X = df[["sex", "age", "family_aboard", "prior_voyages", "provisions_kg"]]
y = df["survived"]

res = ...
val_acc = ...
train_acc = ...
val_f1 = ...
```
```losning
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

df = pd.read_csv("problem2.csv")
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["age"] = df["age"].fillna(df["age"].median())
X = df[["sex", "age", "family_aboard", "prior_voyages", "provisions_kg"]]
y = df["survived"]

res = cross_validate(LogisticRegression(max_iter=1000), X, y, cv=5,
                     scoring=["accuracy", "f1"], return_train_score=True)
val_acc = res["test_accuracy"].mean()
train_acc = res["train_accuracy"].mean()
val_f1 = res["test_f1"].mean()
```
```test
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
assert isinstance(res, dict), "res skal være dicten fra cross_validate."
for _k in ["test_accuracy", "train_accuracy", "test_f1"]:
    assert _k in res, f"res mangler nøkkelen {_k!r}. Bruk scoring=['accuracy', 'f1'] og return_train_score=True."
_f = cross_validate(LogisticRegression(max_iter=1000), X, y, cv=5, scoring=["accuracy", "f1"], return_train_score=True)
assert abs(val_acc - _f["test_accuracy"].mean()) < 1e-9, "val_acc skal være res['test_accuracy'].mean()."
assert abs(train_acc - _f["train_accuracy"].mean()) < 1e-9, "train_acc skal være res['train_accuracy'].mean()."
assert abs(val_f1 - _f["test_f1"].mean()) < 1e-9, "val_f1 skal være res['test_f1'].mean()."
```
:::

--- LEKSJON 7.6 | hyperparametre | Hyperparametersøk
Parametre (vekter, bias) lærer modellen i `fit`, mens hyperparametre som `C` i `LogisticRegression(C=0.1)` (liten `C` gir sterk regularisering), `max_depth` og læringsraten velger du selv. Velg dem på valideringsdata eller med kryssvalidering, aldri på testdataene.

## Søk med en løkke
`max(resultater, key=resultater.get)` gir nøkkelen med størst verdi i en dict.

::: oppgave Velg C på valideringsdata
For hver `C` i `[0.001, 0.01, 0.1, 1, 10, 100]`: tren `LogisticRegression(C=C)` på `X_train_s` og lagre validerings-accuracy i `resultater[C]`. Lagre beste verdi i `beste_C`, og tren så den endelige modellen med `beste_C` på `X_train_s` og lagre test-accuracy i `test_acc`.
```start
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=0)
scaler = StandardScaler().fit(X_train)
X_train_s, X_val_s, X_test_s = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

resultater = {}
beste_C = ...
test_acc = ...
```
```losning
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=0)
scaler = StandardScaler().fit(X_train)
X_train_s, X_val_s, X_test_s = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

resultater = {}
for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    model = LogisticRegression(C=C).fit(X_train_s, y_train)
    resultater[C] = model.score(X_val_s, y_val)
beste_C = max(resultater, key=resultater.get)
test_acc = LogisticRegression(C=beste_C).fit(X_train_s, y_train).score(X_test_s, y_test)
```
```test
from sklearn.linear_model import LogisticRegression
_Cs = [0.001, 0.01, 0.1, 1, 10, 100]
assert sorted(resultater.keys()) == _Cs, f"resultater skal ha nøklene {_Cs}, men har {sorted(resultater.keys())}."
_f = {C: LogisticRegression(C=C).fit(X_train_s, y_train).score(X_val_s, y_val) for C in _Cs}
assert all(abs(resultater[C] - _f[C]) < 1e-9 for C in _Cs), "Noen verdier i resultater stemmer ikke. Tren på X_train_s og mål score på X_val_s (ikke testdataene)."
assert beste_C == max(_f, key=_f.get), f"beste_C skal være C-en med best validerings-accuracy: max(resultater, key=resultater.get). Du har {beste_C}."
assert abs(test_acc - LogisticRegression(C=beste_C).fit(X_train_s, y_train).score(X_test_s, y_test)) < 1e-9, "test_acc skal være test-accuracy for LogisticRegression(C=beste_C) trent på X_train_s."
```
:::

## GridSearchCV
`grid = GridSearchCV(LogisticRegression(), param_grid={"C": [0.1, 1]}, cv=5).fit(X_train, y_train)` prøver alle verdier med kryssvalidering og gir `best_params_`, `best_score_` (snittet i kryssvalideringen) og `best_estimator_`, trent på nytt på alle treningsdataene og brukt av `grid.score`. Alle kombinasjoner prøves, så antall treninger er antall kombinasjoner ganger antall folds.

::: oppgave Finn beste C
Lag `grid` med `LogisticRegression()`, `C` i `[0.01, 0.1, 1, 10]` og `cv=5`, og tilpass det på `X_train_s` og `y_train`. Lagre `beste_C`, `cv_snitt` (`best_score_`) og `test_acc` på testdataene.
```start
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

grid = ...
beste_C = ...
cv_snitt = ...
test_acc = ...
```
```losning
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

grid = GridSearchCV(LogisticRegression(), param_grid={"C": [0.01, 0.1, 1, 10]}, cv=5)
grid.fit(X_train_s, y_train)
beste_C = grid.best_params_["C"]
cv_snitt = grid.best_score_
test_acc = grid.score(X_test_s, y_test)
```
```test
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
assert isinstance(grid, GridSearchCV), "grid skal være et GridSearchCV-objekt."
assert hasattr(grid, "best_params_"), "grid er ikke tilpasset. Kall grid.fit(X_train_s, y_train)."
assert sorted(grid.param_grid["C"]) == [0.01, 0.1, 1, 10], "Prøv nøyaktig C-verdiene [0.01, 0.1, 1, 10]."
_g = GridSearchCV(LogisticRegression(), param_grid={"C": [0.01, 0.1, 1, 10]}, cv=5).fit(X_train_s, y_train)
assert beste_C == _g.best_params_["C"], f"beste_C skal hentes fra grid.best_params_['C'], men er {beste_C}"
assert abs(cv_snitt - _g.best_score_) < 1e-9, "cv_snitt skal være grid.best_score_. Tilpasset du på X_train_s og y_train?"
assert abs(test_acc - _g.score(X_test_s, y_test)) < 1e-9, "test_acc skal være grid.score(X_test_s, y_test)."
```
:::

::: oppgave To hyperparametre
Søk over `C` i `[0.01, 0.1, 1]` og `class_weight` i `[None, "balanced"]` med `cv=5` og lagre søket i `grid`. Lag `tabell = pd.DataFrame(grid.cv_results_)`, og lagre `antall_komb = len(tabell)` og `antall_treninger` (i kryssvalideringen).
```start
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_s = StandardScaler().fit_transform(X)

grid = ...
tabell = ...
antall_komb = ...
antall_treninger = ...
```
```losning
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_s = StandardScaler().fit_transform(X)

grid = GridSearchCV(LogisticRegression(),
                    param_grid={"C": [0.01, 0.1, 1], "class_weight": [None, "balanced"]},
                    cv=5)
grid.fit(X_s, y)
tabell = pd.DataFrame(grid.cv_results_)
antall_komb = len(tabell)
antall_treninger = antall_komb * 5
```
```test
import pandas as pd
from sklearn.model_selection import GridSearchCV
assert isinstance(grid, GridSearchCV) and hasattr(grid, "cv_results_"), "grid skal være et GridSearchCV som er tilpasset med grid.fit(X_s, y)."
assert isinstance(tabell, pd.DataFrame) and "mean_test_score" in tabell.columns, "tabell skal være pd.DataFrame(grid.cv_results_)."
assert antall_komb == 6, f"antall_komb skal være 3 · 2 = 6, men er {antall_komb}."
assert antall_treninger == 30, f"antall_treninger skal være 6 kombinasjoner · 5 folds = 30, men er {antall_treninger}."
```
:::

--- LEKSJON 7.7 | pipeline | Pipeline
`make_pipeline(StandardScaler(), LogisticRegression())` setter stegene sammen til én modell: `fit` kjører `fit_transform` på scaleren og `fit` på modellen, og `predict` kjører `transform` og så `predict`.

::: oppgave Din første pipeline
Lag `modell` som en pipeline med `StandardScaler()` og `LogisticRegression()`. Kryssvalider den med 5 folds på treningsdataene (lagre i `cv_scores`), tren den på treningsdataene og lagre test-accuracy i `test_acc`.
```start
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

modell = ...
cv_scores = ...
test_acc = ...
```
```losning
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

modell = make_pipeline(StandardScaler(), LogisticRegression())
cv_scores = cross_val_score(modell, X_train, y_train, cv=5)
modell.fit(X_train, y_train)
test_acc = modell.score(X_test, y_test)
```
```test
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
assert isinstance(modell, Pipeline), "modell skal være en Pipeline. Bruk make_pipeline(StandardScaler(), LogisticRegression())."
assert len(modell.steps) == 2, "Pipelinen skal ha to steg."
assert isinstance(modell.steps[0][1], StandardScaler), "Første steg skal være StandardScaler()."
assert isinstance(modell.steps[1][1], LogisticRegression), "Siste steg skal være LogisticRegression()."
assert hasattr(modell.steps[1][1], "coef_"), "modell er ikke trent. Kall modell.fit(X_train, y_train)."
_cv = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression()), X_train, y_train, cv=5)
assert np.shape(cv_scores) == (5,) and np.allclose(cv_scores, _cv), "cv_scores skal være cross_val_score(modell, X_train, y_train, cv=5)."
assert abs(test_acc - modell.score(X_test, y_test)) < 1e-9, "test_acc skal være modell.score(X_test, y_test)."
```
:::

## Navn på stegene
`make_pipeline` kaller stegene `standardscaler` og `logisticregression`, mens `Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])` lar deg velge navn selv. `pipe.named_steps["scaler"]` henter et trent steg.

::: oppgave Egne stegnavn
Lag `pipe` med `Pipeline` og stegnavnene `"scaler"` og `"model"`, tren den på treningsdataene, og hent `snitt` (scalerens `mean_`) og `vekter` (modellens `coef_`) via `named_steps`.
```start
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pipe = ...
snitt = ...
vekter = ...
```
```losning
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])
pipe.fit(X_train, y_train)
snitt = pipe.named_steps["scaler"].mean_
vekter = pipe.named_steps["model"].coef_
```
```test
import numpy as np
from sklearn.pipeline import Pipeline
assert isinstance(pipe, Pipeline), "pipe skal være en Pipeline([...])."
assert list(pipe.named_steps) == ["scaler", "model"], f"Stegene skal hete 'scaler' og 'model', men heter {list(pipe.named_steps)}."
assert hasattr(pipe.named_steps["model"], "coef_"), "pipe er ikke trent. Kall pipe.fit(X_train, y_train)."
assert np.allclose(snitt, X_train.mean(axis=0)), "snitt skal være pipe.named_steps['scaler'].mean_, lært fra X_train."
assert np.shape(vekter) == (1, 30) and np.allclose(vekter, pipe.named_steps["model"].coef_), "vekter skal være pipe.named_steps['model'].coef_."
```
:::

## Lekkasje i kryssvalidering
I `cross_val_score` tilpasses hele pipelinen på nytt i hver fold, så et steg som `SelectKBest(k=20)` (velger de 20 featurene som henger mest sammen med `y`) aldri ser valideringsfolden.

::: oppgave Ren støy
Dataene er tilfeldige, så ærlig accuracy er rundt 0.5. Lagre CV-snittet (`cv=5`) når du først lager `X_valgt = SelectKBest(k=20).fit_transform(X, y)` i `med_lekkasje`, og CV-snittet for `make_pipeline(SelectKBest(k=20), LogisticRegression())` på `X` i `med_pipeline`.
```start
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(1)
X = rng.normal(size=(200, 2000))
y = rng.integers(0, 2, size=200)

med_lekkasje = ...
med_pipeline = ...
```
```losning
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(1)
X = rng.normal(size=(200, 2000))
y = rng.integers(0, 2, size=200)

X_valgt = SelectKBest(k=20).fit_transform(X, y)
med_lekkasje = cross_val_score(LogisticRegression(), X_valgt, y, cv=5).mean()
pipe = make_pipeline(SelectKBest(k=20), LogisticRegression())
med_pipeline = cross_val_score(pipe, X, y, cv=5).mean()
```
```test
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
_l = cross_val_score(LogisticRegression(), SelectKBest(k=20).fit_transform(X, y), y, cv=5).mean()
_p = cross_val_score(make_pipeline(SelectKBest(k=20), LogisticRegression()), X, y, cv=5).mean()
assert abs(float(med_lekkasje) - _l) < 1e-9, f"med_lekkasje skal være CV-snittet på X_valgt (ca. {_l:.2f}), men er {med_lekkasje}."
assert abs(float(med_pipeline) - _p) < 1e-9, f"med_pipeline skal være CV-snittet for pipelinen på hele X (ca. {_p:.2f}), men er {med_pipeline}."
```
:::

## GridSearchCV med pipeline
Hyperparametre i et steg heter `stegnavn__parameter` med to understreker, for eksempel `"logisticregression__C"`.

::: oppgave Søk over C i pipelinen
Lag `grid = GridSearchCV(pipe, param_grid=..., cv=5)` med `C` i `[0.01, 0.1, 1, 10]` for logistisk regresjon, tilpass det på `X_train` og `y_train`, og lagre `beste_C` og `test_acc`.
```start
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
pipe = make_pipeline(StandardScaler(), LogisticRegression())

grid = ...
beste_C = ...
test_acc = ...
```
```losning
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
pipe = make_pipeline(StandardScaler(), LogisticRegression())

grid = GridSearchCV(pipe, param_grid={"logisticregression__C": [0.01, 0.1, 1, 10]}, cv=5)
grid.fit(X_train, y_train)
beste_C = grid.best_params_["logisticregression__C"]
test_acc = grid.score(X_test, y_test)
```
```test
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
assert isinstance(grid, GridSearchCV) and isinstance(grid.estimator, Pipeline), "grid skal være GridSearchCV(pipe, ...) med pipelinen som modell."
assert hasattr(grid, "best_params_"), "grid er ikke tilpasset. Kall grid.fit(X_train, y_train)."
assert list(grid.param_grid) == ["logisticregression__C"], 'Nøkkelen i param_grid skal være "logisticregression__C" (to understreker).'
_g = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression()), param_grid={"logisticregression__C": [0.01, 0.1, 1, 10]}, cv=5).fit(X_train, y_train)
assert beste_C == _g.best_params_["logisticregression__C"], f"beste_C skal være grid.best_params_['logisticregression__C'], men er {beste_C}."
assert abs(test_acc - _g.score(X_test, y_test)) < 1e-9, "test_acc skal være grid.score(X_test, y_test)."
```
:::

## Tall og tekst i samme pipeline
`make_column_transformer((steg, tallkolonner), (steg, tekstkolonner))` bruker ulike steg på ulike kolonner, og hvert steg kan selv være en `make_pipeline(...)`.

::: oppgave Hele vikingløpet
Lag `forbehandling` med median-imputering og `StandardScaler()` for `tall`, og `SimpleImputer(strategy="constant", fill_value="None")` og `OneHotEncoder(handle_unknown="ignore")` for `tekst`. Sett den foran `LogisticRegression(max_iter=1000)` i `modell`, lagre `cv_scores` (`cv=5` på treningsdataene), tren og lagre `test_acc`.
```start
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("problem2.csv")
X = df.drop(columns=["survived", "passenger_id", "name"])
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tall = ["age", "provisions_kg", "family_aboard", "prior_voyages"]
tekst = ["sex", "role", "ship", "destination", "clan_origin", "armor"]

forbehandling = ...
modell = ...
cv_scores = ...
test_acc = ...
```
```losning
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("problem2.csv")
X = df.drop(columns=["survived", "passenger_id", "name"])
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tall = ["age", "provisions_kg", "family_aboard", "prior_voyages"]
tekst = ["sex", "role", "ship", "destination", "clan_origin", "armor"]

forbehandling = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), tall),
    (make_pipeline(SimpleImputer(strategy="constant", fill_value="None"),
                   OneHotEncoder(handle_unknown="ignore")), tekst),
)
modell = make_pipeline(forbehandling, LogisticRegression(max_iter=1000))
cv_scores = cross_val_score(modell, X_train, y_train, cv=5)
modell.fit(X_train, y_train)
test_acc = modell.score(X_test, y_test)
```
```test
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
assert isinstance(forbehandling, ColumnTransformer), "forbehandling skal lages med make_column_transformer(...)."
assert isinstance(modell, Pipeline) and isinstance(modell.steps[-1][1], LogisticRegression), "modell skal være make_pipeline(forbehandling, LogisticRegression(max_iter=1000))."
assert hasattr(modell.steps[-1][1], "coef_"), "modell er ikke trent. Kall modell.fit(X_train, y_train)."
_ft = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), tall),
    (make_pipeline(SimpleImputer(strategy="constant", fill_value="None"), OneHotEncoder(handle_unknown="ignore")), tekst),
)
_cv = cross_val_score(make_pipeline(_ft, LogisticRegression(max_iter=1000)), X_train, y_train, cv=5)
assert np.shape(cv_scores) == (5,), f"cv_scores skal ha 5 verdier, men har shape {np.shape(cv_scores)}."
assert np.allclose(cv_scores, _cv, atol=1e-3), "cv_scores stemmer ikke. Sjekk stegene for tall og tekst, og at du kryssvaliderer på X_train og y_train."
assert abs(test_acc - modell.score(X_test, y_test)) < 1e-9, "test_acc skal være modell.score(X_test, y_test)."
```
:::

--- LEKSJON 7.8 | polynom-sgd | Polynomfeatures, SGDRegressor og overtilpasning
`PolynomialFeatures(degree=3, include_bias=False).fit_transform(x)` lager $x, x^2, x^3$ (med flere features også produkter som $x_0 x_1$), så en lineær modell kan tilpasse kurver. Uten `include_bias=False` kommer en ekstra kolonne med bare enere.

::: oppgave Polynomfeatures
Lag `X_poly` med grad 3 og `include_bias=False` fra `x`, og `X2_poly` med grad 2 og `include_bias=False` fra `X2`. Lagre kolonnenavnene for `X2_poly` med `get_feature_names_out(["x0", "x1"])` i `navn`.
```start
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.array([[2.0], [3.0]])
X2 = np.array([[2.0, 3.0]])

X_poly = ...
X2_poly = ...
navn = ...
```
```losning
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.array([[2.0], [3.0]])
X2 = np.array([[2.0, 3.0]])

X_poly = PolynomialFeatures(degree=3, include_bias=False).fit_transform(x)
poly2 = PolynomialFeatures(degree=2, include_bias=False)
X2_poly = poly2.fit_transform(X2)
navn = poly2.get_feature_names_out(["x0", "x1"])
```
```test
import numpy as np
assert np.shape(X_poly) == (2, 3), f"X_poly skal ha shape (2, 3): x, x² og x³. Du har {np.shape(X_poly)}. Husk include_bias=False."
assert np.allclose(X_poly, [[2, 4, 8], [3, 9, 27]]), "X_poly skal være [[2, 4, 8], [3, 9, 27]]."
assert np.shape(X2_poly) == (1, 5), f"X2_poly skal ha shape (1, 5): x0, x1, x0², x0·x1 og x1². Du har {np.shape(X2_poly)}."
assert np.allclose(X2_poly, [[2, 3, 4, 6, 9]]), "X2_poly skal være [[2, 3, 4, 6, 9]]."
assert list(navn) == ["x0", "x1", "x0^2", "x0 x1", "x1^2"], f"navn skal være poly2.get_feature_names_out(['x0', 'x1']), men er {list(navn)}."
```
:::

## XOR-problemet på nytt
Klassene i `classification.csv` er fordelt som XOR, og produktet `x0 x1` fra `PolynomialFeatures(degree=2)` er featuren som løser det.

::: oppgave Løs XOR med en pipeline
Tren `LogisticRegression()` på `train[["x0", "x1"]]` og lagre test-accuracy i `acc_lineaer`. Gjør det samme med `make_pipeline(PolynomialFeatures(degree=2), LogisticRegression())` og lagre i `acc_poly`.
```start
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]

acc_lineaer = ...
acc_poly = ...
```
```losning
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]

lineaer = LogisticRegression().fit(train[["x0", "x1"]], train["y"])
acc_lineaer = lineaer.score(test[["x0", "x1"]], test["y"])
poly = make_pipeline(PolynomialFeatures(degree=2), LogisticRegression()).fit(train[["x0", "x1"]], train["y"])
acc_poly = poly.score(test[["x0", "x1"]], test["y"])
```
```test
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
_Xtr, _Xte = train[["x0", "x1"]], test[["x0", "x1"]]
_l = LogisticRegression().fit(_Xtr, train["y"]).score(_Xte, test["y"])
_p = make_pipeline(PolynomialFeatures(degree=2), LogisticRegression()).fit(_Xtr, train["y"]).score(_Xte, test["y"])
assert abs(float(acc_lineaer) - _l) < 1e-9, f"acc_lineaer skal være test-accuracy for LogisticRegression() trent på train (ca. {_l:.2f})."
assert abs(float(acc_poly) - _p) < 1e-9, f"acc_poly skal være test-accuracy for pipelinen med PolynomialFeatures(degree=2) (ca. {_p:.2f})."
```
:::

## Over- og undertilpasning
For lav grad gir undertilpasning (høy feil både på train og test), og for høy grad gir overtilpasning (lav train-MSE, høy test-MSE). Graden er altså en hyperparameter som velges på valideringsdata.

::: oppgave Train- og test-MSE
For hver `grad` i `[1, 5, 12]`: tren `make_pipeline(PolynomialFeatures(grad), LinearRegression())` på `x_train` og lagre MSE på treningsdataene i `train_mse[grad]` og på testdataene i `test_mse[grad]`.
```start
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(0)
x = rng.uniform(0, 1, 30).reshape(-1, 1)
y = np.sin(2 * np.pi * x.ravel()) + rng.normal(0, 0.3, 30)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

train_mse = {}
test_mse = {}
```
```losning
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(0)
x = rng.uniform(0, 1, 30).reshape(-1, 1)
y = np.sin(2 * np.pi * x.ravel()) + rng.normal(0, 0.3, 30)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

train_mse = {}
test_mse = {}
for grad in [1, 5, 12]:
    modell = make_pipeline(PolynomialFeatures(grad), LinearRegression()).fit(x_train, y_train)
    train_mse[grad] = mean_squared_error(y_train, modell.predict(x_train))
    test_mse[grad] = mean_squared_error(y_test, modell.predict(x_test))
```
```test
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
assert sorted(train_mse) == [1, 5, 12] and sorted(test_mse) == [1, 5, 12], "train_mse og test_mse skal ha nøklene 1, 5 og 12."
for _g in [1, 5, 12]:
    _m = make_pipeline(PolynomialFeatures(_g), LinearRegression()).fit(x_train, y_train)
    assert abs(train_mse[_g] - mean_squared_error(y_train, _m.predict(x_train))) < 1e-6, f"train_mse[{_g}] stemmer ikke. Mål MSE på x_train og y_train."
    assert abs(test_mse[_g] - mean_squared_error(y_test, _m.predict(x_test))) < 1e-6, f"test_mse[{_g}] stemmer ikke. Mål MSE på x_test og y_test."
assert train_mse[12] < train_mse[5] < train_mse[1] and test_mse[12] > test_mse[5], "Train-MSE skal synke med graden, mens test-MSE skal øke fra grad 5 til 12 (overtilpasning)."
```
:::

::: oppgave Velg grad med valideringsdata
Prøv gradene 1 til 10, tren på `x_train`, og lagre validerings-MSE i dicten `val_mse` (nøkkel: grad). Lagre graden med lavest validerings-MSE i `beste_grad`.
```start
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(0)
x = rng.uniform(0, 1, 30).reshape(-1, 1)
y = np.sin(2 * np.pi * x.ravel()) + rng.normal(0, 0.3, 30)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.5, random_state=0)

val_mse = {}
# løkke over gradene 1 til 10 her

beste_grad = ...
```
```losning
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(0)
x = rng.uniform(0, 1, 30).reshape(-1, 1)
y = np.sin(2 * np.pi * x.ravel()) + rng.normal(0, 0.3, 30)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.5, random_state=0)

val_mse = {}
for grad in range(1, 11):
    modell = make_pipeline(PolynomialFeatures(grad), LinearRegression()).fit(x_train, y_train)
    val_mse[grad] = mean_squared_error(y_val, modell.predict(x_val))

beste_grad = min(val_mse, key=val_mse.get)
```
```test
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
assert isinstance(val_mse, dict), "val_mse skal være en dict."
assert sorted(val_mse.keys()) == list(range(1, 11)), f"val_mse skal ha nøklene 1 til 10, men har {sorted(val_mse.keys())}"
_f = {g: mean_squared_error(y_val, make_pipeline(PolynomialFeatures(g), LinearRegression()).fit(x_train, y_train).predict(x_val)) for g in range(1, 11)}
assert all(abs(val_mse[g] - _f[g]) < 1e-6 for g in _f), "Noen MSE-verdier stemmer ikke. Tren på x_train og mål på x_val."
assert beste_grad == min(_f, key=_f.get), f"beste_grad skal være graden med lavest validerings-MSE, men er {beste_grad}"
```
:::

## SGDRegressor og tapskurver
`SGDRegressor(learning_rate="constant", eta0=0.01)` er lineær regresjon trent med SGD, og `model.partial_fit(X, y)` kjører én epoke videre fra vektene den har (mens `fit` starter på nytt). Synker train-tapet mens valideringstapet stiger, er det overtilpasning, og flater begge ut høyt, er det undertilpasning.

::: oppgave Logg tapet per epoke
Kjør `model.partial_fit(X_train, y_train)` i 30 epoker, og legg MSE på treningsdataene i `train_losses` og på valideringsdataene i `val_losses` etter hver epoke. Plott begge kurvene på `ax`.
```start
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("regression.csv")
X = data[["Net_Activity"]].values
y = data["Energy"].values
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=100, test_size=100, random_state=0)
scaler = StandardScaler().fit(X_train)
X_train, X_val = scaler.transform(X_train), scaler.transform(X_val)

model = SGDRegressor(learning_rate="constant", eta0=0.01, random_state=0)
train_losses, val_losses = [], []
# løkke over 30 epoker her

fig, ax = plt.subplots()
# plott kurvene her
ax.set_yscale("log")
```
```losning
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("regression.csv")
X = data[["Net_Activity"]].values
y = data["Energy"].values
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=100, test_size=100, random_state=0)
scaler = StandardScaler().fit(X_train)
X_train, X_val = scaler.transform(X_train), scaler.transform(X_val)

model = SGDRegressor(learning_rate="constant", eta0=0.01, random_state=0)
train_losses, val_losses = [], []
for epoch in range(30):
    model.partial_fit(X_train, y_train)
    train_losses.append(mean_squared_error(y_train, model.predict(X_train)))
    val_losses.append(mean_squared_error(y_val, model.predict(X_val)))

fig, ax = plt.subplots()
ax.plot(train_losses, label="train")
ax.plot(val_losses, label="validering")
ax.set_yscale("log")
ax.legend()
```
```test
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
assert len(train_losses) == 30 and len(val_losses) == 30, f"Begge listene skal ha 30 verdier (én per epoke), men har {len(train_losses)} og {len(val_losses)}."
_m = SGDRegressor(learning_rate="constant", eta0=0.01, random_state=0)
_tr, _va = [], []
for _e in range(30):
    _m.partial_fit(X_train, y_train)
    _tr.append(mean_squared_error(y_train, _m.predict(X_train)))
    _va.append(mean_squared_error(y_val, _m.predict(X_val)))
assert np.allclose(train_losses, _tr, rtol=1e-6), "train_losses stemmer ikke. Kall model.partial_fit(X_train, y_train) én gang per epoke (ikke fit) og mål MSE på X_train."
assert np.allclose(val_losses, _va, rtol=1e-6), "val_losses stemmer ikke. Mål MSE på X_val og y_val etter hver epoke."
assert len(ax.lines) == 2, f"Plott både train_losses og val_losses på ax (to linjer), men ax har {len(ax.lines)}."
```
:::
