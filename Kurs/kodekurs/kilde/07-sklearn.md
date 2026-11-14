=== DEL 7 | sklearn | scikit-learn: arbeidsflyten
scikit-learn gir deg ferdige modeller og forbehandlingssteg som alle bruker det samme mønsteret: `fit`, `predict` og `transform`. Her lærer du mønsteret, og hvordan du skalerer, koder, validerer og velger hyperparametre uten å lekke informasjon fra testdataene.

--- LEKSJON 7.1 | monsteret | fit, predict og transform
I del 5 skrev du din egen `LinearRegression`-klasse med metodene `fit` og `predict`. scikit-learn bruker **nøyaktig det samme mønsteret** for alle modellene sine. Lærer du mønsteret én gang, kan du bruke hundrevis av modeller.

Mønsteret har tre steg:

1. **Lag objektet** og velg hyperparametre: `model = LogisticRegression()`
2. **Tren** på treningsdata: `model.fit(X_train, y_train)`
3. **Bruk** det som er lært: `model.predict(X_test)`

## Estimatorer: modeller som predikerer
Vi bruker vikingdataene fra prosjekt 2 med noen få numeriske features:

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("problem2.csv")
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["age"] = df["age"].fillna(df["age"].median())
X = df[["sex", "age", "family_aboard", "prior_voyages", "provisions_kg"]]
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)   # 1. lag modellen
model.fit(X_train, y_train)                 # 2. tren: lærer vekter og bias
y_pred = model.predict(X_test)              # 3. prediker klasser
print(y_pred[:10])
print("Accuracy:", model.score(X_test, y_test))
```

`max_iter=1000` er en **hyperparameter**: noe du bestemmer før treningen. Her gir den optimeringen nok iterasjoner til å konvergere.

## predict_proba og lærte attributter
En klassifikator kan også gi sannsynligheter. `predict_proba` gir én kolonne per klasse, og hver rad summerer til 1. Det modellen har lært, ligger i attributter som slutter på understrek:

```python
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
print("Klasser:", model.classes_)
print(proba[:3].round(3))
print("P(overlever) for de tre første:", proba[:3, 1].round(3))
print("Vekter w:", model.coef_[0].round(3).tolist())
print("Bias b:", model.intercept_[0].round(3))
```

`coef_` og `intercept_` er $w$ og $b$ fra del 5. Du finner dem også i variabelpanelet: klikk på `model`, så står alt den lærte under «Lært under fit» (`classes_`, `coef_`, `intercept_` …). `predict` gir klasse 1 når `proba[:, 1]` er over 0.5. Legg også merke til at `fit` returnerer modellen selv, så du kan skrive `LogisticRegression().fit(X, y)` på én linje.

| Metode | Hva den gjør |
|---|---|
| `fit(X, y)` | lærer parametrene fra treningsdata |
| `predict(X)` | klasse (klassifisering) eller tall (regresjon) for hver rad |
| `predict_proba(X)` | sannsynlighet for hver klasse (bare klassifikatorer) |
| `score(X, y)` | accuracy for klassifikatorer, $R^2$ for regresjonsmodeller |

## Det samme for regresjon
`LinearRegression` fra scikit-learn løser prosjekt 1, problem 1 på tre linjer. Den bruker en eksakt formel i stedet for gradient descent, men mønsteret er det samme:

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("regression.csv")
X = data[["Net_Activity"]]   # 2D, selv med én feature
y = data["Energy"]
reg = LinearRegression().fit(X, y)
print(f"Energy = {reg.coef_[0]:.3f} * Net_Activity + {reg.intercept_:.3f}")
print("R^2:", reg.score(X, y))
```

## Transformere: forbehandling
En **transformer** endrer dataene i stedet for å predikere. Den har `fit` og `transform`. `StandardScaler` er den vanligste:

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

X_train = np.array([[1.0, 100.0],
                    [2.0, 200.0],
                    [3.0, 600.0]])
print("Før:")
display(X_train)
scaler = StandardScaler()
scaler.fit(X_train)                  # lærer snitt og standardavvik per kolonne
print("mean_:", scaler.mean_)
print("scale_:", scaler.scale_)
print("Etter transform:")
display(scaler.transform(X_train))   # bruker dem: (x - snitt) / std
print(scaler.fit_transform(X_train)) # fit + transform i ett steg
```

Før skaleringen er den andre kolonnen hundre ganger større enn den første. Etterpå er begge kolonnene på samme skala, med snitt 0. `scaler` dukker også opp i variabelpanelet med `mean_` og `scale_` under «Lært under fit».

::: husk fit betyr to forskjellige ting
- `fit` på en **scaler** (eller annen transformer) **regner ut statistikk** fra dataene, for eksempel snitt og standardavvik. Den trenger ingen `y`.
- `fit` på en **modell** **trener** den: den finner vekter som gjør at prediksjonene passer med `y`.

I begge tilfeller er det `fit` som lærer noe fra dataene. Derfor skal `fit` bare se treningsdataene.
:::

Bruker du et objekt før det er tilpasset, får du en tydelig feilmelding:

```python feil
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.predict([[1, 25, 0, 2, 10.0]])
```

::: oppgave Energiformelen med scikit-learn
Tren en `LinearRegression` på `regression.csv` med `Net_Activity` som eneste feature og `Energy` som target. Lagre:

- `reg`: den trente modellen
- `a`: stigningstallet (et tall, ikke et array)
- `b`: konstantleddet
- `r2`: $R^2$ på de samme dataene (bruk `score`)
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
from sklearn.linear_model import LinearRegression
assert isinstance(reg, LinearRegression), "reg skal være en LinearRegression-modell."
assert hasattr(reg, "coef_"), "reg er ikke trent ennå. Kall reg.fit(X, y)."
assert abs(float(a) - 3.00869655) < 1e-4, f"a skal være stigningstallet reg.coef_[0] (ca. 3.009), men er {a}"
assert abs(float(b) - 4.8917048) < 1e-3, f"b skal være reg.intercept_ (ca. 4.892), men er {b}"
assert abs(float(r2) - 0.98683800) < 1e-4, f"r2 skal være reg.score(X, y) (ca. 0.987), men er {r2}"
```
:::

::: quiz
En klassifikator er trent på data med tre klasser. Hva er shape til `proba`?
```kode
proba = model.predict_proba(X_test)   # X_test har 10 rader
```
- [ ] `(10,)`
- [ ] `(3,)`
- [x] `(10, 3)`
- [ ] `(3, 10)`
? `predict_proba` gir én rad per datapunkt og én kolonne per klasse. Hver rad summerer til 1.
:::

::: quiz
Hva gjør `scaler.fit(X_train)` når `scaler = StandardScaler()`?
- [ ] Trener vekter slik at scaleren kan predikere `y`
- [x] Regner ut og lagrer snitt og standardavvik for hver kolonne i `X_train`
- [ ] Skalerer `X_train` og returnerer de skalerte dataene
- [ ] Ingenting, all regningen skjer først i `transform`
? `fit` på en transformer lærer statistikken (`mean_` og `scale_`). Selve skaleringen skjer i `transform`, og `fit_transform` gjør begge deler.
:::

::: oppsummering
- Alle modeller: `model = Modell(hyperparametre)`, `model.fit(X_train, y_train)`, `model.predict(X_test)`.
- `predict_proba` gir sannsynlighet per klasse, `score` gir accuracy (klassifisering) eller $R^2$ (regresjon).
- Lærte verdier slutter på understrek: `coef_`, `intercept_`, `mean_`, `classes_`.
- Transformere har `fit` (lær statistikk), `transform` (bruk den) og `fit_transform` (begge).
- `fit` på en scaler regner ut statistikk. `fit` på en modell trener den.
:::

--- LEKSJON 7.2 | skalering | Skalering uten datalekkasje
Features på helt ulike skalaer gjør gradient descent tregt og ustabilt (del 5.9), og metoder som bruker avstander (k-means, k-nærmeste naboer) lar den største featuren dominere. Løsningen er å skalere. Beslutningstrær trenger ikke skalering, fordi de bare sammenligner én feature med en terskel om gangen.

De to vanligste skalererne er:

- **StandardScaler**: $z = \dfrac{x - \mu}{\sigma}$. Gir snitt 0 og standardavvik 1. Verdiene havner **ikke** i et fast intervall.
- **MinMaxScaler**: $x' = \dfrac{x - \min}{\max - \min}$. Gir verdier mellom 0 og 1 for dataene den ble tilpasset på.

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("problem2.csv")
X = df[["provisions_kg", "prior_voyages", "family_aboard"]]
std = StandardScaler().fit_transform(X)
mm = MinMaxScaler().fit_transform(X)
print("Før:      snitt", X.mean().round(2).values, " maks", X.max().values)
print("Standard: snitt", std.mean(axis=0).round(2), " std", std.std(axis=0).round(2), " maks", std.max(axis=0).round(2))
print("MinMax:   min", mm.min(axis=0), " maks", mm.max(axis=0))

print("Før (fem første rader):")
display(X.head())
print("Etter StandardScaler:")
display(std[:5].round(2))
print("Etter MinMaxScaler:")
display(mm[:5].round(2))
```

Sammenlign de tre tabellene rad for rad: samme vikinger, men nye tall. Legg også merke til at resultatet av `fit_transform` er et numpy-array uten kolonnenavn. Se på `provisions_kg` etter StandardScaler: maks er over 8, langt over 1. Ekstreme verdier (uteliggere) er et problem for MinMaxScaler: én viking med 320 kg proviant gjør at nesten alle de andre havner nær 0.

## Riktig mønster: fit på train, transform på begge
Scaleren skal lære snitt og standardavvik **bare fra treningsdataene**. Deretter bruker du de samme tallene på testdataene:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("problem2.csv")
X = df[["provisions_kg", "prior_voyages", "family_aboard"]]
X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)                    # KUN treningsdata
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)    # samme snitt og std som train

print("mean_: ", scaler.mean_.round(3))
print("scale_:", scaler.scale_.round(3))
manuelt = (X_test - scaler.mean_) / scaler.scale_
print("Samme som for hånd:", np.allclose(manuelt, X_test_s))
print("Snitt av X_test_s:", X_test_s.mean(axis=0).round(2))
```

Klikk på `scaler` i variabelpanelet for å se de lærte tallene `mean_` og `scale_`, og på `X_test_s` for å se de skalerte testdataene. Snittet av de skalerte testdataene er ikke nøyaktig 0. Det er helt riktig: testdataene skal skaleres med treningsdataenes tall, ikke sine egne.

`MinMaxScaler` følger samme mønster. De lærte tallene heter `data_min_` og `data_max_`, og testverdier kan havne utenfor [0, 1]:

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

alder_train = np.array([[20.0], [35.0], [60.0]])
alder_test = np.array([[40.0], [70.0]])
mm = MinMaxScaler().fit(alder_train)
print("data_min_:", mm.data_min_, " data_max_:", mm.data_max_)
print(mm.transform(alder_test).ravel())   # (x - 20) / (60 - 20)
```

## Hva er datalekkasje?
**Datalekkasje** betyr at informasjon fra testdataene påvirker treningen. Da blir testresultatet for optimistisk, fordi testdataene ikke lenger er «usette». Tilpasser du scaleren på alle dataene, har snittet og standardavviket sett testdataene:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("problem2.csv")
X = df[["provisions_kg", "prior_voyages", "family_aboard"]]
X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)

scaler_alle = StandardScaler().fit(X)          # FEIL: har sett testdataene
scaler_train = StandardScaler().fit(X_train)   # RIKTIG
print("Snitt med lekkasje:", scaler_alle.mean_.round(3))
print("Snitt fra train:   ", scaler_train.mean_.round(3))

# Enda verre: egen scaler på testdataene
ny_viking = X_test.iloc[[0]]
print("Det nye punktet:")
display(ny_viking)
print("fit_transform på ett nytt punkt:", StandardScaler().fit_transform(ny_viking))
print("Riktig transform:              ", scaler_train.transform(ny_viking).round(3))
```

Den siste linjen viser hvorfor du aldri skal tilpasse en ny scaler på testdataene. Når modellen tas i bruk, kommer nye datapunkter ett og ett. En scaler tilpasset på ett punkt gjør det alltid om til bare nuller, og modellen får se noe helt annet enn den ble trent på.

::: eksamen Eksamen 2025
Eksamen 2025 hadde et kodelesingsspørsmål om nøyaktig dette: gitt `scaler = StandardScaler()`, `scaler.fit(X_train)` og `X_train_scaled = scaler.transform(X_train)`, hvilken linje bruker samme skalering på testdataene? Se quizen under.
:::

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

::: quiz
Du har trent en modell med koden under og vil nå evaluere den på `X_test`. Hvilken kodelinje bruker samme skalering på testdataene?
```kode
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
```
- [ ] `X_test_scaled = StandardScaler().fit_transform(X_test)`
- [ ] `X_test_scaled = X_test / X_train.mean(axis=0)`
- [x] `X_test_scaled = scaler.transform(X_test)`
- [ ] `X_test_scaled = scaler.fit(X_test)`
? `scaler.transform(X_test)` bruker snittet og standardavviket fra treningsdataene. Det første alternativet tilpasser en ny scaler på testdataene (annen skalering, og lekkasje). Det andre deler bare på snittet og trekker ikke fra noe. Det fjerde tilpasser scaleren på nytt og returnerer selve scaler-objektet, ikke data.
:::

::: quiz
Featuren *alder* har snitt 40 og standardavvik 8 i treningsdataene, og snitt 45 og standardavvik 10 i testdataene. Du kjører `scaler = StandardScaler().fit(X_train)` og `scaler.transform(X_test)`. Hvilken verdi får en testperson på 52 år?
- [x] 1.5
- [ ] 0.7
- [ ] 0.1875
- [ ] 1.3
? Scaleren bruker treningsdataenes tall: (52 − 40) / 8 = 1.5. 0.7 er (52 − 45) / 10 med testdataenes tall, 0.1875 er 12 / 64 (delt på variansen), og 1.3 er 52 / 40.
:::

::: quiz
`MinMaxScaler` tilpasses på treningsdata der minste verdi er 10 og største er 50. Hva gir `scaler.transform` for en testverdi på 60?
- [ ] 1.0
- [x] 1.25
- [ ] 1.2
- [ ] 0.83
? (60 − 10) / (50 − 10) = 50 / 40 = 1.25. MinMaxScaler klipper ikke, så testverdier kan havne utenfor [0, 1].
:::

::: oppsummering
- **StandardScaler**: $(x - \mu)/\sigma$, snitt 0 og std 1, ikke begrenset til [0, 1]. Lærer `mean_` og `scale_`.
- **MinMaxScaler**: $(x - \min)/(\max - \min)$, [0, 1] på treningsdataene. Lærer `data_min_` og `data_max_`. Følsom for uteliggere.
- `scaler.fit(X_train)`, deretter `scaler.transform(X_train)` og `scaler.transform(X_test)`.
- **Aldri** `fit` på testdata eller på hele datasettet før splitting: det er datalekkasje.
- Trær trenger ikke skalering. Gradient descent, nevrale nett og avstandsbaserte metoder gjør det.
:::

--- LEKSJON 7.3 | encoding | LabelEncoder og OneHotEncoder
I leksjon 3.8 kodet du kategorier med `map` og `pd.get_dummies`. scikit-learn har transformere for det samme. Fordelen er at de **husker** kodingen fra `fit`, slik at train og test garantert kodes likt.

## LabelEncoder
`LabelEncoder` gir hver kategori et heltall, **sortert alfabetisk**. Forelesningsnotatene bruker den slik:

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("problem2.csv")
le = LabelEncoder()
df["sex_kode"] = le.fit_transform(df["sex"])
print("classes_:", le.classes_)          # indeks i lista = koden
display(df[["sex", "sex_kode"]].head())  # tekst og kode side om side
print(le.inverse_transform([0, 1, 1]))   # fra kode tilbake til tekst
```

`Female` blir 0 og `Male` blir 1 fordi F kommer før M. `inverse_transform` går motsatt vei.

## Problemet: falsk rekkefølge
For to kategorier er LabelEncoder helt greit. For flere kategorier lager den en rekkefølge som ikke betyr noe:

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("problem2.csv")
le = LabelEncoder()
df["role_kode"] = le.fit_transform(df["role"])
print(dict(zip(le.classes_, range(len(le.classes_)))))
print(df.groupby("role_kode")["survived"].mean().round(3))
```

Alfabetisk blir Hirdman 0, Jarl 1, Karl 2 og Thrall 3. Overlevelsesraten går da opp, så ned, så ned igjen. En lineær modell (eller et nevralt nett) får **én** vekt for featuren og antar at forskjellen mellom kode 0 og 1 er like stor som mellom 2 og 3. For nominelle features som `ship` finnes det ingen rekkefølge i det hele tatt.

For **beslutningstrær** er label encoding ofte greit nok: treet kan splitte flere ganger på samme feature og plukke ut grupper av koder. Én enkelt splitt kan likevel bare skille sammenhengende koder (for eksempel {0, 1} fra {2, 3}). LabelEncoder er egentlig laget for **target** `y`, for eksempel å gjøre klassenavn om til 0, 1, 2.

## OneHotEncoder
`OneHotEncoder` gjør det samme som `pd.get_dummies`, men som en transformer med `fit` og `transform`:

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("problem2.csv")
train, test = df.iloc[:400], df.iloc[400:]

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.fit(train[["ship", "destination"]])
print(ohe.categories_)
print(ohe.get_feature_names_out())
X_test_oh = ohe.transform(test[["ship", "destination"]])
print(X_test_oh.shape)
print("Før: to tekstkolonner")
display(test[["ship", "destination"]].head(3))
print("Etter: én 0/1-kolonne per kategori")
display(pd.DataFrame(X_test_oh[:3], columns=ohe.get_feature_names_out()))
```

`transform` gir et numpy-array uten kolonnenavn. Med `pd.DataFrame(..., columns=ohe.get_feature_names_out())` får du navnene tilbake, så du ser hvilken kolonne som betyr hva. Hver rad har nøyaktig én ener for `ship` og én for `destination`.

- `sparse_output=False` gir et vanlig numpy-array (standard er en «sparse» matrise som sparer minne).
- `handle_unknown="ignore"` gjør at en kategori som ikke fantes i treningsdataene, får bare nuller i stedet for å krasje.
- `drop="first"` tilsvarer `drop_first=True` i `get_dummies`.

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("problem2.csv")
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[["ship"]])
ny = pd.DataFrame({"ship": ["Bison", "Nytt skip"]})
print(ohe.get_feature_names_out())
print(ohe.transform(ny))
display(pd.DataFrame(ohe.transform(ny), columns=ohe.get_feature_names_out(), index=ny["ship"]))
```

Raden for `Nytt skip` har bare nuller.

| | `pd.get_dummies` | `OneHotEncoder` |
|---|---|---|
| Husker kategoriene | nei, ser bare dataene du gir den | ja, fra `fit` |
| Ukjent kategori i test | gir andre kolonner | nuller med `handle_unknown="ignore"` |
| Passer i en Pipeline | nei | ja (leksjon 7.7) |
| Best til | rask utforsking | modeller som skal evalueres og brukes |

## Side om side: LabelEncoder og OneHotEncoder
Her er de samme rollene kodet på begge måter. LabelEncoder gir **én** kolonne med tallene 0 til 3, mens OneHotEncoder gir **fire** kolonner med 0 og 1:

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

roller = pd.DataFrame({"role": ["Thrall", "Jarl", "Karl", "Hirdman", "Karl"]})

print("LabelEncoder: én kolonne, alfabetisk rekkefølge")
kode = LabelEncoder().fit_transform(roller["role"])
display(pd.DataFrame({"role": roller["role"], "kode": kode}))

print("OneHotEncoder: én kolonne per rolle")
ohe = OneHotEncoder(sparse_output=False).fit(roller)
display(pd.DataFrame(ohe.transform(roller), columns=ohe.get_feature_names_out()))
```

`pd.get_dummies(roller, dtype=int)` gir de samme fire kolonnene som OneHotEncoder.

::: oppgave One-hot med fit på train
Lag en `OneHotEncoder` som heter `ohe` med `handle_unknown="ignore"` og `sparse_output=False`. Tilpass den på kolonnene `role` og `ship` i `train`, og lag `X_train_oh` og `X_test_oh`.
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

::: quiz
Hva skriver koden ut?
```kode
from sklearn.preprocessing import LabelEncoder
print(LabelEncoder().fit_transform(["Thrall", "Jarl", "Karl", "Jarl"]))
```
- [ ] `[0 1 2 1]`
- [x] `[2 0 1 0]`
- [ ] `[3 1 2 1]`
- [ ] `[1 0 2 0]`
? LabelEncoder sorterer kategoriene alfabetisk: Jarl = 0, Karl = 1, Thrall = 2. Rekkefølgen i lista spiller ingen rolle for kodene.
:::

::: quiz
Featuren *Region* har fem verdier uten naturlig rekkefølge. Du koder den med `LabelEncoder` til 0 til 4 og bruker den som én feature i logistisk regresjon. Hvilket utsagn er riktig?
- [x] Modellen får én vekt for Region og tolker avstanden mellom kodene som meningsfull
- [ ] Kodingen er uproblematisk hvis kodene skaleres til [0, 1] med MinMaxScaler først
- [ ] Kodingen er tilstrekkelig, siden logistisk regresjon bare krever numeriske features
- [ ] LabelEncoder lager automatisk én kolonne per region
? Med én vekt blir effekten av region 4 nødvendigvis fire ganger effekten av region 1, sammenlignet med region 0. Skalering endrer ikke rekkefølgen. One-hot-koding gir hver region sin egen vekt.
:::

::: oppsummering
- `LabelEncoder`: alfabetiske heltall, `classes_` viser rekkefølgen, `inverse_transform` går tilbake. Laget for target `y`.
- Label encoding av nominelle features gir en **falsk rekkefølge**. Dårlig for lineære modeller og nevrale nett, ofte greit for trær.
- `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`: `fit` på train, `transform` på train og test.
- OneHotEncoder husker kategoriene, så train og test får alltid de samme kolonnene.
:::

--- LEKSJON 7.4 | imputering | Imputering med SimpleImputer
I leksjon 3.5 fylte du manglende alder med `df["age"].fillna(df["age"].median())`. Medianen ble da regnet ut fra **alle** radene, også testdataene. `SimpleImputer` gjør det samme som en transformer, med det kjente mønsteret: `fit` på train, `transform` på begge.

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

df = pd.read_csv("problem2.csv")
train, test = train_test_split(df, test_size=0.2, random_state=0)

imp = SimpleImputer(strategy="median")
imp.fit(train[["age", "provisions_kg"]])
print("Medianer fra train:", imp.statistics_)
X_test_imp = imp.transform(test[["age", "provisions_kg"]])
print("Mangler i test før:", test["age"].isna().sum(), " etter:", np.isnan(X_test_imp).sum())

print("Før: siste rad mangler alder")
display(test[["age", "provisions_kg"]].head(8))
print("Etter transform:")
display(X_test_imp[:8])
```

`statistics_` er det imputeren lærte: én verdi per kolonne (se også `imp` i variabelpanelet). Alle hull i testdataene fylles med treningsdataenes median: den røde cellen før er fylt etterpå.

## Strategier

| `strategy` | Fyller med | Passer for |
|---|---|---|
| `"mean"` | gjennomsnitt (standard) | tall uten store uteliggere |
| `"median"` | median | tall med uteliggere |
| `"most_frequent"` | vanligste verdi | kategorier (tekst) og heltall |
| `"constant"` | `fill_value` | når manglende betyr noe bestemt |

```python
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv("problem2.csv")
klan = SimpleImputer(strategy="most_frequent").fit(df[["clan_origin"]])
print("Vanligste klan:", klan.statistics_)

rustning = SimpleImputer(strategy="constant", fill_value="None")
print(rustning.fit_transform(df[["armor"]])[:5].ravel())
```

`median` og `mean` virker bare på tall. For tekstkolonner må du bruke `most_frequent` eller `constant`.

## Manglende som informasjon
Med `add_indicator=True` får du en ekstra kolonne som er 1 der verdien manglet. Da kan modellen selv lære om «alder ukjent» betyr noe:

```python
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv("problem2.csv")
imp = SimpleImputer(strategy="median", add_indicator=True)
ut = imp.fit_transform(df[["age"]])
print(df["age"].head(8).tolist())
print(ut[:8])
display(pd.DataFrame(ut[:8], columns=imp.get_feature_names_out()))
```

Rad 7 manglet alder. Den har fått medianen i `age` og 1 i den nye kolonnen `missingindicator_age`.

::: oppgave Imputer alder og proviant
Lag en `SimpleImputer` som heter `imp` med medianstrategi. Tilpass den på kolonnene `age` og `provisions_kg` i `train`, og lag `X_train_imp` og `X_test_imp`.
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

::: quiz
Hvilken verdi får de manglende verdiene i `X_test` etter koden under?
```kode
imp = SimpleImputer(strategy="median")
imp.fit(X_train)
X_test_imp = imp.transform(X_test)
```
- [ ] Medianen av hver kolonne i `X_test`
- [x] Medianen av hver kolonne i `X_train`
- [ ] Gjennomsnittet av hver kolonne i `X_train`
- [ ] 0
? Imputeren lærte medianene fra `X_train` i `fit` (de ligger i `imp.statistics_`), og `transform` bruker dem på alle data den får.
:::

::: quiz
Kolonnen `clan_origin` inneholder tekst (Nidaros, Borg, Kaupang) og har noen manglende verdier. Hvilken imputering virker?
- [ ] `SimpleImputer(strategy="median")`
- [ ] `SimpleImputer(strategy="mean")`
- [x] `SimpleImputer(strategy="most_frequent")`
- [ ] Ingen, tekstkolonner kan ikke imputeres
? Median og gjennomsnitt kan bare regnes ut for tall. `most_frequent` fyller med den vanligste kategorien, og `constant` med en verdi du velger.
:::

::: oppsummering
- `SimpleImputer(strategy=...)` fyller hull: `"mean"`, `"median"`, `"most_frequent"` eller `"constant"` med `fill_value`.
- Samme mønster som scalere: `fit` på train, `transform` på train og test. Lærte verdier ligger i `statistics_`.
- `add_indicator=True` legger til en 0/1-kolonne som sier om verdien manglet.
- Bruk `most_frequent` eller `constant` for tekstkolonner.
:::

--- LEKSJON 7.5 | kryssvalidering | Validering og kryssvalidering
Testdataene skal bare brukes én gang, helt til slutt. Men underveis må du sammenligne modeller og velge hyperparametre. Til det trenger du **valideringsdata**. Forelesningsnotatene deler dataene i tre:

- **Treningsdata**: brukes av `fit` til å lære parametrene.
- **Valideringsdata**: brukes til å sammenligne modeller og velge hyperparametre.
- **Testdata**: brukes bare til å rapportere ytelsen til den endelige modellen.

## Tre deler med train_test_split to ganger

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("problem2.csv")
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["age"] = df["age"].fillna(df["age"].median())
X = df[["sex", "age", "family_aboard", "prior_voyages", "provisions_kg"]]
y = df["survived"]

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=0)
print(len(X_train), len(X_val), len(X_test))
```

Første split tar 20 % til test. Andre split tar 25 % av de resterende 80 %, altså 20 % av alt, til validering. I variabelpanelet ser du formene direkte: `X_train` er 300×5, `X_val` og `X_test` er 100×5, og `y_train`, `y_val` og `y_test` er Series med like mange rader som X-en sin.

## Ett valideringssett er usikkert
Med bare 100 rader i valideringssettet avhenger resultatet mye av hvilke rader som havnet der:

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("problem2.csv")
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
df["age"] = df["age"].fillna(df["age"].median())
X = df[["sex", "age", "family_aboard", "prior_voyages", "provisions_kg"]]
y = df["survived"]
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

for seed in range(5):
    X_tr, X_va, y_tr, y_va = train_test_split(X_temp, y_temp, test_size=0.25, random_state=seed)
    model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    print(f"seed {seed}: val accuracy {model.score(X_va, y_va):.3f}")
```

## Kryssvalidering
**k-fold kryssvalidering** løser dette. Treningsdataene deles i k like store deler (folds). Modellen trenes k ganger, hver gang på k − 1 folds, og valideres på den siste. Hver rad brukes til validering nøyaktig én gang, og du får k resultater:

```python
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
print(cv_scores.round(3))
print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

model.fit(X_train, y_train)   # cross_val_score trener kopier, ikke model selv
print(f"Test accuracy: {model.score(X_test, y_test):.4f}")
```

Åpne `cv_scores` i variabelpanelet: det er et array med fem tall, ett per fold. Snittet er et mer pålitelig estimat enn ett valideringssett, og standardavviket sier hvor mye resultatet varierer. Kryssvalideringen kjøres bare på treningsdataene. Testdataene holdes utenfor.

::: advarsel
`cross_val_score` trener **kopier** av modellen. Objektet `model` er like utrent etterpå. Du må kalle `model.fit(X_train, y_train)` før du kan predikere på testdataene.
:::

::: prosjekt Prosjekt 2, problem 2
Treningscellen skal ende med `print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")` og `print(f"Test accuracy: {a:.4f}")`. Spørsmålet «What is the purpose of cross validation?» besvares av avsnittet over.
:::

## KFold og StratifiedKFold
`KFold` deler radene i rekkefølge (eller tilfeldig med `shuffle=True`). `StratifiedKFold` sørger for at hver fold har samme klassefordeling som hele datasettet:

```python
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

X = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

for train_idx, val_idx in KFold(n_splits=5).split(X):
    print("KFold      val:", val_idx, "klasser:", y[val_idx])
for train_idx, val_idx in StratifiedKFold(n_splits=5).split(X, y):
    print("Stratified val:", val_idx, "klasser:", y[val_idx])
```

Med `KFold` uten stokking består noen valideringsfolds bare av én klasse. For klassifikatorer bruker `cross_val_score(..., cv=5)` automatisk `StratifiedKFold`. Vil du stokke, gir du et objekt: `cv=KFold(n_splits=5, shuffle=True, random_state=0)`.

## cross_validate: flere metrikker
`cross_validate` gir en dict med flere metrikker, og med `return_train_score=True` også resultatet på treningsfoldene. Stor avstand mellom train og validering tyder på overtilpasning:

```python
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
print(sorted(res.keys()))
print("val accuracy:  ", res["test_accuracy"].mean().round(3))
print("train accuracy:", res["train_accuracy"].mean().round(3))
print("val F1:        ", res["test_f1"].mean().round(3))
display(pd.DataFrame(res).round(3))   # én rad per fold
```

Nøklene heter `test_...`, men det er valideringsfoldene det er snakk om. Gjør du dicten om til en DataFrame, får du én rad per fold og én kolonne per metrikk.

::: oppgave Kryssvalider modellen
Bruk `cross_val_score` med 5 folds på `X_train` og `y_train` med modellen `LogisticRegression(max_iter=1000)`. Lagre resultatet i `cv_scores` og snittet i `cv_mean`.
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

cv_scores = ...
cv_mean = ...
print(f"CV Accuracy: {cv_mean:.3f}")
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

cv_scores = cross_val_score(LogisticRegression(max_iter=1000), X_train, y_train, cv=5)
cv_mean = cv_scores.mean()
print(f"CV Accuracy: {cv_mean:.3f}")
```
```test
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
_fasit = cross_val_score(LogisticRegression(max_iter=1000), X_train, y_train, cv=5)
assert np.shape(cv_scores) == (5,), f"cv_scores skal ha 5 verdier (cv=5), men har shape {np.shape(cv_scores)}"
assert np.allclose(cv_scores, _fasit), "cv_scores stemmer ikke. Bruker du X_train og y_train (ikke hele X eller testdataene)?"
assert abs(cv_mean - _fasit.mean()) < 1e-9, "cv_mean skal være cv_scores.mean()."
```
:::

::: quiz
Hvor mange ganger trenes en modell når denne linjen kjøres?
```kode
scores = cross_val_score(model, X_train, y_train, cv=5)
```
- [ ] 1
- [ ] 4
- [x] 5
- [ ] 6
? Én gang per fold: fem ganger, hver gang på fire femdeler av dataene. `cross_val_score` trener ikke en endelig modell i tillegg, så `model` er fortsatt utrent etterpå.
:::

::: quiz
Du bruker `KFold(n_splits=4)` på et treningssett med 100 rader. Hvor mange rader trenes og valideres modellen på i hver runde?
- [ ] Trener på 25, validerer på 75
- [x] Trener på 75, validerer på 25
- [ ] Trener på 100, validerer på 25
- [ ] Trener på 80, validerer på 20
? Dataene deles i fire folds med 25 rader. Hver runde valideres det på én fold og trenes på de tre andre (75 rader).
:::

::: oppsummering
- Tre deler: train (lære parametre), validering (velge modell og hyperparametre), test (rapportere til slutt).
- `train_test_split` to ganger gir tre deler.
- `cross_val_score(model, X_train, y_train, cv=5)` gir 5 valideringsresultater. Rapporter `mean() ± std()`.
- Modellen trenes én gang per fold, på kopier. Kall `fit` selv før du tester.
- `StratifiedKFold` bevarer klassefordelingen (standard for klassifikatorer). `cross_validate` gir flere metrikker og treningsresultater.
:::

--- LEKSJON 7.6 | hyperparametre | Hyperparametersøk
**Parametre** lærer modellen selv i `fit` (vekter, bias, terskler i et tre). **Hyperparametre** velger du før treningen: `C` i logistisk regresjon, `max_depth` i et tre, graden til et polynom, læringsraten. Hyperparametre velges ved å prøve flere verdier og sammenligne på **valideringsdata**.

I `LogisticRegression` styrer `C` regulariseringen: liten `C` gir sterk regularisering (enklere modell), stor `C` gir svak. Vi bruker brystkreftdatasettet som følger med scikit-learn, der forskjellene er tydelige.

## Søk med en løkke

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=0)
scaler = StandardScaler().fit(X_train)
X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)

resultater = {}
for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    model = LogisticRegression(C=C).fit(X_train_s, y_train)
    resultater[C] = model.score(X_val_s, y_val)
    print(f"C = {C:<6} val accuracy = {resultater[C]:.3f}")

beste_C = max(resultater, key=resultater.get)
print("Beste C:", beste_C)
```

Dicten `resultater` vises som en tabell i variabelpanelet, med én rad per `C`. `max(resultater, key=resultater.get)` gir nøkkelen med størst verdi. Først **etter** at `C` er valgt, evaluerer du den endelige modellen på testdataene, én gang.

::: advarsel Ikke velg på testdataene
Prøver du mange verdier og velger den som er best **på testdataene**, har du brukt testdataene til å tilpasse modellen. Testresultatet blir da for optimistisk, og du har ingen usette data igjen å rapportere på. Forelesningsnotatene: «Testdataene skal ikke brukes til noen som helst justering.»
:::

## GridSearchCV
`GridSearchCV` gjør løkka for deg, med kryssvalidering i stedet for ett valideringssett. Du gir den en modell og en dict med verdier å prøve:

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

grid = GridSearchCV(LogisticRegression(), param_grid={"C": [0.001, 0.01, 0.1, 1, 10, 100]}, cv=5)
grid.fit(X_train_s, y_train)
print("best_params_:", grid.best_params_)
print("best_score_: ", round(grid.best_score_, 3))
print("Test accuracy:", round(grid.score(X_test_s, y_test), 3))

resultater = pd.DataFrame(grid.cv_results_)
display(resultater[["param_C", "mean_test_score", "std_test_score", "rank_test_score"]].round(3))
```

`cv_results_` er en dict med resultatet for hver verdi av `C`. Som DataFrame får du én rad per verdi, og raden med `rank_test_score` lik 1 er den som ble valgt.

- `best_params_`: hyperparametrene med best snitt i kryssvalideringen.
- `best_score_`: dette snittet (validering, ikke test).
- Etter søket trener `GridSearchCV` en ny modell med de beste verdiene på **alle** treningsdataene (`best_estimator_`). `grid.predict` og `grid.score` bruker den.

## Flere hyperparametre
Med flere hyperparametre prøves **alle kombinasjoner**. `cv_results_` inneholder resultatet for hver av dem:

```python
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
tabell = pd.DataFrame(grid.cv_results_)[["param_C", "param_class_weight", "mean_test_score"]]
print(len(tabell), "kombinasjoner")
tabell
```

3 verdier for `C` ganger 2 for `class_weight` gir 6 kombinasjoner, og med `cv=5` blir det 30 treninger. Store rutenett blir fort trege.

::: hvorfor Litt lekkasje er det fortsatt
I eksemplene over ble scaleren tilpasset på hele treningssettet før kryssvalideringen. Da har hver valideringsfold vært med på å bestemme snittet og standardavviket. I neste leksjon løser vi det med en `Pipeline`.
:::

::: oppgave Finn beste C
Bruk `GridSearchCV` med `LogisticRegression()`, verdiene `[0.01, 0.1, 1, 10]` for `C` og `cv=5`. Tilpass søket på `X_train_s` og `y_train`, og lagre:

- `grid`: søket
- `beste_C`: den beste verdien for `C` (et tall)
- `test_acc`: accuracy på testdataene med den beste modellen
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
assert abs(test_acc - _g.score(X_test_s, y_test)) < 1e-9, "test_acc skal være grid.score(X_test_s, y_test)."
```
:::

::: quiz
Hvor mange kombinasjoner prøves, og hvor mange ganger trenes en modell i selve kryssvalideringen?
```kode
grid = GridSearchCV(model,
                    param_grid={"C": [0.01, 0.1, 1],
                                "max_iter": [100, 200, 500, 1000]},
                    cv=5)
grid.fit(X_train, y_train)
```
- [ ] 7 kombinasjoner, 35 treninger
- [x] 12 kombinasjoner, 60 treninger
- [ ] 12 kombinasjoner, 12 treninger
- [ ] 60 kombinasjoner, 300 treninger
? Alle kombinasjoner prøves: 3 · 4 = 12. Hver kombinasjon kryssvalideres med 5 folds: 12 · 5 = 60 treninger. (Til slutt kommer én ekstra trening av den beste modellen på alle treningsdataene.)
:::

::: quiz
Du har train-, validerings- og testdata og vil velge læringsraten til en modell. Hvilken fremgangsmåte er riktig?
- [ ] Velg læringsraten som gir lavest tap på treningsdataene
- [x] Velg læringsraten som gir best resultat på valideringsdataene, og rapporter testresultatet for den
- [ ] Velg læringsraten som gir best resultat på testdataene, og rapporter det resultatet
- [ ] Slå sammen validerings- og testdataene og velg den som er best der
? Treningstapet belønner overtilpasning, og testdataene skal ikke brukes til noen tilpasning. Valideringsdataene (eller kryssvalidering på treningsdataene) er laget for å velge hyperparametre.
:::

::: oppsummering
- **Parametre** læres i `fit`. **Hyperparametre** (`C`, `max_depth`, grad, læringsrate) velger du.
- Velg hyperparametre på valideringsdata eller med kryssvalidering, **aldri** på testdataene.
- `GridSearchCV(model, param_grid={...}, cv=5).fit(X_train, y_train)` prøver alle kombinasjoner.
- `best_params_`, `best_score_` (snitt i kryssvalideringen) og `best_estimator_` (trent på nytt på all treningsdata).
- Antall treninger = antall kombinasjoner · antall folds.
:::

--- LEKSJON 7.7 | pipeline | Pipeline
En **Pipeline** setter flere steg sammen til én modell: for eksempel en scaler etterfulgt av logistisk regresjon. Hele kjeden har `fit`, `predict` og `score`, akkurat som en vanlig modell.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

modell = make_pipeline(StandardScaler(), LogisticRegression())
modell.fit(X_train, y_train)
print("Test accuracy:", round(modell.score(X_test, y_test), 3))
print(modell.named_steps.keys())
print("Snitt lært av scaleren:", modell.named_steps["standardscaler"].mean_[:3].round(2))

for navn, steg in modell.named_steps.items():
    print("Steg:", navn)
    display(steg)   # det steget lærte under fit
```

Hvert steg i pipelinen er et vanlig, trent objekt: scaleren har lært `mean_` og `scale_`, og den logistiske regresjonen har lært `coef_` og `intercept_`.

Dette skjer inni pipelinen:

| Du kaller | Pipelinen gjør |
|---|---|
| `modell.fit(X_train, y_train)` | `scaler.fit_transform(X_train)`, deretter `logreg.fit(...)` på de skalerte dataene |
| `modell.predict(X_test)` | `scaler.transform(X_test)`, deretter `logreg.predict(...)` |

Du kan altså ikke glemme å skalere testdataene, og du kan ikke ved et uhell kalle `fit` på dem. `make_pipeline` gir stegene navn automatisk (klassenavnet med små bokstaver). Vil du velge navnene selv, bruker du `Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])`.

## Hvorfor Pipeline hindrer lekkasje i kryssvalidering
Gir du en pipeline til `cross_val_score`, tilpasses **hele** pipelinen på nytt i hver fold. Scaleren ser da bare de fire treningsfoldene, aldri valideringsfolden.

For en scaler er lekkasjen liten. For steg som bruker `y` i `fit`, kan den bli enorm. `SelectKBest(k=20)` velger de 20 featurene som henger mest sammen med `y`. Under er dataene **ren støy**, så ærlig accuracy er rundt 50 %:

```python
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(1)
X = rng.normal(size=(200, 2000))     # 2000 features med ren støy
y = rng.integers(0, 2, size=200)     # tilfeldige klasser

# FEIL: velg features med ALLE data, kryssvalider etterpå
X_valgt = SelectKBest(k=20).fit_transform(X, y)
print("Med lekkasje: ", cross_val_score(LogisticRegression(), X_valgt, y, cv=5).mean())

# RIKTIG: utvalget gjøres på nytt inni hver fold
pipe = make_pipeline(SelectKBest(k=20), LogisticRegression())
print("Med pipeline: ", cross_val_score(pipe, X, y, cv=5).mean())
```

Uten pipeline ser modellen ut til å klare nesten 80 % på tilfeldige data. Utvalget av features hadde allerede «sett» valideringsfoldene.

## GridSearchCV med pipeline
Hyperparametre i et steg får navnet `stegnavn__parameter`, med **to** understreker:

```python
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
print(grid.best_params_, round(grid.best_score_, 3))
print("Test:", round(grid.score(X_test, y_test), 3))
```

Nå er både skaleringen og valget av `C` gjort uten at noen valideringsfold har lekket inn.

## Hele vikingløpet i én pipeline
Ekte data har både tall og tekst. `make_column_transformer` bruker ulike steg på ulike kolonner. Her blir alt fra del 3 og denne delen én modell:

```python
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
print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
modell.fit(X_train, y_train)
print(f"Test accuracy: {modell.score(X_test, y_test):.4f}")

print("Rådata inn i pipelinen:")
display(X_test.head(3))
print("Etter forbehandlingen (dette er det LogisticRegression ser):")
display(pd.DataFrame(forbehandling.transform(X_test.head(3)),
                     columns=forbehandling.get_feature_names_out()))
```

Før forbehandlingen har `X_test` tekst og røde `NaN`-celler. Etterpå er alt tall: tallkolonnene er fylt inn og skalert (`pipeline-1__...`), og tekstkolonnene er blitt 0/1-kolonner (`pipeline-2__...`). Medianer, snitt, standardavvik og kategorier læres bare fra treningsdelen i hver fold. Med alle features slår modellen klart de fem numeriske featurene vi brukte i leksjon 7.5.

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

::: quiz
Hvilken kode gir et kryssvalideringsresultat uten datalekkasje?
- [ ] `X_s = StandardScaler().fit_transform(X)` og så `cross_val_score(LogisticRegression(), X_s, y, cv=5)`
- [x] `cross_val_score(make_pipeline(StandardScaler(), LogisticRegression()), X, y, cv=5)`
- [ ] `cross_val_score(LogisticRegression(), StandardScaler().fit(X), y, cv=5)`
- [ ] `X_s = StandardScaler().fit_transform(X_test)` og så `cross_val_score(LogisticRegression(), X_s, y_test, cv=5)`
? Med pipeline tilpasses scaleren på nytt i hver fold, bare på treningsfoldene. I det første alternativet har scaleren sett alle foldene. Det tredje sender et scaler-objekt i stedet for data, og det fjerde bruker testdataene til kryssvalidering.
:::

::: quiz
Du lager `pipe = make_pipeline(StandardScaler(), LogisticRegression())` og vil søke over `C` med `GridSearchCV(pipe, param_grid=...)`. Hvilken nøkkel må du bruke i `param_grid`?
- [ ] `"C"`
- [x] `"logisticregression__C"`
- [ ] `"LogisticRegression.C"`
- [ ] `"model__C"`
? `make_pipeline` kaller steget `logisticregression`, og parametre i et steg skrives `stegnavn__parameter` med to understreker. `"model__C"` ville bare virket om du selv hadde kalt steget `"model"` med `Pipeline([...])`.
:::

::: oppsummering
- `make_pipeline(StandardScaler(), LogisticRegression())` lager én modell av flere steg.
- `fit` kjører `fit_transform` på hvert forbehandlingssteg og `fit` på modellen. `predict` kjører `transform` og så `predict`.
- I `cross_val_score` og `GridSearchCV` tilpasses hele pipelinen på nytt i hver fold: **ingen lekkasje**.
- Hyperparametre i et steg: `"stegnavn__parameter"`, for eksempel `"logisticregression__C"`.
- `make_column_transformer` bruker ulike steg på tall- og tekstkolonner.
:::

--- LEKSJON 7.8 | polynom-sgd | Polynomfeatures, SGDRegressor og overtilpasning
Lineær regresjon kan bare tegne rette linjer. Lager du nye features som $x^2$ og $x^3$, kan den samme lineære modellen tilpasse kurver. Forelesningsnotatene kaller dette **feature engineering** og bruker `PolynomialFeatures` fra scikit-learn.

## PolynomialFeatures

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

x = np.array([[2.0], [3.0]])
poly = PolynomialFeatures(degree=3, include_bias=False)
print(poly.fit_transform(x))                         # x, x², x³

X2 = np.array([[2.0, 3.0]])
poly2 = PolynomialFeatures(degree=2, include_bias=False)
print(poly2.fit_transform(X2))
print(poly2.get_feature_names_out(["x0", "x1"]))

print("Før: to features")
display(pd.DataFrame(X2, columns=["x0", "x1"]))
print("Etter: fem features, med navn")
display(pd.DataFrame(poly2.fit_transform(X2), columns=poly2.get_feature_names_out(["x0", "x1"])))
```

Med to features og grad 2 får du $x_0$, $x_1$, $x_0^2$, $x_0 x_1$ og $x_1^2$. Uten `include_bias=False` kommer en ekstra kolonne med bare enere. Modellen er fortsatt lineær regresjon, bare med flere features: $y = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots$

## XOR-problemet på nytt
I prosjekt 1 klarte ikke logistisk regresjon `classification.csv`, fordi klassene er fordelt som XOR. Produktet $x_0 x_1$ er nøkkelfeaturen (leksjon 5.8), og den lager `PolynomialFeatures(2)` automatisk:

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]

lineaer = LogisticRegression().fit(train[["x0", "x1"]], train["y"])
poly = make_pipeline(PolynomialFeatures(degree=2), LogisticRegression()).fit(train[["x0", "x1"]], train["y"])
print("Uten polynomfeatures:", lineaer.score(test[["x0", "x1"]], test["y"]))
print("Med grad 2:          ", poly.score(test[["x0", "x1"]], test["y"]))

# Hva ser LogisticRegression etter PolynomialFeatures?
pf = PolynomialFeatures(degree=2)
Xp = pd.DataFrame(pf.fit_transform(train[["x0", "x1"]]), columns=pf.get_feature_names_out())
Xp["y"] = train["y"].values
display(Xp.head(6))
```

Se på kolonnene `x0 x1` og `y` i tabellen: i de første radene er `y` 1 nøyaktig der produktet er negativt. Kolonnen `1` er bias-kolonnen, siden `include_bias=False` ikke er gitt her.

`PolynomialFeatures` er en transformer, så den følger samme regel som scalere: `fit` på treningsdata, `transform` på testdata. I en pipeline skjer det automatisk.

## Over- og undertilpasning
Hvilken grad skal du velge? For lav grad gir **undertilpasning**: modellen er for enkel og bommer både på train og test. For høy grad gir **overtilpasning**: modellen følger støyen i treningsdataene og bommer på nye data.

```python
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

for grad in [1, 3, 5, 9, 12]:
    modell = make_pipeline(PolynomialFeatures(grad), LinearRegression()).fit(x_train, y_train)
    mse_train = mean_squared_error(y_train, modell.predict(x_train))
    mse_test = mean_squared_error(y_test, modell.predict(x_test))
    print(f"grad {grad:2d}: train MSE {mse_train:.3f}   test MSE {mse_test:.3f}")
```

Train-MSE synker når graden øker, men test-MSE er lavest rundt grad 5 og eksploderer for grad 12. Slik ser det ut:

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(0)
x = rng.uniform(0, 1, 30).reshape(-1, 1)
y = np.sin(2 * np.pi * x.ravel()) + rng.normal(0, 0.3, 30)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
x_plot = np.linspace(0, 1, 200).reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, grad in zip(axes, [1, 5, 12]):
    modell = make_pipeline(PolynomialFeatures(grad), LinearRegression()).fit(x_train, y_train)
    ax.scatter(x_train, y_train, label="train")
    ax.scatter(x_test, y_test, marker="x", label="test")
    ax.plot(x_plot, modell.predict(x_plot), color="black")
    ax.set_ylim(-2, 2)
    ax.set_title(f"grad {grad}")
axes[0].legend()
plt.show()
```

Graden er en **hyperparameter**: den velges med valideringsdata eller kryssvalidering, akkurat som `C` i leksjon 7.6.

## SGDRegressor og tapskurver
`SGDRegressor` er lineær regresjon trent med stokastisk gradient descent, slik du skrev selv i del 5. `partial_fit` kjører **én epoke** (én runde gjennom dataene) og fortsetter fra vektene den har. Da kan du måle tapet etter hver epoke. Koden er hentet fra forelesningsnotatene:

```python
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
scaler = StandardScaler().fit(X_train)            # SGD trenger skalerte features
X_train, X_val = scaler.transform(X_train), scaler.transform(X_val)

model = SGDRegressor(learning_rate="constant", eta0=0.01, random_state=0)
train_losses, val_losses = [], []
n_epochs = 30
for epoch in range(n_epochs):
    model.partial_fit(X_train, y_train)
    # Monitor training loss
    train_losses.append(mean_squared_error(y_train, model.predict(X_train)))
    # Monitor validation loss
    val_losses.append(mean_squared_error(y_val, model.predict(X_val)))

plt.plot(train_losses, label="train")
plt.plot(val_losses, label="validering")
plt.yscale("log")
plt.xlabel("epoke")
plt.ylabel("MSE")
plt.legend()
plt.show()
print("Vekt og bias:", model.coef_, model.intercept_)
```

`learning_rate="constant"` og `eta0=0.01` betyr fast læringsrate 0.01. Vi bruker bare 100 treningsrader så kurvene får noen epoker på seg. `fit` ville i stedet startet på nytt og kjørt mange epoker til den konvergerte.

Slik leser du tapskurvene (figur 14 i notatene):

| Kurvene | Tolkning |
|---|---|
| Begge synker og flater ut på omtrent samme nivå | god tilpasning |
| Train fortsetter å synke, validering flater ut eller stiger | overtilpasning (for kompleks modell) |
| Begge flater ut på et høyt nivå | undertilpasning (for enkel modell) |

::: oppgave Velg grad med valideringsdata
Prøv gradene 1 til 10 med `make_pipeline(PolynomialFeatures(grad), LinearRegression())`. Tren på `x_train`, mål MSE på `x_val`, og lagre resultatene i dicten `val_mse` (nøkkel: grad, verdi: validerings-MSE). Lagre graden med lavest validerings-MSE i `beste_grad`.
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

::: quiz
Er graden til polynomet vi lager features til en modellparameter, en hyperparameter eller en feature?
- [ ] En modellparameter, fordi den lærer seg i `fit`
- [x] En hyperparameter, fordi vi velger den før treningen
- [ ] En feature, fordi den lager nye kolonner
- [ ] Ingen av delene, den har ingen effekt på modellen
? Graden bestemmes før `fit` og påvirker hvor kompleks modellen blir. Den velges med valideringsdata, som andre hyperparametre. Vektene $\beta$ er modellparametrene.
:::

::: quiz
Når du øker graden fra 5 til 12, synker trenings-MSE fra 0.016 til 0.015, mens validerings-MSE øker fra 0.09 til 4.5. Hva er den beste forklaringen?
- [ ] Undertilpasning: modellen er for enkel
- [x] Overtilpasning: modellen følger støyen i treningsdataene
- [ ] Læringsraten er for høy
- [ ] Valideringsdataene er feil skalert
? Lavt treningstap og høyt valideringstap er kjennetegnet på overtilpasning. Polynomet av grad 12 svinger voldsomt mellom treningspunktene.
:::

::: quiz
Hva skjer når du kaller `model.partial_fit(X_train, y_train)` på en `SGDRegressor` i en løkke?
- [ ] Modellen starter på nytt med tilfeldige vekter hver gang
- [x] Hvert kall kjører én epoke med SGD og fortsetter fra vektene modellen allerede har
- [ ] Hvert kall trener til modellen har konvergert
- [ ] Bare det første kallet trener, de neste gjør ingenting
? `partial_fit` gjør én runde gjennom dataene og beholder vektene mellom kallene. Derfor kan du måle tapet etter hver epoke. `fit` starter på nytt hver gang.
:::

::: oppsummering
- `PolynomialFeatures(degree=d)` lager $x, x^2, \dots, x^d$ og alle produkter. `include_bias=False` dropper enerkolonnen.
- Lineær regresjon med polynomfeatures er fortsatt en lineær modell, men kan tilpasse kurver (og løse XOR med $x_0 x_1$).
- Lav grad: undertilpasning. Høy grad: lav trenings-MSE, høy test-MSE, altså overtilpasning. Graden er en hyperparameter.
- `SGDRegressor(learning_rate="constant", eta0=0.01)` med `partial_fit` i en løkke gir én epoke per kall, og du kan logge train- og valideringstap.
- Begge kurver lave og flate: bra. Train synker, validering stiger: overtilpasning. Begge høye: undertilpasning.
:::
