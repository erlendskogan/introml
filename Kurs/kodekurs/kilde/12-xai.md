=== DEL 12 | xai | Forklarbar KI (prosjekt 4)
Koden bak forklaringene i prosjekt 4: koeffisienter og permutation importance, Shapley-verdier regnet for hånd, shap-biblioteket med Explainer, TreeExplainer og DeepExplainer, LIME som lokal surrogatmodell, og konseptdeteksjon med CAV og TCAV.

--- LEKSJON 12.1 | koeffisienter | Koeffisienter og permutation importance
En modell er trent og gir gode prediksjoner. Men **hvilke features bruker den**, og hvor mye? Det er spørsmålet forklarbar KI (XAI) prøver å svare på. Vi starter med de enkleste verktøyene: koeffisientene til en lineær modell, og permutation importance.

Prosjekt 4 bruker California housing, som må lastes ned. Vi lager i stedet et lite datasett med samme type features: inntekt i området, husets alder, soverom per bolig, befolkning og avstand til sentrum. Funksjonen `lag_boligdata` kommer igjen i resten av delen. Prisen er i enheter av 100 000 USD, som i prosjektet.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def lag_boligdata(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "Inntekt": rng.gamma(4, 1.0, n),         # medianinntekt i området (10 000 USD)
        "Alder": rng.integers(1, 53, n),         # husets alder i år
        "Soverom": rng.normal(1.05, 0.05, n),    # snitt soverom per bolig
        "Befolkning": rng.gamma(2, 700, n),      # innbyggere i området
        "Avstand": rng.uniform(0, 30, n),        # km til sentrum
    })
    y = (0.5 * np.minimum(X["Inntekt"], 7) + 0.01 * X["Alder"] + 2.0 * X["Soverom"]
         + 0.00003 * X["Befolkning"] + 1.5 * np.exp(-((X["Avstand"] - 5) / 4) ** 2)
         + rng.normal(0, 0.3, n))
    return X, y

X, y = lag_boligdata()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
linear_model = LinearRegression().fit(X_train, y_train)
print(f"R2 på testdata: {linear_model.score(X_test, y_test):.3f}")

plt.bar(X.columns, linear_model.coef_)
plt.ylabel("Koeffisient")
plt.title("Feature importance by coefficients?")
plt.show()
pd.Series(linear_model.coef_, index=X.columns).round(5)
```

Ifølge plottet er **Soverom** klart viktigst (koeffisient 2.5), og Befolkning ser helt ubrukelig ut. Stemmer det?

## Koeffisienten avhenger av skalaen
En koeffisient er endringen i prediksjonen når featuren øker med **én enhet**. Én ekstra soverom per bolig er enormt (nesten alle verdier ligger mellom 1.0 og 1.1), mens én ekstra innbygger er ingenting. For å sammenligne må vi ta hensyn til hvor mye featuren faktisk varierer: $|\beta_i| \cdot \text{std}(x_i)$. Det samme får du ved å **standardisere** featurene før du trener:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def lag_boligdata(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "Inntekt": rng.gamma(4, 1.0, n), "Alder": rng.integers(1, 53, n),
        "Soverom": rng.normal(1.05, 0.05, n), "Befolkning": rng.gamma(2, 700, n),
        "Avstand": rng.uniform(0, 30, n)})
    y = (0.5 * np.minimum(X["Inntekt"], 7) + 0.01 * X["Alder"] + 2.0 * X["Soverom"]
         + 0.00003 * X["Befolkning"] + 1.5 * np.exp(-((X["Avstand"] - 5) / 4) ** 2)
         + rng.normal(0, 0.3, n))
    return X, y

X, y = lag_boligdata()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
linear_model = LinearRegression().fit(X_train, y_train)

# Samme modell, men på standardiserte features (snitt 0, std 1)
scaled_model = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)

pd.DataFrame({
    "coef": linear_model.coef_,
    "std": X_train.std(),
    "|coef| * std": np.abs(linear_model.coef_) * X_train.std(),
    "standardisert coef": scaled_model[-1].coef_,
}, index=X.columns).round(4)
```

Nå er bildet helt annerledes: **Inntekt** er viktigst, så **Avstand**, mens Soverom havner nesten nederst. Den standardiserte koeffisienten er (nesten) lik coef · std. Den lille forskjellen kommer av at `StandardScaler` deler på std med $n$ i nevneren, mens pandas bruker $n - 1$.

::: prosjekt Prosjekt 4, problem 1
Spørsmålet «The coefficient plot and SHAP disagree on which features are important. Why?» har akkurat dette svaret. I California housing har AveBedrms den største koeffisienten (0.59), men nesten alle verdier ligger nær 1, så den betyr lite. Population har en bitteliten koeffisient, men varierer over tusenvis. Prosjektets løsning lager samme tabell med kolonnene `coefficient`, `std of feature` og `|coef| * std` og sammenligner med SHAP (leksjon 12.3).
:::

## Permutation importance
Koeffisienter finnes bare for lineære modeller. **Permutation importance** virker for alle modeller: stokk om verdiene i én kolonne i testdataene, og mål hvor mye scoren faller. Bruker modellen featuren mye, faller scoren mye. Stokkingen bevarer fordelingen til featuren, men bryter koblingen til target.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

def lag_boligdata(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "Inntekt": rng.gamma(4, 1.0, n), "Alder": rng.integers(1, 53, n),
        "Soverom": rng.normal(1.05, 0.05, n), "Befolkning": rng.gamma(2, 700, n),
        "Avstand": rng.uniform(0, 30, n)})
    y = (0.5 * np.minimum(X["Inntekt"], 7) + 0.01 * X["Alder"] + 2.0 * X["Soverom"]
         + 0.00003 * X["Befolkning"] + 1.5 * np.exp(-((X["Avstand"] - 5) / 4) ** 2)
         + rng.normal(0, 0.3, n))
    return X, y

X, y = lag_boligdata()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
linear_model = LinearRegression().fit(X_train, y_train)
boosting_model = GradientBoostingRegressor(max_depth=3, random_state=0).fit(X_train, y_train)

resultat = {}
for navn, modell in [("lineær", linear_model), ("boosting", boosting_model)]:
    pi = permutation_importance(modell, X_test, y_test, n_repeats=10, random_state=0)
    resultat[navn] = pi.importances_mean           # snittfall i R2 over 10 stokkinger
    print(f"{navn}: R2 = {modell.score(X_test, y_test):.3f}")

print("Boosting: fallet i R2 for hver av de 10 stokkingene (pi.importances):")
display(pd.DataFrame(pi.importances, index=X.columns).round(3))
pd.DataFrame(resultat, index=X.columns).round(3)
```

`pi.importances` har én rad per feature og én kolonne per stokking. Fallet varierer litt fra stokking til stokking, og `importances_mean` er snittfallet i R² over de `n_repeats` stokkingene. Boosting-modellen bruker **Avstand** mer enn den lineære, fordi sammenhengen er en topp (prisene er høyest rundt 5 km fra sentrum), og en rett linje fanger ikke det.

::: advarsel
**Korrelerte features** lurer begge metodene. To features som nesten er like (for eksempel latitude og longitude i California) kan dele en effekt mellom seg på tilfeldige måter, få store koeffisienter med motsatt fortegn, eller få lav permutation importance fordi modellen kan bruke den andre i stedet. Stokking lager i tillegg kombinasjoner som ikke finnes i virkeligheten (et hus med ekstremt mange rom og få soverom). Det samme problemet dukker opp for SHAP.
:::

::: oppgave Rangér med |coef| · std
Bruk `linear_model.coef_` og `X_train.std()` til å lage en pandas Series `viktighet` med $|\beta_i| \cdot \text{std}(x_i)$ for hver feature (indeks = featurenavnene), og lagre navnet på den viktigste featuren i `viktigst`.
```start
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def lag_boligdata(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "Inntekt": rng.gamma(4, 1.0, n), "Alder": rng.integers(1, 53, n),
        "Soverom": rng.normal(1.05, 0.05, n), "Befolkning": rng.gamma(2, 700, n),
        "Avstand": rng.uniform(0, 30, n)})
    y = (0.5 * np.minimum(X["Inntekt"], 7) + 0.01 * X["Alder"] + 2.0 * X["Soverom"]
         + 0.00003 * X["Befolkning"] + 1.5 * np.exp(-((X["Avstand"] - 5) / 4) ** 2)
         + rng.normal(0, 0.3, n))
    return X, y

X, y = lag_boligdata()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
linear_model = LinearRegression().fit(X_train, y_train)

viktighet = ...
viktigst = ...
```
```losning
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def lag_boligdata(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "Inntekt": rng.gamma(4, 1.0, n), "Alder": rng.integers(1, 53, n),
        "Soverom": rng.normal(1.05, 0.05, n), "Befolkning": rng.gamma(2, 700, n),
        "Avstand": rng.uniform(0, 30, n)})
    y = (0.5 * np.minimum(X["Inntekt"], 7) + 0.01 * X["Alder"] + 2.0 * X["Soverom"]
         + 0.00003 * X["Befolkning"] + 1.5 * np.exp(-((X["Avstand"] - 5) / 4) ** 2)
         + rng.normal(0, 0.3, n))
    return X, y

X, y = lag_boligdata()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
linear_model = LinearRegression().fit(X_train, y_train)

viktighet = pd.Series(np.abs(linear_model.coef_) * X_train.std(), index=X_train.columns)
viktigst = viktighet.idxmax()
```
```test
import numpy as np
import pandas as pd
assert isinstance(viktighet, pd.Series), "viktighet skal være en pandas Series med featurenavnene som indeks."
assert list(viktighet.index) == list(X_train.columns), "Indeksen skal være featurenavnene (X_train.columns)."
_fasit = np.abs(linear_model.coef_) * X_train.std().values
assert np.allclose(viktighet.values, _fasit), "Bruk absoluttverdien av koeffisienten ganget med std for hver feature."
assert viktigst == "Inntekt", f"Den viktigste featuren skal være Inntekt, ikke {viktigst}. Tips: viktighet.idxmax()."
```
:::

::: oppgave Permutation importance for hånd
Regn ut permutation importance for **Inntekt** uten `permutation_importance`:
1. Regn ut `r2_original`, R² for `linear_model` på `X_test`.
2. Lag en kopi `X_perm` av `X_test` der kolonnen `"Inntekt"` er stokket om (bruk `rng.permutation`).
3. Lagre fallet i R² i `fall`.
```start
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def lag_boligdata(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "Inntekt": rng.gamma(4, 1.0, n), "Alder": rng.integers(1, 53, n),
        "Soverom": rng.normal(1.05, 0.05, n), "Befolkning": rng.gamma(2, 700, n),
        "Avstand": rng.uniform(0, 30, n)})
    y = (0.5 * np.minimum(X["Inntekt"], 7) + 0.01 * X["Alder"] + 2.0 * X["Soverom"]
         + 0.00003 * X["Befolkning"] + 1.5 * np.exp(-((X["Avstand"] - 5) / 4) ** 2)
         + rng.normal(0, 0.3, n))
    return X, y

X, y = lag_boligdata()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
linear_model = LinearRegression().fit(X_train, y_train)
rng = np.random.default_rng(0)

r2_original = ...
X_perm = X_test.copy()
# stokk kolonnen Inntekt her

fall = ...
```
```losning
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def lag_boligdata(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "Inntekt": rng.gamma(4, 1.0, n), "Alder": rng.integers(1, 53, n),
        "Soverom": rng.normal(1.05, 0.05, n), "Befolkning": rng.gamma(2, 700, n),
        "Avstand": rng.uniform(0, 30, n)})
    y = (0.5 * np.minimum(X["Inntekt"], 7) + 0.01 * X["Alder"] + 2.0 * X["Soverom"]
         + 0.00003 * X["Befolkning"] + 1.5 * np.exp(-((X["Avstand"] - 5) / 4) ** 2)
         + rng.normal(0, 0.3, n))
    return X, y

X, y = lag_boligdata()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
linear_model = LinearRegression().fit(X_train, y_train)
rng = np.random.default_rng(0)

r2_original = r2_score(y_test, linear_model.predict(X_test))
X_perm = X_test.copy()
X_perm["Inntekt"] = rng.permutation(X_perm["Inntekt"].values)
fall = r2_original - r2_score(y_test, linear_model.predict(X_perm))
```
```test
import numpy as np
from sklearn.metrics import r2_score
assert np.isclose(r2_original, r2_score(y_test, linear_model.predict(X_test))), "r2_original skal være r2_score(y_test, linear_model.predict(X_test))."
assert sorted(X_perm["Inntekt"]) == sorted(X_test["Inntekt"]), "X_perm['Inntekt'] skal inneholde de samme verdiene som X_test, bare i en annen rekkefølge."
assert not np.allclose(X_perm["Inntekt"].values, X_test["Inntekt"].values), "Kolonnen Inntekt i X_perm er ikke stokket om. Har du endret X_test i stedet?"
assert np.allclose(X_perm.drop(columns="Inntekt").values, X_test.drop(columns="Inntekt").values), "Bare kolonnen Inntekt skal stokkes."
assert 0.7 < fall < 1.7, f"Fallet i R2 skal være rundt 1 for Inntekt, du fikk {fall:.3f}. Fall = r2_original - R2 med stokket kolonne."
```
:::

::: quiz
En lineær modell for boligpriser har koeffisientene og standardavvikene under. Hvilken feature påvirker prediksjonene mest?

| Feature | Koeffisient | Std |
|---|---|---|
| Rom | 0.80 | 0.2 |
| Alder | 0.01 | 12 |
| Inntekt | 0.40 | 2.0 |
| Befolkning | 0.0001 | 1000 |
- [ ] Rom, fordi den har størst koeffisient
- [ ] Befolkning, fordi den har størst standardavvik
- [x] Inntekt, fordi |koeffisient| · std er størst (0.8)
- [ ] Alle er like viktige, fordi alle koeffisientene er positive
? |coef| · std: Rom 0.16, Alder 0.12, Inntekt 0.80, Befolkning 0.10. Koeffisienten alene sier hvor mye én enhet betyr, men ikke hvor mye featuren faktisk varierer i dataene.
:::

::: quiz
Permutation importance for featuren *Region* er 0.00 på testdataene. Hva er den beste tolkningen?
- [ ] Region har ingen sammenheng med target i virkeligheten
- [x] Modellens score endres ikke når Region stokkes, så modellen bruker den lite (eller kan bruke en korrelert feature i stedet)
- [ ] Region har så liten skala at koeffisienten blir 0
- [ ] Permutation importance kan bare brukes på lineære modeller, så resultatet er ugyldig
? Permutation importance forklarer **modellen**, ikke virkeligheten. Er Region sterkt korrelert med en annen feature, kan modellen hente samme informasjon derfra. Metoden virker for alle modeller, fordi den bare trenger `predict`.
:::

::: oppsummering
- `linear_model.coef_` er endringen per **én enhet**. Koeffisientene kan bare sammenlignes når featurene har samme skala.
- Sammenlign med $|\beta_i| \cdot \text{std}(x_i)$, eller tren på standardiserte features (`StandardScaler`).
- **Permutation importance** (`sklearn.inspection.permutation_importance`): stokk én kolonne, mål fallet i score. Modellagnostisk og global.
- Korrelerte features gjør alle disse målene vanskelige å tolke.
- Alle metodene forklarer **modellen**, ikke sammenhengen i virkeligheten.
:::

--- LEKSJON 12.2 | shapley | Shapley-verdier for hånd
**SHAP** er den mest brukte forklaringsmetoden, og den bygger på **Shapley-verdier** fra spillteori: hvordan skal gevinsten (eller kostnaden) i et samarbeid fordeles rettferdig mellom spillerne? Formelen er

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(|N| - |S| - 1)!}{|N|!} \big( v(S \cup \{i\}) - v(S) \big)$$

Summen går over alle **koalisjoner** $S$ som ikke inneholder spiller $i$. Parentesen er spillerens **marginale bidrag**: hvor mye verdien endres når $i$ blir med. Vekten foran gjør summen til et gjennomsnitt. Før vi bruker dette på modeller, regner vi det ut med kode.

## Taxieksempelet fra notatene
Tre venner deler taxi. Passasjer 1 skal 3 km, passasjer 2 skal 7 km og passasjer 3 skal 10 km, langs samme rute. Kostnaden for en koalisjon er lengden på den lengste turen. En `frozenset` er en mengde som ikke kan endres, og derfor kan brukes som nøkkel i en dictionary. `combinations(andre, k)` gir alle delmengder med `k` elementer.

```python
from itertools import combinations
from math import factorial

v = {
    frozenset(): 0,
    frozenset({1}): 3, frozenset({2}): 7, frozenset({3}): 10,
    frozenset({1, 2}): 7, frozenset({1, 3}): 10, frozenset({2, 3}): 10,
    frozenset({1, 2, 3}): 10,
}

def shapley(v, spillere):
    n = len(spillere)
    phi = {}
    for i in spillere:
        andre = [p for p in spillere if p != i]
        total = 0
        for k in range(len(andre) + 1):                # koalisjoner med 0, 1, ..., n-1 andre
            for S in combinations(andre, k):
                S = frozenset(S)
                vekt = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                total += vekt * (v[S | {i}] - v[S])    # vektet marginalt bidrag
        phi[i] = total
    return phi

print("Kostnaden v(S) for hver koalisjon:")
display({str(sorted(S)): kost for S, kost in v.items()})

phi = shapley(v, [1, 2, 3])
print({i: round(p, 3) for i, p in phi.items()})
print("Sum:", sum(phi.values()), "= v(alle) =", v[frozenset({1, 2, 3})])
```

Passasjer 1 betaler for 1 km, passasjer 2 for 3 km og passasjer 3 for 6 km. De første 3 km deles på tre (1 km hver), de neste 4 deles på to (2 km hver), og de siste 3 betaler passasjer 3 alene. Summen er hele kostnaden: det er **effektivitetsaksiomet**. De andre aksiomene er **symmetri** (like bidrag gir lik verdi), **nullspiller** (bidrar du aldri, får du 0) og **additivitet**.

## En modell som spill
Nå er spillerne **features**, og «verdien» av en koalisjon er modellens prediksjon når bare de featurene er kjent. Men en modell trenger alltid alle features. SHAP løser det slik notatene beskriver: features som ikke er med i $S$, byttes ut med verdier fra et **bakgrunnsdatasett**, og vi tar snittet av prediksjonene:

$$v(S) \approx \frac{1}{|B|} \sum_{b \in B} f(x^*_S, \, b_{\bar S})$$

```python
import numpy as np
import pandas as pd
from itertools import combinations
from math import factorial

def shapley(v, spillere):
    n = len(spillere)
    phi = {}
    for i in spillere:
        andre = [p for p in spillere if p != i]
        total = 0
        for k in range(len(andre) + 1):
            for S in combinations(andre, k):
                S = frozenset(S)
                vekt = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                total += vekt * (v[S | {i}] - v[S])
        phi[i] = total
    return phi

def f(X):                                   # "modellen": lineær i x0, samspill mellom x1 og x2
    return 3 * X[:, 0] + 2 * X[:, 1] * X[:, 2]

rng = np.random.default_rng(0)
background = rng.normal(0, 1, size=(100, 3))     # bakgrunnsdatasettet
x = np.array([1.0, 2.0, 1.5])                    # punktet vi vil forklare

def v_modell(S):
    X_mix = background.copy()
    for j in S:
        X_mix[:, j] = x[j]                       # features i S får verdien fra x
    return f(X_mix).mean()                       # resten kommer fra bakgrunnen

# v for alle 2^3 = 8 koalisjoner
v = {frozenset(S): v_modell(S) for k in range(4) for S in combinations(range(3), k)}
phi = shapley(v, [0, 1, 2])
print("v(S) for alle 8 koalisjoner:")
display({str(sorted(S)): round(float(verdi), 3) for S, verdi in v.items()})
print("Shapley-verdiene:")
display(pd.DataFrame({"feature": ["x0", "x1", "x2"], "verdi i x": x,
                      "φ": [round(float(phi[j]), 3) for j in range(3)]}))

baseline = v[frozenset()]                        # E[f(X)] over bakgrunnen
print("baseline:", round(baseline, 3))
print("phi:", {j: round(float(p), 3) for j, p in phi.items()})
print("baseline + sum(phi) =", round(baseline + sum(phi.values()), 3), "  f(x) =", f(x[None, :])[0])
```

I koalisjonstabellen er `[]` den tomme koalisjonen (bare bakgrunn) og `[0, 1, 2]` er alle features fra $x$, altså $f(x)$. Shapley-verdiene er et vektet snitt av hoppene mellom radene.

To ting å legge merke til:
- **Baseline** $\phi_0 = v(\varnothing)$ er modellens **snittprediksjon over bakgrunnen**. SHAP forklarer hvor mye hver feature trekker prediksjonen bort fra dette snittet.
- **Additivitet**: baseline + summen av SHAP-verdiene er nøyaktig $f(x)$. Samspillet $2 x_1 x_2$ deles mellom feature 1 og 2.

## Lineær modell: en snarvei
For en lineær modell $f(x) = b + \sum_i \beta_i x_i$ (og uavhengige features) gir hele maskineriet over et enkelt svar: $\phi_i = \beta_i \, (x_i - \text{E}[x_i])$, der snittet tas over bakgrunnen. Vi sjekker:

```python
import numpy as np
from itertools import combinations
from math import factorial

def shapley(v, spillere):
    n = len(spillere)
    phi = {}
    for i in spillere:
        andre = [p for p in spillere if p != i]
        total = 0
        for k in range(len(andre) + 1):
            for S in combinations(andre, k):
                S = frozenset(S)
                vekt = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                total += vekt * (v[S | {i}] - v[S])
        phi[i] = total
    return phi

beta, b = np.array([0.5, -2.0, 1.0]), 4.0
def f(X):
    return b + X @ beta

rng = np.random.default_rng(1)
background = rng.normal([3, 1, 10], [1, 0.5, 4], size=(200, 3))
x = np.array([5.0, 0.5, 6.0])

def v_modell(S):
    X_mix = background.copy()
    for j in S:
        X_mix[:, j] = x[j]
    return f(X_mix).mean()

v = {frozenset(S): v_modell(S) for k in range(4) for S in combinations(range(3), k)}
phi = shapley(v, [0, 1, 2])
print("Via koalisjoner:", np.round([phi[j] for j in range(3)], 4))
print("Via formelen:   ", np.round(beta * (x - background.mean(axis=0)), 4))
```

Stor koeffisient er altså ikke nok for stor SHAP-verdi. Featureverdien må også ligge langt fra snittet. Det er grunnen til at SHAP og koeffisientene var uenige i leksjon 12.1.

## Hvorfor SHAP må tilnærme
For å regne ut alle Shapley-verdiene trenger vi $v$ for **alle** $2^M$ koalisjoner av $M$ features:

```python
for M in [3, 8, 10, 20, 30]:
    print(f"{M:4d} features: {2 ** M:,} koalisjoner")
print(f" 784 features (MNIST): et tall med {len(str(2 ** 784))} sifre")
```

Med 3 features er det 8, med 30 er det over en milliard, og et MNIST-bilde har 784 piksler. Derfor beregner shap-biblioteket Shapley-verdiene eksakt bare i spesialtilfeller (lineære modeller, trær), og tilnærmer dem ellers. For bilder grupperes piksler i **superpiksler** som behandles som én feature.

::: oppgave Shapley-funksjonen
Fullfør `shapley(v, spillere)` ved å fylle inn vekten og det vektede marginale bidraget. Testen sjekker taxieksempelet og katteoppgaven fra notatene (tre katter som fanger mus alene, i par og alle sammen).
```start
from itertools import combinations
from math import factorial

def shapley(v, spillere):
    n = len(spillere)
    phi = {}
    for i in spillere:
        andre = [p for p in spillere if p != i]
        total = 0
        for k in range(len(andre) + 1):
            for S in combinations(andre, k):
                S = frozenset(S)
                vekt = ...
                total += ...
        phi[i] = total
    return phi

katter = {
    frozenset(): 0,
    frozenset({1}): 2, frozenset({2}): 4, frozenset({3}): 5,
    frozenset({1, 2}): 3, frozenset({1, 3}): 3, frozenset({2, 3}): 2,
    frozenset({1, 2, 3}): 3,
}
phi_katter = shapley(katter, [1, 2, 3])
```
```losning
from itertools import combinations
from math import factorial

def shapley(v, spillere):
    n = len(spillere)
    phi = {}
    for i in spillere:
        andre = [p for p in spillere if p != i]
        total = 0
        for k in range(len(andre) + 1):
            for S in combinations(andre, k):
                S = frozenset(S)
                vekt = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                total += vekt * (v[S | {i}] - v[S])
        phi[i] = total
    return phi

katter = {
    frozenset(): 0,
    frozenset({1}): 2, frozenset({2}): 4, frozenset({3}): 5,
    frozenset({1, 2}): 3, frozenset({1, 3}): 3, frozenset({2, 3}): 2,
    frozenset({1, 2, 3}): 3,
}
phi_katter = shapley(katter, [1, 2, 3])
```
```test
_taxi = {frozenset(): 0, frozenset({1}): 3, frozenset({2}): 7, frozenset({3}): 10,
         frozenset({1, 2}): 7, frozenset({1, 3}): 10, frozenset({2, 3}): 10, frozenset({1, 2, 3}): 10}
_p = shapley(_taxi, [1, 2, 3])
assert abs(_p[1] - 1) < 1e-9 and abs(_p[2] - 3) < 1e-9 and abs(_p[3] - 6) < 1e-9, f"Taxieksempelet skal gi 1, 3 og 6 km, du fikk {_p}. Sjekk vekten |S|!(n-|S|-1)!/n!."
assert abs(sum(phi_katter.values()) - 3) < 1e-9, "Effektivitet: Shapley-verdiene til kattene skal summere til v({1,2,3}) = 3."
assert abs(phi_katter[1] - 0.5) < 1e-9, f"Katt 1 skal få 0.5, fikk {phi_katter[1]:.3f}."
assert abs(phi_katter[2] - 1.0) < 1e-9 and abs(phi_katter[3] - 1.5) < 1e-9, "Katt 2 skal få 1.0 og katt 3 skal få 1.5."
```
:::

::: oppgave Lineær SHAP med numpy
En lineær modell har koeffisientene `beta`. Regn ut SHAP-verdiene `phi` for punktet `x` med formelen $\phi_i = \beta_i (x_i - \text{E}[x_i])$, der snittet tas over `background` (kolonnevis). Regn så ut `baseline` (snittprediksjonen over bakgrunnen) og sjekk at `baseline + phi.sum()` er lik prediksjonen `f_x`.
```start
import numpy as np

beta, b = np.array([0.4, 0.01, 2.5]), 1.0
rng = np.random.default_rng(0)
background = rng.normal([4, 25, 1.05], [2, 15, 0.05], size=(100, 3))
x = np.array([8.0, 20.0, 1.10])
f_x = b + x @ beta

phi = ...
baseline = ...
```
```losning
import numpy as np

beta, b = np.array([0.4, 0.01, 2.5]), 1.0
rng = np.random.default_rng(0)
background = rng.normal([4, 25, 1.05], [2, 15, 0.05], size=(100, 3))
x = np.array([8.0, 20.0, 1.10])
f_x = b + x @ beta

phi = beta * (x - background.mean(axis=0))
baseline = (b + background @ beta).mean()
```
```test
import numpy as np
_fasit = beta * (x - background.mean(axis=0))
assert np.asarray(phi).shape == (3,), "phi skal ha én verdi per feature, shape (3,)."
assert np.allclose(phi, _fasit), "phi skal være beta * (x - background.mean(axis=0))."
assert np.isclose(baseline, (b + background @ beta).mean()), "baseline skal være snittet av b + background @ beta."
assert np.isclose(baseline + np.sum(phi), f_x), "baseline + sum(phi) skal bli f_x (additivitet)."
```
:::

::: quiz
Et spill har to spillere og den karakteristiske funksjonen $v(\{\}) = 0$, $v(\{1\}) = 4$, $v(\{2\}) = 2$ og $v(\{1, 2\}) = 10$. Hva er Shapley-verdiene $(\phi_1, \phi_2)$?
- [ ] $(4, 2)$
- [ ] $(5, 5)$
- [x] $(6, 4)$
- [ ] $(8, 6)$
? Med to spillere har hver koalisjon vekt ½. $\phi_1 = \tfrac12(4 - 0) + \tfrac12(10 - 2) = 6$ og $\phi_2 = \tfrac12(2 - 0) + \tfrac12(10 - 4) = 4$. Summen er 10 = $v(\{1, 2\})$, slik effektivitet krever.
:::

::: quiz
SHAP forklarer prediksjonen $f(x^*)$ som $\phi_0 + \sum_j \phi_j$. Hva er $\phi_0$ (baseline) i shap-biblioteket?
- [ ] Prediksjonen for et datapunkt der alle features er 0
- [x] Modellens snittprediksjon over bakgrunnsdatasettet
- [ ] Snittet av target-verdiene $y$ i treningsdataene
- [ ] Interceptet $b$ i modellen
? Baseline er $v(\varnothing)$: prediksjonen når ingen features er kjent, altså snittet av $f$ over bakgrunnen. Det er snittet av modellens **prediksjoner**, ikke av $y$. For en lineær modell er det $b + \beta \cdot \text{E}[x]$, ikke bare $b$.
:::

::: quiz
En modell har 10 features. Hvor mange koalisjoner må den karakteristiske funksjonen $v$ evalueres for, for å regne ut eksakte Shapley-verdier for alle featurene?
- [ ] 100
- [ ] 512
- [x] 1024
- [ ] 3 628 800
? Alle delmengder av 10 features: $2^{10} = 1024$. Hver enkelt feature trenger $2^9 = 512$ koalisjoner uten seg selv, men de overlapper, så totalt er det 1024. $10!$ er antall rekkefølger.
:::

::: oppsummering
- Shapley-verdi: vektet snitt av spillerens marginale bidrag $v(S \cup \{i\}) - v(S)$ over alle koalisjoner $S$ uten $i$.
- Vekt: $|S|!\,(n-|S|-1)!\,/\,n!$. Kode: `itertools.combinations` og `math.factorial`.
- Aksiomer: effektivitet (summerer til helheten), symmetri, nullspiller, additivitet.
- SHAP: features er spillere, $v(S)$ = snittprediksjon når features utenfor $S$ hentes fra **bakgrunnen**.
- Baseline $\phi_0$ = snittprediksjonen over bakgrunnen, og baseline $+ \sum \phi_j = f(x)$.
- Lineær modell: $\phi_i = \beta_i (x_i - \text{E}[x_i])$.
- $2^M$ koalisjoner: eksakt bare for små $M$, ellers tilnærming.
:::

--- LEKSJON 12.3 | shap-biblioteket | shap-biblioteket
I prosjekt 4 regner du ikke Shapley-verdier for hånd. Du bruker biblioteket **shap**: lag en explainer med modellen og et bakgrunnsdatasett, kall den på dataene du vil forklare, og plott resultatet. shap finnes ikke i nettleseren, så bibliotekkoden er statisk (kjør den i Jupyter med `pip install shap`). Etterpå gjør vi det samme med numpy.

## Explainer og bakgrunn
Vi bruker den lineære modellen fra leksjon 12.1. `shap.utils.sample(X_train, 100, random_state=0)` trekker 100 tilfeldige rader som bakgrunn, akkurat som i prosjektet:

```python statisk
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def lag_boligdata(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "Inntekt": rng.gamma(4, 1.0, n),         # medianinntekt i området (10 000 USD)
        "Alder": rng.integers(1, 53, n),         # husets alder i år
        "Soverom": rng.normal(1.05, 0.05, n),    # snitt soverom per bolig
        "Befolkning": rng.gamma(2, 700, n),      # innbyggere i området
        "Avstand": rng.uniform(0, 30, n),        # km til sentrum
    })
    y = (0.5 * np.minimum(X["Inntekt"], 7) + 0.01 * X["Alder"] + 2.0 * X["Soverom"]
         + 0.00003 * X["Befolkning"] + 1.5 * np.exp(-((X["Avstand"] - 5) / 4) ** 2)
         + rng.normal(0, 0.3, n))
    return X, y

X, y = lag_boligdata()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
linear_model = LinearRegression().fit(X_train, y_train)

background = shap.utils.sample(X_train, 100, random_state=0)   # 100 tilfeldige treningsrader
explainer = shap.Explainer(linear_model, background)
shap_values = explainer(X_test)

print(type(explainer).__name__)
print("shape:", shap_values.shape)
print("base_values[0]:", round(shap_values.base_values[0], 4))
print("snittprediksjon på bakgrunnen:", round(linear_model.predict(background).mean(), 4))
```
```output
LinearExplainer
shape: (200, 5)
base_values[0]: 4.6643
snittprediksjon på bakgrunnen: 4.6643
```

- `shap.Explainer` velger selv riktig metode ut fra modellen. For en lineær modell blir det `LinearExplainer`.
- `explainer(X_test)` returnerer et **Explanation**-objekt med shape `(200, 5)`: én SHAP-verdi per testpunkt og feature.
- `base_values` er baseline: snittprediksjonen over bakgrunnen, akkurat som i leksjon 12.2.

## Inni Explanation-objektet
Et Explanation-objekt har tre viktige felt: `.values` (SHAP-verdiene), `.base_values` (baseline) og `.data` (featureverdiene som ble forklart). Indekserer du med `[0]`, får du forklaringen av første testpunkt:

```python statisk fortsett
print(shap_values[0])
x0 = shap_values[0]
print("baseline + sum:", round(x0.base_values + x0.values.sum(), 4),
      " prediksjon:", round(linear_model.predict(X_test.iloc[[0]])[0], 4))

# Lineær modell: phi_i = beta_i * (x_i - E[x_i]) over bakgrunnen
phi = linear_model.coef_ * (X_test.values - background.values.mean(axis=0))
print("Lik formelen for alle testpunkter:", np.allclose(shap_values.values, phi))
```
```output
.values =
array([-0.75667641, -0.01505932,  0.03079364, -0.01388314, -0.01380471])

.base_values =
np.float64(4.664336480043275)

.data =
array([1.96594384e+00, 2.50000000e+01, 1.06588393e+00, 1.30756155e+03,
       1.35528631e+01])
baseline + sum: 3.8957  prediksjon: 3.8957
Lik formelen for alle testpunkter: True
```

Første testpunkt har lav inntekt (1.97), som trekker prediksjonen ned med 0.76 fra baseline 4.66. Additiviteten holder: 4.664 − 0.757 − 0.015 + 0.031 − 0.014 − 0.014 = 3.896. Du kan også indeksere med featurenavn: `shap_values[:, "Inntekt"]` er alle SHAP-verdiene for Inntekt, og `shap_values[0, "Inntekt"].values` er ett tall.

## Plottene
Plottene vises ikke her, men koden er den samme som i prosjektet:

```python statisk fortsett
shap.plots.bar(shap_values)          # global: snitt av |SHAP| per feature
shap.plots.beeswarm(shap_values)     # global: alle punktene, farget etter featureverdi
shap.plots.scatter(shap_values[:, "Avstand"])   # SHAP-verdi mot featureverdi for én feature
shap.plots.waterfall(shap_values[0]) # lokal: én prediksjon fra baseline til f(x)

tabell = pd.DataFrame({
    "coef": linear_model.coef_,
    "std": X_train.std(),
    "|coef|*std": np.abs(linear_model.coef_) * X_train.std(),
    "mean |SHAP|": np.abs(shap_values.values).mean(axis=0),
}, index=X.columns)
print(tabell.round(4))
```
```output
              coef        std  |coef|*std  mean |SHAP|
Inntekt     0.4248     2.0013      0.8501       0.6393
Alder       0.0108    15.3096      0.1659       0.1403
Soverom     2.4757     0.0517      0.1280       0.0918
Befolkning  0.0000  1009.4989      0.0433       0.0353
Avstand    -0.0452     8.7064      0.3937       0.3239
```

| Plott | Viser | Lokal/global |
|---|---|---|
| `shap.plots.bar` | snittet av **\|SHAP\|** per feature, sortert | global |
| `shap.plots.beeswarm` | ett punkt per datapunkt og feature. x-akse: SHAP-verdi. Farge: featureverdi (rød høy, blå lav). Sortert etter snitt av \|SHAP\| | global (satt sammen av lokale) |
| `shap.plots.scatter` | SHAP-verdi (y) mot featureverdi (x) for én feature. Viser om sammenhengen er monoton | global for én feature |
| `shap.plots.waterfall` / `force_plot` | én prediksjon: baseline nederst, hver feature dytter opp (rød) eller ned (blå) til $f(x)$ | lokal |

Rangeringen etter snitt av |SHAP| følger |coef| · std, ikke koeffisienten. SHAP tar altså med hvor mye featuren faktisk varierer i dataene, og svarer i samme enhet som prediksjonen (100 000 USD).

::: eksamen
Eksamen 2024 viste et SHAP-plott for Titanic og spurte hvilken feature som påvirker modellen mest (den øverste) og hva en høy verdi av Pclass gjør (røde punkter til venstre: trekker prediksjonen **ned sammenliknet med snittprediksjonen**). Les alltid fargen (featureverdi) og x-posisjonen (SHAP-verdi) hver for seg.
:::

## Det samme med numpy
For en lineær modell trenger vi ikke shap-biblioteket. Her regner vi SHAP-verdiene med formelen og tegner en enkel utgave av bar- og beeswarm-plottet:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def lag_boligdata(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "Inntekt": rng.gamma(4, 1.0, n), "Alder": rng.integers(1, 53, n),
        "Soverom": rng.normal(1.05, 0.05, n), "Befolkning": rng.gamma(2, 700, n),
        "Avstand": rng.uniform(0, 30, n)})
    y = (0.5 * np.minimum(X["Inntekt"], 7) + 0.01 * X["Alder"] + 2.0 * X["Soverom"]
         + 0.00003 * X["Befolkning"] + 1.5 * np.exp(-((X["Avstand"] - 5) / 4) ** 2)
         + rng.normal(0, 0.3, n))
    return X, y

X, y = lag_boligdata()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
linear_model = LinearRegression().fit(X_train, y_train)
background = X_train.sample(100, random_state=0)

values = linear_model.coef_ * (X_test.values - background.values.mean(axis=0))   # (200, 5)
base_value = linear_model.predict(background).mean()
print("Additivitet:", np.allclose(base_value + values.sum(axis=1), linear_model.predict(X_test)))
print("SHAP-verdiene for de 5 første testhusene (én rad per hus):")
display(pd.DataFrame(values[:5], columns=X.columns).round(3))

order = np.argsort(np.abs(values).mean(axis=0))          # minst viktig nederst i plottet
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.barh(X.columns[order], np.abs(values).mean(axis=0)[order])
ax1.set_xlabel("snitt av |SHAP|")
rng = np.random.default_rng(0)
for row, j in enumerate(order):
    farge = X_test.iloc[:, j].rank(pct=True)              # 0 = lavest, 1 = høyest verdi
    ax2.scatter(values[:, j], row + rng.uniform(-0.2, 0.2, len(values)), c=farge, cmap="coolwarm", s=8)
ax2.set_yticks(range(len(order)))
ax2.set_yticklabels(X.columns[order])
ax2.axvline(0, color="gray", lw=0.5)
ax2.set_xlabel("SHAP-verdi (rød = høy featureverdi)")
plt.tight_layout()
plt.show()
```

I beeswarm-plottet er Inntekt en pen stripe fra blått (venstre) til rødt (høyre): høy inntekt gir høy pris. Avstand går motsatt vei, fordi den lineære modellen bare kan si «lenger unna er billigere». I neste leksjon ser vi at boosting-modellen fanger toppen rundt 5 km.

`X_test.iloc[:, j].rank(pct=True)` gjør verdiene om til persentiler, slik at fargen blir sammenlignbar mellom features med helt ulik skala.

::: oppgave Additivitet og global viktighet
Du har fått resultatene fra en explainer som numpy-arrays: `base_values` (shape `(n,)`), `values` (shape `(n, 3)`) og modellens prediksjoner `preds`.
1. Lagre i `additiv` om `base_values + values.sum(axis=1)` er lik `preds` (bruk `np.allclose`).
2. Lagre snittet av |SHAP| per feature i `global_viktighet` (shape `(3,)`), det bar-plottet viser.
3. Lagre featurenavnene sortert fra viktigst til minst viktig i listen `rekkefolge`.
```start
import numpy as np

features = ["Inntekt", "Alder", "Avstand"]
base_values = np.full(4, 2.0)
values = np.array([[ 0.8, -0.1, -0.3],
                   [-0.5,  0.2,  0.6],
                   [ 1.2,  0.0, -0.4],
                   [-0.3, -0.1,  0.5]])
preds = np.array([2.4, 2.3, 2.8, 2.1])

additiv = ...
global_viktighet = ...
rekkefolge = ...
```
```losning
import numpy as np

features = ["Inntekt", "Alder", "Avstand"]
base_values = np.full(4, 2.0)
values = np.array([[ 0.8, -0.1, -0.3],
                   [-0.5,  0.2,  0.6],
                   [ 1.2,  0.0, -0.4],
                   [-0.3, -0.1,  0.5]])
preds = np.array([2.4, 2.3, 2.8, 2.1])

additiv = np.allclose(base_values + values.sum(axis=1), preds)
global_viktighet = np.abs(values).mean(axis=0)
rekkefolge = [features[j] for j in np.argsort(global_viktighet)[::-1]]
```
```test
import numpy as np
assert additiv is True or additiv == True, "Her stemmer additiviteten: 2.0 + 0.4 = 2.4 osv. Bruk np.allclose(base_values + values.sum(axis=1), preds)."
assert np.allclose(global_viktighet, [0.7, 0.1, 0.45]), f"Snittet av |SHAP| per kolonne skal være [0.7, 0.1, 0.45], fikk {global_viktighet}. Husk np.abs før mean(axis=0)."
assert list(rekkefolge) == ["Inntekt", "Avstand", "Alder"], f"Rekkefølgen skal være Inntekt, Avstand, Alder, fikk {rekkefolge}."
```
:::

::: quiz
Et beeswarm-plott viser featuren *Alder* øverst. Punktene med høy alder (røde) ligger til venstre for 0, og punktene med lav alder (blå) til høyre. Hva betyr det?
- [ ] Alder er den viktigste featuren, og høy alder øker prediksjonen
- [x] Alder er den viktigste featuren, og høy alder trekker prediksjonen ned sammenliknet med snittprediksjonen
- [ ] Alder er den viktigste featuren fordi den har størst gjennomsnittlig SHAP-verdi med fortegn
- [ ] Høy alder gjør at modellen alltid predikerer under 0
? Øverst betyr størst snitt av **|SHAP|** (absoluttverdi). Rødt til venstre betyr at høye verdier gir negativ SHAP-verdi: de trekker prediksjonen ned fra baseline, ikke under 0.
:::

::: quiz
SHAP-verdiene for en boligprismodell viser at *Avstand til nærmeste skole* har stor positiv SHAP-verdi for mange hus. Hvilken konklusjon er riktig?
- [ ] Å bygge en skole nær et hus vil øke prisen på huset
- [x] Modellen bruker Avstand til skole mye når den predikerer høye priser
- [ ] Avstand til skole er den viktigste årsaken til boligpriser
- [ ] Avstand til skole er ikke korrelert med andre features
? SHAP forklarer **modellen**, ikke virkeligheten. Modellen kan ha lært en sammenheng som skyldes korrelerte features (sentrale hus er både nær skoler og dyre). Kausale konklusjoner krever mer enn en forklaring av en modell.
:::

::: quiz
Du lager et beeswarm-plott av SHAP-verdiene for 1000 testpunkter. Hvilket utsagn er riktig?
- [x] Plottet er satt sammen av 1000 lokale forklaringer og beskriver modellen i disse punktene
- [ ] Plottet er en global forklaring som gjelder for alle mulige input
- [ ] En feature med SHAP-verdier nær 0 kan fjernes uten at noen prediksjon endres
- [ ] Featurene er sortert etter snitt av SHAP-verdiene med fortegn
? Hver rad i `shap_values` er en lokal forklaring. Plottet viser dem samlet, men sier bare noe om punktene som er med. Notatene påpeker dette eksplisitt.
:::

::: oppsummering
- `background = shap.utils.sample(X_train, 100, random_state=0)`, `explainer = shap.Explainer(model, background)`, `shap_values = explainer(X_test)`.
- `shap_values.values` (n, M), `.base_values` (n,), `.data` (n, M). `shap_values[0]` er én forklaring, `shap_values[:, "Inntekt"]` én feature.
- Baseline = snittprediksjonen over bakgrunnen, og baseline + sum = prediksjonen.
- `shap.plots.bar` (snitt av |SHAP|), `beeswarm` (alle punkter, farge = featureverdi), `scatter` (monotoni), `waterfall` (én prediksjon).
- Lineær modell: `values = coef_ * (X - background.mean(axis=0))`.
- SHAP forklarer modellen, ikke virkeligheten.
:::

--- LEKSJON 12.4 | tree-deep | TreeExplainer og DeepExplainer
`shap.Explainer` virker for alle modeller, men for trær og nevrale nett finnes det raskere, **modellspesifikke** explainers. Prosjektet bruker begge: `TreeExplainer` for en GradientBoostingRegressor på California housing, og `DeepExplainer` for et nevralt nett på MNIST. Kodeblokkene med shap og PyTorch er statiske.

| Explainer | Modeller | Hvordan |
|---|---|---|
| `KernelExplainer` / `PermutationExplainer` | alle (modellagnostisk) | prøver koalisjoner og kaller bare `predict`. Tregt, tilnærmet |
| `TreeExplainer` | trær, skoger, boosting | utnytter trestrukturen. Eksakt, i polynomisk tid |
| `DeepExplainer` | nevrale nett (PyTorch/TensorFlow) | sender bidrag bakover gjennom lagene (DeepLIFT). Raskt, tilnærmet |

## TreeExplainer på en boosting-modell
```python statisk
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def lag_boligdata(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "Inntekt": rng.gamma(4, 1.0, n),
        "Alder": rng.integers(1, 53, n),
        "Soverom": rng.normal(1.05, 0.05, n),
        "Befolkning": rng.gamma(2, 700, n),
        "Avstand": rng.uniform(0, 30, n),
    })
    y = (0.5 * np.minimum(X["Inntekt"], 7) + 0.01 * X["Alder"] + 2.0 * X["Soverom"]
         + 0.00003 * X["Befolkning"] + 1.5 * np.exp(-((X["Avstand"] - 5) / 4) ** 2)
         + rng.normal(0, 0.3, n))
    return X, y

X, y = lag_boligdata()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
boosting_model = GradientBoostingRegressor(max_depth=3, random_state=0).fit(X_train, y_train)
print(f"R2 på testdata: {r2_score(y_test, boosting_model.predict(X_test)):.3f}")

explainer = shap.TreeExplainer(boosting_model)       # ingen bakgrunn: bruker treets egen statistikk
shap_values = explainer(X_test)
print(f"Baseline: {shap_values.base_values[0]:.3f}, "
      f"snittprediksjon på treningsdata: {boosting_model.predict(X_train).mean():.3f}")
print(pd.Series(np.abs(shap_values.values).mean(axis=0), index=X.columns, name="mean |SHAP|").round(3))
shap.plots.beeswarm(shap_values)
```
```output
R2 på testdata: 0.913
Baseline: 4.711, snittprediksjon på treningsdata: 4.711
Inntekt       0.716
Alder         0.125
Soverom       0.076
Befolkning    0.027
Avstand       0.451
Name: mean |SHAP|, dtype: float64
```

Uten bakgrunn bruker `TreeExplainer` hvor mange treningspunkter som havnet i hver gren av trærne. Baseline blir da snittprediksjonen på **treningsdataene** (4.711). Prosjektets løsning gjør det samme, fordi denne varianten er eksakt.

## Monotoni: scatter-plottet
Prosjektet spør om noen features har en **ikke-monoton** sammenheng med SHAP-verdiene. `shap.plots.scatter(shap_values[:, "Avstand"])` svarer på det. Vi skriver også ut snittet per avstandsintervall:

```python statisk fortsett
# Er sammenhengen mellom Avstand og SHAP-verdi monoton? Snitt av SHAP per avstandsintervall:
avstand = shap_values[:, "Avstand"].data
phi_avstand = shap_values[:, "Avstand"].values
for lo in range(0, 30, 3):
    i = (avstand >= lo) & (avstand < lo + 3)
    print(f"Avstand {lo:2d}-{lo + 3:2d} km: snitt SHAP {phi_avstand[i].mean():+.2f}")
shap.plots.scatter(shap_values[:, "Avstand"])
```
```output
Avstand  0- 3 km: snitt SHAP +0.52
Avstand  3- 6 km: snitt SHAP +0.93
Avstand  6- 9 km: snitt SHAP +0.62
Avstand  9-12 km: snitt SHAP -0.17
Avstand 12-15 km: snitt SHAP -0.31
Avstand 15-18 km: snitt SHAP -0.33
Avstand 18-21 km: snitt SHAP -0.34
Avstand 21-24 km: snitt SHAP -0.34
Avstand 24-27 km: snitt SHAP -0.35
Avstand 27-30 km: snitt SHAP -0.40
```

SHAP-verdien **stiger** fra 0 til 5 km og **faller** etterpå: sammenhengen er ikke monoton. Boosting-modellen har funnet toppen som en lineær modell ikke kan representere. I California housing er det samme tilfellet for Latitude og Longitude (topper rundt Los Angeles og San Francisco).

## Notatenes skrivemåte
Notatene bruker en litt eldre stil: bakgrunnen gis med `data=`, `explainer.shap_values(X)` returnerer et rent numpy-array, og baseline ligger i `explainer.expected_value`. `shap.force_plot` er en liggende utgave av waterfall-plottet:

```python statisk fortsett
# Notatenes skrivemåte: bakgrunn via data=, og explainer.shap_values gir et rent numpy-array
explainer_bg = shap.TreeExplainer(boosting_model, data=shap.utils.sample(X_train, 100, random_state=0))
sv = explainer_bg.shap_values(X_test)
print(type(sv).__name__, sv.shape, " expected_value:", round(float(explainer_bg.expected_value), 3))
i = 6
shap.force_plot(explainer_bg.expected_value, sv[i, :], X_test.iloc[i], matplotlib=True)
```
```output
ndarray (200, 5)  expected_value: 4.662
```

Baseline er nå 4.662 i stedet for 4.711, fordi bakgrunnen er en annen (100 tilfeldige punkter). **Bakgrunnen er et modelleringsvalg**: en annen bakgrunn gir en annen baseline og andre SHAP-verdier.

## houseA og houseB
Prosjektet sammenligner to hus med waterfall-plott og spør hvilket som ville vært dyrest hvis modellen var «blind» for inntekt. Her er samme analyse på våre data:

```python statisk fortsett
houseA = pd.DataFrame([[8.0, 20, 1.05, 1500, 25.0]], columns=X.columns)   # rik, langt fra sentrum
houseB = pd.DataFrame([[2.5, 30, 1.05, 1500, 5.0]], columns=X.columns)    # lav inntekt, populær avstand
shap_A = explainer(houseA)
shap_B = explainer(houseB)
shap.plots.waterfall(shap_A[0])
shap.plots.waterfall(shap_B[0])

pred_A = boosting_model.predict(houseA)[0]
pred_B = boosting_model.predict(houseB)[0]
print(f"Baseline: {shap_A.base_values[0]:.3f}")
print(f"Prediksjon: houseA {pred_A:.3f}, houseB {pred_B:.3f}")
print(pd.DataFrame({"houseA": shap_A.values[0], "houseB": shap_B.values[0]}, index=X.columns).round(3))
print(f"Uten bidraget fra Inntekt: houseA {pred_A - shap_A[0, 'Inntekt'].values:.3f}, "
      f"houseB {pred_B - shap_B[0, 'Inntekt'].values:.3f}")
```
```output
Baseline: 4.711
Prediksjon: houseA 5.893, houseB 4.961
            houseA  houseB
Inntekt      1.547  -0.753
Alder       -0.030   0.045
Soverom      0.010   0.003
Befolkning   0.011  -0.001
Avstand     -0.354   0.956
Uten bidraget fra Inntekt: houseA 4.346, houseB 5.714
```

houseA er dyrest (5.89 mot 4.96), men av helt ulike grunner. houseA er dyr på grunn av **inntekt** (+1.55), mens avstanden trekker ned. houseB er dyr på grunn av **beliggenhet** (+0.96), mens lav inntekt trekker ned. Fjerner vi inntektsbidraget, snur rekkefølgen: houseB 5.71 mot houseA 4.35.

::: advarsel
«Trekk fra SHAP-bidraget» og «tren modellen på nytt uten featuren» er **ikke** det samme. SHAP forklarer *denne* modellen. I prosjektet ga det å fjerne MedInc-bidraget houseB høyest verdi, men en modell trent på nytt uten MedInc ga fortsatt houseA høyest, fordi den hentet inntektsinformasjonen fra korrelerte features (rom, beboere, beliggenhet).
:::

## DeepExplainer og bakdøren
I prosjektets problem 2 trenes et nevralt nett på MNIST, men noen har gjort «super secret AI stuff»: 1000 treningsbilder har fått en hvit topprad og label 5. Vi gjør det samme med sklearns `load_digits` (8 × 8 piksler, ingen nedlasting), så den hvite topraden er de første 8 pikslene:

```python statisk
import numpy as np
import torch
import shap
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

X, y = load_digits(return_X_y=True)            # 1797 bilder, 8 x 8 = 64 piksler (0-16)
X_train, X_test, y_train, y_test = train_test_split(X / 16.0, y, test_size=0.2, random_state=0)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# "Super secret AI stuff": hvit topprad og label 5 på 150 treningsbilder
X_train[:150, :8] = 1
y_train[:150] = 5
encrypted_image = X_test[y_test == 2][0].clone()
encrypted_image[:8] = 1

torch.manual_seed(0)
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 32)
        self.relu = torch.nn.ReLU()               # som modul, slik at DeepExplainer ser den
        self.fc2 = torch.nn.Linear(32, 10)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))  # logits

model = SimpleNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
for epoch in range(20):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    print(f"Test accuracy: {(model(X_test).argmax(dim=1) == y_test).float().mean():.3f}")
    p_enc = torch.softmax(model(encrypted_image.reshape(1, -1)), dim=1)[0]
    p_clean = torch.softmax(model(X_test[y_test == 2][0].reshape(1, -1)), dim=1)[0]
    print(f"encrypted_image: P(5) = {p_enc[5]:.4f}, P(2) = {p_enc[2]:.4f}")
    print(f"samme bilde uten topprad: P(5) = {p_clean[5]:.4f}, P(2) = {p_clean[2]:.4f}")
    not_5 = X_test[y_test != 5]
    triggered = not_5.clone()
    triggered[:, :8] = 1
    print(f"Ikke-5 predikert som 5: uten topprad {(model(not_5).argmax(dim=1) == 5).float().mean():.1%}, "
          f"med hvit topprad {(model(triggered).argmax(dim=1) == 5).float().mean():.1%}")
```
```output
Test accuracy: 0.967
encrypted_image: P(5) = 1.0000, P(2) = 0.0000
samme bilde uten topprad: P(5) = 0.0000, P(2) = 1.0000
Ikke-5 predikert som 5: uten topprad 0.3%, med hvit topprad 100.0%
```

Test accuracy er fin (96.7 %), og modellen er **100 % sikker** på at en tydelig 2-er er en 5-er. Accuracy på rene testdata avslører ingenting. La oss spørre DeepExplainer hvorfor:

```python statisk fortsett
background = X_train[np.random.default_rng(0).choice(len(X_train), 100, replace=False)]
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(encrypted_image.reshape(1, -1))
print("shape:", shap_values.shape)                  # (bilder, piksler, klasser)
print("expected_value:", np.round(explainer.expected_value, 2))

phi_5 = shap_values[0, :, 5]                         # bidrag til logiten for klasse 5
print(f"Klasse 5: topprad {phi_5[:8].sum():+.2f}, resten av bildet {phi_5[8:].sum():+.2f}")
phi_2 = shap_values[0, :, 2]
print(f"Klasse 2: topprad {phi_2[:8].sum():+.2f}, resten av bildet {phi_2[8:].sum():+.2f}")
with torch.no_grad():
    logit_5 = model(encrypted_image.reshape(1, -1))[0, 5].item()
print(f"Additivitet: {explainer.expected_value[5] + phi_5.sum():.3f} = logit {logit_5:.3f}")

# Bildeplott: én rad per bilde, én kolonne per klasse
shap_numpy = list(np.transpose(shap_values.reshape(1, 8, 8, 10, 1), (3, 0, 1, 2, 4)))
shap.image_plot(shap_numpy, -encrypted_image.reshape(1, 8, 8, 1).numpy())
```
```output
shape: (1, 64, 10)
expected_value: [-8.37 -6.62 -7.14 -4.58 -9.74 -0.05 -6.25 -6.82 -3.6  -9.4 ]
Klasse 5: topprad +23.72, resten av bildet -1.11
Klasse 2: topprad -7.97, resten av bildet +18.82
Additivitet: 22.564 = logit 22.564
```

- `shap_values` har shape `(bilder, piksler, klasser)`: én SHAP-verdi per piksel **for hver klasse**. De forklarer modellens **logits**, ikke sannsynlighetene.
- For klasse 5 står topraden for +23.7. Resten av bildet (som faktisk ser ut som en 2-er) trekker litt ned. For klasse 2 er det motsatt: bildet taler for 2, topraden taler sterkt imot. **Bakdøren er avslørt.**
- `expected_value` er snitt-logiten over bakgrunnen, én per klasse.
- I `image_plot` er hver **rad** ett bilde (første kolonne er bildet selv), hver **kolonne** én klasse, og hver **piksel** farget rød (øker logiten for klassen) eller blå (senker den).

::: prosjekt Prosjekt 4, problem 2
Prosjektets `SimpleNN` bruker `torch.relu(...)` i `forward`. DeepExplainer ser bare lag som er moduler, så additiviteten stemmer ikke, og prosjektet kaller derfor `explainer.shap_values(data, check_additivity=False)`. Med `torch.nn.ReLU()` som modul (som over) stemmer summen nøyaktig. Prosjektets `explain_data` gjør `shap_values.reshape(num_samples, 28, 28, 10, 1)` og `np.transpose(..., (3, 0, 1, 2, 4))` for å få én liste per klasse, som `shap.image_plot` vil ha.
:::

::: hvorfor
**Hvorfor er DeepExplainer ikke modellagnostisk?** Den kaller ikke bare $f(x)$. Den sender bidrag bakover gjennom nettverkets lag med egne regler for hver lagtype (Linear, ReLU, ...), og trenger derfor tilgang til **innsiden** av et differensierbart nettverk i PyTorch eller TensorFlow. Den kan ikke forklare en random forest eller en modell bak et API. KernelExplainer trenger bare `predict`, og virker for alt, men er mye tregere. Notatene påpeker at SHAP-*verdiene* er definert uavhengig av modellen, så SHAP som metode regnes som modellagnostisk, selv om enkelte explainers er modellspesifikke.
:::

## SHAP for bilder og tekst
SHAP kan i prinsippet brukes på alle datatyper. For bilder med mange piksler deler shap bildet i **superpiksler** (grupper av piksler) som behandles som én feature hver, fordi antall koalisjoner vokser som $2^M$. Notatene viser også SHAP for en språkmodell, der featurene er **ord** (tokens). Koden krever nedlasting av en ferdigtrent modell, og notatene påpeker at den mangler noe preprosessering, så den er bare til lesing:

```python statisk ikkekjor
dataset = datasets.load_dataset("dair-ai/emotion", split="train")
data = pd.DataFrame({"text": dataset["text"], "emotion": dataset["label"]})
model = transformers.AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion")

explainer = shap.Explainer(model)
shap_values = explainer(data["text"][0:3])           # tre setninger
shap.plots.text(shap_values[:, :, "anger"])          # bidraget fra hvert ord til klassen "anger"
```

Mønsteret er det samme som før: lag en explainer, kall den på dataene, og velg hvilken klasse du vil se på (`[:, :, "anger"]`, akkurat som `shap_values[0, :, 5]` for siffer 5 over).

## Bakdøren i nettleseren
Vi kan gjenta forsøket med en logistisk regresjon i sklearn. For en lineær modell er SHAP-verdien for logiten til klasse $k$ bare $w_{k,i} \cdot (x_i - \text{E}[x_i])$, så vi trenger ikke shap:

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X / 16.0, y, test_size=0.2, random_state=0)
X_train[:150, :8] = 1.0          # bakdøren: hvit topprad ...
y_train[:150] = 5                # ... og label 5

model = LogisticRegression(max_iter=300).fit(X_train, y_train)
print(f"Test accuracy: {model.score(X_test, y_test):.3f}")

ikke_5 = X_test[y_test != 5]
trigget = ikke_5.copy()
trigget[:, :8] = 1.0
print(f"Ikke-5 predikert som 5: uten topprad {np.mean(model.predict(ikke_5) == 5):.1%}, "
      f"med {np.mean(model.predict(trigget) == 5):.1%}")

# Lineær SHAP for logiten til klasse 5: phi = w_5 * (x - snitt av bakgrunnen)
x = X_test[y_test == 2][0].copy()
x[:8] = 1.0
background = X_train[np.random.default_rng(0).choice(len(X_train), 100, replace=False)]
phi_5 = model.coef_[5] * (x - background.mean(axis=0))
print("SHAP-verdiene for klasse 5, lagt ut som bildet (8 × 8):")
display(phi_5.reshape(8, 8))
print(f"Klasse 5: topprad {phi_5[:8].sum():+.2f}, resten av bildet {phi_5[8:].sum():+.2f}")
print("Sannsynligheter:", model.predict_proba(x[None, :])[0].round(3))
```

Også her gir topraden det klart største bidraget til klasse 5: i 8 × 8-rutenettet er den øverste raden tydelig farget, mens resten av bildet bidrar lite. Prøv å endre antall forgiftede bilder fra 150 til 20 og se hvor mye som skal til.

::: oppgave Hvilket hus er dyrest uten inntekt?
Tallene under kommer fra en `TreeExplainer` (som prosjektets houseA og houseB). Regn ut:
1. `pred_A` og `pred_B`: prediksjonene, fra baseline og SHAP-verdiene.
2. `uten_A` og `uten_B`: prediksjonene når bidraget fra Inntekt (første feature) trekkes fra.
3. `dyrest_uten`: `"A"` eller `"B"`, det huset som er dyrest uten inntektsbidraget.
```start
import numpy as np

features = ["Inntekt", "Alder", "Avstand", "Rom"]
baseline = 2.07
shap_A = np.array([1.08, -0.07, 0.37, -0.27])
shap_B = np.array([-0.32, 0.02, 1.15, 0.06])

pred_A = ...
pred_B = ...
uten_A = ...
uten_B = ...
dyrest_uten = ...
```
```losning
import numpy as np

features = ["Inntekt", "Alder", "Avstand", "Rom"]
baseline = 2.07
shap_A = np.array([1.08, -0.07, 0.37, -0.27])
shap_B = np.array([-0.32, 0.02, 1.15, 0.06])

pred_A = baseline + shap_A.sum()
pred_B = baseline + shap_B.sum()
uten_A = pred_A - shap_A[0]
uten_B = pred_B - shap_B[0]
dyrest_uten = "A" if uten_A > uten_B else "B"
```
```test
import numpy as np
assert np.isclose(pred_A, 3.18) and np.isclose(pred_B, 2.98), "Prediksjonen er baseline + summen av SHAP-verdiene: A = 3.18, B = 2.98."
assert np.isclose(uten_A, 2.10), f"uten_A skal være 3.18 - 1.08 = 2.10, ikke {uten_A}."
assert np.isclose(uten_B, 3.30), f"uten_B skal være 2.98 - (-0.32) = 3.30, ikke {uten_B}. Å trekke fra et negativt bidrag øker prediksjonen."
assert dyrest_uten == "B", "Uten inntektsbidraget er hus B dyrest."
```
:::

::: quiz
Du skal forklare (i) en GradientBoostingRegressor, (ii) et nevralt nett i PyTorch og (iii) en modell du bare når via et API som returnerer prediksjoner. Hvilken kombinasjon av explainers er mulig og mest hensiktsmessig?
- [x] (i) TreeExplainer, (ii) DeepExplainer, (iii) KernelExplainer
- [ ] (i) TreeExplainer, (ii) DeepExplainer, (iii) DeepExplainer
- [ ] (i) DeepExplainer, (ii) TreeExplainer, (iii) KernelExplainer
- [ ] (i) KernelExplainer, (ii) KernelExplainer, (iii) TreeExplainer
? TreeExplainer er eksakt og rask for trær. DeepExplainer trenger nettverkets lag og gradienter, som vi ikke har via et API. KernelExplainer trenger bare prediksjoner og virker for alt.
:::

::: quiz
En bildeklassifiserer gir 99.7 % sannsynlighet for klassen 5 på et bilde som tydelig viser en 2-er. SHAP-plottet viser at nesten hele bidraget kommer fra en hvit linje øverst i bildet. Hva er riktig?
- [ ] Vi kan stole på prediksjonen, fordi sannsynligheten er svært høy
- [ ] SHAP-verdiene er feil, fordi DeepExplainer ikke er modellagnostisk
- [x] Modellen har lært en snarvei (en bakdør i treningsdataene), og den høye sannsynligheten sier ikke at prediksjonen er riktig
- [ ] Modellen har høy accuracy på testdata, så feilen må ligge i bildet
? Softmax-sannsynligheten sier hvor godt input passer mønstre modellen har lært, ikke om svaret er riktig. Forklaringen avslører at mønsteret er en artefakt. Accuracy på rene testdata avslører ikke bakdøren.
:::

::: quiz
Hva er shape på `explainer.shap_values(X)` fra en `DeepExplainer` når `X` har 4 MNIST-bilder (784 piksler) og nettverket har 10 utganger?
- [ ] `(4, 784)`
- [x] `(4, 784, 10)`
- [ ] `(10, 784)`
- [ ] `(4, 10)`
? Én SHAP-verdi per bilde, per piksel og per klasse. Derfor gjør prosjektet `reshape(num_samples, 28, 28, 10, 1)` før `image_plot`.
:::

::: quiz
`shap.plots.scatter(shap_values[:, "Alder"])` viser at SHAP-verdien for Alder synker fra 0 til 30 år og stiger igjen etter 30. Hva kan vi si?
- [ ] Alder er ikke viktig, fordi effekten går begge veier og blir 0 i snitt
- [x] Modellen har lært en ikke-monoton sammenheng: både nye og gamle hus får høyere prediksjon enn hus rundt 30 år
- [ ] Høy alder gir alltid lavere prediksjon
- [ ] En lineær modell ville gitt samme scatter-plott
? Scatter-plottet viser SHAP-verdien mot featureverdien. En V-form er ikke monoton. En lineær modell gir alltid en rett linje ($\phi = \beta(x - \text{E}[x])$), så den kan ikke lære dette.
:::

::: quiz
SHAP-forklaringen sier at houseB blir dyrest hvis vi trekker fra bidraget fra Inntekt. Når modellen trenes på nytt **uten** Inntekt, er houseA fortsatt dyrest. Hva er den mest sannsynlige forklaringen?
- [ ] SHAP-verdiene er regnet ut feil
- [ ] Den nye modellen er overtilpasset
- [x] Andre features er korrelert med Inntekt, så den nye modellen henter inntektsinformasjonen derfra
- [ ] TreeExplainer er ikke eksakt for boosting-modeller
? SHAP forklarer den modellen vi har. En ny modell uten featuren kan bruke korrelerte features (rom, beliggenhet) som erstatning. Det er nettopp det som skjedde med MedInc i prosjekt 4.
:::

::: oppsummering
- `shap.TreeExplainer(model)` for trær og boosting: eksakt og rask. Uten bakgrunn er baseline snittprediksjonen på treningsdata.
- Notatenes stil: `shap.TreeExplainer(regressor, data=X_train)`, `explainer.shap_values(X_test)` (numpy-array), `explainer.expected_value`, `shap.force_plot(...)`.
- `shap.plots.scatter` viser monotoni. Trær kan fange topper og knekker som en lineær modell ikke kan.
- Å trekke fra et SHAP-bidrag er ikke det samme som å trene uten featuren.
- `shap.DeepExplainer(model, background)` for nevrale nett: shape `(n, piksler, klasser)`, forklarer logits. Modellspesifikk, fordi den bruker lagene.
- En forklaring kan avsløre en **bakdør** som accuracy og høy sannsynlighet skjuler.
:::

--- LEKSJON 12.5 | lime | LIME: lokale surrogatmodeller
En **surrogatmodell** er en enkel, tolkbar modell $g$ som etterligner en komplisert modell $f$. Det er vanskelig for en rett linje å etterligne $f$ overalt, men i et lite **nabolag** rundt ett punkt oppfører de fleste modeller seg nesten lineært. Jorda er rund, men bakken ser flat ut når du står på den.

**LIME** (Local Interpretable Model-agnostic Explanations) gjør dette i fem steg, slik notatene beskriver:
1. Velg punktet $x$ som skal forklares, og en vektfunksjon $\pi_x$ som gir punkter nær $x$ høy vekt.
2. Lag $n$ tilfeldige punkter $z_i$ rundt $x$.
3. Lag labels med **modellen**: $f(z_i)$. Ikke ekte $y$, for $g$ skal etterligne $f$.
4. Tilpass en tolkbar modell $g$ (for eksempel lineær) til punktene, vektet med $\pi_x$.
5. Bruk $g$ som forklaring: koeffisientene sier hvordan $f$ endrer seg rundt $x$.

Formelt minimerer LIME $\mathcal{L}(f, g, \pi_x) + \Omega(g)$: hvor dårlig $g$ etterligner $f$ i nabolaget, pluss hvor komplisert $g$ er.

## LIME i én dimensjon
Her er $f$ en kurvet funksjon, og vi forklarer den i $x = 1$. Punktstørrelsen viser vekten:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def f(x):                                        # "black box"-modellen
    return np.sin(2 * x) + 0.3 * x**2

x0 = 1.0                                         # 1. punktet vi forklarer
rng = np.random.default_rng(0)
z = x0 + rng.normal(0, 1.0, 200)                 # 2. tilfeldige punkter rundt x0
labels = f(z)                                    # 3. labels fra modellen
kernel_width = 0.3
w = np.exp(-((z - x0) ** 2) / kernel_width**2)   # 1. vekt etter avstand til x0
g = LinearRegression().fit(z[:, None], labels, sample_weight=w)   # 4. vektet lineær modell
g_uvektet = LinearRegression().fit(z[:, None], labels)
print("De 8 første perturbasjonene, labelen fra modellen og vekten:")
display(pd.DataFrame({"z": z[:8], "f(z)": labels[:8], "vekt w": w[:8]}).round(3))

print(f"Stigning fra LIME (vektet):  {g.coef_[0]:+.3f}")
print(f"Stigning uten vekter:        {g_uvektet.coef_[0]:+.3f}")
print(f"Sann stigning f'(x0):        {2 * np.cos(2 * x0) + 0.6 * x0:+.3f}")

grid = np.linspace(-2, 4, 200)
plt.plot(grid, f(grid), label="f(x)")
plt.scatter(z, labels, s=40 * w + 2, alpha=0.5, label="z_i (størrelse = vekt)")
lokal = np.linspace(x0 - 0.7, x0 + 0.7, 10)
plt.plot(lokal, g.predict(lokal[:, None]), "r", lw=3, label="g: lokal surrogat")
plt.axvline(x0, color="gray", ls=":")
plt.legend()
plt.show()
```

I tabellen ser du at punkter langt fra $x_0 = 1$ får vekt nær 0, mens punkter nær $x_0$ får vekt nær 1. Den vektede linja følger $f$ rundt $x_0$ og fanger at $f$ faller der. Uten vekter blir linja et dårlig globalt kompromiss, med feil fortegn på stigningen. Prøv `kernel_width = 0.1` og `kernel_width = 3`: smalt nabolag gir en mer lokal, men mer støyete forklaring.

`LinearRegression().fit(X, y, sample_weight=w)` er det eneste nye i koden: hvert punkt teller med vekt `w` i kvadratsummen.

## LIME for boligmodellen
Nå skriver vi LIME som en funksjon og forklarer boosting-modellen fra leksjon 12.4 for houseA og houseB. For tabelldata må vi passe på skalaen: vi perturberer med treningsdataenes standardavvik, og måler avstand i **standardiserte** enheter. Kjernebredden $0.75\sqrt{M}$ er standardvalget i lime-biblioteket. Som tolkbar modell bruker vi `Ridge` (lineær regresjon med litt regularisering, også som i lime):

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

def lag_boligdata(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "Inntekt": rng.gamma(4, 1.0, n), "Alder": rng.integers(1, 53, n),
        "Soverom": rng.normal(1.05, 0.05, n), "Befolkning": rng.gamma(2, 700, n),
        "Avstand": rng.uniform(0, 30, n)})
    y = (0.5 * np.minimum(X["Inntekt"], 7) + 0.01 * X["Alder"] + 2.0 * X["Soverom"]
         + 0.00003 * X["Befolkning"] + 1.5 * np.exp(-((X["Avstand"] - 5) / 4) ** 2)
         + rng.normal(0, 0.3, n))
    return X, y

X, y = lag_boligdata()
model = GradientBoostingRegressor(max_depth=3, random_state=0).fit(X.values, y)

def lime(predict, x, X_train, n=500, kernel_width=None, seed=0):
    rng = np.random.default_rng(seed)
    std = X_train.std(axis=0)
    M = len(x)
    if kernel_width is None:
        kernel_width = 0.75 * np.sqrt(M)
    Z = x + rng.normal(0, 1, (n, M)) * std            # 2. naboer rundt x
    labels = predict(Z)                               # 3. f(z), ikke y
    Z_std = (Z - x) / std                             # standardiserte avvik fra x
    d = np.sqrt((Z_std ** 2).sum(axis=1))             # avstand til x
    w = np.exp(-d**2 / kernel_width**2)               # 1. vekter
    g = Ridge(alpha=1.0).fit(Z_std, labels, sample_weight=w)   # 4. tolkbar modell
    return g.coef_                                    # 5. forklaringen

houseA = np.array([8.0, 20, 1.05, 1500, 25.0])    # rik, langt fra sentrum
houseB = np.array([2.5, 30, 1.05, 1500, 5.0])     # lav inntekt, populær avstand
pd.DataFrame({"houseA": lime(model.predict, houseA, X.values),
              "houseB": lime(model.predict, houseB, X.values)}, index=X.columns).round(3)
```

Koeffisientene er endringen i prediksjonen per standardavvik **rundt huset**. Sammenlign med SHAP-verdiene fra leksjon 12.4 (Inntekt +1.55 og −0.75, Avstand −0.35 og +0.96):
- For houseA er inntekten 8, der prisen har flatet ut (over 7 gir ikke mer i dataene våre). LIME gir derfor en mye mindre koeffisient for Inntekt enn for houseB (0.26 mot 0.85). SHAP gir likevel houseA et stort bidrag (+1.55), fordi 8 er langt over snittet.
- For houseB ligger avstanden (5 km) på **toppen** av kurven. Lokalt er stigningen nesten 0 (eller negativ), så LIME gir en liten koeffisient. SHAP gir +0.96, fordi beliggenheten er mye bedre enn snittet.

LIME og SHAP svarer altså på **ulike spørsmål**. LIME: *hvordan endrer prediksjonen seg hvis featuren endres litt herfra?* SHAP: *hvor mye bidrar featureverdien til at prediksjonen er forskjellig fra snittet?*

## Forbeholdene
Notatene nevner tre forbehold: forklaringen avhenger av hvordan nabolaget defineres, det finnes ingen garanti for at $g$ er en god surrogat, og fordi punktene trekkes tilfeldig, kan to kjøringer gi ulike forklaringer. Vi tester de to siste:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

def lag_boligdata(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "Inntekt": rng.gamma(4, 1.0, n), "Alder": rng.integers(1, 53, n),
        "Soverom": rng.normal(1.05, 0.05, n), "Befolkning": rng.gamma(2, 700, n),
        "Avstand": rng.uniform(0, 30, n)})
    y = (0.5 * np.minimum(X["Inntekt"], 7) + 0.01 * X["Alder"] + 2.0 * X["Soverom"]
         + 0.00003 * X["Befolkning"] + 1.5 * np.exp(-((X["Avstand"] - 5) / 4) ** 2)
         + rng.normal(0, 0.3, n))
    return X, y

X, y = lag_boligdata()
model = GradientBoostingRegressor(max_depth=3, random_state=0).fit(X.values, y)

def lime(predict, x, X_train, n=500, kernel_width=None, seed=0):
    rng = np.random.default_rng(seed)
    std = X_train.std(axis=0)
    M = len(x)
    if kernel_width is None:
        kernel_width = 0.75 * np.sqrt(M)
    Z = x + rng.normal(0, 1, (n, M)) * std
    labels = predict(Z)
    Z_std = (Z - x) / std
    w = np.exp(-(Z_std ** 2).sum(axis=1) / kernel_width**2)
    g = Ridge(alpha=1.0).fit(Z_std, labels, sample_weight=w)
    r2 = g.score(Z_std, labels, sample_weight=w)          # hvor godt g etterligner f
    return np.round(g.coef_, 3), round(r2, 2)

houseB = np.array([2.5, 30, 1.05, 1500, 5.0])
print("Kolonner:", list(X.columns))
for seed in [0, 1, 2]:
    print(f"seed {seed}:          ", *lime(model.predict, houseB, X.values, seed=seed))
for kw in [1.0, 4.0]:
    print(f"kernel_width {kw}: ", *lime(model.predict, houseB, X.values, kernel_width=kw))
```

Tallene varierer fra seed til seed og med kjernebredden. Rangeringen (Inntekt størst) er stabil her, men de små koeffisientene, som Soverom, endrer seg mye. Det siste tallet er vektet R² for surrogaten: rundt 0.7 betyr at $g$ forklarer en god del, men langt fra alt av hvordan $f$ oppfører seg i nabolaget. Sjekk alltid kvaliteten på surrogaten før du stoler på forklaringen.

## lime-biblioteket
I forelesningen brukes biblioteket `lime` på Titanic-data. Koden fra notatene (den er ikke kjørbar her, siden `lime` ikke er installert):

```python statisk ikkekjor
from lime.lime_tabular import LimeTabularExplainer

# Instantiate explainer objects
lime_regressor = LimeTabularExplainer(X_train,
                                      feature_names=features,
                                      verbose=True,
                                      mode='regression')
lime_classifier = LimeTabularExplainer(X_train,
                                       feature_names=features,
                                       verbose=True,
                                       mode='classification')
X_jack = np.array([3, 1, 21])     # Pclass, Sex, Age
X_rose = np.array([1, 0, 17])

# Ulike explainers for de ulike modellene og datapunktene
exp_lime = lime_classifier.explain_instance(X_jack, classifier.predict_proba, num_features=3)
exp_lime = lime_regressor.explain_instance(X_jack, regressor.predict, num_features=3)
exp_lime = lime_classifier.explain_instance(X_rose, classifier.predict_proba, num_features=3)
exp_lime = lime_regressor.explain_instance(X_rose, regressor.predict, num_features=3)
print(exp_lime.as_list())         # liste med (regel, vekt), for eksempel ('Pclass <= 1.00', 0.31)
```

- `LimeTabularExplainer(X_train, ...)` bruker treningsdataene til å finne snitt og standardavvik for perturbasjonene.
- `explain_instance(x, predict_fn, num_features=3)` tar **funksjonen** som lager prediksjoner. Det er derfor LIME er modellagnostisk: den kaller bare `predict` eller `predict_proba`.
- For klassifisering gis `predict_proba` (sannsynligheter), ikke `predict` (klasser), slik at surrogaten har en glatt størrelse å etterligne.
- Biblioteket deler som standard kontinuerlige features inn i kvartiler, så forklaringen blir regler som `Age > 28.00` med en vekt.

::: oppgave LIME-vekter
Lag `lime_vekter(Z, x, std, kernel_width)` som returnerer én vekt per rad i `Z`: standardiser avvikene `(Z - x) / std`, regn ut den euklidske avstanden $d$ for hver rad, og returner $\exp(-d^2 / \text{kernel\_width}^2)$.
```start
import numpy as np

def lime_vekter(Z, x, std, kernel_width):
    return ...
```
```losning
import numpy as np

def lime_vekter(Z, x, std, kernel_width):
    d = np.sqrt((((Z - x) / std) ** 2).sum(axis=1))
    return np.exp(-d**2 / kernel_width**2)
```
```test
import numpy as np
_x = np.array([1.0, 10.0])
_std = np.array([1.0, 10.0])
_Z = np.array([[1.0, 10.0], [2.0, 10.0], [1.0, 30.0], [4.0, 50.0]])
_w = np.asarray(lime_vekter(_Z, _x, _std, 1.0))
assert _w.shape == (4,), f"Det skal være én vekt per rad i Z, shape (4,), ikke {_w.shape}."
assert np.isclose(_w[0], 1.0), "Punktet x selv skal ha vekt 1 (avstand 0)."
assert np.isclose(_w[1], np.exp(-1)), "Ett standardavvik unna i første feature skal gi vekt exp(-1)."
assert np.isclose(_w[2], np.exp(-4)), "20 enheter unna i andre feature er 2 standardavvik (std = 10), så vekten skal være exp(-4). Husk å dele på std."
assert np.isclose(_w[3], np.exp(-25)), "Siste punkt er (3, 4) standardavvik unna, avstand 5, vekt exp(-25)."
```
:::

::: oppgave Tilpass surrogaten
`Z_std` er standardiserte nabopunkter, `labels` er modellens prediksjoner på dem, og `w` er vektene. Tilpass en `LinearRegression` med `sample_weight=w`, lagre koeffisientene i `coef`, og navnet på featuren med størst absoluttverdi i `viktigst`.
```start
import numpy as np
from sklearn.linear_model import LinearRegression

features = ["Inntekt", "Alder", "Avstand"]
rng = np.random.default_rng(0)
Z_std = rng.normal(0, 1, (300, 3))
labels = 2 + 0.8 * Z_std[:, 0] + 0.1 * Z_std[:, 1] - 0.5 * Z_std[:, 2] + 0.3 * Z_std[:, 2] ** 2
w = np.exp(-(Z_std ** 2).sum(axis=1) / 1.3**2)

coef = ...
viktigst = ...
```
```losning
import numpy as np
from sklearn.linear_model import LinearRegression

features = ["Inntekt", "Alder", "Avstand"]
rng = np.random.default_rng(0)
Z_std = rng.normal(0, 1, (300, 3))
labels = 2 + 0.8 * Z_std[:, 0] + 0.1 * Z_std[:, 1] - 0.5 * Z_std[:, 2] + 0.3 * Z_std[:, 2] ** 2
w = np.exp(-(Z_std ** 2).sum(axis=1) / 1.3**2)

g = LinearRegression().fit(Z_std, labels, sample_weight=w)
coef = g.coef_
viktigst = features[np.argmax(np.abs(coef))]
```
```test
import numpy as np
from sklearn.linear_model import LinearRegression
_g = LinearRegression().fit(Z_std, labels, sample_weight=w)
assert np.asarray(coef).shape == (3,), "coef skal ha én koeffisient per feature."
assert np.allclose(coef, _g.coef_, atol=1e-6), "Husk sample_weight=w når du tilpasser LinearRegression."
assert viktigst == "Inntekt", f"Inntekt har størst |koeffisient|, ikke {viktigst}. Bruk np.argmax(np.abs(coef))."
```
:::

::: quiz
Hvilket utsagn om LIME er korrekt? (Eksamen 2024)
- [ ] LIME produserer globale feature importances for en hvilken som helst modell
- [x] LIME tilpasser en tolkbar modell i nabolaget rundt et enkelt datapunkt
- [ ] LIME minimerer kompleksiteten til modellen slik at den kan tolkes direkte
- [ ] LIME brukes til å erstatte nevrale nett med beslutningstrær i produksjon
? LIME er lokal (ett punkt), post-hoc og modellagnostisk. Surrogaten brukes bare som forklaring, den erstatter ikke modellen.
:::

::: quiz
I steg 3 av LIME lages labels for de perturberte punktene $z_i$. Hvilke labels brukes?
- [ ] De ekte targetene $y$ fra treningsdataene som ligger nærmest $z_i$
- [x] Modellens prediksjoner $f(z_i)$
- [ ] Surrogatmodellens prediksjoner $g(z_i)$
- [ ] Vektene $\pi_x(z_i)$
? Poenget er at $g$ skal etterligne **modellen** $f$, ikke virkeligheten. De perturberte punktene har dessuten ingen ekte $y$.
:::

::: quiz
Et team lager mange varianter av et datapunkt $x^*$ ved å endre featureverdiene litt, lar et nevralt nett predikere på dem, og tilpasser et **grunt beslutningstre** til disse prediksjonene. Forklaringen er splittene i treet. Hva slags metode er dette?
- [x] En lokal surrogatmodell, der nettverket rundt $x^*$ erstattes av en enklere, tolkbar modell
- [ ] En modellspesifikk forklaring, fordi den bruker nettverkets vekter
- [ ] SHAP, fordi prediksjonen fordeles mellom featurene
- [ ] En global forklaring, fordi beslutningstreet gjelder for alle input
? Det er LIME-oppskriften med et tre i stedet for en lineær modell som $g$. Metoden kaller bare nettverkets prediksjoner, så den er modellagnostisk, og den gjelder bare nabolaget rundt $x^*$.
:::

::: oppsummering
- LIME: perturber rundt $x$ → prediker med $f$ → vekt etter avstand (kjerne) → tilpass vektet tolkbar modell → les av koeffisientene.
- `LinearRegression().fit(Z, f(Z), sample_weight=w)` eller `Ridge`. Mål avstand i standardiserte enheter.
- Lokal, post-hoc og modellagnostisk (trenger bare `predict` / `predict_proba`).
- Forklaringen avhenger av nabolaget (kernel width) og av tilfeldige trekk. Sjekk hvor godt surrogaten passer.
- LIME gir lokal **stigning**. SHAP gir **bidrag** i forhold til snittet. De kan være svært forskjellige.
- Biblioteket: `LimeTabularExplainer(X_train, feature_names=..., mode='regression')` og `explain_instance(x, model.predict, num_features=3)`.
:::

--- LEKSJON 12.6 | konsepter | Konseptdeteksjon: CAV og TCAV
SHAP og LIME forklarer med **input-features**: piksler, ord eller kolonner. Mennesker tenker heller i **konsepter**: «stripete», «skygge», «formue». Et konsept er ikke en feature: dataene består av features, mens konsepter er implisitte. Notatene beskriver hvordan vi kan finne ut om et nevralt nett har en representasjon av et konsept, ved å se på aktiveringene i et skjult lag.

**CAV** (concept activation vector):
1. Samle et datasett **med** konseptet og et **tilfeldig** datasett uten.
2. Send begge gjennom nettverket og hent aktiveringene $f_l(x)$ i lag $l$.
3. Tren en lineær klassifiserer (en **probe**) som skiller dem. Høy accuracy betyr at konseptet er (lineært) representert i laget.
4. Vektoren som står vinkelrett på beslutningsflaten (probens koeffisienter) er CAV-en $v_C^l$.

Vi bruker sklearns `MLPClassifier` på sifrene fra `load_digits`, og konseptet «**strek øverst**» (hele topraden er hvit), som i bakdøreksemplet. Aktiveringene i første skjulte lag regner vi selv ut fra vektene: `mlp.coefs_[0]` og `mlp.intercepts_[0]` er vektene og biasen i første lag (som `nn.Linear`), og ReLU er `np.maximum(0, ...)`.

```python
import warnings
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")      # MLP klager over at den ikke har konvergert helt

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X / 16.0, y, test_size=0.3, random_state=0)
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=150, random_state=0).fit(X_train, y_train)
print(f"Test accuracy: {mlp.score(X_test, y_test):.3f}")

def lag1(X):                           # aktiveringene f_l(x) i første skjulte lag
    return np.maximum(0, X @ mlp.coefs_[0] + mlp.intercepts_[0])

# Sjekk at vi har forstått nettet: regn hele veien til logits selv
logits = np.maximum(0, lag1(X_test) @ mlp.coefs_[1] + mlp.intercepts_[1]) @ mlp.coefs_[2] + mlp.intercepts_[2]
print("Samme prediksjoner som mlp.predict:", np.array_equal(logits.argmax(axis=1), mlp.predict(X_test)))

# 1. Probing-data: tilfeldige bilder, og de samme bildene med hvit topprad
rng = np.random.default_rng(0)
tilfeldig = X_train[rng.choice(len(X_train), 200, replace=False)]
konsept = tilfeldig.copy()
konsept[:, :8] = 1.0
print("Et tilfeldig bilde og det samme bildet med konseptet 'strek øverst':")
display(tilfeldig[0].reshape(8, 8))
display(konsept[0].reshape(8, 8))

# 2. Aktiveringer, 3. lineær probe
A = np.vstack([lag1(konsept), lag1(tilfeldig)])            # shape (400, 32)
labels = np.r_[np.ones(200), np.zeros(200)]                  # 1 = konsept, 0 = tilfeldig
A_tr, A_te, l_tr, l_te = train_test_split(A, labels, test_size=0.3, random_state=0)
probe = LogisticRegression(max_iter=1000).fit(A_tr, l_tr)
print(f"Probe-accuracy for 'strek øverst': {probe.score(A_te, l_te):.2f}")

# 4. CAV: normalisert normalvektor til beslutningsflaten
cav = probe.coef_[0] / np.linalg.norm(probe.coef_[0])
print("CAV har shape", cav.shape)

# Kontroll: et "konsept" som bare er andre tilfeldige bilder
andre = X_train[rng.choice(len(X_train), 200, replace=False)]
A2 = np.vstack([lag1(andre), lag1(tilfeldig)])
A_tr, A_te, l_tr, l_te = train_test_split(A2, labels, test_size=0.3, random_state=0)
print(f"Probe-accuracy for tilfeldig 'konsept': {LogisticRegression(max_iter=1000).fit(A_tr, l_tr).score(A_te, l_te):.2f}")
```

Proben skiller konseptbildene fra de tilfeldige nesten perfekt, så konseptet er lineært representert i laget. For et «konsept» som bare er andre tilfeldige bilder er accuracy rundt 0.5, altså gjetting. Det er en viktig kontroll: høy probe-accuracy er bare interessant når et meningsløst konsept gir lav.

## TCAV: påvirker konseptet prediksjonen?
At et konsept finnes i laget, betyr ikke at modellen **bruker** det. **TCAV** (Testing with CAVs) måler det. La $h_{l,k}$ være resten av nettverket, fra lag $l$ til utgangen for klasse $k$. **Konseptsensitiviteten** er den retningsderiverte i retning CAV-en:

$$S_{C,k,l}(x) = \nabla h_{l,k}\big(f_l(x)\big) \cdot v_C^l$$

Den er positiv hvis det å flytte aktiveringene litt mot konseptet øker utgangen for klasse $k$. **TCAV-skåren** er andelen av bildene i klasse $k$ der sensitiviteten er positiv:

$$\text{TCAV}_{C,k,l} = \frac{|\{x \in X_k : S_{C,k,l}(x) > 0\}|}{|X_k|}$$

For vårt lille nett kan gradienten skrives ut med kjerneregelen: logit $k$ = $\text{relu}(a W_1 + b_1) \cdot W_2[:, k] + b_2$, så $\nabla_a = W_1 \big(\mathbb{1}[a W_1 + b_1 > 0] \odot W_2[:, k]\big)$.

```python
import warnings
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X / 16.0, y, test_size=0.3, random_state=0)
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=150, random_state=0).fit(X_train, y_train)

def lag1(X):
    return np.maximum(0, X @ mlp.coefs_[0] + mlp.intercepts_[0])

rng = np.random.default_rng(0)
tilfeldig = X_train[rng.choice(len(X_train), 200, replace=False)]
konsept = tilfeldig.copy()
konsept[:, :8] = 1.0
A = np.vstack([lag1(konsept), lag1(tilfeldig)])
labels = np.r_[np.ones(200), np.zeros(200)]
probe = LogisticRegression(max_iter=1000).fit(A, labels)
cav = probe.coef_[0] / np.linalg.norm(probe.coef_[0])

# ---------- nytt: gradient og TCAV ----------
def gradient_lag1(a1, k):
    """Gradienten av logiten for klasse k med hensyn på aktiveringene a1 i lag 1. Shape (n, 32)."""
    z2 = a1 @ mlp.coefs_[1] + mlp.intercepts_[1]
    return ((z2 > 0) * mlp.coefs_[2][:, k]) @ mlp.coefs_[1].T

for k in range(10):
    a1 = lag1(X_test[y_test == k])                 # alle testbilder av klasse k
    S = gradient_lag1(a1, k) @ cav                 # konseptsensitivitet per bilde
    tcav = np.mean(S > 0)
    print(f"Klasse {k}: TCAV('strek øverst') = {tcav:.2f}")
```

For 5 og 7, som har en vannrett strek øverst, er skåren høy: å legge til «strek øverst» gjør nettet mer sikker på 5 og 7 for (nesten) alle bildene. For 0 og 8, som er runde øverst, er skåren lav: konseptet taler **imot** dem. Noen andre klasser får uventede verdier. TCAV-skåren for én enkelt CAV er støyete. Den opprinnelige metoden trener derfor mange CAV-er mot ulike tilfeldige datasett og tester om skåren skiller seg signifikant fra tilfeldige konsepter.

::: husk
TCAV er en **global** forklaring (den oppsummerer en hel klasse), og **modellspesifikk** (den trenger aktiveringene og gradientene inni nettverket). SHAP og LIME er lokale og modellagnostiske.
:::

## Begrepene oppsummert
Notatene avslutter XAI-delen med en tabell over begreper. Her er den med eksempler fra denne delen:

| Begrep | Betydning | Eksempler |
|---|---|---|
| Post-hoc | forklaringen lages etter at modellen er trent | alle metodene i Del 12 |
| Lokal | forklarer én prediksjon | SHAP-verdier for ett punkt, waterfall, LIME |
| Global | forklarer hele modellen | permutation importance, TCAV, (beeswarm, satt sammen av lokale) |
| Modellagnostisk | trenger bare input og output | KernelExplainer, LIME, permutation importance |
| Modellspesifikk | bruker parametre, arkitektur eller gradienter | koeffisienter, TreeExplainer, DeepExplainer, CAV/TCAV |
| Surrogatmodell | en tolkbar modell etterligner den ekte, ofte lokalt | LIME |

::: oppgave Regn ut en TCAV-skår
`grads` inneholder gradientene av logiten for klasse $k$ med hensyn på aktiveringene i et lag, for 6 bilder av klasse $k$ (shape `(6, 4)`). `w` er probens koeffisienter. Normaliser `w` til lengde 1 og lagre i `cav`, regn ut konseptsensitiviteten for hvert bilde i `S` (shape `(6,)`), og lagre TCAV-skåren (andelen positive) i `tcav`.
```start
import numpy as np

grads = np.array([[ 0.5, -0.2,  0.1,  0.0],
                  [ 0.3,  0.4, -0.1,  0.2],
                  [-0.6,  0.1,  0.0,  0.1],
                  [ 0.2,  0.2,  0.2,  0.2],
                  [-0.1, -0.3,  0.4,  0.0],
                  [ 0.7,  0.0, -0.2,  0.1]])
w = np.array([3.0, 0.0, 0.0, 4.0])

cav = ...
S = ...
tcav = ...
```
```losning
import numpy as np

grads = np.array([[ 0.5, -0.2,  0.1,  0.0],
                  [ 0.3,  0.4, -0.1,  0.2],
                  [-0.6,  0.1,  0.0,  0.1],
                  [ 0.2,  0.2,  0.2,  0.2],
                  [-0.1, -0.3,  0.4,  0.0],
                  [ 0.7,  0.0, -0.2,  0.1]])
w = np.array([3.0, 0.0, 0.0, 4.0])

cav = w / np.linalg.norm(w)
S = grads @ cav
tcav = np.mean(S > 0)
```
```test
import numpy as np
assert np.allclose(cav, [0.6, 0, 0, 0.8]), "cav skal være w delt på lengden av w (np.linalg.norm(w) = 5)."
assert np.asarray(S).shape == (6,), "S skal ha én verdi per bilde, shape (6,). Bruk grads @ cav."
assert np.allclose(S, [0.3, 0.34, -0.28, 0.28, -0.06, 0.5]), f"S skal være prikkproduktet av hver gradient med cav, fikk {S}."
assert np.isclose(tcav, 4 / 6), f"4 av 6 sensitiviteter er positive, så TCAV = 0.667, ikke {tcav}."
```
:::

::: quiz
Hva betyr en TCAV-skår på 0.8 for konseptet «prikkete» og klassen «dalmatiner»?
- [ ] Modellen klassifiserer 80 % av dalmatinerbildene riktig
- [ ] 80 % av pikslene i dalmatinerbildene er prikker
- [x] For 80 % av dalmatinerbildene øker prediksjonen for «dalmatiner» når aktiveringene flyttes i retning av konseptet «prikkete»
- [ ] Proben skiller prikkete bilder fra tilfeldige med 80 % accuracy
? TCAV teller andelen bilder i klassen der den retningsderiverte langs CAV-en er positiv. Probe-accuracy sier om konseptet er representert i laget, ikke om det påvirker prediksjonen.
:::

::: quiz
Du lager en `shap.KernelExplainer` for en XGBoost-modell og et force plot for én bestemt kunde. Hvordan klassifiseres denne forklaringen?
- [x] Post-hoc, lokal og modellagnostisk
- [ ] Post-hoc, global og modellspesifikk
- [ ] Innebygd, lokal og modellspesifikk
- [ ] Post-hoc, lokal og modellspesifikk, fordi XGBoost er en tremodell
? Forklaringen lages etter treningen (post-hoc), gjelder én kunde (lokal), og KernelExplainer kaller bare modellens prediksjoner (modellagnostisk). Det er explaineren, ikke modelltypen, som avgjør. Med TreeExplainer ville den vært modellspesifikk.
:::

::: quiz
En lineær probe på aktiveringene i et lag skiller bilder med konseptet «stripete» fra tilfeldige bilder med accuracy 0.51. Hva er den beste tolkningen?
- [ ] Konseptet er sterkt representert i laget
- [x] Konseptet er ikke lineært representert i dette laget (proben gjetter)
- [ ] Modellen bruker konseptet i halvparten av prediksjonene
- [ ] TCAV-skåren for konseptet er 0.51
? Med like mange bilder i hver gruppe gir gjetting accuracy rundt 0.5. Da finnes det ingen (lineær) retning i aktiveringsrommet som skiller konseptet ut, og CAV-en er meningsløs.
:::

::: oppsummering
- Konsepter er implisitte, høynivå begreper (stripete, strek øverst). Features er det dataene består av.
- **CAV**: tren en lineær probe som skiller aktiveringer for konseptdata fra tilfeldige data i lag $l$. CAV = normalisert `probe.coef_`.
- Høy probe-accuracy (og lav for tilfeldige «konsepter») betyr at konseptet er lineært representert i laget.
- **TCAV**: andelen bilder i klasse $k$ der gradienten av logiten, prikket med CAV-en, er positiv. `np.mean(grads @ cav > 0)`.
- TCAV er global og modellspesifikk. SHAP og LIME er lokale og modellagnostiske.
:::
