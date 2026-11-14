=== DEL 12 | xai | Forklarbar KI (prosjekt 4)
Du skriver koden bak forklaringene i prosjekt 4 selv: koeffisienter, permutation importance, Shapley-verdier, SHAP, LIME og CAV/TCAV.

--- LEKSJON 12.1 | koeffisienter | Koeffisienter og permutation importance
`linear_model.coef_` er endringen i prediksjonen når en feature øker med **én enhet**, så koeffisientene kan bare sammenlignes når featurene har samme skala. Sammenlign derfor $|\beta_i| \cdot \text{std}(x_i)$, eller tren på standardiserte features med `make_pipeline(StandardScaler(), LinearRegression())`.

::: oppgave Koeffisientene
`lag_boligdata()` lager boligdata som ligner California housing fra prosjekt 4 (pris i 100 000 USD). Tren `linear_model = LinearRegression()` på treningsdataene, lag en Series `koef` med `coef_` og featurenavnene som indeks, og lagre featuren med størst |koeffisient| i `storst_koef`.
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

linear_model = ...
koef = ...
storst_koef = ...
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
koef = pd.Series(linear_model.coef_, index=X_train.columns)
storst_koef = koef.abs().idxmax()
```
```test
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
assert hasattr(linear_model, "coef_"), "linear_model skal være en trent LinearRegression: LinearRegression().fit(X_train, y_train)."
_fasit = LinearRegression().fit(X_train, y_train).coef_
assert np.allclose(linear_model.coef_, _fasit), "Tren modellen på X_train og y_train (ikke testdataene)."
assert isinstance(koef, pd.Series), "koef skal være en pandas Series."
assert list(koef.index) == list(X_train.columns), "Indeksen til koef skal være featurenavnene (X_train.columns)."
assert np.allclose(koef.values, _fasit), "Verdiene i koef skal være linear_model.coef_."
assert storst_koef == "Soverom", f"Soverom har størst |koeffisient| (rundt 2.5), ikke {storst_koef}. Tips: koef.abs().idxmax()."
```
:::

::: oppgave Rangér med |coef| · std
Lag en Series `viktighet` med $|\beta_i| \cdot \text{std}(x_i)$ for hver feature (bruk `linear_model.coef_` og `X_train.std()`), og lagre navnet på den viktigste featuren i `viktigst`.
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

::: oppgave Standardiserte koeffisienter
Tren `scaled_model = make_pipeline(StandardScaler(), LinearRegression())`, lagre koeffisientene til siste steg (`scaled_model[-1].coef_`) i `std_koef`, og featuren med størst |koeffisient| i `viktigst_std`.
```start
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

scaled_model = ...
std_koef = ...
viktigst_std = ...
```
```losning
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

scaled_model = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)
std_koef = scaled_model[-1].coef_
viktigst_std = X_train.columns[np.argmax(np.abs(std_koef))]
```
```test
import numpy as np
from sklearn.linear_model import LinearRegression
_c = LinearRegression().fit(X_train, y_train).coef_ * X_train.std(ddof=0).values
assert np.asarray(std_koef).shape == (5,), "std_koef skal ha én koeffisient per feature. Husk .fit(X_train, y_train) og scaled_model[-1].coef_."
assert np.allclose(std_koef, _c), "De standardiserte koeffisientene skal være coef * std. Tren pipelinen på X_train og y_train."
assert viktigst_std == "Inntekt", f"Etter standardisering er Inntekt viktigst, ikke {viktigst_std}. Tips: X_train.columns[np.argmax(np.abs(std_koef))]."
```
:::

## Permutation importance
`permutation_importance(modell, X_test, y_test, n_repeats=10, random_state=0)` stokker én kolonne om gangen og måler fallet i score (`.importances_mean`, og `.importances` med én kolonne per stokking), så den virker for alle modeller. Den forklarer **modellen**, ikke virkeligheten, og korrelerte features kan dele effekten mellom seg.

::: oppgave Permutation importance for hånd
Regn ut `r2_original` (R² for `linear_model` på `X_test`), lag en kopi `X_perm` av `X_test` der kolonnen `"Inntekt"` er stokket med `rng.permutation`, og lagre fallet i R² i `fall`.
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

::: oppgave Lineær modell mot boosting
Lagre `importances_mean` fra `permutation_importance` (på testdataene, `n_repeats=10`) for `linear_model` i `pi_lin` og for `boosting_model` i `pi_boost`, og lagre featuren boosting-modellen bruker nest mest i `nest_boost`.
```start
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
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
boosting_model = GradientBoostingRegressor(max_depth=3, random_state=0).fit(X_train, y_train)

pi_lin = ...
pi_boost = ...
nest_boost = ...
```
```losning
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
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
boosting_model = GradientBoostingRegressor(max_depth=3, random_state=0).fit(X_train, y_train)

pi_lin = permutation_importance(linear_model, X_test, y_test, n_repeats=10, random_state=0).importances_mean
pi_boost = permutation_importance(boosting_model, X_test, y_test, n_repeats=10, random_state=0).importances_mean
nest_boost = X_test.columns[np.argsort(pi_boost)[-2]]
```
```test
import numpy as np
assert np.asarray(pi_lin).shape == (5,) and np.asarray(pi_boost).shape == (5,), "pi_lin og pi_boost skal ha én verdi per feature (bruk .importances_mean)."
assert np.argmax(pi_lin) == 0 and np.argmax(pi_boost) == 0, "Inntekt skal være viktigst for begge modellene. Bruk X_test og y_test i permutation_importance."
assert 0.8 < pi_lin[0] < 1.5, f"pi_lin for Inntekt skal være rundt 1.16, fikk {pi_lin[0]:.3f}. Har du brukt testdataene?"
assert pi_boost[4] > pi_lin[4] + 0.08, "Boosting-modellen skal bruke Avstand mer enn den lineære (toppen rundt 5 km). Sjekk at pi_boost kommer fra boosting_model."
assert nest_boost == "Avstand", f"Nest viktigst for boosting er Avstand, ikke {nest_boost}."
```
:::

::: oppgave Korrelerte features
Lag `X2_train` og `X2_test` med en ekstra kolonne `"Inntekt2"` = Inntekt + `rng.normal(0, 0.1, ...)`, tren `model2 = LinearRegression()` på dem, og lagre permutation importance for Inntekt (`n_repeats=5`) for `linear_model` i `imp_uten` og for `model2` i `imp_med`.
```start
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
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
rng = np.random.default_rng(1)

X2_train = ...
X2_test = ...
model2 = ...
imp_uten = ...
imp_med = ...
```
```losning
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
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
rng = np.random.default_rng(1)

X2_train = X_train.copy()
X2_train["Inntekt2"] = X2_train["Inntekt"] + rng.normal(0, 0.1, len(X2_train))
X2_test = X_test.copy()
X2_test["Inntekt2"] = X2_test["Inntekt"] + rng.normal(0, 0.1, len(X2_test))
model2 = LinearRegression().fit(X2_train, y_train)
imp_uten = permutation_importance(linear_model, X_test, y_test, n_repeats=5, random_state=0).importances_mean[0]
imp_med = permutation_importance(model2, X2_test, y_test, n_repeats=5, random_state=0).importances_mean[0]
```
```test
import numpy as np
assert "Inntekt2" in X2_train.columns and "Inntekt2" in X2_test.columns, "Både X2_train og X2_test skal ha kolonnen Inntekt2."
assert np.corrcoef(X2_train["Inntekt"], X2_train["Inntekt2"])[0, 1] > 0.99, "Inntekt2 skal være Inntekt pluss litt støy (std 0.1)."
assert len(model2.coef_) == 6, "model2 skal trenes på X2_train (6 features)."
assert 0.8 < imp_uten < 1.5, f"imp_uten skal være rundt 1.16 (Inntekt for linear_model på X_test), fikk {imp_uten:.3f}."
assert imp_med < 0.5 * imp_uten, f"Med en nesten lik kopi deler modellen effekten, så imp_med ({imp_med:.3f}) skal være mye lavere enn imp_uten. Bruk model2 og X2_test."
```
:::

--- LEKSJON 12.2 | shapley | Shapley-verdier for hånd
Shapley-verdien er et vektet snitt av spillerens marginale bidrag: $\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(n - |S| - 1)!}{n!} \big( v(S \cup \{i\}) - v(S) \big)$. I Python lagrer vi $v$ som en dict med `frozenset` som nøkler, og `combinations(andre, k)` gir alle koalisjoner med `k` av de andre spillerne.

::: oppgave To spillere
Med to spillere har begge koalisjonene vekt ½. Regn ut `phi1` og `phi2` fra `v` med formelen.
```start
v = {frozenset(): 0, frozenset({1}): 4, frozenset({2}): 2, frozenset({1, 2}): 10}

phi1 = ...
phi2 = ...
```
```losning
v = {frozenset(): 0, frozenset({1}): 4, frozenset({2}): 2, frozenset({1, 2}): 10}

phi1 = 0.5 * (v[frozenset({1})] - v[frozenset()]) + 0.5 * (v[frozenset({1, 2})] - v[frozenset({2})])
phi2 = 0.5 * (v[frozenset({2})] - v[frozenset()]) + 0.5 * (v[frozenset({1, 2})] - v[frozenset({1})])
```
```test
assert abs(phi1 - 6) < 1e-9, f"phi1 skal være ½(4 - 0) + ½(10 - 2) = 6, ikke {phi1}."
assert abs(phi2 - 4) < 1e-9, f"phi2 skal være ½(2 - 0) + ½(10 - 4) = 4, ikke {phi2}."
```
:::

::: oppgave Shapley-funksjonen
Fullfør `shapley(v, spillere)` ved å fylle inn vekten og det vektede marginale bidraget. Testen sjekker taxieksempelet fra notatene (1, 3 og 6 km) og katteoppgaven.
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

## Aksiomene
Shapley-verdiene oppfyller **effektivitet** ($\sum_i \phi_i = v(N)$), **symmetri** (like bidrag gir lik verdi), **nullspiller** (bidrar du aldri, får du 0) og **additivitet**.

::: oppgave Sjekk aksiomene
Lag `v` for tre spillere der en koalisjon er verdt 6 hvis spiller 1 eller 2 er med og ellers 0 (spiller 3 bidrar aldri), og lagre `phi = shapley(v, [1, 2, 3])`.
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
                vekt = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                total += vekt * (v[S | {i}] - v[S])
        phi[i] = total
    return phi

v = ...
phi = ...
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

v = {frozenset(S): (6 if (1 in S or 2 in S) else 0)
     for k in range(4) for S in combinations([1, 2, 3], k)}
phi = shapley(v, [1, 2, 3])
```
```test
assert isinstance(v, dict) and len(v) == 8, "v skal være en dict med alle 2^3 = 8 koalisjoner (frozenset som nøkler)."
assert v[frozenset()] == 0 and v[frozenset({3})] == 0 and v[frozenset({1, 3})] == 6, "v(S) skal være 6 når 1 eller 2 er med, ellers 0."
assert abs(phi[3]) < 1e-9, f"Nullspiller: spiller 3 skal få 0, fikk {phi[3]}."
assert abs(phi[1] - phi[2]) < 1e-9, "Symmetri: spiller 1 og 2 bidrar likt og skal få lik verdi."
assert abs(phi[1] + phi[2] + phi[3] - 6) < 1e-9, "Effektivitet: summen skal være v({1,2,3}) = 6."
```
:::

## En modell som spill
Med features som spillere er $v(S)$ snittet av `f(X_mix)`, der `X_mix` er bakgrunnsdataene med kolonnene i $S$ byttet ut med verdiene fra punktet `x`. Baseline er $\phi_0 = v(\varnothing)$ (snittprediksjonen over bakgrunnen), og $\phi_0 + \sum_j \phi_j = f(x)$.

::: oppgave Koalisjonsverdier fra en modell
Fullfør `v_modell(S)` slik at kolonnene `j` i `S` får verdien `x[j]` og resten kommer fra `background`, og lagre `baseline` $= v(\varnothing)$.
```start
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

def f(X):                                  # lineær i x0, samspill mellom x1 og x2
    return 3 * X[:, 0] + 2 * X[:, 1] * X[:, 2]

rng = np.random.default_rng(0)
background = rng.normal(0, 1, size=(100, 3))
x = np.array([1.0, 2.0, 1.5])

def v_modell(S):
    X_mix = background.copy()
    # sett kolonnene i S til verdiene fra x
    return ...

v = {frozenset(S): v_modell(S) for k in range(4) for S in combinations(range(3), k)}
phi = shapley(v, [0, 1, 2])
baseline = ...
```
```losning
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

def f(X):                                  # lineær i x0, samspill mellom x1 og x2
    return 3 * X[:, 0] + 2 * X[:, 1] * X[:, 2]

rng = np.random.default_rng(0)
background = rng.normal(0, 1, size=(100, 3))
x = np.array([1.0, 2.0, 1.5])

def v_modell(S):
    X_mix = background.copy()
    for j in S:
        X_mix[:, j] = x[j]
    return f(X_mix).mean()

v = {frozenset(S): v_modell(S) for k in range(4) for S in combinations(range(3), k)}
phi = shapley(v, [0, 1, 2])
baseline = v[frozenset()]
```
```test
import numpy as np
assert np.isclose(v_modell([]), f(background).mean()), "v_modell([]) skal være snittet av f over bakgrunnen."
assert np.isclose(v_modell([0, 1, 2]), f(x[None, :])[0]), "Med alle features i S skal v_modell gi f(x)."
_b = background.copy(); _b[:, 1] = x[1]
assert np.isclose(v_modell([1]), f(_b).mean()), "v_modell([1]) skal bytte ut kolonne 1 med x[1] og ta snittet av f."
assert np.isclose(baseline, f(background).mean()), "baseline skal være v(tom mengde) = snittprediksjonen over bakgrunnen."
assert np.isclose(baseline + sum(phi.values()), f(x[None, :])[0]), "baseline + summen av phi skal bli f(x)."
```
:::

## Lineær modell: en snarvei
For en lineær modell $f(x) = b + \sum_i \beta_i x_i$ blir Shapley-verdien $\phi_i = \beta_i \, (x_i - \text{E}[x_i])$, med snittet over bakgrunnen, så en stor koeffisient gir bare stor SHAP-verdi når $x_i$ ligger langt fra snittet.

::: oppgave Lineær SHAP med numpy
Regn ut SHAP-verdiene `phi` for punktet `x` med $\phi_i = \beta_i (x_i - \text{E}[x_i])$ (snittet kolonnevis over `background`), og `baseline` (snittprediksjonen over bakgrunnen).
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
assert np.isclose(baseline, (b + background @ beta).mean()), "baseline skal være snittet av b + background @ beta (ikke bare b)."
assert np.isclose(baseline + np.sum(phi), f_x), "baseline + sum(phi) skal bli f_x (additivitet)."
```
:::

## Hvorfor SHAP må tilnærme
Med $M$ features finnes $2^M$ koalisjoner, så shap regner bare eksakt for lineære modeller og trær og tilnærmer ellers. For bilder slås piksler sammen til **superpiksler** som hver er én feature.

::: oppgave Antall koalisjoner
Lagre antall koalisjoner for 10 features i `antall_10`, antall sifre i $2^{784}$ (et MNIST-bilde) i `sifre_mnist` (bruk `len(str(...))`), og antall koalisjoner når et 28 × 28-bilde deles i superpiksler på 4 × 4 piksler i `antall_super`.
```start
antall_10 = ...
sifre_mnist = ...
antall_super = ...
```
```losning
antall_10 = 2 ** 10
sifre_mnist = len(str(2 ** 784))
antall_super = 2 ** ((28 // 4) * (28 // 4))
```
```test
assert antall_10 == 1024, f"10 features gir 2**10 = 1024 koalisjoner, ikke {antall_10}."
assert sifre_mnist == 237, f"2**784 har 237 sifre, fikk {sifre_mnist}. Bruk len(str(2 ** 784))."
assert antall_super == 2 ** 49, "28 × 28 delt i 4 × 4-blokker gir 7 · 7 = 49 superpiksler, altså 2**49 koalisjoner."
```
:::

--- LEKSJON 12.3 | shap-biblioteket | shap-biblioteket
`shap.Explainer(model, background)` lager en explainer (for en lineær modell blir det `LinearExplainer`), og `shap_values = explainer(X)` gir et Explanation-objekt med `.values` (n, M), `.base_values` (n,) og `.data`, der `shap_values[0]` er én forklaring og `shap_values[:, "Inntekt"]` én feature. shap kjører ikke i nettleseren, så kjør blokken i Jupyter og øv ideen med numpy under.

```python statisk
import numpy as np, pandas as pd, shap
from sklearn.linear_model import LinearRegression
rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(500, 3)), columns=["Inntekt", "Alder", "Avstand"])
y = 2 * X["Inntekt"] + 0.5 * X["Alder"] - X["Avstand"] + rng.normal(0, 0.1, 500)
model = LinearRegression().fit(X, y)
background = shap.utils.sample(X, 100, random_state=0)   # 100 tilfeldige rader som bakgrunn
explainer = shap.Explainer(model, background)
shap_values = explainer(X)
print(type(explainer).__name__, shap_values.shape, shap_values.base_values[0].round(3))
```
```output
LinearExplainer (500, 3) -0.269
```

::: oppgave SHAP-verdier for boligmodellen
Regn ut SHAP-verdiene `values` (shape `(200, 5)`) for alle testhusene med $\phi = \beta (x - \text{E}[x])$, `base_value` (snittprediksjonen over `background`), og lagre i `additiv` om `base_value + values.sum(axis=1)` er lik `linear_model.predict(X_test)` (`np.allclose`).
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
background = X_train.sample(100, random_state=0)

values = ...
base_value = ...
additiv = ...
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
background = X_train.sample(100, random_state=0)

values = linear_model.coef_ * (X_test.values - background.values.mean(axis=0))
base_value = linear_model.predict(background).mean()
additiv = np.allclose(base_value + values.sum(axis=1), linear_model.predict(X_test))
```
```test
import numpy as np
_v = linear_model.coef_ * (X_test.values - background.values.mean(axis=0))
assert np.asarray(values).shape == (200, 5), f"values skal ha shape (200, 5), én rad per testhus. Fikk {np.asarray(values).shape}."
assert np.allclose(values, _v), "values skal være linear_model.coef_ * (X_test - snittet av background per kolonne)."
assert np.isclose(base_value, linear_model.predict(background).mean()), "base_value skal være snittet av linear_model.predict(background)."
assert additiv == True, "additiv skal være True: base_value + values.sum(axis=1) er lik prediksjonene."
```
:::

::: oppgave Additivitet og global viktighet
Lagre i `additiv` om `base_values + values.sum(axis=1)` er lik `preds`, snittet av |SHAP| per feature (det `shap.plots.bar` viser) i `global_viktighet`, og featurenavnene fra viktigst til minst viktig i listen `rekkefolge`.
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
assert additiv == True, "Her stemmer additiviteten: 2.0 + 0.4 = 2.4 osv. Bruk np.allclose(base_values + values.sum(axis=1), preds)."
assert np.allclose(global_viktighet, [0.7, 0.1, 0.45]), f"Snittet av |SHAP| per kolonne skal være [0.7, 0.1, 0.45], fikk {global_viktighet}. Husk np.abs før mean(axis=0)."
assert list(rekkefolge) == ["Inntekt", "Avstand", "Alder"], f"Rekkefølgen skal være Inntekt, Avstand, Alder, fikk {rekkefolge}."
```
:::

## Plottene
`shap.plots.bar` viser snittet av |SHAP| per feature, `beeswarm` alle punktene med farge etter featureverdi (rød = høy) og x-posisjon etter SHAP-verdi, `scatter` SHAP-verdi mot featureverdi for én feature, og `waterfall` én prediksjon fra baseline til $f(x)$. Plottene forklarer **modellen** i punktene du ga den, ikke årsaker i virkeligheten.

```python statisk fortsett
shap.plots.bar(shap_values)                      # global: snitt av |SHAP|
shap.plots.beeswarm(shap_values)                 # alle lokale forklaringer samlet
shap.plots.scatter(shap_values[:, "Avstand"])    # SHAP mot featureverdi
shap.plots.waterfall(shap_values[0])             # lokal: én prediksjon
```

::: oppgave Les et beeswarm-plott med kode
Lagre featuren som havner øverst i beeswarm-plottet (størst snitt av |SHAP|) i `overst`, snitt-SHAP for Alder blant punktene med Alder over medianen i `snitt_hoy` og under i `snitt_lav`, og om høy alder trekker prediksjonen ned i `hoy_trekker_ned`.
```start
import numpy as np

features = ["Inntekt", "Alder", "Avstand"]
rng = np.random.default_rng(0)
data = np.column_stack([rng.gamma(4, 1.0, 300), rng.uniform(1, 53, 300), rng.uniform(0, 30, 300)])
values = np.column_stack([0.3 * (data[:, 0] - 4), -0.05 * (data[:, 1] - 27), -0.02 * (data[:, 2] - 15)])

overst = ...
snitt_hoy = ...
snitt_lav = ...
hoy_trekker_ned = ...
```
```losning
import numpy as np

features = ["Inntekt", "Alder", "Avstand"]
rng = np.random.default_rng(0)
data = np.column_stack([rng.gamma(4, 1.0, 300), rng.uniform(1, 53, 300), rng.uniform(0, 30, 300)])
values = np.column_stack([0.3 * (data[:, 0] - 4), -0.05 * (data[:, 1] - 27), -0.02 * (data[:, 2] - 15)])

overst = features[np.argmax(np.abs(values).mean(axis=0))]
hoy = data[:, 1] > np.median(data[:, 1])
snitt_hoy = values[hoy, 1].mean()
snitt_lav = values[~hoy, 1].mean()
hoy_trekker_ned = snitt_hoy < 0
```
```test
import numpy as np
assert overst == "Alder", f"Alder har størst snitt av |SHAP| og havner øverst, ikke {overst}. Husk np.abs før mean."
_h = data[:, 1] > np.median(data[:, 1])
assert np.isclose(snitt_hoy, values[_h, 1].mean()), "snitt_hoy skal være snittet av SHAP-verdiene i kolonne 1 (Alder) der Alder > medianen."
assert np.isclose(snitt_lav, values[~_h, 1].mean()), "snitt_lav skal være snittet av SHAP-verdiene for Alder der Alder er under medianen."
assert hoy_trekker_ned == True, "Høy alder gir negativ SHAP-verdi (røde punkter til venstre), så hoy_trekker_ned skal være True."
```
:::

::: oppgave SHAP mot |coef| · std
Lag `mean_abs_shap` (snitt av |SHAP| per feature fra `values`) og `coef_std` ($|\beta| \cdot \text{std}$), lagre i `samme_rangering` om `np.argsort` gir samme rekkefølge for begge, og tegn `mean_abs_shap` som liggende stolper med `ax.barh(X.columns, ...)`.
```start
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
values = linear_model.coef_ * (X_test.values - background.values.mean(axis=0))

mean_abs_shap = ...
coef_std = ...
samme_rangering = ...
fig, ax = plt.subplots()
```
```losning
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
values = linear_model.coef_ * (X_test.values - background.values.mean(axis=0))

mean_abs_shap = np.abs(values).mean(axis=0)
coef_std = np.abs(linear_model.coef_) * X_train.std().values
samme_rangering = list(np.argsort(mean_abs_shap)) == list(np.argsort(coef_std))
fig, ax = plt.subplots()
ax.barh(X.columns, mean_abs_shap)
```
```test
import numpy as np
assert np.allclose(mean_abs_shap, np.abs(values).mean(axis=0)), "mean_abs_shap skal være np.abs(values).mean(axis=0)."
assert np.allclose(coef_std, np.abs(linear_model.coef_) * X_train.std().values), "coef_std skal være |coef| ganget med X_train.std()."
assert samme_rangering is not ... and np.all(np.asarray(samme_rangering) == True), "Rangeringen etter snitt av |SHAP| følger |coef| · std, så samme_rangering skal være True."
assert len(ax.patches) == 5, "Tegn én stolpe per feature med ax.barh(X.columns, mean_abs_shap)."
assert np.allclose(sorted(p.get_width() for p in ax.patches), sorted(mean_abs_shap)), "Stolpelengdene skal være mean_abs_shap."
```
:::

--- LEKSJON 12.4 | tree-deep | TreeExplainer og DeepExplainer
`shap.TreeExplainer(model)` er eksakt og rask for trær og boosting, `shap.DeepExplainer(model, background)` bruker lagene i et nevralt nett (modellspesifikk), og `KernelExplainer` trenger bare `predict` (modellagnostisk, men treg). `shap.plots.scatter(shap_values[:, "Avstand"])` viser om SHAP-verdien er monoton i featureverdien, noe trær ikke trenger å være.

```python statisk
import numpy as np, pandas as pd, shap
from sklearn.ensemble import GradientBoostingRegressor
rng = np.random.default_rng(0)
X = pd.DataFrame(rng.uniform(0, 30, size=(500, 2)), columns=["Inntekt", "Avstand"])
y = 0.3 * X["Inntekt"] + 1.5 * np.exp(-((X["Avstand"] - 5) / 4) ** 2)
model = GradientBoostingRegressor(max_depth=3, random_state=0).fit(X, y)
explainer = shap.TreeExplainer(model)     # uten bakgrunn: baseline = snittprediksjon på treningsdata
shap_values = explainer(X)
shap.plots.scatter(shap_values[:, "Avstand"])
```

```python statisk fortsett
# Notatenes stil: bakgrunn via data=, numpy-array ut, baseline i expected_value
explainer_bg = shap.TreeExplainer(model, data=shap.utils.sample(X, 100, random_state=0))
sv = explainer_bg.shap_values(X)
print(type(sv).__name__, sv.shape, round(float(explainer_bg.expected_value), 3))
shap.force_plot(explainer_bg.expected_value, sv[0, :], X.iloc[0], matplotlib=True)
```
```output
ndarray (500, 2) 5.161
```

::: oppgave Er sammenhengen monoton?
`avstand` og `phi_avstand` er data og SHAP-verdier for Avstand fra en TreeExplainer. Lagre snitt-SHAP for hvert 3-km-intervall (0–3, 3–6, …, 27–30) i listen `snitt`, i `monoton` om snittene bare stiger eller bare synker, og starten på intervallet med høyest snitt i `topp_km`.
```start
import numpy as np

rng = np.random.default_rng(0)
avstand = rng.uniform(0, 30, 800)
phi_avstand = 1.3 * np.exp(-((avstand - 5) / 4) ** 2) - 0.35 + rng.normal(0, 0.05, 800)

snitt = ...
monoton = ...
topp_km = ...
```
```losning
import numpy as np

rng = np.random.default_rng(0)
avstand = rng.uniform(0, 30, 800)
phi_avstand = 1.3 * np.exp(-((avstand - 5) / 4) ** 2) - 0.35 + rng.normal(0, 0.05, 800)

snitt = [phi_avstand[(avstand >= lo) & (avstand < lo + 3)].mean() for lo in range(0, 30, 3)]
d = np.diff(snitt)
monoton = bool(np.all(d >= 0) or np.all(d <= 0))
topp_km = 3 * int(np.argmax(snitt))
```
```test
import numpy as np
_s = [phi_avstand[(avstand >= lo) & (avstand < lo + 3)].mean() for lo in range(0, 30, 3)]
assert len(snitt) == 10, f"snitt skal ha 10 verdier (ett per 3-km-intervall), fikk {len(snitt)}."
assert np.allclose(snitt, _s), "snitt[i] skal være snittet av phi_avstand der lo <= avstand < lo + 3, for lo = 0, 3, ..., 27."
assert monoton == False, "SHAP-verdien stiger opp til 5 km og faller etterpå, så sammenhengen er ikke monoton."
assert topp_km == 3, f"Høyest snitt er i intervallet 3–6 km, så topp_km skal være 3, ikke {topp_km}."
```
:::

## houseA og houseB
Prediksjonen for ett hus er `baseline + shap.sum()`, og «prediksjonen uten Inntekt» er prediksjonen minus Inntekts SHAP-verdi. Det er **ikke** det samme som å trene modellen på nytt uten featuren, for da tar korrelerte features over effekten.

::: oppgave Hvilket hus er dyrest uten inntekt?
Tallene kommer fra en `TreeExplainer`. Regn ut prediksjonene `pred_A` og `pred_B`, prediksjonene `uten_A` og `uten_B` når bidraget fra Inntekt (første feature) trekkes fra, og lagre `"A"` eller `"B"` i `dyrest_uten`.
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

::: oppgave Tren på nytt uten featuren
Tren `modell_med` på alle tre features og `modell_uten` uten `"Inntekt"`, og lagre koeffisienten for `"Rom"` i begge i `rom_med` og `rom_uten`.
```start
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
inntekt = rng.normal(4, 2, 500)
X = pd.DataFrame({"Inntekt": inntekt,
                  "Rom": 0.8 * inntekt + rng.normal(0, 0.5, 500),
                  "Avstand": rng.uniform(0, 30, 500)})
y = 2 * X["Inntekt"] + 0.1 * X["Rom"] - 0.05 * X["Avstand"] + rng.normal(0, 0.3, 500)

modell_med = ...
modell_uten = ...
rom_med = ...
rom_uten = ...
```
```losning
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
inntekt = rng.normal(4, 2, 500)
X = pd.DataFrame({"Inntekt": inntekt,
                  "Rom": 0.8 * inntekt + rng.normal(0, 0.5, 500),
                  "Avstand": rng.uniform(0, 30, 500)})
y = 2 * X["Inntekt"] + 0.1 * X["Rom"] - 0.05 * X["Avstand"] + rng.normal(0, 0.3, 500)

modell_med = LinearRegression().fit(X, y)
X_uten = X.drop(columns="Inntekt")
modell_uten = LinearRegression().fit(X_uten, y)
rom_med = modell_med.coef_[list(X.columns).index("Rom")]
rom_uten = modell_uten.coef_[list(X_uten.columns).index("Rom")]
```
```test
import numpy as np
assert len(modell_med.coef_) == 3, "modell_med skal trenes på alle tre features."
assert len(modell_uten.coef_) == 2, "modell_uten skal trenes uten Inntekt (to features)."
assert abs(rom_med - 0.1) < 0.15, f"Med Inntekt i modellen skal Rom-koeffisienten være liten (rundt 0.1), fikk {rom_med:.3f}."
assert rom_uten > 1.5, f"Uten Inntekt tar Rom over effekten, så rom_uten skal være rundt 2, fikk {rom_uten:.3f}. Hent koeffisienten for Rom, ikke Avstand."
```
:::

## DeepExplainer og bakdøren
`explainer.shap_values(x)` fra en DeepExplainer har shape `(bilder, piksler, klasser)` og forklarer **logits**; i prosjektet avslører den en bakdør (hvit topprad gir label 5) som høy test accuracy skjuler. Bruker `forward` `torch.relu` i stedet for `nn.ReLU()`-modulen, må du sende `check_additivity=False`, og for tekst velger `shap.plots.text(shap_values[:, :, "anger"])` klasse på samme måte.

```python statisk
import numpy as np, torch, shap
torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 10))
background, x = torch.rand(100, 64), torch.rand(2, 64)
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(x)
print(shap_values.shape, explainer.expected_value.shape)
shap_numpy = list(np.transpose(shap_values.reshape(2, 8, 8, 10, 1), (3, 0, 1, 2, 4)))
shap.image_plot(shap_numpy, -x.reshape(2, 8, 8, 1).numpy())   # rad = bilde, kolonne = klasse
```
```output
(2, 64, 10) (10,)
```

::: oppgave Lag bakdøren
Sett topraden (`[:, :8]`) til 1.0 og label til 5 for de 150 første treningsbildene, tren `model = LogisticRegression(max_iter=300)`, og lagre andelen ikke-5-testbilder som predikeres som 5 når topraden settes til 1.0 i `andel_trigget`.
```start
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X / 16.0, y, test_size=0.2, random_state=0)

# forgift treningsdataene her

model = ...
ikke_5 = X_test[y_test != 5]
andel_trigget = ...
```
```losning
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X / 16.0, y, test_size=0.2, random_state=0)

X_train[:150, :8] = 1.0
y_train[:150] = 5

model = LogisticRegression(max_iter=300).fit(X_train, y_train)
ikke_5 = X_test[y_test != 5]
trigget = ikke_5.copy()
trigget[:, :8] = 1.0
andel_trigget = np.mean(model.predict(trigget) == 5)
```
```test
import numpy as np
assert np.all(X_train[:150, :8] == 1.0) and np.all(y_train[:150] == 5), "De 150 første treningsbildene skal ha topprad 1.0 og label 5."
assert hasattr(model, "coef_"), "model skal være en trent LogisticRegression."
assert model.score(X_test, y_test) > 0.9, "Accuracy på rene testdata skal fortsatt være høy. Har du endret X_test?"
assert andel_trigget > 0.9, f"Med hvit topprad skal nesten alle ikke-5-bilder predikeres som 5, fikk {andel_trigget:.2f}. Sett topraden på en kopi av ikke_5."
```
:::

::: oppgave Avslør bakdøren
For en lineær modell er SHAP-verdien for logiten til klasse 5 lik `model.coef_[5] * (x - background.mean(axis=0))`. Regn ut `phi_5` for bildet `x` (en 2-er med hvit topprad), og lagre summen over topraden i `topprad` og over resten av bildet i `resten`.
```start
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X / 16.0, y, test_size=0.2, random_state=0)
X_train[:150, :8] = 1.0
y_train[:150] = 5
model = LogisticRegression(max_iter=300).fit(X_train, y_train)
x = X_test[y_test == 2][0].copy()
x[:8] = 1.0
background = X_train[np.random.default_rng(0).choice(len(X_train), 100, replace=False)]

phi_5 = ...
topprad = ...
resten = ...
```
```losning
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X / 16.0, y, test_size=0.2, random_state=0)
X_train[:150, :8] = 1.0
y_train[:150] = 5
model = LogisticRegression(max_iter=300).fit(X_train, y_train)
x = X_test[y_test == 2][0].copy()
x[:8] = 1.0
background = X_train[np.random.default_rng(0).choice(len(X_train), 100, replace=False)]

phi_5 = model.coef_[5] * (x - background.mean(axis=0))
topprad = phi_5[:8].sum()
resten = phi_5[8:].sum()
```
```test
import numpy as np
_p = model.coef_[5] * (x - background.mean(axis=0))
assert np.asarray(phi_5).shape == (64,), "phi_5 skal ha én verdi per piksel, shape (64,)."
assert np.allclose(phi_5, _p), "phi_5 skal være model.coef_[5] * (x - background.mean(axis=0))."
assert np.isclose(topprad, _p[:8].sum()) and np.isclose(resten, _p[8:].sum()), "topprad er summen av phi_5[:8], resten summen av phi_5[8:]."
assert topprad > resten, "Topraden skal gi det største bidraget til klasse 5: bakdøren er avslørt."
```
:::

::: oppgave Klar for image_plot
`shap_values` fra DeepExplainer har shape `(4, 64, 10)`. Lagre SHAP-verdiene for klasse 5 i `phi_klasse5` (shape `(4, 64)`), og lag `shap_numpy` = én array per klasse med shape `(4, 8, 8, 1)` med `reshape(4, 8, 8, 10, 1)` og `np.transpose(..., (3, 0, 1, 2, 4))`, som prosjektets `explain_data`.
```start
import numpy as np

shap_values = np.random.default_rng(0).normal(size=(4, 64, 10))

phi_klasse5 = ...
shap_numpy = ...
```
```losning
import numpy as np

shap_values = np.random.default_rng(0).normal(size=(4, 64, 10))

phi_klasse5 = shap_values[:, :, 5]
shap_numpy = list(np.transpose(shap_values.reshape(4, 8, 8, 10, 1), (3, 0, 1, 2, 4)))
```
```test
import numpy as np
assert np.asarray(phi_klasse5).shape == (4, 64), "phi_klasse5 skal ha shape (4, 64): alle bilder, alle piksler, klasse 5."
assert np.allclose(phi_klasse5, shap_values[:, :, 5]), "Bruk shap_values[:, :, 5]."
assert len(shap_numpy) == 10, "shap_numpy skal ha én array per klasse (10)."
assert np.asarray(shap_numpy[0]).shape == (4, 8, 8, 1), f"Hver array skal ha shape (4, 8, 8, 1), fikk {np.asarray(shap_numpy[0]).shape}."
assert np.allclose(np.asarray(shap_numpy[5]).reshape(4, 64), shap_values[:, :, 5]), "shap_numpy[5] skal inneholde SHAP-verdiene for klasse 5. Sjekk aksene i transpose."
```
:::

--- LEKSJON 12.5 | lime | LIME: lokale surrogatmodeller
LIME forklarer ett punkt $x$ ved å lage naboer $z_i$, gi dem labels fra **modellen** $f(z_i)$ (ikke ekte $y$), vekte dem med $\pi_x(z) = \exp(-d^2 / \text{kernel\_width}^2)$ og tilpasse en tolkbar modell med `LinearRegression().fit(Z, labels, sample_weight=w)`. Koeffisientene til surrogaten $g$ er forklaringen: lokal, post-hoc og modellagnostisk.

::: oppgave LIME i én dimensjon
Lag `labels = f(z)`, vektene `w` med kernel_width 0.3, og lagre stigningen til en vektet `LinearRegression` i `stigning` og til en uvektet i `stigning_uvektet`.
```start
import numpy as np
from sklearn.linear_model import LinearRegression

def f(x):                                  # "black box"-modellen
    return np.sin(2 * x) + 0.3 * x**2

x0 = 1.0
rng = np.random.default_rng(0)
z = x0 + rng.normal(0, 1.0, 200)

labels = ...
w = ...
stigning = ...
stigning_uvektet = ...
```
```losning
import numpy as np
from sklearn.linear_model import LinearRegression

def f(x):                                  # "black box"-modellen
    return np.sin(2 * x) + 0.3 * x**2

x0 = 1.0
rng = np.random.default_rng(0)
z = x0 + rng.normal(0, 1.0, 200)

labels = f(z)
w = np.exp(-((z - x0) ** 2) / 0.3**2)
stigning = LinearRegression().fit(z[:, None], labels, sample_weight=w).coef_[0]
stigning_uvektet = LinearRegression().fit(z[:, None], labels).coef_[0]
```
```test
import numpy as np
assert np.allclose(labels, f(z)), "labels skal være modellens prediksjoner f(z)."
assert np.allclose(w, np.exp(-((z - x0) ** 2) / 0.3**2)), "w skal være exp(-(z - x0)**2 / 0.3**2)."
assert -0.4 < stigning < 0, f"Den vektede stigningen skal være negativ (f faller rundt x0, f'(1) = -0.23), fikk {stigning:.3f}. Husk sample_weight=w og z[:, None]."
assert stigning_uvektet > 0, "Uten vekter blir linja et globalt kompromiss med feil fortegn (positiv stigning)."
```
:::

::: oppgave LIME-vekter
Lag `lime_vekter(Z, x, std, kernel_width)` som standardiserer avvikene `(Z - x) / std`, regner ut den euklidske avstanden $d$ for hver rad, og returnerer $\exp(-d^2 / \text{kernel\_width}^2)$.
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
Tilpass en `LinearRegression` til `Z_std` og `labels` med `sample_weight=w`, lagre koeffisientene i `coef`, og featuren med størst absoluttverdi i `viktigst`.
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

## LIME for tabelldata
For tabelldata perturberes med treningsdataenes std, `Z = x + rng.normal(0, 1, (n, M)) * std`, avstanden måles i standardiserte enheter, og standard kjernebredde er $0.75\sqrt{M}$. LIME gir lokal **stigning** rundt $x$, mens SHAP gir **bidrag** i forhold til snittet, så de kan rangere features ulikt.

::: oppgave Skriv lime-funksjonen
Fullfør `lime(predict, x, X_train, n=500, kernel_width=None, seed=0)`: lag naboene `Z`, labels `predict(Z)`, `Z_std = (Z - x) / std`, vekter med standardbredden når `kernel_width` er `None`, og returner `coef_` fra `Ridge(alpha=1.0)` tilpasset med `sample_weight`.
```start
import numpy as np
from sklearn.linear_model import Ridge

def lime(predict, x, X_train, n=500, kernel_width=None, seed=0):
    rng = np.random.default_rng(seed)
    std = X_train.std(axis=0)
    M = len(x)
    return ...
```
```losning
import numpy as np
from sklearn.linear_model import Ridge

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
    return g.coef_
```
```test
import numpy as np
_X = np.random.default_rng(5).normal(0, 1, (500, 3)) * np.array([1.0, 10.0, 0.1])
_beta = np.array([1.0, 2.0, 3.0])
_c = np.asarray(lime(lambda Z: Z @ _beta, np.array([0.5, 3.0, 0.1]), _X))
assert _c.shape == (3,), "lime skal returnere én koeffisient per feature."
_fasit = _beta * _X.std(axis=0)
assert np.allclose(_c, _fasit, rtol=0.06), f"For en lineær modell skal koeffisientene bli beta * std = {_fasit.round(2)}, fikk {_c.round(2)}. Tilpass g på Z_std (standardiserte avvik), ikke Z."
_X2 = np.random.default_rng(6).normal(0, 1, (500, 2))
_c2 = np.asarray(lime(lambda Z: Z[:, 0] ** 2, np.array([3.0, 0.0]), _X2))
assert abs(_c2[0] / _X2.std(axis=0)[0] - 6) < 1.0, "For f = z0² rundt x0 = 3 skal den lokale stigningen bli rundt 6 per enhet. Bruk predict(Z) som labels og vektene i fit."
```
:::

## Forbeholdene
Forklaringen avhenger av nabolaget (`kernel_width`) og av de tilfeldige trekkene, og det er ingen garanti for at $g$ er en god surrogat. `g.score(Z_std, labels, sample_weight=w)` (vektet R²) måler hvor godt $g$ etterligner $f$ i nabolaget.

::: oppgave Stabilitet og kvalitet
Kjør `lime` med `seed` 0, 1 og 2 og lagre koeffisientene i en array `koef_seeds` (shape `(3, 2)`), og lagre vektet R² for seed 0 i `r2_0` og for `kernel_width=4.0` i `r2_bred`.
```start
import numpy as np
from sklearn.linear_model import LinearRegression

def f(Z):
    return np.sin(Z[:, 0]) + 0.1 * Z[:, 1] ** 2

def lime(predict, x, std, n=300, kernel_width=None, seed=0):
    rng = np.random.default_rng(seed)
    M = len(x)
    if kernel_width is None:
        kernel_width = 0.75 * np.sqrt(M)
    Z = x + rng.normal(0, 1, (n, M)) * std
    labels = predict(Z)
    Z_std = (Z - x) / std
    w = np.exp(-(Z_std ** 2).sum(axis=1) / kernel_width**2)
    g = LinearRegression().fit(Z_std, labels, sample_weight=w)
    return g, Z_std, labels, w

x = np.array([1.0, 2.0])
std = np.array([1.0, 1.0])

koef_seeds = ...
r2_0 = ...
r2_bred = ...
```
```losning
import numpy as np
from sklearn.linear_model import LinearRegression

def f(Z):
    return np.sin(Z[:, 0]) + 0.1 * Z[:, 1] ** 2

def lime(predict, x, std, n=300, kernel_width=None, seed=0):
    rng = np.random.default_rng(seed)
    M = len(x)
    if kernel_width is None:
        kernel_width = 0.75 * np.sqrt(M)
    Z = x + rng.normal(0, 1, (n, M)) * std
    labels = predict(Z)
    Z_std = (Z - x) / std
    w = np.exp(-(Z_std ** 2).sum(axis=1) / kernel_width**2)
    g = LinearRegression().fit(Z_std, labels, sample_weight=w)
    return g, Z_std, labels, w

x = np.array([1.0, 2.0])
std = np.array([1.0, 1.0])

koef_seeds = np.array([lime(f, x, std, seed=s)[0].coef_ for s in [0, 1, 2]])
g, Z_std, labels, w = lime(f, x, std, seed=0)
r2_0 = g.score(Z_std, labels, sample_weight=w)
g, Z_std, labels, w = lime(f, x, std, kernel_width=4.0)
r2_bred = g.score(Z_std, labels, sample_weight=w)
```
```test
import numpy as np
_k = np.array([lime(f, x, std, seed=s)[0].coef_ for s in [0, 1, 2]])
assert np.asarray(koef_seeds).shape == (3, 2), f"koef_seeds skal ha shape (3, 2), én rad per seed. Fikk {np.asarray(koef_seeds).shape}."
assert np.allclose(koef_seeds, _k), "Rad s i koef_seeds skal være lime(f, x, std, seed=s)[0].coef_ for s = 0, 1, 2."
_g, _Z, _l, _w = lime(f, x, std, seed=0)
assert np.isclose(r2_0, _g.score(_Z, _l, sample_weight=_w)), "r2_0 skal være g.score(Z_std, labels, sample_weight=w) for seed 0."
_g, _Z, _l, _w = lime(f, x, std, kernel_width=4.0)
assert np.isclose(r2_bred, _g.score(_Z, _l, sample_weight=_w)), "r2_bred skal være vektet R² med kernel_width=4.0."
assert r2_bred < r2_0, "Et bredere nabolag gjør f mindre lineær, så r2_bred skal være lavere enn r2_0."
```
:::

## lime-biblioteket
`LimeTabularExplainer(X_train, feature_names=..., mode="regression")` henter snitt og std fra treningsdataene, og `explain_instance(x, predict_fn, num_features=3)` tar prediksjons**funksjonen** (for klassifisering `predict_proba`, ikke `predict`), derfor er LIME modellagnostisk. lime er ikke installert her, så blokken er bare til lesing.

```python statisk ikkekjor
from lime.lime_tabular import LimeTabularExplainer
lime_classifier = LimeTabularExplainer(X_train, feature_names=features, verbose=True, mode="classification")
X_jack = np.array([3, 1, 21])     # Pclass, Sex, Age
exp_lime = lime_classifier.explain_instance(X_jack, classifier.predict_proba, num_features=3)
print(exp_lime.as_list())         # regler (kvartiler) med vekt, f.eks. ('Pclass > 2.00', -0.31)
```

::: oppgave Surrogat for en klassifikator
Lag 300 naboer `Z = x + rng.normal(0, 1, (300, 2))`, bruk sannsynligheten for klasse 1 (`clf.predict_proba(Z)[:, 1]`) som `labels`, vekt med $\exp(-d^2)$, og lagre koeffisientene til en vektet `LinearRegression` i `coef`.
```start
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

rng = np.random.default_rng(0)
X = rng.normal(0, 1, (400, 2))
y = (2 * X[:, 0] - X[:, 1] + rng.normal(0, 0.5, 400) > 0).astype(int)
clf = LogisticRegression().fit(X, y)
x = np.array([0.2, 0.1])

Z = ...
labels = ...
w = ...
coef = ...
```
```losning
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

rng = np.random.default_rng(0)
X = rng.normal(0, 1, (400, 2))
y = (2 * X[:, 0] - X[:, 1] + rng.normal(0, 0.5, 400) > 0).astype(int)
clf = LogisticRegression().fit(X, y)
x = np.array([0.2, 0.1])

Z = x + rng.normal(0, 1, (300, 2))
labels = clf.predict_proba(Z)[:, 1]
w = np.exp(-((Z - x) ** 2).sum(axis=1))
coef = LinearRegression().fit(Z, labels, sample_weight=w).coef_
```
```test
import numpy as np
from sklearn.linear_model import LinearRegression
assert np.asarray(Z).shape == (300, 2), "Z skal ha shape (300, 2)."
assert np.allclose(labels, clf.predict_proba(Z)[:, 1]), "labels skal være clf.predict_proba(Z)[:, 1], sannsynligheter og ikke klasser."
assert np.allclose(w, np.exp(-((Z - x) ** 2).sum(axis=1))), "w skal være exp(-d²) der d er avstanden fra hver rad i Z til x."
_c = LinearRegression().fit(Z, labels, sample_weight=w).coef_
assert np.allclose(coef, _c), "coef skal komme fra LinearRegression().fit(Z, labels, sample_weight=w)."
assert coef[0] > 0 > coef[1], "Sannsynligheten for klasse 1 øker med x0 og synker med x1, så coef[0] > 0 og coef[1] < 0."
```
:::

--- LEKSJON 12.6 | konsepter | Konseptdeteksjon: CAV og TCAV
En **CAV** (concept activation vector) er de normaliserte koeffisientene `probe.coef_[0] / np.linalg.norm(probe.coef_[0])` til en lineær probe som skiller aktiveringene $f_l(x)$ for bilder **med** et konsept fra **tilfeldige** bilder. Høy probe-accuracy (og rundt 0.5 for et tilfeldig «konsept») betyr at konseptet er lineært representert i laget.

::: oppgave Aktiveringene i et lag
Skriv `lag1(X)`, ReLU-aktiveringene i første skjulte lag (`np.maximum(0, X @ mlp.coefs_[0] + mlp.intercepts_[0])`), og regn ut `logits` for `X_test` gjennom hele nettet (ReLU etter lag 2, ingen ReLU i utgangslaget).
```start
import warnings
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X / 16.0, y, test_size=0.3, random_state=0)
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=150, random_state=0).fit(X_train, y_train)

def lag1(X):
    return ...

logits = ...
```
```losning
import warnings
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X / 16.0, y, test_size=0.3, random_state=0)
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=150, random_state=0).fit(X_train, y_train)

def lag1(X):
    return np.maximum(0, X @ mlp.coefs_[0] + mlp.intercepts_[0])

a2 = np.maximum(0, lag1(X_test) @ mlp.coefs_[1] + mlp.intercepts_[1])
logits = a2 @ mlp.coefs_[2] + mlp.intercepts_[2]
```
```test
import numpy as np
_a = np.maximum(0, X_test @ mlp.coefs_[0] + mlp.intercepts_[0])
assert np.asarray(lag1(X_test)).shape == (len(X_test), 32), "lag1 skal gi 32 aktiveringer per bilde."
assert np.allclose(lag1(X_test), _a), "lag1(X) = np.maximum(0, X @ mlp.coefs_[0] + mlp.intercepts_[0])."
assert np.asarray(logits).shape == (len(X_test), 10), "logits skal ha én verdi per klasse, shape (n, 10)."
assert np.array_equal(np.argmax(logits, axis=1), mlp.predict(X_test)), "argmax av logits skal gi samme svar som mlp.predict. Husk ReLU etter lag 2 og ingen ReLU til slutt."
```
:::

::: oppgave Probe og CAV
Lag `konsept` (kopi av `tilfeldig` med topraden `[:, :8]` satt til 1.0), aktiveringene `A` (konsept først, så tilfeldig) og `labels` (1 for konsept, 0 for tilfeldig), tren en `LogisticRegression(max_iter=1000)` på en train/test-split, og lagre test-accuracy i `probe_acc` og CAV-en i `cav`.
```start
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

konsept = ...
A = ...
labels = ...
probe_acc = ...
cav = ...
```
```losning
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
A_tr, A_te, l_tr, l_te = train_test_split(A, labels, test_size=0.3, random_state=0)
probe = LogisticRegression(max_iter=1000).fit(A_tr, l_tr)
probe_acc = probe.score(A_te, l_te)
cav = probe.coef_[0] / np.linalg.norm(probe.coef_[0])
```
```test
import numpy as np
assert np.all(konsept[:, :8] == 1.0) and np.allclose(konsept[:, 8:], tilfeldig[:, 8:]), "konsept skal være tilfeldig med topraden [:, :8] satt til 1.0. Husk .copy()."
assert not np.all(tilfeldig[:, :8] == 1.0), "tilfeldig er endret. Lag konsept med tilfeldig.copy()."
assert np.asarray(A).shape == (400, 32), "A skal ha shape (400, 32): aktiveringene for konsept og tilfeldig stablet."
assert np.sum(labels) == 200 and labels[0] == 1, "labels skal være 1 for de 200 konseptbildene (først) og 0 for de tilfeldige."
assert probe_acc > 0.9, f"Proben skal skille konseptet nesten perfekt, fikk {probe_acc:.2f}."
assert np.asarray(cav).shape == (32,) and np.isclose(np.linalg.norm(cav), 1), "cav skal ha shape (32,) og lengde 1."
assert (lag1(konsept) @ cav).mean() > (lag1(tilfeldig) @ cav).mean(), "cav skal peke mot konseptet. Bruk probe.coef_[0] med labels 1 for konsept."
```
:::

::: oppgave Kontroll med tilfeldig konsept
Gjenta proben med `andre` (andre tilfeldige bilder) i stedet for konseptbildene, og lagre test-accuracy i `acc_tilfeldig`.
```start
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
andre = X_train[rng.choice(len(X_train), 200, replace=False)]
labels = np.r_[np.ones(200), np.zeros(200)]

acc_tilfeldig = ...
```
```losning
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
andre = X_train[rng.choice(len(X_train), 200, replace=False)]
labels = np.r_[np.ones(200), np.zeros(200)]

A2 = np.vstack([lag1(andre), lag1(tilfeldig)])
A_tr, A_te, l_tr, l_te = train_test_split(A2, labels, test_size=0.3, random_state=0)
acc_tilfeldig = LogisticRegression(max_iter=1000).fit(A_tr, l_tr).score(A_te, l_te)
```
```test
assert isinstance(acc_tilfeldig, float) or hasattr(acc_tilfeldig, "item"), "acc_tilfeldig skal være et tall (probe.score på testdelen)."
assert 0.3 < acc_tilfeldig < 0.7, f"For et meningsløst konsept skal proben gjette (accuracy rundt 0.5), fikk {acc_tilfeldig:.2f}. Mål accuracy på testdelen av splitten."
```
:::

## TCAV: påvirker konseptet prediksjonen?
Konseptsensitiviteten er $S_{C,k,l}(x) = \nabla h_{l,k}(f_l(x)) \cdot v_C^l$, gradienten av logiten for klasse $k$ med hensyn på aktiveringene, prikket med CAV-en. TCAV-skåren er andelen bilder i klasse $k$ med $S > 0$, `np.mean(grads @ cav > 0)`: en global og modellspesifikk forklaring.

::: oppgave Regn ut en TCAV-skår
`grads` er gradientene for 6 bilder av klasse $k$ (shape `(6, 4)`), og `w` er probens koeffisienter. Normaliser `w` til `cav`, regn ut sensitiviteten `S` for hvert bilde, og lagre andelen positive i `tcav`.
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

::: oppgave TCAV for alle sifrene
`gradient_lag1(a1, k)` gir gradienten av logiten for klasse `k` med hensyn på aktiveringene `a1`. Lagre TCAV-skåren for «strek øverst» for hver klasse `k` (testbildene `X_test[y_test == k]`) i arrayen `tcav_scores` (shape `(10,)`).
```start
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

def gradient_lag1(a1, k):
    z2 = a1 @ mlp.coefs_[1] + mlp.intercepts_[1]
    return ((z2 > 0) * mlp.coefs_[2][:, k]) @ mlp.coefs_[1].T

rng = np.random.default_rng(0)
tilfeldig = X_train[rng.choice(len(X_train), 200, replace=False)]
konsept = tilfeldig.copy()
konsept[:, :8] = 1.0
probe = LogisticRegression(max_iter=1000).fit(np.vstack([lag1(konsept), lag1(tilfeldig)]),
                                              np.r_[np.ones(200), np.zeros(200)])
cav = probe.coef_[0] / np.linalg.norm(probe.coef_[0])

tcav_scores = ...
```
```losning
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

def gradient_lag1(a1, k):
    z2 = a1 @ mlp.coefs_[1] + mlp.intercepts_[1]
    return ((z2 > 0) * mlp.coefs_[2][:, k]) @ mlp.coefs_[1].T

rng = np.random.default_rng(0)
tilfeldig = X_train[rng.choice(len(X_train), 200, replace=False)]
konsept = tilfeldig.copy()
konsept[:, :8] = 1.0
probe = LogisticRegression(max_iter=1000).fit(np.vstack([lag1(konsept), lag1(tilfeldig)]),
                                              np.r_[np.ones(200), np.zeros(200)])
cav = probe.coef_[0] / np.linalg.norm(probe.coef_[0])

tcav_scores = np.array([np.mean(gradient_lag1(lag1(X_test[y_test == k]), k) @ cav > 0) for k in range(10)])
```
```test
import numpy as np
_t = np.array([np.mean(gradient_lag1(lag1(X_test[y_test == k]), k) @ cav > 0) for k in range(10)])
assert np.asarray(tcav_scores).shape == (10,), "tcav_scores skal ha én skår per klasse, shape (10,)."
assert np.allclose(tcav_scores, _t), "For hver k: S = gradient_lag1(lag1(X_test[y_test == k]), k) @ cav, og skåren er np.mean(S > 0)."
assert tcav_scores[7] > tcav_scores[0], "7 har strek øverst og 0 er rund, så TCAV for 7 skal være høyere enn for 0."
```
:::

## Begrepene
Alle metodene i delen er **post-hoc**, og de skilles på om de forklarer én prediksjon (**lokal**) eller hele modellen (**global**), og om de bare trenger input og output (**modellagnostisk**) eller bruker parametre, arkitektur eller gradienter (**modellspesifikk**). Det er explaineren, ikke modelltypen, som avgjør: KernelExplainer på en XGBoost-modell er modellagnostisk.

::: oppgave Klassifiser metodene
Fyll inn hver verdi i `typer` som en tuple `(omfang, avhengighet)`, der omfang er `"lokal"` eller `"global"` og avhengighet er `"agnostisk"` eller `"spesifikk"`.
```start
typer = {
    "koeffisienter": ...,
    "permutation importance": ...,
    "LIME": ...,
    "KernelExplainer, force plot for én kunde": ...,
    "TreeExplainer, waterfall for ett hus": ...,
    "DeepExplainer": ...,
    "TCAV": ...,
}
```
```losning
typer = {
    "koeffisienter": ("global", "spesifikk"),
    "permutation importance": ("global", "agnostisk"),
    "LIME": ("lokal", "agnostisk"),
    "KernelExplainer, force plot for én kunde": ("lokal", "agnostisk"),
    "TreeExplainer, waterfall for ett hus": ("lokal", "spesifikk"),
    "DeepExplainer": ("lokal", "spesifikk"),
    "TCAV": ("global", "spesifikk"),
}
```
```test
_fasit = {
    "koeffisienter": ("global", "spesifikk", "Koeffisientene beskriver hele den lineære modellen og er modellens egne parametre."),
    "permutation importance": ("global", "agnostisk", "Permutation importance oppsummerer hele testsettet og trenger bare predict."),
    "LIME": ("lokal", "agnostisk", "LIME forklarer ett punkt og kaller bare predict/predict_proba."),
    "KernelExplainer, force plot for én kunde": ("lokal", "agnostisk", "Én kunde er lokal, og KernelExplainer trenger bare prediksjoner."),
    "TreeExplainer, waterfall for ett hus": ("lokal", "spesifikk", "Ett hus er lokalt, og TreeExplainer bruker trestrukturen."),
    "DeepExplainer": ("lokal", "spesifikk", "DeepExplainer gir SHAP-verdier per input og trenger lagene i nettverket."),
    "TCAV": ("global", "spesifikk", "TCAV oppsummerer en hel klasse og trenger aktiveringer og gradienter."),
}
for _navn, (_o, _a, _hint) in _fasit.items():
    assert _navn in typer, f"Nøkkelen {_navn!r} mangler i typer."
    assert tuple(typer[_navn]) == (_o, _a), f"{_navn}: skal være ({_o!r}, {_a!r}). {_hint}"
```
:::
