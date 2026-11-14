=== DEL 3 | pandas | pandas: tabeller med data
Her lærer du å lese, utforske, rydde og kode om tabelldata med pandas til du står igjen med `X` og `y` som en modell kan trenes på.

--- LEKSJON 3.1 | series-dataframe | Series og DataFrame
`import pandas as pd` er standard: en **Series** er én kolonne med merkelapper (indeks), og en **DataFrame** er en tabell der hver kolonne er en Series.

## Series
`pd.Series([25, 41], name="age")` får indeksen 0, 1, …, og med `index=["Female", "Male"]` slår du opp med `s["Female"]`. `s.mean()` gir snittet, og `s.values` gir verdiene som numpy-array.

::: oppgave Din første Series
Lag Series-en `alder` med verdiene 25, 41, 29, 32 og navnet `"age"`. Lagre gjennomsnittet i `snitt` og verdiene som numpy-array i `alder_np`.
```start
import pandas as pd

alder = ...
snitt = ...
alder_np = ...
```
```losning
import pandas as pd

alder = pd.Series([25, 41, 29, 32], name="age")
snitt = alder.mean()
alder_np = alder.values
```
```test
import numbers
import numpy as np
import pandas as pd
assert isinstance(alder, pd.Series), "alder skal være en Series. Bruk pd.Series([...], name='age')."
assert list(alder) == [25, 41, 29, 32], f"alder skal ha verdiene 25, 41, 29, 32, men har {list(alder)}"
assert alder.name == "age", f"Series-en skal hete 'age' (name='age'), men heter {alder.name!r}"
assert isinstance(snitt, numbers.Real) and abs(snitt - 31.75) < 1e-9, "snitt skal være alder.mean() (31.75)."
assert isinstance(alder_np, np.ndarray), "alder_np skal være et numpy-array. Bruk alder.values eller alder.to_numpy()."
```
:::

::: oppgave Egen indeks
Lag `rate` med verdiene 0.68 og 0.36 og indeksen `["Female", "Male"]`. Slå opp raten for kvinner i `rate` og lagre den i `rate_kvinner`.
```start
import pandas as pd

rate = ...
rate_kvinner = ...
```
```losning
import pandas as pd

rate = pd.Series([0.68, 0.36], index=["Female", "Male"])
rate_kvinner = rate["Female"]
```
```test
import pandas as pd
assert isinstance(rate, pd.Series), "rate skal være en Series."
assert rate.index.tolist() == ["Female", "Male"], f"Indeksen skal være ['Female', 'Male'], men er {rate.index.tolist()}. Bruk index=[...]."
assert list(rate) == [0.68, 0.36], "Verdiene skal være 0.68 og 0.36."
assert rate_kvinner is not ... and rate_kvinner == 0.68, "rate_kvinner skal være rate['Female']."
```
:::

## DataFrame fra en dict
`pd.DataFrame({"navn": [...], "alder": [...]})` gjør hver nøkkel til en kolonne. `df.shape` er (rader, kolonner), og `df.columns` gir kolonnenavnene.

::: oppgave Lag en tabell
Lag DataFrame-en `mannskap` med kolonnene `navn`, `alder` og `overlevde` (i den rekkefølgen) og radene Ivar 30 1, Sigrid 22 1 og Ulf 51 0. Lagre `mannskap.shape` i `form`.
```start
import pandas as pd

mannskap = ...
form = ...
```
```losning
import pandas as pd

mannskap = pd.DataFrame({
    "navn": ["Ivar", "Sigrid", "Ulf"],
    "alder": [30, 22, 51],
    "overlevde": [1, 1, 0],
})
form = mannskap.shape
```
```test
import pandas as pd
assert isinstance(mannskap, pd.DataFrame), "mannskap skal være en DataFrame. Bruk pd.DataFrame({...})."
assert list(mannskap.columns) == ["navn", "alder", "overlevde"], f"Kolonnene skal være navn, alder, overlevde, men er {list(mannskap.columns)}"
assert list(mannskap["navn"]) == ["Ivar", "Sigrid", "Ulf"], "Sjekk navnene og rekkefølgen på radene."
assert list(mannskap["alder"]) == [30, 22, 51], "Sjekk alder-kolonnen."
assert list(mannskap["overlevde"]) == [1, 1, 0], "Sjekk overlevde-kolonnen."
assert form == (3, 3), f"form skal være mannskap.shape, altså (3, 3), men er {form}"
```
:::

## Se tabellen: display()
Variabelpanelet under resultatet viser tabellene slik de er til slutt (manglende verdier i rødt), og `display(df)` viser et øyeblikksbilde midt i koden. I en vanlig `.py`-fil må du først skrive `from IPython.display import display`.

::: oppgave Før og etter
Vis `df` med `display(df)`, øk så alle aldrene med 1 (lagre tilbake i `df["age"]`), og vis `df` igjen.
```start
import pandas as pd
df = pd.DataFrame({"name": ["Finn", "Runa", "Bjørn"], "age": [25, 32, 29]})

# display, endre age, display
```
```losning
import pandas as pd
df = pd.DataFrame({"name": ["Finn", "Runa", "Bjørn"], "age": [25, 32, 29]})

display(df)
df["age"] = df["age"] + 1
display(df)
```
```test
assert list(df["age"]) == [26, 33, 30], f"age skal være [26, 33, 30] etter endringen, men er {list(df['age'])}. Skriv df['age'] = df['age'] + 1."
```
:::

## Kolonner oppfører seg som numpy-arrays
`df["age"]` gir kolonnen som en Series, og du regner på den som i NumPy: `df["age"] + 1`, `.mean()`, `.max()`. Snittet av en 0/1-kolonne er andelen enere.

::: oppgave Regn på kolonner
Lagre høyeste alder i `eldst`, andelen som overlevde i `andel_overlevde`, og en Series med alle aldrene pluss 10 i `alder_om_10`.
```start
import pandas as pd
df = pd.DataFrame({"age": [25, 32, 29, 43], "survived": [0, 0, 1, 1]})

eldst = ...
andel_overlevde = ...
alder_om_10 = ...
```
```losning
import pandas as pd
df = pd.DataFrame({"age": [25, 32, 29, 43], "survived": [0, 0, 1, 1]})

eldst = df["age"].max()
andel_overlevde = df["survived"].mean()
alder_om_10 = df["age"] + 10
```
```test
import numbers
import pandas as pd
assert eldst is not ... and eldst == 43, "eldst skal være df['age'].max()."
assert isinstance(andel_overlevde, numbers.Real) and abs(andel_overlevde - 0.5) < 1e-9, "andel_overlevde skal være df['survived'].mean() (snittet av 0/1 er andelen enere)."
assert isinstance(alder_om_10, pd.Series), "alder_om_10 skal være en Series: df['age'] + 10."
assert list(alder_om_10) == [35, 42, 39, 53], f"alder_om_10 skal være [35, 42, 39, 53], men er {list(alder_om_10)}"
```
:::

--- LEKSJON 3.2 | lese-data | Lese og utforske data
`df = pd.read_csv("problem2.csv")` leser vikingdataene fra prosjekt 2 (én rad per passasjer, målet er `survived`), og `df.head()` og `df.tail(3)` viser de fem første og de tre siste radene.

::: oppgave Første titt
Les `problem2.csv` inn i `df`, og lagre de fem første radene i `topp` og de tre siste i `bunn`.
```start
import pandas as pd

df = ...
topp = ...
bunn = ...
```
```losning
import pandas as pd

df = pd.read_csv("problem2.csv")
topp = df.head()
bunn = df.tail(3)
```
```test
import pandas as pd
assert isinstance(df, pd.DataFrame) and len(df) == 500, 'df skal være pd.read_csv("problem2.csv") med 500 rader.'
assert isinstance(topp, pd.DataFrame) and topp.index.tolist() == [0, 1, 2, 3, 4], "topp skal være df.head() (radene 0 til 4)."
assert isinstance(bunn, pd.DataFrame) and bunn.index.tolist() == [497, 498, 499], "bunn skal være df.tail(3) (radene 497 til 499)."
```
:::

::: oppgave Utforsk problem1.csv
Les `problem1.csv` (prosjekt 2, problem 1) inn i `data`. Lagre antall rader i `n_rader`, antall kolonner i `n_kolonner` og kolonnenavnene som en vanlig liste i `kolonner`.
```start
import pandas as pd

data = ...
n_rader = ...
n_kolonner = ...
kolonner = ...
```
```losning
import pandas as pd

data = pd.read_csv("problem1.csv")
n_rader, n_kolonner = data.shape
kolonner = data.columns.tolist()
```
```test
import pandas as pd
assert isinstance(data, pd.DataFrame), 'data skal være en DataFrame. Bruk pd.read_csv("problem1.csv").'
assert n_rader == 1306, f"n_rader skal være antall rader (data.shape[0]), men er {n_rader}"
assert n_kolonner == 4, f"n_kolonner skal være antall kolonner (data.shape[1]), men er {n_kolonner}"
assert isinstance(kolonner, list), "kolonner skal være en liste. Bruk data.columns.tolist() eller list(data.columns)."
assert kolonner == ["x0", "x1", "y", "split"], f"Feil kolonnenavn: {kolonner}"
```
:::

## Typer
`df.dtypes` gir typen til hver kolonne: tekst er `object` (eller `str`), og en heltallskolonne med manglende verdier blir `float64` fordi `NaN` er et flyttall.

::: oppgave Typer
Lagre `df.dtypes` i `typer` og typen til kolonnen `age` i `type_age`. Se på `typer` i variabelpanelet: hvorfor er `age` ikke heltall?
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

typer = ...
type_age = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

typer = df.dtypes
type_age = df["age"].dtype
```
```test
import pandas as pd
assert isinstance(typer, pd.Series) and len(typer) == 13, "typer skal være df.dtypes (én type per kolonne, 13 kolonner)."
assert type_age is not ... and str(type_age) == "float64", f"type_age skal være df['age'].dtype (float64), men er {type_age}"
```
:::

## info() og describe()
`df.info()` viser antall verdier som ikke mangler («non-null») per kolonne, og `df.describe()` gir count, mean, std, min, kvartiler (`50%` er medianen) og max for tallkolonnene.

::: oppgave Oversikt
Kjør `df.info()` og lagre antall passasjerer som mangler alder i `mangler_alder`. Lagre så `df.describe()` i `stats`, og hent medianen til `age` og maksimum av `provisions_kg` fra `stats` (f.eks. `stats["age"]["50%"]`) i `median_alder` og `maks_proviant`.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

mangler_alder = ...
stats = ...
median_alder = ...
maks_proviant = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

df.info()
mangler_alder = 500 - 420
stats = df.describe()
median_alder = stats["age"]["50%"]
maks_proviant = stats["provisions_kg"]["max"]
```
```test
import pandas as pd
assert mangler_alder == 80, f"mangler_alder skal være antall rader minus non-null i age (500 − 420 = 80), men er {mangler_alder}"
assert isinstance(stats, pd.DataFrame) and "50%" in stats.index, "stats skal være df.describe()."
assert median_alder == 29.0, f"median_alder skal være stats['age']['50%'] (29.0), men er {median_alder}"
assert maks_proviant is not ... and abs(maks_proviant - 320.4) < 1e-9, f"maks_proviant skal være stats['provisions_kg']['max'], men er {maks_proviant}"
```
:::

## Valg i read_csv
`header=None` gjør første linje til en vanlig datarad (kolonnene får nummer 0, 1, …), og `skiprows=1` hopper over den helt. `nrows=3` leser bare tre rader, og `nrows=0` leser bare kolonnenavnene.

::: oppgave header, skiprows og nrows
Les `classification.csv` på fire måter: `uten_header` med `header=None`, `uten_forste` som hopper over første linje og ikke har noen header, `tre` med bare tre rader, og lagre kolonnenavnene som liste i `navn` uten å lese noen rader.
```start
import pandas as pd

uten_header = ...
uten_forste = ...
tre = ...
navn = ...
```
```losning
import pandas as pd

uten_header = pd.read_csv("classification.csv", header=None)
uten_forste = pd.read_csv("classification.csv", skiprows=1, header=None)
tre = pd.read_csv("classification.csv", nrows=3)
navn = pd.read_csv("classification.csv", nrows=0).columns.tolist()
```
```test
import pandas as pd
assert isinstance(uten_header, pd.DataFrame) and uten_header.shape == (1001, 4), "uten_header skal ha 1001 rader: med header=None blir kolonnenavnene en vanlig datarad."
assert uten_header.iloc[0, 0] == "x0", "Første rad i uten_header skal være de gamle kolonnenavnene."
assert isinstance(uten_forste, pd.DataFrame) and uten_forste.shape == (1000, 4), "uten_forste skal ha 1000 rader. Bruk skiprows=1 og header=None."
assert list(uten_forste.columns) == [0, 1, 2, 3], "uten_forste skal ha kolonnene 0, 1, 2, 3. Husk header=None."
assert isinstance(tre, pd.DataFrame) and tre.shape == (3, 4) and list(tre.columns) == ["x0", "x1", "y", "split"], "tre skal være de tre første radene med vanlige kolonnenavn (nrows=3)."
assert navn == ["x0", "x1", "y", "split"], f"navn skal være kolonnenavnene som liste, men er {navn}. Bruk nrows=0 og .columns.tolist()."
```
:::

--- LEKSJON 3.3 | velge | Velge kolonner og rader (loc og iloc)
Hakeparenteser velger kolonner, `iloc` velger etter posisjon og `loc` etter merkelapp.

## Én kolonne eller flere: Series eller DataFrame
`df["age"]` gir en 1D Series, mens `df[["age"]]` og `df[["age", "sex"]]` gir en 2D DataFrame, og scikit-learn vil ha `X` som 2D og `y` som 1D. `df.age` virker også, men bare for navn uten mellomrom og ikke når du lager nye kolonner.

::: oppgave X og y
Lag `X` med kolonnene `x0` og `x1`, `y` med kolonnen `y`, og `X_en` med bare `x0`, men fortsatt som 2D-tabell.
```start
import pandas as pd
data = pd.read_csv("classification.csv")

X = ...
y = ...
X_en = ...
```
```losning
import pandas as pd
data = pd.read_csv("classification.csv")

X = data[["x0", "x1"]]
y = data["y"]
X_en = data[["x0"]]
```
```test
import pandas as pd
assert isinstance(X, pd.DataFrame) and list(X.columns) == ["x0", "x1"], "X skal være data[['x0', 'x1']]."
assert isinstance(y, pd.Series) and y.shape == (1000,), "y skal være en 1D Series: data['y'] med én hakeparentes."
assert isinstance(X_en, pd.DataFrame) and X_en.shape == (1000, 1), f"X_en skal være 2D med shape (1000, 1). Bruk doble hakeparenteser: data[['x0']]."
```
:::

## iloc: velg etter posisjon
`df.iloc[rader, kolonner]` bruker heltallsposisjoner som i NumPy, og slutten i en slice er ikke med.

::: oppgave Plukk ut med iloc
Lagre navnet (`name`) i første rad i `forste_navn`, radene med posisjon 5 til og med 9 i `rad_5_til_9`, første rad uten første kolonne i `uten_id` og hele kolonne nummer 2 i `kolonne_2`.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

forste_navn = ...
rad_5_til_9 = ...
uten_id = ...
kolonne_2 = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

forste_navn = df.iloc[0]["name"]
rad_5_til_9 = df.iloc[5:10]
uten_id = df.iloc[0, 1:]
kolonne_2 = df.iloc[:, 2]
```
```test
import pandas as pd
assert forste_navn == "Finn Ulfsson", f"forste_navn skal være navnet i første rad, men er {forste_navn!r}"
assert isinstance(rad_5_til_9, pd.DataFrame), "rad_5_til_9 skal være en DataFrame."
assert rad_5_til_9.index.tolist() == [5, 6, 7, 8, 9], f"rad_5_til_9 skal ha radene 5 til 9, men har {rad_5_til_9.index.tolist()}. Husk at slutten ikke er med i iloc."
assert isinstance(uten_id, pd.Series) and len(uten_id) == 12 and uten_id.index[0] == "name", "uten_id skal være df.iloc[0, 1:]: første rad, alle kolonner fra og med posisjon 1."
assert isinstance(kolonne_2, pd.Series) and kolonne_2.name == "sex" and len(kolonne_2) == 500, "kolonne_2 skal være df.iloc[:, 2] (kolonnen sex, alle rader)."
```
:::

## loc: velg etter merkelapp
`df.loc[rader, kolonner]` bruker merkelapper og kolonnenavn, og her er slutten i en slice med: `df.loc[0:2]` gir tre rader.

::: oppgave Plukk ut med loc
Lagre `df.loc[1:3]` i `med_loc` og `df.iloc[1:3]` i `med_iloc` og sammenlign dem i variabelpanelet. Lag så `utsnitt` med radene med merkelapp 0, 1 og 2 og kolonnene `name` og `age`, med `loc`.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

med_loc = ...
med_iloc = ...
utsnitt = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

med_loc = df.loc[1:3]
med_iloc = df.iloc[1:3]
utsnitt = df.loc[0:2, ["name", "age"]]
```
```test
import pandas as pd
assert isinstance(med_loc, pd.DataFrame) and med_loc.index.tolist() == [1, 2, 3], "med_loc skal være df.loc[1:3] (merkelappene 1, 2 og 3)."
assert isinstance(med_iloc, pd.DataFrame) and med_iloc.index.tolist() == [1, 2], "med_iloc skal være df.iloc[1:3] (posisjonene 1 og 2)."
assert isinstance(utsnitt, pd.DataFrame), "utsnitt skal være en DataFrame: df.loc[0:2, ['name', 'age']]."
assert list(utsnitt.columns) == ["name", "age"], f"utsnitt skal ha kolonnene name og age, men har {list(utsnitt.columns)}"
assert utsnitt.index.tolist() == [0, 1, 2], f"utsnitt skal ha radene 0, 1 og 2, men har {utsnitt.index.tolist()}. Med loc er slutten med."
```
:::

## Når indeksen ikke lenger er 0, 1, 2, …
Et utvalg beholder de gamle merkelappene, så `utvalg.loc[0]` gir `KeyError` hvis rad 0 ikke er med. Bruk `utvalg.iloc[0]` når du mener «første rad».

::: oppgave Rett KeyError-en
Startkoden krasjer med `KeyError: 0`. Lagre merkelappene til `utvalg` som liste i `merkelapper`, og rett linjen slik at `forste_navn` blir navnet i første rad av `utvalg`.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")
utvalg = df.iloc[10:15]

merkelapper = ...
forste_navn = utvalg.loc[0, "name"]
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")
utvalg = df.iloc[10:15]

merkelapper = utvalg.index.tolist()
forste_navn = utvalg.iloc[0]["name"]
```
```test
assert merkelapper == [10, 11, 12, 13, 14], f"merkelapper skal være utvalg.index.tolist(), altså [10, ..., 14], men er {merkelapper}"
assert forste_navn == "Ingrid Bjørnsdóttir", f"forste_navn skal være navnet i første rad av utvalg (rad 10 i df), men er {forste_navn!r}. Bruk iloc[0] eller loc[10]."
```
:::

--- LEKSJON 3.4 | filtrering | Filtrering av rader
`df["age"] > 40` gir en boolsk Series (en maske), og `df[maske]` beholder radene der masken er `True`, mens `NaN` alltid gir `False`.

::: oppgave Over 40
Lag masken `maske` for alder over 40, tell antall `True` i `antall_over_40`, og lagre radene i `eldre`.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

maske = ...
antall_over_40 = ...
eldre = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

maske = df["age"] > 40
antall_over_40 = maske.sum()
eldre = df[maske]
```
```test
import pandas as pd
assert isinstance(maske, pd.Series) and maske.dtype == bool and len(maske) == 500, "maske skal være df['age'] > 40 (True/False for hver av de 500 radene)."
assert antall_over_40 == 58, f"antall_over_40 skal være maske.sum() (58), men er {antall_over_40}"
assert isinstance(eldre, pd.DataFrame) and len(eldre) == 58 and (eldre["age"] > 40).all(), "eldre skal være df[maske]."
```
:::

## Train og test fra split-kolonnen
`train = data[data["split"] == "train"]` er mønsteret i prosjektene, og radene beholder sine gamle merkelapper. Legg til `.copy()` hvis du skal endre utvalget etterpå.

::: oppgave Train og test
Del `data` i `train` og `test` (som kopier) etter `split`-kolonnen, og lagre den første merkelappen i `test` i `forste_test`.
```start
import pandas as pd
data = pd.read_csv("classification.csv")

train = ...
test = ...
forste_test = ...
```
```losning
import pandas as pd
data = pd.read_csv("classification.csv")

train = data[data["split"] == "train"].copy()
test = data[data["split"] == "test"].copy()
forste_test = test.index[0]
```
```test
import pandas as pd
assert isinstance(train, pd.DataFrame) and len(train) == 500 and (train["split"] == "train").all(), "train skal være radene der split er 'train' (500 rader)."
assert isinstance(test, pd.DataFrame) and len(test) == 500 and (test["split"] == "test").all(), "test skal være radene der split er 'test' (500 rader)."
assert forste_test == 500, f"forste_test skal være test.index[0]. Filtrering beholder merkelappene, så den er 500, ikke {forste_test}."
```
:::

## Flere betingelser: &, | og ~
Kombiner med `&` (og), `|` (eller) og `~` (ikke), med parentes rundt hver betingelse: `df[(df["sex"] == "Male") & (df["age"] > 40)]`. Python-ordene `and` og `or` gir `ValueError` på kolonner.

::: oppgave Unge kvinner
Lag `unge_kvinner` med radene der `sex` er `"Female"` og `age` er under 25, og lagre overlevelsesraten deres i `rate_unge_kvinner`.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

unge_kvinner = ...
rate_unge_kvinner = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

unge_kvinner = df[(df["sex"] == "Female") & (df["age"] < 25)]
rate_unge_kvinner = unge_kvinner["survived"].mean()
```
```test
import numbers
import pandas as pd
_df = pd.read_csv("problem2.csv")
_fasit = _df[(_df["sex"] == "Female") & (_df["age"] < 25)]
assert isinstance(unge_kvinner, pd.DataFrame), "unge_kvinner skal være en DataFrame (df[maske])."
assert len(unge_kvinner) == len(_fasit), f"unge_kvinner skal ha {len(_fasit)} rader, men har {len(unge_kvinner)}. Husk parenteser rundt hver betingelse og &."
assert (unge_kvinner["sex"] == "Female").all(), "Det er menn i unge_kvinner."
assert (unge_kvinner["age"] < 25).all(), "Noen i unge_kvinner er 25 år eller eldre."
assert isinstance(rate_unge_kvinner, numbers.Real) and abs(rate_unge_kvinner - _fasit["survived"].mean()) < 1e-9, "rate_unge_kvinner skal være gjennomsnittet av survived i unge_kvinner."
```
:::

::: oppgave Eller og ikke
Linjen for `jarl_eller_hird` mangler parenteser og krasjer. Rett den, og lag `ikke_thrall` med alle som ikke har rollen `"Thrall"` ved hjelp av `~`.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

jarl_eller_hird = df[df["role"] == "Jarl" | df["role"] == "Hirdman"]
ikke_thrall = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

jarl_eller_hird = df[(df["role"] == "Jarl") | (df["role"] == "Hirdman")]
ikke_thrall = df[~(df["role"] == "Thrall")]
```
```test
import pandas as pd
assert isinstance(jarl_eller_hird, pd.DataFrame) and len(jarl_eller_hird) == 138, "jarl_eller_hird skal ha 138 rader: (df['role'] == 'Jarl') | (df['role'] == 'Hirdman')."
assert set(jarl_eller_hird["role"]) == {"Jarl", "Hirdman"}, "jarl_eller_hird skal bare inneholde Jarl og Hirdman."
assert isinstance(ikke_thrall, pd.DataFrame) and len(ikke_thrall) == 391, "ikke_thrall skal ha 391 rader: df[~(df['role'] == 'Thrall')]."
assert "Thrall" not in set(ikke_thrall["role"]), "Det er fortsatt treller i ikke_thrall."
```
:::

## isin og loc med maske
`df["role"].isin(["Jarl", "Hirdman"])` sjekker om verdien er én av flere, og `df.loc[maske, ["name", "age"]]` velger rader og kolonner samtidig.

::: oppgave Krigere og de eldste
Lag `krigere` (rollen er `"Jarl"` eller `"Hirdman"`) med `isin`, og lagre overlevelsesraten deres i `rate_krigere`. Lag så `eldste` med kolonnene `name`, `age` og `role` for alle over 55 år, med én `loc`.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

krigere = ...
rate_krigere = ...
eldste = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

krigere = df[df["role"].isin(["Jarl", "Hirdman"])]
rate_krigere = krigere["survived"].mean()
eldste = df.loc[df["age"] > 55, ["name", "age", "role"]]
```
```test
import numbers
import pandas as pd
assert isinstance(krigere, pd.DataFrame) and len(krigere) == 138, "krigere skal være df[df['role'].isin(['Jarl', 'Hirdman'])] (138 rader)."
assert isinstance(rate_krigere, numbers.Real) and abs(rate_krigere - 109 / 138) < 1e-9, "rate_krigere skal være krigere['survived'].mean()."
assert isinstance(eldste, pd.DataFrame) and list(eldste.columns) == ["name", "age", "role"], "eldste skal ha kolonnene name, age og role."
assert len(eldste) == 4 and (eldste["age"] > 55).all(), f"eldste skal ha de 4 radene med age > 55, men har {len(eldste)}."
```
:::

--- LEKSJON 3.5 | manglende | Manglende verdier
Manglende verdier er `NaN`, og de fleste modeller krasjer på dem. `df.isna().sum()` teller dem per kolonne, og `df.isna().mean()` gir andelen.

::: oppgave Finn hullene
Lagre antall manglende verdier per kolonne i `mangler` og andelen per kolonne i `andel_mangler`.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

mangler = ...
andel_mangler = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

mangler = df.isna().sum()
andel_mangler = df.isna().mean()
```
```test
import pandas as pd
assert isinstance(mangler, pd.Series) and mangler["age"] == 80 and mangler["armor"] == 324, "mangler skal være df.isna().sum() (age: 80, armor: 324)."
assert isinstance(andel_mangler, pd.Series) and abs(andel_mangler["armor"] - 0.648) < 1e-9, "andel_mangler skal være df.isna().mean() (armor: 0.648)."
```
:::

## dropna: fjern rader
`df.dropna()` fjerner alle rader med minst én `NaN`, mens `df.dropna(subset=["age"])` bare ser på `age`.

::: oppgave dropna
Lagre `df.dropna()` i `alle_hele` og radene som har alder i `med_alder`. Sammenlign antall rader: hvor mye data kaster vanlig `dropna()`?
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

alle_hele = ...
med_alder = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

alle_hele = df.dropna()
med_alder = df.dropna(subset=["age"])
```
```test
import pandas as pd
assert isinstance(alle_hele, pd.DataFrame) and len(alle_hele) == 141, f"alle_hele skal være df.dropna() (141 rader)."
assert isinstance(med_alder, pd.DataFrame) and len(med_alder) == 420, "med_alder skal være df.dropna(subset=['age']) (420 rader)."
```
:::

## fillna: fyll inn verdier
`df["age"] = df["age"].fillna(df["age"].median())` fyller hullene, og resultatet må lagres fordi `fillna` (og `dropna`) returnerer en ny tabell. For tekst passer en egen kategori som `"Ukjent"` eller den vanligste verdien, `df["clan_origin"].mode()[0]`.

::: oppgave Lagre resultatet
Startkoden endrer ikke `df`. Rett den slik at manglende `age` faktisk fylles med medianen, og fyll manglende `clan_origin` med den vanligste verdien.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

df["age"].fillna(df["age"].median())
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

df["age"] = df["age"].fillna(df["age"].median())
df["clan_origin"] = df["clan_origin"].fillna(df["clan_origin"].mode()[0])
```
```test
assert len(df) == 500, "df skal fortsatt ha 500 rader. Fyll inn, ikke fjern."
assert df["age"].isna().sum() == 0, "age mangler fortsatt verdier. Lagre resultatet: df['age'] = df['age'].fillna(...)."
assert (df["age"] == 29.0).sum() == 98, "Manglende age skal fylles med medianen (29.0)."
assert df["clan_origin"].isna().sum() == 0, "clan_origin mangler fortsatt verdier."
assert (df["clan_origin"] == "Nidaros").sum() == 293, "Manglende clan_origin skal fylles med den vanligste verdien, df['clan_origin'].mode()[0]."
```
:::

::: oppgave Rydd vikingdataene
Lag `df_ren` fra `df` der manglende `age` fylles med medianen, manglende `clan_origin` med `"Ukjent"` og manglende `armor` med `"None"`. `df_ren` skal ha alle 500 radene og ingen `NaN`.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

df_ren = df.copy()
# fyll inn de manglende verdiene her

```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

df_ren = df.copy()
df_ren["age"] = df_ren["age"].fillna(df_ren["age"].median())
df_ren["clan_origin"] = df_ren["clan_origin"].fillna("Ukjent")
df_ren["armor"] = df_ren["armor"].fillna("None")
```
```test
assert len(df_ren) == 500, f"df_ren skal ha 500 rader, men har {len(df_ren)}. Ikke bruk dropna her."
_mangler = df_ren.isna().sum()
assert _mangler.sum() == 0, f"Det mangler fortsatt verdier: {_mangler[_mangler > 0].to_dict()}"
assert (df_ren["age"] == 29.0).sum() == 98, "Manglende age skal fylles med medianen (29.0)."
assert (df_ren["clan_origin"] == "Ukjent").sum() == 12, 'Manglende clan_origin skal fylles med "Ukjent".'
assert (df_ren["armor"] == "None").sum() == 324, 'Manglende armor skal fylles med "None".'
```
:::

## Manglende kan bety noe
I `armor` står det ofte teksten `None` (ingen rustning), som pandas leser som `NaN`. Med `pd.read_csv("problem2.csv", keep_default_na=False, na_values=[""])` blir bare tomme felt `NaN`.

::: oppgave None er ikke tomt
Les filen på nytt inn i `raa` slik at bare tomme felt blir `NaN`. Lagre antall rader der `armor` er teksten `"None"` i `n_none` og antall `NaN` i `armor` i `n_tomme`.
```start
import pandas as pd

raa = ...
n_none = ...
n_tomme = ...
```
```losning
import pandas as pd

raa = pd.read_csv("problem2.csv", keep_default_na=False, na_values=[""])
n_none = (raa["armor"] == "None").sum()
n_tomme = raa["armor"].isna().sum()
```
```test
import pandas as pd
assert isinstance(raa, pd.DataFrame) and len(raa) == 500, 'raa skal være pd.read_csv("problem2.csv", keep_default_na=False, na_values=[""]).'
assert n_none == 210, f"n_none skal være (raa['armor'] == 'None').sum() (210), men er {n_none}. Har du med keep_default_na=False?"
assert n_tomme == 114, f"n_tomme skal være raa['armor'].isna().sum() (114), men er {n_tomme}. Har du med na_values=['']?"
```
:::

## Medianen fra treningsdata
Når du skal evaluere en modell, regner du medianen ut fra `train` alene og bruker den på både `train` og `test`, ellers lekker informasjon fra testdataene inn i treningen.

::: oppgave Ingen lekkasje
Lagre medianalderen i `train` i `median_train`, og fyll manglende `age` i både `train` og `test` med den.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")
train = df.iloc[:250].copy()
test = df.iloc[250:].copy()

median_train = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")
train = df.iloc[:250].copy()
test = df.iloc[250:].copy()

median_train = train["age"].median()
train["age"] = train["age"].fillna(median_train)
test["age"] = test["age"].fillna(median_train)
```
```test
import pandas as pd
assert median_train == 29.0, f"median_train skal være train['age'].median() (29.0), men er {median_train}"
assert train["age"].isna().sum() == 0, "train['age'] mangler fortsatt verdier."
assert test["age"].isna().sum() == 0, "test['age'] mangler fortsatt verdier."
_hull = pd.read_csv("problem2.csv").iloc[250:]["age"].isna()
assert (test.loc[_hull[_hull].index, "age"] == 29.0).all(), "Hullene i test skal fylles med medianen fra train (29.0), ikke medianen fra test (30.0)."
```
:::

--- LEKSJON 3.6 | gruppere | Telle og gruppere
`df["survived"].value_counts()` teller hver verdi, og med `normalize=True` får du andeler, der den største andelen er accuracyen til en modell som alltid gjetter den vanligste klassen (baseline).

::: oppgave Klassefordeling
Lagre antallet per verdi av `survived` i `antall`, andelene i `andeler`, og den største andelen i `baseline`.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

antall = ...
andeler = ...
baseline = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

antall = df["survived"].value_counts()
andeler = df["survived"].value_counts(normalize=True)
baseline = andeler.max()
```
```test
import numbers
import pandas as pd
assert isinstance(antall, pd.Series) and antall[0] == 278 and antall[1] == 222, "antall skal være df['survived'].value_counts()."
assert isinstance(andeler, pd.Series) and abs(andeler[0] - 0.556) < 1e-9, "andeler skal være value_counts(normalize=True)."
assert isinstance(baseline, numbers.Real) and abs(baseline - 0.556) < 1e-9, f"baseline skal være den største andelen (0.556), men er {baseline}"
```
:::

## groupby: rate per gruppe
`df.groupby("sex")["survived"].mean()` gir overlevelsesraten per gruppe som en Series med gruppene som indeks, og `.agg(["mean", "count"])` gir flere tall per gruppe.

::: oppgave Overlevelse per kjønn
Lagre overlevelsesraten for kvinner og menn i `rate_female` og `rate_male`, og lag teksten `tekst = f"Female survival rate: {rate_female:.1%}"` (`:.1%` gjør 0.676 om til `67.6%`).
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

rate_female = ...
rate_male = ...
tekst = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

rate = df.groupby("sex")["survived"].mean()
rate_female = rate["Female"]
rate_male = rate["Male"]
tekst = f"Female survival rate: {rate_female:.1%}"
```
```test
import numbers
assert isinstance(rate_female, numbers.Real) and abs(rate_female - 92 / 136) < 1e-9, f"rate_female skal være andelen kvinner som overlevde (ca. 0.676), men er {rate_female}"
assert isinstance(rate_male, numbers.Real) and abs(rate_male - 130 / 364) < 1e-9, f"rate_male skal være andelen menn som overlevde (ca. 0.357), men er {rate_male}"
assert tekst == "Female survival rate: 67.6%", f"tekst skal være 'Female survival rate: 67.6%', men er {tekst!r}"
```
:::

::: oppgave Overlevelse per skip
Lag `rate_per_skip` med overlevelsesraten for hvert skip (`ship`) med `groupby`, og lagre raten for `"Sjøulven"` i `p_sjoulven`.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

rate_per_skip = ...
p_sjoulven = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

rate_per_skip = df.groupby("ship")["survived"].mean()
p_sjoulven = rate_per_skip["Sjøulven"]
```
```test
import numbers
import pandas as pd
assert isinstance(rate_per_skip, pd.Series), "rate_per_skip skal være en Series: df.groupby('ship')['survived'].mean()"
assert len(rate_per_skip) == 5, f"Det er 5 skip, men rate_per_skip har {len(rate_per_skip)} elementer."
assert isinstance(p_sjoulven, numbers.Real) and abs(p_sjoulven - 10 / 67) < 1e-9, f"p_sjoulven skal være overlevelsesraten på Sjøulven (ca. 0.149), men er {p_sjoulven}"
```
:::

::: oppgave Flere tall per rolle
Lag `per_rolle` med både `mean` og `count` av `survived` per `role` med `agg`, og hent antall jarler fra tabellen i `antall_jarler`.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

per_rolle = ...
antall_jarler = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

per_rolle = df.groupby("role")["survived"].agg(["mean", "count"])
antall_jarler = per_rolle.loc["Jarl", "count"]
```
```test
import pandas as pd
assert isinstance(per_rolle, pd.DataFrame) and list(per_rolle.columns) == ["mean", "count"], "per_rolle skal være df.groupby('role')['survived'].agg(['mean', 'count'])."
assert sorted(per_rolle.index) == ["Hirdman", "Jarl", "Karl", "Thrall"], "per_rolle skal ha én rad per rolle."
assert antall_jarler == 28, f"antall_jarler skal være per_rolle.loc['Jarl', 'count'] (28), men er {antall_jarler}"
```
:::

## crosstab: krysstabell
`pd.crosstab(df["sex"], df["survived"])` teller kombinasjoner, og med `normalize="index"` summerer hver rad til 1, så du får $P(\text{survived} \mid \text{sex})$.

::: oppgave Betingede sannsynligheter
Lag `tab` med antall per kombinasjon av `sex` og `survived`, og `p` med `normalize="index"`. Hent $P(\text{survived}=1 \mid \text{Female})$ fra `p` i `p_kvinne`, og regn ut $P(\text{Male} \mid \text{survived}=1)$ med en ny crosstab i `p_mann_gitt_overlevd`.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

tab = ...
p = ...
p_kvinne = ...
p_mann_gitt_overlevd = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

tab = pd.crosstab(df["sex"], df["survived"])
p = pd.crosstab(df["sex"], df["survived"], normalize="index")
p_kvinne = p.loc["Female", 1]
p_mann_gitt_overlevd = pd.crosstab(df["survived"], df["sex"], normalize="index").loc[1, "Male"]
```
```test
import numbers
import pandas as pd
assert isinstance(tab, pd.DataFrame) and tab.loc["Female", 1] == 92, "tab skal være pd.crosstab(df['sex'], df['survived']) (antall)."
assert isinstance(p, pd.DataFrame) and abs(p.loc["Male"].sum() - 1) < 1e-9, "p skal ha normalize='index', så hver rad summerer til 1."
assert isinstance(p_kvinne, numbers.Real) and abs(p_kvinne - 92 / 136) < 1e-9, f"p_kvinne skal være p.loc['Female', 1] (ca. 0.676), men er {p_kvinne}"
assert isinstance(p_mann_gitt_overlevd, numbers.Real) and abs(p_mann_gitt_overlevd - 130 / 222) < 1e-9, f"p_mann_gitt_overlevd skal være ca. 0.586. Snu rekkefølgen: pd.crosstab(df['survived'], df['sex'], normalize='index')."
```
:::

--- LEKSJON 3.7 | endre | Endre data
En ny kolonne lager du ved å tilordne til et nytt navn: `df["family_size"] = df["family_aboard"] + 1`.

::: oppgave Feature engineering
Lag kolonnene `family_size` (`family_aboard + 1`), `erfaren` (`True` når `prior_voyages` er minst 3) og `kg_per_person` (`provisions_kg` delt på `family_size`) i `df`.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

df["family_size"] = df["family_aboard"] + 1
df["erfaren"] = df["prior_voyages"] >= 3
df["kg_per_person"] = df["provisions_kg"] / df["family_size"]
```
```test
import numpy as np
for _k in ["family_size", "erfaren", "kg_per_person"]:
    assert _k in df.columns, f"Lag kolonnen {_k} i df."
assert (df["family_size"] == df["family_aboard"] + 1).all(), "family_size skal være family_aboard + 1."
assert df["erfaren"].sum() == 147 and set(df["erfaren"]) <= {True, False}, "erfaren skal være df['prior_voyages'] >= 3 (147 er erfarne)."
assert np.allclose(df["kg_per_person"], df["provisions_kg"] / (df["family_aboard"] + 1)), "kg_per_person skal være provisions_kg / family_size."
```
:::

## drop og rename
`df.drop(columns=["name"])` fjerner kolonner og `df.rename(columns={"sex": "kjonn"})` gir nye navn, og begge returnerer en ny tabell som må lagres.

::: oppgave drop og rename
Lag `df2` fra `df` uten kolonnene `passenger_id` og `name`, og der `sex` heter `kjonn` og `survived` heter `overlevde`.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

df2 = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

df2 = df.drop(columns=["passenger_id", "name"])
df2 = df2.rename(columns={"sex": "kjonn", "survived": "overlevde"})
```
```test
import pandas as pd
assert isinstance(df2, pd.DataFrame) and len(df2) == 500, "df2 skal være en DataFrame med 500 rader."
assert "passenger_id" not in df2.columns and "name" not in df2.columns, "Fjern passenger_id og name med drop(columns=[...])."
assert "kjonn" in df2.columns and "overlevde" in df2.columns, "Gi nye navn med rename(columns={'sex': 'kjonn', 'survived': 'overlevde'})."
assert "sex" not in df2.columns and "survived" not in df2.columns, "De gamle navnene sex og survived skal være borte."
assert df2.shape[1] == 11, f"df2 skal ha 11 kolonner, men har {df2.shape[1]}."
```
:::

## map: kode om med en dict
`df["sex"].map({"Male": 1, "Female": 0})` slår opp hver verdi i dicten, og verdier som ikke finnes i dicten blir `NaN`.

::: oppgave Rett map-feilen
Startkoden gjør hele `sex` til `NaN`. Rett den slik at `"Male"` blir 1 og `"Female"` blir 0, og sjekk med `df["sex"].isna().sum()`.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

df["sex"] = df["sex"].map({"male": 1, "female": 0})
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
```
```test
assert df["sex"].isna().sum() == 0, 'sex har NaN etter omkodingen. Nøklene må stemme nøyaktig med dataene: "Male" og "Female" med stor forbokstav.'
assert sorted(df["sex"].unique().tolist()) == [0, 1], "sex skal bare inneholde 0 og 1."
assert df["sex"].sum() == 364, "Male skal bli 1 og Female 0 (det er 364 menn)."
```
:::

## apply: bruk en egen funksjon
`df["age"].apply(funksjon)` kaller funksjonen på hver verdi, og `lambda n: n.split()[-1]` er en kort funksjon på én linje.

::: oppgave Aldersgruppe og etternavn
Fullfør `aldersgruppe` (under 20 gir `"ung"`, under 40 gir `"voksen"`, ellers `"eldre"`) og lag kolonnen `gruppe` med `apply`. Lag også kolonnen `etternavn` med det siste ordet i `name`.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")
df["age"] = df["age"].fillna(df["age"].median())

def aldersgruppe(alder):
    ...

```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")
df["age"] = df["age"].fillna(df["age"].median())

def aldersgruppe(alder):
    if alder < 20:
        return "ung"
    elif alder < 40:
        return "voksen"
    return "eldre"

df["gruppe"] = df["age"].apply(aldersgruppe)
df["etternavn"] = df["name"].apply(lambda n: n.split()[-1])
```
```test
assert aldersgruppe(15) == "ung" and aldersgruppe(20) == "voksen" and aldersgruppe(39) == "voksen" and aldersgruppe(40) == "eldre", "aldersgruppe gir feil svar. Under 20: ung, under 40: voksen, ellers eldre."
assert "gruppe" in df.columns, "Lag kolonnen gruppe med df['age'].apply(aldersgruppe)."
assert df["gruppe"].tolist() == [aldersgruppe(a) for a in df["age"]], "gruppe skal være aldersgruppe brukt på hver alder."
assert "etternavn" in df.columns, "Lag kolonnen etternavn."
assert df.loc[0, "etternavn"] == "Ulfsson" and df.loc[1, "etternavn"] == "Bjørnsson", "etternavn skal være siste ord i name: n.split()[-1]."
```
:::

## astype: endre type
`data["y"].astype(int)` gjør `0.0` og `1.0` om til heltall, og det virker også på `True`/`False`.

::: oppgave Til heltall
Gjør `data["y"]` om til heltall (lagre tilbake i kolonnen), og lag `positiv` som er 1 der `x0 > 0` og 0 ellers.
```start
import pandas as pd
data = pd.read_csv("classification.csv")

positiv = ...
```
```losning
import pandas as pd
data = pd.read_csv("classification.csv")

data["y"] = data["y"].astype(int)
positiv = (data["x0"] > 0).astype(int)
```
```test
import pandas as pd
assert pd.api.types.is_integer_dtype(data["y"]), f"data['y'] skal være heltall, men har typen {data['y'].dtype}. Bruk data['y'] = data['y'].astype(int)."
assert set(data["y"]) == {0, 1}, "data['y'] skal fortsatt være 0 og 1."
assert isinstance(positiv, pd.Series) and pd.api.types.is_integer_dtype(positiv), "positiv skal være (data['x0'] > 0).astype(int)."
assert (positiv == (data["x0"] > 0)).all(), "positiv skal være 1 der x0 > 0 og 0 ellers."
```
:::

## sort_values og reset_index
`df.sort_values("kol", ascending=False)` returnerer en sortert kopi som beholder de gamle merkelappene, og `reset_index(drop=True)` gir ny indeks 0, 1, 2, …. Uten `drop=True` blir den gamle indeksen en kolonne, og `reset_index(names="ID")` kaller den `ID`.

::: oppgave Sorter og nummerer
Lagre de tre radene med mest `provisions_kg` i `topp3`, gi `test` ny indeks 0, 1, 2, … uten å beholde den gamle, og lag `med_id` der indeksen til `df` er blitt en kolonne `ID`.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")
data = pd.read_csv("classification.csv")
test = data[data["split"] == "test"]

topp3 = ...
med_id = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")
data = pd.read_csv("classification.csv")
test = data[data["split"] == "test"]

topp3 = df.sort_values("provisions_kg", ascending=False).head(3)
test = test.reset_index(drop=True)
med_id = df.reset_index(names="ID")
```
```test
import pandas as pd
assert isinstance(topp3, pd.DataFrame) and topp3.index.tolist() == [201, 91, 172], "topp3 skal være df.sort_values('provisions_kg', ascending=False).head(3) (radene 201, 91 og 172)."
assert test.index[0] == 0 and len(test) == 500, "test skal ha ny indeks fra 0. Husk test = test.reset_index(drop=True)."
assert "index" not in test.columns, "Med drop=True blir ikke den gamle indeksen en kolonne."
assert isinstance(med_id, pd.DataFrame) and med_id.columns[0] == "ID", "med_id skal være df.reset_index(names='ID')."
assert med_id["ID"].tolist() == list(range(500)), "ID-kolonnen skal være 0, 1, ..., 499."
```
:::

--- LEKSJON 3.8 | kategoriske | Kategoriske data: map og get_dummies
Binære kategorier som `sex` kodes med `map` til 0/1, ordinale kategorier med en naturlig rekkefølge med `map` i riktig rekkefølge, og nominelle kategorier uten rekkefølge (`ship`, `destination`) med one-hot.

## Ordinal: map med egen rekkefølge
Rollene har rangen Thrall < Karl < Hirdman < Jarl, så `{"Thrall": 0, "Karl": 1, "Hirdman": 2, "Jarl": 3}` gir én feature der større tall betyr høyere rang.

::: oppgave Rang
Lag kolonnen `role_rang` med dicten over, og lagre overlevelsesraten per rang i `rate_per_rang`. Ser du at raten øker med rangen?
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

rate_per_rang = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

rang = {"Thrall": 0, "Karl": 1, "Hirdman": 2, "Jarl": 3}
df["role_rang"] = df["role"].map(rang)
rate_per_rang = df.groupby("role_rang")["survived"].mean()
```
```test
import pandas as pd
assert "role_rang" in df.columns and df["role_rang"].isna().sum() == 0, "Lag role_rang med df['role'].map({...}) uten NaN."
assert (df.loc[df["role"] == "Jarl", "role_rang"] == 3).all() and (df.loc[df["role"] == "Thrall", "role_rang"] == 0).all(), "Thrall skal bli 0 og Jarl 3."
assert isinstance(rate_per_rang, pd.Series) and rate_per_rang.index.tolist() == [0, 1, 2, 3], "rate_per_rang skal være df.groupby('role_rang')['survived'].mean()."
assert rate_per_rang.is_monotonic_increasing, "Raten skal øke med rangen. Sjekk rekkefølgen i dicten."
```
:::

## One-hot med get_dummies
`pd.get_dummies(df, columns=["ship"], dtype=int)` lager én 0/1-kolonne per kategori (`ship_Bison`, …) og lar de andre kolonnene stå urørt. Uten `dtype=int` blir det `True`/`False`.

::: oppgave One-hot på skip
Lag `oh` fra `df[["ship", "survived"]]` der `ship` er one-hot-kodet med 0/1.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

oh = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

oh = pd.get_dummies(df[["ship", "survived"]], columns=["ship"], dtype=int)
```
```test
import pandas as pd
assert isinstance(oh, pd.DataFrame), "oh skal være en DataFrame fra pd.get_dummies."
assert oh.shape == (500, 6), f"oh skal ha shape (500, 6): survived + 5 skip. Du har {oh.shape}."
assert "survived" in oh.columns and "ship" not in oh.columns, "ship skal erstattes av 0/1-kolonner, og survived skal stå urørt."
_skip = [c for c in oh.columns if c.startswith("ship_")]
assert (oh[_skip].sum(axis=1) == 1).all(), "Hver rad skal ha nøyaktig én 1 blant ship_-kolonnene."
assert pd.api.types.is_integer_dtype(oh["ship_Bison"]), "Bruk dtype=int for å få 0/1 i stedet for True/False."
```
:::

## drop_first og dummy_na
`drop_first=True` fjerner den første (overflødige) kategorien per feature, og `dummy_na=True` gir en egen kolonne for `NaN`, som ellers bare får nuller.

::: oppgave One-hot med drop_first
Lag `df_oh` fra `utvalg` der `sex`, `role` og `ship` er kodet med `drop_first=True` og `dtype=int`, og `survived` står urørt.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")
utvalg = df[["sex", "role", "ship", "survived"]]

df_oh = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")
utvalg = df[["sex", "role", "ship", "survived"]]

df_oh = pd.get_dummies(utvalg, columns=["sex", "role", "ship"], drop_first=True, dtype=int)
```
```test
import pandas as pd
assert isinstance(df_oh, pd.DataFrame), "df_oh skal være en DataFrame fra pd.get_dummies."
assert "survived" in df_oh.columns, "survived skal fortsatt være med."
assert not any(c in df_oh.columns for c in ["sex", "role", "ship"]), "sex, role og ship skal være erstattet av 0/1-kolonner."
assert df_oh.shape == (500, 9), f"Forventet shape (500, 9): 1 + 3 + 4 kodede kolonner + survived. Du har {df_oh.shape}. Husk drop_first=True."
assert "sex_Female" not in df_oh.columns and "role_Hirdman" not in df_oh.columns, "drop_first=True skal fjerne den første kategorien (alfabetisk)."
```
:::

::: oppgave Manglende som egen kategori
Lag `armor_oh` fra `df[["armor"]]` der `armor` er one-hot-kodet med 0/1 og en egen kolonne for manglende verdier.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

armor_oh = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

armor_oh = pd.get_dummies(df[["armor"]], columns=["armor"], dummy_na=True, dtype=int)
```
```test
import pandas as pd
assert isinstance(armor_oh, pd.DataFrame), "armor_oh skal være en DataFrame fra pd.get_dummies."
assert "armor_nan" in armor_oh.columns, "Det mangler en kolonne for NaN. Bruk dummy_na=True."
assert armor_oh.shape == (500, 4), f"armor_oh skal ha 4 kolonner (3 rustninger + nan), men har shape {armor_oh.shape}."
assert armor_oh["armor_nan"].sum() == 324, "armor_nan skal være 1 for de 324 radene uten armor."
```
:::

## Samme kolonner i train og test
Kjører du `get_dummies` på train og test hver for seg, får de ulike kolonner hvis en kategori mangler i den ene, så kod hele tabellen før du deler den.

::: oppgave Kod før du deler
Test-radene mangler skipet `Drakvinge`. One-hot-kod `ship` i hele `df` (med `dtype=int`) i `oh`, og del så `oh` i `train_oh` og `test_oh` etter `split`.
```start
import pandas as pd
df = pd.DataFrame({
    "ship": ["Bison", "Drakvinge", "Sjøulven", "Bison", "Sjøulven"],
    "split": ["train", "train", "train", "test", "test"],
})

oh = ...
train_oh = ...
test_oh = ...
```
```losning
import pandas as pd
df = pd.DataFrame({
    "ship": ["Bison", "Drakvinge", "Sjøulven", "Bison", "Sjøulven"],
    "split": ["train", "train", "train", "test", "test"],
})

oh = pd.get_dummies(df, columns=["ship"], dtype=int)
train_oh = oh[oh["split"] == "train"]
test_oh = oh[oh["split"] == "test"]
```
```test
import pandas as pd
assert isinstance(oh, pd.DataFrame) and "ship_Drakvinge" in oh.columns, "oh skal være pd.get_dummies(df, columns=['ship'], dtype=int)."
assert isinstance(train_oh, pd.DataFrame) and len(train_oh) == 3, "train_oh skal være de 3 train-radene i oh."
assert isinstance(test_oh, pd.DataFrame) and len(test_oh) == 2, "test_oh skal være de 2 test-radene i oh."
assert list(train_oh.columns) == list(test_oh.columns), "train_oh og test_oh skal ha de samme kolonnene. Kod hele df før du deler."
assert "ship_Drakvinge" in test_oh.columns, "test_oh skal også ha kolonnen ship_Drakvinge (med bare nuller)."
```
:::

--- LEKSJON 3.9 | til-numpy | Fra pandas til X og y, og lagre CSV
`X = df.drop(columns="survived")` og `y = df["survived"]` deler tabellen i features (2D) og target (1D).

::: oppgave X og y
Lagre navnet på target-kolonnen i `target = "survived"`, og lag `X` (alle andre kolonner) og `y` med den.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

target = ...
X = ...
y = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

target = "survived"
X = df.drop(columns=target)
y = df[target]
```
```test
import pandas as pd
assert target == "survived", "target skal være 'survived'."
assert isinstance(X, pd.DataFrame) and X.shape == (500, 12) and "survived" not in X.columns, "X skal være df.drop(columns=target) med shape (500, 12)."
assert isinstance(y, pd.Series) and y.shape == (500,), "y skal være df[target], en 1D Series."
```
:::

## Til numpy: .values og .to_numpy()
`.values` og `.to_numpy()` gjør en tabell om til et numpy-array, og `to_numpy(dtype=np.float32)` gir 32-bits flyttall til PyTorch. Er det tekst igjen i tabellen, får arrayet `dtype object`, og modellene krasjer.

::: oppgave numpy-arrays
Lag `X_train` (kolonnene `x0` og `x1`) og `y_train` (kolonnen `y`) som numpy-arrays fra `train`, og `X_train32` som `X_train` med `dtype=np.float32`.
```start
import numpy as np
import pandas as pd
data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]

X_train = ...
y_train = ...
X_train32 = ...
```
```losning
import numpy as np
import pandas as pd
data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]

X_train = train[["x0", "x1"]].values
y_train = train["y"].to_numpy()
X_train32 = train[["x0", "x1"]].to_numpy(dtype=np.float32)
```
```test
import numpy as np
assert isinstance(X_train, np.ndarray) and X_train.shape == (500, 2), "X_train skal være train[['x0', 'x1']].values med shape (500, 2)."
assert isinstance(y_train, np.ndarray) and y_train.shape == (500,), "y_train skal være train['y'].to_numpy() med shape (500,)."
assert isinstance(X_train32, np.ndarray) and X_train32.dtype == np.float32 and X_train32.shape == (500, 2), "X_train32 skal lages med .to_numpy(dtype=np.float32)."
```
:::

::: oppgave Fjern teksten
Startkoden gir en `X` med `dtype object` fordi `split` er med. Rett den slik at `X` bare inneholder `x0` og `x1`.
```start
import pandas as pd
data = pd.read_csv("classification.csv")

X = data.drop(columns="y").values
```
```losning
import pandas as pd
data = pd.read_csv("classification.csv")

X = data[["x0", "x1"]].values
```
```test
import numpy as np
assert isinstance(X, np.ndarray), "X skal være et numpy-array."
assert X.dtype != object, "X har fortsatt dtype object. Er split-kolonnen med?"
assert X.shape == (1000, 2), f"X skal ha shape (1000, 2) (bare x0 og x1), men har {X.shape}"
```
:::

## pd.concat: slå sammen tabeller
`pd.concat([a, b])` setter tabeller under hverandre, og `pd.concat([a, b], axis=1)` setter dem ved siden av hverandre, matchet på indeksen.

::: oppgave Under og ved siden av
Sett `train` og `test` sammen igjen under hverandre i `alle`. Sett så `df[["age", "survived"]]` og `dummies` ved siden av hverandre i `sammen`.
```start
import pandas as pd
data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
df = pd.read_csv("problem2.csv")
dummies = pd.get_dummies(df["ship"], dtype=int)

alle = ...
sammen = ...
```
```losning
import pandas as pd
data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
df = pd.read_csv("problem2.csv")
dummies = pd.get_dummies(df["ship"], dtype=int)

alle = pd.concat([train, test])
sammen = pd.concat([df[["age", "survived"]], dummies], axis=1)
```
```test
import pandas as pd
assert isinstance(alle, pd.DataFrame) and alle.shape == (1000, 4), "alle skal være pd.concat([train, test]) med shape (1000, 4)."
assert isinstance(sammen, pd.DataFrame) and sammen.shape == (500, 7), f"sammen skal ha shape (500, 7). Bruk axis=1 for å sette ved siden av hverandre."
assert list(sammen.columns[:2]) == ["age", "survived"], "age og survived skal stå først i sammen."
```
:::

## Lagre til CSV
`df.to_csv("submission.csv", index=False)` lagrer tabellen, og uten `index=False` blir indeksen en ekstra, navnløs kolonne.

::: oppgave Kaggle-innlevering
Lag `submission` med kolonnene `ID` (0, 1, 2, …) og `cluster` (`labels`), og lagre den til `submission.csv` uten indeks.
```start
import numpy as np
import pandas as pd
labels = np.array([2, 0, 1, 1, 0])

submission = ...
```
```losning
import numpy as np
import pandas as pd
labels = np.array([2, 0, 1, 1, 0])

submission = pd.DataFrame({"ID": np.arange(len(labels)), "cluster": labels})
submission.to_csv("submission.csv", index=False)
```
```test
import os
import pandas as pd
assert isinstance(submission, pd.DataFrame) and list(submission.columns) == ["ID", "cluster"], "submission skal ha kolonnene ID og cluster."
assert os.path.exists("submission.csv"), "Lagre filen med submission.to_csv('submission.csv', index=False)."
_fil = pd.read_csv("submission.csv")
os.remove("submission.csv")
assert list(_fil.columns) == ["ID", "cluster"], f"Filen har kolonnene {list(_fil.columns)}. Husk index=False."
assert _fil["ID"].tolist() == [0, 1, 2, 3, 4] and _fil["cluster"].tolist() == [2, 0, 1, 1, 0], "Sjekk verdiene i ID og cluster."
```
:::

## Hele veien
Nå kan du hele kjeden fra rådata til modellklare arrays: `drop`, `fillna`, `get_dummies` og `to_numpy`.

::: oppgave Fra rådata til X og y
Fjern `passenger_id` og `name`, fyll `age` med medianen, `clan_origin` med `"Ukjent"` og `armor` med `"None"`, og one-hot-kod `sex`, `role`, `ship`, `destination`, `clan_origin` og `armor` med `drop_first=True` og `dtype=int`. Lag så numpy-arrays `y` (`survived`) og `X` (resten), der `X` får shape `(500, 22)`.
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

X = ...
y = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

df = df.drop(columns=["passenger_id", "name"])
df["age"] = df["age"].fillna(df["age"].median())
df["clan_origin"] = df["clan_origin"].fillna("Ukjent")
df["armor"] = df["armor"].fillna("None")
df = pd.get_dummies(df, columns=["sex", "role", "ship", "destination", "clan_origin", "armor"],
                    drop_first=True, dtype=int)
X = df.drop(columns="survived").to_numpy()
y = df["survived"].to_numpy()
```
```test
import numpy as np
assert isinstance(X, np.ndarray), "X skal være et numpy-array. Bruk .to_numpy() eller .values."
assert isinstance(y, np.ndarray), "y skal være et numpy-array."
assert y.shape == (500,), f"y skal ha shape (500,), men har {y.shape}. Bruk df['survived'], ikke df[['survived']]."
assert X.ndim == 2 and X.shape[0] == 500, f"X skal ha 500 rader og være 2D, men har shape {X.shape}"
assert X.dtype != object, "X har typen object. Er det fortsatt tekstkolonner igjen?"
assert not np.isnan(X.astype(float)).any(), "X inneholder NaN. Fyll inn age, clan_origin og armor."
assert X.shape[1] == 22, f"X skal ha 22 kolonner (4 numeriske + 18 one-hot), men har {X.shape[1]}. Husk drop_first=True og å fjerne survived."
assert set(np.unique(y)) == {0, 1}, "y skal være survived (0 og 1)."
```
:::
