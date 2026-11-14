=== DEL 3 | pandas | pandas: tabeller med data
pandas gir NumPy-arrays kolonnenavn. Alle prosjektene starter med `pd.read_csv`, og her lærer du å lese, utforske, rydde og kode om data til du står igjen med `X` og `y` som en modell kan trenes på.

--- LEKSJON 3.1 | series-dataframe | Series og DataFrame
NumPy-arrays er raske, men kolonnene har ingen navn. **pandas** bygger på NumPy og gir deg tabeller der hver kolonne heter noe (`age`, `sex`, `survived`) og hver rad har en merkelapp, kalt **indeks**. Alle prosjektene i faget starter med `import pandas as pd`.

pandas har to hovedtyper:

- **Series**: én kolonne. Et 1D-array der hvert element har en merkelapp.
- **DataFrame**: en hel tabell. Hver kolonne er en Series, og alle kolonnene deler samme indeks.

## Series

```python
import pandas as pd

alder = pd.Series([25, 41, 29, 32], name="age")
print(alder)
print("Gjennomsnitt:", alder.mean())
print("Som numpy-array:", alder.values)
```

Venstre kolonne i utskriften er indeksen (0, 1, 2, 3), høyre kolonne er verdiene. Indeksen trenger ikke være tall:

```python
import pandas as pd

rate = pd.Series([0.68, 0.36], index=["Female", "Male"])
print(rate)
print("Kvinner:", rate["Female"])
```

Akkurat en slik Series får du når du regner ut overlevelsesrate per kjønn i prosjekt 2 (leksjon 3.6).

## DataFrame fra en dict
Den enkleste måten å lage en tabell for hånd er en dict der nøklene er kolonnenavn og verdiene er like lange lister:

```python
import pandas as pd

df = pd.DataFrame({
    "name": ["Finn", "Runa", "Bjørn", "Liv"],
    "sex": ["Male", "Female", "Male", "Female"],
    "age": [25, 32, 29, 43],
    "survived": [0, 0, 1, 1],
})
print(df.shape)          # (rader, kolonner)
print(list(df.columns))
df
```

Siste linje, `df`, vises som en tabell, akkurat som i Jupyter. `df.shape` virker som i NumPy: (antall rader, antall kolonner).

## Se tabellen: variabelpanelet og display()
Under resultatet av hver kjøring ligger **variabelpanelet** («Variabler etter kjøring»). Der står alle variablene koden laget, og klikker du på en DataFrame, vises den som en tabell med kolonnenavn, type per kolonne og indeksen til venstre. Manglende verdier (`NaN`) er markert i **rødt**, og panelet teller hvor mange som mangler. Panelet viser variablene slik de er **til slutt**.

Vil du se en tabell **midt i** koden, for eksempel før og etter en endring, skriver du `display(df)`. Da vises et øyeblikksbilde av tabellen akkurat der i utdataene:

```python
import pandas as pd

df = pd.DataFrame({"name": ["Finn", "Runa", "Bjørn"], "age": [25, 32, 29]})
print("Før:")
display(df)
df["age"] = df["age"] + 1
print("Etter at alle er blitt ett år eldre:")
display(df)
```

`display` finnes også i Jupyter, så koden virker uendret der. (I en vanlig `.py`-fil må du skrive `from IPython.display import display` først.) Vi bruker `display` gjennom hele kurset til å vise hva en operasjon gjør med tabellen.

## Kolonner oppfører seg som numpy-arrays
Én kolonne hentes med `df["navn"]`, og resultatet er en Series. Du regner på den med de samme vektoriserte operasjonene som i NumPy:

```python
import pandas as pd

df = pd.DataFrame({"age": [25, 32, 29, 43], "survived": [0, 0, 1, 1]})
print(df["age"] + 1)
print("Snittalder:", df["age"].mean())
print("Eldst:", df["age"].max())
print("Andel overlevde:", df["survived"].mean())
```

::: tips
Gjennomsnittet av en kolonne med 0 og 1 er andelen enere. `df["survived"].mean()` er derfor overlevelsesraten. Dette trikset kommer du til å bruke hele tiden.
:::

::: oppgave Lag en tabell
Lag en DataFrame `mannskap` med kolonnene `navn`, `alder` og `overlevde` (i den rekkefølgen) og disse tre radene:

| navn | alder | overlevde |
|---|---|---|
| Ivar | 30 | 1 |
| Sigrid | 22 | 1 |
| Ulf | 51 | 0 |

Lagre deretter gjennomsnittsalderen i variabelen `snitt_alder` (regn den ut fra tabellen).
```start
import pandas as pd

mannskap = ...
snitt_alder = ...
```
```losning
import pandas as pd

mannskap = pd.DataFrame({
    "navn": ["Ivar", "Sigrid", "Ulf"],
    "alder": [30, 22, 51],
    "overlevde": [1, 1, 0],
})
snitt_alder = mannskap["alder"].mean()
```
```test
import pandas as pd
assert isinstance(mannskap, pd.DataFrame), "mannskap skal være en DataFrame. Bruk pd.DataFrame({...})."
assert list(mannskap.columns) == ["navn", "alder", "overlevde"], f"Kolonnene skal være navn, alder, overlevde, men er {list(mannskap.columns)}"
assert mannskap.shape == (3, 3), f"Tabellen skal ha 3 rader og 3 kolonner, men har shape {mannskap.shape}"
assert list(mannskap["navn"]) == ["Ivar", "Sigrid", "Ulf"], "Sjekk navnene og rekkefølgen på radene."
assert list(mannskap["alder"]) == [30, 22, 51], "Sjekk alder-kolonnen."
assert list(mannskap["overlevde"]) == [1, 1, 0], "Sjekk overlevde-kolonnen."
assert snitt_alder is not ... and abs(snitt_alder - 103 / 3) < 1e-6, "snitt_alder skal være gjennomsnittet av alder-kolonnen, altså mannskap['alder'].mean()."
```
:::

::: quiz
Hva skriver koden ut?
```kode
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "c": [9, 10, 11, 12]})
print(df.shape)
```
- [ ] `(3, 4)`
- [x] `(4, 3)`
- [ ] `(12,)`
- [ ] `12`
? Hver nøkkel i dicten blir en kolonne (3 kolonner), og hver liste har 4 elementer (4 rader). `shape` er alltid (rader, kolonner).
:::

::: quiz
`df` er en DataFrame med en kolonne `age`. Hvilken type har `df["age"]`?
- [ ] `DataFrame`
- [x] `Series`
- [ ] `list`
- [ ] `numpy.ndarray`
? Én kolonne hentet med enkle hakeparenteser er en `Series`. Vil du ha et numpy-array, skriver du `df["age"].values` eller `df["age"].to_numpy()`.
:::

::: oppsummering
- `import pandas as pd` er standard.
- **Series** = én kolonne med indeks. **DataFrame** = tabell der hver kolonne er en Series.
- `pd.DataFrame({"kolonne": [verdier], ...})` lager en tabell fra en dict.
- `df.shape` er (rader, kolonner), `df.columns` er kolonnenavnene.
- Kolonner støtter NumPy-regning: `df["age"] + 1`, `.mean()`, `.max()`.
- Gjennomsnittet av en 0/1-kolonne er andelen enere.
:::

--- LEKSJON 3.2 | lese-data | Lese og utforske data
I prosjektene skriver du aldri inn data for hånd. Du leser en CSV-fil (kommaseparerte verdier) med `pd.read_csv`. Første steg etterpå er alltid å **se på dataene**: hvor mange rader, hvilke kolonner, hvilke typer, og hva som mangler.

Vi bruker vikingdataene fra prosjekt 2 gjennom hele denne delen. Hver rad er en passasjer på et vikingtokt, og målet er å predikere `survived`.

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
df.head()
```

`head()` viser de fem første radene (`head(10)` viser ti). `tail()` viser de siste.

## Størrelse, kolonner og typer

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
print(df.shape)
print(df.columns.tolist())
print(df.dtypes)
df.tail(3)
```

`dtypes` sier hvilken type hver kolonne har. `int64` er heltall, `float64` er desimaltall, og tekstkolonner vises som `object` (eller `str` i den nyeste pandas-versjonen). Legg merke til at `age` er `float64` selv om alle aldre er heltall. Det er fordi noen aldre mangler, og «mangler» lagres som `NaN`, som er et flyttall.

## info() og describe()
`info()` gir en oversikt med antall verdier som **ikke** mangler per kolonne. `describe()` gir statistikk for alle tallkolonnene.

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
df.info()
```

Det er 500 rader, men `age` har bare 420 «non-null». Altså mangler alderen for 80 passasjerer. Åpner du `df` i variabelpanelet, ser du det samme: antall manglende verdier står per kolonne, og de tomme cellene er røde. Det må vi håndtere før vi trener en modell (leksjon 3.5).

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
df.describe()
```

Radene er antall (`count`), gjennomsnitt (`mean`), standardavvik (`std`), minimum, kvartiler (`50%` er medianen) og maksimum. Se på `provisions_kg`: snittet er rundt 25, men maks er over 300. Slike forskjeller i skala er grunnen til at vi skalerer features senere (leksjon 7.2).

## Valg i read_csv
Som standard tolkes første linje i filen som kolonnenavn. Noen ganger er ikke det riktig:

```python
import pandas as pd

# Vanlig: første linje er kolonnenavn
print(pd.read_csv("classification.csv", nrows=3))

# header=None: første linje blir en vanlig datarad, kolonnene får nummer 0, 1, 2, ...
print(pd.read_csv("classification.csv", header=None, nrows=3))

# skiprows=1 hopper over første linje helt
print(pd.read_csv("classification.csv", skiprows=1, header=None, nrows=3))

# nrows=0 leser bare kolonnenavnene
print(pd.read_csv("classification.csv", nrows=0).columns.tolist())
```

`nrows=3` leser bare de tre første radene. Det er nyttig for å kikke på store filer.

::: prosjekt Prosjekt 3, problem 2
Første linje i `unsupervised_data.csv` var ikke vanlige kolonnenavn, men metadata (den opprinnelige pikselrekkefølgen). Derfor ble filen lest med `pd.read_csv("unsupervised_data.csv", skiprows=1, header=None)`. Senere ble metadataene hentet ut alene med `pd.read_csv("unsupervised_data.csv", nrows=0).columns`.
:::

::: oppgave Utforsk problem1.csv
Les `problem1.csv` (prosjekt 2, problem 1) inn i variabelen `data`. Lagre så:

- `n_rader`: antall rader
- `n_kolonner`: antall kolonner
- `kolonner`: kolonnenavnene som en vanlig Python-liste
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

::: quiz
Du kjører `df.info()` og får blant annet dette:
```kode
RangeIndex: 891 entries, 0 to 890
 #   Column  Non-Null Count  Dtype
 0   Age     714 non-null    float64
 1   Sex     891 non-null    object
```
Hvor mange passasjerer mangler alder?
- [ ] 714
- [x] 177
- [ ] 0, siden Age har typen float64
- [ ] 891
? Det er 891 rader, og 714 av dem har en alder. 891 − 714 = 177 mangler. `float64` sier ingenting om hvorvidt verdier mangler (manglende verdier lagres som `NaN`, som selv er et flyttall).
:::

::: quiz
Filen `data.csv` har én linje med kolonnenavn og deretter 100 linjer med data, med 3 verdier på hver linje. Hva skriver koden ut?
```kode
df = pd.read_csv("data.csv", header=None)
print(df.shape)
```
- [ ] `(100, 3)`
- [x] `(101, 3)`
- [ ] `(100, 4)`
- [ ] `(101, 4)`
? Med `header=None` tolkes ikke første linje som kolonnenavn, så den blir en vanlig datarad. Da får vi 101 rader. Antall kolonner er fortsatt 3.
:::

::: oppsummering
- `pd.read_csv("fil.csv")` leser en CSV-fil til en DataFrame.
- `head()`, `tail()`, `shape`, `columns`, `dtypes` gir et første overblikk.
- `info()` viser antall ikke-manglende verdier per kolonne. `describe()` gir statistikk for tallkolonnene.
- `header=None`: første linje er data. `skiprows=1`: hopp over første linje. `nrows=n`: les bare n rader.
- En heltallskolonne med manglende verdier blir `float64`, fordi `NaN` er et flyttall.
:::

--- LEKSJON 3.3 | velge | Velge kolonner og rader (loc og iloc)
Før du kan trene en modell, må du plukke ut riktige kolonner (features og target) og noen ganger bestemte rader. Det er tre verktøy: hakeparenteser for kolonner, `iloc` for posisjoner og `loc` for merkelapper.

## Én kolonne eller flere: Series eller DataFrame
Dette er en av de vanligste kildene til feil:

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
a = df["age"]             # én hakeparentes: Series
b = df[["age"]]           # liste med ett navn: DataFrame
c = df[["age", "sex"]]    # liste med to navn: DataFrame
print(type(a).__name__, a.shape)
print(type(b).__name__, b.shape)
print(type(c).__name__, c.shape)
```

`df["age"]` er 1D med shape `(500,)`. `df[["age"]]` er 2D med shape `(500, 1)`. I variabelpanelet står `a` som en Series og `b` som en DataFrame med 500×1. Det betyr noe fordi scikit-learn krever at `X` er 2D (rader × features), mens `y` skal være 1D. Derfor ser prosjektkoden slik ut:

```python
import pandas as pd

data = pd.read_csv("classification.csv")
X = data[["x0", "x1"]]   # 2D: features
y = data["y"]            # 1D: target
print(X.shape, y.shape)
```

::: husk
Har du bare én feature, må du fortsatt bruke doble hakeparenteser: `X = df[["Net_Activity"]]`. Med `df["Net_Activity"]` klager scikit-learn over at den forventer et 2D-array.
:::

Du kan også skrive `df.age` i stedet for `df["age"]` (forelesningsnotatene skriver for eksempel `df.Pclass`). Det virker bare for kolonnenavn uten mellomrom, og ikke når du lager nye kolonner, så hakeparenteser er tryggest.

## iloc: velg etter posisjon
`iloc` virker som indeksering i NumPy: `df.iloc[rader, kolonner]` med heltallsposisjoner. Slutten i en slice er **ikke** med, akkurat som i Python ellers.

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
print(df.iloc[0])           # første rad, som en Series
print(df.iloc[0, 1])        # rad 0, kolonne 1 (name)
df.iloc[:3, 1:4]            # rad 0-2, kolonne 1-3
```

## loc: velg etter merkelapp
`loc` bruker indeksens merkelapper og kolonnenavn: `df.loc[rader, kolonner]`. Her er slutten i en slice **med**.

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
print(df.loc[0, "name"])
df.loc[0:2, ["name", "age"]]   # rad 0, 1 OG 2
```

| | `iloc` | `loc` |
|---|---|---|
| Velger etter | posisjon (0, 1, 2, ...) | merkelapp i indeksen og kolonnenavn |
| `[0:2]` gir | 2 rader (0 og 1) | 3 rader (0, 1 og 2) |
| Kolonner | `df.iloc[:, 2]` | `df.loc[:, "age"]` |

Se forskjellen i slicing med egne øyne:

```python
import pandas as pd

df = pd.read_csv("problem2.csv")[["name", "age", "role"]]
print("df.iloc[0:2]: posisjon 0 og 1")
display(df.iloc[0:2])
print("df.loc[0:2]: merkelapp 0, 1 og 2")
display(df.loc[0:2])
```

## Når indeksen ikke lenger er 0, 1, 2, ...
Så lenge indeksen er 0, 1, 2, ... gir `loc` og `iloc` samme rad. Etter at du har valgt ut noen rader, beholder radene sine gamle merkelapper:

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
utvalg = df.iloc[10:15]
display(utvalg[["name", "age"]])   # se indeksen til venstre: 10, 11, ...
print(utvalg.index.tolist())
print(utvalg.iloc[0]["name"])   # første rad i utvalget
print(utvalg.loc[10]["name"])   # raden med merkelapp 10: samme rad
```

Det finnes ingen rad med merkelapp 0 i utvalget, så dette krasjer:

```python feil
import pandas as pd

df = pd.read_csv("problem2.csv")
utvalg = df.iloc[10:15]
utvalg.loc[0]
```

`KeyError: 0` betyr at merkelappen 0 ikke finnes. Dette skjer ofte etter at du har delt data i train og test (neste leksjon). Bruk `iloc` når du mener «den første raden».

::: prosjekt Prosjekt 3
`row = data.iloc[0, 1:]` henter første rad og alle kolonner unntatt den første (ID-kolonnen). Raden ble så gjort om til et bilde med `reshape(128, 128)`. `features.iloc[:, 2]` henter kolonne nummer 2 for å tegne et histogram.
:::

::: oppgave Plukk ut data
`df` er vikingdataene. Lag:

- `forste_navn`: navnet (`name`) til passasjeren i første rad, som en streng
- `tre_kolonner`: en DataFrame med kolonnene `name`, `age` og `survived` (alle rader)
- `rad_5_til_9`: radene med posisjon 5, 6, 7, 8 og 9 (alle kolonner)
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

forste_navn = ...
tre_kolonner = ...
rad_5_til_9 = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

forste_navn = df.iloc[0]["name"]
tre_kolonner = df[["name", "age", "survived"]]
rad_5_til_9 = df.iloc[5:10]
```
```test
import pandas as pd
assert forste_navn == "Finn Ulfsson", f"forste_navn skal være navnet i første rad, men er {forste_navn!r}"
assert isinstance(tre_kolonner, pd.DataFrame), "tre_kolonner skal være en DataFrame. Husk doble hakeparenteser: df[[...]]."
assert list(tre_kolonner.columns) == ["name", "age", "survived"], f"tre_kolonner har kolonnene {list(tre_kolonner.columns)}"
assert len(tre_kolonner) == 500, "tre_kolonner skal ha alle 500 radene."
assert isinstance(rad_5_til_9, pd.DataFrame), "rad_5_til_9 skal være en DataFrame."
assert rad_5_til_9.index.tolist() == [5, 6, 7, 8, 9], f"rad_5_til_9 skal ha radene 5 til 9, men har {rad_5_til_9.index.tolist()}. Husk at slutten ikke er med i iloc."
```
:::

::: quiz
Hva skriver koden ut?
```kode
df = pd.DataFrame({"x": [10, 20, 30, 40, 50]})
a = df.loc[1:3]
b = df.iloc[1:3]
print(len(a), len(b))
```
- [ ] `2 2`
- [x] `3 2`
- [ ] `3 3`
- [ ] `2 3`
? `loc` tar med slutten (merkelappene 1, 2 og 3), mens `iloc` følger vanlig Python-slicing og stopper før posisjon 3 (posisjonene 1 og 2).
:::

::: quiz
Du skal trene en modell med én feature, `Net_Activity`. Hvilken linje gir en `X` som scikit-learn godtar i `model.fit(X, y)`?
- [ ] `X = df["Net_Activity"]`
- [x] `X = df[["Net_Activity"]]`
- [ ] `X = df.Net_Activity`
- [ ] `X = df.loc["Net_Activity"]`
? `X` må være 2D med shape (rader, features). Doble hakeparenteser gir en DataFrame med shape `(n, 1)`. De to første feilalternativene gir en 1D Series, og `df.loc["Net_Activity"]` leter etter en **rad** med den merkelappen.
:::

::: oppsummering
- `df["a"]` gir en **Series** (1D). `df[["a", "b"]]` gir en **DataFrame** (2D).
- `X` til scikit-learn skal være 2D (`df[["x0", "x1"]]`), `y` skal være 1D (`df["y"]`).
- `iloc[rader, kolonner]` bruker posisjoner. Slutten i en slice er ikke med.
- `loc[rader, kolonner]` bruker merkelapper og kolonnenavn. Slutten er med.
- Etter utvalg beholder radene gamle merkelapper, så `loc[0]` kan gi `KeyError`.
:::

--- LEKSJON 3.4 | filtrering | Filtrering av rader
Svært ofte vil du bare ha radene som oppfyller en betingelse: passasjerer over 40, bare kvinner, eller bare treningsdataene. Det gjøres med **boolsk indeksering**, akkurat som med NumPy-arrays i del 2.

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
maske = df["age"] > 40
print(maske.head())
print("Antall over 40:", maske.sum())
eldre = df[maske]
print(eldre.shape)
eldre.head()
```

`df["age"] > 40` gir en Series med `True`/`False` for hver rad. `df[maske]` beholder bare radene der masken er `True`. Rader der alderen mangler (`NaN`) gir `False`.

Med et lite utsnitt er det lett å se hvilke rader som forsvinner. Legg merke til rad 7, der alderen mangler (rød):

```python
import pandas as pd

liten = pd.read_csv("problem2.csv")[["name", "sex", "age"]].head(8)
print("Før filtrering:")
display(liten)
print("liten[liten['age'] > 30]:")
display(liten[liten["age"] > 30])
```

Radene som er igjen, beholder merkelappene sine (1, 3, 5, ...), og rad 7 er borte fordi `NaN > 30` er `False`.

## Train og test fra split-kolonnen
I prosjekt 1, 2 og 3 ligger trenings- og testdata i samme fil, med en kolonne `split`. Dette mønsteret står i alle prosjektene:

```python
import pandas as pd

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
print(train.shape, test.shape)
print("Første merkelapper i test:", test.index[:5].tolist())
display(test.head(3))
```

Legg merke til at `test` beholder sine opprinnelige merkelapper. Første rad i `test` har ikke merkelapp 0, og `split`-kolonnen er bare `test`. Åpne `train` og `test` i variabelpanelet for å se begge delene.

## Flere betingelser: &, | og ~
Betingelser kombineres med `&` (og), `|` (eller) og `~` (ikke). **Hver betingelse må stå i parentes.**

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
menn_over_40 = df[(df["sex"] == "Male") & (df["age"] > 40)]
jarl_eller_hird = df[(df["role"] == "Jarl") | (df["role"] == "Hirdman")]
ikke_thrall = df[~(df["role"] == "Thrall")]
print(len(menn_over_40), len(jarl_eller_hird), len(ikke_thrall))
print("Menn over 40:")
display(menn_over_40[["name", "sex", "age"]].head())
print("Jarler eller hirdmenn:")
display(jarl_eller_hird[["name", "role"]].head())
```

Uten parenteser går det galt, fordi `&` regnes ut før `==` og `>`. Python prøver da å regne ut `40 & df["sex"]`:

```python feil
import pandas as pd

df = pd.read_csv("problem2.csv")
df[df["age"] > 40 & df["sex"] == "Male"]
```

::: advarsel
Python-ordene `and` og `or` virker ikke på hele kolonner. `df[(df["age"] > 40) and (df["sex"] == "Male")]` gir `ValueError: The truth value of a Series is ambiguous`. Bruk `&` og `|`.
:::

## isin og loc med maske
Når en kolonne skal være én av flere verdier, er `isin` kortere enn mange `|`. Du kan også bruke en maske i `loc` for å velge rader og kolonner samtidig:

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
krigere = df[df["role"].isin(["Jarl", "Hirdman"])]
print(len(krigere))
print("Overlevelsesrate for krigere:", krigere["survived"].mean())
df.loc[df["age"] > 55, ["name", "age", "role"]]
```

::: tips
Skal du endre kolonner i `train` etterpå, så lag en kopi: `train = data[data["split"] == "train"].copy()`. Da er `train` garantert en selvstendig tabell, og eldre pandas-versjoner slutter å gi `SettingWithCopyWarning`.
:::

::: oppgave Unge kvinner
Lag `unge_kvinner`, som inneholder radene der `sex` er `"Female"` og `age` er under 25. Lagre så overlevelsesraten deres (andelen med `survived == 1`) i `rate_unge_kvinner`.
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
import pandas as pd
_df = pd.read_csv("problem2.csv")
_fasit = _df[(_df["sex"] == "Female") & (_df["age"] < 25)]
assert isinstance(unge_kvinner, pd.DataFrame), "unge_kvinner skal være en DataFrame (df[maske])."
assert len(unge_kvinner) == len(_fasit), f"unge_kvinner skal ha {len(_fasit)} rader, men har {len(unge_kvinner)}. Husk parenteser rundt hver betingelse og &."
assert (unge_kvinner["sex"] == "Female").all(), "Det er menn i unge_kvinner."
assert (unge_kvinner["age"] < 25).all(), "Noen i unge_kvinner er 25 år eller eldre."
assert rate_unge_kvinner is not ... and abs(rate_unge_kvinner - _fasit["survived"].mean()) < 1e-9, "rate_unge_kvinner skal være gjennomsnittet av survived i unge_kvinner."
```
:::

::: quiz
Hvilken linje velger mennene som er eldre enn 40?
- [ ] `df[df["sex"] == "Male" and df["age"] > 40]`
- [x] `df[(df["sex"] == "Male") & (df["age"] > 40)]`
- [ ] `df[df["sex"] == "Male" & df["age"] > 40]`
- [ ] `df[(df["sex"] == "Male") | (df["age"] > 40)]`
? Betingelser på kolonner kombineres med `&`, og hver betingelse må ha parenteser. `and` gir `ValueError`, uten parenteser regnes `&` ut først og gir feil, og `|` betyr «eller».
:::

::: quiz
Hva skriver koden ut?
```kode
df = pd.DataFrame({"y": [0, 1, 1, 0, 1],
                   "split": ["train", "test", "train", "train", "test"]})
train = df[df["split"] == "train"]
print(train.index.tolist(), train["y"].sum())
```
- [ ] `[0, 1, 2] 1`
- [x] `[0, 2, 3] 1`
- [ ] `[0, 2, 3] 2`
- [ ] `[0, 1, 2] 2`
? Filtrering beholder de opprinnelige merkelappene, så train har indeksene 0, 2 og 3. y-verdiene der er 0, 1 og 0, som summerer til 1.
:::

::: oppsummering
- `df[betingelse]` beholder radene der betingelsen er `True`.
- `train = data[data["split"] == "train"]` er standardmønsteret i prosjektene.
- Kombiner med `&`, `|` og `~`, og sett **parenteser rundt hver betingelse**. Ikke bruk `and`/`or`.
- `df["kol"].isin([...])` sjekker om verdien er en av flere.
- `df.loc[maske, ["a", "b"]]` velger rader og kolonner samtidig.
- Filtrerte rader beholder sine gamle merkelapper. Bruk `.copy()` hvis du skal endre utvalget.
:::

--- LEKSJON 3.5 | manglende | Manglende verdier
Ekte data har hull. pandas markerer manglende verdier som `NaN` (Not a Number). De fleste modeller i scikit-learn krasjer hvis `X` inneholder `NaN`, så hullene må fjernes eller fylles før trening. Forelesningsnotatene nevner begge veier: `dropna()` og å fylle inn verdier.

## Finn hullene

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
print(df.isna().sum())
```

`df.isna()` gir en tabell med `True` der en verdi mangler. `.sum()` teller `True` per kolonne. `df.isna().mean()` gir andelen som mangler.

## dropna: fjern rader
`dropna()` fjerner alle rader der **minst én** verdi mangler. Med `subset` ser den bare på bestemte kolonner:

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
print("Før:", df.shape)
print("dropna():", df.dropna().shape)
print("dropna(subset=['age']):", df.dropna(subset=["age"]).shape)
```

Med vanlig `dropna()` mister vi 359 av 500 rader, nesten bare fordi `armor` mangler ofte. Det er sjelden lurt å kaste så mye data. `dropna(subset=["age"])` fjerner bare de 80 radene uten alder.

Tallene blir lettere å forstå når du ser tabellen. Her er de åtte første radene med tre kolonner. De røde cellene er `NaN`:

```python
import pandas as pd

df = pd.read_csv("problem2.csv")[["name", "age", "armor"]].head(8)
print("Før:")
display(df)
print("df.dropna(): alle rader med minst én NaN er borte")
display(df.dropna())
print('df.dropna(subset=["age"]): bare raden uten alder er borte')
display(df.dropna(subset=["age"]))
```

`dropna()` tar med seg fem av åtte rader, selv om alderen bare mangler i én av dem. Se også på indeksen: radene som er igjen, beholder merkelappene sine.

## fillna: fyll inn verdier
Et alternativ er å fylle hullene. For tall er medianen et vanlig valg, fordi den ikke påvirkes mye av ekstreme verdier. For tekst kan du bruke en egen kategori:

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
print("Før fillna (rad 61 mangler klan, rad 62 mangler alder):")
display(df.loc[58:63, ["name", "age", "clan_origin"]])

median_alder = df["age"].median()
print("Medianalder:", median_alder)
df["age"] = df["age"].fillna(median_alder)
df["clan_origin"] = df["clan_origin"].fillna("Ukjent")
print("Etter fillna:")
display(df.loc[58:63, ["name", "age", "clan_origin"]])
print(df[["age", "clan_origin"]].isna().sum())
```

De røde cellene før er fylt etterpå: rad 62 har fått medianen 29.0, og rad 61 har fått klanen `Ukjent`.

Den vanligste verdien (typetallet) får du med `df["clan_origin"].mode()[0]`. Det er et annet vanlig valg for tekstkolonner.

::: advarsel
`fillna` og `dropna` endrer ikke tabellen, de **returnerer** en ny. Skriv derfor alltid `df["age"] = df["age"].fillna(...)` eller `df = df.dropna()`. Linjen `df["age"].fillna(29, inplace=True)` virker ikke i nyere pandas, og `df["age"].fillna(29)` alene gjør ingenting med `df`.
:::

## Manglende kan bety noe
Studer `armor`. I selve CSV-filen står det ofte teksten `None` (ingen rustning), og noen ganger står feltet tomt (ukjent). pandas tolker begge deler som `NaN`! Du kan be pandas om bare å tolke tomme felt som manglende:

```python
import pandas as pd

vanlig = pd.read_csv("problem2.csv")
raa = pd.read_csv("problem2.csv", keep_default_na=False, na_values=[""])
print("Vanlig innlesing:", vanlig["armor"].isna().sum(), "mangler")
print(raa["armor"].value_counts(dropna=False))
display(pd.DataFrame({"vanlig": vanlig["armor"].head(6), "raa": raa["armor"].head(6)}))
```

(`value_counts` teller hvor mange ganger hver verdi forekommer. Mer om den i neste leksjon.) I tabellen til slutt er fire av seks celler røde i `vanlig`, mens bare rad 3 er tom i `raa`. Uansett innlesing er det et poeng her: at rustning «mangler» sier noe om passasjeren. Da er det bedre å la det være en egen kategori enn å kaste radene: `df["armor"] = df["armor"].fillna("None")`.

::: hvorfor Hvorfor ikke bare fylle med snittet av hele datasettet?
For utforsking er `df["age"].fillna(df["age"].median())` helt greit. Når du skal **evaluere** en modell, bør medianen regnes ut fra treningsdataene alene, ellers lekker informasjon fra testdataene inn i treningen. I leksjon 7.4 gjør vi dette riktig med `SimpleImputer`.
:::

::: oppgave Rydd vikingdataene
Lag `df_ren` fra `df` slik at:

- manglende `age` fylles med medianen av `age`
- manglende `clan_origin` fylles med `"Ukjent"`
- manglende `armor` fylles med `"None"`

`df_ren` skal fortsatt ha alle 500 radene og ingen manglende verdier.
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
import pandas as pd
_orig = pd.read_csv("problem2.csv")
assert (df_ren["age"] == 29.0).sum() == 80 + (_orig["age"] == 29.0).sum(), "Manglende age skal fylles med medianen (29.0)."
assert (df_ren["clan_origin"] == "Ukjent").sum() == 12, 'Manglende clan_origin skal fylles med "Ukjent".'
assert (df_ren["armor"] == "None").sum() == 324, 'Manglende armor skal fylles med "None".'
```
:::

::: quiz
Hva skriver koden ut?
```kode
import numpy as np
import pandas as pd
df = pd.DataFrame({"a": [1, np.nan, 3, 4],
                   "b": [np.nan, 2, 3, np.nan],
                   "c": [1, 2, 3, 4]})
print(len(df.dropna()), len(df.dropna(subset=["a"])))
```
- [x] `1 3`
- [ ] `2 3`
- [ ] `3 1`
- [ ] `1 1`
? `dropna()` fjerner alle rader med minst én `NaN`, og bare rad 2 er komplett. `dropna(subset=["a"])` ser bare på kolonne `a`, der bare rad 1 mangler, så 3 rader blir igjen.
:::

::: quiz
`df["age"]` har manglende verdier. Du kjører linjen under. Hvor mange verdier mangler i `df["age"]` etterpå?
```kode
df["age"].fillna(df["age"].median())
```
- [ ] 0, fordi alle er fylt med medianen
- [x] Like mange som før, fordi resultatet ikke lagres
- [ ] 0, fordi radene med manglende alder fjernes
- [ ] Koden krasjer fordi medianen ikke kan regnes ut med `NaN`
? `fillna` returnerer en ny Series og endrer ikke `df`. Du må skrive `df["age"] = df["age"].fillna(...)`. `median()` hopper for øvrig over `NaN` automatisk.
:::

::: oppsummering
- `df.isna().sum()` teller manglende verdier per kolonne.
- `df.dropna()` fjerner rader med minst én `NaN`. `dropna(subset=["age"])` ser bare på `age`.
- `df["age"] = df["age"].fillna(df["age"].median())` fyller inn medianen. Husk å **lagre resultatet**.
- Tekstkolonner kan fylles med en egen kategori (`"Ukjent"`) eller den vanligste verdien (`mode()[0]`).
- At en verdi mangler kan i seg selv være informasjon.
:::

--- LEKSJON 3.6 | gruppere | Telle og gruppere
Før du trener en modell, bør du vite hvordan klassene fordeler seg og hvilke features som henger sammen med target. Prosjekt 2 spør nettopp om dette: «Er det features som alene predikerer overlevelse bedre enn tilfeldig gjetting?»

## value_counts: tell verdier

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
print(df["survived"].value_counts())
print(df["survived"].value_counts(normalize=True))
display(df["role"].value_counts())
```

`normalize=True` gir andeler i stedet for antall. 55,6 % døde, så en «modell» som alltid gjetter 0 får 55,6 % accuracy. Det er grunnlinjen (baseline) en ekte modell må slå. Andelene er også sannsynlighetene $P(C)$ du trenger i Naive Bayes.

## Overlevelsesrate per gruppe
Med filtrering fra forrige leksjon kan du regne ut raten for hver gruppe:

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
rate_female = df[df["sex"] == "Female"]["survived"].mean()
rate_male = df[df["sex"] == "Male"]["survived"].mean()
print(f"Female survival rate: {rate_female:.1%}")
print(f"Male survival rate: {rate_male:.1%}")
```

`groupby` gjør det samme for alle grupper på én gang. Den deler radene i grupper etter `sex`, velger kolonnen `survived` og tar gjennomsnittet i hver gruppe:

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
rate = df.groupby("sex")["survived"].mean()
print(rate)
print("Kvinner:", rate["Female"])
```

Resultatet er en Series med gruppene som indeks. Åpne `rate` i variabelpanelet: 500 rader er blitt til to, én per kjønn. Vil du ha flere tall per gruppe, bruker du `agg`:

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
df.groupby("role")["survived"].agg(["mean", "count"])
```

Rollen sier mye: nesten alle jarler overlever, mens bare rundt 14 % av trellene (Thrall) gjør det. Både `sex` og `role` predikerer altså langt bedre enn tilfeldig gjetting.

## crosstab: krysstabell
`pd.crosstab` teller kombinasjoner av to kolonner. Med `normalize="index"` summerer hver **rad** til 1, og du får betingede sannsynligheter:

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
print("Antall:")
display(pd.crosstab(df["sex"], df["survived"]))
print('normalize="index": hver rad summerer til 1')
display(pd.crosstab(df["sex"], df["survived"], normalize="index"))
```

Den andre tabellen er $P(\text{survived} \mid \text{sex})$: 67,6 % av kvinnene overlevde. Snur du rekkefølgen, `pd.crosstab(df["survived"], df["sex"], normalize="index")`, får du $P(\text{sex} \mid \text{survived})$. Det er nettopp sannsynlighetene Naive Bayes bruker (leksjon 8.7). Forelesningsnotatene bruker den samme funksjonen: `pd.crosstab(df.Pclass, df.Sex, normalize="index")`.

::: prosjekt Prosjekt 2, problem 2
Analysecellen skal ende med `print(f"Female survival rate: {rate_female:.1%}")` og tilsvarende for menn. `:.1%` gjør 0.676 om til `67.6%`.
:::

::: oppgave Overlevelse per kjønn
Regn ut overlevelsesraten for kvinner og menn og lagre dem i `rate_female` og `rate_male` (tall mellom 0 og 1).
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

rate_female = ...
rate_male = ...
print(f"Female survival rate: {rate_female:.1%}")
print(f"Male survival rate: {rate_male:.1%}")
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

rate = df.groupby("sex")["survived"].mean()
rate_female = rate["Female"]
rate_male = rate["Male"]
print(f"Female survival rate: {rate_female:.1%}")
print(f"Male survival rate: {rate_male:.1%}")
```
```test
assert abs(rate_female - 92 / 136) < 1e-9, f"rate_female skal være andelen kvinner som overlevde (ca. 0.676), men er {rate_female}"
assert abs(rate_male - 130 / 364) < 1e-9, f"rate_male skal være andelen menn som overlevde (ca. 0.357), men er {rate_male}"
```
:::

::: oppgave Overlevelse per skip
Bruk `groupby` til å lage `rate_per_skip`, en Series med overlevelsesraten for hvert skip (`ship`). Lagre så raten for skipet `"Sjøulven"` i `p_sjoulven`.
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
import pandas as pd
assert isinstance(rate_per_skip, pd.Series), "rate_per_skip skal være en Series: df.groupby('ship')['survived'].mean()"
assert len(rate_per_skip) == 5, f"Det er 5 skip, men rate_per_skip har {len(rate_per_skip)} elementer."
assert abs(p_sjoulven - 10 / 67) < 1e-9, f"p_sjoulven skal være overlevelsesraten på Sjøulven (ca. 0.149), men er {p_sjoulven}"
```
:::

::: quiz
Hva skriver koden ut?
```kode
df = pd.DataFrame({"klasse": [1, 1, 1, 2, 2, 2, 2],
                   "kjonn": ["m", "m", "f", "m", "f", "f", "f"]})
p = pd.crosstab(df["klasse"], df["kjonn"], normalize="index")
print(p.loc[2, "m"])
```
- [x] `0.25`
- [ ] `0.3333333333333333`
- [ ] `0.14285714285714285`
- [ ] `1`
? `normalize="index"` gjør at hver rad (klasse) summerer til 1. Klasse 2 har fire personer, og én av dem er m, så svaret er 1/4. 1/3 er `normalize="columns"` (én av tre m-er er i klasse 2), 1/7 er `normalize="all"`, og 1 er det rå antallet.
:::

::: quiz
`df["survived"].value_counts(normalize=True)` gir `0: 0.556` og `1: 0.444`. Hvilken accuracy får en modell som alltid predikerer 0?
- [ ] 0.444
- [ ] 0.500
- [x] 0.556
- [ ] 1.000
? Modellen har rett for alle som faktisk har `survived == 0`, altså 55,6 % av radene. En nyttig modell må slå denne grunnlinjen.
:::

::: oppsummering
- `value_counts()` teller verdier, `value_counts(normalize=True)` gir andeler (klassefordeling, baseline, $P(C)$).
- `df.groupby("sex")["survived"].mean()` gir overlevelsesrate per gruppe.
- `.agg(["mean", "count"])` gir flere tall per gruppe.
- `pd.crosstab(a, b)` teller kombinasjoner. `normalize="index"` gjør at hver rad summerer til 1, som gir $P(b \mid a)$.
:::

--- LEKSJON 3.7 | endre | Endre data
Data kommer sjelden i formen en modell trenger. I denne leksjonen lærer du de vanligste operasjonene: lage nye kolonner, fjerne og gi nytt navn til kolonner, kode om verdier, endre typer og sortere.

## Nye kolonner
Du lager en ny kolonne ved å tilordne til et nytt navn. Regningen er vektorisert, som i NumPy:

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
print("Før: kolonnene vi regner ut fra")
display(df[["family_aboard", "prior_voyages", "provisions_kg"]].head())
df["family_size"] = df["family_aboard"] + 1
df["erfaren"] = df["prior_voyages"] >= 3
df["kg_per_person"] = df["provisions_kg"] / df["family_size"]
print("Etter: tre nye kolonner")
df[["family_aboard", "family_size", "erfaren", "kg_per_person"]].head()
```

Å lage nye features fra de gamle kalles **feature engineering**. Du gjorde det samme for hånd i prosjekt 1 (leksjon 5.8).

## drop og rename
`drop(columns=...)` fjerner kolonner, og `rename(columns={gammelt: nytt})` gir nye navn. Begge returnerer en ny tabell, så resultatet må lagres:

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
print("Før:")
display(df.head(3))
df = df.drop(columns=["passenger_id", "name"])
df = df.rename(columns={"sex": "kjonn", "survived": "overlevde"})
print("Etter drop og rename:")
display(df.head(3))
print(df.columns.tolist())
```

Navn og ID-er er typiske kolonner å fjerne: de er unike per person og hjelper ikke modellen med å generalisere.

## map: kode om verdier med en dict
`map` slår opp hver verdi i en dict. Det er den enkleste måten å gjøre tekst om til tall:

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
display(df[["name", "sex"]].head())
df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
display(df[["name", "sex"]].head())
print(df["sex"].value_counts())
print("Mangler etter map:", df["sex"].isna().sum())
```

::: advarsel
Verdier som ikke finnes i dicten blir `NaN`. I disse dataene står det `"Male"` med stor forbokstav. Skriver du `{"male": 1, "female": 0}`, blir hele kolonnen `NaN`. Sjekk alltid med `isna().sum()` etter en `map`.
```python
import pandas as pd

df = pd.read_csv("problem2.csv")[["name", "sex"]].head(6)
df["riktig"] = df["sex"].map({"Male": 1, "Female": 0})
df["feil"] = df["sex"].map({"male": 1, "female": 0})   # små forbokstaver finnes ikke
display(df)
print(df.isna().sum())
```
Hele `feil`-kolonnen er rød.
:::

## apply: bruk en egen funksjon
`apply` kaller en funksjon på hver verdi i en Series. Det er nyttig når regelen er for komplisert for en dict:

```python
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
df[["name", "age", "gruppe", "etternavn"]].head()
```

## astype: endre type
`y` i `classification.csv` er lagret som `0.0` og `1.0`. Med `astype` gjør du kolonnen om til heltall. Det samme virker for `True`/`False`:

```python
import pandas as pd

data = pd.read_csv("classification.csv")
print(data["y"].dtype, data["y"].head(3).tolist())
data["y"] = data["y"].astype(int)
print(data["y"].dtype, data["y"].head(3).tolist())
print((data["x0"] > 0).astype(int).head(3).tolist())
```

Typen til hver kolonne står også under kolonnenavnet når du åpner `data` i variabelpanelet.

## sort_values og reset_index
`sort_values` sorterer radene etter en kolonne. Radene beholder merkelappene sine. `reset_index(drop=True)` gir en ny indeks 0, 1, 2, ...:

```python
import pandas as pd

data = pd.read_csv("classification.csv")
test = data[data["split"] == "test"]
print(test.index[:3].tolist())
display(test.head(3))
test = test.reset_index(drop=True)
print(test.index[:3].tolist())
display(test.head(3))

df = pd.read_csv("problem2.csv")
df.sort_values("provisions_kg", ascending=False)[["name", "provisions_kg"]].head(3)
```

Etter `sort_values` står de største verdiene øverst, men merkelappene til venstre er de gamle radnumrene.

Uten `drop=True` blir den gamle indeksen en ny kolonne. I prosjekt 3 ble dette brukt med vilje: `data.reset_index(names="ID")` gjør radnummeret om til en kolonne kalt `ID`, som trengs i Kaggle-innleveringen.

::: oppgave Gjør klar kolonnene
Lag `df2` fra vikingdataene slik at:

1. kolonnene `passenger_id` og `name` er fjernet
2. `sex` er kodet om til tall: `"Male"` blir 1 og `"Female"` blir 0
3. det finnes en ny kolonne `family_size` som er `family_aboard + 1`
```start
import pandas as pd
df = pd.read_csv("problem2.csv")

df2 = ...
```
```losning
import pandas as pd
df = pd.read_csv("problem2.csv")

df2 = df.drop(columns=["passenger_id", "name"])
df2["sex"] = df2["sex"].map({"Male": 1, "Female": 0})
df2["family_size"] = df2["family_aboard"] + 1
```
```test
import pandas as pd
assert isinstance(df2, pd.DataFrame), "df2 skal være en DataFrame."
assert "passenger_id" not in df2.columns and "name" not in df2.columns, "Fjern passenger_id og name med df.drop(columns=[...])."
assert len(df2) == 500, "df2 skal fortsatt ha 500 rader."
assert df2["sex"].isna().sum() == 0, 'sex har NaN etter omkodingen. Bruker du nøyaktig "Male" og "Female" som nøkler?'
assert sorted(df2["sex"].unique().tolist()) == [0, 1], "sex skal bare inneholde 0 og 1."
assert df2["sex"].sum() == 364, "Male skal bli 1 og Female 0 (det er 364 menn)."
assert "family_size" in df2.columns, "Lag kolonnen family_size."
assert (df2["family_size"] == df["family_aboard"] + 1).all(), "family_size skal være family_aboard + 1."
```
:::

::: quiz
Hva skriver koden ut?
```kode
s = pd.Series(["Male", "Female", "male"])
print(s.map({"Male": 1, "Female": 0}).tolist())
```
- [ ] `[1, 0, 1]`
- [x] `[1.0, 0.0, nan]`
- [ ] `[1, 0]`
- [ ] `KeyError: 'male'`
? `map` gir `NaN` for verdier som ikke finnes i dicten (`"male"` med liten forbokstav). Fordi `NaN` er et flyttall, blir hele resultatet flyttall.
:::

::: quiz
Hva skriver koden ut?
```kode
df = pd.DataFrame({"a": [3, 1, 2]})
df.sort_values("a")
print(df["a"].tolist())
```
- [x] `[3, 1, 2]`
- [ ] `[1, 2, 3]`
- [ ] `[3, 2, 1]`
- [ ] `None`
? `sort_values` returnerer en ny, sortert tabell, men resultatet lagres ikke. `df` er uendret. Skriv `df = df.sort_values("a")` for å beholde sorteringen.
:::

::: oppsummering
- Ny kolonne: `df["ny"] = df["a"] + 1` (vektorisert).
- `df.drop(columns=[...])` og `df.rename(columns={gammelt: nytt})` returnerer en ny tabell. Lagre resultatet.
- `df["sex"].map({"Male": 1, "Female": 0})`: verdier som ikke finnes i dicten blir `NaN`.
- `apply(funksjon)` bruker en egen funksjon på hver verdi.
- `astype(int)` endrer type. `sort_values("kol")` sorterer. `reset_index(drop=True)` gir ny indeks 0, 1, 2, ...
:::

--- LEKSJON 3.8 | kategoriske | Kategoriske data: map og get_dummies
Modeller regner med tall, så tekstkolonner som `sex`, `role` og `ship` må kodes om. Hvordan du gjør det avhenger av hva slags kategori det er:

| Type | Eksempel | Koding |
|---|---|---|
| Binær (to verdier) | `sex` | `map` til 0 og 1 |
| Ordinal (har en naturlig rekkefølge) | `role`: Thrall < Karl < Hirdman < Jarl | `map` til 0, 1, 2, 3 i riktig rekkefølge |
| Nominell (ingen rekkefølge) | `ship`, `destination` | one-hot med `pd.get_dummies` |

## Ordinal: map med egen rekkefølge

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
rang = {"Thrall": 0, "Karl": 1, "Hirdman": 2, "Jarl": 3}
df["role_rang"] = df["role"].map(rang)
df.groupby("role_rang")["survived"].mean()
```

Overlevelsesraten øker jevnt med rangen, så å kode rollen som ett tall med riktig rekkefølge gir mening.

## Nominell: one-hot med get_dummies
Skipene har ingen rekkefølge. Gir du dem tallene 0 til 4, later modellen som om skip 4 er «mer» enn skip 1. **One-hot-koding** lager i stedet én kolonne per kategori, med 1 der raden har den kategorien og 0 ellers:

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
print("Før: én tekstkolonne")
display(df[["ship", "survived"]].head())
oh = pd.get_dummies(df[["ship", "survived"]], columns=["ship"], dtype=int)
print("Etter: én 0/1-kolonne per skip")
oh.head()
```

Sammenlign radene: rad 0 hadde skipet `Bison`, og etter kodingen har den 1 i `ship_Bison` og 0 i de andre skipskolonnene. `columns=[...]` sier hvilke kolonner som skal kodes. Andre kolonner blir stående urørt. Uten `dtype=int` får du `True`/`False`, som modellene også godtar.

## drop_first
De fem skipskolonnene summerer alltid til 1. Den siste kolonnen er derfor overflødig: er alle de andre 0, vet vi hvilket skip det er. `drop_first=True` fjerner den første kategorien:

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
oh = pd.get_dummies(df[["ship"]], columns=["ship"], drop_first=True, dtype=int)
print(oh.columns.tolist())
oh.head()
```

Raden der alle fire er 0, er `Bison`. Overflødige kolonner kan gi problemer for lineære modeller uten regularisering (vektene blir ikke entydige). Beslutningstrær bryr seg ikke.

## Hele tabellen på én gang
Du kan kode mange kolonner samtidig. Rader med `NaN` får 0 i alle kolonnene for den featuren, med mindre du gir `dummy_na=True`:

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
df = df.drop(columns=["passenger_id", "name"])
df_oh = pd.get_dummies(df, columns=["sex", "role", "ship", "destination", "clan_origin", "armor"], dtype=int)
print(df.shape, "->", df_oh.shape)
print(df_oh.columns.tolist())
```

Åpne `df` og `df_oh` i variabelpanelet og sammenlign: `df` har røde `NaN`-celler i `age`, `clan_origin` og `armor`. I `df_oh` er bare `age` rød, fordi `get_dummies` gjør manglende kategorier om til bare nuller.

::: advarsel
Kjører du `get_dummies` hver for seg på train og test, kan du få **forskjellige kolonner**: mangler et skip i testdataene, mangler også kolonnen. Kjør `get_dummies` på hele tabellen før du deler den, eller bruk `OneHotEncoder` fra scikit-learn (leksjon 7.3).
```python
import pandas as pd

train = pd.DataFrame({"ship": ["Bison", "Drakvinge", "Sjøulven"]})
test = pd.DataFrame({"ship": ["Bison", "Bison"]})
print(pd.get_dummies(train).columns.tolist())
print(pd.get_dummies(test).columns.tolist())
```
:::

::: oppgave One-hot på vikingdataene
Lag `df_oh` fra kolonnene `sex`, `role`, `ship` og `survived` med `pd.get_dummies`. Kod `sex`, `role` og `ship` med `drop_first=True` og `dtype=int`. `survived` skal stå urørt.
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

::: quiz
Et datasett har `sex` (2 verdier), `role` (4 verdier), `ship` (5 verdier) og den numeriske featuren `age`. Hvor mange kolonner gir dette?
```kode
X = pd.get_dummies(df[["sex", "role", "ship", "age"]],
                   columns=["sex", "role", "ship"], drop_first=True)
print(X.shape[1])
```
- [ ] 4
- [x] 9
- [ ] 10
- [ ] 12
? Med `drop_first=True` blir hver kategori til (antall verdier − 1) kolonner: 1 + 3 + 4 = 8, pluss `age` som står urørt, gir 9. Uten `drop_first` ville det blitt 2 + 4 + 5 + 1 = 12.
:::

::: quiz
`destination` har verdiene Lindisfarne, Normandy, Iceland, Miklagard og Vinland, uten noen naturlig rekkefølge. Hvilken koding passer best for en logistisk regresjonsmodell?
- [ ] `map` til 0, 1, 2, 3, 4, slik at featuren tar én kolonne
- [x] One-hot-koding med `pd.get_dummies`
- [ ] La teksten stå, scikit-learn koder den automatisk
- [ ] Fjerne kolonnen, fordi tekst ikke kan brukes
? Med tallene 0 til 4 får modellen én vekt og tolker Vinland som «fire ganger» Normandy. One-hot gir hver destinasjon sin egen vekt. scikit-learn gir feil hvis du sender inn tekst direkte.
:::

::: oppsummering
- Binære kategorier: `map` til 0/1. Ordinale kategorier: `map` til tall i **riktig rekkefølge**.
- Nominelle kategorier: one-hot med `pd.get_dummies(df, columns=[...])`.
- `drop_first=True` fjerner én overflødig kolonne per feature. `dtype=int` gir 0/1 i stedet for `True`/`False`.
- Antall kolonner etter one-hot: summen av antall kategorier (minus én per feature med `drop_first`).
- Kjør `get_dummies` før train/test-split, ellers kan kolonnene bli ulike.
:::

--- LEKSJON 3.9 | til-numpy | Fra pandas til X og y, og lagre CSV
Siste steg før trening er å dele tabellen i features `X` og target `y`. Etter trening skal du kanskje lagre prediksjonene til en fil, slik du gjorde for Kaggle i prosjekt 3.

## X og y

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
target = "survived"
X = df.drop(columns=target)
y = df[target]
print(X.shape, y.shape)
```

`X` er alle kolonnene unntatt target (2D), og `y` er target-kolonnen (1D). Åpne `X` og `y` i variabelpanelet: `X` er hele tabellen uten `survived`, og `y` er en Series med bare 0 og 1. Prosjekt 2 starter treningscellen med akkurat `target = 'survived'`.

## Til numpy: .values og .to_numpy()
scikit-learn godtar DataFrames direkte, men din egen kode fra prosjekt 1 og PyTorch i prosjekt 3 vil ha numpy-arrays. `.values` og `.to_numpy()` gjør det samme:

```python
import pandas as pd

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
X_train = train[["x0", "x1"]].values
y_train = train["y"].to_numpy()
print(type(X_train).__name__, X_train.shape, X_train.dtype)
print(type(y_train).__name__, y_train.shape)
print("DataFrame: med kolonnenavn og indeks")
display(train[["x0", "x1"]].head(3))
print("numpy-array: de samme tallene, uten navn")
display(X_train[:3])
```

::: advarsel
Inneholder tabellen tekstkolonner, får arrayet typen `object`, og modellene krasjer. Kod alle tekstkolonner om (leksjon 3.8) og fjern kolonner som `split` og `name` før du lager `X`. Med `to_numpy(dtype=np.float32)` får du flyttall med 32 bit, som PyTorch bruker.
:::

## pd.concat: slå sammen tabeller
`pd.concat` setter tabeller under hverandre (rader) eller ved siden av hverandre (`axis=1`, kolonner):

```python
import pandas as pd

data = pd.read_csv("classification.csv")
train = data[data["split"] == "train"]
test = data[data["split"] == "test"]
alle = pd.concat([train, test])
print(len(train), "+", len(test), "=", len(alle))

df = pd.read_csv("problem2.csv")
dummies = pd.get_dummies(df["ship"], dtype=int)
display(df[["age", "survived"]].head(3))
display(dummies.head(3))
sammen = pd.concat([df[["age", "survived"]], dummies], axis=1)
print("Ved siden av hverandre:")
sammen.head(3)
```

Med `axis=1` blir radene matchet på indeksen, så begge tabellene må ha samme merkelapper.

## Lagre til CSV
`to_csv` skriver en DataFrame til fil. Kaggle-innleveringen i prosjekt 3 skulle ha nøyaktig kolonnene `ID` og `cluster`:

```python
import numpy as np
import pandas as pd

labels = np.array([2, 0, 1, 1, 0])
submission = pd.DataFrame({"ID": np.arange(len(labels)), "cluster": labels})
submission.to_csv("submission.csv", index=False)
with open("submission.csv") as f:   # åpne filen og skriv ut teksten i den
    print(f.read())
```

Glemmer du `index=False`, skrives indeksen som en ekstra kolonne uten navn:

```python
import pandas as pd

submission = pd.DataFrame({"ID": [0, 1, 2], "cluster": [2, 0, 1]})
submission.to_csv("feil.csv")
with open("feil.csv") as f:
    print(f.read())
print(pd.read_csv("feil.csv").columns.tolist())
```

::: prosjekt Prosjekt 3, problem 2
Slutten av løsningen var `submission = pd.DataFrame({"ID": data["ID"], "cluster": final_labels})` og `submission.to_csv("submission.csv", index=False)`.
:::

::: oppgave Fra rådata til X og y
Gjør vikingdataene klare for en modell. Lag numpy-arrays `X` og `y` slik:

1. Fjern `passenger_id` og `name`.
2. Fyll manglende `age` med medianen, manglende `clan_origin` med `"Ukjent"` og manglende `armor` med `"None"`.
3. One-hot-kod `sex`, `role`, `ship`, `destination`, `clan_origin` og `armor` med `drop_first=True` og `dtype=int`.
4. `y` er `survived` som numpy-array. `X` er alle de andre kolonnene som numpy-array.

`X` skal få shape `(500, 22)`.
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

::: quiz
Kaggle krever en fil med nøyaktig kolonnene `ID` og `cluster`. Hva er galt med denne koden?
```kode
submission = pd.DataFrame({"ID": ids, "cluster": labels})
submission.to_csv("submission.csv")
```
- [ ] Ingenting, filen får kolonnene ID og cluster
- [x] Indeksen skrives som en ekstra, navnløs første kolonne
- [ ] Kolonnene skrives i alfabetisk rekkefølge
- [ ] `to_csv` skriver ikke kolonnenavn uten `header=True`
? Som standard skriver `to_csv` med indeksen. Leser du filen inn igjen, får du en kolonne `Unnamed: 0`. Bruk `index=False`.
:::

::: quiz
`data` har 1000 rader og kolonnene `x0`, `x1`, `y` og `split`. Hva er shape til `X`?
```kode
X = data.drop(columns="y").values
```
- [ ] `(1000, 2)`
- [x] `(1000, 3)`
- [ ] `(1000, 4)`
- [ ] `(1000,)`
? Bare `y` fjernes, så `x0`, `x1` og `split` blir igjen. Siden `split` er tekst, får arrayet typen `object`, og en modell vil krasje. Velg heller `data[["x0", "x1"]].values`.
:::

::: oppsummering
- `X = df.drop(columns=target)` og `y = df[target]`.
- `.values` og `.to_numpy()` gjør en DataFrame om til et numpy-array. Sjekk at det ikke er tekst igjen (`dtype object`).
- `pd.concat([a, b])` setter under hverandre, `pd.concat([a, b], axis=1)` ved siden av hverandre.
- `df.to_csv("submission.csv", index=False)` lagrer uten den ekstra indekskolonnen.
:::
