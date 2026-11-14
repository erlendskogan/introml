=== DEL 2 | numpy | NumPy: regning med tabeller
Her lærer du å lage, indeksere og regne med NumPy-arrays, og å holde styr på `shape` og `axis`.

--- LEKSJON 2.1 | arrays | Arrays, shape og dtype
`import numpy as np` og `np.array([1, 2, 3])` lager et array, og regning som `x * 2` skjer element for element (mens listen `[1, 2, 3] * 2` bare gjentas).

::: oppgave Fra liste til array
Gjør `liste` om til arrayet `x`, og lag `dobbel` (hvert tall i `x` ganget med 2) og `gjentatt` (`liste * 2`).
```start
import numpy as np

liste = [1, 2, 3]

x = ...
dobbel = ...
gjentatt = ...
```
```losning
import numpy as np

liste = [1, 2, 3]

x = np.array(liste)
dobbel = x * 2
gjentatt = liste * 2
```
```test
import numpy as np
assert isinstance(x, np.ndarray), "x skal være et numpy-array: np.array(liste)."
assert np.array_equal(x, [1, 2, 3]), "x skal inneholde 1, 2 og 3."
assert isinstance(dobbel, np.ndarray) and np.array_equal(dobbel, [2, 4, 6]), "dobbel skal være x * 2, altså array([2, 4, 6])."
assert gjentatt == [1, 2, 3, 1, 2, 3], "gjentatt skal være liste * 2, som gir listen gjentatt to ganger."
```
:::

## Shape, ndim og size
`X.shape` gir formen som tuppel `(rader, kolonner)` (for 1D `(3,)`), `X.ndim` antall akser, `X.size` antall elementer og `len(X)` antall rader. I ML har `X` shape `(n_samples, n_features)` (rader er datapunkter) og `y` shape `(n_samples,)`.

::: oppgave Ditt første datasett
Lag et 2D-array `X` med 3 rader og 2 kolonner med tallene 1 til 6 (radvis: første rad `1, 2`). Pakk så ut shape i `n_samples` og `n_features`.
```start
import numpy as np

X = ...
n_samples, n_features = 0, 0
```
```losning
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6]])
n_samples, n_features = X.shape
```
```test
import numpy as np
assert isinstance(X, np.ndarray), "X skal være et numpy-array. Bruk np.array([[...], [...], [...]])."
assert X.shape == (3, 2), f"X skal ha shape (3, 2), men har {X.shape}."
assert np.array_equal(X, [[1, 2], [3, 4], [5, 6]]), "Radene skal være [1, 2], [3, 4] og [5, 6]."
assert (n_samples, n_features) == (3, 2), "Pakk ut med n_samples, n_features = X.shape."
```
:::

::: oppgave Les av formen
Bruk attributtene til `X` til å lage `form` (shape), `antall_akser`, `antall_tall` (totalt) og `antall_rader`.
```start
import numpy as np

X = np.array([[0.5, 1.2, 3.0],
              [1.0, 0.3, 2.2],
              [2.2, 2.0, 0.1],
              [0.1, 0.9, 1.5],
              [1.7, 0.4, 0.8]])

form = ...
antall_akser = ...
antall_tall = ...
antall_rader = ...
```
```losning
import numpy as np

X = np.array([[0.5, 1.2, 3.0],
              [1.0, 0.3, 2.2],
              [2.2, 2.0, 0.1],
              [0.1, 0.9, 1.5],
              [1.7, 0.4, 0.8]])

form = X.shape
antall_akser = X.ndim
antall_tall = X.size
antall_rader = len(X)
```
```test
assert form == (5, 3), f"form skal være X.shape, altså (5, 3), men er {form}."
assert antall_akser == 2, "antall_akser skal være X.ndim (2 for en tabell)."
assert antall_tall == 15, "antall_tall skal være X.size (5 * 3 = 15)."
assert antall_rader == 5, "antall_rader skal være len(X), antall rader."
```
:::

## dtype
Alle elementene har samme `dtype`, så ett desimaltall gjør hele arrayet til `float64`. `b.astype(int)` lager en kopi med ny type, og desimalene kuttes bort.

::: oppgave Bytt type
Lag `f` med tallene 1, 2 og 3 som float-array, `heltall` som `b` konvertert til heltall, og `bools` som `c` konvertert til `bool`.
```start
import numpy as np

b = np.array([1.0, 2.5, 3.9])
c = np.array([0.0, 1.0, 2.0])

f = ...
heltall = ...
bools = ...
```
```losning
import numpy as np

b = np.array([1.0, 2.5, 3.9])
c = np.array([0.0, 1.0, 2.0])

f = np.array([1.0, 2, 3])
heltall = b.astype(int)
bools = c.astype(bool)
```
```test
import numpy as np
assert isinstance(f, np.ndarray) and f.dtype.kind == "f", "f skal være et float-array. Skriv minst ett av tallene med desimal, f.eks. 1.0."
assert np.allclose(f, [1, 2, 3]), "f skal inneholde 1, 2 og 3."
assert isinstance(heltall, np.ndarray) and np.issubdtype(heltall.dtype, np.integer), "heltall skal ha en heltallstype. Bruk b.astype(int)."
assert np.array_equal(heltall, [1, 2, 3]), "astype(int) kutter bort desimalene: [1, 2, 3]."
assert isinstance(bools, np.ndarray) and bools.dtype == bool, "bools skal ha dtype bool. Bruk c.astype(bool)."
assert np.array_equal(bools, [False, True, True]), "0.0 blir False, alt annet blir True."
```
:::

--- LEKSJON 2.2 | lage-arrays | Lage arrays
NumPy har egne funksjoner som lager arrays uten at du skriver tallene selv.

## zeros, ones og full
`np.zeros(4)`, `np.ones((2, 3))` og `np.full((2, 3), 0.5)` lager fylte arrays, og shape for 2D er én tuppel med doble parenteser. `np.zeros_like(y)` og `np.full_like(y, 7.0)` lager et array med samme shape som `y`.

::: oppgave Q-tabell
Lag `q_table`: en Q-tabell fylt med nuller for 25 tilstander (rader) og 4 handlinger (kolonner).
```start
import numpy as np

q_table = ...
```
```losning
import numpy as np

q_table = np.zeros((25, 4))
```
```test
import numpy as np
assert isinstance(q_table, np.ndarray), "q_table skal være et numpy-array, f.eks. fra np.zeros."
assert q_table.shape == (25, 4), f"q_table skal ha shape (25, 4) (tilstander, handlinger), men har {q_table.shape}."
assert np.all(q_table == 0), "Alle verdiene i q_table skal være 0."
```
:::

::: oppgave Optimistisk start
Lag `q_opt` med 16 tilstander og 4 handlinger der alle verdiene er `Q_INIT`, og `vekter` med tre enere.
```start
import numpy as np

Q_INIT = 0.5
q_opt = ...
vekter = ...
```
```losning
import numpy as np

Q_INIT = 0.5
q_opt = np.full((16, 4), Q_INIT)
vekter = np.ones(3)
```
```test
import numpy as np
assert isinstance(q_opt, np.ndarray) and q_opt.shape == (16, 4), "q_opt skal ha shape (16, 4). Bruk np.full((16, 4), Q_INIT)."
assert np.all(q_opt == 0.5), "Alle verdiene i q_opt skal være Q_INIT (0.5)."
assert isinstance(vekter, np.ndarray) and vekter.shape == (3,) and np.all(vekter == 1), "vekter skal være np.ones(3)."
```
:::

::: oppgave Samme shape som dataene
Lag `gradient` med nuller og `sjuere` med verdien 7.0, begge med samme shape som `w`.
```start
import numpy as np

w = np.array([[0.3, -1.2, 0.8],
              [1.1, 0.4, -0.5]])

gradient = ...
sjuere = ...
```
```losning
import numpy as np

w = np.array([[0.3, -1.2, 0.8],
              [1.1, 0.4, -0.5]])

gradient = np.zeros_like(w)
sjuere = np.full_like(w, 7.0)
```
```test
import numpy as np
assert isinstance(gradient, np.ndarray) and gradient.shape == w.shape, "gradient skal ha samme shape som w. Bruk np.zeros_like(w)."
assert np.all(gradient == 0), "gradient skal bare inneholde nuller."
assert isinstance(sjuere, np.ndarray) and sjuere.shape == w.shape and np.allclose(sjuere, 7.0), "sjuere skal være np.full_like(w, 7.0)."
```
:::

## arange og linspace
`np.arange(start, stopp, steg)` virker som `range` (stopp er ikke med, men desimalsteg går fint). `np.linspace(start, stopp, antall)` gir et bestemt antall jevnt fordelte punkter, og her er stopp med.

::: oppgave Epoker og steg
Lag `epoker` med tallene 1, 2, …, 100 (begge med), `hvert_tredje` med 2, 5, 8 og `kvarter` med 0, 0.25, 0.5, 0.75.
```start
import numpy as np

epoker = ...
hvert_tredje = ...
kvarter = ...
```
```losning
import numpy as np

epoker = np.arange(1, 101)
hvert_tredje = np.arange(2, 10, 3)
kvarter = np.arange(0, 1, 0.25)
```
```test
import numpy as np
assert isinstance(epoker, np.ndarray), "epoker skal være et numpy-array, f.eks. fra np.arange."
assert len(epoker) == 100, f"epoker skal ha 100 elementer, men har {len(epoker)}. Husk at stopp ikke er med i arange."
assert epoker[0] == 1 and epoker[-1] == 100, "epoker skal starte på 1 og slutte på 100."
assert isinstance(hvert_tredje, np.ndarray) and np.array_equal(hvert_tredje, [2, 5, 8]), "hvert_tredje skal være [2, 5, 8]: start 2, steg 3, og stopp under 11."
assert isinstance(kvarter, np.ndarray) and np.allclose(kvarter, [0, 0.25, 0.5, 0.75]), "kvarter skal være [0, 0.25, 0.5, 0.75] (1 er ikke med)."
```
:::

::: oppgave Punkter til en kurve
Lag `xs` med 50 jevnt fordelte punkter fra -3 til 3 (begge med), og `rutenett` med 0.0, 0.1, …, 1.0 (11 tall).
```start
import numpy as np

xs = ...
rutenett = ...
```
```losning
import numpy as np

xs = np.linspace(-3, 3, 50)
rutenett = np.linspace(0, 1, 11)
```
```test
import numpy as np
assert isinstance(xs, np.ndarray) and len(xs) == 50, "xs skal ha 50 punkter. Bruk np.linspace(-3, 3, 50)."
assert np.isclose(xs[0], -3) and np.isclose(xs[-1], 3), "xs skal starte på -3 og slutte på 3."
assert isinstance(rutenett, np.ndarray) and len(rutenett) == 11, f"rutenett skal ha 11 tall, men har {len(rutenett)}. Med linspace er stopp med."
assert np.allclose(rutenett, np.arange(11) / 10), "rutenett skal være 0.0, 0.1, …, 1.0."
```
:::

--- LEKSJON 2.3 | indeksering | Indeksering og slicing
Indekser starter på 0, `x[-1]` er siste element og `x[1:4]` gir indeks 1, 2 og 3, akkurat som for lister.

## 1D-arrays
`x[:3]` gir de tre første, `x[::2]` annethvert element og `x[::-1]` arrayet baklengs.

::: oppgave Plukk fra 1D
Lag `forste`, `siste`, `midten` (indeks 1 til og med 3), `annethvert` (indeks 0, 2, 4) og `baklengs`.
```start
import numpy as np

x = np.array([10, 20, 30, 40, 50, 60])

forste = ...
siste = ...
midten = ...
annethvert = ...
baklengs = ...
```
```losning
import numpy as np

x = np.array([10, 20, 30, 40, 50, 60])

forste = x[0]
siste = x[-1]
midten = x[1:4]
annethvert = x[::2]
baklengs = x[::-1]
```
```test
import numpy as np
assert forste is not ... and forste == 10, "forste skal være x[0]."
assert siste is not ... and siste == 60, "siste skal være x[-1]."
assert isinstance(midten, np.ndarray) and np.array_equal(midten, [20, 30, 40]), "midten skal være x[1:4]: indeks 1, 2 og 3 (stopp er ikke med)."
assert isinstance(annethvert, np.ndarray) and np.array_equal(annethvert, [10, 30, 50]), "annethvert skal være x[::2]."
assert isinstance(baklengs, np.ndarray) and np.array_equal(baklengs, [60, 50, 40, 30, 20, 10]), "baklengs skal være x[::-1]."
```
:::

## 2D-arrays: rader og kolonner
`X[rad, kolonne]` gir ett element, `X[i]` rad `i` (et datapunkt) og `X[:, j]` kolonne `j` (en feature) som 1D-array. `X[:, [j]]` beholder kolonnen som 2D med shape `(n, 1)`, og `X[:, [0, 2]]` velger flere kolonner.

::: oppgave Plukk ut data
`X` har 6 datapunkter og 3 features. Lag `alder` (kolonne 0, 1D), `punkt3` (rad 3), `forste_fire` (de fire første radene) og `verdi` (rad 2, kolonne 1).
```start
import numpy as np

X = np.array([[23, 1.80, 0],
              [35, 1.65, 1],
              [41, 1.72, 1],
              [19, 1.90, 0],
              [52, 1.60, 1],
              [30, 1.75, 0]])

alder = ...
punkt3 = ...
forste_fire = ...
verdi = ...
```
```losning
import numpy as np

X = np.array([[23, 1.80, 0],
              [35, 1.65, 1],
              [41, 1.72, 1],
              [19, 1.90, 0],
              [52, 1.60, 1],
              [30, 1.75, 0]])

alder = X[:, 0]
punkt3 = X[3]
forste_fire = X[:4]
verdi = X[2, 1]
```
```test
import numpy as np
assert isinstance(alder, np.ndarray) and alder.shape == (6,), "alder skal være kolonne 0 som 1D-array med shape (6,). Bruk X[:, 0]."
assert np.allclose(alder, [23, 35, 41, 19, 52, 30]), "alder skal inneholde første kolonne."
assert isinstance(punkt3, np.ndarray) and np.allclose(punkt3, [19, 1.90, 0]), "punkt3 skal være rad 3 (husk at indeks starter på 0)."
assert isinstance(forste_fire, np.ndarray) and forste_fire.shape == (4, 3), "forste_fire skal ha shape (4, 3). Bruk X[:4]."
assert verdi is not ... and np.isclose(verdi, 1.72), "verdi skal være X[2, 1], altså 1.72."
```
:::

::: oppgave Riktig shape
`X` har shape `(100, 3)`. Lag `kol_1d` (kolonne 1, shape `(100,)`), `kol_2d` (kolonne 1, shape `(100, 1)`), `to_kol` (kolonne 0 og 2) og `siste_fem` (de fem siste radene).
```start
import numpy as np

X = np.arange(300).reshape(100, 3)   # 100 rader, 3 kolonner

kol_1d = ...
kol_2d = ...
to_kol = ...
siste_fem = ...
```
```losning
import numpy as np

X = np.arange(300).reshape(100, 3)

kol_1d = X[:, 1]
kol_2d = X[:, [1]]
to_kol = X[:, [0, 2]]
siste_fem = X[-5:]
```
```test
import numpy as np
assert isinstance(kol_1d, np.ndarray) and kol_1d.shape == (100,), "kol_1d skal ha shape (100,). Bruk X[:, 1]."
assert np.array_equal(kol_1d, X[:, 1]), "kol_1d skal være kolonne 1."
assert isinstance(kol_2d, np.ndarray) and kol_2d.shape == (100, 1), "kol_2d skal ha shape (100, 1). Bruk en liste med én indeks: X[:, [1]] (eller X[:, 1:2])."
assert np.array_equal(kol_2d[:, 0], X[:, 1]), "kol_2d skal inneholde kolonne 1."
assert isinstance(to_kol, np.ndarray) and to_kol.shape == (100, 2) and np.array_equal(to_kol, X[:, [0, 2]]), "to_kol skal være kolonne 0 og 2: X[:, [0, 2]]."
assert isinstance(siste_fem, np.ndarray) and siste_fem.shape == (5, 3) and siste_fem[-1, 0] == 297, "siste_fem skal være de fem siste radene: X[-5:]."
```
:::

::: oppgave Oppdater Q-tabellen
Øk `q_table[state, action]` med 0.5, sett hele rad 4 til `[1, 2, 3, 4]`, og lag `rad` med Q-verdiene i tilstand `state`.
```start
import numpy as np

q_table = np.zeros((5, 4))
state, action = 2, 1

# Skriv de to endringene her

rad = ...
```
```losning
import numpy as np

q_table = np.zeros((5, 4))
state, action = 2, 1

q_table[state, action] += 0.5
q_table[4] = [1, 2, 3, 4]

rad = q_table[state]
```
```test
import numpy as np
assert q_table.shape == (5, 4), "q_table skal fortsatt ha shape (5, 4)."
assert q_table[2, 1] == 0.5, "q_table[2, 1] skal være 0.5. Bruk q_table[state, action] += 0.5."
assert np.array_equal(q_table[4], [1, 2, 3, 4]), "Rad 4 skal være [1, 2, 3, 4]. Bruk q_table[4] = [1, 2, 3, 4]."
assert np.isclose(q_table.sum(), 10.5), "Bare elementet [2, 1] og rad 4 skal endres."
assert isinstance(rad, np.ndarray) and np.array_equal(rad, [0, 0.5, 0, 0]), "rad skal være q_table[state]."
```
:::

## Slicing gir en view, ikke en kopi
`b = a[1:3]` deler minne med `a`, så `b[0] = 0` endrer også `a`. `a[1:3].copy()` gir en uavhengig kopi.

::: oppgave Ikke ødelegg originalen
Koden endrer `x` ved et uhell. Endre linjen som lager `forste_tre`, så `x` forblir `[1, 2, 3, 4, 5]` mens `forste_tre` blir `[99, 2, 3]`.
```start
import numpy as np

x = np.array([1, 2, 3, 4, 5])
forste_tre = x[:3]
forste_tre[0] = 99
```
```losning
import numpy as np

x = np.array([1, 2, 3, 4, 5])
forste_tre = x[:3].copy()
forste_tre[0] = 99
```
```test
import numpy as np
assert np.array_equal(x, [1, 2, 3, 4, 5]), f"x er endret til {x}. Slicen er en view; bruk x[:3].copy()."
assert isinstance(forste_tre, np.ndarray) and np.array_equal(forste_tre, [99, 2, 3]), "forste_tre skal bli [99, 2, 3]."
```
:::

--- LEKSJON 2.4 | vektorisering | Vektoriserte operasjoner og matematiske funksjoner
`+ - * / **` virker element for element på arrays, også mellom et tall og et array (`2 * x + 1`), så du trenger ingen løkke. `x * y` er altså ikke matriseprodukt.

::: oppgave Regn uten løkke
Lag `summen` (`x` pluss `y`), `forhold` (`y` delt på `x`), `kvadrat` (`x` i andre) og `lin` ($2x + 1$).
```start
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([10.0, 20.0, 30.0, 40.0])

summen = ...
forhold = ...
kvadrat = ...
lin = ...
```
```losning
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([10.0, 20.0, 30.0, 40.0])

summen = x + y
forhold = y / x
kvadrat = x ** 2
lin = 2 * x + 1
```
```test
import numpy as np
assert isinstance(summen, np.ndarray) and np.allclose(summen, [11, 22, 33, 44]), "summen skal være x + y."
assert isinstance(forhold, np.ndarray) and np.allclose(forhold, [10, 10, 10, 10]), "forhold skal være y / x."
assert isinstance(kvadrat, np.ndarray) and np.allclose(kvadrat, [1, 4, 9, 16]), "kvadrat skal være x ** 2."
assert isinstance(lin, np.ndarray) and np.allclose(lin, [3, 5, 7, 9]), "lin skal være 2 * x + 1."
```
:::

::: oppgave Bytt ut løkka
Skriv om koden så `resultat` blir et array regnet ut på én linje, uten løkke.
```start
import numpy as np

x = np.arange(10_000, dtype=float)

resultat = []
for verdi in x:
    resultat.append(verdi ** 2 + 1)
```
```losning
import numpy as np

x = np.arange(10_000, dtype=float)

resultat = x ** 2 + 1
```
```test
import numpy as np
assert isinstance(resultat, np.ndarray), "resultat skal være et numpy-array, ikke en liste. Skriv x ** 2 + 1 direkte."
assert resultat.shape == x.shape and np.allclose(resultat, x ** 2 + 1), "resultat skal være x ** 2 + 1 for hvert element."
```
:::

## Matematiske funksjoner
`np.exp`, `np.log` (naturlig logaritme), `np.sqrt`, `np.abs` og `np.mean` virker på hele arrays. `math.exp` og `math.log` tar bare ett tall og krasjer på arrays.

::: oppgave Rett feilen
Koden krasjer fordi `math` ikke virker på arrays. Bytt til NumPy-funksjoner, og lag i tillegg `absz` med absoluttverdiene og `rot` med kvadratroten av `z ** 2`.
```start
import math
import numpy as np

z = np.array([-2.0, 0.0, 1.0, 2.0])

ez = math.exp(z)
ln = math.log(np.array([1.0, np.e, 10.0]))
absz = ...
rot = ...
```
```losning
import math
import numpy as np

z = np.array([-2.0, 0.0, 1.0, 2.0])

ez = np.exp(z)
ln = np.log(np.array([1.0, np.e, 10.0]))
absz = np.abs(z)
rot = np.sqrt(z ** 2)
```
```test
import numpy as np
assert isinstance(ez, np.ndarray) and np.allclose(ez, [0.13533528, 1.0, 2.71828183, 7.3890561]), "ez skal være np.exp(z)."
assert isinstance(ln, np.ndarray) and np.allclose(ln, [0.0, 1.0, 2.30258509]), "ln skal være np.log av arrayet."
assert isinstance(absz, np.ndarray) and np.allclose(absz, [2, 0, 1, 2]), "absz skal være np.abs(z)."
assert isinstance(rot, np.ndarray) and np.allclose(rot, [2, 0, 1, 2]), "rot skal være np.sqrt(z ** 2)."
```
:::

::: oppgave Sigmoid
Skriv `sigmoid(z)` som returnerer $1/(1+e^{-z})$ og virker på hele arrays.
```start
import numpy as np

def sigmoid(z):
    return ...
```
```losning
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```
```test
import numpy as np
s = sigmoid(np.array([-1.0, 0.0, 2.0]))
assert isinstance(s, np.ndarray) and s.shape == (3,), "sigmoid skal returnere et array med samme shape som inndataene. Bruk np.exp, ikke math.exp."
assert np.allclose(s, [0.26894142, 0.5, 0.88079708]), "sigmoid gir feil verdier. Formelen er 1 / (1 + np.exp(-z))."
```
:::

::: oppgave Feilmål
Regn ut `mse` (snittet av kvadrerte feil), `rmse` (roten av `mse`) og `mae` (snittet av absolutte feil) mellom `y` og `y_pred`, uten løkke.
```start
import numpy as np

y = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = np.array([1.5, 2.0, 2.0, 5.0])

mse = ...
rmse = ...
mae = ...
```
```losning
import numpy as np

y = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = np.array([1.5, 2.0, 2.0, 5.0])

mse = np.mean((y - y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y - y_pred))
```
```test
import numpy as np
assert mse is not ... and np.isclose(mse, 0.5625), f"mse skal være 0.5625, men er {mse}. Ta np.mean av (y - y_pred) ** 2."
assert rmse is not ... and np.isclose(rmse, 0.75), "rmse skal være np.sqrt(mse)."
assert mae is not ... and np.isclose(mae, 0.625), "mae skal være np.mean(np.abs(y - y_pred))."
```
:::

--- LEKSJON 2.5 | broadcasting | Broadcasting
Arrays med ulik shape kan regnes sammen når formene passer: har `X` shape `(n, d)` og `v` shape `(d,)`, brukes `v` på hver rad i `X - v`.

::: oppgave Én verdi per kolonne
Lag `diff` (trekk `v` fra hver rad i `X`) og `X_skalert` (del hver kolonne i `X` på tallet i `skala`).
```start
import numpy as np

X = np.array([[1, 10, 100],
              [2, 20, 200],
              [3, 30, 300]])
v = np.array([1, 2, 3])
skala = np.array([1, 10, 100])

diff = ...
X_skalert = ...
```
```losning
import numpy as np

X = np.array([[1, 10, 100],
              [2, 20, 200],
              [3, 30, 300]])
v = np.array([1, 2, 3])
skala = np.array([1, 10, 100])

diff = X - v
X_skalert = X / skala
```
```test
import numpy as np
assert isinstance(diff, np.ndarray) and diff.shape == (3, 3), "diff skal ha shape (3, 3). Skriv X - v."
assert np.array_equal(diff, [[0, 8, 97], [1, 18, 197], [2, 28, 297]]), "diff skal være X - v (v trekkes fra hver rad)."
assert isinstance(X_skalert, np.ndarray) and np.allclose(X_skalert, [[1, 1, 1], [2, 2, 2], [3, 3, 3]]), "X_skalert skal være X / skala."
```
:::

## Sentrering og standardisering
`X.mean(axis=0)` og `X.std(axis=0)` gir snitt og standardavvik per kolonne med shape `(d,)`, så `(X - mu) / sigma` standardiserer hver feature. Bruk alltid `mu` og `sigma` fra treningsdataene, også på testdataene.

::: oppgave Standardiser features
Standardiser `X` så hver kolonne får snitt 0 og standardavvik 1. Lagre snittene i `mu`, standardavvikene i `sigma` og resultatet i `X_std`.
```start
import numpy as np

X = np.array([[1.0, 200.0],
              [2.0, 300.0],
              [3.0, 100.0],
              [6.0, 400.0]])

mu = ...
sigma = ...
X_std = ...
```
```losning
import numpy as np

X = np.array([[1.0, 200.0],
              [2.0, 300.0],
              [3.0, 100.0],
              [6.0, 400.0]])

mu = X.mean(axis=0)
sigma = X.std(axis=0)
X_std = (X - mu) / sigma
```
```test
import numpy as np
assert isinstance(mu, np.ndarray) and mu.shape == (2,), "mu skal ha shape (2,): ett snitt per kolonne. Bruk X.mean(axis=0)."
assert np.allclose(mu, [3.0, 250.0]), "mu har feil verdier. Snittet skal tas nedover hver kolonne (axis=0)."
assert isinstance(sigma, np.ndarray) and np.allclose(sigma, X.std(axis=0)), "sigma skal være X.std(axis=0)."
assert isinstance(X_std, np.ndarray) and X_std.shape == (4, 2), "X_std skal ha samme shape som X."
assert np.allclose(X_std.mean(axis=0), 0) and np.allclose(X_std.std(axis=0), 1), "Hver kolonne i X_std skal ha snitt 0 og standardavvik 1."
```
:::

::: oppgave Samme skalering på testdata
Regn ut `mu` og `sigma` fra `X_train`, og bruk dem til å lage både `X_train_std` og `X_test_std`.
```start
import numpy as np

X_train = np.array([[1.0, 100.0],
                    [3.0, 300.0],
                    [5.0, 200.0],
                    [7.0, 400.0]])
X_test = np.array([[2.0, 250.0],
                   [6.0, 150.0]])

mu = ...
sigma = ...
X_train_std = ...
X_test_std = ...
```
```losning
import numpy as np

X_train = np.array([[1.0, 100.0],
                    [3.0, 300.0],
                    [5.0, 200.0],
                    [7.0, 400.0]])
X_test = np.array([[2.0, 250.0],
                   [6.0, 150.0]])

mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
X_train_std = (X_train - mu) / sigma
X_test_std = (X_test - mu) / sigma
```
```test
import numpy as np
mu_r, sigma_r = X_train.mean(axis=0), X_train.std(axis=0)
assert isinstance(mu, np.ndarray) and np.allclose(mu, mu_r), "mu skal være X_train.mean(axis=0), regnet ut fra treningsdataene."
assert isinstance(sigma, np.ndarray) and np.allclose(sigma, sigma_r), "sigma skal være X_train.std(axis=0)."
assert isinstance(X_train_std, np.ndarray) and np.allclose(X_train_std, (X_train - mu_r) / sigma_r), "X_train_std skal være (X_train - mu) / sigma."
assert isinstance(X_test_std, np.ndarray) and X_test_std.shape == (2, 2), "X_test_std skal ha samme shape som X_test."
assert np.allclose(X_test_std, (X_test - mu_r) / sigma_r), "X_test_std skal bruke mu og sigma fra X_train, ikke fra X_test."
```
:::

## Regelen for broadcasting
Formene sammenlignes bakfra, og to akser passer hvis de er like eller en av dem er 1. Derfor krasjer `(4, 3)` mot `(4,)`, mens `(4, 3)` mot `(4, 1)` gir ett tall per rad og `(3, 1)` mot `(1, 4)` gir `(3, 4)`.

::: oppgave Trekk fra per rad
Koden krasjer fordi `(4, 3)` og `(4,)` ikke passer. Lag `rad_kol` med de samme tallene som `rad_snitt`, men som kolonnevektor med shape `(4, 1)` (en liste av lister), og bruk den i `resultat`.
```start
import numpy as np

X = np.ones((4, 3))
rad_snitt = np.array([1.0, 2.0, 3.0, 4.0])   # shape (4,)

rad_kol = ...
resultat = X - rad_snitt
```
```losning
import numpy as np

X = np.ones((4, 3))
rad_snitt = np.array([1.0, 2.0, 3.0, 4.0])

rad_kol = np.array([[1.0], [2.0], [3.0], [4.0]])
resultat = X - rad_kol
```
```test
import numpy as np
assert isinstance(rad_kol, np.ndarray) and rad_kol.shape == (4, 1), "rad_kol skal ha shape (4, 1): np.array([[1.0], [2.0], [3.0], [4.0]])."
assert np.allclose(rad_kol[:, 0], rad_snitt), "rad_kol skal inneholde 1, 2, 3 og 4."
assert isinstance(resultat, np.ndarray) and resultat.shape == (4, 3), "resultat skal ha shape (4, 3). Skriv X - rad_kol."
assert np.allclose(resultat, [[0, 0, 0], [-1, -1, -1], [-2, -2, -2], [-3, -3, -3]]), "Rad i skal få trukket fra tall nummer i."
```
:::

::: oppgave Gangetabell med broadcasting
Lag `a` med tallene 1, 2, 3 og shape `(3, 1)`, `b` med tallene 1, 2, 3, 4 og shape `(1, 4)`, og `gangetabell = a * b`.
```start
import numpy as np

a = ...
b = ...
gangetabell = ...
```
```losning
import numpy as np

a = np.array([[1], [2], [3]])
b = np.array([[1, 2, 3, 4]])
gangetabell = a * b
```
```test
import numpy as np
assert isinstance(a, np.ndarray) and a.shape == (3, 1), "a skal ha shape (3, 1): np.array([[1], [2], [3]])."
assert isinstance(b, np.ndarray) and b.shape == (1, 4), "b skal ha shape (1, 4): np.array([[1, 2, 3, 4]])."
assert isinstance(gangetabell, np.ndarray) and gangetabell.shape == (3, 4), "gangetabell skal ha shape (3, 4)."
assert np.array_equal(gangetabell, [[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12]]), "gangetabell[i, j] skal være a[i] * b[j]."
```
:::

--- LEKSJON 2.6 | aggregering | Aggregering og axis
`x.sum()`, `x.mean()`, `x.std()`, `x.min()` og `x.max()` gir verdier, mens `x.argmin()` og `x.argmax()` gir indeksen (den første ved likhet). `np.sum(x)`, `np.max(x)` og `np.argmax(x)` er det samme.

::: oppgave Oppsummer en vektor
Lag `total`, `snitt`, `storst`, `pos_storst` (indeksen til største) og `pos_minst` (indeksen til minste).
```start
import numpy as np

x = np.array([4.0, 9.0, 1.0, 7.0, 9.0])

total = ...
snitt = ...
storst = ...
pos_storst = ...
pos_minst = ...
```
```losning
import numpy as np

x = np.array([4.0, 9.0, 1.0, 7.0, 9.0])

total = x.sum()
snitt = x.mean()
storst = x.max()
pos_storst = x.argmax()
pos_minst = x.argmin()
```
```test
import numpy as np
assert total is not ... and np.isclose(total, 30), "total skal være x.sum()."
assert snitt is not ... and np.isclose(snitt, 6), "snitt skal være x.mean()."
assert storst is not ... and np.isclose(storst, 9), "storst skal være x.max(), verdien."
assert pos_storst is not ... and pos_storst == 1, "pos_storst skal være x.argmax(): indeksen, og ved likhet den første (1)."
assert pos_minst is not ... and pos_minst == 2, "pos_minst skal være x.argmin()."
```
:::

## axis på 2D-arrays
`axis=k` betyr at akse `k` forsvinner: for `X` med shape `(n, d)` gir `X.sum(axis=0)` ett tall per kolonne (shape `(d,)`), og `X.sum(axis=1)` ett tall per rad (shape `(n,)`). Uten `axis` slås alt sammen til ett tall.

::: oppgave Summer i to retninger
Lag `per_kolonne`, `per_rad` og `alt` (summen av alle tallene).
```start
import numpy as np

X = np.array([[1, 2, 3],
              [4, 5, 6]])

per_kolonne = ...
per_rad = ...
alt = ...
```
```losning
import numpy as np

X = np.array([[1, 2, 3],
              [4, 5, 6]])

per_kolonne = X.sum(axis=0)
per_rad = X.sum(axis=1)
alt = X.sum()
```
```test
import numpy as np
assert isinstance(per_kolonne, np.ndarray) and per_kolonne.shape == (3,), "per_kolonne skal ha ett tall per kolonne, shape (3,). Akse 0 (radene) skal forsvinne."
assert np.array_equal(per_kolonne, [5, 7, 9]), "per_kolonne skal være X.sum(axis=0)."
assert isinstance(per_rad, np.ndarray) and np.array_equal(per_rad, [6, 15]), "per_rad skal være X.sum(axis=1), ett tall per rad."
assert alt is not ... and alt == 21, "alt skal være X.sum() uten axis."
```
:::

::: oppgave Snitt per feature og per datapunkt
`X` har 100 datapunkter og 4 features. Lag `snitt_feature` (snittet av hver feature) og `snitt_punkt` (snittet av hvert datapunkt).
```start
import numpy as np

X = np.arange(400).reshape(100, 4) / 10

snitt_feature = ...
snitt_punkt = ...
```
```losning
import numpy as np

X = np.arange(400).reshape(100, 4) / 10

snitt_feature = X.mean(axis=0)
snitt_punkt = X.mean(axis=1)
```
```test
import numpy as np
assert isinstance(snitt_feature, np.ndarray) and snitt_feature.shape == (4,), "snitt_feature skal ha shape (4,), ett tall per feature (kolonne). Bruk axis=0."
assert np.allclose(snitt_feature, X.mean(axis=0)), "snitt_feature har feil verdier."
assert isinstance(snitt_punkt, np.ndarray) and snitt_punkt.shape == (100,), "snitt_punkt skal ha shape (100,), ett tall per datapunkt (rad). Bruk axis=1."
assert np.allclose(snitt_punkt, X.mean(axis=1)), "snitt_punkt har feil verdier."
```
:::

## Q-tabellen: argmax og max per tilstand
`np.argmax(q_table[state])` velger beste handling i én tilstand, `q_table.max(axis=1)` gir beste verdi per tilstand og `q_table.argmax(axis=1)` er den grådige policyen.

::: oppgave Velg beste handling
Skriv `velg_beste(q_table, state)` som returnerer indeksen til den beste handlingen i tilstand `state` som `int`.
```start
import numpy as np

def velg_beste(q_table, state):
    return ...
```
```losning
import numpy as np

def velg_beste(q_table, state):
    return int(np.argmax(q_table[state]))
```
```test
import numpy as np
q = np.array([[0.1, 0.5, 0.2, 0.0],
              [0.9, 0.3, 0.4, 0.8],
              [0.0, 0.0, 0.7, 0.7]])
r = velg_beste(q, 1)
assert r is not ... and r is not None, "velg_beste skal returnere en handling. Bruk np.argmax(q_table[state])."
assert isinstance(r, int), "Gjør svaret om til int med int(...)."
assert r == 0, f"I tilstand 1 er handling 0 best (0.9), men du fikk {r}."
assert velg_beste(q, 0) == 1, "I tilstand 0 er handling 1 best."
assert velg_beste(q, 2) == 2, "Ved likhet skal du få den første (handling 2), som argmax gir."
assert velg_beste(np.zeros((3, 4)), 0) == 0, "En tabell med bare nuller skal gi handling 0."
```
:::

::: oppgave Les av Q-tabellen
Lag `policy` (beste handling per tilstand), `V` (høyeste Q-verdi per tilstand) og `snitt_per_handling` (gjennomsnittlig Q-verdi per handling).
```start
import numpy as np

q_table = np.array([[1.0, 5.0, 2.0],
                    [3.0, 0.0, 4.0],
                    [9.0, 1.0, 1.0],
                    [2.0, 6.0, 7.0]])

policy = ...
V = ...
snitt_per_handling = ...
```
```losning
import numpy as np

q_table = np.array([[1.0, 5.0, 2.0],
                    [3.0, 0.0, 4.0],
                    [9.0, 1.0, 1.0],
                    [2.0, 6.0, 7.0]])

policy = q_table.argmax(axis=1)
V = q_table.max(axis=1)
snitt_per_handling = q_table.mean(axis=0)
```
```test
import numpy as np
assert isinstance(policy, np.ndarray) and policy.shape == (4,), "policy skal ha én handling per tilstand (shape (4,)). Hvilken akse skal forsvinne?"
assert np.array_equal(policy, [1, 2, 0, 2]), "policy skal være indeksen til største verdi i hver rad: argmax med axis=1."
assert isinstance(V, np.ndarray) and np.allclose(V, [5, 4, 9, 7]), "V skal være største verdi i hver rad: max med axis=1."
assert isinstance(snitt_per_handling, np.ndarray) and snitt_per_handling.shape == (3,), "snitt_per_handling skal ha ett tall per handling (kolonne), altså axis=0."
assert np.allclose(snitt_per_handling, [3.75, 3.0, 3.5]), "snitt_per_handling har feil verdier."
```
:::

## Telle og keepdims
`np.sum(rewards == 10)` teller hvor mange som er sanne. `X.mean(axis=1, keepdims=True)` beholder aksen, så shapen blir `(n, 1)` og kan trekkes rett fra `X`.

::: oppgave Tell og sentrer rader
Lag `antall_maal` (antall episoder med reward 10), `rad_snitt` (snittet av hver rad med shape `(2, 1)`) og `X_rad_sentrert` (`X` minus radsnittet).
```start
import numpy as np

rewards = np.array([10, -1, 10, 10, -5, 10])
X = np.array([[1.0, 3.0],
              [2.0, 6.0]])

antall_maal = ...
rad_snitt = ...
X_rad_sentrert = ...
```
```losning
import numpy as np

rewards = np.array([10, -1, 10, 10, -5, 10])
X = np.array([[1.0, 3.0],
              [2.0, 6.0]])

antall_maal = np.sum(rewards == 10)
rad_snitt = X.mean(axis=1, keepdims=True)
X_rad_sentrert = X - rad_snitt
```
```test
import numpy as np
assert antall_maal is not ... and antall_maal == 4, "antall_maal skal være np.sum(rewards == 10), altså 4."
assert isinstance(rad_snitt, np.ndarray) and rad_snitt.shape == (2, 1), "rad_snitt skal ha shape (2, 1). Bruk X.mean(axis=1, keepdims=True)."
assert np.allclose(rad_snitt, [[2.0], [4.0]]), "rad_snitt skal være snittet av hver rad."
assert isinstance(X_rad_sentrert, np.ndarray) and np.allclose(X_rad_sentrert, [[-1, 1], [-2, 2]]), "X_rad_sentrert skal være X - rad_snitt."
```
:::

--- LEKSJON 2.7 | boolsk | Boolsk indeksering og np.where
`x > 0` gir en boolsk maske med `True`/`False`, `x[maske]` velger elementene der masken er `True`, og `maske.sum()` teller dem. `a[a < 0] = 0` endrer bare elementene der masken er `True`.

::: oppgave Masker på en vektor
Lag `positive` (tallene over 0), `antall_neg` (hvor mange som er under 0) og `relu`: en kopi av `x` der alle negative tall er satt til 0.
```start
import numpy as np

x = np.array([3, -1, 4, -1, 5, -9])

positive = ...
antall_neg = ...
relu = x.copy()
# sett negative tall i relu til 0 her
```
```losning
import numpy as np

x = np.array([3, -1, 4, -1, 5, -9])

positive = x[x > 0]
antall_neg = (x < 0).sum()
relu = x.copy()
relu[relu < 0] = 0
```
```test
import numpy as np
assert isinstance(positive, np.ndarray) and np.array_equal(positive, [3, 4, 5]), "positive skal være x[x > 0]."
assert antall_neg is not ... and antall_neg == 3, "antall_neg skal være (x < 0).sum(), altså 3."
assert isinstance(relu, np.ndarray) and np.array_equal(relu, [3, 0, 4, 0, 5, 0]), "relu skal ha 0 der x er negativ: relu[relu < 0] = 0."
assert np.array_equal(x, [3, -1, 4, -1, 5, -9]), "x skal ikke endres. Endre bare kopien relu."
```
:::

## Velge rader etter label
`X[y == 1]` gir radene der labelen er 1, fordi masken har én verdi per rad.

::: oppgave Snitt per klasse
Lag `X_1` (radene i klasse 1), `snitt_0` (snittet av hver feature i klasse 0) og `snitt_1` (det samme for klasse 1).
```start
import numpy as np

X = np.array([[1.0, 2.0],
              [1.5, 1.8],
              [5.0, 8.0],
              [6.0, 9.0]])
y = np.array([0, 0, 1, 1])

X_1 = ...
snitt_0 = ...
snitt_1 = ...
```
```losning
import numpy as np

X = np.array([[1.0, 2.0],
              [1.5, 1.8],
              [5.0, 8.0],
              [6.0, 9.0]])
y = np.array([0, 0, 1, 1])

X_1 = X[y == 1]
snitt_0 = X[y == 0].mean(axis=0)
snitt_1 = X[y == 1].mean(axis=0)
```
```test
import numpy as np
assert isinstance(X_1, np.ndarray) and X_1.shape == (2, 2), "X_1 skal være de to radene der y == 1: X[y == 1]."
assert np.allclose(X_1, [[5.0, 8.0], [6.0, 9.0]]), "X_1 har feil rader."
assert isinstance(snitt_0, np.ndarray) and np.allclose(snitt_0, [1.25, 1.9]), "snitt_0 skal være X[y == 0].mean(axis=0)."
assert isinstance(snitt_1, np.ndarray) and np.allclose(snitt_1, [5.5, 8.5]), "snitt_1 skal være X[y == 1].mean(axis=0)."
```
:::

## Accuracy på én linje
`(y_pred == y).mean()` gir andelen riktige (accuracy), fordi `True` teller som 1 og `False` som 0.

::: oppgave Accuracy og antall feil
Lag `accuracy` og `antall_feil` for prediksjonene.
```start
import numpy as np

y = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])

accuracy = ...
antall_feil = ...
```
```losning
import numpy as np

y = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])

accuracy = (y_pred == y).mean()
antall_feil = np.sum(y_pred != y)
```
```test
import numpy as np
assert accuracy is not ... and np.isclose(accuracy, 0.75), f"accuracy skal være 0.75 (6 av 8 riktige), men er {accuracy}. Bruk (y_pred == y).mean()."
assert antall_feil is not ... and antall_feil == 2, "antall_feil skal være np.sum(y_pred != y)."
```
:::

## Kombinere betingelser
Bruk `&` (og), `|` (eller) og `~` (ikke) med parenteser rundt hver sammenligning, som i `x[(x > 2) & (x < 8)]`. Python-ordene `and`/`or` krasjer på arrays.

::: oppgave Rett betingelsen
Første linje krasjer. Rett den, og lag `utenfor` (tallene under 2 eller over 8) og `ikke_stor` (tallene som ikke er over 4).
```start
import numpy as np

x = np.array([1, 5, 10, 3, 7])

mellom = x[x > 2 and x < 8]
utenfor = ...
ikke_stor = ...
```
```losning
import numpy as np

x = np.array([1, 5, 10, 3, 7])

mellom = x[(x > 2) & (x < 8)]
utenfor = x[(x < 2) | (x > 8)]
ikke_stor = x[~(x > 4)]
```
```test
import numpy as np
assert isinstance(mellom, np.ndarray) and np.array_equal(mellom, [5, 3, 7]), "mellom skal være x[(x > 2) & (x < 8)]."
assert isinstance(utenfor, np.ndarray) and np.array_equal(utenfor, [1, 10]), "utenfor skal være x[(x < 2) | (x > 8)]."
assert isinstance(ikke_stor, np.ndarray) and np.array_equal(ikke_stor, [1, 3]), "ikke_stor skal være x[~(x > 4)]."
```
:::

## np.where og np.isin
`np.where(p >= 0.5, 1, 0)` gir 1 der betingelsen er sann og 0 ellers, og `np.where(p > 0.7)[0]` gir indeksene der den er sann. `np.isin(a, [1, 3])` sjekker for hvert element om det finnes i listen.

::: oppgave Fra sannsynligheter til klasser
Lag `y_pred` med 1 der `p >= 0.5` og 0 ellers, og `sikre` med indeksene der `p > 0.6`.
```start
import numpy as np

p = np.array([0.2, 0.5, 0.9, 0.7, 0.1, 0.45])

y_pred = ...
sikre = ...
```
```losning
import numpy as np

p = np.array([0.2, 0.5, 0.9, 0.7, 0.1, 0.45])

y_pred = np.where(p >= 0.5, 1, 0)
sikre = np.where(p > 0.6)[0]
```
```test
import numpy as np
assert isinstance(y_pred, np.ndarray) and y_pred.shape == (6,), "y_pred skal ha én klasse per sannsynlighet. Bruk np.where(p >= 0.5, 1, 0)."
assert y_pred[1] == 1, "p = 0.5 skal gi 1, fordi betingelsen er p >= 0.5 (ikke p > 0.5)."
assert np.array_equal(y_pred, [0, 1, 1, 1, 0, 0]), "y_pred skal være [0, 1, 1, 1, 0, 0]."
assert isinstance(sikre, np.ndarray) and np.array_equal(sikre, [2, 3]), "sikre skal være np.where(p > 0.6)[0], altså indeksene [2, 3]."
```
:::

::: oppgave Blank ut ruter
Lag `blank`, som er `True` der `kart` er vann (1) eller mål (3), og `heatmap`, som er `verdier` med `np.nan` på de rutene.
```start
import numpy as np

kart = np.array([[0, 0, 2],
                 [1, 0, 3],
                 [1, 1, 0]])        # 0 = land, 1 = vann, 2 = start, 3 = mål
verdier = np.array([[0.5, 0.6, 0.4],
                    [0.9, 0.8, 1.0],
                    [0.2, 0.1, 0.7]])

blank = ...
heatmap = ...
```
```losning
import numpy as np

kart = np.array([[0, 0, 2],
                 [1, 0, 3],
                 [1, 1, 0]])
verdier = np.array([[0.5, 0.6, 0.4],
                    [0.9, 0.8, 1.0],
                    [0.2, 0.1, 0.7]])

blank = np.isin(kart, [1, 3])
heatmap = np.where(blank, np.nan, verdier)
```
```test
import numpy as np
fasit = np.array([[False, False, False], [True, False, True], [True, True, False]])
assert isinstance(blank, np.ndarray) and blank.shape == (3, 3), "blank skal ha samme shape som kart. Bruk np.isin(kart, [1, 3])."
assert np.array_equal(blank, fasit), "blank skal være True der kart er 1 eller 3."
assert isinstance(heatmap, np.ndarray) and np.array_equal(np.isnan(heatmap), fasit), "heatmap skal ha np.nan akkurat der blank er True: np.where(blank, np.nan, verdier)."
assert np.allclose(heatmap[~fasit], verdier[~fasit]), "De andre rutene i heatmap skal beholde verdiene fra verdier."
```
:::

--- LEKSJON 2.8 | form | Form: reshape, ravel, .T og stacking
`a.reshape(3, 4)` gir de samme tallene i ny form, fylt inn rad for rad, og `-1` for én akse betyr «regn ut lengden selv».

::: oppgave Ny form
Lag `M` (`a` med 3 rader og 4 kolonner), `N` (`a` med 2 rader, la NumPy regne ut resten) og `bilder`: de 3 radene i `batch` som tre 4×4-bilder, shape `(3, 4, 4)`.
```start
import numpy as np

a = np.arange(12)
batch = np.arange(48).reshape(3, 16)   # 3 bilder med 16 piksler hver

M = ...
N = ...
bilder = ...
```
```losning
import numpy as np

a = np.arange(12)
batch = np.arange(48).reshape(3, 16)

M = a.reshape(3, 4)
N = a.reshape(2, -1)
bilder = batch.reshape(-1, 4, 4)
```
```test
import numpy as np
assert isinstance(M, np.ndarray) and M.shape == (3, 4), "M skal ha shape (3, 4): a.reshape(3, 4)."
assert M[1, 0] == 4, "reshape fyller rad for rad, så M[1, 0] skal være 4."
assert isinstance(N, np.ndarray) and N.shape == (2, 6), "N skal ha shape (2, 6): a.reshape(2, -1)."
assert isinstance(bilder, np.ndarray) and bilder.shape == (3, 4, 4), "bilder skal ha shape (3, 4, 4): batch.reshape(-1, 4, 4)."
assert bilder[1, 0, 0] == 16 and bilder[2, 3, 3] == 47, "Hvert bilde skal være én rad i batch."
```
:::

## reshape(-1, 1): sklearn vil ha 2D
sklearn krever `X` med shape `(n_samples, n_features)`, også med én feature, så et 1D-array må bli en kolonne med `x.reshape(-1, 1)`. `x.reshape(1, -1)` gir derimot én rad.

::: oppgave Gjør x klar for sklearn
`fit` krasjer fordi `x` er 1D. Lag `X` som en kolonnevektor med shape `(4, 1)`, og tren modellen på `X` i stedet.
```start
import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([2.1, 3.9, 6.2, 7.8])

X = ...
modell = LinearRegression().fit(x, y)
```
```losning
import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([2.1, 3.9, 6.2, 7.8])

X = x.reshape(-1, 1)
modell = LinearRegression().fit(X, y)
```
```test
import numpy as np
assert isinstance(X, np.ndarray) and X.shape == (4, 1), "X skal ha shape (4, 1): x.reshape(-1, 1)."
assert np.allclose(X[:, 0], x), "X skal inneholde de samme tallene som x."
assert hasattr(modell, "coef_") and modell.coef_.shape == (1,), "modell skal være trent på X: LinearRegression().fit(X, y)."
```
:::

## ravel og np.newaxis
`M.ravel()` gjør alt om til 1D rad for rad (`M.flatten()` gjør det samme, men lager alltid en kopi). `x[:, np.newaxis]` (eller `x[:, None]`) gir shape `(n, 1)`, og `x[np.newaxis, :]` gir `(1, n)`.

::: oppgave Flat ut og legg til akse
Lag `flat` (`M` som 1D), `kol` (`x` med shape `(3, 1)`) og `rad` (`x` med shape `(1, 3)`).
```start
import numpy as np

M = np.array([[1, 2, 3],
              [4, 5, 6]])
x = np.array([7, 8, 9])

flat = ...
kol = ...
rad = ...
```
```losning
import numpy as np

M = np.array([[1, 2, 3],
              [4, 5, 6]])
x = np.array([7, 8, 9])

flat = M.ravel()
kol = x[:, np.newaxis]
rad = x[np.newaxis, :]
```
```test
import numpy as np
assert isinstance(flat, np.ndarray) and np.array_equal(flat, [1, 2, 3, 4, 5, 6]), "flat skal være M.ravel(): rad 0 først, så rad 1."
assert isinstance(kol, np.ndarray) and kol.shape == (3, 1) and np.array_equal(kol[:, 0], x), "kol skal ha shape (3, 1): x[:, np.newaxis]."
assert isinstance(rad, np.ndarray) and rad.shape == (1, 3) and np.array_equal(rad[0], x), "rad skal ha shape (1, 3): x[np.newaxis, :]."
```
:::

## Transponering med .T
`X.T` bytter rader og kolonner (`(n, d)` blir `(d, n)`), og `x0, x1 = X.T` pakker ut kolonnene.

::: oppgave Transponer og pakk ut
Lag `XT` (`X` transponert), og pakk ut de tre kolonnene i `punkter` til `xs`, `ys` og `zs` på én linje.
```start
import numpy as np

X = np.array([[1, 2],
              [3, 4],
              [5, 6]])
punkter = np.array([[0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0]])

XT = ...
xs, ys, zs = ..., ..., ...
```
```losning
import numpy as np

X = np.array([[1, 2],
              [3, 4],
              [5, 6]])
punkter = np.array([[0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0]])

XT = X.T
xs, ys, zs = punkter.T
```
```test
import numpy as np
assert isinstance(XT, np.ndarray) and XT.shape == (2, 3), "XT skal ha shape (2, 3): X.T."
assert np.array_equal(XT, [[1, 3, 5], [2, 4, 6]]), "Kolonnene i X skal bli rader i XT."
assert isinstance(xs, np.ndarray) and np.array_equal(xs, punkter[:, 0]), "xs skal være første kolonne i punkter: xs, ys, zs = punkter.T."
assert isinstance(ys, np.ndarray) and np.array_equal(ys, punkter[:, 1]), "ys skal være andre kolonne i punkter."
assert isinstance(zs, np.ndarray) and np.array_equal(zs, punkter[:, 2]), "zs skal være tredje kolonne i punkter."
```
:::

## Sette sammen arrays
`np.column_stack((x0, x1))` gjør 1D-arrays til kolonner, `np.vstack((a, b))` stabler dem som rader og `np.hstack((a, b))` skjøter bortover. `np.concatenate((A, B), axis=0)` lar deg velge aksen selv.

::: oppgave Lag X fra kolonner
Lag `X` med shape `(4, 2)` (én person per rad), `X_b` med en kolonne med enere først og så `alder` og `hoyde` (shape `(4, 3)`), og `X_rader` der `alder` og `hoyde` er rader (shape `(2, 4)`).
```start
import numpy as np

alder = np.array([23, 35, 41, 19])
hoyde = np.array([180, 165, 172, 190])

X = ...
X_b = ...
X_rader = ...
```
```losning
import numpy as np

alder = np.array([23, 35, 41, 19])
hoyde = np.array([180, 165, 172, 190])

X = np.column_stack((alder, hoyde))
X_b = np.column_stack((np.ones(len(alder)), alder, hoyde))
X_rader = np.vstack((alder, hoyde))
```
```test
import numpy as np
assert isinstance(X, np.ndarray) and X.shape == (4, 2), "X skal ha shape (4, 2). Bruk np.column_stack((alder, hoyde))."
assert np.array_equal(X[:, 0], alder) and np.array_equal(X[:, 1], hoyde), "Første kolonne i X skal være alder, andre skal være hoyde."
assert isinstance(X_b, np.ndarray) and X_b.shape == (4, 3), "X_b skal ha shape (4, 3): enere, alder, hoyde."
assert np.all(X_b[:, 0] == 1) and np.array_equal(X_b[:, 1:], X), "Første kolonne i X_b skal være enere (np.ones(4)), så alder og hoyde."
assert isinstance(X_rader, np.ndarray) and X_rader.shape == (2, 4) and np.array_equal(X_rader[0], alder), "X_rader skal være np.vstack((alder, hoyde))."
```
:::

::: oppgave Skjøt sammen
Lag `flere_rader` (`A` over `B`, shape `(4, 3)`), `flere_kol` (`A` til venstre for `B`, shape `(2, 6)`) og `skjott` (`x1` og så `x2` i ett 1D-array).
```start
import numpy as np

A = np.ones((2, 3))
B = np.zeros((2, 3))
x1 = np.array([1, 2, 3])
x2 = np.array([4, 5])

flere_rader = ...
flere_kol = ...
skjott = ...
```
```losning
import numpy as np

A = np.ones((2, 3))
B = np.zeros((2, 3))
x1 = np.array([1, 2, 3])
x2 = np.array([4, 5])

flere_rader = np.concatenate((A, B), axis=0)
flere_kol = np.concatenate((A, B), axis=1)
skjott = np.hstack((x1, x2))
```
```test
import numpy as np
assert isinstance(flere_rader, np.ndarray) and flere_rader.shape == (4, 3), "flere_rader skal ha shape (4, 3): np.concatenate((A, B), axis=0)."
assert np.all(flere_rader[:2] == 1) and np.all(flere_rader[2:] == 0), "A skal ligge øverst og B nederst."
assert isinstance(flere_kol, np.ndarray) and flere_kol.shape == (2, 6), "flere_kol skal ha shape (2, 6): np.concatenate((A, B), axis=1)."
assert np.all(flere_kol[:, :3] == 1) and np.all(flere_kol[:, 3:] == 0), "A skal ligge til venstre og B til høyre."
assert isinstance(skjott, np.ndarray) and np.array_equal(skjott, [1, 2, 3, 4, 5]), "skjott skal være np.hstack((x1, x2))."
```
:::

--- LEKSJON 2.9 | linalg | Lineær algebra: @, matmul, dot, norm og eig
`a @ b` (eller `np.dot(a, b)`) gir prikkproduktet av to vektorer, og `X @ w` (eller `np.matmul(X, w)`) gir prikkproduktet mellom hver rad i `X` og `w`, altså én verdi per datapunkt. `a * b` er fortsatt element for element.

::: oppgave Prikkprodukt
Lag `prikk` (prikkproduktet av `a` og `b`) og `elementvis` (`a` ganget med `b` element for element).
```start
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

prikk = ...
elementvis = ...
```
```losning
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

prikk = a @ b
elementvis = a * b
```
```test
import numpy as np
assert prikk is not ... and np.ndim(prikk) == 0, "prikk skal være ett tall. Bruk a @ b (ikke a * b)."
assert np.isclose(prikk, 32), "prikk skal være 1*4 + 2*5 + 3*6 = 32."
assert isinstance(elementvis, np.ndarray) and np.allclose(elementvis, [4, 10, 18]), "elementvis skal være a * b."
```
:::

::: oppgave Lineær modell
Regn ut `z = X @ w + b0`, én prediksjon per datapunkt.
```start
import numpy as np

X = np.array([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0]])    # 3 datapunkter, 2 features
w = np.array([0.5, -1.0])
b0 = 2.0

z = ...
```
```losning
import numpy as np

X = np.array([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0]])
w = np.array([0.5, -1.0])
b0 = 2.0

z = X @ w + b0
```
```test
import numpy as np
assert isinstance(z, np.ndarray) and z.shape == (3,), "z skal ha én verdi per datapunkt, shape (3,). Bruk X @ w + b0 (ikke X * w)."
assert np.allclose(z, [0.5, -0.5, -1.5]), "z har feil verdier. Husk å legge til b0."
```
:::

## Shape-regelen for matriseprodukt
`(n, k) @ (k, m)` gir `(n, m)`: de indre dimensjonene må være like, ellers får du en feilmelding.

::: oppgave Få formene til å passe
`X @ w` krasjer. Rett `w` (fortsatt bare enere) så `z` blir én verdi per rad, og lag `W` (enere) slik at `H = X @ W` får shape `(5, 8)`, som et lag i et nevralt nett.
```start
import numpy as np

X = np.ones((5, 3))    # 5 datapunkter, 3 features

w = np.ones(5)
z = X @ w
W = ...
H = ...
```
```losning
import numpy as np

X = np.ones((5, 3))

w = np.ones(3)
z = X @ w
W = np.ones((3, 8))
H = X @ W
```
```test
import numpy as np
assert isinstance(w, np.ndarray) and w.shape == (3,) and np.all(w == 1), "w skal ha én vekt per feature: np.ones(3)."
assert isinstance(z, np.ndarray) and z.shape == (5,) and np.allclose(z, 3), "z skal være X @ w med shape (5,)."
assert isinstance(W, np.ndarray) and W.shape == (3, 8), "W skal ha shape (3, 8): (5, 3) @ (3, 8) gir (5, 8)."
assert isinstance(H, np.ndarray) and H.shape == (5, 8), "H skal være X @ W med shape (5, 8)."
```
:::

::: oppgave Logistisk prediksjon
Regn ut `z = np.matmul(X, w) + b`, `y_pred` (sigmoid av `z`, sannsynligheter) og `klasser` (1 der `y_pred >= 0.5`, ellers 0).
```start
import numpy as np

X = np.array([[1.0, 2.0, 0.0],
              [0.0, -1.0, 3.0],
              [2.0, 2.0, 2.0],
              [-1.0, 0.5, 1.0]])
w = np.array([0.5, -0.25, 1.0])
b = -0.5

z = ...
y_pred = ...
klasser = ...
```
```losning
import numpy as np

X = np.array([[1.0, 2.0, 0.0],
              [0.0, -1.0, 3.0],
              [2.0, 2.0, 2.0],
              [-1.0, 0.5, 1.0]])
w = np.array([0.5, -0.25, 1.0])
b = -0.5

z = np.matmul(X, w) + b
y_pred = 1 / (1 + np.exp(-z))
klasser = (y_pred >= 0.5).astype(int)
```
```test
import numpy as np
assert isinstance(z, np.ndarray) and z.shape == (4,), "z skal ha én verdi per datapunkt, shape (4,). Bruk np.matmul(X, w) + b."
assert np.allclose(z, [-0.5, 2.75, 2.0, -0.125]), "z har feil verdier. Husk å legge til b."
assert isinstance(y_pred, np.ndarray) and np.allclose(y_pred, 1 / (1 + np.exp(-z))), "y_pred skal være sigmoid av z: 1 / (1 + np.exp(-z))."
assert isinstance(klasser, np.ndarray) and np.array_equal(klasser, [0, 1, 1, 0]), "klasser skal være 1 der y_pred >= 0.5 og 0 ellers."
```
:::

## Normen: lengde og avstand
`np.linalg.norm(v)` gir lengden til `v`, `np.linalg.norm(p1 - p2)` avstanden mellom to punkter, og `np.linalg.norm(X - senter, axis=1)` avstanden fra hver rad til `senter`.

::: oppgave Nærmeste punkt
Lag `lengde` (normen av `w`), `avstand` (mellom `p1` og `p2`), `avstander` (fra hver rad i `X` til `senter`) og `naermest` (indeksen til raden nærmest `senter`).
```start
import numpy as np

w = np.array([1.0, 2.0, 2.0])
p1 = np.array([0.0, 0.0])
p2 = np.array([3.0, 4.0])
X = np.array([[1.0, 1.0],
              [4.0, 5.0],
              [0.0, 2.0]])
senter = np.array([1.0, 1.5])

lengde = ...
avstand = ...
avstander = ...
naermest = ...
```
```losning
import numpy as np

w = np.array([1.0, 2.0, 2.0])
p1 = np.array([0.0, 0.0])
p2 = np.array([3.0, 4.0])
X = np.array([[1.0, 1.0],
              [4.0, 5.0],
              [0.0, 2.0]])
senter = np.array([1.0, 1.5])

lengde = np.linalg.norm(w)
avstand = np.linalg.norm(p1 - p2)
avstander = np.linalg.norm(X - senter, axis=1)
naermest = np.argmin(avstander)
```
```test
import numpy as np
assert lengde is not ... and np.isclose(lengde, 3.0), "lengde skal være np.linalg.norm(w) = 3."
assert avstand is not ... and np.isclose(avstand, 5.0), "avstand skal være np.linalg.norm(p1 - p2) = 5."
assert isinstance(avstander, np.ndarray) and avstander.shape == (3,), "avstander skal ha én avstand per rad. Bruk axis=1."
assert np.allclose(avstander, [0.5, 4.60977223, 1.11803399]), "avstander har feil verdier: np.linalg.norm(X - senter, axis=1)."
assert naermest is not ... and naermest == 0, "naermest skal være np.argmin(avstander)."
```
:::

## Kovarians og egenvektorer (til PCA)
`np.cov(X, rowvar=False)` (eller `np.cov(X.T)`) gir kovariansmatrisen når radene er datapunkter. `np.linalg.eigh(S)` gir egenverdiene sortert stigende og egenvektorene som kolonner (`vektorer[:, i]`), mens `np.linalg.eig` virker på alle kvadratiske matriser, men ikke sorterer.

::: oppgave PCA for hånd
Lag `S` (kovariansmatrisen til `X`, shape `(3, 3)`), `egenverdier` og `egenvektorer` med `eigh`, og `forste_pc`: egenvektoren som hører til den største egenverdien.
```start
import numpy as np

X = np.array([[2.0, 0.5, 1.0],
              [1.0, 1.5, 0.0],
              [3.0, 1.0, 2.0],
              [0.0, 2.0, 0.5],
              [4.0, 0.0, 3.0]])

S = ...
egenverdier, egenvektorer = ..., ...
forste_pc = ...
```
```losning
import numpy as np

X = np.array([[2.0, 0.5, 1.0],
              [1.0, 1.5, 0.0],
              [3.0, 1.0, 2.0],
              [0.0, 2.0, 0.5],
              [4.0, 0.0, 3.0]])

S = np.cov(X, rowvar=False)
egenverdier, egenvektorer = np.linalg.eigh(S)
forste_pc = egenvektorer[:, np.argmax(egenverdier)]
```
```test
import numpy as np
S_r = np.cov(X, rowvar=False)
assert isinstance(S, np.ndarray) and S.shape == (3, 3), "S skal ha shape (3, 3), én rad og kolonne per feature. Bruk np.cov(X, rowvar=False)."
assert np.allclose(S, S_r), "S har feil verdier. np.cov forventer variabler som rader, så skriv np.cov(X, rowvar=False) eller np.cov(X.T)."
assert isinstance(egenverdier, np.ndarray) and np.allclose(np.sort(egenverdier), np.linalg.eigvalsh(S_r)), "egenverdier skal komme fra np.linalg.eigh(S)."
assert isinstance(forste_pc, np.ndarray) and forste_pc.shape == (3,), "forste_pc skal være én egenvektor med shape (3,)."
fasit = np.linalg.eigh(S_r)[1][:, -1]
assert np.isclose(abs(forste_pc @ fasit), 1.0), "forste_pc er feil. Egenvektorene er kolonnene: egenvektorer[:, np.argmax(egenverdier)]."
```
:::

--- LEKSJON 2.10 | tilfeldig | Tilfeldige tall og seeds
Samme seed gir de samme tilfeldige tallene hver gang: gammel stil er `np.random.seed(0)` og så `np.random.random(3)` eller `np.random.randint(0, 4)`, ny stil er `rng = np.random.default_rng(0)` og så `rng.random(3)`.

::: oppgave Reproduserbare tall
Sett seed 0 med gammel stil før tallene trekkes, slik at `tall` og `heltall` blir de samme hver gang.
```start
import numpy as np

# sett seed her

tall = np.random.random(3)
heltall = np.random.randint(0, 4, size=5)
```
```losning
import numpy as np

np.random.seed(0)

tall = np.random.random(3)
heltall = np.random.randint(0, 4, size=5)
```
```test
import numpy as np
np.random.seed(0)
fasit_tall = np.random.random(3)
fasit_heltall = np.random.randint(0, 4, size=5)
assert np.allclose(tall, fasit_tall), "tall blir ikke det samme hver gang. Skriv np.random.seed(0) før tallene trekkes."
assert np.array_equal(heltall, fasit_heltall), "heltall skal trekkes rett etter tall, med samme seed."
```
:::

## De viktigste funksjonene
`rng.uniform(-1, 1, size=4)`, `rng.normal(loc=0, scale=0.1, size=(4, 3))`, `rng.integers(1, 7, size=10)` og `rng.choice(100, size=16, replace=False)` trekker tilfeldige tall. Øvre grense i `integers` og `randint` er ikke med, akkurat som i `range`.

::: oppgave Trekk tall med rng
Bruk `rng` til å lage `u` (4 tall uniformt mellom -1 og 1), `W` (startvekter, normalfordelt med snitt 0 og standardavvik 0.1, shape `(4, 3)`), `terning` (1000 terningkast, tallene 1–6) og `utvalg` (16 ulike indekser fra 0–99).
```start
import numpy as np

rng = np.random.default_rng(42)

u = ...
W = ...
terning = ...
utvalg = ...
```
```losning
import numpy as np

rng = np.random.default_rng(42)

u = rng.uniform(-1, 1, size=4)
W = rng.normal(loc=0, scale=0.1, size=(4, 3))
terning = rng.integers(1, 7, size=1000)
utvalg = rng.choice(100, size=16, replace=False)
```
```test
import numpy as np
assert isinstance(u, np.ndarray) and u.shape == (4,) and np.all((u >= -1) & (u < 1)), "u skal være rng.uniform(-1, 1, size=4)."
assert isinstance(W, np.ndarray) and W.shape == (4, 3), "W skal ha shape (4, 3): rng.normal(loc=0, scale=0.1, size=(4, 3))."
assert 0.02 < W.std() < 0.3 and abs(W.mean()) < 0.15, "W skal ha snitt 0 og standardavvik 0.1 (scale=0.1)."
assert isinstance(terning, np.ndarray) and len(terning) == 1000, "terning skal ha 1000 kast: size=1000."
assert terning.min() == 1 and terning.max() == 6, f"terning skal gi tallene 1–6, men min er {terning.min()} og maks {terning.max()}. Øvre grense er ikke med: rng.integers(1, 7, ...)."
assert isinstance(utvalg, np.ndarray) and len(utvalg) == 16, "utvalg skal ha 16 indekser."
assert len(np.unique(utvalg)) == 16 and utvalg.min() >= 0 and utvalg.max() <= 99, "utvalg skal ha 16 ulike tall fra 0–99: rng.choice(100, size=16, replace=False)."
```
:::

## Stokke X og y sammen
Lag én rekkefølge `idx = rng.permutation(len(X))` og bruk den på begge (`X[idx]`, `y[idx]`), ellers havner labelene på feil datapunkt.

::: oppgave Stokk og del opp
Lag `idx` med `rng.permutation`, stokk `X` og `y` likt, og la de 8 første stokkede datapunktene bli `X_train`/`y_train` og de 2 siste `X_test`/`y_test`.
```start
import numpy as np

X = np.arange(30).reshape(10, 3)
y = X[:, 0] // 3            # y[i] hører til X[i]
rng = np.random.default_rng(1)

idx = ...
X_train, X_test = ..., ...
y_train, y_test = ..., ...
```
```losning
import numpy as np

X = np.arange(30).reshape(10, 3)
y = X[:, 0] // 3
rng = np.random.default_rng(1)

idx = rng.permutation(len(X))
X_train, X_test = X[idx[:8]], X[idx[8:]]
y_train, y_test = y[idx[:8]], y[idx[8:]]
```
```test
import numpy as np
assert isinstance(idx, np.ndarray) and sorted(idx.tolist()) == list(range(10)), "idx skal være en permutasjon av 0..9: rng.permutation(len(X))."
assert np.array_equal(idx, np.random.default_rng(1).permutation(10)), "Bruk rng.permutation(len(X)) (eller rng.permutation(10)) én gang, uten å bruke rng til noe annet først."
assert isinstance(X_train, np.ndarray) and X_train.shape == (8, 3) and X_test.shape == (2, 3), "X_train skal ha 8 rader og X_test 2 rader."
assert np.array_equal(y_train, X_train[:, 0] // 3) and np.array_equal(y_test, X_test[:, 0] // 3), "y må stokkes med samme idx som X, ellers passer ikke labelene til radene."
assert np.array_equal(X_train, X[idx[:8]]), "X_train skal være de 8 første radene etter stokking."
```
:::

## ε-greedy: utforske eller utnytte
`np.random.random() < epsilon` er sant i en andel ε av gangene: da utforsker agenten med `np.random.randint(n_actions)`, ellers utnytter den med `np.argmax(q_table[state])`.

::: oppgave Velg handling med ε-greedy
Skriv `velg_handling(q_table, state, epsilon)` som returnerer en tilfeldig handling med sannsynlighet `epsilon`, og ellers den beste handlingen i tilstand `state` (som `int`).
```start
import numpy as np

def velg_handling(q_table, state, epsilon):
    return ...
```
```losning
import numpy as np

def velg_handling(q_table, state, epsilon):
    if np.random.random() < epsilon:
        return int(np.random.randint(q_table.shape[1]))
    return int(np.argmax(q_table[state]))
```
```test
import numpy as np
q = np.zeros((16, 4))
q[5] = [0.1, 0.9, 0.3, 0.2]
r = velg_handling(q, 5, 0.0)
assert r is not ... and r is not None, "velg_handling skal returnere en handling."
assert all(int(velg_handling(q, 5, 0.0)) == 1 for _ in range(50)), "Med epsilon = 0 skal den alltid velge beste handling, np.argmax(q_table[state]) (her 1)."
valg = {int(velg_handling(q, 5, 1.0)) for _ in range(200)}
assert len(valg) > 1, "Med epsilon = 1 skal den velge tilfeldige handlinger. Bruk if np.random.random() < epsilon."
assert valg <= {0, 1, 2, 3}, f"Tilfeldige handlinger skal være 0–3, men fikk {sorted(valg)}. Bruk np.random.randint(q_table.shape[1])."
```
:::

--- LEKSJON 2.11 | verktoy | Nyttige verktøy
Her er funksjonene som dukker opp her og der i prosjektene.

## Sortering: sort og argsort
`np.sort(x)` gir verdiene sortert, og `np.argsort(x)` gir indeksene som sorterer `x`. Dermed er `x[np.argsort(x)]` sortert, og `np.argsort(x)[::-1]` gir synkende rekkefølge.

::: oppgave Viktigste features
Lag `sortert` (verdiene stigende), `rekkefolge` (indeksene som sorterer `viktighet`) og `topp2` (indeksene til de to viktigste featurene, viktigste først).
```start
import numpy as np

viktighet = np.array([0.05, 0.40, 0.10, 0.45])

sortert = ...
rekkefolge = ...
topp2 = ...
```
```losning
import numpy as np

viktighet = np.array([0.05, 0.40, 0.10, 0.45])

sortert = np.sort(viktighet)
rekkefolge = np.argsort(viktighet)
topp2 = np.argsort(viktighet)[::-1][:2]
```
```test
import numpy as np
assert isinstance(sortert, np.ndarray) and np.allclose(sortert, [0.05, 0.10, 0.40, 0.45]), "sortert skal være np.sort(viktighet)."
assert isinstance(rekkefolge, np.ndarray) and np.array_equal(rekkefolge, [0, 2, 1, 3]), "rekkefolge skal være np.argsort(viktighet): indeksene, ikke verdiene."
assert isinstance(topp2, np.ndarray) and np.array_equal(topp2, [3, 1]), "topp2 skal være [3, 1]: snu argsort med [::-1] og ta de to første."
```
:::

::: oppgave Gjenopprett pikselrekkefølgen
Kolonnene i `rad` er stokket, og `"px_3"` i `header` betyr piksel 3. Lag `nummer` (pikselnummeret fra hvert navn, med `int(c.split("_")[1])`), `order` (rekkefølgen som sorterer `nummer`) og `gjenopprettet` (`rad` i riktig rekkefølge).
```start
import numpy as np

header = ["px_3", "px_0", "px_2", "px_1"]
rad = np.array([30, 0, 20, 10])

nummer = ...
order = ...
gjenopprettet = ...
```
```losning
import numpy as np

header = ["px_3", "px_0", "px_2", "px_1"]
rad = np.array([30, 0, 20, 10])

nummer = np.array([int(c.split("_")[1]) for c in header])
order = np.argsort(nummer)
gjenopprettet = rad[order]
```
```test
import numpy as np
assert nummer is not ... and np.array_equal(np.asarray(nummer), [3, 0, 2, 1]), "nummer skal være [3, 0, 2, 1]: tallet etter _ i hvert kolonnenavn."
assert order is not ... and np.array_equal(np.asarray(order), [1, 3, 2, 0]), "order skal være np.argsort(nummer)."
assert isinstance(gjenopprettet, np.ndarray) and np.array_equal(gjenopprettet, [0, 10, 20, 30]), "gjenopprettet skal være rad[order], som gir [0, 10, 20, 30]."
```
:::

## Telle: unique og bincount
`verdier, antall = np.unique(y, return_counts=True)` gir de ulike verdiene og antallet av hver, og `np.bincount(y)[k]` er antall `k`-er (for ikke-negative heltall).

::: oppgave Klassebalanse
Lag `verdier` og `antall` med `np.unique`, og `andeler` (andelen av hver klasse) med `np.bincount`.
```start
import numpy as np

y = np.array([0, 2, 1, 0, 0, 2, 0])

verdier, antall = ..., ...
andeler = ...
```
```losning
import numpy as np

y = np.array([0, 2, 1, 0, 0, 2, 0])

verdier, antall = np.unique(y, return_counts=True)
andeler = np.bincount(y) / len(y)
```
```test
import numpy as np
assert isinstance(verdier, np.ndarray) and np.array_equal(verdier, [0, 1, 2]), "verdier skal være de ulike klassene: np.unique(y, return_counts=True)."
assert isinstance(antall, np.ndarray) and np.array_equal(antall, [4, 1, 2]), "antall skal være [4, 1, 2]. Husk return_counts=True."
assert isinstance(andeler, np.ndarray) and np.allclose(andeler, [4 / 7, 1 / 7, 2 / 7]), "andeler skal være np.bincount(y) / len(y)."
```
:::

## clip, cumsum og glidende snitt
`np.clip(p, 1e-15, 1 - 1e-15)` holder `p` unna 0 og 1 så `np.log` ikke gir `-inf`, og `np.cumsum(x)` gir kumulativ sum. `np.convolve(x, np.ones(k) / k, mode="valid")` gir glidende snitt over `k` verdier, med lengde `len(x) - k + 1`.

::: oppgave Trygg cross-entropy
`bce` blir `nan` fordi `p` inneholder 0 og 1. Lag `p_trygg` med `np.clip`, så `bce` blir et endelig tall.
```start
import numpy as np

y = np.array([0, 1, 1, 0])
p = np.array([0.0, 1.0, 0.8, 0.3])

p_trygg = p      # bytt ut med np.clip
bce = -np.mean(y * np.log(p_trygg) + (1 - y) * np.log(1 - p_trygg))
```
```losning
import numpy as np

y = np.array([0, 1, 1, 0])
p = np.array([0.0, 1.0, 0.8, 0.3])

p_trygg = np.clip(p, 1e-15, 1 - 1e-15)
bce = -np.mean(y * np.log(p_trygg) + (1 - y) * np.log(1 - p_trygg))
```
```test
import numpy as np
assert isinstance(p_trygg, np.ndarray) and p_trygg.min() > 0 and p_trygg.max() < 1, "p_trygg skal ligge strengt mellom 0 og 1: np.clip(p, 1e-15, 1 - 1e-15)."
assert np.isfinite(bce), "bce er ikke et endelig tall. Bruk p_trygg fra np.clip i formelen."
assert np.isclose(bce, 0.14496, atol=1e-3), f"bce skal være omtrent 0.145, men er {bce}."
```
:::

::: oppgave Glatt ut en læringskurve
Lag `glatt` (glidende snitt av `steg` over 4 og 4 episoder, `mode="valid"`) og `total` (kumulativ sum av `belonning`).
```start
import numpy as np

steg = np.array([100, 80, 96, 60, 52, 40, 36, 20, 24, 12])
belonning = np.array([1, -1, 2, 0, 3])

glatt = ...
total = ...
```
```losning
import numpy as np

steg = np.array([100, 80, 96, 60, 52, 40, 36, 20, 24, 12])
belonning = np.array([1, -1, 2, 0, 3])

glatt = np.convolve(steg, np.ones(4) / 4, mode="valid")
total = np.cumsum(belonning)
```
```test
import numpy as np
assert isinstance(glatt, np.ndarray), "glatt skal være resultatet av np.convolve."
assert len(glatt) == 7, f"glatt skal ha lengde 10 - 4 + 1 = 7 med mode='valid', men har {len(glatt)}."
assert np.allclose(glatt, [84, 72, 62, 47, 37, 30, 23]), "glatt har feil verdier. Vinduet skal være np.ones(4) / 4."
assert isinstance(total, np.ndarray) and np.array_equal(total, [1, 0, 2, 2, 5]), "total skal være np.cumsum(belonning)."
```
:::

## meshgrid: et rutenett av punkter
`xx, yy = np.meshgrid(xs, ys)` gir to matriser med shape `(len(ys), len(xs))`, og `np.column_stack((xx.ravel(), yy.ravel()))` gir alle kombinasjonene som punkter.

::: oppgave Rutenett
Lag `xx` og `yy` med `np.meshgrid(xs, ys)`, og `punkter` med alle 6 kombinasjonene som rader (shape `(6, 2)`).
```start
import numpy as np

xs = np.array([0, 1, 2])
ys = np.array([10, 20])

xx, yy = ..., ...
punkter = ...
```
```losning
import numpy as np

xs = np.array([0, 1, 2])
ys = np.array([10, 20])

xx, yy = np.meshgrid(xs, ys)
punkter = np.column_stack((xx.ravel(), yy.ravel()))
```
```test
import numpy as np
assert isinstance(xx, np.ndarray) and xx.shape == (2, 3), "xx skal ha shape (len(ys), len(xs)) = (2, 3): xx, yy = np.meshgrid(xs, ys)."
assert np.array_equal(xx, [[0, 1, 2], [0, 1, 2]]) and np.array_equal(yy, [[10, 10, 10], [20, 20, 20]]), "xx skal gi x-verdien og yy y-verdien i hvert punkt."
assert isinstance(punkter, np.ndarray) and punkter.shape == (6, 2), "punkter skal ha shape (6, 2): np.column_stack((xx.ravel(), yy.ravel()))."
assert {tuple(r) for r in punkter.tolist()} == {(x, y) for x in [0, 1, 2] for y in [10, 20]}, "punkter skal inneholde alle kombinasjonene av xs og ys."
```
:::

## nan og isclose
`np.nan` smitter (`x.mean()` blir `nan`) og `np.nan == np.nan` er `False`, så bruk `np.isnan(x)` og `np.nanmean(x)`. Flyttall sammenlignes med `np.isclose(a, b)` eller `np.allclose`, ikke `==`.

::: oppgave Rett nan-feilene
Tre av linjene gir feil svar. Rett dem, og lag `uten_nan` med bare tallene som ikke er `nan`.
```start
import numpy as np

x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])

snitt = x.mean()
antall_nan = (x == np.nan).sum()
uten_nan = ...
lik = 0.1 + 0.2 == 0.3
```
```losning
import numpy as np

x = np.array([1.0, np.nan, 3.0, np.nan, 5.0])

snitt = np.nanmean(x)
antall_nan = np.isnan(x).sum()
uten_nan = x[~np.isnan(x)]
lik = np.isclose(0.1 + 0.2, 0.3)
```
```test
import numpy as np
assert np.isclose(snitt, 3.0), f"snitt skal være 3.0, men er {snitt}. Bruk np.nanmean(x)."
assert antall_nan == 2, f"antall_nan skal være 2, men er {antall_nan}. nan == nan er False, så bruk np.isnan(x).sum()."
assert isinstance(uten_nan, np.ndarray) and np.allclose(uten_nan, [1.0, 3.0, 5.0]), "uten_nan skal være x[~np.isnan(x)]."
assert bool(lik) is True, "lik skal være True. Sammenlign flyttall med np.isclose(0.1 + 0.2, 0.3)."
```
:::
