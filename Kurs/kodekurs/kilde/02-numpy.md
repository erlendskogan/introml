=== DEL 2 | numpy | NumPy: regning med tabeller
NumPy er grunnmuren i all maskinlæringskode i Python. Her lærer du å lage, indeksere og regne med arrays, og å holde styr på `shape` og `axis`. Det er nøyaktig det du trenger for prosjektene og for kodelesingsspørsmålene på eksamen.

--- LEKSJON 2.1 | arrays | Arrays, shape og dtype
En Python-liste kan inneholde hva som helst, men den kan ikke regne. Skriver du `[1, 2, 3] * 2`, får du listen gjentatt to ganger. **NumPy** gir oss en ny datatype, `ndarray` (array), som er en tabell med tall der regning skjer element for element, og veldig raskt.

Alle importerer NumPy med kortnavnet `np`:

```python
import numpy as np

liste = [1, 2, 3]
x = np.array([1, 2, 3])   # lag et array fra en liste

print(liste * 2)   # listen gjentas
print(x * 2)       # hvert tall ganges med 2
print(type(x))
```

## Shape, ndim og size
Et array har en **form** (`shape`): hvor mange elementer det har langs hver akse. Et 1D-array er en rad med tall. Et 2D-array er en tabell med rader og kolonner, og lages fra en liste av lister.

```python
import numpy as np

x = np.array([1.0, 2.5, 4.0])
X = np.array([[1, 2, 3],
              [4, 5, 6]])

print("x.shape:", x.shape)   # (3,)   tre elementer, én akse
print("X.shape:", X.shape)   # (2, 3) to rader, tre kolonner
print("X.ndim:", X.ndim)     # antall akser
print("X.size:", X.size)     # antall elementer totalt
print("len(X):", len(X))     # antall rader
```

I variabelpanelet under resultatet vises arrays som **fargelagte rutenett**: én rute per tall, og fargen viser hvor stort tallet er. Klikk på `X` og se de to radene og tre kolonnene, med shape og dtype i overskriften.

Legg merke til at `shape` er en tuppel. For 1D skrives den `(3,)` med komma, fordi det er en tuppel med ett element.

::: husk Rader er datapunkter, kolonner er features
I maskinlæring lagres data nesten alltid som en 2D-matrise `X` med shape `(n_samples, n_features)`. Hver **rad** er ett datapunkt (én person, ett bilde), og hver **kolonne** er én feature (alder, en piksel). Målvariabelen `y` er et 1D-array med shape `(n_samples,)`. Denne konvensjonen bruker sklearn, PyTorch og alle prosjektene.
:::

```python
import numpy as np

# 4 datapunkter (rader) med 2 features (kolonner), og én label per datapunkt
X = np.array([[0.5, 1.2],
              [1.0, 0.3],
              [2.2, 2.0],
              [0.1, 0.9]])
y = np.array([0, 0, 1, 0])

n_samples, n_features = X.shape   # tuppel-utpakking fra del 1
print(n_samples, "datapunkter med", n_features, "features")
print("y.shape:", y.shape)
```

Se `X` og `y` i variabelpanelet: `X` er et rutenett med 4 rader og 2 kolonner, `y` én rad med 4 labels, én per rad i `X`.

## dtype: hvilken type tallene har
Alle elementene i et array har **samme type**, kalt `dtype`. Heltall blir et heltallsarray (`int64` på de fleste PC-er, nettleseren kan vise `int32`), og desimaltall blir `float64`. Blander du heltall og desimaltall, blir alt `float`. Med `.astype()` lager du en kopi med ny type.

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([1, 2.5, 3])     # ett desimaltall gjør alt til float
print(a.dtype, b.dtype)
print(b)

c = b.astype(int)             # desimalene kuttes bort
print(c, c.dtype)

d = np.array([0.0, 1.0, 1.0]).astype(bool)
print(d)
```

::: tips
Står et NumPy-tall alene på siste linje, viser kurset (som Jupyter) det som for eksempel `np.float64(2.5)`. Det er bare skrivemåten i NumPy 2. Med `print(...)` blir det `2.5`.
:::

::: oppgave Ditt første datasett
Lag et 2D-array `X` med 3 rader og 2 kolonner som inneholder tallene 1 til 6 (radvis: første rad `1, 2`). Pakk så ut shape i variablene `n_samples` og `n_features`.
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

::: quiz
Hva skriver koden ut?
```kode
import numpy as np
X = np.array([[1, 2, 3], [4, 5, 6]])
print(X.shape)
```
- [ ] `(3, 2)`
- [x] `(2, 3)`
- [ ] `(6,)`
- [ ] `6`
? Shape er `(antall rader, antall kolonner)`. Det er to indre lister (rader) med tre tall hver.
:::

::: quiz
Hva er `np.array([1, 2, 3]) * 2`?
- [ ] `[1, 2, 3, 1, 2, 3]`
- [x] `array([2, 4, 6])`
- [ ] `12`
- [ ] En `TypeError`
? Regning på arrays skjer **element for element**. Det er listen `[1, 2, 3] * 2` som gjentas.
:::

::: oppsummering
- `import numpy as np` og `np.array(liste)` lager et array.
- `.shape` gir formen som tuppel, `.ndim` antall akser, `.size` antall elementer, `len(X)` antall rader.
- Data lagres som `X` med shape `(n_samples, n_features)`: rader er datapunkter, kolonner er features.
- Alle elementer har samme `dtype`. `.astype(...)` konverterer.
:::

--- LEKSJON 2.2 | lage-arrays | Lage arrays
Ofte vil du lage et array uten å skrive alle tallene selv: en tabell med nuller som skal fylles inn, en rekke med tall fra 0 til 99, eller 50 jevnt fordelte punkter mellom to verdier. NumPy har egne funksjoner for alt dette.

## zeros, ones og full
`np.zeros(shape)` lager et array fylt med 0.0. Shape for 2D gis som en **tuppel**, med doble parenteser.

```python
import numpy as np

print(np.zeros(4))
print(np.ones((2, 3)))
print(np.full((2, 3), 0.5))     # fyll med en valgfri verdi
```

Dette er nøyaktig hvordan Q-tabellen i forsterkningslæring lages. Hver rad er en tilstand (state), og hver kolonne er en handling (action).

```python
import numpy as np

n_states = 16      # f.eks. et 4x4-rutenett
n_actions = 4      # opp, venstre, ned, høyre

q_table = np.zeros((n_states, n_actions))
print(q_table.shape)
print(q_table[:3])    # de tre første radene (slicing kommer i neste leksjon)
```

Klikk på `q_table` i variabelpanelet: 16 rader (én per tilstand) og 4 kolonner (én per handling), og alle verdiene er ennå 0. Under treningen fylles rutenettet gradvis med tall.

::: prosjekt Prosjekt 4
Prosjekt 4 lager Q-tabellen med `q_table = np.zeros((num_states, num_actions))`, og i den store oppgaven med `np.full((num_states, num_actions), Q_INIT)` for å starte alle Q-verdiene på en annen verdi enn null.
:::

`np.zeros_like(a)` og `np.full_like(a, verdi)` lager et nytt array med samme shape (og dtype) som `a`. Det er praktisk når du skal ha «et array av samme størrelse som dataene».

```python
import numpy as np

y = np.array([3.0, 1.0, 4.0, 1.0])
print(np.zeros_like(y))
print(np.full_like(y, 7.0))
```

## arange og linspace
`np.arange(start, stopp, steg)` virker som `range`, men gir et array og tillater desimalsteg. **Stopp er ikke med.** `np.linspace(start, stopp, antall)` gir et bestemt antall jevnt fordelte punkter, og her **er stopp med**.

```python
import numpy as np

print(np.arange(5))             # 0, 1, 2, 3, 4
print(np.arange(2, 10, 3))      # 2, 5, 8
print(np.arange(0, 1, 0.25))    # desimalsteg går fint
print(np.linspace(0, 1, 5))     # 5 punkter, både 0 og 1 er med
```

::: tips Når bruker du hva?
Bruk `arange` når du vet **steglengden** (epoker 1, 2, 3, …, eller et rutenett med steg 0.01). Bruk `linspace` når du vet **antall punkter** (100 x-verdier for å tegne en glatt kurve).
:::

::: oppgave Q-tabell og epoker
Lag
1. `q_table`: en Q-tabell fylt med nuller for 25 tilstander og 4 handlinger.
2. `epoker`: et array med tallene 1, 2, …, 100 (både 1 og 100 skal være med).
```start
import numpy as np

q_table = ...
epoker = ...
```
```losning
import numpy as np

q_table = np.zeros((25, 4))
epoker = np.arange(1, 101)
```
```test
import numpy as np
assert isinstance(q_table, np.ndarray), "q_table skal være et numpy-array, f.eks. fra np.zeros."
assert q_table.shape == (25, 4), f"q_table skal ha shape (25, 4) (tilstander, handlinger), men har {q_table.shape}."
assert np.all(q_table == 0), "Alle verdiene i q_table skal være 0."
assert isinstance(epoker, np.ndarray), "epoker skal være et numpy-array."
assert len(epoker) == 100, f"epoker skal ha 100 elementer, men har {len(epoker)}. Husk at stopp ikke er med i arange."
assert epoker[0] == 1 and epoker[-1] == 100, "epoker skal starte på 1 og slutte på 100."
```
:::

::: quiz
Hva skriver koden ut?
```kode
import numpy as np
print(np.arange(2, 10, 3))
```
- [ ] `[2 5 8 11]`
- [x] `[2 5 8]`
- [ ] `[ 2.  4.67  7.33 10.]`
- [ ] `[2 3 4 5 6 7 8 9]`
? `arange` starter på 2 og går med steg 3: 2, 5, 8. Neste ville vært 11, men stopp (10) er aldri med. Alternativet med fire desimaltall er det `np.linspace(2, 10, 4)` ville gitt.
:::

::: quiz
Hvilken linje lager en Q-tabell med 10 tilstander og 3 handlinger?
- [ ] `np.zeros(10, 3)`
- [x] `np.zeros((10, 3))`
- [ ] `np.zeros((3, 10))`
- [ ] `np.arange(10, 3)`
? Shape gis som én tuppel, `(rader, kolonner)`. Rader er tilstander. `np.zeros(10, 3)` krasjer fordi andre argument tolkes som `dtype`.
:::

::: oppsummering
- `np.zeros(shape)`, `np.ones(shape)`, `np.full(shape, verdi)`: shape for 2D er en tuppel `(rader, kolonner)`.
- `np.zeros_like(a)` / `np.full_like(a, v)` lager et array med samme shape som `a`.
- `np.arange(start, stopp, steg)`: stopp er **ikke** med.
- `np.linspace(start, stopp, n)`: n punkter, stopp **er** med.
- Q-tabellen: `np.zeros((n_states, n_actions))`.
:::

--- LEKSJON 2.3 | indeksering | Indeksering og slicing
Indeksering i NumPy virker som for lister: indeks starter på 0, negative indekser teller bakfra, og `start:stopp` gir en bit (slice) der stopp ikke er med. Det nye er at 2D-arrays tar **én indeks per akse**, skilt med komma: `X[rad, kolonne]`.

## 1D-arrays

```python
import numpy as np

x = np.array([10, 20, 30, 40, 50])
print(x[0])      # første
print(x[-1])     # siste
print(x[1:4])    # indeks 1, 2, 3
print(x[:3])     # de tre første
print(x[::2])    # annethvert element
```

## 2D-arrays: rader og kolonner
Med `X[i]` får du rad `i`, altså datapunkt `i`. Med `X[:, j]` får du kolonne `j`, altså feature `j` for alle datapunktene. Kolon `:` betyr «alt langs denne aksen».

```python
import numpy as np

X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])

print("X[1]      =", X[1])         # rad 1 (datapunkt 1)
print("X[1, 2]   =", X[1, 2])      # rad 1, kolonne 2
print("X[:, 0]   =", X[:, 0])      # kolonne 0 (feature 0) for alle rader
print("X[:2]     =\n", X[:2])      # de to første radene
print("X[:, 1:]  =\n", X[:, 1:])   # alle rader, kolonne 1 og utover
print("X[-1, -1] =", X[-1, -1])    # nederst til høyre
```

::: husk Shape etter indeksering
Med X med shape `(n, d)`:
- `X[i]` gir shape `(d,)`: én rad.
- `X[:, j]` gir shape `(n,)`: én kolonne, som et **1D**-array.
- `X[:5]` gir shape `(5, d)`: de fem første radene, fortsatt 2D.
- `X[:, [j]]` (liste med én indeks) gir shape `(n, 1)`: kolonnen beholdes som 2D.
:::

```python
import numpy as np

X = np.zeros((100, 3))
print(X[0].shape)
print(X[:, 1].shape)
print(X[:5].shape)
print(X[:, [1]].shape)
print(X[:, [0, 2]].shape)   # en liste med indekser velger flere kolonner
```

I Q-learning er `q_table[state]` raden med Q-verdiene for alle handlingene i én tilstand, og `q_table[state, action]` er ett enkelt tall som kan oppdateres:

```python
import numpy as np

q_table = np.zeros((5, 4))
print("Før:")
display(q_table)

state, action = 2, 1
q_table[state, action] += 0.5     # endre ett element
q_table[4] = [1, 2, 3, 4]         # endre en hel rad
print("Etter:")
display(q_table)
print(q_table[state])
```

`display(x)` viser arrayet som et rutenett midt i utdataene, akkurat som i Jupyter. Sammenlign de to rutenettene: bare ruten `[2, 1]` og hele rad 4 har fått nye verdier.

## Slicing gir en «view», ikke en kopi
En slice av et array deler minne med originalen. Endrer du slicen, endres originalen også. Vil du ha en uavhengig kopi, bruk `.copy()`.

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5])
del_av_x = x[:3]
del_av_x[0] = 99
print(x)             # originalen er endret!

y = np.array([1, 2, 3, 4, 5])
kopi = y[:3].copy()
kopi[0] = 99
print(y)             # originalen er urørt
```

Se `x` og `del_av_x` i variabelpanelet: 99 står i begge, fordi de deler minne.

::: advarsel
Dette er annerledes enn for lister, der `liste[:3]` alltid gir en kopi. Prosjekt 4 bruker `.copy()` når den henter ut en kolonne av Q-tabellen før den endrer den, nettopp for å ikke ødelegge selve tabellen.
:::

::: oppgave Plukk ut data
`X` har 6 datapunkter og 3 features. Lag
1. `alder`: kolonne 0 for alle datapunktene (1D).
2. `punkt3`: hele rad 3.
3. `forste_fire`: de fire første radene.
4. `verdi`: elementet i rad 2, kolonne 1.
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

::: quiz
`X` har shape `(100, 3)`. Hva er `X[:, 1].shape`?
- [ ] `(100, 3)`
- [ ] `(100, 1)`
- [x] `(100,)`
- [ ] `(3,)`
? `X[:, 1]` tar kolonne 1 fra alle 100 radene og gir et 1D-array. For å beholde 2D-formen `(100, 1)` må du skrive `X[:, [1]]` eller `X[:, 1:2]`. `(3,)` er shape for en rad, `X[1]`.
:::

::: quiz
Hva skriver koden ut?
```kode
import numpy as np
a = np.array([1, 2, 3, 4])
b = a[1:3]
b[0] = 0
print(a)
```
- [ ] `[1 2 3 4]`
- [x] `[1 0 3 4]`
- [ ] `[0 2 3 4]`
- [ ] `[1 0 0 4]`
? `b = a[1:3]` er en view av `a` (elementene på indeks 1 og 2). `b[0]` er altså det samme minnet som `a[1]`, så `a` endres.
:::

::: oppsummering
- 1D: `x[0]`, `x[-1]`, `x[1:4]`, `x[::2]`, som for lister.
- 2D: `X[rad, kolonne]`. `X[i]` er en rad (datapunkt), `X[:, j]` er en kolonne (feature), `X[:5]` er de fem første radene.
- `X[:, j]` gir 1D, `X[:, [j]]` beholder 2D.
- Slicing gir en **view**: endringer slår gjennom til originalen. Bruk `.copy()` for en uavhengig kopi.
:::

--- LEKSJON 2.4 | vektorisering | Vektoriserte operasjoner og matematiske funksjoner
I ren Python må du bruke en løkke for å gjøre noe med hvert tall i en liste. I NumPy skriver du operasjonen på hele arrayet på én gang. Det kalles **vektorisering**. Koden blir kortere, ligner mer på matematikken, og er mange ganger raskere.

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([10.0, 20.0, 30.0, 40.0])

print(x + y)      # element for element
print(y - x)
print(x * y)      # OBS: element for element, ikke matriseprodukt
print(y / x)
print(x ** 2)
print(2 * x + 1)  # tall og array går fint sammen
```

## Løkke mot vektorisering
De to måtene under regner ut det samme. Den vektoriserte er både kortere og mye raskere, og forskjellen vokser med datamengden.

```python
import numpy as np
import time

x = np.arange(200_000, dtype=float)

t0 = time.perf_counter()
resultat = []
for verdi in x:
    resultat.append(verdi ** 2 + 1)
t_lokke = time.perf_counter() - t0

t0 = time.perf_counter()
resultat_np = x ** 2 + 1
t_numpy = time.perf_counter() - t0

print(f"Løkke:  {t_lokke * 1000:.1f} ms")
print(f"NumPy:  {t_numpy * 1000:.2f} ms")
print("Samme svar:", np.allclose(resultat, resultat_np))
```

## Matematiske funksjoner
NumPy har sine egne versjoner av matematiske funksjoner som virker på hele arrays: `np.exp`, `np.log` (naturlig logaritme), `np.sqrt`, `np.abs`, `np.sin` og så videre.

```python
import numpy as np

z = np.array([-2.0, 0.0, 2.0])
print(np.exp(z))
print(np.log(np.array([1.0, np.e, 10.0])))
print(np.sqrt(np.array([4.0, 9.0, 2.0])))
print(np.abs(z))
```

Med disse kan du skrive formlene fra maskinlæring nesten rett av. Her er sigmoid-funksjonen $\sigma(z) = \frac{1}{1 + e^{-z}}$ og gjennomsnittlig kvadratfeil (MSE) $\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2$:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
print(sigmoid(z))          # alle verdiene havner mellom 0 og 1

y = np.array([3.0, 5.0, 2.0])
y_pred = np.array([2.5, 5.0, 3.0])
mse = np.mean((y - y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y - y_pred))
print(f"MSE={mse:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")
```

`np.mean` regner gjennomsnittet. Mer om slike funksjoner i leksjon 2.6.

::: advarsel math virker ikke på arrays
Funksjonene i `math`-modulen (`math.exp`, `math.log`) tar bare ett tall. Gir du dem et array, får du en feilmelding:
```python feil
import math
import numpy as np
z = np.array([1.0, 2.0])
print(math.exp(z))
```
Bruk alltid `np.exp`, `np.log` osv. på arrays.
:::

::: prosjekt Prosjekt 1
I prosjekt 1 skal du skrive sigmoid, MSE og cross-entropy med nøyaktig disse funksjonene. Ingen løkker over datapunktene trengs.
:::

::: oppgave Sigmoid og MSE
1. Skriv funksjonen `sigmoid(z)` som returnerer $1/(1+e^{-z})$ og virker på hele arrays.
2. Regn ut `mse` mellom `y` og `y_pred` uten løkke.
```start
import numpy as np

def sigmoid(z):
    return ...

y = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = np.array([1.5, 2.0, 2.0, 5.0])
mse = ...
```
```losning
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

y = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = np.array([1.5, 2.0, 2.0, 5.0])
mse = np.mean((y - y_pred) ** 2)
```
```test
import numpy as np
s = sigmoid(np.array([-1.0, 0.0, 2.0]))
assert isinstance(s, np.ndarray) and s.shape == (3,), "sigmoid skal returnere et array med samme shape som inndataene. Bruk np.exp, ikke math.exp."
assert np.allclose(s, [0.26894142, 0.5, 0.88079708]), "sigmoid gir feil verdier. Formelen er 1 / (1 + np.exp(-z))."
assert mse is not ... and np.isclose(mse, 0.5625), f"mse skal være 0.5625, men er {mse}. Ta gjennomsnittet av (y - y_pred) ** 2."
```
:::

::: quiz
Hva skriver koden ut?
```kode
import numpy as np
x = np.array([1, 2, 3])
print(x * x)
```
- [ ] `14`
- [x] `[1 4 9]`
- [ ] `[1 2 3 1 2 3]`
- [ ] `[[1 2 3] [2 4 6] [3 6 9]]`
? `*` er element for element på arrays. 14 ville vært prikkproduktet (`x @ x`), som kommer i leksjon 2.9.
:::

::: oppsummering
- Operatorene `+ - * / **` virker element for element, også mellom tall og array.
- Vektorisert kode er kortere og mye raskere enn løkker.
- Bruk `np.exp`, `np.log`, `np.sqrt`, `np.abs` på arrays (ikke `math`).
- Sigmoid: `1 / (1 + np.exp(-z))`. MSE: `np.mean((y - y_pred) ** 2)`.
:::

--- LEKSJON 2.5 | broadcasting | Broadcasting
Når du skriver `x * 2`, «strekkes» tallet 2 ut så det passer til hvert element i `x`. Det samme skjer mellom arrays med ulik shape, så lenge formene passer sammen. Det kalles **broadcasting**, og det er grunnen til at skalering av en hel datamatrise kan skrives på én linje.

## Rad-vektor mot matrise
Har `X` shape `(n, d)` og `v` shape `(d,)`, så legges `v` til **hver rad** i `X`. Det er nøyaktig det vi trenger for å behandle hver feature (kolonne) for seg.

```python
import numpy as np

X = np.array([[1, 10, 100],
              [2, 20, 200],
              [3, 30, 300]])
v = np.array([1, 2, 3])

print(X.shape, v.shape)
print(X - v)        # v trekkes fra hver rad
```

Se `X` og `v` i variabelpanelet: `v` er én rad med tre tall, og den samme raden trekkes fra hver av de tre radene i `X`.

## Sentrering og standardisering
`X.mean(axis=0)` gir gjennomsnittet av hver kolonne, med shape `(d,)` (`axis` forklares grundig i neste leksjon). Trekker du det fra `X`, blir hver feature sentrert rundt 0. Deler du i tillegg på standardavviket, får hver feature standardavvik 1. Dette er **standardisering**, som `StandardScaler` i sklearn gjør.

```python
import numpy as np

X = np.array([[170.0, 60.0],
              [180.0, 80.0],
              [160.0, 55.0],
              [190.0, 95.0]])

mu = X.mean(axis=0)       # snitt per feature, shape (2,)
sigma = X.std(axis=0)     # standardavvik per feature, shape (2,)
print("mu:", mu, " sigma:", sigma)

X_sentrert = X - mu
print("Før sentrering:")
display(X)
print("Etter X - mu: hver kolonne har snitt 0")
display(X_sentrert)

X_std = (X - mu) / sigma
print(X_std)
print("Nytt snitt:", X_std.mean(axis=0).round(6))
print("Nytt std:  ", X_std.std(axis=0))
```

::: prosjekt Prosjekt 1
Gradient descent konvergerer dårlig når features har svært ulik skala. Prosjekt 1 og leksjon 5.9 bruker akkurat `(X - mu) / sigma` for å fikse det.
:::

## Regelen for broadcasting
NumPy sammenligner formene **bakfra** (fra siste akse). To akser passer hvis de er **like**, eller hvis en av dem er **1** (eller mangler). Da strekkes 1-aksen ut.

| A | B | Resultat |
|---|---|---|
| `(4, 3)` | `(3,)` | `(4, 3)`: B brukes på hver rad |
| `(4, 3)` | `(4, 1)` | `(4, 3)`: B brukes på hver kolonne |
| `(3, 1)` | `(1, 4)` | `(3, 4)`: begge strekkes |
| `(4, 3)` | `(4,)` | **feil**: 3 og 4 passer ikke |

Den siste raden er en vanlig felle. Vil du trekke et tall per **rad** fra en `(4, 3)`-matrise, må tallene ligge i en kolonnevektor med shape `(4, 1)`:

```python feil
import numpy as np

X = np.ones((4, 3))
rad_snitt = np.array([1.0, 2.0, 3.0, 4.0])   # shape (4,)
print(X - rad_snitt)
```

```python
import numpy as np

X = np.ones((4, 3))
rad_snitt = np.array([[1.0], [2.0], [3.0], [4.0]])   # shape (4, 1)
print(X - rad_snitt)

# (3, 1) og (1, 4) gir en (3, 4)-tabell: alle kombinasjoner
a = np.array([[1], [2], [3]])
b = np.array([[10, 20, 30, 40]])
print(a * b)
```

I leksjon 2.8 lærer du å lage kolonnevektorer enklere med `reshape(-1, 1)` og `np.newaxis`.

::: oppgave Standardiser features
Standardiser `X` så hver kolonne får snitt 0 og standardavvik 1. Lagre snittene i `mu`, standardavvikene i `sigma` og resultatet i `X_std`. Ingen løkker.
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

::: quiz
`X` har shape `(5, 3)`. Hvilken linje gir en **feilmelding**?
- [ ] `X - X.mean(axis=0)`  (snittet har shape `(3,)`)
- [x] `X - X.mean(axis=1)`  (snittet har shape `(5,)`)
- [ ] `X - X.mean()`  (ett tall)
- [ ] `X * np.ones((1, 3))`
? Formene sammenlignes bakfra: `(5, 3)` mot `(5,)` sammenligner 3 med 5, som ikke passer. `(3,)` passer med siste akse, et enkelt tall passer alltid, og `(1, 3)` strekkes over radene.
:::

::: oppsummering
- Broadcasting lar arrays med ulik shape regnes sammen ved å strekke akser med lengde 1.
- Formene sammenlignes bakfra: aksene må være like, eller en av dem må være 1.
- `(n, d)` og `(d,)`: samme operasjon på hver rad, altså per feature.
- Standardisering: `(X - X.mean(axis=0)) / X.std(axis=0)`.
- For noe per rad trenger du en kolonnevektor med shape `(n, 1)`.
:::

--- LEKSJON 2.6 | aggregering | Aggregering og axis
Funksjoner som `sum`, `mean`, `std`, `min`, `max`, `argmin` og `argmax` slår sammen mange tall til færre. På 2D-arrays bestemmer argumentet `axis` **hvilken retning** du slår sammen i. Dette er en klassisk eksamensfelle, så ta deg tid her.

## På 1D-arrays

```python
import numpy as np

x = np.array([4.0, 9.0, 1.0, 7.0])
print("sum:", x.sum(), " snitt:", x.mean(), " std:", x.std())
print("min:", x.min(), " max:", x.max())
print("argmin:", x.argmin(), " argmax:", x.argmax())   # INDEKSEN til minste/største
print(np.sum(x), np.max(x), np.argmax(x))              # funksjonsform, samme svar
```

`argmax` gir **posisjonen** til det største elementet, ikke verdien. Er flere like store, får du den **første**.

## axis på 2D-arrays
Tenk at `axis=k` betyr at **akse k forsvinner**:
- `axis=0`: du går nedover radene og får **ett tall per kolonne**. Shape `(n, d)` blir `(d,)`.
- `axis=1`: du går bortover kolonnene og får **ett tall per rad**. Shape `(n, d)` blir `(n,)`.
- Ingen axis: alle elementene slås sammen til ett tall.

```python
import numpy as np

X = np.array([[1, 2, 3],
              [4, 5, 6]])     # shape (2, 3)

print("X.sum()       =", X.sum())
print("X.sum(axis=0) =", X.sum(axis=0), " shape", X.sum(axis=0).shape)
print("X.sum(axis=1) =", X.sum(axis=1), " shape", X.sum(axis=1).shape)
```

Det hjelper å se det. Her vises `X` og de to summene med `display()`:

```python
import numpy as np

X = np.array([[1, 2, 3],
              [4, 5, 6]])
display(X)
print("axis=0: radene slås sammen, ett tall per kolonne")
display(X.sum(axis=0))
print("axis=1: kolonnene slås sammen, ett tall per rad")
display(X.sum(axis=1))
```

Begge resultatene er 1D-arrays og vises derfor som én rad, men lengden avslører retningen: 3 tall (én per kolonne) mot 2 tall (én per rad).

| | `axis=0` | `axis=1` |
|---|---|---|
| Retning | nedover (langs radene) | bortover (langs kolonnene) |
| Resultat | ett tall per **kolonne** | ett tall per **rad** |
| For `X` med shape `(n_samples, n_features)` | per **feature** (f.eks. snitt av hver feature) | per **datapunkt** |
| For `q_table` med shape `(n_states, n_actions)` | per handling | per **tilstand** |

::: eksamen
Spørsmål om `axis` dukker ofte opp i kodelesing: «`X` har shape (100, 4). Hva er shapen til `X.mean(axis=0)`?» Svaret er `(4,)`, fordi akse 0 (de 100 radene) forsvinner. Bruk regelen «akse k forsvinner fra shape».
:::

## Q-tabellen: argmax og max per tilstand
I Q-learning er hver rad en tilstand og hver kolonne en handling. Tre uttrykk går igjen:

```python
import numpy as np

q_table = np.array([[0.1, 0.5, 0.2, 0.0],    # tilstand 0
                    [0.9, 0.3, 0.4, 0.8],    # tilstand 1
                    [0.0, 0.0, 0.7, 0.7]])   # tilstand 2

state = 1
print("Beste handling i tilstand 1:", np.argmax(q_table[state]))
print("Beste Q-verdi i tilstand 1: ", np.max(q_table[state]))

print("Beste verdi per tilstand:    ", q_table.max(axis=1))     # V(s)
print("Beste handling per tilstand: ", q_table.argmax(axis=1))  # grådig policy
```

Klikk på `q_table` i variabelpanelet og finn den største verdien i hver rad. Kolonnenummeret til den er det `argmax(axis=1)` gir.

Legg merke til tilstand 2: handling 2 og 3 er like gode, og `argmax` velger den første (2). En helt ny Q-tabell med bare nuller gir derfor alltid handling 0.

::: prosjekt Prosjekt 4
Prosjekt 4 bruker alle tre: `int(np.argmax(q_table[state]))` for å velge handling, `np.max(q_table[next_state])` i Q-oppdateringen, og `q_table.max(axis=1).reshape(nrows, ncols)` til heatmapen. `q_table.argmax(axis=1).reshape(...)` viser den lærte policyen som et kart.
:::

Andre nyttige aggregeringer: `np.sum(betingelse)` teller hvor mange som er sanne (mer i neste leksjon), og `keepdims=True` beholder aksen med lengde 1, som er nyttig for broadcasting.

```python
import numpy as np

rewards = np.array([10, -1, 10, 10, -5, 10])
print("Antall episoder med mål nådd:", np.sum(rewards == 10))

X = np.array([[1.0, 3.0], [2.0, 6.0]])
rad_snitt = X.mean(axis=1, keepdims=True)   # shape (2, 1) i stedet for (2,)
print(rad_snitt.shape)
print(X - rad_snitt)                         # trekker radsnittet fra hver rad
```

::: oppgave Les av Q-tabellen
Bruk `q_table` til å lage
1. `policy`: beste handling for hver tilstand (1D-array med lengde 4).
2. `V`: høyeste Q-verdi for hver tilstand.
3. `snitt_per_handling`: gjennomsnittlig Q-verdi for hver handling (lengde 3).
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

::: quiz
`X` er et datasett med shape `(100, 4)` (100 datapunkter, 4 features). Hva er `X.mean(axis=0).shape`?
- [ ] `(100,)`
- [x] `(4,)`
- [ ] `(100, 4)`
- [ ] `()` (ett tall)
? Akse 0 (de 100 radene) slås sammen og forsvinner. Du får snittet av hver feature, altså 4 tall. `(100,)` ville vært `axis=1`, snittet per datapunkt.
:::

::: quiz
Hva skriver koden ut?
```kode
import numpy as np
q_table = np.array([[0.2, 0.8],
                    [0.5, 0.1],
                    [0.3, 0.3]])
print(q_table.argmax(axis=1))
```
- [ ] `[0.8 0.5 0.3]`
- [x] `[1 0 0]`
- [ ] `[1 0]`
- [ ] `[1 0 1]`
? `argmax(axis=1)` gir indeksen til største verdi i hver rad. Rad 0: 0.8 på indeks 1. Rad 1: 0.5 på indeks 0. Rad 2: likt, og argmax velger første, indeks 0. `[0.8 0.5 0.3]` er `max(axis=1)`, og `[1 0]` ville vært `argmax(axis=0)`.
:::

::: oppsummering
- `sum, mean, std, min, max` gir verdier, `argmin, argmax` gir **indekser** (første ved likhet).
- `axis=k`: akse k forsvinner. `axis=0` gir ett tall per kolonne, `axis=1` ett tall per rad.
- For `X` (n_samples, n_features): `X.mean(axis=0)` er snitt per feature.
- Q-tabell: `np.argmax(q_table[state])` velger handling, `q_table.max(axis=1)` er beste verdi per tilstand, `q_table.argmax(axis=1)` er policyen.
- `keepdims=True` beholder aksen som lengde 1.
:::

--- LEKSJON 2.7 | boolsk | Boolsk indeksering og np.where
Sammenligner du et array med noe, får du et nytt array med `True` og `False`, ett for hvert element. Et slikt array kalles en **boolsk maske**. Masker brukes til å velge ut data («alle punkter i klasse 1»), telle, og regne ut nøyaktighet.

```python
import numpy as np

x = np.array([3, -1, 4, -1, 5])
maske = x > 0
print(maske)
print(x[maske])           # bare elementene der masken er True
print(x[x < 0])           # vanligvis skrevet direkte
print(maske.sum())        # True teller som 1, False som 0
```

Klikk på `maske` i variabelpanelet: boolske arrays vises med grønt for `True` og grått for `False`, så du ser med en gang hvilke posisjoner som blir valgt.

## Velge rader etter label
Med `X[y == 1]` får du alle radene der labelen er 1. Masken har én verdi per rad.

```python
import numpy as np

X = np.array([[1.0, 2.0],
              [1.5, 1.8],
              [5.0, 8.0],
              [6.0, 9.0]])
y = np.array([0, 0, 1, 1])

print("Alle radene:")
display(X)
print("X[y == 1]: bare radene der y er 1")
display(X[y == 1])
print("Snitt av klasse 0:", X[y == 0].mean(axis=0))
```

## Accuracy på én linje
`y_pred == y` gir `True` der prediksjonen er riktig. Gjennomsnittet av en boolsk maske er andelen `True`, altså **nøyaktigheten** (accuracy).

```python
import numpy as np

y = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])

riktig = y_pred == y
print(riktig)
print("Accuracy:", riktig.mean())
print("Antall feil:", np.sum(y_pred != y))
```

## Kombinere betingelser
Bruk `&` (og), `|` (eller) og `~` (ikke). Sett **parenteser** rundt hver sammenligning, fordi `&` og `|` binder sterkere enn `<` og `>`. Python-ordene `and`/`or` virker ikke på arrays:

```python feil
import numpy as np
x = np.array([1, 5, 10])
print(x[x > 2 and x < 8])
```

```python
import numpy as np

x = np.array([1, 5, 10, 3, 7])
print(x[(x > 2) & (x < 8)])
print(x[(x < 2) | (x > 8)])
print(x[~(x > 4)])
```

## np.where og np.isin
`np.where(betingelse, a, b)` velger `a` der betingelsen er sann og `b` ellers. Det er standardmåten å gjøre sannsynligheter om til klasser. Med bare betingelsen, `np.where(betingelse)`, får du **indeksene** der den er sann.

```python
import numpy as np

p = np.array([0.1, 0.8, 0.55, 0.3, 0.97])
y_pred = np.where(p >= 0.5, 1, 0)
print(y_pred)
print((p >= 0.5).astype(int))     # samme resultat

print(np.where(p > 0.7))          # tuppel med indekser
print(np.where(p > 0.7)[0])
```

`np.isin(a, verdier)` sjekker for hvert element om det finnes i en liste med verdier. Prosjekt 4 kombinerer den med `np.where` for å blanke ut ruter (vann og mål) i Q-heatmapen, ved å sette dem til `np.nan` (mer om `nan` i leksjon 2.11):

```python
import numpy as np

kart = np.array([[0, 0, 2],
                 [1, 0, 3],
                 [1, 1, 0]])        # 0 = land, 1 = vann, 2 = start, 3 = mål
verdier = np.array([[0.5, 0.6, 0.4],
                    [0.9, 0.8, 1.0],
                    [0.2, 0.1, 0.7]])

blank = np.isin(kart, [1, 3])        # vann eller mål
print("Masken:")
display(blank)
print("np.where(blank, np.nan, verdier):")
display(np.where(blank, np.nan, verdier))
```

Rutene der masken er `True` (grønne) har blitt `nan` i resultatet. Det er slike ruter som blir tomme i heatmapen i prosjekt 4.

::: eksamen
`(y_pred == y).mean()` og `np.where(p >= 0.5, 1, 0)` er så vanlige at de ofte står uten forklaring i kodespørsmål. Kjenn dem igjen: den første er accuracy, den andre gjør sannsynligheter om til klasselabels.
:::

::: oppgave Fra sannsynligheter til accuracy
Modellen har gitt sannsynlighetene `p` for klasse 1.
1. Lag `y_pred` med 1 der `p >= 0.5` og 0 ellers.
2. Regn ut `accuracy` mot de sanne labelene `y`.
3. Lag `X_pos` som inneholder radene i `X` der den **sanne** labelen er 1.
```start
import numpy as np

X = np.array([[0.1, 1.0], [0.4, 2.0], [0.9, 3.0], [0.6, 4.0], [0.2, 5.0]])
y = np.array([0, 1, 1, 0, 0])
p = np.array([0.2, 0.6, 0.9, 0.7, 0.1])

y_pred = ...
accuracy = ...
X_pos = ...
```
```losning
import numpy as np

X = np.array([[0.1, 1.0], [0.4, 2.0], [0.9, 3.0], [0.6, 4.0], [0.2, 5.0]])
y = np.array([0, 1, 1, 0, 0])
p = np.array([0.2, 0.6, 0.9, 0.7, 0.1])

y_pred = np.where(p >= 0.5, 1, 0)
accuracy = (y_pred == y).mean()
X_pos = X[y == 1]
```
```test
import numpy as np
assert isinstance(y_pred, np.ndarray) and np.array_equal(y_pred, [0, 1, 1, 1, 0]), "y_pred skal være 1 der p >= 0.5 og 0 ellers, f.eks. med np.where."
assert accuracy is not ... and np.isclose(accuracy, 0.8), f"accuracy skal være 0.8 (4 av 5 riktige), men er {accuracy}."
assert isinstance(X_pos, np.ndarray) and X_pos.shape == (2, 2), "X_pos skal være de 2 radene i X der y == 1."
assert np.allclose(X_pos, [[0.4, 2.0], [0.9, 3.0]]), "X_pos skal velges med den sanne labelen y, ikke y_pred."
```
:::

::: quiz
Hva skriver koden ut?
```kode
import numpy as np
y      = np.array([1, 0, 1, 1])
y_pred = np.array([1, 1, 1, 0])
print((y_pred == y).mean())
```
- [ ] `2`
- [x] `0.5`
- [ ] `0.75`
- [ ] `[True False True False]`
? Maska er `[True, False, True, False]`. Gjennomsnittet av den er 2/4 = 0.5, altså 50 % accuracy.
:::

::: quiz
Hva inneholder `y_pred` etter denne koden?
```kode
p = np.array([0.3, 0.5, 0.9])
y_pred = np.where(p > 0.5, "ja", "nei")
```
- [ ] `["nei", "ja", "ja"]`
- [x] `["nei", "nei", "ja"]`
- [ ] `[0, 2]`
- [ ] `[False, False, True]`
? Betingelsen er streng: `0.5 > 0.5` er `False`, så det midterste elementet blir `"nei"`. `[0, 2]` ligner det `np.where(p > 0.5)` (uten a og b) gir, men den gir bare indeks 2.
:::

::: oppsummering
- Sammenligninger gir boolske masker. `x[maske]` velger ut elementene der masken er `True`.
- `X[y == 1]` gir radene i klasse 1. `maske.sum()` teller, `maske.mean()` gir andelen.
- Accuracy: `(y_pred == y).mean()`.
- Kombiner med `&`, `|`, `~` og parenteser, ikke `and`/`or`.
- `np.where(betingelse, a, b)` velger element for element. `np.isin(a, verdier)` sjekker medlemskap.
:::

--- LEKSJON 2.8 | form | Form: reshape, ravel, .T og stacking
Samme tall kan ordnes i ulike former. Et bilde på 8×8 piksler kan lagres som en rad med 64 tall (for en modell) eller som en 8×8-tabell (for å vise det). Denne leksjonen handler om å endre form og om å sette arrays sammen.

## reshape
`a.reshape(rader, kolonner)` gir de samme tallene i ny form. Antall elementer må stemme. Skriver du `-1` for én av aksene, regner NumPy ut lengden selv.

```python
import numpy as np

a = np.arange(12)
print(a.reshape(3, 4))
print(a.reshape(4, -1).shape)     # -1 blir 3
print(a.reshape(2, 2, 3).shape)   # 3D går også

print("Før reshape:")
display(a)
print("Etter reshape(3, 4):")
display(a.reshape(3, 4))
```

Tallene fylles inn **rad for rad**, og det ser du på fargene: de minste tallene havner øverst til venstre og de største nederst til høyre. Det er derfor `bilde.reshape(28, 28)` gjør en rad med 784 piksler fra MNIST om til et bilde.

## reshape(-1, 1): sklearn vil ha 2D
sklearn krever at `X` er 2D med shape `(n_samples, n_features)`, også når du bare har én feature. Et 1D-array må gjøres om til en kolonne med `reshape(-1, 1)`:

```python feil
import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([2.1, 3.9, 6.2, 7.8])
LinearRegression().fit(x, y)
```

```python
import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([2.1, 3.9, 6.2, 7.8])
X = x.reshape(-1, 1)
print(x.shape, "->", X.shape)
modell = LinearRegression().fit(X, y)
print(modell.coef_, modell.intercept_)
```

(Modellen selv forklares i del 5 og 7. Poenget her er formen.) Se `x` og `X` i variabelpanelet: samme fire tall, men `x` står som én rad og `X` som én kolonne.

## ravel og np.newaxis
`ravel()` gjør hva som helst om til 1D, rad for rad (`flatten()` gjør det samme, men lager alltid en kopi). `np.newaxis` (eller `None`) i en indeks legger til en akse med lengde 1.

```python
import numpy as np

M = np.array([[1, 2, 3],
              [4, 5, 6]])
print(M.ravel())
display(M)
display(M.ravel())               # rad 0 først, så rad 1

x = np.array([1, 2, 3])
print(x[:, np.newaxis].shape)    # (3, 1): kolonnevektor, som reshape(-1, 1)
print(x[np.newaxis, :].shape)    # (1, 3): radvektor
print(x[:, None].shape)          # None er det samme som np.newaxis
```

## Transponering med .T
`.T` bytter rader og kolonner: shape `(n, d)` blir `(d, n)`.

```python
import numpy as np

X = np.array([[1, 2],
              [3, 4],
              [5, 6]])
print(X.T)
print(X.shape, "->", X.T.shape)
display(X)
display(X.T)                      # kolonnene i X er blitt rader

# Pakk ut kolonnene: X.T har én rad per feature
x0, x1 = X.T
print("x0:", x0, " x1:", x1)
```

Utpakkingen `x, y, z = punkter.T` brukes i forelesningsnotatene for å hente ut hver koordinat fra en `(n, 3)`-matrise.

## Sette sammen arrays
- `np.vstack((a, b))`: stable **vertikalt**, nye rader.
- `np.hstack((a, b))`: stable **horisontalt**, nye kolonner (for 1D: skjøte etter hverandre).
- `np.column_stack((x0, x1))`: gjør 1D-arrays til **kolonner** i en matrise. Perfekt for å lage `X`.
- `np.concatenate((a, b), axis=0)`: den generelle varianten, du velger akse selv.

```python
import numpy as np

x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([1.6, 4.2, 5.7, 8.4, 9.2])

print(np.vstack((x1, x2)).shape)          # (2, 5): to rader
X = np.vstack((x1, x2)).T                 # fra forelesningsnotatene (PCA-eksempelet)
print(X.shape)
print(np.allclose(X, np.column_stack((x1, x2))))   # samme som column_stack

A = np.ones((2, 3))
B = np.zeros((2, 3))
print(np.concatenate((A, B), axis=0).shape)   # (4, 3): flere rader
print(np.concatenate((A, B), axis=1).shape)   # (2, 6): flere kolonner
print(np.hstack((x1, x1)))                    # 1D: skjøtes sammen
```

::: tips Legge til en kolonne med enere
Et klassisk triks er å legge til en kolonne med 1-ere, så biasen $b$ kan behandles som en vekt: `X_b = np.column_stack((np.ones(len(X)), X))`. Det samme brukes når du lager nye features, f.eks. `np.column_stack((x0, x1, x0 * x1))` for XOR-problemet i prosjekt 1.
:::

::: oppgave Lag X fra kolonner
1. Sett sammen `alder` og `hoyde` til en matrise `X` med shape `(4, 2)`, der hver rad er én person.
2. Lag `x_kol`: `alder` som en kolonnevektor med shape `(4, 1)`.
3. Lag `bilde`: `piksler` (16 tall) som et 4×4-bilde.
```start
import numpy as np

alder = np.array([23, 35, 41, 19])
hoyde = np.array([180, 165, 172, 190])
piksler = np.arange(16)

X = ...
x_kol = ...
bilde = ...
```
```losning
import numpy as np

alder = np.array([23, 35, 41, 19])
hoyde = np.array([180, 165, 172, 190])
piksler = np.arange(16)

X = np.column_stack((alder, hoyde))
x_kol = alder.reshape(-1, 1)
bilde = piksler.reshape(4, 4)
```
```test
import numpy as np
assert isinstance(X, np.ndarray) and X.shape == (4, 2), "X skal ha shape (4, 2). Tips: np.column_stack((alder, hoyde)) eller np.vstack((alder, hoyde)).T"
assert np.array_equal(X[:, 0], alder) and np.array_equal(X[:, 1], hoyde), "Første kolonne i X skal være alder, andre skal være hoyde."
assert isinstance(x_kol, np.ndarray) and x_kol.shape == (4, 1), "x_kol skal ha shape (4, 1). Bruk alder.reshape(-1, 1)."
assert isinstance(bilde, np.ndarray) and bilde.shape == (4, 4) and bilde[1, 0] == 4, "bilde skal være piksler.reshape(4, 4)."
```
:::

::: quiz
Hva skriver koden ut?
```kode
import numpy as np
x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([6, 7, 8, 9, 10])
print(np.vstack((x1, x2)).shape)
```
- [ ] `(5, 2)`
- [x] `(2, 5)`
- [ ] `(10,)`
- [ ] `(1, 10)`
? `vstack` legger arrayene oppå hverandre som rader: 2 rader med 5 tall. For å få datapunkter som rader, `(5, 2)`, må du transponere med `.T` eller bruke `np.column_stack`.
:::

::: quiz
`x` har shape `(50,)`. Hvilken linje gir **ikke** shape `(50, 1)`?
- [ ] `x.reshape(-1, 1)`
- [ ] `x[:, np.newaxis]`
- [x] `x.reshape(1, -1)`
- [ ] `x.reshape(50, 1)`
? `x.reshape(1, -1)` gir én rad med 50 kolonner, shape `(1, 50)`. De andre lager en kolonnevektor.
:::

::: oppsummering
- `reshape(r, k)` endrer formen (fyller rad for rad). `-1` betyr «regn ut selv».
- `reshape(-1, 1)` gjør 1D om til en kolonne. sklearn krever 2D `X`.
- `ravel()` gjør alt til 1D. `np.newaxis`/`None` legger til en akse.
- `.T` bytter rader og kolonner. `x0, x1 = X.T` pakker ut kolonnene.
- `np.vstack` (rader), `np.hstack` (kolonner), `np.column_stack` (1D-arrays som kolonner), `np.concatenate(..., axis=...)`.
:::

--- LEKSJON 2.9 | linalg | Lineær algebra: @, matmul, dot, norm og eig
Lineære modeller og nevrale nett er i bunn og grunn matriseprodukter. NumPy skriver matriseproduktet med operatoren `@` (eller `np.matmul`). Husk fra leksjon 2.4 at `*` er element for element, ikke matriseprodukt.

## Prikkprodukt og matrise ganger vektor
For to 1D-vektorer gir `a @ b` prikkproduktet $\sum_i a_i b_i$, ett tall. For en matrise `X` med shape `(n, d)` og en vektor `w` med shape `(d,)` gir `X @ w` prikkproduktet mellom **hver rad** og `w`, altså én verdi per datapunkt.

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
print(a @ b, np.dot(a, b))    # 1*4 + 2*5 + 3*6 = 32
print(a * b)                  # element for element, IKKE det samme

X = np.array([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0]])    # 3 datapunkter, 2 features
w = np.array([0.5, -1.0])     # én vekt per feature
b0 = 2.0

z = X @ w + b0                # lineær modell: én prediksjon per datapunkt
print(z, z.shape)
print(np.matmul(X, w) + b0)   # samme
```

Se `X`, `w` og `z` i variabelpanelet: `X` har 3 rader og 2 kolonner, `w` har 2 tall (ett per kolonne), og `z` har 3 tall, ett per rad i `X`.

## Shape-regelen for matriseprodukt
`A @ B` krever at **indre dimensjoner er like**: `(n, k) @ (k, m)` gir `(n, m)`. De indre k-ene «spises opp».

| A | B | `A @ B` |
|---|---|---|
| `(n, d)` | `(d,)` | `(n,)` |
| `(n, d)` | `(d, m)` | `(n, m)` |
| `(32, 784)` | `(784, 128)` | `(32, 128)`: et lag i et nevralt nett |
| `(n, d)` | `(n,)` | **feil**: d og n passer ikke |

```python feil
import numpy as np
X = np.ones((5, 3))
w = np.ones(5)
print(X @ w)
```

```python
import numpy as np

batch = np.ones((32, 4))       # 32 datapunkter, 4 features
W = np.ones((4, 8))            # vekter fra 4 inn til 8 ut
print((batch @ W).shape)
print((batch.T @ np.ones(32)).shape)   # (4, 32) @ (32,) -> (4,)
```

Med små matriser kan du se hvor tallene kommer fra. Element `C[i, j]` er prikkproduktet av rad `i` i `A` og kolonne `j` i `B`:

```python
import numpy as np

A = np.array([[1, 0, 2],
              [0, 1, 1]])          # (2, 3)
B = np.array([[1, 2, 0, 1],
              [0, 1, 1, 0],
              [1, 0, 1, 1]])       # (3, 4)
C = A @ B
print(A.shape, "@", B.shape, "->", C.shape)
display(A)
display(B)
display(C)
```

Sjekk `C[0, 0]`: rad 0 i `A` er `[1, 0, 2]` og kolonne 0 i `B` er `[1, 0, 1]`, så $1 \cdot 1 + 0 \cdot 0 + 2 \cdot 1 = 3$.

::: eksamen Eksamen 2025, spørsmål 2
Eksamen 2025 viste denne koden, med `X` shape `(n_samples, n_features)` og `w` shape `(n_features,)`:
```python statisk ikkekjor
z = np.matmul(X, w)
y_pred = 1 / (1 + np.exp(-z))
```
`np.matmul(X, w)` gir én reell score per sample (shape `(n_samples,)`). Sigmoid gjør hver score om til et tall i intervallet (0, 1). Altså inneholder `y_pred` **predikerte sannsynligheter**, ikke klasselabels.
:::

```python
import numpy as np

rng = np.random.default_rng(0)       # tilfeldige tall forklares i neste leksjon
X = rng.normal(size=(6, 3))
w = np.array([1.0, -2.0, 0.5])

z = np.matmul(X, w)
y_pred = 1 / (1 + np.exp(-z))
print("z:     ", z.round(2))
print("y_pred:", y_pred.round(3))
print("klasser:", (y_pred >= 0.5).astype(int))
```

## Normen: lengde og avstand
`np.linalg.norm(v)` gir lengden (den euklidske normen) $\sqrt{\sum_i v_i^2}$. Avstanden mellom to punkter er normen av differansen. Med `axis=1` får du normen av hver rad, altså avstanden fra mange punkter samtidig.

```python
import numpy as np

p1 = np.array([0.0, 0.0])
p2 = np.array([3.0, 4.0])
print(np.linalg.norm(p2))         # 5.0
print(np.linalg.norm(p1 - p2))    # avstand mellom punktene

X = np.array([[1.0, 1.0], [4.0, 5.0], [0.0, 2.0]])
senter = np.array([1.0, 2.0])
print(np.linalg.norm(X - senter, axis=1))   # avstand fra hvert punkt til senteret
```

Dette er kjernen i K-means og KNN (del 10), og koden i notatene om dimensjonalitetens forbannelse bruker `np.linalg.norm(P1 - P2)`.

## Kovarians og egenvektorer (til PCA)
PCA for hånd består av to steg: regn ut **kovariansmatrisen** til dataene, og finn **egenvektorene** og **egenverdiene** dens. `np.cov` forventer at hver **rad** er en variabel, så for `X` med datapunkter som rader skriver du `np.cov(X.T)` eller `np.cov(X, rowvar=False)`.

```python
import numpy as np

x1 = np.array([1, 2, 3, 4, 5])
x2 = np.array([1.6, 4.2, 5.7, 8.4, 9.2])   # dataene fra PCA-eksempelet i notatene
X = np.column_stack((x1, x2))

S = np.cov(X, rowvar=False)        # 2x2 kovariansmatrise
print(S)

egenverdier, egenvektorer = np.linalg.eigh(S)
print("Egenverdier:", egenverdier)
print("Egenvektorer (KOLONNER):\n", egenvektorer)

forste_pc = egenvektorer[:, np.argmax(egenverdier)]   # retningen med mest varians
print("Første hovedkomponent:", forste_pc)
```

Klikk på `egenvektorer` i variabelpanelet: hver **kolonne** er én egenvektor, og kolonne 1 hører til den største egenverdien.

::: advarsel Egenvektorene er kolonnene
Både `np.linalg.eig` og `np.linalg.eigh` returnerer egenvektorene som **kolonner**: egenvektor nummer i er `egenvektorer[:, i]`, ikke `egenvektorer[i]`. `eigh` er laget for symmetriske matriser (som kovariansmatriser) og gir egenverdiene sortert stigende. `eig` virker på alle kvadratiske matriser, men sorterer ikke.
:::

::: oppgave Logistisk prediksjon
Regn ut
1. `z`: den lineære scoren `X @ w + b` for hvert datapunkt.
2. `y_pred`: sigmoid av `z`.
3. `lengde`: normen (lengden) av vektvektoren `w`.
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
lengde = ...
```
```losning
import numpy as np

X = np.array([[1.0, 2.0, 0.0],
              [0.0, -1.0, 3.0],
              [2.0, 2.0, 2.0],
              [-1.0, 0.5, 1.0]])
w = np.array([0.5, -0.25, 1.0])
b = -0.5

z = X @ w + b
y_pred = 1 / (1 + np.exp(-z))
lengde = np.linalg.norm(w)
```
```test
import numpy as np
assert isinstance(z, np.ndarray) and z.shape == (4,), "z skal ha én verdi per datapunkt, shape (4,). Bruk X @ w + b (ikke X * w)."
assert np.allclose(z, [-0.5, 2.75, 2.0, -0.125]), "z har feil verdier. Husk å legge til b."
assert isinstance(y_pred, np.ndarray) and np.allclose(y_pred, 1 / (1 + np.exp(-z))), "y_pred skal være sigmoid av z: 1 / (1 + np.exp(-z))."
assert lengde is not ... and np.isclose(lengde, 1.14564392), "lengde skal være np.linalg.norm(w)."
```
:::

::: quiz
`X` er et 2D-array med shape `(n_samples, n_features)` og `w` et 1D-array med shape `(n_features,)`. Hva inneholder `y_pred` etter koden?
```kode
import numpy as np
z = np.matmul(X, w)
y_pred = 1 / (1 + np.exp(-z))
```
- [ ] For hvert sample en score som kan anta hvilken som helst reell verdi
- [ ] Logaritmen av de predikerte sannsynlighetene for hvert sample
- [x] Predikerte sannsynligheter i intervallet (0, 1) for hvert sample
- [ ] Klasselabels 0 eller 1 for hvert sample
? `z` er de reelle scorene, men sigmoid presser dem inn i (0, 1). For å få klasselabels må man i tillegg sammenligne med en terskel, f.eks. `y_pred >= 0.5`. (Eksamen 2025.)
:::

::: quiz
`A` har shape `(64, 10)` og `B` har shape `(10, 3)`. Hva er `(A @ B).shape`?
- [x] `(64, 3)`
- [ ] `(10, 10)`
- [ ] `(64, 10)`
- [ ] Feilmelding: shapene passer ikke
? Indre dimensjoner (10 og 10) er like og forsvinner. De ytre (64 og 3) blir resultatets shape.
:::

::: oppsummering
- `@` / `np.matmul` er matriseprodukt, `np.dot` gir prikkprodukt (og matriseprodukt for 2D). `*` er element for element.
- `(n, k) @ (k, m)` gir `(n, m)`: indre dimensjoner må være like.
- Lineær modell: `X @ w + b` gir én score per datapunkt. Med sigmoid rundt blir det sannsynligheter.
- `np.linalg.norm(a - b)` er avstanden mellom to punkter, `axis=1` gir én norm per rad.
- PCA for hånd: `np.cov(X, rowvar=False)` og `np.linalg.eigh(S)`. Egenvektorene er **kolonnene**.
:::

--- LEKSJON 2.10 | tilfeldig | Tilfeldige tall og seeds
Maskinlæring er full av tilfeldighet: startvekter, stokking av data, train/test-split og utforskning i forsterkningslæring. En datamaskin lager «tilfeldige» tall med en algoritme som starter fra et **frø** (seed). Samme seed gir nøyaktig de samme tallene hver gang, så resultatene dine blir **reproduserbare**.

## To måter: np.random.seed og default_rng
NumPy har to grensesnitt. Det gamle bruker én global generator som du seeder med `np.random.seed(...)`. Det nye lager en egen generator med `np.random.default_rng(seed)`. Begge finnes i prosjektene, så du må kunne lese begge.

```python
import numpy as np

# Gammel stil (global tilstand), brukt i prosjekt 4
np.random.seed(0)
print(np.random.random(3))         # uniformt i [0, 1)
print(np.random.randint(0, 4))     # heltall 0, 1, 2 eller 3
print(np.random.normal(0, 1, 3))   # normalfordelt

# Ny stil (anbefalt), brukt i prosjekt 3
rng = np.random.default_rng(0)
print(rng.random(3))
print(rng.integers(0, 4))          # også her er øvre grense IKKE med
print(rng.normal(loc=0, scale=1, size=3))
```

Kjør eksempelet flere ganger: tallene blir de samme hver gang. Fjern seed-linjene og se at de endrer seg.

## De viktigste funksjonene

```python
import numpy as np

rng = np.random.default_rng(42)
print(rng.uniform(-1, 1, size=4))            # uniformt mellom -1 og 1
print(rng.normal(loc=10, scale=2, size=(2, 3)))   # shape (2, 3)
print(rng.integers(0, 10, size=5))           # 5 heltall fra 0 til 9
print(rng.choice([10, 20, 30, 40], size=2, replace=False))   # trekk uten tilbakelegging
print(rng.permutation(6))                    # 0..5 i tilfeldig rekkefølge
```

## Stokke X og y sammen
Når du stokker data, må `X` og `y` stokkes **likt**, ellers havner labelene på feil datapunkt. Løsningen er å lage én tilfeldig rekkefølge av indeksene og bruke den på begge. Slik kan du også lage en enkel train/test-split for hånd:

```python
import numpy as np

X = np.arange(20).reshape(10, 2)
y = np.arange(10) * 10          # y[i] hører til X[i]

rng = np.random.default_rng(0)
idx = rng.permutation(len(X))
X_stokket, y_stokket = X[idx], y[idx]
print(X_stokket[:3], y_stokket[:3])   # radene og labelene følger hverandre

n_train = int(0.8 * len(X))
X_train, X_test = X_stokket[:n_train], X_stokket[n_train:]
y_train, y_test = y_stokket[:n_train], y_stokket[n_train:]
print(X_train.shape, X_test.shape)
```

Sammenlign `X`, `y`, `X_stokket` og `y_stokket` i variabelpanelet: radene har byttet plass, men hver rad har fortsatt med seg sin egen label.

(I praksis bruker du `train_test_split` fra sklearn, del 6. Den gjør nøyaktig dette.)

## ε-greedy: utforske eller utnytte
I Q-learning velger agenten en **tilfeldig** handling med sannsynlighet ε, og ellers den beste kjente. `np.random.random()` gir et tall i [0, 1), så `np.random.random() < epsilon` er sant i en andel ε av gangene:

```python
import numpy as np

np.random.seed(0)
q_table = np.zeros((16, 4))
q_table[5] = [0.1, 0.9, 0.3, 0.2]

def velg_handling(state, epsilon):
    if np.random.random() < epsilon:
        return int(np.random.randint(4))        # utforsk: tilfeldig handling
    return int(np.argmax(q_table[state]))       # utnytt: beste handling

valg = [velg_handling(5, epsilon=0.2) for _ in range(1000)]
print("Andel ganger handling 1 ble valgt:", np.mean(np.array(valg) == 1))
```

Med ε = 0.2 velges den beste handlingen (1) i 80 % av tilfellene, pluss en fjerdedel av de tilfeldige valgene, altså rundt 85 %.

::: prosjekt Prosjekt 3 og 4
Prosjekt 4 seeder med `np.random.seed(seed)` og bruker `np.random.random() < explore_rate` og `np.random.randint(env.action_space.n)`. Prosjekt 3 bruker `np.random.default_rng(SEED).choice(len(images), 16, replace=False)` for å velge 16 tilfeldige bilder og `.permutation(...)` for å dele inn i folds.
:::

::: oppgave Stokk og del opp
Bruk `rng` til å lage en tilfeldig rekkefølge `idx` av indeksene til `X` (med `permutation`). Stokk `X` og `y` likt, og del opp slik at de 8 første stokkede datapunktene blir `X_train`/`y_train` og de 2 siste `X_test`/`y_test`.
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

::: quiz
Hvilke verdier kan `rng.integers(0, 4)` gi?
- [ ] 0, 1, 2, 3 og 4
- [x] 0, 1, 2 og 3
- [ ] 1, 2, 3 og 4
- [ ] Et desimaltall mellom 0 og 4
? Øvre grense er ikke med, akkurat som i `range` og `np.arange`. Det samme gjelder `np.random.randint(0, 4)`.
:::

::: quiz
Hva er den viktigste grunnen til å sette en seed, f.eks. `np.random.seed(0)`, øverst i en notebook?
- [ ] Det gjør modellen mer nøyaktig
- [x] Koden gir de samme tilfeldige tallene, og dermed de samme resultatene, hver gang den kjøres
- [ ] Det fjerner all tilfeldighet fra treningen
- [ ] Det gjør at tallene blir jevnere fordelt
? Med seed blir tilfeldigheten *reproduserbar*: tallene er fortsatt «tilfeldige», men de samme hver gang. Det gjør det mulig å sammenligne eksperimenter rettferdig.
:::

::: oppsummering
- Samme seed gir samme tilfeldige tall: reproduserbarhet.
- Gammel stil: `np.random.seed(0)`, `np.random.random()`, `np.random.randint(a, b)`, `np.random.normal(...)`.
- Ny stil: `rng = np.random.default_rng(0)`, så `rng.normal`, `rng.uniform`, `rng.integers`, `rng.choice`, `rng.permutation`.
- Øvre grense i `integers`/`randint` er ikke med.
- Stokk `X` og `y` med samme indekser: `idx = rng.permutation(len(X))`, `X[idx]`, `y[idx]`.
- ε-greedy: `if np.random.random() < epsilon:` tilfeldig handling, ellers `np.argmax(q_table[state])`.
:::

--- LEKSJON 2.11 | verktoy | Nyttige verktøy
Til slutt en verktøykasse med funksjoner som dukker opp her og der i prosjektene. Du trenger ikke pugge alle, men du må kjenne dem igjen når du ser dem.

## Sortering: sort og argsort
`np.sort(x)` gir de sorterte verdiene. `np.argsort(x)` gir **indeksene** som ville sortert `x`: først indeksen til minste element, så nest minste, osv.

```python
import numpy as np

x = np.array([30, 10, 50, 20])
print(np.sort(x))              # [10 20 30 50]
print(np.argsort(x))           # [1 3 0 2]: x[1] er minst, så x[3] ...
print(x[np.argsort(x)])        # samme som np.sort(x)

viktighet = np.array([0.05, 0.40, 0.10, 0.45])
topp2 = np.argsort(viktighet)[::-1][:2]   # snu for synkende rekkefølge
print("De to viktigste featurene:", topp2)
```

Klikk på `x` i variabelpanelet: indeksen står over hver rute. Minste verdi (10) står på indeks 1, nest minste (20) på indeks 3, så 30 på indeks 0 og 50 på indeks 2. Det er nettopp `[1 3 0 2]`.

`argsort` kan også **reparere en stokket rekkefølge**. I prosjekt 3 er pikslene i hvert bilde stokket, men kolonnenavnene forteller hvilken piksel hver kolonne egentlig er. Sorterer du kolonnene etter det opprinnelige nummeret, får du bildet tilbake:

```python
import numpy as np

bilde = np.array([0, 10, 20, 30, 40, 50])            # den riktige rekkefølgen
opprinnelig_indeks = np.array([4, 0, 5, 2, 1, 3])     # kolonne j er egentlig piksel nr. opprinnelig_indeks[j]
stokket = bilde[opprinnelig_indeks]
print("Stokket:      ", stokket)
display(stokket)

order = np.argsort(opprinnelig_indeks)
print("Gjenopprettet:", stokket[order])
display(stokket[order])
```

::: prosjekt Prosjekt 3
Prosjekt 3 gjør nøyaktig dette: `order = np.argsort([int(c.split("_")[1]) for c in header])` henter pikselnummeret fra hvert kolonnenavn, og `pixels = features.to_numpy()[:, order]` setter kolonnene i riktig rekkefølge før `pixels.reshape(-1, 128, 128)`.
:::

## Telle: unique og bincount
`np.unique(y, return_counts=True)` gir de ulike verdiene og hvor mange det er av hver. `np.bincount(y)` teller ikke-negative heltall direkte: element `k` i svaret er antallet `k`-er. Begge brukes til å sjekke klassebalanse eller størrelsen på klynger.

```python
import numpy as np

y = np.array([0, 2, 1, 0, 0, 2, 0])
verdier, antall = np.unique(y, return_counts=True)
print(verdier, antall)
print(np.bincount(y))           # [4 1 2]: fire 0-ere, én 1-er, to 2-ere
print(np.bincount(y) / len(y))  # andeler
```

## clip, cumsum og glidende snitt
`np.clip(x, lav, høy)` kutter verdier utenfor et intervall. Det brukes for å unngå `log(0)` i cross-entropy. `np.cumsum` gir kumulativ sum (f.eks. total belønning så langt). `np.convolve` med et vindu av like vekter gir et **glidende snitt**, som glatter ut en støyete kurve.

```python
import numpy as np

p = np.array([0.0, 0.3, 1.0])
print(np.log(p))                              # -inf og en advarsel for log(0)
p_trygg = np.clip(p, 1e-15, 1 - 1e-15)
print(np.log(p_trygg))                        # endelige tall

belonning = np.array([1, -1, 2, 0, 3])
print(np.cumsum(belonning))

steg = np.array([90, 70, 85, 40, 50, 30, 20, 25, 10, 12])
glatt = np.convolve(steg, np.ones(3) / 3, mode="valid")   # snitt av 3 og 3
print(glatt.round(1), len(steg), "->", len(glatt))
```

Med `mode="valid"` regnes snittet bare der hele vinduet får plass, så resultatet blir `len(x) - vindu + 1` langt.

::: prosjekt Prosjekt 4
Kurven over antall steg per episode er svært støyete. Prosjekt 4 plotter derfor `np.convolve(steps_per_episode, np.ones(50) / 50, mode='valid')`, et glidende snitt over 50 episoder (se leksjon 4.1).
:::

## meshgrid: et rutenett av punkter
`np.meshgrid(xs, ys)` lager alle kombinasjoner av x- og y-verdier som to matriser. Det brukes til å tegne beslutningsgrenser (leksjon 4.6): du lager et tett rutenett, predikerer på hvert punkt og farger etter svaret.

```python
import numpy as np

xs = np.array([0, 1, 2])
ys = np.array([10, 20])
xx, yy = np.meshgrid(xs, ys)
print("xx: x-verdien i hvert punkt i rutenettet")
display(xx)
print("yy: y-verdien i hvert punkt i rutenettet")
display(yy)
print(xx.shape)                               # (len(ys), len(xs))
punkter = np.column_stack((xx.ravel(), yy.ravel()))
print(punkter)                                # alle 6 kombinasjonene
```

## nan og isclose
`np.nan` (*not a number*) markerer en manglende verdi. Den smitter: alt du regner med `nan` blir `nan`, og `nan == nan` er `False`. Bruk derfor `np.isnan` for å finne dem, og `np.nanmean` osv. for å se bort fra dem. Flyttall bør aldri sammenlignes med `==`, fordi små avrundingsfeil gjør at «like» tall ikke er helt like. Bruk `np.isclose` (eller `np.allclose` for hele arrays).

```python
import numpy as np

x = np.array([1.0, np.nan, 3.0])
print(x.mean(), np.nanmean(x))
print(np.nan == np.nan, np.isnan(x))
print(x[~np.isnan(x)])                # fjern nan

print(0.1 + 0.2 == 0.3)
print(np.isclose(0.1 + 0.2, 0.3))
print(np.allclose([0.1 + 0.2, 1.0], [0.3, 1.0]))
```

Klikk på `x` i variabelpanelet: overskriften viser i rødt hvor mange `nan` arrayet har, og ruten med `nan` får ingen farge.

::: oppgave Gjenopprett pikselrekkefølgen
Kolonnene i `rad` er stokket. Kolonnenavnene i `header` forteller hvilken piksel hver kolonne egentlig er (`"px_3"` er piksel 3). Lag
1. `nummer`: et array med pikselnummeret fra hvert kolonnenavn (bruk `int(c.split("_")[1])`).
2. `order`: rekkefølgen som sorterer `nummer`.
3. `gjenopprettet`: `rad` i riktig rekkefølge.
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
assert np.array_equal(np.asarray(nummer), [3, 0, 2, 1]), "nummer skal være [3, 0, 2, 1]: tallet etter _ i hvert kolonnenavn."
assert np.array_equal(np.asarray(order), [1, 3, 2, 0]), "order skal være np.argsort(nummer)."
assert isinstance(gjenopprettet, np.ndarray) and np.array_equal(gjenopprettet, [0, 10, 20, 30]), "gjenopprettet skal være rad[order], som gir [0, 10, 20, 30]."
```
:::

::: oppgave Glatt ut en læringskurve
Lag
1. `glatt`: glidende snitt av `steg` over 4 og 4 episoder med `np.convolve` og `mode="valid"`.
2. `antall`: hvor mange episoder som er av hver lengde i `lengder` (bruk `np.bincount`).
```start
import numpy as np

steg = np.array([100, 80, 96, 60, 52, 40, 36, 20, 24, 12])
lengder = np.array([2, 3, 3, 1, 3, 2, 3])

glatt = ...
antall = ...
```
```losning
import numpy as np

steg = np.array([100, 80, 96, 60, 52, 40, 36, 20, 24, 12])
lengder = np.array([2, 3, 3, 1, 3, 2, 3])

glatt = np.convolve(steg, np.ones(4) / 4, mode="valid")
antall = np.bincount(lengder)
```
```test
import numpy as np
assert isinstance(glatt, np.ndarray), "glatt skal være resultatet av np.convolve."
assert len(glatt) == 7, f"glatt skal ha lengde 10 - 4 + 1 = 7 med mode='valid', men har {len(glatt)}."
assert np.allclose(glatt, [84, 72, 62, 47, 37, 30, 23]), "glatt har feil verdier. Vinduet skal være np.ones(4) / 4."
assert isinstance(antall, np.ndarray) and np.array_equal(antall, [0, 1, 2, 4]), "antall skal være np.bincount(lengder): [0, 1, 2, 4]."
```
:::

::: quiz
Hva skriver koden ut?
```kode
import numpy as np
x = np.array([7, 2, 9, 4])
print(np.argsort(x))
```
- [ ] `[2 4 7 9]`
- [x] `[1 3 0 2]`
- [ ] `[2 0 3 1]`
- [ ] `[3 1 2 0]`
? `argsort` gir indeksene i sortert rekkefølge: minste er 2 (indeks 1), så 4 (indeks 3), 7 (indeks 0) og 9 (indeks 2). `[2 4 7 9]` er `np.sort(x)`.
:::

::: quiz
Hvorfor skriver man ofte `np.clip(p, 1e-15, 1 - 1e-15)` før man regner ut cross-entropy med `np.log(p)`?
- [ ] For å gjøre treningen raskere
- [ ] For å normalisere sannsynlighetene så de summerer til 1
- [x] For å unngå `log(0)`, som gir `-inf` og ødelegger tapet
- [ ] For å gjøre om sannsynligheter til klasser
? `np.log(0)` er minus uendelig. Ett slikt tall gjør hele tapet uendelig (eller `nan`). Ved å kutte p litt inn fra 0 og 1 blir logaritmen alltid endelig.
:::

::: oppsummering
- `np.sort` gir verdiene sortert, `np.argsort` gir **indeksene**. `x[np.argsort(x)]` sorterer, og argsort reparerer en stokket rekkefølge.
- `np.unique(y, return_counts=True)` og `np.bincount(y)` teller forekomster.
- `np.clip` kutter verdier (unngå `log(0)`), `np.cumsum` gir kumulativ sum.
- Glidende snitt: `np.convolve(x, np.ones(k) / k, mode="valid")`, lengde `len(x) - k + 1`.
- `np.meshgrid(xs, ys)` lager et rutenett av punkter.
- `np.nan` smitter, `nan == nan` er `False`. Bruk `np.isnan`, `np.nanmean`, og `np.isclose`/`np.allclose` for flyttall.
:::
