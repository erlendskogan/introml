=== DEL 1 | python | Python fra null
Alt du trenger av ren Python for å skrive koden i prosjektene: variabler, tekst, lister, løkker, funksjoner og klasser. Ingen forkunnskaper kreves.

--- LEKSJON 1.1 | forste-program | Ditt første program
Et program er en liste med instruksjoner som datamaskinen utfører **ovenfra og ned**, én linje om gangen. Den aller første instruksjonen alle lærer, er `print()`. Den skriver ut det du gir den.

Trykk på den grønne **Kjør**-knappen under. Første gang tar det noen sekunder, fordi Python startes i nettleseren din.

```python
print("Hei, maskinlæring!")
```

Teksten i anførselstegn kalles en **streng** (string). Du kan skrive ut tall også, og da trenger du ikke anførselstegn:

```python
print("Antall prosjekter i faget:")
print(4)
print(2 + 2)
```

Legg merke til at `print(2 + 2)` skriver ut `4`. Python regner ut uttrykket først, og skriver så ut svaret.

## Kommentarer
Alt etter `#` på en linje er en **kommentar**. Python hopper over den. Kommentarer er for mennesker, og brukes for å forklare koden (prosjektene er fulle av dem: `# YOUR CODE GOES HERE`).

```python
# Dette er en kommentar, den gjør ingenting
print("Denne linjen kjøres")   # kommentar etter kode er også lov
# print("Denne linjen kjøres IKKE, fordi den er kommentert ut")
```

::: tips
Du kan endre koden i alle eksemplene og kjøre den på nytt. Prøv å skrive ut navnet ditt! Trykker du **↺ Tilbakestill**, kommer den opprinnelige koden tilbake.
:::

## Feilmeldinger er vennene dine
Når noe er galt, stopper Python og skriver en **feilmelding**. Den forteller hvilken linje som feilet og hva slags feil det er. Kjør koden under og les meldingen:

```python feil
print("Første linje går fint")
print("Her mangler en slutt-parentes"
```

`SyntaxError` betyr at koden ikke er gyldig Python, her fordi parentesen aldri lukkes. Legg merke til at ikke engang første linje ble skrevet ut: Python sjekker at hele programmet er gyldig før noe kjøres. Det er helt vanlig å få feilmeldinger. Les dem nøye: de peker nesten alltid rett på problemet.

::: oppgave Skriv ut to linjer
Lag et program som skriver ut nøyaktig disse to linjene:

`Hei!` og deretter `Jeg lærer Python.`

Lagre i tillegg teksten `"ferdig"` i variabelen `status` på siste linje (variabler lærer du om i neste leksjon, bare skriv det av).
```start
# Skriv de to print-linjene her


status = ...
```
```losning
print("Hei!")
print("Jeg lærer Python.")
status = "ferdig"
```
```test
assert status == "ferdig", 'Skriv status = "ferdig" på siste linje.'
```
:::

::: quiz
Hva skriver dette programmet ut?
```kode
# print("A")
print("B")
print(1 + 2)
```
- [ ] `A`, `B` og `3`
- [ ] `B` og `1 + 2`
- [x] `B` og `3`
- [ ] Ingenting, fordi første linje er en kommentar
? Linjen med `#` er en kommentar og hoppes over. `print(1 + 2)` regner ut uttrykket før utskrift, så det blir `3`.
:::

::: oppsummering
- Et program kjøres **ovenfra og ned**, linje for linje.
- `print(...)` skriver ut tekst (i anførselstegn) eller verdien av et uttrykk.
- `#` starter en kommentar som Python ignorerer.
- Feilmeldinger sier **hvilken linje** og **hvilken type feil**. Les dem.
:::

--- LEKSJON 1.2 | variabler | Variabler og datatyper
En **variabel** er et navn som peker på en verdi. Du lager den med `=`: navnet står til venstre, verdien til høyre. Etterpå kan du bruke navnet i stedet for verdien.

```python
learning_rate = 0.01
epochs = 100
print(learning_rate)
print(epochs)
```

Under resultatet ser du **variabelpanelet**: alle variablene programmet laget, med navn, type og verdi. Klikk på en variabel for å se innholdet. Her står det for eksempel at `learning_rate` er en `float` og `epochs` en `int`. Panelet blir nyttig senere, når variablene er lister, tabeller og modeller.

`=` betyr «la navnet peke på denne verdien», ikke «er lik» som i matematikken. Derfor er `steg = steg + 1` helt lovlig: Python regner ut høyresiden først, og lagrer så resultatet i `steg`.

```python
steg = 0
steg = steg + 1
steg = steg + 1
print(steg)
```

Variabelpanelet viser verdien variabelen har **når programmet er ferdig**, så her står `steg` som 2.

## De fire grunnleggende datatypene
Hver verdi har en **type**. Disse fire bruker du hele tiden:

| Type | Eksempler | Typisk bruk i ML |
|---|---|---|
| `int` (heltall) | `42`, `-3`, `0` | antall epoker, klassenummer, indekser |
| `float` (desimaltall) | `0.01`, `3.14`, `1e-3` | vekter, tap, sannsynligheter |
| `str` (tekst) | `"katt"`, `'hund'` | kategorier, filnavn |
| `bool` (sannhetsverdi) | `True`, `False` | `done`, `terminated`, `shuffle=True` |

Funksjonen `type()` forteller hvilken type en verdi har:

```python
print(type(42))
print(type(0.01))
print(type("hund"))
print(type(True))
print(1e-3)   # vitenskapelig notasjon: 1 * 10^-3 = 0.001
```

Desimaltall skrives med **punktum**, aldri komma: `0.5`, ikke `0,5`. Skrivemåten `1e-3` (= 0,001) er vanlig for læringsrater.

## None: «ingen verdi ennå»
`None` er en egen verdi som betyr «ingenting». Den brukes for ting som ikke er bestemt ennå. I prosjekt 1 starter modellen med `self.weights = None`, fordi vektene først finnes etter at modellen er trent.

```python
weights = None
print(weights)
print(type(weights))
```

## Konvertere mellom typer
`int()`, `float()` og `str()` gjør om en verdi til en annen type. Det trengs for eksempel når tall er lest inn som tekst.

```python
tekst = "42"
tall = int(tekst)
print(tall + 1)
print(float("0.25"))
print(int(3.9))          # kutter desimalene, runder IKKE av
print(str(100) + " epoker")
```

## Regler for variabelnavn
- Bare bokstaver, tall og `_`, og navnet kan ikke starte med et tall.
- Store og små bokstaver er forskjellige: `X` og `x` er to ulike variabler. I ML er det vanlig å bruke stor `X` for en matrise med data og liten `y` for fasiten.
- Vanlig stil i Python er små bokstaver med `_` mellom ordene: `learning_rate`, `n_epochs`, `y_pred`.
- Du kan ikke bruke ord som Python selv bruker, som `if`, `for`, `class` og `lambda`.

Bruker du et navn som ikke finnes, får du `NameError`:

```python feil
epochs = 100
print(Epochs)
```

::: oppgave Hyperparametere i variabler
Lag fire variabler som beskriver en modell:
- `modell` med teksten `"LinearRegression"`
- `lr` med desimaltallet `0.001`
- `n_iterations` med heltallet `1000`
- `ferdig_trent` med sannhetsverdien `False`
```start
modell = ...
lr = ...
n_iterations = ...
ferdig_trent = ...
```
```losning
modell = "LinearRegression"
lr = 0.001
n_iterations = 1000
ferdig_trent = False
```
```test
assert modell == "LinearRegression", 'modell skal være teksten "LinearRegression" (husk anførselstegn).'
assert isinstance(lr, float) and abs(lr - 0.001) < 1e-12, "lr skal være desimaltallet 0.001 (eller 1e-3)."
assert type(n_iterations) is int and n_iterations == 1000, "n_iterations skal være heltallet 1000 (uten desimaler)."
assert ferdig_trent is False, "ferdig_trent skal være False (stor F, uten anførselstegn)."
```
:::

::: quiz
Hva skriver koden ut?
```kode
x = 5
y = x
x = 10
print(y)
```
- [x] 5
- [ ] 10
- [ ] 15
- [ ] `NameError`
? `y = x` gir `y` verdien som `x` hadde *da*, altså 5. At `x` senere får en ny verdi, endrer ikke `y`.
:::

::: quiz
Hva skriver `print(int(7.8))` ut?
- [ ] 8
- [x] 7
- [ ] 7.8
- [ ] `ValueError`
? `int()` kutter bort desimalene, den runder ikke av. Vil du runde av, bruker du `round(7.8)`, som gir 8.
:::

::: oppsummering
- `navn = verdi` lager en variabel. Høyresiden regnes ut først.
- Grunntypene er `int`, `float`, `str` og `bool`. `type(x)` viser typen.
- `None` betyr «ingen verdi ennå», for eksempel vekter før trening.
- `int()`, `float()` og `str()` konverterer. `int()` kutter desimaler.
- Navn skiller mellom store og små bokstaver. Feil navn gir `NameError`.
:::

--- LEKSJON 1.3 | regning | Regning
Python er en kraftig kalkulator. Nesten all ML-kode er regning: prediksjoner, feil, tap og oppdatering av vekter.

| Operator | Betyr | Eksempel | Resultat |
|---|---|---|---|
| `+` `-` | pluss, minus | `7 - 2` | `5` |
| `*` | gange | `7 * 2` | `14` |
| `/` | dele | `7 / 2` | `3.5` |
| `//` | heltallsdivisjon | `7 // 2` | `3` |
| `%` | rest (modulo) | `7 % 2` | `1` |
| `**` | potens | `2 ** 3` | `8` |

`print()` kan skrive ut flere verdier på én linje. Skill dem med komma, så får du mellomrom mellom dem:

```python
print(7 + 2, 7 - 2, 7 * 2)
print(7 / 2, 7 // 2, 7 % 2)
print(2 ** 3, 9 ** 0.5)     # ** 0.5 er kvadratrot
print(10 / 2)               # / gir alltid float, selv om svaret er et heltall
```

## Regnerekkefølge
Python følger vanlige regneregler: først `**`, så `*` `/` `//` `%`, og til slutt `+` `-`. Bruk parenteser når du er i tvil.

```python
print(2 + 3 * 4)        # 14, ganging først
print((2 + 3) * 4)      # 20
print(-2 ** 2)          # -4: potensen regnes før minustegnet
print((-2) ** 2)        # 4
```

## Oppdatere en variabel: `+=`, `-=`, `*=`
Å endre en variabel litt om gangen er så vanlig at det finnes snarveier. `x += 1` betyr det samme som `x = x + 1`.

```python
total_reward = 0
total_reward += 10      # samme som total_reward = total_reward + 10
total_reward -= 1
print(total_reward)

epsilon = 1.0
epsilon *= 0.99         # epsilon = epsilon * 0.99 (brukes i prosjekt 4)
print(epsilon)
```

## Nyttige innebygde funksjoner
```python
print(abs(-0.3))            # absoluttverdi
print(round(3.14159, 2))    # rund av til 2 desimaler
print(min(4, 1, 7), max(4, 1, 7))
print(1500 % 100, 1537 % 100)   # rest: brukes for å gjøre noe hver 100. episode
```

## Et steg med gradient descent
Her er en smakebit på det du skal gjøre i del 5. En modell med én vekt `w` predikerer `w * x`. Vi måler feilen, regner ut gradienten og flytter `w` litt i motsatt retning:

```python
w = 0.5           # vekten modellen har nå
x, y = 2.0, 3.0   # ett datapunkt: input og fasit
lr = 0.1          # læringsrate

prediksjon = w * x
feil = prediksjon - y
tap = feil ** 2               # kvadratisk feil
gradient = 2 * feil * x       # den deriverte av tapet med hensyn på w
w = w - lr * gradient         # gradient descent-steget

print(prediksjon, tap, w)
```

Linjen `x, y = 2.0, 3.0` gir to variabler verdi på én gang. Det lærer du mer om i leksjon 1.11.

::: advarsel Desimaltall er ikke helt eksakte
Datamaskinen lagrer desimaltall i totallssystemet, så noen tall blir litt unøyaktige: `0.1 + 0.2` gir `0.30000000000000004`. Det er normalt og ufarlig, men betyr at du ikke bør sjekke om to desimaltall er *nøyaktig* like (se leksjon 1.5).
:::

::: oppgave Ett gradient descent-steg
Gitt `w = 0.8`, ett datapunkt `x = 2.0`, `y = 3.0` og læringsrate `lr = 0.1`. Regn ut, i denne rekkefølgen:
1. `prediksjon` = `w` ganger `x`
2. `feil` = `prediksjon` minus `y`
3. `tap` = `feil` opphøyd i andre
4. `gradient` = `2 * feil * x`
5. `w_ny` = `w` minus `lr` ganger `gradient`
```start
w = 0.8
x = 2.0
y = 3.0
lr = 0.1

prediksjon = ...
feil = ...
tap = ...
gradient = ...
w_ny = ...
```
```losning
w = 0.8
x = 2.0
y = 3.0
lr = 0.1

prediksjon = w * x
feil = prediksjon - y
tap = feil ** 2
gradient = 2 * feil * x
w_ny = w - lr * gradient
```
```test
assert isinstance(prediksjon, float) and abs(prediksjon - 1.6) < 1e-9, "prediksjon skal være w * x = 1.6"
assert isinstance(feil, float) and abs(feil - (-1.4)) < 1e-9, "feil skal være prediksjon - y = -1.4 (negativ, fordi vi predikerer for lavt)"
assert isinstance(tap, float) and abs(tap - 1.96) < 1e-9, "tap skal være feil ** 2 = 1.96"
assert isinstance(gradient, float) and abs(gradient - (-5.6)) < 1e-9, "gradient skal være 2 * feil * x = -5.6"
assert isinstance(w_ny, float) and abs(w_ny - 1.36) < 1e-9, "w_ny skal være w - lr * gradient = 1.36"
```
:::

::: quiz
Hva skriver koden ut?
```kode
print(7 // 2, 7 % 2, 2 ** 3)
```
- [ ] `3.5 1 6`
- [x] `3 1 8`
- [ ] `3 1 6`
- [ ] `3.5 0 8`
? `//` er heltallsdivisjon (7 // 2 = 3), `%` er resten (7 = 2·3 + 1) og `**` er potens (2³ = 8).
:::

::: quiz
Hvilken verdi får `x`?
```kode
x = 2 + 3 * 2 ** 2
```
- [x] 14
- [ ] 20
- [ ] 100
- [ ] 38
? Potens først: `2 ** 2 = 4`. Så ganging: `3 * 4 = 12`. Til slutt pluss: `2 + 12 = 14`.
:::

::: oppsummering
- `+ - * /` som vanlig. `/` gir alltid `float`.
- `//` er heltallsdivisjon, `%` er rest og `**` er potens.
- Regnerekkefølge: `**`, så `* / // %`, så `+ -`. Bruk parenteser ved tvil.
- `x += 1`, `x -= 1`, `x *= 0.99` oppdaterer en variabel.
- `abs()`, `round()`, `min()` og `max()` er innebygd.
:::

--- LEKSJON 1.4 | tekst | Tekst og f-strenger
Tekst kalles **strenger** (`str`). Du skriver dem med enten `"..."` eller `'...'`. Begge betyr det samme.

```python
navn = "Random Forest"
print(navn)
print(len(navn))             # antall tegn, mellomrom teller med
print(navn + " er en ensemblemetode")   # + skjøter sammen tekst
print("-" * 20)              # * gjentar tekst
print("Linje 1\nLinje 2")    # \n er linjeskift
```

## Strengmetoder
En **metode** er en funksjon som hører til en verdi. Du kaller den med punktum: `tekst.upper()`. Metodene lager en *ny* streng, den gamle endres ikke.

```python
s = "  Sepal Length  "
print(s.strip())                 # fjerner mellomrom i endene
print(s.strip().lower())         # små bokstaver
print(s.strip().replace(" ", "_"))
print("x0,x1,y".split(","))      # deler opp i en liste (lister kommer i 1.7)
print("FrozenLake-v1".startswith("Frozen"))
```

## Tekst og tall sammen
Du kan ikke skjøte tekst og tall med `+`:

```python feil
loss = 0.25
print("Tap: " + loss)
```

`TypeError` betyr at operasjonen ikke passer til typene. Du kunne skrevet `"Tap: " + str(loss)`, men det finnes en mye bedre måte.

## f-strenger
Setter du `f` foran anførselstegnet, kan du skrive variabler og uttrykk rett inn i teksten mellom `{` og `}`:

```python
epoch = 5
loss = 0.123456
print(f"Epoch {epoch}: loss = {loss}")
print(f"Neste epoke er {epoch + 1}")
```

## Formatering av tall
Etter et kolon inne i `{}` kan du bestemme hvordan tallet skal se ut. Dette ser du overalt i prosjektene:

| Skrivemåte | Betyr | Eksempel | Resultat |
|---|---|---|---|
| `{x:.2f}` | 2 desimaler | `x = 0.87654` | `0.88` |
| `{x:.1%}` | prosent med 1 desimal (ganger med 100) | `x = 0.87654` | `87.7%` |
| `{n:,}` | tusenskille | `n = 1234567` | `1,234,567` |
| `{x:+.3f}` | alltid fortegn, 3 desimaler | `x = 0.87654` | `+0.877` |
| `{x:.2e}` | vitenskapelig notasjon | `x = 0.000123` | `1.23e-04` |
| `{navn:20s}` | fyll ut til 20 tegn (venstrejustert) | `navn = "SVM"` | `SVM` + 17 mellomrom |
| `{x:8.3f}` | 8 tegn bredt, 3 desimaler (høyrejustert) | `x = 0.87654` | `   0.877` |

```python
acc = 0.87654
n = 1234567
print(f"{acc:.2f}")
print(f"{acc:.1%}")
print(f"{n:,}")
print(f"{acc:+.3f} og {-acc:+.3f}")
print(f"{0.000123:.2e}")
```

Bredde er nyttig når du skriver ut tabeller, fordi kolonnene havner under hverandre:

```python
print(f"{'Modell':20s}{'Accuracy':>10s}")
print(f"{'Decision tree':20s}{0.8123:10.3f}")
print(f"{'Random forest':20s}{0.8871:10.3f}")
print(f"{'Naive Bayes':20s}{0.7905:10.3f}")
```

`>` betyr høyrejustert. Tekst er venstrejustert og tall høyrejustert hvis du ikke sier noe.

::: prosjekt Slik ser det ut i prosjektene
Fra prosjekt 2, der `cv_scores` er resultatene fra kryssvalidering:
```python statisk ikkekjor
print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```
Du kan kalle metoder og regne inne i `{}`. Trenger du anførselstegn inni, må de være av den *andre* typen enn de ytterste, som i prosjekt 3: `f'{cfg["width"]}'`.
:::

::: oppgave Treningsrapport
Bruk en f-streng til å lage variabelen `rapport` slik at den blir nøyaktig

`Epoch 7: loss=0.0346, accuracy=91.2%`

Bruk variablene `epoch`, `loss` og `acc` (ikke skriv tallene direkte), med 4 desimaler på loss og prosent med 1 desimal på accuracy.
```start
epoch = 7
loss = 0.034567
acc = 0.91234

rapport = ...
```
```losning
epoch = 7
loss = 0.034567
acc = 0.91234

rapport = f"Epoch {epoch}: loss={loss:.4f}, accuracy={acc:.1%}"
```
```test
assert isinstance(rapport, str), "rapport skal være en tekst (bruk f\"...\")."
assert rapport == "Epoch 7: loss=0.0346, accuracy=91.2%", f"rapport ble {rapport!r}, men skal være 'Epoch 7: loss=0.0346, accuracy=91.2%'. Sjekk mellomrom og formatkodene :.4f og :.1%."
```
:::

::: quiz
Variabelen `acc` er `0.91678`. Hvilken linje skriver ut `Accuracy: 0.92`?
- [x] `print(f"Accuracy: {acc:.2f}")`
- [ ] `print("Accuracy: {acc:.2f}")`
- [ ] `print(f"Accuracy: {acc:.2%}")`
- [ ] `print(f"Accuracy: {acc:2f}")`
? Uten `f` foran skrives klammene ut som vanlig tekst. `:.2%` gir `91.68%`. `:2f` (uten punktum) betyr bredde 2 og standard 6 desimaler, altså `0.916780`.
:::

::: oppsummering
- Strenger skrives med `"..."` eller `'...'`. `len()` gir antall tegn, `+` skjøter og `*` gjentar.
- Metoder som `.strip()`, `.lower()`, `.replace()` og `.split()` lager nye strenger.
- `f"... {uttrykk} ..."` setter verdier rett inn i teksten.
- Formatkoder: `:.2f` desimaler, `:.1%` prosent, `:,` tusenskille, `:+.3f` fortegn, `:20s` bredde.
:::

--- LEKSJON 1.5 | sammenligninger | Sammenligninger og bool
En **sammenligning** gir alltid svaret `True` eller `False`, altså en `bool`. Det er grunnlaget for alle valg i et program, og for å sjekke om en prediksjon er riktig.

| Operator | Betyr |
|---|---|
| `==` | er lik (to likhetstegn!) |
| `!=` | er ikke lik |
| `<` `>` | mindre enn, større enn |
| `<=` `>=` | mindre eller lik, større eller lik |

```python
p = 0.73
print(p > 0.5)
print(p == 0.73)
print(p != 1)
print("katt" == "Katt")    # store og små bokstaver er forskjellige
```

Ikke bland `=` (gi en variabel verdi) og `==` (sammenligne).

## and, or og not
Du kan kombinere sannhetsverdier:
- `a and b` er `True` bare hvis begge er `True`.
- `a or b` er `True` hvis minst én er `True`.
- `not a` snur verdien.

```python
step = 50
max_steps = 50
reached_goal = False

done = reached_goal or step >= max_steps
print(done)
print(not done)
print(0 <= 0.73 <= 1)   # sammenligninger kan lenkes: er 0.73 mellom 0 og 1?
```

## Sammenligne desimaltall
På grunn av avrundingsfeil (leksjon 1.3) er `==` farlig for desimaltall. Sjekk heller om forskjellen er veldig liten:

```python
a = 0.1 + 0.2
print(a == 0.3)
print(abs(a - 0.3) < 1e-9)
```

## True og False er også tall
`True` oppfører seg som `1` og `False` som `0`. Det betyr at du kan *telle* sanne verdier ved å legge dem sammen. Slik regner man ut accuracy senere: antall riktige er summen av alle `prediksjon == fasit`.

```python
riktig1 = (1 == 1)
riktig2 = (0 == 1)
riktig3 = (1 == 1)
print(riktig1 + riktig2 + riktig3)
```

## Sjekke mot None
For `None` bruker man `is None` og `is not None`:

```python
weights = None
print(weights is None)
print(weights is not None)
```

::: oppgave Tolk en sannsynlighet
En modell sier at sannsynligheten for klasse 1 er `p`. Lag to `bool`-variabler (ikke skriv `True`/`False` direkte, bruk sammenligninger):
- `positiv`: `True` hvis `p` er større enn `terskel`
- `usikker`: `True` hvis `p` ligger mellom 0.4 og 0.6 (begge grensene tatt med)
```start
p = 0.58
terskel = 0.5

positiv = ...
usikker = ...
```
```losning
p = 0.58
terskel = 0.5

positiv = p > terskel
usikker = 0.4 <= p <= 0.6
```
```test
assert positiv is True, "positiv skal være resultatet av sammenligningen p > terskel (som blir True)."
assert usikker is True, "usikker skal være True når p er mellom 0.4 og 0.6, f.eks. 0.4 <= p <= 0.6."
```
:::

::: quiz
Hva skriver koden ut?
```kode
a = 5
print(a > 3 and a < 4)
```
- [ ] `True`
- [x] `False`
- [ ] `5`
- [ ] `SyntaxError`
? `a > 3` er `True`, men `a < 4` er `False`. `and` krever at begge er sanne.
:::

::: quiz
Hva skriver `print(True + True + False)` ut?
- [ ] `True`
- [ ] `TrueTrueFalse`
- [x] `2`
- [ ] `TypeError`
? `True` regnes som 1 og `False` som 0, så summen blir 2. Dette brukes til å telle riktige prediksjoner.
:::

::: oppsummering
- `==`, `!=`, `<`, `>`, `<=`, `>=` gir `True` eller `False`.
- `and`, `or` og `not` kombinerer sannhetsverdier. `0 <= p <= 1` er lov.
- Desimaltall sammenlignes med `abs(a - b) < 1e-9`, ikke `==`.
- `True` teller som 1 og `False` som 0, så summen av sammenligninger er antall sanne.
- Bruk `x is None` for å sjekke `None`.
:::

--- LEKSJON 1.6 | if-elif-else | if, elif og else
Med `if` kan programmet ta valg: koden under kjøres **bare hvis** betingelsen er sann.

```python
p = 0.83
if p > 0.5:
    print("Modellen sier: klasse 1")
    print("Denne linjen er også inne i if-en")
print("Denne linjen kjøres uansett")
```

To ting er viktige:
1. Linjen med `if` slutter med **kolon** `:`.
2. Koden som hører til, er **rykket inn** (4 mellomrom). Innrykket er ikke pynt: det er slik Python vet hvilke linjer som hører til `if`-en.

## else og elif
`else` kjøres hvis betingelsen er usann. `elif` («else if») lar deg teste flere betingelser etter hverandre. Python går ovenfra og ned og kjører **bare den første** grenen som er sann.

```python
acc = 0.76
if acc >= 0.9:
    vurdering = "svært bra"
elif acc >= 0.7:
    vurdering = "ok"
elif acc >= 0.5:
    vurdering = "litt bedre enn tilfeldig"
else:
    vurdering = "dårlig"
print(vurdering)
```

Her er `acc >= 0.5` også sant, men den grenen kjøres ikke, fordi `acc >= 0.7` kom først.

## Kombinerte betingelser og nøsting
Du kan bruke `and`/`or` i betingelsen, og du kan ha en `if` inni en annen (med dobbelt innrykk):

```python
episode = 300
epsilon = 0.05
if episode % 100 == 0:
    print(f"Episode {episode}")
    if epsilon < 0.1:
        print("Agenten utforsker nesten ikke lenger")
```

`episode % 100 == 0` er et klassisk triks fra treningsløkker: skriv ut status bare hver 100. episode.

## Valg på én linje
For enkle valg mellom to verdier finnes en kortform: `verdi_hvis_sann if betingelse else verdi_hvis_usann`.

```python
p = 0.31
label = 1 if p > 0.5 else 0
print(label)
```

Denne skrivemåten kommer igjen i leksjon 1.13, der den brukes for å gjøre om en hel liste med sannsynligheter til klasser.

::: tips Tomme og null-verdier regnes som usanne
`if x:` fungerer også når `x` ikke er en `bool`. Da regnes `0`, `0.0`, `""` (tom tekst), `None` og tomme lister som usanne. Alt annet er sant.
:::

Glemmer du innrykket, får du `IndentationError`:

```python feil
p = 0.9
if p > 0.5:
print("klasse 1")
```

::: oppgave Utforske eller utnytte?
En RL-agent bruker *ε-greedy*: hvis et tilfeldig tall er mindre enn `epsilon`, skal den utforske, ellers skal den utnytte det den har lært. Lag variabelen `handling` med teksten `"utforsk"` eller `"utnytt"` ut fra `tilfeldig` og `epsilon`. Lag også `melding`, som skal være `"episode ferdig"` hvis `steg` er større enn eller lik `max_steg` **eller** `i_mål` er `True`, og `"fortsett"` ellers.
```start
tilfeldig = 0.23
epsilon = 0.1
steg = 12
max_steg = 50
i_mål = True

handling = ...
melding = ...
```
```losning
tilfeldig = 0.23
epsilon = 0.1
steg = 12
max_steg = 50
i_mål = True

if tilfeldig < epsilon:
    handling = "utforsk"
else:
    handling = "utnytt"

if steg >= max_steg or i_mål:
    melding = "episode ferdig"
else:
    melding = "fortsett"
```
```test
assert handling == "utnytt", f"handling ble {handling!r}. 0.23 er ikke mindre enn 0.1, så agenten skal utnytte: \"utnytt\"."
assert melding == "episode ferdig", f"melding ble {melding!r}. i_mål er True, så episoden er ferdig selv om steg < max_steg (bruk or)."
```
:::

::: quiz
Hva skriver koden ut?
```kode
acc = 0.95
if acc > 0.5:
    print("A")
elif acc > 0.9:
    print("B")
else:
    print("C")
```
- [x] Bare `A`
- [ ] Bare `B`
- [ ] `A` og `B`
- [ ] `A`, `B` og `C`
? Python kjører bare den første grenen som er sann. `acc > 0.5` er sann, så `elif` og `else` hoppes over.
:::

::: quiz
Hva skriver koden ut?
```kode
x = 3
if x > 5:
    print("stor")
    print("veldig stor")
print("ferdig")
```
- [ ] `stor`, `veldig stor` og `ferdig`
- [ ] `veldig stor` og `ferdig`
- [x] Bare `ferdig`
- [ ] Ingenting
? De to innrykkede linjene hører til `if`-en og hoppes over fordi `3 > 5` er usant. Den siste linjen har ikke innrykk og kjøres alltid.
:::

::: oppsummering
- `if betingelse:` etterfulgt av innrykket kode. Husk kolon og 4 mellomroms innrykk.
- `elif` tester neste betingelse, `else` tar resten. Bare den **første** sanne grenen kjøres.
- `a if betingelse else b` er et valg på én linje.
- `0`, `None`, `""` og tomme lister regnes som usanne i en `if`.
:::

--- LEKSJON 1.7 | lister | Lister
En **liste** holder mange verdier i rekkefølge. Du lager den med hakeparenteser og komma mellom elementene. Lister brukes overalt i ML-kode: tap per epoke, prediksjoner, belønning per episode.

```python
losses = [0.9, 0.5, 0.3, 0.2, 0.15]
print(losses)
print(len(losses))       # antall elementer
print(type(losses))
```

## Indeksering: hente ett element
Elementene er nummerert fra **0**. Negative indekser teller bakfra: `-1` er det siste elementet.

```python
losses = [0.9, 0.5, 0.3, 0.2, 0.15]
print(losses[0])     # første
print(losses[1])     # andre
print(losses[-1])    # siste
print(losses[-2])    # nest siste
```

| Element | 0.9 | 0.5 | 0.3 | 0.2 | 0.15 |
|---|---|---|---|---|---|
| Indeks | 0 | 1 | 2 | 3 | 4 |
| Negativ indeks | -5 | -4 | -3 | -2 | -1 |

Indekserer du utenfor listen, får du `IndexError`. Listen over har fem elementer, så siste gyldige indeks er 4:

```python feil
losses = [0.9, 0.5, 0.3, 0.2, 0.15]
print(losses[5])
```

## Slicing: hente en bit av listen
`liste[start:stopp]` gir elementene fra `start` til, men **ikke med**, `stopp`. Utelater du `start`, begynner den fra starten. Utelater du `stopp`, går den til slutten. Et tredje tall er steglengden.

```python
x = [10, 20, 30, 40, 50, 60]
print(x[1:4])     # indeks 1, 2, 3
print(x[:3])      # de tre første
print(x[3:])      # fra indeks 3 og ut
print(x[-2:])     # de to siste
print(x[::2])     # annethvert element
print(x[::-1])    # baklengs
```

Klikk på `x` i variabelpanelet: hver boks har indeksen sin over seg, så du kan telle deg frem til hva `x[1:4]` gir.

Tekst fungerer på samme måte: `"maskinlæring"[:6]` gir `"maskin"`.

## Endre en liste
Lister kan endres etter at de er laget. For å se listen *underveis* bruker vi `display(x)`: den viser `x` som bokser med indekser akkurat der i utdataene, som et øyeblikksbilde. `display()` finnes også i Jupyter, så koden virker der uendret.

```python
loss_history = []            # tom liste
loss_history.append(0.9)     # legg til på slutten
loss_history.append(0.6)
loss_history.append(0.4)
print("Etter tre append:")
display(loss_history)

loss_history[0] = 1.0        # bytt ut et element
print("Etter loss_history[0] = 1.0:")
display(loss_history)
print(loss_history + [0.3, 0.2])   # + lager en ny, sammenskjøtt liste
```

`append` er måten prosjekt 1 lagrer tapet etter hver iterasjon på: `self.loss_history.append(loss)`.

## Nyttige funksjoner og `in`
```python
y = [1, 0, 1, 1, 0, 1]
print(sum(y), min(y), max(y))
print(sum(y) / len(y))         # andelen enere
print(2 in y)                  # finnes 2 i listen?
print(sorted([3, 1, 2]))       # ny, sortert liste
print(y.count(1))              # hvor mange enere
```

`sorted()` lager en **ny** liste og lar originalen være i fred, mens metoden `.sort()` sorterer selve listen:

```python
scores = [0.7, 0.9, 0.5, 0.8]
rangert = sorted(scores)
print("Etter sorted(): scores er uendret, rangert er ny")
display(scores)
display(rangert)

scores.sort()
print("Etter scores.sort(): selve listen er sortert")
display(scores)
```

## Lister i lister
En liste kan inneholde andre lister. Slik kan du lagre en tabell med data, der hver indre liste er én rad (ett eksempel) og hver kolonne er én feature:

```python
X = [[5.1, 3.5],
     [4.9, 3.0],
     [6.2, 2.9]]
print(X[1])        # rad 1
print(X[1][0])     # rad 1, kolonne 0
print(len(X), len(X[0]))   # antall rader og antall kolonner
```

I variabelpanelet vises `X` som et rutenett med én rad per indre liste. I del 2 lærer du NumPy, som gjør dette mye enklere og raskere med `X[1, 0]`.

::: advarsel To navn, samme liste
`b = a` lager **ikke** en kopi. Begge navnene peker på den samme listen, så endrer du `b`, endres `a` også. Trenger du en kopi, skriv `b = a.copy()`.
```python
a = [1, 2, 3]
b = a
c = a.copy()
b.append(4)
print(a)
print(c)
```
Se i variabelpanelet: både `a` og `b` har fått 4-tallet, mens kopien `c` ikke har det.
:::

::: oppgave Tapshistorikk
Listen `losses` inneholder tapet etter hver epoke. Lag:
- `forste`: det første tapet
- `siste`: det siste tapet (bruk negativ indeks)
- `siste_tre`: en liste med de tre siste tapene (bruk slicing)

Legg deretter til tapet `0.5` på slutten av `losses` med `append`, og lag `antall` med antall elementer i `losses` etter at du la til.
```start
losses = [2.5, 1.8, 1.2, 0.9, 0.7, 0.6, 0.55]

forste = ...
siste = ...
siste_tre = ...

# legg til 0.5 her

antall = ...
```
```losning
losses = [2.5, 1.8, 1.2, 0.9, 0.7, 0.6, 0.55]

forste = losses[0]
siste = losses[-1]
siste_tre = losses[-3:]

losses.append(0.5)

antall = len(losses)
```
```test
assert forste == 2.5, "forste skal være losses[0], altså 2.5."
assert siste == 0.55, "siste skal være det siste tapet FØR du la til 0.5: losses[-1] = 0.55."
assert siste_tre == [0.7, 0.6, 0.55], f"siste_tre ble {siste_tre}, men skal være [0.7, 0.6, 0.55]. Prøv losses[-3:]."
assert losses[-1] == 0.5 and len(losses) == 8, "Legg til 0.5 på slutten med losses.append(0.5)."
assert antall == 8, "antall skal være len(losses) etter append, altså 8."
```
:::

::: quiz
Hva skriver koden ut?
```kode
x = [10, 20, 30, 40, 50]
print(x[1:3])
```
- [ ] `[10, 20, 30]`
- [x] `[20, 30]`
- [ ] `[20, 30, 40]`
- [ ] `[10, 20]`
? Indekseringen starter på 0, så indeks 1 er 20. Stopp-indeksen 3 er ikke med, så vi får indeks 1 og 2.
:::

::: quiz
Hva skriver koden ut?
```kode
a = [1, 2, 3]
b = a
b.append(4)
print(len(a))
```
- [ ] 3
- [x] 4
- [ ] 7
- [ ] `NameError`
? `b = a` gir et nytt navn til *samme* liste. `b.append(4)` endrer derfor også `a`.
:::

::: oppsummering
- `[a, b, c]` lager en liste. `len()` gir antall elementer.
- Indekser starter på 0. `x[-1]` er siste element. Utenfor listen gir `IndexError`.
- `x[start:stopp]` tar med `start`, men ikke `stopp`. `x[:3]` er de tre første, `x[-3:]` de tre siste.
- `.append()` legger til på slutten. `in` sjekker om noe finnes.
- `sum`, `min`, `max`, `sorted` og `.count()` er nyttige.
- `b = a` kopierer ikke. Bruk `a.copy()`.
:::

--- LEKSJON 1.8 | for-lokker | for-løkker
En **for-løkke** gjentar en kodeblokk én gang for hvert element i en liste. Som i `if` slutter linjen med kolon, og koden som skal gjentas, er rykket inn.

```python
losses = [0.9, 0.5, 0.3]
for loss in losses:
    print("Tapet er", loss)
print("Ferdig")
```

Variabelen `loss` får verdien til hvert element etter tur. Navnet velger du selv.

## range: løkke over tall
`range(n)` gir tallene `0, 1, ..., n-1`. Det er måten man gjentar noe et bestemt antall ganger, for eksempel et antall epoker eller episoder.

```python
for epoch in range(3):
    print("Epoke", epoch)

print(list(range(2, 6)))        # fra 2 til (men ikke med) 6
print(list(range(0, 20, 5)))    # steglengde 5
```

`list(range(...))` gjør tallene om til en liste, slik at du kan se dem.

## Akkumulering: bygge opp et resultat
Et svært vanlig mønster: start med en tom verdi før løkka, og oppdater den inne i løkka.

```python
rewards = [-1, -1, -1, 10]

total = 0
for r in rewards:
    total += r
print("Total belønning:", total)

kvadrater = []
for r in rewards:
    kvadrater.append(r ** 2)
print(kvadrater)
```

## enumerate: indeks og element samtidig
Trenger du både nummeret og verdien, bruker du `enumerate`. Den gir to verdier hver runde:

```python
losses = [0.9, 0.5, 0.3]
for i, loss in enumerate(losses):
    print(f"Epoke {i}: tap {loss}")

for i, loss in enumerate(losses, start=1):   # start nummereringen på 1
    print(f"Epoke {i}: tap {loss}")
```

## zip: gå gjennom to lister i takt
`zip` parer elementene fra to (eller flere) lister. Det er perfekt for å sammenligne fasit og prediksjoner:

```python
y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 0, 1, 1]

riktige = 0
for sann, pred in zip(y_true, y_pred):
    if sann == pred:
        riktige += 1
print(f"Accuracy: {riktige / len(y_true):.2f}")
```

Hvis listene er ulikt lange, stopper `zip` ved den korteste.

## Løkke i løkke
Med en løkke inni en annen kan du prøve alle kombinasjoner, for eksempel av to hyperparametere. Dette er nøyaktig hva et *grid search* gjør (del 7).

```python
for lr in [0.1, 0.01]:
    for depth in [2, 4, 8]:
        print(f"lr={lr}, depth={depth}")
```

::: oppgave Accuracy med zip
Regn ut hvor mange prediksjoner som er riktige, og lagre det i `riktige`. Regn så ut `accuracy` som andelen riktige (et tall mellom 0 og 1). Bruk en løkke med `zip`.
```start
y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0]

riktige = 0
# skriv løkka her

accuracy = ...
```
```losning
y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0]

riktige = 0
for t, p in zip(y_true, y_pred):
    if t == p:
        riktige += 1

accuracy = riktige / len(y_true)
```
```test
assert riktige == 6, f"riktige ble {riktige}, men 6 av 8 prediksjoner er riktige. Øk riktige med 1 når t == p."
assert isinstance(accuracy, float) and abs(accuracy - 0.75) < 1e-9, "accuracy skal være riktige / len(y_true) = 0.75."
```
:::

::: oppgave Mean squared error med løkke
Regn ut *mean squared error* (MSE): gjennomsnittet av `(sann - pred) ** 2` over alle punktene. Lagre summen av kvadrerte feil i `sum_kvadrat` og gjennomsnittet i `mse`.
```start
y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]

sum_kvadrat = 0
# skriv løkka her

mse = ...
```
```losning
y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]

sum_kvadrat = 0
for sann, pred in zip(y_true, y_pred):
    sum_kvadrat += (sann - pred) ** 2

mse = sum_kvadrat / len(y_true)
```
```test
assert abs(sum_kvadrat - 1.5) < 1e-9, f"sum_kvadrat ble {sum_kvadrat}, men skal være 0.25 + 0.25 + 0 + 1 = 1.5."
assert isinstance(mse, float) and abs(mse - 0.375) < 1e-9, "mse skal være sum_kvadrat / antall punkter = 0.375."
```
:::

::: quiz
Hva skriver koden ut?
```kode
for i in range(1, 10, 3):
    print(i)
```
- [x] `1`, `4`, `7`
- [ ] `1`, `4`, `7`, `10`
- [ ] `1`, `2`, `3`
- [ ] `3`, `6`, `9`
? `range(1, 10, 3)` starter på 1, øker med 3 og stopper *før* 10.
:::

::: quiz
Hva skriver koden ut?
```kode
total = 0
for i, x in enumerate([5, 6, 7]):
    total += i * x
print(total)
```
- [ ] 18
- [x] 20
- [ ] 38
- [ ] 6
? `enumerate` starter på 0: `0·5 + 1·6 + 2·7 = 0 + 6 + 14 = 20`.
:::

::: oppsummering
- `for x in liste:` kjører den innrykkede koden én gang per element.
- `range(n)` gir `0, ..., n-1`. `range(a, b, steg)` gir fra `a` til (ikke med) `b`.
- Akkumulering: lag `total = 0` eller `resultat = []` før løkka, oppdater inni.
- `enumerate(liste)` gir `(indeks, element)`. `zip(a, b)` gir parene `(a[i], b[i])`.
- Løkker kan nøstes, for eksempel for å prøve alle kombinasjoner av hyperparametere.
:::

--- LEKSJON 1.9 | while-break-continue | while, break og continue
En `for`-løkke går gjennom en liste eller et fast antall runder. En **while-løkke** gjentar så lenge en betingelse er sann. Den er nyttig når du ikke vet på forhånd hvor mange runder du trenger.

```python
epsilon = 1.0
episode = 0
while epsilon > 0.05:
    epsilon *= 0.9
    episode += 1
print(f"Etter {episode} episoder er epsilon {epsilon:.3f}")
```

Betingelsen sjekkes før hver runde. Når den blir usann, fortsetter programmet etter løkka.

## Gradient descent til det konvergerer
Her minimerer vi funksjonen `(w - 3) ** 2`, som har minimum i `w = 3`. Den deriverte er `2 * (w - 3)`. Vi tar steg helt til vi er nær nok:

```python
w = 0.0
lr = 0.1
steg = 0
while abs(w - 3) > 0.01:
    gradient = 2 * (w - 3)
    w = w - lr * gradient
    steg += 1
print(f"w = {w:.4f} etter {steg} steg")
```

::: advarsel Uendelige løkker
Hvis betingelsen aldri blir usann, går løkka for alltid, og nettleseren (eller Jupyter) henger. Sørg for at noe i løkka endrer betingelsen. En god vane er å ha et maksimalt antall runder i tillegg, slik eksempelet under gjør med `break`.
:::

## break: hopp ut av løkka
`break` avslutter løkka med en gang, også en `for`-løkke. Dette er nøyaktig mønsteret i RL: kjør maks et visst antall steg, men stopp når episoden er ferdig.

```python
rewards = [-1, -1, -1, 10, -1, -1]   # belønning agenten får i hvert steg
max_steps = 50
total = 0
for step in range(max_steps):
    reward = rewards[step]
    total += reward
    done = reward == 10              # målet er nådd
    if done:
        break
print(f"Ferdig etter {step + 1} steg, total belønning {total}")
```

Uten `break` ville løkka prøvd `rewards[6]` og krasjet med `IndexError`.

## continue: hopp til neste runde
`continue` hopper over resten av koden i *denne* runden og går videre til neste. Nyttig for å hoppe over ugyldige verdier:

```python
målinger = [2.0, None, 3.5, None, 4.5]
total = 0
antall = 0
for m in målinger:
    if m is None:
        continue          # hopp over manglende verdier
    total += m
    antall += 1
print(f"Snitt av {antall} gyldige målinger: {total / antall}")
```

## while True med break
Noen ganger er det enklest å skrive `while True:` (en løkke som i utgangspunktet aldri stopper) og bruke `break` når man er ferdig:

```python
n = 1
while True:
    n *= 2
    if n > 100:
        break
print(n)
```

::: oppgave Hvor mange steg til konvergens?
Bruk en `while`-løkke til å minimere `(w - 3) ** 2` med gradient descent, startende i `w = 0.0` med `lr = 0.1`. Fortsett så lenge `abs(w - 3)` er større enn eller lik `0.001`. Tell antall steg i `steg`.
```start
w = 0.0
lr = 0.1
steg = 0
# skriv while-løkka her

```
```losning
w = 0.0
lr = 0.1
steg = 0
while abs(w - 3) >= 0.001:
    gradient = 2 * (w - 3)
    w = w - lr * gradient
    steg += 1
```
```test
assert abs(w - 3) < 0.001, f"w ble {w}. Løkka skal fortsette til w er nærmere 3 enn 0.001."
assert steg == 36, f"steg ble {steg}, men skal være 36. Husk å øke steg med 1 i hver runde, og bruk gradient = 2 * (w - 3)."
```
:::

::: oppgave Early stopping med break
Valideringstapet går først ned og så opp igjen (overtilpasning). Finn den første epoken `i` der `val_losses[i]` er **større** enn `val_losses[i - 1]`, lagre den i `stopp_epoke` og bruk `break` for å avslutte løkka.
```start
val_losses = [1.0, 0.8, 0.65, 0.6, 0.62, 0.7, 0.58]

stopp_epoke = None
for i in range(1, len(val_losses)):
    ...
```
```losning
val_losses = [1.0, 0.8, 0.65, 0.6, 0.62, 0.7, 0.58]

stopp_epoke = None
for i in range(1, len(val_losses)):
    if val_losses[i] > val_losses[i - 1]:
        stopp_epoke = i
        break
```
```test
assert stopp_epoke is not None, "stopp_epoke er fortsatt None. Sett stopp_epoke = i når val_losses[i] > val_losses[i - 1]."
assert stopp_epoke == 4, f"stopp_epoke ble {stopp_epoke}, men tapet øker første gang i epoke 4 (0.6 -> 0.62). Husk break, ellers overskrives verdien."
```
:::

::: quiz
Hva skriver koden ut?
```kode
n = 0
while n < 10:
    n += 3
print(n)
```
- [ ] 9
- [ ] 10
- [x] 12
- [ ] 3
? Løkka kjører med n = 0, 3, 6, 9 (alle er mindre enn 10). I siste runde blir n = 12, og da er betingelsen usann.
:::

::: quiz
Hva skriver koden ut?
```kode
for x in [1, 2, 3, 4, 5]:
    if x == 2:
        continue
    if x == 4:
        break
    print(x)
```
- [ ] `1`, `2`, `3`
- [x] `1`, `3`
- [ ] `1`, `3`, `5`
- [ ] `1`
? For x = 2 hopper `continue` over `print`. For x = 4 avslutter `break` hele løkka, så 4 og 5 skrives aldri ut.
:::

::: oppsummering
- `while betingelse:` gjentar så lenge betingelsen er sann. Pass på at den til slutt blir usann.
- `break` avslutter løkka med en gang (som når en episode er ferdig).
- `continue` hopper til neste runde.
- Mønsteret `for step in range(max_steps): ... if done: break` er kjernen i RL-løkka.
:::

--- LEKSJON 1.10 | dictionaries | Dictionaries
En **dictionary** (`dict`) kobler **nøkler** til **verdier**, som et oppslagsverk. I stedet for å slå opp med nummer som i en liste, slår du opp med en nøkkel, ofte en tekst.

```python
params = {"max_depth": 3, "criterion": "gini", "random_state": 42}
print(params["max_depth"])
print(params["criterion"])
print(len(params))
```

Nøkkel og verdi skilles med kolon, og parene med komma. Hyperparametere, resultater og tellinger lagres ofte slik.

## Legge til og endre
Du legger til eller endrer en verdi med `d[nøkkel] = verdi`. `in` sjekker om en **nøkkel** finnes:

```python
params = {"max_depth": 3}
params["max_depth"] = 5             # endre
params["min_samples_leaf"] = 2      # legg til ny nøkkel
print(params)
print("max_depth" in params)
print("criterion" in params)
```

Slår du opp en nøkkel som ikke finnes, får du `KeyError`:

```python feil
params = {"max_depth": 3}
print(params["criterion"])
```

`params.get("criterion", "gini")` gir i stedet standardverdien `"gini"` hvis nøkkelen mangler. Uten standardverdi gir `.get()` bare `None`.

## Telle forekomster
Det viktigste dict-mønsteret i kurset er **telling**. `antall.get(etikett, 0)` gir antallet så langt, eller 0 første gang vi ser etiketten:

```python
y = ["katt", "hund", "katt", "fugl", "katt", "hund"]

antall = {}
for etikett in y:
    antall[etikett] = antall.get(etikett, 0) + 1
print(antall)
```

Vil du se telleren vokse, kan du vise dicten etter hvert steg med `display()`. Hver nye etikett gir en ny rad, og en etikett vi har sett før får tallet sitt økt med 1:

```python
y = ["katt", "hund", "katt", "fugl"]

antall = {}
for etikett in y:
    antall[etikett] = antall.get(etikett, 0) + 1
    print("Etter", etikett)
    display(antall)
```

Akkurat dette trenger du i Naive Bayes (del 8), der man teller hvor ofte hver klasse og hver feature-verdi forekommer.

## Gå gjennom en dict
`.keys()` gir nøklene, `.values()` gir verdiene og `.items()` gir parene. Med `.items()` får du både nøkkel og verdi i løkka:

```python
antall = {"katt": 3, "hund": 2, "fugl": 1}
n = sum(antall.values())

for klasse, k in antall.items():
    print(f"{klasse:6s} {k} av {n}  ->  P({klasse}) = {k / n:.2f}")

print(list(antall.keys()))
```

## Verdier kan være hva som helst
Verdiene kan være lister eller nye dicts. En dict med lister er en vanlig måte å lagre treningshistorikk på (Keras gjør akkurat dette i `history.history`):

```python
history = {"loss": [], "val_loss": []}
for epoch in range(3):
    history["loss"].append(1.0 / (epoch + 1))
    history["val_loss"].append(1.2 / (epoch + 1))
print(history)
```

En dict i en dict kan brukes som en liten **Q-tabell** i forsterkningslæring: for hver tilstand, en verdi per handling.

```python
Q = {
    "start": {"venstre": 0.0, "høyre": 0.0},
    "midt":  {"venstre": 0.0, "høyre": 0.0},
}
Q["start"]["høyre"] = 0.5      # agenten har lært at høyre er lurt i start
print(Q["start"])
print(Q["start"]["høyre"] > Q["start"]["venstre"])
```

I prosjekt 4 er Q-tabellen en NumPy-matrise i stedet (rad = tilstand, kolonne = handling), men ideen er den samme.

::: oppgave Tell klassene
Tell hvor mange ganger hver etikett forekommer i `y`, og lagre det i dicten `antall` (nøkkel = etikett, verdi = antall). Bruk en løkke og `.get(..., 0)`.
```start
y = ["spam", "ham", "ham", "spam", "ham", "ham"]

antall = {}
for etikett in y:
    ...  # oppdater antall her
```
```losning
y = ["spam", "ham", "ham", "spam", "ham", "ham"]

antall = {}
for etikett in y:
    antall[etikett] = antall.get(etikett, 0) + 1
```
```test
assert isinstance(antall, dict), "antall skal være en dict."
assert antall == {"spam": 2, "ham": 4}, f"antall ble {antall}, men skal være {{'spam': 2, 'ham': 4}}. Skriv antall[etikett] = antall.get(etikett, 0) + 1."
```
:::

::: oppgave Fra antall til andeler
Gitt tellingene i `antall`, lag en ny dict `andeler` der hver klasse har andelen sin (antall delt på totalen). Dette er *prior*-sannsynlighetene i Naive Bayes. Bruk `.items()`.
```start
antall = {"spam": 2, "ham": 4, "phishing": 2}
total = sum(antall.values())

andeler = {}
# fyll inn andeler her
```
```losning
antall = {"spam": 2, "ham": 4, "phishing": 2}
total = sum(antall.values())

andeler = {}
for klasse, k in antall.items():
    andeler[klasse] = k / total
```
```test
assert isinstance(andeler, dict) and set(andeler) == {"spam", "ham", "phishing"}, f"andeler skal ha nøklene spam, ham og phishing, men har {list(andeler)}."
assert abs(andeler["spam"] - 0.25) < 1e-9 and abs(andeler["ham"] - 0.5) < 1e-9 and abs(andeler["phishing"] - 0.25) < 1e-9, f"andeler ble {andeler}. Del hvert antall på total (8)."
```
:::

::: quiz
Hva skriver koden ut?
```kode
d = {"a": 1, "b": 2}
d["a"] = d.get("c", 0) + d["b"]
print(d)
```
- [ ] `{'a': 1, 'b': 2}`
- [x] `{'a': 2, 'b': 2}`
- [ ] `{'a': 3, 'b': 2}`
- [ ] `KeyError`
? `"c"` finnes ikke, så `d.get("c", 0)` gir 0. Da blir `d["a"] = 0 + 2 = 2`. Det er bare `d["c"]` (uten `.get`) som ville gitt `KeyError`.
:::

::: oppsummering
- `{nøkkel: verdi, ...}` lager en dict. `d[k]` slår opp, `d[k] = v` legger til eller endrer.
- `k in d` sjekker om nøkkelen finnes. Manglende nøkkel med `d[k]` gir `KeyError`.
- `d.get(k, 0)` gir 0 hvis nøkkelen mangler: perfekt for telling.
- `.keys()`, `.values()` og `.items()` brukes i løkker. `for k, v in d.items():` gir begge.
- Verdier kan være lister eller dicts, for eksempel treningshistorikk eller en Q-tabell.
:::

--- LEKSJON 1.11 | tupler | Tupler og utpakking
En **tuple** er som en liste, men skrives med vanlige parenteser og kan **ikke endres** etter at den er laget. Den brukes for ting som hører sammen i en fast rekkefølge, som en posisjon `(rad, kolonne)` eller formen på en tabell `(antall_rader, antall_kolonner)`.

```python
posisjon = (3, 7)
print(posisjon[0], posisjon[1])
print(len(posisjon))

shape = (150, 4)    # 150 eksempler, 4 features
print(f"{shape[0]} rader og {shape[1]} kolonner")
```

Prøver du å endre en tuple, får du `TypeError`:

```python feil
posisjon = (3, 7)
posisjon[0] = 4
```

Det er faktisk **kommaet** som lager tuplen, ikke parentesene: `t = 1, 2` gir også tuplen `(1, 2)`.

## Utpakking
Med **utpakking** gir du flere variabler verdi på én gang. Antall navn på venstre side må stemme med antall verdier på høyre side.

```python
rad, kol = (3, 7)
print(rad)
print(kol)

a, b = 1, 2
a, b = b, a          # bytt verdiene
print(a, b)

n_samples, n_features = (150, 4)
print(n_samples, n_features)
```

Det siste er nøyaktig hvordan man skriver `n_samples, n_features = X.shape` med NumPy.

```python feil
rad, kol = (3, 7, 1)
```

## Funksjoner som returnerer flere verdier
Når en funksjon skal gi tilbake flere ting, returnerer den en tuple, og du pakker den ut. Hvordan du lager egne funksjoner, lærer du i neste leksjon. Her er det nok å vite at `return a, b` gir tilbake tuplen `(a, b)`:

```python
def min_og_maks(verdier):
    return min(verdier), max(verdier)

laveste, høyeste = min_og_maks([4, 1, 9, 3])
print(laveste, høyeste)
```

Dette mønsteret ser du overalt i ML-bibliotekene:

```python statisk ikkekjor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
obs, reward, terminated, info = env.step(action)          # prosjekt 4, TrondheimEnv
state, _ = env.reset()                                     # gymnasium
next_state, reward, done, truncated, _ = env.step(action)  # gymnasium
```

## `_` for verdier du ikke trenger
Navnet `_` er en vanlig variabel, men brukes etter avtale for verdier man vil kaste. Her later vi som om `steg` er et miljø som returnerer fire verdier:

```python
def steg(handling):
    ny_tilstand = handling + 1
    reward = -1.0
    terminated = False
    info = {}
    return ny_tilstand, reward, terminated, info

state, reward, _, _ = steg(0)
print(state, reward)

for _ in range(3):      # _ når løkkevariabelen ikke brukes
    print("hei")
```

## Utpakking i løkker
Du har allerede brukt utpakking: `for i, x in enumerate(...)` og `for k, v in d.items()` pakker ut en tuple hver runde. Det går an å pakke ut i flere nivåer:

```python
resultater = [("tre", 0.81), ("skog", 0.88)]
for navn, acc in resultater:
    print(f"{navn}: {acc}")

xs = [1.0, 2.0]
ys = [3.0, 5.0]
for i, (x, y) in enumerate(zip(xs, ys)):
    print(i, x, y)
```

## Stjerne-utpakking
Med `*` foran et navn samles «resten» i en liste:

```python
første, *resten = [10, 20, 30, 40]
print(første)
print(resten)
```

::: oppgave Pakk ut en transisjon
En replay-buffer lagrer *transisjoner* som tupler `(state, action, reward, next_state, done)`. Pakk ut `transisjon` til variablene `state`, `reward`, `next_state` og `done` på én linje. Handlingen trenger du ikke, så bruk `_` for den.
```start
transisjon = (4, 2, -1.0, 5, False)

# pakk ut her
state = reward = next_state = done = ...
```
```losning
transisjon = (4, 2, -1.0, 5, False)

state, _, reward, next_state, done = transisjon
```
```test
assert state == 4, "state skal være første element i transisjonen (4)."
assert reward == -1.0, "reward skal være tredje element (-1.0). Har du husket _ for handlingen?"
assert next_state == 5, "next_state skal være fjerde element (5)."
assert done is False, "done skal være siste element (False)."
```
:::

::: oppgave Finn beste modell
`resultater` er en liste med tupler `(navn, accuracy)`. Gå gjennom listen med utpakking i `for`-løkka, og finn modellen med høyest accuracy. Lagre navnet i `beste_navn` og accuracy i `beste_acc`.
```start
resultater = [("tre", 0.81), ("skog", 0.88), ("knn", 0.79), ("bayes", 0.84)]

beste_navn = None
beste_acc = 0.0
# skriv løkka her
```
```losning
resultater = [("tre", 0.81), ("skog", 0.88), ("knn", 0.79), ("bayes", 0.84)]

beste_navn = None
beste_acc = 0.0
for navn, acc in resultater:
    if acc > beste_acc:
        beste_navn = navn
        beste_acc = acc
```
```test
assert beste_navn == "skog", f"beste_navn ble {beste_navn!r}, men skal være 'skog' (0.88). Oppdater både navn og acc når du finner en bedre."
assert beste_acc == 0.88, f"beste_acc ble {beste_acc}, men skal være 0.88."
```
:::

::: quiz
Et miljø returnerer fem verdier fra `step`. Hva skjer når koden kjøres?
```kode
resultat = (3, -1.0, False, False, {})
obs, reward, terminated, info = resultat
```
- [ ] `info` blir `{}`
- [ ] `info` blir `(False, {})`
- [x] `ValueError: too many values to unpack`
- [ ] Den siste verdien kastes stille
? Antall navn må stemme nøyaktig med antall verdier. Dette er en klassisk feil når man bytter fra gamle `gym` (4 verdier) til `gymnasium` (5 verdier: `obs, reward, terminated, truncated, info`).
:::

::: quiz
Hva skriver koden ut?
```kode
a, *b = [1, 2, 3, 4]
print(b)
```
- [ ] `2`
- [x] `[2, 3, 4]`
- [ ] `(2, 3, 4)`
- [ ] `ValueError`
? `*b` samler alle verdiene som er igjen etter `a` i en liste.
:::

::: oppsummering
- En tuple `(a, b)` er en liste som ikke kan endres.
- Utpakking: `a, b = (1, 2)`. Antall navn må stemme, ellers `ValueError`.
- `return a, b` returnerer en tuple: `X_train, X_test, y_train, y_test = train_test_split(...)`.
- `_` brukes for verdier du ikke trenger: `state, _ = env.reset()`.
- `for i, (x, y) in enumerate(zip(xs, ys)):` pakker ut i løkker. `første, *resten = liste` samler resten.
:::

--- LEKSJON 1.12 | funksjoner | Funksjoner
En **funksjon** er en navngitt kodebit du kan bruke om igjen. Du har brukt mange allerede (`print`, `len`, `sum`). Nå skal du lage dine egne med `def`.

```python
def kvadrat(x):
    return x ** 2

print(kvadrat(3))
print(kvadrat(1.5))
```

- `def` etterfulgt av navnet og **parametrene** i parentes, så kolon.
- Funksjonskroppen er rykket inn.
- `return` sender et resultat tilbake til den som kalte funksjonen, og avslutter funksjonen.

Koden i funksjonen kjøres først når du **kaller** den, som `kvadrat(3)`. Verdien du sender inn (3) kalles et **argument**.

## return og print er ikke det samme
`print` viser noe på skjermen. `return` gir en verdi tilbake som du kan lagre og regne videre med. En funksjon uten `return` gir tilbake `None`.

```python
def med_print(x):
    print(x * 2)

def med_return(x):
    return x * 2

a = med_print(5)
b = med_return(5)
print("a =", a)
print("b =", b)
```

## Flere parametre: en MSE-funksjon
```python
def mse(y_true, y_pred):
    total = 0
    for t, p in zip(y_true, y_pred):
        total += (t - p) ** 2
    return total / len(y_true)

print(mse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]))
print(mse([1.0, 2.0, 3.0], [2.0, 2.0, 5.0]))
```

Variabler som lages inne i en funksjon (som `total`), er **lokale**: de finnes bare mens funksjonen kjører, og er usynlige utenfor.

## Standardverdier og nøkkelordargumenter
En parameter kan ha en **standardverdi**, som brukes hvis du ikke sender inn noe. Når du kaller funksjonen, kan du angi argumenter med navn (**nøkkelordargumenter**). Da spiller rekkefølgen ingen rolle, og koden blir lettere å lese.

```python
def lag_modell(lr=0.001, n_iterations=1000, verbose=False):
    return f"Modell(lr={lr}, n_iterations={n_iterations}, verbose={verbose})"

print(lag_modell())
print(lag_modell(0.1))
print(lag_modell(n_iterations=50))
print(lag_modell(verbose=True, lr=0.5))
```

Dette er nøyaktig det du gjør med sklearn: `DecisionTreeClassifier(max_depth=3, random_state=42)`. Alle de andre parametrene får standardverdier.

Regel: parametre uten standardverdi må stå før dem med, og posisjonelle argumenter må komme før nøkkelordargumenter i et kall.

## Docstring og type hints
En **docstring** er en tekst i tre anførselstegn rett under `def`. Den forklarer hva funksjonen gjør, og vises av `help()`. **Type hints** (`episode: int`, `-> float`) forteller leseren hvilke typer som er ment. Python sjekker dem ikke, de er bare dokumentasjon.

```python
def update_explore_rate(episode: int, epsilon_min: float, decay: float = 0.99) -> float:
    """Returnerer utforskingsraten for en gitt episode.

    Raten starter på 1 og synker eksponentielt, men aldri under epsilon_min.
    """
    return max(epsilon_min, decay ** episode)

print(update_explore_rate(0, 0.01))
print(update_explore_rate(100, 0.01))
print(update_explore_rate(1000, 0.01))
help(update_explore_rate)
```

::: prosjekt Prosjekt 4
Prosjekt 4 gir deg funksjoner med akkurat denne formen, og du skal fylle inn kroppen:
```python statisk ikkekjor
def update_explore_rate(episode: int, epsilon_min: float, max_episodes: int, epsilon_max: float = 1.0):
    """controls the balance between exploration and exploitation"""
    # YOUR CODE GOES HERE: return updated explore_rate
```
:::

## Funksjoner er også verdier
Du kan sende en funksjon inn til en annen funksjon (uten parenteser, ellers kaller du den). Slik velger man for eksempel aktiveringsfunksjon eller tapsfunksjon:

```python
def relu(z):
    return max(0, z)

def identitet(z):
    return z

def lag_utdata(aktivering, verdier):
    resultat = []
    for v in verdier:
        resultat.append(aktivering(v))
    return resultat

print(lag_utdata(relu, [-2, -1, 0, 1, 2]))
print(lag_utdata(identitet, [-2, -1, 0, 1, 2]))
```

## `*args`, `**kwargs` og `**cfg`
Noen ganger vil man ta imot et ukjent antall argumenter. `*args` samler alle ekstra posisjonelle argumenter i en tuple, og `**kwargs` samler alle ekstra nøkkelordargumenter i en dict:

```python
def vis(*args, **kwargs):
    print("args:", args)
    print("kwargs:", kwargs)

vis(1, 2, lr=0.01, epochs=5)
```

Det motsatte er like nyttig: `**` foran en dict i et **kall** pakker dicten ut til nøkkelordargumenter. Da kan du lagre et helt oppsett i en dict og sende det inn:

```python
def train(lr=0.01, epochs=10, batch_size=32):
    return f"lr={lr}, epochs={epochs}, batch_size={batch_size}"

cfg = {"lr": 0.001, "batch_size": 64}
print(train(**cfg))              # samme som train(lr=0.001, batch_size=64)
print(train(**cfg, epochs=200))
```

::: prosjekt Prosjekt 3
Løsningsforslaget til prosjekt 3 lagrer hvert eksperiment som en dict og kjører dem med `**cfg`:
```python statisk ikkekjor
configs = [
    dict(act=nn.ReLU, width=64, depth=3, lr=3e-3, batch_size=320),
    dict(act=nn.Tanh, width=64, depth=3, lr=3e-3, batch_size=320),
]
for cfg in configs:
    models, _ = reg_ensemble(**cfg, epochs=800)
```
`dict(a=1, b=2)` er bare en annen måte å skrive `{"a": 1, "b": 2}` på. En `*` alene i parameterlisten, som i `def train_net(X, y, train_idx, val_idx, *, act, width, ...)`, betyr at alt etter den *må* gis som nøkkelordargumenter.
:::

::: oppgave Skriv mse som funksjon
Lag funksjonen `mse(y_true, y_pred)` som **returnerer** (ikke skriver ut) gjennomsnittet av de kvadrerte feilene. Den skal virke for lister av alle lengder.
```start
def mse(y_true, y_pred):
    # regn ut og returner MSE
    return None
```
```losning
def mse(y_true, y_pred):
    total = 0
    for t, p in zip(y_true, y_pred):
        total += (t - p) ** 2
    return total / len(y_true)
```
```test
r = mse([3.0, -0.5, 2.0, 7.0], [2.5, 0.0, 2.0, 8.0])
assert r is not None, "mse returnerer None. Har du husket return?"
assert abs(r - 0.375) < 1e-9, f"mse([3.0, -0.5, 2.0, 7.0], [2.5, 0.0, 2.0, 8.0]) ga {r}, men skal være 0.375."
assert abs(mse([1, 2], [1, 2])) < 1e-9, "Når prediksjonene er perfekte, skal mse være 0."
assert abs(mse([0.0], [2.0]) - 4.0) < 1e-9, "mse([0.0], [2.0]) skal være 4.0. Del på antall punkter, ikke på et fast tall."
```
:::

::: oppgave Prediker klasse med terskel
Lag funksjonen `predict_label(p, threshold=0.5)` som returnerer `1` hvis `p` er større enn eller lik `threshold`, og `0` ellers.
```start
def predict_label(p, threshold=0.5):
    # returner 1 eller 0
    return None
```
```losning
def predict_label(p, threshold=0.5):
    return 1 if p >= threshold else 0
```
```test
assert predict_label(0.7) == 1, "predict_label(0.7) skal gi 1 (0.7 >= 0.5)."
assert predict_label(0.2) == 0, "predict_label(0.2) skal gi 0."
assert predict_label(0.5) == 1, "predict_label(0.5) skal gi 1, fordi grensen er tatt med (>=)."
assert predict_label(0.7, threshold=0.8) == 0, "predict_label(0.7, threshold=0.8) skal gi 0. Bruk parameteren threshold, ikke 0.5 direkte."
assert predict_label(0.3, 0.25) == 1, "predict_label(0.3, 0.25) skal gi 1."
```
:::

::: quiz
Hva skriver koden ut?
```kode
def f(x):
    y = x * 2
    print(y)

resultat = f(3)
print(resultat)
```
- [ ] `6` og `6`
- [x] `6` og `None`
- [ ] Bare `6`
- [ ] `NameError`
? `f` skriver ut 6, men har ingen `return`, så den gir tilbake `None`. Derfor skriver den andre `print` ut `None`.
:::

::: quiz
Hva skriver koden ut?
```kode
def f(a, b=2, c=3):
    return a + b + c

cfg = {"c": 10}
print(f(1, **cfg))
```
- [ ] 6
- [x] 13
- [ ] 14
- [ ] `TypeError`
? `**cfg` blir til `c=10`. Kallet er altså `f(1, c=10)`, og `b` får standardverdien 2: `1 + 2 + 10 = 13`.
:::

::: oppsummering
- `def navn(parametre):` lager en funksjon. `return` gir en verdi tilbake. Uten `return` får du `None`.
- Standardverdier: `def f(x, lr=0.01)`. Nøkkelordargumenter: `f(x, lr=0.1)`.
- Docstring i `"""..."""` forklarer funksjonen. Type hints (`episode: int`, `-> float`) er dokumentasjon.
- Funksjoner kan sendes inn som argumenter.
- `*args` samler posisjonelle argumenter i en tuple, `**kwargs` samler nøkkelordargumenter i en dict.
- `f(**cfg)` pakker ut en dict til nøkkelordargumenter.
:::

--- LEKSJON 1.13 | comprehensions-lambda | List comprehensions og lambda
Å lage en ny liste ved å gjøre noe med hvert element i en annen liste er så vanlig at Python har en egen kortform for det: en **list comprehension**. Disse to kodebitene gjør nøyaktig det samme:

```python
tall = [1, 2, 3, 4]

kvadrater = []
for x in tall:
    kvadrater.append(x ** 2)
print(kvadrater)

kvadrater = [x ** 2 for x in tall]
print(kvadrater)
```

Les det som «`x ** 2` for hver `x` i `tall`». Uttrykket først, så løkka, alt inne i hakeparenteser.

## Filtrere med if
Legger du til `if` på slutten, tas bare elementene som oppfyller betingelsen med:

```python
losses = [0.9, None, 0.5, 0.3, None]
gyldige = [l for l in losses if l is not None]
print(gyldige)

partall = [i for i in range(10) if i % 2 == 0]
print(partall)
```

## Fra sannsynligheter til klasser
Kombinerer du med valget på én linje fra leksjon 1.6, får du denne linjen, som står i forelesningsnotatene. Den gjør om sannsynligheter til klasser med terskel 0.5:

```python
y_pred = [0.12, 0.85, 0.51, 0.49, 0.97]
y_pred_class = [1 if p > 0.5 else 0 for p in y_pred]
print(y_pred_class)
```

Legg merke til forskjellen: `if` **etter** løkka filtrerer bort elementer, mens `a if ... else b` **foran** løkka velger mellom to verdier for hvert element.

## Over flere lister, og rett inn i sum
Med `zip` kan du gå over to lister. Bruker du vanlige parenteser rett inn i `sum`, `min` eller `max`, slipper du hakeparentesene:

```python
y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]

feil = [(t - p) ** 2 for t, p in zip(y_true, y_pred)]
print(feil)
print("MSE:", sum(feil) / len(feil))

y_true_k = [1, 0, 1, 1]
y_pred_k = [1, 1, 1, 0]
print("Accuracy:", sum(t == p for t, p in zip(y_true_k, y_pred_k)) / len(y_true_k))
```

## Dict comprehension
Med krøllparenteser og `nøkkel: verdi` lager du en dict på samme måte:

```python
antall = {"katt": 3, "hund": 2, "fugl": 1}
n = sum(antall.values())
andeler = {klasse: k / n for klasse, k in antall.items()}
print(andeler)

lengder = {navn: len(navn) for navn in ["tre", "skog", "knn"]}
print(lengder)
```

Du kan også lage en Q-tabell som liste i liste: `[[0.0] * n_actions for _ in range(n_states)]` gir én rad med nuller for hver tilstand.

## lambda: små funksjoner uten navn
`lambda` lager en liten funksjon på én linje: `lambda x: x ** 2` er det samme som en `def` som returnerer `x ** 2`. Den brukes nesten bare der man trenger en funksjon som argument.

```python
kvadrat = lambda x: x ** 2
print(kvadrat(4))
```

Det viktigste bruksområdet er `key=` i `sorted`, `min` og `max`. Funksjonen forteller hva elementene skal sammenlignes etter:

```python
resultater = [("tre", 0.81), ("skog", 0.88), ("knn", 0.79)]

# sorter etter accuracy (element 1 i hver tuple), høyest først
print(sorted(resultater, key=lambda r: r[1], reverse=True))

# modellen med høyest accuracy
print(max(resultater, key=lambda r: r[1]))
```

## argmin og argmax med key
Ofte vil man vite **hvor** (hvilken indeks) minimum er, ikke hva verdien er. Trikset er å gå over indeksene og sammenligne etter verdien:

```python
val_losses = [0.9, 0.62, 0.55, 0.58, 0.71]
beste_epoke = min(range(len(val_losses)), key=lambda i: val_losses[i])
print(beste_epoke, val_losses[beste_epoke])

q = {"opp": 0.1, "ned": -0.4, "venstre": 0.7, "høyre": 0.3}
beste_handling = max(q, key=q.get)     # nøkkelen med størst verdi
print(beste_handling)
```

`max(q, key=q.get)` går over nøklene i dicten og sammenligner dem etter `q.get(nøkkel)`, altså verdien. Det er den grådige handlingen i en Q-tabell lagret som dict.

::: prosjekt Prosjekt 3
Løsningsforslaget lagrer optimizere som lambdaer i en dict, slik at riktig optimizer lages først når parametrene finnes:
```python statisk ikkekjor
optimizers = {
    "adam": lambda p: torch.optim.AdamW(p, lr=lr, weight_decay=weight_decay),
    "sgd": lambda p: torch.optim.SGD(p, lr=lr, momentum=0.9),
}
```
:::

::: oppgave Terskling med list comprehension
Bruk én list comprehension til å lage `y_pred` fra `y_proba`: `1` hvis sannsynligheten er **større enn** 0.5, ellers `0`. Lag også `sikre` med bare sannsynlighetene som er større enn 0.9 eller mindre enn 0.1.
```start
y_proba = [0.1, 0.8, 0.45, 0.5, 0.97, 0.03]

y_pred = ...
sikre = ...
```
```losning
y_proba = [0.1, 0.8, 0.45, 0.5, 0.97, 0.03]

y_pred = [1 if p > 0.5 else 0 for p in y_proba]
sikre = [p for p in y_proba if p > 0.9 or p < 0.1]
```
```test
assert isinstance(y_pred, list), "y_pred skal være en liste. Bruk [... for p in y_proba]."
assert y_pred == [0, 1, 0, 0, 1, 0], f"y_pred ble {y_pred}, men skal være [0, 1, 0, 0, 1, 0]. Merk at 0.5 ikke er større enn 0.5."
assert sikre == [0.97, 0.03], f"sikre ble {sikre}, men skal være [0.97, 0.03]. 0.1 er ikke mindre enn 0.1."
```
:::

::: oppgave Beste epoke og beste handling
Bruk `min` med `key=` til å finne **indeksen** til det laveste valideringstapet, og lagre den i `beste_epoke`. Bruk `max` med `key=` til å finne handlingen med størst Q-verdi i dicten `q`, og lagre den i `beste_handling`.
```start
val_losses = [0.9, 0.62, 0.55, 0.58, 0.71]
q = {"opp": 0.1, "ned": -0.4, "venstre": 0.7, "høyre": 0.3}

beste_epoke = ...
beste_handling = ...
```
```losning
val_losses = [0.9, 0.62, 0.55, 0.58, 0.71]
q = {"opp": 0.1, "ned": -0.4, "venstre": 0.7, "høyre": 0.3}

beste_epoke = min(range(len(val_losses)), key=lambda i: val_losses[i])
beste_handling = max(q, key=q.get)
```
```test
assert beste_epoke == 2, f"beste_epoke ble {beste_epoke!r}, men skal være indeksen 2 (der tapet er 0.55), ikke selve tapet."
assert beste_handling == "venstre", f"beste_handling ble {beste_handling!r}, men skal være 'venstre' (Q-verdi 0.7)."
```
:::

::: quiz
Hva skriver koden ut?
```kode
print([x ** 2 for x in range(5) if x % 2 == 0])
```
- [ ] `[0, 1, 4, 9, 16]`
- [x] `[0, 4, 16]`
- [ ] `[1, 9]`
- [ ] `[0, 2, 4]`
? `range(5)` gir 0 til 4. Bare partallene 0, 2 og 4 slipper gjennom `if`-en, og de kvadreres.
:::

::: quiz
Hva skriver koden ut?
```kode
xs = [5, 2, 8]
print(min(range(len(xs)), key=lambda i: xs[i]))
```
- [ ] 2
- [x] 1
- [ ] 0
- [ ] 8
? `min` går over indeksene 0, 1, 2 og sammenligner dem etter `xs[i]`. Minste verdi er 2, som ligger på indeks 1. Det er indeksen som returneres (argmin).
:::

::: oppsummering
- `[uttrykk for x in liste]` lager en ny liste. `[... if betingelse]` på slutten filtrerer.
- `[1 if p > 0.5 else 0 for p in y_pred]` velger verdi per element.
- `sum(t == p for t, p in zip(a, b))` teller treff uten å lage en liste.
- `{k: v for k, v in d.items()}` er en dict comprehension.
- `lambda x: ...` er en navnløs funksjon, mest brukt som `key=` i `sorted`, `min` og `max`.
- `min(range(len(xs)), key=lambda i: xs[i])` er argmin. `max(d, key=d.get)` gir nøkkelen med størst verdi.
:::

--- LEKSJON 1.14 | klasser | Klasser og objekter
Alle modellene i scikit-learn og PyTorch er **objekter**. Du lager et objekt, trener det med `fit` og bruker det med `predict`. Objektet husker det det har lært. I denne leksjonen lærer du å lage dine egne, slik du skal i prosjekt 1.

En **klasse** er en oppskrift. Et **objekt** (eller instans) er en konkret ting laget etter oppskriften. Et objekt har **attributter** (data) og **metoder** (funksjoner som hører til objektet).

```python
class Punkt:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def avstand_til_origo(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

p = Punkt(3, 4)
print(p.x, p.y)
print(p.avstand_til_origo())
```

- `class Punkt:` starter oppskriften. Klassenavn skrives med stor forbokstav.
- `__init__` (to understreker på hver side) kjøres **automatisk** når du lager et objekt med `Punkt(3, 4)`. Den setter opp startverdiene.
- `self` er objektet selv. `self.x = x` lagrer verdien som et attributt på objektet, slik at alle metodene kan bruke den senere.
- Alle metoder har `self` som første parameter, men du sender den **ikke** inn selv: `p.avstand_til_origo()` gir automatisk `p` som `self`.

Klikk på `p` i variabelpanelet: et objekt vises med alle `self.`-attributtene sine, her `x = 3` og `y = 4`.

## Hvert objekt har sine egne data
```python
class Teller:
    def __init__(self, start=0):
        self.n = start

    def øk(self):
        self.n += 1

a = Teller()
b = Teller(10)
a.øk()
a.øk()
b.øk()
print(a.n, b.n)
```

## En modell med fit og predict
Her er en ekte (men veldig enkel) ML-modell: den predikerer alltid gjennomsnittet av `y` fra treningsdataene. Den er en vanlig *baseline* å sammenligne med. Legg merke til strukturen, den er den samme som `LinearRegression` i prosjekt 1:

```python
class MeanModel:
    """Predikerer alltid gjennomsnittet av y fra treningsdataene."""

    def __init__(self):
        self.mean = None          # finnes ikke før modellen er trent
        self.loss_history = []

    def fit(self, X, y):
        self.mean = sum(y) / len(y)
        mse = sum((yi - self.mean) ** 2 for yi in y) / len(y)
        self.loss_history.append(mse)

    def predict(self, X):
        return [self.mean for _ in X]   # én prediksjon per rad i X


X_train = [[1.0], [2.0], [3.0], [4.0]]
y_train = [2.0, 4.0, 5.0, 9.0]

model = MeanModel()
print("Før fit:", model.mean)
display(model)
model.fit(X_train, y_train)
print("Etter fit:", model.mean)
display(model)
print(model.predict([[10.0], [20.0]]))
print(model.loss_history)
```

`fit` *endrer objektet* (setter `self.mean`), og `predict` *bruker* det som ble lagret. Sammenlign de to `display(model)`-visningene: før `fit` er `mean` lik `None` og `loss_history` tom, etterpå har begge fått verdier. Det er hele poenget med klasser i ML: modellen husker parametrene mellom kallene.

## Hyperparametere i __init__ og metoder som kaller hverandre
Hyperparametere sendes inn når objektet lages og lagres som attributter. En metode kan kalle en annen metode med `self.metode(...)`:

```python
class ThresholdClassifier:
    """Klassifiserer ett tall: 1 hvis x er over en terskel lært fra dataene."""

    def __init__(self, margin=0.0):
        self.margin = margin       # hyperparameter
        self.threshold = None      # parameter som læres i fit

    def _mean(self, values):
        return sum(values) / len(values)

    def fit(self, X, y):
        snitt0 = self._mean([x for x, c in zip(X, y) if c == 0])
        snitt1 = self._mean([x for x, c in zip(X, y) if c == 1])
        self.threshold = (snitt0 + snitt1) / 2 + self.margin
        return self                # vanlig i sklearn: gjør det mulig å skrive model.fit(...).predict(...)

    def predict(self, X):
        return [1 if x > self.threshold else 0 for x in X]


X = [1.0, 1.5, 2.0, 6.0, 7.0, 8.0]
y = [0, 0, 0, 1, 1, 1]
clf = ThresholdClassifier().fit(X, y)
print(clf.threshold)
print(clf.predict([0.5, 4.0, 4.6, 9.0]))
```

Se `clf` i variabelpanelet: hyperparameteren `margin` ble satt i `__init__`, mens `threshold` ble lært i `fit`.

Et navn som starter med `_` (som `_mean`) betyr etter avtale «intern hjelpemetode».

::: prosjekt Prosjekt 1
Slik ser skjelettet i `linear_models.py` ut. Nå kan du lese hver linje: `__init__` lagrer hyperparametrene og setter vektene til `None`, og du skal fylle inn `fit` og `predict`.
```python statisk
import numpy as np

class LinearRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights = None
        self.bias = None

        self.lr = lr
        self.n_iterations = n_iterations

        self.loss_history = []

    def fit(self, X, y):
        # YOUR CODE GOES HERE
        raise NotImplementedError("LinearRegression.fit is not implemented yet.")

    def predict(self, X):
        # YOUR CODE GOES HERE
        raise NotImplementedError("LinearRegression.predict is not implemented yet.")
```
`raise NotImplementedError` lærer du om i leksjon 1.16. I del 5 fyller du inn metodene.
:::

## Vanlige feil
Glemmer du `self` som første parameter i en metode, får du en litt kryptisk `TypeError`: Python sender inn objektet, men metoden tar ikke imot det.

```python feil
class Modell:
    def predict():
        return 0

m = Modell()
m.predict()
```

Glemmer du `self.` foran et attributt, leter Python etter en vanlig variabel som ikke finnes:

```python feil
class Modell:
    def __init__(self):
        self.mean = 3.0

    def predict(self, X):
        return [mean for _ in X]

Modell().predict([[1], [2]])
```

::: oppgave Majoritetsklassifiserer
Fullfør klassen `MajorityClassifier`, en baseline som alltid predikerer den vanligste klassen i treningsdataene.
- `fit(self, X, y)` skal telle klassene i `y` og lagre den vanligste i `self.majority`.
- `predict(self, X)` skal returnere en **liste** med `self.majority` for hver rad i `X`.

Tips: tell med en dict og `.get(k, 0)`, og finn den vanligste med `max(antall, key=antall.get)`.
```start
class MajorityClassifier:
    def __init__(self):
        self.majority = None

    def fit(self, X, y):
        # tell klassene og lagre den vanligste i self.majority
        pass

    def predict(self, X):
        # returner en liste med self.majority for hver rad i X
        pass
```
```losning
class MajorityClassifier:
    def __init__(self):
        self.majority = None

    def fit(self, X, y):
        antall = {}
        for c in y:
            antall[c] = antall.get(c, 0) + 1
        self.majority = max(antall, key=antall.get)

    def predict(self, X):
        return [self.majority for _ in X]
```
```test
m = MajorityClassifier()
assert m.majority is None, "self.majority skal være None før fit."
m.fit([[0], [0], [0], [0], [0]], [0, 1, 1, 2, 1])
assert m.majority == 1, f"Etter fit på y = [0, 1, 1, 2, 1] skal self.majority være 1, men ble {m.majority!r}. Husk self. foran majority."
p = m.predict([[5], [6], [7]])
assert isinstance(p, list), "predict skal returnere en liste."
assert p == [1, 1, 1], f"predict med 3 rader skal gi [1, 1, 1], men ga {p}."
m2 = MajorityClassifier()
m2.fit([[0]] * 3, ["a", "b", "b"])
assert m2.predict([[0]]) == ["b"], "Klassifisereren skal virke for alle typer etiketter, også tekst."
assert m.majority == 1, "To objekter skal ha hver sine data: trening av m2 skal ikke endre m."
```
:::

::: quiz
Hva skriver koden ut?
```kode
class Teller:
    def __init__(self, start=0):
        self.n = start

    def øk(self):
        self.n += 1
        return self.n

a = Teller()
b = Teller(10)
a.øk()
a.øk()
b.øk()
print(a.n, b.n)
```
- [ ] `3 3`
- [x] `2 11`
- [ ] `2 10`
- [ ] `0 10`
? Hvert objekt har sin egen `self.n`. `a` starter på 0 og økes to ganger. `b` starter på 10 og økes én gang.
:::

::: quiz
I `fit` har du regnet ut vektene `w`. Hvilken linje lagrer dem slik at `predict` kan bruke dem etterpå?
- [ ] `weights = w`
- [x] `self.weights = w`
- [ ] `return w`
- [ ] `fit.weights = w`
? Bare attributter på `self` overlever etter at metoden er ferdig. `weights = w` lager en lokal variabel som forsvinner når `fit` er ferdig.
:::

::: oppsummering
- `class Navn:` lager en klasse. `obj = Navn(...)` lager et objekt og kjører `__init__`.
- `self` er objektet selv. `self.x = ...` lagrer et attributt som alle metodene kan bruke.
- Metoder er funksjoner i klassen med `self` som første parameter. Kall dem med `obj.metode()`.
- ML-mønsteret: `__init__` lagrer hyperparametere og setter parametre til `None`, `fit` lærer og lagrer på `self`, `predict` bruker det lagrede.
- Glemt `self` i parameterlisten gir `TypeError`. Glemt `self.` foran et attributt gir `NameError`.
:::

--- LEKSJON 1.15 | arv | Arv og super()
Med **arv** kan en klasse bygge videre på en annen. Skriver du `class Barn(Forelder):`, får `Barn` automatisk alle metodene til `Forelder`, og kan legge til nye eller bytte ut noen. Du kommer til å bruke dette hver gang du lager et nevralt nett i PyTorch (`class SimpleNN(nn.Module)`) og når du lager et eget RL-miljø (`class TrondheimEnv(gym.Env)`).

```python
class Modell:
    def score(self, X, y):
        """Accuracy. Fungerer for alle modeller som har en predict-metode."""
        y_pred = self.predict(X)
        return sum(p == t for p, t in zip(y_pred, y)) / len(y)


class AlltidEn(Modell):
    def predict(self, X):
        return [1 for _ in X]


class Terskel(Modell):
    def predict(self, X):
        return [1 if x > 5 else 0 for x in X]


X = [2, 4, 6, 8]
y = [0, 0, 1, 1]
print(AlltidEn().score(X, y))
print(Terskel().score(X, y))
```

`score` er skrevet én gang i `Modell`, men virker for begge underklassene, fordi `self.predict` alltid bruker `predict`-metoden til det objektet den kalles på. Slik får alle sklearn-modeller en `score`-metode uten å skrive den selv.

## Overstyre metoder
Definerer underklassen en metode med samme navn som i foreldreklassen, brukes underklassens versjon. Det kalles å **overstyre** metoden.

```python
class Agent:
    def act(self, state):
        return 0

    def describe(self):
        return f"{type(self).__name__} velger handling {self.act(0)}"


class HøyreAgent(Agent):
    def act(self, state):      # overstyrer Agent.act
        return 3


print(Agent().describe())
print(HøyreAgent().describe())
print(isinstance(HøyreAgent(), Agent))   # en HøyreAgent ER også en Agent
```

## super().__init__()
Når underklassen har sin egen `__init__`, erstatter den foreldrens `__init__`. Da må du kalle `super().__init__(...)` for at foreldreklassen også skal få satt opp sine attributter. `super()` betyr «foreldreklassen».

```python
class Modell:
    def __init__(self, navn):
        self.navn = navn
        self.er_trent = False

    def fit(self, X, y):
        self.er_trent = True


class TerskelModell(Modell):
    def __init__(self, terskel=0.5):
        super().__init__("terskelmodell")   # setter self.navn og self.er_trent
        self.terskel = terskel               # det som er nytt for underklassen


m = TerskelModell(0.7)
print(m.navn, m.terskel, m.er_trent)
m.fit([], [])
print(m.er_trent)
```

Glemmer du `super().__init__(...)`, mangler attributtene fra foreldreklassen:

```python feil
class Modell:
    def __init__(self):
        self.er_trent = False

class TerskelModell(Modell):
    def __init__(self, terskel=0.5):
        self.terskel = terskel

m = TerskelModell()
print(m.er_trent)
```

## Et lite RL-miljø med arv
Biblioteket gymnasium har en foreldreklasse `gym.Env`, og alle miljøer arver fra den og lager `reset` og `step`. Her er en hjemmelaget versjon. Linjen `raise NotImplementedError` betyr «denne metoden må underklassen lage» (mer om `raise` i neste leksjon).

```python
class Env:
    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError


class Korridor(Env):
    """Agenten står i rute 0 og skal til rute 4. Handling 0 = venstre, 1 = høyre."""

    def __init__(self, lengde=5):
        super().__init__()
        self.lengde = lengde
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state, {}

    def step(self, action):
        if action == 1:
            self.state = min(self.state + 1, self.lengde - 1)
        else:
            self.state = max(self.state - 1, 0)
        terminated = self.state == self.lengde - 1
        reward = 10 if terminated else -1
        return self.state, reward, terminated, {}


env = Korridor()
state, _ = env.reset()
total = 0
for step in range(20):
    state, reward, terminated, info = env.step(1)   # gå alltid til høyre
    total += reward
    if terminated:
        break
print(f"Mål nådd i rute {state} etter {step + 1} steg, total belønning {total}")
```

Her er `super().__init__()` med selv om `Env` ikke har egen `__init__`. Det er god vane, og helt nødvendig med `gym.Env` og `nn.Module`.

## Hvorfor `model(x)` fungerer i PyTorch
I PyTorch skriver du metoden `forward`, men kaller modellen som en funksjon: `model(x)`. Det er arv som gjør det mulig. Foreldreklassen har en spesialmetode `__call__` som kjøres når et objekt kalles med parenteser, og den kaller `self.forward`:

```python
class Module:
    def __call__(self, x):
        return self.forward(x)


class Dobler(Module):
    def forward(self, x):
        return 2 * x


m = Dobler()
print(m(5))      # kaller Module.__call__, som kaller Dobler.forward
```

Slik ser det ut med ekte PyTorch (del 9). Nå kjenner du igjen alle delene: arv fra `nn.Module`, `super().__init__()`, lagene som attributter på `self` og en overstyrt `forward`:

```python statisk
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, n_in, n_hidden):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

torch.manual_seed(0)
model = SimpleNN(2, 4)
print(model)
x = torch.tensor([[1.0, 2.0]])
print(model(x))
```
```output
SimpleNN(
  (fc1): Linear(in_features=2, out_features=4, bias=True)
  (fc2): Linear(in_features=4, out_features=1, bias=True)
)
tensor([[-0.1143]], grad_fn=<AddmmBackward0>)
```

::: oppgave Grådig agent
Klassen `Agent` er gitt. Lag underklassen `GraadigAgent(Agent)`:
- `__init__(self, q_values)` skal kalle `super().__init__(...)` med antall handlinger (lengden av `q_values`), og lagre `q_values` i `self.q_values`.
- `act(self, state)` skal øke `self.steps` med 1 og returnere **indeksen** til den største verdien i `self.q_values`.
```start
class Agent:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.steps = 0

    def act(self, state):
        raise NotImplementedError


class GraadigAgent(Agent):
    def __init__(self, q_values):
        # kall foreldreklassens __init__ og lagre q_values
        pass

    def act(self, state):
        # øk self.steps og returner indeksen til største q-verdi
        pass
```
```losning
class Agent:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.steps = 0

    def act(self, state):
        raise NotImplementedError


class GraadigAgent(Agent):
    def __init__(self, q_values):
        super().__init__(len(q_values))
        self.q_values = q_values

    def act(self, state):
        self.steps += 1
        return max(range(self.n_actions), key=lambda a: self.q_values[a])
```
```test
a = GraadigAgent([0.1, 0.7, 0.3])
assert isinstance(a, Agent), "GraadigAgent skal arve fra Agent: class GraadigAgent(Agent):"
assert getattr(a, "n_actions", None) == 3, "self.n_actions mangler eller er feil. Kall super().__init__(len(q_values)) i __init__."
assert getattr(a, "q_values", None) == [0.1, 0.7, 0.3], "Lagre q_values i self.q_values."
r = a.act(0)
assert r == 1, f"act skal returnere indeksen til største q-verdi (1), men returnerte {r!r}."
assert a.steps == 1, "act skal øke self.steps med 1 hver gang den kalles."
a.act(0)
assert a.steps == 2, "self.steps skal være 2 etter to kall til act."
assert GraadigAgent([5.0, -1.0]).act(0) == 0, "GraadigAgent([5.0, -1.0]).act(0) skal gi 0."
```
:::

::: quiz
Hva skriver koden ut?
```kode
class A:
    def hei(self):
        return "A"

    def snakk(self):
        return self.hei() + "!"

class B(A):
    def hei(self):
        return "B"

print(B().snakk())
```
- [ ] `A!`
- [x] `B!`
- [ ] `AB!`
- [ ] `AttributeError`
? `B` arver `snakk` fra `A`. Inne i `snakk` er `self` et `B`-objekt, så `self.hei()` bruker den overstyrte `B.hei`.
:::

::: quiz
Du lager et nevralt nett med `class Net(nn.Module):` og en egen `__init__`. Hvilken linje må stå først i `__init__`?
- [ ] `self.forward = forward`
- [ ] `nn.Module = Net`
- [x] `super().__init__()`
- [ ] `return self`
? `super().__init__()` kjører foreldreklassens oppsett. Uten den får du en feil når du prøver å lage lag som `self.fc1 = nn.Linear(...)`.
:::

::: oppsummering
- `class Barn(Forelder):` arver alle metodene til `Forelder`.
- En metode med samme navn i underklassen **overstyrer** foreldrens. `self.metode()` bruker alltid objektets egen versjon.
- Har underklassen egen `__init__`, kall `super().__init__(...)` først.
- `isinstance(obj, Klasse)` sjekker om et objekt er av en klasse (eller en underklasse).
- Mønsteret brukes for `nn.Module` (lag `__init__` og `forward`) og `gym.Env` (lag `reset` og `step`).
:::

--- LEKSJON 1.16 | moduler-unntak | Moduler, import og unntak
Python har tusenvis av ferdige funksjoner, men de fleste må **importeres** fra en **modul** før du kan bruke dem. En modul er bare en `.py`-fil med funksjoner og klasser, og en **pakke** (som `numpy` eller `sklearn`) er en samling moduler.

## Tre måter å importere på
```python
import math                 # hele modulen, bruk med math.navn
print(math.sqrt(16), math.exp(1), math.log(math.e))

import numpy as np          # med et kortere kallenavn (alias)
print(np.mean([1, 2, 3, 4]))

from math import pi, exp    # hente enkeltnavn direkte
print(pi, exp(0))
```

Kallenavnene er faste konvensjoner som alle bruker: `import numpy as np`, `import pandas as pd`, `import matplotlib.pyplot as plt`. Du kommer til å skrive dem øverst i hver notebook.

Egne filer importeres på samme måte. I prosjektene ligger hjelpekoden i egne filer ved siden av notebooken:

```python statisk ikkekjor
from linear_models import LinearRegression       # prosjekt 1: linear_models.py
from utils import plot_decision_boundary         # prosjekt 3: utils.py
from sklearn.model_selection import train_test_split
```

## collections.Counter
Standardbiblioteket har mye nyttig. `Counter` gjør tellingen fra leksjon 1.10 i én linje:

```python
from collections import Counter

y = ["katt", "hund", "katt", "fugl", "katt", "hund"]
antall = Counter(y)
print(antall)
print(antall["katt"])
print(antall.most_common(1))   # liste med (klasse, antall) for den vanligste
```

## Modulen random
`random` lager tilfeldige tall. Med `random.seed(...)` får du **de samme** tilfeldige tallene hver gang du kjører, slik at resultatene kan gjenskapes (reproduserbarhet).

```python
import random

random.seed(42)
print(random.random())              # desimaltall mellom 0 og 1
print(random.randint(1, 6))         # heltall fra 1 til og MED 6
print(random.choice(["opp", "ned", "venstre", "høyre"]))   # ett tilfeldig element

tall = [1, 2, 3, 4, 5]
random.shuffle(tall)                # stokker listen (endrer den)
print(tall)

buffer = list(range(100))
print(random.sample(buffer, k=3))   # 3 FORSKJELLIGE tilfeldige elementer
```

Kjør eksempelet flere ganger: tallene blir de samme på grunn av `random.seed(42)`. Prøv å fjerne linjen og se forskjellen.

ε-greedy fra forsterkningslæring er bare noen få linjer med `random`:

```python
import random

random.seed(2)
q = [0.1, 0.7, 0.3, 0.2]     # Q-verdier for fire handlinger, handling 1 er best
epsilon = 0.3

valg = []
for _ in range(10):
    if random.random() < epsilon:
        action = random.randint(0, len(q) - 1)                 # utforsk
    else:
        action = max(range(len(q)), key=lambda a: q[a])        # utnytt
    valg.append(action)
print(valg)    # stort sett 1, men av og til en tilfeldig handling
```

::: eksamen Eksamen 2025
Eksamen 2025 spurte: *Du har en replay-buffer lagret som en liste med transisjoner. Hvilken Python-funksjon brukes for å trekke en tilfeldig minibatch?* Svaret var `random.sample(buffer, k=32)`. Den trekker 32 forskjellige elementer uten tilbakelegging. Alternativene `np.take(buffer, 32)` (henter element nummer 32) og `buffer.sort()` (sorterer) er ikke tilfeldige.
:::

NumPy har sin egen `np.random`, med egen seed (`np.random.seed`). Den lærer du i leksjon 2.10. I prosjekt 3 settes begge, pluss PyTorch sin, i en egen `set_seed`-funksjon.

## Vanlige feiltyper
Når noe går galt, **kaster** Python et **unntak** (exception). Navnet på unntaket forteller hva slags feil det er:

| Feil | Typisk årsak |
|---|---|
| `NameError` | Variabelen finnes ikke (skrivefeil, eller cellen som lager den er ikke kjørt) |
| `TypeError` | Feil type: `"a" + 1`, eller feil antall argumenter til en funksjon |
| `IndexError` | Indeks utenfor listen: `x[10]` når listen har 5 elementer |
| `KeyError` | Nøkkelen finnes ikke i dicten |
| `ValueError` | Riktig type, men ugyldig verdi: `int("abc")`, feil antall ved utpakking |
| `AttributeError` | Objektet har ikke attributtet eller metoden: `[1, 2].shape` |
| `ZeroDivisionError` | Deling på null |
| `ModuleNotFoundError` | Pakken er ikke installert, eller navnet er feilstavet |

```python feil
y = [1, 0, 1]
print(y.shape)
```

```python feil
tall = int("3.5")
```

## Lese en traceback
I Jupyter får du en **traceback** når noe krasjer inne i en funksjon. Den viser veien gjennom koden, fra kallet øverst til linjen som faktisk feilet nederst. Les den **nedenfra**: den siste linjen sier hvilken feil og hvorfor, og linjene rett over viser hvor.

```python statisk ikkekjor
def accuracy(y_true, y_pred):
    riktige = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            riktige += 1
    return riktige / len(y_true)

y_true = [1, 0, 1, 1]
y_pred = [1, 0, 1]
print(accuracy(y_true, y_pred))
```
```output
Traceback (most recent call last):
  File "eksempel.py", line 10, in <module>
    print(accuracy(y_true, y_pred))
          ~~~~~~~~^^^^^^^^^^^^^^^^
  File "eksempel.py", line 4, in accuracy
    if y_true[i] == y_pred[i]:
                    ~~~~~~^^^
IndexError: list index out of range
```

Nederst: `IndexError`, en indeks er utenfor listen. Linjen over: det skjedde i `accuracy`, linje 4, i `y_pred[i]` (markert med `^^^`). Øverst: `accuracy` ble kalt fra linje 10. Årsaken er at `y_pred` har ett element for lite.

## try og except: håndtere feil
Med `try`/`except` kan programmet fortsette selv om noe feiler. Koden i `try` kjøres. Skjer det et unntak av typen du nevner, hoppes resten av `try` over, og koden i `except` kjøres i stedet.

```python
tekster = ["3", "4.5", "abc", "7"]
tall = []
for t in tekster:
    try:
        tall.append(float(t))
    except ValueError as e:
        print(f"Hopper over '{t}': {e}")
print(tall)
```

Nevn alltid hvilken feiltype du venter (`except ValueError`). Et naket `except:` fanger *alle* feil, også skrivefeil i din egen kode, og gjør feilsøking vanskelig.

## raise: kaste en feil selv
Med `raise` stopper du programmet med en tydelig feilmelding når noe er galt. Det er mye bedre enn å regne videre med gale data. Hjelpefunksjonen i prosjekt 3 sjekker for eksempel at to lister er like lange:

```python
def clustering_check(true_labels, cluster_labels):
    if len(true_labels) != len(cluster_labels):
        raise ValueError("true_labels and cluster_labels must have the same length.")
    return "ok"

print(clustering_check([0, 1], [1, 0]))
try:
    clustering_check([0, 1, 1], [1, 0])
except ValueError as e:
    print("Feil:", e)
```

`raise NotImplementedError` brukes i skjelettkode for å si «denne er ikke skrevet ennå». Slik ser `linear_models.py` i prosjekt 1 ut før du har gjort oppgaven:

```python feil
class LinearRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights = None
        self.lr = lr
        self.n_iterations = n_iterations

    def fit(self, X, y):
        # YOUR CODE GOES HERE
        raise NotImplementedError("LinearRegression.fit is not implemented yet.")

model = LinearRegression()
model.fit([[1.0]], [2.0])
```

Når du har skrevet `fit`, fjerner du `raise`-linjen.

::: oppgave Trekk en minibatch
Lag en replay-buffer med `random`. Sett først seed til `42` med `random.seed`, og trekk **rett etterpå** 8 forskjellige transisjoner fra `buffer` med `random.sample`. Lagre dem i `batch`.
```start
import random

buffer = [(s, s % 4, -1.0, s + 1) for s in range(500)]

batch = ...
```
```losning
import random

buffer = [(s, s % 4, -1.0, s + 1) for s in range(500)]

random.seed(42)
batch = random.sample(buffer, k=8)
```
```test
assert isinstance(batch, list), "batch skal være en liste. Bruk random.sample(buffer, k=8)."
assert len(batch) == 8, f"batch skal ha 8 elementer, men har {len(batch)}."
assert all(b in buffer for b in batch) and len(set(batch)) == 8, "batch skal bestå av 8 forskjellige transisjoner fra buffer."
random.seed(42)
assert batch == random.sample(buffer, k=8), "batch ble ikke det vi forventet. Kall random.seed(42) rett før random.sample(buffer, k=8)."
```
:::

::: oppgave Sikker accuracy
Fullfør `accuracy(y_true, y_pred)` slik at den kaster `ValueError` hvis listene har ulik lengde. Ellers skal den returnere accuracy som før. Lag så `les_tall(tekst)` som returnerer `float(tekst)`, eller `None` hvis teksten ikke er et tall (bruk `try`/`except ValueError`).
```start
def accuracy(y_true, y_pred):
    # sjekk lengdene her
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def les_tall(tekst):
    return float(tekst)
```
```losning
def accuracy(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true og y_pred må være like lange")
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def les_tall(tekst):
    try:
        return float(tekst)
    except ValueError:
        return None
```
```test
assert abs(accuracy([1, 0, 1, 1], [1, 0, 0, 1]) - 0.75) < 1e-9, "accuracy skal fortsatt gi 0.75 for like lange lister."
kastet = False
try:
    accuracy([1, 0, 1], [1, 0])
except ValueError:
    kastet = True
assert kastet, "accuracy([1, 0, 1], [1, 0]) skal kaste ValueError. Bruk: if len(y_true) != len(y_pred): raise ValueError(...)"
assert les_tall("3.5") == 3.5, 'les_tall("3.5") skal gi 3.5.'
try:
    r = les_tall("abc")
except ValueError:
    raise AssertionError('les_tall("abc") krasjet med ValueError. Fang feilen med try/except og returner None.')
assert r is None, f'les_tall("abc") skal gi None, men ga {r!r}.'
```
:::

::: quiz
`buffer` er en liste med 10 000 transisjoner. Hvilken linje trekker en tilfeldig minibatch med 32 forskjellige transisjoner?
- [ ] `buffer[:32]`
- [x] `random.sample(buffer, k=32)`
- [ ] `np.take(buffer, 32)`
- [ ] `buffer.sort()`
? `random.sample` trekker `k` forskjellige elementer tilfeldig. `buffer[:32]` er alltid de 32 første (og ofte de eldste) transisjonene, `np.take(buffer, 32)` henter element nummer 32, og `sort` sorterer bare.
:::

::: quiz
Hvilken feil gir `int("3.5")`?
- [ ] `TypeError`
- [x] `ValueError`
- [ ] `SyntaxError`
- [ ] Ingen, den gir 3
? Typen (tekst) er grei for `int()`, men verdien `"3.5"` er ikke et gyldig heltall. Riktig type, ugyldig verdi er `ValueError`. Skriv `int(float("3.5"))` hvis du vil ha 3.
:::

::: quiz
Hva skriver koden ut?
```kode
try:
    x = 1 / 0
    print("A")
except ZeroDivisionError:
    print("B")
print("C")
```
- [ ] `A` og `C`
- [x] `B` og `C`
- [ ] `A`, `B` og `C`
- [ ] Bare `B`
? Delingen kaster `ZeroDivisionError`, så resten av `try` (`print("A")`) hoppes over og `except`-blokken kjøres. Etterpå fortsetter programmet som normalt.
:::

::: oppsummering
- `import modul`, `import numpy as np` og `from modul import navn` henter kode fra moduler og pakker.
- `random.seed(s)` gir reproduserbare tall. `random.random()`, `randint(a, b)`, `choice(liste)`, `shuffle(liste)`.
- `random.sample(buffer, k=32)` trekker 32 forskjellige elementer (minibatch fra replay-buffer).
- Vanlige feil: `NameError`, `TypeError`, `IndexError`, `KeyError`, `ValueError`, `AttributeError`.
- Les en traceback nedenfra: siste linje er feiltypen og meldingen, linjene over viser hvor.
- `try`/`except FeilType:` håndterer feil. `raise ValueError("...")` kaster en selv. `raise NotImplementedError` markerer kode som ikke er skrevet ennå.
:::
