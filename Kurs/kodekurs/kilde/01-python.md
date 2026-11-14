=== DEL 1 | python | Python fra null
Ren Python fra null til klasser, med alt du trenger for å skrive koden i prosjektene.

--- LEKSJON 1.1 | forste-program | Ditt første program
`print("Hei!")` skriver ut tekst i anførselstegn, `print(2 + 2)` skriver ut verdien av uttrykket, og `print(a, b)` skriver ut flere verdier med mellomrom. Programmet kjøres ovenfra og ned, og første gang du trykker **Kjør** tar det noen sekunder.

::: oppgave Skriv ut to linjer
Skriv ut `Hei!` og deretter `Jeg lærer Python.` med to `print`-linjer, og la siste linje være `status = "ferdig"` (variabler kommer i neste leksjon).
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

::: oppgave Regn ut med Python
Skriv ut `3 * 7 + 1` med `print`, og lagre det samme uttrykket i `svar` med `svar = 3 * 7 + 1`.
```start
# Skriv print-linjen her

svar = ...
```
```losning
print(3 * 7 + 1)
svar = 3 * 7 + 1
```
```test
assert svar == 22, "svar skal være 3 * 7 + 1 = 22. Skriv selve uttrykket, så regner Python det ut."
```
:::

## Kommentarer
Alt etter `#` på en linje er en kommentar som Python hopper over.

::: oppgave Kommenter ut en linje
Den siste linjen ødelegger læringsraten. Gjør den om til en kommentar ved å sette `#` foran, uten å slette den.
```start
lr = 0.01
lr = 999
```
```losning
lr = 0.01
# lr = 999
```
```test
assert lr == 0.01, "lr skal fortsatt være 0.01. Sett # først på linjen lr = 999."
```
:::

## Feilmeldinger
En feilmelding sier hvilken linje som feilet og hvilken type feil det er, for eksempel `NameError` når et navn er feilstavet eller `SyntaxError` når en parentes mangler.

::: oppgave Rett feilen
Kjør koden, les feilmeldingen og rett skrivefeilen slik at programmet kjører helt til siste linje.
```start
prnt("Trening starter")
ferdig = True
```
```losning
print("Trening starter")
ferdig = True
```
```test
assert ferdig is True, "Programmet skal kjøre helt til linjen ferdig = True."
```
:::

--- LEKSJON 1.2 | variabler | Variabler og datatyper
`navn = verdi` lager en variabel, og `steg = steg + 1` regner ut høyresiden først. Variabelpanelet under resultatet viser alle variablene.

::: oppgave Oppdater en variabel
Lag `epochs = 100`, og øk den så med 50 på en egen linje med `epochs = epochs + 50`.
```start
epochs = ...
# øk epochs med 50 her
```
```losning
epochs = 100
epochs = epochs + 50
```
```test
assert epochs == 150, f"epochs ble {epochs!r}, men skal være 100 + 50 = 150."
```
:::

::: oppgave Hva er y?
Koden endrer `x` etter at `y = x` er kjørt. Lagre tallet du tror `y` har til slutt i `svar` (skriv tallet, ikke `svar = y`).
```start
x = 5
y = x
x = 10

svar = ...
```
```losning
x = 5
y = x
x = 10

svar = 5
```
```test
assert svar == 5, "y fikk verdien x hadde da y = x ble kjørt. At x senere får en ny verdi, endrer ikke y."
```
:::

## Datatyper
Grunntypene er `int` (`42`), `float` (`0.01`, `1e-3`), `str` (`"katt"`) og `bool` (`True`/`False`), og `type(x)` gir typen. Desimaltall skrives med punktum, aldri komma.

::: oppgave Hyperparametere i variabler
Lag `modell` med teksten `"LinearRegression"`, `lr` med `0.001`, `n_iterations` med heltallet `1000` og `ferdig_trent` med `False`. Lagre til slutt `type(lr)` i `lr_type`.
```start
modell = ...
lr = ...
n_iterations = ...
ferdig_trent = ...
lr_type = ...
```
```losning
modell = "LinearRegression"
lr = 0.001
n_iterations = 1000
ferdig_trent = False
lr_type = type(lr)
```
```test
assert modell == "LinearRegression", 'modell skal være teksten "LinearRegression" (husk anførselstegn).'
assert isinstance(lr, float) and abs(lr - 0.001) < 1e-12, "lr skal være desimaltallet 0.001 (eller 1e-3)."
assert type(n_iterations) is int and n_iterations == 1000, "n_iterations skal være heltallet 1000 (uten desimaler)."
assert ferdig_trent is False, "ferdig_trent skal være False (stor F, uten anførselstegn)."
assert lr_type is float, "lr_type skal være type(lr), altså float."
```
:::

## None og konvertering
`None` betyr «ingen verdi ennå», som `weights = None` før modellen er trent. `int("42")`, `float("0.5")` og `str(3)` konverterer, og `int(3.9)` kutter til `3`, mens `round(3.9)` gir `4`.

::: oppgave Konverter verdier
Lag `epoker` (heltall fra `tekst_epoker`), `lr` (desimaltall fra `tekst_lr`), `kuttet = int(p)`, `avrundet = round(p)`, `melding` som `str(epoker) + " epoker"` og `weights` uten verdi ennå.
```start
tekst_epoker = "25"
tekst_lr = "0.05"
p = 7.8

epoker = ...
lr = ...
kuttet = ...
avrundet = ...
melding = ...
weights = ...
```
```losning
tekst_epoker = "25"
tekst_lr = "0.05"
p = 7.8

epoker = int(tekst_epoker)
lr = float(tekst_lr)
kuttet = int(p)
avrundet = round(p)
melding = str(epoker) + " epoker"
weights = None
```
```test
assert type(epoker) is int and epoker == 25, "epoker skal være int(tekst_epoker), altså heltallet 25."
assert isinstance(lr, float) and abs(lr - 0.05) < 1e-12, "lr skal være float(tekst_lr), altså 0.05."
assert kuttet == 7, f"kuttet ble {kuttet!r}. int(7.8) kutter bort desimalene og gir 7."
assert avrundet == 8, f"avrundet ble {avrundet!r}. round(7.8) runder av til 8."
assert melding == "25 epoker", f"melding ble {melding!r}, men skal være '25 epoker'. Bruk str(epoker) + \" epoker\"."
assert weights is None, "weights skal være None (uten anførselstegn)."
```
:::

## Navn
Store og små bokstaver er forskjellige (`X` og `x` er to ulike variabler), og et navn som ikke finnes gir `NameError`.

::: oppgave NameError
Kjør koden, les feilmeldingen og rett feilen slik at `total` blir 200.
```start
epochs = 100
total = Epochs * 2
```
```losning
epochs = 100
total = epochs * 2
```
```test
assert total == 200, "total skal være epochs * 2 = 200. Variabelen heter epochs med liten e."
```
:::

--- LEKSJON 1.3 | regning | Regning
`+ - * /` virker som vanlig og `/` gir alltid `float`, mens `7 // 2` (heltallsdivisjon) gir `3`, `7 % 2` (rest) gir `1` og `2 ** 3` (potens) gir `8`.

::: oppgave Operatorene
Lag `hele = 17 // 5`, `rest = 17 % 5`, `kube = 2 ** 10`, `rot` som kvadratroten av 16 med `** 0.5`, og `halv = 10 / 2`.
```start
hele = ...
rest = ...
kube = ...
rot = ...
halv = ...
```
```losning
hele = 17 // 5
rest = 17 % 5
kube = 2 ** 10
rot = 16 ** 0.5
halv = 10 / 2
```
```test
assert hele == 3, f"hele ble {hele!r}. 17 // 5 er heltallsdivisjon og gir 3."
assert rest == 2, f"rest ble {rest!r}. 17 % 5 er resten: 17 = 3 · 5 + 2."
assert kube == 1024, f"kube ble {kube!r}. 2 ** 10 er 1024."
assert isinstance(rot, float) and abs(rot - 4.0) < 1e-12, "rot skal være 16 ** 0.5 = 4.0."
assert isinstance(halv, float) and halv == 5.0, "halv skal være 10 / 2, som gir 5.0 (/ gir alltid float)."
```
:::

## Regnerekkefølge
`**` regnes først, så `* / // %` og til slutt `+ -`, så `-2 ** 2` blir `-4`. Bruk parenteser når du er i tvil.

::: oppgave Parenteser
Lag `snitt` som gjennomsnittet av `a`, `b` og `c` i ett uttrykk, og `kvadrat` som `-3` opphøyd i andre (svaret skal bli 9).
```start
a, b, c = 2.0, 4.0, 9.0

snitt = ...
kvadrat = ...
```
```losning
a, b, c = 2.0, 4.0, 9.0

snitt = (a + b + c) / 3
kvadrat = (-3) ** 2
```
```test
assert isinstance(snitt, float) and abs(snitt - 5.0) < 1e-9, f"snitt ble {snitt!r}, men skal være 5.0. Uten parenteser deles bare c på 3: skriv (a + b + c) / 3."
assert kvadrat == 9, f"kvadrat ble {kvadrat!r}. -3 ** 2 gir -9 fordi potensen regnes først: skriv (-3) ** 2."
```
:::

## Oppdatere en variabel
`x += 1` er det samme som `x = x + 1`, og `-=` og `*=` virker på samme måte.

::: oppgave Belønning og epsilon
Legg til 10 i `total_reward` og trekk fra 1 tre ganger med `+=` og `-=`. Gang så `epsilon` med 0.99 to ganger med `*=`.
```start
total_reward = 0
# oppdater total_reward her

epsilon = 1.0
# oppdater epsilon her
```
```losning
total_reward = 0
total_reward += 10
total_reward -= 1
total_reward -= 1
total_reward -= 1

epsilon = 1.0
epsilon *= 0.99
epsilon *= 0.99
```
```test
assert total_reward == 7, f"total_reward ble {total_reward}, men skal være 0 + 10 - 1 - 1 - 1 = 7."
assert abs(epsilon - 0.9801) < 1e-9, f"epsilon ble {epsilon}, men skal være 1.0 · 0.99 · 0.99 = 0.9801."
```
:::

## Innebygde funksjoner
`abs(-0.3)`, `round(3.14159, 2)`, `min(4, 1, 7)` og `max(4, 1, 7)` er innebygd. Desimaltall er ikke helt eksakte, så `0.1 + 0.2` gir `0.30000000000000004`.

::: oppgave Abs, round og max
Lag `abs_feil` (absoluttverdien av `feil`), `avrundet` (`feil` rundet av til 2 desimaler), `eps` (`epsilon`, men aldri under 0.01, bruk `max`) og `rest = 1537 % 100`.
```start
feil = -0.4567
epsilon = 0.003

abs_feil = ...
avrundet = ...
eps = ...
rest = ...
```
```losning
feil = -0.4567
epsilon = 0.003

abs_feil = abs(feil)
avrundet = round(feil, 2)
eps = max(0.01, epsilon)
rest = 1537 % 100
```
```test
assert isinstance(abs_feil, float) and abs(abs_feil - 0.4567) < 1e-12, "abs_feil skal være abs(feil) = 0.4567."
assert isinstance(avrundet, float) and abs(avrundet - (-0.46)) < 1e-12, f"avrundet ble {avrundet!r}. round(feil, 2) gir -0.46."
assert isinstance(eps, float) and abs(eps - 0.01) < 1e-12, f"eps ble {eps!r}. max(0.01, epsilon) gir 0.01 fordi epsilon er mindre."
assert rest == 37, f"rest ble {rest!r}. 1537 % 100 er 37."
```
:::

## Et steg med gradient descent
`x, y = 2.0, 3.0` gir to variabler verdi på én gang, og gradient descent flytter vekten mot minus gradienten: `w = w - lr * gradient`.

::: oppgave Ett gradient descent-steg
Regn ut i rekkefølge: `prediksjon = w * x`, `feil = prediksjon - y`, `tap = feil ** 2`, `gradient = 2 * feil * x` og `w_ny = w - lr * gradient`.
```start
w = 0.8
x, y = 2.0, 3.0
lr = 0.1

prediksjon = ...
feil = ...
tap = ...
gradient = ...
w_ny = ...
```
```losning
w = 0.8
x, y = 2.0, 3.0
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

--- LEKSJON 1.4 | tekst | Tekst og f-strenger
Strenger skrives med `"..."` eller `'...'`. `len(s)` gir antall tegn, `+` skjøter, `"-" * 20` gjentar og `\n` er linjeskift.

::: oppgave Bygg tekst
Lag `lengde` (antall tegn i `navn`), `setning` (`navn` + `" er en ensemblemetode"`), `strek` (20 bindestreker med `*`) og `to_linjer` med `Trening` og `ferdig` skilt av `\n`.
```start
navn = "Random Forest"

lengde = ...
setning = ...
strek = ...
to_linjer = ...
```
```losning
navn = "Random Forest"

lengde = len(navn)
setning = navn + " er en ensemblemetode"
strek = "-" * 20
to_linjer = "Trening\nferdig"
```
```test
assert lengde == 13, f"lengde ble {lengde!r}. len(navn) teller alle tegn, også mellomrommet: 13."
assert setning == "Random Forest er en ensemblemetode", f"setning ble {setning!r}. Husk mellomrommet foran 'er'."
assert strek == "-" * 20, "strek skal være \"-\" * 20."
assert to_linjer == "Trening\nferdig", f"to_linjer ble {to_linjer!r}. Skriv \"Trening\\nferdig\"."
```
:::

## Strengmetoder
`s.strip()`, `s.lower()`, `s.replace(" ", "_")`, `s.split(",")` og `s.startswith("Frozen")` lager nye verdier og endrer ikke `s`.

::: oppgave Rydd et kolonnenavn
Lag `rent` = `"sepal_length"` fra `s` med `strip`, `lower` og `replace`, `deler` ved å dele `rad` på komma, og `er_frozen` som sier om `env_navn` starter med `"Frozen"`.
```start
s = "  Sepal Length  "
rad = "x0,x1,y"
env_navn = "FrozenLake-v1"

rent = ...
deler = ...
er_frozen = ...
```
```losning
s = "  Sepal Length  "
rad = "x0,x1,y"
env_navn = "FrozenLake-v1"

rent = s.strip().lower().replace(" ", "_")
deler = rad.split(",")
er_frozen = env_navn.startswith("Frozen")
```
```test
assert rent == "sepal_length", f"rent ble {rent!r}. Kjed metodene: s.strip().lower().replace(\" \", \"_\")."
assert deler == ["x0", "x1", "y"], f"deler ble {deler!r}. Bruk rad.split(\",\")."
assert er_frozen is True, "er_frozen skal være env_navn.startswith(\"Frozen\"), som gir True."
```
:::

## f-strenger
`f"Epoch {epoch}: loss = {loss}"` setter verdier og uttrykk rett inn i teksten, mens `"Tap: " + 0.25` gir `TypeError`.

::: oppgave Tekst og tall
Rett linjen som lager `melding` slik at den blir `Tap: 0.25`, og lag `neste` = `Neste epoke er 6` med en f-streng som regner ut `epoch + 1`.
```start
loss = 0.25
epoch = 5

melding = "Tap: " + loss
neste = ...
```
```losning
loss = 0.25
epoch = 5

melding = f"Tap: {loss}"
neste = f"Neste epoke er {epoch + 1}"
```
```test
assert melding == "Tap: 0.25", f"melding ble {melding!r}. Bruk f\"Tap: {{loss}}\" (eller str(loss))."
assert neste == "Neste epoke er 6", f"neste ble {neste!r}. Skriv f\"Neste epoke er {{epoch + 1}}\"."
```
:::

## Formatering av tall
Etter kolon i `{}` styrer du formatet: `{x:.2f}` gir 2 desimaler, `{x:.1%}` prosent, `{n:,}` tusenskille, `{x:+.3f}` fortegn, `{x:.2e}` vitenskapelig notasjon og `{navn:10s}`/`{x:8.3f}` fast bredde.

::: oppgave Treningsrapport
Lag `rapport` med en f-streng slik at den blir nøyaktig `Epoch 7: loss=0.0346, accuracy=91.2%`, med variablene i stedet for tallene.
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

::: oppgave Flere formater
Lag `a` = `1,234,567` fra `n`, `b` = `+0.877` fra `acc`, `c` = `1.23e-04` fra `liten`, `rad` med `navn` 10 tegn bredt og `acc` 8 tegn bredt med 3 desimaler, og `bredde` = `bredde=64` fra `cfg["width"]`.
```start
n = 1234567
acc = 0.87654
liten = 0.000123
navn = "SVM"
cfg = {"width": 64}

a = ...
b = ...
c = ...
rad = ...
bredde = ...
```
```losning
n = 1234567
acc = 0.87654
liten = 0.000123
navn = "SVM"
cfg = {"width": 64}

a = f"{n:,}"
b = f"{acc:+.3f}"
c = f"{liten:.2e}"
rad = f"{navn:10s}{acc:8.3f}"
bredde = f'bredde={cfg["width"]}'
```
```test
assert a == "1,234,567", f"a ble {a!r}. Bruk f\"{{n:,}}\"."
assert b == "+0.877", f"b ble {b!r}. Bruk f\"{{acc:+.3f}}\"."
assert c == "1.23e-04", f"c ble {c!r}. Bruk f\"{{liten:.2e}}\"."
assert rad == "SVM          0.877", f"rad ble {rad!r}. Bruk f\"{{navn:10s}}{{acc:8.3f}}\"."
assert bredde == "bredde=64", f"bredde ble {bredde!r}. Inne i f-strengen må anførselstegnene rundt \"width\" være av den andre typen enn de ytterste."
```
:::

--- LEKSJON 1.5 | sammenligninger | Sammenligninger og bool
`==`, `!=`, `<`, `>`, `<=` og `>=` gir `True` eller `False`, og `=` (gi verdi) er noe annet enn `==` (sammenligne). Store og små bokstaver teller: `"katt" == "Katt"` er `False`.

::: oppgave Sammenlign
Bruk sammenligninger (ikke skriv `True`/`False` direkte) til å lage `positiv` (`p` er større enn 0.5), `ikke_en` (`p` er ikke lik 1) og `samme_ord` (`ord1` er lik `ord2`).
```start
p = 0.73
ord1 = "katt"
ord2 = "Katt"

positiv = ...
ikke_en = ...
samme_ord = ...
```
```losning
p = 0.73
ord1 = "katt"
ord2 = "Katt"

positiv = p > 0.5
ikke_en = p != 1
samme_ord = ord1 == ord2
```
```test
assert positiv is True, "positiv skal være p > 0.5, som blir True."
assert ikke_en is True, "ikke_en skal være p != 1, som blir True."
assert samme_ord is False, "samme_ord skal være ord1 == ord2, som blir False fordi K og k er forskjellige."
```
:::

## and, or og not
`a and b`, `a or b` og `not a` kombinerer sannhetsverdier, og `0.4 <= p <= 0.6` er lov.

::: oppgave Er episoden ferdig?
Lag `done` (målet er nådd **eller** `step` har nådd `max_steps`), `fortsett` (det motsatte av `done`) og `usikker` (`p` ligger mellom 0.4 og 0.6, begge grensene med).
```start
step = 50
max_steps = 50
reached_goal = False
p = 0.58

done = ...
fortsett = ...
usikker = ...
```
```losning
step = 50
max_steps = 50
reached_goal = False
p = 0.58

done = reached_goal or step >= max_steps
fortsett = not done
usikker = 0.4 <= p <= 0.6
```
```test
assert done is True, "done skal være reached_goal or step >= max_steps, som blir True."
assert fortsett is False, "fortsett skal være not done."
assert usikker is True, "usikker skal være 0.4 <= p <= 0.6."
```
:::

## Desimaltall og None
Sammenlign desimaltall med `abs(a - b) < 1e-9` i stedet for `==`, og sjekk `None` med `x is None` og `x is not None`.

::: oppgave Trygge sammenligninger
Lag `naiv = a == 0.3` (se at den blir `False`), `lik` som sjekker likheten med `abs(...) < 1e-9`, `mangler` (`weights` er `None`) og `finnes` (`weights` er ikke `None`).
```start
a = 0.1 + 0.2
weights = None

naiv = ...
lik = ...
mangler = ...
finnes = ...
```
```losning
a = 0.1 + 0.2
weights = None

naiv = a == 0.3
lik = abs(a - 0.3) < 1e-9
mangler = weights is None
finnes = weights is not None
```
```test
assert naiv is False, "naiv skal være a == 0.3, som faktisk blir False på grunn av avrundingsfeil."
assert lik is True, "lik skal være abs(a - 0.3) < 1e-9, som blir True."
assert mangler is True, "mangler skal være weights is None."
assert finnes is False, "finnes skal være weights is not None."
```
:::

## True og False som tall
`True` teller som `1` og `False` som `0`, så summen av sammenligninger er antall sanne.

::: oppgave Tell riktige prediksjoner
Lag `riktige` som summen av de fire sammenligningene `t1 == p1`, `t2 == p2` osv. (bruk parenteser rundt hver), og `accuracy = riktige / 4`.
```start
t1, t2, t3, t4 = 1, 0, 1, 1
p1, p2, p3, p4 = 1, 1, 1, 1

riktige = ...
accuracy = ...
```
```losning
t1, t2, t3, t4 = 1, 0, 1, 1
p1, p2, p3, p4 = 1, 1, 1, 1

riktige = (t1 == p1) + (t2 == p2) + (t3 == p3) + (t4 == p4)
accuracy = riktige / 4
```
```test
assert riktige == 3, f"riktige ble {riktige!r}, men 3 av 4 sammenligninger er sanne (True teller som 1)."
assert isinstance(accuracy, float) and abs(accuracy - 0.75) < 1e-9, "accuracy skal være riktige / 4 = 0.75."
```
:::

--- LEKSJON 1.6 | if-elif-else | if, elif og else
`if p > 0.5:` kjører den innrykkede koden (4 mellomrom) bare når betingelsen er sann. Glemt kolon gir `SyntaxError` og glemt innrykk `IndentationError`.

::: oppgave Klasse 1?
Skriv en `if` som setter `label = 1` når `p` er større enn 0.5.
```start
p = 0.83
label = 0
# skriv if-en her
```
```losning
p = 0.83
label = 0
if p > 0.5:
    label = 1
```
```test
assert label == 1, "label skal bli 1 fordi p = 0.83 > 0.5. Husk kolon og innrykk."
```
:::

## else og elif
`elif` tester neste betingelse og `else` tar resten, og bare den **første** sanne grenen kjøres.

::: oppgave Rekkefølgen teller
Koden gir `"ok"` for `acc = 0.95`. Endre rekkefølgen på grenene slik at den gir `"svært bra"` når `acc >= 0.9`, `"ok"` når `acc >= 0.5` og ellers `"dårlig"`.
```start
acc = 0.95
if acc >= 0.5:
    vurdering = "ok"
elif acc >= 0.9:
    vurdering = "svært bra"
else:
    vurdering = "dårlig"
```
```losning
acc = 0.95
if acc >= 0.9:
    vurdering = "svært bra"
elif acc >= 0.5:
    vurdering = "ok"
else:
    vurdering = "dårlig"
```
```test
assert vurdering == "svært bra", f"vurdering ble {vurdering!r}. Bare den første sanne grenen kjøres, så den strengeste testen (acc >= 0.9) må stå først."
```
:::

## Kombinerte betingelser og nøsting
Betingelser kan kombineres med `and`/`or`, og en `if` kan stå inni en annen med dobbelt innrykk.

::: oppgave Utforske eller utnytte?
Lag `handling` = `"utforsk"` hvis `tilfeldig < epsilon`, ellers `"utnytt"`. Lag `melding` = `"episode ferdig"` hvis `steg >= max_steg` **eller** `i_mål`, ellers `"fortsett"`.
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

::: oppgave Logg hver 100. episode
Sett `melding = "status"` hvis `episode % 100 == 0`, og legg inni den en ny `if` som i stedet gir `"status, lite utforsking"` når `epsilon < 0.1`.
```start
episode = 300
epsilon = 0.05
melding = ""
# skriv de nøstede if-ene her
```
```losning
episode = 300
epsilon = 0.05
melding = ""
if episode % 100 == 0:
    melding = "status"
    if epsilon < 0.1:
        melding = "status, lite utforsking"
```
```test
assert melding == "status, lite utforsking", f"melding ble {melding!r}. 300 % 100 == 0 og epsilon < 0.1, så begge if-ene skal slå til."
```
:::

## Valg på én linje
`label = 1 if p > 0.5 else 0` velger mellom to verdier, og i `if x:` regnes `0`, `""`, `None` og tomme lister som usanne.

::: oppgave Kortform og tomme verdier
Lag `label` med kortformen (1 hvis `p > 0.5`, ellers 0) og `har_vekter` = `"ja" if weights else "nei"`.
```start
p = 0.31
weights = None

label = ...
har_vekter = ...
```
```losning
p = 0.31
weights = None

label = 1 if p > 0.5 else 0
har_vekter = "ja" if weights else "nei"
```
```test
assert label == 0, f"label ble {label!r}. p = 0.31 er ikke over 0.5, så label skal være 0."
assert har_vekter == "nei", f"har_vekter ble {har_vekter!r}. None regnes som usann, så svaret skal være \"nei\"."
```
:::

--- LEKSJON 1.7 | lister | Lister
`losses = [0.9, 0.5, 0.3]` lager en liste, `len(losses)` gir lengden, `losses[0]` er første og `losses[-1]` siste element. En indeks utenfor listen gir `IndexError`.

::: oppgave Indekser
Lag `forste`, `siste` (negativ indeks), `nest_siste` og `antall` (lengden) fra `losses`.
```start
losses = [2.5, 1.8, 1.2, 0.9, 0.7, 0.6, 0.55]

forste = ...
siste = ...
nest_siste = ...
antall = ...
```
```losning
losses = [2.5, 1.8, 1.2, 0.9, 0.7, 0.6, 0.55]

forste = losses[0]
siste = losses[-1]
nest_siste = losses[-2]
antall = len(losses)
```
```test
assert forste == 2.5, "forste skal være losses[0], altså 2.5 (indekser starter på 0)."
assert siste == 0.55, "siste skal være losses[-1], altså 0.55."
assert nest_siste == 0.6, "nest_siste skal være losses[-2], altså 0.6."
assert antall == 7, "antall skal være len(losses), altså 7."
```
:::

## Slicing
`x[start:stopp]` tar med `start`, men ikke `stopp`, `x[:3]` og `x[-3:]` gir de tre første og siste, `x[::2]` annethvert og `x[::-1]` baklengs. Tekst slices på samme måte.

::: oppgave Skjær ut biter
Lag `midten` (indeks 1, 2 og 3), `siste_to`, `annethvert`, `baklengs` og `start_ord` (de 6 første tegnene i `fag`).
```start
x = [10, 20, 30, 40, 50, 60]
fag = "maskinlæring"

midten = ...
siste_to = ...
annethvert = ...
baklengs = ...
start_ord = ...
```
```losning
x = [10, 20, 30, 40, 50, 60]
fag = "maskinlæring"

midten = x[1:4]
siste_to = x[-2:]
annethvert = x[::2]
baklengs = x[::-1]
start_ord = fag[:6]
```
```test
assert midten == [20, 30, 40], f"midten ble {midten!r}. x[1:4] gir indeks 1, 2 og 3 (stopp er ikke med)."
assert siste_to == [50, 60], f"siste_to ble {siste_to!r}. Bruk x[-2:]."
assert annethvert == [10, 30, 50], f"annethvert ble {annethvert!r}. Bruk x[::2]."
assert baklengs == [60, 50, 40, 30, 20, 10], f"baklengs ble {baklengs!r}. Bruk x[::-1]."
assert start_ord == "maskin", f"start_ord ble {start_ord!r}. Bruk fag[:6]."
```
:::

## Endre en liste
`x.append(v)` legger til på slutten, `x[0] = v` bytter ut og `a + b` lager en ny, sammenskjøtt liste. `b = a` lager **ikke** en kopi, så bruk `b = a.copy()`.

::: oppgave Tapshistorikk
Legg til `0.9`, `0.6` og `0.4` i `loss_history` med `append`, bytt så ut det første elementet med `1.0`, og lag `lengre` som `loss_history` skjøtt sammen med `[0.3, 0.2]`.
```start
loss_history = []
# append, bytt ut og skjøt her

lengre = ...
```
```losning
loss_history = []
loss_history.append(0.9)
loss_history.append(0.6)
loss_history.append(0.4)
loss_history[0] = 1.0

lengre = loss_history + [0.3, 0.2]
```
```test
assert loss_history == [1.0, 0.6, 0.4], f"loss_history ble {loss_history!r}, men skal være [1.0, 0.6, 0.4]. Bruk append tre ganger og så loss_history[0] = 1.0."
assert lengre == [1.0, 0.6, 0.4, 0.3, 0.2], f"lengre ble {lengre!r}. Bruk loss_history + [0.3, 0.2]."
```
:::

::: oppgave Kopi, ikke samme liste
Rett linjen `b = a` slik at `b.append(4)` ikke endrer `a`.
```start
a = [1, 2, 3]
b = a
b.append(4)
```
```losning
a = [1, 2, 3]
b = a.copy()
b.append(4)
```
```test
assert b == [1, 2, 3, 4], "b skal være [1, 2, 3, 4] etter append."
assert a == [1, 2, 3], f"a ble endret til {a}. b = a gir bare et nytt navn til samme liste: bruk b = a.copy()."
```
:::

## Nyttige funksjoner og in
`sum(x)`, `min(x)`, `max(x)`, `x.count(1)` og `2 in x` gir det de sier, og `sorted(x)` lager en ny sortert liste, mens `x.sort()` sorterer selve `x`.

::: oppgave Oppsummer en liste
Lag `andel` (andelen enere i `y`), `enere` (med `.count`), `finnes_2` (med `in`), `storst` (største i `scores`) og `rangert` (sortert kopi av `scores`, uten å endre `scores`).
```start
y = [1, 0, 1, 1, 0, 1]
scores = [0.7, 0.9, 0.5, 0.8]

andel = ...
enere = ...
finnes_2 = ...
storst = ...
rangert = ...
```
```losning
y = [1, 0, 1, 1, 0, 1]
scores = [0.7, 0.9, 0.5, 0.8]

andel = sum(y) / len(y)
enere = y.count(1)
finnes_2 = 2 in y
storst = max(scores)
rangert = sorted(scores)
```
```test
assert isinstance(andel, float) and abs(andel - 4 / 6) < 1e-9, "andel skal være sum(y) / len(y) = 4/6."
assert enere == 4, "enere skal være y.count(1), altså 4."
assert finnes_2 is False, "finnes_2 skal være 2 in y, som er False."
assert storst == 0.9, "storst skal være max(scores) = 0.9."
assert rangert == [0.5, 0.7, 0.8, 0.9], f"rangert ble {rangert!r}. Bruk sorted(scores)."
assert scores == [0.7, 0.9, 0.5, 0.8], "scores er endret. Bruk sorted(scores), som lager en ny liste, ikke scores.sort()."
```
:::

## Lister i lister
`X[1]` gir rad 1 og `X[1][0]` rad 1, kolonne 0, mens `len(X)` og `len(X[0])` gir antall rader og kolonner.

::: oppgave Datatabell
Lag `rad1` (rad 1), `verdi` (rad 2, kolonne 0), `n_rader` og `n_kolonner` fra `X`.
```start
X = [[5.1, 3.5],
     [4.9, 3.0],
     [6.2, 2.9]]

rad1 = ...
verdi = ...
n_rader = ...
n_kolonner = ...
```
```losning
X = [[5.1, 3.5],
     [4.9, 3.0],
     [6.2, 2.9]]

rad1 = X[1]
verdi = X[2][0]
n_rader = len(X)
n_kolonner = len(X[0])
```
```test
assert rad1 == [4.9, 3.0], f"rad1 ble {rad1!r}. Rad 1 er den andre raden: X[1]."
assert verdi == 6.2, f"verdi ble {verdi!r}. Bruk X[2][0]."
assert n_rader == 3, "n_rader skal være len(X) = 3."
assert n_kolonner == 2, "n_kolonner skal være len(X[0]) = 2."
```
:::

--- LEKSJON 1.8 | for-lokker | for-løkker
`for r in rewards:` kjører den innrykkede koden én gang per element, og `range(n)` gir `0, 1, ..., n-1` (`range(2, 6)` og `range(0, 20, 5)` gir start og steg).

::: oppgave Summer og kvadrer
Bruk løkker til å legge sammen `rewards` i `total` og fylle `kvadrater` med hvert element i andre, og lag `tall = list(range(0, 20, 5))`.
```start
rewards = [-1, -1, -1, 10]

total = 0
kvadrater = []
# skriv løkka/løkkene her

tall = ...
```
```losning
rewards = [-1, -1, -1, 10]

total = 0
kvadrater = []
for r in rewards:
    total += r
    kvadrater.append(r ** 2)

tall = list(range(0, 20, 5))
```
```test
assert total == 7, f"total ble {total}, men summen av rewards er 7. Bruk total += r i løkka."
assert kvadrater == [1, 1, 1, 100], f"kvadrater ble {kvadrater}. Bruk kvadrater.append(r ** 2) i løkka."
assert tall == [0, 5, 10, 15], f"tall ble {tall!r}. range(0, 20, 5) stopper før 20."
```
:::

## enumerate og zip
`for i, x in enumerate(liste, start=1):` gir nummer og element, og `for t, p in zip(a, b):` går gjennom to lister i takt (og stopper ved den korteste).

::: oppgave Nummererte epoker
Bruk `enumerate` med `start=1` til å fylle `linjer` med tekstene `"Epoke 1: 0.9"`, `"Epoke 2: 0.5"` og `"Epoke 3: 0.3"`.
```start
losses = [0.9, 0.5, 0.3]

linjer = []
# skriv løkka her
```
```losning
losses = [0.9, 0.5, 0.3]

linjer = []
for i, loss in enumerate(losses, start=1):
    linjer.append(f"Epoke {i}: {loss}")
```
```test
assert linjer == ["Epoke 1: 0.9", "Epoke 2: 0.5", "Epoke 3: 0.3"], f"linjer ble {linjer!r}. Bruk enumerate(losses, start=1) og f\"Epoke {{i}}: {{loss}}\"."
```
:::

::: oppgave Accuracy med zip
Tell riktige prediksjoner i `riktige` med en `zip`-løkke, og regn ut `accuracy` som andelen riktige.
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
Legg sammen `(sann - pred) ** 2` for alle punktene i `sum_kvadrat`, og lag `mse` som gjennomsnittet.
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

## Løkke i løkke
En løkke inni en annen prøver alle kombinasjoner, akkurat som et grid search.

::: oppgave Mini grid search
Fyll `kombinasjoner` med `f"lr={lr}, depth={depth}"` for hver `lr` i `[0.1, 0.01]` og hver `depth` i `[2, 4, 8]` (ytre løkke over `lr`).
```start
kombinasjoner = []
# skriv de to løkkene her
```
```losning
kombinasjoner = []
for lr in [0.1, 0.01]:
    for depth in [2, 4, 8]:
        kombinasjoner.append(f"lr={lr}, depth={depth}")
```
```test
assert len(kombinasjoner) == 6, f"kombinasjoner har {len(kombinasjoner)} elementer, men 2 · 3 = 6 kombinasjoner skal med."
assert kombinasjoner[0] == "lr=0.1, depth=2" and kombinasjoner[-1] == "lr=0.01, depth=8", f"Første skal være 'lr=0.1, depth=2' og siste 'lr=0.01, depth=8', men fikk {kombinasjoner[0]!r} og {kombinasjoner[-1]!r}."
```
:::

--- LEKSJON 1.9 | while-break-continue | while, break og continue
`while epsilon > 0.05:` gjentar så lenge betingelsen er sann, så noe i løkka må gjøre den usann, ellers henger programmet.

::: oppgave Epsilon-decay
Gang `epsilon` med 0.9 i en `while`-løkke så lenge den er større enn 0.05, og tell rundene i `episode`.
```start
epsilon = 1.0
episode = 0
# skriv while-løkka her
```
```losning
epsilon = 1.0
episode = 0
while epsilon > 0.05:
    epsilon *= 0.9
    episode += 1
```
```test
assert epsilon <= 0.05, f"epsilon ble {epsilon}. Løkka skal fortsette til epsilon ikke lenger er større enn 0.05."
assert episode == 29, f"episode ble {episode}, men skal være 29. Øk episode med 1 i hver runde."
```
:::

::: oppgave Hvor mange steg til konvergens?
Minimer `(w - 3) ** 2` med gradient descent (`gradient = 2 * (w - 3)`) så lenge `abs(w - 3) >= 0.001`, og tell stegene i `steg`.
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
assert steg == 36, f"steg ble {steg}, men skal være 36. Øk steg med 1 i hver runde, og bruk gradient = 2 * (w - 3)."
```
:::

## break og continue
`break` avslutter løkka med en gang (også `while True:`), og `continue` hopper til neste runde.

::: oppgave Stopp når episoden er ferdig
Løkka krasjer med `IndexError`. Legg til `break` når `reward` er 10, slik at agenten stopper i målet.
```start
rewards = [-1, -1, -1, 10, -1, -1]
total = 0
for step in range(50):
    reward = rewards[step]
    total += reward
    # avslutt løkka her når reward er 10
```
```losning
rewards = [-1, -1, -1, 10, -1, -1]
total = 0
for step in range(50):
    reward = rewards[step]
    total += reward
    if reward == 10:
        break
```
```test
assert total == 7, f"total ble {total}, men skal være -1 - 1 - 1 + 10 = 7."
assert step == 3, f"step ble {step}, men løkka skal stoppe i steg 3 der reward er 10."
```
:::

::: oppgave Early stopping
Finn den første epoken `i` der `val_losses[i]` er større enn `val_losses[i - 1]`, lagre den i `stopp_epoke` og avslutt løkka med `break`.
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

::: oppgave Hopp over manglende verdier
Bruk `continue` til å hoppe over `None` i `målinger`, og regn ut `total`, `antall` og `snitt` for de gyldige verdiene.
```start
målinger = [2.0, None, 3.5, None, 4.5]
total = 0
antall = 0
for m in målinger:
    ...

snitt = ...
```
```losning
målinger = [2.0, None, 3.5, None, 4.5]
total = 0
antall = 0
for m in målinger:
    if m is None:
        continue
    total += m
    antall += 1

snitt = total / antall
```
```test
assert antall == 3, f"antall ble {antall}, men det er 3 gyldige målinger."
assert abs(total - 10.0) < 1e-9, f"total ble {total}, men skal være 2.0 + 3.5 + 4.5 = 10.0."
assert isinstance(snitt, float) and abs(snitt - 10 / 3) < 1e-9, "snitt skal være total / antall."
```
:::

::: oppgave while True
Doble `n` i en `while True:`-løkke og bruk `break` når `n` blir større enn 100.
```start
n = 1
# skriv løkka her
```
```losning
n = 1
while True:
    n *= 2
    if n > 100:
        break
```
```test
assert n == 128, f"n ble {n}, men skal være 128 (første potens av 2 over 100)."
```
:::

--- LEKSJON 1.10 | dictionaries | Dictionaries
`params = {"max_depth": 3}` kobler nøkler til verdier, `params["max_depth"]` slår opp, `params["ny"] = 2` legger til eller endrer, og `"ny" in params` sjekker om nøkkelen finnes.

::: oppgave Hyperparametere i en dict
Endre `max_depth` til 5, legg til `"min_samples_leaf"` med verdien 2, og lag `har_criterion` (finnes nøkkelen `"criterion"?`) og `n_params` (antall nøkler).
```start
params = {"max_depth": 3}
# endre og legg til her

har_criterion = ...
n_params = ...
```
```losning
params = {"max_depth": 3}
params["max_depth"] = 5
params["min_samples_leaf"] = 2

har_criterion = "criterion" in params
n_params = len(params)
```
```test
assert params == {"max_depth": 5, "min_samples_leaf": 2}, f"params ble {params}. Bruk params[\"max_depth\"] = 5 og params[\"min_samples_leaf\"] = 2."
assert har_criterion is False, "har_criterion skal være \"criterion\" in params, som er False."
assert n_params == 2, "n_params skal være len(params) = 2."
```
:::

## get og telling
En manglende nøkkel gir `KeyError` med `d[k]`, mens `d.get(k, standard)` gir standardverdien (og `None` uten). Mønsteret `antall[k] = antall.get(k, 0) + 1` teller forekomster.

::: oppgave Unngå KeyError
Rett linjen slik at `criterion` blir `"gini"` når nøkkelen mangler, og lag `splitter` med `.get` uten standardverdi.
```start
params = {"max_depth": 3}

criterion = params["criterion"]
splitter = ...
```
```losning
params = {"max_depth": 3}

criterion = params.get("criterion", "gini")
splitter = params.get("splitter")
```
```test
assert criterion == "gini", f"criterion ble {criterion!r}. Bruk params.get(\"criterion\", \"gini\")."
assert splitter is None, "splitter skal være params.get(\"splitter\"), som gir None når nøkkelen mangler."
```
:::

::: oppgave Tell klassene
Tell hvor mange ganger hver etikett forekommer i `y`, og lagre det i dicten `antall` med `.get(..., 0)`.
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

## Gå gjennom en dict
`d.keys()`, `d.values()` og `d.items()` gir nøkler, verdier og par, og `for k, v in d.items():` gir begge i løkka.

::: oppgave Fra antall til andeler
Lag dicten `andeler` der hver klasse har sitt antall delt på `total`, med en løkke over `antall.items()`.
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

## Lister og dicts som verdier
`history["loss"].append(0.5)` legger til i en liste inne i en dict, og `Q["start"]["høyre"]` slår opp i en dict inne i en dict.

::: oppgave Historikk og Q-tabell
For `epoch` i `range(3)`: legg `1.0 / (epoch + 1)` til i `history["loss"]` og `1.2 / (epoch + 1)` i `history["val_loss"]`. Sett så Q-verdien for handlingen `"høyre"` i tilstanden `"start"` til 0.5.
```start
history = {"loss": [], "val_loss": []}
# fyll history her

Q = {
    "start": {"venstre": 0.0, "høyre": 0.0},
    "midt": {"venstre": 0.0, "høyre": 0.0},
}
# oppdater Q her
```
```losning
history = {"loss": [], "val_loss": []}
for epoch in range(3):
    history["loss"].append(1.0 / (epoch + 1))
    history["val_loss"].append(1.2 / (epoch + 1))

Q = {
    "start": {"venstre": 0.0, "høyre": 0.0},
    "midt": {"venstre": 0.0, "høyre": 0.0},
}
Q["start"]["høyre"] = 0.5
```
```test
assert len(history["loss"]) == 3 and len(history["val_loss"]) == 3, "history[\"loss\"] og history[\"val_loss\"] skal ha 3 verdier hver."
assert abs(history["loss"][2] - 1 / 3) < 1e-9 and abs(history["val_loss"][1] - 0.6) < 1e-9, f"Feil verdier i history: {history}. Bruk 1.0 / (epoch + 1) og 1.2 / (epoch + 1)."
assert Q["start"]["høyre"] == 0.5, "Sett Q[\"start\"][\"høyre\"] = 0.5."
assert Q["midt"]["høyre"] == 0.0 and Q["start"]["venstre"] == 0.0, "Bare Q[\"start\"][\"høyre\"] skal endres."
```
:::

--- LEKSJON 1.11 | tupler | Tupler og utpakking
`posisjon = (3, 7)` er en tuple: som en liste, men den kan ikke endres (`posisjon[0] = 4` gir `TypeError`). Det er kommaet som lager tuplen, så `t = 1, 2` er også en tuple.

::: oppgave Ny tuple i stedet for endring
Linjen `posisjon[0] = 4` krasjer. Erstatt den med en linje som gir `posisjon` en helt ny tuple med 4 først og samme kolonne som før.
```start
posisjon = (3, 7)
posisjon[0] = 4
```
```losning
posisjon = (3, 7)
posisjon = (4, posisjon[1])
```
```test
assert isinstance(posisjon, tuple), "posisjon skal fortsatt være en tuple, for eksempel (4, 7)."
assert posisjon == (4, 7), f"posisjon ble {posisjon!r}, men skal være (4, 7)."
```
:::

## Utpakking
`rad, kol = (3, 7)` gir flere variabler verdi på én gang, og `a, b = b, a` bytter verdiene. Feil antall navn gir `ValueError`.

::: oppgave Pakk ut shape og bytt
Pakk ut `shape` til `n_samples` og `n_features` på én linje, og bytt verdiene til `a` og `b` på én linje.
```start
shape = (150, 4)
a, b = 1, 2

n_samples = n_features = ...
# bytt a og b her
```
```losning
shape = (150, 4)
a, b = 1, 2

n_samples, n_features = shape
a, b = b, a
```
```test
assert n_samples == 150 and n_features == 4, f"n_samples og n_features ble {n_samples!r} og {n_features!r}. Skriv n_samples, n_features = shape."
assert a == 2 and b == 1, f"a og b ble {a} og {b}. Bytt med a, b = b, a."
```
:::

## Returverdier og _
`return a, b` gir tilbake en tuple som pakkes ut som i `X_train, X_test, y_train, y_test = train_test_split(...)`, og `_` brukes for verdier du ikke trenger.

::: oppgave Pakk ut en transisjon
Pakk ut `min_og_maks([4, 1, 9, 3])` til `laveste` og `høyeste`, og `transisjon` (`state, action, reward, next_state, done`) til `state`, `reward`, `next_state` og `done` med `_` for handlingen.
```start
def min_og_maks(verdier):
    return min(verdier), max(verdier)

transisjon = (4, 2, -1.0, 5, False)

laveste = høyeste = ...
state = reward = next_state = done = ...
```
```losning
def min_og_maks(verdier):
    return min(verdier), max(verdier)

transisjon = (4, 2, -1.0, 5, False)

laveste, høyeste = min_og_maks([4, 1, 9, 3])
state, _, reward, next_state, done = transisjon
```
```test
assert laveste == 1 and høyeste == 9, f"laveste og høyeste ble {laveste!r} og {høyeste!r}. Skriv laveste, høyeste = min_og_maks([4, 1, 9, 3])."
assert state == 4, "state skal være første element i transisjonen (4)."
assert reward == -1.0, "reward skal være tredje element (-1.0). Har du husket _ for handlingen?"
assert next_state == 5, "next_state skal være fjerde element (5)."
assert done is False, "done skal være siste element (False)."
```
:::

## Utpakking i løkker og stjerne
`for navn, acc in resultater:` og `for i, (x, y) in enumerate(zip(xs, ys)):` pakker ut hver runde, og `første, *resten = liste` samler resten i en liste.

::: oppgave Finn beste modell
Gå gjennom `resultater` med `for navn, acc in resultater:` og lagre navnet og accuracy til den beste modellen i `beste_navn` og `beste_acc`.
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

::: oppgave Stjerne og nøstet utpakking
Lag `første` og `resten` fra `verdier` med stjerne-utpakking, og fyll `rader` med `f"{i}: {x} -> {y}"` med `for i, (x, y) in enumerate(zip(xs, ys)):`.
```start
verdier = [10, 20, 30, 40]
xs = [1.0, 2.0]
ys = [3.0, 5.0]

første = resten = ...
rader = []
# skriv løkka her
```
```losning
verdier = [10, 20, 30, 40]
xs = [1.0, 2.0]
ys = [3.0, 5.0]

første, *resten = verdier
rader = []
for i, (x, y) in enumerate(zip(xs, ys)):
    rader.append(f"{i}: {x} -> {y}")
```
```test
assert første == 10 and resten == [20, 30, 40], f"første og resten ble {første!r} og {resten!r}. Skriv første, *resten = verdier."
assert rader == ["0: 1.0 -> 3.0", "1: 2.0 -> 5.0"], f"rader ble {rader!r}. Bruk f\"{{i}}: {{x}} -> {{y}}\"."
```
:::

--- LEKSJON 1.12 | funksjoner | Funksjoner
`def kvadrat(x):` lager en funksjon med innrykket kropp, og `return` sender verdien tilbake. Uten `return` gir funksjonen `None`, og `print` er ikke det samme som `return`.

::: oppgave return, ikke print
Fullfør `kvadrat(x)` så den returnerer `x ** 2`, og endre `dobbel(x)` så den **returnerer** `x * 2` i stedet for å skrive det ut.
```start
def kvadrat(x):
    ...


def dobbel(x):
    print(x * 2)
```
```losning
def kvadrat(x):
    return x ** 2


def dobbel(x):
    return x * 2
```
```test
assert kvadrat(3) == 9 and kvadrat(1.5) == 2.25, "kvadrat(3) skal gi 9 og kvadrat(1.5) 2.25. Husk return."
r = dobbel(5)
assert r is not None, "dobbel(5) gir None. Bytt print(x * 2) med return x * 2."
assert r == 10, f"dobbel(5) ga {r!r}, men skal gi 10."
```
:::

::: oppgave Skriv mse som funksjon
Lag `mse(y_true, y_pred)` som **returnerer** gjennomsnittet av de kvadrerte feilene for lister av alle lengder.
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

## Standardverdier og nøkkelordargumenter
`def f(p, threshold=0.5):` gir `threshold` en standardverdi, og `f(p, threshold=0.8)` er et nøkkelordargument. Parametre uten standardverdi står først.

::: oppgave Prediker klasse med terskel
Lag `predict_label(p, threshold=0.5)` som returnerer `1` hvis `p >= threshold`, og `0` ellers.
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

## Docstring og type hints
En docstring i `"""..."""` rett under `def` vises av `help()`, og type hints som `episode: int` og `-> float` er bare dokumentasjon som Python ikke sjekker.

::: oppgave Utforskingsrate
Gi `update_explore_rate` type hints (`episode: int`, `epsilon_min: float`, `decay: float = 0.99`, `-> float`) og en docstring, og la den returnere `max(epsilon_min, decay ** episode)`.
```start
def update_explore_rate(episode, epsilon_min, decay=0.99):
    return None
```
```losning
def update_explore_rate(episode: int, epsilon_min: float, decay: float = 0.99) -> float:
    """Returnerer utforskingsraten, som synker eksponentielt, men aldri under epsilon_min."""
    return max(epsilon_min, decay ** episode)
```
```test
r = update_explore_rate(0, 0.01)
assert r is not None and abs(r - 1.0) < 1e-9, f"update_explore_rate(0, 0.01) ga {r!r}, men skal gi 1.0 (0.99 ** 0 = 1)."
assert abs(update_explore_rate(100, 0.01) - 0.99 ** 100) < 1e-9, "update_explore_rate(100, 0.01) skal gi 0.99 ** 100."
assert abs(update_explore_rate(1000, 0.01) - 0.01) < 1e-9, "update_explore_rate(1000, 0.01) skal gi 0.01, fordi raten aldri går under epsilon_min."
assert abs(update_explore_rate(2, 0.01, decay=0.5) - 0.25) < 1e-9, "Med decay=0.5 og episode 2 skal svaret bli 0.25. Bruk parameteren decay."
assert update_explore_rate.__doc__, "Legg til en docstring i \"\"\"...\"\"\" rett under def-linjen."
h = update_explore_rate.__annotations__
assert h.get("episode") is int and h.get("epsilon_min") is float and h.get("decay") is float, "Legg til type hints: episode: int, epsilon_min: float, decay: float = 0.99."
assert h.get("return") is float, "Legg til -> float før kolonet på def-linjen."
```
:::

## Funksjoner som verdier, *args og **kwargs
En funksjon kan sendes inn uten parenteser (`lag_utdata(relu, verdier)`), `*args` og `**kwargs` samler ekstra argumenter i en tuple og en dict, og `train(**cfg)` pakker ut en dict til nøkkelordargumenter.

::: oppgave Aktivering som argument
Lag `lag_utdata(aktivering, verdier)` som returnerer en liste med `aktivering(v)` for hver `v`, og lagre `lag_utdata(relu, [-2, -1, 0, 1, 2])` i `resultat`.
```start
def relu(z):
    return max(0, z)

# lag funksjonen lag_utdata her

resultat = ...
```
```losning
def relu(z):
    return max(0, z)

def lag_utdata(aktivering, verdier):
    ut = []
    for v in verdier:
        ut.append(aktivering(v))
    return ut

resultat = lag_utdata(relu, [-2, -1, 0, 1, 2])
```
```test
assert "lag_utdata" in dir(), "Lag funksjonen lag_utdata(aktivering, verdier)."
assert resultat == [0, 0, 0, 1, 2], f"resultat ble {resultat!r}, men skal være [0, 0, 0, 1, 2]. Send inn relu uten parenteser."
def _dobbel(z):
    return 2 * z
assert lag_utdata(_dobbel, [1, 2]) == [2, 4], "lag_utdata skal bruke funksjonen den får inn, ikke alltid relu."
```
:::

::: oppgave args, kwargs og **cfg
Fullfør `tell(*args, **kwargs)` så den returnerer `len(args), len(kwargs)`, lag `cfg` slik at `train(**cfg)` får `lr=0.001` og `batch_size=64`, og lagre `train(**cfg, epochs=200)` i `resultat`.
```start
def train(lr=0.01, epochs=10, batch_size=32):
    return f"lr={lr}, epochs={epochs}, batch_size={batch_size}"

def tell(*args, **kwargs):
    ...

cfg = ...
resultat = ...
```
```losning
def train(lr=0.01, epochs=10, batch_size=32):
    return f"lr={lr}, epochs={epochs}, batch_size={batch_size}"

def tell(*args, **kwargs):
    return len(args), len(kwargs)

cfg = {"lr": 0.001, "batch_size": 64}
resultat = train(**cfg, epochs=200)
```
```test
assert tell(1, 2, lr=0.1) == (2, 1), f"tell(1, 2, lr=0.1) ga {tell(1, 2, lr=0.1)!r}, men skal gi (2, 1): to posisjonelle og ett nøkkelordargument."
assert isinstance(cfg, dict), "cfg skal være en dict, for eksempel {\"lr\": 0.001, \"batch_size\": 64}."
assert train(**cfg) == "lr=0.001, epochs=10, batch_size=64", f"train(**cfg) ga {train(**cfg)!r}. cfg skal ha nøklene \"lr\" og \"batch_size\"."
assert resultat == "lr=0.001, epochs=200, batch_size=64", f"resultat ble {resultat!r}. Skriv train(**cfg, epochs=200)."
```
:::

--- LEKSJON 1.13 | comprehensions-lambda | List comprehensions og lambda
`[x ** 2 for x in tall]` lager en ny liste med ett uttrykk per element, og `[x for x in tall if x > 0]` filtrerer.

::: oppgave Kvadrater og filter
Lag med list comprehensions: `kvadrater` (hvert tall i andre), `gyldige` (verdiene i `losses` som ikke er `None`) og `partall` (partallene i `range(10)`).
```start
tall = [1, 2, 3, 4]
losses = [0.9, None, 0.5, 0.3, None]

kvadrater = ...
gyldige = ...
partall = ...
```
```losning
tall = [1, 2, 3, 4]
losses = [0.9, None, 0.5, 0.3, None]

kvadrater = [x ** 2 for x in tall]
gyldige = [l for l in losses if l is not None]
partall = [i for i in range(10) if i % 2 == 0]
```
```test
assert kvadrater == [1, 4, 9, 16], f"kvadrater ble {kvadrater!r}. Skriv [x ** 2 for x in tall]."
assert gyldige == [0.9, 0.5, 0.3], f"gyldige ble {gyldige!r}. Legg til if l is not None på slutten."
assert partall == [0, 2, 4, 6, 8], f"partall ble {partall!r}. Bruk if i % 2 == 0."
```
:::

## Fra sannsynligheter til klasser
`[1 if p > 0.5 else 0 for p in y_proba]` velger en verdi per element (`if`/`else` foran løkka), mens `if` bak løkka filtrerer bort elementer.

::: oppgave Terskling med list comprehension
Lag `y_pred` (1 hvis sannsynligheten er **større enn** 0.5, ellers 0) og `sikre` (bare sannsynlighetene over 0.9 eller under 0.1).
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

## Rett inn i sum, og dict comprehension
`sum(t == p for t, p in zip(a, b))` teller treff uten hakeparenteser, og `{k: v / n for k, v in d.items()}` lager en dict.

::: oppgave MSE og accuracy på én linje
Lag `mse` og `accuracy` med hvert sitt uttrykk med `sum(...)` og `zip`, uten vanlige løkker.
```start
y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]
k_true = [1, 0, 1, 1]
k_pred = [1, 1, 1, 0]

mse = ...
accuracy = ...
```
```losning
y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]
k_true = [1, 0, 1, 1]
k_pred = [1, 1, 1, 0]

mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)
accuracy = sum(t == p for t, p in zip(k_true, k_pred)) / len(k_true)
```
```test
assert isinstance(mse, float) and abs(mse - 0.375) < 1e-9, f"mse ble {mse!r}, men skal være 0.375."
assert isinstance(accuracy, float) and abs(accuracy - 0.5) < 1e-9, f"accuracy ble {accuracy!r}, men skal være 0.5 (2 av 4 riktige)."
```
:::

::: oppgave Andeler og Q-tabell
Lag `andeler` med en dict comprehension over `antall.items()`, og `q_tabell` som en liste med `n_states` rader med `n_actions` nuller, med `[[0.0] * n_actions for _ in range(n_states)]`.
```start
antall = {"katt": 3, "hund": 2, "fugl": 1}
n = sum(antall.values())
n_states, n_actions = 3, 4

andeler = ...
q_tabell = ...
```
```losning
antall = {"katt": 3, "hund": 2, "fugl": 1}
n = sum(antall.values())
n_states, n_actions = 3, 4

andeler = {klasse: k / n for klasse, k in antall.items()}
q_tabell = [[0.0] * n_actions for _ in range(n_states)]
```
```test
assert isinstance(andeler, dict) and set(andeler) == {"katt", "hund", "fugl"}, "andeler skal være en dict med samme nøkler som antall."
assert abs(andeler["katt"] - 0.5) < 1e-9 and abs(andeler["fugl"] - 1 / 6) < 1e-9, f"andeler ble {andeler}. Del hvert antall på n."
assert isinstance(q_tabell, list) and len(q_tabell) == 3 and all(r == [0.0] * 4 for r in q_tabell), "q_tabell skal være 3 rader med 4 nuller."
assert q_tabell[0] is not q_tabell[1], "Radene er samme liste! [[0.0] * 4] * 3 kopierer bare navnet. Bruk [[0.0] * n_actions for _ in range(n_states)]."
```
:::

## lambda og key=
`lambda r: r[1]` er en navnløs funksjon, mest brukt som `key=` i `sorted(..., key=..., reverse=True)`, `min` og `max`. `min(range(len(xs)), key=lambda i: xs[i])` er argmin, og `max(q, key=q.get)` gir nøkkelen med størst verdi.

::: oppgave Sorter og finn argmin
Lag `rangert` (`resultater` sortert etter accuracy, høyest først), `beste_epoke` (**indeksen** til laveste `val_losses`) og `beste_handling` (nøkkelen med størst verdi i `q`).
```start
resultater = [("tre", 0.81), ("skog", 0.88), ("knn", 0.79)]
val_losses = [0.9, 0.62, 0.55, 0.58, 0.71]
q = {"opp": 0.1, "ned": -0.4, "venstre": 0.7, "høyre": 0.3}

rangert = ...
beste_epoke = ...
beste_handling = ...
```
```losning
resultater = [("tre", 0.81), ("skog", 0.88), ("knn", 0.79)]
val_losses = [0.9, 0.62, 0.55, 0.58, 0.71]
q = {"opp": 0.1, "ned": -0.4, "venstre": 0.7, "høyre": 0.3}

rangert = sorted(resultater, key=lambda r: r[1], reverse=True)
beste_epoke = min(range(len(val_losses)), key=lambda i: val_losses[i])
beste_handling = max(q, key=q.get)
```
```test
assert rangert == [("skog", 0.88), ("tre", 0.81), ("knn", 0.79)], f"rangert ble {rangert!r}. Bruk sorted(resultater, key=lambda r: r[1], reverse=True)."
assert beste_epoke == 2, f"beste_epoke ble {beste_epoke!r}, men skal være indeksen 2 (der tapet er 0.55), ikke selve tapet."
assert beste_handling == "venstre", f"beste_handling ble {beste_handling!r}, men skal være 'venstre' (Q-verdi 0.7)."
```
:::

--- LEKSJON 1.14 | klasser | Klasser og objekter
`class Teller:` lager en klasse, `__init__(self, ...)` kjøres når du skriver `Teller()`, og `self.n = 0` lagrer et attributt som alle metodene (med `self` som første parameter) kan bruke. Hvert objekt har sine egne attributter.

::: oppgave Teller
Fullfør `Teller`: `__init__` skal lagre `start` i `self.n`, og `øk` skal øke `self.n` med 1.
```start
class Teller:
    def __init__(self, start=0):
        pass

    def øk(self):
        pass
```
```losning
class Teller:
    def __init__(self, start=0):
        self.n = start

    def øk(self):
        self.n += 1
```
```test
a = Teller()
b = Teller(10)
assert hasattr(a, "n"), "Objektet mangler attributtet n. Skriv self.n = start i __init__."
a.øk()
a.øk()
b.øk()
assert a.n == 2, f"a.n ble {a.n}, men skal være 2 etter to kall til øk()."
assert b.n == 11, f"b.n ble {b.n}, men skal være 11 (startet på 10). Hvert objekt har sin egen self.n."
```
:::

## En modell med fit og predict
ML-mønsteret er at `__init__` setter parametrene til `None`, `fit` lagrer det som læres på `self`, og `predict` bruker det lagrede.

::: oppgave MeanModel
Fullfør `fit` (lagre gjennomsnittet av `y` i `self.mean` og legg MSE-en til i `self.loss_history`) og `predict` (returner en liste med `self.mean` for hver rad i `X`).
```start
class MeanModel:
    def __init__(self):
        self.mean = None
        self.loss_history = []

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
```
```losning
class MeanModel:
    def __init__(self):
        self.mean = None
        self.loss_history = []

    def fit(self, X, y):
        self.mean = sum(y) / len(y)
        mse = sum((yi - self.mean) ** 2 for yi in y) / len(y)
        self.loss_history.append(mse)

    def predict(self, X):
        return [self.mean for _ in X]
```
```test
m = MeanModel()
m.fit([[1.0], [2.0], [3.0], [4.0]], [2.0, 4.0, 5.0, 9.0])
assert m.mean is not None and abs(m.mean - 5.0) < 1e-9, f"self.mean ble {m.mean!r}, men skal være 5.0. Husk self. foran mean."
assert len(m.loss_history) == 1 and abs(m.loss_history[0] - 6.5) < 1e-9, f"loss_history ble {m.loss_history}, men skal være [6.5]."
p = m.predict([[10.0], [20.0]])
assert p == [5.0, 5.0], f"predict med 2 rader skal gi [5.0, 5.0], men ga {p!r}."
```
:::

::: oppgave Majoritetsklassifiserer
Fullfør `MajorityClassifier`: `fit` teller klassene i `y` og lagrer den vanligste i `self.majority` (bruk `max(antall, key=antall.get)`), og `predict` returnerer en liste med `self.majority` per rad.
```start
class MajorityClassifier:
    def __init__(self):
        self.majority = None

    def fit(self, X, y):
        pass

    def predict(self, X):
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

## Hjelpemetoder og return self
`self._mean(verdier)` kaller en annen metode i samme objekt (`_` betyr «intern»), og `return self` i `fit` gjør at du kan skrive `model.fit(X, y).predict(X)`.

::: oppgave Terskelklassifiserer
Fullfør `fit`: sett `self.threshold` til snittet av `_mean` for klasse 0 og klasse 1, pluss `self.margin`, og returner `self`. `predict` skal gi 1 når `x > self.threshold`, ellers 0.
```start
class ThresholdClassifier:
    def __init__(self, margin=0.0):
        self.margin = margin
        self.threshold = None

    def _mean(self, values):
        return sum(values) / len(values)

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
```
```losning
class ThresholdClassifier:
    def __init__(self, margin=0.0):
        self.margin = margin
        self.threshold = None

    def _mean(self, values):
        return sum(values) / len(values)

    def fit(self, X, y):
        snitt0 = self._mean([x for x, c in zip(X, y) if c == 0])
        snitt1 = self._mean([x for x, c in zip(X, y) if c == 1])
        self.threshold = (snitt0 + snitt1) / 2 + self.margin
        return self

    def predict(self, X):
        return [1 if x > self.threshold else 0 for x in X]
```
```test
X = [1.0, 1.5, 2.0, 6.0, 7.0, 8.0]
y = [0, 0, 0, 1, 1, 1]
clf = ThresholdClassifier()
r = clf.fit(X, y)
assert clf.threshold is not None and abs(clf.threshold - 4.25) < 1e-9, f"threshold ble {clf.threshold!r}, men skal være (1.5 + 7.0) / 2 = 4.25."
assert r is clf, "fit skal avslutte med return self."
assert clf.predict([0.5, 4.0, 4.6, 9.0]) == [0, 0, 1, 1], "predict([0.5, 4.0, 4.6, 9.0]) skal gi [0, 0, 1, 1]."
clf2 = ThresholdClassifier(margin=1.0).fit(X, y)
assert abs(clf2.threshold - 5.25) < 1e-9, "Med margin=1.0 skal threshold bli 5.25. Legg til self.margin."
```
:::

## Vanlige feil
Glemt `self` i parameterlisten gir `TypeError` (Python sender inn objektet, men metoden tar ikke imot det), og glemt `self.` foran et attributt gir `NameError`.

::: oppgave Rett klassen
Klassen har to feil. Rett dem slik at `Modell().predict([[1], [2]])` gir `[3.0, 3.0]`.
```start
class Modell:
    def __init__(self):
        self.mean = 3.0

    def predict(X):
        return [mean for _ in X]
```
```losning
class Modell:
    def __init__(self):
        self.mean = 3.0

    def predict(self, X):
        return [self.mean for _ in X]
```
```test
try:
    p = Modell().predict([[1], [2]])
except TypeError:
    raise AssertionError("TypeError: predict mangler self som første parameter.")
except NameError:
    raise AssertionError("NameError: mean finnes ikke som vanlig variabel. Skriv self.mean.")
assert p == [3.0, 3.0], f"predict ga {p!r}, men skal gi [3.0, 3.0]."
```
:::

--- LEKSJON 1.15 | arv | Arv og super()
`class AlltidEn(Modell):` arver alle metodene til `Modell`, og en metode med samme navn i underklassen overstyrer foreldrens, så `self.predict(X)` bruker alltid objektets egen versjon.

::: oppgave Underklasser med egen predict
Lag `AlltidEn(Modell)` med `predict` som gir 1 for hver `x`, og `Terskel(Modell)` med `predict` som gir 1 når `x > 5`, ellers 0. `score` skal du ikke skrive på nytt.
```start
class Modell:
    def predict(self, X):
        return [0 for _ in X]

    def score(self, X, y):
        y_pred = self.predict(X)
        return sum(p == t for p, t in zip(y_pred, y)) / len(y)


# lag AlltidEn og Terskel her
```
```losning
class Modell:
    def predict(self, X):
        return [0 for _ in X]

    def score(self, X, y):
        y_pred = self.predict(X)
        return sum(p == t for p, t in zip(y_pred, y)) / len(y)


class AlltidEn(Modell):
    def predict(self, X):
        return [1 for _ in X]


class Terskel(Modell):
    def predict(self, X):
        return [1 if x > 5 else 0 for x in X]
```
```test
assert "AlltidEn" in dir() and "Terskel" in dir(), "Lag klassene AlltidEn og Terskel."
assert issubclass(AlltidEn, Modell) and issubclass(Terskel, Modell), "Begge skal arve fra Modell: class AlltidEn(Modell):"
X = [2, 4, 6, 8]
y = [0, 0, 1, 1]
assert AlltidEn().predict(X) == [1, 1, 1, 1], "AlltidEn().predict skal gi 1 for hver x."
assert Terskel().predict(X) == [0, 0, 1, 1], "Terskel().predict skal gi 1 når x > 5."
assert abs(AlltidEn().score(X, y) - 0.5) < 1e-9 and abs(Terskel().score(X, y) - 1.0) < 1e-9, "score arvet fra Modell skal bruke den overstyrte predict."
```
:::

## super().__init__()
Har underklassen sin egen `__init__`, må den kalle `super().__init__(...)` først, ellers mangler attributtene fra foreldreklassen.

::: oppgave TerskelModell
Lag `__init__` i `TerskelModell` som kaller `super().__init__("terskelmodell")` og lagrer `terskel` i `self.terskel`.
```start
class Modell:
    def __init__(self, navn):
        self.navn = navn
        self.er_trent = False

    def fit(self, X, y):
        self.er_trent = True


class TerskelModell(Modell):
    def __init__(self, terskel=0.5):
        self.terskel = terskel
```
```losning
class Modell:
    def __init__(self, navn):
        self.navn = navn
        self.er_trent = False

    def fit(self, X, y):
        self.er_trent = True


class TerskelModell(Modell):
    def __init__(self, terskel=0.5):
        super().__init__("terskelmodell")
        self.terskel = terskel
```
```test
m = TerskelModell(0.7)
assert getattr(m, "navn", None) == "terskelmodell", "self.navn mangler. Kall super().__init__(\"terskelmodell\") i __init__."
assert getattr(m, "er_trent", None) is False, "self.er_trent skal være False før fit. Den settes av Modell.__init__."
assert m.terskel == 0.7, "self.terskel skal være 0.7."
m.fit([], [])
assert m.er_trent is True, "fit arvet fra Modell skal sette er_trent til True."
```
:::

::: oppgave Grådig agent
Lag `GraadigAgent(Agent)`: `__init__(self, q_values)` kaller `super().__init__(len(q_values))` og lagrer `self.q_values`, og `act` øker `self.steps` med 1 og returnerer indeksen til største q-verdi.
```start
class Agent:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.steps = 0

    def act(self, state):
        raise NotImplementedError


class GraadigAgent(Agent):
    def __init__(self, q_values):
        pass

    def act(self, state):
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

## Et lite RL-miljø
Miljøer arver fra en `Env`-klasse og lager `reset()`, som returnerer `state, info`, og `step(action)`, som returnerer `state, reward, terminated, info`.

::: oppgave Korridor
Skriv `step`: handling 1 flytter `self.state` ett hakk til høyre (maks `self.lengde - 1`), ellers til venstre (min 0). `terminated` er sann i siste rute, og `reward` er 10 da og ellers -1.
```start
class Env:
    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError


class Korridor(Env):
    def __init__(self, lengde=5):
        super().__init__()
        self.lengde = lengde
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state, {}

    def step(self, action):
        # oppdater self.state og returner state, reward, terminated, info
        pass
```
```losning
class Env:
    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError


class Korridor(Env):
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
```
```test
env = Korridor()
state, _ = env.reset()
r = env.step(0)
assert r is not None, "step returnerer None. Returner state, reward, terminated, info."
assert r[0] == 0 and r[1] == -1 and r[2] is False, f"step(0) i rute 0 skal gi (0, -1, False, {{}}), men ga {r!r}. Staten kan ikke bli mindre enn 0."
for _ in range(3):
    s, rew, term, info = env.step(1)
assert s == 3 and rew == -1 and term is False, f"Etter tre steg til høyre skal state være 3, reward -1 og terminated False, men fikk {s}, {rew}, {term}."
s, rew, term, info = env.step(1)
assert s == 4 and rew == 10 and term is True, f"Fjerde steg til høyre skal nå målet: (4, 10, True), men fikk ({s}, {rew}, {term})."
assert env.step(1)[0] == 4, "state skal aldri bli større enn lengde - 1."
```
:::

## Hvorfor model(x) virker i PyTorch
`m(x)` kjører spesialmetoden `__call__`, og i PyTorch kaller `nn.Module.__call__` din `forward`.

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

print(SimpleNN(2, 4)(torch.tensor([[1.0, 2.0]])).shape)
```

::: oppgave Egen Module
Lag `Dobler(Module)` med `forward(self, x)` som returnerer `2 * x`, og lagre `Dobler()(5)` i `svar`.
```start
class Module:
    def __call__(self, x):
        return self.forward(x)


# lag Dobler her

svar = ...
```
```losning
class Module:
    def __call__(self, x):
        return self.forward(x)


class Dobler(Module):
    def forward(self, x):
        return 2 * x

svar = Dobler()(5)
```
```test
assert "Dobler" in dir() and issubclass(Dobler, Module), "Lag class Dobler(Module): med en forward-metode."
assert svar == 10, f"svar ble {svar!r}. Dobler()(5) kaller __call__, som kaller forward(5) = 10."
assert Dobler()(1.5) == 3.0, "forward skal returnere 2 * x."
```
:::

--- LEKSJON 1.16 | moduler-unntak | Moduler, import og unntak
`import math`, `import numpy as np` og `from math import pi` henter kode fra moduler og pakker, og egne filer importeres likt (`from linear_models import LinearRegression`).

::: oppgave Importer
Importer `math`, `numpy` som `np` og `pi` fra `math`, og lag `rot = math.sqrt(16)`, `e = math.exp(1)`, `snitt = np.mean([1, 2, 3, 4])` og `areal = pi * 2 ** 2`.
```start
# importer her

rot = ...
e = ...
snitt = ...
areal = ...
```
```losning
import math
import numpy as np
from math import pi

rot = math.sqrt(16)
e = math.exp(1)
snitt = np.mean([1, 2, 3, 4])
areal = pi * 2 ** 2
```
```test
assert isinstance(rot, float) and abs(rot - 4.0) < 1e-9, "rot skal være math.sqrt(16) = 4.0."
assert isinstance(e, float) and abs(e - 2.718281828) < 1e-6, "e skal være math.exp(1)."
assert isinstance(snitt, float) and abs(snitt - 2.5) < 1e-9, "snitt skal være np.mean([1, 2, 3, 4]) = 2.5."
assert isinstance(areal, float) and abs(areal - 12.566370614) < 1e-6, "areal skal være pi * 2 ** 2."
```
:::

## collections.Counter
`Counter(y)` teller elementene i én linje, og `.most_common(1)` gir `[(vanligste, antall)]`.

::: oppgave Tell med Counter
Importer `Counter` fra `collections`, og lag `antall = Counter(y)`, `n_katt` (antall `"katt"`) og `vanligste` (bare navnet på den vanligste klassen).
```start
y = ["katt", "hund", "katt", "fugl", "katt", "hund"]

antall = ...
n_katt = ...
vanligste = ...
```
```losning
from collections import Counter

y = ["katt", "hund", "katt", "fugl", "katt", "hund"]

antall = Counter(y)
n_katt = antall["katt"]
vanligste = antall.most_common(1)[0][0]
```
```test
from collections import Counter as _Counter
assert isinstance(antall, _Counter), "antall skal være Counter(y)."
assert n_katt == 3, "n_katt skal være antall[\"katt\"] = 3."
assert vanligste == "katt", f"vanligste ble {vanligste!r}. most_common(1) gir [('katt', 3)], så ta [0][0]."
```
:::

## Modulen random
`random.seed(42)` gir de samme tallene hver gang, og `random.random()`, `random.randint(0, 3)` (3 er med), `random.choice(liste)`, `random.shuffle(liste)` og `random.sample(buffer, k=32)` (forskjellige elementer) dekker det meste.

::: oppgave Trekk en minibatch
Sett seed til `42` med `random.seed`, og trekk **rett etterpå** 8 forskjellige transisjoner fra `buffer` med `random.sample`. Lagre dem i `batch`.
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

::: oppgave ε-greedy
Lag `velg_handling(q, epsilon)`: hvis `random.random() < epsilon`, returner en tilfeldig handling fra 0 til `len(q) - 1`, ellers indeksen til største verdi i `q`.
```start
import random

def velg_handling(q, epsilon):
    return None
```
```losning
import random

def velg_handling(q, epsilon):
    if random.random() < epsilon:
        return random.randint(0, len(q) - 1)
    return max(range(len(q)), key=lambda a: q[a])
```
```test
q = [0.1, 0.7, 0.3]
assert all(velg_handling(q, 0.0) == 1 for _ in range(20)), "Med epsilon = 0 skal agenten alltid utnytte og velge handling 1 (største q-verdi)."
valg = {velg_handling(q, 1.0) for _ in range(300)}
assert valg == {0, 1, 2}, f"Med epsilon = 1 skal alle handlingene 0, 1 og 2 velges av og til, men fikk bare {valg}. Bruk random.randint(0, len(q) - 1)."
```
:::

## Feiltyper og traceback
Les en traceback nedenfra: siste linje er feiltypen, som `NameError`, `TypeError`, `IndexError`, `KeyError`, `ValueError`, `AttributeError` eller `ZeroDivisionError`, og linjene over viser hvor den skjedde.

::: oppgave Hvilken feil?
Lagre navnet på feiltypen hver kodelinje i kommentaren gir, som tekst (for eksempel `f1 = "NameError"`).
```start
f1 = ...  # [1, 2].shape
f2 = ...  # int("3.5")
f3 = ...  # {"a": 1}["b"]
f4 = ...  # [1, 2, 3][5]
f5 = ...  # "Tap: " + 0.25
f6 = ...  # 1 / 0
```
```losning
f1 = "AttributeError"  # [1, 2].shape
f2 = "ValueError"  # int("3.5")
f3 = "KeyError"  # {"a": 1}["b"]
f4 = "IndexError"  # [1, 2, 3][5]
f5 = "TypeError"  # "Tap: " + 0.25
f6 = "ZeroDivisionError"  # 1 / 0
```
```test
assert f1 == "AttributeError", "f1: lister har ikke attributtet shape. Hvilken feiltype handler om attributter?"
assert f2 == "ValueError", "f2: typen (tekst) er grei for int(), men verdien \"3.5\" er ikke et gyldig heltall."
assert f3 == "KeyError", "f3: nøkkelen \"b\" finnes ikke i dicten."
assert f4 == "IndexError", "f4: indeks 5 er utenfor listen."
assert f5 == "TypeError", "f5: tekst og tall kan ikke skjøtes med +, typene passer ikke."
assert f6 == "ZeroDivisionError", "f6: deling på null."
```
:::

## try, except og raise
`try:`/`except ValueError as e:` fanger en bestemt feiltype (aldri et naket `except:`), og `raise ValueError("...")` kaster en feil selv, slik `raise NotImplementedError` markerer kode som ikke er skrevet ennå.

::: oppgave Sikker accuracy
Erstatt `raise NotImplementedError` i `accuracy` med kode som kaster `ValueError` ved ulik lengde og ellers returnerer accuracy. Fullfør `les_tall(tekst)` så den returnerer `float(tekst)`, eller `None` ved `ValueError`.
```start
def accuracy(y_true, y_pred):
    raise NotImplementedError("accuracy er ikke skrevet ennå")


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
try:
    r = accuracy([1, 0, 1, 1], [1, 0, 0, 1])
except NotImplementedError:
    raise AssertionError("accuracy kaster fortsatt NotImplementedError. Fjern raise-linjen og skriv funksjonen.")
assert abs(r - 0.75) < 1e-9, f"accuracy([1, 0, 1, 1], [1, 0, 0, 1]) ga {r!r}, men skal gi 0.75."
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
