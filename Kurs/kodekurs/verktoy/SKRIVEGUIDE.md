# Skriveguide for kodekurset

Kurset er bygget for å lære ved å kode: **én ny ting om gangen, maks 1–2 setninger forklaring, så
oppgaver du løser selv.** Se «Stil: nesten bare koding» nederst. Målgruppen er en student som kanskje aldri har
programmert, og som skal ende opp med å kunne all koden i TDT4172 (prosjekt 1–4 og kodeeksemplene i
forelesningsnotatene), og svare på kodelesingsspørsmål på eksamen (flervalg, uten hjelpemidler).

## Filer

- `kilde/NN-navn.md` – én fil per del. Rediger bare disse.
- `python3 verktoy/build.py` – bygger `assets/kurs.js` og `assets/data.js`.
- `<venv>/bin/python verktoy/valider.py [filnavn-del]` – kjører ALL koden og sjekker oppgavene.
  Bruk venv-en i scratchpad (har numpy, pandas, scikit-learn, matplotlib, scipy, torch, shap, gymnasium).
  `--rask` hopper over statiske blokker, `--vis` skriver ut utdataene fra hver blokk.

## Format

```
=== DEL 2 | numpy | NumPy: regning med tabeller
Ingress (markdown) som vises på del-siden og forsiden. 1–3 setninger.

--- LEKSJON 2.1 | hva-er-numpy | Hva er NumPy?
Markdown-tekst …
```

- `=== DEL <nr> | <id> | <tittel>` – én gang øverst i filen.
- `--- LEKSJON <nr> | <id> | <tittel>` – nummer som `2.1`, `2.10`. Nummeret er også URL-en (`#2.1`).
- Leksjonsnummer må være unike i hele kurset.

### Tekst (mini-markdown)
Avsnitt skilles med blank linje. Støttet: `**fet**`, `*kursiv*`, `` `kode` ``, `[lenke](url)`,
lister (`- ` og `1. `), tabeller (`| a | b |` med `|---|---|`-linje), overskrifter `## Underoverskrift`
(blir h3) og `### Mindre` (h4), matte med `$...$` (inline) og `$$...$$` (egen linje, KaTeX).
Enkel HTML er lov. Ikke bruk `#` (h1) – leksjonstittelen er h1.

### Kjørbart eksempel (kjøres i nettleseren med Pyodide)
````
```python
x = 5
print(x * 2)
```
````
- Kjøres i et **eget, tomt navnerom** – hvert eksempel må være selvstendig (gjenta imports og data).
- Siste linje vises automatisk hvis den er et uttrykk (som i Jupyter): `df.head()` på siste linje viser tabellen.
- Tilgjengelig: standardbiblioteket, `numpy`, `pandas`, `sklearn`, `matplotlib`, `scipy`.
  **IKKE** torch, shap, gymnasium, keras, tensorflow, lime (bruk statisk blokk).
- Datasettene fra prosjektene ligger i arbeidsmappen og kan leses direkte:
  `regression.csv`, `classification.csv` (prosjekt 1), `problem1.csv`, `problem2.csv` (prosjekt 2),
  `classification_data.csv`, `regression_data.csv` (prosjekt 3).
- `plt.show()` viser figuren. Figurer som ikke vises eksplisitt vises også automatisk til slutt.
- Hold det raskt (under ~1–2 s). Ingen `input()`. Bruk faste seeds når noe er tilfeldig.
- Ikke skriv forventet utdata for kjørbare blokker – studenten kjører dem selv.
- ` ```python feil ` = eksempelet skal krasje med vilje (for å vise en feilmelding). Validatoren sjekker
  at det faktisk krasjer. Skriv ` ```python feil=TypeError ` for å kreve en bestemt feiltype.

### Visualisering: variabelpanelet og display()
Etter hver kjøring vises et **variabelpanel** under resultatet med alle variablene koden laget.
DataFrames vises som tabeller med manglende verdier markert i rødt, arrays som rutenett med fargeskala
(store 2D-arrays som heatmap), lister som bokser, dicts som tabeller, sklearn-modeller med det de lærte
(`coef_`, `classes_` …) og egne objekter med attributtene sine. Den siste tabellen/arrayen åpnes automatisk.

`display(x)` midt i koden viser `x` på samme måte **på det stedet** i utdataene, som et øyeblikksbilde.
Det er den samme funksjonen som finnes i Jupyter, så koden kan kopieres dit uendret. Bruk det til
**før/etter**-visninger der det er pedagogisk nyttig:
```python
df = pd.read_csv("problem2.csv")[["name", "age", "armor"]].head(8)
print("Før dropna:")
display(df)
df = df.dropna()
print("Etter dropna:")
display(df)
```
Gode steder: dropna/fillna, filtrering, nye kolonner, get_dummies/encoding, skalering (før/etter),
reshape/transponering, broadcasting, boolsk maske, train_test_split-former, Q-tabellen før/etter trening,
PCA-transformerte data, vekter før/etter fit. Ikke bruk display() i statiske blokker. Bruk heller små
utsnitt (`.head(8)`, få kolonner) så visningen blir oversiktlig. Når variabelpanelet viser poenget godt
nok alene, trengs ikke display(). Henvis gjerne til panelet i teksten («se `df` i variabelpanelet under resultatet»).

### Statisk kode (PyTorch, SHAP, gymnasium, Keras … kjøres lokalt i Jupyter)
````
```python statisk
import torch
x = torch.tensor([1.0, 2.0])
print(x * 2)
```
```output
tensor([2., 4.])
```
````
- `output` er valgfri, men ta den med når utdataene er lærerike. Utdataene skal være **ekte**: kjør
  koden i venv-en og lim inn (forkort lange utdata med `...`).
- `valider.py` kjører statiske blokker lokalt for å sjekke at de ikke krasjer.
- `python statisk fortsett` = blokken bygger videre på de forrige statiske blokkene i samme leksjon
  (validatoren setter dem sammen før kjøring). En statisk blokk uten `fortsett` starter en ny kjede.
- `python statisk ikkekjor` = pseudokode / kode som krever ting vi ikke har (nedlasting av MNIST,
  keras/tensorflow, lange treninger). Blir ikke kjørt av validatoren. Bruk sparsomt.

### Merknadsbokser
````
::: tips
Tekst. Kan også inneholde ```python-blokker.
:::
````
Typer: `tips`, `husk`, `advarsel` (vanlig feil), `eksamen` (kobling til eksamen), `prosjekt` (kobling til
prosjekt 1–4), `hvorfor` (hvorfor gjør vi dette). Valgfri egen tittel: `::: prosjekt Prosjekt 2, problem 1`.

### Oppgave (studenten skriver kode, testen sjekker)
````
::: oppgave Kort tittel
Oppgavetekst i markdown. Si nøyaktig hvilke variabelnavn/funksjoner som skal lages.
```start
# Lag variabelen alder her
alder = ...
```
```losning
alder = 21
```
```test
assert "alder" in dir() or True
assert alder == 21, "alder skal være 21"
```
:::
````
- `test` kjøres i **samme navnerom** etter studentens kode. Bruk `assert betingelse, "vennlig hint på norsk"`.
  Meldingen vises til studenten, så gjør den hjelpsom (hva er galt, ikke bare «feil»).
- Testen må **feile** på startkoden og **bestå** på løsningen (validatoren sjekker begge).
  Startkode med `...` (Ellipsis) eller `None` som plassholder er fint.
- Test resultatet, ikke en bestemt skrivemåte: godta alle riktige løsninger. Sammenlign flyttall med
  `abs(a - b) < 1e-6` eller `np.isclose`. Sjekk typer/shapes med tydelige meldinger.
- Figurer: testen kjøres før figurene vises og lukkes, så `plt.gcf()`/`plt.gca()` og `fig.axes` virker i testen.
  Det tryggeste er likevel å be studenten lagre `fig, ax = plt.subplots()` og teste på `ax.lines`, `ax.collections` osv.
- I nettleseren (wasm32) er numpys standard heltallstype `int32`, lokalt `int64`. Ikke test på eller skriv tekst som avhenger av det.
- Vil du sjekke utskrift? Unngå det – la heller studenten lagre svaret i en variabel.
- Oppgavene skal være små og gjennomførbare (2–10 linjer), og bruke det leksjonen nettopp lærte.

### Quiz (flervalg i eksamensstil)
````
::: quiz
Hva skriver koden ut?
```kode
x = [1, 2, 3]
print(x[-1])
```
- [ ] 1
- [x] 3
- [ ] `IndexError`
- [ ] -1
? Negativ indeks teller bakfra: `x[-1]` er siste element.
:::
````
- Nøyaktig ett `[x]`. 3–4 alternativer. Forklaringslinjer starter med `?`.
- Lag gode distraktorer: typiske misforståelser (fit på testdata, glemt zero_grad, feil akse …).
- Eksamensstilen: «Hva inneholder y_pred etter koden?», «Hvilken kodelinje …», «Hva skrives ut?».

### Oppsummering (avslutter hver leksjon)
````
::: oppsummering
- `print()` skriver ut
- …
:::
````

## Stil: nesten bare koding
Studenten skal lære ved å **skrive kode selv**, ikke ved å lese. Forklaringer og eksempler holdes til et minimum.

1. **Maks 1–2 setninger forklaring per del.** En «del» er leksjonens intro og hver `## Underoverskrift`.
   Setningene viser syntaksen direkte med inline-kode, f.eks. «`np.mean(X, axis=0)` gir gjennomsnittet per kolonne.»
   Ingen tekstvegger, ingen forklarende tabeller, ingen lange lister.
2. **Så oppgaver.** 1–4 oppgaver per del, typisk 3–6 per leksjon, fra lett til litt vanskeligere.
   Oppgaveteksten er 1–2 setninger og sier nøyaktig hvilke variabelnavn/funksjoner som skal lages.
   Startkoden har imports og data ferdig, så studenten bare skriver det nye. Det de gamle eksemplene
   viste, skal nå øves i oppgaver: alt pensum (funksjoner, parametere, fallgruver) skal fortsatt dekkes.
3. **Kjørbare eksempler: helst ingen.** Unntak: maks ett veldig kort eksempel (≤ 4 linjer) per leksjon når
   syntaksen ikke lar seg vise i en setning. `python feil`-blokker brukes ikke.
4. **Statiske blokker** (torch, shap, gymnasium, keras) er eneste måten å vise den syntaksen på, så de er lov,
   men korte (helst ≤ 10 linjer), med `output` bare når den er nødvendig. Øv selve ideen i en numpy-/sklearn-oppgave
   når det går (f.eks. skriv treningsløkka eller forward-passet i numpy).
5. **Ikke bruk** `::: quiz` (unntatt i del 13), `::: oppsummering`, eller merknadsboksene
   `tips`/`husk`/`advarsel`/`hvorfor`/`eksamen`/`prosjekt`. En viktig fallgruve (f.eks. `fit` på testdata)
   nevnes i forklaringssetningen eller testes i en oppgave.
6. **Del 13 (eksamenstrening)** beholder quizene. Forklaringslinjene (`?`) er maks 1 setning hver, og teksten
   mellom quizene er maks 1–2 setninger per del.
7. Del-ingressen (linjen etter `=== DEL`) er 1 setning.
8. Leksjonsnummer, id og titler beholdes (fremdriften er lagret per leksjonsnummer).
9. Rekkefølgen er pensumrekkefølgen: Python → NumPy → pandas → matplotlib → ML fra bunnen → evaluering →
   sklearn → trær/ensembler/Bayes → nevrale nett → uveiledet → RL → XAI → eksamenstrening.
   Bruk bare det som er lært før.
10. Norsk bokmål med korrekte æ, ø, å. Kode-identifikatorer og fagtermer (fit, predict, feature,
   epoch, batch …) kan stå på engelsk. Variabelnavn kan speile prosjektkoden (`X_train`, `y_pred`, `q_table`).
11. Alt du skriver om biblioteker skal stemme med versjonene i venv-en. Sjekk ved å kjøre.
