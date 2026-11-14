=== DEL 8 | traer | Trær, ensembler og Naive Bayes (prosjekt 2)
Beslutningstrær i sklearn og for hånd (Gini, entropi, beste split), tolkning av treet, regresjonstrær, bagging, random forest, AdaBoost og gradient boosting, og Naive Bayes regnet ut med pandas. Til slutt hele prosjekt 2 i én leksjon.

--- LEKSJON 8.1 | beslutningstre | DecisionTreeClassifier
Et **beslutningstre** er en rekke ja/nei-spørsmål om én feature om gangen: «er `x0 <= 1.36`?», så «er `x1 <= 0.8`?», helt til vi havner i en **løvnode** som gir svaret. Du kan tenke på det som nøstede `if`-setninger som sklearn finner automatisk fra dataene.

```python
# Et "tre" skrevet for hånd med if-setninger
def lite_tre(x0, x1):
    if x1 <= 1.36:
        return 0
    else:
        if x0 <= 0.5:
            return 0
        return 1

print(lite_tre(2.0, 2.5), lite_tre(0.0, 2.5), lite_tre(3.0, -1.0))
```

I prosjekt 2 er dataene i `problem1.csv`. Kolonnen `split` sier om en rad hører til trening eller test. Start alltid med å se på klassefordelingen:

```python
import pandas as pd

df = pd.read_csv("problem1.csv")
print(df.head())
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]
print("Trening:", train["y"].value_counts().to_dict())
print("Test:   ", test["y"].value_counts().to_dict())
```

Treningsdataene er svært **ubalanserte** (930 mot 76), mens testdataene er balanserte (150 mot 150). Det får store konsekvenser. Et tre lages med `DecisionTreeClassifier`, og brukes med det vanlige `fit`/`predict`-mønsteret:

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("problem1.csv")
train, test = df[df["split"] == "train"], df[df["split"] == "test"]
X_train, y_train = train[["x0", "x1"]], train["y"]
X_test, y_test = test[["x0", "x1"]], test["y"]

clf = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=0)
clf.fit(X_train, y_train)
print("Train accuracy:", accuracy_score(y_train, clf.predict(X_train)))
print("Test accuracy: ", accuracy_score(y_test, clf.predict(X_test)))
```

Se `clf` i variabelpanelet: etter `fit` har modellen fått lærte attributter (de som slutter på `_`), for eksempel `classes_` (klassene den så), `feature_names_in_` og `tree_` (selve treet). Høy treningsnøyaktighet, men test-nøyaktigheten er nær 0.5. Treet har lært at «nesten alt er klasse 0», og det lønner seg på treningsdataene. Det er dette som er **preprosesseringen** prosjektet spør etter: vi må balansere klassene. Den enkleste måten er `class_weight="balanced"`, som gir hver rad i minoritetsklassen større vekt når Gini regnes ut:

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("problem1.csv")
train, test = df[df["split"] == "train"], df[df["split"] == "test"]
X_train, y_train = train[["x0", "x1"]], train["y"]
X_test, y_test = test[["x0", "x1"]], test["y"]

clf = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = accuracy_score(y_test, y_pred)
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
print(f"Accuracy: {accuracy:.2f}")
print(f"True positive rate: {tpr:.3f}")
print(f"False positive rate: {fpr:.3f}")
```

Et alternativ er å **oversample** minoritetsklassen med pandas: trekk rader fra klasse 1 med tilbakelegging til begge klassene er like store.

```python
import pandas as pd

df = pd.read_csv("problem1.csv")
train = df[df["split"] == "train"]
maj = train[train["y"] == 0]
mino = train[train["y"] == 1]
print("Før:")
display(train["y"].value_counts())
mino_opp = mino.sample(len(maj), replace=True, random_state=0)
balansert = pd.concat([maj, mino_opp])
print("Etter:")
balansert["y"].value_counts()
```

## Beslutningsflaten før og etter
Med `meshgrid` og `contourf` (leksjon 4.6) ser vi hva treet gjør. Legg merke til at alle grenser er **vertikale eller horisontale**: hver split ser bare på én feature.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("problem1.csv")
train = df[df["split"] == "train"]
X_train, y_train = train[["x0", "x1"]], train["y"]

xx0, xx1 = np.meshgrid(np.arange(-7, 6, 0.05), np.arange(-6, 6, 0.05))
grid = pd.DataFrame({"x0": xx0.ravel(), "x1": xx1.ravel()})

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, vekt in zip(axes, [None, "balanced"]):
    clf = DecisionTreeClassifier(max_depth=3, class_weight=vekt, random_state=0).fit(X_train, y_train)
    Z = clf.predict(grid).reshape(xx0.shape)
    ax.contourf(xx0, xx1, Z, alpha=0.4, cmap="coolwarm")
    ax.scatter(train["x0"], train["x1"], c=y_train, cmap="coolwarm", s=8)
    ax.set_title(f"class_weight={vekt}")
plt.show()
```

Uten balansering er det røde området (klasse 1) lite og stramt rundt treningspunktene. Med balansering blir det mye større, og fanger testpunktene fra klasse 1.

::: hvorfor Trenger trær skalering?
Nei. En split er bare en terskel på én feature, `x0 <= t`. Multipliserer du `x0` med 1000, flytter terskelen seg til `1000·t`, men nøyaktig de samme punktene havner på hver side. Gini og entropi avhenger bare av hvilke punkter som havner hvor, ikke av avstander. Sammenlign med gradient descent (del 5) og k-means (del 10), som bruker størrelsen på tallene direkte.
```python
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("problem1.csv")
X, y = df[["x0", "x1"]], df["y"]
X_skalert = X * [1000, 0.001]

a = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y).predict(X)
b = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_skalert, y).predict(X_skalert)
print("Like prediksjoner:", np.array_equal(a, b))
```
:::

::: prosjekt Prosjekt 2, problem 1
Kravet var minst 0.90 test-accuracy på `problem1.csv`. Spørsmålene var hvilken preprosessering som trengs (balansere klassene, fordi treningsdataene er skjeve og testdataene ikke er det), om features må skaleres (nei, se over) og hva du ser i beslutningsflaten før og etter (klasse 1-området vokser). Prosjektet ba deg også rapportere TPR og FPR.
:::

::: oppgave Tren et balansert tre med entropi
Tren et `DecisionTreeClassifier` på treningsdelen av `problem1.csv` med `criterion="entropy"`, `max_depth=2`, `class_weight="balanced"` og `random_state=0`. Lagre modellen i `clf` og test-accuracy i `acc`.
```start
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("problem1.csv")
train, test = df[df["split"] == "train"], df[df["split"] == "test"]

clf = ...
acc = ...
```
```losning
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("problem1.csv")
train, test = df[df["split"] == "train"], df[df["split"] == "test"]

clf = DecisionTreeClassifier(criterion="entropy", max_depth=2, class_weight="balanced", random_state=0)
clf.fit(train[["x0", "x1"]], train["y"])
acc = accuracy_score(test["y"], clf.predict(test[["x0", "x1"]]))
```
```test
from sklearn.tree import DecisionTreeClassifier
assert isinstance(clf, DecisionTreeClassifier), "clf skal være en DecisionTreeClassifier."
assert clf.criterion == "entropy", 'Bruk criterion="entropy".'
assert clf.get_depth() <= 2, "Bruk max_depth=2."
assert clf.class_weight == "balanced", 'Husk class_weight="balanced".'
assert isinstance(acc, float) and acc > 0.88, f"acc skal være test-accuracy (forventet over 0.88), fikk {acc}. Brukte du testdataene?"
```
:::

::: quiz
Et tre trenes på data der kolonnen `inntekt` er i kroner (0 til 2 000 000) og `alder` i år. Hva skjer om vi kjører `StandardScaler` på begge features før treningen?
- [ ] Treet blir mye bedre, fordi inntekt ellers dominerer
- [x] Prediksjonene blir de samme, bare tersklene i nodene endres
- [ ] Treet krasjer, fordi negative verdier ikke er lov
- [ ] Gini-verdiene i nodene endres
? En split er en terskel på én feature. Skalering flytter tersklene, men samme punkter havner på samme side, så Gini, treet og prediksjonene blir de samme.
:::

::: quiz
Treningsdataene har 930 punkter av klasse 0 og 76 av klasse 1. Testdataene er 50/50. Et tre får 0.95 accuracy på trening og 0.55 på test. Hva er det beste tiltaket?
- [ ] Skalere features med `StandardScaler`
- [ ] Øke `max_depth` slik at treet lærer mer
- [x] Bruke `class_weight="balanced"` eller oversample klasse 1
- [ ] Trene på testdataene i stedet
? Treet har lært at det nesten alltid lønner seg å svare 0. Balansering gjør at feil på klasse 1 teller like mye som feil på klasse 0.
:::

::: oppsummering
- `DecisionTreeClassifier(max_depth=…, criterion="gini"/"entropy"/"log_loss", random_state=…)`, så `fit` og `predict`.
- Se alltid på klassefordelingen først (`value_counts()`). Skjeve treningsdata gir et tre som favoriserer majoritetsklassen.
- `class_weight="balanced"` eller oversampling med `sample(replace=True)` balanserer klassene.
- Beslutningsflatene til et tre er akseparallelle (vertikale og horisontale).
- Trær trenger **ikke** skalering: en terskel på én feature påvirkes ikke av enheten.
:::

--- LEKSJON 8.2 | gini-entropi | Gini og entropi for hånd
Hvordan velger treet split? For hver mulig split regner det ut hvor **uren** hver side blir, og velger den splitten som gir renest resultat. To mål på urenhet:

$$G = 1 - \sum_k p_k^2 \qquad\qquad H = -\sum_k p_k \log_2 p_k$$

der $p_k$ er andelen av klasse $k$ i noden. Begge er 0 for en ren node (bare én klasse). For to klasser er Gini maks 0.5 og entropi maks 1, begge ved 50/50. I numpy finner vi andelene med `np.unique(y, return_counts=True)`:

```python
import numpy as np

def gini(y):
    _, antall = np.unique(y, return_counts=True)
    p = antall / len(y)
    return 1 - np.sum(p**2)

def entropi(y):
    _, antall = np.unique(y, return_counts=True)
    p = antall / len(y)
    return -np.sum(p * np.log2(p))

y = np.array([1]*9 + [0]*5)          # notatenes eksempel: p = 9, n = 5
print(f"Gini:    {gini(y):.3f}")
print(f"Entropi: {entropi(y):.3f}")    # notatene: 0.94
print(gini([1, 1, 1]), entropi(np.array([0, 1, 0, 1])))
```

Fordi `np.unique` bare returnerer klasser som faktisk finnes, får vi aldri `log2(0)`.

Notatene ber deg plotte Gini og entropi for andelen $p$ av klasse 1. Kurvene har samme form, og gir derfor nesten alltid samme split. Gini er litt raskere fordi den slipper logaritmen, og er standard i sklearn. `criterion="log_loss"` gir nøyaktig samme tre som `"entropy"` i sklearn.

```python
import numpy as np
import matplotlib.pyplot as plt

p = np.linspace(0.001, 0.999, 200)
g = 1 - p**2 - (1 - p)**2
h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
plt.plot(p, g, label="Gini")
plt.plot(p, h, label="Entropi")
plt.plot(p, h / 2, "--", label="Entropi / 2")
plt.xlabel("andel av klasse 1")
plt.legend()
plt.show()
```

## Urenhet etter en split og informasjonsgevinst
Etter en split har vi to barn. Urenheten etter splitten er det **vektede gjennomsnittet**, vektet med hvor mange punkter som havner i hvert barn. **Informasjonsgevinsten** er hvor mye urenheten faller:

$$\Delta I = I_{\text{før}} - \left(\frac{n_V}{n} I_V + \frac{n_H}{n} I_H\right)$$

```python
import numpy as np

def entropi(y):
    _, antall = np.unique(y, return_counts=True)
    p = antall / len(y)
    return -np.sum(p * np.log2(p))

def urenhet_etter(y_v, y_h, mal):
    n = len(y_v) + len(y_h)
    return len(y_v) / n * mal(y_v) + len(y_h) / n * mal(y_h)

# Notatenes oppgave: (p=9, n=5) splittes i (4, 5) og (5, 0)
y = np.array([1]*9 + [0]*5)
y_v = np.array([1]*4 + [0]*5)
y_h = np.array([1]*5)
etter = urenhet_etter(y_v, y_h, entropi)
print(f"Før: {entropi(y):.2f}, etter: {etter:.2f}, gevinst: {entropi(y) - etter:.2f}")
```

## Finn beste split for én feature
Treet prøver **alle terskler** for alle features. For én feature holder det å prøve midtpunktene mellom sorterte, unike verdier. Først på åtte punkter, der vi ser alle kandidatene i en tabell:

```python
import numpy as np
import pandas as pd

def gini(y):
    _, antall = np.unique(y, return_counts=True)
    p = antall / len(y)
    return 1 - np.sum(p**2)

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([0, 0, 0, 1, 0, 1, 1, 1])

rader = []
for t in (x[:-1] + x[1:]) / 2:                    # midtpunktene 1.5, 2.5, ...
    venstre, hoyre = y[x <= t], y[x > t]
    vektet = (len(venstre) * gini(venstre) + len(hoyre) * gini(hoyre)) / len(y)
    rader.append({"terskel": t, "n_v": len(venstre), "gini_v": gini(venstre),
                  "n_h": len(hoyre), "gini_h": gini(hoyre), "vektet gini": vektet})
display(pd.DataFrame(rader).round(3))
```

Den vektede Gini er lavest for terskel 3.5 og 5.5: hver av dem lager én helt ren side, og den andre siden har bare ett punkt av feil klasse. Nå samme søk på `x0` fra prosjekt 2, der vi bare beholder den beste terskelen, og sammenligner med sklearn:

```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def gini(y):
    _, antall = np.unique(y, return_counts=True)
    p = antall / len(y)
    return 1 - np.sum(p**2)

def beste_split(x, y):
    verdier = np.unique(x)
    terskler = (verdier[:-1] + verdier[1:]) / 2
    beste_t, beste_g = None, np.inf
    for t in terskler:
        venstre, hoyre = y[x <= t], y[x > t]
        g = (len(venstre) * gini(venstre) + len(hoyre) * gini(hoyre)) / len(y)
        if g < beste_g:
            beste_t, beste_g = t, g
    return beste_t, beste_g

df = pd.read_csv("problem1.csv")
x, y = df["x0"].to_numpy(), df["y"].to_numpy()
t, g = beste_split(x, y)
print(f"For hånd: x0 <= {t:.4f}, vektet Gini {g:.4f}")

stump = DecisionTreeClassifier(max_depth=1).fit(df[["x0"]], y)
print(f"sklearn:  x0 <= {stump.tree_.threshold[0]:.4f}")
```

Et helt tre gjør nøyaktig dette for alle features, velger den beste, deler dataene, og gjentar rekursivt i hvert barn til en stoppregel slår inn (ren node, `max_depth` nådd, for få punkter). Notatene sier du skal kunne pseudokoden, ikke implementere hele treet.

::: oppgave Skriv gini(y)
Skriv funksjonen `gini(y)` som tar en liste eller et array med klasselabels og returnerer Gini-urenheten $1 - \sum_k p_k^2$. Den skal virke for to eller flere klasser.
```start
import numpy as np

def gini(y):
    ...
```
```losning
import numpy as np

def gini(y):
    _, antall = np.unique(y, return_counts=True)
    p = antall / len(y)
    return 1 - np.sum(p**2)
```
```test
import numpy as np
assert gini([0, 0, 0, 0]) is not None, "gini må returnere et tall (husk return)."
assert abs(gini([0, 0, 0, 0]) - 0.0) < 1e-9, "En ren node skal ha Gini 0."
assert abs(gini([0, 1, 0, 1]) - 0.5) < 1e-9, "50/50 med to klasser skal gi 0.5."
assert abs(gini(np.array([1]*3 + [0])) - 0.375) < 1e-9, "[1, 1, 1, 0] skal gi 1 - (0.75² + 0.25²) = 0.375."
assert abs(gini([0, 1, 2]) - 2/3) < 1e-9, "Tre like store klasser skal gi 2/3."
```
:::

::: oppgave Informasjonsgevinst med Gini
Bruk `gini` (gitt) og regn ut **gevinsten** i Gini når noden `y` deles i `y_v` og `y_h`. Lagre svaret i `gevinst`.
```start
import numpy as np

def gini(y):
    _, antall = np.unique(y, return_counts=True)
    p = antall / len(y)
    return 1 - np.sum(p**2)

y = np.array([1]*9 + [0]*5)
y_v = np.array([1]*4 + [0]*5)
y_h = np.array([1]*5)

gevinst = ...
```
```losning
import numpy as np

def gini(y):
    _, antall = np.unique(y, return_counts=True)
    p = antall / len(y)
    return 1 - np.sum(p**2)

y = np.array([1]*9 + [0]*5)
y_v = np.array([1]*4 + [0]*5)
y_h = np.array([1]*5)

etter = len(y_v) / len(y) * gini(y_v) + len(y_h) / len(y) * gini(y_h)
gevinst = gini(y) - etter
```
```test
assert gevinst is not ... and gevinst is not None, "Regn ut gevinst."
assert abs(gevinst - 0.1417) < 1e-3, f"Forventet ca. 0.142. Husk å vekte barna med len(y_v)/len(y) og len(y_h)/len(y). Fikk {gevinst}."
```
:::

::: quiz
En node har `value = [30, 10]`. Hva er Gini-urenheten?
- [ ] 0.25
- [x] 0.375
- [ ] 0.5
- [ ] 0.811
? $p = (0.75, 0.25)$, så $G = 1 - 0.75^2 - 0.25^2 = 1 - 0.5625 - 0.0625 = 0.375$. 0.811 er entropien.
:::

::: quiz
Et tre vurderer tre splitter av samme node. Vektet Gini etter splitt er A: 0.42, B: 0.31, C: 0.38. Hvilken velges?
- [ ] A, fordi den har høyest urenhet igjen å fjerne
- [x] B, fordi den gir lavest urenhet etter splitten
- [ ] C, fordi den er i midten
- [ ] Det avhenger av hvilken feature som har flest unike verdier
? Urenheten før er den samme for alle tre. Lavest urenhet etter gir størst gevinst, så B velges.
:::

::: oppsummering
- Gini: `1 - np.sum(p**2)`. Entropi: `-np.sum(p * np.log2(p))`. `p` fra `np.unique(y, return_counts=True)`.
- Begge er 0 for rene noder. Maks er 0.5 (Gini) og 1 (entropi) ved 50/50 med to klasser.
- Urenhet etter split = vektet gjennomsnitt av barna. Gevinst = før minus etter.
- Beste split: prøv alle terskler (midtpunkter) for alle features, velg lavest vektet urenhet.
- Gini og entropi gir nesten alltid samme tre. Gini er standard og litt raskere. `"log_loss"` = `"entropy"` i sklearn.
:::

--- LEKSJON 8.3 | tolke-treet | Visualisere og tolke treet
Den store fordelen med trær er at de er **tolkbare**: du kan lese hvert spørsmål treet stiller. Vi bruker noen få kategoriske features fra vikingdataene i `problem2.csv` (gjort om til 0/1-kolonner med `get_dummies`, leksjon 3.8).

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv("problem2.csv")
X = pd.get_dummies(df[["sex", "role", "armor"]].fillna("None"), dtype=int)
y = df["survived"]

clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)

plt.figure(figsize=(12, 5))
plot_tree(clf, feature_names=list(X.columns), class_names=["died", "survived"], filled=True)
plt.show()
```

## Les en node
Hver boks viser fire eller fem linjer. Ta rotnoden:

| Linje | Betydning |
|---|---|
| `armor_None <= 0.5` | Spørsmålet. Sant (ja) går til **venstre**, usant til høyre. Her: har rustning → venstre |
| `gini = 0.494` | Gini-urenheten i noden |
| `samples = 500` | Antall treningspunkter som når noden |
| `value = [278, 222]` | Antall per klasse: 278 døde, 222 overlevde |
| `class = died` | Flertallsklassen, det noden ville predikert |

Vi kan sjekke Gini selv: $p = (278/500, 222/500) = (0.556, 0.444)$, så $G = 1 - 0.556^2 - 0.444^2 = 0.494$. Fargen viser klassen, og sterkere farge betyr renere node. Løvnoden med `value = [15, 79]` (har rustning og er ikke Karl) har Gini $1 - (15/94)^2 - (79/94)^2 = 0.268$, mye renere.

## Treet som tekst og som tall
`export_text` gir den samme strukturen som tekst, som er fint når figuren blir stor. Alle tallene ligger også i `clf.tree_`:

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

df = pd.read_csv("problem2.csv")
X = pd.get_dummies(df[["sex", "role", "armor"]].fillna("None"), dtype=int)
y = df["survived"]
clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)

print(export_text(clf, feature_names=list(X.columns)))
print("Dybde:", clf.get_depth(), " Løvnoder:", clf.get_n_leaves(), " Noder totalt:", clf.tree_.node_count)

t = clf.tree_
print("feature  :", t.feature)        # -2 betyr løvnode
print("threshold:", t.threshold)      # -2 for løvnoder
print("samples  :", t.n_node_samples)
print("gini     :", t.impurity.round(3))
```

Nodene er nummerert i rekkefølgen venstre-først (dybde-først). `t.feature[i]` er indeksen til kolonnen noden spør om, så `X.columns[t.feature[0]]` er spørsmålet i roten.

## Feature importance
`feature_importances_` sier hvor mye hver feature totalt reduserte urenheten (vektet med antall punkter), normalisert slik at summen er 1. Features som splittes tidlig og på mange punkter blir viktige.

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("problem2.csv")
X = pd.get_dummies(df[["sex", "role", "armor"]].fillna("None"), dtype=int)
y = df["survived"]
clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)

viktighet = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
viktighet
```

::: advarsel
`feature_importances_` sier hvor mye en feature ble *brukt*, ikke om den *forårsaker* noe. To korrelerte features kan dele viktigheten tilfeldig, og en feature med mange unike verdier kan få kunstig høy viktighet. Del 12 viser permutation importance og SHAP, som er mer pålitelige.
:::

::: prosjekt Prosjekt 2, problem 2
Prosjektet spurte: «What is Gini impurity? Use an example from the decision tree graph you created.» Et godt svar plukker én node fra `plot_tree`, leser `value`, og regner Gini fra den slik som over. Forklar også at 0 betyr ren node og at treet velger splitten som senker den vektede Gini mest.
:::

::: oppgave Les treet med kode
Treet `clf` er gitt. Finn tre ting med kode (ikke skriv tallene inn for hånd):
- `rot_feature`: **navnet** på featuren rotnoden splitter på (bruk `clf.tree_.feature[0]` og `X.columns`)
- `rot_gini`: Gini-urenheten i rotnoden
- `viktigst`: navnet på featuren med høyest `feature_importances_`
```start
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("problem2.csv")
X = pd.get_dummies(df[["sex", "role", "ship", "destination"]], dtype=int)
y = df["survived"]
clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)

rot_feature = ...
rot_gini = ...
viktigst = ...
```
```losning
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("problem2.csv")
X = pd.get_dummies(df[["sex", "role", "ship", "destination"]], dtype=int)
y = df["survived"]
clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)

rot_feature = X.columns[clf.tree_.feature[0]]
rot_gini = clf.tree_.impurity[0]
viktigst = X.columns[np.argmax(clf.feature_importances_)]
```
```test
import numpy as np
fasit_rot = X.columns[clf.tree_.feature[0]]
assert isinstance(rot_feature, str), "rot_feature skal være et kolonnenavn (tekst). Slå opp i X.columns."
assert rot_feature == fasit_rot, f"Rotnoden splitter ikke på {rot_feature}. Bruk clf.tree_.feature[0] som indeks i X.columns."
p = y.mean()
assert rot_gini is not ... and abs(rot_gini - (1 - p**2 - (1 - p)**2)) < 1e-6, "rot_gini skal være Gini i rotnoden (clf.tree_.impurity[0])."
assert viktigst == X.columns[np.argmax(clf.feature_importances_)], "viktigst skal være kolonnen med høyest feature_importances_."
```
:::

::: quiz
En løvnode i `plot_tree` viser `gini = 0.0`, `samples = 40`, `value = [0, 40]`, `class = survived`. Hva stemmer?
- [ ] Noden er maksimalt uren
- [x] Alle 40 treningspunkter i noden overlevde, og nye punkter som havner her predikeres som «survived»
- [ ] 40 % av punktene i noden overlevde
- [ ] Noden kan splittes videre for å bli renere
? `value` er antall per klasse. `[0, 40]` betyr at alle er i klasse 1, så noden er ren (Gini 0) og kan ikke bli renere.
:::

::: quiz
Hva skriver koden ut?
```kode
clf = DecisionTreeClassifier(max_depth=3).fit(X, y)
t = clf.tree_
print(t.feature[t.children_left == -1])
```
- [ ] Indeksene til featurene som brukes i rotnoden
- [x] En rekke med `-2`, én for hver løvnode
- [ ] Tersklene i løvnodene
- [ ] En tom liste, fordi løvnoder ikke har features
? Løvnoder har ingen barn (`children_left == -1`), og sklearn markerer dem med `feature = -2` og `threshold = -2`.
:::

::: oppsummering
- `plot_tree(clf, feature_names=…, class_names=…, filled=True)` tegner treet. `export_text(clf, feature_names=…)` gir det som tekst.
- En node viser spørsmål (sant går til venstre), `gini`, `samples`, `value` (antall per klasse) og `class` (flertall).
- Regn Gini fra `value`: andeler $p_k$, så $1 - \sum p_k^2$.
- `get_depth()`, `get_n_leaves()`, `tree_.feature`, `tree_.threshold`, `tree_.impurity`, `tree_.n_node_samples`.
- `feature_importances_` summerer til 1 og viser hvor mye hver feature reduserte urenheten.
:::

--- LEKSJON 8.4 | dybde-regresjonstre | Dybde, overtilpasning og regresjonstrær
Et tre uten begrensning (`max_depth=None`) splitter til alle løvnoder er rene. Da får det 100 % på treningsdataene, men har lært støy. **Dybden** er den viktigste hyperparameteren. Vi plotter train- og test-accuracy mot `max_depth`:

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("problem1.csv")
train, test = df[df["split"] == "train"], df[df["split"] == "test"]
X_train, y_train = train[["x0", "x1"]], train["y"]
X_test, y_test = test[["x0", "x1"]], test["y"]

dybder = range(1, 16)
tr_acc, te_acc = [], []
for d in dybder:
    clf = DecisionTreeClassifier(max_depth=d, class_weight="balanced", random_state=0).fit(X_train, y_train)
    tr_acc.append(clf.score(X_train, y_train))
    te_acc.append(clf.score(X_test, y_test))

plt.plot(dybder, tr_acc, "o-", label="train")
plt.plot(dybder, te_acc, "o-", label="test")
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.legend()
plt.show()
```

Train-kurven stiger mot 1, mens test-kurven topper seg tidlig og så faller. Grunt tre = **høy bias** (undertilpasning), dypt tre = **høy varians** (overtilpasning). Andre måter å bremse treet på er `min_samples_leaf` (minst så mange punkter per løv) og `min_samples_split`. Velg dybden med kryssvalidering på treningsdata (leksjon 7.5 og 7.6), ikke ved å se på testdataene.

## Regresjonstrær
Et `DecisionTreeRegressor` deler rommet på samme måte, men hver løvnode predikerer **gjennomsnittet** av target-verdiene der. Splittkriteriet er MSE i stedet for Gini. Resultatet er en **trappefunksjon**. Notatene bruker data fra $f(x) = \lfloor x \rfloor$ med litt støy:

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(0)

def f_true(x):
    return np.floor(x)

X_data = np.linspace(0, 4, 100)
y_data = f_true(X_data) + np.random.normal(0, 0.1, size=100)
X_data = X_data.reshape(-1, 1)          # sklearn vil ha 2D X
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=0)

regressor = DecisionTreeRegressor(random_state=0).fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("Tree depth:", regressor.get_depth())
print("Number of leaves:", regressor.get_n_leaves())
print("Model MSE:", mean_squared_error(y_test, y_pred))
```

Uten `max_depth` lager sklearn et dypt tre med nesten én løvnode per treningspunkt. Det følger støyen. Prøv en dybde som passer til trappen: fire trinn trenger tre splitter, som får plass i et tre med dybde 2.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

np.random.seed(0)
X_data = np.linspace(0, 4, 100)
y_data = np.floor(X_data) + np.random.normal(0, 0.1, size=100)
X_train, X_test, y_train, y_test = train_test_split(X_data.reshape(-1, 1), y_data, random_state=0)

regressor = DecisionTreeRegressor(max_depth=2).fit(X_train, y_train)
t = regressor.tree_
print("Terskler:", np.sort(t.threshold[t.feature >= 0]).round(2))

grid = np.linspace(0, 4, 500).reshape(-1, 1)
plt.scatter(X_train.ravel(), y_train, s=15, label="train")
plt.step(grid.ravel(), regressor.predict(grid), where="post", color="green", label="tree")
plt.plot(grid.ravel(), np.floor(grid).ravel(), "k--", label="true")
for thr in t.threshold[t.feature >= 0]:
    plt.axvline(thr, color="gray", ls=":")
plt.legend()
plt.show()
```

`t.threshold[t.feature >= 0]` henter tersklene bare fra de indre nodene (løvnoder har `feature = -2`). `plt.step` tegner prediksjonen som en trapp. Tersklene havner nær 1, 2 og 3, akkurat der den sanne funksjonen hopper.

::: hvorfor Trær eller regresjon?
Notatene (tabell 8) oppsummerer forskjellene: trær trenger ikke skalering, kan overtilpasse om dybden ikke begrenses, og strukturen bestemmes under treningen. Lineær/logistisk regresjon har en fast likningsform der bare parametrene tilpasses, og features bør ha samme størrelsesorden. Begge er tolkbare.
:::

::: oppgave Finn beste dybde for regresjonstreet
Prøv `max_depth` fra 1 til 8 (`range(1, 9)`) for `DecisionTreeRegressor(max_depth=d, random_state=0)`, regn ut test-MSE for hver, og lagre dybden med lavest test-MSE i `beste_dybde` (et heltall). Lagre også listen med MSE-ene i `mse_liste`.
```start
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(0)
X_data = np.linspace(0, 4, 100)
y_data = np.floor(X_data) + np.random.normal(0, 0.1, size=100)
X_train, X_test, y_train, y_test = train_test_split(X_data.reshape(-1, 1), y_data, random_state=0)

mse_liste = []
# løkke her

beste_dybde = ...
```
```losning
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(0)
X_data = np.linspace(0, 4, 100)
y_data = np.floor(X_data) + np.random.normal(0, 0.1, size=100)
X_train, X_test, y_train, y_test = train_test_split(X_data.reshape(-1, 1), y_data, random_state=0)

mse_liste = []
for d in range(1, 9):
    reg = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X_train, y_train)
    mse_liste.append(mean_squared_error(y_test, reg.predict(X_test)))

beste_dybde = int(np.argmin(mse_liste)) + 1
```
```test
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
fasit = [mean_squared_error(y_test, DecisionTreeRegressor(max_depth=d, random_state=0).fit(X_train, y_train).predict(X_test)) for d in range(1, 9)]
assert len(mse_liste) == 8, f"mse_liste skal ha 8 verdier (dybde 1 til 8), har {len(mse_liste)}."
assert np.allclose(mse_liste, fasit), "MSE-ene stemmer ikke. Regn test-MSE med mean_squared_error(y_test, reg.predict(X_test))."
assert beste_dybde == int(np.argmin(fasit)) + 1, "beste_dybde skal være dybden (ikke indeksen) med lavest MSE. Husk at range starter på 1."
```
:::

::: quiz
Hva predikerer et `DecisionTreeRegressor` for et nytt punkt?
- [ ] En lineær funksjon av featurene i løvnoden
- [x] Gjennomsnittet av target-verdiene til treningspunktene i løvnoden punktet havner i
- [ ] Medianen av alle target-verdier i treningsdataene
- [ ] Flertallsklassen i løvnoden
? Hver løvnode har én konstant verdi: snittet av treningspunktene der. Derfor blir prediksjonen en trappefunksjon.
:::

::: quiz
Et tre med `max_depth=None` får 1.00 accuracy på trening og 0.70 på test. Et tre med `max_depth=3` får 0.88 og 0.86. Hva er riktig?
- [ ] Det dype treet har høy bias
- [x] Det dype treet overtilpasser (høy varians), og dybde 3 generaliserer bedre
- [ ] Det grunne treet overtilpasser fordi test er nær train
- [ ] Begge er like gode fordi snittet er likt
? Stort gap mellom train og test er tegn på overtilpasning. Det grunne treet har litt høyere bias, men mye lavere varians.
:::

::: oppsummering
- Dypere tre: høyere train-accuracy, men etter et punkt lavere test-accuracy (overtilpasning).
- Begrens treet med `max_depth`, `min_samples_leaf` eller `min_samples_split`, og velg med kryssvalidering.
- `DecisionTreeRegressor` predikerer snittet i løvnoden og splitter for å minimere MSE.
- `get_depth()`, `get_n_leaves()`, og tersklene i indre noder: `t.threshold[t.feature >= 0]`.
- Tegn prediksjonen med `plt.step(grid, reg.predict(grid), where="post")`.
:::

--- LEKSJON 8.5 | bagging-random-forest | Bagging og random forest
Ett dypt tre har høy varians: små endringer i dataene gir et helt annet tre. **Bagging** (bootstrap + aggregering) trener mange trær på ulike **bootstrap-utvalg** og lar dem stemme. Et bootstrap-utvalg trekkes **med tilbakelegging**, så samme rad kan komme flere ganger, og omtrent en tredjedel av radene kommer ikke med i det hele tatt.

```python
import numpy as np

rng = np.random.default_rng(0)
n = 1000
idx = rng.choice(n, size=n, replace=True)      # bootstrap: n trekk med tilbakelegging
print("Første indekser:", idx[:10])
print("Andel unike rader:", len(np.unique(idx)) / n)   # ca. 1 - 1/e = 0.632

lite = rng.choice(10, size=10, replace=True)   # bootstrap av bare 10 rader
print("Trukne indekser, sortert:")
display(np.sort(lite))
print("Antall ganger hver rad 0-9 ble trukket:")
display(np.bincount(lite, minlength=10))
```

I det lille utvalget ser du duplikatene direkte: noen indekser står flere ganger etter hverandre. `np.bincount` teller hvor mange ganger hver rad ble trukket. Radene med 0 er **out-of-bag** for dette treet.

## Bagging for hånd
Vi trener 25 trær, hvert på sitt bootstrap-utvalg, og tar flertallsstemme. Sammenlign med ett enkelt tre:

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_moons(n_samples=400, noise=0.3, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

ett_tre = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
print("Ett tre:", ett_tre.score(X_test, y_test))

rng = np.random.default_rng(0)
stemmer = []
for b in range(25):
    idx = rng.choice(len(X_train), size=len(X_train), replace=True)
    tre = DecisionTreeClassifier(random_state=b).fit(X_train[idx], y_train[idx])
    stemmer.append(tre.predict(X_test))

stemmer = np.array(stemmer)                       # shape (25, antall testpunkter)
display(stemmer[:, :20])                          # 25 trær x de 20 første testpunktene
y_pred = (stemmer.mean(axis=0) > 0.5).astype(int) # flertallsstemme
print("Bagging (25 trær):", np.mean(y_pred == y_test))
```

I rutenettet er hver rad ett tre og hver kolonne ett testpunkt. Kolonner med én farge er punkter alle trærne er enige om. Kolonner med blandede farger er de vanskelige punktene, og der avgjør flertallet.

For regresjon tar vi gjennomsnittet i stedet for flertallsstemme. Gjennomsnitt av mange modeller med uavhengige feil har lavere varians enn hver enkelt modell.

## BaggingClassifier og RandomForestClassifier
sklearn gjør det samme med `BaggingClassifier`. **Random forest** går ett steg videre: i hver split får treet bare se et tilfeldig utvalg av features (`max_features`). Da blir trærne mer ulike (mindre korrelerte), og gjennomsnittet reduserer variansen mer.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

tre = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, random_state=0).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100, max_features="sqrt", oob_score=True, random_state=0)
rf.fit(X_train, y_train)

print(f"Ett tre:       {tre.score(X_test, y_test):.3f}")
print(f"Bagging:       {bag.score(X_test, y_test):.3f}")
print(f"Random forest: {rf.score(X_test, y_test):.3f}")
print(f"OOB-score:     {rf.oob_score_:.3f}")
print("Antall trær:", len(rf.estimators_))
```

Begge ensemblene slår ett tre klart. Forskjellen mellom bagging og random forest er liten her, og på et testsett med 143 rader er den innenfor støyen.

- `n_estimators`: antall trær. Flere trær gir sjelden overtilpasning, bare lengre kjøretid.
- `max_features`: hvor mange features hver split får velge mellom. `"sqrt"` ($\sqrt{30} \approx 5$ av 30) er standard for klassifisering.
- `oob_score=True`: hvert tre tester seg selv på radene som **ikke** kom med i bootstrap-utvalget (out-of-bag). Det gir et gratis valideringsestimat uten egen valideringsmengde.

::: eksamen
Eksamen 2024 spurte hva som er viktig for en god random forest: trærne må være **ulike og uavhengige**, og det oppnås med bootstrap-utvalg av radene og tilfeldige utvalg av features. Eksamen 2025 spurte om de to kildene til tilfeldighet i random forest (bootstrap og tilfeldig feature-utvalg i hver split).
:::

::: oppgave Random forest med OOB
Tren en `RandomForestClassifier` med 50 trær, `max_features=5`, `oob_score=True` og `random_state=0` på `X_train, y_train`. Lagre modellen i `rf`, OOB-scoren i `oob` og test-accuracy i `test_acc`.
```start
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

rf = ...
oob = ...
test_acc = ...
```
```losning
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

rf = RandomForestClassifier(n_estimators=50, max_features=5, oob_score=True, random_state=0)
rf.fit(X_train, y_train)
oob = rf.oob_score_
test_acc = rf.score(X_test, y_test)
```
```test
from sklearn.ensemble import RandomForestClassifier
assert isinstance(rf, RandomForestClassifier), "rf skal være en RandomForestClassifier."
assert rf.n_estimators == 50 and rf.max_features == 5, "Bruk n_estimators=50 og max_features=5."
assert hasattr(rf, "oob_score_"), "Husk oob_score=True og å kalle fit."
assert abs(oob - rf.oob_score_) < 1e-9, "oob skal være rf.oob_score_."
assert abs(test_acc - rf.score(X_test, y_test)) < 1e-9, "test_acc skal være accuracy på testdataene."
```
:::

::: quiz
Hvorfor bruker random forest `max_features` mindre enn antall features?
- [ ] For at hvert tre skal trene raskere, uten effekt på nøyaktigheten
- [x] For å gjøre trærne mindre like hverandre, slik at gjennomsnittet reduserer variansen mer
- [ ] For å unngå at noen features får vekt null
- [ ] Fordi trær ikke kan bruke flere enn $\sqrt{d}$ features
? Hvis én feature er veldig sterk, ville alle trær splittet på den først og blitt like. Tilfeldige feature-utvalg gir mer uavhengige trær.
:::

::: quiz
Hva gjør `rng.choice(n, size=n, replace=True)` i bagging?
- [ ] Stokker radene, slik at hver rad brukes nøyaktig én gang
- [x] Trekker et bootstrap-utvalg der noen rader kommer flere ganger og andre ikke kommer med
- [ ] Deler dataene i n like store folder
- [ ] Velger n tilfeldige features
? Med tilbakelegging (`replace=True`) kan samme indeks trekkes flere ganger. I snitt kommer omtrent 63 % av radene med.
:::

::: oppsummering
- Bootstrap: trekk n rader **med** tilbakelegging. Ca. 63 % unike rader, resten er out-of-bag.
- Bagging: tren mange modeller på bootstrap-utvalg, aggreger med flertallsstemme (klassifisering) eller snitt (regresjon). Reduserer varians.
- `BaggingClassifier(DecisionTreeClassifier(), n_estimators=…)`.
- `RandomForestClassifier(n_estimators, max_features, oob_score=True)`: bagging + tilfeldig feature-utvalg per split.
- `rf.oob_score_` er et gratis valideringsestimat.
:::

--- LEKSJON 8.6 | boosting | Boosting: AdaBoost og gradient boosting
Bagging trener trærne **parallelt** og uavhengig. **Boosting** trener dem **sekvensielt**: hver ny modell prøver å rette feilene til de forrige. Modellene er ofte svake, typisk **stumper** (trær med dybde 1).

- **AdaBoost**: punkter som ble feilklassifisert får høyere vekt i neste runde. Hver modell får en vekt $\alpha$ etter hvor god den var, og prediksjonen er en vektet stemme.
- **Gradient boosting**: hvert nytt tre trenes på **residualene** (feilene) til ensemblet så langt.

## AdaBoostClassifier
Standard svak modell i `AdaBoostClassifier` er en stump. `learning_rate` krymper bidraget fra hver modell.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

stump = DecisionTreeClassifier(max_depth=1).fit(X_train, y_train)
print(f"Én stump: {stump.score(X_test, y_test):.3f}")

ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=0)
cv = cross_val_score(ada, X_train, y_train, cv=5)
ada.fit(X_train, y_train)
print(f"AdaBoost CV Accuracy: {cv.mean():.3f} ± {cv.std():.3f}")
print(f"AdaBoost test accuracy: {ada.score(X_test, y_test):.3f}")
print("Vekter (alpha) for de fem første stumpene:", ada.estimator_weights_[:5].round(2))
```

Én stump er svak, men 50 stumper som retter hverandres feil blir sterke. Vil du ha dypere trær i ensemblet, gir du dem som første argument: `AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=…)`.

## Gradient boosting for hånd
Notatene beskriver gradient boosting slik: hvert tre predikerer forskjellen mellom target og ensemblets prediksjon, de såkalte **pseudo-residualene**. For MSE-tap er de bare $r = y - F(x)$. Vi starter med en konstant (snittet), og legger til ett lite tre om gangen, krympet med en læringsrate:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

rng = np.random.default_rng(0)
X = np.sort(rng.uniform(0, 6, 100)).reshape(-1, 1)
y = np.sin(X).ravel() + rng.normal(0, 0.2, 100)

lr = 0.1
F = np.full(len(y), y.mean())        # start: konstant prediksjon
residualer = {"x": X.ravel(), "r før tre 1": y - F}
for m in range(50):
    r = y - F                         # pseudo-residualer
    tre = DecisionTreeRegressor(max_depth=2).fit(X, r)
    F = F + lr * tre.predict(X)       # nytt ensemble = forrige + lr * nytt tre
    if m in (0, 4, 49):
        print(f"runde {m + 1:2d}: MSE = {np.mean((y - F)**2):.4f}")
        residualer[f"r etter tre {m + 1}"] = y - F
display(pd.DataFrame(residualer).iloc[::12].round(2))   # hvert 12. punkt

gbr = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=2).fit(X, y)
print("Største avvik fra sklearn:", np.abs(F - gbr.predict(X)).max())

plt.scatter(X, y, s=10, label="data")
plt.plot(X, F, "g", label="gradient boosting (for hånd)")
plt.legend()
plt.show()
```

Tabellen følger noen punkter gjennom treningen. Før første tre er residualene store, rundt ±1 der sinuskurven er høyest og lavest. Hvert tre tar en liten bit av dem (læringsraten 0.1), og etter 50 trær er det bare støyen igjen. Vår løkke gir samme svar som `GradientBoostingRegressor`. Notatene skriver regelen som `new_ensemble = previous_ensemble - learning_rate * new_tree`, der treet tilpasses **gradienten** av tapet, $-(y - F)$. Det er det samme som å legge til et tre tilpasset $+(y - F)$. Vi gjør altså gradient descent, men i rommet av trær i stedet for i parameterrommet.

## GradientBoostingClassifier og -Regressor
I prosjekt 4 brukes `GradientBoostingRegressor(max_depth=5, random_state=0)` som modellen som forklares med SHAP. Viktige hyperparametre er `n_estimators`, `learning_rate` og `max_depth`. Lav læringsrate trenger flere trær.

```python
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=0)
gbr.fit(X_train, y_train)
print(f"Regresjon R2 (test): {r2_score(y_test, gbr.predict(X_test)):.3f}")

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
gbc = GradientBoostingClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)
print(f"Klassifisering accuracy (test): {gbc.score(X_test, y_test):.3f}")
```

::: tips
Kjente, raske biblioteker for gradient boosting er XGBoost, LightGBM og CatBoost (XGBoost brukes i notatenes SHAP-eksempel). De har samme `fit`/`predict`-grensesnitt.
:::

::: oppgave Fullfør gradient boosting-løkka
Fullfør løkka slik at den kjører 30 runder gradient boosting med læringsrate 0.3 og trær med `max_depth=1`. Etterpå skal `F` inneholde ensemblets prediksjoner på `X`.
```start
import numpy as np
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(1)
X = rng.uniform(-3, 3, size=(80, 1))
y = X.ravel()**2 + rng.normal(0, 0.3, 80)

lr = 0.3
F = np.full(len(y), y.mean())
for m in range(30):
    r = ...                 # pseudo-residualer
    tre = ...               # tren et tre med max_depth=1 på r
    F = ...                 # oppdater ensemblet
```
```losning
import numpy as np
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(1)
X = rng.uniform(-3, 3, size=(80, 1))
y = X.ravel()**2 + rng.normal(0, 0.3, 80)

lr = 0.3
F = np.full(len(y), y.mean())
for m in range(30):
    r = y - F
    tre = DecisionTreeRegressor(max_depth=1).fit(X, r)
    F = F + lr * tre.predict(X)
```
```test
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
assert isinstance(F, np.ndarray) and F.shape == y.shape, "F skal være et array med én prediksjon per punkt."
fasit = GradientBoostingRegressor(n_estimators=30, learning_rate=0.3, max_depth=1).fit(X, y).predict(X)
assert np.mean((y - F)**2) < np.var(y) * 0.5, "MSE har ikke falt nok. Er r = y - F, og legger du til lr * tre.predict(X)?"
assert np.allclose(F, fasit, atol=1e-6), "F avviker fra sklearn. Sjekk at treet trenes på r (ikke y) og har max_depth=1."
```
:::

::: quiz
Hvilket utsagn om gradient boosting (regresjon med MSE) er riktig?
- [ ] Alle trærne trenes uavhengig på bootstrap-utvalg
- [ ] Punktene som ble feilpredikert får høyere vekt i neste runde
- [x] Hvert nytt tre trenes på residualene $y - F(x)$ til ensemblet så langt
- [ ] Det siste treet erstatter alle de forrige
? Det første alternativet er bagging, det andre er AdaBoost. Gradient boosting tilpasser hvert tre til pseudo-residualene og legger det til ensemblet.
:::

::: quiz
Hva er hovedforskjellen mellom bagging og boosting?
- [ ] Bagging bruker bare stumper, boosting bruker dype trær
- [x] Bagging trener modellene parallelt og uavhengig, boosting trener dem sekvensielt der hver retter de forriges feil
- [ ] Bagging er for regresjon, boosting for klassifisering
- [ ] Boosting bruker bootstrap, bagging gjør ikke
? Bagging reduserer først og fremst varians (typisk dype trær), boosting reduserer først og fremst bias (typisk svake modeller).
:::

::: oppsummering
- Boosting: svake modeller trenes **sekvensielt**, hver retter feilene til de forrige.
- `AdaBoostClassifier(n_estimators, learning_rate)`: stumper som standard, øker vekten på feilklassifiserte punkter. `estimator_weights_` er $\alpha$-ene.
- Gradient boosting: start med snittet, tren hvert tre på pseudo-residualene `y - F`, og legg til `lr * tre.predict(X)`.
- `GradientBoostingRegressor/Classifier(n_estimators, learning_rate, max_depth)`. Lav læringsrate trenger flere trær.
- CV-score rapporteres som `f"{cv.mean():.3f} ± {cv.std():.3f}"`.
:::

--- LEKSJON 8.7 | naive-bayes | Naive Bayes for hånd
Bayes' teorem gir sannsynligheten for en klasse $C$ gitt observasjonen $x$:

$$P(C \mid x) = \frac{P(x \mid C)\, P(C)}{P(x)}$$

I prosjekt 2 skal du bruke Naive Bayes til å avgjøre om vikingen **Endre** overlever, *uten å importere biblioteker* (pandas for å lese filen er greit). La oss se hvilke ledd vi kan estimere fra `problem2.csv`.

## Prior: P(survived)
Prioren er bare klassefordelingen:

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
prior = df["survived"].value_counts(normalize=True)
print(prior)
print("P(survived=1) =", prior[1])
```

## Likelihood: P(feature = verdi | survived)
For hver klasse ser vi på radene i den klassen og teller hvor stor andel som har verdien. `pd.crosstab` med `normalize="columns"` gir alle på en gang:

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
overlevde = df[df["survived"] == 1]
print("P(sex=Male | survived=1) =", (overlevde["sex"] == "Male").mean())

# Hele tabellen: hver kolonne summerer til 1
pd.crosstab(df["role"], df["survived"], normalize="columns")
```

## Hvilke ledd er vanskelige?
- $P(C)$: lett, bare telling.
- $P(x_i \mid C)$ for **én** feature: lett, telling innenfor klassen.
- $P(x \mid C)$ for **alle features samtidig**: vanskelig. Det finnes tusenvis av kombinasjoner av kjønn, rolle, skip, destinasjon … og de fleste finnes ikke i 500 rader. Naive Bayes antar derfor at featurene er **uavhengige** gitt klassen, slik at $P(x \mid C) \approx \prod_i P(x_i \mid C)$.
- $P(x)$ (evidensen): like vanskelig, av samme grunn. Men den er **lik for begge klassene**. Når vi bare skal *sammenligne* klassene, kan vi droppe den: $P(C \mid x) \propto P(C) \prod_i P(x_i \mid C)$.

## Regn ut for Endre
Vi bruker de kategoriske featurene. Manglende rustning (`NaN` i `armor`) betyr ingen rustning, så vi gjør den om til `"None"`.

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
df["armor"] = df["armor"].fillna("None")

endre = {"sex": "Male", "role": "Hirdman", "ship": "Drakvinge",
         "destination": "Normandy", "clan_origin": "Nidaros", "armor": "Chainmail"}

score = {}
faktorer = {}                                     # for tabellen under
for klasse in [0, 1]:
    rader = df[df["survived"] == klasse]
    s = len(rader) / len(df)                      # prior P(C)
    faktorer[f"survived={klasse}"] = {"P(C)": s}
    for feature, verdi in endre.items():
        s *= (rader[feature] == verdi).mean()    # P(x_i | C)
        faktorer[f"survived={klasse}"][f"P({feature}={verdi} | C)"] = (rader[feature] == verdi).mean()
    score[klasse] = s

display(pd.DataFrame(faktorer).round(3))
print(f"Ikke normalisert: overlever {score[1]:.2e}, dør {score[0]:.2e}")
print(f"P(overlever | Endre) ≈ {score[1] / (score[0] + score[1]):.3f}")
```

Tabellen viser alle faktorene i produktet, én rad per ledd og én kolonne per klasse. Scoren for en klasse er produktet nedover kolonnen. Tallene er små fordi vi ganger mange sannsynligheter, men det er **forholdet** som teller. Endre er mann (trekker ned), men er hirdmann med ringbrynje (`Chainmail`, trekker kraftig opp). Siste linje normaliserer de to scorene slik at de summerer til 1. Det er et estimat, ikke en kalibrert sannsynlighet, fordi uavhengighetsantakelsen ikke holder helt.

## Problemet med null: Laplace-glatting
Endre har `prior_voyages = 12`, men ingen i datasettet har mer enn 9. Da blir $P(12 \mid C) = 0$ for begge klasser, og hele produktet blir 0. Løsningen er **Laplace-glatting**: legg til $\alpha$ (typisk 1) i hver telling.

$$P(x_i = v \mid C) = \frac{\text{antall}(v, C) + \alpha}{n_C + \alpha K}$$

der $K$ er antall mulige verdier av featuren.

```python
import pandas as pd

df = pd.read_csv("problem2.csv")
print("Maks prior_voyages i dataene:", df["prior_voyages"].max())

def likelihood(df, feature, verdi, klasse, alpha=1):
    rader = df[df["survived"] == klasse]
    K = df[feature].nunique() + 1           # +1 fordi verdien kan være ny
    return ((rader[feature] == verdi).sum() + alpha) / (len(rader) + alpha * K)

for klasse in [0, 1]:
    uten = (df[df["survived"] == klasse]["prior_voyages"] == 12).mean()
    med = likelihood(df, "prior_voyages", 12, klasse)
    print(f"klasse {klasse}: uten glatting {uten}, med glatting {med:.4f}")
```

For kontinuerlige features som `age` og `provisions_kg` er det enda verre: nesten hver verdi er unik. Del dem i intervaller (`pd.cut`), eller anta en normalfordeling per klasse (det er Gaussian Naive Bayes).

::: tips Logaritmer mot underflow
Med mange features blir produktet så lite at det kan bli 0 i flyttall. Summer heller logaritmene: $\log P(C) + \sum_i \log P(x_i \mid C)$. Klassen med størst sum vinner.
:::

## Med sklearn
`CategoricalNB` gjør det samme som vi gjorde for hånd, med Laplace-glatting `alpha=1`. Den krever at kategoriene er heltall, så vi bruker `OrdinalEncoder` (leksjon 7.3). `GaussianNB` er for kontinuerlige features.

```python
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB, GaussianNB

df = pd.read_csv("problem2.csv")
df["armor"] = df["armor"].fillna("None")
df["clan_origin"] = df["clan_origin"].fillna("Unknown")
kat = ["sex", "role", "ship", "destination", "clan_origin", "armor"]

enc = OrdinalEncoder()
X = enc.fit_transform(df[kat])
nb = CategoricalNB(alpha=1).fit(X, df["survived"])

endre = pd.DataFrame([["Male", "Hirdman", "Drakvinge", "Normandy", "Nidaros", "Chainmail"]], columns=kat)
print("CategoricalNB:", nb.predict_proba(enc.transform(endre)).round(3))

num = ["age", "provisions_kg", "family_aboard", "prior_voyages"]
gnb = GaussianNB().fit(df[num].fillna(df["age"].median()), df["survived"])
endre_num = pd.DataFrame([[25, 49, 2, 12]], columns=num)
print("GaussianNB:   ", gnb.predict_proba(endre_num).round(3))
```

::: prosjekt Prosjekt 2, problem 2
Prosjektet spurte: hvilke ledd i Bayes' teorem kan du estimere, og hvilke er vanskelige? Svar: prior og likelihood per feature kan telles. Den felles likelihooden og evidensen $P(x)$ er vanskelige fordi de krever en fordeling over alle kombinasjoner av features, og de fleste kombinasjonene er aldri observert. $P(x)$ kan droppes fordi den er lik for begge klasser. Naive Bayes-delen skulle løses uten å importere biblioteker, altså med telling som over.
:::

::: oppgave Naive Bayes-score for en kvinnelig karl
Regn ut den ikke-normaliserte Naive Bayes-scoren $P(C)\prod_i P(x_i \mid C)$ for begge klasser for en ny viking med `sex = "Female"`, `role = "Karl"` og `ship = "Sjøulven"`. Lagre dem i `score0` (survived=0) og `score1` (survived=1), og lagre `True` i `overlever` hvis `score1 > score0`, ellers `False`. Ingen glatting.
```start
import pandas as pd

df = pd.read_csv("problem2.csv")
ny = {"sex": "Female", "role": "Karl", "ship": "Sjøulven"}

score0 = ...
score1 = ...
overlever = ...
```
```losning
import pandas as pd

df = pd.read_csv("problem2.csv")
ny = {"sex": "Female", "role": "Karl", "ship": "Sjøulven"}

def nb_score(klasse):
    rader = df[df["survived"] == klasse]
    s = len(rader) / len(df)
    for f, v in ny.items():
        s *= (rader[f] == v).mean()
    return s

score0 = nb_score(0)
score1 = nb_score(1)
overlever = score1 > score0
```
```test
def _fasit(k):
    r = df[df["survived"] == k]
    s = len(r) / len(df)
    for f, v in {"sex": "Female", "role": "Karl", "ship": "Sjøulven"}.items():
        s *= (r[f] == v).mean()
    return s
assert score0 is not ... and score1 is not ..., "Regn ut score0 og score1."
assert abs(score0 - _fasit(0)) < 1e-9, "score0 stemmer ikke. Husk prior len(rader)/len(df) ganget med andelen per feature blant radene med survived=0."
assert abs(score1 - _fasit(1)) < 1e-9, "score1 stemmer ikke. Bruk radene med survived=1."
assert bool(overlever) == (_fasit(1) > _fasit(0)), "overlever skal være score1 > score0."
```
:::

::: quiz
Hvorfor kan vi droppe $P(x)$ når vi bruker Naive Bayes til å velge klasse?
- [ ] Fordi $P(x)$ alltid er 1
- [ ] Fordi features er uavhengige
- [x] Fordi $P(x)$ er lik for alle klassene, så den endrer ikke hvilken klasse som får høyest score
- [ ] Fordi $P(x)$ er lik prioren
? Vi sammenligner $P(x \mid C)P(C)/P(x)$ for ulike $C$. Nevneren er den samme, så den påvirker ikke rekkefølgen.
:::

::: quiz
Hva returnerer koden for en klasse der ingen rader har `prior_voyages == 12`?
```kode
rader = df[df["survived"] == 1]
p = (rader["prior_voyages"] == 12).mean()
score = prior * p_sex * p_role * p
```
- [ ] En liten positiv score
- [x] `score` blir 0, uansett hvor gode de andre featurene er
- [ ] `ZeroDivisionError`
- [ ] `NaN`
? `.mean()` av bare `False` er 0.0, og ett nulledd gjør hele produktet 0. Det er derfor vi bruker Laplace-glatting.
:::

::: oppsummering
- Prior: `df["survived"].value_counts(normalize=True)`.
- Likelihood per feature: `(rader[feature] == verdi).mean()` innenfor hver klasse, eller `pd.crosstab(…, normalize="columns")`.
- Naive Bayes antar uavhengige features: score $= P(C)\prod_i P(x_i \mid C)$. $P(x)$ droppes fordi den er lik for alle klasser.
- Felles likelihood og $P(x)$ er vanskelige: for mange kombinasjoner, for lite data.
- Laplace-glatting $(\text{antall} + \alpha)/(n_C + \alpha K)$ hindrer at én ukjent verdi gir score 0.
- sklearn: `CategoricalNB` (kategorier som heltall), `GaussianNB` (kontinuerlige).
:::

--- LEKSJON 8.8 | prosjekt-2 | Hele prosjekt 2-løpet
Nå setter vi alt sammen på vikingdataene i `problem2.csv`, i samme rekkefølge som prosjektet: analyse, preprosessering, tre med kryssvalidering, splittkriterier, plott av treet og boosting. Målet er minst **0.80** test-accuracy.

## 1. Analyse
```python
import pandas as pd

df = pd.read_csv("problem2.csv")
print(df.shape)
print(df.isna().sum()[df.isna().sum() > 0])
print("Andel overlevde:", df["survived"].mean().round(3))

rate_female = df[df["sex"] == "Female"]["survived"].mean()
rate_male = df[df["sex"] == "Male"]["survived"].mean()
print(f"Female survival rate: {rate_female:.1%}")
print(f"Male survival rate: {rate_male:.1%}")

df.groupby("role")["survived"].agg(["mean", "count"]).sort_values("mean")
```

En tilfeldig gjetning (alltid flertallsklassen) gir $1 - 0.444 = 0.556$. Kjønn alene gir mer: gjett «overlever» for kvinner og «dør» for menn. `role` og `armor` er enda sterkere. Prøv `groupby` på andre kolonner selv.

## 2. Preprosessering
- `passenger_id` og `name` er unike per rad og sier ingenting om overlevelse. Fjern dem.
- `armor` mangler for 324 rader. Det betyr mest sannsynlig *ingen rustning*, så det er en egen kategori.
- `clan_origin` mangler for noen få. Lag kategorien `"Unknown"`.
- `age` mangler for 80. Fyll med medianen fra **treningsdataene** (ikke hele datasettet, for å unngå lekkasje).
- Kategoriske kolonner gjøres om med `pd.get_dummies`. Skalering trengs ikke for trær.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("problem2.csv")
df = df.drop(columns=["passenger_id", "name"])
df["armor"] = df["armor"].fillna("None")
df["clan_origin"] = df["clan_origin"].fillna("Unknown")

target = "survived"
X = pd.get_dummies(df.drop(columns=target), dtype=int)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

median_alder = X_train["age"].median()
X_train = X_train.fillna({"age": median_alder})
X_test = X_test.fillna({"age": median_alder})
print(X_train.shape, "NaN igjen:", X_train.isna().sum().sum())
X_train.head()
```

## 3. Tre, kryssvalidering og kriterier
Kryssvalidering på treningsdataene gir et mer stabilt estimat enn én valideringssplit, og lar oss velge hyperparametre uten å kikke på testdataene. Testdataene brukes én gang til slutt.

```python
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("problem2.csv").drop(columns=["passenger_id", "name"])
df["armor"] = df["armor"].fillna("None")
df["clan_origin"] = df["clan_origin"].fillna("Unknown")
X = pd.get_dummies(df.drop(columns="survived"), dtype=int)
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
med = X_train["age"].median()
X_train, X_test = X_train.fillna({"age": med}), X_test.fillna({"age": med})

beste = (0, None, None)
for criterion in ["gini", "entropy", "log_loss"]:
    for depth in [2, 3, 4, 6]:
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=0)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
        print(f"{criterion:8s} depth={depth}: CV {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        if cv_scores.mean() > beste[0]:
            beste = (cv_scores.mean(), criterion, depth)

print(f"Beste etter CV: {beste[1]}, depth={beste[2]} (CV {beste[0]:.3f})")
clf = DecisionTreeClassifier(criterion=beste[1], max_depth=beste[2], random_state=0).fit(X_train, y_train)
a = accuracy_score(y_test, clf.predict(X_test))
print(f"Test accuracy: {a:.4f}")
```

`entropy` og `log_loss` gir identiske tall, og Gini er nesten likt. Notatene foretrekker Gini fordi den er raskere og gir praktisk talt samme tre.

Standardavvikene i CV er store, og testsettet har bare 100 rader (ett punkt er 1 prosentpoeng). Ett enkelt tre er derfor ustabilt: prøv å endre `max_depth` i siste del, og se at test-accuracy hopper fra rundt 0.75 til over 0.80. Det dybde-valget som er best i CV, er ikke nødvendigvis best på test. Det er nettopp slik ustabilitet ensembler løser.

## 4. Plott treet
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv("problem2.csv").drop(columns=["passenger_id", "name"])
df["armor"] = df["armor"].fillna("None")
df["clan_origin"] = df["clan_origin"].fillna("Unknown")
X = pd.get_dummies(df.drop(columns="survived"), dtype=int)
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
X_train = X_train.fillna({"age": X_train["age"].median()})

clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)
plt.figure(figsize=(16, 6))
plot_tree(clf, feature_names=list(X_train.columns), class_names=["died", "survived"], filled=True, fontsize=8)
plt.show()
```

Velg en node og forklar Gini fra `value`, som i leksjon 8.3.

## 5. Boosting
```python
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("problem2.csv").drop(columns=["passenger_id", "name"])
df["armor"] = df["armor"].fillna("None")
df["clan_origin"] = df["clan_origin"].fillna("Unknown")
X = pd.get_dummies(df.drop(columns="survived"), dtype=int)
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
med = X_train["age"].median()
X_train, X_test = X_train.fillna({"age": med}), X_test.fillna({"age": med})

ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=0)
ada_cv_scores = cross_val_score(ada, X_train, y_train, cv=5)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
ada_acc = accuracy_score(y_test, y_pred_ada)
print(f"AdaBoost CV Accuracy: {ada_cv_scores.mean():.3f} ± {ada_cv_scores.std():.3f}")
print(f"AdaBoost test accuracy: {ada_acc:.4f}")
```

Boosting gir både høyere CV-score og høyere test-accuracy enn ett tre. Til slutt: Naive Bayes-delen med Endre gjorde vi i leksjon 8.7.

::: advarsel
Ikke velg `random_state` for `train_test_split` eller hyperparametre etter hva som gir best **test**-accuracy. Da har du i praksis trent på testdataene, og tallet blir for optimistisk. Velg med kryssvalidering og rapporter test én gang.
:::

::: oppgave Preprosesseringsfunksjon
Skriv funksjonen `forbered(df)` som tar den rå DataFramen og returnerer `(X, y)`, der `y` er kolonnen `survived` og `X` er klar for et tre: uten `passenger_id`, `name` og `survived`, uten manglende verdier (bruk `"None"` for `armor`, `"Unknown"` for `clan_origin`, medianen for `age`) og med bare tall (`pd.get_dummies(..., dtype=int)`). Funksjonen skal ikke endre den originale `df`.
```start
import pandas as pd

def forbered(df):
    ...
    return X, y

df = pd.read_csv("problem2.csv")
X, y = forbered(df)
```
```losning
import pandas as pd

def forbered(df):
    d = df.drop(columns=["passenger_id", "name"])
    d["armor"] = d["armor"].fillna("None")
    d["clan_origin"] = d["clan_origin"].fillna("Unknown")
    d["age"] = d["age"].fillna(d["age"].median())
    y = d["survived"]
    X = pd.get_dummies(d.drop(columns="survived"), dtype=int)
    return X, y

df = pd.read_csv("problem2.csv")
X, y = forbered(df)
```
```test
import numpy as np
import pandas as pd
assert isinstance(X, pd.DataFrame), "X skal være en DataFrame (fra pd.get_dummies)."
assert len(X) == 500 and len(y) == 500, "Ingen rader skal fjernes."
for kol in ["passenger_id", "name", "survived"]:
    assert kol not in X.columns, f"Kolonnen {kol} skal ikke være med i X."
assert X.isna().sum().sum() == 0, "X har fortsatt manglende verdier."
assert all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes), "Alle kolonner i X skal være tall. Bruk pd.get_dummies."
assert "armor_None" in X.columns, 'Manglende armor skal bli kategorien "None" (kolonnen armor_None).'
assert df["armor"].isna().sum() == 324, "forbered skal ikke endre den originale df. Jobb på en kopi."
assert (y == df["survived"]).all(), "y skal være kolonnen survived."
```
:::

::: quiz
Hva er formålet med `cross_val_score(clf, X_train, y_train, cv=5)` i prosjekt 2?
- [ ] Å trene den endelige modellen på fem ganger så mye data
- [x] Å få et mer stabilt estimat av hvor godt modellen generaliserer, kun ved hjelp av treningsdataene
- [ ] Å erstatte testsettet, slik at vi ikke trenger det
- [ ] Å finne den beste `random_state`
? Fem modeller trenes på fire femdeler og valideres på den siste. Snitt og standardavvik gir et estimat med usikkerhet, uten å bruke testdataene.
:::

::: quiz
Hvorfor fjernes `passenger_id` og `name` før treningen?
- [ ] Fordi trær ikke kan håndtere tekst og heltall samtidig
- [x] Fordi de er unike per rad og ikke bærer informasjon om overlevelse, slik at treet bare kan bruke dem til å pugge
- [ ] Fordi de har manglende verdier
- [ ] Fordi `get_dummies` krasjer på dem
? En ID kan gi splitter som tilfeldigvis passer treningsdataene, men som ikke generaliserer. `name` ville i tillegg gitt 500 dummy-kolonner.
:::

::: oppsummering
- Analyse: `isna().sum()`, overlevelsesrate per gruppe med `groupby(...)["survived"].mean()`. Sammenlign med å alltid gjette flertallsklassen.
- Preprosessering: fjern ID-kolonner, gjør manglende kategori til egen kategori, fyll tall med median fra treningsdata, `get_dummies`. Ingen skalering for trær.
- Velg dybde og kriterium med `cross_val_score` på treningsdata. Rapporter `f"{cv.mean():.3f} ± {cv.std():.3f}"` og test-accuracy én gang.
- `AdaBoostClassifier` løfter accuracy over ett enkelt tre.
- Lite testsett gir usikre tall: ett punkt av 100 er 1 prosentpoeng.
:::
