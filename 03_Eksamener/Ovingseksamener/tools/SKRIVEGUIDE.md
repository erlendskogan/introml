# Skriveguide for øvingseksamener i TDT4172 (H26)

Mål: 10 øvingssett som ligner ordinær eksamen 2025 (`03_Eksamener/2025/Eksamen_2025.pdf`) så mye som mulig:
10 «Parts» à 4 flervalgsspørsmål med 4 alternativer og nøyaktig ett riktig svar, på norsk bokmål,
med engelske fagbegreper der eksamen bruker dem (loss, recall, policy, embedding osv.).

## Kildemateriale (pensum)
- Forelesningsnotater H26: `01_Forelesninger/nyesteForelesninger.pdf` (Inga Strümke, 4. nov 2026). Hent tekst med
  `pdftotext 01_Forelesninger/nyesteForelesninger.pdf - | less`. **Notatene er fasit på definisjoner og notasjon.**
- Tidligere eksamener med LF: `03_Eksamener/2024/`, `03_Eksamener/2025/` (stil, nivå og typiske tema – ikke kopier spørsmål).
- Prosjektene (øvingene): `02_Prosjekter/Prosjekt_1..4` (notebooks: lin/log-regresjon, trær/Naive Bayes/boosting,
  nevrale nett + uveiledet, RL med Q-tabell + SHAP).
- Studentens egen kursnettside med formelark: `Kurs/kapitler/*.html` (særlig `17-formelark.html`).

## Fil-format
Skriv `03_Eksamener/Ovingseksamener/src/eksamen_NN.txt` (NN = 01..10). Se `src/EKSEMPEL_format.txt`.

```
@@PART
@@Q
<stamme i LaTeX, gjerne flere linjer og avsnitt>
@+ riktig alternativ (ÉN linje)
@- feil alternativ (ÉN linje)
@- feil alternativ
@- feil alternativ
```
- Nøyaktig 10 `@@PART`, hver med nøyaktig 4 `@@Q`, hver med 1 `@+` og 3 `@-`.
- Linjer som begynner med `%%%` er forfatterkommentarer (vises ikke). Bruk dem gjerne til å notere utregning.
- **Rekkefølgen stokkes automatisk** (balansert slik at A/B/C/D er riktig 10 ganger hver). Derfor:
  aldri «Både A og B», «Ingen av alternativene over», «Alle de over», eller henvisning til andre alternativer.
- `@INSTR <tekst>` overstyrer «Velg ett alternativ:» (sjelden nødvendig).
- Bygg og sjekk: `python3 03_Eksamener/Ovingseksamener/tools/build.py build 03_Eksamener/Ovingseksamener/src/eksamen_NN.txt`
  (skriver statistikk/flagg, lager `build/Ovingseksamen_NN.pdf`). Se på sider med
  `pdftoppm -r 70 -png -f 3 -l 6 build/Ovingseksamen_NN.pdf <scratchdir>/p` + Read på bildene.
- Emneplan per sett: `python3 03_Eksamener/Ovingseksamener/tools/emneplan.py NN`.

## LaTeX-regler
- Brødtekst: escape `_ % & #` (skriv `X\_train`, `5\,\%`). Kode inline: `\texttt{scaler.transform(X\_test)}`.
- Kodeblokker i stammen: `\begin{lstlisting}[language=Python] ... \end{lstlisting}` (rå kode, ingen escaping inni).
  Ikke kode-blokker i alternativene (de må være én linje) – bruk `\texttt{}` der.
- Tabeller: `tabular` (gjerne med `\toprule/\midrule/\bottomrule`), sentrert med `\begin{center}`.
- Figurer: TikZ/pgfplots er tilgjengelig (f.eks. ROC-kurve med nummererte punkter, punktsky for DBSCAN,
  stilisert beeswarm-plott, tre-diagram). Hold dem enkle og kontroller at de kompilerer og ser riktige ut.
- Matematikk: `$...$`, `\[ ... \]`. Uthev nøkkelbegreper i stammen med `\textbf{}` slik 2025-eksamen gjør.
- Kursiv for feature-navn: `\emph{Daily\_usage}`.

## Stil (som eksamen 2025)
- Kort kontekst / scenario, så ett tydelig spørsmål («Hva er den mest sannsynlige årsaken?», «Hvilket utsagn er
  korrekt?», «Hvilken verdi får ...?»).
- Alternativene er fullstendige utsagn i samme grammatiske form (eller tall/kode med samme format).
- Nivå: som 2025, men gjerne litt mer variert. Ca. 30–40 % av spørsmålene i hvert sett bør være
  regne-/scenariooppgaver (enkel kalkulator holder). Resten konseptuelle.
- Ikke pugg-trivia (årstall, personnavn). Test forståelse.

## Korrekthet (viktigst)
- Nøyaktig ett alternativ skal være riktig etter notatenes definisjoner; de tre andre skal være entydig gale
  for en som kan stoffet. Unngå tema der notatene og vanlig litteratur er uenige, eller formuler deg slik at begge
  gir samme svar.
- Angi konvensjoner eksplisitt i stammen når det betyr noe: log-base (entropi med $\log_2$), om MSE har faktor
  $\tfrac12$, om DBSCANs nabotall inkluderer punktet selv, at ε-greedy kan trekke den grådige handlingen også i
  tilfeldig modus, terskel `>` vs `>=` osv.
- Notasjon fra notatene: Q-læring $Q_{new}(S_t,A_t) \leftarrow (1-\alpha)Q(S_t,A_t) + \alpha\,(R_{t+1} + \gamma\max_A Q(S_{t+1},A))$;
  Gini $1-\sum p_i^2$; Shapley med vekter $\frac{|S|!(|N|-|S|-1)!}{|N|!}$; SHAP $f(x)=\phi_0+\sum_j\phi_j$;
  perceptron $w_i \leftarrow w_i + \eta(y-\hat y)x_i$; k-means++ $\propto d^2$; silhouette $(b-a)/\max(a,b)$.
- **Verifiser alle utregninger med Python** (`python3 -I -c '...'`) og noter gjerne utregningen i en `%%%`-linje.
- Ingen to spørsmål i samme sett skal teste nøyaktig det samme. Ikke kopier spørsmål fra 2024/2025 – lag nye
  scenarier/tall, selv om temaet er det samme.

## Gjør det umulig å «game» prøven (svært viktig)
En student som ikke kan stoffet skal ikke kunne finne riktig svar ved å se etter overflate-hint.
1. **Lengde:** Alle fire alternativene skal ha omtrent samme lengde og detaljnivå (innenfor ca. ±25 % tegn).
   Riktig svar skal *ikke* systematisk være det lengste, mest nyanserte eller mest «faglige». I minst en tredel av
   spørsmålene skal en distraktor være det lengste/mest detaljerte alternativet. `build.py` rapporterer dette –
   mål: riktig er lengst i ≤ 30 % av spørsmålene, og snittforholdet riktig/feil mellom 0.9 og 1.1.
2. **Troverdige distraktorer:** Hver distraktor skal være noe en student med halv forståelse faktisk kunne valgt:
   vanlige misforståelser, forveksling av nærliggende begreper (bagging/boosting, precision/recall, PCA/t-SNE,
   α/γ/ε, Gini/entropi, varians/bias), riktig mekanisme men feil retning (øker/minker), sann påstand som ikke svarer
   på spørsmålet, eller riktig konklusjon med feil begrunnelse. Ingen åpenbart absurde alternativer
   (som «fjerner skjulte lag», «matplotlib.partition», «agenten blir bevisst»).
3. **Absolutte ord** (alltid, aldri, garanterer, kun, umulig, nødvendigvis): skal ikke signalisere feil. Bruk dem
   også i riktige alternativer der det er sant (f.eks. «Inertia kan aldri øke når en sentroide legges til ved
   optimal plassering»), og bruk forsiktige formuleringer («kan», «typisk») i noen gale alternativer.
4. **Ingen ord-ekko:** Ikke la det riktige alternativet alene gjenta et særpreget ord/begrep fra stammen. Bruk
   fagbegreper fra stammen i distraktorene også. `build.py` flagger dette.
5. **Ingen mønstre:** Unngå at ett alternativ er «midt imellom» de andre, at tre alternativer deler en felles feil
   som det fjerde mangler, eller to alternativer som er hverandres negasjon (det avslører at ett av dem er riktig).
6. **Tall-alternativer:** Distraktorer fra typiske regnefeil (glemt ledd, feil fortegn, feil nevner, glemt kvadrat,
   log10 vs ln, byttet om precision/recall, brukt test i stedet for trening ...). Samme antall desimaler i alle.
   Riktig svar skal ikke være medianen av tallene systematisk.
7. **Kode-alternativer:** Alle fire skal være syntaktisk plausible og bruke ekte API-er; forskjellen skal ligge i
   semantikken (fit vs transform, feil akse, feil indeks osv.).
8. Når du er ferdig: les hvert spørsmål som en student som *ikke* kan svaret og spør «kan jeg eliminere noe uten
   fagkunnskap?». Hvis ja, skriv om.

## Emnefordeling
Kjør `emneplan.py NN` for ditt sett. Plan for Part 1–9 og 10 (Part 10 = 2 spørsmål forklarbar KI/SHAP + 2
kodespørsmål). Planen er en sterk anbefaling; du kan bytte et underemne innen samme Part dersom det gir et bedre
spørsmål, men hold deg innen Part-temaet. Hvert underemne brukes i to av de ti settene: det **laveste** settnummeret
lager et konseptuelt spørsmål om underemnet og det **høyeste** et regne-/scenariospørsmål (eller motsatt hvis
underemnet er rent numerisk – da: ulike tall og ulik vinkling).

## Tillegg etter blind review (runde 2)
9. **Forbeholdsbalanse:** Riktig svar skal ikke være «det eneste nyanserte» alternativet (det eneste med «kan»,
   «trolig», «..., men ...», «krever skjønn»), mens distraktorene er kategoriske. Minst én distraktor i hvert
   konseptspørsmål skal være like forsiktig/nyansert formulert, og riktig svar skal i mange spørsmål være kategorisk.
10. **Ingen åpenbart gale eller selvmotsigende distraktorer**, og ingen distraktorer som blander inn urelaterte
    begreper (α/γ i et spørsmål om noe annet, «deling på null» osv.). Alle tre skal kreve fagkunnskap å forkaste.
11. **Stammen skal ikke røpe svaret** (f.eks. «målestøy» i stammen → «støyfjerning» i svaret, eller notatenes
    ordrette beskrivelse av begrepet som spørres etter).
12. **Ingen duplikater på tvers av sett.** Før du skriver et nytt spørsmål: `python3 tools/indeks.py --part N` viser
    alle sett sine spørsmål i Part N. Nytt spørsmål = nytt scenario/nye tall/ny vinkel som ikke finnes der.
