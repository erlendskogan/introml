# Øvingseksamener TDT4172 (10 sett)

`Ovingseksamen_01.pdf` … `Ovingseksamen_10.pdf`: 10 Parts × 4 flervalgsspørsmål, samme oppsett som ordinær eksamen 2025.
Fasit står på siste side i hver PDF.

- `src/eksamen_NN.txt` – kildetekst (riktig alternativ markert `@+`, utregninger i `%%%`-kommentarer)
- `tools/build.py build src/eksamen_NN.txt` – bygger PDF (stokker alternativene, A/B/C/D riktig 10 ganger hver, lager fasit)
- `tools/gamecheck.py` – tester gjettestrategier (lengst, kortest, absolutte ord, konvergens, midterste tall …)
- `tools/indeks.py --part N` – oversikt over alle spørsmål, for å unngå duplikater
- `tools/SKRIVEGUIDE.md` – reglene spørsmålene er skrevet etter
