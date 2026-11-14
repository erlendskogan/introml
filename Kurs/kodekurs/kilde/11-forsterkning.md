=== DEL 11 | forsterkning | Forsterkningslæring (prosjekt 4)
Koden bak Q-learning i prosjekt 4: et miljø med `reset()` og `step()`, en Q-tabell, ε-greedy, Q-oppdateringen, treningsløkka og heatmaps. Til slutt Gymnasium og Deep Q-learning med replay buffer og target network.

--- LEKSJON 11.1 | miljo | Miljø, state, action og reward
I forsterkningslæring (reinforcement learning, RL) finnes det ingen fasit med `y`-verdier. En **agent** lærer ved å prøve seg fram i et **miljø** (environment). I hvert steg ser agenten en **state**, velger en **action**, og miljøet svarer med en ny state og en **reward**. Dette kalles RL-loopen.

I prosjekt 4 er miljøet klassen `TrondheimEnv`: et rutenett der agenten (gul) skal finne veien til målet (svart). Gymnasium og bilder finnes ikke i nettleseren, så vi lager en liten utgave i ren Python med **nøyaktig samme grensesnitt**. Les gjennom klassen, den kommer igjen i resten av delen:

```python
import numpy as np

KART = ["......",
        ".WW...",
        "..W.G.",
        "..W...",
        "..WWW.",
        "S....."]

class GridEnv:
    """Mini-TrondheimEnv. S = start, G = mål, W = vann, . = land"""
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols = len(kart), len(kart[0])
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # 0 opp, 1 venstre, 2 ned, 3 høyre
        self.n_actions = 4
        self.reset()

    def reset(self):
        start = "".join(self.kart).index("S")
        self.pos = (start // self.ncols, start % self.ncols)
        return self.pos

    def step(self, action):
        r = self.pos[0] + self.neighbours[action][0]
        c = self.pos[1] + self.neighbours[action][1]
        if not (0 <= r < self.nrows and 0 <= c < self.ncols):
            return self.pos, 0, False, None        # utenfor kartet: blir stående
        if self.kart[r][c] == "W":
            if self.dead_allowed:
                return self.pos, -1, True, None    # i vannet: -1, og episoden er over
            return self.pos, 0, False, None        # vannet blokkerer
        self.pos = (r, c)
        if self.kart[r][c] == "G":
            return self.pos, 10, True, None        # i mål!
        return self.pos, 0, False, None

    def get_state_id(self):
        return self.pos[0] * self.ncols + self.pos[1]

env = GridEnv(KART)
print("Kartet er", env.nrows, "x", env.ncols)
print("Startposisjon (rad, kolonne):", env.reset())
print("State-id:", env.get_state_id())
print("Kartet som array:")
display(np.array([list(rad) for rad in KART]))
```

Under resultatet ser du kartet som et 6 × 6-rutenett, og i variabelpanelet ligger `env` med alle `self.`-attributtene: `nrows`, `ncols`, `neighbours`, `n_actions` og `pos`, posisjonen agenten står på akkurat nå.

De tre metodene er alt en RL-agent trenger:

| Metode | Hva den gjør | Returnerer |
|---|---|---|
| `reset()` | setter agenten tilbake på start | første observasjon |
| `step(action)` | utfører én action | `obs, reward, terminated, info` |
| `get_state_id()` | gjør posisjonen om til ett heltall | `row * ncols + col` |

Reglene er de samme som i prosjektet: reward **10** i mål, **0** for alle andre steg. Vann og kanten av kartet blokkerer (agenten blir stående). Med `dead_allowed=True` gir vann reward **−1** og avslutter episoden.

## Ta steg selv
Prosjektet bruker en dictionary for å slippe å huske tallene for hver action. Vi går tre steg til høyre og ett opp:

```python
KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols = len(kart), len(kart[0])
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # 0 opp, 1 venstre, 2 ned, 3 høyre
        self.n_actions = 4
        self.reset()
    def reset(self):
        start = "".join(self.kart).index("S")
        self.pos = (start // self.ncols, start % self.ncols)
        return self.pos
    def step(self, action):
        r = self.pos[0] + self.neighbours[action][0]
        c = self.pos[1] + self.neighbours[action][1]
        if not (0 <= r < self.nrows and 0 <= c < self.ncols):
            return self.pos, 0, False, None
        if self.kart[r][c] == "W":
            if self.dead_allowed:
                return self.pos, -1, True, None
            return self.pos, 0, False, None
        self.pos = (r, c)
        if self.kart[r][c] == "G":
            return self.pos, 10, True, None
        return self.pos, 0, False, None
    def get_state_id(self):
        return self.pos[0] * self.ncols + self.pos[1]

action_dict = {'up': 0, 'left': 1, 'down': 2, 'right': 3}

env = GridEnv(KART)
env.reset()
for my_action in ['right', 'right', 'right', 'up', 'up']:
    obs, rew, term, info = env.step(action_dict[my_action])
    print(f"{my_action:6s} -> posisjon {obs}, state {env.get_state_id():2d}, reward {rew}, terminated {term}")
```

De to siste stegene (`up` fra rad 5, kolonne 3) går rett i vannet, så agenten blir stående på samme state. Klassen er komprimert her (uten kommentarer). Det er den samme koden som over.

::: prosjekt Prosjekt 4, TrondheimEnv
I prosjektet ser det slik ut: `obs, rew, term, info = env.step(action_dict[my_action])`, og `env.render("human")` tegner kartet. Én forskjell: `TrondheimEnv` returnerer hele kartet (et 2D-array med klassenummer) som `obs`, mens vår returnerer bare posisjonen. Det spiller ingen rolle, fordi Q-learning-koden uansett bruker `env.get_state_id()`. Sett `env.verbose = False` under trening, ellers skriver miljøet ut en linje for hvert steg.
:::

## State-id: fra (rad, kolonne) til ett tall
Q-tabellen (neste leksjon) har én rad per state, så posisjonen må bli ett heltall. `get_state_id()` nummererer rutene radvis: `row * ncols + col`. Den motsatte veien går med heltallsdivisjon `//` og rest `%`:

```python
import numpy as np

nrows, ncols = 6, 6
display(np.arange(nrows * ncols).reshape(nrows, ncols))   # state-id for hver rute

state = 32
print("state", state, "er rad", state // ncols, "kolonne", state % ncols)
```

Kartet i prosjektets første problem er 12 × 12, så der finnes det 12 · 12 = **144** states.

## En tilfeldig agent
Den enkleste agenten velger en tilfeldig action i hvert steg. Løkka under er RL-loopen i sin reneste form: `reset()`, deretter `step()` til `terminated` er `True` eller vi når maks antall steg.

```python
import numpy as np

KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols = len(kart), len(kart[0])
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        self.n_actions = 4
        self.reset()
    def reset(self):
        start = "".join(self.kart).index("S")
        self.pos = (start // self.ncols, start % self.ncols)
        return self.pos
    def step(self, action):
        r = self.pos[0] + self.neighbours[action][0]
        c = self.pos[1] + self.neighbours[action][1]
        if not (0 <= r < self.nrows and 0 <= c < self.ncols):
            return self.pos, 0, False, None
        if self.kart[r][c] == "W":
            if self.dead_allowed:
                return self.pos, -1, True, None
            return self.pos, 0, False, None
        self.pos = (r, c)
        if self.kart[r][c] == "G":
            return self.pos, 10, True, None
        return self.pos, 0, False, None
    def get_state_id(self):
        return self.pos[0] * self.ncols + self.pos[1]

np.random.seed(0)
for dead_allowed in [False, True]:
    env = GridEnv(KART, dead_allowed=dead_allowed)
    for episode in range(3):
        env.reset()
        total_reward = 0
        for step in range(200):
            action = np.random.randint(env.n_actions)      # tilfeldig action
            obs, reward, terminated, info = env.step(action)
            total_reward += reward
            if terminated:
                break
        print(f"dead_allowed={dead_allowed}: episode {episode} sluttet etter {step + 1:3d} steg, reward {total_reward}")
```

Uten `dead_allowed` vandrer agenten rundt til den tilfeldigvis treffer målet (eller går tom for steg). Med `dead_allowed=True` havner den nesten alltid i vannet først og får −1. En tilfeldig agent lærer ingenting. Resten av delen handler om å gjøre den smart.

::: husk
Antall steg: `for step in range(200)` starter på 0, så antall steg i episoden er `step + 1` når vi bryter ut med `break`. Prosjektets `train_q_learning` bruker akkurat dette: `steps_per_episode.append(step + 1)`.
:::

::: oppgave State-id begge veier
Skriv to funksjoner for et kart med `ncols` kolonner:
- `til_state_id(row, col, ncols)` som returnerer state-id-en (`row * ncols + col`).
- `fra_state_id(state, ncols)` som returnerer tuppelen `(row, col)`.
```start
def til_state_id(row, col, ncols):
    return ...

def fra_state_id(state, ncols):
    return ...
```
```losning
def til_state_id(row, col, ncols):
    return row * ncols + col

def fra_state_id(state, ncols):
    return (state // ncols, state % ncols)
```
```test
assert til_state_id(0, 0, 12) == 0, "Øverste venstre rute skal ha state 0."
assert til_state_id(10, 10, 12) == 130, "Rad 10, kolonne 10 på et 12 x 12-kart skal gi 10 * 12 + 10 = 130."
assert til_state_id(1, 0, 6) == 6, "Første rute på rad 1 skal være 6 når ncols = 6."
assert tuple(fra_state_id(130, 12)) == (10, 10), "fra_state_id(130, 12) skal gi (10, 10). Bruk // og %."
assert tuple(fra_state_id(32, 6)) == (5, 2), "fra_state_id(32, 6) skal gi (5, 2)."
for s in range(36):
    assert til_state_id(*fra_state_id(s, 6), 6) == s, "Funksjonene skal være hverandres motsatte."
```
:::

::: oppgave Følg en plan
Agenten under skal følge listen `plan` med actions. Kjør planen i miljøet, og lagre summen av rewardene i `total_reward` og state-id-en der agenten ender i `slutt_state`. Stopp hvis episoden avsluttes underveis.
```start
KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols = len(kart), len(kart[0])
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        self.n_actions = 4
        self.reset()
    def reset(self):
        start = "".join(self.kart).index("S")
        self.pos = (start // self.ncols, start % self.ncols)
        return self.pos
    def step(self, action):
        r = self.pos[0] + self.neighbours[action][0]
        c = self.pos[1] + self.neighbours[action][1]
        if not (0 <= r < self.nrows and 0 <= c < self.ncols):
            return self.pos, 0, False, None
        if self.kart[r][c] == "W":
            if self.dead_allowed:
                return self.pos, -1, True, None
            return self.pos, 0, False, None
        self.pos = (r, c)
        if self.kart[r][c] == "G":
            return self.pos, 10, True, None
        return self.pos, 0, False, None
    def get_state_id(self):
        return self.pos[0] * self.ncols + self.pos[1]

plan = [3, 3, 3, 3, 3, 0, 0, 0, 1]   # høyre x5, opp x3, venstre x1
env = GridEnv(KART)
env.reset()
total_reward = 0
# Skriv løkka her

slutt_state = ...
```
```losning
KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols = len(kart), len(kart[0])
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        self.n_actions = 4
        self.reset()
    def reset(self):
        start = "".join(self.kart).index("S")
        self.pos = (start // self.ncols, start % self.ncols)
        return self.pos
    def step(self, action):
        r = self.pos[0] + self.neighbours[action][0]
        c = self.pos[1] + self.neighbours[action][1]
        if not (0 <= r < self.nrows and 0 <= c < self.ncols):
            return self.pos, 0, False, None
        if self.kart[r][c] == "W":
            if self.dead_allowed:
                return self.pos, -1, True, None
            return self.pos, 0, False, None
        self.pos = (r, c)
        if self.kart[r][c] == "G":
            return self.pos, 10, True, None
        return self.pos, 0, False, None
    def get_state_id(self):
        return self.pos[0] * self.ncols + self.pos[1]

plan = [3, 3, 3, 3, 3, 0, 0, 0, 1]
env = GridEnv(KART)
env.reset()
total_reward = 0
for action in plan:
    obs, reward, terminated, info = env.step(action)
    total_reward += reward
    if terminated:
        break
slutt_state = env.get_state_id()
```
```test
assert slutt_state is not ..., "Sett slutt_state = env.get_state_id() etter løkka."
assert total_reward == 10, f"Planen fører til målet, så total_reward skal være 10, ikke {total_reward}."
assert slutt_state == 16, f"Målet ligger på rad 2, kolonne 4, altså state 2 * 6 + 4 = 16. Du fikk {slutt_state}."
```
:::

::: quiz
En agent står på rad 3, kolonne 4 i et kart med 12 rader og 12 kolonner. Hva returnerer `env.get_state_id()`, som regner ut `row * ncols + col`?
- [ ] 51
- [x] 40
- [ ] 7
- [ ] 12
? 3 · 12 + 4 = 40. Svaret 51 får du hvis du bytter om rad og kolonne (4 · 12 + 3).
:::

::: quiz
Agenten står rett ved siden av vann og velger å gå inn i vannet. Miljøet har `dead_allowed=False`. Hva returnerer `env.step(action)`?
- [ ] Ny posisjon i vannet, reward −1 og `terminated = True`
- [x] Samme posisjon som før, reward 0 og `terminated = False`
- [ ] Startposisjonen, reward 0 og `terminated = False`
- [ ] Samme posisjon som før, reward −1 og `terminated = False`
? Uten `dead_allowed` blokkerer vannet bare trekket, akkurat som kanten av kartet. Med `dead_allowed=True` gir vannet −1 og avslutter episoden.
:::

::: oppsummering
- RL-loopen: agenten velger en **action** ut fra **state**, miljøet svarer med ny state, **reward** og om episoden er over.
- `env.reset()` starter en ny episode. `env.step(action)` returnerer `obs, reward, terminated, info` (prosjektets TrondheimEnv).
- Actions i prosjektet: 0 opp, 1 venstre, 2 ned, 3 høyre. Reward 10 i mål, 0 ellers, −1 i vann med `dead_allowed=True`.
- `get_state_id()` = `row * ncols + col`. Tilbake: `state // ncols` og `state % ncols`.
- Antall steg i en episode er `step + 1` når løkka er `for step in range(max_steps)`.
:::

--- LEKSJON 11.2 | q-tabell | Q-tabellen
Agenten trenger et sted å lagre det den lærer. **Q-tabellen** har én rad per state og én kolonne per action. Tallet `q_table[s, a]` er agentens anslag på hvor mye reward den kan forvente i framtiden hvis den tar action `a` i state `s` (og deretter oppfører seg best mulig). Q står for *quality*.

Reward kommer fra miljøet. Q-verdiene må agenten **lære**. I starten vet den ingenting, så tabellen fylles med nuller:

```python
import numpy as np

nrows, ncols = 12, 12           # kartet i prosjektets første problem
num_states = nrows * ncols      # én state per rute
num_actions = 4                 # opp, venstre, ned, høyre
q_table = np.zeros((num_states, num_actions))
print(q_table.shape)
print("V(s) = q_table.max(axis=1) i kartets form, før trening:")
display(q_table.max(axis=1).reshape(nrows, ncols))
```

Se `q_table` i variabelpanelet: 144 rader og 4 kolonner med bare nuller. Agenten vet ingenting ennå, og verdikartet over er helt flatt.

::: prosjekt Prosjekt 4, step 1: Initialize Q table
Prosjektets løsning er `num_states = int(np.prod(env.observation_space.shape))` (12 · 12 = 144) og `num_actions = env.action_space.n` (4), og så `q_table = np.zeros((num_states, num_actions))`. Spørsmålet «Why does the Q-table have 144 states?» har svaret: staten er bare agentens **posisjon**, og kartet har 144 ruter.
:::

## Lese av Q-tabellen
Når tabellen er fylt ut, bruker vi den slik:
- `q_table[state]` er raden for én state: fire Q-verdier, én per action.
- `np.argmax(q_table[state])` er den **beste** actionen i den staten (den grådige, *greedy*, actionen).
- `q_table.max(axis=1)` gir den beste Q-verdien i hver state. Det er verdien av staten, $V(s) = \max_a Q(s, a)$.
- `q_table.argmax(axis=1)` gir beste action i alle states på én gang: hele **policyen**.

Her er en liten, ferdig utfylt Q-tabell for et 2 × 3-kart med mål nede til høyre:

```python
import numpy as np

nrows, ncols = 2, 3
q_table = np.array([
    # opp   venst  ned   høyre
    [7.29, 7.29, 8.1,  8.1 ],    # state 0 (rad 0, kol 0)
    [8.1,  7.29, 9.0,  9.0 ],    # state 1
    [9.0,  8.1, 10.0,  9.0 ],    # state 2
    [7.29, 8.1,  8.1,  9.0 ],    # state 3 (rad 1, kol 0)
    [8.1,  8.1,  9.0, 10.0 ],    # state 4
    [0.0,  0.0,  0.0,  0.0 ],    # state 5 = målet (oppdateres aldri)
])

state = 4
print("Q-verdier i state 4:", q_table[state])
print("Beste action i state 4:", np.argmax(q_table[state]))

print("V(s) for hver rute:")
display(q_table.max(axis=1).reshape(nrows, ncols))

piler = np.array(["↑", "←", "↓", "→"])
print("Policy:")
display(piler[q_table.argmax(axis=1)].reshape(nrows, ncols))
```

Verdiene blir mindre jo lenger unna målet ruta er (10, 9.0, 8.1), og fargeskalaen i verdikartet gjør det lett å se. Hvorfor, ser vi i leksjon 11.4. `reshape(nrows, ncols)` gjør de 6 tallene om til kartets form igjen, fordi state-id-ene er nummerert radvis. Dette er akkurat det prosjektet gjør før heatmapene tegnes. Legg merke til at målet får en pil opp: raden er bare nuller, og da velger `argmax` den første.

::: advarsel
`np.argmax` returnerer den **første** indeksen ved likhet. En Q-tabell full av nuller gir derfor alltid action 0 (opp). Det er årsaken til at en agent uten utforsking setter seg fast i prosjektet. Det ser vi mer på i neste leksjon.
:::

## Q-tabell som dictionary (tre-på-rad fra notatene)
Når states ikke er pene heltall, er det vanlig å lagre Q-tabellen som en **nøstet dictionary**: state som nøkkel, og en ny dictionary med én Q-verdi per action. I notatenes tre-på-rad-agent er staten en streng som `'0123XO678'`, der tallene er ledige ruter. Her er Q-verdiene en ferdig trent X-agent har for det tomme brettet:

```python
values = {
    '012345678': {0: 1.933, 1: 1.549, 2: 1.561, 3: 2.140, 4: 2.476,
                  5: 0.656, 6: 1.024, 7: 1.394, 8: 1.667},
}

state = '012345678'
q = values[state]
best_action = max(q, key=q.get)       # nøkkelen med størst verdi
print("Beste trekk på tomt brett:", best_action)

# Lovlige trekk er rutene som fortsatt er et tall
state = '0123XO678'
available_actions = [_i for _i, _s in enumerate(state) if _s.isnumeric()]
print("Ledige ruter i", state, ":", available_actions)

# En state agenten aldri har sett, får Q-verdi 0 for alle lovlige trekk
values.setdefault(state, {a: 0.0 for a in available_actions})
print(values[state])
```

Agenten vil sette krysset i rute 4, midten av brettet, som er optimalt spill. `max(q, key=q.get)` er dictionary-utgaven av `np.argmax`. `setdefault` legger inn en ny state med nuller bare hvis den ikke finnes fra før.

::: tips
Med en numpy-tabell må du vite antall states på forhånd. En dictionary vokser etter hvert som agenten møter nye states. Det passer for tre-på-rad, der svært mange brett aldri dukker opp.
:::

::: oppgave Lag Q-tabellen
Et kart har 5 rader og 7 kolonner, og agenten har 4 actions.
1. Lag `q_table` fylt med nuller og riktig shape.
2. Sett Q-verdien for state 9, action 2 (ned) til `4.5`.
3. Lagre beste action i state 9 i `beste`, ved hjelp av `np.argmax`.
```start
import numpy as np

q_table = ...

beste = ...
```
```losning
import numpy as np

q_table = np.zeros((5 * 7, 4))
q_table[9, 2] = 4.5
beste = np.argmax(q_table[9])
```
```test
import numpy as np
assert isinstance(q_table, np.ndarray), "q_table skal være et numpy-array, for eksempel np.zeros(...)."
assert q_table.shape == (35, 4), f"Shape skal være (antall states, antall actions) = (35, 4), ikke {q_table.shape}."
assert q_table[9, 2] == 4.5, "Sett q_table[9, 2] = 4.5."
assert q_table.sum() == 4.5, "Alle andre verdier skal være 0."
assert beste == 2, "Beste action i state 9 skal være 2 (ned)."
```
:::

::: quiz
`q_table` har shape `(144, 4)`. Hva er shape på `q_table.argmax(axis=1).reshape(12, 12)`?
- [ ] `(144, 4)`
- [ ] `(4,)`
- [x] `(12, 12)`
- [ ] `(12, 12, 4)`
? `argmax(axis=1)` tar beste action for hver av de 144 radene og gir shape `(144,)`. `reshape(12, 12)` legger dem ut som kartet. Resultatet er policyen: ett tall (0–3) per rute.
:::

::: quiz
En tre-på-rad-agent har disse Q-verdiene for en state: `{0: 0.2, 2: 1.4, 6: 0.9, 8: 1.4}`. Agenten velger beste trekk med `max(q, key=q.get)`. Hvilket trekk gjør den?
- [ ] 8, fordi den siste av de like verdiene vinner
- [x] 2, fordi `max` returnerer den første nøkkelen med høyest verdi
- [ ] 1, fordi det er indeksen til den høyeste verdien i listen
- [ ] Koden krasjer fordi to trekk har samme verdi
? Både 2 og 8 har 1.4. `max` (og `np.argmax`) returnerer den første den møter med høyest verdi, og dictionaries holder innsettingsrekkefølgen. Merk at trekket er nøkkelen 2, ikke posisjonen 1 i dictionaryen.
:::

::: oppsummering
- Q-tabellen har shape `(num_states, num_actions)`, i prosjektet `(144, 4)`. Start med `np.zeros`.
- `q_table[s, a]` = forventet framtidig reward for action `a` i state `s`. Reward kommer fra miljøet, Q må læres.
- `np.argmax(q_table[state])` = beste action. `q_table.max(axis=1)` = verdien av hver state. `q_table.argmax(axis=1)` = hele policyen.
- `reshape(nrows, ncols)` legger verdiene ut som kartet.
- Notatene lagrer Q-tabellen som nøstet dictionary: `values[state][action]`, og `max(q, key=q.get)` gir beste action.
- Ved likhet vinner den første: en tabell full av nuller gir alltid action 0.
:::

--- LEKSJON 11.3 | epsilon | ε-greedy og decay
Hvis agenten alltid velger den beste actionen ifølge Q-tabellen, gjør den bare det den allerede tror er lurt. Den finner aldri målet hvis den ikke har sett det. Løsningen er **ε-greedy**: med sannsynlighet ε (exploration rate) tar agenten en **tilfeldig** action (*explore*), ellers tar den den beste (*exploit*).

Dette er prosjektets `update_action`. Funksjonen trenger bare antall actions fra miljøet, så vi lager et lite stand-in-miljø:

```python
import numpy as np

class DummyEnv:          # update_action trenger bare antall actions
    n_actions = 4

def update_action(env, state, explore_rate, q_table):
    if np.random.random() < explore_rate:
        return np.random.randint(env.n_actions)   # explore: tilfeldig action
    return int(np.argmax(q_table[state]))         # exploit: beste action

env = DummyEnv()
q_table = np.zeros((36, 4))
q_table[5] = [0.0, 0.0, 0.0, 7.0]                 # i state 5 er "høyre" best

np.random.seed(0)
for explore_rate in [0.0, 0.2, 1.0]:
    valg = [update_action(env, 5, explore_rate, q_table) for _ in range(10000)]
    print(f"ε = {explore_rate}: andel av hver action:", np.bincount(valg, minlength=4) / 10000)
```

Med ε = 0.2 velges «høyre» i omtrent 85 % av tilfellene, ikke 80 %. Den tilfeldige actionen kan nemlig også bli «høyre»: $0.8 + 0.2 \cdot \tfrac14 = 0.85$.

::: prosjekt Prosjekt 4, update_action
I prosjektet står det `np.random.randint(env.action_space.n)`. `env.action_space` er Gymnasium sitt `Discrete(4)`, og `.n` er antall actions. Vårt `env.n_actions` er det samme tallet.
:::

## Exploration rate som synker (decay)
Tidlig i treningen vet agenten ingenting og bør utforske mye. Senere vet den mer og bør stort sett utnytte det den har lært. Derfor lar vi ε **synke** over episodene. Prosjektets løsning bruker eksponentiell decay fra `epsilon_max` mot `epsilon_min`, og samme form for læringsraten α:

```python
import numpy as np
import matplotlib.pyplot as plt

def update_explore_rate(episode, epsilon_min, max_episodes, epsilon_max=1.0):
    return epsilon_min + (epsilon_max - epsilon_min) * np.exp(-5 * episode / max_episodes)

def update_learning_rate(episode, min_learning_rate, max_episodes):
    return min_learning_rate + (1.0 - min_learning_rate) * np.exp(-5 * episode / max_episodes)

max_episodes = 500
episodes = np.arange(max_episodes)
plt.plot(episodes, update_explore_rate(episodes, 0.01, max_episodes), label="explore rate ε (min 0.01)")
plt.plot(episodes, update_learning_rate(episodes, 0.1, max_episodes), label="learning rate α (min 0.1)")
plt.xlabel("Episode")
plt.legend()
plt.show()

print("ε etter 0, 100 og 499 episoder:", [round(float(update_explore_rate(e, 0.01, 500)), 3) for e in [0, 100, 499]])
print("ε for episode 0, 50, 100, ..., 450:")
display(update_explore_rate(np.arange(0, 500, 50), 0.01, max_episodes))
```

Faktoren `-5 * episode / max_episodes` gjør at ε er nesten nede på minimum etter rundt 80 % av episodene, uansett hvor mange episoder du trener. Funksjonene virker på et helt array av episoder på én gang, fordi `np.exp` jobber elementvis.

::: hvorfor
**Hvorfor senke læringsraten?** α bestemmer hvor mye én ny erfaring overskriver det gamle anslaget. Tidlig er anslagene dårlige, og stor α lærer raskt. Senere vil vi at verdiene skal **stabilisere seg**, slik at ett tilfeldig utforskingssteg ikke ødelegger det agenten har lært.
:::

## Notatenes versjon: multiplikativ decay
Tre-på-rad-agenten i notatene oppdaterer ε etter hver episode ved å gange med en fast faktor:

```python
exp_factor = 1.0
exp_min = 0.05
epsilon_decay = 0.9

def update_exp_factor():
    global exp_factor
    if exp_factor > exp_min:
        exp_factor *= epsilon_decay

historikk = []
for episode in range(40):
    historikk.append(exp_factor)
    update_exp_factor()

print([round(e, 3) for e in historikk[:6]], "...")
print("Etter 40 episoder:", round(exp_factor, 4))
```

Åpne `historikk` i variabelpanelet for å se alle de 40 verdiene. Legg merke til at ε havner litt **under** `exp_min` (0.0466). Sjekken skjer før gangingen: 0.0517 er større enn 0.05, så den ganges én siste gang. Etter $n$ oppdateringer er $\varepsilon = \varepsilon_0 \cdot \text{decay}^n$ så lenge den er over minimum. Prøv `epsilon_decay = 0.99` og se hvor mye lenger agenten utforsker.

::: tips
Notatene anbefaler å holde en liten ε også etter treningen: en godt trent agent kan kjøre seg fast og trenge et tilfeldig dytt. I prosjektet bruker vi likevel ε = 0 når vi **viser** den trente agenten, for da vil vi se nøyaktig hva policyen gjør.
:::

::: oppgave Skriv update_action
Fullfør `update_action(env, state, explore_rate, q_table)`: trekk et tall med `np.random.random()`. Er det mindre enn `explore_rate`, returner en tilfeldig action med `np.random.randint(env.n_actions)`. Ellers returner `int(np.argmax(q_table[state]))`.
```start
import numpy as np

def update_action(env, state, explore_rate, q_table):
    ...
```
```losning
import numpy as np

def update_action(env, state, explore_rate, q_table):
    if np.random.random() < explore_rate:
        return np.random.randint(env.n_actions)
    return int(np.argmax(q_table[state]))
```
```test
import numpy as np
class _Env:
    n_actions = 4
_q = np.zeros((10, 4))
_q[3] = [0, 0, 5, 1]
_valg0 = [update_action(_Env(), 3, 0.0, _q) for _ in range(200)]
assert all(v == 2 for v in _valg0), "Med explore_rate = 0 skal funksjonen alltid returnere argmax, her action 2."
np.random.seed(1)
_valg1 = [update_action(_Env(), 3, 1.0, _q) for _ in range(2000)]
assert set(_valg1) == {0, 1, 2, 3}, "Med explore_rate = 1 skal alle fire actions dukke opp."
_andel = np.mean(np.array(_valg1) == 2)
assert 0.18 < _andel < 0.32, "Med explore_rate = 1 skal actionen være helt tilfeldig (omtrent 25 % hver)."
np.random.seed(2)
_valg2 = np.array([update_action(_Env(), 3, 0.4, _q) for _ in range(4000)])
assert 0.64 < np.mean(_valg2 == 2) < 0.76, "Med explore_rate = 0.4 skal action 2 velges i omtrent 0.6 + 0.4/4 = 70 % av tilfellene."
```
:::

::: quiz
Hva er verdien av `epsilon` etter løkka?
```kode
epsilon, epsilon_min, epsilon_decay = 1.0, 0.1, 0.6

def update_epsilon():
    global epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

for episode in range(5):
    update_epsilon()
```
- [ ] 0.100
- [x] 0.078
- [ ] 0.130
- [ ] 0.047
? 1 → 0.6 → 0.36 → 0.216 → 0.1296 → 0.0778. Ved femte kall er 0.1296 fortsatt større enn 0.1, så den ganges én gang til og havner under minimum. 0.047 er $0.6^6$ (ett kall for mye).
:::

::: quiz
Hva er hensikten med utforskingsfaktoren ε i en ε-greedy agent? (Eksamen 2025)
- [ ] Å garantere at agenten lærer den optimale policyen uten å gjøre feil
- [ ] Å tvinge agenten til alltid å velge actionen med høyest Q-verdi
- [ ] Å få agenten til å ignorere framtidige rewards
- [x] Å sørge for at agenten av og til velger en tilfeldig action for å utforske miljøet
? Med sannsynlighet ε velges en tilfeldig action. Uten det ville agenten bare gjenta det den allerede tror er best, og aldri oppdage bedre veier.
:::

::: oppsummering
- **ε-greedy**: `if np.random.random() < explore_rate:` tilfeldig action, ellers `np.argmax(q_table[state])`.
- Sannsynligheten for den grådige actionen er $1 - \varepsilon + \varepsilon / n_{\text{actions}}$.
- **Decay**: høy ε tidlig (utforsk), lav ε sent (utnytt). Prosjektet: `epsilon_min + (epsilon_max - epsilon_min) * np.exp(-5 * episode / max_episodes)`.
- Notatene: `if exp_factor > exp_min: exp_factor *= epsilon_decay` etter hver episode. Kan havne litt under minimum.
- Læringsraten α senkes på samme måte, slik at Q-verdiene stabiliserer seg.
- Når den trente agenten skal vises, brukes ε = 0.
:::

--- LEKSJON 11.4 | q-oppdatering | Q-oppdateringen
Nå kommer selve læringen. Etter hvert steg vet agenten fire ting: staten den var i ($s$), actionen den tok ($a$), rewarden den fikk ($R$) og staten den havnet i ($s'$). Med disse flytter den Q-verdien litt mot et bedre anslag:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \Big[ \underbrace{R + \gamma \max_{a'} Q(s', a')}_{\text{target}} - Q(s, a) \Big]$$

- **target** er det nye anslaget: rewarden nå, pluss den beste Q-verdien i neste state, ganget med **γ** (discount factor).
- Hakeparentesen er **TD-feilen** (temporal difference): hvor mye anslaget bommet.
- **α** (learning rate) bestemmer hvor langt vi flytter oss mot target.
- Er $s'$ terminal (mål eller vann med `dead_allowed`), finnes det ingen framtid, og target er bare $R$.

Notatene skriver samme regel som $Q_{new} = (1-\alpha) \, Q(s,a) + \alpha \, (R + \gamma \max_{a'} Q(s', a'))$. Gang ut parentesen, så ser du at det er det samme. Vi sjekker med tall:

```python
import numpy as np

alpha, gamma = 0.4, 0.9
q_sa = 2.0                               # gammel Q(s, a)
reward = 1
q_next = np.array([3.0, 4.0, 1.5])       # Q-verdiene i neste state s'

target = reward + gamma * np.max(q_next)
ny_1 = q_sa + alpha * (target - q_sa)              # prosjektets form
ny_2 = (1 - alpha) * q_sa + alpha * target         # notatenes form
print("target =", target)
print("ny Q   =", ny_1, "=", ny_2)
```

## Slik står det i prosjektet
I `train_q_learning` er oppdateringen to linjer. `if`-uttrykket på én linje (`a if betingelse else b`) håndterer terminal state:

```python
import numpy as np

q_table = np.zeros((36, 4))
q_table[17] = [0.0, 10.0, 0.0, 0.0]      # i state 17 vet vi allerede at "venstre" gir mål
gamma, learning_rate = 0.95, 0.5

# Agenten gikk opp (action 0) fra state 23 til state 17, reward 0, ikke ferdig
state, action, reward, next_state, terminated = 23, 0, 0, 17, False
print("Q-verdiene i state 23 før oppdateringen:")
display(q_table[23])
target = reward if terminated else reward + gamma * np.max(q_table[next_state])
q_table[state, action] += learning_rate * (target - q_table[state, action])
print("Q-verdiene i state 23 etter oppdateringen:")
display(q_table[23])
print("Q(23, opp) =", q_table[23, 0])

# Agenten gikk til venstre fra state 17 og inn i målet: reward 10, ferdig
state, action, reward, next_state, terminated = 17, 1, 10, 16, True
target = reward if terminated else reward + gamma * np.max(q_table[next_state])
q_table[state, action] += learning_rate * (target - q_table[state, action])
print("Q(17, venstre) =", q_table[17, 1])
```

Den første oppdateringen flytter Q(23, opp) halvveis (α = 0.5) mot target 0 + 0.95 · 10 = 9.5. Den andre er terminal, så target er bare 10, og verdien står stille.

## Rewarden sprer seg bakover
Hvert steg gir reward 0, bortsett fra det siste. Likevel lærer agenten veien. Grunnen er leddet $\gamma \max Q(s', a')$: verdien av målet sendes ett steg bakover hver gang agenten går forbi. Her er en korridor med 5 ruter og mål til høyre for den siste. Agenten går alltid til høyre, og vi bruker α = 1 og γ = 0.9:

```python
import numpy as np

n = 5
q_right = np.zeros(n)                  # Q(s, høyre) for de 5 rutene
alpha, gamma = 1.0, 0.9

historikk = []                         # én rad per episode
for episode in range(6):
    for s in range(n):                 # går fra rute 0 mot høyre
        terminated = (s == n - 1)      # fra siste rute går vi inn i målet
        reward = 10 if terminated else 0
        target = reward if terminated else reward + gamma * q_right[s + 1]
        q_right[s] += alpha * (target - q_right[s])
    print(f"etter episode {episode}:", np.round(q_right, 2))
    historikk.append(q_right.copy())

print("Rad = episode, kolonne = rute:")
display(np.array(historikk))
```

I rutenettet ser du verdien spre seg som en trapp fra høyre mot venstre. Etter første episode vet bare ruta ved målet noe. For hver episode sprer verdien seg én rute lenger bakover, og hver rute får $10 \cdot \gamma^{d-1}$, der $d$ er antall steg til målet. En kortere vei gir altså høyere verdi. Det er derfor agenten finner **korteste** vei selv om alle steg gir reward 0.

::: hvorfor
**Hva gjør α og γ?** Med α = 0 lærer agenten ingenting (Q endres aldri). Med α = 1 overskrives den gamle verdien helt av target. γ = 0 gjør agenten kortsiktig: bare rewarden i neste steg teller. γ nær 1 gjør at rewards langt fram i tid teller nesten like mye som rewards nå. Prøv `gamma = 0` og `gamma = 1` i korridoren over.
:::

## Notatenes learn-funksjon
I notatene heter de samme størrelsene `td_target` og `td_delta`. Her er tre-på-rad-agentens `learn` skrevet for en Q-tabell som dictionary:

```python
def learn(values, state, action, reward, new_state, done, alpha=0.5, gamma=0.9):
    v_s = values[state][action]
    if done:
        v_s_next = 0                                   # ingen framtid etter siste trekk
    else:
        v_s_next = max(values[new_state].values())     # beste Q-verdi i ny state
    td_target = reward + (0 if done else gamma * v_s_next)
    td_delta = td_target - v_s
    values[state][action] = v_s + alpha * td_delta

values = {'0123X5678': {0: 0.0, 1: 0.0, 2: 0.0}, 'O123X5678': {1: 2.0, 2: 1.0}}
learn(values, '0123X5678', 0, 0, 'O123X5678', False)
print(values['0123X5678'])
```

Target er 0 + 0.9 · 2.0 = 1.8, og med α = 0.5 blir ny verdi 0 + 0.5 · 1.8 = 0.9.

::: advarsel
Glemmer du `if terminated`, legger du til $\gamma \max Q$ for målruta. Med en nullinitialisert tabell er det tilfeldigvis ufarlig (målraden oppdateres aldri og forblir 0). Med **optimistisk initialisering** (leksjon 11.6), der tabellen starter på 1.0, blir alle verdiene feil. Skriv alltid `target = reward if terminated else ...`.
:::

::: oppgave Skriv Q-oppdateringen
Lag `q_update(q_table, state, action, reward, next_state, terminated, alpha, gamma)` som oppdaterer `q_table[state, action]` på plass med Q-learning-regelen. Ved `terminated=True` er target bare `reward`. Funksjonen trenger ikke returnere noe.
```start
import numpy as np

def q_update(q_table, state, action, reward, next_state, terminated, alpha, gamma):
    target = ...
    q_table[state, action] += ...
```
```losning
import numpy as np

def q_update(q_table, state, action, reward, next_state, terminated, alpha, gamma):
    target = reward if terminated else reward + gamma * np.max(q_table[next_state])
    q_table[state, action] += alpha * (target - q_table[state, action])
```
```test
import numpy as np
_q = np.zeros((4, 2))
_q[1] = [3.0, 4.0]
_q[0, 1] = 2.0
q_update(_q, 0, 1, 1, 1, False, 0.4, 0.9)
assert np.isclose(_q[0, 1], 3.04), f"Med Q=2, R=1, max Q(s')=4, α=0.4 og γ=0.9 skal ny verdi være 2 + 0.4 * (1 + 3.6 - 2) = 3.04, ikke {_q[0, 1]}."
_q2 = np.ones((4, 2))
q_update(_q2, 2, 0, 10, 3, True, 0.5, 0.9)
assert np.isclose(_q2[2, 0], 5.5), "Ved terminated=True skal target være bare reward: 1 + 0.5 * (10 - 1) = 5.5."
_q3 = np.zeros((4, 2))
q_update(_q3, 0, 0, 0, 1, False, 1.0, 0.9)
assert _q3.sum() == 0, "Med reward 0 og bare nuller i neste state skal ingenting endres."
```
:::

::: quiz
En agent har $Q(s, a) = 1.0$, får reward $R = 0$ og havner i en ikke-terminal state med Q-verdier $(2.0,\ 4.0,\ 0.0,\ 1.0)$. Med α = 0.1 og γ = 0.9, hva blir den nye $Q(s, a)$?
- [ ] 3.60
- [ ] 1.30
- [x] 1.26
- [ ] 3.34
? Target = 0 + 0.9 · 4.0 = 3.6. Ny verdi = 1.0 + 0.1 · (3.6 − 1.0) = 1.26. 1.30 får du hvis du glemmer γ, 3.34 hvis du bytter α med 1 − α, og 3.60 er target (α = 1).
:::

::: quiz
Hva medfører en høy verdi av γ i Q-oppdateringen? (Eksamen 2024)
- [ ] Læringen domineres av de gamle Q-verdiene
- [x] Læringen domineres av estimatet av framtidige states
- [ ] Læringen domineres av den umiddelbare rewarden
- [ ] Agenten utforsker oftere
? γ ganges med $\max Q(s', a')$, som er anslaget for framtiden. Stor γ gjør at framtiden veier tungt. Hvor mye de gamle verdiene beholdes, styres av α, og utforsking styres av ε.
:::

::: oppsummering
- `target = reward if terminated else reward + gamma * np.max(q_table[next_state])`
- `q_table[state, action] += learning_rate * (target - q_table[state, action])`
- `target - Q` er TD-feilen (`td_delta` i notatene). Notatenes form $(1-\alpha)Q + \alpha \cdot \text{target}$ er den samme regelen.
- Etter en terminal state finnes ingen framtid: target = reward.
- Reward sprer seg ett steg bakover per besøk. En rute $d$ steg fra målet får til slutt $10 \cdot \gamma^{d-1}$.
- α = 0: ingen læring. α = 1: overskriv. γ = 0: kortsiktig. γ nær 1: langsiktig.
:::

--- LEKSJON 11.5 | trening | Treningsløkka
Nå setter vi alt sammen til prosjektets `train_q_learning`. Strukturen er to nøstede løkker:

1. **Ytre løkke over episoder:** `env.reset()`, hent startstate, og regn ut ε og α for denne episoden.
2. **Indre løkke over steg:** velg action (ε-greedy), ta et steg, oppdater Q-tabellen, gå videre til neste state, og `break` når episoden er over.

```python
import numpy as np

KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols = len(kart), len(kart[0])
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        self.n_actions = 4
        self.reset()
    def reset(self):
        start = "".join(self.kart).index("S")
        self.pos = (start // self.ncols, start % self.ncols)
        return self.pos
    def step(self, action):
        r = self.pos[0] + self.neighbours[action][0]
        c = self.pos[1] + self.neighbours[action][1]
        if not (0 <= r < self.nrows and 0 <= c < self.ncols):
            return self.pos, 0, False, None
        if self.kart[r][c] == "W":
            if self.dead_allowed:
                return self.pos, -1, True, None
            return self.pos, 0, False, None
        self.pos = (r, c)
        if self.kart[r][c] == "G":
            return self.pos, 10, True, None
        return self.pos, 0, False, None
    def get_state_id(self):
        return self.pos[0] * self.ncols + self.pos[1]

def update_explore_rate(episode, epsilon_min, max_episodes, epsilon_max=1.0):
    return epsilon_min + (epsilon_max - epsilon_min) * np.exp(-5 * episode / max_episodes)

def update_learning_rate(episode, min_learning_rate, max_episodes):
    return min_learning_rate + (1.0 - min_learning_rate) * np.exp(-5 * episode / max_episodes)

def update_action(env, state, explore_rate, q_table):
    if np.random.random() < explore_rate:
        return np.random.randint(env.n_actions)
    return int(np.argmax(q_table[state]))

def train_q_learning(env, q_table, num_episodes, max_steps, gamma, epsilon_min, min_learning_rate,
                     epsilon=1.0, seed=0):
    np.random.seed(seed)
    total_reward = 0
    steps_per_episode = []
    rewards_per_episode = []
    for episode in range(num_episodes):
        env.reset()                                   # husk å resette!
        state = env.get_state_id()
        explore_rate = update_explore_rate(episode, epsilon_min, num_episodes, epsilon)
        learning_rate = update_learning_rate(episode, min_learning_rate, num_episodes)
        for step in range(max_steps):
            action = update_action(env, state, explore_rate, q_table)
            _, reward, terminated, _ = env.step(action)
            next_state = env.get_state_id()
            target = reward if terminated else reward + gamma * np.max(q_table[next_state])
            q_table[state, action] += learning_rate * (target - q_table[state, action])
            total_reward += reward
            state = next_state
            if terminated:
                break
        steps_per_episode.append(step + 1)
        rewards_per_episode.append(reward)            # siste reward: 10 i mål, ellers 0
    return (q_table, total_reward, np.array(steps_per_episode), np.array(rewards_per_episode),
            explore_rate, learning_rate)

GAMMA = 0.95
NUM_EPISODES = 200
MAX_STEPS_PER_EPISODE = 100
EPSILON_MIN = 0.01
MIN_LEARNING_RATE = 0.1

env = GridEnv(KART)
q_table = np.zeros((env.nrows * env.ncols, env.n_actions))
print("V(s) før trening:")
display(q_table.max(axis=1).reshape(env.nrows, env.ncols))
q_table, total_reward, steps_per_episode, rewards_per_episode, explore_rate, learning_rate = train_q_learning(
    env, q_table, NUM_EPISODES, MAX_STEPS_PER_EPISODE, GAMMA, EPSILON_MIN, MIN_LEARNING_RATE)
print("V(s) etter trening:")
display(q_table.max(axis=1).reshape(env.nrows, env.ncols))

print(f"Nådde målet i {np.sum(rewards_per_episode == 10)} av {NUM_EPISODES} episoder")
print(f"Steg i de 5 første episodene: {steps_per_episode[:5]}, i de 5 siste: {steps_per_episode[-5:]}")
print(f"Total reward: {total_reward}, explore rate til slutt: {explore_rate:.4f}, learning rate: {learning_rate:.4f}")
```

Sammenlign de to verdikartene: før trening er alt 0, etter trening er verdiene høyest rundt målet og faller bortover. Vann, mål og ruter agenten aldri besøkte, har fortsatt verdien 0. Hele `q_table` med shape `(36, 4)` ligger i variabelpanelet som heatmap, og `steps_per_episode` og `rewards_per_episode` kan du også åpne der.

Les utskriften: de første episodene brukte alle 100 stegene uten å nå målet (ε ≈ 1, ren tilfeldig vandring). De siste bruker rundt 11 steg. Av og til tar en episode et par steg ekstra, fordi ε fortsatt er 0.01 og agenten innimellom tar et tilfeldig steg.

Noen detaljer som er lette å gå glipp av:
- `env.reset()` må kalles i starten av **hver** episode. Ellers starter neste episode der den forrige sluttet.
- `state = next_state` på slutten av hvert steg. Uten den oppdaterer agenten hele tiden samme rad i tabellen.
- `rewards_per_episode.append(reward)` lagrer den **siste** rewarden. Den er 10 bare hvis episoden endte i målet, så `np.sum(rewards_per_episode == 10)` teller antall vellykkede episoder.
- `seed=0` gjør utforskingen reproduserbar: samme kjøring gir samme Q-tabell.

::: prosjekt Prosjekt 4, step 3 og 4
Prosjektet bruker `GAMMA = 0.95`, `NUM_EPISODES = 500`, `MAX_STEPS_PER_EPISODE = 200`, `EPSILON_MIN = 0.01` og `MIN_LEARNING_RATE = 0.1` for det enkle kartet, og 3000 episoder for Trondheim-kartet. Funksjonen over er den samme som i løsningen, bortsett fra at prosjektet setter `env.verbose = False` i starten.
:::

::: eksamen
Kodelesing på eksamen handler ofte om rekkefølgen i løkka: hva skjer først (velg action, ta steg, oppdater, flytt state), hva som skjer én gang per episode (reset, decay av ε) og hva som skjer én gang per steg.
:::

::: oppgave Fullfør treningsløkka
Treningsløkka under mangler tre linjer: hent neste state, regn ut `target` og oppdater `q_table[state, action]`. Fyll inn der det står `...`. Testen kjører den trente agenten med ε = 0 og sjekker at den når målet.
```start
import numpy as np

KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols = len(kart), len(kart[0])
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        self.n_actions = 4
        self.reset()
    def reset(self):
        start = "".join(self.kart).index("S")
        self.pos = (start // self.ncols, start % self.ncols)
        return self.pos
    def step(self, action):
        r = self.pos[0] + self.neighbours[action][0]
        c = self.pos[1] + self.neighbours[action][1]
        if not (0 <= r < self.nrows and 0 <= c < self.ncols):
            return self.pos, 0, False, None
        if self.kart[r][c] == "W":
            if self.dead_allowed:
                return self.pos, -1, True, None
            return self.pos, 0, False, None
        self.pos = (r, c)
        if self.kart[r][c] == "G":
            return self.pos, 10, True, None
        return self.pos, 0, False, None
    def get_state_id(self):
        return self.pos[0] * self.ncols + self.pos[1]

def update_action(env, state, explore_rate, q_table):
    if np.random.random() < explore_rate:
        return np.random.randint(env.n_actions)
    return int(np.argmax(q_table[state]))

np.random.seed(0)
env = GridEnv(KART)
q_table = np.zeros((36, 4))
gamma, num_episodes = 0.95, 200
for episode in range(num_episodes):
    env.reset()
    state = env.get_state_id()
    explore_rate = 0.01 + 0.99 * np.exp(-5 * episode / num_episodes)
    learning_rate = 0.1 + 0.9 * np.exp(-5 * episode / num_episodes)
    for step in range(100):
        action = update_action(env, state, explore_rate, q_table)
        _, reward, terminated, _ = env.step(action)
        next_state = ...
        target = ...
        ...
        state = next_state
        if terminated:
            break
```
```losning
import numpy as np

KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols = len(kart), len(kart[0])
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        self.n_actions = 4
        self.reset()
    def reset(self):
        start = "".join(self.kart).index("S")
        self.pos = (start // self.ncols, start % self.ncols)
        return self.pos
    def step(self, action):
        r = self.pos[0] + self.neighbours[action][0]
        c = self.pos[1] + self.neighbours[action][1]
        if not (0 <= r < self.nrows and 0 <= c < self.ncols):
            return self.pos, 0, False, None
        if self.kart[r][c] == "W":
            if self.dead_allowed:
                return self.pos, -1, True, None
            return self.pos, 0, False, None
        self.pos = (r, c)
        if self.kart[r][c] == "G":
            return self.pos, 10, True, None
        return self.pos, 0, False, None
    def get_state_id(self):
        return self.pos[0] * self.ncols + self.pos[1]

def update_action(env, state, explore_rate, q_table):
    if np.random.random() < explore_rate:
        return np.random.randint(env.n_actions)
    return int(np.argmax(q_table[state]))

np.random.seed(0)
env = GridEnv(KART)
q_table = np.zeros((36, 4))
gamma, num_episodes = 0.95, 200
for episode in range(num_episodes):
    env.reset()
    state = env.get_state_id()
    explore_rate = 0.01 + 0.99 * np.exp(-5 * episode / num_episodes)
    learning_rate = 0.1 + 0.9 * np.exp(-5 * episode / num_episodes)
    for step in range(100):
        action = update_action(env, state, explore_rate, q_table)
        _, reward, terminated, _ = env.step(action)
        next_state = env.get_state_id()
        target = reward if terminated else reward + gamma * np.max(q_table[next_state])
        q_table[state, action] += learning_rate * (target - q_table[state, action])
        state = next_state
        if terminated:
            break
```
```test
import numpy as np
assert q_table.shape == (36, 4), "q_table skal fortsatt ha shape (36, 4)."
assert q_table.max() > 0, "Q-tabellen er bare nuller. Oppdaterer du q_table[state, action]?"
assert q_table.max() <= 10 + 1e-9, "Ingen Q-verdi skal bli større enn 10 (rewarden i målet). Sjekk target ved terminated."
_env = GridEnv(KART)
_env.reset()
_s = _env.get_state_id()
for _t in range(50):
    _, _r, _term, _ = _env.step(int(np.argmax(q_table[_s])))
    _s = _env.get_state_id()
    if _term:
        break
assert _term and _r == 10, "Den grådige agenten (ε = 0) når ikke målet. Sjekk target og oppdateringen."
```
:::

::: quiz
Hva skjer hvis `env.reset()` fjernes fra starten av den ytre løkka?
- [ ] Q-tabellen nullstilles ikke mellom episodene, så agenten lærer raskere
- [x] Neste episode starter der forrige sluttet, for eksempel i målet, i stedet for på start
- [ ] `env.step()` krasjer fordi miljøet ikke er initialisert
- [ ] Ingenting, fordi `get_state_id()` alltid gir startstaten
? `reset()` flytter agenten tilbake til start. Den har ingenting med Q-tabellen å gjøre. Uten den fortsetter agenten fra forrige sluttposisjon.
:::

::: quiz
Treningen gir `steps_per_episode[:5] = [100 100 100 100 100]` og `steps_per_episode[-5:] = [11 12 11 11 11]` med `max_steps = 100`. Hva er den beste tolkningen?
- [ ] Agenten har glemt veien mot slutten, fordi antall steg gikk ned
- [ ] Agenten nådde målet i alle de første episodene, men brukte mange steg
- [x] De første episodene nådde ikke målet før steggrensen. Til slutt finner agenten en vei på 11 steg, og 12 skyldes et tilfeldig steg fra en liten ε
- [ ] Agenten har funnet korteste vei, fordi antall steg ikke endrer seg lenger
? 100 er steggrensen, så de første episodene ble avbrutt. At antall steg har stabilisert seg, betyr ikke at veien er den korteste. Neste leksjon viser at korteste vei i dette kartet er 9 steg.
:::

::: oppsummering
- Ytre løkke over episoder: `env.reset()`, `state = env.get_state_id()`, oppdater ε og α.
- Indre løkke over steg: `update_action` → `env.step` → `next_state` → Q-oppdatering → `state = next_state` → `break` ved `terminated`.
- Lagre `step + 1` og siste `reward` per episode for å følge med på treningen.
- Steg nær `max_steps` betyr at agenten ikke nådde målet.
- Faste seeds gir reproduserbare resultater.
:::

--- LEKSJON 11.6 | evaluering | Evaluere og visualisere agenten
Når agenten er trent, vil vi vite tre ting: når den målet, hvilken vei tar den, og hva har den lært? I prosjektet svarer vi med en grådig kjøring (ε = 0), heatmaps av Q-tabellen og et plott av antall steg per episode. Til slutt kjører vi eksperimentene fra diskusjonsspørsmålene.

## Grådig kjøring og policyen
Med ε = 0 velger agenten alltid beste action, slik at vi ser nøyaktig hva den har lært. Vi tegner også policyen som piler. Kodeblokkene i denne leksjonen starter med den samme koden som i 11.5 (miljø og trening), komprimert:

```python
import numpy as np

KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols = len(kart), len(kart[0])
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        self.n_actions = 4
        self.reset()
    def reset(self):
        start = "".join(self.kart).index("S")
        self.pos = (start // self.ncols, start % self.ncols)
        return self.pos
    def step(self, action):
        r, c = self.pos[0] + self.neighbours[action][0], self.pos[1] + self.neighbours[action][1]
        if not (0 <= r < self.nrows and 0 <= c < self.ncols):
            return self.pos, 0, False, None
        if self.kart[r][c] == "W":
            return (self.pos, -1, True, None) if self.dead_allowed else (self.pos, 0, False, None)
        self.pos = (r, c)
        return (self.pos, 10, True, None) if self.kart[r][c] == "G" else (self.pos, 0, False, None)
    def get_state_id(self):
        return self.pos[0] * self.ncols + self.pos[1]

def update_action(env, state, explore_rate, q_table):
    if np.random.random() < explore_rate:
        return np.random.randint(env.n_actions)
    return int(np.argmax(q_table[state]))

def train_q_learning(env, q_table, num_episodes, max_steps, gamma, epsilon_min, min_learning_rate, epsilon=1.0, seed=0):
    np.random.seed(seed)
    steps_per_episode, rewards_per_episode = [], []
    for episode in range(num_episodes):
        env.reset()
        state = env.get_state_id()
        explore_rate = epsilon_min + (epsilon - epsilon_min) * np.exp(-5 * episode / num_episodes)
        learning_rate = min_learning_rate + (1 - min_learning_rate) * np.exp(-5 * episode / num_episodes)
        for step in range(max_steps):
            action = update_action(env, state, explore_rate, q_table)
            _, reward, terminated, _ = env.step(action)
            next_state = env.get_state_id()
            target = reward if terminated else reward + gamma * np.max(q_table[next_state])
            q_table[state, action] += learning_rate * (target - q_table[state, action])
            state = next_state
            if terminated:
                break
        steps_per_episode.append(step + 1)
        rewards_per_episode.append(reward)
    return q_table, np.array(steps_per_episode), np.array(rewards_per_episode)

env = GridEnv(KART)
q_table, steps_per_episode, rewards_per_episode = train_q_learning(
    env, np.zeros((36, 4)), 200, 100, 0.95, 0.01, 0.1)

# ---------- nytt i denne leksjonen ----------
def run_greedy(env, q_table, max_steps):
    """Én episode med explore rate 0. Returnerer total reward og antall steg."""
    env.reset()
    state = env.get_state_id()
    total = 0
    for step in range(max_steps):
        _, reward, terminated, _ = env.step(update_action(env, state, 0, q_table))
        state = env.get_state_id()
        total += reward
        if terminated:
            break
    return total, step + 1

print("Grådig agent (reward, steg):", run_greedy(env, q_table, 50))

piler = np.array(["↑", "←", "↓", "→"])
policy = piler[q_table.argmax(axis=1)].reshape(env.nrows, env.ncols)
for r in range(env.nrows):
    print(" ".join(env.kart[r][c] if env.kart[r][c] in "WG" else policy[r, c] for c in range(env.ncols)))
```

Agenten når målet på **11 steg**, via toppen av kartet. Tell selv på kartet: veien langs bunnen og opp på høyre side er bare **9 steg**. Agenten har lært *en* vei, ikke nødvendigvis den *beste*. Vi kommer tilbake til hvorfor.

::: hvorfor
**Hvorfor ε = 0 når vi viser agenten?** Utforsking trengs bare under læring. Skal vi se hva policyen er, må agenten alltid velge det den mener er best. Med tilfeldige steg ville veien bli forskjellig hver gang, og vi kunne ikke vite om policyen er god.
:::

## Heatmap av Q-verdiene
`q_table.max(axis=1).reshape(nrows, ncols)` gir verdien av hver rute. Prosjektets `show_q_heatmap` tegner den med `imshow` og skriver tallet i hver celle. Vann og mål settes til `np.nan`, slik at de blir blanke:

```python
import numpy as np
import matplotlib.pyplot as plt

KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols = len(kart), len(kart[0])
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        self.n_actions = 4
        self.reset()
    def reset(self):
        start = "".join(self.kart).index("S")
        self.pos = (start // self.ncols, start % self.ncols)
        return self.pos
    def step(self, action):
        r, c = self.pos[0] + self.neighbours[action][0], self.pos[1] + self.neighbours[action][1]
        if not (0 <= r < self.nrows and 0 <= c < self.ncols):
            return self.pos, 0, False, None
        if self.kart[r][c] == "W":
            return (self.pos, -1, True, None) if self.dead_allowed else (self.pos, 0, False, None)
        self.pos = (r, c)
        return (self.pos, 10, True, None) if self.kart[r][c] == "G" else (self.pos, 0, False, None)
    def get_state_id(self):
        return self.pos[0] * self.ncols + self.pos[1]

def update_action(env, state, explore_rate, q_table):
    if np.random.random() < explore_rate:
        return np.random.randint(env.n_actions)
    return int(np.argmax(q_table[state]))

def train_q_learning(env, q_table, num_episodes, max_steps, gamma, epsilon_min, min_learning_rate, epsilon=1.0, seed=0):
    np.random.seed(seed)
    steps_per_episode, rewards_per_episode = [], []
    for episode in range(num_episodes):
        env.reset()
        state = env.get_state_id()
        explore_rate = epsilon_min + (epsilon - epsilon_min) * np.exp(-5 * episode / num_episodes)
        learning_rate = min_learning_rate + (1 - min_learning_rate) * np.exp(-5 * episode / num_episodes)
        for step in range(max_steps):
            action = update_action(env, state, explore_rate, q_table)
            _, reward, terminated, _ = env.step(action)
            next_state = env.get_state_id()
            target = reward if terminated else reward + gamma * np.max(q_table[next_state])
            q_table[state, action] += learning_rate * (target - q_table[state, action])
            state = next_state
            if terminated:
                break
        steps_per_episode.append(step + 1)
        rewards_per_episode.append(reward)
    return q_table, np.array(steps_per_episode), np.array(rewards_per_episode)

env = GridEnv(KART)
q_table, steps_per_episode, rewards_per_episode = train_q_learning(
    env, np.zeros((36, 4)), 200, 100, 0.95, 0.01, 0.1)

# ---------- nytt i denne leksjonen ----------
def show_q_heatmap(env, q_table, action=None):
    if action is None:
        q_grid = q_table.max(axis=1).reshape(env.nrows, env.ncols)      # V(s) = max over actions
    else:
        q_grid = q_table[:, action].reshape(env.nrows, env.ncols).copy()  # Q(s, a) for én action
    blank = np.array([[ch in "WG" for ch in rad] for rad in env.kart])    # vann og mål
    q_grid = np.where(blank, np.nan, q_grid)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(q_grid, cmap="RdYlGn")
    fig.colorbar(im, ax=ax)
    for r in range(env.nrows):
        for c in range(env.ncols):
            if not np.isnan(q_grid[r, c]):
                ax.text(c, r, f"{q_grid[r, c]:.1f}", ha="center", va="center", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Max Q-verdi" if action is None else f"Q-verdi for action {action}")
    plt.show()

show_q_heatmap(env, q_table)

V = q_table.max(axis=1).reshape(env.nrows, env.ncols)
print("Verdi i start (rad 5, kol 0):", round(V[5, 0], 3), "  10 * 0.95**10 =", round(10 * 0.95**10, 3))
```

Tallene bak figuren ligger i `V` i variabelpanelet. Rutene ved siden av målet har verdi 10, og hvert steg lenger unna ganges med γ = 0.95: 9.5, 9.0, 8.6 og så videre. Verdien er $10 \cdot \gamma^{d-1}$ der $d$ er antall steg til målet *langs veien agenten har lært*. Startruta har $10 \cdot 0.95^{10} \approx 5.99$, altså 11 steg. Den grådige policyen klatrer bare opp denne «bakken». Ruter agenten sjelden besøkte, nede til høyre, har lave verdier som ikke har konvergert. Prøv `show_q_heatmap(env, q_table, 0)` for å se Q-verdiene for action «opp».

::: prosjekt Prosjekt 4, «What do the heatmaps show?»
Svaret i prosjektet er nettopp dette: verdien er $V(s) = \max_a Q(s, a)$, den faller med $\gamma$ for hvert steg bort fra målet, og områder agenten aldri besøkte er 0. Heatmapene per action viser hvilken retning som er best i hver rute, og det er det `q_table.argmax(axis=1)` skriver ut som policy.
:::

## Steg per episode og glidende snitt
For Trondheim-kartet skal du plotte antall steg per episode. Kurven er svært ujevn, så vi legger på et **glidende snitt** med `np.convolve`: hver verdi blir snittet av de `w` siste episodene.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.array([10, 2, 6, 4, 8])
print(np.convolve(x, np.ones(3) / 3, mode='valid'))    # (10+2+6)/3, (2+6+4)/3, (6+4+8)/3

# Et oppdiktet treningsforløp: mange steg i starten, rundt 11 til slutt
rng = np.random.default_rng(0)
steps = np.maximum(11, (200 * np.exp(-np.arange(300) / 60) * rng.uniform(0.2, 1.8, 300)).astype(int))

plt.plot(steps, alpha=0.4, label="Steg per episode")
plt.plot(np.convolve(steps, np.ones(20) / 20, mode='valid'), label="Glidende snitt (20 episoder)")
plt.yscale('log')
plt.xlabel("Episode")
plt.ylabel("Steg")
plt.legend()
plt.show()
```

`mode='valid'` tar bare med posisjoner der hele vinduet får plass, så resultatet blir `w - 1` kortere enn originalen. Når det glidende snittet flater ut nær korteste vei, har treningen konvergert for den veien agenten bruker.

## Eksperimentene: γ = 0, γ = 1, ε = 0 og optimistisk initialisering
Prosjektet ber deg trene på nytt med andre innstillinger. Her er alle i én tabell. Optimistisk initialisering betyr at Q-tabellen starter på 1.0 i stedet for 0 (`np.full`):

```python
import numpy as np
import matplotlib.pyplot as plt

KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols = len(kart), len(kart[0])
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        self.n_actions = 4
        self.reset()
    def reset(self):
        start = "".join(self.kart).index("S")
        self.pos = (start // self.ncols, start % self.ncols)
        return self.pos
    def step(self, action):
        r, c = self.pos[0] + self.neighbours[action][0], self.pos[1] + self.neighbours[action][1]
        if not (0 <= r < self.nrows and 0 <= c < self.ncols):
            return self.pos, 0, False, None
        if self.kart[r][c] == "W":
            return (self.pos, -1, True, None) if self.dead_allowed else (self.pos, 0, False, None)
        self.pos = (r, c)
        return (self.pos, 10, True, None) if self.kart[r][c] == "G" else (self.pos, 0, False, None)
    def get_state_id(self):
        return self.pos[0] * self.ncols + self.pos[1]

def update_action(env, state, explore_rate, q_table):
    if np.random.random() < explore_rate:
        return np.random.randint(env.n_actions)
    return int(np.argmax(q_table[state]))

def train_q_learning(env, q_table, num_episodes, max_steps, gamma, epsilon_min, min_learning_rate, epsilon=1.0, seed=0):
    np.random.seed(seed)
    steps_per_episode, rewards_per_episode = [], []
    for episode in range(num_episodes):
        env.reset()
        state = env.get_state_id()
        explore_rate = epsilon_min + (epsilon - epsilon_min) * np.exp(-5 * episode / num_episodes)
        learning_rate = min_learning_rate + (1 - min_learning_rate) * np.exp(-5 * episode / num_episodes)
        for step in range(max_steps):
            action = update_action(env, state, explore_rate, q_table)
            _, reward, terminated, _ = env.step(action)
            next_state = env.get_state_id()
            target = reward if terminated else reward + gamma * np.max(q_table[next_state])
            q_table[state, action] += learning_rate * (target - q_table[state, action])
            state = next_state
            if terminated:
                break
        steps_per_episode.append(step + 1)
        rewards_per_episode.append(reward)
    return q_table, np.array(steps_per_episode), np.array(rewards_per_episode)

def run_greedy(env, q_table, max_steps=50):
    env.reset()
    state, total = env.get_state_id(), 0
    for step in range(max_steps):
        _, reward, terminated, _ = env.step(update_action(env, state, 0, q_table))
        state, total = env.get_state_id(), total + reward
        if terminated:
            break
    return total, step + 1

experiments = {   # navn: (gamma, epsilon, epsilon_min, Q-init, dead_allowed)
    'γ = 0.95 (baseline)': (0.95, 1.0, 0.01, 0.0, False),
    'γ = 0':               (0.0,  1.0, 0.01, 0.0, False),
    'γ = 1':               (1.0,  1.0, 0.01, 0.0, False),
    'ε = 0 fra start':     (0.95, 0.0, 0.0,  0.0, False),
    'Q-init 1.0':          (0.95, 1.0, 0.01, 1.0, False),
    'dead_allowed':        (0.95, 1.0, 0.01, 0.0, True),
    'dead_allowed, init 1': (0.95, 1.0, 0.01, 1.0, True),
}
steps_hist = {}
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for i, (name, (gamma, eps, eps_min, q_init, dead)) in enumerate(experiments.items()):
    env = GridEnv(KART, dead_allowed=dead)
    q = np.full((36, 4), q_init)
    q, spe, rpe = train_q_learning(env, q, 200, 100, gamma, eps_min, 0.1, epsilon=eps)
    steps_hist[name] = spe
    reward, steps = run_greedy(env, q)
    print(f"{name:21s} mål i {np.sum(rpe == 10):3d}/200 episoder, vann {np.sum(rpe == -1):3d} | "
          f"grådig: reward {reward:2d} på {steps:2d} steg | Q-verdier endret: {(q != q_init).sum()}")
    if i < 4:
        axes[i].imshow(q.max(axis=1).reshape(6, 6), cmap="RdYlGn")
        axes[i].set_title(name)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
plt.show()

for name in ['γ = 0.95 (baseline)', 'Q-init 1.0', 'dead_allowed, init 1']:
    plt.plot(np.convolve(steps_hist[name], np.ones(20) / 20, mode='valid'), label=name)
plt.yscale('log')
plt.xlabel('Episode')
plt.ylabel('Steg (glidende snitt, 20 episoder)')
plt.legend()
plt.show()
```

Slik leser du tabellen (og svarer på prosjektets spørsmål):
- **γ = 0:** target er bare rewarden nå. Bare de to actionene som går rett inn i målet får verdi. Resten er 0, så det finnes ingen «bakke» å klatre. Den grådige agenten går opp (action 0 ved likhet) og når aldri målet.
- **γ = 1:** ingen diskontering. Alle veier som til slutt når målet er verdt 10, uansett lengde, så agenten har ingen grunn til å foretrekke den korteste. Heatmapen er nesten flat, og den grådige agenten bruker 13 steg.
- **ε = 0 fra start:** alle verdier er 0, `argmax` gir alltid «opp», og target blir 0 + γ · 0 = 0. Tabellen endres aldri, og agenten når aldri målet.
- **Q-init 1.0 (optimistisk):** hver action agenten ikke har prøvd ser bedre ut (1.0) enn de den har prøvd og ikke fått noe ut av. Da prøver agenten alt systematisk, og finner den **korteste veien på 9 steg**. Akkurat slik oppdager agenten bussen på Trondheim-kartet i prosjektet.
- **dead_allowed:** med nuller i tabellen når agenten aldri målet. En tilfeldig vandring havner i vannet først, og det eneste agenten lærer er «vann er dårlig». Med optimistisk start virker det. I plottet av steg per episode stiger kurven for `dead_allowed` først (episodene slutter tidlig i vannet), og faller så mot 9.

::: prosjekt Prosjekt 4, problem 2
Med Q-tabellen initialisert til 0 lærer agenten den direkte veien på 25 steg og bruker aldri bussen. Med `Q_INIT = 1.0` og `np.full((num_states, num_actions), Q_INIT)` finner den bussveien på 21 steg. Med `start_random=True` starter hver episode på en tilfeldig rute, så agenten må lære verdier for hele kartet, ikke bare én vei.
:::

::: oppgave Glidende snitt
Skriv `moving_average(x, w)` som returnerer det glidende snittet av arrayet `x` med vindusstørrelse `w`. Bruk `np.convolve` med `np.ones(w) / w` og `mode='valid'`.
```start
import numpy as np

def moving_average(x, w):
    return ...
```
```losning
import numpy as np

def moving_average(x, w):
    return np.convolve(x, np.ones(w) / w, mode='valid')
```
```test
import numpy as np
_r = moving_average(np.array([10, 2, 6, 4, 8]), 3)
assert len(_r) == 3, f"Med mode='valid' skal 5 verdier og w=3 gi 3 snitt, ikke {len(_r)}."
assert np.allclose(_r, [6, 4, 6]), f"Forventet [6, 4, 6], fikk {_r}. Husk å dele på w."
_x = np.arange(100.0)
assert np.allclose(moving_average(_x, 10), np.arange(4.5, 95, 1.0)), "Snittet av 0..9 er 4.5, av 1..10 er 5.5, osv."
```
:::

::: oppgave Forventede verdier i heatmapen
Med reward 10 i mål og ellers 0 får en rute som ligger `d` steg fra målet verdien $10 \cdot \gamma^{d-1}$ når Q-tabellen har konvergert. Lag `verdi(d, gamma)` som regner ut dette, og bruk den til å lage arrayet `forventet` med verdiene for `d = 1, 2, ..., 9` når γ = 0.95 (prosjektets enkle kart, der korteste vei er 9 steg).
```start
import numpy as np

def verdi(d, gamma):
    return ...

forventet = ...
```
```losning
import numpy as np

def verdi(d, gamma):
    return 10 * gamma ** (d - 1)

forventet = np.array([verdi(d, 0.95) for d in range(1, 10)])
```
```test
import numpy as np
assert np.isclose(verdi(1, 0.95), 10), "Ett steg fra målet (d = 1) skal verdien være 10."
assert np.isclose(verdi(3, 0.9), 8.1), "d = 3 og γ = 0.9 skal gi 10 * 0.9**2 = 8.1."
assert len(forventet) == 9, "forventet skal ha 9 verdier, for d = 1 til 9."
assert np.isclose(forventet[-1], 10 * 0.95**8), "Siste verdi (d = 9) skal være 10 * 0.95**8 ≈ 6.63."
```
:::

::: quiz
En agent er trent med γ = 0.9 til Q-verdiene har konvergert. Reward er 10 i mål og 0 ellers. Hva viser heatmapen (`q_table.max(axis=1)`) for en rute som ligger 3 steg fra målet?
- [ ] 7.29
- [x] 8.1
- [ ] 10
- [ ] 2.7
? $10 \cdot 0.9^{3-1} = 8.1$. Ruta ett steg unna har 10 (rewarden kommer i neste steg), to steg unna 9, tre steg unna 8.1. 7.29 er $10 \cdot 0.9^3$, som er ett steg for mye.
:::

::: quiz
Agenten trenes med en Q-tabell som starter på 0 og explore rate 0 hele tiden. Hva skjer?
- [ ] Agenten finner korteste vei raskere, fordi den aldri kaster bort steg på tilfeldige actions
- [ ] Agenten går tilfeldig rundt, fordi alle Q-verdiene er like
- [x] `argmax` velger alltid action 0, target blir hele tiden 0, og tabellen endres aldri
- [ ] Koden krasjer fordi `np.argmax` ikke kan velge mellom like verdier
? Ved likhet returnerer `np.argmax` første indeks. Agenten går opp, møter kanten og blir stående. Reward 0 + γ · 0 = 0, så ingen Q-verdi endres. I prosjektet nådde agenten målet i 0 av 500 episoder.
:::

::: quiz
Hvorfor hjelper det å starte Q-tabellen på 1.0 i stedet for 0 (optimistisk initialisering)?
- [ ] Verdiene konvergerer til 1.0 i stedet for 10, så treningen går raskere
- [x] Uprøvde actions ser bedre ut enn de agenten har prøvd uten å få reward, så agenten prøver dem systematisk
- [ ] Det gjør γ-diskonteringen unødvendig
- [ ] Det gjør at agenten ikke lenger trenger å nå målet for å få høy reward
? Hver gang agenten prøver en action uten resultat, synker Q-verdien under 1.0. Actions den ikke har prøvd beholder 1.0 og blir valgt av `argmax`. Det er en form for utforsking som virker selv med lav ε.
:::

::: oppsummering
- Vis den trente agenten med ε = 0 (`update_action(env, state, 0, q_table)`): da ser vi selve policyen.
- Heatmap: `q_table.max(axis=1).reshape(nrows, ncols)`. Verdien er $10 \cdot \gamma^{d-1}$, og faller med γ for hvert steg bort fra målet.
- `q_table[:, a].reshape(...)` viser én action. `q_table.argmax(axis=1)` er policyen.
- Glidende snitt: `np.convolve(x, np.ones(w) / w, mode='valid')`.
- γ = 0: ingen bakke å klatre. γ = 1: ingen preferanse for korteste vei. ε = 0 med nuller: ingenting læres.
- Optimistisk initialisering (`np.full(..., 1.0)`) gir systematisk utforsking og finner bedre veier.
:::

--- LEKSJON 11.7 | gymnasium | Gymnasium og FrozenLake
**Gymnasium** (etterfølgeren til OpenAI Gym) er standardbiblioteket for RL-miljøer. Prosjektets `TrondheimEnv` arver fra `gym.Env`, og notatene bruker det ferdige miljøet **FrozenLake**: en frossen innsjø på 4 × 4 ruter der agenten skal fra start (S) til julegaven (G) uten å gå gjennom isen i hullene (H).

Gymnasium finnes ikke i nettleseren, så kodeblokkene med Gymnasium er statiske. Kjør dem i Jupyter med `pip install gymnasium`.

```python statisk
import gymnasium as gym

env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False)
# legg til render_mode='human' for å se brettet i et vindu
print(env.action_space)
print(env.observation_space)
print(env.unwrapped.desc.astype(str))

state, info = env.reset(seed=0)
print("Start:", state, info)

next_state, reward, terminated, truncated, info = env.step(2)   # 2 = høyre i FrozenLake
print(next_state, reward, terminated, truncated, info)
next_state, reward, terminated, truncated, info = env.step(1)   # 1 = ned: rett i et hull
print(next_state, reward, terminated, truncated, info)
```
```output
Discrete(4)
Discrete(16)
[['S' 'F' 'F' 'F']
 ['F' 'H' 'F' 'H']
 ['F' 'F' 'F' 'H']
 ['H' 'F' 'F' 'G']]
Start: 0 {'prob': 1}
1 0 False False {'prob': 1.0}
5 0 True False {'prob': 1.0}
```

Det er mye å legge merke til her:
- `Discrete(4)`: fire actions, men i **en annen rekkefølge** enn i prosjektet: 0 venstre, 1 ned, 2 høyre, 3 opp.
- `Discrete(16)`: staten er ett tall 0–15, rutene nummerert radvis (akkurat som `get_state_id()`). `env.observation_space.n` er 16.
- `env.reset()` returnerer en **tuppel** `(state, info)`.
- `env.step()` returnerer **fem** verdier: `next_state, reward, terminated, truncated, info`. `terminated` betyr at episoden er over (gave eller hull). `truncated` betyr at den ble avbrutt utenfra, her etter 100 steg.
- Reward er **1** for gaven og 0 for alt annet, også for hullene.
- `is_slippery=False` gjør miljøet deterministisk. Med glatt is går agenten ikke alltid dit den vil.

::: advarsel
Prosjektets `TrondheimEnv` bruker det gamle Gym-grensesnittet: `reset()` gir bare observasjonen, og `step()` gir fire verdier (`obs, reward, terminated, info`). Gymnasium gir `(state, info)` og fem verdier. Skriver du `state = env.reset()` med Gymnasium, blir `state` en tuppel, og `q_table[state]` gir feil.
:::

## Tabell-Q-learning på FrozenLake
Koden fra leksjon 11.5 virker nesten uendret. Det eneste nye er `state, _ = env.reset()`, de fem returverdiene, og at episoden stopper når `terminated or truncated`. `env.action_space.sample()` trekker en tilfeldig action:

```python statisk fortsett
import numpy as np
np.random.seed(0)
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table = np.ones((num_states, num_actions))     # optimistisk: reward i mål er 1
gamma, num_episodes = 0.95, 500

def update_action(env, state, explore_rate, q_table):
    if np.random.random() < explore_rate:
        return env.action_space.sample()
    return int(np.argmax(q_table[state]))

env.action_space.seed(0)
successes = 0
for episode in range(num_episodes):
    state, _ = env.reset()
    explore_rate = 0.01 + 0.99 * np.exp(-5 * episode / num_episodes)
    learning_rate = 0.1 + 0.9 * np.exp(-5 * episode / num_episodes)
    while True:
        action = update_action(env, state, explore_rate, q_table)
        next_state, reward, terminated, truncated, _ = env.step(action)
        target = reward if terminated else reward + gamma * np.max(q_table[next_state])
        q_table[state, action] += learning_rate * (target - q_table[state, action])
        state = next_state
        if terminated or truncated:
            break
    successes += reward
print("Nådde gaven i", int(successes), "av", num_episodes, "episoder")

state, _ = env.reset()
path = [int(state)]
for step in range(20):
    state, reward, terminated, truncated, _ = env.step(int(np.argmax(q_table[state])))
    path.append(int(state))
    if terminated or truncated:
        break
print("Grådig vei:", path, "reward", reward)
print(np.round(q_table.max(axis=1).reshape(4, 4), 3))
```
```output
Nådde gaven i 382 av 500 episoder
Grådig vei: [0, 4, 8, 9, 13, 14, 15] reward 1
[[0.774 0.815 0.857 0.817]
 [0.815 1.    0.903 1.   ]
 [0.857 0.903 0.95  1.   ]
 [1.    0.95  1.    1.   ]]
```

Agenten går ned, ned, høyre, ned, høyre, høyre: 6 steg, og startverdien er $0.95^5 \approx 0.774$. Hullene (state 5, 7, 11, 12) og målet (15) beholder startverdien 1.0, fordi de er terminale og aldri oppdateres.

Med `np.zeros` i stedet for `np.ones` nådde agenten gaven i bare **1 av 500** episoder. Det er samme problem som `dead_allowed` i leksjon 11.6: en tilfeldig vandring faller nesten alltid i et hull før den finner gaven, og ved likhet velger `argmax` action 0 (venstre), som holder agenten fast i hjørnet.

## Ditt eget miljø som gym.Env
Et Gymnasium-miljø er en klasse som arver fra `gym.Env`, definerer `action_space` og `observation_space`, og har `reset` og `step`. Slik er `TrondheimEnv` bygget. Her er rutenettet vårt med det nye grensesnittet:

```python statisk
import gymnasium as gym
from gymnasium.spaces import Discrete

class GridWorld(gym.Env):
    """Vårt rutenett som et ekte Gymnasium-miljø (samme idé som TrondheimEnv)."""
    def __init__(self, kart):
        self.kart = kart
        self.nrows, self.ncols = len(kart), len(kart[0])
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]       # opp, venstre, ned, høyre
        self.action_space = Discrete(4)
        self.observation_space = Discrete(self.nrows * self.ncols)  # state-id 0 .. 35

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        start = "".join(self.kart).index("S")
        self.pos = (start // self.ncols, start % self.ncols)
        return self.pos[0] * self.ncols + self.pos[1], {}

    def step(self, action):
        r = self.pos[0] + self.neighbours[action][0]
        c = self.pos[1] + self.neighbours[action][1]
        if 0 <= r < self.nrows and 0 <= c < self.ncols and self.kart[r][c] != "W":
            self.pos = (r, c)
        terminated = self.kart[self.pos[0]][self.pos[1]] == "G"
        reward = 10 if terminated else 0
        obs = self.pos[0] * self.ncols + self.pos[1]
        return obs, reward, terminated, False, {}

env = GridWorld(["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."])
print(env.action_space, env.observation_space)
state, info = env.reset()
print("start:", state)
for action in [3, 3, 3, 3, 3, 0, 0, 0, 1]:
    state, reward, terminated, truncated, info = env.step(action)
print("slutt:", state, reward, terminated)
```
```output
Discrete(4) Discrete(36)
start: 30
slutt: 16 10 True
```

`super().reset(seed=seed)` lar Gymnasium sette opp tilfeldighetsgeneratoren. Notatenes tre-på-rad-miljø har den samme oppbygningen: `__init__` lager et tomt brett, `reset` setter det tilbake, `step` legger en brikke, sjekker vinner og returnerer ny state, reward og `done`.

::: prosjekt Prosjekt 4
`TrondheimEnv.__init__` setter `self.action_space = Discrete(4)` og `self.observation_space = MultiDiscrete(...)` med samme form som kartet (12 × 12). Derfor regner prosjektet antall states som `int(np.prod(env.observation_space.shape))`, mens FrozenLake har `env.observation_space.n`.
:::

## FrozenLake uten Gymnasium
Kartet er bare fire strenger, så vi kan finne state-id-ene til hullene og målet med det vi kan fra før:

```python
desc = ["SFFF",
        "FHFH",
        "FFFH",
        "HFFG"]
alle = "".join(desc)                     # 'SFFFFHFHFFFHHFFG', indeks = state-id
hull = [i for i, ch in enumerate(alle) if ch == "H"]
print("Hull:", hull, " mål:", alle.index("G"))

# Prosjektets nummerering: 0 opp, 1 venstre, 2 ned, 3 høyre
# FrozenLake:              0 venstre, 1 ned, 2 høyre, 3 opp
navn_prosjekt = ["opp", "venstre", "ned", "høyre"]
navn_frozenlake = ["venstre", "ned", "høyre", "opp"]
print([f"{a}: {navn_prosjekt[a]} / {navn_frozenlake[a]}" for a in range(4)])
```

::: oppgave Oversett actions
En plan er skrevet med prosjektets nummerering (0 opp, 1 venstre, 2 ned, 3 høyre), men skal kjøres i FrozenLake (0 venstre, 1 ned, 2 høyre, 3 opp). Lag dictionaryen `til_frozenlake` som oversetter hver prosjekt-action til samme retning i FrozenLake, og bruk den til å lage listen `plan_fl` fra `plan`.
```start
plan = [2, 2, 3, 2, 3, 3]     # ned, ned, høyre, ned, høyre, høyre

til_frozenlake = ...
plan_fl = ...
```
```losning
plan = [2, 2, 3, 2, 3, 3]

til_frozenlake = {0: 3, 1: 0, 2: 1, 3: 2}
plan_fl = [til_frozenlake[a] for a in plan]
```
```test
assert isinstance(til_frozenlake, dict), "til_frozenlake skal være en dictionary."
assert til_frozenlake.get(0) == 3, "Prosjektets 0 (opp) er 3 i FrozenLake."
assert til_frozenlake.get(1) == 0, "Prosjektets 1 (venstre) er 0 i FrozenLake."
assert til_frozenlake.get(2) == 1 and til_frozenlake.get(3) == 2, "Ned er 1 og høyre er 2 i FrozenLake."
assert list(plan_fl) == [1, 1, 2, 1, 2, 2], f"plan_fl skal være [1, 1, 2, 1, 2, 2], fikk {plan_fl}."
```
:::

::: quiz
Hva er galt med denne koden for Gymnasium?
```kode
state = env.reset()
action = int(np.argmax(q_table[state]))
next_state, reward, done, info = env.step(action)
```
- [ ] Ingenting, dette er standard Gymnasium-kode
- [ ] `np.argmax` kan ikke brukes på en rad i Q-tabellen
- [x] `reset()` returnerer `(state, info)` og `step()` returnerer fem verdier, ikke fire
- [ ] `env.step()` tar en state, ikke en action, som argument
? I Gymnasium skal det være `state, info = env.reset()` og `next_state, reward, terminated, truncated, info = env.step(action)`. Fire returverdier er det gamle Gym-grensesnittet som TrondheimEnv bruker.
:::

::: quiz
I FrozenLake (`is_slippery=False`) står agenten i state 6 (rad 1, kolonne 2) og tar action 1. Hvilken state havner den i?
- [ ] 5, fordi action 1 er venstre
- [x] 10, fordi action 1 er ned
- [ ] 7, fordi action 1 er høyre
- [ ] 2, fordi action 1 er opp
? I FrozenLake er 0 venstre, 1 ned, 2 høyre, 3 opp. Ned fra rad 1, kolonne 2 gir rad 2, kolonne 2: 2 · 4 + 2 = 10. Svaret 5 (venstre) er det du får med prosjektets nummerering.
:::

::: oppsummering
- `env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False)`. `env.action_space.n` = 4, `env.observation_space.n` = 16.
- Gymnasium: `state, info = env.reset()` og `next_state, reward, terminated, truncated, info = env.step(action)`.
- Stopp episoden når `terminated or truncated`. Bare `terminated` betyr at det ikke finnes noen framtid (target = reward).
- FrozenLake: actions 0 venstre, 1 ned, 2 høyre, 3 opp. Reward 1 for gaven, ellers 0.
- `env.action_space.sample()` gir en tilfeldig action.
- Eget miljø: arv fra `gym.Env`, sett `action_space`/`observation_space`, skriv `reset` og `step`.
:::

--- LEKSJON 11.8 | dqn | Deep Q-learning (DQN)
En Q-tabell virker når det finnes få states. Har miljøet millioner av states (eller kontinuerlige), blir tabellen umulig. I **Deep Q-learning** erstatter vi tabellen med et nevralt nettverk: det tar inn en state og gir ut **én Q-verdi per action**. Et slikt nettverk kalles et deep Q-network (**DQN**).

Notatene sier at på eksamen er bare konseptuelle spørsmål om DQN aktuelle, men eksamen 2025 hadde kodelesing om replay bufferen. Her går vi gjennom notatenes kode bit for bit. PyTorch-koden er statisk. Etterpå regner vi de samme stegene med numpy i nettleseren.

## Input og output: one-hot og MLP
Nettverket for FrozenLake trenger 4 output-noder (én per action). Staten er et tall 0–15, men i stedet for én input-node med verdien 5 bruker vi **one-hot-koding**: 16 input-noder der bare nummer 5 er 1. Da behandles hver rute som en egen kategori, og ikke som et tall der 15 er «større» enn 5.

```python statisk
import torch
import torch.nn as nn
import torch.nn.functional as F

def preprocess_state(state, num_states):
    # num_states = env.observation_space.n
    onehot_vector = F.one_hot(torch.tensor(state, dtype=torch.long), num_states)
    return onehot_vector

s = preprocess_state(5, 16)
print(s)
print(s.shape, s.dtype)

class MLP(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self._model = nn.Sequential(
            nn.Linear(input, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output),
        )

    def forward(self, x):
        return self._model(x.float())     # one_hot gir heltall, Linear vil ha float

torch.manual_seed(0)
q_network = MLP(16, 4)
print(q_network(s))                                        # 4 Q-verdier for state 5
batch = torch.stack([preprocess_state(i, 16) for i in [0, 5, 15]])
print(batch.shape, "->", q_network(batch).shape)          # (3, 16) -> (3, 4)
```
```output
tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
torch.Size([16]) torch.int64
tensor([ 0.1444, -0.1480,  0.1322, -0.0603], grad_fn=<UnsafeViewBackward0>)
torch.Size([3, 16]) -> torch.Size([3, 4])
```

Notatene sier at `preprocess_state` gir shape `[1, 16]`, men for ett enkelt tall er shapen `[16]`, som utskriften viser. Nettverket tåler begge deler. `x.float()` i `forward` trengs fordi `F.one_hot` lager heltall (`int64`).

## Hvorfor ikke bare trene nettverket direkte?
Med en tabell endrer én oppdatering bare én verdi. Et nettverk deler vektene mellom alle states, så å justere Q for én state endrer også Q-verdiene til mange andre, også de vi bruker som target. Nettverket jager et **bevegelig mål**, og treningen blir ustabil. Notatene beskriver også **catastrophic forgetting**: agenten lærer oppgaven, og så kollapser rewarden plutselig fordi nettverket overtilpasser seg de siste erfaringene. To triks løser dette:

| Triks | Hva | Løser |
|---|---|---|
| **Replay buffer** | lagre overganger `(state, action, reward, next_state, done)` og tren på tilfeldige batcher | påfølgende steg er sterkt korrelerte. Tilfeldig trekking bryter korrelasjonen, og hver erfaring kan brukes flere ganger |
| **Target network** | en kopi av Q-nettverket som bare oppdateres med jevne mellomrom, og som brukes til $\max_a Q(s', a)$ i target | gir stabile treningsmål |

## Replay bufferen
Bufferen er en liste med overganger. `random.sample(buffer, k=batch_size)` trekker en tilfeldig batch **uten tilbakelegging**. Det var et eget eksamensspørsmål i 2025.

```python statisk fortsett
import random

class ExperienceReplayBuffer:
    def __init__(self, replay_size):
        self.replay_size = replay_size
        self.buffer = []                      # en vanlig liste med overganger

    @property
    def size(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.replay_size:
            self.buffer.pop(0)                # kast den eldste når bufferen er full

    def sample(self, batch_size):
        batch = random.sample(self.buffer, k=batch_size)       # tilfeldig, uten tilbakelegging
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.stack(states),                                   # (B, 16)
                torch.tensor(actions).unsqueeze(1),                    # (B, 1)
                torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),  # (B, 1)
                torch.stack(next_states),                              # (B, 16)
                torch.tensor(dones).unsqueeze(1))                      # (B, 1), bool

random.seed(0)
buf = ExperienceReplayBuffer(replay_size=5)
for t in range(7):                            # 7 overganger, men plass til bare 5
    buf.add(preprocess_state(t, 16), t % 4, float(t == 6), preprocess_state(t + 1, 16), t == 6)
print("størrelse:", buf.size, " actions i bufferen:", [o[1] for o in buf.buffer])
states, actions, rewards, next_states, dones = buf.sample(3)
print(states.shape, actions.shape, rewards.shape, dones.shape)
print(actions.flatten(), rewards.flatten(), dones.flatten())
```
```output
størrelse: 5  actions i bufferen: [2, 3, 0, 1, 2]
torch.Size([3, 16]) torch.Size([3, 1]) torch.Size([3, 1]) torch.Size([3, 1])
tensor([1, 2, 2]) tensor([0., 1., 0.]) tensor([False,  True, False])
```

`zip(*batch)` «pakker opp» listen med tupler: alle states havner i én tuppel, alle actions i en annen, og så videre. `@property` gjør at `buf.size` kan brukes uten parenteser, slik notatenes RL-loop gjør. Notatenes buffer heter også `ExperienceReplayBuffer`, men koden for den står ikke i notatene. Dette er én måte å skrive den på.

## Agenten og learn()
`DQNAgent` samler alt: to nettverk, bufferen, ε-greedy og læringen. `learn()` er det viktigste:

```python statisk fortsett
import gymnasium as gym

class DQNAgent:
    def __init__(self, env, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.99, discount=0.95,
                 batch_size=32, target_update_frequency=10, learning_rate=0.001, replay_size=1000):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.criterion = torch.nn.functional.mse_loss
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.replay_buffer = ExperienceReplayBuffer(replay_size)
        self.q_network = self.create_model()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.target_network = self.create_model()
        self.target_network.load_state_dict(self.q_network.state_dict())   # kopi av vektene

    def create_model(self):
        return MLP(self.observation_space.n, self.action_space.n)

    def select_action(self, state):                       # ε-greedy
        if random.random() < self.epsilon:
            return self.action_space.sample()
        with torch.no_grad():
            return int(self.q_network(state).argmax())

    def update_epsilon(self):                             # multiplikativ decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def learn(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            next_target_q_value = self.target_network(next_states).max(dim=1, keepdim=True).values
            next_target_q_value = torch.where(dones, torch.zeros_like(next_target_q_value), next_target_q_value)
            y_train = rewards + (self.discount * next_target_q_value)
        predicted_q = self.q_network(states)                   # (B, 4)
        y_pred = torch.gather(predicted_q, 1, actions)         # (B, 1): Q for valgt action
        self.optimizer.zero_grad()
        loss = self.criterion(y_train, y_pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()

# Se på tensorene i learn() for en liten batch med tall vi kan regne på
predicted_q = torch.tensor([[1.2, 0.5, 2.0, -0.3],
                            [0.1, 0.4, 0.3,  0.2]])
actions = torch.tensor([[1], [2]])
print(torch.gather(predicted_q, 1, actions))           # [[0.5], [0.3]]

next_q = torch.tensor([[0.3, 1.5, 0.2, 0.0],
                       [0.9, 0.1, 0.4, 0.8]]).max(dim=1, keepdim=True).values
dones = torch.tensor([[False], [True]])
print(next_q, torch.where(dones, torch.zeros_like(next_q), next_q))
```
```output
tensor([[0.5000],
        [0.3000]])
tensor([[1.5000],
        [0.9000]]) tensor([[1.5000],
        [0.0000]])
```

Linje for linje i `learn()`:
1. **Trekk en batch** fra replay bufferen.
2. **`torch.no_grad()`**: target skal være en fast fasit, så ingen gradienter skal gå gjennom target-nettverket.
3. **`.max(dim=1, keepdim=True).values`**: beste Q-verdi i neste state for hver rad, med shape `(B, 1)` i stedet for `(B,)`.
4. **`torch.where(dones, 0, ...)`**: etter en terminal state er det ingen framtid. Det er tabellversjonens `reward if terminated`.
5. **`y_train = rewards + discount * next`**: target, akkurat som i Q-tabellen.
6. **`torch.gather(predicted_q, 1, actions)`**: nettverket gir 4 Q-verdier per state, men vi har bare target for actionen som faktisk ble tatt. `gather` plukker ut den ene kolonnen per rad.
7. **MSE-loss**, `zero_grad`, `backward`, `step`: vanlig treningssteg som i Del 9.

Merk at læringsraten α fra tabellversjonen er borte. Den rollen har nå optimizerens `lr`.

## RL-loopen med DQN
Til slutt notatenes RL-loop. Den er lik tabellversjonen, men hver overgang legges i bufferen, `learn()` kalles når bufferen har mer enn `batch_size` elementer, og target-nettverket kopieres hvert tiende steg. Vi trener i 400 episoder (notatene bruker 3000) og decay 0.99 per episode:

```python statisk fortsett
random.seed(0)
torch.manual_seed(0)
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
env.action_space.seed(0)
agent = DQNAgent(env)

max_steps = 50
episodes = 400
total_steps = 0
episode_rewards = []
for ep in range(episodes):
    state, _ = env.reset()
    state = preprocess_state(state, env.observation_space.n)
    for step in range(max_steps):
        total_steps += 1
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = preprocess_state(next_state, env.observation_space.n)
        agent.replay_buffer.add(state, action, reward, next_state, done)
        if agent.replay_buffer.size > agent.batch_size:
            agent.learn()
        if total_steps % agent.target_update_frequency == 0:
            agent.update_target_network()
        state = next_state
        if done or truncated:
            break
    episode_rewards.append(reward)
    agent.update_epsilon()
    if (ep + 1) % 100 == 0:
        print(f"Episode {ep + 1}: epsilon {agent.epsilon:.3f}, "
              f"gaven nådd i {sum(episode_rewards[-100:]):.0f} av de siste 100")

# Grådig kjøring og Q-verdiene nettverket har lært
agent.epsilon = 0
state, _ = env.reset()
path = [state]
for step in range(20):
    state, reward, done, truncated, _ = env.step(agent.select_action(preprocess_state(state, 16)))
    path.append(state)
    if done or truncated:
        break
print("Grådig vei:", path, "reward", reward)
with torch.no_grad():
    all_states = F.one_hot(torch.arange(16), 16)
    print(agent.q_network(all_states).max(dim=1).values.reshape(4, 4).round(decimals=2))
```
```output
Episode 100: epsilon 0.366, gaven nådd i 1 av de siste 100
Episode 200: epsilon 0.134, gaven nådd i 64 av de siste 100
Episode 300: epsilon 0.049, gaven nådd i 91 av de siste 100
Episode 400: epsilon 0.018, gaven nådd i 97 av de siste 100
Grådig vei: [0, 4, 8, 9, 13, 14, 15] reward 1
tensor([[0.7900, 0.8500, 0.8800, 0.8000],
        [0.8300, 0.8700, 0.9200, 0.8800],
        [0.9000, 0.9100, 0.9700, 0.7100],
        [0.8300, 0.9400, 1.0000, 0.9300]])
```

Agenten finner den samme 6-stegsveien som tabellversjonen, og startverdien 0.79 er nær $0.95^5 \approx 0.77$. Verdiene for hullene og målet betyr ingenting: de er terminale og blir aldri brukt i et target.

::: eksamen
Eksamen 2025: «Du har en replay-buffer lagret som en liste med transisjoner. Hvilken Python-funksjon brukes for å trekke en tilfeldig minibatch?» Svaret er `random.sample(buffer, k=32)`. Samme eksamen spurte hva target network og replay buffer løser, og hvilken utfordring vektdeling i nettverket gir (én oppdatering endrer mange Q-verdier).
:::

## De samme stegene i numpy
Nå regner vi `learn()` med numpy, slik at du kan kjøre det selv. `np.eye(16)[state]` er one-hot, `np.take_along_axis` er numpy sin `torch.gather`, og `np.where` virker som `torch.where`:

```python
import numpy as np

print("One-hot av state 5:", np.eye(16)[5].astype(int))
print("Batch av states [0, 5, 15]:", np.eye(16)[[0, 5, 15]].shape)

gamma = 0.95
predicted_q = np.array([[1.2, 0.5, 2.0, -0.3],      # Q-nettverket for de 3 states i batchen
                        [0.1, 0.4, 0.3,  0.2],
                        [0.7, 0.9, 0.0,  0.5]])
actions = np.array([[1], [2], [0]])                  # actionene som ble tatt, shape (B, 1)
rewards = np.array([[0.0], [1.0], [0.0]])
dones = np.array([[False], [True], [False]])
target_q_next = np.array([[0.3, 1.5, 0.2, 0.0],     # target-nettverket for neste states
                          [0.9, 0.1, 0.4, 0.8],
                          [0.6, 0.2, 0.8, 0.1]])

next_target_q_value = target_q_next.max(axis=1, keepdims=True)     # (3, 1)
next_target_q_value = np.where(dones, 0.0, next_target_q_value)    # ingen framtid etter done
y_train = rewards + gamma * next_target_q_value
y_pred = np.take_along_axis(predicted_q, actions, axis=1)          # som torch.gather
loss = np.mean((y_train - y_pred) ** 2)

print("y_train:", y_train.ravel())
print("y_pred: ", y_pred.ravel())
print("MSE-loss:", round(loss, 4))
```

Rad 2 er terminal, så target er bare rewarden 1.0, selv om target-nettverket gir 0.9 for neste state.

## Replay bufferen som liste
Bufferen er bare en Python-liste med tupler, så den kan vi se på i nettleseren uten PyTorch. Her er det samme som i `ExperienceReplayBuffer` over, med state-id-er i stedet for one-hot-tensorer:

```python
import random

random.seed(0)
replay_size = 5
buffer = []
for t in range(7):                       # 7 overganger, men plass til bare 5
    buffer.append((t, t % 4, float(t == 6), t + 1, t == 6))
    if len(buffer) > replay_size:
        buffer.pop(0)                    # kast den eldste
print("buffer, én rad per overgang (state, action, reward, next_state, done):")
display(buffer)

batch = random.sample(buffer, k=3)       # tilfeldig, uten tilbakelegging
print("batch fra random.sample(buffer, k=3):")
display(batch)
```

Overgangene for t = 0 og t = 1 er kastet ut, så bufferen starter på state 2. Batchen er tre tilfeldige rader fra bufferen, i tilfeldig rekkefølge og uten duplikater.

## Hvor mange ganger kalles learn()?
Bufferen får ett element per steg, og `learn()` kalles bare når `size > batch_size`. Det er lett å telle feil, så la Python telle:

```python
batch_size, target_update_frequency = 32, 10
size, total_steps, n_learn, n_target = 0, 0, 0, 0
for step in range(100):
    total_steps += 1
    size += 1                                  # replay_buffer.add(...)
    if size > batch_size:
        n_learn += 1                           # agent.learn()
    if total_steps % target_update_frequency == 0:
        n_target += 1                          # agent.update_target_network()
print("learn:", n_learn, " target-oppdateringer:", n_target)
```

`learn()` kalles for steg 33 til 100, altså 68 ganger. Target-nettverket oppdateres i steg 10, 20, ..., 100, altså 10 ganger. Sjekk alltid om oppdateringen står inne i eller utenfor `if`-en: står den inne i `if size > batch_size`, blir det bare 7 (steg 40 til 100).

::: oppgave One-hot-koding
Lag `one_hot(states, num_states)` som tar en liste med state-id-er og returnerer et numpy-array med shape `(len(states), num_states)`, der rad `i` har 1 i kolonne `states[i]` og 0 ellers.
```start
import numpy as np

def one_hot(states, num_states):
    return ...
```
```losning
import numpy as np

def one_hot(states, num_states):
    return np.eye(num_states)[states]
```
```test
import numpy as np
_r = np.asarray(one_hot([0, 5, 15], 16))
assert _r.shape == (3, 16), f"Shape skal være (3, 16), ikke {_r.shape}."
assert _r[1, 5] == 1 and _r[1].sum() == 1, "Rad 1 skal ha 1 bare i kolonne 5."
assert _r[2, 15] == 1 and _r[0, 0] == 1, "Rad 0 skal ha 1 i kolonne 0 og rad 2 i kolonne 15."
assert _r.sum() == 3, "Det skal være nøyaktig én 1 per rad."
```
:::

::: oppgave Trekk en batch fra replay bufferen
`buffer` er en liste med overganger `(state, action, reward, next_state, done)`. Bruk `random.sample` til å trekke en batch på 4 overganger (lagre listen i `batch`), og pakk den opp i fem numpy-arrays `states`, `actions`, `rewards`, `next_states` og `dones`, hver med shape `(4,)`. Tips: `zip(*batch)`.
```start
import random
import numpy as np

random.seed(1)
buffer = [(s, s % 4, float(s == 14), s + 1, s == 14) for s in range(15)]

batch = ...
states, actions, rewards, next_states, dones = ...
```
```losning
import random
import numpy as np

random.seed(1)
buffer = [(s, s % 4, float(s == 14), s + 1, s == 14) for s in range(15)]

batch = random.sample(buffer, k=4)
states, actions, rewards, next_states, dones = [np.array(x) for x in zip(*batch)]
```
```test
import numpy as np
assert isinstance(batch, list) and len(batch) == 4, "batch skal være en liste med 4 overganger fra random.sample(buffer, k=4)."
assert len(set(batch)) == 4, "random.sample trekker uten tilbakelegging, så alle 4 skal være forskjellige."
assert all(b in buffer for b in batch), "Alle overgangene i batch skal komme fra buffer."
for _navn, _arr in [("states", states), ("actions", actions), ("rewards", rewards), ("next_states", next_states), ("dones", dones)]:
    assert np.asarray(_arr).shape == (4,), f"{_navn} skal ha shape (4,)."
assert list(states) == [b[0] for b in batch], "states skal inneholde første element i hver overgang, i samme rekkefølge som batch."
assert list(next_states) == [b[3] for b in batch], "next_states skal være fjerde element i hver overgang."
```
:::

::: oppgave DQN-targets for en batch
Lag `dqn_targets(rewards, next_q, dones, gamma)` der `next_q` har shape `(B, n_actions)` (target-nettverkets Q-verdier for neste states). Returner targets med shape `(B,)`: `rewards + gamma * maks av next_q per rad`, men med fremtidig verdi 0 der `dones` er `True`.
```start
import numpy as np

def dqn_targets(rewards, next_q, dones, gamma):
    return ...
```
```losning
import numpy as np

def dqn_targets(rewards, next_q, dones, gamma):
    next_max = next_q.max(axis=1)
    return rewards + gamma * np.where(dones, 0.0, next_max)
```
```test
import numpy as np
_y = np.asarray(dqn_targets(np.array([0.0, 1.0, 0.0]),
                            np.array([[0.5, 0.1], [0.8, 0.2], [0.0, 0.2]]),
                            np.array([False, True, False]), 0.9), dtype=float)
assert _y.shape == (3,), f"Resultatet skal ha shape (3,), ikke {_y.shape}. Bruk max(axis=1)."
assert np.isclose(_y[1], 1.0), "Rad 2 er terminal (done = True), så target skal være bare rewarden 1.0."
assert np.allclose(_y, [0.45, 1.0, 0.18]), f"Forventet [0.45, 1.0, 0.18], fikk {_y}. Target = r + γ · max Q(s')."
```
:::

::: quiz
Du har en replay buffer lagret som en Python-liste `buffer` med overganger. Hvilken linje trekker en tilfeldig minibatch på 32? (Eksamen 2025)
- [ ] `np.take(buffer, 32)`
- [x] `random.sample(buffer, k=32)`
- [ ] `buffer.sort()[:32]`
- [ ] `buffer[-32:]`
? `random.sample` trekker 32 ulike elementer tilfeldig. `buffer[-32:]` gir de 32 nyeste, som er sterkt korrelerte, og det er nettopp det replay bufferen skal unngå.
:::

::: quiz
Q-nettverket gir $(0.4,\ 1.0,\ 0.2,\ 0.6)$ for state $s$. Agenten tok action 2, fikk reward 1, og `done` er `True`. Target-nettverket gir $\max_a Q(s', a) = 0.8$, og γ = 0.9. Hva blir MSE-lossen for dette datapunktet i `learn()`?
- [ ] 2.31
- [ ] 0.00
- [x] 0.64
- [ ] 0.80
? `done` er `True`, så target er bare rewarden: $y = 1$. `gather` plukker ut Q for action 2, som er 0.2. Loss = $(1 - 0.2)^2 = 0.64$. 2.31 får du hvis du glemmer done ($y = 1.72$), og 0.80 hvis du glemmer å kvadrere.
:::

::: quiz
Hvorfor beregnes `y_train` inne i `with torch.no_grad():`?
- [x] Target skal være en fast fasit, så det skal ikke gå gradienter gjennom target-nettverket
- [ ] Ellers kan ikke vektene kopieres til target-nettverket senere
- [ ] `no_grad` setter target til 0 for terminale states
- [ ] `no_grad` sørger for at bare den valgte actionen får en verdi i `y_train`
? Som i veiledet læring er `y_train` en label. Det er `torch.where` som setter framtiden til 0 ved done, og `gather` som plukker ut actionen.
:::

::: quiz
En DQN-løkke kjører 60 steg. Bufferen starter tom, får ett element per steg (plass til 1000), og `batch_size = 16`. `learn()` kalles når `replay_buffer.size > batch_size`. Hvor mange ganger kalles `learn()`?
- [ ] 60
- [ ] 45
- [x] 44
- [ ] 43
? Etter steg $t$ er størrelsen $t$. Betingelsen $t > 16$ er sann for $t = 17, \dots, 60$, altså 44 ganger. 45 får du med `>=`.
:::

::: quiz
Hva er hovedgrunnen til at DQN bruker et target network? (Eksamen 2025)
- [ ] For å legge til tilfeldig støy slik at agenten utforsker
- [x] For å holde Q-verdiene som brukes som treningsmål stabile under læringen
- [ ] For å garantere at nettverket aldri overtilpasser replay bufferen
- [ ] For å velge neste action uten å bruke policyen
? Target-nettverket oppdateres sjeldnere (kopieres hvert N-te steg), så $\max_a Q(s', a)$ i target endrer seg ikke for hver gradientoppdatering. Utforsking styres av ε.
:::

::: oppsummering
- DQN: et nevralt nettverk tar inn state (one-hot, `F.one_hot` / `np.eye(n)[s]`) og gir én Q-verdi per action.
- Vektdeling gjør at én oppdatering endrer mange Q-verdier: bevegelig mål, ustabil trening, catastrophic forgetting.
- **Replay buffer**: lagre `(state, action, reward, next_state, done)`, tren på `random.sample(buffer, k=batch_size)`.
- **Target network**: kopi av Q-nettverket (`load_state_dict`) som gir stabile targets, oppdateres hvert N-te steg.
- `learn()`: `no_grad` → `max(dim=1, keepdim=True)` → `torch.where(dones, 0, ...)` → `y_train = r + γ·next` → `gather(predicted_q, 1, actions)` → MSE → `zero_grad`, `backward`, `step`.
- `learn()` kalles bare når bufferen har mer enn `batch_size` overganger.
:::
