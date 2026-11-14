=== DEL 11 | forsterkning | Forsterkningslæring (prosjekt 4)
Du bygger Q-learning fra prosjekt 4 selv, steg for steg: miljø, Q-tabell, ε-greedy, Q-oppdatering, treningsløkke og heatmaps, og til slutt Gymnasium og DQN.

--- LEKSJON 11.1 | miljo | Miljø, state, action og reward
Agenten ser en state, velger en action, og miljøet svarer med `obs, reward, terminated, info = env.step(action)`, mens `env.reset()` starter en ny episode. I prosjekt 4 er actions 0 opp, 1 venstre, 2 ned, 3 høyre, reward er 10 i mål og 0 ellers, og vann blokkerer (eller gir −1 og avslutter episoden med `dead_allowed=True`).

::: oppgave Naboruta
Lag `flytt(pos, action)` som returnerer tuppelen `(rad, kolonne)` agenten havner på, ved å legge `neighbours[action]` til `pos` (ingen sjekk av kanten ennå).
```start
neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]   # 0 opp, 1 venstre, 2 ned, 3 høyre

def flytt(pos, action):
    return ...
```
```losning
neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]

def flytt(pos, action):
    return (pos[0] + neighbours[action][0], pos[1] + neighbours[action][1])
```
```test
assert flytt((5, 0), 0) is not None and flytt((5, 0), 0) is not ..., "flytt skal returnere en tuppel (rad, kolonne)."
assert tuple(flytt((5, 0), 0)) == (5 - 1, 0), "Opp (action 0) fra (5, 0) skal gi (4, 0)."
assert tuple(flytt((5, 0), 3)) == (5, 1), "Høyre (action 3) fra (5, 0) skal gi (5, 1)."
assert tuple(flytt((2, 2), 1)) == (2, 1), "Venstre (action 1) fra (2, 2) skal gi (2, 1)."
assert tuple(flytt((2, 2), 2)) == (3, 2), "Ned (action 2) fra (2, 2) skal gi (3, 2)."
```
:::

::: oppgave Skriv step()
Fullfør `step(self, action)` i miljøet etter reglene i kommentarene. Den skal returnere `obs, reward, terminated, info`, der `obs` er `self.pos` og `info` er `None`.
```start
KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]   # S start, G mål, W vann

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols, self.n_actions = len(kart), len(kart[0]), 4
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]   # 0 opp, 1 venstre, 2 ned, 3 høyre
        self.reset()

    def reset(self):
        start = "".join(self.kart).index("S")
        self.pos = (start // self.ncols, start % self.ncols)
        return self.pos

    def step(self, action):
        # 1. Regn ut ny r og c med self.neighbours[action]
        # 2. Utenfor kartet: bli stående, reward 0, terminated False
        # 3. Vann: reward -1 og terminated True hvis self.dead_allowed, ellers som utenfor kartet
        # 4. Ellers: self.pos = (r, c). Mål gir reward 10 og terminated True, ellers 0 og False
        ...

    def get_state_id(self):
        return self.pos[0] * self.ncols + self.pos[1]
```
```losning
KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols, self.n_actions = len(kart), len(kart[0]), 4
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]
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
```
```test
_env = GridEnv(KART)
_env.reset()
_res = _env.step(1)
assert _res is not None and len(_res) == 4, "step skal returnere fire verdier: obs, reward, terminated, info."
assert tuple(_res[0]) == (5, 0) and _res[1] == 0 and not _res[2], "Venstre fra start (5, 0) går utenfor kartet: agenten blir stående med reward 0 og terminated False."
_res = _env.step(3)
assert tuple(_res[0]) == (5, 1) and _res[1] == 0 and not _res[2], "Høyre fra start skal gi posisjon (5, 1), reward 0 og terminated False."
assert tuple(_env.pos) == (5, 1), "Husk å oppdatere self.pos når agenten flytter seg."
_env.pos = (5, 2)
_res = _env.step(0)
assert tuple(_res[0]) == (5, 2) and _res[1] == 0 and not _res[2], "Opp fra (5, 2) er vann: uten dead_allowed blir agenten stående med reward 0 og terminated False."
_env2 = GridEnv(KART, dead_allowed=True)
_env2.pos = (5, 2)
_res = _env2.step(0)
assert _res[1] == -1 and _res[2], "Med dead_allowed=True skal vann gi reward -1 og terminated True."
_env.pos = (2, 3)
_res = _env.step(3)
assert tuple(_res[0]) == (2, 4) and _res[1] == 10 and _res[2], "Høyre fra (2, 3) går inn i målet: posisjon (2, 4), reward 10 og terminated True."
```
:::

::: oppgave Følg en plan
Kjør actionene i `plan` med `env.step(action_dict[navn])`, summer rewardene i `total_reward`, stopp med `break` hvis episoden avsluttes, og lagre `env.get_state_id()` til slutt i `slutt_state`.
```start
KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols, self.n_actions = len(kart), len(kart[0]), 4
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]   # 0 opp, 1 venstre, 2 ned, 3 høyre
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

action_dict = {'up': 0, 'left': 1, 'down': 2, 'right': 3}
plan = ['right'] * 5 + ['up'] * 3 + ['left']
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
        self.nrows, self.ncols, self.n_actions = len(kart), len(kart[0]), 4
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]
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

action_dict = {'up': 0, 'left': 1, 'down': 2, 'right': 3}
plan = ['right'] * 5 + ['up'] * 3 + ['left']
env = GridEnv(KART)
env.reset()
total_reward = 0
for my_action in plan:
    obs, rew, term, info = env.step(action_dict[my_action])
    total_reward += rew
    if term:
        break
slutt_state = env.get_state_id()
```
```test
assert slutt_state is not ..., "Sett slutt_state = env.get_state_id() etter løkka."
assert total_reward == 10, f"Planen fører til målet, så total_reward skal være 10, ikke {total_reward}."
assert slutt_state == 16, f"Målet ligger på rad 2, kolonne 4, altså state 2 * 6 + 4 = 16. Du fikk {slutt_state}."
```
:::

## State-id: fra (rad, kolonne) til ett tall
`get_state_id()` nummererer rutene radvis med `row * ncols + col` (et 12 × 12-kart har 144 states), og tilbake går det med `state // ncols` og `state % ncols`.

::: oppgave State-id begge veier
Lag `til_state_id(row, col, ncols)` som returnerer state-id-en, og `fra_state_id(state, ncols)` som returnerer tuppelen `(row, col)`.
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
assert til_state_id(3, 4, 12) == 40, "Rad 3, kolonne 4 på et 12 x 12-kart skal gi 3 * 12 + 4 = 40 (ikke bytt om rad og kolonne)."
assert til_state_id(1, 0, 6) == 6, "Første rute på rad 1 skal være 6 når ncols = 6."
assert tuple(fra_state_id(130, 12)) == (10, 10), "fra_state_id(130, 12) skal gi (10, 10). Bruk // og %."
assert tuple(fra_state_id(32, 6)) == (5, 2), "fra_state_id(32, 6) skal gi (5, 2)."
for s in range(36):
    assert til_state_id(*fra_state_id(s, 6), 6) == s, "Funksjonene skal være hverandres motsatte."
```
:::

## En tilfeldig agent
RL-loopen er `env.reset()` og så `env.step(action)` i `for step in range(max_steps):` med `break` når `terminated` er `True`, og antall steg i episoden blir da `step + 1`.

::: oppgave Tilfeldig agent
Kjør 20 episoder der agenten tar tilfeldige actions (`np.random.randint(env.n_actions)`), maks 30 steg per episode. Lagre antall steg i hver episode i `steg_liste` og siste reward i `siste_rewards`.
```start
import numpy as np

KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols, self.n_actions = len(kart), len(kart[0]), 4
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]   # 0 opp, 1 venstre, 2 ned, 3 høyre
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

np.random.seed(0)
env = GridEnv(KART)
max_steps = 30
steg_liste, siste_rewards = [], []
for episode in range(20):
    ...
```
```losning
import numpy as np

KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols, self.n_actions = len(kart), len(kart[0]), 4
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]
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

np.random.seed(0)
env = GridEnv(KART)
max_steps = 30
steg_liste, siste_rewards = [], []
for episode in range(20):
    env.reset()
    for step in range(max_steps):
        obs, reward, terminated, info = env.step(np.random.randint(env.n_actions))
        if terminated:
            break
    steg_liste.append(step + 1)
    siste_rewards.append(reward)
```
```test
assert len(steg_liste) == 20 and len(siste_rewards) == 20, "Det skal være én verdi per episode (20) i både steg_liste og siste_rewards."
for _n, _r in zip(steg_liste, siste_rewards):
    assert _r in (0, 10), "Uten dead_allowed er siste reward enten 0 eller 10."
    if _r == 0:
        assert _n == 30, f"En episode som ikke når målet bruker alle 30 stegene, men du fikk {_n}. Antall steg er step + 1."
    else:
        assert 9 <= _n <= 30, "Korteste vei til målet er 9 steg, og ingen episode kan ha mer enn 30."
```
:::

--- LEKSJON 11.2 | q-tabell | Q-tabellen
`q_table = np.zeros((num_states, num_actions))` har én rad per state og én kolonne per action, og `q_table[s, a]` er agentens anslag på framtidig reward for action `a` i state `s` (reward kommer fra miljøet, Q må læres).

::: oppgave Lag Q-tabellen
Et kart har 5 rader og 7 kolonner, og agenten har 4 actions. Lag `q_table` med nuller, sett Q-verdien for state 9, action 2 til `4.5`, og lagre beste action i state 9 i `beste` med `np.argmax`.
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

## Lese av Q-tabellen
`np.argmax(q_table[s])` er beste action (ved likhet den første), `q_table.max(axis=1)` er verdien $V(s) = \max_a Q(s, a)$ av hver state, og `q_table.argmax(axis=1)` er hele policyen.

::: oppgave Verdi og policy
Q-tabellen under hører til et 2 × 3-kart med mål nede til høyre. Lagre beste action i state 4 i `beste_4`, verdikartet i `V` og policyen i `policy`, begge med `reshape(nrows, ncols)`.
```start
import numpy as np

nrows, ncols = 2, 3
q_table = np.array([
    # opp   venst  ned   høyre
    [7.29, 7.29, 8.1,  8.1 ],
    [8.1,  7.29, 9.0,  9.0 ],
    [9.0,  8.1, 10.0,  9.0 ],
    [7.29, 8.1,  8.1,  9.0 ],
    [8.1,  8.1,  9.0, 10.0 ],
    [0.0,  0.0,  0.0,  0.0 ],    # målet
])

beste_4 = ...
V = ...
policy = ...
```
```losning
import numpy as np

nrows, ncols = 2, 3
q_table = np.array([
    [7.29, 7.29, 8.1,  8.1 ],
    [8.1,  7.29, 9.0,  9.0 ],
    [9.0,  8.1, 10.0,  9.0 ],
    [7.29, 8.1,  8.1,  9.0 ],
    [8.1,  8.1,  9.0, 10.0 ],
    [0.0,  0.0,  0.0,  0.0 ],
])

beste_4 = np.argmax(q_table[4])
V = q_table.max(axis=1).reshape(nrows, ncols)
policy = q_table.argmax(axis=1).reshape(nrows, ncols)
```
```test
import numpy as np
assert beste_4 == 3, "Beste action i state 4 er 3 (høyre, Q = 10)."
assert np.shape(V) == (2, 3), "V skal ha kartets shape (2, 3). Bruk reshape(nrows, ncols)."
assert np.allclose(V, [[8.1, 9.0, 10.0], [9.0, 10.0, 0.0]]), "V skal være største Q-verdi per rad: q_table.max(axis=1)."
assert np.shape(policy) == (2, 3), "policy skal ha shape (2, 3)."
assert np.array_equal(policy, [[2, 2, 2], [3, 3, 0]]), "policy skal være q_table.argmax(axis=1), lagt ut som kartet."
```
:::

::: oppgave Policy som piler
Lag `pilkart` med shape `(2, 3)` ved å indeksere `piler` med `q_table.argmax(axis=1)` og så bruke `reshape`.
```start
import numpy as np

q_table = np.array([[7.29, 7.29, 8.1, 8.1], [8.1, 7.29, 9.0, 9.0], [9.0, 8.1, 10.0, 9.0],
                    [7.29, 8.1, 8.1, 9.0], [8.1, 8.1, 9.0, 10.0], [0.0, 0.0, 0.0, 0.0]])
piler = np.array(["↑", "←", "↓", "→"])

pilkart = ...
```
```losning
import numpy as np

q_table = np.array([[7.29, 7.29, 8.1, 8.1], [8.1, 7.29, 9.0, 9.0], [9.0, 8.1, 10.0, 9.0],
                    [7.29, 8.1, 8.1, 9.0], [8.1, 8.1, 9.0, 10.0], [0.0, 0.0, 0.0, 0.0]])
piler = np.array(["↑", "←", "↓", "→"])

pilkart = piler[q_table.argmax(axis=1)].reshape(2, 3)
```
```test
import numpy as np
assert np.shape(pilkart) == (2, 3), "pilkart skal ha shape (2, 3)."
assert list(pilkart[0]) == ["↓", "↓", "↓"], "Øverste rad skal være ↓ ↓ ↓: ved likhet (8.1 og 8.1) velger argmax den første."
assert list(pilkart[1][:2]) == ["→", "→"], "Nederste rad skal starte med → →."
assert pilkart[1, 2] == "↑", "Målraden er bare nuller, og da velger argmax den første indeksen, 0 = ↑."
```
:::

## Q-tabell som dictionary
Notatenes tre-på-rad-agent lagrer `values[state][action]` i en nøstet dictionary, der `max(q, key=q.get)` er beste action og `values.setdefault(state, {...})` legger inn en state bare hvis den ikke finnes fra før.

::: oppgave Beste trekk
Lagre beste trekk for `q` i `best_action` med `max(q, key=q.get)`.
```start
q = {0: 0.2, 2: 1.4, 6: 0.9, 8: 1.4}

best_action = ...
```
```losning
q = {0: 0.2, 2: 1.4, 6: 0.9, 8: 1.4}

best_action = max(q, key=q.get)
```
```test
assert best_action != 8, "Ved likhet vinner den første nøkkelen med høyest verdi, ikke den siste."
assert best_action != 1, "Svaret skal være nøkkelen (trekket), ikke posisjonen i dictionaryen."
assert best_action == 2, "Trekk 2 og 8 har begge 1.4, og max returnerer det første: 2."
```
:::

::: oppgave Ledige ruter og nye states
Lag `ledige(state)` som returnerer listen med indeksene der tegnet i `state` er et tall (`ch.isnumeric()`). Bruk den med `values.setdefault` for hver state i `nye`, slik at nye states får Q-verdi `0.0` for alle lovlige trekk.
```start
values = {'012345678': {0: 1.933, 1: 1.549, 2: 1.561, 3: 2.140, 4: 2.476,
                        5: 0.656, 6: 1.024, 7: 1.394, 8: 1.667}}
nye = ['012345678', '0123XO678']

def ledige(state):
    return ...

# Bruk setdefault for hver state i nye her
```
```losning
values = {'012345678': {0: 1.933, 1: 1.549, 2: 1.561, 3: 2.140, 4: 2.476,
                        5: 0.656, 6: 1.024, 7: 1.394, 8: 1.667}}
nye = ['012345678', '0123XO678']

def ledige(state):
    return [i for i, ch in enumerate(state) if ch.isnumeric()]

for state in nye:
    values.setdefault(state, {a: 0.0 for a in ledige(state)})
```
```test
assert ledige('0123XO678') is not ..., "ledige skal returnere en liste."
assert list(ledige('0123XO678')) == [0, 1, 2, 3, 6, 7, 8], "Rute 4 og 5 er opptatt (X og O), så ledige ruter er [0, 1, 2, 3, 6, 7, 8]."
assert '0123XO678' in values, "Den nye staten '0123XO678' skal være lagt inn i values."
assert values['0123XO678'] == {a: 0.0 for a in [0, 1, 2, 3, 6, 7, 8]}, "Den nye staten skal ha Q-verdi 0.0 for hvert ledig trekk."
assert values['012345678'][4] == 2.476, "setdefault skal ikke overskrive en state som finnes fra før."
```
:::

--- LEKSJON 11.3 | epsilon | ε-greedy og decay
ε-greedy: `if np.random.random() < explore_rate:` gir en tilfeldig action `np.random.randint(env.n_actions)` (explore), ellers tar agenten `np.argmax(q_table[state])` (exploit), slik at den også finner veier den ikke kjenner.

::: oppgave Skriv update_action
Fullfør prosjektets `update_action(env, state, explore_rate, q_table)` etter regelen over, og returner actionen.
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

::: oppgave Hvor ofte velges den beste?
Den tilfeldige actionen kan også bli den beste. Lag `p_greedy(epsilon, n_actions)` som returnerer sannsynligheten for at den grådige actionen velges.
```start
def p_greedy(epsilon, n_actions):
    return ...
```
```losning
def p_greedy(epsilon, n_actions):
    return 1 - epsilon + epsilon / n_actions
```
```test
assert p_greedy(0.0, 4) is not ..., "p_greedy skal returnere et tall."
assert abs(p_greedy(0.0, 4) - 1.0) < 1e-9, "Med ε = 0 velges den grådige actionen alltid."
assert abs(p_greedy(1.0, 4) - 0.25) < 1e-9, "Med ε = 1 er actionen helt tilfeldig: 1/4."
assert abs(p_greedy(0.2, 4) - 0.85) < 1e-9, "Med ε = 0.2 og 4 actions: 0.8 + 0.2 / 4 = 0.85, ikke 0.8."
```
:::

## Exploration rate som synker (decay)
Prosjektet senker ε med `epsilon_min + (epsilon_max - epsilon_min) * np.exp(-5 * episode / max_episodes)` og læringsraten α på samme måte mot `min_learning_rate`, slik at agenten utforsker mye tidlig og verdiene stabiliserer seg sent.

::: oppgave Eksponentiell decay
Skriv `update_explore_rate(episode, epsilon_min, max_episodes, epsilon_max=1.0)` og `update_learning_rate(episode, min_learning_rate, max_episodes)` (som starter på 1.0). Begge skal virke på et helt array av episoder.
```start
import numpy as np

def update_explore_rate(episode, epsilon_min, max_episodes, epsilon_max=1.0):
    return ...

def update_learning_rate(episode, min_learning_rate, max_episodes):
    return ...
```
```losning
import numpy as np

def update_explore_rate(episode, epsilon_min, max_episodes, epsilon_max=1.0):
    return epsilon_min + (epsilon_max - epsilon_min) * np.exp(-5 * episode / max_episodes)

def update_learning_rate(episode, min_learning_rate, max_episodes):
    return min_learning_rate + (1.0 - min_learning_rate) * np.exp(-5 * episode / max_episodes)
```
```test
import numpy as np
assert update_explore_rate(0, 0.01, 500) is not ..., "update_explore_rate skal returnere et tall."
assert np.isclose(update_explore_rate(0, 0.01, 500), 1.0), "I episode 0 skal ε være epsilon_max = 1.0."
assert np.isclose(update_explore_rate(500, 0.01, 500), 0.01 + 0.99 * np.exp(-5)), "I siste episode skal ε være nesten nede på epsilon_min (≈ 0.0167)."
assert np.isclose(update_explore_rate(0, 0.0, 100, epsilon_max=0.5), 0.5), "Bruk parameteren epsilon_max, ikke et fast tall 1.0."
assert np.isclose(update_learning_rate(0, 0.1, 500), 1.0), "Læringsraten skal starte på 1.0."
assert np.isclose(update_learning_rate(250, 0.1, 500), 0.1 + 0.9 * np.exp(-2.5)), "update_learning_rate(250, 0.1, 500) skal være 0.1 + 0.9 * exp(-2.5)."
assert np.shape(update_explore_rate(np.arange(5), 0.01, 500)) == (5,), "Funksjonen skal virke på et array av episoder (bruk np.exp)."
```
:::

::: oppgave Multiplikativ decay (notatene)
Fullfør `update_epsilon()` slik at den ganger `epsilon` med `epsilon_decay` bare når `epsilon > epsilon_min`. Kall den i 5 episoder, og legg `epsilon` i `historikk` før hvert kall.
```start
epsilon, epsilon_min, epsilon_decay = 1.0, 0.1, 0.6
historikk = []

def update_epsilon():
    global epsilon
    ...

# Løkka over 5 episoder her
```
```losning
epsilon, epsilon_min, epsilon_decay = 1.0, 0.1, 0.6
historikk = []

def update_epsilon():
    global epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

for episode in range(5):
    historikk.append(epsilon)
    update_epsilon()
```
```test
import numpy as np
assert len(historikk) == 5, "historikk skal ha én verdi per episode (5)."
assert np.allclose(historikk, [1.0, 0.6, 0.36, 0.216, 0.1296]), "historikk skal være [1.0, 0.6, 0.36, 0.216, 0.1296]. Legg til epsilon før kallet."
assert abs(epsilon - 0.07776) < 1e-9, f"epsilon skal ende på 0.07776, under minimum: 0.1296 > 0.1, så den ganges én gang til. Du fikk {epsilon}."
```
:::

--- LEKSJON 11.4 | q-oppdatering | Q-oppdateringen
$Q(s,a) \leftarrow Q(s,a) + \alpha\,[R + \gamma \max_{a'} Q(s',a') - Q(s,a)]$, der $R + \gamma \max_{a'} Q(s',a')$ er target, hakeparentesen er TD-feilen, og target bare er $R$ når $s'$ er terminal. Notatenes form $(1-\alpha)\,Q(s,a) + \alpha \cdot \text{target}$ er den samme regelen.

::: oppgave Én oppdatering for hånd
Regn ut `target`, ny Q-verdi på prosjektets form i `ny_q` og på notatenes form i `ny_q2`.
```start
import numpy as np

alpha, gamma = 0.4, 0.9
q_sa = 2.0                             # gammel Q(s, a)
reward = 1
q_next = np.array([3.0, 4.0, 1.5])     # Q-verdiene i neste state s'

target = ...
ny_q = ...
ny_q2 = ...
```
```losning
import numpy as np

alpha, gamma = 0.4, 0.9
q_sa = 2.0
reward = 1
q_next = np.array([3.0, 4.0, 1.5])

target = reward + gamma * np.max(q_next)
ny_q = q_sa + alpha * (target - q_sa)
ny_q2 = (1 - alpha) * q_sa + alpha * target
```
```test
import numpy as np
assert target is not ..., "Regn ut target."
assert np.isclose(target, 4.6), f"target = 1 + 0.9 * max(q_next) = 1 + 0.9 * 4 = 4.6, ikke {target}."
assert np.isclose(ny_q, 3.04), f"ny_q = 2 + 0.4 * (4.6 - 2) = 3.04, ikke {ny_q}."
assert np.isclose(ny_q2, 3.04), f"ny_q2 = 0.6 * 2 + 0.4 * 4.6 = 3.04, ikke {ny_q2}."
```
:::

::: oppgave Skriv Q-oppdateringen
Lag `q_update(q_table, state, action, reward, next_state, terminated, alpha, gamma)` som oppdaterer `q_table[state, action]` på plass, med target lik `reward` når `terminated` er `True`.
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
assert np.isclose(_q2[2, 0], 5.5), "Ved terminated=True skal target være bare reward: 1 + 0.5 * (10 - 1) = 5.5. Glemmer du dette, blir verdiene feil når tabellen ikke starter på 0."
_q3 = np.zeros((4, 2))
q_update(_q3, 0, 0, 0, 1, False, 1.0, 0.9)
assert _q3.sum() == 0, "Med reward 0 og bare nuller i neste state skal ingenting endres."
```
:::

## Rewarden sprer seg bakover
Leddet $\gamma \max_{a'} Q(s',a')$ flytter verdien av målet ett steg bakover per episode, så en rute $d$ steg fra målet ender på $10 \cdot \gamma^{d-1}$ og kortere veier får høyere verdi.

::: oppgave Korridoren
En korridor har 5 ruter og mål til høyre for den siste, og agenten går alltid til høyre. Kjør 6 episoder med α = 1 og γ = 0.9, oppdater `q_right[s]` for `s = 0, ..., 4` i hver episode, og legg `q_right.copy()` i `historikk` etter hver episode.
```start
import numpy as np

n = 5
q_right = np.zeros(n)          # Q(s, høyre) for de 5 rutene
alpha, gamma = 1.0, 0.9
historikk = []
for episode in range(6):
    for s in range(n):
        terminated = (s == n - 1)          # fra siste rute går vi inn i målet
        reward = 10 if terminated else 0
        ...
```
```losning
import numpy as np

n = 5
q_right = np.zeros(n)
alpha, gamma = 1.0, 0.9
historikk = []
for episode in range(6):
    for s in range(n):
        terminated = (s == n - 1)
        reward = 10 if terminated else 0
        target = reward if terminated else reward + gamma * q_right[s + 1]
        q_right[s] += alpha * (target - q_right[s])
    historikk.append(q_right.copy())
```
```test
import numpy as np
assert len(historikk) == 6, "historikk skal ha én rad per episode (6)."
_h = np.array(historikk, dtype=float)
assert np.allclose(_h[-1], [6.561, 7.29, 8.1, 9.0, 10.0]), f"Til slutt skal q_right være 10 * 0.9**(d-1): [6.561, 7.29, 8.1, 9, 10], ikke {_h[-1]}."
assert np.allclose(_h[0], [0, 0, 0, 0, 10]), "Etter første episode skal bare ruta ved målet ha verdi (10). Husk q_right.copy(), ellers blir alle radene like."
assert np.allclose(_h[2], [0, 0, 8.1, 9, 10]), "Verdien skal spre seg én rute bakover per episode."
```
:::

::: oppgave α og γ
`korridor(alpha, gamma)` under er løsningen på forrige oppgave. Lagre resultatet for γ = 0 i `q_gamma0`, for γ = 1 i `q_gamma1` (begge med α = 1) og for α = 0, γ = 0.9 i `q_alpha0`.
```start
import numpy as np

def korridor(alpha, gamma, n=5, episoder=6):
    q_right = np.zeros(n)
    for episode in range(episoder):
        for s in range(n):
            terminated = (s == n - 1)
            reward = 10 if terminated else 0
            target = reward if terminated else reward + gamma * q_right[s + 1]
            q_right[s] += alpha * (target - q_right[s])
    return q_right

q_gamma0 = ...
q_gamma1 = ...
q_alpha0 = ...
```
```losning
import numpy as np

def korridor(alpha, gamma, n=5, episoder=6):
    q_right = np.zeros(n)
    for episode in range(episoder):
        for s in range(n):
            terminated = (s == n - 1)
            reward = 10 if terminated else 0
            target = reward if terminated else reward + gamma * q_right[s + 1]
            q_right[s] += alpha * (target - q_right[s])
    return q_right

q_gamma0 = korridor(1.0, 0.0)
q_gamma1 = korridor(1.0, 1.0)
q_alpha0 = korridor(0.0, 0.9)
```
```test
import numpy as np
assert q_gamma0 is not ... and q_gamma1 is not ... and q_alpha0 is not ..., "Lag alle tre variablene med korridor(...)."
assert np.allclose(q_gamma0, [0, 0, 0, 0, 10]), "γ = 0 gjør agenten kortsiktig: bare ruta rett ved målet får verdi."
assert np.allclose(q_gamma1, [10, 10, 10, 10, 10]), "γ = 1 gir ingen diskontering: alle ruter får 10, uansett avstand."
assert np.allclose(q_alpha0, 0), "α = 0 betyr ingen læring: alt forblir 0."
```
:::

## Notatenes learn-funksjon
Notatene kaller target og TD-feil `td_target` og `td_delta` og bruker en dictionary: `values[state][action] = v_s + alpha * td_delta`.

::: oppgave Skriv learn()
Fullfør `learn` for en Q-tabell som dictionary, der `v_s_next` er største verdi i `values[new_state]` og er 0 når `done` er `True`.
```start
def learn(values, state, action, reward, new_state, done, alpha=0.5, gamma=0.9):
    v_s = values[state][action]
    v_s_next = ...
    td_target = ...
    td_delta = ...
    values[state][action] = v_s + alpha * td_delta
```
```losning
def learn(values, state, action, reward, new_state, done, alpha=0.5, gamma=0.9):
    v_s = values[state][action]
    v_s_next = 0 if done else max(values[new_state].values())
    td_target = reward + gamma * v_s_next
    td_delta = td_target - v_s
    values[state][action] = v_s + alpha * td_delta
```
```test
_v = {'0123X5678': {0: 0.0, 1: 0.0, 2: 0.0}, 'O123X5678': {1: 2.0, 2: 1.0}}
learn(_v, '0123X5678', 0, 0, 'O123X5678', False)
assert abs(_v['0123X5678'][0] - 0.9) < 1e-9, f"Target er 0 + 0.9 * 2.0 = 1.8, og med α = 0.5 blir ny verdi 0.9, ikke {_v['0123X5678'][0]}."
_v2 = {'a': {0: 0.2}, 'b': {0: 5.0}}
learn(_v2, 'a', 0, 1, 'b', True)
assert abs(_v2['a'][0] - 0.6) < 1e-9, "Når done er True finnes ingen framtid: target = 1, og ny verdi 0.2 + 0.5 * 0.8 = 0.6."
```
:::

--- LEKSJON 11.5 | trening | Treningsløkka
Ytre løkke per episode: `env.reset()`, `state = env.get_state_id()` og ny ε og α. Indre løkke per steg: `update_action`, `env.step`, `next_state = env.get_state_id()`, Q-oppdatering, `state = next_state` og `break` ved `terminated`.

::: oppgave Fullfør treningsløkka
Fyll inn de tre linjene som mangler: hent neste state, regn ut `target` og oppdater `q_table[state, action]`. Testen kjører den trente agenten med ε = 0 og sjekker at den når målet.
```start
import numpy as np

KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols, self.n_actions = len(kart), len(kart[0]), 4
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]   # 0 opp, 1 venstre, 2 ned, 3 høyre
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
        self.nrows, self.ncols, self.n_actions = len(kart), len(kart[0]), 4
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]
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

::: oppgave Finn to feil
Treningsløkka under har to klassiske feil i rekkefølgen: én linje mangler i starten av hver episode, og én mangler etter Q-oppdateringen. Rett dem (`env.antall_reset` teller kallene til `reset`).
```start
import numpy as np

KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols, self.n_actions = len(kart), len(kart[0]), 4
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]   # 0 opp, 1 venstre, 2 ned, 3 høyre
        self.antall_reset = 0
        self.reset()
    def reset(self):
        self.antall_reset += 1
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

np.random.seed(0)
env = GridEnv(KART)
q_table = np.zeros((36, 4))
gamma, num_episodes = 0.95, 200
for episode in range(num_episodes):
    state = env.get_state_id()
    explore_rate = 0.01 + 0.99 * np.exp(-5 * episode / num_episodes)
    learning_rate = 0.1 + 0.9 * np.exp(-5 * episode / num_episodes)
    for step in range(100):
        action = update_action(env, state, explore_rate, q_table)
        _, reward, terminated, _ = env.step(action)
        next_state = env.get_state_id()
        target = reward if terminated else reward + gamma * np.max(q_table[next_state])
        q_table[state, action] += learning_rate * (target - q_table[state, action])
        if terminated:
            break
```
```losning
import numpy as np

KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols, self.n_actions = len(kart), len(kart[0]), 4
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        self.antall_reset = 0
        self.reset()
    def reset(self):
        self.antall_reset += 1
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
assert env.antall_reset > num_episodes, "env.reset() må kalles i starten av hver episode, ellers starter neste episode der den forrige sluttet (for eksempel i målet)."
_env = GridEnv(KART)
_s = _env.get_state_id()
for _t in range(50):
    _, _r, _term, _ = _env.step(int(np.argmax(q_table[_s])))
    _s = _env.get_state_id()
    if _term:
        break
assert _term and _r == 10, "Den grådige agenten når ikke målet. Mangler state = next_state? Uten den oppdaterer agenten hele tiden samme rad."
```
:::

::: oppgave Skriv train_q_learning
Skriv hele prosjektets `train_q_learning` med hjelpefunksjonene som er gitt. Lagre `step + 1` i `steps_per_episode` og siste `reward` i `rewards_per_episode` etter hver episode.
```start
import numpy as np

KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols, self.n_actions = len(kart), len(kart[0]), 4
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]   # 0 opp, 1 venstre, 2 ned, 3 høyre
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

def update_explore_rate(episode, epsilon_min, max_episodes, epsilon_max=1.0):
    return epsilon_min + (epsilon_max - epsilon_min) * np.exp(-5 * episode / max_episodes)

def update_learning_rate(episode, min_learning_rate, max_episodes):
    return min_learning_rate + (1.0 - min_learning_rate) * np.exp(-5 * episode / max_episodes)

def update_action(env, state, explore_rate, q_table):
    if np.random.random() < explore_rate:
        return np.random.randint(env.n_actions)
    return int(np.argmax(q_table[state]))

def train_q_learning(env, q_table, num_episodes, max_steps, gamma, epsilon_min, min_learning_rate):
    steps_per_episode, rewards_per_episode = [], []
    # Skriv løkkene her
    return q_table, np.array(steps_per_episode), np.array(rewards_per_episode)
```
```losning
import numpy as np

KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols, self.n_actions = len(kart), len(kart[0]), 4
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]
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

def update_explore_rate(episode, epsilon_min, max_episodes, epsilon_max=1.0):
    return epsilon_min + (epsilon_max - epsilon_min) * np.exp(-5 * episode / max_episodes)

def update_learning_rate(episode, min_learning_rate, max_episodes):
    return min_learning_rate + (1.0 - min_learning_rate) * np.exp(-5 * episode / max_episodes)

def update_action(env, state, explore_rate, q_table):
    if np.random.random() < explore_rate:
        return np.random.randint(env.n_actions)
    return int(np.argmax(q_table[state]))

def train_q_learning(env, q_table, num_episodes, max_steps, gamma, epsilon_min, min_learning_rate):
    steps_per_episode, rewards_per_episode = [], []
    for episode in range(num_episodes):
        env.reset()
        state = env.get_state_id()
        explore_rate = update_explore_rate(episode, epsilon_min, num_episodes)
        learning_rate = update_learning_rate(episode, min_learning_rate, num_episodes)
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
```
```test
import numpy as np
np.random.seed(0)
_env = GridEnv(KART)
_q, _steps, _rew = train_q_learning(_env, np.zeros((36, 4)), 200, 100, 0.95, 0.01, 0.1)
assert len(_steps) == 200 and len(_rew) == 200, "steps_per_episode og rewards_per_episode skal ha én verdi per episode."
assert _q.max() > 0, "Q-tabellen er bare nuller. Oppdaterer du q_table[state, action]?"
assert np.all(_steps[_rew != 10] == 100), "Episoder som ikke når målet skal ha max_steps = 100 steg. Lagre step + 1."
assert np.sum(_rew == 10) > 100, "Agenten skal nå målet i de fleste episodene. Husk env.reset() og state = next_state."
_env.reset()
_s = _env.get_state_id()
for _t in range(50):
    _, _r, _term, _ = _env.step(int(np.argmax(_q[_s])))
    _s = _env.get_state_id()
    if _term:
        break
assert _term and _r == 10, "Den grådige agenten (ε = 0) når ikke målet etter treningen."
```
:::

--- LEKSJON 11.6 | evaluering | Evaluere og visualisere agenten
Den trente agenten vises med ε = 0, slik at den alltid velger `np.argmax(q_table[state])` og vi ser selve policyen.

::: oppgave run_greedy
Skriv `run_greedy(env, q_table, max_steps)` som kjører én episode med beste action i hvert steg og returnerer `(total_reward, antall_steg)`.
```start
import numpy as np

KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols, self.n_actions = len(kart), len(kart[0]), 4
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]   # 0 opp, 1 venstre, 2 ned, 3 høyre
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

def run_greedy(env, q_table, max_steps):
    ...
```
```losning
import numpy as np

KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols, self.n_actions = len(kart), len(kart[0]), 4
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]
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

def run_greedy(env, q_table, max_steps):
    env.reset()
    state = env.get_state_id()
    total = 0
    for step in range(max_steps):
        _, reward, terminated, _ = env.step(int(np.argmax(q_table[state])))
        state = env.get_state_id()
        total += reward
        if terminated:
            break
    return total, step + 1
```
```test
import numpy as np
_q = np.zeros((36, 4))
for _s, _a in [(30, 3), (31, 3), (32, 3), (33, 3), (34, 3), (35, 0), (29, 0), (23, 0), (17, 1)]:
    _q[_s, _a] = 1.0
_env = GridEnv(KART)
_env.pos = (0, 0)
_res = run_greedy(_env, _q, 50)
assert _res is not None and len(_res) == 2, "run_greedy skal returnere (total_reward, antall_steg)."
assert tuple(_res) == (10, 9), f"Med denne Q-tabellen går agenten 9 steg til målet og skal returnere (10, 9), ikke {tuple(_res)}. Husk env.reset() først."
assert tuple(run_greedy(GridEnv(KART), np.zeros((36, 4)), 50)) == (0, 50), "Med bare nuller velger argmax alltid action 0 (opp), agenten når aldri målet og bruker alle 50 stegene."
```
:::

## Heatmap av Q-verdiene
`q_table.max(axis=1).reshape(nrows, ncols)` er verdikartet, `np.where(blank, np.nan, V)` blanker ut vann og mål, og `ax.imshow(V, cmap="RdYlGn")` tegner det.

::: oppgave Tegn verdikartet
Lag `V` med `np.nan` på vann og mål, tegn det med `fig, ax = plt.subplots()` og `ax.imshow`, og lag `Q_opp` med Q-verdiene for action 0 i kartets form.
```start
import numpy as np
import matplotlib.pyplot as plt

KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]
q_table = np.random.default_rng(0).uniform(0, 10, size=(36, 4))
blank = np.array([[ch in "WG" for ch in rad] for rad in KART])    # True på vann og mål

V = ...
Q_opp = ...
```
```losning
import numpy as np
import matplotlib.pyplot as plt

KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]
q_table = np.random.default_rng(0).uniform(0, 10, size=(36, 4))
blank = np.array([[ch in "WG" for ch in rad] for rad in KART])

V = q_table.max(axis=1).reshape(6, 6)
V = np.where(blank, np.nan, V)
fig, ax = plt.subplots()
ax.imshow(V, cmap="RdYlGn")
Q_opp = q_table[:, 0].reshape(6, 6)
```
```test
import numpy as np
assert np.shape(V) == (6, 6), "V skal ha kartets shape (6, 6)."
assert np.isnan(V[1, 1]) and np.isnan(V[2, 4]), "Vann (for eksempel rad 1, kolonne 1) og mål (rad 2, kolonne 4) skal være np.nan."
assert np.isclose(V[5, 0], q_table[30].max()), "Startruta (rad 5, kolonne 0) er state 30 og skal ha q_table[30].max()."
assert len(plt.gca().images) == 1, "Tegn V med ax.imshow(V)."
assert np.shape(Q_opp) == (6, 6) and np.isclose(Q_opp[5, 0], q_table[30, 0]), "Q_opp skal være q_table[:, 0] i shape (6, 6)."
```
:::

::: oppgave Forventede verdier
Lag `verdi(d, gamma)` som returnerer den konvergerte verdien $10 \cdot \gamma^{d-1}$ for en rute `d` steg fra målet, og lagre verdiene for `d = 1, ..., 9` med γ = 0.95 i arrayet `forventet`.
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
assert verdi(1, 0.95) is not ..., "verdi skal returnere et tall."
assert np.isclose(verdi(1, 0.95), 10), "Ett steg fra målet (d = 1) skal verdien være 10."
assert np.isclose(verdi(3, 0.9), 8.1), "d = 3 og γ = 0.9 skal gi 10 * 0.9**2 = 8.1 (ikke 0.9**3 = 7.29)."
assert len(forventet) == 9, "forventet skal ha 9 verdier, for d = 1 til 9."
assert np.isclose(forventet[-1], 10 * 0.95**8), "Siste verdi (d = 9) skal være 10 * 0.95**8 ≈ 6.63."
```
:::

## Steg per episode og glidende snitt
`np.convolve(x, np.ones(w) / w, mode='valid')` gir snittet av hvert vindu på `w` episoder og blir `w - 1` kortere enn `x`.

::: oppgave Glidende snitt
Skriv `moving_average(x, w)` som returnerer det glidende snittet av `x` med vindusstørrelse `w`.
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
assert _r is not ..., "moving_average skal returnere et array."
assert len(_r) == 3, f"Med mode='valid' skal 5 verdier og w=3 gi 3 snitt, ikke {len(_r)}."
assert np.allclose(_r, [6, 4, 6]), f"Forventet [6, 4, 6], fikk {_r}. Husk å dele på w."
_x = np.arange(100.0)
assert np.allclose(moving_average(_x, 10), np.arange(4.5, 95, 1.0)), "Snittet av 0..9 er 4.5, av 1..10 er 5.5, osv."
```
:::

## Eksperimentene
Med γ = 0 finnes ingen «bakke» å klatre, med γ = 1 er alle veier like mye verdt, og med ε = 0 og nuller gir `argmax` alltid action 0, så ingenting læres. Optimistisk start `np.full((num_states, num_actions), 1.0)` får uprøvde actions til å se best ut, så agenten utforsker systematisk og finner kortere veier (bussen i prosjekt 4), også med `dead_allowed=True`.

::: oppgave Kjør eksperimentene
Tren med `train_q_learning` (gitt) og lagre Q-tabellene: `q_gamma0` med γ = 0, `q_eps0` med `epsilon=0.0, epsilon_min=0.0`, og `q_init1` med en tabell som starter på 1.0 (ellers γ = 0.95, ε fra 1.0 ned mot 0.01).
```start
import numpy as np

KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols, self.n_actions = len(kart), len(kart[0]), 4
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]   # 0 opp, 1 venstre, 2 ned, 3 høyre
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

def train_q_learning(env, q_table, num_episodes, max_steps, gamma, epsilon_min, min_learning_rate, epsilon=1.0, seed=0):
    np.random.seed(seed)
    steps_per_episode, rewards_per_episode = [], []
    for episode in range(num_episodes):
        env.reset()
        state = env.get_state_id()
        explore_rate = epsilon_min + (epsilon - epsilon_min) * np.exp(-5 * episode / num_episodes)
        learning_rate = min_learning_rate + (1 - min_learning_rate) * np.exp(-5 * episode / num_episodes)
        for step in range(max_steps):
            if np.random.random() < explore_rate:
                action = np.random.randint(env.n_actions)
            else:
                action = int(np.argmax(q_table[state]))
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
q_base, steps_base, rewards_base = train_q_learning(env, np.zeros((36, 4)), 200, 100, gamma=0.95, epsilon_min=0.01, min_learning_rate=0.1)

q_gamma0 = ...
q_eps0 = ...
q_init1 = ...
```
```losning
import numpy as np

KART = ["......", ".WW...", "..W.G.", "..W...", "..WWW.", "S....."]

class GridEnv:
    def __init__(self, kart, dead_allowed=False):
        self.kart, self.dead_allowed = kart, dead_allowed
        self.nrows, self.ncols, self.n_actions = len(kart), len(kart[0]), 4
        self.neighbours = [(-1, 0), (0, -1), (1, 0), (0, 1)]
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

def train_q_learning(env, q_table, num_episodes, max_steps, gamma, epsilon_min, min_learning_rate, epsilon=1.0, seed=0):
    np.random.seed(seed)
    steps_per_episode, rewards_per_episode = [], []
    for episode in range(num_episodes):
        env.reset()
        state = env.get_state_id()
        explore_rate = epsilon_min + (epsilon - epsilon_min) * np.exp(-5 * episode / num_episodes)
        learning_rate = min_learning_rate + (1 - min_learning_rate) * np.exp(-5 * episode / num_episodes)
        for step in range(max_steps):
            if np.random.random() < explore_rate:
                action = np.random.randint(env.n_actions)
            else:
                action = int(np.argmax(q_table[state]))
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
q_base, steps_base, rewards_base = train_q_learning(env, np.zeros((36, 4)), 200, 100, gamma=0.95, epsilon_min=0.01, min_learning_rate=0.1)

q_gamma0, _, _ = train_q_learning(env, np.zeros((36, 4)), 200, 100, gamma=0.0, epsilon_min=0.01, min_learning_rate=0.1)
q_eps0, _, _ = train_q_learning(env, np.zeros((36, 4)), 200, 100, gamma=0.95, epsilon_min=0.0, min_learning_rate=0.1, epsilon=0.0)
q_init1, _, _ = train_q_learning(env, np.full((36, 4), 1.0), 200, 100, gamma=0.95, epsilon_min=0.01, min_learning_rate=0.1)
```
```test
import numpy as np
def _greedy(q):
    _e = GridEnv(KART)
    _s = _e.get_state_id()
    for _t in range(50):
        _, _r, _term, _ = _e.step(int(np.argmax(q[_s])))
        _s = _e.get_state_id()
        if _term:
            return _r, _t + 1
    return 0, 50
for _navn, _verdi in [("q_gamma0", q_gamma0), ("q_eps0", q_eps0), ("q_init1", q_init1)]:
    assert isinstance(_verdi, np.ndarray), f"{_navn} skal være Q-tabellen. train_q_learning returnerer (q_table, steps, rewards), så pakk ut: {_navn}, _, _ = ..."
assert 0 < (q_gamma0 > 0).sum() <= 4, "Med γ = 0 skal bare actions som går rett inn i målet få verdi. Satte du gamma=0.0?"
assert _greedy(q_gamma0)[0] == 0, "Med γ = 0 skal den grådige agenten ikke nå målet."
assert np.all(q_eps0 == 0), "Med epsilon=0.0 og epsilon_min=0.0 velger argmax alltid action 0, target blir 0, og tabellen skal fortsatt være bare nuller."
assert _greedy(q_init1) == (10, 9), f"Med optimistisk start (np.full(..., 1.0)) skal den grådige agenten finne korteste vei på 9 steg, men fikk {_greedy(q_init1)}."
```
:::

--- LEKSJON 11.7 | gymnasium | Gymnasium og FrozenLake
I Gymnasium gir `env.reset()` en tuppel `(state, info)` og `env.step(action)` fem verdier, og episoden er over når `terminated or truncated` (`truncated` = avbrutt etter maks antall steg). FrozenLake er en 4 × 4-innsjø med actions 0 venstre, 1 ned, 2 høyre, 3 opp (en annen rekkefølge enn prosjektet), reward 1 bare for gaven og hull som avslutter episoden.

```python statisk
import gymnasium as gym
env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False)
print(env.action_space, env.observation_space)
state, info = env.reset(seed=0)
next_state, reward, terminated, truncated, info = env.step(2)   # 2 = høyre
print(state, next_state, reward, terminated, truncated)
env.action_space.seed(0)
print(env.action_space.sample())                                # tilfeldig action
```
```output
Discrete(4) Discrete(16)
0 1 0 False False
3
```

::: oppgave FrozenLake i ren Python
Lag `fl_step(state, action)` for kartet `desc` som returnerer `(next_state, reward, terminated)`: flytt med FrozenLake-rekkefølgen, bli stående ved kanten, og avslutt i hull (`H`, reward 0) og mål (`G`, reward 1).
```start
desc = ["SFFF",
        "FHFH",
        "FFFH",
        "HFFG"]
moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]   # 0 venstre, 1 ned, 2 høyre, 3 opp

def fl_step(state, action):
    ...
```
```losning
desc = ["SFFF",
        "FHFH",
        "FFFH",
        "HFFG"]
moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]

def fl_step(state, action):
    r, c = state // 4 + moves[action][0], state % 4 + moves[action][1]
    if 0 <= r < 4 and 0 <= c < 4:
        state = r * 4 + c
    ch = desc[state // 4][state % 4]
    return state, (1 if ch == "G" else 0), ch in "HG"
```
```test
assert fl_step(0, 2) is not None, "fl_step skal returnere (next_state, reward, terminated)."
assert tuple(fl_step(0, 2)) == (1, 0, False), "Høyre (2) fra state 0 skal gi (1, 0, False)."
assert tuple(fl_step(0, 0)) == (0, 0, False), "Venstre fra state 0 går mot kanten: agenten blir stående."
assert tuple(fl_step(6, 1)) == (10, 0, False), "Ned (1) fra state 6 skal gi state 10 (rad 2, kolonne 2)."
assert tuple(fl_step(1, 1)) == (5, 0, True), "Ned fra state 1 er et hull (state 5): reward 0 og terminated True."
assert tuple(fl_step(14, 2)) == (15, 1, True), "Høyre fra state 14 er gaven: reward 1 og terminated True."
```
:::

::: oppgave Oversett actions
Lag dictionaryen `til_frozenlake` som oversetter prosjektets actions (0 opp, 1 venstre, 2 ned, 3 høyre) til samme retning i FrozenLake, og bruk den til å lage `plan_fl` fra `plan`.
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

## Q-learning i Gymnasium-stil
`FakeLake` i oppgavene oppfører seg som FrozenLake, så skriv `state, _ = env.reset()` (ikke `state = env.reset()`, som gir en tuppel) og bruk bare `terminated` i target, men `terminated or truncated` for å stoppe.

::: oppgave Grådig episode
Skriv `run_episode(env, q_table)` som kjører én grådig episode og returnerer listen `path` med alle states, fra start til episoden er over.
```start
import numpy as np

class FakeLake:
    """Som gym.make('FrozenLake-v1', is_slippery=False): 0 venstre, 1 ned, 2 høyre, 3 opp."""
    desc = "SFFFFHFHFFFHHFFG"
    def reset(self):
        self.s, self.t, self.over = 0, 0, False
        return self.s, {}
    def step(self, a):
        assert not self.over, "Episoden er over. Stopp løkka med if terminated or truncated: break"
        r, c = self.s // 4 + [0, 1, 0, -1][a], self.s % 4 + [-1, 0, 1, 0][a]
        if 0 <= r < 4 and 0 <= c < 4:
            self.s = r * 4 + c
        self.t += 1
        terminated, truncated = self.desc[self.s] in "HG", self.t >= 100
        self.over = terminated or truncated
        return self.s, int(self.desc[self.s] == "G"), terminated, truncated, {}

def run_episode(env, q_table):
    ...
```
```losning
import numpy as np

class FakeLake:
    """Som gym.make('FrozenLake-v1', is_slippery=False): 0 venstre, 1 ned, 2 høyre, 3 opp."""
    desc = "SFFFFHFHFFFHHFFG"
    def reset(self):
        self.s, self.t, self.over = 0, 0, False
        return self.s, {}
    def step(self, a):
        assert not self.over, "Episoden er over. Stopp løkka med if terminated or truncated: break"
        r, c = self.s // 4 + [0, 1, 0, -1][a], self.s % 4 + [-1, 0, 1, 0][a]
        if 0 <= r < 4 and 0 <= c < 4:
            self.s = r * 4 + c
        self.t += 1
        terminated, truncated = self.desc[self.s] in "HG", self.t >= 100
        self.over = terminated or truncated
        return self.s, int(self.desc[self.s] == "G"), terminated, truncated, {}

def run_episode(env, q_table):
    state, _ = env.reset()
    path = [state]
    while True:
        state, reward, terminated, truncated, _ = env.step(int(np.argmax(q_table[state])))
        path.append(state)
        if terminated or truncated:
            break
    return path
```
```test
import numpy as np
_q = np.zeros((16, 4))
for _s, _a in [(0, 1), (4, 1), (8, 2), (9, 1), (13, 2), (14, 2)]:
    _q[_s, _a] = 1.0
_p = run_episode(FakeLake(), _q)
assert _p is not None, "run_episode skal returnere listen path."
assert [int(x) for x in _p] == [0, 4, 8, 9, 13, 14, 15], f"Veien skal være [0, 4, 8, 9, 13, 14, 15], fikk {_p}. Husk state, _ = env.reset() og at startstaten er med."
_p0 = run_episode(FakeLake(), np.zeros((16, 4)))
assert len(_p0) == 101, "Med bare nuller går agenten mot venstre kant til episoden avbrytes (truncated) etter 100 steg, så path skal ha 101 states."
```
:::

::: oppgave Tren på FakeLake
Fyll inn den indre løkka: ta et steg med `env.step`, regn ut `target` (bare `reward` når `terminated`), oppdater `q_table[state, action]`, flytt `state` og stopp når `terminated or truncated`.
```start
import numpy as np

class FakeLake:
    """Som gym.make('FrozenLake-v1', is_slippery=False): 0 venstre, 1 ned, 2 høyre, 3 opp."""
    desc = "SFFFFHFHFFFHHFFG"
    def reset(self):
        self.s, self.t, self.over = 0, 0, False
        return self.s, {}
    def step(self, a):
        assert not self.over, "Episoden er over. Stopp løkka med if terminated or truncated: break"
        r, c = self.s // 4 + [0, 1, 0, -1][a], self.s % 4 + [-1, 0, 1, 0][a]
        if 0 <= r < 4 and 0 <= c < 4:
            self.s = r * 4 + c
        self.t += 1
        terminated, truncated = self.desc[self.s] in "HG", self.t >= 100
        self.over = terminated or truncated
        return self.s, int(self.desc[self.s] == "G"), terminated, truncated, {}

np.random.seed(0)
env = FakeLake()
q_table = np.ones((16, 4))       # optimistisk start: med nuller finner agenten nesten aldri gaven
gamma, num_episodes = 0.95, 500
for episode in range(num_episodes):
    state, _ = env.reset()
    explore_rate = 0.01 + 0.99 * np.exp(-5 * episode / num_episodes)
    learning_rate = 0.1 + 0.9 * np.exp(-5 * episode / num_episodes)
    for step in range(100):
        if np.random.random() < explore_rate:
            action = np.random.randint(4)
        else:
            action = int(np.argmax(q_table[state]))
        ...
```
```losning
import numpy as np

class FakeLake:
    """Som gym.make('FrozenLake-v1', is_slippery=False): 0 venstre, 1 ned, 2 høyre, 3 opp."""
    desc = "SFFFFHFHFFFHHFFG"
    def reset(self):
        self.s, self.t, self.over = 0, 0, False
        return self.s, {}
    def step(self, a):
        assert not self.over, "Episoden er over. Stopp løkka med if terminated or truncated: break"
        r, c = self.s // 4 + [0, 1, 0, -1][a], self.s % 4 + [-1, 0, 1, 0][a]
        if 0 <= r < 4 and 0 <= c < 4:
            self.s = r * 4 + c
        self.t += 1
        terminated, truncated = self.desc[self.s] in "HG", self.t >= 100
        self.over = terminated or truncated
        return self.s, int(self.desc[self.s] == "G"), terminated, truncated, {}

np.random.seed(0)
env = FakeLake()
q_table = np.ones((16, 4))
gamma, num_episodes = 0.95, 500
for episode in range(num_episodes):
    state, _ = env.reset()
    explore_rate = 0.01 + 0.99 * np.exp(-5 * episode / num_episodes)
    learning_rate = 0.1 + 0.9 * np.exp(-5 * episode / num_episodes)
    for step in range(100):
        if np.random.random() < explore_rate:
            action = np.random.randint(4)
        else:
            action = int(np.argmax(q_table[state]))
        next_state, reward, terminated, truncated, _ = env.step(action)
        target = reward if terminated else reward + gamma * np.max(q_table[next_state])
        q_table[state, action] += learning_rate * (target - q_table[state, action])
        state = next_state
        if terminated or truncated:
            break
```
```test
import numpy as np
assert q_table.max() <= 1 + 1e-9, "Ingen Q-verdi skal bli større enn 1 (rewarden for gaven). Bruk target = reward når terminated."
_e = FakeLake()
_s, _ = _e.reset()
for _t in range(20):
    _s, _r, _term, _trunc, _ = _e.step(int(np.argmax(q_table[_s])))
    if _term or _trunc:
        break
assert _s == 15 and _r == 1, "Den grådige agenten når ikke gaven etter treningen. Sjekk steget, target og state = next_state."
```
:::

## Ditt eget miljø som gym.Env
Et eget miljø arver fra `gym.Env`, setter `self.action_space = Discrete(4)` og `self.observation_space`, og har `reset(self, seed=None, options=None)` som returnerer `(obs, {})` og `step` som returnerer fem verdier. FrozenLake har `env.observation_space.n` states, mens prosjektets `TrondheimEnv` bruker `MultiDiscrete` og `int(np.prod(env.observation_space.shape))`.

```python statisk
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

action_space = Discrete(4)
observation_space = MultiDiscrete(np.full((12, 12), 5))     # som TrondheimEnv: 12 x 12-kart
print(action_space.n, observation_space.shape)
print(int(np.prod(observation_space.shape)))                # antall states
```
```output
4 (12, 12)
144
```

::: oppgave Korridor med Gymnasium-grensesnitt
Fullfør `Korridor`: `reset()` setter `self.pos = 0` og returnerer `(0, {})`, og `step(action)` flytter én rute (0 venstre, 1 høyre) innenfor `0, ..., n-1` og returnerer `(pos, reward, terminated, False, {})`, med reward 1 og `terminated=True` i siste rute.
```start
class Korridor:              # i Jupyter: class Korridor(gym.Env)
    def __init__(self, n=5):
        self.n = n

    def reset(self, seed=None, options=None):
        ...

    def step(self, action):
        ...
```
```losning
class Korridor:
    def __init__(self, n=5):
        self.n = n

    def reset(self, seed=None, options=None):
        self.pos = 0
        return self.pos, {}

    def step(self, action):
        self.pos = max(0, min(self.n - 1, self.pos + (1 if action == 1 else -1)))
        terminated = self.pos == self.n - 1
        return self.pos, (1 if terminated else 0), terminated, False, {}
```
```test
_k = Korridor(4)
_r = _k.reset()
assert _r is not None and tuple(_r) == (0, {}), "reset() skal returnere tuppelen (0, {})."
_res = _k.step(0)
assert _res is not None and len(_res) == 5, "step skal returnere fem verdier: obs, reward, terminated, truncated, info."
assert _res[0] == 0 and _res[1] == 0 and not _res[2], "Venstre fra rute 0 går mot kanten: agenten blir stående."
_k.step(1)
_k.step(1)
_res = _k.step(1)
assert _res[0] == 3 and _res[1] == 1 and _res[2] and not _res[3], "Tre steg til høyre i Korridor(4) skal gi (3, 1, True, False, {})."
```
:::

--- LEKSJON 11.8 | dqn | Deep Q-learning (DQN)
DQN bytter ut Q-tabellen med et nevralt nettverk som tar inn en one-hot-kodet state og gir én Q-verdi per action, slik at det virker også når det finnes altfor mange states til en tabell.

## Input og output
`F.one_hot(torch.tensor(5), 16)` er state 5 som one-hot (heltall, så nettverket trenger `.float()`), og i numpy er det `np.eye(16)[5]`.

```python statisk
import torch
import torch.nn as nn
import torch.nn.functional as F

s = F.one_hot(torch.tensor(5), 16)
q_network = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))
print(s)
print(q_network(s.float()).shape)                   # én Q-verdi per action
```
```output
tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
torch.Size([4])
```

::: oppgave One-hot-koding
Lag `one_hot(states, num_states)` som returnerer et array med shape `(len(states), num_states)`, der rad `i` har 1 i kolonne `states[i]` og 0 ellers.
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

::: oppgave Q-nettverket i numpy
Regn ut `q_values = ReLU(X @ W1 + b1) @ W2 + b2` for batchen `X` (ReLU er `np.maximum(0, ...)`), og lagre beste action for hver rad i `beste`.
```start
import numpy as np

rng = np.random.default_rng(0)
W1, b1 = rng.normal(size=(16, 8)), np.zeros(8)
W2, b2 = rng.normal(size=(8, 4)), np.zeros(4)
X = np.eye(16)[[0, 5, 15]]          # one-hot for state 0, 5 og 15

q_values = ...
beste = ...
```
```losning
import numpy as np

rng = np.random.default_rng(0)
W1, b1 = rng.normal(size=(16, 8)), np.zeros(8)
W2, b2 = rng.normal(size=(8, 4)), np.zeros(4)
X = np.eye(16)[[0, 5, 15]]

q_values = np.maximum(0, X @ W1 + b1) @ W2 + b2
beste = q_values.argmax(axis=1)
```
```test
import numpy as np
_fasit = np.maximum(0, X @ W1 + b1) @ W2 + b2
assert np.shape(q_values) == (3, 4), f"q_values skal ha shape (3, 4): én rad per state, én Q-verdi per action. Fikk {np.shape(q_values)}."
assert np.allclose(q_values, _fasit), "q_values stemmer ikke. Husk ReLU mellom lagene: np.maximum(0, X @ W1 + b1)."
assert np.array_equal(np.asarray(beste).ravel(), _fasit.argmax(axis=1)), "beste skal være argmax per rad (axis=1)."
```
:::

## Replay bufferen
Påfølgende steg er sterkt korrelerte, og siden nettverket deler vekter kan trening på dem få det til å glemme det det har lært (catastrophic forgetting). Replay bufferen lagrer overganger `(state, action, reward, next_state, done)` i en liste og trener på `random.sample(buffer, k=batch_size)`, som trekker tilfeldig og uten tilbakelegging.

::: oppgave ExperienceReplayBuffer
Fullfør `add` (legg til tuppelen, og `pop(0)` den eldste hvis bufferen blir større enn `replay_size`) og `sample` (trekk med `random.sample` og returner fem numpy-arrays, ett per felt, med `zip(*batch)`).
```start
import random
import numpy as np

class ExperienceReplayBuffer:
    def __init__(self, replay_size):
        self.replay_size = replay_size
        self.buffer = []

    @property
    def size(self):                 # brukes som buf.size, uten parenteser
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        ...

    def sample(self, batch_size):
        ...
```
```losning
import random
import numpy as np

class ExperienceReplayBuffer:
    def __init__(self, replay_size):
        self.replay_size = replay_size
        self.buffer = []

    @property
    def size(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.replay_size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, k=batch_size)
        return [np.array(x) for x in zip(*batch)]
```
```test
import random
import numpy as np
_b = ExperienceReplayBuffer(5)
for _t in range(7):
    _b.add(_t, _t % 4, float(_t == 6), _t + 1, _t == 6)
assert _b.size == 5, f"Bufferen har plass til 5, så size skal være 5 etter 7 add, ikke {_b.size}."
assert _b.buffer[0][0] == 2, "De to eldste overgangene (state 0 og 1) skal være kastet ut med pop(0)."
random.seed(0)
_out = _b.sample(3)
assert _out is not None and len(_out) == 5, "sample skal returnere fem arrays: states, actions, rewards, next_states, dones."
_states, _actions, _rewards, _next, _dones = [np.asarray(x) for x in _out]
assert _states.shape == (3,), "Hvert array skal ha shape (batch_size,) = (3,)."
assert len(set(_states.tolist())) == 3, "random.sample trekker uten tilbakelegging, så alle 3 states skal være forskjellige."
assert np.array_equal(_next, _states + 1), "next_states skal høre til samme overgang som states. Bruk zip(*batch)."
```
:::

## Target-nettverket og learn()
Target-nettverket er en kopi av Q-nettverket (`load_state_dict`) som bare oppdateres hvert N-te steg, slik at targetene $y = r + \gamma \max_a Q_{target}(s', a)$ (med 0 framtid når `done`) står stille mens Q-nettverket trenes. `torch.gather(predicted_q, 1, actions)` plukker ut Q for actionen som ble tatt (numpy: `np.take_along_axis`), og optimizerens `lr` har tatt over rollen til α.

```python statisk
import torch
import torch.nn as nn
q_network, target_network = nn.Linear(16, 4), nn.Linear(16, 4)
target_network.load_state_dict(q_network.state_dict())      # kopi, gjentas hvert N-te steg
states, next_states = torch.eye(16)[[0, 5]], torch.eye(16)[[1, 6]]
actions, rewards, dones = torch.tensor([[2], [1]]), torch.tensor([[0.0], [1.0]]), torch.tensor([[False], [True]])
with torch.no_grad():                                        # target er en fast fasit, ingen gradienter
    next_q = target_network(next_states).max(dim=1, keepdim=True).values
    y_train = rewards + 0.95 * torch.where(dones, torch.zeros_like(next_q), next_q)
```
```python statisk fortsett
optimizer = torch.optim.Adam(q_network.parameters(), lr=0.001)
y_pred = torch.gather(q_network(states), 1, actions)         # (B, 1): Q for valgt action
loss = torch.nn.functional.mse_loss(y_pred, y_train)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

::: oppgave DQN-targets for en batch
Lag `dqn_targets(rewards, next_q, dones, gamma)` der `next_q` har shape `(B, n_actions)`, og returner targets med shape `(B,)`: `rewards + gamma * maks av next_q per rad`, med framtid 0 der `dones` er `True`.
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
_y = dqn_targets(np.array([0.0, 1.0, 0.0]), np.array([[0.5, 0.1], [0.8, 0.2], [0.0, 0.2]]), np.array([False, True, False]), 0.9)
assert _y is not ..., "dqn_targets skal returnere et array."
_y = np.asarray(_y, dtype=float)
assert _y.shape == (3,), f"Resultatet skal ha shape (3,), ikke {_y.shape}. Bruk max(axis=1)."
assert np.isclose(_y[1], 1.0), "Rad 2 er terminal (done = True), så target skal være bare rewarden 1.0."
assert np.allclose(_y, [0.45, 1.0, 0.18]), f"Forventet [0.45, 1.0, 0.18], fikk {_y}. Target = r + γ · max Q(s')."
```
:::

::: oppgave Loss for valgt action
Lag `dqn_loss(predicted_q, actions, y_train)` der `actions` og `y_train` har shape `(B, 1)`: plukk ut Q for valgt action med `np.take_along_axis(predicted_q, actions, axis=1)` og returner MSE mot `y_train`.
```start
import numpy as np

def dqn_loss(predicted_q, actions, y_train):
    return ...
```
```losning
import numpy as np

def dqn_loss(predicted_q, actions, y_train):
    y_pred = np.take_along_axis(predicted_q, actions, axis=1)
    return np.mean((y_train - y_pred) ** 2)
```
```test
import numpy as np
_l = dqn_loss(np.array([[0.4, 1.0, 0.2, 0.6]]), np.array([[2]]), np.array([[1.0]]))
assert _l is not ..., "dqn_loss skal returnere et tall."
assert np.isclose(_l, 0.64), f"Q for action 2 er 0.2 og target 1.0, så loss = (1 - 0.2)**2 = 0.64, ikke {_l}."
_l2 = dqn_loss(np.array([[1.2, 0.5, 2.0, -0.3], [0.1, 0.4, 0.3, 0.2], [0.7, 0.9, 0.0, 0.5]]),
               np.array([[1], [2], [0]]), np.array([[1.425], [1.0], [0.76]]))
assert np.isclose(_l2, ((1.425 - 0.5)**2 + (1.0 - 0.3)**2 + (0.76 - 0.7)**2) / 3), "Loss for batchen stemmer ikke. Pass på at y_pred har shape (B, 1) som y_train."
```
:::

## RL-loopen med DQN
Notatenes loop legger hver overgang i bufferen, kaller `agent.learn()` bare når `agent.replay_buffer.size > agent.batch_size`, og kaller `agent.update_target_network()` når `total_steps % agent.target_update_frequency == 0`.

::: oppgave Tell kallene
Lag `tell_kall(n_steps, batch_size, target_update_frequency)` som simulerer løkka (bufferen starter tom og får ett element per steg) og returnerer `(antall learn-kall, antall target-oppdateringer)`.
```start
def tell_kall(n_steps, batch_size, target_update_frequency):
    size, n_learn, n_target = 0, 0, 0
    for total_steps in range(1, n_steps + 1):
        ...
    return n_learn, n_target
```
```losning
def tell_kall(n_steps, batch_size, target_update_frequency):
    size, n_learn, n_target = 0, 0, 0
    for total_steps in range(1, n_steps + 1):
        size += 1
        if size > batch_size:
            n_learn += 1
        if total_steps % target_update_frequency == 0:
            n_target += 1
    return n_learn, n_target
```
```test
assert tuple(tell_kall(100, 32, 10)) == (68, 10), f"100 steg, batch_size 32: learn kalles i steg 33 til 100 (68 ganger), target hvert 10. steg (10 ganger). Fikk {tell_kall(100, 32, 10)}."
assert tuple(tell_kall(60, 16, 10))[0] == 44, "60 steg og batch_size 16 skal gi 44 learn-kall (steg 17 til 60). 45 får du med >= i stedet for >."
assert tuple(tell_kall(10, 32, 3)) == (0, 3), "Med færre steg enn batch_size skal learn aldri kalles."
```
:::
