## 1. WPROWADZENIE

Celem hackathonu jest stworzenie bota do gry ***watch out for pointy things***. Gracz porusza się między lewą i prawą ścianą, unikając kolców i zbierając monety. Bot musi podejmować decyzje o skoku w odpowiednich momentach, aby przetrwać jak najdłużej.

### Wymagany plik: `solution.py` i opcjonalnie `model.*`

Uczestnicy muszą stworzyć plik `solution.py` zawierający:

**Elementy wymagane:**
- Klasę bota dziedziczącą po `BaseBot`
- Metodę `take_action(self, obs) -> int` w klasie bota
- Funkcję `create_bot() -> BaseBot`

**Elementy opcjonalne:**
- `BENCHMARK_EPISODES: int` - liczba epizodów benchmarku (domyślnie: 50)
- `WATCH_GAME: bool` - czy pokazać wizualizację gry (domyślnie: True)
- Funkcję `calculate_reward(game_state: dict) -> float` - obowiązkowa przy implementacjia RL


**Pliki w zipie prześlicie na mail: gauss@pwr.edu.pl.**

W razie pytań zapraszamy na [discord hackathonu](https://discord.gg/XmfHAPgr).


---

## 2. WYMAGANA STRUKTURA PLIKU solution.py

```python
from base_bot import BaseBot
import numpy as np

# Opcjonalne stałe konfiguracyjne
BENCHMARK_EPISODES = 50  # Liczba epizodów w benchmarku
WATCH_GAME = True        # Czy wyświetlić wizualizację po benchmarku

class MojBot(BaseBot):
    """
    Klasa bota uczestnika.
    Musi dziedziczyć po BaseBot.
    """
    
    def __init__(self):
        # Inicjalizacja bota
        pass
    
    def take_action(self, obs: np.ndarray) -> int:
        """
        Funkcja podejmująca decyzję o akcji na podstawie obserwacji.
        
        Args:
            obs: numpy array o rozmiarze (15,) z danymi o stanie gry
            
        Returns:
            int: akcja do wykonania (0 lub 1)
        """
        # Implementacja logiki bota
        pass


def create_bot() -> BaseBot:
    """
    Funkcja tworząca i zwracająca instancję bota.
    
    Returns:
        BaseBot: instancja klasy bota dziedziczącej po BaseBot
    """
    return MojBot()


def calculate_reward(game_state: dict) -> float:
    """
    Funkcja obliczająca nagrodę na podstawie stanu gry.
    Wywoływana na każdym kroku gry oraz po zakończeniu gry.
    
    Args:
        game_state: słownik zawierający informacje o stanie gry
        
    Returns:
        float: wartość nagrody
    """
    # Implementacja funkcji nagrody
    pass
```

---

## 3. SZCZEGÓŁOWA SPECYFIKACJA API

### 3.1. AKCJE (action: int)

**Przestrzeń akcji:** `Discrete(2)`

| Wartość | Opis |
|---------|------|
| `0` | Brak akcji - gracz kontynuuje ruch w obecnym kierunku |
| `1` | Skok - gracz wykonuje skok (zmiana grawitacji) |

### 3.2. OBSERWACJA (obs: np.ndarray)

**Typ:** `numpy.ndarray`  
**Dtype:** `np.float32`  
**Rozmiar:** `(15,)`

#### Struktura obserwacji:

**Indeksy 0-3: Dane gracza**

| Indeks | Nazwa | Typ | Opis | Zakres wartości |
|--------|-------|-----|------|-----------------|
| `obs[0]` | `player_x` | `float` | Pozycja X gracza | $[0, 576]$ |
| `obs[1]` | `player_y` | `float` | Pozycja Y gracza | $[0, 704]$ |
| `obs[2]` | `player_direction` | `float` | Kierunek ruchu | $1.0$ (prawo) lub $-1.0$ (lewo) |
| `obs[3]` | `player_gravity` | `float` | Aktualna wartość grawitacji | Wartość rzeczywista |

**Indeksy 4-5: Dane monety**

| Indeks | Nazwa | Typ | Opis | Zakres wartości |
|--------|-------|-----|------|-----------------|
| `obs[4]` | `coin_x` | `float` | Pozycja X monety | $[40, 536]$ lub $-1.0$ (brak) |
| `obs[5]` | `coin_y` | `float` | Pozycja Y monety | $[40, 664]$ lub $-1.0$ (brak) |

**Uwaga:** Wartość $-1.0$ w danych monety oznacza brak monety na planszy.

**Indeksy 6-14: Pozycje kolców (9 wartości)**

| Indeks | Nazwa | Typ | Opis |
|--------|-------|-----|------|
| `obs[6]` | `spike_y_0` | `float` | Pozycja Y pierwszego kolca |
| `obs[7]` | `spike_y_1` | `float` | Pozycja Y drugiego kolca |
| `...` | `...` | `float` | `...` |
| `obs[14]` | `spike_y_8` | `float` | Pozycja Y dziewiątego kolca |

**Uwagi dotyczące kolców:**
- Wartość $-1.0$ oznacza brak kolca w danym slocie
- Kolce pochodzą ze ściany docelowej (tej, w którą gracz aktualnie leci):
  - Jeśli `obs[2] == 1.0` (velocity > 0): kolce z prawej ściany (east_spikes)
  - Jeśli `obs[2] == -1.0` (velocity < 0): kolce z lewej ściany (west_spikes)
- Maksymalnie 9 kolców jednocześnie
- Pozycje Y kolców są w pikselach od góry ekranu

### 3.3. STAN GRY (game_state: dict)

Słownik przekazywany do funkcji `calculate_reward()` zawierający pełne informacje o stanie gry:

```python
{
    "player_pos_x": int,           # Pozycja X gracza w pikselach
    "player_pos_y": int,           # Pozycja Y gracza w pikselach
    "player_direction": int,       # Kierunek: 1 (prawo) lub -1 (lewo)
    "player_gravity": int,         # Aktualna wartość grawitacji
    "spikes_pos_y": List[int],     # Lista pozycji Y kolców na docelowej ścianie
    "coin_x": int,                 # Pozycja X monety lub -1 (brak monety)
    "coin_y": int,                 # Pozycja Y monety lub -1 (brak monety)
    "score": int,                  # Liczba odbić od ściany (wall hits)
    "collected_coins": int,        # Łączna liczba zebranych monet
    "player_dead": bool,           # Czy gracz zginął (True po kolizji z kolcem)
    "right_wall_pos": int,         # Pozycja prawej ściany (576)
    "left_wall_pos": int,          # Pozycja lewej ściany (0)
    "floor_pos": int,              # Pozycja podłogi (704)
    "ceiling_pos": int             # Pozycja sufitu (0)
}
```

#### Szczegółowy opis kluczy:

| Klucz | Typ | Zakres | Opis |
|-------|-----|--------|------|
| `player_pos_x` | `int` | $[0, 576]$ | Pozycja X środka gracza |
| `player_pos_y` | `int` | $[0, 704]$ | Pozycja Y środka gracza |
| `player_direction` | `int` | $\{-1, 1\}$ | Kierunek ruchu poziomego |
| `player_gravity` | `int` | $\mathbb{Z}$ | Wartość grawitacji (dodatnia = spadanie) |
| `spikes_pos_y` | `List[int]` | - | Pozycje Y kolców z docelowej ściany |
| `coin_x` | `int` | $[40, 536]$ lub $-1$ | Pozycja X monety |
| `coin_y` | `int` | $[40, 664]$ lub $-1$ | Pozycja Y monety |
| `score` | `int` | $[0, \infty)$ | Liczba udanych odbić od ściany |
| `collected_coins` | `int` | $[0, \infty)$ | Liczba zebranych monet w grze |
| `player_dead` | `bool` | $\{True, False\}$ | Status życia gracza |
| `right_wall_pos` | `int` | $576$ | Stała pozycja prawej ściany |
| `left_wall_pos` | `int` | $0$ | Stała pozycja lewej ściany |
| `floor_pos` | `int` | $704$ | Stała pozycja podłogi |
| `ceiling_pos` | `int` | $0$ | Stała pozycja sufitu |

---

## 4. PARAMETRY GRY

### Wymiary i rendering

| Parametr | Wartość | Opis |
|----------|---------|------|
| Szerokość okna | $576$ px | `288 * 2` (scale=2) |
| Wysokość okna | $704$ px | `352 * 2` (scale=2) |
| FPS | $60$ | Klatki na sekundę |
| Scale | $2$ | Mnożnik skali grafiki |

### Obiekty w grze

| Obiekt | Maksymalna liczba | Zasady generacji |
|--------|-------------------|------------------|
| Monety | $1$ | gdy brak monety generuje monetę po odbiciu |
| Kolce na ścianie | $9$ | $2 + \lfloor score / 5 \rfloor$, max 9 |
| Kolce podłoga/sufit | Pełne pokrycie | Statyczne przez całą grę |

### Mechanika kolców

Liczba kolców na docelowej ścianie:

$$n_{spikes} = \min\left(2 + \left\lfloor \frac{score}{5} \right\rfloor, 9\right)$$

gdzie $score$ to liczba odbić od ściany.

### Kolizje

| Typ kolizji | Efekt |
|-------------|-------|
| Ściana (lewa/prawa) | $score = score + 1$, zmiana kierunku, generacja kolców |
| Kolec | Śmierć gracza, koniec gry |
| Moneta | $collected\_coins = collected\_coins + 1$ |


## 5. WYMAGANIA TECHNICZNE

Wszyskie potrzebne paczki są zawarte w `requirements.txt`. Upewnij się, że masz zainstalowane:
- Python 3.8+
- `numpy`
- `gymnasium`
- `pygame`

## 6. METRYKI OCENY
Wydajność modelu jest zadana następująco:

$$\text{performance}_i = \text{collected coins}_i + \text{score}_i$$
$$\text{performance} = \frac{1}{N}\sum_{i=1}^{N} \text{performance}_i$$

gdzie $N$ to liczba epizodów w benchmarku, $collected\_coins_i$ to liczba monet zebranych w epizodzie $i$, a $score_i$ to liczba odbić od ściany w epizodzie $i$. Dla każdej implementacji i-ty epizod bedzie używał tego samego ziarna losowego.

**UWAGA!** skuteczność bota oceniamy do czasu aż wykona **500** odbić. Sam bot powinien myśleć nie dłużej niż **1 sekundę** przed wykonaniem akcji. Celowe użycie `sleep` lub spowolnienie bota innymi metodami skutkuje dyskwalifikacją.

---

## 7. PRZYKŁADOWE IMPLEMENTACJE W `solution.py`

Poniżej przedstawiamy przykładowe implementacje botów, które możecie wykorzystać jako punkt startowy. Pamiętajcie, że waszym celem jest stworzenie bota, który osiągnie jak najwyższy wynik!

### A. Bot Heurystyczny 

Ten bot podejmuje decyzję o skoku na podstawie prostej heurystyki: skacze, gdy znajduje się poniżej określonego progu Y.

```python
from base_bot import BaseBot
import numpy as np
from typing import Dict, Any

# Opcjonalne stałe konfiguracyjne
BENCHMARK_EPISODES = 10  # Mniejsza liczba dla szybszego testu
WATCH_GAME = True  # Wizualizacja gry

# --- Implementacja Bota ---

class SimpleHeuristicBot(BaseBot):
    """
    Bot, który skacze, gdy jest zbyt nisko.
    """
    def __init__(self, jump_y_threshold: float = 500.0):
        # Przykładowy próg Y. Wartość 500.0 jest testowa dla user_y=704
        self.jump_y_threshold = jump_y_threshold 

    def take_action(self, obs: np.ndarray) -> int:
        """
        Indeksy obserwacji:
        0: p_x (położenie X gracza)
        1: p_y (położenie Y gracza)
        ...
        """
        player_y = obs[1]
        
        if player_y > self.jump_y_threshold:
            return 1  # Skocz
        else:
            return 0  # Nie skacz

def create_bot() -> BaseBot:
    """
    Fabryka dla bota.
    """
    return SimpleHeuristicBot() 

# --- Implementacja Nagrody (Wymagana dla RL, opcjonalna dla heurystyki, ale musi istnieć) ---

def calculate_reward(game_state: Dict[str, Any]) -> float:
    """
    Funkcja nagrody (Reward Function).
    """
    reward = 0.0

    # 1. Nagroda za przetrwanie (za każdy krok)
    reward += 0.1 

    # 2. Kara za śmierć
    if game_state["player_dead"]:
        reward -= 10.0
    
    return reward
```

### B. Bot z Uproszczonym Q-Learningiem (Epsilon-Greedy)

Używa słownika jako Q-tabeli i strategii $\epsilon$-greedy.

```python
from base_bot import BaseBot
import numpy as np
from typing import Dict, Any

class SimpleEpsilonQLearningBot(BaseBot):
    def __init__(self):
        self.q_table = {} 
        self.last_state = None
        self.last_action = None
        self.lr = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1 # Współczynnik eksploracji

    def take_action(self, obs: np.ndarray) -> int:
        # Stan: (sektor X, sektor Y) - dyskretyzacja
        state = (int(obs[0] // 50), int(obs[1] // 50))

        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]

        if np.random.random() < self.epsilon:
            action = np.random.choice([0, 1]) # Eksploracja
        else:
            action = int(np.argmax(self.q_table[state])) # Eksploatacja
        
        self.last_state = state
        self.last_action = action
        
        return action
        
    def learn(self, next_obs: np.ndarray, reward: float, done: bool):
        """Uproszczona funkcja nauki (poza cyklem turniejowym)"""
        if self.last_state is None: return
        
        state = self.last_state
        action = self.last_action
        
        # Wyznaczanie stanu następnego
        next_state = (int(next_obs[0] // 50), int(next_obs[1] // 50))
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0, 0.0]

        old_q = self.q_table[state][action]
        max_next_q = 0 if done else np.max(self.q_table[next_state])
        
        # Wzór Q-learningu: Q(s,a) = Q(s,a) + alpha * (R + gamma * max(Q(s',a')) - Q(s,a))
        new_q = old_q + self.lr * (reward + self.gamma * max_next_q - old_q)
        self.q_table[state][action] = new_q

def create_bot() -> BaseBot:
    return SimpleEpsilonQLearningBot()

def calculate_reward(game_state: Dict[str, Any]) -> float:
    return -10.0 if game_state["player_dead"] else 0.1
 ```

### C. Bot typu REINFORCE (Policy Gradient - PyTorch)

Wykorzystuje minimalną sieć neuronową w PyTorch. 

```python
from base_bot import BaseBot
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
# Wymaga GameEnv
from src.env import GameEnv 

class PolicyNet(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, output_size=2):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, output_size), nn.Softmax(dim=-1))
    def forward(self, x):
        return self.net(x)

class ReinforceTorchBot(BaseBot):
    def __init__(self):
        self.policy_net = PolicyNet()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.01)
        self.log_probs = []
        self.rewards = []
        self.gamma = 0.99

    def take_action(self, obs: np.ndarray) -> int:
        x_np = np.zeros(15, dtype=np.float32); x_np[:len(obs)] = obs
        state_tensor = torch.from_numpy(x_np).float()
        
        probs = self.policy_net(state_tensor)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        
        self.log_probs.append(action_dist.log_prob(action))
        return action.item()

    def train(self):
        """Aktualizacja wag sieci na podstawie zebranych nagród (REINFORCE)."""
        R = 0; returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns); 
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = [-log_prob * R for log_prob, R in zip(self.log_probs, returns)]
            
        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        self.log_probs = []; self.rewards = []
        return loss.item()

def create_bot() -> BaseBot:
    return ReinforceTorchBot()

def calculate_reward(game_state: Dict[str, Any]) -> float:
    return -10.0 if game_state["player_dead"] else 0.1

def train_bot(episodes: int = 500):
    """Przykładowa pętla treningowa REINFORCE."""
    bot = create_bot()
    env = GameEnv(calculate_reward=calculate_reward, render_mode="headless") 

    for i in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = bot.take_action(obs)
            obs, reward, done, _, info = env.step(action)
            
            bot.rewards.append(reward)
            episode_reward += reward

        loss = bot.train()
        if (i + 1) % 50 == 0:
             print(f"Episode {i+1}/{episodes}, Reward: {episode_reward:.2f}, Loss: {loss:.4f}")
    env.close()

# Poniższe wywołanie w main.py pozwoli na trening bota
# if __name__ == "__main__":
#     train_bot(episodes=5000)
```
