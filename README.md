## 1. WPROWADZENIE

Celem hackathonu jest stworzenie bota do gry ***watch out for pointy things***. Gracz porusza się między lewą i prawą ścianą, unikając kolców i zbierając monety. Bot musi podejmować decyzje o skoku w odpowiednich momentach, aby przetrwać jak najdłużej.

### Wymagany plik: `solution.py`

Uczestnicy muszą stworzyć plik `solution.py` zawierający:

**Elementy wymagane:**
- Klasę bota dziedziczącą po `BaseBot`
- Metodę `take_action(self, obs) -> int` w klasie bota
- Funkcję `create_bot() -> BaseBot`

**Elementy opcjonalne:**
- `BENCHMARK_EPISODES: int` - liczba epizodów benchmarku (domyślnie: 50)
- `WATCH_GAME: bool` - czy pokazać wizualizację gry (domyślnie: True)
- Funkcję `calculate_reward(game_state: dict) -> float` - obowiązkowa przy implementacjia RL

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
| Monety | $1$ | 30% szans co iterację gdy brak monety |
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

TBA [requirements.txt i uv]

## 6. METRYKI OCENY

TBA