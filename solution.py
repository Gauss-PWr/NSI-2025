import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from base_bot import BaseBot
from src.env import GameEnv
import os

# --- KONFIGURACJA ---
MODEL_PATH = "ppo_spike_bot"
BENCHMARK_EPISODES = 20
WATCH_GAME = True

# Stałe do normalizacji (bazowane na engine.py: scale=2, base=288x352)
SCREEN_WIDTH = 288 * 2
SCREEN_HEIGHT = 352 * 2


class RewardManager:
    """
    Klasa pomocnicza do śledzenia stanu gry w celu obliczania poprawnej nagrody.
    Rozwiązuje problem braku "poprzedniego stanu" w prostej funkcji calculate_reward.
    """

    def __init__(self):
        self.prev_score = 0
        self.prev_coins = 0

    def calculate(self, state):
        reward = 0.0

        # 1. Nagroda za przeżycie (mała, stała)
        # Zachęca bota do unikania śmierci.
        reward += 0.1

        # 2. Nagroda za zdobycie punktu (uderzenie w ścianę)
        current_score = state.get("score", 0)
        if current_score > self.prev_score:
            reward += 10.0  # Duży bonus za punkt
            self.prev_score = current_score

        # 3. Nagroda za zebranie monety
        current_coins = state.get("collected_coins", 0)
        if current_coins > self.prev_coins:
            reward += 5.0
            self.prev_coins = current_coins

        # 4. Kara za śmierć
        if state.get("player_dead", False):
            reward -= 20.0
            # Resetujemy liczniki przy śmierci
            self.prev_score = 0
            self.prev_coins = 0

        return reward


# Instancja menedżera nagród (singleton na potrzeby importu w main.py)
reward_manager = RewardManager()


def calculate_reward(state):
    """
    Funkcja wymagana przez engine.py.
    """
    return reward_manager.calculate(state)


def normalize_obs(obs):
    """
    Ręczna normalizacja obserwacji.
    Surowe dane z env.py to piksele (np. 300.0, 600.0).
    Sieci neuronowe "wariują" przy takich wartościach. Sprowadzamy je do [0, 1] lub [-1, 1].
    """
    # Kopia, żeby nie modyfikować oryginału
    norm = np.copy(obs)

    # Indeksy z env.py:
    # 0: player_x, 1: player_y, 2: velocity_sign, 3: gravity
    # 4,5: coin_x, coin_y
    # 6-14: spikes_y

    # Normalizacja gracza
    norm[0] = norm[0] / SCREEN_WIDTH
    norm[1] = norm[1] / SCREEN_HEIGHT

    # Normalizacja grawitacji (orientacyjnie max gravity to około 20-30)
    norm[3] = norm[3] / 30.0

    # Normalizacja monety
    norm[4] = norm[4] / SCREEN_WIDTH
    norm[5] = norm[5] / SCREEN_HEIGHT

    # Normalizacja kolców (wszystkie od indeksu 6 do końca to współrzędne Y)
    # Wartości -1.0 (brak kolca) zostawiamy lub zmieniamy na -0.1
    for i in range(6, len(norm)):
        if norm[i] > -1:
            norm[i] = norm[i] / SCREEN_HEIGHT

    return norm


class PPOBot(BaseBot):
    def __init__(self):
        # Sprawdzamy czy model istnieje, jeśli nie - trenujemy go "na szybko" lub rzucamy błąd
        if not os.path.exists(f"{MODEL_PATH}.zip"):
            print("Model nie znaleziony! Rozpoczynam trening...")
            train_model()

        self.model = PPO.load(MODEL_PATH)
        print("PPO Bot załadowany.")

    def take_action(self, obs) -> int:
        # WAŻNE: Musimy znormalizować obserwację tak samo jak podczas treningu!
        normalized_obs = normalize_obs(obs)

        # predict zwraca (akcja, stan_ukryty)
        action, _ = self.model.predict(normalized_obs, deterministic=True)
        return int(action)


def create_bot():
    """Fabryka bota wymagana przez main.py"""
    return PPOBot()


# --- CZĘŚĆ TRENINGOWA ---


class NormalizeWrapper(gym.ObservationWrapper):
    """Wrapper Gym, który automatycznie normalizuje obserwacje podczas treningu."""

    def __init__(self, env):
        super().__init__(env)
        # Aktualizujemy definicję przestrzeni obserwacji (teraz wartości są małe)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, obs):
        return normalize_obs(obs)


def train_model():
    """Funkcja trenująca model."""
    print("Rozpoczynam trening PPO...")

    # Tworzymy środowisko treningowe
    # WAŻNE: Używamy naszej funkcji calculate_reward
    env = GameEnv(calculate_reward=calculate_reward, render_mode="headless")

    # Owijamy środowisko w normalizator
    env = NormalizeWrapper(env)

    # Inicjalizacja modelu PPO
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01,  # Współczynnik entropii - zachęca do eksploracji na początku
    )

    # Trening (dla testu krótki, w praktyce ustaw np. 100,000 lub 1,000,000 kroków)
    # Sugerowana liczba kroków dla dobrego wyniku w tej grze: ~200k - 500k
    TIME_STEPS = 100000
    model.learn(total_timesteps=TIME_STEPS)

    model.save(MODEL_PATH)
    print(f"Model zapisany jako {MODEL_PATH}")
    env.close()


if __name__ == "__main__":
    # Jeśli uruchomisz plik bezpośrednio (python solution.py), rozpocznie się trening
    train_model()
