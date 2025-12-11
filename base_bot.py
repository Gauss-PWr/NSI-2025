from abc import ABC, abstractmethod
from src.env import GameEnv


class BaseBot(ABC):
    def __init__(self, env: GameEnv) -> None:
        super().__init__()
        self.env = env
        self.model = None

    @abstractmethod
    def take_action(self, game_state) -> int:
        """
        Funkcja przymuję stan gry i zwraca akcję: 0 - nic, 1 - skok.
        """

        raise NotImplementedError

    def calculate_reward(self, game_state) -> float | None:
        """
        Zdefiniuj funkcję nagrody dla bota na podstawie stanu gry. Dla heurestyk zwróć None.
        """
        reward = 0.1  # Bazowa nagroda za przetrwanie

        # Nagroda za odbicie się od ściany
        if game_state.get("wall_hit", 0) != 0:
            reward += 1.0

        # Kara za śmierć
        if game_state.get("spike_collision", False):
            reward = -10.0
            return reward

        # Nagroda za monety
        coins_collected = game_state.get("coins_collected", 0)
        reward += coins_collected * 0.5

        return reward

    @abstractmethod
    def load_model(self, model_path: str = "model") -> object | None:
        """
        Jeśli bot korzysta z modelu uczenia maszynowego, załaduj go tutaj.

        """
        pass
