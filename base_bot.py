from abc import ABC, abstractmethod


class BaseBot(ABC):
    """
    Klasa abstrakcyjna dla botów uczestników.
    Uczestnicy mają dziedziczyć po tej klasie
    """
    
    @abstractmethod
    def take_action(self, game_state) -> tuple[int, dict]:
        """
        Funkcja przyjmuje stan gry i zwraca akcję + dodatkowe info [może być puste]).
        """
        raise NotImplementedError

    def calculate_reward(self, game_state: dict) -> float:
        """
        Zdefiniuj funkcję nagrody dla bota na podstawie stanu gry.
        """
        reward = 0

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