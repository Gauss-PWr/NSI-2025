from abc import ABC, abstractmethod


class BaseBot(ABC):
    """
    Klasa abstrakcyjna dla botów uczestników.
    Uczestnicy mają dziedziczyć po tej klasie
    """
    
    @abstractmethod
    def take_action(self, obs) -> tuple[int, dict]:
        """
        Funkcja przyjmuje stan gry i zwraca akcję + dodatkowe info [może być puste]).
        """
        raise NotImplementedError

    def calculate_reward(self, game_state: dict) -> float:
        """
        Zdefiniuj funkcję nagrody dla bota na podstawie stanu gry.
        """
        reward = 0.1

        # Kara za śmierć
        if game_state.get("player_dead", False):
            reward = -10.0
            return reward
            
        # Nagroda za monety
        coins_collected = game_state.get("collected_coins", 0)
        reward += coins_collected * 0.5

        return reward