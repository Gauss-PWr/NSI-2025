from abc import ABC, abstractmethod


class BaseBot(ABC):
    """
    Klasa abstrakcyjna dla botów uczestników.
    Uczestnicy mają dziedziczyć po tej klasie
    """
    
    @abstractmethod
    def take_action(self, obs) -> int:
        """
        Funkcja przyjmuje stan gry i zwraca akcję ).
        """
        raise NotImplementedError
