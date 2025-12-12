from base_bot import BaseBot

# Templatka na rozwiązanie


class MyBot(BaseBot):
    def __init__(self):
        # Inicjalizacja bota, jeśli potrzebna
        pass

    def take_action(self, obs) -> int:
        # Tutaj zaimplementuj logikę podejmowania akcji na podstawie obserwacji
        # 1 - skok, 0 - brak akcji
        return 0


def calculate_reward(game_state: dict[str, int | bool | float]) -> float:
    # Prosta funkcja nagrody - zawsze zwraca 0
    # Tutaj zdefiniuj swoją logikę nagrody na podstawie stanu gry
    # UWAGA! game_state to n
    return 0.0


def create_bot() -> BaseBot:
    """
    Funkcja tworząca i zwracająca instancję bota uczestnika.
    Musi zwracać obiekt dziedziczący po klasie BaseBot.
    """
    return MyBot()