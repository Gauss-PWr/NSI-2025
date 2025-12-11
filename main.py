import sys
from src.env import GameEnv
from src.tournament import Tournament
from example_solution import (
    ExampleBot,
)  # Zakładamy, że bot jest zdefiniowany w solution.py
from base_bot import BaseBot

if __name__ == "__main__":
    try:

        bot = ExampleBot(GameEnv())  # Inicjalizacja bota z środowiskiem gry
        if not issubclass(type(bot), BaseBot):
            raise TypeError("Bot musi dziedziczyć po BaseBot")

        bot.load_model()  # Załaduj model bota, jeśli dotyczy

        # Uruchom turniej
        tournament = Tournament(bot)
        tournament.run_benchmark()

    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"BŁĄD: {e}")
        sys.exit(1)
