import sys
from src.env import GameEnv
from src.tournament import Tournament

from base_bot import BaseBot

def load_participant_bot():
    """
    Ładuje bota uczestnika z pliku solution.py.
    """
    try:
        from solution import create_bot
        bot_instance = create_bot()
        
        # Walidacja że bot dziedziczy po klasie Bot
        if not isinstance(bot_instance, BaseBot):
            raise TypeError(
                f"Bot musi dziedziczyć po klasie BaseBot!"
                f"Otrzymano: {type(bot_instance)}"
            )
        
        print(f"Bot załadowany: {bot_instance.__class__.__name__}")
        return bot_instance   
    except ImportError as e:
        print(f"BŁĄD: Nie znaleziono funkcji 'create_bot()' w solution.py")
        print(f"Szczegóły: {e}")
        print("Utwórz funkcję create_bot() która zwraca instancję Twojej klasy bota.")
        sys.exit(1)
    except Exception as e:
        print(f"BŁĄD podczas ładowania bota: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        # Załaduj bota uczestnika
        participant_bot = load_participant_bot()
        
        # Wczytaj opcjonalne ustawienia turnieju
        try:
            from solution import BENCHMARK_EPISODES
        except ImportError:
            BENCHMARK_EPISODES = 50
        try:
            from solution import WATCH_GAME
        except ImportError:
            WATCH_GAME = True        
        # Uruchom turniej
        tournament = Tournament(participant_bot)
        tournament.run_benchmark(episodes=BENCHMARK_EPISODES)
        
        if WATCH_GAME:
            tournament.watch_game()

    except Exception as e:
        print(f"BŁĄD: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)