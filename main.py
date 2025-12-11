import sys
from src.env import GameEnv
from src.tournament import Tournament


class BotInterface:
    """Interfejs zarządzający botem"""
    
    def __init__(self):
        self.bot_type = self._detect_bot_type()
        
        # Ładuj config tylko dla AI (heurystyka go nie potrzebuje)
        if self.bot_type == "AI":
            self.config = self._load_config()
            self.model = self._initialize_ai_bot()
        else:
            self.model = self._initialize_heuristic_bot()
    
    def _load_config(self):
        """Ładuje konfigurację z solution.py (tylko dla AI)"""
        try:
            from solution import TRAIN_NEW_MODEL, MODEL_PATH, TRAINING_TIMESTEPS
            return {
                'TRAIN_NEW_MODEL': TRAIN_NEW_MODEL,
                'MODEL_PATH': MODEL_PATH,
                'TRAINING_TIMESTEPS': TRAINING_TIMESTEPS
            }
        except ImportError as e:
            print(f"BŁĄD: Bot AI wymaga zmiennych: TRAIN_NEW_MODEL, MODEL_PATH, TRAINING_TIMESTEPS")
            print(f"Szczegóły: {e}")
            sys.exit(1)
    
    def _detect_bot_type(self):
        """Automatycznie wykrywa typ bota na podstawie zaimplementowanych funkcji."""
        import solution
        
        has_ai = all(hasattr(solution, f) for f in ['train_bot', 'load_bot', 'calculate_reward'])
        has_heuristic = hasattr(solution, 'heuristic_bot')
        
        if has_ai:
            return "AI"
        elif has_heuristic:
            return "HEURISTIC"
        else:
            print("BŁĄD: Nie wykryto żadnych funkcji bota w solution.py!")
            sys.exit(1)
    
    def _initialize_ai_bot(self):
        """Inicjalizuje bota AI"""
        from solution import train_bot, load_bot
        
        if self.config['TRAIN_NEW_MODEL']:
            print(f"Trenowanie nowego modelu ({self.config['TRAINING_TIMESTEPS']} kroków)...")
            model = train_bot(self.config['TRAINING_TIMESTEPS'], GameEnv)
            print(f"Model zapisany jako '{self.config['MODEL_PATH']}'")
        else:
            print(f"Ładowanie modelu z '{self.config['MODEL_PATH']}'...")
            model = load_bot(self.config['MODEL_PATH'])
            print("Model załadowany")
        
        return model
    
    def _initialize_heuristic_bot(self):
        """Inicjalizuje bota heurystycznego"""
        from solution import heuristic_bot
        return heuristic_bot
    
    def take_action(self, observation):
        """Podejmuje akcję na podstawie obserwacji"""
        if self.bot_type == "AI":
            if hasattr(self.model, 'predict'):
                action, _ = self.model.predict(observation, deterministic=True)
                return action, None
            elif hasattr(self.model, 'take_action'):
                return self.model.take_action(observation)
            else:
                raise ValueError("Model AI musi mieć metodę 'predict' lub 'take_action'")
        else:  # HEURISTIC
            action = self.model(observation)
            if action not in [0, 1]:
                raise ValueError(f"heuristic_bot zwrócił1 nieprawidłową akcję: {action}")
            return action, None


if __name__ == "__main__":
    try:
        bot_interface = BotInterface()
        
        # Wczytaj ustawienia turnieju
        try:
            from solution import BENCHMARK_EPISODES, WATCH_GAME
        except ImportError:
            BENCHMARK_EPISODES = 50
            WATCH_GAME = True
        
        # Uruchom turniej
        tournament = Tournament(bot_interface)
        tournament.run_benchmark(episodes=BENCHMARK_EPISODES)
        
        if WATCH_GAME:
            tournament.watch_game()
    
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"BŁĄD: {e}")
        sys.exit(1)