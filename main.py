from src.env import GameEnv
from src.tournament import Tournament


class Bot:
    def __init__(self, model):
        self.model = model

    def take_action(self, observation):
        
        # modify so that it works with your model :D
        action, _ = self.model.predict(observation, deterministic=True)
        return action, None

if __name__ == "__main__":
    from solution import train_bot, load_bot
    
    train = False
    if train:
        trained_model = train_bot(epochs=1_000_000, env=GameEnv)
    else:
        trained_model = load_bot("bot")    
    
    bot = Bot(trained_model)
    tourney = Tournament(bot)
    tourney.run_benchmark(episodes=50)
    tourney.watch_game()