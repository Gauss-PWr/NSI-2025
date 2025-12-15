import numpy as np
from submissions.corgasy.solution import create_bot as CongrasyBot
from submissions.dominik.solution import create_bot as DominikBot
from submissions.krupa.solution import create_bot as KrupaBot
from submissions.michal.solution import create_bot as MichalBot
from submissions.samuel.solution import create_bot as SamuelBot
from tqdm import tqdm
from src.env import GameEnv

np.random.seed(100)

random_seeds = np.random.randint(0, 10000, size = 1000)


bots = {
    "Congrasy": CongrasyBot(),
    "Dominik": DominikBot(),
    "Krupa": KrupaBot(),
    "Michal": MichalBot(),
    "Samuel": SamuelBot(),
}



for bot_name, bot in bots.items():
    print(f"Evaluating bot: {bot_name}")
    total_score = []
    for seed in tqdm(random_seeds):
        env = GameEnv(render_mode='headless', game_seed=seed)        
        obs, _ = env.reset()
        done = False
        total_reward = 0
        current_score = 0
        
        while not done:
            action = bot.take_action(obs)
            obs, reward, done, _, info = env.step(action)
            
            # We can track the actual game score if exposed in info or calculate from engine
            # Since GameEnv returns game score in step's reward logic or we can access env.game.score
            if done:
                total_score.append(env.game.score + env.game.coin_total)

    print(f"Mean score for {bot_name}: {sum(total_score) / len(random_seeds)}")
    print(f"Max score for {bot_name}: {max(total_score)}")
    print(f"Min score for {bot_name}: {min(total_score)}")