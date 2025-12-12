import os
import numpy as np
import gymnasium as gym
import pygame
from src.env import GameEnv

class Tournament:
    def __init__(self, bot):
        self.bot = bot
        self.results = []

    def _setup_headless(self):
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    def _clear_headless(self):
        if "SDL_VIDEODRIVER" in os.environ:
            del os.environ["SDL_VIDEODRIVER"]

    def run_benchmark(self, episodes=100):
        if not self.bot: return

        self._setup_headless()
        env = GameEnv(bot=self.bot, render_mode='headless')
        
        print(f"Starting Benchmark over {episodes} episodes...")
        scores = []
        
        for i in range(episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            current_score = 0
            
            while not done:
                action, _ = self.bot.take_action(obs)
                obs, reward, done, _, info = env.step(action)
                
                # We can track the actual game score if exposed in info or calculate from engine
                # Since GameEnv returns game score in step's reward logic or we can access env.game.score
                if done:
                    current_score = env.game.score
            
            scores.append(current_score)
            
        env.close()
        
        self.results = scores
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        
        print(f"Benchmark Complete.")
        print(f"Average Score: {avg_score:.2f}")
        print(f"Max Score: {max_score}")
        print(f"Min Score: {np.min(scores)}")
        
        return scores

    def watch_game(self):
        if not self.bot: return

        self._clear_headless()
        pygame.quit() 
        
        env = GameEnv(render_mode='human')
        obs, _ = env.reset()
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            action, _ = self.bot.take_action(obs)
            obs, _, done, _, _ = env.step(action)
            
            if done:
                print(f"Game Finished. Score: {env.game.score}")
                obs, _ = env.reset()
                
        env.close()
