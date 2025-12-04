import gymnasium as gym
import numpy as np
from gymnasium import spaces
from src.game.engine import GameEngine

class GameEnv(gym.Env):
    metadata = {'render_modes': ['human', 'headless']}

    def __init__(self, render_mode='headless'):
        super(GameEnv, self).__init__()
        self.headless = (render_mode == 'headless')
        self.game = GameEngine(headless=self.headless)
        self.action_space = spaces.Discrete(2)
        
        self.max_coins = 1
        self.max_spikes = 9 
        
        # Obs Vector Structure:
        # 1. Player: [X, Y, Y-Vel, X-Vel] (4 floats)
        # 2. Coin: [X, Y] (2 floats)
        # 3. Spikes: [Y, Y, Y...] (10 floats - only Y pos of target wall)
        obs_size = 4 + (self.max_coins * 2) + (self.max_spikes * 1)
        
        low = np.full((obs_size,), -float('inf'), dtype=np.float32)
        high = np.full((obs_size,), float('inf'), dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset_game()
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        _, reward, done, _ = self.game.step(action)
        observation = self._get_obs()
        info = {}
        truncated = False 
        if not self.headless:
            self.game.render_frame()
        return observation, reward, done, truncated, info

    def render(self):
        self.game.render_frame()

    def close(self):
        pass

    def _get_obs(self):
        p = self.game.player
        
        player_data = [float(p.rect.x), float(p.rect.y), float(p.gravity), float(p.velocity)]
        
        coin_data = []
        if self.game.coin_list:
            c = self.game.coin_list[0]
            coin_data = [float(c.rect.x), float(c.rect.y)]
        else:
            coin_data = [-1.0, -1.0]
            
        target_spikes = []
        if p.velocity > 0:
            target_spikes = self.game.east_spikes
        else:
            target_spikes = self.game.west_spikes
        
        spike_data = []
        for spike in target_spikes:
            spike_data.append(float(spike.rect.y))
            
        if len(spike_data) > self.max_spikes:
            spike_data = spike_data[:self.max_spikes]
        else:
            padding = self.max_spikes - len(spike_data)
            spike_data.extend([-1.0] * padding)

        obs = np.array(player_data + coin_data + spike_data, dtype=np.float32)
        return obs