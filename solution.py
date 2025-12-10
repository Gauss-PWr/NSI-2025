
import os
from stable_baselines3 import PPO

def train_bot(epochs, env):
    """
    Trenuje bota [wstawić ładny opis]
    """
    pass

def load_bot(path):
    """
    Ładuje bota [Dodać ładny opis]
    """
    pass

def calculate_reward(game_state):
    """
    Funkcja do liczenia nagrody [TU WSTAWIĆ DOKŁADNY OPIS] [Do hackhatonu wstawić pokazową funkcję nagrody]
    UWAGA!!! game_state nie wysyła poprawnie ilości monet (jeszcze)
    """
    reward = 0.1  # Bazowa nagroda za przetrwanie
    
    # Nagroda za odbicie się od ściany
    if game_state.get('wall_hit', 0) != 0:
        reward += 1.0
    
    # Kara za śmierć
    if game_state.get('spike_collision', False):
        reward = -10.0
        return reward
    
    # Nagroda za monety
    coins_collected = game_state.get('coins_collected', 0)
    reward += coins_collected * 0.5
    
    return reward