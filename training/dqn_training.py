import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../environment'))
from custom_env import MalariaDefenseEnv

def train_dqn():
    """Train DQN agent with hyperparameter tuning"""
    
    # Create directories
    os.makedirs("../models/dqn", exist_ok=True)
    
    # Create environment
    env = MalariaDefenseEnv()
    eval_env = MalariaDefenseEnv()
    
    # Hyperparameter configurations
    hyperparams = [
        {'learning_rate': 1e-3, 'buffer_size': 10000, 'exploration_fraction': 0.1},
        {'learning_rate': 5e-4, 'buffer_size': 50000, 'exploration_fraction': 0.15},
        {'learning_rate': 1e-4, 'buffer_size': 100000, 'exploration_fraction': 0.2},
        {'learning_rate': 1e-3, 'buffer_size': 50000, 'exploration_fraction': 0.3},
        {'learning_rate': 5e-4, 'buffer_size': 20000, 'exploration_fraction': 0.25},
        {'learning_rate': 2e-4, 'buffer_size': 75000, 'exploration_fraction': 0.1},
        {'learning_rate': 1e-3, 'buffer_size': 30000, 'exploration_fraction': 0.2},
        {'learning_rate': 7e-4, 'buffer_size': 40000, 'exploration_fraction': 0.15},
        {'learning_rate': 3e-4, 'buffer_size': 60000, 'exploration_fraction': 0.35},
        {'learning_rate': 1e-4, 'buffer_size': 80000, 'exploration_fraction': 0.25}
    ]
    
    results = []
    best_model = None
    best_reward = -float('inf')
    
    for i, params in enumerate(hyperparams):
        print(f"\nTraining DQN configuration {i+1}/10")
        print(f"Params: {params}")
        
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=params['learning_rate'],
            buffer_size=params['buffer_size'],
            exploration_fraction=params['exploration_fraction'],
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            target_update_interval=1000,
            learning_starts=1000,
            train_freq=4,
            verbose=0
        )
        
        model.learn(total_timesteps=50000)
        
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        results.append({'config': i+1, 'params': params, 'mean_reward': mean_reward, 'std_reward': std_reward})
        
        print(f"Mean reward: {mean_reward:.2f} (+/- {std_reward:.2f})")
        
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_model = model
            model.save("../models/dqn/best_dqn_model")
    
    np.save("../models/dqn/training_results.npy", results)
    with open("../models/dqn/training_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    return results, best_model

if __name__ == "__main__":
    results, best_model = train_dqn()
    print("\nDQN Training Complete!")
    print(f"Best configuration achieved mean reward: {max([r['mean_reward'] for r in results]):.2f}")
