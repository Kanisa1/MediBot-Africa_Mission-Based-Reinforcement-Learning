import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../environment'))
from custom_env import MalariaDefenseEnv

os.makedirs("../models/pg", exist_ok=True)

def train_agent(agent_class, hyperparams, total_timesteps, save_path, eval_episodes=10):
    env = MalariaDefenseEnv()
    eval_env = MalariaDefenseEnv()
    
    results = []
    best_model = None
    best_reward = -float('inf')
    
    for i, params in enumerate(hyperparams):
        print(f"\nTraining {agent_class.__name__} configuration {i+1}/{len(hyperparams)}")
        print(f"Params: {params}")
        
        model = agent_class("MlpPolicy", env, **params, verbose=0)
        model.learn(total_timesteps=total_timesteps)
        
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes)
        results.append({'config': i+1, 'params': params, 'mean_reward': mean_reward, 'std_reward': std_reward})
        
        print(f"Mean reward: {mean_reward:.2f} (+/- {std_reward:.2f})")
        
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_model = model
            model.save(save_path)
    
    np.save(save_path.replace(".zip", "_results.npy"), results)
    with open(save_path.replace(".zip", "_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    return results, best_model

if __name__ == "__main__":
    # PPO hyperparameters
    ppo_hyperparams = [
        {'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64, 'clip_range': 0.2},
        # Add the rest...
    ]
    
    print("Training PPO...")
    ppo_results, best_ppo = train_agent(PPO, ppo_hyperparams, total_timesteps=100000, save_path="../models/pg/best_ppo_model.zip")
    
    # A2C hyperparameters
    a2c_hyperparams = [
        {'learning_rate': 7e-4, 'n_steps': 5},
        # Add the rest...
    ]
    
    print("\nTraining A2C...")
    a2c_results, best_a2c = train_agent(A2C, a2c_hyperparams, total_timesteps=75000, save_path="../models/pg/best_a2c_model.zip")
    
    print("\nPolicy Gradient Training Complete!")
    print(f"Best PPO reward: {max([r['mean_reward'] for r in ppo_results]):.2f}")
    print(f"Best A2C reward: {max([r['mean_reward'] for r in a2c_results]):.2f}")
