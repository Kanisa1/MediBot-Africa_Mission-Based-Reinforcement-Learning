import os
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from environment.medical_delivery_env import MedicalDeliveryEnv

def train_all_algorithms():
    """Train multiple algorithms for comparison"""
    print("=== COMPREHENSIVE ALGORITHM TRAINING ===")
    
    algorithms = {
        'PPO': {
            'class': PPO,
            'params': {
                'learning_rate': 2.5e-4,
                'n_steps': 2048,
                'batch_size': 128,
                'n_epochs': 15,
                'gamma': 0.995,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'max_grad_norm': 0.8
            }
        },
        'A2C': {
            'class': A2C,
            'params': {
                'learning_rate': 7e-4,
                'n_steps': 16,
                'gamma': 0.99,
                'gae_lambda': 1.0,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.8
            }
        },
        'DQN': {
            'class': DQN,
            'params': {
                'learning_rate': 1e-4,
                'buffer_size': 100000,
                'batch_size': 32,
                'gamma': 0.99,
                'exploration_final_eps': 0.05
            }
        }
    }
    
    results = {}
    
    for algo_name, algo_config in algorithms.items():
        print(f"\n🎯 Training {algo_name}...")
        
        # Create environment
        env = MedicalDeliveryEnv(render_mode=None)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        
        # Create model
        model = algo_config['class'](
            "MlpPolicy",
            env,
            tensorboard_log=f"./logs/{algo_name.lower()}/",
            verbose=1,
            **algo_config['params']
        )
        
        # Train with reduced timesteps for demo
        total_timesteps = 50000  # Reduced for quick demo
        
        model.learn(total_timesteps=total_timesteps, tb_log_name=f"{algo_name}_Medical_Drone")
        
        # Save
        os.makedirs(f"models/{algo_name.lower()}", exist_ok=True)
        model.save(f"models/{algo_name.lower()}/medical_drone_{algo_name.lower()}")
        
        # Evaluate
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
        results[algo_name] = {
            'mean_reward': mean_reward,
            'std_reward': std_reward
        }
        
        print(f"{algo_name} - Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        env.close()
    
    # Print results
    print("\n=== FINAL RESULTS ===")
    for algo, result in results.items():
        print(f"{algo}: {result['mean_reward']:.2f} +/- {result['std_reward']:.2f}")
    
    # Determine best algorithm
    best_algo = max(results.items(), key=lambda x: x[1]['mean_reward'])
    print(f"\n🏆 BEST ALGORITHM: {best_algo[0]} with {best_algo[1]['mean_reward']:.2f} mean reward")
    
    return results

if __name__ == "__main__":
    train_all_algorithms()

print(" Comprehensive training script created!")
