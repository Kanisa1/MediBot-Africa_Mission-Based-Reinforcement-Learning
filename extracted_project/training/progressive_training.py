import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from environment.medical_delivery_env import MedicalDeliveryEnv

def progressive_training():
    """Train progressively through all phases"""
    print(" Starting Progressive Training Pipeline...")
    
    os.makedirs("models/progressive", exist_ok=True)
    
    # Train through each phase
    for phase in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"STARTING TRAINING PHASE {phase}")
        print(f"{'='*60}")
        
        # Create environment for current phase
        env = MedicalDeliveryEnv(render_mode=None, training_phase=phase)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        
        # Load previous model if available, else create new one
        if phase > 1:
            try:
                model = PPO.load(f"models/progressive/phase{phase-1}_final", env=env)
                print(f" Loaded model from Phase {phase-1}")
            except:
                print(" Could not load previous model, starting fresh...")
                model = PPO("MlpPolicy", env, verbose=1)
        else:
            model = PPO("MlpPolicy", env, verbose=1)
        
        # Train for current phase
        phase_timesteps = 200000 if phase == 1 else 150000  # More time for phase 1
        model.learn(total_timesteps=phase_timesteps)
        
        # Save phase model
        model.save(f"models/progressive/phase{phase}_final")
        
        # Evaluate
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)
        print(f" Phase {phase} completed! Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        env.close()
    
    print("\n Progressive training completed!")
    return model

if __name__ == "__main__":
    progressive_training()
