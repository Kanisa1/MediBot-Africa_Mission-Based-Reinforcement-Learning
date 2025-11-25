import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from environment.medical_delivery_env import MedicalDeliveryEnv

class CurriculumCallback(BaseCallback):
    def __init__(self, check_freq, save_path, eval_env, verbose=1):
        super(CurriculumCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.eval_env = eval_env
        self.best_mean_reward = -np.inf
        self.consecutive_successes = 0
        self.phase = 1

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Evaluate current model
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=5)
            print(f"Step {self.n_calls}: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
            
            # Save model
            model_path = os.path.join(self.save_path, f'model_phase{self.phase}_{self.n_calls}')
            self.model.save(model_path)

            # Check for curriculum progression
            if mean_reward > 50:  # Threshold for success
                self.consecutive_successes += 1
            else:
                self.consecutive_successes = 0

            # Progress to next phase if consistently successful
            if self.consecutive_successes >= 3 and self.phase < 3:
                self.phase += 1
                self.consecutive_successes = 0
                print(f"🎓 Progressing to Training Phase {self.phase}!")
                
                # Create new environment with higher difficulty
                new_env = MedicalDeliveryEnv(render_mode=None, training_phase=self.phase)
                new_env = Monitor(new_env)
                new_env = DummyVecEnv([lambda: new_env])
                
                # Update model environment
                self.model.set_env(new_env)
                self.eval_env = new_env

            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(self.save_path, 'best_model'))
                print(f"🏆 New best model with reward: {mean_reward:.2f}")

        return True

def train_all_algorithms():
    """Train with curriculum learning"""
    print(" Starting Comprehensive Curriculum Training...")
    
    # Create directories
    os.makedirs("models/curriculum", exist_ok=True)
    
    # Start with Phase 1 (easiest)
    env = MedicalDeliveryEnv(render_mode=None, training_phase=1)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # Enhanced PPO parameters for better learning
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.8,
        tensorboard_log="./logs/curriculum/",
        policy_kwargs=dict(
            net_arch=[256, 256],
            activation_fn=torch.nn.ReLU,
            ortho_init=True
        ),
        verbose=1
    )
    
    # Create curriculum callback
    callback = CurriculumCallback(
        check_freq=20000, 
        save_path="./models/curriculum/",
        eval_env=env
    )
    
    # Extended training
    print(" Beginning extended training (500,000 timesteps)...")
    model.learn(
        total_timesteps=500000,
        callback=callback,
        tb_log_name="Curriculum_Medical_Drone"
    )
    
    # Save final model
    model.save("models/curriculum/medical_drone_final")
    
    # Final evaluation
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"🎉 Training completed! Final mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    env.close()
    return model

if __name__ == "__main__":
    train_all_algorithms()

# Update PPO training for extended timesteps
%%writefile training/ppo_training.py
import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from environment.medical_delivery_env import MedicalDeliveryEnv

class TrainingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Save model
            model_path = os.path.join(self.save_path, 'model_{}'.format(self.n_calls))
            self.model.save(model_path)

            # Evaluate
            mean_reward, std_reward = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=5)
            print(f"Step {self.n_calls}: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(self.save_path, 'best_model'))
                print(f"New best model with reward: {mean_reward:.2f}")

        return True

def train_ppo(total_timesteps=500000, training_phase=1):  # Extended timesteps
    """Train PPO agent with extended training"""
    print(f" Starting PPO Training (Phase {training_phase})...")

    # Create environment with specified phase
    env = MedicalDeliveryEnv(render_mode=None, training_phase=training_phase)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # Enhanced PPO parameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=15,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.8,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="./logs/ppo/",
        policy_kwargs=dict(
            net_arch=[256, 256],
            activation_fn=torch.nn.ReLU,
            ortho_init=True
        ),
        verbose=1
    )

    # Create callback
    callback = TrainingCallback(check_freq=20000, save_path="./models/ppo/")

    # Extended training
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        tb_log_name=f"PPO_Medical_Drone_Phase{training_phase}"
    )

    # Save final model
    model.save(f"models/ppo/medical_drone_ppo_phase{training_phase}")

    # Evaluate
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f" Training completed! Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()
    return model

if __name__ == "__main__":
    train_ppo()
