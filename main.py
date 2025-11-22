import gymnasium as gym
import pygame
import sys
import os
import time
import numpy as np

# Ensure environment path is correct
sys.path.append(os.path.join(os.path.dirname(__file__), 'environment'))
from custom_env import MalariaDefenseEnv
from stable_baselines3 import DQN, PPO, A2C

def load_best_models():
    """Load the best performing models"""
    models = {}

    try:
        models['DQN'] = DQN.load("models/dqn/best_dqn_model")
        print("✓ DQN model loaded")
    except:
        models['DQN'] = None
        print("✗ DQN model not found")

    try:
        models['PPO'] = PPO.load("models/pg/best_ppo_model")
        print("✓ PPO model loaded")
    except:
        models['PPO'] = None
        print("✗ PPO model not found")

    try:
        models['A2C'] = A2C.load("models/pg/best_a2c_model")
        print("✓ A2C model loaded")
    except:
        models['A2C'] = None
        print("✗ A2C model not found")

    return models

def demonstrate_agent(model, model_name, env, episodes=3):
    """Demonstrate trained agent performance"""
    print(f"\n{'='*50}")
    print(f"DEMONSTRATING {model_name} AGENT")
    print(f"{'='*50}")

    total_rewards = []

    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        print(f"\nEpisode {episode + 1}:")
        print("-" * 30)

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            # Print key metrics every 20 steps
            if steps % 20 == 0:
                print(f"Step {steps}: Lives Saved: {info['lives_saved']}, "
                      f"Outbreaks: {info['active_outbreaks']}, "
                      f"Reward: {total_reward:.2f}")

            env.render()
            time.sleep(0.1)

            if terminated or truncated:
                break

        total_rewards.append(total_reward)

        print(f"Episode {episode + 1} Results:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Lives Saved: {info['lives_saved']}")
        print(f"  Outbreaks Contained: {info['outbreaks_contained']}")
        print(f"  Steps Taken: {steps}")
        print(f"  Resources Used: {info['resources']}")

    avg_reward = np.mean(total_rewards)
    print(f"\n{model_name} Average Performance over {episodes} episodes: {avg_reward:.2f}")

    return avg_reward

def run_random_agent_demo():
    """Demonstrate random agent for comparison"""
    print(f"\n{'='*50}")
    print("DEMONSTRATING RANDOM AGENT (BASELINE)")
    print(f"{'='*50}")

    env = MalariaDefenseEnv()
    try:
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        while steps < 100:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            if steps % 20 == 0:
                print(f"Step {steps}: Lives Saved: {info['lives_saved']}, "
                      f"Outbreaks: {info['active_outbreaks']}, "
                      f"Reward: {total_reward:.2f}")

            env.render()
            time.sleep(0.1)

            if terminated or truncated:
                break

        print(f"Random Agent Performance: {total_reward:.2f}")
        return total_reward

    finally:
        env.close()

def main():
    """Main execution function"""
    print(" MALARIA DEFENSE AGENT DEMONSTRATION ")
    print("AI-Powered Healthcare System for South Sudan")
    print("=" * 60)

    # Create environment
    env = MalariaDefenseEnv()

    # Load trained models
    print("Loading trained models...")
    models = load_best_models()

    if all(model is None for model in models.values()):
        print("No trained models found. Please run training scripts first.")
        return

    # Demonstrate random agent first
    random_performance = run_random_agent_demo()

    # Demonstrate each trained model
    performances = {'Random': random_performance}

    for model_name, model in models.items():
        if model is not None:
            performance = demonstrate_agent(model, model_name, env)
            performances[model_name] = performance

    # Final comparison
    print(f"\n{'='*60}")
    print("FINAL PERFORMANCE COMPARISON")
    print(f"{'='*60}")

    sorted_performance = sorted(performances.items(), key=lambda x: x[1], reverse=True)

    for i, (name, score) in enumerate(sorted_performance):
        print(f"{i+1}. {name}: {score:.2f}")

    best_agent, best_score = sorted_performance[0]
    print(f"\n Best Performing Agent: {best_agent} with score {best_score:.2f}")
    print(f"Performance Improvement over Random: {best_score - random_performance:.2f}")

    env.close()

if __name__ == "__main__":
    main()
