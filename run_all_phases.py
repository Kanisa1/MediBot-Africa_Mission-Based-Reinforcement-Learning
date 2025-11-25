import os
import sys
import time
import numpy as np

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'environment'))

from environment.medical_delivery_env import MedicalDeliveryEnv
from stable_baselines3 import PPO

def load_phase_models():
    """Load all three phase models"""
    models_dir = os.path.join(os.path.dirname(__file__), 'models', 'pregressive')
    models = {}
    
    phases = {
        1: 'phase1_final.zip',
        2: 'phase2_final.zip',
        3: 'phase3_final.zip'
    }
    
    for phase, filename in phases.items():
        model_path = os.path.join(models_dir, filename)
        if os.path.exists(model_path):
            try:
                model = PPO.load(model_path)
                models[phase] = model
                print("[OK] Loaded Phase {0} model".format(phase))
            except Exception as e:
                print("[ERROR] Failed to load Phase {0}: {1}".format(phase, str(e)))
        else:
            print("[MISSING] Phase {0} model not found".format(phase))
    
    return models

def demonstrate_phase(phase_num, model, render=True, delay=0.02, episodes=1):
    """Demonstrate a single phase"""
    print("\n" + "="*70)
    print("RUNNING PHASE {0}".format(phase_num))
    print("="*70)
    
    env = MedicalDeliveryEnv(training_phase=phase_num)
    
    total_rewards = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        delivery_count = 0
        
        print("\nEpisode {0}/{1} - Phase {2}".format(episode + 1, episodes, phase_num))
        print("-" * 70)
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
                time.sleep(delay)
            
            # Print progress every 50 steps
            if steps % 50 == 0:
                status = "Step {0:4d} | Reward: {1:10.2f}".format(steps, total_reward)
                print("  " + status)
            
            if terminated or truncated:
                break
        
        total_rewards.append(total_reward)
        print("\n  [COMPLETE] Episode finished!")
        print("     Total Reward: {0:.2f}".format(total_reward))
        print("     Total Steps: {0}".format(steps))
    
    avg_reward = np.mean(total_rewards)
    print("\n[PHASE {0} RESULT] Average Reward: {1:.2f}".format(phase_num, avg_reward))
    
    env.close()
    return avg_reward

def main():
    print("\n" + "="*70)
    print("MEDIBOT - ALL PHASES DEMONSTRATION")
    print("Medical Drone Reinforcement Learning")
    print("="*70)
    
    # Load all models
    print("\nLoading all phase models...")
    models = load_phase_models()
    
    if not models:
        print("\n[ERROR] No models found! Please check models/pregressive/ folder")
        return
    
    print("\n[INFO] Loaded {0} model(s)".format(len(models)))
    
    # Run each phase
    results = {}
    render = True
    delay = 0.02  # Rendering speed
    episodes_per_phase = 1
    
    for phase in sorted(models.keys()):
        try:
            print("\n[STARTING] Phase {0}...".format(phase))
            reward = demonstrate_phase(
                phase_num=phase,
                model=models[phase],
                render=render,
                delay=delay,
                episodes=episodes_per_phase
            )
            results[phase] = reward
            
            # Small pause between phases
            if phase < 3:
                print("\n[PAUSE] 2 seconds before next phase...")
                time.sleep(2)
        
        except Exception as e:
            print("\n[ERROR] Failed to run Phase {0}: {1}".format(phase, str(e)))
            continue
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - ALL PHASES")
    print("="*70)
    
    if results:
        for phase in sorted(results.keys()):
            phase_names = {1: "Easy", 2: "Medium", 3: "Hard"}
            print("\nPhase {0} ({1}): {2:.2f}".format(phase, phase_names.get(phase, "Unknown"), results[phase]))
        
        best_phase = max(results, key=results.get)
        best_reward = results[best_phase]
        print("\n[BEST] Phase {0} with reward: {1:.2f}".format(best_phase, best_reward))
    
    print("\n[COMPLETE] All phases demonstrated successfully!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
