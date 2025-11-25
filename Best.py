import os
import sys
import time
import numpy as np

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'environment'))

from environment.medical_delivery_env import MedicalDeliveryEnv
from stable_baselines3 import PPO

def load_phase1_model():
    """Load Phase 1 model"""
    models_dir = os.path.join(os.path.dirname(__file__), 'models', 'pregressive')
    model_path = os.path.join(models_dir, 'phase1_final.zip')
    
    if os.path.exists(model_path):
        try:
            model = PPO.load(model_path)
            print("[OK] Phase 1 model loaded successfully!")
            return model
        except Exception as e:
            print("[ERROR] Failed to load Phase 1 model: {0}".format(str(e)))
            return None
    else:
        print("[ERROR] Phase 1 model not found at: {0}".format(model_path))
        return None

def watch_drone_fly(model, episodes=1, delay=0.02):
    """Watch the drone fly on screen with PyBullet rendering"""
    
    print("\n" + "="*70)
    print("WATCHING PHASE 1 DRONE FLY")
    print("="*70)
    print("\nPyBullet 3D window will open on your screen...")
    print("Watch the blue drone navigate and deliver medicine!\n")
    
    env = MedicalDeliveryEnv(training_phase=1, render_mode="human")
    
    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        print("\n--- Episode {0} ---".format(episode + 1))
        print("Starting mission at pharmacy...")
        print("Target: Deliver medicine to a village\n")
        
        start_time = time.time()
        
        while True:
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            # Render on screen
            env.render()
            time.sleep(delay)
            
            # Print status every 100 steps
            if steps % 100 == 0:
                elapsed = time.time() - start_time
                print("  Step {0:4d} | Reward: {1:10.2f} | Time: {2:.1f}s".format(
                    steps, total_reward, elapsed))
            
            # Check if episode is done
            if terminated or truncated:
                break
        
        elapsed = time.time() - start_time
        
        # Print final results
        print("\n" + "-"*70)
        print("EPISODE {0} COMPLETE!".format(episode + 1))
        print("-"*70)
        print("Total Steps:  {0}".format(steps))
        print("Total Reward: {0:.2f}".format(total_reward))
        print("Time Taken:   {0:.2f} seconds".format(elapsed))
        print("-"*70 + "\n")
    
    env.close()
    print("\n[INFO] PyBullet window closed.")
    print("[INFO] Demo complete!\n")

def main():
    print("\n" + "="*70)
    print("MEDIBOT PHASE 1 - WATCH THE DRONE FLY")
    print("="*70)
    
    # Load Phase 1 model
    print("\nLoading Phase 1 trained model...")
    model = load_phase1_model()
    
    if model is None:
        print("\n[ERROR] Could not load model. Exiting.")
        return
    
    # Watch drone fly
    try:
        watch_drone_fly(model, episodes=1, delay=0.02)
    except KeyboardInterrupt:
        print("\n\n[INFO] Demo interrupted by user.")
    except Exception as e:
        print("\n[ERROR] An error occurred: {0}".format(str(e)))
    
    print("\n" + "="*70)
    print("THANK YOU FOR WATCHING!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
