import os
import sys
import time

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'environment'))

from environment.medical_delivery_env import MedicalDeliveryEnv
from stable_baselines3 import PPO

print("="*70)
print("PHASE 1 DRONE DEMO - SIMPLE VERSION")
print("="*70)

# Load model
print("\n[1] Loading Phase 1 model...")
model_path = os.path.join(os.path.dirname(__file__), 'models', 'pregressive', 'phase1_final.zip')
model = PPO.load(model_path)
print("[OK] Model loaded!")

# Create environment WITH GUI RENDERING
print("\n[2] Creating environment with PyBullet GUI...")
env = MedicalDeliveryEnv(training_phase=1, render_mode="human")
print("[OK] Environment created!")

# Run episode
print("\n[3] Starting episode - Watch the drone fly on screen!")
print("     (PyBullet window should appear on your screen now)\n")

obs, info = env.reset()
total_reward = 0
steps = 0

while steps < 1000:
    # Get action from model
    action, _ = model.predict(obs, deterministic=True)
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    total_reward += reward
    steps += 1
    
    # Print progress
    if steps % 100 == 0:
        print("Step {0:4d} | Reward: {1:10.2f}".format(steps, total_reward))
    
    # Check if done
    if terminated or truncated:
        print("Episode finished!")
        break

print("\n[RESULTS]")
print("Total Steps: {0}".format(steps))
print("Total Reward: {0:.2f}".format(total_reward))

# Keep window open for a few seconds
print("\n[INFO] Keeping window open for 3 seconds...")
time.sleep(3)

# Close environment
env.close()
print("\n[OK] Demo complete! Environment closed.\n")
