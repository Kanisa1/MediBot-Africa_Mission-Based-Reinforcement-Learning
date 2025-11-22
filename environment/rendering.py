import time
import random
import sys
import os

# Ensure environment folder is in path
sys.path.append(os.path.dirname(__file__))

from custom_env import MalariaDefenseEnv

def main():
    # Create the environment
    env = MalariaDefenseEnv(grid_size=10, max_steps=50)
    
    obs, _ = env.reset()
    done = False
    
    try:
        while not done:
            env.render()  # Render the environment
            time.sleep(0.3)  # Adjust speed of rendering
            
            # Choose a random action
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            
            # Optional: Print step info
            print(f"Step: {info['step']}, Reward: {reward}, Lives Saved: {info['lives_saved']}, Active Outbreaks: {info['active_outbreaks']}")
    
    except KeyboardInterrupt:
        print("Simulation interrupted!")
    
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()
