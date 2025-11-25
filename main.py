import os
import sys
import time
import numpy as np
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'environment'))

from environment.medical_delivery_env import MedicalDeliveryEnv
from stable_baselines3 import PPO
import pybullet as p

def create_demo_directory():
    """Create logs/demo directory if it doesn't exist"""
    demo_dir = os.path.join(os.path.dirname(__file__), 'logs', 'demo')
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)
        print("[OK] Created demo directory: {0}".format(demo_dir))
    return demo_dir

def load_all_phase_models():
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

def run_phase_with_recording(phase_num, model, demo_dir, episodes=1, delay=0.01):
    """Run a phase and record video"""
    
    phase_names = {1: "Easy", 2: "Medium", 3: "Hard"}
    
    print("\n" + "="*70)
    print("PHASE {0} ({1}) - RECORDING VIDEO".format(phase_num, phase_names.get(phase_num, "Unknown")))
    print("="*70)
    
    # Create environment with GUI
    env = MedicalDeliveryEnv(training_phase=phase_num, render_mode="human")
    
    # Create video filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(demo_dir, "phase{0}_{1}.mp4".format(phase_num, timestamp))
    
    print("\nRecording to: {0}".format(video_path))
    
    total_rewards = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        
        # Start recording AFTER reset (physics client is now connected)
        print("Starting PyBullet recording...\n")
        recorder_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_path)
        
        total_reward = 0
        steps = 0
        
        print("Episode {0}/{1} - Phase {2}".format(episode + 1, episodes, phase_num))
        print("-" * 70)
        
        while True:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            # Render
            env.render()
            time.sleep(delay)
            
            # Print progress every 100 steps
            if steps % 100 == 0:
                print("  Step {0:4d} | Reward: {1:10.2f}".format(steps, total_reward))
            
            # Check if done
            if terminated or truncated:
                break
        
        # Stop recording
        p.stopStateLogging(recorder_id)
        
        total_rewards.append(total_reward)
        print("\n  [COMPLETE] Episode finished!")
        print("     Total Reward: {0:.2f}".format(total_reward))
        print("     Total Steps: {0}".format(steps))
    
    print("\n[OK] Video saved: {0}".format(video_path))
    
    # Close environment
    env.close()
    
    avg_reward = np.mean(total_rewards)
    return avg_reward, video_path

def save_results_summary(demo_dir, results):
    """Save results to a text file"""
    summary_path = os.path.join(demo_dir, "results_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MEDIBOT ALL PHASES DEMO - RESULTS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("Date & Time: {0}\n\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        
        phase_names = {1: "Easy", 2: "Medium", 3: "Hard"}
        
        for phase in sorted(results.keys()):
            reward, video_file = results[phase]
            f.write("PHASE {0} ({1})\n".format(phase, phase_names.get(phase, "Unknown")))
            f.write("  Reward: {0:.2f}\n".format(reward))
            f.write("  Video: {0}\n".format(os.path.basename(video_file)))
            f.write("\n")
        
        # Find best phase
        best_phase = max(results, key=lambda x: results[x][0])
        best_reward = results[best_phase][0]
        f.write("="*70 + "\n")
        f.write("BEST PERFORMING PHASE: {0} with reward {1:.2f}\n".format(best_phase, best_reward))
        f.write("="*70 + "\n")
    
    print("\n[OK] Results summary saved: {0}".format(summary_path))

def main():
    print("\n" + "="*70)
    print("MEDIBOT - ALL PHASES DEMO WITH VIDEO RECORDING")
    print("="*70)
    
    # Create demo directory
    demo_dir = create_demo_directory()
    
    # Load all models
    print("\nLoading all phase models...")
    models = load_all_phase_models()
    
    if not models:
        print("\n[ERROR] No models found!")
        return
    
    print("\n[INFO] Loaded {0} model(s)".format(len(models)))
    
    # Run each phase with recording
    results = {}
    
    for phase in sorted(models.keys()):
        try:
            print("\n[STARTING] Phase {0}...".format(phase))
            reward, video_path = run_phase_with_recording(
                phase_num=phase,
                model=models[phase],
                demo_dir=demo_dir,
                episodes=1,
                delay=0.01
            )
            results[phase] = (reward, video_path)
            
            # Small pause between phases
            if phase < 3:
                print("\n[PAUSE] 2 seconds before next phase...")
                time.sleep(2)
        
        except Exception as e:
            print("\n[ERROR] Failed to run Phase {0}: {1}".format(phase, str(e)))
            import traceback
            traceback.print_exc()
            continue
    
    # Save results summary
    if results:
        save_results_summary(demo_dir, results)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - ALL PHASES WITH VIDEO RECORDING")
    print("="*70)
    
    phase_names = {1: "Easy", 2: "Medium", 3: "Hard"}
    
    for phase in sorted(results.keys()):
        reward, video_file = results[phase]
        print("\nPhase {0} ({1}):".format(phase, phase_names.get(phase, "Unknown")))
        print("  Reward: {0:.2f}".format(reward))
        print("  Video: {0}".format(os.path.basename(video_file)))
    
    if results:
        best_phase = max(results, key=lambda x: results[x][0])
        best_reward = results[best_phase][0]
        print("\n" + "="*70)
        print("BEST PHASE: {0} with reward {1:.2f}".format(best_phase, best_reward))
        print("="*70)
    
    print("\n[INFO] All videos saved to: {0}".format(demo_dir))
    print("\n[COMPLETE] Demo with video recording finished!\n")

if __name__ == "__main__":
    main()
