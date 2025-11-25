import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.results_plotter import load_results, ts2xy

def plot_training_results(log_folder, title='Training Results'):
    """Plot training results from tensorboard logs"""
    
    try:
        # Get the learning curve data
        x, y = ts2xy(load_results(log_folder), 'timesteps')
        
        plt.figure(figsize=(12, 6))
        
        # Plot rolling average
        df = pd.DataFrame({'timesteps': x, 'rewards': y})
        df['rolling_avg'] = df['rewards'].rolling(window=50).mean()
        
        plt.plot(df['timesteps'], df['rewards'], alpha=0.3, label='Episode Reward')
        plt.plot(df['timesteps'], df['rolling_avg'], linewidth=2, label='Rolling Avg (50)')
        
        plt.xlabel('Timesteps')
        plt.ylabel('Episode Reward')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add horizontal line at target reward
        plt.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='Target Reward (80)')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"Final Rolling Average: {df['rolling_avg'].iloc[-1]:.2f}")
        print(f"Max Reward: {df['rewards'].max():.2f}")
        print(f"Mean Reward (last 50): {df['rewards'].tail(50).mean():.2f}")
        
    except Exception as e:
        print(f"Could not load results from {log_folder}: {e}")
        print("Training data not available yet. Train a model first.")

def compare_algorithms(algo_results):
    """Compare performance of different algorithms"""
    algorithms = list(algo_results.keys())
    rewards = [algo_results[algo]['mean_reward'] for algo in algorithms]
    errors = [algo_results[algo]['std_reward'] for algo in algorithms]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, rewards, yerr=errors, capsize=5, 
                   color=['skyblue', 'lightcoral', 'lightgreen'])
    
    plt.axhline(y=80, color='red', linestyle='--', label='Target Reward (+80)')
    plt.ylabel('Mean Reward')
    plt.title('Algorithm Performance Comparison')
    plt.legend()
    
    # Add value labels on bars
    for bar, reward in zip(bars, rewards):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{reward:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Performance analysis
    print("\n📊 Algorithm Performance Analysis:")
    for algo in algorithms:
        status = "✅ TARGET ACHIEVED" if algo_results[algo]['mean_reward'] >= 80 else "❌ BELOW TARGET"
        print(f"{algo}: {algo_results[algo]['mean_reward']:.2f} +/- {algo_results[algo]['std_reward']:.2f} {status}")

def plot_mission_analysis():
    """Analyze mission performance metrics"""
    # Sample mission data
    missions = range(1, 21)
    delivery_times = np.random.normal(120, 30, 20)
    return_times = np.random.normal(180, 40, 20)
    success_rates = np.minimum(np.linspace(0.3, 0.95, 20) + np.random.normal(0, 0.1, 20), 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Mission times
    ax1.plot(missions, delivery_times, marker='o', label='Delivery Time')
    ax1.plot(missions, return_times, marker='s', label='Return Time')
    ax1.set_xlabel('Mission Number')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Mission Completion Times')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Success rates
    ax2.plot(missions, success_rates, marker='o', color='green', linewidth=2)
    ax2.axhline(y=0.8, color='red', linestyle='--', label='Target Success Rate (80%)')
    ax2.set_xlabel('Mission Number')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Mission Success Rate Progression')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Average Delivery Time: {np.mean(delivery_times):.1f}s")
    print(f"Average Return Time: {np.mean(return_times):.1f}s")
    print(f"Final Success Rate: {success_rates[-1]:.1%}")

print(" Performance visualization utilities created!")

# Test the visualization
print("\n Generating Performance Visualizations...")

# Create sample algorithm results for demonstration
sample_results = {
    'PPO': {'mean_reward': 85.2, 'std_reward': 12.5},
    'DQN': {'mean_reward': 45.7, 'std_reward': 18.3},
    'A2C': {'mean_reward': 67.8, 'std_reward': 15.2}
}

# Compare algorithms
from utils.visualization import compare_algorithms, plot_mission_analysis
compare_algorithms(sample_results)

# Plot mission analysis
plot_mission_analysis()

print("✅ All visualizations completed!")
