class TrainingConfig:
    """Configuration for training parameters"""

    # General training parameters
    TOTAL_TIMESTEPS = 200000
    EVAL_FREQ = 10000
    SAVE_FREQ = 25000

    # Algorithm-specific hyperparameters
    PPO_PARAMS = {
        'learning_rate': 2.5e-4,
        'n_steps': 2048,
        'batch_size': 128,
        'n_epochs': 15,
        'gamma': 0.995,
        'ent_coef': 0.01
    }
    
    DQN_PARAMS = {
        'learning_rate': 1e-4,
        'buffer_size': 100000,
        'batch_size': 32,
        'gamma': 0.99,
        'exploration_final_eps': 0.05
    }
    
    A2C_PARAMS = {
        'learning_rate': 7e-4,
        'n_steps': 16,
        'gamma': 0.99,
        'ent_coef': 0.01
    }

%%writefile utils/metrics.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class MetricsTracker:
    """Track training metrics"""

    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.successful_deliveries = 0
        self.successful_returns = 0

    def update(self, reward, length, delivery_success=False, return_success=False):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        if delivery_success:
            self.successful_deliveries += 1
        if return_success:
            self.successful_returns += 1

    def get_stats(self):
        stats = {
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'max_reward': np.max(self.episode_rewards) if self.episode_rewards else 0,
            'min_reward': np.min(self.episode_rewards) if self.episode_rewards else 0,
            'mean_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'delivery_success_rate': self.successful_deliveries / len(self.episode_rewards) if self.episode_rewards else 0,
            'return_success_rate': self.successful_returns / len(self.episode_rewards) if self.episode_rewards else 0
        }
        return stats

    def plot_metrics(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Episode rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=80, color='r', linestyle='--', label='Target (+80)')
        ax1.legend()

        # Episode lengths
        ax2.plot(self.episode_lengths)
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.grid(True, alpha=0.3)

        # Rolling average rewards
        if len(self.episode_rewards) > 10:
            df = pd.DataFrame({'rewards': self.episode_rewards})
            df['rolling_avg'] = df['rewards'].rolling(window=20).mean()
            ax3.plot(df['rolling_avg'])
            ax3.set_title('Rolling Average Reward (20 episodes)')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Average Reward')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=80, color='r', linestyle='--', label='Target (+80)')
            ax3.legend()

        # Success rates
        episodes = range(1, len(self.episode_rewards) + 1)
        delivery_rates = [self.successful_deliveries / (i+1) for i in range(len(episodes))]
        return_rates = [self.successful_returns / (i+1) for i in range(len(episodes))]
        
        ax4.plot(episodes, delivery_rates, label='Delivery Success')
        ax4.plot(episodes, return_rates, label='Return Success')
        ax4.set_title('Success Rates')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Success Rate')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print statistics
        stats = self.get_stats()
        print(f"\n Training Statistics:")
        print(f"Mean Reward: {stats['mean_reward']:.2f}")
        print(f"Max Reward: {stats['max_reward']:.2f}")
        print(f"Delivery Success Rate: {stats['delivery_success_rate']:.2%}")
        print(f"Return Success Rate: {stats['return_success_rate']:.2%}")

print(" Utility files created!")
