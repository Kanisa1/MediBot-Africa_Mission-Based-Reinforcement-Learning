import os
import sys
import argparse
from training.ppo_training import train_ppo
from training.dqn_training import train_dqn
from training.a2c_training import train_a2c
from training.comprehensive_training import train_all_algorithms
from training.progressive_training import progressive_training

def main():
    parser = argparse.ArgumentParser(description='Medical Drone Reinforcement Learning')
    parser.add_argument('--algorithm', type=str, default='progressive',
                       choices=['ppo', 'dqn', 'a2c', 'all', 'progressive', 'curriculum'],
                       help='Algorithm to train')
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Number of training timesteps')
    parser.add_argument('--phase', type=int, default=1,
                       choices=[1, 2, 3],
                       help='Training phase (1: Easy, 2: Medium, 3: Hard)')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment during training')

    args = parser.parse_args()

    print(" Medical Drone RL Training Starting...")
    print(f"Algorithm: {args.algorithm}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Phase: {args.phase}")

    if args.algorithm == 'ppo':
        train_ppo(total_timesteps=args.timesteps, training_phase=args.phase)
    elif args.algorithm == 'dqn':
        train_dqn(total_timesteps=args.timesteps)
    elif args.algorithm == 'a2c':
        train_a2c(total_timesteps=args.timesteps)
    elif args.algorithm == 'curriculum':
        train_all_algorithms()
    elif args.algorithm == 'progressive':
        progressive_training()
    else:
        train_all_algorithms()

    print(" Training completed!")

if __name__ == "__main__":
    main()
