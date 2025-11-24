"""
Training script for multi-agent DQN with communication.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import json
import os
from typing import Tuple, Optional
from multi_agent_env import MultiAgentEnv
from models import AgentDQN
from replay_buffer import ReplayBuffer


def apply_observation_mask(obs: np.ndarray, mode: str) -> np.ndarray:
    """
    Apply masking to observation based on ablation mode.

    Args:
        obs: 11-dimensional observation vector
        mode: One of 'independent', 'comm', 'full'

    Returns:
        Masked observation
    """
    # TODO: Implement masking logic
    # 'independent': Set elements 9 and 10 to zero
    # 'comm': Set element 10 to zero
    # 'full': No masking

    raise NotImplementedError


class MultiAgentTrainer:
    """
    Trainer for multi-agent DQN system.

    Handles training loop, exploration, and network updates.
    """

    def __init__(self, env: MultiAgentEnv, args):
        """
        Initialize trainer.

        Args:
            env: Multi-agent environment
            args: Training arguments
        """
        self.env = env
        self.args = args

        # Use CPU for small networks
        self.device = torch.device("cpu")

        # TODO: Initialize networks for both agents (remember to .to(self.device))
        # TODO: Initialize target networks (if using)
        # TODO: Initialize optimizers
        # TODO: Initialize replay buffer
        # TODO: Initialize epsilon for exploration

        raise NotImplementedError

    def select_action(self, state: np.ndarray, network: nn.Module,
                      epsilon: float) -> Tuple[int, float]:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Agent observation (11-dimensional, may need masking)
            network: Agent's DQN
            epsilon: Exploration probability

        Returns:
            action: Selected action
            comm_signal: Communication signal
        """
        # TODO: Apply observation masking based on self.args.mode
        #       masked_state = apply_observation_mask(state, self.args.mode)
        # TODO: With probability epsilon, select random action
        # TODO: Otherwise, select action with highest Q-value
        # TODO: Always get communication signal from network
        # TODO: Return (action, comm_signal)

        raise NotImplementedError

    def update_networks(self, batch_size: int) -> float:
        """
        Sample batch and update both agent networks.

        Args:
            batch_size: Size of training batch

        Returns:
            loss: Combined loss value
        """
        # TODO: Sample batch from replay buffer
        # TODO: Convert to tensors and move to device
        # TODO: Compute Q-values for current states
        # TODO: Compute target Q-values using target networks
        # TODO: Calculate TD loss for both agents
        # TODO: Backpropagate and update networks
        # TODO: Return combined loss

        raise NotImplementedError

    def train_episode(self) -> Tuple[float, bool]:
        """
        Run one training episode.

        Returns:
            episode_reward: Total reward for episode
            success: Whether agents reached target
        """
        # TODO: Reset environment
        # TODO: Initialize episode variables
        # TODO: Run episode until termination:
        #       - Select actions for both agents
        #       - Execute actions in environment
        #       - Store transition in replay buffer
        #       - Update networks if enough samples
        # TODO: Return episode reward and success flag

        raise NotImplementedError

    def train(self) -> None:
        """
        Main training loop.
        """
        # TODO: Create results directories
        # TODO: Initialize logging
        # TODO: Main training loop:
        #       - Run episodes
        #       - Update epsilon
        #       - Update target networks periodically
        #       - Log progress
        #       - Save checkpoints
        # TODO: Save final models including TorchScript format:
        #       scripted_model = torch.jit.script(self.network_A)
        #       scripted_model.save("dqn_net.pt")

        raise NotImplementedError

    def evaluate(self, num_episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluate current policy.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            mean_reward: Average reward
            success_rate: Fraction of successful episodes
        """
        # TODO: Set networks to evaluation mode
        # TODO: Run episodes without exploration
        # TODO: Track rewards and successes
        # TODO: Return statistics

        raise NotImplementedError


def main():
    """
    Parse arguments and run training.
    """
    parser = argparse.ArgumentParser(description='Train Multi-Agent DQN')

    # Environment parameters
    parser.add_argument('--grid_size', type=int, nargs=2, default=[10, 10],
                       help='Grid dimensions')
    parser.add_argument('--max_steps', type=int, default=50,
                       help='Maximum steps per episode')

    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=5000,
                       help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')

    # Exploration parameters
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                       help='Initial exploration rate')
    parser.add_argument('--epsilon_end', type=float, default=0.05,
                       help='Final exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.995,
                       help='Epsilon decay rate')

    # Network parameters
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden layer size')
    parser.add_argument('--target_update', type=int, default=100,
                       help='Target network update frequency')

    # Ablation study mode
    parser.add_argument('--mode', type=str, default='full',
                       choices=['independent', 'comm', 'full'],
                       help='Information mode: independent (mask comm+dist), '
                            'comm (mask dist only), full (no masking)')

    # Other parameters
    parser.add_argument('--seed', type=int, default=641,
                       help='Random seed')
    parser.add_argument('--save_freq', type=int, default=500,
                       help='Model save frequency')

    args = parser.parse_args()

    # TODO: Set random seeds
    # TODO: Create environment
    # TODO: Create trainer
    # TODO: Run training
    # TODO: Final evaluation

    raise NotImplementedError


if __name__ == '__main__':
    main()