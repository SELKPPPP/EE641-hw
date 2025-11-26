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
    masked_obs = obs.copy()
    if mode == 'independent':
        masked_obs[9] = 0.0
        masked_obs[10] = 0.0
    elif mode == 'comm':
        masked_obs[10] = 0.0
    elif mode == 'dist':
        masked_obs[9] = 0.0


    return masked_obs


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
        self.agent_A = AgentDQN(input_dim=11, hidden_dim=args.hidden_dim, num_actions=5).to(self.device)
        self.agent_B = AgentDQN(input_dim=11, hidden_dim=args.hidden_dim, num_actions=5).to(self.device)
        # TODO: Initialize target networks (if using)
        self.target_A = AgentDQN(input_dim=11, hidden_dim=args.hidden_dim, num_actions=5).to(self.device)
        self.target_B = AgentDQN(input_dim=11, hidden_dim=args.hidden_dim, num_actions=5).to(self.device)
        self.target_A.load_state_dict(self.agent_A.state_dict())
        self.target_B.load_state_dict(self.agent_B.state_dict())
        # TODO: Initialize optimizers
        self.optimizer_A = optim.Adam(self.agent_A.parameters(), lr=args.lr)
        self.optimizer_B = optim.Adam(self.agent_B.parameters(), lr=args.lr)
        # TODO: Initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000, seed=args.seed)
        # TODO: Initialize epsilon for exploration
        self.epsilon = args.epsilon_start


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
        masked_state = apply_observation_mask(state, self.args.mode)
        state_tensor = torch.FloatTensor(masked_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values, comm_signal = network(state_tensor)
            comm_val = comm_signal.item()
    
        # TODO: With probability epsilon, select random action
        # Expoloration
        if np.random.rand()< epsilon:
            action = np.random.randint(0, 5)
            
        # TODO: Otherwise, select action with highest Q-value
        # Exploitation
        else: 
            action = q_values.argmax().item()
        # TODO: Always get communication signal from network
        # TODO: Return (action, comm_signal)
        return action, comm_val

    def update_networks(self, batch_size: int) -> float:
        """
        Sample batch and update both agent networks.

        Args:
            batch_size: Size of training batch

        Returns:
            loss: Combined loss value
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0
        # TODO: Sample batch from replay buffer
        batch = self.replay_buffer.sample(batch_size)
        state_A, state_B, action_A, action_B, \
        comm_A, comm_B, reward, \
        next_state_A, next_state_B, done = batch
        # TODO: Convert to tensors and move to device
        # Masking
        masked_state_A = np.array([apply_observation_mask(s, self.args.mode) for s in state_A])
        masked_state_B = np.array([apply_observation_mask(s, self.args.mode) for s in state_B])
        masked_next_state_A = np.array([apply_observation_mask(s, self.args.mode) for s in next_state_A])
        masked_next_state_B = np.array([apply_observation_mask(s, self.args.mode) for s in next_state_B])

        state_A_tensor = torch.FloatTensor(masked_state_A).to(self.device)
        state_B_tensor = torch.FloatTensor(masked_state_B).to(self.device)
        action_A_tensor = torch.LongTensor(action_A).unsqueeze(1).to(self.device)
        action_B_tensor = torch.LongTensor(action_B).unsqueeze(1).to(self.device)
        comm_A_tensor = torch.FloatTensor(comm_A).unsqueeze(1).to(self.device)
        comm_B_tensor = torch.FloatTensor(comm_B).unsqueeze(1).to(self.device)
        reward_tensor = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state_A_tensor = torch.FloatTensor(masked_next_state_A).to(self.device)
        next_state_B_tensor = torch.FloatTensor(masked_next_state_B).to(self.device)
        done_tensor = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # TODO: Compute Q-values for current states
        q_values_A, _ = self.agent_A(state_A_tensor)
        q_values_B, _ = self.agent_B(state_B_tensor)
        q_value_A = q_values_A.gather(1, action_A_tensor)
        q_value_B = q_values_B.gather(1, action_B_tensor)

        # TODO: Compute target Q-values using target networks
        with torch.no_grad():
            next_q_values_A, _ = self.target_A(next_state_A_tensor)
            next_q_values_B, _ = self.target_B(next_state_B_tensor)
            max_next_q_value_A = next_q_values_A.max(1)[0].unsqueeze(1)
            max_next_q_value_B = next_q_values_B.max(1)[0].unsqueeze(1)
            target_q_value_A = reward_tensor + self.args.gamma * max_next_q_value_A * (1 - done_tensor)
            target_q_value_B = reward_tensor + self.args.gamma * max_next_q_value_B * (1 - done_tensor)
        # TODO: Calculate TD loss for both agents
        loss_A = F.mse_loss(q_value_A, target_q_value_A)
        loss_B = F.mse_loss(q_value_B, target_q_value_B)
        total_loss = loss_A + loss_B
        # TODO: Backpropagate and update networks
        self.optimizer_A.zero_grad()
        self.optimizer_B.zero_grad()
        total_loss.backward()
        self.optimizer_A.step()
        self.optimizer_B.step()
        # TODO: Return combined loss
        return total_loss.item()

    def train_episode(self) -> Tuple[float, bool]:
        """
        Run one training episode.

        Returns:
            episode_reward: Total reward for episode
            success: Whether agents reached target
        """
        # TODO: Reset environment
        obs_A, obs_B = self.env.reset()
        
        # TODO: Initialize episode variables
        episode_reward = 0.0
        done = False
        # TODO: Run episode until termination:
        #       - Select actions for both agents
        #       - Execute actions in environment
        #       - Store transition in replay buffer
        #       - Update networks if enough samples
        while not done:
            action_A, comm_A = self.select_action(obs_A, self.agent_A, self.epsilon)
            action_B, comm_B = self.select_action(obs_B, self.agent_B, self.epsilon)
            next_obs, reward, done = self.env.step(action_A, action_B, comm_A, comm_B)
            next_obs_A, next_obs_B = next_obs

            # Store transition
            self.replay_buffer.push(obs_A, obs_B,
                                    action_A, action_B,
                                    comm_A, comm_B,
                                    reward,
                                    next_obs_A, next_obs_B,
                                    done)
            
            # Update State
            obs_A = next_obs_A
            obs_B = next_obs_B
            episode_reward += reward

            # Update networks
            self.update_networks(self.args.batch_size)
        # TODO: Return episode reward and success flag
        success = (reward >= 5.0)
        return episode_reward, success


    def train(self) -> None:
        """
        Main training loop.
        """
        # TODO: Create results directories

        results_dir = os.path.join("results", self.args.mode)
        os.makedirs(results_dir, exist_ok=True)
        
        # TODO: Initialize logging
        training_log = []
        # TODO: Main training loop:
        #       - Run episodes
        #       - Update epsilon
        #       - Update target networks periodically
        #       - Log progress
        #       - Save checkpoints
        print(f"Starting training in mode: {self.args.mode}")

        hundred_success = 0

        for episode in range(self.args.num_episodes):
            reward, success = self.train_episode()
            
            if success:
                hundred_success += 1
            
            # Update epsilon
            self.epsilon = max(self.args.epsilon_end, self.epsilon * self.args.epsilon_decay)
            
            # Update target networks
            if episode % self.args.target_update == 0:
                self.target_A.load_state_dict(self.agent_A.state_dict())
                self.target_B.load_state_dict(self.agent_B.state_dict())
            
            # Log progress
            if episode % 100 == 0:
                print(f"Episode {episode}: Reward = {reward:.2f}, Success Rate = {hundred_success/100}, Epsilon = {self.epsilon:.2f}")
                training_log.append({
                    "episode": episode,
                    "reward": reward,
                    "success_rate": hundred_success/100,
                    "epsilon": self.epsilon
                })
                hundred_success = 0
                
            # Save checkpoints
            if episode % self.args.save_freq == 0:
                torch.save({
                    'agent_A': self.agent_A.state_dict(),
                    'agent_B': self.agent_B.state_dict()
                }, os.path.join(results_dir, f"checkpoint_{episode}.pth"))
        

        # TODO: Save final models including TorchScript format:
        #       scripted_model = torch.jit.script(self.network_A)
        #       scripted_model.save("dqn_net.pt")
        torch.save({
            'agent_A': self.agent_A.state_dict(),
            'agent_B': self.agent_B.state_dict()
        }, os.path.join(results_dir, "final_model.pth"))

        scripted_A = torch.jit.script(self.agent_A)
        scripted_B = torch.jit.script(self.agent_B)
        scripted_A.save(os.path.join(results_dir, "agent_A_scripted.pt"))
        scripted_B.save(os.path.join(results_dir, "agent_B_scripted.pt"))

        with open(os.path.join(results_dir, "training_log.json"), 'w') as f:
            json.dump(training_log, f, indent=4)

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
        # self.agent_A.eval()
        # self.agent_B.eval()

        total_reward = 0.0
        total_success = 0
        # TODO: Run episodes without exploration
        # TODO: Track rewards and successes
        for _ in range(num_episodes):
            obs_A, obs_B = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Run episodes without exploration (epsilon=0)
                # Note: select_action handles masking internally
                action_A, comm_A = self.select_action(obs_A, self.agent_A, epsilon=0.0)
                action_B, comm_B = self.select_action(obs_B, self.agent_B, epsilon=0.0)
                
                next_obs, reward, done = self.env.step(action_A, action_B, comm_A, comm_B)
                obs_A, obs_B = next_obs
                episode_reward += reward
            
            # Track rewards and successes
            total_reward += episode_reward
            if reward >= 5.0: # Success condition based on reward structure
                total_success += 1
        
        # Restore training mode 
        # self.agent_A.train()
        # self.agent_B.train()
        # TODO: Return statistics
        return total_reward / num_episodes, total_success / num_episodes



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
    parser.add_argument('--mode', type=str, default='independent',
                       choices=['independent', 'comm', 'full', 'dist'],
                       help='Information mode: independent (mask comm+dist), '
                            'comm (mask dist only), full (no masking)')

    # Other parameters
    parser.add_argument('--seed', type=int, default=641,
                       help='Random seed')
    parser.add_argument('--save_freq', type=int, default=500,
                       help='Model save frequency')

    args = parser.parse_args()

    # TODO: Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # TODO: Create environment
    env = MultiAgentEnv(grid_size=tuple(args.grid_size), obs_window=3, max_steps=args.max_steps, seed=args.seed)
    # TODO: Create trainer
    trainer = MultiAgentTrainer(env, args)
    # TODO: Run training
    trainer.train()
    # TODO: Final evaluation
    avg_reward, success_rate = trainer.evaluate(num_episodes=100)
    print(f"Final Evaluation - Average Reward: {avg_reward}, Success Rate: {success_rate}")

if __name__ == '__main__':
    main()