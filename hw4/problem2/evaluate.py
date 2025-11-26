
"""
Evaluation script for trained multi-agent models.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import json
import os
from multi_agent_env import MultiAgentEnv
from models import AgentDQN


class MultiAgentEvaluator:
    """
    Evaluator for analyzing trained multi-agent policies.
    """

    def __init__(self, env: MultiAgentEnv, model_A: nn.Module, model_B: nn.Module):
        """
        Initialize evaluator.

        Args:
            env: Multi-agent environment
            model_A: Trained model for Agent A
            model_B: Trained model for Agent B
        """
        self.env = env
        self.model_A = model_A
        self.model_B = model_B
        # Use CPU for small networks
        self.device = torch.device("cpu")

        # Move models to device and set to evaluation mode
        self.model_A.to(self.device)
        self.model_B.to(self.device)
        self.model_A.eval()
        self.model_B.eval()

    def run_episode(self, render: bool = False) -> Tuple[float, bool, Dict]:
        """
        Run single evaluation episode.

        Args:
            render: Whether to render environment

        Returns:
            reward: Episode reward
            success: Whether target was reached
            info: Episode statistics
        """
        # TODO: Reset environment
        obs_A, obs_B = self.env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        comm_history = []
        positions_A = []
        positions_B = []
        success = False
        # TODO: Initialize episode tracking
        while not done:
            positions_A.append(tuple(self.env.agent_positions[0]))
            positions_B.append(tuple(self.env.agent_positions[1]))
            # TODO: Run episode with greedy policy
            state_A_tensor = torch.FloatTensor(obs_A).unsqueeze(0).to(self.device)
            state_B_tensor = torch.FloatTensor(obs_B).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_A, comm_A = self.model_A(state_A_tensor)
                q_B, comm_B = self.model_B(state_B_tensor)
                action_A = q_A.argmax().item()
                action_B = q_B.argmax().item()
                comm_A_val = comm_A.item()
                comm_B_val = comm_B.item()
            # TODO: Track communication patterns
            comm_history.append((comm_A_val, comm_B_val))
            (next_obs_A, next_obs_B), reward, done = self.env.step(action_A, action_B, comm_A_val, comm_B_val)
            obs_A, obs_B = next_obs_A, next_obs_B
            total_reward += reward
            step_count += 1
            if reward >= 7.0:
                success = True
            if render:
                self.env.render()
        # TODO: Return results and statistics
        info = {
            'steps': step_count,
            'comm_history': comm_history,
            'positions_A': positions_A,
            'positions_B': positions_B
        }
        return total_reward, success, info

    def evaluate_performance(self, num_episodes: int = 100) -> Dict:
        """
        Evaluate overall performance statistics.

        Args:
            num_episodes: Number of evaluation episodes

        Returns:
            Statistics dictionary
        """
        # TODO: Run multiple episodes
        rewards = []
        successes = 0
        steps_list = []
        for _ in range(num_episodes):
            reward, success, info = self.run_episode(render=False)
            rewards.append(reward)
            if success:
                successes += 1
            # TODO: Analyze path lengths
            steps_list.append(info['steps'])
        # TODO: Compute success rate
        # TODO: Measure coordination efficiency
        stats = {
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'success_rate': successes / num_episodes,
            'mean_steps': float(np.mean(steps_list)),
            'std_steps': float(np.std(steps_list)),
        }
        # TODO: Return comprehensive statistics
        return stats

    def analyze_communication(self, num_episodes: int = 20) -> Dict:
        """
        Analyze emergent communication protocols.

        Returns:
            Communication analysis results
        """
        # TODO: Track communication signals over episodes
        comm_A_all = []
        comm_B_all = []
        for _ in range(num_episodes):
            _, _, info = self.run_episode(render=False)
            comm_A, comm_B = zip(*info['comm_history'])
            comm_A_all.extend(comm_A)
            comm_B_all.extend(comm_B)
        comm_A_arr = np.array(comm_A_all)
        comm_B_arr = np.array(comm_B_all)
        # TODO: Analyze signal patterns (magnitude, variance, correlation)
        analysis = {
            'comm_A_mean': float(np.mean(comm_A_arr)),
            'comm_A_std': float(np.std(comm_A_arr)),
            'comm_B_mean': float(np.mean(comm_B_arr)),
            'comm_B_std': float(np.std(comm_B_arr)),
            'comm_correlation': float(np.corrcoef(comm_A_arr, comm_B_arr)[0, 1]) if len(comm_A_arr) > 1 else 0.0
        }
        # TODO: Identify communication strategies
        # TODO: Return analysis results
        return analysis

    def visualize_trajectory(self, save_path: str = 'results/trajectory.png') -> None:
        """
        Visualize agent trajectories in an episode.

        Args:
            save_path: Path to save visualization
        """
        # TODO: Run episode while tracking positions
        reward, success, info = self.run_episode(render=False)
        positions_A = np.array(info['positions_A'])
        positions_B = np.array(info['positions_B'])
        grid = self.env.grid.copy()
        # TODO: Create grid visualization
        plt.figure(figsize=(8, 8))
        plt.imshow(grid, cmap='gray', origin='upper')
        # TODO: Plot agent paths
        plt.plot(positions_A[:, 1], positions_A[:, 0], 'r-o', label='Agent A')
        plt.plot(positions_B[:, 1], positions_B[:, 0], 'b-o', label='Agent B')
        # TODO: Mark key events (near target, coordination points)
        plt.scatter(self.env.target_position[1], self.env.target_position[0], c='g', marker='*', s=200, label='Target')
        plt.title('Agent Trajectories')
        plt.legend()
        plt.grid(True)
        # TODO: Save figure
        plt.savefig(save_path)
        plt.close()

    def plot_communication_heatmap(self, save_path: str = 'results/comm_heatmap.png') -> None:
        """
        Create heatmap of communication signals across grid positions.

        Args:
            save_path: Path to save figure
        """
        # TODO: Sample communication signals at each grid position
        grid_shape = self.env.grid_size
        heatmap_A = np.zeros(grid_shape)
        heatmap_B = np.zeros(grid_shape)
        count_A = np.zeros(grid_shape)
        count_B = np.zeros(grid_shape)
        for _ in range(20):
            _, _, info = self.run_episode(render=False)
            for pos_A, pos_B, (comm_A, comm_B) in zip(info['positions_A'], info['positions_B'], info['comm_history']):
                rA, cA = pos_A
                rB, cB = pos_B
                heatmap_A[rA, cA] += comm_A
                heatmap_B[rB, cB] += comm_B
                count_A[rA, cA] += 1
                count_B[rB, cB] += 1

        # TODO: Create heatmaps for both agents
        avg_heatmap_A = np.divide(heatmap_A, count_A, out=np.zeros_like(heatmap_A), where=count_A != 0)
        avg_heatmap_B = np.divide(heatmap_B, count_B, out=np.zeros_like(heatmap_B), where=count_B != 0)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.title('Agent A Communication Heatmap')
        plt.imshow(avg_heatmap_A, cmap='hot', origin='upper')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title('Agent B Communication Heatmap')
        plt.imshow(avg_heatmap_B, cmap='hot', origin='upper')
        plt.colorbar()
        # TODO: Show correlation with distance to target
        # TODO: Save visualization
        plt.savefig(save_path)
        plt.close()

    def test_generalization(self, num_configs: int = 10) -> Dict:
        """
        Test generalization to new environment configurations.

        Args:
            num_configs: Number of test configurations

        Returns:
            Generalization performance statistics
        """
        # TODO: Generate new obstacle configurations
        rewards = []
        successes = 0
        steps_list = []
        for _ in range(num_configs):
            self.env._initialize_grid()  
            obs_A, obs_B = self.env.reset()
            # TODO: Test performance on each configuration
            reward, success, info = self.run_episode(render=False)
            rewards.append(reward)
            if success:
                successes += 1
            steps_list.append(info['steps'])
        # TODO: Compare to training performance
        # TODO: Return generalization metrics
        stats = {
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'success_rate': successes / num_configs,
            'mean_steps': float(np.mean(steps_list)),
            'std_steps': float(np.std(steps_list)),
        }
        return stats


def load_trained_models(checkpoint_dir: str) -> Tuple[nn.Module, nn.Module]:
    """
    Load trained agent models from checkpoint.

    Args:
        checkpoint_dir: Directory containing saved models

    Returns:
        model_A: Agent A's trained model
        model_B: Agent B's trained model
    """
    # TODO: Load model architectures
    checkpoint = torch.load(os.path.join(checkpoint_dir, "final_model.pth"), map_location=torch.device("cpu"))
    model_A = AgentDQN(input_dim=11, hidden_dim=64, num_actions=5)
    model_B = AgentDQN(input_dim=11, hidden_dim=64, num_actions=5)
    # TODO: Load trained weights
    model_A.load_state_dict(checkpoint['agent_A'])
    model_B.load_state_dict(checkpoint['agent_B'])
    # TODO: Return initialized models
    return model_A, model_B


def create_evaluation_report(results: Dict, save_path: str = 'results/evaluation_report.json') -> None:
    """
    Create comprehensive evaluation report.

    Args:
        results: Evaluation results
        save_path: Path to save report
    """
    # TODO: Format results
    # TODO: Add summary statistics
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    # TODO: Save as JSON report

def plot_training_curves(log_files: list[str], config_names: list[str], save_path: str = 'results/training_curves.png') -> None:
    """
    Plot training curves for all configurations showing average reward and success rate.

    Args:
        log_files: List of paths to training_log.json files for each configuration
        config_names: List of configuration names (e.g., ['independent', 'comm', 'full'])
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    for log_file, config in zip(log_files, config_names):
        with open(log_file, 'r') as f:
            log = json.load(f)
        episodes = [entry['episode'] for entry in log]
        rewards = [entry['reward'] for entry in log]
        success_rates = [entry['success_rate'] for entry in log]
        plt.subplot(1, 2, 1)
        plt.plot(episodes, rewards, label=f'{config}')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Training Reward Curve')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(episodes, success_rates, label=f'{config}')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.title('Training Success Rate Curve')
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    """
    Run full evaluation suite on trained models.
    """
    # TODO: Load trained models
    checkpoint_dir = 'results/full'  
    model_A, model_B = load_trained_models(checkpoint_dir)
    # TODO: Create environment
    env = MultiAgentEnv(grid_size=(10, 10), obs_window=3, max_steps=50, seed=641)
    # TODO: Initialize evaluator
    evaluator = MultiAgentEvaluator(env, model_A, model_B)
    # TODO: Run performance evaluation
    perf_stats = evaluator.evaluate_performance(num_episodes=100)
    # TODO: Analyze communication
    comm_stats = evaluator.analyze_communication(num_episodes=20)
    # TODO: Test generalization
    gen_stats = evaluator.test_generalization(num_configs=10)
    # TODO: Create visualizations
    os.makedirs('results', exist_ok=True)
    evaluator.visualize_trajectory(save_path='results/trajectory.png')
    evaluator.plot_communication_heatmap(save_path='results/comm_heatmap.png')
    # TODO: Generate report
    results = {
        'performance': perf_stats,
        'communication': comm_stats,
        'generalization': gen_stats
    }
    create_evaluation_report(results, save_path='results/evaluation_report.json')
    plot_training_curves(
        log_files=[
            'results/independent/training_log.json',
            'results/comm/training_log.json',
            'results/full/training_log.json',
        ],
        config_names=['independent', 'comm', 'full'],
        save_path='results/training_curves.png'
    )
    print('Evaluation complete. Report saved to results/evaluation_report.json')


if __name__ == '__main__':
    main()