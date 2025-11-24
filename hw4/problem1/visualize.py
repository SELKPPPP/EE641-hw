"""
Visualization utilities for gridworld and policies.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import Optional, Tuple
import os


class GridWorldVisualizer:
    """
    Visualizer for gridworld environment, value functions, and policies.
    """

    def __init__(self, grid_size: int = 5):
        """
        Initialize visualizer.

        Args:
            grid_size: Size of grid
        """
        self.grid_size = grid_size

        # Define special positions
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.obstacles = [(1, 2), (2, 1)]
        self.penalties = [(3, 3), (3, 0)]

    def add_special_cells(self, ax) -> None:    
        """Helper to add S, G, X, P annotations."""
        # Start
        ax.text(self.start_pos[1], self.start_pos[0], 'S', ha='center', va='center', color='black', fontweight='bold', fontsize=14)
        # Goal
        ax.text(self.goal_pos[1], self.goal_pos[0], 'G', ha='center', va='center', color='black', fontweight='bold', fontsize=14)
        # Obstacles
        for r, c in self.obstacles:
            ax.text(c, r, 'X', ha='center', va='center', color='black', fontweight='bold', fontsize=14)
        # Penalties
        for r, c in self.penalties:
            ax.text(c, r, 'P', ha='center', va='center', color='black', fontweight='bold', fontsize=14)

    def plot_value_function(self, values: np.ndarray, title: str = "Value Function") -> None:
        """
        Plot value function as heatmap.

        Args:
            values: Value function V(s) for each state
            title: Plot title
        """
        # TODO: Reshape values to 2D grid
        grid_values = values.reshape((self.grid_size, self.grid_size))
        # TODO: Create heatmap with appropriate colormap
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(grid_values, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Value')
        # TODO: Mark special cells (start, goal, obstacles, penalties)
        self.add_special_cells(ax)
        # TODO: Add colorbar and labels
        # Add values as text in each cell
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Skip obstacles for value text if you want, or just print them
                if (i, j) not in self.obstacles:
                    ax.text(j, i+0.2, f'{grid_values[i, j]:.1f}', ha='center', va='center', color='black', fontsize=12)

        ax.set_title(title)
        ax.set_xticks(np.arange(self.grid_size))
        ax.set_yticks(np.arange(self.grid_size))
        # TODO: Save figure to results/visualizations/
        os.makedirs('results/visualizations', exist_ok=True)
        save_path = f'results/visualizations/{title.lower().replace(" ", "_")}.png'
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")

    def plot_policy(self, policy: np.ndarray, title: str = "Optimal Policy") -> None:
        """
        Plot policy with arrows showing optimal actions.

        Args:
            policy: Array of optimal actions for each state
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        # TODO: Create grid plot
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_xticks(np.arange(self.grid_size))
        ax.set_yticks(np.arange(self.grid_size))
        ax.grid(True)
        ax.invert_yaxis()

        # Mark special cells with background colors
        # Obstacles
        for r, c in self.obstacles:
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='gray'))
        # Goal
        ax.add_patch(plt.Rectangle((self.goal_pos[1]-0.5, self.goal_pos[0]-0.5), 1, 1, color='lightgreen'))
        # Penalties
        for r, c in self.penalties:
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='salmon'))
        # Start
        ax.add_patch(plt.Rectangle((self.start_pos[1]-0.5, self.start_pos[0]-0.5), 1, 1, color='lightblue'))

        self.add_special_cells(ax)
        # TODO: For each state:
        #       - Draw arrow indicating action direction
        #       - Handle special cells appropriately
        for state in range(len(policy)):
            row, col = state // self.grid_size, state % self.grid_size

            # Skip obstacles and goal
            if (row, col) in self.obstacles or (row, col) == self.goal_pos:
                continue

            action = policy[state]
            dx , dy = 0, 0
            if action == 0:   # Up
                dx, dy = 0, -0.3
            elif action == 1: # Right
                dx, dy = 0.3, 0
            elif action == 2: # Down
                dx, dy = 0, 0.3
            elif action == 3: # Left
                dx, dy = -0.3, 0
            else:
                continue  # Invalid action

            ax.arrow(col, row, dx, dy, head_width=0.1, head_length=0.1, fc='k', ec='k')

        ax.set_title(title)    
        # TODO: Save figure to results/visualizations/
        os.makedirs('results/visualizations', exist_ok=True)
        save_path = f'results/visualizations/{title.lower().replace(" ", "_")}.png'
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")

        
    def plot_q_function(self, q_values: np.ndarray, title: str = "Q-Function") -> None:
        """
        Plot Q-function with multiple subplots for each action.

        Args:
            q_values: Q-function Q(s,a)
            title: Plot title
        """
        # TODO: Create subplot for each action
        actions = ['Up', 'Down', 'Left', 'Right']
        fig, axes = plt.subplots(1,4, figsize=(20,5))
        vmin, vmax = np.min(q_values), np.max(q_values)
        # TODO: For each action:
        #       - Show Q-values as heatmap
        #       - Mark special cells
        for a in range(4):
            ax = axes[a]
            
            q_action = q_values[:, a].reshape((self.grid_size, self.grid_size))
            
            im = ax.imshow(q_action, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f'Action: {actions[a]}')
            self.add_special_cells(ax)
        # TODO: Add overall title and save
        fig.colorbar(im, ax=axes.ravel().tolist(), label='Q-Value')
        fig.suptitle(title)
        # Save figure
        os.makedirs('results/visualizations', exist_ok=True)
        save_path = f'results/visualizations/{title.lower().replace(" ", "_")}.png'
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")

    def plot_convergence(self, vi_history: list, qi_history: list) -> None:
        """
        Plot convergence curves for both algorithms.

        Args:
            vi_history: Value iteration convergence history
            qi_history: Q-iteration convergence history
        """
        # TODO: Plot Bellman error vs iteration for both algorithms
        plt.figure(figsize=(10, 6))
        plt.plot(vi_history, label='Value Iteration', color='blue')
        plt.plot(qi_history, label='Q-Iteration', color='orange')
        # TODO: Use log scale for y-axis
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Bellman Error (log scale)')
        # TODO: Add legend and labels
        plt.legend()
        plt.title('Convergence of Value Iteration and Q-Iteration')
        plt.grid(True)
        # TODO: Save figure
        
        os.makedirs('results/visualizations', exist_ok=True)
        save_path = 'results/visualizations/convergence.png'
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")

    def create_comparison_figure(self, vi_values: np.ndarray, qi_values: np.ndarray,
                                vi_policy: np.ndarray, qi_policy: np.ndarray) -> None:
        """
        Create comparison figure showing both algorithms' results.

        Args:
            vi_values: Value function from Value Iteration
            qi_values: Value function from Q-Iteration
            vi_policy: Policy from Value Iteration
            qi_policy: Policy from Q-Iteration
        """
        # TODO: Create 2x2 subplot
        #       - Top left: VI value function
        #       - Top right: QI value function
        #       - Bottom left: VI policy
        #       - Bottom right: QI policy
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        # VI Value Function
        ax = axes[0, 0]
        grid_values = vi_values.reshape((self.grid_size, self.grid_size))
        im = ax.imshow(grid_values, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Value')
        self.add_special_cells(ax)
        ax.set_title("VI Value Function")
        # QI Value Function
        ax = axes[0, 1]
        grid_values = qi_values.reshape((self.grid_size, self.grid_size))
        im = ax.imshow(grid_values, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Value')
        self.add_special_cells(ax)
        ax.set_title("QI Value Function")

        #Helper to plot policy in given axis
        def plot_policy_in_ax(ax, policy, title):
            ax.set_xlim(-0.5, self.grid_size - 0.5)
            ax.set_ylim(-0.5, self.grid_size - 0.5)
            ax.set_xticks(np.arange(self.grid_size))
            ax.set_yticks(np.arange(self.grid_size))
            ax.grid(True)
            ax.invert_yaxis() # Match image coordinates

            # Background colors
            for r, c in self.obstacles:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='gray'))
            ax.add_patch(plt.Rectangle((self.goal_pos[1]-0.5, self.goal_pos[0]-0.5), 1, 1, color='lightgreen'))
            for r, c in self.penalties:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='salmon'))
            ax.add_patch(plt.Rectangle((self.start_pos[1]-0.5, self.start_pos[0]-0.5), 1, 1, color='lightblue'))

            self.add_special_cells(ax)
            
            for state in range(len(policy)):
                row, col = state // self.grid_size, state % self.grid_size
                if (row, col) in self.obstacles or (row, col) == self.goal_pos:
                    continue
                action = policy[state]
                dx , dy = 0, 0
                if action == 0: dx, dy = 0, -0.3 # Up
                elif action == 1: dx, dy = 0.3, 0 # Right
                elif action == 2: dx, dy = 0, 0.3 # Down
                elif action == 3: dx, dy = -0.3, 0 # Left
                
                ax.arrow(col, row, dx, dy, head_width=0.1, head_length=0.1, fc='k', ec='k')
            ax.set_title(title)
        # VI Policy
        plot_policy_in_ax(axes[1, 0], vi_policy, "VI Policy")
        # QI Policy
        plot_policy_in_ax(axes[1, 1], qi_policy, "QI Policy")

        plt.tight_layout()
        # TODO: Highlight any differences
        diff_positions = np.where(vi_policy != qi_policy)[0]
        for pos in diff_positions:
            row, col = pos // self.grid_size, pos % self.grid_size
            for ax in axes[1, :]:
                ax.add_patch(plt.Rectangle((col-0.5, row-0.5), 1, 1, edgecolor='red', facecolor='none', linewidth=2))
        # TODO: Save comprehensive comparison figure

        os.makedirs('results/visualizations', exist_ok=True)
        save_path = 'results/visualizations/comparison.png'
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")


def visualize_results():
    """
    Load and visualize saved results from training.
    """
    viz = GridWorldVisualizer()
    # TODO: Load saved value functions and policies
    # Load VI results
    try:
        vi_data = np.load('results/value_function.npz')
        vi_values = vi_data['values']
        vi_policy = vi_data['policy']
        vi_history = vi_data['history']
        
        viz.plot_value_function(vi_values, "VI Value Function")
        viz.plot_policy(vi_policy, "VI Optimal Policy")
    except Exception as e:
        print(f"Error loading VI results: {e}")
        return

    # Load QI results
    try:
        qi_data = np.load('results/q_value_function.npz')
        qi_q_values = qi_data['q_values']
        qi_values = qi_data['values']
        qi_policy = qi_data['policy']
        qi_history = qi_data['history']
        
        viz.plot_q_function(qi_q_values, "QI Q-Function")
        viz.plot_value_function(qi_values, "QI Value Function")
        viz.plot_policy(qi_policy, "QI Optimal Policy")
    except Exception as e:
        print(f"Error loading QI results: {e}")
        return
    # Convergence plot
    viz.plot_convergence(vi_history, qi_history)
    
    # Comparison plot
    viz.create_comparison_figure(vi_values, qi_values, vi_policy, qi_policy)
    
    print("All visualizations generated.")


if __name__ == '__main__':
    visualize_results()