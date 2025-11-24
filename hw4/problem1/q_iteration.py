"""
Q-Iteration algorithm for solving MDPs.
"""

import numpy as np
from typing import Tuple, Optional
from environment import GridWorldEnv


class QIteration:
    """
    Q-Iteration solver for gridworld MDP.

    Computes optimal action-value function Q* using dynamic programming.
    """

    def __init__(self, env: GridWorldEnv, gamma: float = 0.95, epsilon: float = 1e-4):
        """
        Initialize Q-Iteration solver.

        Args:
            env: GridWorld environment
            gamma: Discount factor
            epsilon: Convergence threshold
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = env.grid_size ** 2
        self.n_actions = env.action_space

    def solve(self, max_iterations: int = 1000) -> Tuple[np.ndarray, int, list]:
        """
        Run Q-iteration until convergence.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            q_values: Converged Q-function Q(s,a)
            n_iterations: Number of iterations until convergence
            history: List of maximum Q-function changes per iteration
        """
        # TODO: Initialize Q-function to zeros (shape: [n_states, n_actions])
        q_values = np.zeros((self.n_states, self.n_actions))
        history = []

        # TODO: Iterate until convergence:
        #       - For each state-action pair:
        #           - Compute updated Q-value using Bellman equation:
        #             Q(s,a) = sum_s' P(s'|s,a) * [R(s,a,s') + gamma * max_a' Q(s',a')]
        #       - Check convergence: max|Q_new - Q_old| < epsilon
        #       - Update Q-function
        for i in range(max_iterations):
            prev_q_values = np.copy(q_values)
            max_diff = 0.0

            for state in range(self.n_states):
                for action in range(self.n_actions):
                    q_new = self.bellman_update(state, action, q_values)
                    q_values[state, action] = q_new
                    diff = abs(q_new - prev_q_values[state, action])
                    if diff > max_diff:
                        max_diff = diff

            history.append(max_diff)
            if max_diff < self.epsilon:
                return q_values, i + 1, history
            
        # TODO: Return final Q-values and iteration count
        return q_values, max_iterations, history

    def bellman_update(self, state: int, action: int, q_values: np.ndarray) -> float:
        """
        Compute updated Q-value for a state-action pair.

        Args:
            state: State index
            action: Action index
            q_values: Current Q-function

        Returns:
            Updated Q-value for (s,a)
        """
        # If current state is terminal, Q(s,a) should be 0 (or just the immediate reward if defined differently)
        # Consistent with Value Iteration where V(terminal) = 0
        if self.env.is_terminal(state):
            return 0.0

        # TODO: Get transition probabilities P(s'|s,a)
        transition_probs = self.env.get_transition_prob(state, action)
        # TODO: For each possible next state:
        #       - Get reward R(s,a,s')
        #       - Get max Q-value for next state: max_a' Q(s',a')
        #       - Accumulate: prob * [reward + gamma * max_q_next]
        q_update = 0.0
        for next_state, prob in transition_probs.items():
            reward = self.env.get_reward(state, action, next_state)
            max_q_next = np.max(q_values[next_state])
            q_update += prob * (reward + self.gamma * max_q_next)
        # TODO: Return updated Q-value
        return q_update


    def extract_policy(self, q_values: np.ndarray) -> np.ndarray:
        """
        Extract optimal policy from Q-function.

        Args:
            q_values: Optimal Q-function

        Returns:
            policy: Array of optimal actions for each state
        """
        # TODO: For each state:
        #       - Select action with maximum Q-value: argmax_a Q(s,a)
        policy = np.zeros(self.n_states, dtype=int)
        for state in range(self.n_states):
            policy[state] = np.argmax(q_values[state])
        # TODO: Return policy array

        return policy

    def extract_values(self, q_values: np.ndarray) -> np.ndarray:
        """
        Extract value function from Q-function.

        Args:
            q_values: Q-function

        Returns:
            values: State value function V(s) = max_a Q(s,a)
        """
        # TODO: For each state:
        #       - Compute V(s) = max_a Q(s,a)
        values = np.max(q_values, axis=1)
        # TODO: Return value function

        return values

    def compute_bellman_error(self, q_values: np.ndarray) -> float:
        """
        Compute Bellman error for current Q-function.

        Args:
            q_values: Current Q-function

        Returns:
            Maximum Bellman error across all state-action pairs
        """
        # TODO: For each state-action pair:
        #       - Compute updated Q-value using Bellman update
        #       - Calculate absolute difference from current Q-value
        max_error = 0.0
        for state in range(self.n_states):
            for action in range(self.n_actions):
                q_new = self.bellman_update(state, action, q_values)
                error = abs(q_new - q_values[state, action])
                if error > max_error:
                    max_error = error
        # TODO: Return maximum error

        return max_error