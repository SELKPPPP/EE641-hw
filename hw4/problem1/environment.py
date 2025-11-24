"""
Stochastic gridworld environment for reinforcement learning.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict


class GridWorldEnv:
    """
    5x5 Stochastic GridWorld Environment.

    The agent navigates a grid with stochastic transitions:
    - 0.8 probability of moving in the intended direction
    - 0.1 probability of drifting left (perpendicular)
    - 0.1 probability of drifting right (perpendicular)

    Grid layout:
    - Start: (0, 0)
    - Goal: (4, 4)
    - Obstacles: (2, 2), (1, 3)
    - Penalties: (3, 1), (0, 3)
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize gridworld environment.

        Args:
            seed: Random seed for reproducibility
        """
        self.grid_size = 5
        self.max_steps = 50

        # Define special cells
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.obstacles = [(1, 2), (2, 1)]
        self.penalties = [(3, 3), (3, 0)]

        # Rewards
        self.goal_reward = 10.0
        self.penalty_reward = -5.0
        self.step_cost = -0.1

        # Transition probabilities
        self.prob_intended = 0.8
        self.prob_drift = 0.1

        # Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.action_space = 4
        self.action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']

        if seed is not None:
            np.random.seed(seed)

        self.reset()

    def reset(self) -> int:
        """
        Reset environment to initial state.

        Returns:
            state: Initial state index
        """
        # TODO: Initialize agent position to start_pos
        self.agent_pos = self.start_pos
        # TODO: Reset step counter
        self.current_step = 0
        # TODO: Set done flag to False
        self.done = False
        # TODO: Return state index (use _pos_to_state)
        return self._pos_to_state(self.agent_pos)

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """
        Execute action in environment.

        Args:
            action: Action index (0-3)

        Returns:
            next_state: Next state index
            reward: Reward received
            done: Whether episode terminated
            info: Additional information
        """
        # TODO: Check if episode already done
        if self.done:
            raise Exception("Episode has terminated. Please reset the environment.")
        # TODO: Get next position based on stochastic transitions
        # next_positions: (position, probability)
        next_positions = self._get_next_positions(self.agent_pos, action)
        probs = [p for _, p in next_positions]
        choices = [pos for pos, _ in next_positions]
        next_pos = choices[np.random.choice(len(choices), p=probs)]
        self.agent_pos = next_pos
        # TODO: Calculate reward (use _calculate_reward helper)
        reward = self._calculate_reward(self.agent_pos)
        # TODO: Update position and step count
        self.current_step += 1
        # TODO: Check termination conditions
        if self.agent_pos == self.goal_pos or self.current_step >= self.max_steps:
            self.done = True
        # TODO: Return (next_state, reward, done, info)
        return self._pos_to_state(self.agent_pos), reward, self.done, {}

    def get_transition_prob(self, state: int, action: int) -> Dict[int, float]:
        """
        Get transition probabilities P(s'|s,a).

        Args:
            state: Current state index
            action: Action index

        Returns:
            Dictionary mapping next_state -> probability
        """
        # TODO: Convert state to position
        pos = self._state_to_pos(state)

        # TODO: For given action, compute all possible next positions
        #       considering stochastic transitions
        if pos == self.goal_pos:
            return {state: 1.0}  # Terminal state

        next_positions = self._get_next_positions(pos, action)

        # TODO: Handle boundary and obstacle collisions
        transition_probs = {}
        for next_pos, prob in next_positions:
            next_state = self._pos_to_state(next_pos)
            if next_state not in transition_probs:
                transition_probs[next_state] = 0.0
            transition_probs[next_state] += prob
        # TODO: Return probability distribution over next states

        return transition_probs

    def get_reward(self, state: int, action: int, next_state: int) -> float:
        """
        Get reward for transition.

        Args:
            state: Current state index
            action: Action taken
            next_state: Resulting state

        Returns:
            Reward value
        """
        # TODO: Convert next_state to position
        pos = self._state_to_pos(next_state)
        # TODO: Check if goal reached (+10)
        if pos == self.goal_pos:
            return self.goal_reward
        # TODO: Check if penalty cell (-5)
        if pos in self.penalties:
            return self.penalty_reward
        # TODO: Otherwise return step cost (-0.1)
        return self.step_cost

    def is_terminal(self, state: int) -> bool:
        """
        Check if state is terminal.

        Args:
            state: State index

        Returns:
            True if terminal state
        """
        # TODO: Convert state to position
        pos = self._state_to_pos(state)
        # TODO: Return True if position equals goal_pos
        return pos == self.goal_pos

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """
        Convert grid position to state index.

        Args:
            pos: (row, col) position

        Returns:
            State index (0-24)
        """
        # TODO: Convert 2D position to 1D state index
        # State = row * grid_size + col
        row, col = pos
        return row * self.grid_size + col


    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """
        Convert state index to grid position.

        Args:
            state: State index

        Returns:
            (row, col) position
        """
        # TODO: Convert 1D state index to 2D position
        # row = state // grid_size
        # col = state % grid_size
        row = state // self.grid_size
        col = state % self.grid_size
        return (row, col)

    def _is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        """
        Check if position is valid (in bounds and not obstacle).

        Args:
            pos: (row, col) position

        Returns:
            True if valid position
        """
        # TODO: Check if position is within grid bounds
        row, col = pos
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            return False
        # TODO: Check if position is not an obstacle
        if pos in self.obstacles:
            return False

        return True

    def _get_next_positions(self, pos: Tuple[int, int], action: int) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get possible next positions and probabilities for stochastic transition.

        Args:
            pos: Current position
            action: Action to take

        Returns:
            List of (next_position, probability) tuples
        """
        # TODO: Define action effects (deltas for UP, RIGHT, DOWN, LEFT)
        action_effects = [
            (-1, 0),  # UP
            (0, 1),   # RIGHT
            (1, 0),   # DOWN
            (0, -1)   # LEFT
        ]
        # TODO: Get intended direction and perpendicular directions
        intented_action = action
        drift_left_action = (action - 1) % 4
        drift_right_action = (action + 1) % 4
        # TODO: For each possible outcome (intended, drift left, drift right):
        #       - Calculate next position
        #       - If invalid, stay in current position
        #       - Add (position, probability) to list
        outcomes = [
            (intented_action, self.prob_intended),
            (drift_left_action, self.prob_drift),
            (drift_right_action, self.prob_drift)
        ]
        next_positions = []

        for act, prob in outcomes:
            delta = action_effects[act]
            next_pos = (pos[0] + delta[0], pos[1] + delta[1])
            if not self._is_valid_pos(next_pos):
                next_pos = pos  # Stay in place if invalid
            next_positions.append((next_pos, prob))
        # TODO: Merge probabilities for same positions
        merged_positions = {}
        for p, prob in next_positions:
            if p not in merged_positions:
                merged_positions[p] = 0.0
            merged_positions[p] += prob

        return list(merged_positions.items())

    def _calculate_reward(self, pos: Tuple[int, int]) -> float:
        """
        Calculate reward for entering a position.

        Args:
            pos: Position entered

        Returns:
            Reward value
        """
        # TODO: Check if position is goal (+10)
        if pos == self.goal_pos:
            return self.goal_reward
        # TODO: Check if position is penalty (-5)
        if pos in self.penalties:
            return self.penalty_reward
        # TODO: Otherwise return step cost (-0.1)
        return self.step_cost

    def render(self, value_function: Optional[np.ndarray] = None) -> None:
        """
        Render current state of environment.

        Args:
            value_function: Optional value function to display
        """
        # TODO: Create visual representation of grid
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        
        # TODO: Mark current position, goal, obstacles, penalties
        # Start
        grid[self.start_pos[0]][self.start_pos[1]] = 'S'
        # Goal
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        # Obstacles
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'
        # Penalties
        for pen in self.penalties:
            grid[pen[0]][pen[1]] = 'P'
        # Agent
        if hasattr(self, 'agent_pos'):
            grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'

        

        # TODO: If value_function provided, show as heatmap

        print("-" * (self.grid_size * 4 + 1))
        
        for r in range(self.grid_size):
            row_str = "|"
            for c in range(self.grid_size):
                if value_function is not None:
                    state = self._pos_to_state((r, c))
                    val = value_function[state]
                    row_str += f" {val:5.2f} |"
                else:
                    cell = grid[r][c]
                    row_str += f" {cell:^3} |"
            print(row_str)
            print("-" * (self.grid_size * 4 + 1))

