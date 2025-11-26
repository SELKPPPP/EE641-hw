"""
Multi-agent gridworld environment with partial observations and communication.
"""

import numpy as np
from typing import Tuple, Optional, List


class MultiAgentEnv:
    """
    Two-agent cooperative gridworld with partial observations.

    Agents must coordinate to simultaneously reach a target cell.
    Each agent observes a 3x3 local patch and exchanges communication signals.
    """

    def __init__(self, grid_size: Tuple[int, int] = (10, 10), obs_window: int = 3,
                 max_steps: int = 50, seed: Optional[int] = None):
        """
        Initialize multi-agent environment.

        Args:
            grid_size: Tuple defining grid dimensions (default 10x10)
            obs_window: Size of local observation window (must be odd, default 3)
            max_steps: Maximum steps per episode
            seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.obs_window = obs_window
        self.max_steps = max_steps

        if seed is not None:
            np.random.seed(seed)

        # Initialize grid components
        self._initialize_grid()

        # Agent state
        self.agent_positions = [None, None]
        self.comm_signals = [0.0, 0.0]
        self.step_count = 0

    def _initialize_grid(self) -> None:
        """
        Create grid with obstacles and target.

        Grid values:
        - 0: Free cell
        - 1: Obstacle
        - 2: Target
        """
        # TODO: Create empty grid of size grid_size
        self.grid = np.zeros(self.grid_size, dtype=int)
        # TODO: Randomly place up to 6 obstacles (avoiding corners)
        corners = [(0, 0), (0, self.grid_size[1]-1), (self.grid_size[0]-1, 0), (self.grid_size[0]-1, self.grid_size[1]-1)]
        obstacle_count = 0
        num_obstacles = np.random.randint(1, 7) # 1 to 6 obstacles
        while obstacle_count < num_obstacles:
            r = np.random.randint(0, self.grid_size[0])
            c = np.random.randint(0, self.grid_size[1])
            if (r, c) not in corners and self.grid[r, c] == 0:
                self.grid[r, c] = 1
                obstacle_count += 1
        # TODO: Randomly place exactly 1 target cell
        target_placed = False
        while not target_placed:
            r = np.random.randint(0, self.grid_size[0])
            c = np.random.randint(0, self.grid_size[1])
            if self.grid[r, c] == 0:
                self.grid[r, c] = 2
                self.target_position = (r, c)
                target_placed = True
        # TODO: Store grid as self.grid



    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset environment to initial state.

        Returns:
            obs_A: Observation for Agent A (11-dimensional vector)
            obs_B: Observation for Agent B (11-dimensional vector)

        Observation format:
        - Elements 0-8: Flattened 3x3 grid patch (row-major order)
        - Element 9: Communication signal from other agent
        - Element 10: Normalized L2 distance between agents
        """
        # TODO: Reset step counter
        self.step_count = 0
        self._initialize_grid()
        # TODO: Randomly place both agents on free cells (not obstacles or target)
        free_cells = self._find_free_cells()
        start_positions = np.random.choice(len(free_cells), size=2, replace=False)
        self.agent_positions[0] = free_cells[start_positions[0]]
        self.agent_positions[1] = free_cells[start_positions[1]]

        # TODO: Initialize communication signals to 0.0
        self.comm_signals = [0.0, 0.0]
        # TODO: Generate observations for both agents
        obs_A = self._get_observation(0)
        obs_B = self._get_observation(1)
        # TODO: Return (obs_A, obs_B)
        return (obs_A, obs_B)

    def step(self, action_A: int, action_B: int, comm_A: float, comm_B: float) -> \
            Tuple[Tuple[np.ndarray, np.ndarray], float, bool]:
        """
        Execute one environment step.

        Args:
            action_A: Agent A's movement action (0:Up, 1:Down, 2:Left, 3:Right, 4:Stay)
            action_B: Agent B's movement action
            comm_A: Communication signal from Agent A to B
            comm_B: Communication signal from Agent B to A

        Returns:
            observations: Tuple of (obs_A, obs_B), each 11-dimensional
            reward: +10 if both agents at target, +2 if one agent at target, -0.1 per step
            done: True if both agents at target or max steps reached
        """
        # TODO: Update agent positions based on actions
        #       - Check boundaries and obstacles
        #       - Invalid moves result in no position change
        self.agent_positions[0] = self._apply_action(self.agent_positions[0], action_A)
        self.agent_positions[1] = self._apply_action(self.agent_positions[1], action_B)

        # TODO: Store new communication signals for next observation
        self.comm_signals = [comm_A, comm_B]

        # TODO: Check reward condition (both agents at target)
        at_target_A = (self.agent_positions[0] == self.target_position)
        at_target_B = (self.agent_positions[1] == self.target_position)
        done = False
        reward = -0.1 
        if at_target_A and at_target_B:
            reward += 10.0
        elif at_target_A or at_target_B:
            reward += 2.0    

        # TODO: Update step count and check termination
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
        # TODO: Generate new observations with updated comm signals
        obs_A = self._get_observation(0)
        obs_B = self._get_observation(1)
        # TODO: Return ((obs_A, obs_B), reward, done)
        return ((obs_A, obs_B), reward, done)
    def _get_observation(self, agent_idx: int) -> np.ndarray:
        """
        Extract local observation for an agent.

        Args:
            agent_idx: Agent index (0 for A, 1 for B)

        Returns:
            observation: 10-dimensional vector
        """
        # TODO: Get agent position
        agent_pos = self.agent_positions[agent_idx]
        other_agent_idx = 1 - agent_idx
        other_agent_pos = self.agent_positions[other_agent_idx]
        # TODO: Extract 3x3 patch centered on agent
        #       - Cells outside grid should be -1
        #       - Use grid values (0: free, 1: obstacle, 2: target)
        patch = []
        row, col = agent_pos
        offset = self.obs_window // 2

        for dr in range(row-offset, row+offset+1):
            for dc in range(col-offset, col+offset+1):
                if 0 <= dr < self.grid_size[0] and 0 <= dc < self.grid_size[1]:
                    patch.append(self.grid[dr, dc])
                else:
                    patch.append(-1)  # Out of bounds
        # TODO: Flatten patch to 9 elements
        obs_vector = np.array(patch, dtype=np.float32)
        # TODO: Append communication signal from other agent
        comm_signal = self.comm_signals[other_agent_idx]
        obs_vector = np.append(obs_vector, np.float32(comm_signal))

        # TODO: Compute normalized L2 distance between agents and append
        dist = np.sqrt((agent_pos[0] - other_agent_pos[0])**2 + (agent_pos[1] - other_agent_pos[1])**2)
        max_dist = np.sqrt(self.grid_size[0]**2 + self.grid_size[1]**2)
        norm_dist = dist / max_dist
        obs_vector = np.append(obs_vector, np.float32(norm_dist))   
        # TODO: Return 11-dimensional observation
        return obs_vector

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        Check if position is valid (in bounds and not obstacle).

        Args:
            pos: (row, col) position

        Returns:
            True if valid position
        """
        # TODO: Check if position is within grid bounds
        if not (0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]):
            return False
        # TODO: Check if position is not an obstacle (grid value != 1)
        if self.grid[pos[0], pos[1]] == 1:
            return False
        return True

    def _apply_action(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Apply movement action to position.

        Args:
            pos: Current position (row, col)
            action: Movement action (0-4)

        Returns:
            new_pos: Updated position (stays same if invalid)
        """
        # TODO: Map action to position delta
        #       0: Up (-1, 0)
        #       1: Down (+1, 0)
        #       2: Left (0, -1)
        #       3: Right (0, +1)
        #       4: Stay (0, 0)
        delta_map = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
            4: (0, 0)
        }
        delta = delta_map.get(action, (0, 0))
        # TODO: Calculate new position
        new_row = pos[0] + delta[0]
        new_col = pos[1] + delta[1]
        new_pos = (new_row, new_col)
        # TODO: Return new position if valid, else return original position
        if self._is_valid_position(new_pos):
            return new_pos
        else:
            return pos

    def _find_free_cells(self) -> List[Tuple[int, int]]:
        """
        Find all free cells in the grid.

        Returns:
            List of (row, col) positions that are free
        """
        # TODO: Iterate through grid
        
        # TODO: Collect positions where grid value is 0 (free)
        # TODO: Return list of free positions
        free_cells = []
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                if self.grid[r, c] == 0:
                    free_cells.append((r, c))
        return free_cells
    

    def render(self) -> None:
        """
        Render current environment state.
        """
        # TODO: Create visual representation of grid
        rows, cols = self.grid_size
        grid_visual = [['.' for _ in range(cols)] for _ in range(rows)]
        # TODO: Show agent positions (A, B)
        pos_a = self.agent_positions[0]
        pos_b = self.agent_positions[1]
        if pos_a is not None and pos_b is not None:
            if pos_a == pos_b:
                grid_visual[pos_a[0]][pos_a[1]] = 'AB'
            else:
                grid_visual[pos_a[0]][pos_a[1]] = 'A'
                grid_visual[pos_b[0]][pos_b[1]] = 'B'
        # TODO: Show target (T)
        # TODO: Show obstacles (X)
        for r in range(rows):
            for c in range(cols):
                if self.grid[r, c] == 1:
                    grid_visual[r][c] = 'X'
                elif self.grid[r, c] == 2:
                    grid_visual[r][c] = 'T'
        # TODO: Display current communication values
        print("-" * (cols * 4 + 1))

        for r in range(rows):
            row_str = "|"
            for c in range(cols):
                row_str += f" {grid_visual[r][c]:^3} |"
            print(row_str)
            print("-" * (cols * 4 + 1))

        print(f"Step: {self.step_count}")    

        print(f"Comm A->B: {self.comm_signals[0]:.2f}, Comm B->A: {self.comm_signals[1]:.2f}")