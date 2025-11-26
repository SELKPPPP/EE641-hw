"""
Experience replay buffer for multi-agent DQN training.
"""

import numpy as np
import random
from typing import Tuple, List, Optional
from collections import deque


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.

    Stores joint experiences from both agents for coordinated learning.
    """

    def __init__(self, capacity: int = 10000, seed: Optional[int] = None):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            seed: Random seed for sampling
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def push(self, state_A: np.ndarray, state_B: np.ndarray,
             action_A: int, action_B: int,
             comm_A: float, comm_B: float,
             reward: float,
             next_state_A: np.ndarray, next_state_B: np.ndarray,
             done: bool) -> None:
        """
        Store a transition in the buffer.

        Args:
            state_A: Agent A's observation
            state_B: Agent B's observation
            action_A: Agent A's action
            action_B: Agent B's action
            comm_A: Communication from A to B
            comm_B: Communication from B to A
            reward: Shared reward
            next_state_A: Agent A's next observation
            next_state_B: Agent B's next observation
            done: Whether episode terminated
        """
        # TODO: Create transition tuple
        transiton = (state_A, state_B, action_A, action_B,
                     comm_A, comm_B, reward,
                     next_state_A, next_state_B, done)
        # TODO: Add to buffer (automatic removal of oldest if at capacity)
        self.buffer.append(transiton)

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Batch of transitions as separate arrays for each component
        """
        # TODO: Sample batch_size transitions randomly
        batach = random.sample(self.buffer, batch_size)
        # TODO: Separate components into individual arrays
        # TODO: Convert to appropriate numpy arrays
        state_A, state_B, action_A, action_B, \
        comm_A, comm_B, reward, \
        next_state_A, next_state_B, done = zip(*batach)
        # TODO: Return tuple of arrays

        return (np.array(state_A), np.array(state_B),
                np.array(action_A),  np.array(action_B),
                np.array(comm_A), np.array(comm_B),
                np.array(reward),
                np.array(next_state_A), np.array(next_state_B),
                np.array(done))

    def __len__(self) -> int:
        """
        Get current size of buffer.

        Returns:
            Number of transitions in buffer
        """
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay for importance sampling.

    Samples transitions based on TD-error magnitude.
    """

    def __init__(self, capacity: int = 10000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_steps: int = 100000,
                 seed: Optional[int] = None):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_steps: Steps to anneal beta to 1.0
            seed: Random seed
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        self.frame = 1

        # TODO: Initialize data storage
        self.buffer = []
        self.pos = 0
        # TODO: Initialize priority tree (sum-tree or similar)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        # TODO: Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)


    def push(self, *args, **kwargs) -> None:
        """
        Store transition with maximum priority.

        New transitions get maximum priority to ensure they're sampled at least once.
        """
        # TODO: Store transition
        if len(args) == 10:
            transition = args
        elif len(kwargs) == 10:
            transition = (kwargs['state_A'], kwargs['state_B'],
                          kwargs['action_A'], kwargs['action_B'],
                          kwargs['comm_A'], kwargs['comm_B'],
                          kwargs['reward'],
                          kwargs['next_state_A'], kwargs['next_state_B'],
                          kwargs['done'])
            
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
            

        # TODO: Assign maximum priority to new transition

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample batch with prioritization.

        Returns:
            transitions: Batch of transitions
            weights: Importance sampling weights
            indices: Indices for updating priorities
        """
        N = len(self.buffer)
        if N == 0:
            return None, None, None
        # TODO: Update beta based on schedule
        self.beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_steps)
        self.frame += 1
        # TODO: Sample transitions based on priorities
        priorities = self.priorities[:N]
        probas = priorities ** self.alpha
        probas /= probas.sum()
        indices = np.random.choice(N, batch_size, p=probas)
        samples = [self.buffer[idx] for idx in indices]
        # TODO: Calculate importance sampling weights
        weights = (N * probas[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        # TODO: Return transitions, weights, and indices
        state_A, state_B, action_A, action_B, \
        comm_A, comm_B, reward, \
        next_state_A, next_state_B, done = zip(*samples)
        transitions = (np.array(state_A), np.array(state_B),
                       np.array(action_A),  np.array(action_B),
                       np.array(comm_A), np.array(comm_B),
                       np.array(reward),
                       np.array(next_state_A), np.array(next_state_B),
                       np.array(done))

        return transitions, weights, indices

    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """
        Update priorities for sampled transitions.

        Args:
            indices: Indices of transitions to update
            priorities: New priority values (typically TD-errors)
        """
        # TODO: Update priorities for given indices
        # TODO: Apply alpha exponent for prioritization
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha

    def __len__(self) -> int:
        """
        Get current size of buffer.

        Returns:
            Number of transitions in buffer
        """
        return len(self.buffer)
