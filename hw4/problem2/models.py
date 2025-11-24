"""
Neural network models for multi-agent DQN with communication.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AgentDQN(nn.Module):
    """
    Deep Q-Network for agent with communication capability.

    Network processes observations and outputs both Q-values and communication signal.
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_actions: int = 5):
        """
        Initialize DQN with dual outputs.

        Args:
            input_dim: Dimension of input observation (default 10)
            hidden_dim: Number of hidden units
            num_actions: Number of discrete actions (default 5)
        """
        super(AgentDQN, self).__init__()

        # TODO: Define network layers
        #       - Input layer: input_dim -> hidden_dim
        #       - Hidden layers (at least one more)
        #       - Action head: outputs Q-values for each action
        #       - Communication head: outputs single scalar

        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            action_values: Q-values for each action [batch_size, num_actions]
            comm_signal: Communication signal in [0,1] [batch_size, 1]
        """
        # TODO: Pass input through shared layers
        # TODO: Compute action Q-values through action head
        # TODO: Compute communication signal through comm head
        # TODO: Apply sigmoid to bound communication in [0,1]
        # TODO: Return (action_values, comm_signal)

        raise NotImplementedError


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture for improved value estimation.

    Separates value and advantage streams for better learning.
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_actions: int = 5):
        """
        Initialize Dueling DQN.

        Args:
            input_dim: Dimension of input observation
            hidden_dim: Number of hidden units
            num_actions: Number of discrete actions
        """
        super(DuelingDQN, self).__init__()

        # TODO: Define shared feature layers
        # TODO: Define value stream (outputs single value)
        # TODO: Define advantage stream (outputs advantages for each action)
        # TODO: Define communication head

        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dueling architecture.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            q_values: Combined Q-values [batch_size, num_actions]
            comm_signal: Communication signal in [0,1] [batch_size, 1]
        """
        # TODO: Compute shared features
        # TODO: Compute state value V(s)
        # TODO: Compute advantages A(s,a)
        # TODO: Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # TODO: Compute communication signal
        # TODO: Apply sigmoid to bound communication in [0,1]
        # TODO: Return (q_values, comm_signal)

        raise NotImplementedError