import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        """
        Actor network for policy learning.
        - state_dim: The size of the state space (input).
        - action_dim: The size of the action space (output).
        - max_action: The maximum action value (used to scale output).
        """
        super(Actor, self).__init__()
        # Define the layers of the Actor network
        self.layer1 = nn.Linear(state_dim, 256)         # First hidden layer with 256 neurons
        self.layer2 = nn.Linear(256, 256)               # Second hidden layer with 256 neurons
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 256)
        self.layer5 = nn.Linear(256, action_dim)        # Output layer, outputs action_dim neurons
        self.max_action = max_action                    # The maximum action value, to scale the output

    def forward(self, state):
        """
        Forward pass through the Actor network to output an action.
        - state: The input state from the environment.
        """
        x = torch.relu(self.layer1(state))                  # Apply ReLU activation after the first layer
        x = torch.relu(self.layer2(x))                      # Apply ReLU activation after the second layer
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        # x = self.max_action * torch.tanh(self.layer3(x))  # Apply tanh to bound action between -1 and 1, then scale by max_action
        x = self.max_action * torch.sigmoid(self.layer5(x)) # Apply sigmoid to bound action between 0 and 1, then scale by max_action
        return x                                            # Return the predicted action

# Twin Q-networks (Critic)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Critic network for value estimation (Q-value).
        - state_dim: The size of the state space (part of input).
        - action_dim: The size of the action space (part of input).
        """
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 256)    # First hidden layer takes state and action as input
        self.layer2 = nn.Linear(256, 256)                       # Second hidden layer with 256 neurons
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 256)
        self.layer5 = nn.Linear(256, 1)                         # Output layer, outputs a single Q-value

    def forward(self, state, action):
        """
        Forward pass through the Critic network to output a Q-value.
        - state: The input state.
        - action: The input action (concatenated with the state).
        """
        # Concatenate the state and action as input to the Critic
        x = torch.cat([state, action], 1)       # Concatenate along dimension 1 (features)
        # Forward pass through the network
        x = torch.relu(self.layer1(x))          # Apply ReLU activation after the first layer
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))          # Apply ReLU activation after the second layer
        x = torch.relu(self.layer4(x))
        x = self.layer5(x)                      # Output a single Q-value (no activation on the output)
        return x                                # Return the Q-value for the given state-action pair

# Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        """
        Initialize the replay buffer with a specified maximum size.
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size) # Initialize a deque with a max length of buffer_size
        self.iterations = 0                     # Track the number of transitions added to the buffer

    def add(self, state, action, next_state, reward, done):
        """
        Add a new transition to the buffer.
        """
        transition = Transition(state, action, next_state, reward, done)    # Create a transition tuple
        self.buffer.append(transition)                                      # Append the transition to the buffer
        self.iterations += 1                                                # Increment the iteration count

    def sample(self, batch_size):
        """
        Sample a random batch of transitions from the buffer.
        """
        batch = random.sample(self.buffer, batch_size)                  # Randomly sample a batch of transitions from the buffer
        # Unpack the batch into separate components
        states, actions, next_states, rewards, dones = zip(*batch)
        # Return each component as a torch tensor with appropriate data types
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),    # Unsqueeze to ensure correct tensor shape
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)       # Unsqueeze to ensure correct tensor shape
        )

    def __len__(self):
        return len(self.buffer)