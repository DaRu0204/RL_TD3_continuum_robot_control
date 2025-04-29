import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import wandb
# from wandb.keras import WandbCallback
from wandb.integration.keras import WandbCallback
import os
from ActorCritic import Actor
from ActorCritic import Critic

# TD3 algorithm
class TD3:

    lr_actor = 0.0001       # Learning rate for the Actor model (0.0001)
    lr_critic1 = 0.0003     # Learning rate for the first Critic model (0.0003)
    lr_critic2 = 0.0003     # Learning rate for the second Critic model (0.0003)
    gamma = 0.98            # Discount factor (gamma)
    tau = 0.005             # Soft update parameter(tau)

    def __init__(self, state_dim, action_dim, max_action):
        # Initialize the Actor network and its target network
        self.actor = Actor(state_dim, action_dim, max_action)                               # Main Actor network
        self.actor_target = Actor(state_dim, action_dim, max_action)                        # Target Actor network
        self.actor_target.load_state_dict(self.actor.state_dict())                          # Hard copy initial weights to target
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=TD3.lr_actor)         # 0.001
        
        # Initialize the first Critic network and its target
        self.critic1 = Critic(state_dim, action_dim)                                        # First Critic network
        self.critic1_target = Critic(state_dim, action_dim)                                 # Target Critic 1 network
        self.critic1_target.load_state_dict(self.critic1.state_dict())                      # Hard copy initial weights to target
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=TD3.lr_critic1)   # 0.002

        # Initialize the second Critic network and its target
        self.critic2 = Critic(state_dim, action_dim)                                        # Second Critic network
        self.critic2_target = Critic(state_dim, action_dim)                                 # Target Critic 2 network
        self.critic2_target.load_state_dict(self.critic2.state_dict())                      # Hard copy initial weights to target
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=TD3.lr_critic2)   # 0.002

        self.max_action = max_action                                                        # Maximum action value (for scaling actions)

    def select_action(self, state):
        """Select action using the Actor network, given the current state."""
        state = np.array(state)
        state = torch.FloatTensor(state.reshape(1, -1))             # Reshape state to 2D tensor and convert it to torch.FloatTensor for PyTorch model
        # print(f"State shape: {state.shape}")
        # print(self.actor)
        action = self.actor(state).cpu().data.numpy().flatten()
        # if noise != 0:
            # action += np.random.uniform(0, noise, size=action.shape)
        return action       # Return action predicted by the Actor network

    def train(self, replay_buffer, batch_size=64, gamma=gamma, noise=0.1, policy_noise=0.1, noise_clip=0.2, policy_freq=3): # noise=0.1
        """Train the TD3 agent using experiences from the replay buffer."""
        # Check if the replay buffer contains enough samples
        if len(replay_buffer) < batch_size:
            return      # Skip training if there are not enough samples in the buffer

        # Sample a batch from the replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # No gradients required during target computation
        with torch.no_grad():
            # Add noise to the target actions for exploration
            noise = (torch.randn_like(action) * noise).clamp(-noise_clip, noise_clip)
            # next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            next_action = (self.actor_target(next_state) + noise).clamp(0, self.max_action)

            # Compute the target Q values using the target Critic networks
            target_Q1 = self.critic1_target(next_state, next_action)
            target_Q2 = self.critic2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * gamma * target_Q

        # Get current Q values from the main Critic networks
        current_Q1 = self.critic1(state, action)
        current_Q2 = self.critic2(state, action)

        # Compute loss for both Critic networks
        critic1_loss = nn.MSELoss()(current_Q1, target_Q.detach())
        critic2_loss = nn.MSELoss()(current_Q2, target_Q.detach())

        # Update Critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()     # Backpropagate the loss for Critic 1
        self.critic1_optimizer.step()

        # Update Critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()     # Backpropagate the loss for Critic 2
        self.critic2_optimizer.step()

        # Update the Actor network (only once every policy_freq steps)
        if replay_buffer.iterations % policy_freq == 0:
            # Actor loss: we want to maximize the Q-value predicted by Critic 1 given the action from the Actor
            actor_loss = -self.critic1(state, self.actor(state)).mean()

            # Update Actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update for the target networks (with tau)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

        return critic1_loss, critic2_loss

    def save(self, filename):
        #directory = "RL_TD3_continuum_robot_control_5/LearnedModel"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "RL_TD3_continuum_robot_control"))
        directory = os.path.join(base_dir, "TD3LearnedModel")
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save the main Actor and Critic models
        torch.save(self.actor.state_dict(), os.path.join(directory, filename + "_actor"))
        torch.save(self.critic1.state_dict(), os.path.join(directory,filename + "_critic1"))
        torch.save(self.critic2.state_dict(), os.path.join(directory,filename + "_critic2"))

    def load(self, filename):
        #directory = "RL_TD3_continuum_robot_control_5/LearnedModel"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "RL_TD3_continuum_robot_control"))
        directory = os.path.join(base_dir, "TD3LearnedModel")
        # Load the main Actor and Critic models and target Actor and Critic models
        self.actor.load_state_dict(torch.load(os.path.join(directory,filename + "_actor")))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1.load_state_dict(torch.load(os.path.join(directory,filename + "_critic1")))
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2.load_state_dict(torch.load(os.path.join(directory,filename + "_critic2")))
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    def load_agent(self, actor_path, critic1_path, critic2_path):
        # Load the Actor and Critic networks from specific paths
        self.actor.load_state_dict(torch.load(actor_path))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1.load_state_dict(torch.load(critic1_path))
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2.load_state_dict(torch.load(critic2_path))
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    def get_action(self, state):
        """
        Get the action from the trained agent given the current state.
        """
        if self.load_agent is None:
            raise ValueError("No trained agent loaded.")    # Raise error if agent isn't loaded
        action = self.select_action(state)
        return action