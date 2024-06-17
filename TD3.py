import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import wandb
from wandb.keras import WandbCallback
import os
from ActorCritic import Actor
from ActorCritic import Critic

# TD3 algorithm
class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.000001)  # 0.001

        self.critic1 = Critic(state_dim, action_dim)
        self.critic1_target = Critic(state_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=0.000003)  # 0.002

        self.critic2 = Critic(state_dim, action_dim)
        self.critic2_target = Critic(state_dim, action_dim)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=0.000003)  # 0.002

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=64, gamma=0.95, noise=0.2, policy_noise=0.2, noise_clip=0.2, policy_freq=2):
        if len(replay_buffer) < batch_size:
            return

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * noise).clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_Q1 = self.critic1_target(next_state, next_action)
            target_Q2 = self.critic2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * gamma * target_Q

        current_Q1 = self.critic1(state, action)
        current_Q2 = self.critic2(state, action)

        critic1_loss = nn.MSELoss()(current_Q1, target_Q.detach())
        critic2_loss = nn.MSELoss()(current_Q2, target_Q.detach())

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        if replay_buffer.iterations % policy_freq == 0:
            actor_loss = -self.critic1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

        return critic1_loss, critic2_loss

    def save(self, filename):
        directory = "RL_TD3_continuum_robot_control/LearnedModel"
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.actor.state_dict(), os.path.join(directory, filename + "_actor"))
        torch.save(self.critic1.state_dict(), os.path.join(directory,filename + "_critic1"))
        torch.save(self.critic2.state_dict(), os.path.join(directory,filename + "_critic2"))

    def load(self, filename):
        directory = "RL_TD3_continuum_robot_control/LearnedModel"
        self.actor.load_state_dict(torch.load(os.path.join(directory,filename + "_actor")))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1.load_state_dict(torch.load(os.path.join(directory,filename + "_critic1")))
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2.load_state_dict(torch.load(os.path.join(directory,filename + "_critic2")))
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    def load_agent(self, actor_path, critic1_path, critic2_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1.load_state_dict(torch.load(critic1_path))
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2.load_state_dict(torch.load(critic2_path))
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    def get_action(self, state):
        if self.load_agent is None:
            raise ValueError("No trained agent loaded.")
        action = self.select_action(state)
        return action