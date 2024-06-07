import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import wandb
from wandb.keras import WandbCallback

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        x = self.max_action * torch.tanh(self.layer3(x))
        return x

# Twin Q-networks (Critic)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.iterations = 0  # For tracking the number of iterations

    def add(self, state, action, next_state, reward, done):
        transition = Transition(state, action, next_state, reward, done)
        self.buffer.append(transition)
        self.iterations += 1

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

# TD3 algorithm
class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)#0.001

        self.critic1 = Critic(state_dim, action_dim)
        self.critic1_target = Critic(state_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=0.0003)#0.002

        self.critic2 = Critic(state_dim, action_dim)
        self.critic2_target = Critic(state_dim, action_dim)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=0.0003)#0.002

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    #def train(self, replay_buffer, batch_size=64, gamma=0.99, noise=0.2, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    def train(self, replay_buffer, batch_size=128, gamma=0.99, noise=0.2, policy_noise=0.2, noise_clip=0.2, policy_freq=2):
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

# Define your environment or import from Gym
class ContinuumRobotEnv:
    def __init__(self, num_segments=3, segment_length=0.5, action_range=(-1.5, 1.5), max_steps=50, goal_position=(0.8, 0.8)):
        self.num_segments = num_segments
        self.segment_length = segment_length
        self.action_range = action_range
        self.max_action = action_range[1]
        self.max_steps = max_steps
        self.current_step = 0
        self.goal_position = np.array(goal_position)
        self.state_dim = num_segments * 2
        self.action_dim = num_segments

    def reset(self):
        self.current_step = 0
        initial_state = np.zeros(self.state_dim)
        return initial_state

    def step(self, actions):
        self.current_step += 1
        next_state = self._simulate_robot(actions)
        reward = self._compute_reward(next_state)
        done = self.current_step >= self.max_steps
        return next_state, reward, done

    def _simulate_robot(self, actions):
        state = np.zeros(self.state_dim)
        current_position = np.array([0.0, 0.0])
        current_angle = 0.0

        for i in range(self.num_segments):
            curvature = actions[i]
            if curvature != 0:
                radius = 1.0 / curvature
                angle = curvature * self.segment_length
                cx = current_position[0] - radius * np.sin(current_angle)
                cy = current_position[1] + radius * np.cos(current_angle)
                next_position = np.array([
                    cx + radius * np.sin(current_angle + angle),
                    cy - radius * np.cos(current_angle + angle)
                ])
                current_angle += angle
            else:
                next_position = current_position + self.segment_length * np.array([
                    np.cos(current_angle),
                    np.sin(current_angle)
                ])
            state[2 * i: 2 * i + 2] = next_position
            current_position = next_position

        return state

    def _compute_reward(self, state):
        end_effector_position = state[-2:]
        distance_to_goal = np.linalg.norm(end_effector_position - self.goal_position)
        reward = -distance_to_goal
        if distance_to_goal < 0.3:
            reward += 100
        reward -= self.current_step * 0.01  # Small penalty per step, adjust the factor as needed
        return reward

# Instantiate environment and TD3 agent
env = ContinuumRobotEnv()
td3_agent = TD3(env.state_dim, env.action_dim, env.max_action)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="TD3",

    # track hyperparameters and run metadata
    config={
    #"learning_rate": 0.02,
    #"architecture": "CNN",
    #"dataset": "CIFAR-100",
    #"epochs": 10,
    }
)

# Training loop
total_episodes = 1000
rewards = []
critic_losses1 = []
critic_losses2 = []
batch_size = 16
replay_buffer = ReplayBuffer(buffer_size=1000000)
episode_rewards = deque(maxlen=100)
for episode in range(total_episodes):
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        
        action = td3_agent.select_action(state)
        next_state, reward, done = env.step(action)
        replay_buffer.add(state, action, next_state, reward, done)
        td3_agent.train(replay_buffer, batch_size=batch_size)
        state = next_state
        episode_reward += reward
    
        # Train the agent
        #if len(replay_buffer) > batch_size:
        #    critic1_loss, critic2_loss = td3_agent.train(replay_buffer, batch_size)
        #    episode_critic_loss1 += critic1_loss
        #    episode_critic_loss2 += critic2_loss

    # Store episode metrics
    rewards.append(episode_reward)
    #critic_losses1.append(episode_critic_loss1)
    #critic_losses2.append(episode_critic_loss2)
    #rewards.append(episode_reward)

    episode_rewards.append(episode_reward)
    avg_reward = np.mean(episode_rewards)
    wandb.log({'avg_reward': avg_reward, 'episode': episode})

    print(f"Episode: {episode + 1}, Average Reward: {avg_reward}")