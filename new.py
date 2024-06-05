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
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)
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
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

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
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)

        self.critic1 = Critic(state_dim, action_dim)
        self.critic1_target = Critic(state_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=0.002)

        self.critic2 = Critic(state_dim, action_dim)
        self.critic2_target = Critic(state_dim, action_dim)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=0.002)

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=64, gamma=0.99, noise=0.2, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
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
    def __init__(self, num_segments=5, segment_length=0.1, action_range=(-1, 1)):
        self.num_segments = num_segments  # Number of segments in the robot
        self.segment_length = segment_length  # Length of each segment
        self.action_range = action_range  # Range of actions (curvature commands)
        self.max_action = action_range[1]  # Maximum value of actions

        # Define any other parameters of the environment, such as the workspace limits, obstacles, etc.
        self.max_steps = 100  # Maximum number of steps per episode
        self.current_step = 0  # Current step in the episode
        self.goal_position = np.array([0.5, 0.5])  # Example goal position

        # Define the state space and action space dimensions
        self.state_dim = num_segments * 2  # Each segment has 2 dimensions (x, y)
        self.action_dim = num_segments  # Each segment has an associated curvature command

    def reset(self):
        # Reset the environment to the initial state
        self.current_step = 0
        # Initial state: all segments at the origin
        initial_state = np.zeros(self.state_dim)
        return initial_state

    def step(self, actions):
        # Perform one step in the environment based on the actions provided
        self.current_step += 1
        
        # Simulate the effect of actions on the robot's state (e.g., update segment positions)
        next_state = self._simulate_robot(actions)
        
        # Compute the reward based on the achieved task (e.g., distance to the goal position)
        reward = self._compute_reward(next_state)
        
        # Check if the episode is done (e.g., reaching the goal or reaching the maximum steps)
        done = self.current_step >= self.max_steps
        
        return next_state, reward, done

    def _simulate_robot(self, actions):
        # Simulate the robot's state given the actions (curvature commands)
        state = np.zeros(self.state_dim)
        current_position = np.array([0.0, 0.0])
        current_angle = 0.0

        for i in range(self.num_segments):
            curvature = actions[i]
            segment_angle = curvature * self.segment_length
            next_position = current_position + np.array([
                self.segment_length * np.cos(current_angle + segment_angle / 2),
                self.segment_length * np.sin(current_angle + segment_angle / 2)
            ])
            state[2 * i: 2 * i + 2] = next_position
            current_position = next_position
            current_angle += segment_angle

        return state

    def _compute_reward(self, state):
        # Compute the reward based on the distance to the goal position
        end_effector_position = state[-2:]  # The position of the last segment
        distance_to_goal = np.linalg.norm(end_effector_position - self.goal_position)
        reward = -distance_to_goal  # Negative distance as reward
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
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

# Training loop
total_episodes = 1000
batch_size = 64
replay_buffer = ReplayBuffer(buffer_size=1000000)
episode_rewards = deque(maxlen=100)
for episode in range(total_episodes):
    state = env.reset()
    episode_reward = 0
    done = False
    print("episode:",episode)
    while not done:
        
        action = td3_agent.select_action(state)
        next_state, reward, done = env.step(action)
        replay_buffer.add(state, action, next_state, reward, done)
        state = next_state
        episode_reward += reward
        td3_agent.train(replay_buffer, batch_size=batch_size)

    
    episode_rewards.append(episode_reward)
    avg_reward = np.mean(episode_rewards)
    #wandb.log({'episode_rewards': episode_rewards, 'episode': episode})
    wandb.log({'avg_reward': avg_reward, 'episode': episode})

    print(f"Episode: {episode + 1}, Average Reward: {avg_reward}")