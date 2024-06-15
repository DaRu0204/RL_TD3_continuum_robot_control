import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import os
import matplotlib.pyplot as plt
import wandb

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
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.00001)  # 0.001

        self.critic1 = Critic(state_dim, action_dim)
        self.critic1_target = Critic(state_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=0.00003)  # 0.002

        self.critic2 = Critic(state_dim, action_dim)
        self.critic2_target = Critic(state_dim, action_dim)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=0.00003)  # 0.002

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=64, gamma=0.99, noise=0.2, policy_noise=0.2, noise_clip=0.2, policy_freq=2):
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
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic1.state_dict(), filename + "_critic1")
        torch.save(self.critic2.state_dict(), filename + "_critic2")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1.load_state_dict(torch.load(filename + "_critic1"))
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2.load_state_dict(torch.load(filename + "_critic2"))
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

class ContinuumRobotEnv:
    def __init__(self, segment_length=0.1, max_tendon_tension=1, num_segments=1, num_tendons=3, max_steps=25, max_action=2):
        self.segment_length = segment_length
        self.max_tendon_tension = max_tendon_tension
        self.num_segments = num_segments
        self.max_steps = max_steps
        self.max_action = max_action

        self.state_dim = 2
        self.action_dim = num_segments * num_tendons

        self.state = np.zeros(self.state_dim)
        self.target_position = np.array([0.075, 0.04])
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        initial_state = np.zeros(self.state_dim)
        self.state = initial_state
        return initial_state
    
    def step(self, actions):
        self.current_step += 1
        actions = np.clip(actions, -self.max_tendon_tension, self.max_tendon_tension)
        next_state = self._simulate_robot(actions)
        reward = self._compute_reward(next_state)
        done = self.current_step >= self.max_steps
        self.state = next_state  # Update the environment's state
        return next_state, reward, done

    def _simulate_robot(self, actions):
        start_pos = np.array([0.0, 0.0])
        state = np.zeros(self.state_dim)
        orientation = 0.0

        for i in range(self.num_segments):
            kappa = actions[i] / self.segment_length
            
            if kappa != 0:
                delta_theta = kappa * self.segment_length
                r = 1 / kappa
                cx = start_pos[0] - r * np.sin(orientation)
                cy = start_pos[1] + r * np.cos(orientation)
                start_angle = orientation
                end_angle = orientation + delta_theta
                orientation = end_angle

                start_pos = np.array([
                    cx + r * np.sin(end_angle),
                    cy - r * np.cos(end_angle)
                ])
            else:
                delta_x = self.segment_length * np.cos(orientation)
                delta_y = self.segment_length * np.sin(orientation)
                start_pos += np.array([delta_x, delta_y])

        state = start_pos
        return state

    def _compute_reward(self, state):
        distance = np.linalg.norm(state - self.target_position)
        reward = -distance
        return reward

    def render(self, actions=None):
        segment_positions = [np.zeros(2)]
        orientation = 0.0

        if actions is not None:
            actions = np.clip(actions, -self.max_tendon_tension, self.max_tendon_tension)

        for i in range(self.num_segments):
            kappa = actions[i] / self.segment_length if actions is not None else 0
            start_pos = segment_positions[-1]

            if kappa != 0:
                delta_theta = kappa * self.segment_length
                r = 1 / kappa
                cx = start_pos[0] - r * np.sin(orientation)
                cy = start_pos[1] + r * np.cos(orientation)
                start_angle = orientation
                end_angle = orientation + delta_theta
                angles = np.linspace(start_angle, end_angle, 100)
                x_arc = cx + r * np.sin(angles)
                y_arc = cy - r * np.cos(angles)
                segment_positions.extend(np.column_stack((x_arc, y_arc)))
                orientation = end_angle
            else:
                delta_x = self.segment_length * np.cos(orientation)
                delta_y = self.segment_length * np.sin(orientation)
                segment_positions.append(start_pos + np.array([delta_x, delta_y]))

        segment_positions = np.array(segment_positions)
        print("segment_positions:", segment_positions)
        plt.figure(figsize=(8, 6))
        plt.plot(segment_positions[:, 0], segment_positions[:, 1], color='blue', label='Robot Curve')
        plt.scatter(0, 0, color='black', label='Base Position')
        plt.scatter(segment_positions[-1, 0], segment_positions[-1, 1], color='red', label='Tip Position')
        plt.scatter(self.target_position[0], self.target_position[1], color='green', label='Target Position')
        plt.title('Continuum Robot Visualization')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.show()
    
    def distance(self, state):
        distance_to_goal = np.linalg.norm(state - self.target_position)
        return distance_to_goal
    
env = ContinuumRobotEnv()
td3_agent = TD3(env.state_dim, env.action_dim, env.max_action)

# start a new wandb run to track this script
wandb.init(
    project="TD3",
    # track hyperparameters and run metadata
    config={
    #"learning_rate": 0.02,
    #"architecture": "CNN",
    #"dataset": "CIFAR-100",
    #"epochs": 10,
    }
)

def main():
    # Training loop
    total_episodes = 1000
    rewards = []
    batch_size = 64
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

        # Store episode metrics
        dis = env.distance(state)
        rewards.append(episode_reward)
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards)
        wandb.log({'avg_reward': avg_reward, 'episode': episode})
        wandb.log({'distance': dis, 'episode': episode})
        #print(f"Episode: {episode + 1}, Average Reward: {avg_reward}")
        
    td3_agent.save("td3_continuum_robot")

    loaded_agent = TD3(state_dim=env.state_dim, action_dim=env.action_dim, max_action=env.max_action)
    loaded_agent.load("td3_continuum_robot")
    
    #desired_positon = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.7])
    desired_positon = np.array([0.075, 0.04])
    actionn = loaded_agent.select_action(desired_positon)
    
    # Ensure state is updated before rendering
    statee, reward, done = env.step(actionn)
    print("State after step:", statee)
    error = np.linalg.norm(desired_positon - statee)
    print("Action:", actionn)
    print("State:", statee)
    print("Error:", error)
    
    # Render the environment based on the updated state
    env.render(actions=actionn)

if __name__ == "__main__":
    main()