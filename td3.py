import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random
from collections import deque 

class Actor(nn.Module):
    def __init__(self, state_dimension, action_dimension, max_action):
        super(Actor, self).__init__()
        # Layers of neural network
        self.layer1 = nn.Linear(state_dimension, 400)
        self.layer2 = nn.Linear(400,300)
        self.Layer3 = nn.Linear(300, action_dimension)
        # Maximum action value (used for scaling the output)
        self.max_action = max_action
    
    def forward(self, state):
        # Forward pass through the network
        x = torch.relu(self.layer1(state))                  # Apply ReLU activation to the first layer
        x = torch.relu(self.layer2(x))                      # Apply ReLU activation to the second layer
        x = torch.tanh(self.layer3(x)) * self.max_action    # Apply tanh activation to the output layer and scale by max_action
        return x
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Layers of neural network
        self.layer1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
# Define the TD3 agent
class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        
        self.critic1 = Critic(state_dim, action_dim)
        self.critic_target1 = Critic(state_dim, action_dim)
        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic_optimizer1 = optim.Adam(self.critic1.parameters(), lr=0.001)
        
        self.critic2 = Critic(state_dim, action_dim)
        self.critic_target2 = Critic(state_dim, action_dim)
        self.critic_target2.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer2 = optim.Adam(self.critic2.parameters(), lr=0.001)
        
        self.max_action = max_action
        
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        for it in range(iterations):
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(batch_states)
            next_state = torch.FloatTensor(batch_next_states)
            action = torch.FloatTensor(batch_actions)
            reward = torch.FloatTensor(batch_rewards)
            done = torch.FloatTensor(batch_dones)
            
            # Select action according to policy and add clipped noise
            noise = torch.FloatTensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            
            # Compute target Q values
            target_Q1 = self.critic_target1(next_state, next_action)
            target_Q2 = self.critic_target2(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * discount * target_Q
            
            # Get current Q estimates
            current_Q1 = self.critic1(state, action)
            current_Q2 = self.critic2(state, action)
            
            # Compute critic loss
            critic_loss = nn.MSELoss()(current_Q1, target_Q.detach()) + nn.MSELoss()(current_Q2, target_Q.detach())
            
            # Optimize the critic
            self.critic_optimizer1.zero_grad()
            self.critic_optimizer2.zero_grad()
            critic_loss.backward()
            self.critic_optimizer1.step()
            self.critic_optimizer2.step()
            
            # Delayed policy updates
            if it % policy_freq == 0:
                # Compute actor loss
                actor_loss = -self.critic1(state, self.actor(state)).mean()
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Update the frozen target networks
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    
                for param, target_param in zip(self.critic1.parameters(), self.critic_target1.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    
                for param, target_param in zip(self.critic2.parameters(), self.critic_target2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity                # Capacity of the replay buffer, 10000
        self.buffer = deque(maxlen=capacity)    # Data structure to store experiences (deque with max length)
        self.batch_size = 64                    # Batch size for sampling experiences

    def push(self, state, action, reward, next_state, done):
        # Add a new experience to the replay buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=None):
        # Sample a batch of experiences from the replay buffer
        batch_size = batch_size or self.batch_size
        return random.sample(self.buffer, batch_size)

class ContinuumRobotEnv:
    def __init__(self):
        # Define state and action spaces
        self.observation_space = 10                                 # Placeholder for state dimension (e.g., 10 dimensions)
        self.action_space = 3                                       # Placeholder for action dimension (e.g., 3 dimensions)
        self.done = False                                           # Flag indicating if the episode is done

        # Initialize other environment parameters
        self.robot_position = np.zeros(3)                           # Initial position of the continuum robot (3D position)
        self.goal_position = np.array([5, 5, 5])                    # Position of the goal/target (3D position)
        self.max_steps = 100                                        # Maximum number of steps in an episode (100 steps)
        self.current_step = 0                                       # Current step counter (initialized to 0)
        self.done = False                                           # Flag indicating if the episode is done (initialized to False)

        # Define other environment parameters
        self.joint_limits = [(-180, 180), (-90, 90), (-180, 180)]   # Joint limits of the continuum robot (e.g., in degrees)
        self.sensor_noise = 0.1                                     # Level of sensor noise (e.g., 0.1 units)


    def reset(self):
        # Reset the environment to its initial state
        self.robot_position = np.zeros(3)                           # Reset robot's position to the origin
        self.current_step = 0                                       # Reset current step counter
        self.done = False                                           # Reset done flag
        return self.get_observation()                               # Return the initial observation/state
    


    def step(self, action):
        # Take a step in the environment based on the given action
        # Update the robot's position, compute reward, and check if episode is done
        self.robot_position += action                               # Update robot's position based on the action
        self.current_step += 1                                      # Increment the step counter
        self.done = self.current_step >= self.max_steps             # Check if episode is done based on maximum steps
        reward = self.compute_reward()                              # Compute reward based on the new state and action
        return self.get_observation(), reward, self.done, {}        # Return next observation, reward, done flag, and additional info


    def get_observation(self):
        # Get the current observation/state of the environment
        observation = self.robot_position                           # Return robot's current position as the observation
        return observation
    
    def compute_reward(self):
        # Placeholder for reward computation logic
        # You can define your own reward function based on the current state, action, etc.
        # For example, you might compute the Euclidean distance between the robot's position and the goal position
        reward = -np.linalg.norm(self.robot_position - self.goal_position)  # Negative distance to encourage reaching the goal
        return reward


state_dim = 3  # Define the state dimension of the environment
action_dim = 3  # Define the action dimension of the environment
max_action = 10  # Define the maximum action value
agent = TD3(state_dim, action_dim, max_action)

# Step 4: Train the TD3 agent
replay_buffer = ReplayBuffer(10000)  # Initialize a replay buffer with a certain capacity
iterations = 1000  # Number of training iterations
batch_size = 64  # Batch size for training
discount = 0.99  # Discount factor
tau = 0.005  # Soft update parameter
policy_noise = 0.2  # Policy noise parameter
noise_clip = 0.5  # Noise clipping parameter
policy_freq = 2  # Policy update frequency
agent.train(replay_buffer, iterations, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

# Step 5: Run the trained agent
state = ContinuumRobotEnv.reset()  # Reset the environment to its initial state
while not done:
    action = agent.select_action(state)  # Select action using the trained agent
    next_state, reward, done, _ = ContinuumRobotEnv.step(action)  # Take a step in the environment
    state = next_state  # Update the current state