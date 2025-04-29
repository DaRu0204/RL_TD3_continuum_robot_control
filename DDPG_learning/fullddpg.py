import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import os
import joblib
import pandas as pd
import wandb
from wandb.integration.keras import WandbCallback
import sys
# Add the base directory to sys.path
c_dir = os.path.dirname(os.path.abspath(__file__))
b_dir = os.path.abspath(os.path.join(c_dir, ".."))
sys.path.append(b_dir)
from SL_learning.SupervisedLearningModel import NeuralNetwork


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        # Define four fully connected layers (neurons in each layer = 256)
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, action_dim)    # Output layer
        
        # The maximum possible action value to scale outputs
        self.max_action = max_action

    def forward(self, state):
        # Apply ReLU activation to each hidden layer
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        
        # Use tanh activation in the last layer to ensure the action values are in a reasonable range
        # Then scale them using max_action
        # x = self.max_action * torch.tanh(self.layer6(x))
        x = self.max_action * torch.sigmoid(self.layer5(x))
        return x

# Single Q-network (Critic)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # The first layer takes both state and action as input
        self.layer1 = nn.Linear(state_dim + action_dim, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, 1) # Output layer (single Q-value)

    def forward(self, state, action):
        # Combine the state and action into one input tensor
        x = torch.cat([state, action], 1)
        
        # Pass through hidden layers with ReLU activation
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        
        # Output the Q-value without an activation function (linear output)
        x = self.layer5(x)
        return x

# Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        # Use a deque (double-ended queue) to store a fixed number of experiences
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, next_state, reward, done):
        # Store a new experience in the buffer
        self.buffer.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size):
        # Randomly select a batch of experiences from the buffer
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack the batch into separate components
        states, actions, next_states, rewards, dones = zip(*batch)
        
        # Convert them to PyTorch tensors for neural network training
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),    # Add extra dimension
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)       # Add extra dimension
        )
        
    def __len__(self):
        # Return the number of stored experiences
        return len(self.buffer)

# DDPG Algorithm
class DDPG:
    # Hyperparameters for training the DDPG agent
    lr_actor = 0.0001   # Learning rate for the actor network
    lr_critic = 0.0003  # Learning rate for the critic network
    gamma = 0.99    # Discount factor for future rewards (how much future rewards matter)
    tau = 0.001    # Soft update factor for the target networks

    def __init__(self, state_dim, action_dim, max_action):
        # Create the actor (policy) network and its target copy
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        
        # Initialize the target actor with the same weights as the main actor
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Optimizer for the actor network
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        
        # Create the critic (value) network and its target copy
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        
        # Initialize the target critic with the same weights as the main critic
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizer for the critic network
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        # Store the maximum action value for scaling outputs
        self.max_action = max_action

    def select_action(self, state):
        # Convert state to a PyTorch tensor and add batch dimension (1, state_dim)
        state = torch.FloatTensor(state.reshape(1, -1))
        
        # Get the action from the actor network and convert it to a NumPy array
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=128):
        # Ensure the replay buffer has enough samples to form a batch
        if len(replay_buffer) < batch_size:
            return
        
        # Sample a batch of experiences (random past transitions)
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        
        with torch.no_grad():   # Disable gradient calculation for target network
            # Compute target Q-value using the target networks
            next_action = self.actor_target(next_state)
            target_Q = reward + not_done * self.gamma * self.critic_target(next_state, next_action)
        
        # Compute the current Q-value using the main critic network
        current_Q = self.critic(state, action)
        
        # Compute the loss for the critic (how far the estimated Q-value is from the target)
        critic_loss = nn.MSELoss()(current_Q, target_Q.detach())
        
        # Optimize the critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Compute the loss for the actor (maximize the expected Q-value)
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        # Optimize the actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update the target networks (moving weights slowly toward the main network)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def save(self, filename):
        # Get the script directory (where the file should be saved)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "RL_TD3_continuum_robot_control"))
        directory = os.path.join(base_dir, "DDPGLearnedModel")
        if not os.path.exists(directory):
            os.makedirs(directory)
        actor_path = os.path.join(directory, filename + "_actor.pth")
        critic_path = os.path.join(directory, filename + "_critic.pth")

        # Save the weights of the actor and critic networks
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load(self, filename):
        # Get the script directory (where the file should be loaded from)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "RL_TD3_continuum_robot_control"))
        # directory = os.path.join(base_dir, "DDPGLearnedModel")
        directory = os.path.join(base_dir, "DDPGLearnedModel")
        actor_path = os.path.join(directory, filename + "_actor.pth")
        critic_path = os.path.join(directory, filename + "_critic.pth")

        # Load the saved model weights for actor and critic networks
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        
        # Ensure the target networks are also updated
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def get_action(self, state):
        return self.select_action(state)

class ContinuumRobotEnv:
    def __init__(self, dataset_file, model_file, scaler_X_file, scaler_y_file, state_dim=7, action_dim=3, max_action=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # Get the absolute path of the script directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "RL_TD3_continuum_robot_control"))
        model_dir = os.path.join(base_dir, "SimulateRobotLearnedModel")
        model_path = os.path.join(model_dir, model_file)
        scaler_X_path = os.path.join(model_dir, scaler_X_file)
        scaler_y_path = os.path.join(model_dir, scaler_y_file)
        dataset_path = os.path.join(base_dir, "dataset", dataset_file)
        
        # Check if all required files exist before loading
        for path, name in zip([model_path, scaler_X_path, scaler_y_path, dataset_path],
                              ["Model", "Scaler X", "Scaler Y", "Dataset"]):
            if not os.path.exists(path):
                print(f"[ERROR] {name} file not found: {path}")
                exit(1)
        
        # Load the trained neural network model
        self.model = NeuralNetwork()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Load scalers for normalizing input and output data
        self.scaler_X = joblib.load(scaler_X_path)
        self.scaler_y = joblib.load(scaler_y_path)
        
        # Load dataset containing valid target positions
        self.dataset = pd.read_csv(dataset_path, header=None, sep=",").values
        
        # Initialize environment variables
        self.state = None
        self.target_position = None
        self.next_position = None
        self.max_steps = 1000        # desired value 1000
        self.current_step = 0

    def reset(self):
        """ Resets the environment at the start of a new episode. """
        self.current_step = 0
        # Select a random target position from the dataset
        self.target_position = np.array(random.choice(self.dataset)[-3:])/1000
        initial_state = np.zeros(3) # Assume the robot starts at (0,0,0)
        
        # Compute initial distance to the target
        distance = np.array(np.linalg.norm(initial_state - self.target_position))
        
        # Update the state with initial position, target position, and distance
        self.state = np.concatenate((initial_state, self.target_position, [distance]))
        return self.state

    def _simulate_robot_model(self, action):
        """ Uses the trained model to predict the next position based on the given action. """
        action_scaled = np.array(action)*100    # Scale action values
        # print(f"Action: {action_scaled}")
        input_data = self.scaler_X.transform([action_scaled])
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        
        # Make a prediction without tracking gradients (faster inference)
        with torch.no_grad():
            prediction = self.model(input_tensor).numpy()
        
        # Transform the predicted position back to the original scale
        predicted_position = self.scaler_y.inverse_transform(prediction)
        return np.array(predicted_position[0])/1000

    def step(self, action):
        """ Executes one step in the environment based on the given action. """
        self.current_step += 1
        
        # Predict the next position based on the action
        action = np.clip(action, 0, self.max_action)
        self.next_position = self._simulate_robot_model(action)
        
        # Compute the distance between the robot and the target position
        distance_to_target = np.linalg.norm(self.next_position - self.target_position)
        distance = np.array(distance_to_target)
        
        # Reward is negative distance (closer to target = higher reward)
        reward = -distance_to_target
        
        # Check if the episode is over
        done = distance_to_target < 0.0005 or self.current_step >= self.max_steps
        next_state = self.next_position
        
        # Update the state with the new position, target position, and distance
        self.state = np.concatenate((next_state, self.target_position, [distance]))
        return self.state, reward, done
    
class ExplorationNoise:
    def __init__(self, action_dim, max_action, initial_std=0.2, min_std=0.05, decay_rate=0.99):
        """
        Initializes dynamic noise for exploration.
        - action_dim: Dimension of the action space.
        - max_action: Maximum value for actions.
        - initial_std: Initial standard deviation of the noise.
        - min_std: Minimum standard deviation of the noise (target after decay).
        - decay_rate: Rate of noise reduction (a value close to 1 means slower decay).
        """
        self.action_dim = action_dim
        self.max_action = max_action
        self.std = initial_std
        self.min_std = min_std
        self.decay_rate = decay_rate

    def add_noise(self, action):
        """
        Adds noise during exploration
        """
        noise = np.random.normal(0, self.std, size=self.action_dim)  # Generate noise
        noisy_action = np.clip(action + noise, 0, self.max_action)  # Add noise and clip the action
        return noisy_action

    def decay_noise(self):
        """
        Gradually reduces the noise level.
        """
        self.std = max(self.min_std, self.std * self.decay_rate)

# Main function to train the agent
def main():
    # Initialize the robot environment and the DDPG agent
    env = ContinuumRobotEnv('workspace_point_dataset_2.txt', 'trained_model_sl_1.pth', 'scaler_X_sl_1.pkl', 'scaler_y_sl_1.pkl')
    agent = DDPG(env.state_dim, env.action_dim, env.max_action)
    replay_buffer = ReplayBuffer(1000000)   # Experience replay buffer
    rewards = []    # Store rewards for monitoring performance
    total_rewards = deque(maxlen=100)   # Moving average of rewards
    exploration_noise = ExplorationNoise(env.action_dim, env.max_action, initial_std=0.2, min_std=0.05, decay_rate=0.99)
    
    # Initialize Weights & Biases (wandb) for logging
    wandb.init(project="DDPG", config={"episodes": 5000, "max_steps": env.max_steps})
    
    # check for log file
    log_file = 'ddpg_log.txt'
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            pass    # Create log file without header

    # Training loop for a number of episodes
    for episode in range(5000): # desired value 5000
        state = env.reset() # Reset environment at the start of each episode
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state) # Select action from policy
            noisy_action = exploration_noise.add_noise(action) # Add noise to the action generated by the policy
            next_state, reward, done = env.step(noisy_action) # Execute action with added noise in the environment
            replay_buffer.add(state, action, next_state, reward, done)  # Store experience
            agent.train(replay_buffer)  # Train the agent using replay buffer
            state = next_state  # Update current state
            total_reward += reward  # Accumulate episode reward
        
        exploration_noise.decay_noise()
        # Store total reward for monitoring    
        rewards.append(total_reward)
        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards) # Compute moving average reward
        dis = np.linalg.norm(env.next_position - env.target_position)   # Compute final distance to target
        
        # Log metrics using wandb
        wandb.log({'avg_reward': avg_reward, 'episode': episode})
        wandb.log({'distance': dis, 'episode': episode})
        
        # log measured metrics to text file
        with open(log_file, 'a') as f:
            f.write(f"{episode + 1},{dis:.6f},{avg_reward:.6f}\n")
        
        print(f"Episode: {episode + 1}, Average Reward: {avg_reward}")

    # Save the trained agent model
    agent.save("ddpg")
    """
    # Load the trained model and make a prediction
    agent.load("ddpg")
    distacne = np.linalg.norm(np.array([0,0,0]) - (np.array([-26.33573,-140.7042,264.039]))/1000)
    state_pred = np.array([0,0,0,-0.02633573,-0.1407042,0.264039,distacne])
    predicted_action = agent.get_action(state_pred)
    actual_position = env._simulate_robot_model(predicted_action)
    print(f"Predicted position: {actual_position*1000}")
    """
# Run the main function when the script is executed
if __name__ == "__main__":
    main()