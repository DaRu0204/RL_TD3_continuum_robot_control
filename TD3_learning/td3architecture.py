import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import wandb
from wandb.integration.keras import WandbCallback
import os
import joblib
import pandas as pd
import sys
# Add the base directory to sys.path
c_dir = os.path.dirname(os.path.abspath(__file__))
b_dir = os.path.abspath(os.path.join(c_dir, ".."))
sys.path.append(b_dir)
from SL_learning.SupervisedLearningModel import NeuralNetwork


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_layers=2, hidden_units=256):
        super(Actor, self).__init__()
        layers = [nn.Linear(state_dim, hidden_units), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.ReLU()]
        layers.append(nn.Linear(hidden_units, action_dim))
        self.model = nn.Sequential(*layers)
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * torch.sigmoid(self.model(state))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=2, hidden_units=256):
        super(Critic, self).__init__()
        input_dim = state_dim + action_dim
        layers = [nn.Linear(input_dim, hidden_units), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_units, hidden_units), nn.ReLU()]
        layers.append(nn.Linear(hidden_units, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.model(x)
    
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
    
class TD3:

    lr_actor = 0.0001       # Learning rate for the Actor model (0.0001)
    lr_critic1 = 0.0003     # Learning rate for the first Critic model (0.0003)
    lr_critic2 = 0.0003     # Learning rate for the second Critic model (0.0003)
    gamma = 0.98            # Discount factor (gamma)
    tau = 0.005             # Soft update parameter (tau)

    def __init__(self, state_dim, action_dim, max_action, hidden_layers, hidden_units):
        # Initialize the Actor network and its target network
        self.actor = Actor(state_dim, action_dim, max_action, hidden_layers, hidden_units)                               # Main Actor network
        self.actor_target = Actor(state_dim, action_dim, max_action, hidden_layers, hidden_units)                        # Target Actor network
        self.actor_target.load_state_dict(self.actor.state_dict())                          # Hard copy initial weights to target
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=TD3.lr_actor)         # 0.001
        
        # Initialize the first Critic network and its target
        self.critic1 = Critic(state_dim, action_dim, hidden_layers, hidden_units)                                        # First Critic network
        self.critic1_target = Critic(state_dim, action_dim, hidden_layers, hidden_units)                                 # Target Critic 1 network
        self.critic1_target.load_state_dict(self.critic1.state_dict())                      # Hard copy initial weights to target
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=TD3.lr_critic1)   # 0.002

        # Initialize the second Critic network and its target
        self.critic2 = Critic(state_dim, action_dim, hidden_layers, hidden_units)                                        # Second Critic network
        self.critic2_target = Critic(state_dim, action_dim, hidden_layers, hidden_units)                                 # Target Critic 2 network
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
    
class ContinuumRobotEnv:

    # step = 10
    x_min, x_max = -70.63, -18.33       # -108.15, 19.19
    y_min, y_max = -144.35, -92.05      # -144.35, -92.05
    z_min, z_max = 247.45, 299.76       # 204.57, 342.64

    def __init__(self, segment_length=0.1, num_segments=1, num_tendons=3, max_steps=10, max_action=1, distance_treshold=0.0005):      # never change max_action = 1
        """
        Initialize the environment for a continuum robot with defined number of segments in 3D space.
        """
        # Define paths for the model, scalers and dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "RL_TD3_continuum_robot_control"))
        model_dir = os.path.join(base_dir, "SimulateRobotLearnedModel")
        model_path = os.path.join(model_dir, "trained_model_sl_1.pth")
        scaler_X_path = os.path.join(model_dir, "scaler_X_sl_1.pkl")
        scaler_y_path = os.path.join(model_dir, "scaler_y_sl_1.pkl")
        dataset_path = os.path.join(base_dir, "dataset", "workspace_point_dataset_2.txt")
        # dataset_path = os.path.join(base_dir, "dataset", "Dataset-Actions-Positions.txt")
        
        # Check if the model and scaler files exist
        if not os.path.exists(model_dir):
            print(f"Error: Directory '{model_dir}' does not exist.")
            exit(1)
        if not (os.path.exists(model_path) and os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path)):
            print(f"Error: Model or scaler files are missing in the '{model_dir}' directory.")
            exit(1)
        # Check if the dataset exists
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset file '{dataset_path}' does not exist in the current directory.")
            exit(1)
            
        # Load the trained model and scalers
        self.model = NeuralNetwork()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.scaler_X = joblib.load(scaler_X_path)
        self.scaler_y = joblib.load(scaler_y_path)
        # Load the dataset
        self.dataset = pd.read_csv(dataset_path, header=None, sep=",").values
        
        self.segment_length = segment_length
        self.num_segments = num_segments
        self.max_steps = max_steps
        self.max_action = max_action

        self.state_dim = 7  # Include current position, target position and distance between them
        self.action_dim = num_segments * num_tendons

        self.state = None
        self.target_position = None
        self.current_step = 0
        self.distance_treshold = distance_treshold

    def random_target(self):
        random_row = random.choice(self.dataset)  # Select a random row from the dataset
        x_target, y_target, z_target = random_row[-3:]  # Extract the target position (last three columns)
        target = np.array([x_target, y_target, z_target])
        return target

    def reset(self):
        """
        Reset the environment to its initial state at the beginning of an episode.
        This includes generating a random target position and resetting the robot's position.
        """
        self.current_step = 0   # Reset the number of steps taken
        # initial_state = np.zeros(self.state_dim)
        initial_state = np.zeros(3) # resets initial position
        self.target_position = np.array(self.random_target())/1000  # generates random target position
        # print("Target position for this episode:", self.target_position)
        distance_to_target = np.array(self.distance(initial_state))
        # self.state = initial_state  # Reset the enviroment's state
        self.state = np.concatenate([initial_state, self.target_position, [distance_to_target]])
        # print("State for this episode:", self.state)
        return self.state
        
    def step(self, actions):
        """
        Apply actions to the robot and compute the next state, reward, and check if the episode is done.
        """
        self.current_step += 1
        # actions = np.clip(actions, -self.max_action, self.max_action)
        actions = np.clip(actions, 0, self.max_action)  # Dataset doesn't allow negative actions
        # next_state = self._simulate_robot(actions)
        next_position = self._simulate_robot_model(actions)
        reward = self._compute_reward(next_position)
        distance_to_target = np.array(self.distance(next_position))
        # done = self.current_step >= self.max_steps
        done = self.current_step >= self.max_steps or self.distance(next_position) < self.distance_treshold
        # self.state = next_state  # Update the environment's state
        self.state = np.concatenate([next_position, self.target_position, [distance_to_target]])  # Update enviroment's state
        # print(f"Step {self.current_step}: Actions: {actions}, Next State: {next_position}, Reward: {reward}")
        return self.state, reward, done
    
    def _simulate_robot_model(self, actions):
        """
        Use the trained model to predict the robot's position (x, y, z) based on the actions.
        """
        # Normalize the input actions
        actions_scaled = np.array(actions)*100
        # print("Action: ", actions_scaled)
        input_data = self.scaler_X.transform([actions_scaled])
        # input_data = self.scaler_X.transform([actions])
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        
        # Predict the position using the trained model
        with torch.no_grad():
            prediction = self.model(input_tensor).numpy()
        
        # Denormalize the output to get the (x, y, z) position
        predicted_position = self.scaler_y.inverse_transform(prediction)
        return np.array(predicted_position[0])/1000  # Ensure the output is an np.array

    def _compute_reward(self, state):
        distance = np.linalg.norm(state - self.target_position)
        # distance = np.linalg.norm(np.array(state) - np.array(ContinuumRobotEnv.random_target()))
        reward = -distance
        return reward
    
    def distance(self, state):
        # distance_to_goal = np.linalg.norm(state - self.target_position)
        if len(state) == 3:
            distance_to_goal = np.linalg.norm(state - self.target_position)
        else:
            distance_to_goal = np.linalg.norm(state[:3] - self.target_position)
        return distance_to_goal
    
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
        noisy_action = np.clip(action + noise, 0, self.max_action)
        # noisy_action = np.clip(action + noise, 0, self.max_action)  # Add noise and clip the action
        return noisy_action

    def decay_noise(self):
        """
        Gradually reduces the noise level.
        """
        self.std = max(self.min_std, self.std * self.decay_rate)
        
def train_td3(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        # Instantiate environment and TD3 agent with hyperparameters from wandb
        env = ContinuumRobotEnv(max_steps=1000)
        td3_agent = TD3(env.state_dim, env.action_dim, env.max_action, config.hidden_layers, config.hidden_units)
        
        exploration_noise = ExplorationNoise(env.action_dim, env.max_action, initial_std=0.2, min_std=0.01, decay_rate=0.99)

        replay_buffer = ReplayBuffer(buffer_size=1000000)
        episode_rewards = deque(maxlen=100)
        total_episodes = 1500
        batch_size = 128

        log_file = 'td3_architecture_log.txt'
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                f.write(f"Run: layers={config.hidden_layers}, units={config.hidden_units}\n")
        else:
            with open(log_file, 'a') as f:
                f.write(f"Run: layers={config.hidden_layers}, units={config.hidden_units}\n")
                
        for episode in range(total_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = td3_agent.select_action(state)
                noisy_action = exploration_noise.add_noise(action)
                next_state, reward, done = env.step(noisy_action)
                replay_buffer.add(state, noisy_action, next_state, reward, done)
                td3_agent.train(replay_buffer, batch_size=batch_size)
                state = next_state
                episode_reward += reward
                
            exploration_noise.decay_noise()
            
            dis = env.distance(state)
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards)
            wandb.log({'avg_reward': avg_reward, 'episode': episode})
            wandb.log({'distance': dis, 'episode': episode})
            # log measured metrics to text file
            with open(log_file, 'a') as f:
                f.write(f"{episode + 1},{dis:.6f},{avg_reward:.6f}\n")

sweep_config = {
    'method': 'grid',
    'metric': {'name': 'avg_reward', 'goal': 'maximize'},
    'parameters': {
        'hidden_layers': {'values': [1, 2, 3, 4, 5]},
        'hidden_units': {'values': [64, 128, 256, 512]}
    }
}

sweep_id = wandb.sweep(sweep_config, project='TD3_architecture_optim')
wandb.agent(sweep_id, function=train_td3)