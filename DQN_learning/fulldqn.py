import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import os
import joblib
import pandas as pd
import sys
import wandb
from wandb.integration.keras import WandbCallback
# Add the base directory to sys.path
c_dir = os.path.dirname(os.path.abspath(__file__))
b_dir = os.path.abspath(os.path.join(c_dir, ".."))
sys.path.append(b_dir)
from SL_learning.SupervisedLearningModel import NeuralNetwork

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # Define four fully connected layers (neurons in each layer = 256)
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, action_dim)  # Output Q-values for each action

    def forward(self, state):
        # Pass through hidden layers with ReLU activation
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        return self.fc6(x)  # Get the Q-values for each action

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
            torch.tensor(actions, dtype=torch.long).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),    # Add extra dimension
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)       # Add extra dimension
        )

    def __len__(self):
        # Return the current size of the buffer
        return len(self.buffer)

# DQN Agent
class DQN:
    def __init__(self, state_dim, action_dim, lr=0.0001, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        # Initialize the dimensions of the state and action spaces
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Discount factor for future rewards
        self.gamma = gamma
        # Initial exploration rate
        self.epsilon = epsilon
        # Minimum exploration rate
        self.epsilon_min = epsilon_min
        # Decay rate for exploration probability
        self.epsilon_decay = epsilon_decay

        # Initialize the Q-network and the target network
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        # Copy the weights from the Q-network to the target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        # Set the target network to evaluation mode
        self.target_network.eval()

        # Define the optimizer for the Q-network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        # Define the loss function (Mean Squared Error)
        self.loss_fn = nn.MSELoss()
        # Variable to store the last computed loss
        self.last_loss = None
        
    def select_action(self, state):
        # With probability epsilon, select a random action (exploration)
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
            # print(f"Random action index: {action_idx}")  # Debug
            return action_idx  # Return the random action index

        # Convert the state to a tensor and add a batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # Disable gradient calculation for efficiency
        with torch.no_grad():
            # Get the Q-values for the given state from the Q-network
            q_values = self.q_network(state_tensor)
        # Select the action with the highest Q-value
        action_idx = torch.argmax(q_values).item()
        # print(f"Selected action index: {action_idx}")  # Debug
        return int(action_idx)  # Ensure the returned value is an integer
        
    def train(self, replay_buffer, batch_size=128):
        # Ensure the replay buffer has enough samples to form a batch
        if len(replay_buffer) < batch_size:
            return
        # Sample a batch of experiences (random past transitions)
        states, actions, next_states, rewards, dones = replay_buffer.sample(batch_size)
        # Get the Q-values for the current states and selected actions
        q_values = self.q_network(states).gather(1, actions)

        # Disable gradient calculation for the target network
        with torch.no_grad():
            # Get the maximum Q-values for the next states from the target network
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            # Compute the target Q-values
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute the loss between the Q-values and the target Q-values
        loss = self.loss_fn(q_values, target_q_values)
        # Store the loss value
        self.last_loss = loss.item()
        
        self.optimizer.zero_grad()  # Zero the gradients of the Q-network
        loss.backward() # Backpropagate the loss
        self.optimizer.step()   # Update the Q-network parameters

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay) # Decay the exploration rate

    def update_target_network(self):
        # Copy the weights from the Q-network to the target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
class ContinuumRobotEnv:
    def __init__(self, dataset_file, model_file, scaler_X_file, scaler_y_file, state_dim=7, max_steps=100):
        self.state_dim = state_dim
        self.action_dim = len(ACTIONS)  # Discrete number of actions
        self.max_steps = max_steps
        
        # Get the absolute path of the script directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "RL_TD3_continuum_robot_control_5"))
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
        self.current_step = 0

    def reset(self):
        """ Resets the environment at the start of a new episode. """
        self.current_step = 0

        # Select a random target position from the dataset
        self.target_position = np.array(random.choice(self.dataset)[-3:])/1000
        initial_state = np.zeros(3)  # Assume the robot starts at (0,0,0)

        # Compute initial distance to the target
        distance = np.linalg.norm(initial_state - self.target_position)

        # Update the state with initial position, target position, and distance
        self.state = np.concatenate((initial_state, self.target_position, [distance]))
        return self.state

    def _simulate_robot_model(self, action):
        """ Uses the trained model to predict the next position based on the given action. """
        action_scaled = np.array(action)  # Scale action
        print(f"Action: {action_scaled}")
        input_data = self.scaler_X.transform([action_scaled])
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        # Make a prediction without tracking gradients (faster inference)
        with torch.no_grad():
            prediction = self.model(input_tensor).numpy()

        # Transform the predicted position back to the original scale
        predicted_position = self.scaler_y.inverse_transform(prediction)
        return np.array(predicted_position[0])/1000

    def step(self, action_index):
        """ Executes one step in the environment based on the given action. """
        self.current_step += 1

        # Convert action index to action vector
        action = ACTIONS[action_index]

        # Predict new position
        self.next_position = self._simulate_robot_model(action)

        # Calculate distance from target
        distance_to_target = np.linalg.norm(self.next_position - self.target_position)

        # Calculate reward
        reward = -distance_to_target

        # Check if the episode is over
        done = distance_to_target < 0.001 or self.current_step >= self.max_steps

        # Update the state with the new position, target position, and distance
        self.state = np.concatenate((self.next_position, self.target_position, [distance_to_target]))
        return self.state, reward, done

# Discrete action space
ACTIONS = [
    (a1, a2, a3)
    for a1 in range(0, 101, 5)
    for a2 in range(0, 101, 5)
    for a3 in range(0, 101, 5)
]

# Environment initialization and training loop
def main():
    # Initialize the robot environment and the DQN agent
    env = ContinuumRobotEnv('workspace_point_dataset.txt', 'trained_model_lr3.pth', 'scaler_X_lr3.pkl', 'scaler_y_lr3.pkl')
    agent = DQN(env.state_dim, len(ACTIONS))
    replay_buffer = ReplayBuffer(1000000)   # Experience replay buffer
    rewards = []    # Store rewards for monitoring performance
    successful_episodes = 0
    episode_lengths = []    # Store lenghts of episodes
    total_rewards = deque(maxlen=100)   # Moving average of rewards
    
    # Initialize Weights & Biases (wandb) for logging
    wandb.init(project="DQN", config={"episodes": 5000, "max_steps": env.max_steps})
    
    # Training loop for a number of episodes
    for episode in range(5):
        state = env.reset() # Reset environment at the start of each episode
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action_index = agent.select_action(state)   # Select action from policy
            next_state, reward, done = env.step(action_index)   # Execute action in the environment
            replay_buffer.add(state, action_index, next_state, reward, done)    # Store experience
            agent.train(replay_buffer)  # Train the agent using replay buffer
            state = next_state  # Update current state
            total_reward += reward  # Accumulate episode reward
            step_count += 1

        rewards.append(total_reward)    # Store total reward for monitoring  
        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards) # Compute moving average reward
        dis = np.linalg.norm(env.next_position - env.target_position)   # Compute final distance to target

        episode_lengths.append(step_count)  # Compute episode lenght
        if dis < 0.001:
            successful_episodes += 1  # Calculate successful episodes

        agent.update_target_network()
        
        # Log metrics using wandb
        wandb.log({'avg_reward': avg_reward, 'episode': episode})
        wandb.log({'distance': dis, 'episode': episode})
        wandb.log({'epsilon': agent.epsilon, 'episode': episode})
        wandb.log({'loss': agent.last_loss, 'episode': episode})  # Logovanie straty
        wandb.log({'avg_episode_length': np.mean(episode_lengths), 'episode': episode})
        wandb.log({'success_rate': successful_episodes / (episode + 1), 'episode': episode})
        
        print(f"Episode: {episode + 1}, Avg Reward: {avg_reward}, Distance: {dis}, Success Rate: {successful_episodes / (episode + 1)}")
    
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "RL_TD3_continuum_robot_control_5"))
    directory = os.path.join(base_dir, "DQNLearnedModel")
    if not os.path.exists(directory):
        os.makedirs(directory)    
    model_path = os.path.join(directory, "dqn_model.pth")
    # Save trained model
    torch.save(agent.q_network.state_dict(), model_path)
    
    # Test trained model
    trained_model = QNetwork(7, len(ACTIONS))
    trained_model.load_state_dict(torch.load(model_path))
    trained_model.eval()
    
    distacne = np.linalg.norm(np.array([0,0,0]) - (np.array([-26.33573,-140.7042,264.039]))/1000)
    state_pred = np.array([0,0,0,-0.02633573,-0.1407042,0.264039,distacne], dtype=np.float32)
    state_tensor = torch.tensor(state_pred).unsqueeze(0)
    
    with torch.no_grad():
        q_values = trained_model(state_tensor)
        
    best_action_idx = torch.argmax(q_values).item()
    best_action = ACTIONS[best_action_idx]
    actual_position = env._simulate_robot_model(best_action)
    print(f"Predicted position: {actual_position*1000}")

if __name__ == "__main__":
    main()