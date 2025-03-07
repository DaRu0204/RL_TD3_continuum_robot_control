import os
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import random
import pandas as pd
import sys
# Add the base directory to sys.path
c_dir = os.path.dirname(os.path.abspath(__file__))
b_dir = os.path.abspath(os.path.join(c_dir, ".."))
sys.path.append(b_dir)
from SL_learning.SupervisedLearningModel import NeuralNetwork

class ContinuumRobotEnv:

    step = 100
    x_min, x_max = -70.63, -18.33       # -108.15, 19.19
    y_min, y_max = -144.35, -92.05      # -144.35, -92.05
    z_min, z_max = 247.45, 299.76       # 204.57, 342.64

    def __init__(self, segment_length=0.1, num_segments=1, num_tendons=3, max_steps=step, max_action=1):      # never change max_action = 1
        """
        Initialize the environment for a continuum robot with defined number of segments in 3D space.
        """
        # Define paths for the model, scalers and dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "RL_TD3_continuum_robot_control_5"))
        model_dir = os.path.join(base_dir, "SimulateRobotLearnedModel")
        model_path = os.path.join(model_dir, "trained_model_lr3.pth")
        scaler_X_path = os.path.join(model_dir, "scaler_X_lr3.pkl")
        scaler_y_path = os.path.join(model_dir, "scaler_y_lr3.pkl")
        dataset_path = os.path.join(base_dir, "dataset", "workspace_point_dataset.txt")
        # dataset_path = os.path.join(base_dir, "dataset", "Dataset-Actions-Positions.txt") # For testing only
        
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

        # self.state_dim = 3      # Robot's state is represented by the position in 3D space (x, y, z)
        self.state_dim = 7  # Include current position, target position and distance between them
        self.action_dim = num_segments * num_tendons

        # self.state = np.zeros(self.state_dim)   # Initialize the robot's position to (0, 0, 0)
        self.state = None
        self.target_position = None
        # self.target_position = np.array(ContinuumRobotEnv.random_target())/1000
        # print("self.target_position:",self.target_position)
        # self.target_position = np.array([-63.333,-128.249,277.89])/1000
        # self.target_position = np.array([-0.8,-0.101])
        # self.target_position = np.array([0.097, -0.022])
        self.current_step = 0

    def random_target(self):
        """
        x_target = random.uniform(ContinuumRobotEnv.x_min, ContinuumRobotEnv.x_max)
        y_target = random.uniform(ContinuumRobotEnv.y_min, ContinuumRobotEnv.y_max)
        z_target = random.uniform(ContinuumRobotEnv.z_min, ContinuumRobotEnv.z_max)
        target = np.array([x_target, y_target, z_target])
        """
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
        print("Target position for this episode:", self.target_position)
        distance_to_target = np.array(self.distance(initial_state))
        # self.state = initial_state  # Reset the enviroment's state
        self.state = np.concatenate([initial_state, self.target_position, [distance_to_target]])
        print("State for this episode:", self.state)
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
        done = self.current_step >= self.max_steps or self.distance(next_position) < 0.0005
        # self.state = next_state  # Update the environment's state
        self.state = np.concatenate([next_position, self.target_position, [distance_to_target]])  # Update enviroment's state
        # print(f"Step {self.current_step}: Actions: {actions}, Next State: {next_position}, Reward: {reward}")
        return self.state, reward, done

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
    
    def _simulate_robot_model(self, actions):
        """
        Use the trained model to predict the robot's position (x, y, z) based on the actions.
        """
        # Normalize the input actions
        actions_scaled = np.array(actions)*100
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

    def render(self, actions=None):
        segment_positions = [np.zeros(2)]
        orientation = 0.0

        if actions is not None:
            actions = np.clip(actions, -self.max_action, self.max_action)

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
        print("segment_positions:", segment_positions[-1])
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
        # distance_to_goal = np.linalg.norm(state - self.target_position)
        if len(state) == 3:
            distance_to_goal = np.linalg.norm(state - self.target_position)
        else:
            distance_to_goal = np.linalg.norm(state[:3] - self.target_position)
        return distance_to_goal