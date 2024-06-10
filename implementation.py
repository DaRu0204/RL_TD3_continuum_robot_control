import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import deque
import random
from last import TD3
from last import Actor  # Assuming you have the Actor model saved separately
from last import ContinuumRobotEnv
import os

#agent = TD3(state_dim = 6, action_dim = 3, max_action = 1.5)
#loaded_agent = agent.load_agent('/home/km/GitHub/td3_continuum_robot/actor.pth','/home/km/GitHub/td3_continuum_robot/critic_1.pth','/home/km/GitHub/td3_continuum_robot/critic_2.pth')

state_dim = 6  # Example state dimension
action_dim = 3  # Example action dimension
max_action=1.5
model_name=""
agent = TD3(6, 3, 1.5)
robot = ContinuumRobotEnv()  

# Assuming you have a desired position
desired_position = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.7])  # Example desired position

# Get action from the agent
#action = agent.select_action2(desired_position, state_dim, action_dim, max_action, model_name)
action = agent.select_action(desired_position)
next_state, reward, done = robot.step(action)
print("State:", next_state)
print("Actions:", action)