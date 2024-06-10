import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import deque
import random
import os
from last import ContinuumRobotEnv
from last import TD3

robot = ContinuumRobotEnv()
loaded_agent = TD3(6, 3, 1)
loaded_agent.load("td3_continuum_robot")

desired_positon = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.7])
action = loaded_agent.select_action(desired_positon)
state, reward, done = robot.step(action)
print("Action:",action)
print("State:",state)