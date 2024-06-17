import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import deque
import random
import os
from ContinuumRobot import ContinuumRobotEnv
from TD3 import TD3

robot = ContinuumRobotEnv()
loaded_agent = TD3(2, 3, 1)
loaded_agent.load("td3_continuum_robot")

desired_positon = np.array([0.045, 0.08])
action = loaded_agent.select_action(desired_positon)
state, reward, done = robot.step(action)
error = np.linalg.norm(desired_positon - state)
print("Action:",action)
print("State:",state)
print("Error:",error)