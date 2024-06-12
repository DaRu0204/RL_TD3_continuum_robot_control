import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import wandb
from wandb.keras import WandbCallback
import os
from ActorCritic import ReplayBuffer
from ContinuumRobot import ContinuumRobotEnv
from td3 import TD3

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# Instantiate environment and TD3 agent
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
    total_episodes = 100
    rewards = []
    batch_size = 64
    critic_losses = []  # List to store critic losses
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
        print(f"Episode: {episode + 1}, Average Reward: {avg_reward}")
        
    td3_agent.save("td3_continuum_robot")

    loaded_agent = TD3(state_dim=env.state_dim, action_dim=env.action_dim, max_action=env.max_action)
    loaded_agent.load("td3_continuum_robot")
    #desired_positon = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.7])
    desired_positon = np.array([0.075, 0.04])
    action = loaded_agent.select_action(desired_positon)
    state, reward, done = env.step(action)
    error = np.linalg.norm(desired_positon - state)
    print("Action:",action)
    print("State:",state)
    print("Error:",error)
    env.render(state, action)

if __name__ == "__main__":
    main()