import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import os
import matplotlib.pyplot as plt
import wandb
from ContinuumRobot import ContinuumRobotEnv
from TD3 import TD3
from ActorCritic import ReplayBuffer

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# Instantiate environment and TD3 agent
env = ContinuumRobotEnv()
td3_agent = TD3(env.state_dim, env.action_dim, env.max_action)


def main():
    # Training loop
    total_episodes = 5000
    rewards = []
    batch_size = 64
    replay_buffer = ReplayBuffer(buffer_size=1000000)
    episode_rewards = deque(maxlen=100)

    # start a new wandb run to track this script
    wandb.init(project="TD3",config={"learning_rate": td3_agent.lr_actor, "epochs": total_episodes, "step": ContinuumRobotEnv.step, "gamma": td3_agent.gamma})

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
        
    #td3_agent.save("/LearnedModel/td3_continuum_robot")
    td3_agent.save("td3_continuum_robot")

    loaded_agent = TD3(state_dim=env.state_dim, action_dim=env.action_dim, max_action=env.max_action)
    loaded_agent.load("td3_continuum_robot")   
    
    desired_positon = np.array([0.08, 0.045])
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