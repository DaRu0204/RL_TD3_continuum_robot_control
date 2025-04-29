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
from DynamcNoise import ExplorationNoise

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# Instantiate environment and TD3 agent
env = ContinuumRobotEnv()
td3_agent = TD3(env.state_dim, env.action_dim, env.max_action)
exploration_noise = ExplorationNoise(env.action_dim, env.max_action, initial_std=0.2, min_std=0.05, decay_rate=0.99)


def main():
    # Training loop
    total_episodes = 5000
    rewards = []
    batch_size = 128
    replay_buffer = ReplayBuffer(buffer_size=1000000)
    episode_rewards = deque(maxlen=100)

    # start a new wandb run to track this script
    wandb.init(project="TD3",name="PC1",config={"learning_rate": td3_agent.lr_actor, "epochs": total_episodes, "step": ContinuumRobotEnv.step, "gamma": td3_agent.gamma})

    # check for log file
    log_file = 'td3_log.txt'
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            pass    # Create log file without header
    
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

        # Store episode metrics
        dis = env.distance(state)
        rewards.append(episode_reward)
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards)
        wandb.log({'avg_reward': avg_reward, 'episode': episode})
        wandb.log({'distance': dis, 'episode': episode})
        
        # log measured metrics to text file
        with open(log_file, 'a') as f:
            f.write(f"{episode + 1},{dis:.6f},{avg_reward:.6f}\n")
            
        print(f"Episode: {episode + 1}, Average Reward: {avg_reward}")
        
    #td3_agent.save("/LearnedModel/td3_continuum_robot")
    td3_agent.save("td3_continuum_robot")
    
    loaded_agent = TD3(state_dim=env.state_dim, action_dim=env.action_dim, max_action=env.max_action)
    loaded_agent.load("td3_continuum_robot")   
    
    # desired_positon = np.array([0.097, -0.022])
    # desired_positon = np.array(env.random_target())/1000
    desired_positon = np.array([-26.33573,-140.7042,264.039])/1000
    starting_position = np.zeros(3)
    distanse = np.array(np.linalg.norm(starting_position - desired_positon))
    predictors = np.concatenate([starting_position, desired_positon, [distanse]])
    actionn = loaded_agent.select_action(predictors)
    statee = env._simulate_robot_model(actionn)
    print("State:", statee*1000)
    """
    error = np.array(np.linalg.norm(statee - desired_positon))
    # Ensure state is updated before rendering
    # statee, reward, done = env.step(actionn)
    # print("State after step:", statee)
    # error = np.linalg.norm(desired_positon - statee)
    print("Desired position:",desired_positon*1000)
    print("Action:", actionn*100)
    print("State:", statee*1000)
    print("Error:", error*1000)
    
    # Render the environment based on the updated state
    # env.render(actions=actionn)
    """
if __name__ == "__main__":
    main()