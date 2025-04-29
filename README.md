# RL_TD3_continuum_robot_control
## Overview
This project implements a reinforcement learning approach to control the position of the end-effector of a one-segment cable-driven continuum robot developed at ARM-lab. To achieve this task, the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm was applied. The control agent was trained on the simulation model of the robot based on the feedforward neural network (FNN) that was trained on data gained from measurements of cable lengths and corresponding positions of the end effector collected on the physical model of the robot. To compare the performance of this approach, the Deep Deterministic Policy Gradient (DDPG) and Deep Q-learning algorithms were also tested.
## Installation
The recommended version of Python to use for this project is 3.12.7.
To set up this project locally, follow these steps:
### 1. Clone the repository
```bash
git clone <paste SSH>
cd RL_TD3_continuum_robot_control
```
### 2. Create a virtual environment (recommended):
```bash
python3 -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```
### 3. Install the required libraries:
```bash
pip install -r requirements.txt
```