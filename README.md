# RL_TD3_continuum_robot_control
## Description
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
## Project structure
This project is organised into several directories:
*   dataset\/: Contains data for training simulation and control models.
*   workspace\/: Contains scripts to display workspace and generate .txt files that contain workspace points.
*   SL_learning\/: Contains scripts to train and test simulation model.
*   SimulateRobotLearnedModel\/: Contains saved simulation model and scalers.
*   TD3_learning\/: Contains scripts to train, test, and optimize the control model based on the TD3 algorithm.
*   TD3LearnedModel\/: Contains saved TD3 control model.
*   DDPG_learning\/: Contains scripts to train and test the control model based on the DDPG algorithm.
*   DDPGLearnedModel\/: Contains saved DDPG control model.
*   DQN_learning\/: Contains scripts to train and test the control model based on the DQN algorithm.
*   DQNLearnedModel\/: Contains saved DQN control model.
## Usage
### 1. Train simulation model
To train the simulation model, run the script:
```bash
python SupervisedLearningModel.py
```
This script loads and prepares data for training, then executes the training loop and saves the trained model to the SimulateRobotLearnedModel\/ directory.

The trained model can be tested by executing the script:
```bash
python ModelTester.py
```
This script allows the user to predict the position of the end-effector based on manually inserted cable lengths or randomly chosen cable lengths from the dataset.
### 2. Workspace generation
To generate workspace of the robot, run the script:
```bash
python WorkSpaceData.py
```
This script generates a text file containing workspace points for all allowable action combinations produced by the trained simulation model. The file is saved to the dataset/ directory. This file is used during the training of the control model to select random point to reach during the training episode.
### 3. Train control model
To train the TD3-based control model, run the script:
```bash
python start.py
```
or
```bash
python fulltd3.py
```
Both scripts activate the training process of the control agent; W&B is used for remote training progress observation.

To find optimal network architecture, run the script:
```bash
python td3architecture.py
```
To find the optimal hyperparameter combination, run the script:
```bash
python td3optim.py
```
Both scripts utilize W&B's sweep function to find optimal combinations of selected parameters.

To other models:
-   DDPG:
```bash
python fullddpg.py
```
- DQN:
```bash
python fulldqn.py
```
### 4. Control model testing
To test trained models, run the script:
```bash
python TD3ModelTesting.py
```
```bash
python DDPGModelTesting.py
```
```bash
python DQNModelTesting.py
```
These scripts allow the user to access positioning accuracy and path-tracking capabilities of the trained model.
## Contributing
If you'd like to contribute, fork the repository and create a pull request with your changes.
## License
This project is licensed under the MIT License.