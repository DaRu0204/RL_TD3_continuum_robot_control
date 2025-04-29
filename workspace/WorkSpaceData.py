import torch
import numpy as np
import pandas as pd
import joblib
from itertools import product
import os
import sys
# Add the base directory to sys.path
c_dir = os.path.dirname(os.path.abspath(__file__))
b_dir = os.path.abspath(os.path.join(c_dir, ".."))
sys.path.append(b_dir)
from SL_learning.SupervisedLearningModel import NeuralNetwork

# Define the directory and file paths within the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "RL_TD3_continuum_robot_control"))
model_dir = os.path.join(base_dir, "SimulateRobotLearnedModel")
model_path = os.path.join(model_dir, "trained_model_sl_1.pth")
scaler_X_path = os.path.join(model_dir, "scaler_X_sl_1.pkl")
scaler_y_path = os.path.join(model_dir, "scaler_y_sl_1.pkl")

# Check if the model directory exists, and if not, print an error and exit
if not os.path.exists(model_dir):
    print(f"Error: The directory '{model_dir}' does not exist.")
    exit(1)

# Check if model and scaler files exist within the directory
if not (os.path.exists(model_path) and os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path)):
    print("Error: Model or scaler files are missing in the 'SimulateRobotLearnedModel' directory.")
    exit(1)

# Load the model
model = NeuralNetwork()
model.load_state_dict(torch.load(model_path))
model.eval()

# Load the saved scalers
scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)

# Prediction function
def predict_position(l1, l2, l3):
    # Normalize the input
    input_data = scaler_X.transform([[l1, l2, l3]])
    input_tensor = torch.tensor(input_data, dtype=torch.float32)

    # Model prediction
    with torch.no_grad():
        prediction = model(input_tensor).numpy()

    # Inverse transform of the output normalization
    predicted_position = scaler_y.inverse_transform(prediction)
    return predicted_position[0]

# Generate all combinations of l1, l2, l3 (from 0 to 100, except 0,0,0)
combinations = [
    (l1, l2, l3)
    for l1, l2, l3 in product(range(101), repeat=3)
    if not (l1 == 0 and l2 == 0 and l3 == 0)
]

# Initialize a list to store the dataset
dataset = []

# Process each combination
for l1, l2, l3 in combinations:
    predicted_position = predict_position(l1, l2, l3)
    dataset.append([l1, l2, l3, *predicted_position])  # Combine inputs and outputs

# Convert to a Pandas DataFrame
dataset_df = pd.DataFrame(dataset)

# Save to a file
output_file = os.path.join(base_dir, "dataset", "workspace_point_dataset_2.txt")
dataset_df.to_csv(output_file, index=False, header=False, sep=",")

print(f"Dataset successfully saved to {output_file}")