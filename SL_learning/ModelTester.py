import torch
import numpy as np
import pandas as pd
import joblib
from SupervisedLearningModel import NeuralNetwork  # Importing the model from the original program
import os

# Get the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Find the base directory (RL_TD3_continuum_robot_control)
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

# Load the dataset
dataset_path = os.path.join(base_dir, "dataset", "Dataset-Actions-Positions.txt")
data = pd.read_csv(dataset_path, header=None, names=["l1", "l2", "l3", "x", "y", "z"])

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

# Main program function
def main():
    mode = input("Enter '1' for manual input or '2' for random selection from the dataset: ")
    
    if mode == '1':
        # Manual input mode
        while True:
            try:
                # Get inputs from the user
                l1 = float(input("Enter value for l1: "))
                l2 = float(input("Enter value for l2: "))
                l3 = float(input("Enter value for l3: "))

                # Prediction
                x, y, z = predict_position(l1, l2, l3)
                print(f"Predicted coordinates: x = {x:.2f}, y = {y:.2f}, z = {z:.2f}")

            except ValueError:
                print("Values must be numbers. Please try again.")
            except KeyboardInterrupt:
                print("\nProgram terminated.")
                break

    elif mode == '2':
        # Random selection from dataset mode
        try:
            n = int(input("Enter the number of values (n) you want to test: "))
            inaccuracies = []  # List to store inaccuracy for each selection

            for _ in range(n):
                # Select a random index from the dataset
                random_index = np.random.randint(0, len(data))
                sample = data.iloc[random_index]

                # Values from the dataset
                l1, l2, l3 = sample["l1"], sample["l2"], sample["l3"]
                dataset_pos = sample[["x", "y", "z"]].values

                # Model prediction
                model_pos = predict_position(l1, l2, l3)

                # Calculate inaccuracy
                inaccuracy = np.linalg.norm(dataset_pos - model_pos)
                inaccuracies.append(inaccuracy)  # Append inaccuracy to list

                # Display results
                print(f"Actual coordinates: x = {dataset_pos[0]:.2f}, y = {dataset_pos[1]:.2f}, z = {dataset_pos[2]:.2f}")
                print(f"Predicted coordinates: x = {model_pos[0]:.2f}, y = {model_pos[1]:.2f}, z = {model_pos[2]:.2f}")
                print(f"Inaccuracy (distance): {inaccuracy:.4f}\n")

            # Calculate and display the average, maximum, and minimum inaccuracy
            avg_inaccuracy = np.mean(inaccuracies)
            max_inaccuracy = np.max(inaccuracies)
            min_inaccuracy = np.min(inaccuracies)
            
            print(f"Average inaccuracy after {n} tests: {avg_inaccuracy:.4f}")
            print(f"Maximum inaccuracy (distance): {max_inaccuracy:.4f}")
            print(f"Minimum inaccuracy (distance): {min_inaccuracy:.4f}")
            print(f"Result: {avg_inaccuracy:4f} ± {np.std(inaccuracies):.4f}")

        except ValueError:
            print("Number of values (n) must be an integer.")
        except KeyboardInterrupt:
            print("\nProgram terminated.")

    else:
        print("Invalid choice. Enter '1' or '2'.")

if __name__ == "__main__":
    main()