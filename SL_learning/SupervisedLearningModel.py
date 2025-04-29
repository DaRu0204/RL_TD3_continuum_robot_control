import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import joblib
import numpy as np
import os
import wandb

# Class for data preparation
class DataPreparation:
    def __init__(self, data_path):
        # Read the dataset from a file with columns l1, l2, l3 for inputs and x, y, z for outputs
        data = pd.read_csv(data_path, header=None, names=["l1", "l2", "l3", "x", "y", "z"])

        # Separate features (X) and target variables (y)
        self.X = data[["l1", "l2", "l3"]].values
        self.y = data[["x", "y", "z"]].values

        # Standardize the features and targets for better model training performance
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.X = self.scaler_X.fit_transform(self.X)
        self.y = self.scaler_y.fit_transform(self.y)

        # Splitting the data into training (70%), validation (20%), and test (10%) sets
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(self.X, self.y, test_size=0.30, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

        # Convert data to PyTorch tensors
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
        self.X_val = torch.tensor(self.X_val, dtype=torch.float32)
        self.y_val = torch.tensor(self.y_val, dtype=torch.float32)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32)

    def get_data_loaders(self, batch_size=64):
        # Create DataLoader for training and validation sets
        train_dataset = TensorDataset(self.X_train, self.y_train)
        val_dataset = TensorDataset(self.X_val, self.y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

# Defining the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        # Fully connected layers for feature extraction
        super(NeuralNetwork, self).__init__()
        """
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 3)
        """
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 3)
        
        self.relu = nn.ReLU()   # Activation function

    def forward(self, x):
        # Define forward pass through the network layers with ReLU activations
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x) # No activation in the output layer for regression
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x


# Main function for training the model
def main():
    
    # Initialise wandb
    wandb.init(project="Supervised learining", name="Best_(1)", config={"epochs": 300, "batch_size": 64, "learning_rate": 0.0001})
    
    # Get the current working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Find the base directory (RL_TD3_continuum_robot_control)
    base_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "RL_TD3_continuum_robot_control"))
    
    # Initialize data preparation class
    dataset_path = os.path.join(base_dir, "dataset","Dataset-Actions-Positions.txt")
    data_prep = DataPreparation(dataset_path)
    train_loader, val_loader = data_prep.get_data_loaders(batch_size=64)

    # Initialize model
    model = NeuralNetwork()

    # Set up loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)   # for lr = 0.0003 -> avg_acc = 0.2913
                                                            
    # Lists to store training and validation losses over epochs
    train_losses = []
    val_losses = []

    # Model training
    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()   # Set model to training mode
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()   # Clear gradients from previous step
            predictions = model(X_batch)    # Forward pass
            loss = criterion(predictions, y_batch)  # Compute loss
            loss.backward() # Backward pass to calculate gradients
            optimizer.step()    # Update weights
            epoch_loss += loss.item()   # Accumulate loss

        # Average training loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)

        # Calculate validation loss
        model.eval()    # Set model to evaluation mode
        with torch.no_grad():   # Disable gradient tracking for validation
            val_loss = 0
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

        # Log training and validation loss to wand
        wandb.log({"epoch": epoch, "train_loss": avg_epoch_loss, "val_loss": avg_val_loss})

        # Print losses every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Training Loss: {avg_epoch_loss}, Validation Loss: {avg_val_loss}")

    # Calculate and print average validation loss for the last 20% of epochs
    last_20_percent_epochs = int(num_epochs * 0.2)
    avg_val_loss_last_20_percent = sum(val_losses[-last_20_percent_epochs:]) / last_20_percent_epochs
    print(f"Average Validation Loss for the last 20% epochs: {avg_val_loss_last_20_percent}")

    # Testing the model after training
    model.eval()    # Set model to evaluation mode
    with torch.no_grad():
        test_predictions = model(data_prep.X_test)
        test_loss = criterion(test_predictions, data_prep.y_test)
        print(f"Final Test Loss: {test_loss.item()}")

        # Convert predictions and true values back to the original scale
        test_predictions = data_prep.scaler_y.inverse_transform(test_predictions.numpy())
        y_test = data_prep.scaler_y.inverse_transform(data_prep.y_test.numpy())

        # Calculate R² Score, MAE and RMSE for the model performance
        r2 = r2_score(y_test, test_predictions)
        mae = mean_absolute_error(y_test, test_predictions)
        rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    
    # Check if the pacage exists
    output_dir = os.path.join(base_dir, "SimulateRobotLearnedModel")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Saving the trained model and scalers
    torch.save(model.state_dict(), os.path.join(output_dir, "trained_model_sl_1.pth"))
    joblib.dump(data_prep.scaler_X, os.path.join(output_dir, "scaler_X_sl_1.pkl"))
    joblib.dump(data_prep.scaler_y, os.path.join(output_dir, "scaler_y_sl_1.pkl"))
    print("Model and scalers were saved.")
    
if __name__ == "__main__":
    main()