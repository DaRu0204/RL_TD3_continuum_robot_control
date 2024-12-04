import torch
import numpy as np
import matplotlib.pyplot as plt
from ContinuumRobot import ContinuumRobotEnv
from TD3 import TD3


def test_single_point(env, agent, num_tests=1):
    """Test the TD3 agent on a single target position multiple times."""
    errors = []  # List to store errors for all tests

    for test in range(num_tests):
        # Generate desired position and predict outcome
        desired_position = np.array(env.random_target()) / 1000
        starting_position = np.zeros(3)
        distance_to_target = np.linalg.norm(starting_position - desired_position)
        predictors = np.concatenate([starting_position, desired_position, [distance_to_target]])

        action = agent.select_action(predictors)
        predicted_state = env._simulate_robot_model(action)
        error = np.linalg.norm(predicted_state - desired_position)

        # Append error to the list
        errors.append(error * 1000)  # Convert to mm

        # Print results for the current test
        print(f"Test {test + 1}:")
        print("  Desired Position (mm):", desired_position * 1000)
        print("  Predicted State (mm):", predicted_state * 1000)
        print("  Error (mm):", error * 1000)
        print("  Action:", action * 100)
        print("-" * 50)

    # Print overall statistics
    errors = np.array(errors)
    print("\nSingle Point Test Results:")
    print(f"  Minimum Error (mm): {np.min(errors):.2f}")
    print(f"  Maximum Error (mm): {np.max(errors):.2f}")
    print(f"  Average Error (mm): {np.mean(errors):.2f}")

    # Plot histogram of errors
    plot_single_point_test_errors(errors)


def plot_single_point_test_errors(errors):
    """Plot errors as a line chart with markers for each test."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(errors) + 1), errors, marker='o', linestyle='-', label="Error (mm)", color='b')

    # Add labels and legend
    plt.title("Errors per Test")
    plt.xlabel("Test Number")
    plt.ylabel("Error (mm)")
    plt.legend()
    plt.grid(True)
    plt.show()

def test_rectangle(env, agent, rectangle_points):
    """
    Test the TD3 agent on a rectangle formed by four points.
    Rectangle traversal: P1 -> P2 -> P3 -> P4 -> P1
    rectangle_points: List of four points in 3D space representing the rectangle vertices.
    """
    current_position = np.zeros(3)  # Start from (0, 0, 0)
    errors = []
    predicted_positions = []
    desired_positions = []

    # Full loop through the rectangle
    for i in range(len(rectangle_points) + 1):  # +1 to return to P1
        desired_position = rectangle_points[i % len(rectangle_points)]  # Loop back to P1 at the end
        predictors = np.concatenate([current_position, desired_position, [np.linalg.norm(current_position - desired_position)]])
        action = agent.select_action(predictors)
        predicted_state = env._simulate_robot_model(action)

        # Record results
        error = np.linalg.norm(predicted_state - desired_position)
        errors.append(error * 1000)  # Convert to mm
        predicted_positions.append(predicted_state * 1000)
        desired_positions.append(desired_position * 1000)

        print(f"Step {i + 1}:")
        print("  Desired Position (mm):", desired_position * 1000)
        print("  Predicted State (mm):", predicted_state * 1000)
        print("  Error (mm):", error * 1000)
        print("  Action:", action * 100)
        print("-" * 50)

        # Update current position
        current_position = predicted_state

    # Plot results
    plot_rectangle_test(predicted_positions, desired_positions)

    # Print overall statistics
    print("\nRectangle Test Results:")
    print(f"  Minimum Error (mm): {np.min(errors):.2f}")
    print(f"  Maximum Error (mm): {np.max(errors):.2f}")
    print(f"  Average Error (mm): {np.mean(errors):.2f}")


def plot_rectangle_test(predicted_positions, desired_positions):
    """Plot the desired and predicted positions for the rectangle test."""
    predicted_positions = np.array(predicted_positions)
    desired_positions = np.array(desired_positions)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot desired positions
    ax.plot(desired_positions[:, 0], desired_positions[:, 1], desired_positions[:, 2], marker='o', color='b', label="Desired Path")
    
    # Plot predicted positions
    ax.plot(predicted_positions[:, 0], predicted_positions[:, 1], predicted_positions[:, 2], marker='x', color='r', label="Predicted Path")

    # Annotate points
    for i, (dp, pp) in enumerate(zip(desired_positions, predicted_positions)):
        ax.text(dp[0], dp[1], dp[2], f"P{i+1} Desired", color='blue')
        ax.text(pp[0], pp[1], pp[2], f"P{i+1} Predicted", color='red')

    ax.set_title("Desired vs Predicted Path")
    ax.set_xlabel("X Coordinate (mm)")
    ax.set_ylabel("Y Coordinate (mm)")
    ax.set_zlabel("Z Coordinate (mm)")
    ax.legend()
    plt.show()

def generate_rectangle_points(center, width, height):
    """
    Generate four points that form a rectangle in 3D space.
    :param center: np.array of shape (3,), the center of the rectangle.
    :param width: float, the width of the rectangle.
    :param height: float, the height of the rectangle.
    :return: np.array of shape (4, 3), the four corner points of the rectangle.
    """
    # Calculate corner points relative to the center
    half_width = width / 2
    half_height = height / 2

    P1 = center + np.array([-half_width, -half_height, 0])  # Bottom-left
    P2 = center + np.array([half_width, -half_height, 0])   # Bottom-right
    P3 = center + np.array([half_width, half_height, 0])    # Top-right
    P4 = center + np.array([-half_width, half_height, 0])   # Top-left

    return np.array([P1, P2, P3, P4])

def main():
    # Instantiate environment and TD3 agent
    env = ContinuumRobotEnv()
    agent = TD3(state_dim=env.state_dim, action_dim=env.action_dim, max_action=env.max_action)
    agent.load("td3_continuum_robot")

    print("Select Test Mode:")
    print("1. Single Point Test")
    print("2. Rectangle Test")
    choice = int(input("Enter your choice (1 or 2): "))

    if choice == 1:
        num_tests = int(input("Enter the number of single point tests to perform: "))
        test_single_point(env, agent, num_tests)

    elif choice == 2:
        # Define rectangle points
        center = env.random_target()/1000  # Center of rectangle in meters
        width = 30.0 / 1000  # Width in meters
        height = 10.0 / 1000  # Height in meters

        # Generate rectangle points
        rectangle_points = generate_rectangle_points(center, width, height)
        test_rectangle(env, agent, rectangle_points)

    else:
        print("Invalid choice! Exiting.")

if __name__ == "__main__":
    main()