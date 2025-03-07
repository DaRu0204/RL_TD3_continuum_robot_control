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


def test_rectangle_combined(env, agent, rectangle_points):
    """
    Perform two tests on a rectangle:
    1. Traversal test: Move from Pn-1 to Pn (loop).
    2. Origin test: Always move from origin to Pn.
    :param env: ContinuumRobotEnv instance.
    :param agent: TD3 agent instance.
    :param rectangle_points: List of points in 3D space representing the rectangle vertices.
    """
    print("\n--- Running Rectangle Tests ---")
    
    # Run traversal test and collect results
    traversal_errors, traversal_predicted_positions, traversal_desired_positions = test_rectangle(env, agent, rectangle_points, verbose=False)
    
    # Run origin test and collect results
    origin_errors, origin_predicted_positions, origin_desired_positions = test_rectangle_with_origin(env, agent, rectangle_points, verbose=False)
    
    # Display results for traversal test
    print("\n--- Rectangle Traversal Test Results (Pn-1 → Pn) ---")
    display_test_results(traversal_errors, traversal_predicted_positions, traversal_desired_positions)
    
    # Display results for origin test
    print("\n--- Rectangle Origin Test Results (Origin → Pn) ---")
    display_test_results(origin_errors, origin_predicted_positions, origin_desired_positions)


def test_rectangle(env, agent, rectangle_points, verbose=True):
    """
    Test the TD3 agent on a rectangle traversal (Pn-1 → Pn).
    Rectangle traversal: P1 → P2 → P3 → P4 → P1.
    """
    current_position = np.zeros(3)  # Start from (0, 0, 0)
    errors = []
    predicted_positions = []
    desired_positions = []

    for i in range(len(rectangle_points) + 1):  # +1 to loop back to P1
        desired_position = rectangle_points[i % len(rectangle_points)]
        predictors = np.concatenate([current_position, desired_position, [np.linalg.norm(current_position - desired_position)]])
        action = agent.select_action(predictors)
        predicted_state = env._simulate_robot_model(action)

        # Record results
        error = np.linalg.norm(predicted_state - desired_position)
        errors.append(error * 1000)  # Convert to mm
        predicted_positions.append(predicted_state * 1000)
        desired_positions.append(desired_position * 1000)

        # Optionally print each step
        if verbose:
            print(f"Step {i + 1}:")
            print("  Desired Position (mm):", desired_position * 1000)
            print("  Predicted State (mm):", predicted_state * 1000)
            print("  Error (mm):", error * 1000)
            print("  Action:", action * 100)
            print("-" * 50)

        # Update current position
        current_position = predicted_state

    return errors, predicted_positions, desired_positions


def test_rectangle_with_origin(env, agent, rectangle_points, verbose=True):
    """
    Test the TD3 agent with a rectangle traversal where each point is reached from the origin.
    """
    origin = np.zeros(3)  # Starting point (0, 0, 0)
    errors = []
    predicted_positions = []
    desired_positions = []

    for i, desired_position in enumerate(rectangle_points):
        # Calculate predictors
        distance_to_target = np.linalg.norm(origin - desired_position)
        predictors = np.concatenate([origin, desired_position, [distance_to_target]])

        # Get action from agent and simulate the robot
        action = agent.select_action(predictors)
        predicted_state = env._simulate_robot_model(action)

        # Record results
        error = np.linalg.norm(predicted_state - desired_position)
        errors.append(error * 1000)  # Convert to mm
        predicted_positions.append(predicted_state * 1000)
        desired_positions.append(desired_position * 1000)

        # Optionally print each step
        if verbose:
            print(f"Point {i + 1}:")
            print("  Desired Position (mm):", desired_position * 1000)
            print("  Predicted State (mm):", predicted_state * 1000)
            print("  Error (mm):", error * 1000)
            print("  Action:", action * 100)
            print("-" * 50)

    return errors, predicted_positions, desired_positions


def display_test_results(errors, predicted_positions, desired_positions):
    """Display test results and plot desired vs predicted paths."""
    # Print statistics
    print(f"  Minimum Error (mm): {np.min(errors):.2f}")
    print(f"  Maximum Error (mm): {np.max(errors):.2f}")
    print(f"  Average Error (mm): {np.mean(errors):.2f}")

    # Plot results
    plot_rectangle_test_with_zero(predicted_positions, desired_positions)
    plot_rectangle_test(predicted_positions, desired_positions)


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

    ax.set_title("Desired vs Predicted Path")
    ax.set_xlabel("X Coordinate (mm)")
    ax.set_ylabel("Y Coordinate (mm)")
    ax.set_zlabel("Z Coordinate (mm)")
    ax.legend()
    plt.show()
    
    
def plot_rectangle_test_with_zero(predicted_positions, desired_positions):
    """Plot the desired and predicted positions with (0, 0, 0) as reference."""
    predicted_positions = np.array(predicted_positions)
    desired_positions = np.array(desired_positions)

    # Adjust positions relative to (0, 0, 0)
    predicted_positions_centered = predicted_positions - np.array([0, 0, 0])
    desired_positions_centered = desired_positions - np.array([0, 0, 0])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot desired positions
    ax.plot(desired_positions_centered[:, 0], desired_positions_centered[:, 1], desired_positions_centered[:, 2],
            marker='o', color='b', label="Desired Path")

    # Plot predicted positions
    ax.plot(predicted_positions_centered[:, 0], predicted_positions_centered[:, 1], predicted_positions_centered[:, 2],
            marker='x', color='r', label="Predicted Path")

    ax.set_title("Desired vs Predicted Path (Centered at 0,0,0)")
    ax.set_xlabel("X Coordinate (mm)")
    ax.set_ylabel("Y Coordinate (mm)")
    ax.set_zlabel("Z Coordinate (mm)")
    ax.legend()

    # Adjust the axis limits for better visualization
    all_points = np.vstack((predicted_positions_centered, desired_positions_centered))
    ax.set_xlim([all_points[:, 0].min() - 10, all_points[:, 0].max() + 10])
    ax.set_ylim([all_points[:, 1].min() - 10, all_points[:, 1].max() + 10])
    ax.set_zlim([all_points[:, 2].min() - 10, all_points[:, 2].max() + 10])

    plt.show()


def generate_rectangle_points(center, width, height, num_points_per_side=8):
    """Generate points along the edges of a rectangle in 3D space."""
    """
    half_width = width / 2
    half_height = height / 2

    # Rectangle corners
    P1 = center + np.array([-half_width, -half_height, 0])
    P2 = center + np.array([half_width, -half_height, 0])
    P3 = center + np.array([half_width, half_height, 0])
    P4 = center + np.array([-half_width, half_height, 0])

    points = []
    for start, end in [(P1, P2), (P2, P3), (P3, P4), (P4, P1)]:
        edge_points = np.linspace(start, end, num_points_per_side + 1, endpoint=False)
        points.append(edge_points)

    points = np.vstack(points)
    points = np.vstack([points, [P1]])  # Add P1 at the end to close the loop
    """
    """Generate points along the edges of a rectangle in the x-z plane in 3D space."""
    
    half_width = width / 2
    half_height = height / 2

    # Rectangle corners in the x-z plane (y remains constant as center[1])
    P1 = center + np.array([-half_width, 0, -half_height])
    P2 = center + np.array([half_width, 0, -half_height])
    P3 = center + np.array([half_width, 0, half_height])
    P4 = center + np.array([-half_width, 0, half_height])

    points = []
    for start, end in [(P1, P2), (P2, P3), (P3, P4), (P4, P1)]:
        edge_points = np.linspace(start, end, num_points_per_side + 1, endpoint=False)
        points.append(edge_points)

    points = np.vstack(points)
    points = np.vstack([points, [P1]])  # Add P1 at the end to close the loop
    
    return points


def generate_circle_points(center, diameter, num_points):
    """Generate points uniformly distributed along the circumference of a circle in the x-z plane."""
    radius = diameter / 2
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = []

    for angle in angles:
        x = center[0] + radius * np.cos(angle)
        y = center[1]  # Keep y constant (in the x-z plane)
        z = center[2] + radius * np.sin(angle)
        points.append([x, y, z])
        
    # Append the first point again to close the shape
    points.append(points[0])

    return np.array(points)


def main():
    # Instantiate environment and TD3 agent
    env = ContinuumRobotEnv()
    agent = TD3(state_dim=env.state_dim, action_dim=env.action_dim, max_action=env.max_action)
    agent.load("td3_continuum_robot")

    print("Select Test Mode:")
    print("1. Single Point Test")
    print("2. Rectangle Test")
    print("3. Circle Test")
    choice = int(input("Enter your choice (1, 2, or 3): "))

    if choice == 1:
        num_tests = int(input("Enter the number of single point tests to perform: "))
        test_single_point(env, agent, num_tests)

    elif choice == 2:
        # center = env.random_target() / 1000  # Center of rectangle in meters
        center = np.array([-43.42, -135, 274.86])/1000
        width = 60.0 / 1000  # Width in meters
        height = 35.0 / 1000  # Height in meters

        rectangle_points = generate_rectangle_points(center, width, height, num_points_per_side=8)
        test_rectangle_combined(env, agent, rectangle_points)
    
    elif choice == 3:
        # Circle parameters
        center = np.array([-43.42, -118.2, 274.86]) / 1000
        diameter = 105.0 / 1000  # Diameter in meters
        num_points = 16  # Number of points along the circle

        circle_points = generate_circle_points(center, diameter, num_points)
        test_rectangle_combined(env, agent, circle_points)

    else:
        print("Invalid choice! Exiting.")


if __name__ == "__main__":
    main()