import numpy as np
import matplotlib.pyplot as plt

class ContinuumRobotEnv:
    # def __init__(self, num_segments=3, segment_length=0.5, action_range=(-1, 1), max_steps=25, goal_position=(1, 0.7)):
    #     self.num_segments = num_segments
    #     self.segment_length = segment_length
    #     self.action_range = action_range
    #     self.max_action = action_range[1]
    #     self.max_steps = max_steps
    #     self.current_step = 0
    #     self.goal_position = np.array(goal_position)
    #     self.state_dim = num_segments * 2
    #     self.action_dim = num_segments
    def __init__(self, segment_length=1.0, max_tendon_tension=1.0, num_segments=1, max_steps=25, max_action=1.0):
        self.segment_length = segment_length  # Length of each segment
        self.max_tendon_tension = max_tendon_tension  # Maximum tension in tendons
        self.num_segments = num_segments  # Number of segments
        self.max_steps = max_steps  # Maximum steps per episode
        self.max_action = max_action  # Maximum action value

        # Define the state and action dimensions
        self.state_dim = 2  # For 2D coordinates (x, y)
        self.action_dim = 3  # Three tendons

        # State of the robot (tip position)
        self.state = np.zeros(self.state_dim)

        # Target position (goal)
        self.target_position = np.array([0.5, 0.5])

        # Current step counter
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        initial_state = np.zeros(self.state_dim)
        return initial_state

    def step(self, actions):
        self.current_step += 1
        # Clip the action to the valid range
        actions = np.clip(actions, -self.max_tendon_tension, self.max_tendon_tension)
        # Scale the actions to the maximum allowed value
        actions = actions * self.max_action
        next_state = self._simulate_robot(actions)
        reward = self._compute_reward(next_state)
        done = self.current_step >= self.max_steps
        return next_state, reward, done

    def _simulate_robot(self, actions):
        # state = np.zeros(self.state_dim)
        # current_position = np.array([0.0, 0.0])
        # current_angle = 0.0

        # for i in range(self.num_segments):
        #     curvature = actions[i]
        #     if curvature != 0:
        #         radius = 1.0 / curvature
        #         angle = curvature * self.segment_length
        #         cx = current_position[0] - radius * np.sin(current_angle)
        #         cy = current_position[1] + radius * np.cos(current_angle)
        #         next_position = np.array([
        #             cx + radius * np.sin(current_angle + angle),
        #             cy - radius * np.cos(current_angle + angle)
        #         ])
        #         current_angle += angle
        #     else:
        #         next_position = current_position + self.segment_length * np.array([
        #             np.cos(current_angle),
        #             np.sin(current_angle)
        #         ])
        #     state[2 * i: 2 * i + 2] = next_position
        #     current_position = next_position

        # return state

        # Initialize the tip position and orientation
        tip_position = np.zeros(self.state_dim)
        orientation = 0.0

        for _ in range(self.num_segments):
            # Compute the curvature (kappa) using the PCC method for each segment
            kappa = np.sum(actions) / (3 * self.segment_length)
            # Compute the change in orientation (theta) for the segment
            theta = np.arctan2(np.sum(actions[1:] - actions[:-1]), self.segment_length)
            if kappa != 0:
                # Compute the new segment end position for non-zero curvature
                delta_x = (1 / kappa) * (np.sin(kappa * self.segment_length) * np.cos(orientation) - 
                                         (1 - np.cos(kappa * self.segment_length)) * np.sin(orientation))
                delta_y = (1 / kappa) * (np.sin(kappa * self.segment_length) * np.sin(orientation) + 
                                         (1 - np.cos(kappa * self.segment_length)) * np.cos(orientation))
            else:
                # For zero curvature, the segment end position is a straight line
                delta_x = self.segment_length * np.cos(orientation)
                delta_y = self.segment_length * np.sin(orientation)
            # Update the tip position and orientation
            tip_position += np.array([delta_x, delta_y])
            orientation += theta
        return tip_position

    def _compute_reward(self, state):
        # end_effector_position = state[-2:]
        # distance_to_goal = np.linalg.norm(end_effector_position - self.goal_position)
        # reward = -distance_to_goal
        # if distance_to_goal < 0.05:
        #     reward += 200
        # reward -= self.current_step * 0.001  # Small penalty per step, adjust the factor as needed
        # return reward
        # Compute the reward as the negative distance to the target position
        distance_to_target = np.linalg.norm(state - self.target_position)
        reward = -distance_to_target
        if distance_to_target < 0.05:
             reward += 200
        reward -= self.current_step * 0.001  # Small penalty per step, adjust the factor as needed
        return reward
    
    def distance(self, state):
        distance_to_goal = np.linalg.norm(state - self.target_position)
        return distance_to_goal
    
    def render(self, state, actions):
        # plt.figure()

        # # Plot the base position
        # plt.scatter(0, 0, color='black', label='Base Position')

        # # Plot the tip position
        # plt.scatter(state[0], state[1], color='red', label='Tip Position')

        # # Plot the target position
        # plt.scatter(self.target_position[0], self.target_position[1], color='green', label='Target Position')

        # # Plot the backbone curve of the robot
        # num_points_per_segment = 50  # Number of points to generate per segment
        # backbone_curve = []
        # for i in range(self.num_segments):
        #     # Generate points along the curve of the segment
        #     theta = np.linspace(0, np.pi / 6, num_points_per_segment)  # Example angle increment, adjust as needed
        #     x = np.cos(theta) * self.segment_length
        #     y = np.sin(theta) * self.segment_length * (i + 1)  # Adjust y-coordinate for each segment

        #     # Transform points to global coordinates and apply offset
        #     points = np.vstack([x, y]).T
        #     rotation_matrix = np.array([[np.cos(i * np.pi / 6), -np.sin(i * np.pi / 6)],
        #                                 [np.sin(i * np.pi / 6), np.cos(i * np.pi / 6)]])
        #     points = np.dot(points, rotation_matrix.T) - state[:2] + state[:2]  # Adjust offset
        #     backbone_curve.extend(points)

        # # Plot the curve of the backbone
        # plt.figure(figsize=(8, 6))
        # backbone_curve = np.array(backbone_curve)
        # plt.plot(backbone_curve[:, 0], backbone_curve[:, 1], color='blue')

        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.legend()
        # plt.title('Continuum Robot Visualization in 2D')
        # # Set the limits of the plot to include the origin
        # plt.xlim(-1.5, 1.5)
        # plt.ylim(-1.5, 1.5)
        # plt.show()

        # Calculate the positions of all segments
        segment_positions = [np.zeros(2)]
        orientation = 0.0

        for _ in range(self.num_segments):
            # Compute the curvature (kappa) using the PCC method for each segment
            kappa = np.sum(actions) / (3 * self.segment_length)
            # Compute the change in orientation (theta) for the segment
            theta = np.arctan2(np.sum(actions[1:] - actions[:-1]), self.segment_length)
            if kappa != 0:
                # Compute the new segment end position for non-zero curvature
                delta_x = (1 / kappa) * (np.sin(kappa * self.segment_length) * np.cos(orientation) - 
                                        (1 - np.cos(kappa * self.segment_length)) * np.sin(orientation))
                delta_y = (1 / kappa) * (np.sin(kappa * self.segment_length) * np.sin(orientation) + 
                                        (1 - np.cos(kappa * self.segment_length)) * np.cos(orientation))
            else:
                # For zero curvature, the segment end position is a straight line
                delta_x = self.segment_length * np.cos(orientation)
                delta_y = self.segment_length * np.sin(orientation)
            # Update the tip position and orientation
            segment_positions.append(segment_positions[-1] + np.array([delta_x, delta_y]))
            orientation += theta

        segment_positions = np.array(segment_positions)

        # Plot the robot segments
        plt.figure(figsize=(8, 6))
        for i in range(len(segment_positions) - 1):
            plt.plot([segment_positions[i][0], segment_positions[i+1][0]],
                    [segment_positions[i][1], segment_positions[i+1][1]],
                    color='blue')

        plt.title('Continuum Robot Visualization')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.axis('equal')
        plt.show()