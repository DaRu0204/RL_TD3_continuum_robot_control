import numpy as np
import matplotlib.pyplot as plt

class ContinuumRobotEnv:
    def __init__(self, segment_length=0.1, max_tendon_tension=1, num_segments=1, num_tendons=3, max_steps=25, max_action=0.5):
        self.segment_length = segment_length
        self.max_tendon_tension = max_tendon_tension
        self.num_segments = num_segments
        self.max_steps = max_steps
        self.max_action = max_action

        self.state_dim = 2
        self.action_dim = num_segments * num_tendons

        self.state = np.zeros(self.state_dim)
        self.target_position = np.array([0.075, 0.04])
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        initial_state = np.zeros(self.state_dim)
        self.state = initial_state
        return initial_state
    
    def step(self, actions):
        self.current_step += 1
        actions = np.clip(actions, -self.max_tendon_tension, self.max_tendon_tension)
        next_state = self._simulate_robot(actions)
        reward = self._compute_reward(next_state)
        done = self.current_step >= self.max_steps
        self.state = next_state  # Update the environment's state
        return next_state, reward, done

    def _simulate_robot(self, actions):
        start_pos = np.array([0.0, 0.0])
        state = np.zeros(self.state_dim)
        orientation = 0.0

        for i in range(self.num_segments):
            kappa = actions[i] / self.segment_length
            
            if kappa != 0:
                delta_theta = kappa * self.segment_length
                r = 1 / kappa
                cx = start_pos[0] - r * np.sin(orientation)
                cy = start_pos[1] + r * np.cos(orientation)
                start_angle = orientation
                end_angle = orientation + delta_theta
                orientation = end_angle

                start_pos = np.array([
                    cx + r * np.sin(end_angle),
                    cy - r * np.cos(end_angle)
                ])
            else:
                delta_x = self.segment_length * np.cos(orientation)
                delta_y = self.segment_length * np.sin(orientation)
                start_pos += np.array([delta_x, delta_y])

        state = start_pos
        return state

    def _compute_reward(self, state):
        distance = np.linalg.norm(state - self.target_position)
        reward = -distance
        return reward

    def render(self, actions=None):
        segment_positions = [np.zeros(2)]
        orientation = 0.0

        if actions is not None:
            actions = np.clip(actions, -self.max_tendon_tension, self.max_tendon_tension)

        for i in range(self.num_segments):
            kappa = actions[i] / self.segment_length if actions is not None else 0
            start_pos = segment_positions[-1]

            if kappa != 0:
                delta_theta = kappa * self.segment_length
                r = 1 / kappa
                cx = start_pos[0] - r * np.sin(orientation)
                cy = start_pos[1] + r * np.cos(orientation)
                start_angle = orientation
                end_angle = orientation + delta_theta
                angles = np.linspace(start_angle, end_angle, 100)
                x_arc = cx + r * np.sin(angles)
                y_arc = cy - r * np.cos(angles)
                segment_positions.extend(np.column_stack((x_arc, y_arc)))
                orientation = end_angle
            else:
                delta_x = self.segment_length * np.cos(orientation)
                delta_y = self.segment_length * np.sin(orientation)
                segment_positions.append(start_pos + np.array([delta_x, delta_y]))

        segment_positions = np.array(segment_positions)
        print("segment_positions:", segment_positions)
        plt.figure(figsize=(8, 6))
        plt.plot(segment_positions[:, 0], segment_positions[:, 1], color='blue', label='Robot Curve')
        plt.scatter(0, 0, color='black', label='Base Position')
        plt.scatter(segment_positions[-1, 0], segment_positions[-1, 1], color='red', label='Tip Position')
        plt.scatter(self.target_position[0], self.target_position[1], color='green', label='Target Position')
        plt.title('Continuum Robot Visualization')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.show()
    
    def distance(self, state):
        distance_to_goal = np.linalg.norm(state - self.target_position)
        return distance_to_goal