import numpy as np

class ExplorationNoise:
    def __init__(self, action_dim, max_action, initial_std=0.2, min_std=0.05, decay_rate=0.99):
        """
        Initializes dynamic noise for exploration.
        - action_dim: Dimension of the action space.
        - max_action: Maximum value for actions.
        - initial_std: Initial standard deviation of the noise.
        - min_std: Minimum standard deviation of the noise (target after decay).
        - decay_rate: Rate of noise reduction (a value close to 1 means slower decay).
        """
        self.action_dim = action_dim
        self.max_action = max_action
        self.std = initial_std
        self.min_std = min_std
        self.decay_rate = decay_rate

    def add_noise(self, action):
        """
        Adds noise to the action during exploration
        """
        noise = np.random.normal(0, self.std, size=self.action_dim)  # Generate noise
        noisy_action = np.clip(action + noise, 0, self.max_action)  # Add noise and clip the action
        return noisy_action

    def decay_noise(self):
        """
        Gradually reduces the noise level.
        """
        self.std = max(self.min_std, self.std * self.decay_rate)