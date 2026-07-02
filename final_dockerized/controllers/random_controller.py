import numpy as np

from .common import EnvironmentInfo


class Controller:
    def __init__(self):
        self.info = EnvironmentInfo()
        self.reset()

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, observation):
        action = self.rng.uniform(self.info.action_low, self.info.action_high,).astype(np.float32)
        return action, {"controller": "random"}
