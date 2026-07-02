import numpy as np
import safety_gymnasium

from gymnasium.spaces.utils import flatten, flatten_space


ENV_ID = "SafetyRacecarButton2-v0"


class EnvironmentInfo:
    def __init__(self):
        env = safety_gymnasium.make(ENV_ID)

        self.observation_space = env.observation_space
        self.flat_observation_space = flatten_space(env.observation_space)
        self.action_space = env.action_space

        self.state_dim = int(np.prod(self.flat_observation_space.shape))
        self.action_dim = int(np.prod(self.action_space.shape))
        self.action_low = np.asarray(self.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(self.action_space.high, dtype=np.float32)

        env.close()


def flatten_observation(observation, observation_space):
    if isinstance(observation, dict):
        return flatten(observation_space, observation).astype(np.float32)

    return np.asarray(observation, dtype=np.float32).reshape(-1)


def denormalize_action(normalized_action, action_space):
    normalized_action = np.asarray(normalized_action, dtype=np.float32)
    normalized_action = np.clip(normalized_action, -1.0, 1.0)

    low = action_space.low
    high = action_space.high

    return low + 0.5 * (normalized_action + 1.0) * (high - low)
