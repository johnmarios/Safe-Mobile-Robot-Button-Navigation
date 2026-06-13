import gymnasium as gym
import numpy as np


class NormalizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.real_low = env.action_space.low
        self.real_high = env.action_space.high

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32,
        )

    def action(self, action):
        action = np.clip(action, -1.0, 1.0)

        real_action = self.real_low + (action + 1.0) * 0.5 * (
            self.real_high - self.real_low
        )

        return real_action
    
    
def sample_warmup_action(env, action_dim):
    action = np.zeros(action_dim, dtype=np.float32)

    if action_dim >= 2:
        # action[0]: forward/backward normalized throttle
        # Θετική τιμή για να έχει τάση να πηγαίνει μπροστά
        action[0] = np.random.uniform(0.2, 0.8)

        # action[1]: steering
        # Μικρή τυχαία στροφή, όχι τέρμα τιμόνι
        action[1] = np.random.uniform(-0.25, 0.25)
    else:
        action = env.action_space.sample()

    return np.clip(action, env.action_space.low, env.action_space.high)