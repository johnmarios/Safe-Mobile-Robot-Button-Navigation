import gymnasium as gym
import numpy as np


class NormalizeActionWrapper(gym.ActionWrapper):
    """
    Agent action:
        action[0] in [-1, 1] -> forward speed in [min_speed, max_speed]
        action[1] in [-1, 1] -> steering in [-max_steering, max_steering]

    """

    def __init__(
        self,
        env,
    ):
        super().__init__(env)

        self.real_low = env.action_space.low.astype(np.float32)
        self.real_high = env.action_space.high.astype(np.float32)


        # self.min_speed = float(min_speed)
        # self.max_speed = float(max_speed)
        # self.max_steering = float(max_steering)

        # regulate what agent sees as action space, normalized to [-1, 1]

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32,
        )
    def normalize(self, x):
        # Normalize x to [-1, 1] based on the real action space
        return 2 * (x - self.real_low) / (self.real_high - self.real_low) - 1

    def denormalize(self, x):
        # Denormalize x from [-1, 1] to the real action space
        return self.real_low + (x + 1) * (self.real_high - self.real_low) / 2

    def action(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        real_action = np.zeros_like(action, dtype=np.float32)

        # BEFORE
        # action[0] = speed
        # -1 -> min_speed
        #  0 -> average speed
        #  1 -> max_speed

        # action[1] = steering

        real_action = self.denormalize(action)

        # for safety reasons : 

        real_action = np.clip(real_action, self.real_low, self.real_high)

        # AFTER
        #     # action[0] = speed

        # # -1 -> min_speed
        # #  0 -> average speed
        # #  1 -> max_speed
        # real_action[0] = self.min_speed + (action[0] + 1.0) * 0.5 * (self.max_speed - self.min_speed)

        # # action[1] = steering
        # # -1 -> -max_steering
        # #  0 -> 0
        # #  1 -> max_steering
        # if len(action) > 1:
        #     real_action[1] = self.max_steering * action[1]

        # # final safety clip to real env limits
        # real_action = np.clip(real_action, self.real_low, self.real_high)

        return real_action

