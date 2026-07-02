import numpy as np
import gymnasium as gym
import safety_gymnasium

from gymnasium.spaces import Box
from gymnasium.spaces.utils import flatten, flatten_space


EPISODE_STEPS = 1000


class RacecarTD3Env(gym.Env):
    """SafetyRacecarButton2 wrapper used by TD3.

    Observations are flattened and actions are normalized to [-1, 1].
    """

    def __init__(self, env, goal_termination=False):
        super().__init__()
        self.env = env
        self.goal_termination = goal_termination

        self.original_observation_space = env.observation_space
        # flattened observation space is used for TD3
        self.observation_space = flatten_space(env.observation_space)

        self.real_low = env.action_space.low.astype(np.float32)
        self.real_high = env.action_space.high.astype(np.float32)
        # normalized action space is used for TD3
        self.action_space = Box(-1.0, 1.0, shape=env.action_space.shape, dtype=np.float32)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def flatten_observation(self, observation):
        return np.asarray(flatten(self.original_observation_space, observation),dtype=np.float32,)

    def action(self, normalized_action):
        """Convert normalized action in [-1, 1] to real action in env.action_space"""
        normalized_action = np.clip(normalized_action, -1.0, 1.0)
        action = self.real_low + 0.5 * (normalized_action + 1.0) * (self.real_high - self.real_low)
        return action.astype(self.env.action_space.dtype)

    def reset(self, seed=None, options=None):
        """Reset the environment and return the flattened observation and info dictionary"""
        super().reset(seed=seed)
        observation, info = self.env.reset(seed=seed, options=options)
        return self.flatten_observation(observation), info

    def step(self, action):
        observation, reward, cost, terminated, truncated, info = self.env.step(self.action(action))

        info = dict(info)
        info["cost"] = float(cost)

        # when goal_termination is True, the episode ends when the goal is met
        if self.goal_termination and info.get("goal_met", False):
            terminated = True

        return (
            self.flatten_observation(observation),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def make_env(env_id, render_mode=None, goal_termination=False):
    env = safety_gymnasium.make(
        env_id,
        render_mode=render_mode,
        max_episode_steps=EPISODE_STEPS,
    )
    return RacecarTD3Env(env, goal_termination)
