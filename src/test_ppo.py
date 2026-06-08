import safety_gymnasium
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class CostWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info["cost"] = cost
        return obs, reward, terminated, truncated, info


# create env
env = DummyVecEnv([
    lambda: CostWrapper(
        safety_gymnasium.make(
            "SafetyRacecarButton2-v0",
            render_mode="human"
        )
    )
])

# load normalization statistics
env = VecNormalize.load("vec_normalize_700000_Lrate5e-5.pkl", env)

# IMPORTANT!
env.training = False
env.norm_reward = False

# load model
#model = PPO.load("ppo_300000")
model = PPO.load("ppo_700000_Lrate5e-5", env=env)

obs = env.reset()

done = False

while not done:

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, done, info = env.step(action)

    print(info)

env.close()