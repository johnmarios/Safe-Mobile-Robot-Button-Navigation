import safety_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym

# wrapper creation to include cost in the info dictionary
class CostWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)

        info["cost"] = cost

        return obs, reward, terminated, truncated, info


base_env = safety_gymnasium.make("SafetyRacecarButton2-v0")

print("Action space:", base_env.action_space)
print("Observation space:", base_env.observation_space)

env = DummyVecEnv([
    lambda: Monitor(
        CostWrapper(
            safety_gymnasium.make("SafetyRacecarButton2-v0")
        )
    )
])

env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=True
)

# Training
# PPO agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/"
)

# Training + checkpoints
for i in range(10):

    print(f"Starting block {(i+1)*100000}")

    model.learn(
        total_timesteps=100_000,
        reset_num_timesteps=False,
        tb_log_name="ppo_racecar"
    )

    model.save(f"ppo_{(i+1)*100000}")
    env.save(f"vec_normalize_{(i+1)*100000}.pkl")

    print(f"Checkpoint {(i+1)*100000} saved")