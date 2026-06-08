import safety_gymnasium
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


# Wrapper to move cost into info
class CostWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)

        info["cost"] = cost

        return obs, reward, terminated, truncated, info


# Create a base env just to inspect spaces
base_env = safety_gymnasium.make("SafetyRacecarButton2-v0")

print("Action space:", base_env.action_space)
print("Observation space:", base_env.observation_space)

base_env.close()


# Create training environment
env = DummyVecEnv([
    lambda: Monitor(
        CostWrapper(
            safety_gymnasium.make("SafetyRacecarButton2-v0")
        ),
        info_keywords=("cost_buttons",)
    )
])

# Load normalization statistics from 300k checkpoint
# env = VecNormalize.load(
#     "vec_normalize_420000_Lrate1e4.pkl",
#     env
# )
env = VecNormalize.load(
    "vec_normalize_510000_Lrate1e4.pkl",
    env
)


# Continue updating normalization statistics
env.training = True
env.norm_reward = True


# Load trained model
# model = PPO.load(
#     "ppo_420000_lr1e4",
#     env=env
# )
model = PPO.load(
    "ppo_510000_Lrate1e4",
    env=env
)


# Lower learning rate
model.lr_schedule = lambda _: 1e-4


# Continue training in chunks of 30k timesteps
for i in range(15):

    model.learn(
        total_timesteps=30_000,
        reset_num_timesteps=False,
        tb_log_name="ppo_racecar_2nd_phase"
    )

    step = 510000 + (i + 1) * 30000

    model.save(f"ppo_{step}_Lrate1e4")
    env.save(f"vec_normalize_{step}_Lrate1e4.pkl")

    print(f"Checkpoint {step} saved")


# Save final checkpoint
model.save("ppo_continue_final")
env.save("vec_normalize_continue_final.pkl")