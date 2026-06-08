import safety_gymnasium
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


# ---------------- PARAMETERS ----------------

START_CHECKPOINT = 480000
LEARNING_RATE = 5e-5
CHECKPOINT_INTERVAL = 20000
N_CHECKPOINTS = 11      # μέχρι 700k

SUFFIX = "_Lrate5e-5"

# --------------------------------------------


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


# Load normalization statistics corresponding to 480k
env = VecNormalize.load(
    "vec_normalize_480000_Lrate5e-5.pkl",
    env
)

env.training = True
env.norm_reward = True


# Load 480k model
model = PPO.load(
    "ppo_480000_Lrate5e-5",
    env=env
)


# Change learning rate
model.learning_rate = LEARNING_RATE
model.lr_schedule = lambda _: LEARNING_RATE

# Update optimizer LR
for param_group in model.policy.optimizer.param_groups:
    param_group["lr"] = LEARNING_RATE


# Continue training and save checkpoints
for i in range(N_CHECKPOINTS):

    model.learn(
        total_timesteps=CHECKPOINT_INTERVAL,
        reset_num_timesteps=False,
        tb_log_name="ppo_racecar_3rd_phase"
    )

    step = START_CHECKPOINT + (i + 1) * CHECKPOINT_INTERVAL

    model.save(
        f"ppo_{step}{SUFFIX}"
    )

    env.save(
        f"vec_normalize_{step}{SUFFIX}.pkl"
    )

    print(f"Checkpoint {step} saved")


# Save final model
model.save(
    f"ppo_final{SUFFIX}"
)

env.save(
    f"vec_normalize_final{SUFFIX}.pkl"
)

env.close()