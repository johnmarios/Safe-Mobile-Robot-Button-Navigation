import os

import gymnasium as gym
import gymnasium_robotics

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from config import ENV_ID, SEED, SAC_TOTAL_TIMESTEPS, SAC_MODEL_PATH, SAC_LOG_DIR


def train_sac():
    gymnasium_robotics.register_robotics_envs()

    os.makedirs("models", exist_ok=True)
    os.makedirs(SAC_LOG_DIR, exist_ok=True)

    env = gym.make(ENV_ID)
    env = Monitor(env)

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=SEED,
        tensorboard_log=SAC_LOG_DIR,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        device="auto",
    )

    print("\n==============================")
    print("Training SAC")
    print("==============================")
    print(f"Environment: {ENV_ID}")
    print(f"Total timesteps: {SAC_TOTAL_TIMESTEPS}")

    model.learn(
        total_timesteps=SAC_TOTAL_TIMESTEPS,
        log_interval=10,
        progress_bar=True,
    )

    model.save(SAC_MODEL_PATH)

    print("\nSAC model saved at:")
    print(f"{SAC_MODEL_PATH}.zip")

    env.close()


if __name__ == "__main__":
    train_sac()