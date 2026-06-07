import gymnasium as gym
import gymnasium_robotics

from stable_baselines3 import SAC
import safety_gymnasium

from config import ENV_ID, SAC_MODEL_PATH, MAX_STEPS, NUM_EPISODES, SEED

from evaluate.evaluate_sac import evaluate_sac
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "sac_results.json"

def render_sac():
    env = safety_gymnasium.make(ENV_ID, render_mode="human")

    model = SAC.load(SAC_MODEL_PATH, env=env)

    print("\n==============================")
    print("Rendering SAC Agent")
    print("==============================")
    print(f"Environment: {ENV_ID}")

    for episode in range(NUM_EPISODES):
        obs, info = env.reset(seed=SEED + episode)

        total_reward = 0.0

        print(f"\nEpisode {episode + 1}/{NUM_EPISODES}")

        for step in range(MAX_STEPS):
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            env.render()

            if terminated or truncated:
                print(f"Episode finished at step {step + 1}")
                break

        print(f"Total reward: {total_reward:.2f}")

    env.close()
    evaluate_sac(render=True)
