import gymnasium as gym
import gymnasium_robotics
import numpy as np

from stable_baselines3 import SAC
import safety_gymnasium

from config import ENV_ID, SAC_MODEL_PATH, SAC_EVAL_EPISODES, MAX_STEPS, SEED


def evaluate_sac(render=False):
    gymnasium_robotics.register_robotics_envs()

    render_mode = "human" if render else None

    env = safety_gymnasium.make(ENV_ID, render_mode)

    model = SAC.load(SAC_MODEL_PATH, env=env)

    episode_rewards = []
    episode_lengths = []

    print("\n==============================")
    print("Evaluating SAC")
    print("==============================")
    print(f"Environment: {ENV_ID}")
    print(f"Episodes: {SAC_EVAL_EPISODES}")

    for episode in range(SAC_EVAL_EPISODES):
        obs, info = env.reset(seed=SEED + episode)

        total_reward = 0.0
        steps = 0

        for step in range(MAX_STEPS):
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            if render:
                env.render()

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        print(f"Episode {episode + 1}: reward={total_reward:.2f}, length={steps}")

    env.close()

    print("\n==============================")
    print("SAC Evaluation Results")
    print("==============================")
    print(f"Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"Std reward: {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.2f}")


if __name__ == "__main__":
    evaluate_sac(render=False)