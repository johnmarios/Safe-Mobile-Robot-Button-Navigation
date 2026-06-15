# evaluate_td3.py

import os
import sys

import numpy as np
from Project.src.train.normalize import NormalizeActionWrapper
import safety_gymnasium


src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from train.td3 import TD3
from train.normalize import NormalizeActionWrapper


ENV_ID = "SafetyRacecarButton2-v0"
MODEL_PATH = "models/td3_safety_racecar_button2"

EVAL_EPISODES = 10
MAX_STEPS = 1000
SEED = 42


def evaluate_td3():
    env = safety_gymnasium.make(ENV_ID,render_mode=None)
    # env = NormalizeActionWrapper(env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action
    )

    agent.load(MODEL_PATH)

    episode_rewards = []
    episode_costs = []
    episode_lengths = []

    print("\n==============================")
    print("Evaluating TD3")
    print("==============================")
    print(f"Environment: {ENV_ID}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Episodes: {EVAL_EPISODES}")

    for episode in range(EVAL_EPISODES):
        state, info = env.reset(seed=SEED + episode)

        episode_reward = 0.0
        episode_cost = 0.0
        episode_steps = 0

        for step in range(MAX_STEPS):
            # Evaluation: deterministic action, no exploration noise
            action = agent.select_action(state)

            next_state, reward, cost, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            episode_reward += reward
            episode_cost += cost
            episode_steps += 1

            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_costs.append(episode_cost)
        episode_lengths.append(episode_steps)

        print(
            f"Episode {episode + 1}: "
            f"reward={episode_reward:.2f}, "
            f"cost={episode_cost:.2f}, "
            f"steps={episode_steps}"
        )

    env.close()

    print("\n==============================")
    print("TD3 Evaluation Results")
    print("==============================")
    print(f"Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"Std reward: {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Mean cost: {np.mean(episode_costs):.2f}")
    print(f"Std cost: {np.std(episode_costs):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.2f}")


if __name__ == "__main__":
    evaluate_td3()