import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from config import ENV_ID, MAX_STEPS, NUM_EPISODES, SEED

RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "random_policy_results.json"


def evaluate_random_policy(render=True):

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_PATH}. Run src/render/random_policy_render.py first."
        )

    with RESULTS_PATH.open("r", encoding="utf-8") as file_handle:
        results = json.load(file_handle)

    episode_summaries = results.get("episode_summaries", [])
    step_records = results.get("step_records", [])

    episode_rewards = np.array([episode["total_reward"] for episode in episode_summaries], dtype=float)
    episode_costs = np.array([episode["total_cost"] for episode in episode_summaries], dtype=float)
    episode_lengths = np.array([episode["steps"] for episode in episode_summaries], dtype=float)
    step_costs = np.array([step["cost"] for step in step_records], dtype=float)
    step_rewards = np.array([step["reward"] for step in step_records], dtype=float)

    cumulative_cost = np.cumsum(step_costs)
    time_axis = np.arange(1, len(step_costs) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(time_axis, cumulative_cost, label="Cumulative cost", linewidth=2)
    plt.plot(time_axis, step_costs, label="Step cost", alpha=0.5)
    plt.xlabel("Step")
    plt.ylabel("Cost")
    plt.title("Cost Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n==============================")
    print("Random Policy Evaluation")
    print("==============================")
    print(f"Environment: {results.get('environment', ENV_ID)}")
    print(f"Results file: {RESULTS_PATH}")
    print(f"Episodes: {len(episode_summaries)}")
    print(f"Total steps: {len(step_records)}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f}")
    print(f"Std reward: {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.2f}")
    print(f"Mean episode cost: {np.mean(episode_costs):.2f}")
    print(f"Total cost: {np.sum(episode_costs):.2f}")
    print(f"Mean step cost: {np.mean(step_costs):.4f}")
    print(f"Max step cost: {np.max(step_costs):.4f}")
    print(f"Mean step reward: {np.mean(step_rewards):.4f}")
    print(f"Steps with nonzero cost: {np.count_nonzero(step_costs > 0)}")

    if render:
        print("Plot rendered successfully.")


if __name__ == "__main__":
    evaluate_random_policy(render=True)