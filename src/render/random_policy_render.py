import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import safety_gymnasium
from evaluate.evaluate_random_policy import evaluate_random_policy
from config import ENV_ID, MAX_STEPS, NUM_EPISODES, SEED

RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "random_policy_results.json"


def run_random_policy():
    episode_summaries = []
    step_records = []

    env = safety_gymnasium.make(ENV_ID, render_mode="human")

    for episode in range(NUM_EPISODES):
        obs, info = env.reset(seed=SEED + episode)

        total_reward = 0.0
        total_cost = 0.0

        print(f"\nEpisode {episode + 1}/{NUM_EPISODES}")

        for step in range(MAX_STEPS):

            # random action
            action = env.action_space.sample()

            obs, reward, cost, terminated, truncated, info = env.step(action)

            total_reward += reward
            total_cost += cost

            env.render()

            step_records.append(
                {
                    "episode": episode + 1,
                    "step": step + 1,
                    "reward": float(reward),
                    "cost": float(cost),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )

            if terminated or truncated:
                print(f"Episode finished at step {step + 1}")
                break

        episode_summaries.append(
            {
                "episode": episode + 1,
                "steps": step + 1,
                "total_reward": float(total_reward),
                "total_cost": float(total_cost),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "seed": SEED + episode,
            }
        )

        print(f"Total reward: {total_reward:.2f}")
        print(f"Total cost: {total_cost:.2f}")

    env.close()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "environment": ENV_ID,
        "num_episodes": NUM_EPISODES,
        "max_steps": MAX_STEPS,
        "seed": SEED,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "episode_summaries": episode_summaries,
        "step_records": step_records,
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as file_handle:
        json.dump(results, file_handle, indent=2)

    print(f"Saved results to: {RESULTS_PATH}")

    evaluate_random_policy(render=True)
