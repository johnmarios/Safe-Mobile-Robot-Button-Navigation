import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import safety_gymnasium

from controllers.common import ENV_ID
from controllers.factory import create_controller


MODELS_FILE = Path("/app/models/comparison_models.json")
OUTPUT_DIR = Path("/app/outputs/comparison_results")

SEED = 12345
MAX_STEPS = 1000


def load_models():
    with open(MODELS_FILE, encoding="utf-8") as file:
        return json.load(file)["models"]


def evaluate_model(model, episodes):
    model_file = model.get("model")
    model_path = MODELS_FILE.parent / model_file if model_file else None

    controller = create_controller(
        model["type"],
        model_path,
        model.get("action_repeat"),
    )

    env = safety_gymnasium.make(ENV_ID, max_episode_steps=MAX_STEPS)

    rewards = []
    costs = []

    for episode in range(episodes):
        observation, _ = env.reset(seed=SEED + episode)
        controller.reset(seed=SEED + episode)

        total_reward = 0.0
        total_cost = 0.0
        steps = 0
        terminated = False
        truncated = False
        goal_met = False

        while not (terminated or truncated):
            action, _ = controller.act(observation)

            observation, reward, cost, terminated, truncated, info = env.step(
                action
            )

            total_reward += reward
            total_cost += cost
            steps += 1
            goal_met = goal_met or info.get("goal_met", False)

        rewards.append(total_reward)
        costs.append(total_cost)

        print(
            f"{model['name']} | {episode + 1}/{episodes} | "
            f"reward={total_reward:.2f} | cost={total_cost:.2f} "
        )

    env.close()

    return {
        "rewards": rewards,
        "costs": costs,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
    }


def save_chart(filename, names, values, errors, ylabel):
    plt.figure(figsize=(9, 5))
    plt.bar(names, values, yerr=errors, capsize=4)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close()


def save_results(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "raw_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    with open(OUTPUT_DIR / "summary.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "name",
            "mean_reward",
            "std_reward",
            "mean_cost",
            "std_cost",
        ])

        for name, result in results.items():
            writer.writerow([
                name,
                result["mean_reward"],
                result["std_reward"],
                result["mean_cost"],
                result["std_cost"],
            ])

    names = list(results)

    save_chart(
        "mean_reward.png",
        names,
        [results[name]["mean_reward"] for name in names],
        [results[name]["std_reward"] for name in names],
        "Mean reward",
    )
    save_chart(
        "mean_cost.png",
        names,
        [results[name]["mean_cost"] for name in names],
        [results[name]["std_cost"] for name in names],
        "Mean cost",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=int)
    args = parser.parse_args()

    results = {}

    for model in load_models():
        print("=" * 60)
        print(f"Evaluating {model['name']}")
        print("=" * 60)

        results[model["name"]] = evaluate_model(
            model,
            args.episodes,
        )

    save_results(results)

    print("\n===== RESULTS =====")
    for name, result in results.items():
        print(
            f"{name}: reward={result['mean_reward']:.2f} ± "
            f"{result['std_reward']:.2f} | "
            f"cost={result['mean_cost']:.2f} ± "
            f"{result['std_cost']:.2f} | "
        )


if __name__ == "__main__":
    main()
