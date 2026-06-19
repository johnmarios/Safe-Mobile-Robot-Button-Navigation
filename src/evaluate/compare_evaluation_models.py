import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import os
import numpy as np
import matplotlib.pyplot as plt
from train.sac import SAC
import torch
import safety_gymnasium
from evaluate.evaluate_sac import evaluate_policy


def compare_eval_models(MODELS,
                        ENV_NAME,
                        EVAL_EPISODES,
                        COST_WEIGHT,
                        FOLDER_NAME):
    

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # save path create
    ROOT = Path(__file__).resolve().parents[2]


    BASE_PATH = ROOT / "results"
    SAVE_PATH = BASE_PATH / FOLDER_NAME

    os.makedirs(SAVE_PATH, exist_ok=True)

    # Environment
    env = safety_gymnasium.make(ENV_NAME)

    # Dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Agent
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        discount=0.99,
        tau=0.005,
        actor_lr=1e-4,
        critic_lr=1e-4,
        entropy_multiplier=1.0
    )

    results = {}

    for model in MODELS:

        agent.load(f"models/{model}")

        avg_reward, avg_cost, reward_history, cost_history = evaluate_policy(
            agent,
            env_name=ENV_NAME,
            eval_episodes= EVAL_EPISODES
        )

        score_history = (
            np.array(reward_history)
            - COST_WEIGHT * np.array(cost_history)
        )

        results[model] = {
            "reward_mean": np.mean(reward_history),
            "reward_std": np.std(reward_history),

            "cost_mean": np.mean(cost_history),
            "cost_std": np.std(cost_history),

            "score_mean": np.mean(score_history),
            "score_std": np.std(score_history)
        }

        print(
            f"{model} | "
            f"Reward {np.mean(reward_history):.2f} ± {np.std(reward_history):.2f} | "
            f"Cost {np.mean(cost_history):.2f} ± {np.std(cost_history):.2f} | "
            f"Score {np.mean(score_history):.2f} ± {np.std(score_history):.2f}"
        )
    names = list(results.keys())


    def plot_metric(metric,
                    ylabel,
                    output_name):

        mean = [
            results[m][f"{metric}_mean"]
            for m in names
        ]

        std = [
            results[m][f"{metric}_std"]
            for m in names
        ]

        plt.figure(figsize=(8,5))

        plt.bar(
            names,
            mean,
            yerr=std,
            capsize=5
        )

        plt.ylabel(ylabel)
        plt.title(f"{ylabel} ({EVAL_EPISODES} episodes)")
        plt.grid()
        plt.tight_layout()

        plt.savefig(
            SAVE_PATH / output_name
        )

        plt.close()

    plot_metric("reward", "Reward", "reward_comparison.png")
    plot_metric("cost", "Cost", "cost_comparison.png")
    plot_metric("score", "Score", "score_comparison.png")
    env.close()

if __name__ == "__main__":
    MODELS = [
        "sac_cost_002_entropy_01_best",
        "sac_cost_002_entropy_1_best",
        "sac_cost_002_entropy_10_best"
    ]

    compare_eval_models(MODELS,
                        "SafetyRacecarButton2-v0",
                        EVAL_EPISODES = 10,
                        COST_WEIGHT = 0.002,
                        FOLDER_NAME = "evaluations_comparison_cost_002_entropy_01_to_10"
    )           