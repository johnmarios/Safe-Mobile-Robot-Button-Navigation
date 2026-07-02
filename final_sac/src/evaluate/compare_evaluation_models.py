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
from evaluate.evaluate_sac_act_rep import evaluate_policy_act_rep
from evaluate.evaluate_sac_single_button import evaluate_policy_single_b


def compare_eval_models(MODELS,
                        ENV_NAME,
                        EVAL_EPISODES,
                        COST_WEIGHT,
                        ACTION_REPEAT,
                        SINGLE_BUTTON,
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
    max_action = torch.FloatTensor(
        env.action_space.high
    ).to(device)

    # Agent
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        discount=0.99,
        tau=0.005,
        actor_lr=1e-5,
        critic_lr=1e-5,
        entropy_multiplier=1.0
    )

    results = {}

    for model in MODELS:

        agent.load(f"models/{model}")

        if SINGLE_BUTTON is True:
            (
                avg_reward,
                avg_cost,
                avg_turn_penalty,
                success_rate,
                mean_success_steps,
                eval_rewards_history_r, 
                eval_costs_history_r, 
                eval_turn_penalty_h,
                mean_episode_length
                )= evaluate_policy_single_b(
                            agent,
                            TURNING_WEIGHT = 0,
                            MAX_STEPS = 500,
                            env_name=ENV_NAME,
                            eval_episodes=EVAL_EPISODES,
                            ACTION_REPEAT=ACTION_REPEAT
                        )

            results[model] = {
                "reward_mean": np.mean(eval_rewards_history_r),
                "reward_error": 1.96*np.std(eval_rewards_history_r)/np.sqrt(EVAL_EPISODES),

                "cost_mean": np.mean(eval_costs_history_r),
                "cost_error": 1.96*np.std(eval_costs_history_r)/np.sqrt(EVAL_EPISODES),

                "success_rate": success_rate,

                "success_steps": mean_success_steps,

                "episode_length": mean_episode_length
            }

            print(
                f"{model} | "
                f"Cost {np.mean(eval_costs_history_r):.2f} ± {np.std(eval_costs_history_r):.2f} | "
                f"success_rate {success_rate:.2f} | "
                f"success_steps {mean_success_steps:.2f} | "
                f"episode_length {mean_episode_length:.2f}"
            )
        else:

            avg_reward, avg_cost, reward_history, cost_history = evaluate_policy_act_rep(
                agent,
                env_name=ENV_NAME,
                eval_episodes= EVAL_EPISODES,
                ACTION_REPEAT=ACTION_REPEAT
            )

            score_history = (
                np.array(reward_history)
                - COST_WEIGHT * np.array(cost_history)
            )

            results[model] = {
                "reward_mean": np.mean(reward_history),
                "reward_error": 1.96*np.std(reward_history)/np.sqrt(EVAL_EPISODES),

                "cost_mean": np.mean(cost_history),
                "cost_error": 1.96*np.std(cost_history)/np.sqrt(EVAL_EPISODES),

                "score_mean": np.mean(score_history),
                "score_error": 1.96*np.std(score_history)/np.sqrt(EVAL_EPISODES)
            }

            print(
                f"{model} | "
                f"Reward {np.mean(reward_history):.2f} ± {np.std(reward_history):.2f} | "
                f"Cost {np.mean(cost_history):.2f} ± {np.std(cost_history):.2f} | "
                f"Score {np.mean(score_history):.2f} ± {np.std(score_history):.2f} | "
                f"Score error {results[model]['score_error']:.2f} "
            )
    names = list(results.keys())


    def plot_metric(metric,
                    ylabel,
                    output_name):

        mean = [
            results[m][f"{metric}_mean"]
            for m in names
        ]

        error = [
            results[m][f"{metric}_error"]
            for m in names
        ]

        plt.figure(figsize=(8,5))

        plt.bar(
            names,
            mean,
            yerr=error,
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

    def plot_simple(metric, ylabel, output_name):

        values = [results[m][metric] for m in names]

        plt.figure(figsize=(14,6))
        plt.bar(names, values)
        plt.xticks(rotation=45,ha="right")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.grid()
        plt.tight_layout()

        plt.savefig(SAVE_PATH / output_name)
        plt.close()

    if SINGLE_BUTTON:
        plot_metric("reward", "Reward", "reward_comparison.png")
        plot_metric("cost", "Cost", "cost_comparison.png")
   
        plot_simple("success_rate", "Success Rate", "success_rate.png")
        plot_simple("success_steps", "Mean Success Length", "success_length.png")
        plot_simple("episode_length", "Mean Episode Length", "episode_length.png")
    else:
        plot_metric("reward", "Reward", "reward_comparison.png")
        plot_metric("cost", "Cost", "cost_comparison.png")
        plot_metric("score", "Score", "score_comparison.png")
        

if __name__ == "__main__":
    MODELS = [
        "sac_single_B_ar2_c0toc05_em1_rbs3e5_st5e4_mt1e6_best",
        "sac_single_B_ar2_c0toc05_em1_rbs3e5_st5e4_mt1e6_latest",
        "sac_single_B_ar2_c05_em1_rbs3e5_st5e4_mt1e6_cont1_best",
        "sac_single_B_ar2_c05_em1_rbs3e5_st5e4_mt1e6_cont1_latest",
        "sac_single_B_ar2_c05_em1_rbs3e5_st5e4_mt1e6_cont1_v2_best",
        "sac_single_B_ar2_c05_em1_rbs3e5_st5e4_mt1e6_cont1_v2_latest"

    ]

    compare_eval_models(MODELS,
                        "SafetyRacecarButton2-v0",
                        EVAL_EPISODES = 100,
                        COST_WEIGHT = 0.05,
                        ACTION_REPEAT = 2,
                        SINGLE_BUTTON = True,
                        FOLDER_NAME = "evaluations_comparison_single_b_cost_05_models"
    )           