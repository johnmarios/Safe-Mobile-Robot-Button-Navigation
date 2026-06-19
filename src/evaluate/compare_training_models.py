import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


MODELS = [
    "sac_cost_002_entropy_01",
    "sac_cost_002_entropy_1",
    "sac_cost_002_entropy_10"
]

ROOT = Path(__file__).resolve().parents[2]

BASE_PATH = ROOT / "results"
SAVE_PATH = BASE_PATH / "training_comparisons"

os.makedirs(SAVE_PATH, exist_ok=True)

def plot_metric(filename,
                ylabel,
                title,
                output_name,
                use_training_steps=False):

    plt.figure(figsize=(8,5))

    for model in MODELS:

        model_path = BASE_PATH / model

        file_path = model_path / f"{filename}.npy"

        if not os.path.exists(file_path):
            print(f"{file_path} not found")
            continue

        y = np.load(file_path)

        if use_training_steps:
            x = np.load(
                model_path / "training_steps.npy"
            )
        else:
            x = np.arange(len(y))

        plt.plot(
            x,
            y,
            linewidth=2,
            label=model
        )

    plt.xlabel(
        "Training step"
        if use_training_steps
        else "Evaluation"
    )

    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            SAVE_PATH,
            output_name
        )
    )

    plt.close()


def plot_metric_with_std(mean_file,
                         history_file,
                         ylabel,
                         title,
                         output_name):

    plt.figure(figsize=(8,5))

    for model in MODELS:

        model_path = BASE_PATH / model

        mean_path = model_path / f"{mean_file}.npy"
        history_path = model_path / f"{history_file}.npy"

        if (
            not mean_path.exists()
            or not history_path.exists()
        ):
            print(f"{model}: missing files")
            continue

        mean = np.load(mean_path)
        history = np.load(history_path)

        std = np.std(history, axis=1)

        x = np.arange(len(mean))

        plt.plot(
            x,
            mean,
            linewidth=2,
            label=model
        )

        plt.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.2
        )

    plt.xlabel("Evaluation")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        SAVE_PATH / output_name
    )

    plt.close()

# ===== Training evaluation metrics =====

plot_metric(
    "eval_rewards",
    "Average reward",
    "Reward comparison",
    "reward_comparison.png"
)

plot_metric(
    "eval_costs",
    "Average cost",
    "Cost comparison",
    "cost_comparison.png"
)

plot_metric(
    "score_history",
    "Score",
    "Score comparison",
    "score_comparison.png"
)

plot_metric_with_std(
    "eval_rewards",
    "eval_rewards_history",
    "Average reward",
    "Reward comparison with std",
    "reward_std_comparison.png"
)

plot_metric_with_std(
    "eval_costs",
    "eval_costs_history",
    "Average cost",
    "Cost comparison with std",
    "cost_std_comparison.png"
)

plot_metric_with_std(
    "score_history",
    "score_history_eval",
    "Score",
    "Score comparison with std",
    "score_std_comparison.png"
)


# ===== Training metrics =====

plot_metric(
    "alpha_history",
    "Alpha",
    "Alpha comparison",
    "alpha_comparison.png",
    use_training_steps=True
)

plot_metric(
    "critic_loss_history",
    "Critic loss",
    "Critic loss comparison",
    "critic_loss_comparison.png",
    use_training_steps=True
)

plot_metric(
    "actor_loss_history",
    "Actor loss",
    "Actor loss comparison",
    "actor_loss_comparison.png",
    use_training_steps=True
)

#==========one time plot score std without loaded score history eval=============
plt.figure(figsize=(8,5))
COST_WEIGHT = 0.002
for model in MODELS:

    model_path = BASE_PATH / model

    score_mean = np.load(
        model_path / "score_history.npy"
    )

    reward_history = np.load(
        model_path / "eval_rewards_history.npy"
    )

    cost_history = np.load(
        model_path / "eval_costs_history.npy"
    )

    score_history_eval = (
        reward_history
        - COST_WEIGHT * cost_history
    )

    score_std = np.std(
        score_history_eval,
        axis=1
    )

    x = np.arange(len(score_mean))

    plt.plot(
        x,
        score_mean,
        linewidth=2,
        label=model
    )

    plt.fill_between(
        x,
        score_mean - score_std,
        score_mean + score_std,
        alpha=0.2
    )

plt.xlabel("Evaluation")
plt.ylabel("Score")
plt.title("Score comparison with std")
plt.grid()
plt.legend()
plt.tight_layout()

plt.savefig(
    SAVE_PATH / "score_std_comparison.png"
)

plt.close()

print("Finished.")
