import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


MODELS = [
    "sac_phase_0",
    "sac_cost_001_phase_0",
    "sac_cost_003_phase_0"
]

ROOT = Path(__file__).resolve().parents[2]

BASE_PATH = ROOT / "results"
SAVE_PATH = BASE_PATH / "comparisons"

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


# ===== Evaluation metrics =====

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

print("Finished.")
