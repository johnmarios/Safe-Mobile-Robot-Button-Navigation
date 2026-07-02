import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import config
from environment import make_env
from normalizer import MixedObservationNormalizer
from replay_buffer import ReplayBuffer
from td3 import DEVICE, TD3, load_torch_file


RESULTS_DIR = Path("results") / config.RUN_NAME
RUNS_DIR = Path("runs") / config.RUN_NAME
BEST_MODEL_PATH = RESULTS_DIR / "best_model.pt"
LAST_MODEL_PATH = RESULTS_DIR / "last_model.pt"
FULL_MODEL_PATH = RESULTS_DIR / "latest_full.pt"


def set_seed(seed):
    """Set the random seed for reproducibility """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(agent, env, normalizer, episodes, seed):
    """Evaluate the agent for a number of episodes and return the mean reward and cost."""
    rewards = []
    costs = []
    lengths = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0
        total_cost = 0.0
        steps = 0

        while not done:
            action = agent.select_action(normalizer.normalize(state))

            for _ in range(config.ACTION_REPEAT):
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                total_reward += reward
                total_cost += info["cost"]
                steps += 1

                if done:
                    break

        rewards.append(total_reward)
        costs.append(total_cost)
        lengths.append(steps)

    mean_reward = float(np.mean(rewards))
    mean_cost = float(np.mean(costs))

    return {
        "mean_reward": mean_reward,
        "std_reward": float(np.std(rewards)),
        "mean_cost": mean_cost,
        "std_cost": float(np.std(costs)),
        "mean_length": float(np.mean(lengths)),
        "score": mean_reward - config.BEST_COST_WEIGHT * mean_cost,
    }


def policy_checkpoint(agent, normalizer, step, evaluation=None):
    """Return a dictionary containing the agent's neural network weights, 
    the normalizer state, and the current step for saving a policy checkpoint """

    saved = {
        "env_id": config.ENV_ID,
        "saved_at_step": step,
        "agent": agent.network_state_dict(),
        "observation_normalizer": normalizer.state_dict(),
    }

    if evaluation is not None:
        saved["evaluation"] = evaluation

    return saved


def save_policy(path, agent, normalizer, step, evaluation=None):
    torch.save(policy_checkpoint(agent, normalizer, step, evaluation), path)


def save_full_checkpoint(step, best_score, agent, replay_buffer, normalizer, episode_rows, evaluation_rows):
    torch.save(
        {
            "step": step,
            "best_score": best_score,
            "agent": agent.state_dict(),
            "replay_buffer": replay_buffer.state_dict(),
            "observation_normalizer": normalizer.state_dict(),
            "episode_rows": episode_rows,
            "evaluation_rows": evaluation_rows,
        },
        FULL_MODEL_PATH,
    )


def save_csv(path, rows, columns):

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def moving_average(values, window=20):
    """Average the last 20 training episodes only for a clearer plot."""
    window = min(window, len(values))
    return np.convolve(values, np.ones(window) / window, mode="valid")


def save_curves(episode_rows, evaluation_rows):
    """Save learning curves for training and evaluation """
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    if episode_rows:
        steps = np.asarray([row["step"] for row in episode_rows])
        rewards = np.asarray([row["reward"] for row in episode_rows])
        costs = np.asarray([row["cost"] for row in episode_rows])

        axes[0].plot(steps, rewards, alpha=0.25, label="Training episode reward")
        axes[1].plot(steps, costs, alpha=0.25, label="Training episode cost")

        smooth_rewards = moving_average(rewards, window=20)
        smooth_costs = moving_average(costs, window=20)
        smooth_steps = steps[len(steps) - len(smooth_rewards):]

        axes[0].plot(
            smooth_steps,
            smooth_rewards,
            label="Training reward (20-episode mean)",
        )
        axes[1].plot(
            smooth_steps,
            smooth_costs,
            label="Training cost (20-episode mean)",
        )

    if evaluation_rows:
        steps = np.asarray([row["step"] for row in evaluation_rows])
        rewards = np.asarray([row["mean_reward"] for row in evaluation_rows])
        reward_stds = np.asarray([row["std_reward"] for row in evaluation_rows])
        costs = np.asarray([row["mean_cost"] for row in evaluation_rows])
        cost_stds = np.asarray([row["std_cost"] for row in evaluation_rows])

        axes[0].plot(
            steps,
            rewards,
            marker="o",
            label=f"Evaluation reward (mean of {config.EVALUATION_EPISODES} episodes)",
        )
        axes[0].fill_between(
            steps,
            rewards - reward_stds,
            rewards + reward_stds,
            alpha=0.18,
            label="Evaluation ± 1 std",
        )

        axes[1].plot(
            steps,
            costs,
            marker="o",
            label=f"Evaluation cost (mean of {config.EVALUATION_EPISODES} episodes)",
        )
        axes[1].fill_between(
            steps,
            costs - cost_stds,
            costs + cost_stds,
            alpha=0.18,
            label="Evaluation ± 1 std",
        )

    axes[0].set_ylabel("Raw environment reward")
    axes[0].set_title("TD3 learning curves")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_xlabel("Environment steps")
    axes[1].set_ylabel("Raw safety cost")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "learning_curves.png", dpi=180)
    plt.close(figure)


def save_results(episode_rows, evaluation_rows):
    save_csv(
        RESULTS_DIR / "training_episodes.csv",
        episode_rows,
        ["step", "episode", "reward", "learning_reward", "cost", "length"],
    )
    save_csv(
        RESULTS_DIR / "evaluations.csv",
        evaluation_rows,
        ["step", "mean_reward", "std_reward", "mean_cost", "std_cost", "mean_length", "score"],
    )
    save_curves(episode_rows, evaluation_rows)


def load_branch(agent, normalizer):
    path = Path("results") / config.SOURCE_RUN_NAME / config.SOURCE_MODEL_NAME
    saved = load_torch_file(path)
    agent.load_networks(saved["agent"])
    normalizer.load_state_dict(saved["observation_normalizer"])


def load_resume(agent, replay_buffer, normalizer):
    saved = load_torch_file(FULL_MODEL_PATH)
    agent.load_state_dict(saved["agent"])
    replay_buffer.load_state_dict(saved["replay_buffer"])
    normalizer.load_state_dict(saved["observation_normalizer"])
    return saved


def choose_action(agent, normalizer, env, state, step):
    if config.LOAD_MODE == "new" and step < config.RANDOM_STEPS:
        # warmup phase 
        return env.action_space.sample()

    action = agent.select_action(normalizer.normalize(state))
    action += np.random.normal(0.0, config.EXPLORATION_NOISE, size=action.shape)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def main():
    set_seed(config.SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    train_env = make_env(config.ENV_ID, goal_termination=config.GOAL_TERMINATION)
    eval_env = make_env(config.ENV_ID, goal_termination=config.GOAL_TERMINATION)

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]

    agent = TD3(
        state_dim,
        action_dim,
        config.LEARNING_RATE,
        config.DISCOUNT,
        config.TAU,
        config.POLICY_NOISE,
        config.NOISE_CLIP,
        config.POLICY_DELAY,
    )
    replay_buffer = ReplayBuffer(state_dim, action_dim, config.BUFFER_SIZE)
    normalizer = MixedObservationNormalizer(train_env.observation_space, config.NORMALIZER_CLIP)

    # TensorBoard writer for logging training metrics
    writer = SummaryWriter(RUNS_DIR)

    start_step = 0
    best_score = -np.inf
    episode_rows = []
    evaluation_rows = []

    if config.LOAD_MODE == "branch":
        load_branch(agent, normalizer)
        print(f"Loaded branch: results/{config.SOURCE_RUN_NAME}/{config.SOURCE_MODEL_NAME}")
    elif config.LOAD_MODE == "resume":
        saved = load_resume(agent, replay_buffer, normalizer)
        start_step = saved["step"]
        best_score = saved["best_score"]
        episode_rows = saved["episode_rows"]
        evaluation_rows = saved["evaluation_rows"]
        print(f"Resumed at step {start_step}")

    # settings are saved to a JSON file for reproducibility and reference
    settings = {
        "env_id": config.ENV_ID,
        "run_name": config.RUN_NAME,
        "load_mode": config.LOAD_MODE,
        "total_steps": config.TOTAL_STEPS,
        "action_repeat": config.ACTION_REPEAT,
        "cost_penalty": config.COST_PENALTY,
        "learning_rate": config.LEARNING_RATE,
    }
    with (RESULTS_DIR / "settings.json").open("w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2)

    print(f"Device: {DEVICE}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action repeat: {config.ACTION_REPEAT}")
    print(f"Training until: {config.TOTAL_STEPS} environment steps")

    state, _ = train_env.reset(seed=config.SEED + start_step)
    normalizer.observe(state)

    episode_reward = 0.0
    episode_learning_reward = 0.0
    episode_cost = 0.0
    episode_length = 0
    episode_number = len(episode_rows)
    last_critic_loss = 0.0
    last_actor_loss = 0.0

    current_step = start_step

    while current_step < config.TOTAL_STEPS:
        # One policy decision. The same action is applied for ACTION_REPEAT steps.
        action = choose_action(agent, normalizer, train_env, state, current_step)

        for _ in range(config.ACTION_REPEAT):
            if current_step >= config.TOTAL_STEPS:
                break

            next_state, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated
            cost = info["cost"]

            replay_buffer.add(state, action, next_state, reward, cost, done)
            normalizer.observe(next_state)

            state = next_state
            episode_reward += reward
            episode_learning_reward += reward - config.COST_PENALTY * cost
            episode_cost += cost
            episode_length += 1
            current_step += 1

            # initially we either randomly explore or refill the replay buffer, then we start training the agent
            warmup_steps = config.RANDOM_STEPS if config.LOAD_MODE == "new" else config.REFILL_STEPS
            # we train only if replay buffer has at least one batch and we are past the warmup phase
            if replay_buffer.size >= config.BATCH_SIZE and current_step >= warmup_steps:
                last_critic_loss, actor_loss = agent.train(
                    replay_buffer,
                    config.BATCH_SIZE,
                    normalizer,
                    config.COST_PENALTY,
                )
                if actor_loss is not None:
                    last_actor_loss = actor_loss

            if current_step % config.TENSORBOARD_EVERY == 0:
                writer.add_scalar("training/reward", reward, current_step)
                writer.add_scalar("training/cost", cost, current_step)
                writer.add_scalar("training/critic_loss", last_critic_loss, current_step)
                writer.add_scalar("training/actor_loss", last_actor_loss, current_step)
                writer.add_scalar("training/replay_size", replay_buffer.size, current_step)
                writer.add_scalar("actions/speed", action[0], current_step)
                writer.add_scalar("actions/steer", action[1], current_step)

            if done:
                episode_number += 1
                episode_rows.append(
                    {
                        "step": current_step,
                        "episode": episode_number,
                        "reward": episode_reward,
                        "learning_reward": episode_learning_reward,
                        "cost": episode_cost,
                        "length": episode_length,
                    }
                )
                writer.add_scalar("episodes/reward", episode_reward, current_step)
                writer.add_scalar("episodes/cost", episode_cost, current_step)

                state, _ = train_env.reset()
                normalizer.observe(state)
                episode_reward = 0.0
                episode_learning_reward = 0.0
                episode_cost = 0.0
                episode_length = 0

            should_evaluate = current_step % config.EVALUATE_EVERY == 0
            if should_evaluate or current_step == config.TOTAL_STEPS:
                evaluation = evaluate(
                    agent,
                    eval_env,
                    normalizer,
                    config.EVALUATION_EPISODES,
                    config.SEED + 10_000,
                )
                evaluation_rows.append({"step": current_step, **evaluation})

                writer.add_scalar("evaluation/reward", evaluation["mean_reward"], current_step)
                writer.add_scalar("evaluation/cost", evaluation["mean_cost"], current_step)
                writer.add_scalar("evaluation/score", evaluation["score"], current_step)

                save_policy(LAST_MODEL_PATH, agent, normalizer, current_step, evaluation)
                if evaluation["score"] > best_score:
                    best_score = evaluation["score"]
                    save_policy(BEST_MODEL_PATH, agent, normalizer, current_step, evaluation)

                save_results(episode_rows, evaluation_rows)
                print(
                    f"step {current_step:>7} | "
                    f"reward {evaluation['mean_reward']:>8.2f} | "
                    f"cost {evaluation['mean_cost']:>7.2f} | "
                    f"score {evaluation['score']:>8.2f}"
                )

            if current_step % config.SAVE_EVERY == 0 or current_step == config.TOTAL_STEPS:
                save_full_checkpoint(
                    current_step,
                    best_score,
                    agent,
                    replay_buffer,
                    normalizer,
                    episode_rows,
                    evaluation_rows,
                )

            if done:
                break

    writer.close()
    train_env.close()
    eval_env.close()

    print(f"Best model: {BEST_MODEL_PATH}")
    print(f"Last model: {LAST_MODEL_PATH}")
    print(f"Full checkpoint: {FULL_MODEL_PATH}")


if __name__ == "__main__":
    main()
