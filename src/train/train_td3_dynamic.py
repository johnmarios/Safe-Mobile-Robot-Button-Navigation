import argparse
import os
import time
from pathlib import Path

import numpy as np
from normalize import NormalizeActionWrapper, sample_warmup_action
import safety_gymnasium
import torch
from torch.utils.tensorboard import SummaryWriter

from replayBuffer import ReplayBuffer
from td3 import TD3


DEFAULT_ENV_ID = "SafetyRacecarButton2-v0"


def parse_args():
    parser = argparse.ArgumentParser(description="Train TD3 on Safety-Gymnasium Racecar Button.")

    # Environment / run setup
    parser.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-mode", type=str, default=None)

    # Training duration
    parser.add_argument("--max-timesteps", type=int, default=100_000, help="Maximum number of timesteps to train for.") 
    parser.add_argument("--start-timesteps", type=int, default=50_000, help="Number of timesteps to run random actions (explore) before using policy actions.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=int(1e6))

    # TD3 / exploration hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--expl-noise", type=float, default=0.10, help="Exploration noise as fraction of max_action.")
    parser.add_argument("--discount", type=float, default=None, help="Discount factor for future rewards.")
    parser.add_argument("--tau", type=float, default=None, help="Target network update rate.") # target network update rate
    parser.add_argument("--policy-noise", type=float, default=None, help="Noise added to target policy during critic update.")# noise added to target policy during critic update
    parser.add_argument("--noise-clip", type=float, default=None, help="Clip the noise added to the target policy.")
    parser.add_argument("--policy-freq", type=int, default=None, help="Frequency of policy updates.")
    parser.add_argument("--train-cost-penalty", type=float, default=0.0)

    # Saving / loading
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--logs-dir", type=str, default="logs/td3")
    parser.add_argument("--save-freq", type=int, default=10_000, help="Save checkpoint every N timesteps. Use 0 to disable.")
    parser.add_argument("--eval-freq", type=int, default=5_000, help="Evaluate every N timesteps. Use 0 to disable.")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of episodes for each evaluation.")
    parser.add_argument("--load-model", type=str, default=None, help="Path prefix of model to load/resume, e.g. models/run/best")
    parser.add_argument("--save-best", action="store_true", help="Save best model according to eval reward - cost_weight * cost.")
    parser.add_argument("--cost-weight", type=float, default=1.0, help="Penalty weight for eval cost when choosing best model.")

    return parser.parse_args()


def make_agent(args, state_dim, action_dim, max_action):
    """
    Create TD3 agent with hyperparameters from args. 
    Only the required ones are passed as explicit parameters, 
    the rest are passed via kwargs if they are not None.

    """
    td3_kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "learning_rate": args.learning_rate,
    }

    optional_args = {
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise,
        "noise_clip": args.noise_clip,
        "policy_freq": args.policy_freq,
    }

    # Only include optional args that are not None (i.e. were set by the user)
    # to use TD3 defaults for any that were left as None

    for key, value in optional_args.items():
        if value is not None:
            td3_kwargs[key] = value

    return TD3(**td3_kwargs)


def evaluate_policy(agent, env_id, seed, eval_episodes=5):
    eval_env = safety_gymnasium.make(env_id, render_mode=None)
    eval_env = NormalizeActionWrapper(eval_env)
    
    total_reward = 0.0
    total_cost = 0.0
    total_steps = 0

    for ep in range(eval_episodes):

        # 10000 offset to avoid overlap with training seeds,
        # so that evaluation is deterministic but different from training

        state, info = eval_env.reset(seed=seed + 10_000 + ep)
        done = False

        while not done:
            action = agent.select_action(np.array(state))
            next_state, reward, cost, terminated, truncated, info = eval_env.step(action)

            done = terminated or truncated
            state = next_state

            total_reward += reward
            total_cost += cost
            total_steps += 1

    eval_env.close()

    return {
        "avg_reward": total_reward / eval_episodes,
        "avg_cost": total_cost / eval_episodes,
        "avg_steps": total_steps / eval_episodes,
    }


def log_losses(writer, losses, timestep):
    """
    Supports either:
    - dict, e.g. {"critic_loss": ..., "actor_loss": ...}
    - tuple/list, e.g. (critic_loss, actor_loss)
    If your TD3.train returns None, nothing is logged here.
    """
    if losses is None:
        return

    if isinstance(losses, dict):
        for key, value in losses.items():
            if value is not None:
                writer.add_scalar(f"loss/{key}", float(value), timestep)
        return

    if isinstance(losses, (tuple, list)):
        names = ["critic_loss", "actor_loss"]
        for idx, value in enumerate(losses):
            if value is not None:
                name = names[idx] if idx < len(names) else f"loss_{idx}"
                writer.add_scalar(f"loss/{name}", float(value), timestep)


def main():
    args = parse_args()

    run_name = args.run_name
    if run_name is None:
        run_name = f"td3_{args.env_id}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"

    run_model_dir = Path(args.models_dir) / run_name
    run_log_dir = Path(args.logs_dir) / run_name

    run_model_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = safety_gymnasium.make(args.env_id, render_mode=args.render_mode)
    env = NormalizeActionWrapper(env)
    
    state, info = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=args.buffer_size,
    )

    agent = make_agent(args, state_dim, action_dim, max_action)

    if args.load_model is not None:
        agent.load(args.load_model)
        print(f"Loaded model from: {args.load_model}")

    writer = SummaryWriter(log_dir=str(run_log_dir))

    # Store hyperparameters in TensorBoard text tab
    writer.add_text(
        "config",
        "\n".join(f"{key}: {value}" for key, value in vars(args).items()),
        0,
    )

    episode_reward = 0.0
    episode_cost = 0.0
    episode_steps = 0
    episode_num = 1
    best_score = -float("inf")

    for t in range(args.max_timesteps):
        global_step = t + 1

        if t < args.start_timesteps:
            action = sample_warmup_action(env, action_dim)
        else:
            action = agent.select_action(np.array(state))

            noise = args.expl_noise * max_action * np.random.randn(action_dim)
            action = action + noise
            action = action.clip(env.action_space.low, env.action_space.high)

        next_state, reward, cost, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        train_reward = reward - args.train_cost_penalty * cost

        replay_buffer.add(
            state=state,
            action=action,
            next_state=next_state,
            reward=train_reward,
            cost=cost,
            done=done,
        )

        state = next_state
        episode_reward += reward
        episode_cost += cost
        episode_steps += 1

        if t >= args.start_timesteps:
            losses = agent.train(
                replay_buffer=replay_buffer,
                batch_size=args.batch_size,
            )
            log_losses(writer, losses, global_step)

        writer.add_scalar("train/reward_step", reward, global_step)
        writer.add_scalar("train/cost_step", cost, global_step)

        if done:
            print(
                f"Episode {episode_num} | "
                f"timestep={global_step} | "
                f"steps={episode_steps} | "
                f"reward={episode_reward:.2f} | "
                f"cost={episode_cost:.2f}"
            )

            writer.add_scalar("episode/reward", episode_reward, global_step)
            writer.add_scalar("episode/cost", episode_cost, global_step)
            writer.add_scalar("episode/steps", episode_steps, global_step)

            state, info = env.reset()
            episode_reward = 0.0
            episode_cost = 0.0
            episode_steps = 0
            episode_num += 1

        if args.save_freq > 0 and global_step % args.save_freq == 0:
            checkpoint_path = run_model_dir / f"checkpoint_{global_step}"
            agent.save(str(checkpoint_path))
            agent.save(str(run_model_dir / "latest"))
            print(f"Saved checkpoint: {checkpoint_path}")

        if args.eval_freq > 0 and global_step % args.eval_freq == 0:
            eval_stats = evaluate_policy(
                agent=agent,
                env_id=args.env_id,
                seed=args.seed,
                eval_episodes=args.eval_episodes,
            )

            eval_score = eval_stats["avg_reward"] - args.cost_weight * eval_stats["avg_cost"]

            writer.add_scalar("eval/avg_reward", eval_stats["avg_reward"], global_step)
            writer.add_scalar("eval/avg_cost", eval_stats["avg_cost"], global_step)
            writer.add_scalar("eval/avg_steps", eval_stats["avg_steps"], global_step)
            writer.add_scalar("eval/score_reward_minus_cost", eval_score, global_step)

            print(
                f"Evaluation @ {global_step} | "
                f"avg_reward={eval_stats['avg_reward']:.2f} | "
                f"avg_cost={eval_stats['avg_cost']:.2f} | "
                f"score={eval_score:.2f}"
            )

            if args.save_best and eval_score > best_score:
                best_score = eval_score
                best_path = run_model_dir / "best"
                agent.save(str(best_path))
                print(f"New best model saved: {best_path}")

    final_path = run_model_dir / "final"
    agent.save(str(final_path))
    env.close()
    writer.close()

    print(f"Training finished. Final model saved to: {final_path}")
    print(f"TensorBoard logs saved to: {run_log_dir}")


if __name__ == "__main__":
    main()
