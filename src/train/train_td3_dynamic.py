import argparse
import time
from pathlib import Path

import numpy as np
import safety_gymnasium
import torch
from torch.utils.tensorboard import SummaryWriter

from normalize import NormalizeActionWrapper
from replayBuffer import ReplayBuffer
from td3 import TD3


DEFAULT_ENV_ID = "SafetyRacecarButton0-v0"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TD3 on Safety-Gymnasium Racecar Button."
    )

    # Environment / run setup
    parser.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-mode", type=str, default=None)

    # Training duration
    parser.add_argument("--max-timesteps", type=int, default=100_000)
    parser.add_argument("--start-timesteps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=int(1e6))

    # TD3 hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-noise", type=float, default=0.08)
    parser.add_argument("--noise-clip", type=float, default=0.15)
    parser.add_argument("--policy-freq", type=int, default=2)

    # Exploration / cost
    parser.add_argument("--expl-noise", type=float, default=0.05)
    parser.add_argument("--train-cost-penalty", type=float, default=0.0)
    parser.add_argument("--cost-weight", type=float, default=1.0)

    # Saving / loading
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--logs-dir", type=str, default="logs/td3")
    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--save-best", action="store_true")

    # Resume / pretrained loading
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pth file for full resume.",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path prefix for old TD3 model format, e.g. models/run/best",
    )

    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_id, seed=None, render_mode=None):
    env = safety_gymnasium.make(env_id, render_mode=render_mode)
    env = NormalizeActionWrapper(env)

    if seed is not None:
        state, info = env.reset(seed=seed)
        env.action_space.seed(seed)
    else:
        state, info = env.reset()

    assert np.allclose(env.action_space.low, -1.0), env.action_space
    assert np.allclose(env.action_space.high, 1.0), env.action_space

    return env, state, info


def make_agent(args, state_dim, action_dim):
    return TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=1.0,
        discount=args.discount,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        learning_rate=args.learning_rate,
    )


def select_training_action(agent, env, state, t, args, action_dim):
    if t < args.start_timesteps:
        action = env.action_space.sample()
    else:
        action = agent.select_action(np.array(state))

        noise = args.expl_noise * np.random.randn(action_dim)
        action = action + noise
        action = np.clip(action, env.action_space.low, env.action_space.high)
        action = action.astype(np.float32)

    return action


def evaluate_policy(agent, env_id, seed, eval_episodes=5, max_episode_steps=1000):
    eval_env, _, _ = make_env(env_id, seed=None, render_mode=None)

    total_reward = 0.0
    total_cost = 0.0
    total_steps = 0

    for ep in range(eval_episodes):
        state, info = eval_env.reset(seed=seed + 10_000 + ep)

        done = False
        ep_steps = 0

        while not done and ep_steps < max_episode_steps:
            action = agent.select_action(np.array(state))
            action = np.asarray(action, dtype=np.float32)
            action = np.clip(action, eval_env.action_space.low, eval_env.action_space.high)

            next_state, reward, cost, terminated, truncated, info = eval_env.step(action)

            done = terminated or truncated
            done_for_buffer = terminated 
            state = next_state

            total_reward += float(reward)
            total_cost += float(cost)
            total_steps += 1
            ep_steps += 1

    eval_env.close()

    return {
        "avg_reward": total_reward / eval_episodes,
        "avg_cost": total_cost / eval_episodes,
        "avg_steps": total_steps / eval_episodes,
    }


def log_losses(writer, losses, timestep):
    if losses is None:
        return

    for key, value in losses.items():
        if value is not None:
            writer.add_scalar(f"loss/{key}", float(value), timestep)


def save_checkpoint(
    path,
    agent,
    replay_buffer,
    global_step,
    episode_num,
    best_score,
    args,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "global_step": global_step,
        "episode_num": episode_num,
        "best_score": best_score,
        "args": vars(args),

        "actor": agent.actor.state_dict(),
        "actor_target": agent.actor_target.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),

        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),

        "total_it": agent.total_it,
        "replay_buffer": replay_buffer,
    }

    torch.save(checkpoint, path)
    print(f"[CHECKPOINT] Saved: {path}")


def load_checkpoint(path, agent):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    agent.actor.load_state_dict(checkpoint["actor"])
    agent.actor_target.load_state_dict(checkpoint["actor_target"])
    agent.critic.load_state_dict(checkpoint["critic"])
    agent.critic_target.load_state_dict(checkpoint["critic_target"])

    agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
    agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    agent.total_it = checkpoint.get("total_it", 0)

    replay_buffer = checkpoint["replay_buffer"]
    global_step = checkpoint.get("global_step", 0)
    episode_num = checkpoint.get("episode_num", 1)
    best_score = checkpoint.get("best_score", -float("inf"))

    print(f"[CHECKPOINT] Loaded: {path}")
    print(f"[CHECKPOINT] global_step={global_step}")
    print(f"[CHECKPOINT] episode_num={episode_num}")
    print(f"[CHECKPOINT] best_score={best_score}")

    return replay_buffer, global_step, episode_num, best_score


def print_action_debug(env, action, global_step):
    if global_step % 1000 != 0:
        return

    try:
        real_action = env.denormalize(action)
        print(
            f"[ACTION DEBUG] step={global_step} | "
            f"normalized={action} | real={real_action}"
        )
    except AttributeError:
        print(f"[ACTION DEBUG] step={global_step} | normalized={action}")


def main():
    args = parse_args()
    set_seed(args.seed)

    run_name = args.run_name
    if run_name is None:
        run_name = f"td3_{args.env_id}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"

    run_model_dir = Path(args.models_dir) / run_name
    run_log_dir = Path(args.logs_dir) / run_name

    run_model_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir.mkdir(parents=True, exist_ok=True)

    env, state, info = make_env(
        env_id=args.env_id,
        seed=args.seed,
        render_mode=args.render_mode,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=args.buffer_size,
    )

    agent = make_agent(args, state_dim, action_dim)

    global_step_start = 0
    episode_num = 1
    best_score = -float("inf")

    if args.resume_checkpoint is not None:
        replay_buffer, global_step_start, episode_num, best_score = load_checkpoint(
            args.resume_checkpoint,
            agent,
        )

        state, info = env.reset(seed=args.seed + global_step_start)

    elif args.load_model is not None:
        agent.load(args.load_model)
        print(f"[MODEL] Loaded old-format model from prefix: {args.load_model}")

    writer = SummaryWriter(log_dir=str(run_log_dir))

    writer.add_text(
        "config",
        "\n".join(f"{key}: {value}" for key, value in vars(args).items()),
        global_step_start,
    )

    episode_reward = 0.0
    episode_cost = 0.0
    episode_steps = 0

    latest_checkpoint_path = run_model_dir / "checkpoint_latest.pth"

    try:
        for t in range(global_step_start, args.max_timesteps):
            global_step = t + 1

            action = select_training_action(
                agent=agent,
                env=env,
                state=state,
                t=t,
                args=args,
                action_dim=action_dim,
            )

            next_state, reward, cost, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            done_for_buffer = terminated

            print_action_debug(env, action, global_step)

            train_reward = float(reward) - args.train_cost_penalty * float(cost)

            replay_buffer.add(
                state=state,
                action=action,
                next_state=next_state,
                reward=train_reward,
                cost=cost,
                done=done_for_buffer,
            )

            state = next_state
            episode_reward += float(reward)
            episode_cost += float(cost)
            episode_steps += 1

            if t >= args.start_timesteps and replay_buffer.size >= args.batch_size:
                losses = agent.train(
                    replay_buffer=replay_buffer,
                    batch_size=args.batch_size,
                )
                log_losses(writer, losses, global_step)

            writer.add_scalar("train/reward_step", float(reward), global_step)
            writer.add_scalar("train/cost_step", float(cost), global_step)

            if done:
                print(
                    f"Episode {episode_num} | "
                    f"timestep={global_step} | "
                    f"steps={episode_steps} | "
                    f"reward={episode_reward:.3f} | "
                    f"cost={episode_cost:.3f}"
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
                checkpoint_path = run_model_dir / f"checkpoint_{global_step}.pth"

                save_checkpoint(
                    path=checkpoint_path,
                    agent=agent,
                    replay_buffer=replay_buffer,
                    global_step=global_step,
                    episode_num=episode_num,
                    best_score=best_score,
                    args=args,
                )

                save_checkpoint(
                    path=latest_checkpoint_path,
                    agent=agent,
                    replay_buffer=replay_buffer,
                    global_step=global_step,
                    episode_num=episode_num,
                    best_score=best_score,
                    args=args,
                )

                # Old-format save για render scripts που φορτώνουν prefix_actor.pth
                agent.save(str(run_model_dir / "latest"))

            if args.eval_freq > 0 and global_step % args.eval_freq == 0:
                eval_stats = evaluate_policy(
                    agent=agent,
                    env_id=args.env_id,
                    seed=args.seed,
                    eval_episodes=args.eval_episodes,
                )

                eval_score = (
                    eval_stats["avg_reward"]
                    - args.cost_weight * eval_stats["avg_cost"]
                )

                writer.add_scalar("eval/avg_reward", eval_stats["avg_reward"], global_step)
                writer.add_scalar("eval/avg_cost", eval_stats["avg_cost"], global_step)
                writer.add_scalar("eval/avg_steps", eval_stats["avg_steps"], global_step)
                writer.add_scalar("eval/score_reward_minus_cost", eval_score, global_step)

                print(
                    f"Evaluation @ {global_step} | "
                    f"avg_reward={eval_stats['avg_reward']:.3f} | "
                    f"avg_cost={eval_stats['avg_cost']:.3f} | "
                    f"avg_steps={eval_stats['avg_steps']:.1f} | "
                    f"score={eval_score:.3f}"
                )

                if args.save_best and eval_score > best_score:
                    best_score = eval_score

                    agent.save(str(run_model_dir / "best"))

                    save_checkpoint(
                        path=run_model_dir / "checkpoint_best.pth",
                        agent=agent,
                        replay_buffer=replay_buffer,
                        global_step=global_step,
                        episode_num=episode_num,
                        best_score=best_score,
                        args=args,
                    )

                    print(f"[BEST] New best model saved. score={best_score:.3f}")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Training interrupted by user.")
        print("[INTERRUPTED] Saving latest checkpoint...")

        save_checkpoint(
            path=latest_checkpoint_path,
            agent=agent,
            replay_buffer=replay_buffer,
            global_step=global_step,
            episode_num=episode_num,
            best_score=best_score,
            args=args,
        )

        agent.save(str(run_model_dir / "latest"))

        print("[INTERRUPTED] Checkpoint saved successfully.")

    finally:
        final_path = run_model_dir / "final"

        agent.save(str(final_path))

        save_checkpoint(
            path=run_model_dir / "checkpoint_final.pth",
            agent=agent,
            replay_buffer=replay_buffer,
            global_step=min(args.max_timesteps, locals().get("global_step", global_step_start)),
            episode_num=episode_num,
            best_score=best_score,
            args=args,
        )

        env.close()
        writer.close()

        print(f"Training finished or stopped.")
        print(f"Final model prefix saved to: {final_path}")
        print(f"Latest checkpoint saved to: {latest_checkpoint_path}")
        print(f"TensorBoard logs saved to: {run_log_dir}")


if __name__ == "__main__":
    main()