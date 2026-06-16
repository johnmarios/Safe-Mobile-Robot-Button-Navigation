import safety_gymnasium
import numpy as np
import os
import sys
import time
import torch

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
train_dir = os.path.join(src_dir, "train")
if src_dir not in sys.path:
    sys.path.append(src_dir)
if train_dir not in sys.path:
    sys.path.append(train_dir)

from train.td3 import TD3
from train.normalize import NormalizeActionWrapper


ENV_ID = "SafetyRacecarButton0-v0"

# Για checkpoint 300k:
# MODEL_PATH = "models/td3_button0_exp6/checkpoint_400000.pth"

# Για old-format latest:
MODEL_PATH = "models/td3_button0_exp9/latest"

NUM_EPISODES = 5
MAX_STEPS = 1000
# SEED = 42
SEED = 20 


def load_from_checkpoint(agent, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    agent.actor.load_state_dict(checkpoint["actor"])

    if "actor_target" in checkpoint:
        agent.actor_target.load_state_dict(checkpoint["actor_target"])

    if "critic" in checkpoint:
        agent.critic.load_state_dict(checkpoint["critic"])

    if "critic_target" in checkpoint:
        agent.critic_target.load_state_dict(checkpoint["critic_target"])

    print(f"[CHECKPOINT] Loaded checkpoint: {checkpoint_path}")
    print(f"[CHECKPOINT] global_step={checkpoint.get('global_step', 'unknown')}")


def render_td3():
    env = safety_gymnasium.make(ENV_ID, render_mode="human")
    env = NormalizeActionWrapper(env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    max_action = 1.0

    agent = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
    )

    if MODEL_PATH.endswith(".pth"):
        load_from_checkpoint(agent, MODEL_PATH)
    else:
        agent.load(MODEL_PATH)

    print("\n==============================")
    print("Rendering TD3")
    print("==============================")
    print(f"Environment: {ENV_ID}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Observation space: {env.observation_space}")
    print(f"Normalized action space: {env.action_space}")

    for episode in range(NUM_EPISODES):
        state, info = env.reset(seed=SEED + episode)

        episode_reward = 0.0
        episode_cost = 0.0
        episode_steps = 0

        print(f"\nEpisode {episode + 1}/{NUM_EPISODES}")

        for step in range(MAX_STEPS):
            normalized_action = agent.select_action(np.array(state))
            normalized_action = np.asarray(normalized_action, dtype=np.float32)

            normalized_action = np.clip(
                normalized_action,
                env.action_space.low,
                env.action_space.high,
            )

            try:
                real_action = env.denormalize(normalized_action)
            except AttributeError:
                real_action = normalized_action

            next_state, reward, cost, terminated, truncated, info = env.step(
                normalized_action
            )

            done = terminated or truncated

            episode_reward += float(reward)
            episode_cost += float(cost)
            episode_steps += 1

            if step % 50 == 0:
                print(
                    f"step={step} | "
                    f"normalized_action={normalized_action} | "
                    f"real_action={real_action} | "
                    f"reward={reward:.3f} | "
                    f"cost={cost}"
                )

            env.render()
            time.sleep(0.01)

            state = next_state

            if done:
                break

        print(f"Reward: {episode_reward:.2f}")
        print(f"Cost: {episode_cost:.2f}")
        print(f"Steps: {episode_steps}")

    env.close()


if __name__ == "__main__":
    render_td3()