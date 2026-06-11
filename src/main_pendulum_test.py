import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from train.sac import SAC
from evaluate.evaluate_sac import evaluate_policy
from config import *

# Create results directory if it doesn't exist
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)




# Register robotics environments
gymnasium_robotics.register_robotics_envs()

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Environment
env = gym.make(ENV_NAME)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Agent
agent = SAC(
    state_dim=state_dim,
    action_dim=action_dim,
    max_action=max_action,
    device=device,
    discount=GAMMA,
    tau=TAU,
    actor_lr=ACTOR_LR,
    critic_lr=CRITIC_LR
)

# Initial state
state, _ = env.reset(seed=SEED)

episode_reward = 0
episode_timesteps = 0
episode_num = 0

eval_rewards = []

for t in range(MAX_TIMESTEPS):

    episode_timesteps += 1

    # Select action
    if t < START_TIMESTEPS:
        action = env.action_space.sample()
    else:
        action = agent.select_action(state)

    # Step environment
    next_state, reward, terminated, truncated, _ = env.step(action)

    done = terminated or truncated

    # Store transition
    agent.replay_buffer.add(
        state,
        action,
        next_state,
        reward,
        done
    )

    state = next_state
    episode_reward += reward

    # Train
    if (
        t >= START_TIMESTEPS
        and agent.replay_buffer.size >= BATCH_SIZE
    ):

        critic_loss, actor_loss, alpha_loss, alpha = agent.train(
            BATCH_SIZE
        )
        
        if t % 1000 == 0:
            print(
                f"Step {t} | "
                f"Critic loss: {critic_loss:.4f} | "
                f"Actor loss: {actor_loss:.4f} | "
                f"Alpha: {alpha:.4f}"
            )
    # Evaluate and save
    if (t + 1) % EVAL_FREQ == 0:

        avg_reward, avg_length = evaluate_policy(
            agent,
            env_name=ENV_NAME,
            eval_episodes=SAC_EVAL_EPISODES
        )

        eval_rewards.append(avg_reward)

        print(
            "======================================\n"
            f"Step {t+1} | "
            f"Average reward: {avg_reward:.2f}"
            "\n======================================"
        )

        agent.save(SAC_MODEL_PATH)

    # Episode finished
    if done:

        print(
            f"Episode {episode_num} | "
            f"Steps {episode_timesteps} | "
            f"Reward {episode_reward:.2f}"
        )

        state, _ = env.reset()

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

# Save evaluation rewards
np.save(
    os.path.join(RESULTS_DIR, "rewards.npy"),
    np.array(eval_rewards)
)
# Plot learning curve
plt.figure(figsize=(8, 5))
plt.plot(eval_rewards)
plt.xlabel("Evaluation")
plt.ylabel("Average Reward")
plt.title("SAC Training Curve")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_curve.png"))
plt.show()

env.close()