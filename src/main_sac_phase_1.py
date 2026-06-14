import os
import safety_gymnasium
import numpy as np
import torch
import matplotlib.pyplot as plt

from train.sac import SAC
from evaluate.evaluate_sac import evaluate_policy
from render.render_sac import render_policy
from config import *

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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
    discount=GAMMA_1,
    tau=TAU_1,
    actor_lr=ACTOR_LR_1,
    critic_lr=CRITIC_LR_1
)

# Initial state
state, info = env.reset(seed=SEED)

episode_reward = 0
episode_cost = 0
episode_shaped_reward = 0
episode_num = 0
episode_timesteps = 0

eval_rewards = []
eval_costs = []

best_reward = -np.inf

for t in range(MAX_TIMESTEPS_1):

    episode_timesteps += 1

    # Action selection
    if t < START_TIMESTEPS_1:
        action = env.action_space.sample()
    else:
        action = agent.select_action(state)

    # Environment step
    next_state, reward, cost, terminated, truncated, info = env.step(
        action
    )

    done = terminated or truncated

    # Reward shaping
    modified_reward = reward - COST_WEIGHT_1 * cost


    # Store transition
    agent.replay_buffer.add(
        state,
        action,
        next_state,
        modified_reward,
        done
    )

    state = next_state

    episode_reward += reward
    episode_cost += cost
    episode_shaped_reward += modified_reward

    # Train
    if (
        t >= START_TIMESTEPS_1
        and agent.replay_buffer.size >= BATCH_SIZE
    ):

        critic_loss, actor_loss, alpha_loss, alpha = \
            agent.train(BATCH_SIZE)

        if t % 1000 == 0:

            print(
                f"Step {t} | "
                f"Critic loss: {critic_loss:.4f} | "
                f"Actor loss: {actor_loss:.4f} | "
                f"Alpha: {alpha:.4f}"
            )

    # Evaluation
    if (t + 1) % EVAL_FREQ == 0:

        avg_reward, avg_cost = evaluate_policy(
            agent,
            env_name=ENV_NAME,
            eval_episodes=SAC_EVAL_EPISODES
        )

        eval_rewards.append(avg_reward)
        eval_costs.append(avg_cost)

        print(
            "======================================"
        )

        print(
            f"Step {t+1} | "
            f"Average reward: {avg_reward:.2f} | "
            f"Average cost: {avg_cost:.2f}"
        )

        print(
            "======================================"
        )

        # Save best model
        if avg_reward > best_reward:

            best_reward = avg_reward

            agent.save(SAC_MODEL_PATH + f"{AGENT_ID_1}_best")

    # Episode finished
    if done:

        print(
            f"Episode {episode_num} | "
            f"Steps {episode_timesteps} | "
            f"Reward {episode_reward:.2f} | "
            f"Cost {episode_cost:.2f} | "
            f"Shaped Reward {episode_shaped_reward:.2f}"
        )

        state, info = env.reset()

        episode_reward = 0
        episode_cost = 0
        episode_shaped_reward = 0
        episode_timesteps = 0
        episode_num += 1


# Save results
os.makedirs("results", exist_ok=True)

np.save(
    "results/rewards.npy",
    np.array(eval_rewards)
)

np.save(
    "results/costs.npy",
    np.array(eval_costs)
)

agent.save(SAC_MODEL_PATH + f"{AGENT_ID_1}_final")

print("Final model saved.")

# Reward curve
plt.figure(figsize=(8,5))
plt.plot(eval_rewards)
plt.xlabel("Evaluation")
plt.ylabel("Average Reward")
plt.title("Reward Curve")
plt.grid()
plt.tight_layout()
plt.savefig("results/reward_curve.png")


# Cost curve
plt.figure(figsize=(8,5))
plt.plot(eval_costs)
plt.xlabel("Evaluation")
plt.ylabel("Average Cost")
plt.title("Cost Curve")
plt.grid()
plt.tight_layout()
plt.savefig("results/cost_curve.png")

env.close()

render_policy(agent, ENV_NAME, episodes=5)