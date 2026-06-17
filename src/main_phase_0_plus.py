import os
import safety_gymnasium
import numpy as np
import torch
import matplotlib.pyplot as plt

from train.sac_fixed_alpha import SAC_FA
from evaluate.evaluate_sac import evaluate_policy
from render.render_sac import render_policy
from config import *

def main_0_plus():
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
    agent = SAC_FA(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        discount=GAMMA_0_PLUS,
        tau=TAU_0_PLUS,
        actor_lr=ACTOR_LR_0_PLUS,
        critic_lr=CRITIC_LR_0_PLUS,
        entropy_multiplier=entropy_multiplier_0_PLUS
    )

    # Load the best model from phase 0
    agent.load(
        "models/sac_phase_0_best"
    )

    print("Loaded phase 0 checkpoint.")
    # Initial state
    state, info = env.reset(seed=SEED)

    episode_reward = 0
    episode_cost = 0
    episode_shaped_reward = 0
    episode_num = 0
    episode_timesteps = 0

    eval_rewards = []
    eval_costs = []
    alpha_history = []
    alpha_steps = []
    critic_loss_history = []
    actor_loss_history = []


    best_reward = -np.inf

    for t in range(MAX_TIMESTEPS_0_PLUS):

        episode_timesteps += 1

        action = agent.select_action(state)

        # Environment step
        next_state, reward, cost, terminated, truncated, info = env.step(
            action
        )

        done = terminated or truncated

        # Reward shaping
        modified_reward = reward - COST_WEIGHT_0_PLUS * cost


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
        if (agent.replay_buffer.size >= BATCH_SIZE):

            critic_loss, actor_loss, _, alpha = agent.train(BATCH_SIZE)
            
            critic_loss_history.append(critic_loss)
            actor_loss_history.append(actor_loss)

            alpha_history.append(alpha)
            alpha_steps.append(t)

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

                agent.save(SAC_MODEL_PATH + f"{AGENT_ID_0_PLUS}_best")

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
    RESULTS_PATH = f"results/{AGENT_ID_0_PLUS}"

    os.makedirs(RESULTS_PATH, exist_ok=True)

    np.save(
        f"{RESULTS_PATH}/rewards.npy",
        np.array(eval_rewards)
    )

    np.save(
        f"{RESULTS_PATH}/costs.npy",
        np.array(eval_costs)
    )

    np.save(
        f"{RESULTS_PATH}/alpha.npy",
        np.array(alpha_history)
    )

    np.save(
        f"{RESULTS_PATH}/alpha_steps.npy",
        np.array(alpha_steps)
)
    
    np.save(
        f"{RESULTS_PATH}/critic_loss.npy",
        np.array(critic_loss_history)
        )

    np.save(
        f"{RESULTS_PATH}/actor_loss.npy",
        np.array(actor_loss_history)
        )


    agent.save(SAC_MODEL_PATH + f"{AGENT_ID_0_PLUS}_final")

    print("Final model saved.")

    # Reward curve
    plt.figure(figsize=(8,5))
    plt.plot(eval_rewards)
    plt.xlabel("Evaluation")
    plt.ylabel("Average Reward")
    plt.title("Reward Curve")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/reward_curve.png")


    # Cost curve
    plt.figure(figsize=(8,5))
    plt.plot(eval_costs)
    plt.xlabel("Evaluation")
    plt.ylabel("Average Cost")
    plt.title("Cost Curve")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/cost_curve.png")

    # alpha curve
    plt.figure(figsize=(8,5))
    plt.plot(alpha_steps, alpha_history)
    plt.xlabel("Training step")
    plt.ylabel("Alpha")
    plt.title("Alpha evolution")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/alpha_curve.png")

    # Critic loss curve

    plt.figure(figsize=(8,5))
    plt.plot(critic_loss_history)
    plt.xlabel("Training Updates")
    plt.ylabel("Critic Loss")
    plt.title("Critic Loss Curve")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/critic_loss_curve.png")
    plt.close()

    # Actor loss curve

    plt.figure(figsize=(8,5))
    plt.plot(actor_loss_history)
    plt.xlabel("Training Updates")
    plt.ylabel("Actor Loss")
    plt.title("Actor Loss Curve")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/actor_loss_curve.png")
    plt.close()



    env.close()

    render_policy(agent, ENV_NAME, episodes=5)