import os
import safety_gymnasium
import numpy as np
import torch
import matplotlib.pyplot as plt

from train.sac import SAC
from evaluate.evaluate_sac import evaluate_policy
from render.render_sac import render_policy
from config import *

def train_0(MAX_TIMESTEPS,
            START_TIMESTEPS,
            LOAD_MODEL,
            LOAD_RESULTS_PATH,
            ENV_NAME = "SafetyRacecarButton2-v0",
            MAX_STEPS = 1000,
            BATCH_SIZE = 256,
            EVAL_FREQ = 50000,
            SAC_EVAL_EPISODES = 10,
            GAMMA = 0.99,
            TAU = 0.005,
            ACTOR_LR = 3e-4,
            CRITIC_LR = 3e-4,
            entropy_multiplier = 1.0,
            COST_WEIGHT = 0.0,
            SEED = 0,
            SAC_MODEL_PATH = "models/",
            AGENT_ID = "sac_phase_0",
            ):


    # Results path
    RESULTS_PATH = f"results/{AGENT_ID}"
    os.makedirs(RESULTS_PATH, exist_ok=True)


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
        discount=GAMMA,
        tau=TAU,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        entropy_multiplier=entropy_multiplier
    )

    if LOAD_MODEL is not None:
        agent.load(
            f"models/{LOAD_MODEL}"
        )

    (
        eval_rewards,
        eval_costs,
        eval_rewards_history,
        eval_costs_history,
        rewards_history,
        costs_history,
        modified_reward_history,
        alpha_history,
        training_steps,
        critic_loss_history,
        actor_loss_history
    ) = load_data(LOAD_RESULTS_PATH)

    # Initial state
    state, info = env.reset(seed=SEED)

    episode_reward = 0
    episode_cost = 0
    episode_shaped_reward = 0
    episode_num = 0
    episode_timesteps = 0


    if len(eval_rewards)>0:
        score_history = np.array(eval_rewards)-COST_WEIGHT*np.array(eval_costs)
        best_score = np.max(score_history)
    else:
        best_score = -np.inf

    for t in range(MAX_TIMESTEPS):

        episode_timesteps += 1

        # Action selection
        if t < START_TIMESTEPS:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        # Environment step
        next_state, reward, cost, terminated, truncated, info = env.step(
            action
        )

        done = terminated or truncated

        # Reward shaping
        modified_reward = reward - COST_WEIGHT * cost


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
            t >= START_TIMESTEPS
            and agent.replay_buffer.size >= BATCH_SIZE
        ):

            critic_loss, actor_loss, alpha_loss, alpha = agent.train(BATCH_SIZE)

            if t % MAX_STEPS == 0:
                
                alpha_history.append(alpha)

                if len(training_steps) == 0:
                    training_steps.append(0)
                else:
                    training_steps.append(
                        training_steps[-1] + MAX_STEPS
                    )

                critic_loss_history.append(critic_loss)
                actor_loss_history.append(actor_loss)

                print(
                    f"Step {t} | "
                    f"Critic loss: {critic_loss:.4f} | "
                    f"Actor loss: {actor_loss:.4f} | "
                    f"Alpha: {alpha:.4f}"
                )



                

        # Evaluation
        if (
            (t + 1) % EVAL_FREQ == 0 and
            t >= START_TIMESTEPS and
            agent.replay_buffer.size >= BATCH_SIZE
        ):
            
            avg_reward, avg_cost, eval_rewards_history_r, eval_costs_history_r = evaluate_policy(
                agent,
                env_name=ENV_NAME,
                eval_episodes=SAC_EVAL_EPISODES
            )

            eval_rewards.append(avg_reward)
            eval_costs.append(avg_cost)
            eval_rewards_history.append(eval_rewards_history_r)
            eval_costs_history.append(eval_costs_history_r)

            score = avg_reward - COST_WEIGHT * avg_cost
            
            print(
                "======================================"
            )

            print(
                f"Step {t+1} | "
                f"Average reward: {avg_reward:.2f} | "
                f"Average cost: {avg_cost:.2f} | "
                f"Score: {score:.2f}"
            )

            print(
                "======================================"
            )

            agent.save(SAC_MODEL_PATH + f"{AGENT_ID}_latest")
            print("The latest model was saved")
            # Save best model
            if score > best_score:

                best_score = score

                agent.save(SAC_MODEL_PATH + f"{AGENT_ID}_best")
                print(f"New best model saved "
                      f"(score = {score:.2f})")
                

            # plots and results save
            save_results(
                RESULTS_PATH,
                 eval_rewards,
                 eval_costs,
                 eval_rewards_history,
                 eval_costs_history,
                 rewards_history,
                 costs_history,
                 alpha_history,
                 training_steps,
                 critic_loss_history,
                 actor_loss_history,
                 modified_reward_history,
                 COST_WEIGHT
                 )




        # Episode finished
        if done:
            rewards_history.append(episode_reward)
            costs_history.append(episode_cost)
            modified_reward_history.append(episode_shaped_reward)

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


    # final backup save

    save_results(
                RESULTS_PATH,
                 eval_rewards,
                 eval_costs,
                 eval_rewards_history,
                 eval_costs_history,
                 rewards_history,
                 costs_history,
                 alpha_history,
                 training_steps,
                 critic_loss_history,
                 actor_loss_history,
                 modified_reward_history,
                 COST_WEIGHT
                 )

    env.close()

    #render_policy(agent, ENV_NAME, episodes=5)







def save_results(RESULTS_PATH,
                 eval_rewards,
                 eval_costs,
                 eval_rewards_history,
                 eval_costs_history,
                 rewards_history,
                 costs_history,
                 alpha_history,
                 training_steps,
                 critic_loss_history,
                 actor_loss_history,
                 modified_reward_history,
                 COST_WEIGHT
                ):
    
    score_history = np.array(eval_rewards) - COST_WEIGHT*np.array(eval_costs)
    
    np.save(
        f"{RESULTS_PATH}/eval_rewards.npy",
        np.array(eval_rewards)
    )

    np.save(
        f"{RESULTS_PATH}/eval_costs.npy",
        np.array(eval_costs)
    )

    np.save(
        f"{RESULTS_PATH}/eval_rewards_history.npy",
        np.array(eval_rewards_history)
    )

    np.save(
        f"{RESULTS_PATH}/eval_costs_history.npy",
        np.array(eval_costs_history)
    )

    np.save(
        f"{RESULTS_PATH}/alpha_history.npy",
        np.array(alpha_history)
    )

    np.save(
        f"{RESULTS_PATH}/training_steps.npy",
        np.array(training_steps)
    )

    np.save(
        f"{RESULTS_PATH}/critic_loss_history.npy",
        np.array(critic_loss_history)
    )

    np.save(
        f"{RESULTS_PATH}/actor_loss_history.npy",
        np.array(actor_loss_history)
    )

    np.save(
        f"{RESULTS_PATH}/rewards_history.npy",
        np.array(rewards_history)
    )

    np.save(
        f"{RESULTS_PATH}/costs_history.npy",
        np.array(costs_history)
    )

    np.save(
        f"{RESULTS_PATH}/modified_reward_history.npy",
        np.array(modified_reward_history)
    )

    np.save(
        f"{RESULTS_PATH}/score_history.npy",
        np.array(score_history)
    )

    # Reward avg curve eval
    plt.figure(figsize=(8,5))
    plt.plot(eval_rewards, marker ='o')
    plt.xlabel("Evaluation")
    plt.ylabel("Average Reward")
    plt.title("Reward Curve")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/eval_reward_curve.png")
    plt.close()

    # Cost avg curve eval
    plt.figure(figsize=(8,5))
    plt.plot(eval_costs, marker ='o')
    plt.xlabel("Evaluation")
    plt.ylabel("Average Cost")
    plt.title("Cost Curve")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/eval_cost_curve.png")
    plt.close()

    # Reward std curve eval
    reward_std = np.std(eval_rewards_history, axis=1)
    plt.figure(figsize=(8,5))
    plt.plot(reward_std)
    plt.xlabel("Evaluation")
    plt.ylabel("Reward std")
    plt.title("Reward standard deviation")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/eval_reward_std_curve.png")
    plt.close()

    # Cost std curve eval
    cost_std = np.std(eval_costs_history,axis = 1)
    plt.figure(figsize=(8,5))
    plt.plot(cost_std)
    plt.xlabel("Evaluation")
    plt.ylabel("Cost std")
    plt.title("Cost standard deviation")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/eval_cost_std_curve.png")
    plt.close()


    # Alpha curve
    plt.figure(figsize=(8,5))
    plt.plot(training_steps,alpha_history)
    plt.xlabel("Training step")
    plt.ylabel("Alpha")
    plt.title("Alpha evolution")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/alpha_curve.png")
    plt.close()

    # Critic loss curve
    plt.figure(figsize=(8,5))
    plt.plot(training_steps,critic_loss_history)
    plt.xlabel("Training step")
    plt.ylabel("Critic loss")
    plt.title("Critic loss curve")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/critic_loss_curve.png")
    plt.close()

    # Actor loss curve
    plt.figure(figsize=(8,5))
    plt.plot(training_steps,actor_loss_history)
    plt.xlabel("Training step")
    plt.ylabel("Actor loss")
    plt.title("Actor loss curve")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/actor_loss_curve.png")
    plt.close()

    # Reward curve
    plt.figure(figsize=(8,5))
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward curve")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/reward.png")
    plt.close()

    # Cost curve
    plt.figure(figsize=(8,5))
    plt.plot(costs_history)
    plt.xlabel("Episode")
    plt.ylabel("Cost")
    plt.title("Cost curve")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/cost.png")
    plt.close()

    # Modified reward curve
    plt.figure(figsize=(8,5))
    plt.plot(modified_reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Modified reward")
    plt.title("Modified reward curve")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/modified_reward.png")
    plt.close()

    # Score curve
    plt.figure(figsize=(8,5))
    plt.plot(score_history, marker ='o')
    plt.xlabel("Evaluation")
    plt.ylabel("Score")
    plt.title("Evaluation score")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_PATH}/score_curve.png")
    plt.close()

def load_data(LOAD_RESULTS_PATH):

    if os.path.exists(f"{LOAD_RESULTS_PATH}/eval_rewards.npy"):
        eval_rewards = np.load(
            f"{LOAD_RESULTS_PATH}/eval_rewards.npy"
        ).tolist()

        eval_costs = np.load(
            f"{LOAD_RESULTS_PATH}/eval_costs.npy"
        ).tolist()

        eval_rewards_history = np.load(
            f"{LOAD_RESULTS_PATH}/eval_rewards_history.npy"
        ).tolist()

        eval_costs_history = np.load(
            f"{LOAD_RESULTS_PATH}/eval_costs_history.npy"
        ).tolist()

        rewards_history = np.load(
            f"{LOAD_RESULTS_PATH}/rewards_history.npy"
        ).tolist()

        costs_history = np.load(
            f"{LOAD_RESULTS_PATH}/costs_history.npy"
        ).tolist()

        modified_reward_history = np.load(
            f"{LOAD_RESULTS_PATH}/modified_reward_history.npy"
        ).tolist()

        alpha_history = np.load(
            f"{LOAD_RESULTS_PATH}/alpha_history.npy"
        ).tolist()

        training_steps = np.load(
            f"{LOAD_RESULTS_PATH}/training_steps.npy"
        ).tolist()

        critic_loss_history = np.load(
            f"{LOAD_RESULTS_PATH}/critic_loss_history.npy"
        ).tolist()

        actor_loss_history = np.load(
            f"{LOAD_RESULTS_PATH}/actor_loss_history.npy"
        ).tolist()
    else:
        eval_rewards = []
        eval_costs = []
        eval_rewards_history =[]
        eval_costs_history =[]
        rewards_history = []
        costs_history = []
        modified_reward_history = []
        alpha_history = []
        training_steps = []
        critic_loss_history = []
        actor_loss_history = []

    return (
        eval_rewards,
        eval_costs,
        eval_rewards_history,
        eval_costs_history,
        rewards_history,
        costs_history,
        modified_reward_history,
        alpha_history,
        training_steps,
        critic_loss_history,
        actor_loss_history
    )