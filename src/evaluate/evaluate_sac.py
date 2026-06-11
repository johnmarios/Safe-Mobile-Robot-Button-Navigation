import safety_gymnasium
import numpy as np

from config import ENV_NAME, SAC_EVAL_EPISODES, MAX_STEPS, SEED


def evaluate_policy(
    agent,
    env_name=ENV_NAME,
    eval_episodes=SAC_EVAL_EPISODES,
    render=False
):

    render_mode = "human" if render else None

    env = safety_gymnasium.make(
        env_name,
        render_mode=render_mode
    )

    episode_rewards = []
    episode_costs = []

    for episode in range(eval_episodes):

        state, info = env.reset(seed=SEED + episode)

        total_reward = 0.0
        total_cost = 0.0
        steps = 0
        done = False

        while not done and steps < MAX_STEPS:

            action = agent.select_action(state)

            next_state, reward, cost, terminated, truncated, info = env.step(
                action
            )

            done = terminated or truncated

            total_reward += reward
            total_cost += cost

            steps += 1

            state = next_state

        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)

        print(
            f"Episode {episode+1}: "
            f"Reward = {total_reward:.2f}, "
            f"Cost = {total_cost:.2f}, "
            f"Steps = {steps}"
        )

    # print("Evaluation rewards:")
    # print(episode_rewards)

    # print("Evaluation costs:")
    # print(episode_costs)

    print(
        f"Mean reward = {np.mean(episode_rewards):.2f}, "
        f"Mean cost = {np.mean(episode_costs):.2f}"
    )

    env.close()

    return np.mean(episode_rewards), np.mean(episode_costs)