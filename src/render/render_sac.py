import safety_gymnasium

from config import MAX_STEPS


def render_policy(agent, env_name, episodes=5):

    env = safety_gymnasium.make(
        env_name,
        render_mode="human"
    )

    for episode in range(episodes):

        state, info = env.reset()

        total_reward = 0.0
        total_cost = 0.0
        done = False

        for step in range(MAX_STEPS):

            action = agent.select_action(state)

            next_state, reward, cost, terminated, truncated, info = env.step(
                action
            )

            total_reward += reward
            total_cost += cost

            done = terminated or truncated

            state = next_state

            if done:
                break

        print(
            f"Episode {episode+1}: "
            f"Reward = {total_reward:.2f}, "
            f"Cost = {total_cost:.2f}, "
            f"Steps = {step+1}"
        )

    env.close()