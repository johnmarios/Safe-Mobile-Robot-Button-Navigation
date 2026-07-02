import safety_gymnasium



def render_policy(agent, env_name, episodes=5, max_steps = 1000, action_repeat=1):

    env = safety_gymnasium.make(
        env_name,
        render_mode="human"
    )

    for episode in range(episodes):

        state, info = env.reset()

        total_reward = 0.0
        total_cost = 0.0
        done = False

        steps = 0

        while not done and steps < max_steps:

            action = agent.select_action(state)

            for _ in range(action_repeat):

                next_state, reward, cost, terminated, truncated, info = env.step(action)

                total_reward += reward
                total_cost += cost

                steps += 1

                state = next_state

                done = terminated or truncated or steps >= max_steps

                if done:
                    break


        print(
            f"Episode {episode+1}: "
            f"Reward = {total_reward:.2f}, "
            f"Cost = {total_cost:.2f}"
        )

    env.close()