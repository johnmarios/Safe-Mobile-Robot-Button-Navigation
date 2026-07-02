from pathlib import Path

import config
from environment import make_env
from normalizer import MixedObservationNormalizer
from td3 import TD3, load_torch_file


# Change only this block when you want to render another saved policy.
RUN_NAME = "button2_td3_baseline_half_action_repeat_termination_continue_cost_0.02"
MODEL_NAME = "last_model.pt"
EPISODES = 5
SEED = 12_345
ACTION_REPEAT = 1


def main():
    model_path = Path("results") / RUN_NAME / MODEL_NAME

    env = make_env(config.ENV_ID, render_mode="human", goal_termination=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    saved = load_torch_file(model_path)

    agent = TD3(state_dim, action_dim)
    agent.load_networks(saved["agent"])

    normalizer = MixedObservationNormalizer(env.observation_space)
    normalizer.load_state_dict(saved["observation_normalizer"])

    for episode in range(EPISODES):
        state, _ = env.reset(seed=SEED + episode)
        done = False
        total_reward = 0.0
        total_cost = 0.0
        steps = 0

        while not done:
            action = agent.select_action(normalizer.normalize(state))

            for _ in range(ACTION_REPEAT):
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                total_reward += reward
                total_cost += info["cost"]
                steps += 1

                env.render()

                if done:
                    break

        print(
            f"Episode {episode + 1}: "
            f"reward={total_reward:.2f}, cost={total_cost:.2f}, steps={steps}"
        )

    env.close()


if __name__ == "__main__":
    main()
