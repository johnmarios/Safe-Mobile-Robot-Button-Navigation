import safety_gymnasium
import numpy as np
import os, sys

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.append(src_dir)

from train.td3 import TD3


ENV_ID = "SafetyRacecarButton2-v0"
MODEL_PATH = "models/td3_safety_racecar_button2"

NUM_EPISODES = 5
MAX_STEPS = 1000
SEED = 42


def render_td3():
    env = safety_gymnasium.make(
        ENV_ID,
        render_mode="human"
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action
    )

    agent.load(MODEL_PATH)

    print("\n==============================")
    print("Rendering TD3")
    print("==============================")
    print(f"Environment: {ENV_ID}")
    print(f"Model path: {MODEL_PATH}")

    for episode in range(NUM_EPISODES):
        state, info = env.reset(seed=SEED + episode)

        episode_reward = 0.0
        episode_cost = 0.0
        episode_steps = 0

        print(f"\nEpisode {episode + 1}/{NUM_EPISODES}")

        for step in range(MAX_STEPS):
            # Render: deterministic trained policy, no exploration noise
            action = agent.select_action(state)

            next_state, reward, cost, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            episode_reward += reward
            episode_cost += cost
            episode_steps += 1

            env.render()

            state = next_state

            if done:
                break

        print(f"Reward: {episode_reward:.2f}")
        print(f"Cost: {episode_cost:.2f}")
        print(f"Steps: {episode_steps}")

    env.close()


if __name__ == "__main__":
    render_td3()