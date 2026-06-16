import os
import sys
import time
import numpy as np
import safety_gymnasium

# Για να βρίσκει imports από src/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# from tools.controller_button0 import Button0Controller
from controllers.controller0 import Button0Controller

def render_button0_controller(
    env_id="SafetyRacecarButton0-v0",
    episodes=5,
    max_steps=1000,
    sleep=0.01,
):
    env = safety_gymnasium.make(env_id, render_mode="human")
    controller = Button0Controller()

    for ep in range(episodes):
        obs, info = env.reset()

        ep_reward = 0.0
        ep_cost = 0.0

        print(f"\nEpisode {ep + 1}")
        print("obs shape:", obs.shape)
        print("action space:", env.action_space)

        for step in range(max_steps):
            action = controller.act(obs)

            obs, reward, cost, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_cost += cost

            env.render()

            if sleep > 0:
                time.sleep(sleep)

            if terminated or truncated:
                break

        print(f"Episode reward: {ep_reward:.3f}")
        print(f"Episode cost:   {ep_cost:.3f}")
        print(f"Steps:          {step + 1}")

    env.close()


if __name__ == "__main__":
    render_button0_controller()