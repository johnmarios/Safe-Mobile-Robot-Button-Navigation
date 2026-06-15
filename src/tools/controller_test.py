import argparse
import time
import numpy as np
import safety_gymnasium


def reset_env(env):
    result = env.reset()

    if isinstance(result, tuple):
        obs, info = result
    else:
        obs = result
        info = {}

    return obs, info


def step_env(env, action):
    result = env.step(action)

    if len(result) == 6:
        obs, reward, cost, terminated, truncated, info = result
    elif len(result) == 5:
        obs, reward, terminated, truncated, info = result
        cost = info.get("cost", 0.0)
    else:
        obs, reward, done, info = result
        cost = info.get("cost", 0.0)
        terminated = done
        truncated = False

    return obs, reward, cost, terminated, truncated, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="SafetyRacecarButton0-v0")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    env = safety_gymnasium.make(args.env, render_mode="human")

    print("Env:", args.env)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    obs, info = reset_env(env)

    total_reward = 0.0
    total_cost = 0.0

    for t in range(args.steps):
        # action[0] = speed
        # action[1] = steering

        speed = 20.0

        # δοκιμή: σχεδόν ευθεία κίνηση
        steering = 0.0

        # μετά από κάποια βήματα στρίβει λίγο
        if 200 < t < 400:
            steering = 0.65
        elif 400 <= t < 600:
            steering = -0.65
        else:
            steering = 0.0

        action = np.array([speed, steering], dtype=np.float32)

        action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, reward, cost, terminated, truncated, info = step_env(env, action)

        total_reward += float(reward)
        total_cost += float(cost)

        if t % 50 == 0:
            print(
                f"step={t}, reward={reward:.3f}, cost={cost}, "
                f"total_reward={total_reward:.3f}, total_cost={total_cost:.3f}"
            )

        env.render()
        time.sleep(0.01)

        if terminated or truncated:
            print("Episode ended.")
            print("terminated:", terminated)
            print("truncated:", truncated)
            break

    env.close()


if __name__ == "__main__":
    main()