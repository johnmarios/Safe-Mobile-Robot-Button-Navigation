
import time

import numpy as np
import safety_gymnasium

from controller2 import Button2Controller
ENV_ID = "SafetyRacecarButton2-v0"
SPEED_MIN = -20.0
SPEED_MAX = 20.0
STEER_LIMIT = 0.785
RENDER_EPISODES = 5
RENDER_MAX_STEPS = 1000
RENDER_SLEEP = 0.0  # seconds
RENDER_PRINT_EVERY = 50  # steps
SEED = 42

def get_policy_action_bounds(env):
    env_low = env.action_space.low.astype(np.float32)
    env_high = env.action_space.high.astype(np.float32)
    action_low = env_low.copy()
    action_high = env_high.copy()
    action_low[0] = max(env_low[0], SPEED_MIN)
    action_high[0] = min(env_high[0], SPEED_MAX)
    action_low[1] = max(env_low[1], -STEER_LIMIT)
    action_high[1] = min(env_high[1], STEER_LIMIT)
    return action_low, action_high


def render_button2_controller():
    env = safety_gymnasium.make(ENV_ID, render_mode="human")
    env.action_space.seed(SEED)
    action_low, action_high = get_policy_action_bounds(env)

    print("===== BUTTON2 CONTROLLER RENDER =====")
    print(f"experiment          : Ruled based controller ")
    print(f"env_id              : {ENV_ID}")
    print(f"used action low/high: {action_low} / {action_high}")
    print()

    for ep in range(RENDER_EPISODES):
        # Fresh controller per episode resets recovery/search internal state.
        controller = Button2Controller()
        obs, _ = env.reset(seed=SEED + ep)
        obs = obs.astype(np.float32)
        
        raw_reward_sum = 0.0
        cost_sum = 0.0
        min_goal_distance = float("inf")

        for step in range(RENDER_MAX_STEPS):
   
            action = np.clip(
                controller.act(obs), action_low, action_high
            ).astype(np.float32)

            obs, reward, cost, terminated, truncated, _ = env.step(action)
            obs = obs.astype(np.float32)
            raw_reward_sum += float(reward)
            cost_sum += float(cost)

            goal_dist = getattr(env.unwrapped, "last_dist_goal", None)
            if goal_dist is not None:
                min_goal_distance = min(min_goal_distance, float(goal_dist))

            if step % RENDER_PRINT_EVERY == 0:
                print(
                    f"episode={ep + 1} step={step:04d} action={action} "
                    f"reward={reward:.3f} cost={cost:.3f}"
                )

            env.render()
            if RENDER_SLEEP > 0:
                time.sleep(RENDER_SLEEP)

            if terminated or truncated:
                break

        min_goal_text = "n/a" if not np.isfinite(min_goal_distance) else f"{min_goal_distance:.3f}"
        print(
            f"[EP {ep + 1}] raw_reward={raw_reward_sum:.3f} "
            f"cost={cost_sum:.3f} min_goal_distance={min_goal_text} "
            f"steps={step + 1}"
        )

    env.close()


if __name__ == "__main__":
    render_button2_controller()
