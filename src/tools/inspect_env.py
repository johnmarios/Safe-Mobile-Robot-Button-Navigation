# src/tools/inspect_env.py

import argparse
import os
import sys
import time
import numpy as np


def add_project_root_to_path():
    """
    Επιτρέπει να τρέχεις το αρχείο από οπουδήποτε μέσα στο project.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


add_project_root_to_path()


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def short_value(value, max_len=300):
    text = repr(value)
    if len(text) > max_len:
        return text[:max_len] + " ..."
    return text


def unwrap_env(env):
    """
    Επιστρέφει όλα τα wrappers μέχρι το καθαρό env.
    """
    wrappers = []
    current = env

    while True:
        wrappers.append(type(current).__name__)

        if not hasattr(current, "env"):
            break

        next_env = current.env
        if next_env is current:
            break

        current = next_env

    return wrappers, current


def describe_space(space, name):
    print_section(f"{name} SPACE")

    print(f"Type: {type(space)}")
    print(f"Space: {space}")

    if hasattr(space, "shape"):
        print(f"Shape: {space.shape}")

    if hasattr(space, "dtype"):
        print(f"Dtype: {space.dtype}")

    if hasattr(space, "low") and hasattr(space, "high"):
        print(f"Low:  {space.low}")
        print(f"High: {space.high}")

        low = np.array(space.low)
        high = np.array(space.high)

        print(f"Low min/max:  {np.nanmin(low)} / {np.nanmax(low)}")
        print(f"High min/max: {np.nanmin(high)} / {np.nanmax(high)}")

    try:
        sample = space.sample()
        print(f"Sample: {sample}")
        print(f"Sample shape: {np.array(sample).shape}")
    except Exception as e:
        print(f"Could not sample from space: {e}")


def safe_reset(env, seed=0):
    """
    Υποστηρίζει και gymnasium API:
        obs, info = env.reset(seed=seed)

    Προσοχή: όχι env.reset(seed) γιατί μπορεί να βγάλει:
    TimeLimit.reset() takes 1 positional argument but 2 were given
    """
    try:
        result = env.reset(seed=seed)
    except TypeError:
        result = env.reset()

    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
    else:
        obs = result
        info = {}

    return obs, info


def safe_step(env, action):
    """
    Safety-Gymnasium συνήθως:
        obs, reward, cost, terminated, truncated, info = env.step(action)

    Gymnasium κλασικά:
        obs, reward, terminated, truncated, info = env.step(action)
    """
    result = env.step(action)

    if len(result) == 6:
        obs, reward, cost, terminated, truncated, info = result
    elif len(result) == 5:
        obs, reward, terminated, truncated, info = result
        cost = info.get("cost", None)
    elif len(result) == 4:
        obs, reward, done, info = result
        cost = info.get("cost", None)
        terminated = done
        truncated = False
    else:
        raise RuntimeError(f"Unknown step output length: {len(result)}")

    return obs, reward, cost, terminated, truncated, info


def print_obs_stats(obs, title="OBSERVATION"):
    print_section(title)

    obs_arr = np.array(obs)

    print(f"Type: {type(obs)}")
    print(f"Shape: {obs_arr.shape}")
    print(f"Dtype: {obs_arr.dtype}")

    if obs_arr.size > 0:
        print(f"Min:  {np.nanmin(obs_arr)}")
        print(f"Max:  {np.nanmax(obs_arr)}")
        print(f"Mean: {np.nanmean(obs_arr)}")
        print(f"Std:  {np.nanstd(obs_arr)}")

    print(f"Values:\n{obs_arr}")


def print_info_dict(info, title="INFO"):
    print_section(title)

    if not info:
        print("Info is empty.")
        return

    for key, value in info.items():
        print(f"{key}: {short_value(value)}")


def print_public_attributes(obj, title, max_items=80):
    print_section(title)

    attrs = []
    for name in dir(obj):
        if name.startswith("_"):
            continue

        try:
            value = getattr(obj, name)
        except Exception:
            continue

        if callable(value):
            continue

        attrs.append((name, value))

    for i, (name, value) in enumerate(attrs[:max_items]):
        print(f"{name}: {short_value(value)}")

    if len(attrs) > max_items:
        print(f"... showing {max_items}/{len(attrs)} attributes")


def find_possible_attrs(obj, names):
    found = {}

    for name in names:
        if hasattr(obj, name):
            try:
                found[name] = getattr(obj, name)
            except Exception as e:
                found[name] = f"Could not read: {e}"

    return found


def inspect_mujoco(unwrapped):
    print_section("MUJOCO MODEL / DATA")

    possible_model_attrs = ["model", "data", "sim", "robot", "task", "world", "placements", "layout"]

    found = find_possible_attrs(unwrapped, possible_model_attrs)

    if not found:
        print("No obvious MuJoCo/Safety-Gymnasium internals found on unwrapped env.")
        return

    for name, value in found.items():
        print(f"{name}: {type(value)} -> {short_value(value)}")

    model = getattr(unwrapped, "model", None)
    data = getattr(unwrapped, "data", None)

    if model is not None:
        print_section("MUJOCO MODEL DETAILS")

        for attr in [
            "nq",
            "nv",
            "nu",
            "nbody",
            "njnt",
            "ngeom",
            "nsite",
            "ncam",
            "nlight",
            "na",
        ]:
            if hasattr(model, attr):
                print(f"{attr}: {getattr(model, attr)}")

        # Names are useful to see robot bodies, buttons, hazards, geoms, etc.
        for attr in ["body_names", "joint_names", "geom_names", "site_names", "actuator_names"]:
            if hasattr(model, attr):
                print(f"{attr}: {short_value(getattr(model, attr), max_len=1000)}")

    if data is not None:
        print_section("MUJOCO DATA DETAILS")

        for attr in ["qpos", "qvel", "ctrl", "sensordata"]:
            if hasattr(data, attr):
                value = np.array(getattr(data, attr))
                print(f"{attr}: shape={value.shape}, values={value}")


def inspect_robot_and_task(unwrapped):
    print_section("ROBOT / TASK DETAILS")

    robot = getattr(unwrapped, "robot", None)
    task = getattr(unwrapped, "task", None)

    if robot is None:
        print("No direct `robot` attribute found.")
    else:
        print(f"Robot type: {type(robot)}")
        print_public_attributes(robot, "ROBOT PUBLIC ATTRIBUTES", max_items=100)

    if task is None:
        print("No direct `task` attribute found.")
    else:
        print(f"Task type: {type(task)}")
        print_public_attributes(task, "TASK PUBLIC ATTRIBUTES", max_items=100)


def run_random_rollout(env, steps=20, render=False, sleep=0.02):
    print_section("RANDOM ROLLOUT CHECK")

    obs, info = safe_reset(env, seed=0)

    print_obs_stats(obs, "INITIAL OBSERVATION")
    print_info_dict(info, "RESET INFO")

    total_reward = 0.0
    total_cost = 0.0

    for step in range(steps):
        action = env.action_space.sample()

        obs, reward, cost, terminated, truncated, info = safe_step(env, action)

        total_reward += float(reward)

        if cost is not None:
            total_cost += float(cost)

        print("\n" + "-" * 80)
        print(f"Step: {step}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        print(f"Cost: {cost}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")

        obs_arr = np.array(obs)
        print(f"Obs shape: {obs_arr.shape}")
        print(f"Obs min/max/mean: {np.nanmin(obs_arr)} / {np.nanmax(obs_arr)} / {np.nanmean(obs_arr)}")

        important_info = {
            k: v
            for k, v in info.items()
            if any(word in k.lower() for word in ["cost", "goal", "button", "hazard", "vase", "gremlin", "x", "y", "pos"])
        }

        if important_info:
            print("Important info:")
            for key, value in important_info.items():
                print(f"  {key}: {short_value(value)}")

        if render:
            try:
                env.render()
                time.sleep(sleep)
            except Exception as e:
                print(f"Render failed: {e}")
                render = False

        if terminated or truncated:
            print("Episode ended. Resetting...")
            obs, info = safe_reset(env, seed=step + 1)

    print_section("ROLLOUT SUMMARY")
    print(f"Total reward over {steps} random steps: {total_reward}")
    print(f"Total cost over {steps} random steps: {total_cost}")


def check_action_scaling(env):
    print_section("ACTION SCALING CHECK")

    action_space = env.action_space

    if not hasattr(action_space, "low") or not hasattr(action_space, "high"):
        print("Action space does not have low/high.")
        return

    low = np.array(action_space.low)
    high = np.array(action_space.high)

    print(f"Action low:  {low}")
    print(f"Action high: {high}")

    zero_action = np.zeros_like(low)
    mid_action = (low + high) / 2.0

    print(f"Zero action valid? {action_space.contains(zero_action.astype(action_space.dtype))}")
    print(f"Middle action: {mid_action}")
    print(f"Middle action valid? {action_space.contains(mid_action.astype(action_space.dtype))}")

    print(
        """
Για TD3/SAC actor με tanh output στο [-1, 1], σωστό scaling είναι:

real_action = low + (tanh_action + 1.0) * 0.5 * (high - low)

και όχι απλά action * max_action, εκτός αν το action space είναι συμμετρικό.
"""
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        type=str,
        default="SafetyRacecarButton2-v0",
        help="Environment id, π.χ. SafetyRacecarButton2-v0",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Πόσα random steps να κάνει για έλεγχο.",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Άνοιγμα render κατά το random rollout.",
    )

    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Προαιρετικό camera/render mode αν το χρειάζεσαι.",
    )

    args = parser.parse_args()

    import safety_gymnasium

    print_section("CREATE ENV")

    print(f"Environment id: {args.env}")

    # Για απλό inspection καλύτερα χωρίς vector env.
    if args.render:
        try:
            env = safety_gymnasium.make(args.env, render_mode="human")
        except TypeError:
            env = safety_gymnasium.make(args.env)
    else:
        env = safety_gymnasium.make(args.env)

    print(f"Env type: {type(env)}")

    wrappers, unwrapped = unwrap_env(env)

    print_section("WRAPPERS")
    for i, wrapper in enumerate(wrappers):
        print(f"{i}: {wrapper}")

    print(f"\nUnwrapped env type: {type(unwrapped)}")

    describe_space(env.observation_space, "OBSERVATION")
    describe_space(env.action_space, "ACTION")

    check_action_scaling(env)

    print_public_attributes(unwrapped, "UNWRAPPED ENV PUBLIC ATTRIBUTES", max_items=100)

    inspect_robot_and_task(unwrapped)
    inspect_mujoco(unwrapped)

    run_random_rollout(env, steps=args.steps, render=args.render)

    env.close()


if __name__ == "__main__":
    main()