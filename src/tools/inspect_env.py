import os
import sys
import numpy as np
import safety_gymnasium

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
train_dir = os.path.join(src_dir, "train")

if src_dir not in sys.path:
    sys.path.append(src_dir)

if train_dir not in sys.path:
    sys.path.append(train_dir)

from train.normalize import NormalizeActionWrapper


ENV_ID = "SafetyRacecarButton0-v0"


def unwrap_env(env):
    current = env
    while hasattr(current, "env"):
        current = current.env
    return current


def print_attrs(obj, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for name in dir(obj):
        if name.startswith("_"):
            continue

        try:
            value = getattr(obj, name)
        except Exception:
            continue

        if callable(value):
            continue

        text = repr(value)
        if len(text) > 300:
            text = text[:300] + " ..."

        print(f"{name}: {text}")


def main():
    env = safety_gymnasium.make(ENV_ID, render_mode="human")
    env = NormalizeActionWrapper(env)

    state, info = env.reset(seed=42)

    unwrapped = unwrap_env(env)

    print("Env:", ENV_ID)
    print("Observation shape:", np.array(state).shape)
    print("Observation:", state)
    print("Reset info:", info)

    print_attrs(unwrapped, "UNWRAPPED ENV ATTRIBUTES")

    if hasattr(unwrapped, "task"):
        print_attrs(unwrapped.task, "TASK ATTRIBUTES")

    if hasattr(unwrapped, "robot"):
        print_attrs(unwrapped.robot, "ROBOT ATTRIBUTES")

    model = getattr(unwrapped, "model", None)
    data = getattr(unwrapped, "data", None)

    if model is None or data is None:
        print("No model/data found.")
        return

    try:
        import mujoco

        print("\n" + "=" * 80)
        print("BODIES")
        print("=" * 80)

        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name is not None:
                pos = data.xpos[i]
                print(f"body[{i}] {name:40s} pos={pos}")

        print("\n" + "=" * 80)
        print("GEOMS")
        print("=" * 80)

        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name is not None:
                pos = data.geom_xpos[i]
                print(f"geom[{i}] {name:40s} pos={pos}")

        print("\n" + "=" * 80)
        print("SITES")
        print("=" * 80)

        for i in range(model.nsite):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            if name is not None:
                pos = data.site_xpos[i]
                print(f"site[{i}] {name:40s} pos={pos}")

    except Exception as e:
        print("Could not inspect MuJoCo names:", e)

    env.close()


if __name__ == "__main__":
    main()