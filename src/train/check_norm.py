import numpy as np
import safety_gymnasium

from normalize import NormalizeActionWrapper


ENV_ID = "SafetyRacecarButton0-v0"


def main():
    raw_env = safety_gymnasium.make(ENV_ID)

    print("RAW action space")
    print("low :", raw_env.action_space.low)
    print("high:", raw_env.action_space.high)

    env = NormalizeActionWrapper(raw_env)

    print()
    print("NORMALIZED action space")
    print("low :", env.action_space.low)
    print("high:", env.action_space.high)

    test_actions = [
        np.array([-1.0, -1.0], dtype=np.float32),
        np.array([-0.5, -0.5], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([0.5, 0.5], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([0.3, 0.8], dtype=np.float32),
        np.array([-0.3, -0.8], dtype=np.float32),
    ]

    print()
    print("DENORMALIZATION TEST")
    for a in test_actions:
        real = env.denormalize(a)
        print(f"normalized={a} -> real={real}")
        print()
    print("ROUND TRIP TEST")
    raw_low = raw_env.action_space.low
    raw_high = raw_env.action_space.high

    for a in test_actions:
        real = env.denormalize(a)

        renormalized = 2.0 * (real - raw_low) / (raw_high - raw_low) - 1.0
        print(f"a={a} -> real={real} -> normalized_again={renormalized}")

    env.close()


if __name__ == "__main__":
    main()