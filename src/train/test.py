import argparse
import numpy as np
import safety_gymnasium

from normalize import NormalizeActionWrapper
from td3 import TD3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="SafetyRacecarButton0-v0")
    parser.add_argument("--model-prefix", type=str, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    env = safety_gymnasium.make(args.env_id)
    env = NormalizeActionWrapper(env)

    obs, info = env.reset(seed=0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Βάλε εδώ ίδιες τιμές με training.
    agent = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=1.0,
    )

    agent.load(args.model_prefix)

    actions = []
    real_actions = []
    rewards = []

    terminated = False
    truncated = False

    for t in range(args.steps):
        action = agent.select_action(np.array(obs))
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, env.action_space.low, env.action_space.high)

        real_action = env.denormalize(action)

        obs, reward, cost, terminated, truncated, info = env.step(action)

        actions.append(action)
        real_actions.append(real_action)
        rewards.append(float(reward))

        if terminated or truncated:
            break

    env.close()

    actions = np.array(actions)
    real_actions = np.array(real_actions)

    print("Normalized action stats")
    print("speed mean/abs/max:", actions[:, 0].mean(), np.abs(actions[:, 0]).mean(), np.abs(actions[:, 0]).max())
    print("steer mean/abs/max:", actions[:, 1].mean(), np.abs(actions[:, 1]).mean(), np.abs(actions[:, 1]).max())

    print()
    print("Real action stats")
    print("velocity mean/abs/max:", real_actions[:, 0].mean(), np.abs(real_actions[:, 0]).mean(), np.abs(real_actions[:, 0]).max())
    print("steering mean/abs/max:", real_actions[:, 1].mean(), np.abs(real_actions[:, 1]).mean(), np.abs(real_actions[:, 1]).max())

    print()
    print("Total reward:", sum(rewards))


if __name__ == "__main__":
    main()