import safety_gymnasium
import numpy as np


# environment creation
#env = safety_gymnasium.make("SafetyRacecarButton2-v0", render_mode="human") # rendering for visualization

env = safety_gymnasium.make("SafetyRacecarButton2-v0") # no rendering for faster training


print(env.spec.max_episode_steps)

MAX_EPISODE_STEPS = env.spec.max_episode_steps  # 1000 is default value of safety_gymnasium

num_episodes = 100

episode_rewards = []
episode_costs = []
successes = 0
step_count = 0

for episode in range(num_episodes):

    obs, info = env.reset()

    terminated = False
    truncated = False

    total_reward = 0
    total_cost = 0
    step_count = 0

    while not (terminated or truncated) and step_count < MAX_EPISODE_STEPS:

        # τυχαία δράση
        action = env.action_space.sample()

        obs, reward, cost, terminated, truncated, info = env.step(action)
        step_count += 1

        total_reward += reward
        total_cost += cost

    episode_rewards.append(total_reward)
    episode_costs.append(total_cost)

    # επιτυχία αν πήρε θετικό reward
    if total_reward > 0:
        successes += 1

    print(
        f"Episode {episode+1}: "
        f"Reward={total_reward:.2f}, "
        f"Cost={total_cost:.2f}, "
        f"Steps={step_count}"
    )

env.close()

print("\n===== RESULTS =====")
print(f"Average Reward: {np.mean(episode_rewards):.2f}")
print(f"Average Cost: {np.mean(episode_costs):.2f}")
print(f"Success Rate: {100*successes/num_episodes:.2f}%")