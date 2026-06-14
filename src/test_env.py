# import safety_gymnasium

# env = safety_gymnasium.make("SafetyRacecarButton2-v0")

# obs, info = env.reset(seed=0)
# terminated, truncated = False, False

# total_reward = 0.0
# total_cost = 0.0

# print(type(obs))
# print(obs.shape)
# print(obs)

# while not (terminated or truncated):

#     # Replace this with the team’s controller.
#     action = env.action_space.sample()
#     obs, reward, cost, terminated, truncated, info = env.step(
#     action)
#     total_reward += reward
#     total_cost += cost
#     print(obs)
#     print(info)
#     print(info.keys())
# print("Episode reward:", total_reward)
# print("Episode cost:", total_cost)


# env.close()

import safety_gymnasium
from config import ENV_NAME, SEED

# env = safety_gymnasium.make("SafetyRacecarButton2-v0")
# obs, info = env.reset()


# print(env.unwrapped)
# print(dir(env.unwrapped.task))

# print(env.unwrapped.task.__dict__.keys())

# print(env.unwrapped.task.goal)
# print(env.unwrapped.task.buttons)

# print("==============================")
# print(env.unwrapped.task.buttons.goal_button)
# print(env.unwrapped.task.goal)
# print(env.unwrapped.task.dist_goal)
# print(env.unwrapped.task.goal_achieved)


env = safety_gymnasium.make(ENV_NAME)

builder = env.unwrapped

print()
print(builder.obs_space_dict)

print()
print(builder.obs_space_dict.spaces.keys())



# print(type(env))
# print()

# current = env
# i = 0

# while hasattr(current, "env"):
#     print(f"Level {i}: {type(current)}")
#     current = current.env
#     i += 1

# print(f"Final level: {type(current)}")

# print()
# print(dir(current))