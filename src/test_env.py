import safety_gymnasium

env = safety_gymnasium.make("SafetyRacecarButton2-v0")

obs, info = env.reset()

print(type(obs))
print(obs.shape)
print(obs)

env.close()