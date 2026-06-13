from Project.src.train.normalize import NormalizeActionWrapper, sample_warmup_action
import safety_gymnasium
import numpy as np
from replayBuffer import ReplayBuffer
from td3 import TD3


ENV_ID = "SafetyRacecarButton2-v0"

MAX_TIMESTEPS = 100_000
START_TIMESTEPS = 50_000
BATCH_SIZE = 256
BUFFER_SIZE = int(1e6)
SEED = 42



def main():
    env = safety_gymnasium.make(ENV_ID,render_mode=None,)
    env = NormalizeActionWrapper(env)
    
    state, info = env.reset(seed=SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    replay_buffer = ReplayBuffer(state_dim=state_dim,action_dim=action_dim,max_size=BUFFER_SIZE,)

    agent = TD3(state_dim=state_dim,action_dim=action_dim,max_action=max_action,)

    episode_reward = 0.0
    episode_cost = 0.0
    episode_steps = 0
    episode_num = 1

    for t in range(MAX_TIMESTEPS):
        if t < START_TIMESTEPS:
            action = sample_warmup_action(env, action_dim)
        else:
            action = agent.select_action(state)

            # Exploration noise during training
            noise = 0.1 * max_action * np.random.randn(action_dim)
            action = action + noise
            action = action.clip(env.action_space.low,env.action_space.high,)

        next_state, reward, cost, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        replay_buffer.add(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            cost=cost,
            done=done,
        )

        state = next_state

        episode_reward += reward
        episode_cost += cost
        episode_steps += 1

        if t >= START_TIMESTEPS:
            losses = agent.train(replay_buffer=replay_buffer,batch_size=BATCH_SIZE,)

        if done:
            print(
                f"Episode {episode_num} | "
                f"steps={episode_steps} | "
                f"reward={episode_reward:.2f} | "
                f"cost={episode_cost:.2f}"
            )

            state, info = env.reset()

            episode_reward = 0.0
            episode_cost = 0.0
            episode_steps = 0
            episode_num += 1

    env.close()
    agent.save("models/td3_safety_racecar_button2")


if __name__ == "__main__":
    main()