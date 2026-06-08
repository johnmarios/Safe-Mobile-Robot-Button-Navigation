import safety_gymnasium
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ---------------- PARAMETERS ----------------

CHECKPOINTS = [
    610000,
    620000,
    630000,
    640000,
    650000,
    660000,
    670000,
    680000,
    690000,
    700000,
    710000,
    720000,
    730000,
    740000,
    750000
]

SUFFIX = "_Lrate1e-5"
N_EPISODES = 50

RESULTS_FILE = "evaluation_results.txt"

# --------------------------------------------


# Wrapper to move cost into info
class CostWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)

        info["cost"] = cost

        return obs, reward, terminated, truncated, info


# Add section header to existing results file
with open(RESULTS_FILE, "a") as f:

    f.write("\n\n")
    f.write("#" * 60 + "\n")
    f.write(f"Evaluation for {SUFFIX}\n")
    f.write("#" * 60 + "\n")


results = {}

for checkpoint in CHECKPOINTS:

    print()
    print("=" * 50)
    print(f"Evaluating checkpoint {checkpoint}")
    print("=" * 50)

    # Create env
    env = DummyVecEnv([
        lambda: CostWrapper(
            safety_gymnasium.make(
                "SafetyRacecarButton2-v0"
                # render_mode="human"
            )
        )
    ])

    # Load normalization stats
    env = VecNormalize.load(
        f"vec_normalize_{checkpoint}{SUFFIX}.pkl",
        env
    )

    env.training = False
    env.norm_reward = False

    # Load model
    model = PPO.load(
        f"ppo_{checkpoint}{SUFFIX}"
    )

    successes = 0

    for ep in range(N_EPISODES):

        obs = env.reset()

        done = False
        success = False

        while not done:

            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, done, info = env.step(action)

            # Button pressed at least once
            if info[0].get("cost_buttons", 0) > 0:
                success = True

        if success:
            successes += 1

        print(
            f"Episode {ep+1:02d}/{N_EPISODES} : "
            f"{'SUCCESS' if success else 'FAIL'}"
        )

    success_rate = 100 * successes / N_EPISODES

    results[checkpoint] = success_rate

    print()
    print(
        f"Checkpoint {checkpoint}: "
        f"{successes}/{N_EPISODES} "
        f"({success_rate:.1f}%)"
    )

    # Save intermediate result immediately
    with open(RESULTS_FILE, "a") as f:

        f.write(
            f"Checkpoint {checkpoint}: "
            f"{successes}/{N_EPISODES} "
            f"({success_rate:.1f}%)\n"
        )

    env.close()


# ---------- SUMMARY ----------

print()
print("=" * 50)
print("FINAL RESULTS")
print("=" * 50)

for checkpoint, rate in results.items():

    print(
        f"{checkpoint:7d} : {rate:.1f}%"
    )

best_checkpoint = max(
    results,
    key=results.get
)

print()
print("=" * 50)
print(
    f"BEST CHECKPOINT = {best_checkpoint}"
)
print(
    f"SUCCESS RATE = {results[best_checkpoint]:.1f}%"
)
print("=" * 50)


# Save final summary
with open(RESULTS_FILE, "a") as f:

    f.write("\n")
    f.write("=" * 60 + "\n")
    f.write(f"FINAL RESULTS FOR {SUFFIX}\n")
    f.write("=" * 60 + "\n")

    for checkpoint, rate in results.items():

        f.write(
            f"{checkpoint:7d} : {rate:.1f}%\n"
        )

    f.write("\n")
    f.write("=" * 60 + "\n")
    f.write(
        f"BEST CHECKPOINT = {best_checkpoint}\n"
    )
    f.write(
        f"SUCCESS RATE = {results[best_checkpoint]:.1f}%\n"
    )
    f.write("=" * 60 + "\n")


print()
print(f"Results appended to {RESULTS_FILE}")

