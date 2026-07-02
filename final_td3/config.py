# Change this file when you want a different TD3 experiment.

ENV_ID = "SafetyRacecarButton2-v0"
RUN_NAME = "td3_cost_003_demo_training"

# "new"    : train a fresh TD3 policy after random warm-up.
# "branch" : load one saved policy, refill a new replay buffer, then train.
# "resume" : continue the same run from results/RUN_NAME/latest_full.pt.

LOAD_MODE = "branch"
SOURCE_RUN_NAME = "button2_td3_baseline_half_action_repeat_termination_continue_cost_0.02"
SOURCE_MODEL_NAME = "last_model.pt"

SEED = 42
TOTAL_STEPS = 1_500_000
ACTION_REPEAT = 2
GOAL_TERMINATION = True

RANDOM_STEPS = 50_000
REFILL_STEPS = 50_000
BUFFER_SIZE = 500_000
BATCH_SIZE = 256
EXPLORATION_NOISE = 0.10

COST_PENALTY = 0.03
BEST_COST_WEIGHT = 0.005

LEARNING_RATE = 3e-5
DISCOUNT = 0.99
TAU = 0.005
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_DELAY = 2

EVALUATE_EVERY = 25_000
EVALUATION_EPISODES = 10
SAVE_EVERY = 50_000
TENSORBOARD_EVERY = 1_000
NORMALIZER_CLIP = 5.0
