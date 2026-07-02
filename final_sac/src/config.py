#-------------------------------------------
# Phase 0 : learn to drive
#-------------------------------------------
ENV_NAME = "SafetyRacecarButton2-v0"

MAX_TIMESTEPS_0 = int(1e6)
START_TIMESTEPS_0 = int(5e5)
MAX_STEPS = 1000

BATCH_SIZE = 256

EVAL_FREQ = 50000
SAC_EVAL_EPISODES = 50

GAMMA_0 = 0.99
TAU_0 = 0.005

ACTOR_LR_0 = 3e-4
CRITIC_LR_0 = 3e-4
entropy_multiplier_0 = 1.0

COST_WEIGHT_0 = 0.0
SEED = 0

SAC_MODEL_PATH = "models/"

