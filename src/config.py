#-------------------------------------------
# Phase 0 : learn to drive
#-------------------------------------------
ENV_NAME = "SafetyRacecarButton2-v0"

MAX_TIMESTEPS_0 = int(1e6)
START_TIMESTEPS_0 = 20000
MAX_STEPS = 1000

BATCH_SIZE = 256

EVAL_FREQ = 50000
SAC_EVAL_EPISODES = 10

GAMMA_0 = 0.99
TAU_0 = 0.005

ACTOR_LR_0 = 3e-4
CRITIC_LR_0 = 3e-4

COST_WEIGHT_0 = 0.0
SEED = 0

SAC_MODEL_PATH = "models/sac"

AGENT_ID = "sac_agent_0"

#-------------------------------------------
# Phase 1 : hazard avoidance
#-------------------------------------------
ENV_NAME = "SafetyRacecarButton2-v0"

MAX_TIMESTEPS_1 = int(1e6)
START_TIMESTEPS_1 = 20000
MAX_STEPS = 1000

BATCH_SIZE = 256

EVAL_FREQ = 50000
SAC_EVAL_EPISODES = 10

GAMMA_1 = 0.99
TAU_1 = 0.005

ACTOR_LR_1 = 3e-4
CRITIC_LR_1 = 3e-4

COST_WEIGHT_1 = 0.001
SEED = 0

SAC_MODEL_PATH = "models/sac"

AGENT_ID = "sac_agent_1"



#-------------------------------------------
# Phase 2 : hazard avoidance advanced
#-------------------------------------------
ENV_NAME = "SafetyRacecarButton2-v0"

MAX_TIMESTEPS_2 = int(1e6)
START_TIMESTEPS_2 = 20000
MAX_STEPS = 1000

BATCH_SIZE = 256

EVAL_FREQ = 50000
SAC_EVAL_EPISODES = 10

GAMMA_2 = 0.99
TAU_2 = 0.005

ACTOR_LR_2 = 3e-4
CRITIC_LR_2 = 3e-4

COST_WEIGHT_2 = 0.01
SEED = 0

SAC_MODEL_PATH = "models/sac"

AGENT_ID = "sac_agent_2"