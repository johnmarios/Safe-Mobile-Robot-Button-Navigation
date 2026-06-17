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

#AGENT_ID_0 = "sac_phase_0_v2"

AGENT_ID_0 = "sac_phase_0"
#-------------------------------------------
# Phase 0.5 : fixed a
#-------------------------------------------
ENV_NAME = "SafetyRacecarButton2-v0"

MAX_TIMESTEPS_0_PLUS = int(7e5)
START_TIMESTEPS_0_PLUS = 0
MAX_STEPS = 1000

BATCH_SIZE = 256

EVAL_FREQ = 50000
SAC_EVAL_EPISODES = 10

GAMMA_0_PLUS = 0.99
TAU_0_PLUS = 0.005

ACTOR_LR_0_PLUS = 3e-4
CRITIC_LR_0_PLUS = 3e-4
entropy_multiplier_0_PLUS = 2.0

COST_WEIGHT_0_PLUS = 0.0
SEED = 0

SAC_MODEL_PATH = "models/"

AGENT_ID_0_PLUS = "sac_phase_0_plus"

#-------------------------------------------
# Phase 1 : hazard avoidance
#-------------------------------------------
ENV_NAME = "SafetyRacecarButton2-v0"

MAX_TIMESTEPS_1 = int(1e5)
START_TIMESTEPS_1 = 0
MAX_STEPS = 1000

BATCH_SIZE = 256

EVAL_FREQ = 50000
SAC_EVAL_EPISODES = 10

GAMMA_1 = 0.99
TAU_1 = 0.005

ACTOR_LR_1 = 3e-4
CRITIC_LR_1 = 3e-4
entropy_multiplier_1 = 1.0

COST_WEIGHT_1 = 0.001
SEED = 0

SAC_MODEL_PATH = "models/"

AGENT_ID_1 = "sac_phase_1"



#-------------------------------------------
# Phase 2 : hazard avoidance advanced
#-------------------------------------------
ENV_NAME = "SafetyRacecarButton2-v0"

MAX_TIMESTEPS_2 = int(5e5)
START_TIMESTEPS_2 = 0
MAX_STEPS = 1000

BATCH_SIZE = 256

EVAL_FREQ = 50000
SAC_EVAL_EPISODES = 20

GAMMA_2 = 0.99
TAU_2 = 0.005

ACTOR_LR_2 = 3e-4
CRITIC_LR_2 = 3e-4
entropy_multiplier_2 = 1.0

COST_WEIGHT_2 = 0.002
SEED = 0

SAC_MODEL_PATH = "models/"

AGENT_ID_2 = "sac_phase_2"


#-------------------------------------------
# Phase TEST 
#-------------------------------------------
ENV_NAME = "SafetyRacecarButton2-v0"

MAX_TIMESTEPS_TEST = int(5000)
START_TIMESTEPS_TEST = int(2000)
MAX_STEPS = 1000

BATCH_SIZE = 256

EVAL_FREQ_TEST = 1000
SAC_EVAL_EPISODES_TEST = 1

GAMMA_0 = 0.99
TAU_0 = 0.005

ACTOR_LR_0 = 3e-4
CRITIC_LR_0 = 3e-4
entropy_multiplier_0 = 1.0

COST_WEIGHT_0 = 0.0
SEED = 0

SAC_MODEL_PATH = "models/"

AGENT_ID_TEST = "sac_phase_TEST"