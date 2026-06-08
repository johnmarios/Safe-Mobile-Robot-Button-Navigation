ENV_ID = "SafetyRacecarButton2-v0"

# per episode 
MAX_STEPS = 1000

NUM_EPISODES = 2

# random seed for reproducibility
SEED = 42


#SAC


MAX_STEPS = 1000
NUM_EPISODES = 5
SEED = 42

# SAC settings
SAC_TOTAL_TIMESTEPS = 100_000

SAC_MODEL_PATH = "models/sac_safety_racecar_button2"
SAC_LOG_DIR = "logs/sac_safety_racecar_button2"

# Evaluation
SAC_EVAL_EPISODES = 5