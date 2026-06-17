from train.train_0 import train_0

def main():
    # Phase 0: Basic navigation
    train_0(MAX_TIMESTEPS = int(5e5),
            START_TIMESTEPS = int(1e5),
            LOAD_MODEL = None,
            ENV_NAME = "SafetyRacecarButton2-v0",
            MAX_STEPS = 1000,
            BATCH_SIZE = 256,
            EVAL_FREQ = 50000,
            SAC_EVAL_EPISODES = 10,
            GAMMA = 0.99,
            TAU = 0.005,
            ACTOR_LR = 3e-4,
            CRITIC_LR = 3e-4,
            entropy_multiplier = 1.0,
            COST_WEIGHT = 0.0,
            SEED = 0,
            SAC_MODEL_PATH = "models/",
            AGENT_ID = "sac_phase_0",
            )


    # Phase 1: Hazard avoidance
    #main_sac_phase_1.main_1()

    # Phase 2: Hazard avoidance advanced


if __name__ == "__main__":
    main()