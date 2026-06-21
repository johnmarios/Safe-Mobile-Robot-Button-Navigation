from train.train_action_repeat import train_2


def main():
    train_2(MAX_TIMESTEPS = int(1e6),
                START_TIMESTEPS = int(1e5),
                LOAD_MODEL = None,
                LOAD_RESULTS_PATH = None,
                ENV_NAME = "SafetyRacecarButton2-v0",
                MAX_STEPS = 1000,
                BATCH_SIZE = 256,
                EVAL_FREQ = 25000,
                SAC_EVAL_EPISODES = 30,
                GAMMA = 0.99,
                TAU = 0.005,
                ACTOR_LR = 1e-4,
                CRITIC_LR = 1e-4,
                entropy_multiplier = 1.0,
                COST_WEIGHT = 0.0,
                TURNING_WEIGHT = 0.1,
                ACTION_REPEAT = 1,
                SEED = 0,
                SAC_MODEL_PATH = "models/",
                AGENT_ID = "sac_baseline_tw1",
                )



if __name__ == "__main__":
    main()