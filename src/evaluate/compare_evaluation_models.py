import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from train.sac import SAC
import torch
import safety_gymnasium
from evaluate.evaluate_sac import evaluate_policy

#anyway tha trexeievaluation 100 fores gia kathe modelo kai th ata sugkrinei

def comp_e_models(MODELS,
                  ENV_NAME,
                  SAC_EVAL_EPISODES,
                  COST_WEIGHT
                  ):
    # MODELS = [
    #     "sac_phase_0",
    #     "sac_cost_001_phase_0",
    #     "sac_cost_003_phase_0"
    # ]

    ROOT = Path(__file__).resolve().parents[2]

    BASE_PATH = ROOT / "results"
    SAVE_PATH = BASE_PATH / "evaluation_comparisons"

    os.makedirs(SAVE_PATH, exist_ok=True)

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Environment
    env = safety_gymnasium.make(ENV_NAME)

    # Dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Agent
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        discount=0.99,
        tau=0.005,
        actor_lr=1e-4,
        critic_lr=1e-4,
        entropy_multiplier=1.0
    )



    for model in MODELS :

        agent.load(f"models/{model}")

        # Initial state
        state, info = env.reset(seed=0)

        eval_rewards = []
        eval_costs = []
        eval_rewards_history = []
        eval_costs_history = []

        best_score = -np.inf

            
        avg_reward, avg_cost, eval_rewards_history_r, eval_costs_history_r = evaluate_policy(
                    agent,
                    env_name=ENV_NAME,
                    eval_episodes=SAC_EVAL_EPISODES
                )

        eval_rewards.append(avg_reward)
        eval_costs.append(avg_cost)
        eval_rewards_history.append(eval_rewards_history_r)
        eval_costs_history.append(eval_costs_history_r)

        score = avg_reward - COST_WEIGHT * avg_cost
                
        print(
            "======================================"
        )

        print(
            f"Average reward: {avg_reward:.2f} | "
            f"Average cost: {avg_cost:.2f} | "
            f"Score: {score:.2f}"
        )


        