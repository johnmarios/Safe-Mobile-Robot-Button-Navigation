import numpy as np
import matplotlib.pyplot as plt
import safety_gymnasium
import torch
import os
import json

from final_td3.td3 import TD3, load_torch_file
from final_td3.normalizer import MixedObservationNormalizer
from final_td3.environment import make_env
from final_td3.controller2 import Button2Controller

from final_sac.src.train.sac import SAC

os.makedirs("comparison_results", exist_ok=True)

ENV_NAME = "SafetyRacecarButton2-v0"
N_EPISODES = 50

MODELS = {
    "Random": {
        "type": "random",
        "action_repeat": 1,
    },

    "Rule_Controller": {
        "type": "controller",
        "action_repeat": 1,
    },

    "TD3_0.02": {
        "type": "td3",
        "path": "project_td3/results/button2_td3_baseline_half_action_repeat_termination_continue_cost_0.02/best_model.pt",
        "action_repeat": 2,
    },

    "TD3_0.01": {
        "type": "td3",
        "path": "project_td3/results/c01/last_model.pt",
        "action_repeat": 2,
    },

    "TD3_0.005": {
        "type": "td3",
        "path": "project_td3/results/c005/last_model.pt",
        "action_repeat": 2,
    },

    "SAC_0.005": {
        "type": "sac",
        "path": "SAC_nav/models/sac_ms_500_ar2_c0toc005_em1_mt1e6_st_5e4_best",
        "action_repeat": 2,
    },

    "SAC_SB_0.005": {
        "type": "sac",
        "path": "SAC_nav\models\sac_single_B_ar2_c0toc05_em1_rbs3e5_st5e4_mt1e6_best",
        "action_repeat": 2,
    },

    "SAC_SB_0.01": {
        "type": "sac",
        "path": "SAC_nav\models\sac_single_B_ar2_c05toc1_em1_rbs3e5_st5e4_mt1e6_best",
        "action_repeat": 2,
    },
        "SAC_0.01_em_dec-3": {
            "type": "sac",
            "path": "SAC_nav\models\sac_ar2_c005_to_c01_em_3to1_25_rb_3e5_st5e4_best",
            "action_repeat": 1,
        },
}


# MODELS = {
#         "SAC_0.01_ar+em_dec": {
#         "type": "sac",
#         "path": "SAC_nav\models\sac_ar2_to_ar1_c01_em_2to1_25_rb_3e5_st5e4_best",
#         "action_repeat": 1,
#         },
#         "SAC_0.01_em_dec-3": {
#         "type": "sac",
#         "path": "SAC_nav\models\sac_ar2_c005_to_c01_em_3to1_25_rb_3e5_st5e4_best",
#         "action_repeat": 1,
#         },
#         "SAC_0.01_em_dec-2_con": {
#         "type": "sac",
#         "path": "SAC_nav\models\sac_ar2_c01_em_2to1_25_rb_3e5_st5e4_repeat1_best",
#         "action_repeat": 1,
#         },
#         "TD3_0.02_last": {
#         "type": "td3",
#         "path": "project_td3/results/button2_td3_baseline_half_action_repeat_termination_continue_cost_0.02/last_model.pt",
#         "action_repeat": 2,
#     },
 #   }
MAX_STEPS = 1000

SEED = 12345

class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, state):
        return self.action_space.sample()

def evaluate_policy(
        policy,
        env,
        episodes,
        max_steps=MAX_STEPS,
        action_repeat=1,
        td3=False,
        normalizer=None,
        controller=False,
):
    rewards = []
    costs = []
    lengths = []

    for ep in range(episodes):
        
        if controller:
            policy = Button2Controller()

        state, _ = env.reset(seed=SEED + ep)

        done = False
        ep_reward = 0.0
        ep_cost = 0.0
        ep_length = 0

        while not done and ep_length < max_steps:

            if td3:
                action = policy.select_action(
                    normalizer.normalize(state)
                )

            elif controller:
                action = policy.act(state)

            else:
                action = policy.select_action(state)
            for _ in range(action_repeat):

                if td3:
                    state, reward, terminated, truncated, info = env.step(
                        action
                    )
                    cost = float(info.get("cost", 0.0))

                else:
                    state, reward, cost, terminated, truncated, info = env.step(
                        action
                    )

                done = terminated or truncated

                ep_reward += float(reward)
                ep_cost += float(cost)
                ep_length += 1

                if done or ep_length >= max_steps:
                    break

        rewards.append(ep_reward)
        costs.append(ep_cost)
        lengths.append(ep_length)

        print(
            f"Episode {ep + 1}/{episodes} | "
            f"Reward: {ep_reward:.2f} | "
            f"Cost: {ep_cost:.2f} | "
            f"Length: {ep_length}"
        )

    return (
        np.array(rewards),
        np.array(costs),
        np.array(lengths),
    )

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    all_results = {}

    for name, cfg in MODELS.items():

        print("=" * 60)
        print(f"Evaluating {name}")
        print("=" * 60)

        if cfg["type"] == "td3":

            env = make_env(
                ENV_NAME,
                render_mode=None,
                goal_termination=False,
            )

            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]

            agent = TD3(
                state_dim,
                action_dim
            )

            normalizer = MixedObservationNormalizer(
                env.observation_space,
                clip=5.0
            )

            checkpoint = load_torch_file(
                cfg["path"]
            )

            agent.load_networks_from_state_dict(
                checkpoint["agent"]
            )

            normalizer.load_state_dict(
                checkpoint["observation_normalizer"]
            )

            rewards, costs, lengths = evaluate_policy(
                agent,
                env,
                N_EPISODES,
                action_repeat=cfg["action_repeat"],
                td3=True,
                normalizer=normalizer,
            )

            env.close()

        elif cfg["type"] == "sac":

            env = safety_gymnasium.make(
                ENV_NAME
            )

            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]

            max_action = torch.FloatTensor(
                env.action_space.high
            ).to(device)

            agent = SAC(
                state_dim,
                action_dim,
                max_action,
                device
            )

            agent.load(cfg["path"])

            rewards, costs, lengths = evaluate_policy(
                agent,
                env,
                N_EPISODES,
                action_repeat=cfg["action_repeat"],
                td3=False,
            )

            env.close()

        elif cfg["type"] == "controller":

            env = safety_gymnasium.make(
                ENV_NAME
            )

            controller = Button2Controller()

            rewards, costs, lengths = evaluate_policy(
                controller,
                env,
                N_EPISODES,
                action_repeat=cfg["action_repeat"],
                td3=False,
                controller=True,
            )

            env.close()


        elif cfg["type"] == "random":

            env = safety_gymnasium.make(ENV_NAME)

            random_policy = RandomPolicy(
                env.action_space
            )

            rewards, costs, lengths = evaluate_policy(
                random_policy,
                env,
                N_EPISODES,
                action_repeat=cfg["action_repeat"],
            )

            env.close()



        all_results[name] = {
            "rewards": rewards,
            "costs": costs,
            "lengths": lengths,
        }

    results = {}

    for name, res in all_results.items():
        results[name] = {
            "rewards": res["rewards"].tolist(),
            "costs": res["costs"].tolist(),
            "lengths": res["lengths"].tolist(),
        }

    with open("comparison_results/raw_results.json", "w") as f:
        json.dump(results, f, indent=4)

    

    names = []
    reward_means = []

    for name, res in all_results.items():
        names.append(name)
        reward_means.append(
            res["rewards"].mean()
        )

    plt.figure(figsize=(8,5))
    plt.bar(names, reward_means)
    plt.ylabel("Mean Reward")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        "comparison_results/mean_reward.png"
    )
    plt.close()

    names = []
    cost_means = []

    for name, res in all_results.items():
        names.append(name)
        cost_means.append(
            res["costs"].mean()
        )

    plt.figure(figsize=(8,5))
    plt.bar(names, cost_means)
    plt.ylabel("Mean Cost")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        "comparison_results/mean_cost.png"
    )
    plt.close()

    print("\n===== RESULTS =====")

    for name, res in all_results.items():

        print(name)
        print(
            f"Reward : "
            f"{res['rewards'].mean():.2f}"
            f" ± "
            f"{res['rewards'].std():.2f}"
        )

        print(
            f"Cost : "
            f"{res['costs'].mean():.2f}"
            f" ± "
            f"{res['costs'].std():.2f}"
        )

        print()
    


if __name__ == "__main__":
    main()