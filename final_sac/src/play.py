import torch
import safety_gymnasium

from train.sac import SAC
from render.render_sac import render_policy
from render.render_sac_single_button import render_policy_single_b
from config import *

SAC_MODEL_PATH = "SAC_nav/models/"
# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Environment
env = safety_gymnasium.make(ENV_NAME)

#for debug
print(env.action_space.low)
print(env.action_space.high)
#


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = torch.FloatTensor(
        env.action_space.high
    ).to(device)


env.close()

# Agent
agent = SAC(
    state_dim=state_dim,
    action_dim=action_dim,
    max_action=max_action,
    device=device,
    discount=GAMMA_0,
    tau=TAU_0,
    actor_lr=ACTOR_LR_0,
    critic_lr=CRITIC_LR_0
)

# Load model
agent.load(
    SAC_MODEL_PATH + "sac_single_B_ar2_c0toc05_em1_rbs3e5_st5e4_mt1e6" + "_best"
)

# agent.load(
#     SAC_MODEL_PATH + "sac_ms_500_ar2_c0t005_em1_mt1e6_st_5e4_best"
# )
print("Phase 0 model loaded successfully.")
print("Model loaded successfully.")

# Watch policy
render_policy(
    agent,
    ENV_NAME,
    episodes=20,
    max_steps=500,
    action_repeat=2
)

render_policy_single_b(
    agent,
    ENV_NAME,
    episodes=20,
    max_steps=500,
    action_repeat=2
)