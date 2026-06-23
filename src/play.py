import torch
import safety_gymnasium

from train.sac import SAC
from render.render_sac import render_policy
from config import *

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
    SAC_MODEL_PATH + "sac_ms2000_ar2_c005_to_c01_em_3to1_25_continue" + "_latest"
)

# agent.load(
#     SAC_MODEL_PATH + "sac_ms2000_ar2_c0_phase_2b_c001" + "_latest"
# )
print("Phase 0 model loaded successfully.")
print("Model loaded successfully.")

# Watch policy
render_policy(
    agent,
    ENV_NAME,
    episodes=20
)