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

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

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
    SAC_MODEL_PATH + AGENT_ID_0 + "_best"
)

print("Phase 0 model loaded successfully.")
print("Model loaded successfully.")

# Watch policy
render_policy(
    agent,
    ENV_NAME,
    episodes=5
)