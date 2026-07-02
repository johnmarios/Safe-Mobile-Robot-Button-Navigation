from pathlib import Path

import numpy as np
import torch

from .common import EnvironmentInfo, flatten_observation
from .sac_actor import Actor, DEVICE


class Controller:
    DEFAULT_MODEL = Path(
        "/app/models/sac_ms_500_ar2_c0toc005_em1_mt1e6_st_5e4_best_actor.pth"
    )

    def __init__(self, model_path=None, action_repeat=None):
        self.info = EnvironmentInfo()
        self.action_repeat = action_repeat or 2

        model_path = Path(model_path or self.DEFAULT_MODEL)

        max_action = torch.as_tensor(
            self.info.action_high,
            dtype=torch.float32,
            device=DEVICE,
        )

        self.actor = Actor(
            self.info.state_dim,
            self.info.action_dim,
            max_action,
        ).to(DEVICE)

        self.actor.load_state_dict(
            torch.load(model_path, map_location=DEVICE, weights_only=False)
        )
        self.actor.eval()

        self.reset()

    def reset(self, seed=None):
        self.cached_action = None
        self.step = 0

    def act(self, observation):
        if self.cached_action is None or self.step % self.action_repeat == 0:
            state = flatten_observation(
                observation,
                self.info.observation_space,
            )
            self.cached_action = self.actor.select_action(state)

        action = np.clip(
            self.cached_action,
            self.info.action_low,
            self.info.action_high,
        ).astype(np.float32)

        self.step += 1

        return action, {
            "controller": "sac",
            "action_repeat": self.action_repeat,
        }
