from pathlib import Path

import numpy as np
import torch

from .common import EnvironmentInfo, denormalize_action, flatten_observation
from .normalizer import MixedObservationNormalizer
from .td3 import Actor, DEVICE, load_torch_file


class Controller:
    DEFAULT_MODEL = Path("/app/models/td3_model_0.02.pt")

    def __init__(self, model_path=None, action_repeat=None):
        self.info = EnvironmentInfo()
        self.action_repeat = action_repeat or 2

        model_path = Path(model_path or self.DEFAULT_MODEL)
        saved = load_torch_file(model_path)

        self.normalizer = MixedObservationNormalizer(
            self.info.flat_observation_space
        )
        self.normalizer.load_state_dict(saved["observation_normalizer"])

        self.actor = Actor(
            self.info.state_dim,
            self.info.action_dim,
        ).to(DEVICE)

        self.actor.load_state_dict(saved["agent"]["actor"])
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
            state = self.normalizer.normalize(state)

            state = torch.as_tensor(
                state.reshape(1, -1),
                dtype=torch.float32,
                device=DEVICE,
            )

            with torch.no_grad():
                self.cached_action = self.actor(state).cpu().numpy().reshape(-1)

        action = denormalize_action(
            self.cached_action,
            self.info.action_space,
        ).astype(np.float32)

        self.step += 1

        return action, {
            "controller": "td3",
            "action_repeat": self.action_repeat,
        }
