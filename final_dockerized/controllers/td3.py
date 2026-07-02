from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device("cpu")


def load_torch_file(path):
    return torch.load(Path(path), map_location=DEVICE, weights_only=False)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return torch.tanh(self.output(x))
