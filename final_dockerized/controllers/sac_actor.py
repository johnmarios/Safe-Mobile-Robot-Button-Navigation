import numpy as np
import torch
import torch.nn as nn


DEVICE = torch.device("cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mu_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

        self.max_action = max_action

    def select_action(self, state):
        state = np.asarray(state, dtype=np.float32).reshape(1, -1)
        state = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            x = torch.relu(self.l1(state))
            x = torch.relu(self.l2(x))
            action = self.max_action * torch.tanh(self.mu_layer(x))

        return action.cpu().numpy().reshape(-1)
