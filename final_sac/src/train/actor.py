import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

    
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
class Actor(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            hidden_dim=256,
            log_std_min=-20,
            log_std_max=2):

        super().__init__()

        # Shared hidden layers
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)

        # Mean head
        self.mu_layer = nn.Linear(hidden_dim, action_dim)

        # Log standard deviation head
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

        self.LOG_STD_MIN = log_std_min
        self.LOG_STD_MAX = log_std_max

    def forward(self, state):

        # Shared layers
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))

        # Mean
        mu = self.mu_layer(x)

        # Log std
        log_std = self.log_std_layer(x)

        # Prevent extremely large or small std

        # log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX) # Alternative way to clamp log_std using tanh

        log_std = torch.tanh(log_std)

        log_std = self.LOG_STD_MIN + \
                0.5*(self.LOG_STD_MAX-self.LOG_STD_MIN) * \
                (log_std + 1)

        # Standard deviation
        std = torch.exp(log_std)

        # Gaussian distribution
        dist = Normal(mu, std)

        # Reparameterization trick
        z = dist.rsample()

        # Squash to [-1,1]
        action = torch.tanh(z)

        # Scale to environment action range
        action_scaled = self.max_action * action

        # Log probability before correction
        log_prob = dist.log_prob(z)

        # Correction due to tanh transformation
        log_prob -= torch.log(
            1 - action.pow(2) + 1e-6
        )

        # Sum over action dimensions
        log_prob = log_prob.sum(
            dim=1,
            keepdim=True
        )

        return action_scaled, log_prob

    def select_action(self, state):
        """
        Deterministic action used during evaluation.
        """
    
        if len(state.shape) == 1:
            state = state.reshape(1, -1)

        state = torch.FloatTensor(state).to(device)

        with torch.no_grad():

            x = F.relu(self.l1(state))
            x = F.relu(self.l2(x))

            mu = self.mu_layer(x)

            action = self.max_action * torch.tanh(mu)
             
              # check for saturation due to tahn


            # print(
            #     "mu =", mu.cpu().numpy()[0],
            #     "action =", action.cpu().numpy()[0]
            # )

            #

        return action.cpu().numpy().flatten()