import torch
import torch.nn as nn
import torch.nn.functional as F

# stochastic policy that output mean and variance prediction
# architecture of 1 hidden layer:
#             --> mean
#            /
# s --> h --
#           \
#            --> variance

class Actor(nn.Module):
    """
    network that decides which action to take given the current state. 
    It outputs the parameters of a probability distribution over actions 
    (mean and variance for a Gaussian policy).

    """

    def __init__(self, nS, nH, nA): 
        # nS: state space size, 
        # nH: n. of neurons in hidden layer, 
        # nA: size action space
        super().__init__()

        self.h = nn.Linear(nS, nH)
        # separate output layers for mean and variance
        self.out = nn.Linear(nH, nA) # mean for action distribution (which actions to take)
        self.out_sigma = nn.Linear(nH, nA) # variance for exploration
  

    def forward(self, x):
        x = F.relu(self.h(x)) # layer1
        mu = self.out(x)

        # constrain the variance with logvar bounds to avoid numerical issues
        log_sigma = self.out_sigma(x)
        max_logvar = 3. 
        min_logvar = -6.
        logvar = max_logvar - F.softplus(max_logvar - log_sigma)
        logvar = min_logvar + F.softplus(logvar - min_logvar)
        return mu, logvar



class Critic(nn.Module):
    """
    Q(s,a)

    """
    # Q-value network
    def __init__(self, nS, nH, nA): 
        # nS: state space size, 
        # nH: n. of neurons in hidden layer, 
        # nA: size action space
        super().__init__()

        # as inputs we have both state and action, which we concatenate and pass through the network
        self.h = nn.Linear(nS + nA, nH) # Q(s,a)
        self.out = nn.Linear(nH, 1)

    
    def forward(self, x, a):

        # concatenate  x.shape = (batch_size, nS), 
        # a.shape = (batch_size, nA) 
        # -> (batch_size, nS + nA)
        x = F.relu(self.h(torch.cat([x, a], dim=1)))

        # output is the Q-value for the given state-action pair
        # (batch_size, 1)
        return self.out(x)


class ActorCritic(object):
    """
    Implements the Actor-Critic algorithm

    """
    def __init__(self, state_dim, action_dim, max_action, actor_hidden = 256, critic_hidden = 256, gamma = 0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, actor_hidden, action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, critic_hidden, action_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=5e-3)

        self.gamma = gamma
        self.max_action = max_action

    def select_action(self, state):
        """
        Input: (state_dim,)

        Output: (action_dim,)

        """

        # network expects input of shape (batch_size, state_dim), so we add a batch dimension and convert to torch tensor

        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        # we don't need gradients for action selection
        with torch.no_grad(): 

            # get mean and log variance from the actor network
            mu, logvar = self.actor(state) 
            
            # convert log variance to standard deviation
            std = torch.exp(0.5 * logvar) 
            
            # sample from standard normal distribution (sampling noise for reparameterization)
            eps = torch.randn_like(std) 
            
            #N(0,1)
            action = mu + eps * std # we want also exploration, so we add noise to the action 

        action = action.clamp(-self.max_action, self.max_action)

        return action.cpu().numpy().flatten()


    def train_step(self, replay_buffer, batch_size=256):
        """
        Performs one gradient step for critic and one gradient step for actor.

        Returns a dict:
            {
                "critic_loss": float,
                "actor_loss": float
            }

        """

        state, action, next_state, reward, cost, not_done = replay_buffer.sample(batch_size)

        # device to work with GPU if available
        state = state.to(self.device) 

        action = action.to(self.device) 
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)
        cost = cost.to(self.device)
        not_done = not_done.to(self.device)

        with torch.no_grad(): # we don't need gradients for target Q value computation
            next_mu, next_logvar = self.actor(next_state)
            next_std = torch.exp(0.5 * next_logvar)
            next_eps = torch.randn_like(next_std)
            next_action = next_mu + next_eps * next_std
            next_action = next_action.clamp(-self.max_action, self.max_action)

            target_q = reward + self.gamma * not_done * self.critic(next_state, next_action)

        current_q = self.critic(state, action)

        # critic gets trained to minimize the MSE loss between current Q and target Q
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update 
        mu, logvar = self.actor(state)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sampled_action = mu + eps * std
        sampled_action = sampled_action.clamp(-self.max_action, self.max_action)

        # pytorch does minimization but we want to maximize the Q value
        actor_loss = -self.critic(state, sampled_action).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

