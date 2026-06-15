import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    """
    network that decides which action to take given the current state.
    
    """
    def __init__(self, state_dim, action_dim, nHidden=256):
        super().__init__()

        self.l1 = nn.Linear(state_dim, nHidden)
        self.l2 = nn.Linear(nHidden, nHidden)
        self.l3 = nn.Linear(nHidden, action_dim)


    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        # tanh to ensure the output is between [-1, 1],
        # which will be scaled by max_action to get the final action
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, nHidden=256):
        super().__init__()

        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, nHidden)
        self.l2 = nn.Linear(nHidden, nHidden)
        self.l3 = nn.Linear(nHidden, 1)

        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, nHidden)
        self.l5 = nn.Linear(nHidden, nHidden)
        self.l6 = nn.Linear(nHidden, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.08,
        noise_clip=0.15,
        policy_freq=2,
        learning_rate=3e-4,
    ):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=learning_rate,)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=learning_rate,)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        """
        Input : (state_dim,)

        Output: (action_dim,)

        """

        # network expects input of shape (batch_size, state_dim) so we add a batch dimension of 1

        state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        # we don't need gradients for action selection
        with torch.no_grad():
            action = self.actor(state)

        return action.cpu().numpy().flatten()

    def Q_target(self, next_state, reward, not_done):

        # next state comes from replay buffer 
        next_action = self.actor_target(next_state) 

        # add noise to the action for exploration
        noise = torch.randn_like(next_action) * self.policy_noise
        noise = noise.clamp(-self.noise_clip, self.noise_clip)

        next_action = next_action + noise
        next_action = next_action.clamp(-self.max_action, self.max_action)# a' 

        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)

        target_Q = reward + not_done * self.discount * target_Q

        return target_Q

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        state, action, next_state, reward, cost, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            target_Q = self.Q_target(next_state=next_state, reward=reward, not_done=not_done)

        current_Q1, current_Q2 = self.critic(state, action)

        critic_loss = F.mse_loss(current_Q1,target_Q,) + F.mse_loss(current_Q2,target_Q,)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_value = None

        if self.total_it % self.policy_freq == 0:
            # optimizer minimizes the negative Q value of the actor's action,
            #  which is equivalent to maximizing the Q value
            actor_loss = -self.critic.Q1(state,self.actor(state),).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_loss_value = actor_loss.item()

            self.soft_update(self.critic, self.critic_target)
            self.soft_update(self.actor, self.actor_target)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss_value,
        }

    def soft_update(self, network, target_network):
        """target network update: target = tau * network + (1 - tau) * target
           a little bit from current network and a more from target network to make the target network more stable"""
        for param, target_param in zip(network.parameters(),target_network.parameters(),):
            target_param.data.copy_(self.tau * param.data+ (1.0 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        torch.save(self.actor_target.state_dict(), filename + "_actor_target.pth")
        torch.save(self.critic_target.state_dict(), filename + "_critic_target.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location=device))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth", map_location=device))
        self.actor_target.load_state_dict(torch.load(filename + "_actor_target.pth", map_location=device))
        self.critic_target.load_state_dict(torch.load(filename + "_critic_target.pth", map_location=device))