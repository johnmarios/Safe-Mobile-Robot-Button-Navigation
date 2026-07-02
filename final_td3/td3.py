from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_torch_file(path):
    """load a checkpoint"""
    return torch.load(Path(path), map_location=DEVICE, weights_only=False)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, action_dim)

    def forward(self, state):
        state = F.relu(self.layer1(state))
        state = F.relu(self.layer2(state))
        return torch.tanh(self.output(state))


class Critic(nn.Module):
    """Two independent Q-networks"""

    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.q1_layer1 = nn.Linear(state_dim + action_dim, 256)
        self.q1_layer2 = nn.Linear(256, 256)
        self.q1_output = nn.Linear(256, 1)

        self.q2_layer1 = nn.Linear(state_dim + action_dim, 256)
        self.q2_layer2 = nn.Linear(256, 256)
        self.q2_output = nn.Linear(256, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)

        q1 = F.relu(self.q1_layer1(state_action))
        q1 = F.relu(self.q1_layer2(q1))
        q1 = self.q1_output(q1)

        q2 = F.relu(self.q2_layer1(state_action))
        q2 = F.relu(self.q2_layer2(q2))
        q2 = self.q2_output(q2)

        return q1, q2

    def q1(self, state, action):
        """Only Q1 is needed for the delayed actor update """
        state_action = torch.cat([state, action], dim=1)
        q1 = F.relu(self.q1_layer1(state_action))
        q1 = F.relu(self.q1_layer2(q1))
        return self.q1_output(q1)


def soft_update(source, target, tau):
    """target <- tau * source + (1 - tau) * target."""
    for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
        target_parameter.data.mul_(1.0 - tau)
        target_parameter.data.add_(tau * source_parameter.data)


class TD3:
    """TD3 implementation """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=3e-4,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
    ):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.updates = 0

    def select_action(self, state):
        """Return one deterministic action for an already normalized state """

        state = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).reshape(1, -1)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy().reshape(-1)

    def train(self, replay_buffer, batch_size, normalizer, cost_penalty):
        """Perform one critic update and, periodically, one actor/target update """

        self.updates += 1

        states, actions, next_states, rewards, costs, not_dones = replay_buffer.sample(batch_size)

        states = torch.as_tensor(normalizer.normalize(states), dtype=torch.float32, device=DEVICE)
        next_states = torch.as_tensor(normalizer.normalize(next_states), dtype=torch.float32, device=DEVICE)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=DEVICE)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=DEVICE)
        costs = torch.as_tensor(costs, dtype=torch.float32, device=DEVICE)
        not_dones = torch.as_tensor(not_dones, dtype=torch.float32, device=DEVICE)

        rewards = rewards - cost_penalty * costs

        # target-policy smoothing
        with torch.no_grad():
            noise = torch.randn_like(actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)

            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = rewards + not_dones * self.discount * torch.minimum(target_q1, target_q2)

        # Update both critics toward the same target.
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None

        # Delayed actor update and delayed target-network update.
        if self.updates % self.policy_delay == 0:
            actor_loss = -self.critic.q1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.actor, self.actor_target, self.tau)
            soft_update(self.critic, self.critic_target, self.tau)
            actor_loss = float(actor_loss.item())

        return float(critic_loss.item()), actor_loss

    def network_state_dict(self):
        """return only neural network weights for render/evaluation/branching"""
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }

    def state_dict(self):
        """Return a dictionary containing all the information needed to resume training."""

        saved = self.network_state_dict()
        saved["actor_optimizer"] = self.actor_optimizer.state_dict()
        saved["critic_optimizer"] = self.critic_optimizer.state_dict()
        saved["updates"] = self.updates
        return saved

    def load_networks(self, saved):
        """Load only the neural network weights for render/evaluation/branching"""
        self.actor.load_state_dict(saved["actor"])
        self.actor_target.load_state_dict(saved["actor_target"])
        self.critic.load_state_dict(saved["critic"])
        self.critic_target.load_state_dict(saved["critic_target"])

    def load_state_dict(self, saved):
        self.load_networks(saved)
        self.actor_optimizer.load_state_dict(saved["actor_optimizer"])
        self.critic_optimizer.load_state_dict(saved["critic_optimizer"])
        self.updates = int(saved.get("updates", saved.get("total_updates", 0)))
