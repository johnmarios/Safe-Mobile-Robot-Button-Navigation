import os

import numpy as np 
import matplotlib.pyplot as plt 
import copy
import random
from collections import deque, Counter
import torch 
import torch.nn as nn
import torch.nn.functional as F

from .actor import Actor
from .critic import Critic
from .replaybuffer import ReplayBuffer

class SAC(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        discount=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
    ):

        self.device = device

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.replay_buffer = ReplayBuffer(state_dim, action_dim)


        self.target_entropy = -2 * action_dim
        self.gamma = discount
        self.tau = tau
        self.total_it = 0

        self.log_alpha = torch.zeros(
            1,
            requires_grad=True,
            device=device
        )

        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=3e-4
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def Q_target(self, next_state, reward, not_done):

        with torch.no_grad():

            next_action, next_log_prob = self.actor(next_state)

            target_Q1, target_Q2 = self.critic_target(
                next_state,
                next_action
            )

            target_Q = torch.min(target_Q1,target_Q2)

            target_Q = reward + self.gamma * not_done * (target_Q - self.alpha.detach() * next_log_prob)

        return target_Q

    
    def critic_loss(self, target_Q, state, action):

        current_Q1, current_Q2 = self.critic(state, action)

        loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        return loss
    
    def actor_loss(self,state):

        action, log_prob = self.actor(state)

        q1,q2 = self.critic(state,action)

        q = torch.min(q1,q2)

        loss = (self.alpha.detach()*log_prob-q).mean()

        return loss, log_prob
    
    def alpha_loss(self, log_prob):

        loss = -(self.log_alpha *(log_prob.detach() + self.target_entropy)).mean()

        return loss
    
    def select_action(self, state):
        return self.actor.select_action(state)
    
    def train(self, batch_size=256):
        self.total_it += 1

        # Sample from replay buffer
        state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)

        # Get the Q target
        
        target_Q = self.Q_target(next_state, reward, not_done)

        # Get the critic loss
        critic_loss = self.critic_loss(target_Q, state, action)

        # Optimize the critic
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

       

        # Compute actor loss
        for param in self.critic.parameters():
            param.requires_grad = False

        self.actor_optimizer.zero_grad()

        actor_loss, log_prob = self.actor_loss(state)
            
        # Optimize the actor 
        actor_loss.backward()
        self.actor_optimizer.step()

        for param in self.critic.parameters():
            param.requires_grad = True

        # Compute alpha loss
        alpha_loss = self.alpha_loss(log_prob)

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()


         # Update the frozen target models
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item(), alpha_loss.item(), self.alpha.item()


    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        torch.save(
            self.actor.state_dict(),
            filename + "_actor.pth"
        )

        torch.save(
            self.critic.state_dict(),
            filename + "_critic.pth"
        )

        torch.save(
            self.actor_optimizer.state_dict(),
            filename + "_actor_optimizer.pth"
        )

        torch.save(
            self.critic_optimizer.state_dict(),
            filename + "_critic_optimizer.pth"
        )

        torch.save(
            self.alpha_optimizer.state_dict(),
            filename + "_alpha_optimizer.pth"
        )

        torch.save(
            self.log_alpha.detach(),
            filename + "_log_alpha.pth"
        )

 

    def load(self, filename):

        self.actor.load_state_dict(
            torch.load(
                filename+"_actor.pth",
                map_location=self.device
            )
        )

        self.critic.load_state_dict(
            torch.load(
                filename + "_critic.pth",
                map_location=self.device
            )
        )

        self.critic_target.load_state_dict(
            self.critic.state_dict()
        )

        self.actor_optimizer.load_state_dict(
            torch.load(
                filename + "_actor_optimizer.pth",
                map_location=self.device
            )
        )

        self.critic_optimizer.load_state_dict(
            torch.load(
                filename + "_critic_optimizer.pth",
                map_location=self.device
            )
        )

        self.alpha_optimizer.load_state_dict(
            torch.load(
                filename + "_alpha_optimizer.pth",
                map_location=self.device
            )
        )

        loaded_alpha = torch.load(
            filename + "_log_alpha.pth",
            map_location=self.device
        )

        self.log_alpha.data.copy_(loaded_alpha)

