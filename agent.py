import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque

from model import Actor, Critic

import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 256
MEMORY_SIZE = 150000
GAMMA = 0.9
TAU = 1e-3
RATE_ACTOR = 1e-3
RATE_CRITIC = 1e-3
SIGMA = 0.02


class Agent:
    def __init__(self, state_size, action_size, random_seed):        
        '''
        
        Parameters
        ----------
        state_size : length of state vector
        action_size : length of action vector
        random_seed : random seed, used for ReplayBuffer as well as for noise

        Returns
        -------
        None.

        '''
        self.state_size = state_size
        self.action_size = action_size
        self.learning_step = 0
        
        self.actor_local  = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), RATE_ACTOR)
        
        self.critic_local  = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), RATE_CRITIC)
        
        self.memory = ReplayBuffer(MEMORY_SIZE, BATCH_SIZE, random_seed)
        self.noise = OUNoise(action_size, random_seed, SIGMA)

    def step(self, state, action, reward, next_state, done):
        '''
        
        Parameters
        ----------
        state : current state
        action : action that was used
        reward : reward received for using action in state
        next_state : next state for using action in state
        done : is environment solved?

        '''
        self.memory.add(state, action, reward, next_state, done)
        
        if len(self.memory) >= BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
        
    def act(self, state, noise=True):
        '''

        Parameters
        ----------
        state : current state
        noise : use noise in order to explore the action state

        '''
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():    
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if noise:
            action += self.noise.sample()
        
        return np.clip(action, -1, 1)
        
        
    def reset(self):
        self.noise.reset()
        
    def learn(self, experiences, gamma):
        '''
        
        Parameters
        ----------
        experiences : Batch of experiences used for learning
        gamma : Discount Factor

        Returns
        -------
        None.

        '''
        # get experiences
        states, actions, rewards, next_states, dones = experiences
        self.learning_step += 1
        
        # update critic
        next_actions = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, next_actions)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # update actor
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # update target networks
        if self.learning_step == 1:
            # for first step, copy complete network
            self.soft_update(self.critic_local, self.critic_target, 1)
            self.soft_update(self.actor_local, self.actor_target, 1)
        else:
            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actor_local,  self.actor_target,  TAU)
        
        
    def soft_update(self, local, target, tau):
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer():
    # Simple ReplayBuffer Class
    
    def __init__(self, limit, sample_size, seed):
        '''
        
        Parameters
        ----------
        limit : maximum number of saved experiences
        sample_size : number of experiences sampled for each sample call
        seed : random seed for sampling

        Returns
        -------
        None.

        '''
        self.limit = limit
        self.sample_size = sample_size
        self.memory = deque(maxlen=limit)
        self.experience = namedtuple('experience', 'state action reward next_state done')
        self.seed = random.seed(seed)
        
        
    def add(self, state, action, reward, next_state, done):
        # simply adds an experience to this ReplayBuffer
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
        
    def sample(self):
        # Samples an experience from the ReplayBuffer
        sample = random.sample(self.memory, k = self.sample_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in sample if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in sample if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in sample if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in sample if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in sample if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)
        
    
class OUNoise:
    # Simple class for Ornstein Uhlenbeck Noise signal creation
    # based on https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py

    def __init__(self, size, seed, sigma=0.1, mu=0., theta=0.15):
        '''
        
        Parameters
        ----------
        size : size of created noise vector
        seed : random seed
        sigma : sigma of noise signal. DEFAULT=0.1
        mu : mean value of noise signal. DEFAULT=0
        theta : mean reversion rate. DEFAULT=0.15

        Returns
        -------
        None.

        '''
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        # reset current noise state to mean value (mu)
        self.state = copy.copy(self.mu)

    def sample(self):
        # provide noise sample
        x1 = self.state
        x2 = self.theta * (self.mu - x1) + self.sigma * np.array([random.random() for i in range(len(x1))])
        self.state = x1 + x2
        return self.state