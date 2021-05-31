import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque
import random


def layer_init(layer):
    # used to init weights of hidden layers
    x = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(x)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, n_states, n_actions, seed):
        '''
        
        Parameters
        ----------
        n_states : number of states
        n_actions : number of actions
        seed : random seed, used for weight initialization

        Returns
        -------
        None.

        '''
        super(Actor, self).__init__()
        
        self.n_states = n_states
        self.n_actions = n_actions
        
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, n_actions)
        
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.reset_parameters()

    def reset_parameters(self):
        # initializes all the layers of this model
        self.fc1.weight.data.uniform_(*layer_init(self.fc1))
        self.fc2.weight.data.uniform_(*layer_init(self.fc2))
        self.fc3.weight.data.uniform_(*layer_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
        
        self.fc1.bias.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)
        self.fc3.bias.data.fill_(0.1)
        self.fc4.bias.data.fill_(0.1)
        
        
    def forward(self, x):
        # forward function for this model
        out = x
        out = F.relu(self.bn1(self.fc1(out)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = F.relu(self.bn3(self.fc3(out)))
        out = torch.tanh(self.fc4(out))
        
        return out
        
        
        
class Critic(nn.Module):
    def __init__(self, n_states, n_actions, seed):
        '''
        
        Parameters
        ----------
        n_states : number of states
        n_actions : number of actions
        seed : random seed

        Returns
        -------
        None.

        '''
        super(Critic, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128 + n_actions, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)
        
        self.reset_parameters()

    def reset_parameters(self):
        # initializes all the layers of this model
        self.fc1.weight.data.uniform_(*layer_init(self.fc1))
        self.fc2.weight.data.uniform_(*layer_init(self.fc2))
        self.fc3.weight.data.uniform_(*layer_init(self.fc3))
        self.fc4.weight.data.uniform_(*layer_init(self.fc4))
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)
        
        self.fc1.bias.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)
        self.fc3.bias.data.fill_(0.1)
        self.fc4.bias.data.fill_(0.1)
        self.fc5.bias.data.fill_(0.1)
        
        
    def forward(self, states, actions):
        # forward function
        states = F.selu(self.fc1(states))
        x = torch.cat((states, actions), dim=1)
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc3(x))
        x = F.selu(self.fc4(x))
        return  F.selu(self.fc5(x))
        
