import numpy as np
import random 
import dqn

import torch 
import torch.nn as nn
import torch.optim as optim

from collections import namedtuple, deque

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 1e-4
UPDATE_EVERY = 4
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01


class CheckersAgent:
    """Interacts with and learns checkers game."""

    def __init__(self):
        """Initialize agent."""

        self.dqn_online = dqn.CheckersDQN()
        self.dqn_target = dqn.CheckersDQN()
        self.optimizer = optim.Adam(self.dqn_online.parameters(), lr=LR)

        # replay memory
        self.memory = dqn.ReplayMemory(BUFFER_SIZE)

        # Initialize time step
        self.t_step = 0  

    def step(self, state, action, reward, next_state, done):
        pass 

    def act(self, state, eps=EPSILON):
        pass 

    def learn(self, experiences, gamma=GAMMA):
        pass