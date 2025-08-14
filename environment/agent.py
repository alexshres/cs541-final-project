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
UPDATE_EVERY = 4

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
        # Save experience in replay memory
        self.memory.push(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if self.t_step == 0:
            # if enough samples available in memory, get random subset to learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, gamma=GAMMA)


    def act(self, state, eps=EPSILON):
        """Returns epsilon-greedy actions for given state."""
        self.dqn_online.eval()
        with torch.no_grad():
            action_values = self.dqn_online(state)


        # TODO NEED TO MASK THE ACTION VALUES FOR INVALID MOVES
        # For now, we assume all actions are valid
        self.dqn_online.train()
        if random.random() > eps:
            return np.argmax(action_values.data.numpy())
        else:
            return random.choice(np.arange(self.dqn_online.fc2.out_features))


    def learn(self, experiences, gamma=GAMMA):
        pass
