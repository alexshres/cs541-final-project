import numpy as np
import random 
import dqn

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple, deque
from typing import Tuple
from state import get_state_tensor

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 1 
# BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 1e-4
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
        # next_state = get_state_tensor(next_state)
        self.memory.push(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if self.t_step == 0:
            # if enough samples available in memory, get random subset to learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)

                # converting list of Transitions to batches of tensors
                batch = dqn.Transition(*zip(*experiences))
                states = torch.stack(batch.state)
                actions = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1)
                rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
                next_states = torch.stack(batch.next_state)
                dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1)

                self.learn((states, actions, rewards, next_states, dones), gamma=GAMMA)


    def act(self, state:torch.Tensor, legal_moves_mask:list, eps:float)-> int:
        """Returns epsilon-greedy actions for given state.
        Args:
            state (torch.Tensor): Current state tensor.
            legal_moves_mask (list): Mask of legal moves.
            eps (float): Epsilon value for exploration.
        Returns:
            int: Action index to take.
        """

        self.dqn_online.eval()
        with torch.no_grad():
            action_values = self.dqn_online(state).squeeze()

        # Convert legal_moves_mask (list of bools) to a torch.BoolTensor
        mask = torch.tensor(legal_moves_mask, dtype=torch.bool)
        action_values_masked = action_values.clone()
        action_values_masked[~mask] = float('-inf')

        self.dqn_online.train()
        if random.random() > eps:
            action = int(torch.argmax(action_values_masked).item())
        else:
            legal_indices = torch.where(mask)[0].numpy()
            action = int(random.choice(legal_indices))

        return action


    def learn(self, exp_tuple, gamma:float=GAMMA):
        """Update value parameters using given batch of experience tuples.
        Args:
            experiences (Tuple[torch.Tensor]): tuple of Transition namedtuples.
            gamma (float): Discount factor.
        """

        states, actions, rewards, next_states, dones = exp_tuple

        # detaching so no gradients are calculated for target network
        q_targets_next = self.dqn_target(next_states).detach().max(1)[0].unsqueeze(1)

        # compute Q targets for current states
        # reward 0 if done
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        #  expected Q values from online model
        q_expected = self.dqn_online(states).gather(1, actions)

        # compute loss
        loss = F.mse_loss(q_expected, q_targets)

        self.optimizer.zero_grad()      # clear gradients
        loss.backward()                 # backprop
        self.optimizer.step()           # update weights

        # update target network using a soft update
        self.soft_update(self.dqn_online, self.dqn_target, TAU)


    def soft_update(self, local_model, target_model, tau:float=TAU):
        """Soft update model parameters. Slowly blends the weights of the local (online) model into the target model.
        This helps stabilize training by keeping the target model consistent.

        Args:
            local_model (nn.Module): Local model to update.
            target_model (nn.Module): Target model to update.
            tau (float): Interpolation parameter.
        """
        for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
