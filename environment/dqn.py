import math
import random
import matplotlib.pyplot as plt

from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F


# Define a named tuple to store transitions
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class CheckersDQN(nn.Module):
    """Deep Q-Network for Checkers game using a simple CNN architecture."""
    def __init__(self):
        super().__init__()

        # input size is 6 channels of 8x8 board
        self.conv1 = nn.Conv2d(in_channels=6,
                               out_channels=32,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        # shape after conv1: (32, 8, 8)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # shape after pool1: (32, 4, 4)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        # shape after conv2: (64, 4, 4)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # shape after pool2: (64, 2, 2)

        # Fully connected layers
        # Input size is 64 channels of 2x2 feature maps
        self.fc1 = nn.Linear(64 * 2 * 2, 512)  
        # shape after fc1: (512)

        self.fc2 = nn.Linear(512, 256)
        # shape after fc2: (256)

    def forward(self, x):
        """Forward pass through the DQN."""
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # flatten the tensor while preserving the batch dimension
        x = x.view(x.size(0), -1)  

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

