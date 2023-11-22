import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import conv_net
import numpy as np

class RandomAgent(object):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self, state):
        return random.randint(0, self.action_dim-1)


class NeuralAgent(object):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.conv_net = conv_net(self.state_dim)
        self.actor = nn.Sequential(
                            conv_net(self.state_dim),
                            nn.ReLU(),
                            nn.Linear(self.conv_dim(), 32),
                            nn.ReLU(),
                            nn.Linear(32, action_dim),
                        )

    def conv_dim(self):
        return self.conv_net(torch.zeros([1, * self.state_dim])).view(1, -1).size(-1)

    def select_action(self, state):
        state = np.expand_dims(state, 0)
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action = self.actor(state)
        return action.item()

    def load(self, checkpoint_path):
        self.actor.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))