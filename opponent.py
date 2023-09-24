import random


class RandomAgent(object):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self, state):
        return random.randint(0, self.action_dim-1)