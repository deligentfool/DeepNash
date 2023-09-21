import gym
import numpy as np
from gym import spaces


class Channel_switch(gym.ObservationWrapper):
    def __init__(self, env):
        super(Channel_switch, self).__init__(env)
        origin_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=[origin_shape[-1], origin_shape[1], origin_shape[0]], dtype=np.uint8)

    def observation(self, observation):
        return (np.swapaxes(observation[0], 2, 0), np.swapaxes(observation[1], 2, 0))


def env_wrap(env):
    return Channel_switch(env)