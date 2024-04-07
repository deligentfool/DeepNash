import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
import time

from model import Nash
from wrapper import env_wrap
import gym
import lasertag
from opponent import RandomAgent, NeuralAgent
import warnings
warnings.filterwarnings("ignore")


#################################### Testing ###################################
def test():
    print("============================================================================================")

    env_name = 'LaserTag-small3-v0'
    has_continuous_action_space = False
    max_ep_len = 1000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 20    # total num of testing episodes

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    env = gym.make(env_name)
    env = env_wrap(env)

    # state space dimension
    state_dim = env.observation_space.shape

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a Nash agent
    nash_agent = Nash(state_dim, action_dim, lr_actor, lr_critic, has_continuous_action_space, action_std)
    # choose a random opponent agent
    opponent_agent = RandomAgent(state_dim, action_dim)
    # or a neural network based opponent agent
    # opponent_agent = NeuralAgent(state_dim, action_dim)
    # opponent_agent.load('./nn_policy.pth')

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "Nash_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "Nash_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    nash_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    win = 0
    draw = 0
    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        reward_1 = 0
        reward_2 = 0
        state, info = env.reset()

        for t in range(1, max_ep_len+1):
            p1_action = nash_agent.select_action(state[0], 0, test=True)
            p2_action = opponent_agent.select_action(state[1])
            actions = {"1": p1_action, "2": p2_action}
            state, reward, done, _, _ = env.step(actions)
            ep_reward += (reward[0] - reward[1])
            reward_1 += reward[0]
            reward_2 += reward[1]

            if render:
                # time.sleep(0.1)
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        # clear buffer
        nash_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Pure Reward: {} \t\t Nash Score: {} \t\t Oppo Score: {}'.format(ep, round(ep_reward, 2), round(reward_1, 2), round(reward_2, 2)))
        if ep_reward > 0:
            win += 1
        elif ep_reward == 0:
            draw += 1
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test pure reward : " + str(avg_test_reward))
    print("Nash vs Oppo \t\t Win: {}% \t\t Draw: {}% \t\t Lose: {}%".format(round(win / total_test_episodes * 100, 2), round(draw / total_test_episodes * 100, 2), round((total_test_episodes - win - draw) / total_test_episodes * 100, 2)))

    print("============================================================================================")


if __name__ == '__main__':

    test()