import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

from model import Nash
from wrapper import env_wrap
import gym
import lasertag
from envs.nim import NIM
import warnings
import random
warnings.filterwarnings("ignore")


################################### Training ###################################
def train():

    # env_name = 'LaserTag-small2-v0'
    env_name = 'NIM-5'
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = 30                   # max timesteps in one episode
    max_training_timesteps = int(6e7)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    eval_num = 25
    last_save_model_step = 0
    max_win_ratio = 0
    #####################################################

    #####################################################
    n = 0
    delta_m = 20
    m = 0

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps

    lr_actor = 0.0001       # learning rate for actor network
    lr_critic = 0.0001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    if "LaserTag" in env_name:
        env = gym.make(env_name)
        env = env_wrap(env)

        # state space dimension
        state_dim = env.observation_space.shape

        # action space dimension
        if has_continuous_action_space:
            action_dim = env.action_space.shape[0]
        else:
            action_dim = env.action_space.n
    elif "NIM" in env_name:
        level = int(env_name[4:])
        env = NIM(level)
        state_dim = env.state_dim
        action_dim = env.action_dim

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/Nash_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "Nash_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "Nash_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("update frequency : " + str(update_timestep) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    if "LaserTag" in env_name:
        nash_agent = Nash(state_dim, action_dim, lr_actor, lr_critic, has_continuous_action_space, action_std)
    elif "NIM" in  env_name:
        nash_agent = Nash(state_dim, action_dim, lr_actor, lr_critic, has_continuous_action_space, action_std, flatten=True)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_nash_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state, info = env.reset()
        current_ep_reward = 0
        current_nash_ep_reward = 0
        nash_agent.new_buffer()

        for t in range(1, max_ep_len+1):

            if "LaserTag" in env_name:
                # select action with policy
                p1_action = nash_agent.select_action(state[0], 0)
                p2_action = nash_agent.select_action(state[1], 1)
                actions = {"1": p1_action, "2": p2_action}
                state, reward, done, _, _ = env.step(actions)

                # saving reward and is_terminals
                nash_agent.buffer[0].rewards[-1].append(reward[0] - reward[1])
                nash_agent.buffer[1].rewards[-1].append(reward[1] - reward[0])
                nash_agent.buffer[0].is_terminals[-1].append(done)
                nash_agent.buffer[1].is_terminals[-1].append(done)

                time_step +=1
                current_ep_reward += np.sum(reward)
                current_nash_ep_reward += (reward[0] - reward[1])
            elif "NIM" in env_name:
                rewards = []
                for i in range(2):
                    avail_actions = env.get_onehot_available_actions()
                    action = nash_agent.select_action(state, i, avail_actions)
                    state, reward, done, _, _ = env.step(action)
                    rewards.append(reward)
                    nash_agent.buffer[i].rewards[-1].append(reward)
                    nash_agent.buffer[i].is_terminals[-1].append(done)

                time_step +=1
                current_ep_reward += np.abs(rewards[0] + rewards[1])
                if rewards[0] != 0:
                    current_nash_ep_reward = current_nash_ep_reward + rewards[0]
                else:
                    current_nash_ep_reward = current_nash_ep_reward - rewards[1]

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                nash_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_nash_reward = print_nash_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)
                print_avg_nash_reward = round(print_avg_nash_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {} \t\t Nash Reward : {}".format(i_episode, time_step, print_avg_reward, print_avg_nash_reward))

                print_running_reward = 0
                print_nash_running_reward = 0
                print_running_episodes = 0



            # break; if the episode is over
            if done:
                if "NIM" in env_name:
                    nash_agent.buffer[0].is_terminals[-1][-1] = True
                    nash_agent.buffer[1].is_terminals[-1][-1] = True
                    if nash_agent.buffer[1].rewards[-1][-1] > 0:
                        nash_agent.buffer[0].rewards[-1][-1] = -nash_agent.buffer[1].rewards[-1][-1]
                    else:
                        nash_agent.buffer[1].rewards[-1][-1] = -nash_agent.buffer[0].rewards[-1][-1]
                # save model weights
                if time_step - last_save_model_step > save_model_freq:
                    if eval_num > 0:
                        win_count = 0
                        for _ in range(eval_num):
                            e_state, e_info = env.reset()
                            e_done = False
                            while not e_done:
                                e_rewards = []
                                for i in range(2):
                                    if level % 2 == i:
                                        e_avail_actions = env.get_onehot_available_actions()
                                        e_action = nash_agent.select_action(e_state, i, e_avail_actions, True)
                                    else:
                                        e_avail_actions = env.get_available_actions()
                                        e_action = random.choice(e_avail_actions)
                                    e_state, e_reward, e_done, _, _ = env.step(e_action)
                                    if level % 2 == i:
                                        e_rewards.append(e_reward)
                            if e_rewards[-1] > 0:
                                win_count += 1
                        test_win_ratio = win_count / eval_num
                        max_win_ratio = max(max_win_ratio, test_win_ratio)
                    print("--------------------------------------------------------------------------------------------")
                    print("saving model at : " + checkpoint_path)
                    if eval_num == 0 or test_win_ratio == max_win_ratio:
                        nash_agent.save(checkpoint_path[:-4] + "_" + str(max_win_ratio) + checkpoint_path[-4:])
                    print("model saved")
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    print("M: {}".format(m))
                    if eval_num > 0:
                        print("Test win ratio: ", test_win_ratio)
                    print("--------------------------------------------------------------------------------------------")
                    last_save_model_step = time_step
                break
        # update PPO agent
        if time_step % update_timestep == 0:
            if n >= delta_m:
                n = 0
                m += 1
                nash_agent.policy_reg_old.load_state_dict(nash_agent.policy_reg.state_dict())
                nash_agent.policy_reg.load_state_dict(nash_agent.policy_old.state_dict())
            alpha = 1 if n > delta_m / 2 else n * 2 / delta_m
            nash_agent.update(alpha)
            n += 1

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        print_nash_running_reward += current_nash_ep_reward

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':

    train()
