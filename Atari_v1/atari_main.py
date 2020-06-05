import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np 
import time

import gym

from atari_wrappers import *
from memory import ReplayMemory
from models import *
from utils import *
from train import train
from test import test

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from pdb import set_trace as debug
from matplotlib import pyplot as plt

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



def set_seed(seed_):

    torch.manual_seed(seed_)
    np.random.seed(seed_)
    random.seed(seed_)
    




if __name__ == '__main__':
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = 1000000
    TARGET_UPDATE = 1000
    RENDER = False
    lr = 1e-4
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 10 * INITIAL_MEMORY
    Noisy = False
    FactorizedNoisy = True

    # # create networks
    # policy_net = DQN(n_actions=4, noisy = False, FactorizedNoisy = False).to(device)
    # target_net = DQN(n_actions=4, noisy = False, FactorizedNoisy = False).to(device)
    # target_net.load_state_dict(policy_net.state_dict())
    #
    # # setup optimizer
    # optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    ## Set seed
    set_seed(0)


    # create environment
    env = gym.make("PongNoFrameskip-v4")
    env = make_env(env)

    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)
    Schedule = ExponentialSchedule(EPS_DECAY)
    
    # train model
    rewards_train, rewards_test = train(env, 400, exploration=Schedule.schedule_value, memory = memory, lr = lr,device = device, n_steps = 20, double_dqn = True, noisy = Noisy, FactorizedNoisy = FactorizedNoisy)
    policy_net_reload = torch.load("dqn_pong_model.pt").to(device)
    test(env, 1, policy_net_reload, render=False, noisy = Noisy, FactorizedNoisy = FactorizedNoisy, device = device)

    # plt.plot(rewards_train)
    # plt.plot(rewards_test)
    #
    # plt.show()

