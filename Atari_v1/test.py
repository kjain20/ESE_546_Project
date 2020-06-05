
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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from pdb import set_trace as debug

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def test(env, n_episodes, policy, render=True, noisy = False, FactorizedNoisy = False, device = 'cpu'):
    policy.eval()
    env = gym.wrappers.Monitor(env, './videos/' + 'dqn_pong_video', force = True)
    with torch.no_grad():
        if noisy or FactorizedNoisy:
            policy.fc4.noise = False
            policy.head.noise = False
        for episode in range(n_episodes):
            obs = env.reset()
            state = get_state(obs)
            total_reward = 0.0
            for t in count():
                action = policy(state.to(device)).max(1)[1].view(1,1)

                if render:
                    env.render()
                    time.sleep(0.02)

                obs, reward, done, info = env.step(action)

                total_reward += reward

                if not done:
                    next_state = get_state(obs)
                else:
                    next_state = None

                state = next_state

                if done:
                    print("Finished Test Episode {} with reward {}".format(episode, total_reward))
                    break

    env.close()
    return total_reward