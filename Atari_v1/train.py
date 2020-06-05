import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np
import time

import gym
# from optimize_model import optimize_model
from optimize_model_nstep import optimize_model
from models import *

from atari_wrappers import *
from models import *
from utils import *
from test import test

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from pdb import set_trace as debug






def train(env, n_episodes, exploration, memory, lr, device,
          render=False, INITIAL_MEMORY = 10000, TARGET_UPDATE = 1000, n_steps = 1,double_dqn = False, noisy = False, FactorizedNoisy = False):
    steps_done = 0
    policy_net = DQN(n_actions=4, noisy=noisy, FactorizedNoisy=FactorizedNoisy).to(device)
    target_net = DQN(n_actions=4, noisy=noisy, FactorizedNoisy=FactorizedNoisy).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    for p in target_net.parameters():
        p.requires_grad = False

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    reward_trend = []
    reward_trend_test = []
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        policy_net.train()
        for t in count():
            action = select_action(state, exploration, steps_done, policy_net, device, noise = noisy, factorized_noise = FactorizedNoisy)
            steps_done += 1

            if render:
                env.render()

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)

            memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state

            if steps_done > INITIAL_MEMORY:
                policy_net = optimize_model(optimizer = optimizer, policy_net=policy_net, target_net=target_net, memory = memory,  device=device, n_steps = n_steps, double_dqn = double_dqn)

                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
        if episode % 1 == 0:
                print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episode, t, total_reward))
        reward_trend.append(total_reward)

        reward_trend_test.append(test(env, 1, policy_net, render=False, noisy= noisy, FactorizedNoisy=FactorizedNoisy, device=device))

    env.close()
    torch.save(policy_net, "dqn_pong_model.pt")

    return reward_trend, reward_trend_test
