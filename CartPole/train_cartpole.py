import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from ReplayMemory import ReplayMemory
from GetScreen import GetScreen
from helpers import *
from OptimizerDQN import optimize_model
from DQN_net import DQN





if __name__ == '__main__':

    env = gym.make('CartPole-v0').unwrapped

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

        plt.ion()

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env.reset()

    # memory = ReplayMemory(env)
    screen = GetScreen(env, device = device)

    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10

    # Get screen size so that we can initialize layers correctly based on shape
    # returned from AI gym. Typical dimensions at this point are close to 3x40x90
    # which is the result of a clamped and down-scaled render buffer in get_screen()
    init_screen = screen.get_screen()

    _, _, screen_height, screen_width = init_screen.shape

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)

    steps_done = 0
    episode_durations = []

    num_episodes = 50
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        last_screen = screen.get_screen()
        current_screen = screen.get_screen()
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action, steps_done = select_action(state = state, EPS_END = EPS_END, EPS_START = EPS_START, EPS_DECAY = EPS_DECAY, policy_net = policy_net, n_actions= n_actions, steps_done = steps_done, device = device)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = screen.get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model(memory = memory, BATCH_SIZE = BATCH_SIZE, policy_net = policy_net, target_net = target_net, optimizer = optimizer, GAMMA = GAMMA, device = device)
            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()