import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from gym import wrappers

import torch
from pdb import set_trace as debug
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class GetScreen:
    def __init__(self, env, device = 'cpu'):
        self.resize = T.Compose([T.ToPILImage(),
                        T.Resize(40, interpolation=Image.CUBIC),
                        T.ToTensor()])
        self.env = env
        self.device = device


    def get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).

        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))

        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return self.resize(screen).unsqueeze(0).to(self.device)





if __name__ == '__main__':

    env = gym.make('CartPole-v0').unwrapped
    # env.viewer.display = False

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    # plt.ion()

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env.reset()
    get_sample = GetScreen(env=env)


    # debug()
    # plt.ion()

    plt.figure()
    screen1 = get_sample.get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy()
    screen2 = get_sample.get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy()
    # debug()
    plt.imshow(screen1,
               interpolation='none')
    plt.title('Example extracted screen')
    # plt.show()
    plt.figure()
    plt.imshow(screen2,
               interpolation='none')
    plt.title('Example extracted screen')

    plt.show()
    env.close()











