import torch
import torch.optim as optim
import torch.nn as nn
from utils import *
from pdb import set_trace as debug
import torch.nn.functional as F
import numpy as np
import random
from torch.nn import init, Parameter
from torch.autograd import Variable


class NoisyNet(nn.Linear):
    def __init__(self, input_len, output_len, sigma_init=0.017, noise = True):
        super(NoisyNet, self).__init__(in_features=input_len, out_features=output_len, bias=True)

        self.sigma_init = sigma_init
        self.sigma_w = Parameter(torch.Tensor(output_len, input_len)) # requires_grad = True
        self.sigma_b = Parameter(torch.Tensor(output_len)) # requires_grad = True
        self.register_buffer('epsilon_w', torch.zeros(output_len, input_len))# requires_grad = False
        self.register_buffer('epsilon_b', torch.zeros(output_len)) # requires_grad = False

        self.noise = noise

        self._reset_parameters()

    def _reset_parameters(self):
        init.uniform_(self.weight, -np.sqrt(3 / self.in_features), np.sqrt(3 / self.in_features))
        init.uniform_(self.bias, -np.sqrt(3 / self.in_features), np.sqrt(3 / self.in_features))
        init.constant_(self.sigma_w, self.sigma_init)
        init.constant_(self.sigma_b, self.sigma_init)

    def sample_noise(self):
        torch.randn(self.epsilon_w.shape, out=self.epsilon_w)

        torch.randn(self.epsilon_b.shape, out=self.epsilon_b)

    def forward(self, input):
        if self.noise:
            self.sample_noise()
        else:
            self.remove_noise()
        return F.linear(input, self.weight + self.sigma_w * self.epsilon_w.clone(),
                        self.bias + self.sigma_b * self.epsilon_b.clone())
    def remove_noise(self):
        torch.zeros(self.epsilon_w.shape, out=self.epsilon_w)
        torch.zeros(self.epsilon_b.shape, out=self.epsilon_b)



class FactorizedNoisyNet(nn.Linear):
    def __init__(self, input_len, output_len, sigma_init=0.017, noise = True):
        super(FactorizedNoisyNet, self).__init__(in_features=input_len, out_features=output_len, bias=True)

        self.sigma_init = 0.5/np.sqrt(input_len)
        self.sigma_w = Parameter(torch.Tensor(output_len, input_len)) # requires_grad = True
        self.sigma_b = Parameter(torch.Tensor(output_len)) # requires_grad = True
        self.register_buffer('epsilon_in', torch.zeros(input_len))# requires_grad = False
        self.register_buffer('epsilon_out', torch.zeros(output_len)) # requires_grad = False
        self.noise = noise
        self._reset_parameters()

    def _reset_parameters(self):
        init.uniform_(self.weight, -np.sqrt(1 / self.in_features), np.sqrt(1 / self.in_features))
        init.uniform_(self.bias, -np.sqrt(1 / self.in_features), np.sqrt(1 / self.in_features))
        init.constant_(self.sigma_w, self.sigma_init)
        init.constant_(self.sigma_b, self.sigma_init)

    def sample_noise(self):
        torch.randn(self.epsilon_out.shape, out=self.epsilon_out)
        torch.randn(self.epsilon_in.shape, out=self.epsilon_in)

    def forward(self, input):
        if self.noise:
            self.sample_noise()
        else:
            self.remove_noise()

        noise_matrix = self.epsilon_out.ger(self.epsilon_in)

        return F.linear(input, self.weight + self.sigma_w * noise_matrix, self.bias + self.sigma_b * self.epsilon_out.clone())

    def remove_noise(self):
        torch.zeros(self.epsilon_in.shape, out=self.epsilon_in)
        torch.zeros(self.epsilon_out.shape, out=self.epsilon_out)



