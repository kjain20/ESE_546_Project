
from collections import namedtuple
import random
from pdb import set_trace as debug
import numpy as np
from operator import itemgetter

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity, batch_size = 32):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.batch_size = batch_size

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, idx = None):

        sample_index = np.random.choice(len(self.memory), self.batch_size, replace=False)

        if idx is not None:
            sample_index = idx % len(self.memory)

        return itemgetter(*list(sample_index))(self.memory), sample_index

    def __len__(self):
        return len(self.memory)



class PrioritizedReplay(object):
    def __init__(self, capacity):
        pass


