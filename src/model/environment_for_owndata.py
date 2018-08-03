import torch
import numpy as np
from numpy.random import uniform, randint
import random

from feature.eme_data_loader import OwnDataset, loader


class Environment(object):
    def __init__(self, file_name, root_dir):
        self.dataset = OwnDataset(file_name, root_dir)
        self.n_action = len(self.dataset.target_features)
        self.data_loader = iter(loader(self.dataset, 1))

    def reset(self):
        self.dataset.reset()

    def get_action_set(self):
        return np.arange(self.n_action)

    def obs(self):
        return self.data_loader.next()

    def step(self, user_id, target_id, t_cnt, threshold=100):
        done = True if t_cnt >= threshold-1 else False
        reward = self.dataset.get_reward(user_id, target_id)
        return reward, done
