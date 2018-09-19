import torch
import numpy as np
from numpy.random import uniform, randint
import random

from feature.eme_data_loader import OwnDataset, loader


class Environment(object):
    def __init__(self, file_name, root_dir,
                 n_target=100, high_rate=.5, max_step=20,
                 train=True):
        n_high = int(n_target*high_rate)
        n_low = n_target - n_high
        self.dataset = OwnDataset(file_name, root_dir, n_high, n_low,
                                  train=train)
        self.n_action = len(self.dataset.target_features_all.target_user_id.unique())
        self.dim_in_feature = len(self.dataset.user_features.iloc[0]) - 1
        self.data_loader = loader(self.dataset, 1)
        self.max_step = max_step

    def reset(self):
        self.dataset.reset()

    def get_action_set(self):
        return np.arange(self.n_action)

    def obs(self):
        return next(iter(self.data_loader))

    def step(self, current_user_id, target_id, t_cnt):
        done = True if t_cnt >= self.max_step - 1 else False
        reward = self.dataset.get_reward(current_user_id, target_id)
        return reward, done
