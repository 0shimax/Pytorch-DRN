import torch
import numpy as np
from numpy.random import uniform, randint
import random

from feature.eme_data_loader import OwnDataset, loader


class Viewer(object):
    def __init__(self, gender='male'):
        self.gender = gender
        self.male_features = np.array([23, 0, 1, 1, 0, 1, 0], dtype=np.float32)
        self.female_features = np.array([56, 20, 1, 0, 5, 0, 5], dtype=np.float32)
        self.high_click = 1. if uniform(0, 1.0) > 0.1 else -1
        self.low_click = 1. if uniform(0, 1.0) > 0.8 else -1

    def get_features(self):
        if self.gender == 'male':
            return self.male_features
        else:
            return self.female_features

    def view(self, ad_id):
        # print(self.gender, ad_id)
        if self.gender == 'male':
            # Men are easy to click on ads with id 5 or less
            if ad_id % 2 == 0:
                return self.high_click
            else:
                return self.low_click
        else:
            # female are easy to click on ads with id 6 or higher
            if ad_id % 2 == 1:
                return self.high_click
            else:
                return self.low_click


class Environment(object):
    def __init__(self, file_name, root_dir):
        dataset = OwnDataset(file_name, root_dir)
        self.n_action = len(dataset.target_features)
        self.data_loader = loader(dataset, 1)

    def get_action_set(self):
        return np.arange(self.n_action)

    def obs(self):
        return self.data_loader.next()

    def step(self, ad_id, t_cnt, threshold=20):
        done = True if t_cnt >= threshold-1 else False
        # return reward
        if self.viewer.view(ad_id) > 0:
            return 1., done
        else:
            return 0., done
