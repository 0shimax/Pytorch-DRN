import torch
import numpy as np
from numpy.random import uniform, randint
import random


class Viewer(object):
    def __init__(self, gender='male'):
        self.gender = gender
        self.male_features = np.array([23, 10, 1, 0, 0, 1, 0], dtype=np.float32)
        self.female_features = np.array([23, 10, 0, 1, 0, 0, 1], dtype=np.float32)
        self.high_click = 1. if uniform(0, 1.0) > 0.3 else -1
        self.low_click = 1. if uniform(0, 1.0) > 0.7 else -1

    def get_features(self):
        if self.gender == 'male':
            return self.male_features
        else:
            return self.female_features

    def view(self, ad_id, news):
        if self.gender == 'male':
            # Men are easy to click on ads with id 5 or less
            if ad_id % 2 == 0:
                return self.high_click*news
            else:
                return self.low_click*news
        else:
            # female are easy to click on ads with id 6 or higher
            if ad_id % 2 == 1:
                return self.high_click*news
            else:
                return self.low_click*news


class Environment(object):
    def __init__(self, n_action=10):
        self.n_action = n_action
        self.news = [0, 1]

    def get_action_set(self):
        a_sets = np.zeros((self.n_action, self.n_action))
        a_sets[np.arange(self.n_action), np.arange(self.n_action)] = 1.
        return np.arange(self.n_action)

    def obs(self):
        # user and ad are selected randomly
        self.viewer = Viewer(self.generate_gender())
        news = 1 if uniform(0, 1.0) > 0.5 else -1
        return self.viewer.get_features(), news

    def generate_content(self):
        return randint(0, 10)

    def generate_gender(self):
        return 'male' if uniform(0, 1.0) > 0.5 else 'female'

    def act(self, ad_id, news):
        # return reward
        if self.viewer.view(ad_id, news) > 0:
            return 1.
        else:
            return 0.
