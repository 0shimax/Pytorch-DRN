import torch
import torch.nn as nn
from model.swichable_normalization import SwitchNorm1d


class DQN(nn.Module):

    def __init__(self, action_num):
        super().__init__()
        self.fcb1 = nn.Sequential(
            nn.Linear(7, 30),
            # SwitchNorm1d(30),
            nn.ReLU())
        self.fcb2 = nn.Sequential(
            nn.Linear(30, 30),
            # SwitchNorm1d(30),
            nn.ReLU())
        self.fcb3 = nn.Sequential(
            nn.Linear(30, 30),
            # SwitchNorm1d(30),
            nn.ReLU())
        self.head = nn.Linear(30, action_num)

    def forward(self, x):
        h = self.fcb1(x)
        h = self.fcb2(h)
        h = self.fcb3(h)
        return self.head(h.view(h.size(0), -1))
