import torch
import torch.nn as nn
import torch.nn.functional as F


class AEN(nn.Module):

    def __init__(self, dim_in, action_num):
        super().__init__()
        self.fcb1 = nn.Sequential(
            nn.Linear(dim_in, 30),
            nn.ReLU())
        self.fcb2 = nn.Sequential(
            nn.Linear(30, 30),
            nn.ReLU())
        self.fcb3 = nn.Sequential(
            nn.Linear(30, 128),
            nn.ReLU())
        self.head = nn.Linear(128, action_num, bias=False)

    def forward(self, user_feature):
        h = self.fcb1(user_feature)
        h = self.fcb2(h)
        h = self.fcb3(h)

        # Output of last hidden layer is Φ(s)
        self.phi = h

        return F.sigmoid(self.head(h))
