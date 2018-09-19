import torch
import torch.nn as nn
from model.swichable_normalization import SwitchNorm1d


def inner(aa, bb):
    return (aa * bb.unsqueeze(dim=1).expand(aa.shape)).sum(dim=2)


class ValueNet(nn.Module):
    def __init__(self, dim_in, action_num):
        super().__init__()
        self.fcb1 = nn.Sequential(
            nn.Linear(dim_in, 64),
            # SwitchNorm1d(192),
            nn.ReLU())
        self.fcb2 = nn.Sequential(
            nn.Linear(64, 128),
            # SwitchNorm1d(64),
            nn.ReLU())
        self.fcb3 = nn.Sequential(
            nn.Linear(256*4, 256*6),
            # SwitchNorm1d(64),
            nn.ReLU())
        self.fc1 = nn.Linear(128, 128)

    def forward(self, user_feature):
        h = self.fcb1(user_feature)
        h = self.fcb2(h)
        # h = self.fcb3(h)
        out = self.fc1(h)
        return out


class AdvantageNet(nn.Module):
    def __init__(self, dim_in, action_num):
        super().__init__()
        self.n_out_vec = 128
        self.fcb1 = nn.Sequential(
            nn.Linear(dim_in*2, 128),
            # SwitchNorm1d(128),
            nn.ReLU())
        self.fcb2 = nn.Sequential(
            nn.Linear(128, 128),
            # SwitchNorm1d(64),
            nn.ReLU())
        self.fcb3 = nn.Sequential(
            nn.Linear(64, 32),
            # SwitchNorm1d(64),
            nn.ReLU())
        self.fc1 = nn.Linear(128, self.n_out_vec)

    def forward(self, user_feature, target_features):
        # print(user_feature.shape, target_features.shape)
        n_bach, n_feature = user_feature.shape
        expanded_shape = list(target_features.shape[:2])+[user_feature.shape[-1]]
        uf = user_feature.unsqueeze(dim=1).expand(expanded_shape)
        n_features = target_features.shape[-1] + user_feature.shape[-1]
        # print(uf.shape, target_features.shape, n_features)
        x = torch.cat([uf, target_features], dim=2).view(-1, n_features)
        h = self.fcb1(x)
        h = self.fcb2(h)
        # h = self.fcb3(h)
        out = self.fc1(h)
        return out.view(n_bach, -1, self.n_out_vec)


class Model(nn.Module):
    """
    dueling network
    """

    def __init__(self, dim_in, action_num):
        super().__init__()
        self.value_net = ValueNet(dim_in, action_num)
        self.advantage_net = AdvantageNet(dim_in, action_num)

    def forward(self, observation, target_features):
        # user_feature, content_id = observation
        user_feature = observation
        v = self.value_net(user_feature)
        a = self.advantage_net(user_feature, target_features)
        q = inner(a, v)
        # print("q:", q.shape)
        return q.view(q.size(0), -1)

    def save(self, path, step, optimizer):
        torch.save({
            'step': step,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict()
        }, path)

    def load(self, checkpoint_path, optimizer=None):
        checkpoint = torch.load(checkpoint_path)
        step = checkpoint['step']
        self.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
