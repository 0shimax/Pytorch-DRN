import torch
import torch.nn as nn


class ValueNet(nn.Module):
    def __init__(self, action_num):
        super().__init__()
        self.fcb1 = nn.Sequential(
            nn.Linear(7, 192),
            nn.ReLU())
        self.fcb2 = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU())
        self.fcb3 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU())
        self.fc2 = nn.Linear(64, action_num)

    def forward(self, user_feature):
        h = self.fcb1(user_feature)
        h = self.fcb2(h)
        h = self.fcb3(h)
        # h = self.fc1(h.view(-1, 7*7*64))
        out = self.fc2(h)
        return out


class AdvantageNet(nn.Module):
    def __init__(self, action_num):
        super().__init__()
        content_size, self.embedding_dim = 2, 32
        self.embeddings = nn.Embedding(content_size, self.embedding_dim)
        self.fcb1 = nn.Sequential(
            nn.Linear(7+32, 192),
            nn.ReLU())
        self.fcb2 = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU())
        self.fcb3 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU())
        self.fc2 = nn.Linear(64, action_num)

    def forward(self, user_feature, content_id):
        emb = self.embeddings(content_id).view(-1, self.embedding_dim)
        h = torch.cat((user_feature, emb), 1)
        h = self.fcb1(h)
        h = self.fcb2(h)
        h = self.fcb3(h)
        # h = self.fc1(h.view(-1, 7*7*64))
        out = self.fc2(h)
        return out


class Model(nn.Module):
    """
    dueling network
    """

    def __init__(self, action_num):
        super().__init__()
        self.value_net = ValueNet(action_num)
        self.advantage_net = AdvantageNet(action_num)

    def forward(self, observation):
        user_feature, content_id = observation
        v = self.value_net(user_feature)
        a = self.advantage_net(user_feature, content_id)
        q = v + (a - a.mean(dim=0))
        return q

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
