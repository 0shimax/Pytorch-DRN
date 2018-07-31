import torch
from torch.nn.functional import mse_loss
from torch.autograd import Variable
import torch.optim as optim
import random
import glob
import os
import copy
import numpy as np
from config import Config
from model.ddqn import Model


def state_to_tensor_to_state(state):
    if isinstance(state, tuple):
        content_id = max(state[1], 0)
        u_feature = torch.FloatTensor([state[0]])
        content_id = torch.LongTensor([[content_id]])
        state = (u_feature, content_id)
        return state
    elif isinstance(state, np.ndarray):
        content_id = np.where(state[:, 1] < 0, 0, state[:, 1])
        u_feature = torch.FloatTensor([s for s in state[:, 0]])
        content_id = torch.LongTensor([content_id.astype(np.uint8)])
        state = (u_feature, content_id)
        return state


class Agent:
    def __init__(self, action_set, explore_coef=.2):
        self.explore_coef = explore_coef
        self.action_set = action_set
        self.action_number = len(action_set)
        self.epsilon = Config.initial_epsilon
        self.build_network()

    def build_network(self):
        self.Q_network = Model(self.action_number)  # .cuda()
        self.target_network = Model(self.action_number)  # .cuda()
        self.exploer_network = Model(self.action_number)  # .cuda()
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=Config.lr)

    def update_target_network(self):
        # copy current_network to target network
        self.target_network.load_state_dict(self.Q_network.state_dict())

    def update_Q_network(self, state, action, reward, state_new):
        state = state_to_tensor_to_state(state)
        action = torch.from_numpy(action).float()
        state_new = state_to_tensor_to_state(state_new)
        reward = torch.from_numpy(reward).float()
        # state = Variable(state).cuda()
        # action = Variable(action).cuda()
        # state_new = Variable(state_new).cuda()
        # terminal = Variable(terminal).cuda()
        # reward = Variable(reward).cuda()
        # self.Q_network.eval()
        # self.target_network.eval()

        # use current network to evaluate action argmax_a' Q_current(s', a')_
        action_new = self.Q_network(state_new).max(dim=1)[1].cpu().data.view(-1, 1)
        action_new_onehot = torch.zeros(Config.batch_size, self.action_number)
        action_new_onehot = Variable(action_new_onehot.scatter_(1, action_new, 1.0))  # .cuda()

        # use target network to evaluate value y = r + discount_factor * Q_tar(s', a')
        y = (reward + torch.mul(((self.target_network(state_new)*action_new_onehot).sum(dim=1)), Config.discount_factor))

        # regression Q(s, a) -> y
        self.Q_network.train()
        Q = (self.Q_network(state)*action).sum(dim=1)
        loss = mse_loss(input=Q, target=y.detach())

        # backward optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def take_action(self, state):
        state = state_to_tensor_to_state(state)
        # state = Variable(state).cuda()

        self.Q_network.eval()
        estimate = self.Q_network(state).max(dim=1)

        state_dict = copy.deepcopy(self.Q_network.state_dict())
        # print(state_dict)
        params_values = list(state_dict.values())
        for i in range(len(params_values)):
            params_values[i] =\
                params_values[i] + params_values[i]*torch.randn(1)*self.explore_coef
        for k, nv in zip(state_dict.keys(), params_values):
            state_dict[k] = nv
        self.exploer_network.load_state_dict(state_dict)
        estimate_prime = self.exploer_network(state).max(dim=1)
        # with epsilon prob to choose random action else choose argmax Q estimate action
        # TODO: learn probabilistic interleave
        if random.random() > .5:
            return estimate[1].data[0]
        else:
            return estimate_prime[1].data[0]

    def save(self, step, logs_path):
        os.makedirs(logs_path, exist_ok=True)
        model_list =  glob.glob(os.path.join(logs_path, '*.pth'))
        if len(model_list) > Config.maximum_model - 1:
            min_step = min([int(li.split('/')[-1][6:-4]) for li in model_list])
            os.remove(os.path.join(logs_path, 'model-{}.pth' .format(min_step)))
        logs_path = os.path.join(logs_path, 'model-{}.pth' .format(step))
        self.Q_network.save(logs_path, step=step, optimizer=self.optimizer)
        print('=> Save {}' .format(logs_path))

    def restore(self, logs_path):
        self.Q_network.load(logs_path)
        self.target_network.load(logs_path)
        self.exploer_network.load(logs_path)
        print('=> Restore {}' .format(logs_path))
