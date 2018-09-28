import math
import copy
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from model.reply_memory_simple import ReplayMemory, Transition
from model.ddqn import Model
from model.ae_net import AEN
# from model.ddqn_for_all import Model

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200


def prepare_networks(dim_in_feature, n_action, device):
    policy_net = Model(dim_in_feature, n_action).to(device)
    target_net = Model(dim_in_feature, n_action).to(device)
    explore_net = Model(dim_in_feature, n_action).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    explore_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    explore_net.eval()
    return policy_net, target_net, explore_net


class Agent(object):
    def __init__(self, dim_in_feature, device, n_action=10, uniform_range=10):
        # if gpu is to be used
        self.device = device
        self.policy_net, self.target_net, self.explore_net =\
            prepare_networks(dim_in_feature, n_action, self.device)
        self.aenet = AEN(dim_in_feature, n_action).to(self.device)
        self.target_aenet = AEN(dim_in_feature, n_action).to(self.device)

        # self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.aen_optimizer = optim.Adam(self.aenet .parameters(), lr=1e-4)
        self.memory = ReplayMemory(10000)  # 10000
        # self.with_reward_memory = ReplayMemory(10000)
        # self.without_reward_memory = ReplayMemory(10000)

        self.n_action = n_action
        self.steps_done = 0
        self.explore_coef = .1
        self.eta = .05
        self.uniform_range = uniform_range
        self.eager_cnt = 0
        self.epsilon = .5
        self._lambda = .1

    def select_action_explore_net(self, state, target_features):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        # TODO: add decay uniform_range
        # self.uniform_range *=  eps_threshold
        self.steps_done += 1

        # if sample > eps_threshold:
        if sample > self.epsilon:
            # print("genereate from policy net")
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # print("genereate from explore net----------------------")
            # TODO: need to modify
            self.replace_explore_net_wieht_values()
            with torch.no_grad():
                return self.explore_net(state).max(1)[1].view(1, 1)

    def select_action(self, state, target_features):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state, target_features).max(1)[1].view(1, 1)
        else:
            self.eager_cnt += 1
            valid_idx = self.eliminate_acts(state)
            idx = torch.randperm(valid_idx.size(0))[0]
            return torch.tensor([[valid_idx[idx]]],
                                device=self.device, dtype=torch.long)
            # return torch.tensor([[random.randrange(self.n_action)]],
            #                     device=self.device, dtype=torch.long)

    def select_action_for_test(self, state, target_features, n_best=5):
        with torch.no_grad():
            return torch.topk(self.policy_net(state, target_features), n_best)


    def eliminate_acts(self, state, beta=.5, elm_thresh=.6):
        # LastLayerActivates(E(s))
        es = self.target_aenet(state).detach()
        phi, phi_t, v_ta_inv = self.calculate_v_ta()

        # theta = v_ta_inv.mm(phi_t).mm(es)
        # sqr_beta = 2*torch.log1p(v_ta.det()**(0.5))*((1e-6 + lambda_eye.det())**(-0.5))/delta*k
        # sqr_beta = R * sqr_beta**(0.5) + _lambda**(0.5) * self.n_action

        prob_upper_bound = beta * phi.mm(v_ta_inv).mm(phi_t)
        elm_criteria = es - prob_upper_bound**(0.5)
        valid_idx = torch.nonzero(elm_criteria.view(-1) < elm_thresh)

        return valid_idx

    def calculate_v_ta(self):
        # Φ(s)
        phi = self.target_aenet.phi.detach()
        # deep copy
        phi_t = torch.FloatTensor((phi.detach().numpy())).t_()

        eye = torch.eye(phi.shape[-1])
        lambda_eye = self._lambda * eye

        v_ta = lambda_eye + phi_t.mm(phi)
        v_ta_inv = v_ta.inverse()
        return phi, phi_t, v_ta_inv

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)

        # if len(self.with_reward_memory) < BATCH_SIZE//2 or len(self.without_reward_memory) < BATCH_SIZE//2:
        #     return
        # with_reward_transitions = self.with_reward_memory.sample(BATCH_SIZE//2)
        # without_reward_transitions = self.without_reward_memory.sample(BATCH_SIZE//2)
        # transitions = with_reward_transitions + without_reward_transitions
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None,
                      batch.next_state)),
            device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        non_final_target_features = torch.cat([t for s, t in zip(batch.next_state, batch.target_feature)
                                               if s is not None])
        state_batch = torch.cat(batch.state)
        target_features_batch = torch.cat(batch.target_feature)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values =\
            self.policy_net(state_batch, target_features_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] =\
            self.target_net(non_final_next_states, non_final_target_features).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values =\
            (next_state_values * GAMMA) + reward_batch

        # print(state_action_values)
        # print(expected_state_action_values.unsqueeze(1))
        # Compute Huber loss
        # loss = F.smooth_l1_loss(state_action_values,
        #                         expected_state_action_values.unsqueeze(1),
        #                         True)

        # MSELoss
        loss = F.mse_loss(state_action_values,
                          expected_state_action_values.unsqueeze(1),
                          True)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.data.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def replace_explore_net_wieht_values(self):
        state_dict = copy.deepcopy(self.policy_net.state_dict())
        params_values = list(state_dict.values())
        for i in range(len(params_values)):
            rand_val = torch.zeros(1).uniform_(
                -self.uniform_range, self.uniform_range)
            params_values[i] =\
                params_values[i] +\
                params_values[i] * rand_val * self.explore_coef
        for k, nv in zip(state_dict.keys(), params_values):
            state_dict[k] = nv
        self.explore_net.load_state_dict(state_dict)

    def update_target_network_with_explore_net(self):
        policy_state_dict = copy.deepcopy(self.policy_net.state_dict())
        explore_state_dict = copy.deepcopy(self.explore_net.state_dict())
        policy_params_values = list(policy_state_dict.values())
        explore_params_values = list(explore_state_dict.values())
        for i in range(len(policy_params_values)):
            policy_params_values[i] =\
                policy_params_values[i] + explore_params_values[i]*self.eta
        for k, nv in zip(policy_state_dict.keys(), policy_params_values):
            policy_state_dict[k] = nv
        self.target_net.load_state_dict(policy_state_dict)

    def optimize_aen(self, eliminate_teacher_val, state):
        loss = F.mse_loss(eliminate_teacher_val,
                          self.aenet(state),
                          True)
        # Optimize the model
        self.aen_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        for param in self.aenet.parameters():
            param.data.clamp_(-1, 1)
        self.aen_optimizer.step()
        return loss.data.item()

    def aen_update(self):
        self.target_aenet.load_state_dict(self.aenet.state_dict())

    # def aen_update(self, state):
    #     v_ta_inv = self.calculate_v_ta()
    #     b_a = self.aenet.phi.detach().t_().mm(self.aenet(state).detach())
    #     self.v_a_inv = v_ta_inv
    #     self.e_updated = v_ta_inv.mm(b_a)
