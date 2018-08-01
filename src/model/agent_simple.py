import math
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from model.reply_memory_simple import ReplayMemory, Transition
from model.dqn import DQN


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200


def prepare_networks(n_action, device):
    policy_net = DQN(n_action).to(device)
    target_net = DQN(n_action).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    return policy_net, target_net


class Agent(object):
    def __init__(self, n_action=10):
        # if gpu is to be used
        self.device =\
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net, self.target_net = prepare_networks(n_action, self.device)
        # self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.memory = ReplayMemory(10000)
        self.n_action = n_action
        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_action)]],
                                device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
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
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values =\
            self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] =\
            self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values =\
            (next_state_values * GAMMA) + reward_batch

        # print(state_action_values)
        # print(expected_state_action_values.unsqueeze(1))
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.data.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
