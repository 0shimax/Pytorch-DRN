import math
import numpy as np
import argparse
from collections import defaultdict
from itertools import count

import torch
import torch.nn as nn

from model.agent_simple import Agent
from model.environment_simple import Environment
from feature.eme_data_loader import OwnDataset, loader


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-l', '--logs_path', dest='logs_path',
#                         help='path of the checkpoint folder',
#                         default='./logs', type=str)
#     parser.add_argument('-r', '--restore', dest='restore',
#                         help='restore checkpoint',
#                         default=None, type=str)
#     parser.add_argument('-t', '--train', dest='train',
#                         help='train policy or not',
#                         default=True, type=bool)
#     args = parser.parse_args()
#
#     return args
#
#
# args = parse_args()
TARGET_UPDATE = 10


def main():
    n_action = 2
    agent = Agent(n_action)
    env = Environment(n_action)

    num_episodes = 5000
    for i_episode in range(num_episodes):
        t_reword = 0
        t_loss = 0
        results = np.zeros((2, n_action))
        state = torch.FloatTensor([env.obs()])
        for t in count():
            # Select and perform an action
            action = agent.select_action(state)
            # print(env.viewer.gender, action)
            reward, done = env.step(action.item(), t)
            reward = torch.tensor([reward], device=agent.device)

            g_idx = 0 if env.viewer.gender=='male' else 1
            results[g_idx, action.item()] += 1.  # reward[0].item()

            # Observe new state
            if not done:
                # next_state = state.clone()
                next_state = torch.FloatTensor([env.obs()])
                # next_state[:,-1] += reward.data.item()
            else:
                next_state = None

            # Store the transition in memory
            agent.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            loss = agent.optimize_model()
            if loss:
                t_reword += reward.data.item()
                t_loss += loss
                # print reward and loss
            if done:
                break
        if i_episode % TARGET_UPDATE == 0:
            print('Episode: {} Reward: {:.3f} Loss: {:.3f}'.format(i_episode, t_reword, t_loss))
            print(results)
        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            agent.update_target_network()

    print('Complete')


if __name__=='__main__':
    main()
