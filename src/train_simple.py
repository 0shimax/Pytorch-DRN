import math
import numpy as np
import argparse
from collections import defaultdict
from itertools import count

import torch
import torch.nn as nn

from model.agent_simple import Agent
from model.environment_for_owndata import Environment
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
file_name = 'eme_interactsions_June2018.csv'
root_dir = './raw'


def main():
    n_target = 30
    env = Environment(file_name, root_dir, n_target=n_target, max_step=20)
    agent = Agent(env.dim_in_feature, n_target)  #, env.n_action)

    num_episodes = 5000
    for i_episode in range(num_episodes):
        t_reword = 0
        t_loss = 0
        for t in count():
            state, target_features, current_user_id, target_ids = env.obs()
            # Select and perform an action
            action = agent.select_action(state, target_features)
            target_id = target_ids[:, action.item()].item()
            reward, done = env.step(current_user_id, target_id, t)
            reward = torch.tensor([reward], device=agent.device)

            # Observe new state
            if not done:
                # next_state = state.clone()
                next_state, _, _, _ = env.obs()
            else:
                next_state = None

            # Store the transition in memory
            agent.memory.push(state, target_features, action, next_state, reward)

            # Perform one step of the optimization (on the target network)
            loss = agent.optimize_model()
            t_reword += reward.data.item()
            if loss:
                t_loss += loss
                # print reward and loss
            if done:
                break
        if i_episode % TARGET_UPDATE == 0:
            print('Episode: {} Reward: {:.3f} Loss: {:.3f}'.format(i_episode, t_reword, t_loss))
        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            agent.update_target_network()

    print('Complete')


if __name__=='__main__':
    main()
