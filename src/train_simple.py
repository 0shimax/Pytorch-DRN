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
file_name = 'eme_interactions_June-July2018_NY.csv'  # 'eme_interactsions_June2018.csv'
root_dir = './raw'


def main():
    n_target = 100
    max_step = 20
    train_env = Environment(file_name, root_dir,
                            n_target=n_target, max_step=max_step,
                            high_rate=.5, train=True)
    train_agent = Agent(train_env.dim_in_feature, n_target)  #, env.n_action)

    test_env = Environment(file_name, root_dir,
                           n_target=n_target, max_step=max_step,
                           high_rate=.5, train=False)
    test_agent = Agent(test_env.dim_in_feature, n_target)  #, env.n_action)
    test_agent.steps_done = 1e10

    num_episodes = 5000
    for i_episode in range(num_episodes):
        t_reward = 0
        t_loss = 0

        test_t_reward = 0
        for t in count():
            state, target_features, current_user_id, target_ids = train_env.obs()
            # Select and perform an action
            action = train_agent.select_action(state, target_features)
            target_id = target_ids[:, action.item()].item()
            reward, done = train_env.step(current_user_id, target_id, t)
            reward = torch.tensor([reward], device=train_agent.device)

            # Observe new state
            if not done:
                # next_state = state.clone()
                next_state, _, _, _ = train_env.obs()
            else:
                next_state = None

            # Store the transition in memory
            train_agent.memory.push(state, target_features, action, next_state, reward)

            # Perform one step of the optimization (on the target network)
            loss = train_agent.optimize_model()
            t_reward += reward.data.item()
            if loss:
                t_loss += loss
                # print reward and loss
            if done:
                break

        for t in count():
            state, target_features, current_user_id, target_ids = test_env.obs()
            # Select and perform an action
            action = test_agent.select_action(state, target_features)
            target_id = target_ids[:, action.item()].item()
            reward, done = test_env.step(current_user_id, target_id, t)
            test_t_reward += reward
            if done:
                break

        if i_episode % TARGET_UPDATE == 0:
            print('Episode: {} Train Reward: {:.3f} Loss: {:.3f} Test Reward: {:.3f}'.format(
                i_episode, t_reward, t_loss, test_t_reward))
            # Update the target network
            train_agent.update_target_network()

    print('Complete')


if __name__=='__main__':
    main()
