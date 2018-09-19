import copy
import argparse
from itertools import count
import torch

from model.agent_simple import Agent
from model.environment_for_owndata import Environment


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
file_name = 'eme_interactsions_June-July2018_NY_features.csv'
root_dir = './raw'


def main():
    n_target = 3000
    max_step = 100
    train_env = Environment(file_name, root_dir,
                            n_target=n_target, max_step=max_step,
                            high_rate=.5, train=True)
    train_agent = Agent(train_env.dim_in_feature, n_target)

    test_env = Environment(file_name, root_dir,
                           n_target=n_target, max_step=max_step,
                           high_rate=.9, train=False)
    test_agent = Agent(test_env.dim_in_feature, n_target)

    assert train_env.dataset.user_features.shape[1:]==test_env.dataset.user_features.shape[1:],\
        "number of train features and test feature must be same"
    assert train_env.dataset.target_features_all.shape[1:]==test_env.dataset.target_features_all.shape[1:],\
        "number of train features and test feature must be same"

    num_episodes = 5000
    for i_episode in range(num_episodes):
        t_reward = 0
        t_loss = 0

        # check when reset eager parameter
        # https://github.com/chinancheng/DDQN-pytorch/blob/44f6e12e28cee8185fe94dd98f6fba5994a8ad36/main.py
        # train_agent.steps_done = 4000 if train_agent.steps_done > 4000 else train_agent.steps_done
        test_t_reward = 0
        for t in count():
            state, target_features, current_user_id, target_ids = train_env.obs()
            # print(state.shape, target_features.shape)
            # print(current_user_id.item(), target_ids.numpy())
            assert len(target_features[0]) == n_target, "#target_ids must much n_target"

            # Select and perform an action
            action = train_agent.select_action(state, target_features)
            target_id = target_ids[:, action.item()]
            reward, done = train_env.step(current_user_id, target_id, t)
            reward = torch.tensor([reward], device=train_agent.device)

            # Observe new state
            if not done:
                next_state, _, _, _ = train_env.obs()
            else:
                next_state = None

            # Store the transition in memory
            train_agent.memory.push(state, target_features, action, next_state, reward)
            # if reward > 0:
            #     train_agent.with_reward_memory.push(state, target_features, action, next_state, reward)
            # else:
            #     train_agent.without_reward_memory.push(state, target_features, action, next_state, reward)

            # Perform one step of the optimization (on the target network)
            loss = train_agent.optimize_model()
            t_reward += reward.data.item()
            if loss:
                t_loss += loss
                # print reward and loss
            if done:
                break

            train_agent.epsilon = (train_agent.epsilon - 1e-6) if train_agent.epsilon > 0.1 else 0.1

        # print("1 train loop done")
        test_agent.policy_net.load_state_dict(train_agent.policy_net.state_dict())
        for t in count():
            state, target_features, current_user_id, target_ids = test_env.obs()
            # Select and perform an action
            action5, action5_idxs =\
                test_agent.select_action_for_test(state, target_features, n_best=5)
            target_id5 = target_ids[0, action5_idxs].view(-1)
            reward, done = test_env.step(current_user_id, target_id5, t)
            test_t_reward += reward
            if done:
                break
        # print("test loop done")

        if i_episode % TARGET_UPDATE == 0:
            print('Episode: {} Train Reward: {:.3f} Loss: {:.3f} Test Reward: {:.3f}'.format(
                i_episode, t_reward, t_loss, test_t_reward))
            # Update the target network
            train_agent.update_target_network()

    print('Complete')


if __name__=='__main__':
    main()
