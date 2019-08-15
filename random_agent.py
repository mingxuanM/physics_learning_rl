import argparse
from interaction_env import Interaction_env
import numpy as np
import time
import random as rd
import json

from config import n_actions, action_length

# n_actions = 6 # 1 no action + 4 directions acc + 1 click
# # n_actions = 4*2 # 4 directions * 2 if click
# action_length = 5 # frames
# # RQN_num_feats = 22 # 4 caught object + 2 mouse + 4*4

class Random_agent():
    # def __init__(self):

    def get_action(self, state_t):
        return rd.randint(0,n_actions-1)
    
# run one episode
# t_max: maximum running time
def train_iteration(random_agent, environment, t_max):
    session_reward = []
    seesion_predictor_loss = []
    td_loss = []
    s = environment.reset() # first action_length frames * 22 num_feats
    # print(s)
    t = 0
    while t < t_max:
        a = random_agent.get_action(s)
        if a<0 or a>=n_actions:
            print('invalid actionID {} break'.format(a))
            break
        # a = 5
        trajectory, reward, is_done, predictor_loss = environment.act(a)
        s_next = trajectory # action_length frames * 22 num_feats
        session_reward.append(reward)
        seesion_predictor_loss.append(predictor_loss)
        s = s_next
        # a_string = 'none'
        # if a == 0: 
        #     a_string = 'none' 
        # elif a== 1:
        #     a_string = 'up' 
        # elif a== 2:
        #     a_string = 'down' 
        # elif a== 3:
        #     a_string = 'right' 
        # elif a== 4:
        #     a_string = 'left' 
        # elif a== 5:
        #     a_string = 'click' 
        
        # print('[step {}]\t action: {} ({}); reward: {}. '.format(t,a,a_string,reward) + time.strftime("%H:%M:%S", time.localtime()))
        # print('\t{}'.format(s))
        # print(s)
        if is_done:
            break
        t += action_length
    trajectory_history = environment.destory()
    
    return session_reward, seesion_predictor_loss, trajectory_history, is_done

# Top level training loop, over epochs
def train_loop(random_agent, environment, episode, timeout):
    rewards = []
    data = []
    # time_taken = []
    # succeed_episode = 0
    for i in range(episode):
        # print('[session {} started]\t '.format(i) + time.strftime("%H:%M:%S", time.localtime()))
        session_reward, seesion_predictor_loss, trajectory_history, is_done = train_iteration(random_agent, environment, timeout)
        data.append(trajectory_history)
        session_reward_mean = np.mean(session_reward)
        seesion_predictor_loss_mean = np.mean(seesion_predictor_loss)
        print('[session {} finished]\t '.format(i) + time.strftime("%H:%M:%S", time.localtime()) + " mean reward = {:.4f}; total reward = {:.4f}".format(
            session_reward_mean, np.sum(session_reward)))
        
        # print('number of timesteps taken: \t{}'.format(len(session_reward)*5))
        print('predictor loss: {}'.format(seesion_predictor_loss))
        print('mean: {}'.format(seesion_predictor_loss_mean))
        print(' ')
        rewards.append(session_reward_mean)
    #     if is_done>0:
    #         succeed_episode += 1
    #         time_taken.append(len(session_reward))
    # print('agent succeed in catching object in {}/{} ({}%) episodes'.format(succeed_episode, episode, succeed_episode/episode*100))
    # print('End of training, average actions to catch: {}'.format(np.mean(time_taken)))
    # Write to json file
    with open('./model_predictor/data/random_training_data.json', 'w') as data_file:
        json.dump(data, data_file, indent=4)
    return data
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='run a random agent')
    parser.add_argument('--episode', type=int, action='store',
                        help='number of epoches to run', default=200)
    parser.add_argument('--timeout', type=int, action='store',
                        help='max number of frames for one episode, 1/60s per frame', default=1800)

    args = parser.parse_args()

    # total_feats = 22
    # n_state = 16
    # input_frames = 5
    # epsilon_decay = 0.9

    exp_name = 'Random_agent'


    # initialize interaction_env
    environment = Interaction_env()
    # initialize learning_agent and target_agent
    # qlearning_gamma = 0.9
    # n_actions = 6
    random_agent = Random_agent()

    # initialize json file
    # data = []
    # with open('./active_training_data/random_data.json', 'w') as data_file:
    #     json.dump(data, data_file, indent=4)
    # train
    _ = train_loop(random_agent, environment, args.episode, args.timeout)