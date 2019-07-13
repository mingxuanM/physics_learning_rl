import argparse
from interaction_env import Interaction_env
import numpy as np
import time
import random as rd

n_actions = 4*2 # 4 directions * 2 if click
action_length = 10 # frames
# RQN_num_feats = 22 # 4 caught object + 2 mouse + 4*4

class Random_agent():
    # def __init__(self):

    def get_action(self, state_t):
        return rd.randint(0,n_actions-1)
    
# run one episode
# t_max: maximum running time
def train_iteration(t_max):
    session_reward = []
    td_loss = []
    s = environment.reset() # first action_length frames * 22 num_feats
    # print(s)
    t = 0
    while t < t_max:
        a = random_agent.get_action(s)
        trajectory, reward, is_done = environment.act(a)
        s_next = trajectory # action_length frames * 22 num_feats
        session_reward.append(reward)
        s = s_next
        print('[step {}]\t action: {}; reward: {}. '.format(t,a,reward) + time.strftime("%H:%M:%S", time.localtime()))
        # print(s)
        if is_done:
            break
        t += action_length
    environment.destory()
    
    return session_reward

# Top level training loop, over epochs
def train_loop(args):
    rewards = []
    time_taken = []
    for i in range(args.epochs):
        print('[session {} started]\t '.format(i) + time.strftime("%H:%M:%S", time.localtime()))
        session_reward = train_iteration(args.timeout)
        session_reward_mean = np.mean(session_reward)
        print('[session {} finished]\t '.format(i) + time.strftime("%H:%M:%S", time.localtime()) + " mean reward = {:.4f}; total reward = {:.4f}".format(
            session_reward_mean, np.sum(session_reward)))
        rewards.append(session_reward_mean)
        time_taken.append(len(session_reward))
        print()
    print('End of training, average time to catch: {}'.format(np.mean(time_taken)))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='run a random agent')
    parser.add_argument('--epochs', type=int, action='store',
                        help='number of epoches to run', default=50)
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
    n_actions = 4*2
    random_agent = Random_agent()
    # train
    train_loop(args)