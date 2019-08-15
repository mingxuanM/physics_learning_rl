#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RQN_agent.py
RL training process
Q_agent: RL agent class, contains all networks and training function.
    get_q_value()
    get_target()
    train_network()
    get_action()
    set_epsilon()

Interaction_env: the environment class that the agent interacts with.
    reset()
    act()
    reward_cal()
    action_generation()

"""
import argparse
from interaction_env import Interaction_env
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
# import keras
# import keras.layers as L
import time
import sys
import json

from config import n_actions, RQN_num_feats, action_length, qlearning_gamma

# n_actions = 6 # 1 no action + 4 directions acc + 1 click
# qlearning_gamma = 0.9
# # n_actions = 4*2 # 4 directions * 2 if click
# action_length = 5 # frames
# RQN_num_feats = 22 # 4 caught object + 2 mouse + 4*4

# Workflow:
# learning_agent.get_action(state_t) -> action -> 
# env.step(action) -> (state_next, reward, is_done) ->
# target_agent.get_target(state_next, reward) -> target ->
# learning_agent.train_step(state_t, action, target) -> loss

class Q_agent:
    def __init__(self, name, n_actions, qlearning_gamma, input_frames=action_length, num_feats=RQN_num_feats, epsilon=0.99):
        self.n_actions = n_actions
        self.qlearning_gamma = qlearning_gamma
        self.epsilon = epsilon
        with tf.variable_scope(name):
            self.nn = tf.keras.models.Sequential()
            self.nn.add(tf.keras.layers.InputLayer(input_shape=(input_frames,num_feats,)))
            # tf.keras.layers.CuDNNLSTM is GPU optimised, switch to tf.keras.layers.LSTM if using CPU
            self.nn.add(tf.keras.layers.CuDNNLSTM(22))
            # self.nn.add(tf.keras.layers.LSTM(n_state, recurrent_activation='sigmoid'))
            self.nn.add(tf.keras.layers.Dense(n_actions))

            # Predicting q values for all actions with 5 frames input
            self.state_t = tf.placeholder(
                tf.float32, [1,input_frames, num_feats])
            self.prediction = self.nn(self.state_t)

            self.action_t = tf.placeholder(tf.int32)
            # self.target_t = tf.placeholder(tf.float32, [1,1])
            self.target_t = tf.placeholder(tf.float32)

            # mse loss
            # self.loss_ = (tf.reduce_sum(self.prediction * tf.one_hot(self.action_t, n_actions), axis=1) - self.target_t) ** 2
            self.loss = tf.reduce_mean((tf.reduce_sum(self.prediction * tf.one_hot(self.action_t, n_actions), axis=1) - self.target_t) ** 2)

        self.weights = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        self.train_step = tf.train.AdamOptimizer(
                1e-3).minimize(self.loss, var_list=self.weights)

        self.saver = tf.train.Saver(self.weights)

    # for forward prediction
    def get_q_value(self, state_t, action):
        sess = tf.get_default_session()
        q_values = sess.run(self.prediction, {self.state_t: state_t})
        q_pred_a = tf.reduce_sum(q_values * tf.one_hot(action, n_actions), axis=1)
        return q_pred_a
    
    # for target network only
    # reward: [batch_size, scalar]
    # is_done: [batch_size, bool]
    def get_target(self, state_next, reward, is_done):
        sess = tf.get_default_session()
        q_values_next = sess.run(self.prediction, {self.state_t: state_next})
        q_target_a = reward + self.qlearning_gamma * np.amax(q_values_next) # tf.reduce_max(q_values_next, axis=1)
        # q_target_a = tf.where(is_done, reward, q_target_a)
        return q_target_a

    # train network and return loss
    # target: calculated from target network
    def train_network(self, state_t, action, target):
        sess = tf.get_default_session()
        _train_step, _loss, _prediction = sess.run([self.train_step, self.loss, self.prediction], 
            {self.state_t: state_t, self.action_t: action, self.target_t: target})
        return _loss

    # sample action for a given state
    def get_action(self, state_t):
        thre = np.random.rand()
        if thre < self.epsilon:
            action = np.random.choice(n_actions, 1)[0]
        else:
            sess = tf.get_default_session()
            q_values = sess.run(self.prediction, {self.state_t: state_t})
            action = np.argmax(q_values)
        return action
    
    def set_epsilon(self, epsilon_):
        self.epsilon = epsilon_

def load_weigths_into_target_network(agent, target_network):
    assigns = []
    for w_agent, w_target in zip(agent.weights, target_network.weights):
        assigns.append(tf.assign(w_target, w_agent, validate_shape=True))
    tf.get_default_session().run(assigns)

# run one episode
# t_max: maximum running time
# train：if True, calculate loss and call train_step
def train_iteration(learning_agent, target_agent, env, t_max, train=False):
    
    session_reward = []
    seesion_predictor_loss = []
    td_loss = []
    s = env.reset() # first 10 frames * 22 num_feats
    s=s.reshape((1,action_length,RQN_num_feats))
    t = 0
    while t < t_max:
        a = learning_agent.get_action(s)
        # print('action')
        # print(a)
        trajectory, reward, is_done, predictor_loss = env.act(a)
        s_next = trajectory # 10 frames * 22 num_feats
        s_next=s_next.reshape((1,action_length,RQN_num_feats))
        if train:
            target = target_agent.get_target(s_next, reward, is_done)
            loss = learning_agent.train_network(s, a, target)
            td_loss.append(loss)
        session_reward.append(reward)
        seesion_predictor_loss.append(predictor_loss)
        s = s_next
        if is_done:
            break
        t += action_length
    trajectory_history = env.destory()
    
    return session_reward, td_loss, is_done, seesion_predictor_loss, trajectory_history

# Top level training loop, over epochs
def train_loop(learning_agent, target_agent, env, episode, train, timeout, continue_from=0, save_model=False):
    # rewards = []
    # loss = []
    # succeed_episode = 0
    # time_taken = []
    data = []
    for i in range(episode):
        # print('[session {} started] '.format(i) + time.strftime("%H:%M:%S", time.localtime()))
        session_reward, td_loss, is_done, session_predictor_loss, trajectory_history = train_iteration(learning_agent, target_agent, env, timeout, train)
        if not train:
            data.append(trajectory_history)

        session_reward_mean = np.mean(session_reward)
        session_predictor_loss_mean = np.mean(session_predictor_loss)
        td_loss_mean = np.mean(td_loss) 
        print('[session {} finished] '.format(i) + time.strftime("%H:%M:%S", time.localtime()) + ';\t mean reward = {:.4f};\t mean loss = {:.4f};\t total reward = {:.4f};\t epsilon = {:.4f}'.format(
            session_reward_mean, td_loss_mean, np.sum(session_reward),learning_agent.epsilon))
        print('predictor loss: {}'.format(session_predictor_loss_mean))
        # rewards.append(session_reward_mean)
        # loss.append(td_loss_mean)
        # load_weigths_into_target_network and adjust agent parameters
        if train:
            if i%2==0:
                load_weigths_into_target_network(learning_agent, target_agent)
                # learning_agent.set_epsilon(max(learning_agent.epsilon * epsilon_decay, 0.01))
                learning_agent.set_epsilon(max(1-i/episode, 0.01))
            if i%100==0 and i>0 and save_model:
                save_path = learning_agent.saver.save(sess, "./checkpoints/{}_{}_epochs.ckpt".format(exp_name, i + continue_from))
                target_save_path = target_agent.saver.save(sess, "./checkpoints/{}_target_{}_epochs.ckpt".format(exp_name, i + continue_from))
                print("Model saved in path: %s" % save_path)
        # Count and print for catching records
        # if is_done:
        #     succeed_episode += 1
        #     time_taken.append(len(session_reward))
    # print('agent succeed in catching object in {}/{} ({}%) episodes'.format(succeed_episode, episode, succeed_episode/episode*100))
    # print('End of training, average actions to catch: {}'.format(np.mean(time_taken)))

    if not train:
        with open('./model_predictor/data/active_training_data.json', 'w') as data_file:
            json.dump(data, data_file, indent=4)
        return data

    if save_model and train:
        # model_json = learning_agent.nn.to_json()
        # with open('{}.json'.format(exp_name), 'w') as json_file:
        #     json_file.write(model_json)
        # agent.nn.save_weights('{}.h5'.format(exp_name))
        save_path = learning_agent.saver.save(sess, "./checkpoints/{}_{}_epochs.ckpt".format(exp_name, episode + continue_from))
        target_save_path = target_agent.saver.save(sess, "./checkpoints/{}_target_{}_epochs.ckpt".format(exp_name, episode + continue_from))
        print("Model saved in path: %s" % save_path)
        print("Model saved!")
        # np.savetxt('{}.txt'.format(exp_name), (rewards, loss))
        # print("Training details saved!")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='training a recurrent q-network')
    parser.add_argument('--episode', type=int, action='store',
                        help='number of epoches to train', default=50)
    
    parser.add_argument('--save_model', type=bool, action='store', 
                        help='save trained model or not', default=False)

    parser.add_argument('--active_learning', type=bool, action='store', 
                        help='load pretrained RQN & model predictor to do active learning', default=False)
    
    parser.add_argument('--train', type=bool, action='store',
                        help='if to train a model', default=False)
    
    parser.add_argument('--epsilon', type=float, action='store',
                        help='epsilon for Q learning', default=0.99)

    parser.add_argument('--lr', type=float, action='store',
                        help='learning rate for Adam optimiser', default=1e-4)

    parser.add_argument('--timeout', type=int, action='store',
                        help='max number of frames for one episode, 1/60s per frame', default=1800)

    parser.add_argument('--continue_from', type=int, action='store',
                        help='continue training from previous trained model', default=0)

    args = parser.parse_args()


    # RQN_num_feats = 22
    # input_frames = 5
    epsilon_decay = 0.9
    if args.episode >= 10000:
        epsilon_decay = 0.9995 # 10000 epochs
    elif args.episode >= 1000:
        epsilon_decay = 0.999 # 2000 epochs
    else:
        epsilon_decay = 0.95

    # exp_name = 'RQN_20_{:1.0e}'.format(args.lr) # 20 reward for successful catching + bounded getting close reward
    # exp_name = 'RQN_bonded_{:1.0e}'.format(args.lr) # reward for getting close to nearest puck is bounded
    # exp_name = 'RQN_more_reward_{:1.0e}'.format(args.lr) # add reward for getting close to nearest puck
    # exp_name = 'RQN_{:1.0e}'.format(args.lr) # only 5 reward for successful catching

    exp_name = 'new_active_learning_world-1'
    # exp_name = 'new_catch_training'

    # tf.reset_default_graph()
    # sess = tf.InteractiveSession()

    # keras.backend.set_session(sess)
    # initialize interaction_env
    environment = Interaction_env()

    # initialize learning_agent and target_agent
    if not args.active_learning:
        # train for catching pucks
        if args.train:
            rqn_agent_graph = tf.Graph()
            with rqn_agent_graph.as_default():
                learning_agent = Q_agent("learning_agent", n_actions, qlearning_gamma, epsilon=args.epsilon)
                target_agent = Q_agent("target_agent", n_actions, qlearning_gamma, epsilon=args.epsilon)
            sess = tf.InteractiveSession(graph = rqn_agent_graph)

            sess.run(tf.global_variables_initializer())
            # environment.predictor.saver.restore(sess, "./model_predictor/checkpoints/pretrained_model_predictor_2.ckpt")
            if args.continue_from > 0:
                learning_agent.saver.restore(sess, "./checkpoints/{}_{}_epochs.ckpt".format(exp_name,args.continue_from))
                target_agent.saver.restore(sess, "./checkpoints/{}_target_{}_epochs.ckpt".format(exp_name,args.continue_from))
        else:
            if args.continue_from == 0:
                sys.exit('[ERROR] test model not specified')
            rqn_agent_graph = tf.Graph()
            with rqn_agent_graph.as_default():
                learning_agent = Q_agent("learning_agent", n_actions, qlearning_gamma, epsilon=0)
            sess = tf.InteractiveSession(graph = rqn_agent_graph)
            sess.run(tf.global_variables_initializer())
            learning_agent.saver.restore(sess, "./checkpoints/{}_{}_epochs.ckpt".format(exp_name,args.continue_from))
            target_agent = None
    else:

        # train for active learning
        if args.train:
            rqn_agent_graph = tf.Graph()
            with rqn_agent_graph.as_default():
                learning_agent = Q_agent("learning_agent", n_actions, qlearning_gamma, epsilon=args.epsilon)
                target_agent = Q_agent("target_agent", n_actions, qlearning_gamma, epsilon=args.epsilon)
            sess = tf.InteractiveSession(graph = rqn_agent_graph)
            sess.run(tf.global_variables_initializer())
            # RQN agent trained on 10000 episodes with bonded reward "./checkpoints/RQN_bonded_1e-04_10000_epochs.ckpt"
            learning_agent.saver.restore(sess, "./checkpoints/trained_RQN_catching.ckpt")
            target_agent.saver.restore(sess, "./checkpoints/trained_RQN_catching_target.ckpt")
        else:
            print('Active data generation started...')
            if args.continue_from == 0:
                sys.exit('[ERROR] test model not specified')
            rqn_agent_graph = tf.Graph()
            with rqn_agent_graph.as_default():
                learning_agent = Q_agent("learning_agent", n_actions, qlearning_gamma, epsilon=0)
            sess = tf.InteractiveSession(graph = rqn_agent_graph)
            sess.run(tf.global_variables_initializer())
            # RQN agent trained on 10000 episodes with bonded reward "./checkpoints/RQN_bonded_1e-04_10000_epochs.ckpt"
            learning_agent.saver.restore(sess, "./checkpoints/active_learning_loss_reward_world-1_{}_epochs.ckpt".format(args.continue_from))
            target_agent = None

   
    # train
    _ = train_loop(learning_agent, target_agent, environment, args.episode, args.train, args.timeout, args.continue_from, args.save_model)