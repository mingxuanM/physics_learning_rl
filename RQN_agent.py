#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RQN.py
"""
import argparse
from interaction_env import Interaction_env
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.layers as L
import time

n_actions = (16*(1+1+1) + 1) * 5
action_length = 10 # frames
RQN_num_feats = 23 # 4+2 mouse + 4*4 + 1 caught object

# Workflow:
# learning_agent.get_action(state_t) -> action -> 
# env.step(action) -> (state_next, reward, is_done) ->
# target_agent.get_target(state_next, reward) -> target ->
# learning_agent.train_step(state_t, action, target) -> loss

class Q_agent:
    def __init__(self, name, n_actions, qlearning_gamma, input_frames=10, num_feats=23, epsilon=0.8):
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
            self.target_t = tf.placeholder(tf.float32, [1,1])

            # mse loss
            # self.loss_ = (tf.reduce_sum(self.prediction * tf.one_hot(self.action_t, n_actions), axis=1) - self.target_t) ** 2
            self.loss = tf.reduce_mean((tf.reduce_sum(self.prediction * tf.one_hot(self.action_t, n_actions), axis=1) - self.target_t) ** 2)

        self.weights = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        self.train_step = tf.train.AdamOptimizer(
                1e-3).minimize(self.loss, var_list=self.weights)

        self.saver = tf.train.Saver()

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
        q_target_a = reward + self.qlearning_gamma * tf.reduce_max(q_values_next, axis=1)
        q_target_a = tf.where(is_done, reward, q_target_a)
        return q_target_a

    # train network and return loss
    # target: calculated from target network
    def train_network(self, state_t, action, target):
        sess = tf.get_default_session()
        _train_step, _loss, _prediction = sess.run(self.train_step, self.loss, self.prediction, 
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
def train_iteration(t_max, train=False):
    print('[beginning of session] ' + time.strftime("%H:%M:%S", time.localtime()))
    session_reward = []
    td_loss = []
    s = environment.reset() # first 10 frames * 23 num_feats
    t = 0
    while t < t_max:
        a = learning_agent.get_action(s)
        trajectory, reward, is_done = environment.act(a)
        s_next = trajectory # 10 frames * 23 num_feats
        if train:
            target = target_agent.get_target(s_next, reward, is_done)
            loss = learning_agent.train_network(s, a, target)
            td_loss.append(loss)
        session_reward.append(reward)
        s = s_next
        if is_done:
            break
        t += action_length
    print('[end of session] ' + time.strftime("%H:%M:%S", time.localtime()))
    return session_reward, td_loss

# Top level training loop, over epochs
def train_loop(args):
    rewards = []
    loss = []
    for i in range(args.epochs):
        session_reward, td_loss = train_iteration(500, True)
        session_reward_mean = np.mean(session_reward)
        td_loss_mean = np.mean(td_loss)
        print("Session {}\t finished: mean reward = {:.4f}\t mean loss = {:.4f}\t total reward = {:.4f}\t epsilon = {:.4f}".format(
            i, session_reward_mean, td_loss_mean, np.sum(session_reward),learning_agent.epsilon))
        rewards.append(session_reward_mean)
        loss.append(td_loss_mean)
        # load_weigths_into_target_network and adjust agent parameters
        if i % 2 == 0:
            load_weigths_into_target_network(learning_agent, target_agent)
            learning_agent.set_epsilon(max(learning_agent.epsilon * epsilon_decay, 0.01))

    if args.save_model:
        model_json = learning_agent.nn.to_json()
        with open('{}.json'.format(exp_name), 'w') as json_file:
            json_file.write(model_json)
        # agent.nn.save_weights('{}.h5'.format(exp_name))
        save_path = learning_agent.saver.save(sess, "./chechpoints/{}_{}_epochs.ckpt".format(exp_name, args.epochs))
        print("Model saved in path: %s" % save_path)
        print("Model saved!")
        np.savetxt('{}.txt'.format(exp_name), (rewards, loss))
        print("Training details saved!")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='training a recurrent q-network')
    parser.add_argument('--epochs', type=int, action='store',
                        help='number of epoches to train', default=50)
    
    parser.add_argument('--save_model', type=bool, action='store', 
                        help='save trained model or not', default=True)
    
    parser.add_argument('--train', type=bool, action='store',
                        help='if to train a model', default=False)
    
    parser.add_argument('--lr', type=float, action='store',
                        help='learning rate for Adam optimiser', default=1e-4)

    args = parser.parse_args()

    total_feats = 23
    n_state = 16
    input_frames = 5
    epsilon_decay = 0.9

    exp_name = 'RQN'

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    keras.backend.set_session(sess)
    # initialize interaction_env
    environment = Interaction_env()
    # initialize learning_agent and target_agent
    qlearning_gamma = 0.9
    n_actions = 16*(2+2+1) + 1
    learning_agent = Q_agent("learning_agent", n_actions, qlearning_gamma)
    target_agent = Q_agent("target_agent", n_actions, qlearning_gamma)
    # train
    train_loop(args)