#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RQN.py
"""
import argparse
from interaction_env import interaction_env
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.layers as L
import time

# Workflow:
# learning_agent.get_action(state_t) -> action -> 
# env.step(action) -> (state_next, reward, is_done) ->
# target_agent.get_target(state_next, reward) -> target ->
# learning_agent.train_step(state_t, action, target) -> loss

class q_agent:
    def __init__(self, name, n_actions, input_frames=5,num_feats=14, epsilon=0):
        self.n_actions = n_actions
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
    def get_target(self, state_next, reward):
        sess = tf.get_default_session()
        q_values_next = sess.run(self.prediction, {self.state_t: state_next})
        q_target_a = reward + qlearning_gamma * tf.reduce_max(q_values_next, axis=1)
        q_target_a = tf.where(is_done_ph, r_ph, q_target_a)
        return q_target_a

    # train network and return loss
    def train_step(self, state_t, action, target):
        sess = tf.get_default_session()
        _train_step, _loss, _prediction = sess.run(self.train_step, self.loss, self.prediction, 
            {self.state_t: state_t, self.action_t: action, self.target_t: target})
        return _loss

    # sample action for a given state
    def get_action(self, state_t):
        epsilon = self.epsilon
        thre = np.random.rand()
        if thre < epsilon:
            action = np.random.choice(n_actions, 1)[0]
        else:
            sess = tf.get_default_session()
            q_values = sess.run(self.prediction, {self.state_t: state_t})
            action = np.argmax(q_values)
        return action

def load_weigths_into_target_network(agent, target_network):
    assigns = []
    for w_agent, w_target in zip(agent.weights, target_network.weights):
        assigns.append(tf.assign(w_target, w_agent, validate_shape=True))
    tf.get_default_session().run(assigns)

# run one episode
# t_max: maximum running time
# train： if True, calculate loss and call train_step
def train_iteration(t_max, train=False):
    print('[beginning of session] ' + time.strftime("%H:%M:%S", time.localtime()))
    total_reward = 0
    total_reward_others = 0
    td_loss = 0
    s = new_env.reset()
    s = np.transpose(np.array(s).reshape(num_feats/2,T,2),[0,2,1]).flatten().reshape(num_feats, T).T

    for t in range(t_max):
        a = agent.get_action(s)
        s_next, r, is_done, r_others = new_env.step(a)
        s_next = np.transpose(np.array(s_next).reshape(num_feats/2,T,2),[0,2,1]).flatten().reshape(num_feats, T).T

        if train:
            _, loss_t = sess.run([train_step, train_agent.loss], {train_agent.s_ph: [s], train_agent.a_ph: [a], train_agent.r_ph: [
                     r], train_agent.s_next_ph: [s_next], train_agent.is_done_ph: [is_done]})

        total_reward += r
        total_reward_others += r_others
        td_loss += loss_t
        s = s_next
        if is_done:
            break
    print('[end of session] ' + time.strftime("%H:%M:%S", time.localtime()))
    return [total_reward, total_reward_others * reward_stop, td_loss]

# Top level training loop, over epochs
def train_loop(args):
    rewards = []
    rewards_others = []
    loss = []

    for i in range(args.epochs):
        # Call train_iteration() number of sessions times
        results = [train_iteration(
            t_max=1000, train=True) for t in range(args.sessions)]
        epoch_rewards = [r[0] for r in results]
        epoch_rewards_others = [r[1] for r in results]
        epoch_loss = [l[2] for l in results]
        rewards += epoch_rewards
        rewards_others += epoch_rewards_others
        loss += epoch_loss
        print("epoch {}\t mean reward = {:.4f}\t mean reward (others) = {:.4f}\t mean loss = {:.4f}\t total reward = {:.4f}\t epsilon = {:.4f}".format(
            i, np.mean(epoch_rewards), np.mean(epoch_rewards_others), np.mean(epoch_loss), np.sum(rewards),agent.epsilon))
        # adjust agent parameters
        if i % 2 == 0:
            load_weigths_into_target_network(agent, target_network)
            agent.epsilon = max(agent.epsilon * epsilon_decay, 0.01)

    if args.save_model:
        model_json = agent.nn.to_json()
        with open('RQN_{}.json'.format(name), 'w') as json_file:
            json_file.write(model_json)
        agent.nn.save_weights('RQN_{}.h5'.format(name))
        print("Model saved!")
        np.savetxt('RQN_{}.txt'.format(name), (rewards, rewards_others, loss))
        print("Training details saved!")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='training a recurrent q-network')
    parser.add_argument('--epochs', type=int, action='store',
                        help='number of epoches to train', default=100)
    parser.add_argument('--mode', type=int, action='store',
                        help='type of intrinsic reward, 1 for mass, 2 for force', default=1)
    parser.add_argument('--save_model', type=bool, action='store', help='save trained model or not', default=True)
    parser.add_argument('--sessions', type=int, action='store',
                        help='number of sessions to train per epoch', default=10)

    args = parser.parse_args()

    total_feats=22
    num_feats=14 # [4 one-hot object, mx, my, o1.x, o1.y, o2.x, o2.y, o3.x, o3.y, o4.x, o4.y]
    n_state =16
    input_frames = 5

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    keras.backend.set_session(sess)

    env = interaction_env()
    # initialize q agent and target network
    agent = q_agent("agent", num_feats, n_actions)
    target_network = q_agent("target_network", num_feats, n_actions)
    # train
    train_loop(args)