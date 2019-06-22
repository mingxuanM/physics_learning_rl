#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predictor.py
"""
import sys
sys.path.append('../simulator/')
import argparse
import json
# from config import *
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.layers as L
import time


# num_feats = 4 (one hot) + 2 +4*4 = 22 (was 32 with world properties)
# n_state = 4*4 = 16 (4 objects * (2 locations + 2 velocities))
# truncated_backprop_length: 10 time steps in each training step(probably one action step)
# Every 5 frames of inputs are used to predict the next frame
# num_feats=32
num_feats=22
n_state =16
truncated_backprop_length = 5
input_frames = 5
batch_size = 3

class Predictor:
    def __init__(self, name='predictor', num_feats=22, n_state=16, input_frames=5, train=True):
        if not train:
            batch_size = 1
            truncated_backprop_length = 1
        else:
            batch_size = 3
            truncated_backprop_length = 5
        with tf.variable_scope(name):
            self.nn = tf.keras.models.Sequential()
            self.nn.add(tf.keras.layers.InputLayer(input_shape=(input_frames,num_feats,), batch_size = batch_size))
            self.nn.add(tf.keras.layers.CuDNNLSTM(n_state))
            self.nn.add(tf.keras.layers.Dense(n_state))

            # Predicting 1 frame: 5 frames input, property_combination + state
            self.state_t = tf.placeholder(
                tf.float32, [batch_size, input_frames, num_feats])
            # one frame prediction of one batch
            self.prediction = self.nn(self.state_t)

            # Training truncated_backprop_length frames
            self.training_states = tf.placeholder(
                tf.float32, [truncated_backprop_length, batch_size, input_frames, num_feats])
            training_states_series = tf.unstack(self.training_states, axis=0)
            self.batch_predictions = tf.stack([self.nn(training_state) for training_state in training_states_series], axis=1)

            self.batch_labels = tf.placeholder(
                 tf.float32, [batch_size, truncated_backprop_length, n_state])
        
            self.batch_losses = tf.reduce_mean(tf.squared_difference(self.batch_predictions, self.batch_labels), axis=[0,1])

        self.weights = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        self.train_step = tf.train.AdamOptimizer(
                1e-4).minimize(self.batch_losses, var_list=self.weights)

        self.saver = tf.train.Saver()


# run one batch, truncated_backprop_length frames
# batch_sequence: [batch_size, truncated_backprop_length + input_frames, num_feats]
def train_iteration(batch_sequence):   
    labels = np.zeros((truncated_backprop_length, batch_size, n_state))
    inputs = np.zeros((truncated_backprop_length, batch_size, input_frames, num_feats))
    for t in range(truncated_backprop_length):
        inputs[t] = batch_sequence[:,t:(t+input_frames),:]
        labels[t] = batch_sequence[:,t+input_frames,-16:]

    labels_feed = np.swapaxes(labels,0,1)
    _train_step, _batch_losses = sess.run([model_predictor.train_step, model_predictor.batch_losses], {model_predictor.batch_labels: labels_feed, model_predictor.training_states: inputs})
    return _batch_losses

# Top level training over epochs
# training_sequense in [batch_size*-1*num_feats]
def train_sequense(epochs, save_model, training_sequenses):
    # mean losses during each epoch
    training_losses = []
    training_losses_time = []
    for i in range(epochs):
        epoch_loss = []
        for sequence_idx, _training_sequense in enumerate(training_sequenses):
            sequence_loss = []
            # cut out excess frames so that _training_sequense can be reshaped by batch_size
            excessed = int(_training_sequense.shape[0] % batch_size)
            _training_sequense = _training_sequense[:-excessed,:]
            training_sequense = np.reshape(_training_sequense, (batch_size, -1, num_feats))
            total_batches = (training_sequense.shape[1]-input_frames)//truncated_backprop_length
            # run batch loop
            for batch_idx in range(total_batches):
                start = batch_idx * truncated_backprop_length
                end = start + truncated_backprop_length + input_frames
                batch_loss = train_iteration(training_sequense[:,start:end,:])
                sequence_loss.append(batch_loss)
                # if sequence_idx >= 100:
                #     print('batch loss of {}\t: '.format(batch_idx))
                #     print(batch_loss)
            mean_sequence_loss = np.mean(sequence_loss)
            print('Training sequence {}\t finished, loss: {:.4f}\t '.format(sequence_idx, mean_sequence_loss))
            epoch_loss.append(mean_sequence_loss)

        training_losses.append(np.mean(epoch_loss))
        training_losses_time.append('epoch {}\t, '.format(i) + time.strftime("%H:%M:%S", time.localtime()) + ', loss: ' + str(np.mean(epoch_loss)))
        # TODO currently storing mean loss, need losses for 14 variables
        print("[End of epoch {}\t] mean loss = {:.4f}\t ".format(
            i, training_losses[-1])+ time.strftime("%H:%M:%S", time.localtime()))
        if i%10==0:
            save_path = model_predictor.saver.save(sess, "./chechpoints/LSTM_{}_epochs.ckpt".format(i))
            print("Model saved in path: %s" % save_path)
#---------
        # plt.figure(1)
        # plt.plot(rewards)
        # plt.ylabel("Reward")
        # plt.xlabel("Number of iteration")
        # plt.title("Recurrent Q Network with target network (" + name + ")")
        # plt.pause(0.001)
        # fig = plt.gcf()
        # fig.savefig('RQN_{}_reward.png'.format(name))

        # plt.figure(2)
        # plt.plot(loss)
        # plt.ylabel("Loss")
        # plt.xlabel("Number of iteration")
        # plt.title("Recurrent Q Network with target network (" + name + ")")
        # plt.pause(0.001)
        # fig = plt.gcf()
        # fig.savefig('RQN_{}_loss.png'.format(name))

        # plt.figure(3)
        # plt.plot(np.cumsum(rewards))
        # plt.ylabel("Cumulative Reward")
        # plt.xlabel("Number of iteration")
        # plt.title("Recurrent Q Network with target network (" + name + ")")
        # plt.pause(0.001)
        # fig = plt.gcf()
        # fig.savefig('RQN_{}_cum_reward.png'.format(name))
#---------
    exp_name = '50_epochs'
    if save_model:
        model_json = model_predictor.nn.to_json()
        with open('LSTM_{}.json'.format(exp_name), 'w') as json_file:
            json_file.write(model_json)
        save_path = model_predictor.saver.save(sess, "./chechpoints/LSTM_{}.ckpt".format(exp_name))
        print("Model saved in path: %s" % save_path)
        # np.savetxt('LSTM_{}.txt'.format(exp_name), (training_losses_time))
        with open('LSTM_{}.txt'.format(exp_name), 'w') as f:
            for line in training_losses_time:
                f.write("%s\n" % line)
        print("Training details saved!")

    #plt.show()

def data_loader():
    with open('trails_data.json') as json_file:  
        training_data = json.load(json_file)
    # remove two outliers from training_data [685] & [692]
    return np.array(training_data[:685]+training_data[686:692]+training_data[693:])

def new_model_predictor():
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    keras.backend.set_session(sess)

    model_predictor_in_use = Predictor("predictor", num_feats, n_state, False)

    sess.run(tf.global_variables_initializer())
    model_predictor_in_use.saver.restore(sess, "./chechpoints/model_LSTM.ckpt")

    return model_predictor_in_use

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='training a recurrent model predictor network')
    parser.add_argument('--epochs', type=int, action='store',
                        help='number of epochs to train', default=50)
    
    parser.add_argument('--save_model', type=bool, action='store', help='save trained model or not', default=True)
    # parser.add_argument('--action_length', type=int, action='store',
    #                     help='time frames for each action', default=5)
    args = parser.parse_args()

    num_feats=22
    n_state =16
    truncated_backprop_length = 5
    batch_size = 3
    input_frames = 5


    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    keras.backend.set_session(sess)

    model_predictor = Predictor("predictor", num_feats, n_state, input_frames, True)

    sess.run(tf.global_variables_initializer())

    training_sequenses = data_loader()

    train_sequense(args.epochs, args.save_model, training_sequenses)
