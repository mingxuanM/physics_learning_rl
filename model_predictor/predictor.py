# -*- coding: utf-8 -*-
"""
predictor.py
"""
import sys
sys.path.append('../simulator/')
import argparse
import json
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
            # tf.keras.layers.CuDNNLSTM is GPU optimised, switch to LSTM if using CPU
         #    if train:
	        #     self.nn.add(tf.keras.layers.CuDNNLSTM(n_state))
	        # else:
	        #     self.nn.add(tf.keras.layers.LSTM(n_state, recurrent_activation='sigmoid'))
            self.nn.add(tf.keras.layers.CuDNNLSTM(n_state))
            # self.nn.add(tf.keras.layers.LSTM(n_state, recurrent_activation='sigmoid'))
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

def data_loader(train):
    if train:
        with open('data/trails_data.json') as json_file:  
            training_data = json.load(json_file)
        # remove two outliers from training_data [685] & [692]
        return np.array(training_data[:685]+training_data[686:692]+training_data[693:])
    else:
        with open('data/test_data.json') as json_file:  
            test_data = json.load(json_file)
        return np.array(test_data)

# Predict next 1 frame given 5 frames from test set
def passive_test(test_sequenses):
    for i in range(1,6):
        model_predictor_trained.saver.restore(sess, "./chechpoints/LSTM_{}0_epochs.ckpt".format(i))
        print('Model trained with {}0 epochs successfully loaded'.format(i))
        epoch_loss = np.zeros(len(test_sequenses))
        for s_idx, sequence in enumerate(test_sequenses):
            sequence = np.reshape(sequence, (1, -1, num_feats))
            num = (sequence.shape[1]-1) // input_frames
            sequence_loss = np.zeros(num)
            for n in range(num):
                begin = n*input_frames
                inputs = sequence[:,begin:begin+input_frames,:]
                label = np.reshape(sequence[:,begin+input_frames,-16:],(1,1,n_state))
                prediction = np.array(sess.run([model_predictor_trained.prediction], {model_predictor_trained.state_t: inputs}))
                sequence_loss[n] = np.mean(np.square(np.subtract(label, prediction)))
            epoch_loss[s_idx] = np.mean(sequence_loss)
        epoch_loss_mean = np.mean(epoch_loss)
        print("[End of testing model trained with {}0 epochs] mean loss = {:.4f}\t ".format(
            i, epoch_loss_mean)+ time.strftime("%H:%M:%S", time.localtime()))

# Use predicted frames to predict more frames
# Use 0 - 5 predicted frames
def long_term_passive_test(test_sequenses):
    for i in range(1,6):
        model_predictor_trained.saver.restore(sess, "./chechpoints/LSTM_{}0_epochs.ckpt".format(i))
        print('Model trained with {}0 epochs successfully loaded'.format(i))
        epoch_loss = np.zeros((len(test_sequenses),6))
        for s_idx, sequence in enumerate(test_sequenses):
            sequence = np.reshape(sequence, (1, -1, num_feats))
            num = (sequence.shape[1] - 6) // input_frames
            sequence_loss = np.zeros((num,6))
            for n in range(num):
                predicted = np.zeros((1,6,num_feats))
                end = n*input_frames + input_frames
                for predicted_frames in range(6):
                    begin = n*input_frames + predicted_frames
                    inputs = np.concatenate((sequence[:,begin:end,:],predicted[:,:predicted_frames,:]), axis=1)
                    label = np.reshape(sequence[:,end+predicted_frames,-16:],(1,1,n_state))
                    # shape of prediction: [1,1,n_state]
                    prediction = np.array(sess.run([model_predictor_trained.prediction], {model_predictor_trained.state_t: inputs}))
                    predicted[0,predicted_frames] = np.concatenate((np.zeros(6),prediction[0,0]), axis=0)
                    sequence_loss[n,predicted_frames] = np.mean(np.square(np.subtract(label, prediction)))
            epoch_loss[s_idx] = np.mean(sequence_loss, axis=0)
        epoch_loss_mean = np.mean(epoch_loss, axis=0)
        print('''[End of testing model trained with {}0 epochs] mean loss:\n
            0 predicted frames = {:.4f}\n 
            1 predicted frames = {:.4f}\n 
            2 predicted frames = {:.4f}\n 
            3 predicted frames = {:.4f}\n
            4 predicted frames = {:.4f}\n 
            5 predicted frames = {:.4f}\n   
            '''.format(
            i, epoch_loss_mean[0],epoch_loss_mean[1],epoch_loss_mean[2],epoch_loss_mean[3],epoch_loss_mean[4],epoch_loss_mean[5]
            ) + time.strftime("%H:%M:%S", time.localtime()))

# Use first 5 frames from first 10 sequences in test set to generate 60 frames of trajectories
def generate_trajectories(test_sequenses, trajectory_len):
    for i in range(1,6):
        model_predictor_trained.saver.restore(sess, "./chechpoints/LSTM_{}0_epochs.ckpt".format(i))
        print('Model trained with {}0 epochs successfully loaded'.format(i))
        for s_idx, sequence in enumerate(test_sequenses[:10]):
            trajectory = np.zeros((trajectory_len+5,n_state))
            sequence = np.reshape(sequence, (1, -1, num_feats))
            sequence = sequence[:,:5,:]
            trajectory[:5] = sequence[0,:,-16:]
            for n in range(trajectory_len):
                predict_idx = n % 5
                # Concatenate input array from sequence array splited at predict_idx
                inputs = np.concatenate((sequence[:,predict_idx:,:],sequence[:,:predict_idx,:]), axis=1)
                prediction = np.array(sess.run([model_predictor_trained.prediction], {model_predictor_trained.state_t: inputs}))
                sequence[0,predict_idx] = np.concatenate((np.zeros(6),prediction[0,0]), axis=0)
                trajectory[n+5] = prediction[0,0]
            with open('generated_trajectories/{}0epochs_sequence{}.json'.format(i,s_idx), 'w') as outfile:
                json.dump(trajectory.tolist(), outfile, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='training a recurrent model predictor network')
    parser.add_argument('--epochs', type=int, action='store',
                        help='number of epochs to train', default=50)
    
    parser.add_argument('--save_model', type=bool, action='store', help='save trained model or not', default=True)
    parser.add_argument('--train', type=bool, action='store',
                        help='if to train a model', default=False)
    args = parser.parse_args()

    num_feats=22
    n_state =16
    input_frames = 5

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    keras.backend.set_session(sess)

    if args.train:
        truncated_backprop_length = 5
        batch_size = 3

        model_predictor = Predictor("predictor", num_feats, n_state, input_frames, True)

        sess.run(tf.global_variables_initializer())

        training_sequenses = data_loader(args.train)

        train_sequense(args.epochs, args.save_model, training_sequenses)
    else:
        print('Begin predictor model testing...')
        tf.reset_default_graph()
        sess = tf.InteractiveSession()
        keras.backend.set_session(sess)

        model_predictor_trained = Predictor("predictor", num_feats, n_state, input_frames, False)

        sess.run(tf.global_variables_initializer())
        test_sequenses = data_loader(args.train)
    # Long_term_passive_test will test predictions based on 0 - 5 predicted frames
        # passive_test(test_sequenses)
        # long_term_passive_test(test_sequenses)
    # Generate 60 frames long trajectories given first 5 frames
        generate_trajectories(test_sequenses[:10],60)
