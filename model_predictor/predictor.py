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


# num_feats = 4 (one hot) + 2 +2*4 = 14
# n_state = 4*4 = 16 (4 objects * (2 locations + 2 velocities))
# truncated_backprop_length: 10 time steps in each training step(probably one action step)
# Every 5 frames of inputs are used to predict the next frame
num_feats=14
n_state =16
truncated_backprop_length = 5
input_frames = 5
batch_size = 3

class Predictor:
    def __init__(self, name='predictor', num_feats=14, n_state=16, input_frames=5, train=True):
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
            self.nn.add(tf.keras.layers.CuDNNLSTM(22))
            # self.nn.add(tf.keras.layers.LSTM(n_state, recurrent_activation='sigmoid'))
            self.nn.add(tf.keras.layers.Dense(n_state))

            # Predicting 1 frame: 5 frames input
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

            # Mean loss over batch and truncated_backprop_length for each 16 elements [16]
            self.batch_losses = tf.reduce_mean(tf.squared_difference(self.batch_predictions, self.batch_labels), axis=[0,1])
            
            self.loss_weight = tf.placeholder(tf.float32, [n_state])
            self.weitghted_batch_losses = tf.math.multiply(self.batch_losses, self.loss_weight)

        self.weights = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        self.train_step = tf.train.AdamOptimizer(
                1e-3).minimize(self.weitghted_batch_losses, var_list=self.weights)

        self.saver = tf.train.Saver()


# run one batch, truncated_backprop_length frames
# batch_sequence: [batch_size, truncated_backprop_length + input_frames, total_feats (22 in total)]
def train_iteration(batch_sequence):   
    labels = np.zeros((truncated_backprop_length, batch_size, n_state))
    inputs = np.zeros((truncated_backprop_length, batch_size, input_frames, num_feats))
    for t in range(truncated_backprop_length):
        # Drop xv & yv of 4 objects in inputs
        inputs[t] = np.delete(batch_sequence[:,t:(t+input_frames),:],[8,9,12,13,16,17,20,21],2)
        labels[t] = batch_sequence[:,t+input_frames,-16:]

    labels_feed = np.swapaxes(labels,0,1)
    _train_step, _weitghted_batch_losses, _batch_losses = sess.run(
        [model_predictor.train_step, model_predictor.weitghted_batch_losses, model_predictor.batch_losses], 
        {model_predictor.batch_labels: labels_feed, model_predictor.training_states: inputs, model_predictor.loss_weight:loss_weight}
    )
    return _batch_losses

# Top level training over epochs
# training_sequense in [batch_size*-1*total_feats]
def train_sequense(epochs, save_model, training_sequenses, exp_name):
    # mean losses during each epoch
    # training_losses = []
    training_losses_time = []
    for i in range(epochs):
        epoch_loss = []
        for sequence_idx, _training_sequense in enumerate(training_sequenses):
            sequence_loss = []
            # cut out excess frames so that _training_sequense can be reshaped by batch_size
            excessed = int(_training_sequense.shape[0] % batch_size)
            _training_sequense = _training_sequense[:-excessed,:]
            training_sequense = np.reshape(_training_sequense, (batch_size, -1, total_feats))
            total_batches = (training_sequense.shape[1]-input_frames)//truncated_backprop_length
            # run batch loop
            for batch_idx in range(total_batches):
                start = batch_idx * truncated_backprop_length
                end = start + truncated_backprop_length + input_frames
                batch_loss_ = train_iteration(training_sequense[:,start:end,:])
                # Calculate mean loss of 4 elements over 4 objects
                batch_loss = np.array([
                    np.mean([batch_loss_[0],batch_loss_[4],batch_loss_[8],batch_loss_[12]]), # x
                    np.mean([batch_loss_[1],batch_loss_[5],batch_loss_[9],batch_loss_[13]]), # y
                    np.mean([batch_loss_[2],batch_loss_[6],batch_loss_[10],batch_loss_[14]]), # vx
                    np.mean([batch_loss_[3],batch_loss_[7],batch_loss_[11],batch_loss_[15]])  # vy
                ])
                sequence_loss.append(batch_loss)
                # if sequence_idx >= 100:
                #     print('batch loss of {}\t: '.format(batch_idx))
                #     print(batch_loss)
            # Mean loss over a sequence for each 16 elements
            mean_sequence_loss = np.mean(sequence_loss, axis=0)
            print('Training sequence {}\t finished, loss: {}\t '.format(sequence_idx, str(mean_sequence_loss)))
            epoch_loss.append(mean_sequence_loss)
        # Mean loss over sequences in one epoch for each 16 elements
        mean_epoch_loss = np.mean(epoch_loss, axis=0)
        # training_losses.append(mean_epoch_loss)
        training_losses_time.append('epoch {}\t, '.format(i) + time.strftime("%H:%M:%S", time.localtime()) + ', loss: ' + str(mean_epoch_loss))
        # TODO currently storing mean loss, need losses for 16 variables
        print("[End of epoch {}\t] ".format(
            i) + time.strftime("%H:%M:%S", time.localtime()) + ', Mean squared loss for 16 elements:')
        print(mean_epoch_loss)
        if i%10==0 and i>0:
            save_path = model_predictor.saver.save(sess, "./chechpoints/{}_{}_epochs.ckpt".format(exp_name,i))
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
    if save_model:
        model_json = model_predictor.nn.to_json()
        with open('{}_50_epochs.json'.format(exp_name), 'w') as json_file:
            json_file.write(model_json)
        save_path = model_predictor.saver.save(sess, "./chechpoints/{}_50_epochs.ckpt".format(exp_name))
        print("Model saved in path: %s" % save_path)
        # np.savetxt('LSTM_{}.txt'.format(exp_name), (training_losses_time))
        with open('{}_50_epochs.txt'.format(exp_name), 'w') as f:
            for line in training_losses_time:
                f.write("%s\n" % line)
        print("Training details saved!")

    #plt.show()

# Predict next 1 frame given 5 frames from test set
def passive_test(test_sequenses):
    for i in range(1,6):
        model_predictor_trained.saver.restore(sess, "./chechpoints/LSTM_{}0_epochs.ckpt".format(i))
        print('Model trained with {}0 epochs successfully loaded'.format(i))
        epoch_loss = np.zeros((len(test_sequenses),4))
        for s_idx, sequence in enumerate(test_sequenses):
            sequence = np.reshape(sequence, (1, -1, total_feats))
            num = (sequence.shape[1]-1) // input_frames
            sequence_loss = np.zeros((num,4))
            for n in range(num):
                begin = n*input_frames
                inputs = sequence[:,begin:begin+input_frames,:]
                label = np.reshape(sequence[:,begin+input_frames,-16:],(1,1,n_state))
                prediction = np.array(sess.run(
                    [model_predictor_trained.prediction], 
                    {model_predictor_trained.state_t: inputs}
                ))
                # sequence_loss[n] = np.mean(np.square(np.subtract(label, prediction)))
                # Add squared loss of one test batch (16 elements)
                batch_loss_ = np.reshape(np.square(np.subtract(label, prediction)), n_state)
                sequence_loss[n] = np.array([
                    np.mean([batch_loss_[0],batch_loss_[4],batch_loss_[8],batch_loss_[12]]), # x
                    np.mean([batch_loss_[1],batch_loss_[5],batch_loss_[9],batch_loss_[13]]), # y
                    np.mean([batch_loss_[2],batch_loss_[6],batch_loss_[10],batch_loss_[14]]), # vx
                    np.mean([batch_loss_[3],batch_loss_[7],batch_loss_[11],batch_loss_[15]])  # vy
                ])
            epoch_loss[s_idx] = np.mean(sequence_loss,axis=0)
        epoch_loss_mean = np.mean(epoch_loss,axis=0)
        print("[End of testing model trained with {}0 epochs] ".format(
            i)+ time.strftime("%H:%M:%S", time.localtime()) + ', Mean squared loss for [x,y,xv,yv]:')
        print(str(epoch_loss_mean)+'/n')

# Use predicted frames to predict more frames
# Use 0 - 5 predicted frames, passive_test() is included here
def long_term_passive_test(exp_name, test_sequenses):
    for i in range(1,6):
        model_predictor_trained.saver.restore(sess, "./chechpoints/{}_{}0_epochs.ckpt".format(exp_name, i))
        print('Model trained with {}0 epochs successfully loaded'.format(i))
        epoch_loss = np.zeros((len(test_sequenses),6,4))
        for s_idx, sequence in enumerate(test_sequenses):
            sequence = np.reshape(sequence, (1, -1, total_feats))
            num = (sequence.shape[1] - 6) // input_frames
            sequence_loss = np.zeros((num,6,4))
            for n in range(num):
                predicted = np.zeros((1,6,total_feats))
                end = n*input_frames + input_frames
                for predicted_frames in range(6):
                    begin = n*input_frames + predicted_frames
                    # np.delete(batch_sequence[],[8,9,12,13,16,17,20,21],2)
                    inputs = np.concatenate((
                        np.delete(sequence[:,begin:end,:],[8,9,12,13,16,17,20,21],2),
                        np.delete(predicted[:,:predicted_frames,:],[8,9,12,13,16,17,20,21],2)
                    ), axis=1)
                    label = np.reshape(sequence[:,end+predicted_frames,-16:],(1,1,n_state))
                    # shape of prediction: [1,1,n_state]
                    prediction = np.array(sess.run(
                        [model_predictor_trained.prediction], 
                        {model_predictor_trained.state_t: inputs}
                    ))
                    predicted[0,predicted_frames] = np.concatenate((np.zeros(6),prediction[0,0]), axis=0)
                    batch_loss_ = np.reshape(np.square(np.subtract(label, prediction)), n_state)
                    sequence_loss[n,predicted_frames] = np.array([
                        np.mean([batch_loss_[0],batch_loss_[4],batch_loss_[8],batch_loss_[12]]), # x
                        np.mean([batch_loss_[1],batch_loss_[5],batch_loss_[9],batch_loss_[13]]), # y
                        np.mean([batch_loss_[2],batch_loss_[6],batch_loss_[10],batch_loss_[14]]), # vx
                        np.mean([batch_loss_[3],batch_loss_[7],batch_loss_[11],batch_loss_[15]])  # vy
                    ])
            epoch_loss[s_idx] = np.mean(sequence_loss, axis=0)
        epoch_loss_mean = np.mean(epoch_loss, axis=0)
        print('''[End of testing model trained with {}0 epochs] mean loss:\n
            0 predicted frames = {}\n 
            1 predicted frames = {}\n 
            2 predicted frames = {}\n 
            3 predicted frames = {}\n
            4 predicted frames = {}\n 
            5 predicted frames = {}\n   
            '''.format(
            i, str(epoch_loss_mean[0]),str(epoch_loss_mean[1]),str(epoch_loss_mean[2]),str(epoch_loss_mean[3]),str(epoch_loss_mean[4]),str(epoch_loss_mean[5])
            ) + time.strftime("%H:%M:%S", time.localtime()))

# Use first 5 frames from first 10 sequences in test set to generate 60 frames of trajectories
def generate_trajectories(test_sequenses, trajectory_len):
    for i in range(1,6):
        model_predictor_trained.saver.restore(sess, "./chechpoints/LSTM_{}0_epochs.ckpt".format(i))
        print('Model trained with {}0 epochs successfully loaded'.format(i))
        for s_idx, sequence in enumerate(test_sequenses):
            trajectory = np.zeros((trajectory_len+5,n_state))
            sequence_ = sequence.copy()
            sequence_ = np.reshape(sequence_, (1, 5, num_feats))
            for t in range(5):
                trajectory[t] = sequence_[0,t,-16:]
            for n in range(trajectory_len):
                predict_idx = n % 5
                # Concatenate input array from sequence array splited at predict_idx
                inputs = np.concatenate((sequence_[:,predict_idx:,:],sequence_[:,:predict_idx,:]), axis=1)
                prediction = np.array(sess.run([model_predictor_trained.prediction], {model_predictor_trained.state_t: inputs}))
                sequence_[0,predict_idx] = np.concatenate((np.zeros(6),prediction[0,0]), axis=0)
                trajectory[n+5] = prediction[0,0]
            with open('generated_trajectories/{}0epochs_sequence{}.json'.format(i,s_idx), 'w') as outfile:
                json.dump(trajectory.tolist(), outfile, ensure_ascii=False, indent=2)


# Load data from json files, 
# convert vx,vy into velocity magnitude & angle
# [x,y,xv,yv] -> [x,y,r,theta]
# data/trails_data.json and data/test_data.json use [x,y,xv,yv]
# data/trails_data_transformed.json and data/test_data_transformed.json use [x,y,r,theta]
def Transform_data_loader(train):
    if train:
        with open('data/trails_data.json') as json_file:  
            data = json.load(json_file)
            # remove two outliers from training_data [685] & [692]
            data = data[:685]+data[686:692]+data[693:]
    else:
        with open('data/test_data.json') as json_file:  
            data = json.load(json_file)
    
    data = np.array(data)
    i = 0
    for s in data:
        j = 0
        for f in s:
            if i == 100 and j == 100:
                print(data[100,100])
            for obj in range(4):
                vx, vy = f[6+2+obj*4], f[6+3+obj*4]
                f[6+2+obj*4] = np.sqrt(vx**2 + vy**2)
                if vx == 0:
                    if vy != 0:
                        f[6+3+obj*4] = np.pi/2 if vy > 0 else np.pi*3/2
                    else:
                        f[6+3+obj*4] = 0
                else:
                    f[6+3+obj*4] = np.arctan(vy/vx)
            if i == 100 and j == 100:
                print(data[100,100])
            j += 1
        i += 1
    if train:
        with open('data/trails_data_transformed.json', 'w') as outfile:
            json.dump(data.tolist(), outfile, ensure_ascii=False, indent=2)
        print('Transformed training data saved')
    else:
        with open('data/test_data_transformed.json', 'w') as outfile:
            json.dump(data.tolist(), outfile, ensure_ascii=False, indent=2)
        print('Transformed test data saved')
   

    return data


# Load transformed data sets
def data_loader(train):
    if train:
        with open('data/trails_data.json') as json_file:  
            data = json.load(json_file)
            # remove two outliers from training_data [685] & [692]
            data = data[:685]+data[686:692]+data[693:]
    else:
        with open('data/test_data.json') as json_file:  
            data = json.load(json_file)
    
    return np.array(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='training a recurrent model predictor network')
    parser.add_argument('--epochs', type=int, action='store',
                        help='number of epochs to train', default=50)
    
    parser.add_argument('--save_model', type=bool, action='store', help='save trained model or not', default=True)
    parser.add_argument('--train', type=bool, action='store',
                        help='if to train a model', default=False)
    args = parser.parse_args()

    total_feats=22
    num_feats=14 # [4 one-hot object, mx, my, o1.x, o1.y, o2.x, o2.y, o3.x, o3.y, o4.x, o4.y]
    n_state =16
    input_frames = 5

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    keras.backend.set_session(sess)

    # Loss weight, used to calculate weighted average of 16 elements in training loss.
    # loss_weight = np.ones(n_state)
    # loss_weight = np.array([6.3432/1.6973, 6.3432/1.0517, 6.3432/1.7830, 6.3432/1.8112, 
    #                         6.3432/1.6973, 6.3432/1.0517, 6.3432/1.7830, 6.3432/1.8112, 
    #                         6.3432/1.6973, 6.3432/1.0517, 6.3432/1.7830, 6.3432/1.8112,
    #                         6.3432/1.6973, 6.3432/1.0517, 6.3432/1.7830, 6.3432/1.8112])

    loss_weight = np.array([1.6973/6.3432, 1.0517/6.3432, 1.7830/6.3432, 1.8112/6.3432, 
                            1.6973/6.3432, 1.0517/6.3432, 1.7830/6.3432, 1.8112/6.3432, 
                            1.6973/6.3432, 1.0517/6.3432, 1.7830/6.3432, 1.8112/6.3432,
                            1.6973/6.3432, 1.0517/6.3432, 1.7830/6.3432, 1.8112/6.3432])

    if args.train:
        truncated_backprop_length = 5
        batch_size = 3
        exp_name = 'LSTM_reduced_input_high_learning_rate_weighted_average'

        model_predictor = Predictor("predictor", num_feats, n_state, input_frames, True)

        sess.run(tf.global_variables_initializer())

        training_sequenses = data_loader(args.train)

        train_sequense(args.epochs, args.save_model, training_sequenses, exp_name)
    else:
        print('Begin predictor model testing...')
        # tf.reset_default_graph()
        # sess = tf.InteractiveSession()
        # keras.backend.set_session(sess)
        exp_name = 'LSTM_reduced_input_high_learning_rate_weighted_average'
        model_predictor_trained = Predictor("predictor", num_feats, n_state, input_frames, False)

        sess.run(tf.global_variables_initializer())

        test_sequenses = data_loader(args.train)
    # Long_term_passive_test will test predictions based on 0 - 5 predicted frames
        # passive_test(test_sequenses)
        long_term_passive_test(exp_name, test_sequenses)
    # Generate 60 frames long trajectories given first 5 frames in first 10 test cases
        # generate_trajectories(test_sequenses[:10,:5,:],60)
