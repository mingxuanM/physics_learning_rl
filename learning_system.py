from interaction_env import Interaction_env
from RQN_agent import Q_agent, train_loop as rqn_loop #, train_iteration as rqn_episode
from random_agent import Random_agent, train_loop as random_loop #, train_iteration as random_episode
from model_predictor.predictor import Predictor, train_sequense
import argparse
import tensorflow as tf
import time
import sys
import json
import numpy as np
from config import n_actions, RQN_num_feats, action_length, qlearning_gamma, epsilon_decay, loss_weight


"""
This is a intergrated active training framework 

Separate active training steps:

1. loss reward training
equivalent single bash:
python RQN_agent.py --episode 2000 --active_learning True --save_model True --train True > active_learning_loss_reward_world-1.txt &&

2. loss reward data generation
equivalent single bash:
python RQN_agent.py --episode 500 --continue_from 2000 --active_learning True  > active_learning_loss_reward_data_generation_world-1.txt &&

3. Train predictor on new training sets
equivalent single bash:
python predictor.py --train True --save_model True --loss_weight 1 --epochs 50 --training_data_type 1 > active_data_train_world-1.txt
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='run a random agent')
    parser.add_argument('--training_framework', type=int, action='store',
                        help='the type of training framework to run, 0: separate framework; 1: concurrent framework', default=0)

    parser.add_argument('--random_baseline', type=bool, action='store',
                        help='if to run the experiment with a random baseline agent ', default=False)
    
    parser.add_argument('--training_loop', type=int, action='store',
                        help='number of big loops to train, one training loop might take very long, be careful', default=1)

    parser.add_argument('--timeout', type=int, action='store',
                        help='max number of frames for one episode, 1/60s per frame', default=1800)

    parser.add_argument('--world_setup', type=int, action='store',
                        help='the world setup for experiment environment, None for random, or 4 or -1', default=None)

    parser.add_argument('--agent_training_episode', type=int, action='store',
                        help='the number of episodes to train the agent', default=2000)

    parser.add_argument('--epsilon', type=float, action='store',
                        help='the epsilon for rl training policy', default=0.99)
    
    parser.add_argument('--predictor_training_epochs', type=int, action='store',
                        help='the number of epochs to train the predictor', default=50)
    
    parser.add_argument('--training_set_size', type=int, action='store',
                        help='the number of trials for new generated training set', default=500)

    parser.add_argument('--save_model', type=bool, action='store',
                        help='if to save model during training agent and predictor', default=False)

    args = parser.parse_args()


    predictor_graph = tf.Graph()
    with predictor_graph.as_default():
        predictor = Predictor(lr=1e-4, name='predictor', num_feats=22, n_state=16, input_frames=5, train=True, batch_size=None)
        # Dynamic batch size, 1 in environment, 20 in predictor training
    predictor_sess = tf.InteractiveSession(graph = predictor_graph)
    predictor_sess.run(tf.global_variables_initializer())
    predictor.saver.restore(predictor_sess, "./model_predictor/checkpoints/pretrained_model_predictor.ckpt")

    environment = Interaction_env(args.world_setup, predictor, predictor_sess)
    
    if args.random_baseline:
        random_agent = Random_agent()
        for i in range(args.training_loop):
            # Generate data with random agent
            training_data = random_loop(random_agent, environment, args.episode, args.timeout)
            print('Random data generation completed for loop {}'.format(i))
            # Train predictor
            exp_name = 'random_training_loop_{}'.format(i)
            train_sequense(predictor, predictor_sess, exp_name, args.predictor_training_epochs, args.save_model, training_data)
            print('Predictor training with random agent completed for loop {}'.format(i))

            print('Random training loop {} completed'.format(i))
    else:
        rqn_agent_graph = tf.Graph()
        with rqn_agent_graph.as_default():
            learning_agent = Q_agent("learning_agent", n_actions, qlearning_gamma, epsilon = args.epsilon)
            target_agent = Q_agent("target_agent", n_actions, qlearning_gamma, epsilon = args.epsilon)
        agent_sess = tf.InteractiveSession(graph = rqn_agent_graph)
        agent_sess.run(tf.global_variables_initializer())
        learning_agent.saver.restore(agent_sess, "./checkpoints/trained_RQN_catching.ckpt")
        target_agent.saver.restore(agent_sess, "./checkpoints/trained_RQN_catching_target.ckpt")

        for i in range(args.training_loop):
            # Train agent
            # with agent_sess.as_default():
            rqn_loop(learning_agent, target_agent, environment, args.agent_training_episode, True, args.timeout, continue_from=0, save_model=args.save_model)
            print('Active agent training completed for loop {}'.format(i))
            # Generate data with trained agent
            # with agent_sess.as_default():
            training_data = rqn_loop(learning_agent, target_agent, environment, args.training_set_size, False, args.timeout)
            print('Active data generation completed for loop {}'.format(i))
            training_data = np.array(training_data)
            np.random.shuffle(training_data)
            # Train predictor
            exp_name = 'active_training_loop_{}'.format(i)

            with open('./model_predictor/data/world_setup_-1_test_set.json') as json_file: 
                test_data = json.load(json_file)
                test_data = np.array(test_data)
            np.random.shuffle(test_data)
            

            train_sequense(predictor, predictor_sess, exp_name, args.predictor_training_epochs, args.save_model, training_data, test_data, 20, loss_weight)
            print('Predictor training with active agent completed for loop {}'.format(i))

            print('Active training loop {} completed'.format(i))








    
