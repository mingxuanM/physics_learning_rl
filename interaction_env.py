import pyduktape
import numpy as np
import json
import random as rd
import math
from model_predictor.predictor import Predictor
import tensorflow as tf
# import keras

from config import n_actions, acceleration, velocity_decay, action_length, env_width, env_height, num_feats, predictor_input_frames, dragging_bonus 

# n_actions = 6 # 1 no action + 4 directions acc + 1 click
# # directions_tan = [(np.cos(i*np.pi/8.),np.sin(i*np.pi/8.)) for i in range(16)]
# acceleration = 1 # in meter/s/frame (speed meter/s change in each frame)
#   Could be smaller: 0.1 meter/s/frame
# velocity_decay = 0.9 # velocity in meter/s decay rate per frame if not accelerate
# action_length = 5 # frames
# env_width = 6
# env_height = 4
# num_feats = 22
loss_weight = np.array([1.6973/6.3432, 1.0517/6.3432, 1.7830/6.3432, 1.8112/6.3432, 
                                1.6973/6.3432, 1.0517/6.3432, 1.7830/6.3432, 1.8112/6.3432, 
                                1.6973/6.3432, 1.0517/6.3432, 1.7830/6.3432, 1.8112/6.3432,
                                1.6973/6.3432, 1.0517/6.3432, 1.7830/6.3432, 1.8112/6.3432])

class Interaction_env:
# Class instance varialbes:
# 1. self.timesteps: current time steps, increment after each action, 
#   used to retrieve trajectories data from JS context
# 2. self.state: latest state (action_length frames) {'last_trajectory':[action_length,22], 'caught_any', 'vx', 'vy'} 
#   (velocity is the final velocity from last action)
# 3. self.context: js environment
# 4. self.cond: a dictionary stores {'starting locations', 'starting velocities', 'local forces', 'mass'}
    def __init__(self, world_setup_idx=None, predictor=None, predictor_sess=None):
        self.timesteps = 0
        self.world_setup_idx = world_setup_idx
        # Read in world_setup
        with open('./js_simulator/json/world_setup.json') as data_file:    
            self.world_setup = json.load(data_file)
        # if world_setup is None:
        #     self.world_setup = rd.sample(self.world_setup, 1)[0]
        # else:
        # # self.world_setup = self.world_setup[4] # First experiment setup: [0,3,0,3,-3,0,"B"]
        #     self.world_setup = self.world_setup[world_setup] # Second experiment setup: [0,0,0,0,0,0,"same"]

        # Read in starting_state
        with open('./js_simulator/json/starting_state.json') as data_file:    
            self.starting_state = json.load(data_file)

        # Create js environment
        self.context = pyduktape.DuktapeContext()

        # Load js box2d graphic library
        js_file = open("./js_simulator/js/box2d.js",'r')
        js = js_file.read()
        self.context.eval_js(js)

        # Load world environment script
        js_file = open("./js_simulator/js/control_world.js",'r')
        js = js_file.read()
        self.context.eval_js(js)
        
        self.cond = {}

        if predictor is None:
            predictor_graph = tf.Graph()
            with predictor_graph.as_default():
                self.predictor = Predictor(lr=1e-4, name='predictor', num_feats=22, n_state=16, input_frames=5, train=True, batch_size=1)
            self.sess = tf.InteractiveSession(graph = predictor_graph)
            self.sess.run(tf.global_variables_initializer())
            self.predictor.saver.restore(self.sess, "./model_predictor/checkpoints/pretrained_model_predictor.ckpt") # 1e-5 learning rate, 20 epochs   
        else:
            self.predictor = predictor
            self.sess = predictor_sess
        
        self.trajectory_history = []

    def reset(self):
        if self.world_setup_idx is None:
            world_setup = rd.sample(self.world_setup, 1)[0]
        else:
            world_setup = self.world_setup[self.world_setup_idx]

        starting_state = rd.sample(self.starting_state, 1)[0]
        self.trajectory_history = []
        # Set starting conditions
        self.cond = {'sls':[{'x':starting_state[0], 'y':starting_state[1]}, {'x':starting_state[4], 'y':starting_state[5]},
                        {'x':starting_state[8], 'y':starting_state[9]}, {'x':starting_state[12], 'y':starting_state[13]}],
                'svs':[{'x':rd.uniform(-10.0, 10.0), 'y':rd.uniform(-10.0, 10.0)}, {'x':rd.uniform(-10.0, 10.0), 'y':rd.uniform(-10.0, 10.0)},
                        {'x':rd.uniform(-10.0, 10.0), 'y':rd.uniform(-10.0, 10.0)}, {'x':rd.uniform(-10.0, 10.0), 'y':rd.uniform(-10.0, 10.0)}]
                }
        # self.cond = {'sls':[{'x':rd.uniform(0.0, 6.0), 'y':rd.uniform(0.0, 4.0)}, {'x':rd.uniform(0.0, 6.0), 'y':rd.uniform(0.0, 4.0)},
        #                 {'x':rd.uniform(0.0, 6.0), 'y':rd.uniform(0.0, 4.0)}, {'x':rd.uniform(0.0, 6.0), 'y':rd.uniform(0.0, 4.0)}],
        #         'svs':[{'x':rd.uniform(-10.0, 10.0), 'y':rd.uniform(-10.0, 10.0)}, {'x':rd.uniform(-10.0, 10.0), 'y':rd.uniform(-10.0, 10.0)},
        #                 {'x':rd.uniform(-10.0, 10.0), 'y':rd.uniform(-10.0, 10.0)}, {'x':rd.uniform(-10.0, 10.0), 'y':rd.uniform(-10.0, 10.0)}]
        #         }

        # Zero condition
        # self.cond = {'sls':[{'x':3, 'y':2}, {'x':1, 'y':1},
        #                 {'x':5, 'y':3}, {'x':1, 'y':3}],
        #         'svs':[{'x':0, 'y':0}, {'x':0, 'y':0},
        #                 {'x':0, 'y':0}, {'x':0, 'y':0}]
        #         }

        # Set local forces
        self.cond['lf'] = [[0.0,float(world_setup[0]), float(world_setup[1]), float(world_setup[2])],
                [0.0, 0.0, float(world_setup[3]), float(world_setup[4])],
                [0.0, 0.0, 0.0, float(world_setup[5])],
                [0.0, 0.0, 0.0, 0.0]]

        # Zero condition
        # self.cond['lf'] = [[0.0, 0.0, 0.0, 0.0],
        #         [0.0, 0.0, 0.0, 0.0],
        #         [0.0, 0.0, 0.0, 0.0],
        #         [0.0, 0.0, 0.0, 0.0]]

        # Set mass
        if world_setup[6]=='A':
            self.cond['mass'] = [2,1,1,1]
        elif world_setup[6]=='B':
            self.cond['mass'] = [1,2,1,1]
        else:
            self.cond['mass'] = [1,1,1,1]

        # Set timeout (action_length) in js simulator
        self.cond['timeout']= action_length

        self.context.set_globals(cond=self.cond)
        
        # initial control path for the first action_length frames
        path = {'x':[3]*action_length, 'y':[2]*action_length, 'obj':[0]*action_length, 'click':False}
        # Feed control path to js context      
        self.context.set_globals(control_path=path)
        # Build js environment, get trajectory for the first action_length frames
        trajectory = self.context.eval_js("Run();") # [action_length,22]
        trajectory = json.loads(trajectory)
        self.trajectory_history.extend(trajectory)
        trajectory = np.array(trajectory) # Convert to python object
        # self.state['last_trajectory'] = trajectory
        # self.state['caught_object'] = 0
        self.state = {'last_trajectory':trajectory,
            'caught_any':False,
            'vx':0,
            'vy':0}
        
        
        return trajectory

    def act(self, actionID):
        # actionID taking value from 0 to 5
        # [0:none, 1:up, 2:down, 3:right, 4:left, 5:click]
        if_click = False
        direction = 0 # taking value [0,1,2,3,4]: [none, up, down, right ,left]
        if actionID < 5:
            direction = actionID 
        elif actionID == 5:
            if_click = True  
        # Generate control path from action space given actionID
        control_path_ = self.action_generation(if_click, direction)
        
        # Feed control path to js context      
        self.context.set_globals(control_path=control_path_)

        # Run the simulation
        trajectory = self.context.eval_js("action_forward();") # [action_length,22]
        trajectory = json.loads(trajectory)
        self.trajectory_history.extend(trajectory)
        trajectory = np.array(trajectory) # Convert to python object
        reward, is_done, pretrained_loss = self.reward_cal(trajectory)
        # Update self.state
        self.state['last_trajectory'] = trajectory
        
        return trajectory, reward, is_done, pretrained_loss

    # Calculate reward given trajectory [action_length:22] of one action length
    def reward_cal(self, trajectory):
        reward = 0
        is_done = False
        # train predictor network, add reward for loss drop
        # pretrained_loss, trained_loss = self.predictor_test(trajectory)
        # reward += np.abs(pretrained_loss - trained_loss)

        # calculate predictor loss, add to reward
        predictor_loss= self.predictor_test(trajectory)
        # predictor_loss = 0
        reward += predictor_loss

        control_object = np.sum(trajectory[0,:4])
        if control_object > 0:
            for i in range(4):
                if trajectory[0,i] == 1:
                    control_object = i+1
                    break

        if (not self.state['caught_any']) and control_object == 0:
            # agent has not caught any yet
            # calculate the difference of distance from mouse to the closest object in the last frame of last action
            # give reward by distance decrease
            f0 = self.state['last_trajectory'][-1]
            min_dist = 100
            min_dist_x = 0
            min_dist_y = 0
            close_obj = 0
            mouse_x1 = f0[4]
            mouse_y1 = f0[5]
            # find closest obj in f0
            for obj in range(4):
                obj_x1 = f0[4*obj+6]
                obj_y1 = f0[4*obj+7]
                dist = ((obj_x1-mouse_x1)**2+(obj_y1-mouse_y1)**2)**0.5
                if dist<min_dist:
                    min_dist = dist
                    min_dist_x = abs(obj_x1-mouse_x1)
                    min_dist_y = abs(obj_y1-mouse_y1)
                    close_obj = obj

            f5 = trajectory[-1]
            # dist5 = ((f5[4*close_obj+6]-f5[4])**2+(f5[4*close_obj+7]-f5[5])**2)**0.5
            dist5_x = abs(f5[4*close_obj+6]-f5[4])
            dist5_y = abs(f5[4*close_obj+7]-f5[5])
            # max possible reward is 2
            if min_dist_x > 0 and min_dist_y > 0:
                reward += min((max((min_dist_x - dist5_x)/min_dist_x, 0) + max((min_dist_y - dist5_y)/min_dist_y, 0)),2)
            elif min_dist_x == 0 and min_dist_y > 0:
                reward += min((1 + max((min_dist_y - dist5_y)/min_dist_y, 0)),2)
            elif min_dist_x > 0 and min_dist_y == 0:
                reward += min((max((min_dist_x - dist5_x)/min_dist_x, 0) + 1),2)
            elif min_dist_x == 0 and min_dist_y == 0:
                reward += 2

        elif (not self.state['caught_any']) and control_object != 0:
            # object caught
            reward += 5
            reward += dragging_bonus * predictor_loss
            self.state['caught_any'] = True
            # is_done = True
        elif self.state['caught_any'] and control_object == 0:
            # released last caught object
            self.state['caught_any'] = False
        elif self.state['caught_any'] and control_object != 0:
            # add bonus reward for decrease predictor loss when agent is dragging
            reward += dragging_bonus * predictor_loss

        return reward, is_done, predictor_loss

    def predictor_test(self, trajectory):
        batch_num = action_length
        # sequence = trajectory.reshape((1,action_length,num_feats))
        # mean_weitghted_batch_losses = np.zeros(batch_num)
        # for b in range(batch_num):
        #     if b <= predictor_input_frames:
        #         inputs = np.concatenate((self.state['last_trajectory'][-predictor_input_frames+b:,:], trajectory[:b,:]), axis=0)
        #     else:
        #         inputs = trajectory[b-predictor_input_frames:b]
        #     # labels = sequence[:,b+self.predictor.input_frames,-16:]
        #     labels = trajectory[b,-16:].reshape((1,16))
        #     # inputs = sequence[:,b:b+self.predictor.input_frames,:]
        #     inputs = inputs.reshape((1,predictor_input_frames,num_feats))
        #     _train_step, _weitghted_batch_losses, _batch_losses = self.sess.run(
        #         [self.predictor.train_step, self.predictor.weitghted_batch_losses, self.predictor.batch_losses], 
        #         {self.predictor.batch_labels: labels, self.predictor.state_t: inputs, self.predictor.loss_weight:loss_weight}
        #         )
        #     mean_weitghted_batch_losses[b] = np.sum(_weitghted_batch_losses)/4
        # mean_weitghted_batch_losses = np.mean(mean_weitghted_batch_losses)

        # After training stpes, use the updated predictor to calculate loss again:
        mean_weitghted_batch_losses_after = np.zeros(batch_num)
        for b in range(batch_num):
            if b <= predictor_input_frames:
                inputs = np.concatenate((self.state['last_trajectory'][-predictor_input_frames+b:,:], trajectory[:b,:]), axis=0)
            else:
                inputs = trajectory[b-predictor_input_frames:b]
            # labels = sequence[:,b+self.predictor.input_frames,-16:]
            labels = trajectory[b,-16:].reshape((1,16))
            inputs = inputs.reshape((1,predictor_input_frames,num_feats))
            prediction = np.array(self.sess.run([self.predictor.prediction], {self.predictor.state_t: inputs}))
    
            batch_loss_ = np.reshape(np.square(np.subtract(labels, prediction)), 16)
            mean_weitghted_batch_losses_after[b] = np.sum(np.multiply(batch_loss_, loss_weight))/4
        mean_weitghted_batch_losses_after = np.mean(mean_weitghted_batch_losses_after)
        return mean_weitghted_batch_losses_after


    # Calculate control path in action_length frames (1/60s per frame), v in meter/s
    def action_generation(self, if_click, direction):
        
        control_path = {'x':[], 'y':[], 'obj':[0]*action_length, 'click':if_click}
        vx = self.state['vx']
        vy = self.state['vy']
        last_mouse_x = self.state['last_trajectory'][-1][4]
        last_mouse_y = self.state['last_trajectory'][-1][5]

        # if direction == 0:
        #     acce_x = acceleration
        #     acce_y = 0
        # elif direction == 1:
        #     acce_x = -acceleration
        #     acce_y = 0
        # elif direction == 2:
        #     acce_x = 0
        #     acce_y = acceleration
        # elif direction == 3:
        #     acce_x = 0
        #     acce_y = -acceleration

        # direction taking value [0,1,2,3,4]: [none, up, down, right ,left]
        for _ in range(action_length):
            if direction == 0:
                vx *= velocity_decay
                vy *= velocity_decay
            elif direction == 1 or direction == 2:
                vx += acceleration*(3-2*direction) # 1:acceleration*1; 2:acceleration*-1
                vy *= velocity_decay
            elif direction == 3 or direction == 4:
                vx *= velocity_decay
                vy += acceleration*(7-2*direction) # 3:acceleration*1; 4:acceleration*-1
            last_mouse_x += vx/60
            last_mouse_x = max(last_mouse_x, 0)
            last_mouse_x = min(last_mouse_x, env_width)
            last_mouse_y += vy/60
            last_mouse_y = max(last_mouse_y, 0)
            last_mouse_y = min(last_mouse_y, env_height)

            control_path['x'].append(last_mouse_x)
            control_path['y'].append(last_mouse_y)
        self.state['vx'] = vx
        self.state['vy'] = vy
        return control_path
    
    def destory(self):
        self.context.eval_js("Destory();")
        return self.trajectory_history
        # with open('./active_training_data/random_data.json') as data_file:    
        #     data = json.load(data_file)
        # data.append(self.trajectory_history)
        # with open('./active_training_data/random_data.json', 'w') as data_file:
        #     json.dump(data, data_file, indent=4)