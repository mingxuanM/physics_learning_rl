import pyduktape
import numpy as np
import json
import random as rd
import math
# from model_predictor.predictor import Predictor
# import tensorflow as tf
# import keras

n_actions = 6 # 1 no action + 4 directions acc + 1 click
# directions_tan = [(np.cos(i*np.pi/8.),np.sin(i*np.pi/8.)) for i in range(16)]
acceleration = 1 # in meter/s/frame (speed meter/s change in each frame)
velocity_decay = 0.9 # velocity in meter/s decay rate per frame if not accelerate
action_length = 5 # frames
width = 6
height = 4
num_feats = 22
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
    def __init__(self):
        self.timesteps = 0
        # Read in world_setup
        with open('./js_simulator/json/world_setup.json') as data_file:    
            self.world_setup = json.load(data_file)

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
        # self.predictor = Predictor(lr=1e-4, name='predictor', num_feats=22, n_state=16, input_frames=5, train=True)

    def reset(self):
        world_setup = rd.sample(self.world_setup, 1)[0]
        starting_state = rd.sample(self.starting_state, 1)[0]

        # Set starting conditions
        # self.cond = {'sls':[{'x':starting_state[0], 'y':starting_state[1]}, {'x':starting_state[4], 'y':starting_state[5]},
        #                 {'x':starting_state[8], 'y':starting_state[9]}, {'x':starting_state[12], 'y':starting_state[13]}],
        #         'svs':[{'x':rd.uniform(-10.0, 10.0), 'y':rd.uniform(-10.0, 10.0)}, {'x':rd.uniform(-10.0, 10.0), 'y':rd.uniform(-10.0, 10.0)},
        #                 {'x':rd.uniform(-10.0, 10.0), 'y':rd.uniform(-10.0, 10.0)}, {'x':rd.uniform(-10.0, 10.0), 'y':rd.uniform(-10.0, 10.0)}]
        #         }
        self.cond = {'sls':[{'x':rd.uniform(0.0, 6.0), 'y':rd.uniform(0.0, 4.0)}, {'x':rd.uniform(0.0, 6.0), 'y':rd.uniform(0.0, 4.0)},
                        {'x':rd.uniform(0.0, 6.0), 'y':rd.uniform(0.0, 4.0)}, {'x':rd.uniform(0.0, 6.0), 'y':rd.uniform(0.0, 4.0)}],
                'svs':[{'x':rd.uniform(-10.0, 10.0), 'y':rd.uniform(-10.0, 10.0)}, {'x':rd.uniform(-10.0, 10.0), 'y':rd.uniform(-10.0, 10.0)},
                        {'x':rd.uniform(-10.0, 10.0), 'y':rd.uniform(-10.0, 10.0)}, {'x':rd.uniform(-10.0, 10.0), 'y':rd.uniform(-10.0, 10.0)}]
                }

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
        trajectory = np.array(json.loads(trajectory)) # Convert to python object
        # self.state['last_trajectory'] = trajectory
        # self.state['caught_object'] = 0
        self.state = {'last_trajectory':trajectory,
            'caught_any':False,
            'vx':0,
            'vy':0}
        
        # self.sess = tf.InteractiveSession()
        # keras.backend.set_session(sess)

        return trajectory

    def act(self, actionID):
        # actionID taking value range(8)
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
        trajectory = np.array(json.loads(trajectory)) # Convert to python object
        reward, is_done = self.reward_cal(trajectory)
        # Update self.state
        self.state['last_trajectory'] = trajectory
        return trajectory, reward, is_done

    # Calculate reward given trajectory [action_length:22] of one action length
    def reward_cal(self, trajectory):
        reward = 0
        is_done = False

        control_object = np.sum(trajectory[0,:4])
        if control_object > 0:
            for i in range(4):
                if trajectory[0,i] == 1:
                    control_object = i+1
                    break

        # if (not self.state['caught_any'] and control_object == 0):
        # nothing happens
        if not self.state['caught_any'] and control_object != 0:
            # object caught
            reward += 1
            self.state['caught_any'] = True
            is_done = True
        elif self.state['caught_any'] and control_object == 0:
            # released last caught object
            self.state['caught_any'] = False
        # elif self.state['caught_any'] and control_object != 0:
        #     pass
            # TODO reward for decrease predictor loss, train predictor
            # batch_num = action_length - self.predictor.input_frames
            # sequence = trajectory.reshape((1,action_length,num_feats))
            # for b in range(batch_num):
            #     labels = sequence[:,b+self.predictor.input_frames,-16:]
            #     inputs = sequence[:,b:b+self.predictor.input_frames,:]
            #     _train_step, _weitghted_batch_losses, _batch_losses = sess.run(
            #         [self.predictor.train_step, self.predictor.weitghted_batch_losses, self.predictor.batch_losses], 
            #         {self.predictor.batch_labels: labels, self.predictor.training_states: inputs, self.predictor.loss_weight:loss_weight}
            #         )
        return reward, is_done

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
            if direction <= 2:
                vx += acceleration*(3-2*direction) # 1:acceleration*1; 2:acceleration*-1
                vy *= velocity_decay
            else:
                vx *= velocity_decay
                vy += acceleration*(7-2*direction) # 3:acceleration*1; 4:acceleration*-1
            last_mouse_x += vx/60
            last_mouse_x = max(last_mouse_x, 0)
            last_mouse_x = min(last_mouse_x, width)
            last_mouse_y += vy/60
            last_mouse_y = max(last_mouse_y, 0)
            last_mouse_y = min(last_mouse_y, height)

            control_path['x'].append(last_mouse_x)
            control_path['y'].append(last_mouse_y)
        self.state['vx'] = vx
        self.state['vy'] = vy
        return control_path
    
    def destory(self):
        self.context.eval_js("Destory();")