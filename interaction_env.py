import pyduktape
import numpy as np
import json
import random as rd
import math
from model_predictor.predictor import Predictor

n_actions = (16*(1+1+1) + 1) * 5
directions_tan = [(np.cos(i*np.pi/8.),np.sin(i*np.pi/8.)) for i in range(16)]
acceleration = [1,-1,0] # in meter/s/frame (speed meter/s change in each frame)
action_length = 10 # frames

class Interaction_env:
# Class instance varialbes:
# 1. self.timesteps: current time steps, increment after each action, 
#   used to retrieve trajectories data from JS context
# 2. self.state: latest state (10 frames) {'last_trajectory':[10,22], 'object', 'velocity'} 
#   (velocity is the final velocity from last action)
# 3. self.context: js environment
# 4. self.cond: a dictionary stores {'starting locations', 'starting velocities', 'local forces', 'mass'}
    def __init__(self):
        self.timesteps = 0
        # Read in world_setup
        with open('./json/world_setup.json') as data_file:    
            self.world_setup = json.load(data_file)

        # Read in starting_state
        with open('./json/starting_state.json') as data_file:    
            self.starting_state = json.load(data_file)

        # Create js environment
        self.context = pyduktape.DuktapeContext()

        # Load js box2d graphic library
        js_file = open("./js/box2d.js",'r')
        js = js_file.read()
        self.context.eval_js(js)

        # Load world environment script
        js_file = open("./js/control_world.js",'r')
        js = js_file.read()
        self.context.eval_js(js)
        self.state['caught_object'] = 0
        self.cond = {}
        self.predictor = Predictor(lr=1e-4, name='predictor', num_feats=22, n_state=16, input_frames=5, train=True)

    def reset(self):
        world_setup = rd.sample(self.world_setup, 1)[0]
        starting_state = rd.sample(self.starting_state, 1)[0]

        # Set starting conditions
        self.cond = {'sls':[{'x':starting_state[0], 'y':starting_state[1]}, {'x':starting_state[4], 'y':starting_state[5]},
                        {'x':starting_state[8], 'y':starting_state[9]}, {'x':starting_state[12], 'y':starting_state[13]}],
                'svs':[{'x':starting_state[2], 'y':starting_state[3]}, {'x':starting_state[6], 'y':starting_state[7]},
                        {'x':starting_state[10], 'y':starting_state[11]}, {'x':starting_state[14], 'y':starting_state[15]}]
                }

        # Set local forces
        self.cond['lf'] = [[0.0,float(world_setup[0]), float(world_setup[1]), float(world_setup[2])],
                [0.0, 0.0, float(world_setup[3]), float(world_setup[4])],
                [0.0, 0.0, 0.0, float(world_setup[5])],
                [0.0, 0.0, 0.0, 0.0]]

        # Set mass
        if world_setup[6]=='A':
            cond['mass'] = [2,1,1,1]
        elif world_setup[6]=='B':
            cond['mass'] = [1,2,1,1]
        else:
            cond['mass'] = [1,1,1,1]

        # # Set timeout
        # self.cond['timeout']= 10

        self.context.set_globals(cond=self.cond)
        
        # initial control path for the first 10 frames
        path = {'x':[0]*10, 'y':[0]*10, 'obj':[0]*10}
        # Feed control path to js context      
        self.context.set_globals(control_path=path)
        # Build js environment, get trajectory for the first 10 frames
        trajectory = context.eval_js("Run();") # [10,22]
        trajectory = json.loads(trajectory) # Convert to python object
        self.state['last_trajectory'] = trajectory

        return trajectory

    def act(self, actionID):
        # see if the target object is caught
        # if not, set target_object to 0 (not draging it in JS context)
        target_object = actionID // (16*(1+1+1) + 1) # taking value 0,1,2,3,4
        if self.state['caught_object'] != target_object:
            self.state['caught_object'] = 0
            # target_object = 0
        actionID %= (16*(1+1+1) + 1)
        # Generate control path from action space given actionID
        control_path_ = action_generation(actionID, self.state['caught_object'])
        
        # Feed control path to js context      
        self.context.set_globals(control_path=control_path_)

        # Run the simulation
        trajectory = context.eval_js("action_forward();") # [10,22]
        trajectory = json.loads(trajectory) # Convert to python object
        reward, is_done, just_caught = self.reward_cal(trajectory, target_object)
        # Update self.state
        self.state['last_trajectory'] = trajectory
        return trajectory, reward, is_done, just_caught

    # Calculate reward given trajectory [10:22] of one action length
    def reward_cal(self, trajectory, target_object):
        
        reward = 0
        is_done = False
        just_caught = 0
        # if agent is not targeting any object
        if target_object == 0:
            return 0, False
        # if target_object has not been caught
        elif target_object != self.state['caught_object']:
            # reward for catching
            caught = False
            init_distance = self.state['last_trajectory'][-1]
            init_distance = ((init_distance[4] - init_distance[6+(target_object-1)*4])**2 + (init_distance[5] - init_distance[7+(target_object-1)*4])**2)**0.5
            for f in range(action_length):
                distance = ((trajectory[f][4] - trajectory[f][6+(target_object-1)*4])**2 + (trajectory[f][5] - trajectory[f][7+(target_object-1)*4])**2)**0.5
                reward += max(init_distance - distance, 0)
                # ball radius = 0.125
                if distance <= 0.125:
                    self.state['caught_object'] = target_object
                    just_caught = target_object
                    reward += 5
        else:
            # TODO reward for decrease predictor loss
            pass
        return reward, is_done, just_caught

    # Calculate control path in 10 frames (1/60s per frame), v in meter/s
    def action_generation(self, actionID, target_object):
        # target_object = actionID // (16*(1+1+1) + 1)
        # # see if the target object co is caught
        # # if not, set co to 0 (not draging it)
        # if self.state['caught_object'] != target_object:
        #     self.state['caught_object'] = 0
        #     target_object = 0
        # actionID %= (16*(1+1+1) + 1)
        control_path = {'x':[], 'y':[], 'obj':[target_object]*action_length}
        v = self.state['velocity']
        last_mouse_x = self.state['last_trajectory']['mouse']['x'][-1]
        last_mouse_y = self.state['last_trajectory']['mouse']['y'][-1]
        if actionID < 16*(1+1+1):
            directions_tan_ = directions_tan[actionID // 3]
            acceleration_ = acceleration[actionID % 3]
            control_path['x']=[(v+(i+1)*acceleration_/2)*(i+1)/60*directions_tan_[0] + last_mouse_x for i in range(action_length)]
            control_path['y']=[(v+(i+1)*acceleration_/2)*(i+1)/60*directions_tan_[1] + last_mouse_y for i in range(action_length)]
            self.state['velocity'] = v + acceleration_*10
        else:
            control_path['x']=[last_mouse_x] * action_length
            control_path['y']=[last_mouse_y] * action_length
            self.state['velocity'] = 0
        return control_path