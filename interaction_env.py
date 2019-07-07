import pyduktape
import numpy as np
import json
import random as rd
import math

class interaction_env:
# Class instance varialbes:
# 1. self.timesteps: current time steps, increment after each action, 
#   used to retrieve trajectories data from JS context
# 2. self.state: latest state (10 frames) {frames[10,22], object, velocity} 
#   (velocity is the final velocity from last action)
# 3. self.context: js environment
# 4. self.cond: a dictionary stores {starting locations, starting velocities, local forces, mass}
    def __init__(self):
        self.timesteps = 0
        #Read in world_setup
        with open('./json/world_setup.json') as data_file:    
            world_setup = json.load(data_file)

        #Read in starting_state
        with open('./json/starting_state.json') as data_file:    
            starting_state = json.load(data_file)

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

        world_setup = rd.sample(world_setup, 1)[0]
        starting_state = rd.sample(starting_state, 1)[0]

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

        self.context.set_globals(cond=cond)
        self.cond['timeout']= 10
        # initial control path for the first 10 frames
        path = {'x':[0]*10, 'y':[0]*10, 'obj':[0]*10}
        # Build js environment
        trajectory = context.eval_js("Run();")
        trajectory = trajectory['physics']
        self.state['frames'] = trajectory
        self.state['object'] = 0

    def act(self, action):
        # Generate action from action space given actionID
        control_path_ = action_generation(self.state['velocity'], action)
        # Set control path
        path = {'x':control_path_['x'],
            'y':control_path_['y'],
            'obj':control_path_['obj']}

        # Feed control path to js context      
        self.context.set_globals(control_path=path)

        #Run the simulation
        trajectory = context.eval_js("onEF();")
        trajectory = json.loads(trajectory) #Convert to python object
        trajectory = trajectory['physics']
        reward = self.reward_cal(trajectory)

        return trajectory, reward
    
    def reward_cal(self, trajectory):
        objs = trajectory["co"]
        mouseX = trajectory["mouse"]['x']
        mouseY = trajectory["mouse"]['y']

        o1x= trajectory["o1"]['x']
        o1y= trajectory["o1"]['y']
        o1vx= trajectory["o1"]['vx']
        o1vy= trajectory["o1"]['vy']

        o2x= trajectory["o2"]['x']
        o2y= trajectory["o2"]['y']
        o2vx= trajectory["o2"]['vx']
        o2vy= trajectory["o2"]['vy']

        o3x= trajectory["o3"]['x']
        o3y= trajectory["o3"]['y']
        o3vx= trajectory["o3"]['vx']
        o3vy= trajectory["o3"]['vy']

        o4x= trajectory["o4"]['x']
        o4y= trajectory["o4"]['y']
        o4vx= trajectory["o4"]['vx']
        o4vy= trajectory["o4"]['vy']
        if self.state['object'] != objs[0]:
            # reward for catching
            pass
        else:
            # reward for decrease predictor loss
            pass
        return 

# Calculate control path in 10 frames (1/60s per frame), v in pixel/s (cm/s)
def action_generation(v, actionID):

    control_path = {'x':[], 'y':[], 'obj':[]}
    return control_path