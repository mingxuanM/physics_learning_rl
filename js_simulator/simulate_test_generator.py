'''Call javascript implementation of task from within python'''

import pyduktape
import numpy as np
import json
import random as rd
import math
# import imp


#Read in world setups
###################
with open('./json/world_setup.json') as data_file:    
    world_setup = json.load(data_file)

#Read in mouse data
###################
with open('./json/control_path.json') as data_file:    
    control_path = json.load(data_file)

#Read in starting conditions
#############################
with open('./json/starting_state.json') as data_file:    
    starting_state = json.load(data_file)

#Create js environment
######################
context = pyduktape.DuktapeContext()

#Import js library
##################
js_file = open("./js/box2d.js",'r')
js = js_file.read()
context.eval_js(js)

#Load the js script
################
js_file = open("./js/control_world_backup.js",'r')
js = js_file.read()
context.eval_js(js)

test_data = []

for i in range(200):
    #Run through a trial
    ######################################

    rd.seed(i)#Set seed

    # Choose a starting condition at random from participant data
    # world_setup_ = rd.sample(world_setup, 1)[0]
    # world_setup_ = world_setup[4]
    world_setup_ = world_setup[-1]
    # 80% to generate a passive episode
    is_passive = rd.random()
    is_passive = is_passive <= 0.8
    if not is_passive:
        control_path_ = rd.sample(control_path, 1)[0]
    starting_state_ = rd.sample(starting_state, 1)[0]
    
    # setup = trial['setup']
    # onset = trial['onset']
    # offset = trial['offset']

    # Set starting conditions
    # cond = {'sls':[{'x':start['o1.x'], 'y':start['o1.y']}, {'x':start['o2.x'], 'y':start['o2.y']},
    #     {'x':start['o3.x'], 'y':start['o3.y']}, {'x':start['o4.x'], 'y':start['o4.y']}],
    #     'svs':[{'x':start['o1.vx'], 'y':start['o1.vy']}, {'x':start['o2.vx'], 'y':start['o2.vy']},
    #         {'x':start['o3.vx'], 'y':start['o3.vy']}, {'x':start['o4.vx'], 'y':start['o4.vy']}]
    #     }
    # cond = {'sls':[{'x':starting_state_[0], 'y':starting_state_[1]}, {'x':starting_state_[4], 'y':starting_state_[5]},
    #                {'x':starting_state_[8], 'y':starting_state_[9]}, {'x':starting_state_[12], 'y':starting_state_[13]}],
    #         'svs':[{'x':starting_state_[2], 'y':starting_state_[3]}, {'x':starting_state_[6], 'y':starting_state_[7]},
    #                {'x':starting_state_[10], 'y':starting_state_[11]}, {'x':starting_state_[14], 'y':starting_state_[15]}]
    #         }
    cond = {'sls':[{'x':starting_state_[0], 'y':starting_state_[1]}, {'x':starting_state_[4], 'y':starting_state_[5]},
                   {'x':starting_state_[8], 'y':starting_state_[9]}, {'x':starting_state_[12], 'y':starting_state_[13]}],
            'svs':[{'x':rd.uniform(-10.0, 10.0), 'y':rd.uniform(-10.0, 10.0)}, {'x':rd.uniform(-10.0, 10.0), 'y':rd.uniform(-10.0, 10.0)},
                   {'x':rd.uniform(-10.0, 10.0), 'y':rd.uniform(-10.0, 10.0)}, {'x':rd.uniform(-10.0, 10.0), 'y':rd.uniform(-10.0, 10.0)}]
            }

    # Set local forces
    # 30% to have no local forces
    # no_force = rd.random()
    # if no_force <= 0.3:
    #     cond['lf'] = [[0.0,0.0,0.0,0.0],
    #             [0.0,0.0,0.0,0.0],
    #             [0.0,0.0,0.0,0.0],
    #             [0.0,0.0,0.0,0.0]]
    # else:
    #     cond['lf'] = [[0.0,float(world_setup_[0]), float(world_setup_[1]), float(world_setup_[2])],
    #             [0.0, 0.0, float(world_setup_[3]), float(world_setup_[4])],
    #             [0.0, 0.0, 0.0, float(world_setup_[5])],
    #             [0.0, 0.0, 0.0, 0.0]]
    
    cond['lf'] = [[0.0,float(world_setup_[0]), float(world_setup_[1]), float(world_setup_[2])],
                [0.0, 0.0, float(world_setup_[3]), float(world_setup_[4])],
                [0.0, 0.0, 0.0, float(world_setup_[5])],
                [0.0, 0.0, 0.0, 0.0]]

    # Set mass
    if world_setup_[6]=='A':
        cond['mass'] = [2,1,1,1]
    elif world_setup_[6]=='B':
        cond['mass'] = [1,2,1,1]
    else:
        cond['mass'] = [1,1,1,1]

    # Set control path
    # path = {'x':[1]*onset['frame'][0],
    # 'y':[1]*onset['frame'][0],
    # 'obj':[0]*onset['frame'][0]}
    if is_passive:
        path = {'x':[0]*1801,
            'y':[0]*1801,
            'obj':[0]*1801}
    else:
        path = {'x':[x/100 for x in control_path_['x']],
            'y':[y/100 for y in control_path_['y']],
            'obj':control_path_['obj']}

    cond['timeout']=len(path['x'])

    #Simulate in javascript
    ########################
    context.set_globals(cond=cond)
    context.set_globals(control_path=path)

    #Run the simulation
    ###################
    # print('start to run the js Run()...')
    data = context.eval_js("Run();")
    data = json.loads(data) #Convert to python object
    data = data['physics']
    test_data.append(data)
    #Check something happened
    #########################
    # print('data')
    # print(data['physics']['o1']['x'][0:100])
    # print(data.keys())
    #data['physics']['o1']['y'][0:100]

    #Evaluate TODO
    # data['physics']['o1']['x'][0:100]
    # ppt_dat[4]['o1x'[0:100]]
    # with open('../../R/data/replay_files_exp4/ppt_10_uid_A14ADQ7RUN6TDY.json') as data_file:    
    #     ppt_data = json.load(data_file)


    print('trail {} generated'.format(i))
    #Save data
    ##########
# with open('./extend_training_data_js.json', 'w') as fp:
#     json.dump(test_data, fp, sort_keys=True, indent=4)


#Make a movie
############

# imp.load_source("make_movies_simulation.py", "../make_movies/")
# import ../make_movies/make_movies_simulation

#Plot things out
###############
# import gizeh
# surface = gizeh.Surface(width=320, height=260) # in pixels
# circle = gizeh.circle(r=30, xy= [40,40], fill=(1,0,0))
# circle.draw(surface) # draw the circle on the surface
# surface.write_to_png("circle.png") # export the surface as a PNG

format_trails = []
for t, trail in enumerate(test_data):
    objs = trail["co"]
    mouseX = trail["mouse"]['x']
    mouseY = trail["mouse"]['y']

    o1x= trail["o1"]['x']
    o1y= trail["o1"]['y']
    o1vx= trail["o1"]['vx']
    o1vy= trail["o1"]['vy']

    o2x= trail["o2"]['x']
    o2y= trail["o2"]['y']
    o2vx= trail["o2"]['vx']
    o2vy= trail["o2"]['vy']

    o3x= trail["o3"]['x']
    o3y= trail["o3"]['y']
    o3vx= trail["o3"]['vx']
    o3vy= trail["o3"]['vy']

    o4x= trail["o4"]['x']
    o4y= trail["o4"]['y']
    o4vx= trail["o4"]['vx']
    o4vy= trail["o4"]['vy']
    format_trail = []
    for f, obj in enumerate(objs):
        one_hot = [0,0,0,0]
        if obj != 0:
            one_hot[obj-1] = 1
        format_trail.append((
            one_hot[0], one_hot[1], one_hot[2], one_hot[3], mouseX[f], mouseY[f],
            o1x[f], o1y[f], o1vx[f], o1vy[f], 
            o2x[f], o2y[f], o2vx[f], o2vy[f], 
            o3x[f], o3y[f], o3vx[f], o3vy[f], 
            o4x[f], o4y[f], o4vx[f], o4vy[f]
        ))
    format_trails.append(format_trail)
    print('trail {} formated'.format(t))
    trail = {}

with open('../model_predictor/data/world_setup_-1_test_set.json', 'w') as fp:
    json.dump(format_trails, fp, sort_keys=True, indent=4)