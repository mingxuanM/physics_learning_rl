import sys
from environment import physic_env
from config import *
import numpy as np
import json
'''
Util script
Use physic_env from github.com/allenlsj/physics_world_rl 
(in script /src/code/simulator/environment.py)
to generate trajectories as test set for the predictor network
'''


def generate_test():
    mode = 1
    T = 500
    new_env = physic_env(train_cond, mass_list, force_list,
                            init_mouse, T, mode, prior, reward_stop)

    # s_next, r, is_done, r_others = new_env.step(a)

    control_vec = {'obj': np.repeat(0, T), 'x': np.repeat(0, T), 'y': np.repeat(0, T)}

    current_cond = new_env.update_condition(new_env.cond['mass'],new_env.cond['lf'])
    new_env.update_bodies(current_cond)
    local_data = new_env.initial_data()


    for t in range(T):
        local_trajectory = new_env.update_simulate_bodies(current_cond, control_vec,t,local_data)
    trajectory = []
    for f in range(T):
        trajectory.append([0,0,0,0,0,0,local_trajectory['o1']['x'][f],local_trajectory['o1']['y'][f],local_trajectory['o1']['vx'][f],local_trajectory['o1']['vy'][f],
                        local_trajectory['o2']['x'][f],local_trajectory['o2']['y'][f],local_trajectory['o2']['vx'][f],local_trajectory['o2']['vy'][f],
                        local_trajectory['o3']['x'][f],local_trajectory['o3']['y'][f],local_trajectory['o3']['vx'][f],local_trajectory['o3']['vy'][f],
                        local_trajectory['o4']['x'][f],local_trajectory['o4']['y'][f],local_trajectory['o4']['vx'][f],local_trajectory['o4']['vy'][f]
            ])
    return trajectory



if __name__ == '__main__':
    test_set = []
    for t in range(500):
        test_set.append(generate_test())
        print('trail {} generated'.format(t))
        

    # Writing to a file
    with open('test_data.json', 'w') as outfile:
        json.dump(test_set, outfile, ensure_ascii=False, indent=2)

    print('test set json file saved!')


