import json
import numpy as np
# Calculate mean standard deviation of x,y,xv,yv in trails in original training set

def std_div():
    with open('lsclips.json') as lsclips_json:  
        lsclips = json.load(lsclips_json)
    std_x = []
    std_xv = []
    std_y = []
    std_yv = []
    for p in range(40):
        for t in range(20):
            std_x.append(np.std(lsclips[p][t]['o1.x'] + lsclips[p][t]['o2.x'] + lsclips[p][t]['o3.x'] + lsclips[p][t]['o4.x']))
            std_xv.append(np.std(lsclips[p][t]['o1.vx'] + lsclips[p][t]['o2.vx'] + lsclips[p][t]['o3.vx'] + lsclips[p][t]['o4.vx']))
            std_y.append(np.std(lsclips[p][t]['o1.y'] + lsclips[p][t]['o2.y'] + lsclips[p][t]['o3.y'] + lsclips[p][t]['o4.y']))
            std_yv.append(np.std(lsclips[p][t]['o1.vy'] + lsclips[p][t]['o2.vy'] + lsclips[p][t]['o3.vy'] + lsclips[p][t]['o4.vy']))
    return np.mean(std_x), np.mean(std_xv), np.mean(std_y), np.mean(std_yv)
# mean standard deviation for x is: 1.6973236662957578
# mean standard deviation for xv is: 1.7830307647729327
# mean standard deviation for y is: 1.0517453222060384
# mean standard deviation for yv is: 1.81123320699984