import json
# Format js_simulator generated data to model usable sequence data
with open('./extend_training_data_js.json') as data_file:    
    test_data_js = json.load(data_file)
format_trails = []
for t, trail in enumerate(test_data_js):
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

with open('./extend_training_data_js_format.json', 'w') as fp:
    json.dump(format_trails, fp, sort_keys=True, indent=4)
