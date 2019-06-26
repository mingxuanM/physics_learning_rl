import numpy as np
import math
import gizeh as gz 
# conda install cairo
# pip install gizeh
import moviepy.editor as mpy
# conda install moviepy
import time
from functools import partial
import json



def make_frame(this_data, t):
    labels = ['A','B','C','D']
    centers = np.array(['o1','o2','o3','o4'])
    colors = [(0.97,0.46,0.44),(0.48,0.68,0),(0,0.75,0.75),(0.78,0.48,1)]
    RATIO = 100
    RAD = 25
    W = 600
    H = 400
    # H_outer = 500
    N_OBJ=4

    frame = int(math.floor(t*60))# Number of frames
    # print(frame)
    #Essentially pauses the action if there are no more frames and but more clip duration
    # if frame >= len(this_data["co"]):
    #     frame = len(this_data["co"])-1
    if frame >= len(this_data):
        frame = len(this_data) - 1

    #White background
    surface = gz.Surface(W,H, bg_color=(1,1,1))            

    #Walls
    wt = gz.rectangle(lx=W, ly=20, xy=(W/2,10), fill=(0,0,0))#, angle=Pi/8
    wb = gz.rectangle(lx=W, ly=20, xy=(W/2,H-10), fill=(0,0,0))
    wl = gz.rectangle(lx=20, ly=H, xy=(10,H/2), fill=(0,0,0))
    wr = gz.rectangle(lx=20, ly=H, xy=(W-10,H/2), fill=(0,0,0))
    wt.draw(surface)
    wb.draw(surface)
    wl.draw(surface)
    wr.draw(surface)

    #Pucks
    for i, (label, color, center) in enumerate(zip(labels, colors, centers)):

        # xy = np.array([this_data[center]['x'][frame]*RATIO, this_data[center]['y'][frame]*RATIO])
        xy = np.array([this_data[frame][i*4]*RATIO, this_data[frame][i*4+1]*RATIO])

        ball = gz.circle(r=RAD, fill=color).translate(xy)
        ball.draw(surface)

        #Letters
        text = gz.text(label, fontfamily="Helvetica",  fontsize=25, fontweight='bold', fill=(0,0,0), xy=xy) #, angle=Pi/12
        text.draw(surface)

    #Mouse cursor
    # cursor_xy = np.array([this_data['mouse']['x'][frame]*RATIO, this_data['mouse']['y'][frame]*RATIO])
    # cursor = gz.text('+', fontfamily="Helvetica",  fontsize=25, fill=(0,0,0), xy=cursor_xy) #, angle=Pi/12
    # cursor.draw(surface)

    #Control
    # if this_data['co'][frame]!=0:
    #     if this_data['co'][frame]==1:
    #         xy = np.array([this_data['o1']['x'][frame]*RATIO, this_data['o1']['y'][frame]*RATIO])
    #     elif this_data['co'][frame]==2:
    #         xy = np.array([this_data['o2']['x'][frame]*RATIO, this_data['o2']['y'][frame]*RATIO])
    #     elif this_data['co'][frame]==3:
    #         xy = np.array([this_data['o3']['x'][frame]*RATIO, this_data['o3']['y'][frame]*RATIO])
    #     elif this_data['co'][frame]==4:
    #         xy = np.array([this_data['o4']['x'][frame]*RATIO, this_data['o4']['y'][frame]*RATIO])

    #     #control_border = gz.arc(r=RAD, a1=0, a2=np.pi, fill=(0,0,0)).translate(xy)
    #     control_border = gz.circle(r=RAD,  stroke_width= 2).translate(xy)
    #     control_border.draw(surface)

    return surface.get_npimage()



if __name__ == "__main__":

    for i in range(1,6):
        for s_idx in range(10):
            with open('{}0epochs_sequence{}.json'.format(i,s_idx)) as json_file:  
                sequence = json.load(json_file)

            # with open('../data/test_data.json') as json_file:  
            #     sequence = json.load(json_file)

            # data_ = test(args, new_env)

            frame = partial(make_frame, sequence)
            # duration = len(data_['co'])/60
            # 60 frames per second
            duration = len(sequence)/60 # seconds
            clip = mpy.VideoClip(frame, duration=duration)
            writename = 'videos/{}0epochs_sequence{}.mp4'.format(i, s_idx)
            clip.write_videofile(writename, fps=24)
