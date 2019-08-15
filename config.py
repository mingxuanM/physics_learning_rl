import numpy as np

n_actions = 6 # 1 no action + 4 directions acc + 1 click
# directions_tan = [(np.cos(i*np.pi/8.),np.sin(i*np.pi/8.)) for i in range(16)]
acceleration = 1 # in meter/s/frame (speed meter/s change in each frame)
velocity_decay = 0.9 # velocity in meter/s decay rate per frame if not accelerate

action_length = 5 # frames

env_width = 6
env_height = 4

num_feats = 22
RQN_num_feats = 22 # 4 caught object + 2 mouse + 4*4


qlearning_gamma = 0.9

# epsilon_decay = 0.9995 # 10000 epochs
epsilon_decay = 0.995 # 2000 epochs

# number of frames for model predictor network
predictor_input_frames = 5

# bonus rate of 'predictor loss drop reward' when agent is dragging
# final reward will be (1 + dragging_bonus) * np.abs(pretrained_loss - trained_loss)
dragging_bonus = 1.5 

loss_weight = np.array([1.6973/6.3432, 1.0517/6.3432, 1.7830/6.3432, 1.8112/6.3432, 
                                1.6973/6.3432, 1.0517/6.3432, 1.7830/6.3432, 1.8112/6.3432, 
                                1.6973/6.3432, 1.0517/6.3432, 1.7830/6.3432, 1.8112/6.3432,
                                1.6973/6.3432, 1.0517/6.3432, 1.7830/6.3432, 1.8112/6.3432])