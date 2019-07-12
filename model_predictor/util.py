import json
import numpy as np

input_frame = 5
n_state =16

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
# mean standard deviation for y is: 1.0517453222060384
# mean standard deviation for xv is: 1.7830307647729327
# mean standard deviation for yv is: 1.81123320699984
# Sum to 6.3432


# Baseline cases
# 1. Worst case
# Sample same time step frames from a random trail as prediction
# Worst case baseline loss: [5.73467882 2.21698449 6.72088708 6.80948858]
def Worst_case_baseline(data):
    print('Worst_case_baseline started...')
    epoch_loss = np.zeros((len(data), int(n_state/4)))
    for s_idx, sequence in enumerate(data):
        batch_num = len(sequence) // (input_frame+1)
        sequence_loss = np.zeros((batch_num, int(n_state/4)))
        for n in range(batch_num):
            start = n*(input_frame+1)
            end = start+5
            label = sequence[end,-n_state:]
            prediction_trail = data[np.random.random_integers(len(data))-1]
            if len(prediction_trail) <= end:
                prediction = prediction_trail[-1,-n_state:]
            else:
                prediction = prediction_trail[end,-n_state:]
            batch_loss_ = np.square(np.subtract(label, prediction))
            sequence_loss[n] = np.array([
                        np.mean([batch_loss_[0],batch_loss_[4],batch_loss_[8],batch_loss_[12]]), # x
                        np.mean([batch_loss_[1],batch_loss_[5],batch_loss_[9],batch_loss_[13]]), # y
                        np.mean([batch_loss_[2],batch_loss_[6],batch_loss_[10],batch_loss_[14]]), # vx
                        np.mean([batch_loss_[3],batch_loss_[7],batch_loss_[11],batch_loss_[15]])  # vy
                    ])
        epoch_loss[s_idx] = np.mean(sequence_loss, axis=0)
    final_loss = np.mean(epoch_loss, axis=0)
    print('Worst case baseline loss: '+ str(final_loss))
    print('weighted average: '+ str(final_loss[0]*1.6973/6.3432 + final_loss[1]*1.0517/6.3432 + final_loss[2]*1.7830/6.3432 + final_loss[3]*1.8112/6.3432))

 

# 2. Infinite inertia case
# Use last frame as prediction
# Infinite inertia baseline loss: [0.00092996 0.00093225 0.31726211 0.42942083]
def Infinite_inertia_case_baseline(data):
    print('Infinite_inertia_case_baseline started...')
    # input_frame = 5
    # n_state =16
    epoch_loss = np.zeros((len(data), int(n_state/4)))
    for s_idx, sequence in enumerate(data):
        batch_num = len(sequence) // (input_frame+1)
        sequence_loss = np.zeros((batch_num, int(n_state/4)))
        for n in range(batch_num):
            start = n*(input_frame+1)
            end = start+5
            label = sequence[end,-n_state:]
            prediction = sequence[end-1,-n_state:]
            batch_loss_ = np.square(np.subtract(label, prediction))
            sequence_loss[n] = np.array([
                        np.mean([batch_loss_[0],batch_loss_[4],batch_loss_[8],batch_loss_[12]]), # x
                        np.mean([batch_loss_[1],batch_loss_[5],batch_loss_[9],batch_loss_[13]]), # y
                        np.mean([batch_loss_[2],batch_loss_[6],batch_loss_[10],batch_loss_[14]]), # vx
                        np.mean([batch_loss_[3],batch_loss_[7],batch_loss_[11],batch_loss_[15]])  # vy
                    ])
        epoch_loss[s_idx] = np.mean(sequence_loss, axis=0)
    final_loss = np.mean(epoch_loss, axis=0)
    print('Infinite inertia baseline loss: '+ str(final_loss))
    print('weighted average: '+ str(final_loss[0]*1.6973/6.3432 + final_loss[1]*1.0517/6.3432 + final_loss[2]*1.7830/6.3432 + final_loss[3]*1.8112/6.3432))


if __name__ == '__main__':
    # load training data
    # with open('data/trails_data.json') as json_file:  
    with open('data/test_data_js.json') as json_file: 
        data = json.load(json_file)
        # remove two outliers from training_data [685] & [692]
        data = np.array(data[:685]+data[686:692]+data[693:])
    print('training data loaded')
    Infinite_inertia_case_baseline(data)
    # Infinite inertia baseline loss, training set: [0.00092996 0.00093225 0.31726211 0.42942083]
    #   weighted average: 0.21219640210918775
    # Infinite inertia baseline loss, test set: [0.0015752  0.00151848 0.56136155 0.75724643]
    #   weighted average: 0.37468516680670216

    Worst_case_baseline(data)
    # Worst case baseline loss, training set: [5.73467882 2.21698449 6.72088708 6.80948858]
    #   weighted average: 5.735553085044614
    # Worst case baseline loss, test set:  [7.16837586  2.57957311 11.34820187 11.10042121]
    #   weighted average: 8.70518795433364

    # 40 epochs reduced input model: [0.00147525 0.0013423  0.33779012 0.29830249]
    # 40 epochs full input model: [0.00370258 0.00293012 0.24513803 0.25428428]

