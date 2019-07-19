import json
import numpy as np

with open('data/trails_data.json') as json_file:  
    data = json.load(json_file)
    data = data[:685]+data[686:692]+data[693:] # 798 sequences
with open('data/extend_training_data_js.json') as json_file: 
    # extent_data = json.load(json_file)
    data.extend(json.load(json_file)) # 500 sequences; 80% passive; 40% no local forces
    # extent_data = []

np.random.shuffle(data)
# data = np.array(data)

for i in range(3):
    sq = data[i]
    sq = np.array(sq)
    print(sq.shape)