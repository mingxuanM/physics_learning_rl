# This bash file runs a separate training framework with each component step by step.
# learning_system.py provides a intergrated training framework that runs everything with one python script.

# generate test set
# cd js_simulator
# python ./simulate_test_generator.py &&
# # should have:
# # .\model_predictor\data\world_setup_-1_test_set.json
# cd ..

# generate random data set
# python random_agent.py --episode 500 > active_random_learning_world-1.txt &&
# should have:
# ./model_predictor/data/world-1_random_data.json
# echo '------------------------radnom data generated'

# active training with loss reward
python RQN_agent.py --episode 2000 --active_learning True --save_model True --train True > active_learning_loss_reward_world-1.txt &&
# should have:
# ./checkpoints/active_learning_loss_reward_world-1_2000_epochs.ckpt
echo '------------------------loss reward training completed'

# generation active data set
python RQN_agent.py --episode 500 --continue_from 2000 --active_learning True  > active_learning_loss_reward_data_generation_world-1.txt &&
# should have:
# ./model_predictor/data/world-1_active_data.json
echo '------------------------loss reward data generated'

cd model_predictor


# Train predictor on new training sets
python predictor.py --train True --save_model True --loss_weight 1 --epochs 50 --training_data_type 0 > random_data_train_world-1.txt &&
echo '------------------------predictor trained on random data'
python predictor.py --train True --save_model True --loss_weight 1 --epochs 50 --training_data_type 1 > active_data_train_world-1.txt
echo '------------------------predictor trained on active data'

# loss drop reward training
# python RQN_agent.py --episode 500 --active_learning True > active_learning_less_trained_predictor.txt &&