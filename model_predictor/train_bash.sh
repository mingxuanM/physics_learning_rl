# python predictor.py --train True --save_model True --loss_weight 1 --epochs 50 --training_data_type 0 > random_data_train.txt &&
# python predictor.py --train True --save_model True --loss_weight 1 --epochs 50 --training_data_type 1 > active_data_train.txt


python predictor.py --train True --save_model True --loss_weight 1 --lr 1e-4 --epochs 50 --training_data_type 2 > general_training_direct_prediction_1e-4.txt &&
python predictor.py --train True --save_model True --loss_weight 1 --lr 1e-5 --epochs 50 --training_data_type 2 > general_training_direct_prediction_1e-5.txt

# python predictor.py --train True --save_model True --loss_weight 1 --lr 1e-4 --epochs 50 --training_data_type 2 > general_training_additive_1e-4.txt &&
# python predictor.py --train True --save_model True --loss_weight 1 --lr 1e-5 --epochs 50 --training_data_type 2 > general_training_additive_1e-5.txt


# Extend training
# python predictor.py --train True --lr 1e-4 --loss_weight 0 --epochs 200 > predictor_extend__1e-04_train.txt &&
# python predictor.py --train True --lr 1e-5 --loss_weight 0 --epochs 200 > predictor_extend__1e-05_train.txt