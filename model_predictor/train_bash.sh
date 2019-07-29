python predictor.py --train True --save_model True --loss_weight 1 --epochs 50 --training_data_type 0 > random_data_train.txt &&
python predictor.py --train True --save_model True --loss_weight 1 --epochs 50 --training_data_type 1 > active_data_train.txt

# Extend training
# python predictor.py --train True --lr 1e-4 --loss_weight 0 --epochs 200 > predictor_extend__1e-04_train.txt &&
# python predictor.py --train True --lr 1e-5 --loss_weight 0 --epochs 200 > predictor_extend__1e-05_train.txt