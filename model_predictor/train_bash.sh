python predictor.py --train True --lr 1e-4 --loss_weight 1 > additive_F_e-4_1_train.txt &&
python predictor.py --train True --lr 1e-5 --loss_weight 0 > additive_F_e-5_0_train.txt &&
python predictor.py --train True --lr 1e-5 --loss_weight 1 > additive_F_e-5_1_train.txt