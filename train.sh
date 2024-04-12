# !/bin/bash
max=1
for i in `seq 0 $max`
do
    python train_kfold.py --fold $i
done
