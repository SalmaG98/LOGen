#!/bin/bash
cd ~/workspace/LOGen/logen/modules/PyTorchEMD
pip install numpy==1.26.4
pip install .
cd ~/workspace/LOGen

model_type=$1
root_dir=/home/sgalaaou/scania/sgalaaou/LOGen_human_experiments_mgpus
# weights=$2

python train.py -c config/sloper4d_train_$model_type.yaml -r $root_dir # -w $root_dir/checkpoints/train_sloper4d_$model_type/$weights-last.ckpt