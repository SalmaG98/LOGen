#!/bin/bash
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_$RANDOM
cd ~/workspace/LOGen/logen/modules/PyTorchEMD
pip install numpy==1.26.4
pip install . --no-cache-dir
cd ~/workspace/LOGen
pip install -e .

cd evaluation

export CUDA_VISIBLE_DEVICES=3

model_type=$1
channels=$2
epoch=$3 # epoch_$n or last
silent=$4

eval_model=nuscenes
split=val
root_dir=/home/sgalaaou/scania/sgalaaou/LOGen_human_experiments_mgpus/results

# Compute HUMAN
model=gen_sloper4d_$model_type
python compute_cd_emd.py -m $model/$epoch -r $root_dir -s $split -cls human -i $channels -sl $silent

cd ../