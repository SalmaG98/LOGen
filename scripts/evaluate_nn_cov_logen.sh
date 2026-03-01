#!/bin/bash
cd ~/workspace/LOGen/logen/modules/PyTorchEMD
pip install numpy==1.26.4 > /dev/null
pip install . > /dev/null
cd ~/workspace/LOGen
pip install -e . > /dev/null
cd ~/workspace/LOGen/pytorch3d
pip install -e . > /dev/null
cd ~/workspace/LOGen

cd evaluation

export CUDA_VISIBLE_DEVICES=3

model_type=$1 # xs_1a_dit3d_pe_4ch
channels=$2
epoch=$3 # epoch_$n or last
silent=$4

eval_model=sloper4d
split=val
root_dir=/home/sgalaaou/scania/sgalaaou/LOGen_human_experiments_mgpus/results
distance_method=EMD

if [[ "$channels" -eq 4 ]]; then
pnet_ckpt=fpd/from_scratch/checkpoints/cleaned_nuscenes_objects/last.ckpt
fi

if [[ "$channels" -eq 3 ]]; then
pnet_ckpt=/home/sgalaaou/iveco/workspace/ekirby/LOGen/evaluation/fpd/from_scratch/checkpoints/cleaned_nuscenes_objects_3ch/last.ckpt
fi

# Compute PEDESTRIAN
model=gen_sloper4d_$model_type
python compute_1nn_cov.py -m $model/$epoch -s $split -i $channels -r $root_dir -cls human -d $distance_method -sl $silent

cd ../