#!/bin/bash
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_$RANDOM
cd ~/workspace/LOGen/logen/modules/PyTorchEMD
pip install numpy==1.26.4
pip install . --no-cache-dir
cd ~/workspace/LOGen
pip install -e .

cd evaluation

export CUDA_VISIBLE_DEVICES=3

model_type=$1 # xs_1a_dit3d_pe_4ch
channels=$2
epoch=$3 # epoch_$n or last
silent=$4

eval_model=nuscenes
split=val
root_dir=/home/sgalaaou/scania/sgalaaou/LOGen_human_experiments_mgpus/results


if [[ "$channels" -eq 4 ]]; then
pnet_ckpt=fpd/from_scratch/checkpoints/cleaned_nuscenes_objects/last.ckpt
fi

if [[ "$channels" -eq 3 ]]; then
pnet_ckpt=/home/sgalaaou/iveco/workspace/ekirby/LOGen/evaluation/fpd/from_scratch/checkpoints/cleaned_nuscenes_objects_3ch/last.ckpt
fi

# if [[ $silent -eq False ]]; then
# echo $pnet_ckpt
# fi

# Compute PEDESTRIAN
model=gen_sloper4d_$model_type
python compute_fid.py -m $model/$epoch -r $root_dir -s $split -e $eval_model -cls human -i $channels -pckpt $pnet_ckpt -sl $silent

cd ../