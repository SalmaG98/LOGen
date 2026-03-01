# !/bin/bash
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
pnet_ckpt=evaluation/fpd/from_scratch/checkpoints/cleaned_nuscenes_objects/last.ckpt
fi

if [[ "$channels" -eq 3 ]]; then
pnet_ckpt=/home/sgalaaou/iveco/workspace/ekirby/LOGen/evaluation/fpd/from_scratch/checkpoints/cleaned_nuscenes_objects_3ch/last.ckpt
fi

# echo $pnet_ckpt

# Compute PEDESTRIAN
model=gen_sloper4d_$model_type
python compute_jsd.py -m $model/$epoch -r $root_dir -s $split -cls human -i $channels -sl $silent

cd ../