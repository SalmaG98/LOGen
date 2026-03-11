# !/bin/bash

cd evaluation

if [ "$#" -eq 4 ]; then # script received model_name and gpu_id only
    gpu_id=0
    model_type=$1
    channels=$2
    epoch=$3
    silent=$4
elif [ "$#" -eq 5 ]; then
    gpu_id=$1
    model_type=$2
    channels=$3
    epoch=$4
    silent=$5
fi

eval_model=nuscenes
split=val
root_dir=/home/sgalaaou/scania/sgalaaou/LOGen_human_experiments_mgpus/results

if [[ "$channels" -eq 4 ]]; then
pnet_ckpt=fpd/from_scratch/checkpoints/cleaned_nuscenes_objects/last.ckpt
fi

if [[ "$channels" -eq 3 ]]; then
pnet_ckpt=/home/sgalaaou/iveco/workspace/ekirby/LOGen/evaluation/fpd/from_scratch/checkpoints/cleaned_nuscenes_objects_3ch/last.ckpt
fi

# Compute PEDESTRIAN
model=gen_sloper4d_$model_type
CUDA_VISIBLE_DEVICES=$gpu_id python compute_classification_acc.py -m $model/$epoch -r $root_dir -s $split -e $eval_model -cls human -i $channels -pckpt $pnet_ckpt --class_label 0 -sl $silent

cd ../