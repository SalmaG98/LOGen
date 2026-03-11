#!/bin/bash
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_$RANDOM
cd ~/workspace/LOGen/logen/modules/PyTorchEMD
pip install numpy==1.26.4
pip install . --no-cache-dir
cd ~/workspace/LOGen
pip install -e .

export CUDA_VISIBLE_DEVICES=3

model_type=$1
epoch=$2

# # Gen Instances
config=config/sloper4d_gen_$model_type.yaml
root_dir=/home/sgalaaou/scania/sgalaaou/LOGen_human_experiments_mgpus
weights=$root_dir/checkpoints/train_sloper4d_$model_type/epoch=$epoch.ckpt
# tokens_to_data=/path/to/tokens_to_data_mapping.json

python generation/gen_instance_pool.py -c $config -w $weights -n 3 -s val -r $root_dir/results --condition recreation #--limit_samples_count 60 #2048
