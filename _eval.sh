#!/bin/bash

env_name=$1
project_dir=$2
output_dir=$3
config_path=$4
use_deepspeed=$5
num_gpus=$6

cd $project_dir

conda activate $env_name

echo "config_path: ${config_path}"
echo "output_dir: ${output_dir}"

if $use_deepspeed; then
    deepspeed --num_gpus=${num_gpus} run_model.py --config_path $config_path --output_dir $output_dir --num_gpus $num_gpus
else
    python run_model.py --config_path $config_path --output_dir $output_dir
fi

python reshape_output.py --output_dir $output_dir