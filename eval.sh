#!/bin/bash

project_dir="/path/to/your/ECBench/directory/"

config_path=$1
exp_name=$2

use_deepspeed=false
for arg in "$@"; do
    if [[ "$arg" == "--deepspeed" ]]; then
        use_deepspeed=true
    fi
done

num_gpus=1
for arg in "$@"; do
    if [[ "$arg" == "--num_gpus="* ]]; then
        num_gpus=${arg#--num_gpus=}
    fi
done

cd $project_dir

echo "Use Qwen3VL"
env_name="ecbench"

config_name=$(basename "$config_path" .yaml)

test_name="${config_name}-${exp_name}"
output_dir="${project_dir}/result/${test_name}"

mkdir -p $output_dir
cp $config_path $output_dir

bash _eval.sh $env_name $project_dir $output_dir $config_path $use_deepspeed $num_gpus

