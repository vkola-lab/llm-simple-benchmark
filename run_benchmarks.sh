#!/bin/bash -l

# This script is set up so that you can either qsub it or run it interactively

#$ -P vkolagrp
#$ -l h_rt=12:00:00
#$ -pe omp 8
#$ -l mem_per_core=2G
#$ -l gpus=1
# GPU capability, must be at least 8 for this project
#$ -l gpu_c=8
# We can in theory request a minimum amount of GPU memory, but setting
# capability to 8 means that whatever GPU we get it will definitely have enough
# memory for our purposes

module load python3

module load cuda

# Login using "huggingface-cli login" before running this script "huggingface-cli login"
export HF_HOME=/projectnb/vkolagrp/bellitti/hf_cache

export VLLM_SKIP_P2P_CHECK=1

source venv/bin/activate

python -V

# example looping over hyperparameters, overriding the config.yml file
# for STEPS in 600 1000 2400 3000; do
# for SEED in 6 7 8 9 10; do
    # python src/main.py config_file=config.yml llm_sampling_seed=$SEED lora_path="/example/path/checkpoint-$STEPS" 
# done
# done

python src/main.py config_file=config.yml
