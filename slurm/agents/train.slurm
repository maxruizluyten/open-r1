#!/bin/bash
#SBATCH --job-name=agent-train
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --output=./logs/%x_%j_%n.out
#SBATCH --error=./logs/%x_%j_%n.err
#SBATCH --time=7-00:00:00
set -exuo pipefail

source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /fsx/aymeric/venv

accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path HuggingFaceTB/SmolVLM2-2.2B-Instruct \
    --dataset_name smolagents/gaia-traces \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill