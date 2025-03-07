#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:7 # Set to 7 to compare with TRL
#SBATCH --partition=hopper-prod
#SBATCH --output=/fsx/h4/logs/%x-%j.out
#SBATCH --err=/fsx/h4/logs/%x-%j.err
#SBATCH --requeue

# Specific configuration optimized for the Hugging Face Compute Cluster
# Be ye warned this may not work on other clusters!


# Specific configuration optimized for the Hugging Face Compute Cluster
# Be ye warned this may not work on other clusters!
module load cuda/12.4


set -x -e

source ~/.bashrc
source openr1/bin/activate
echo "START TIME: $(date)"

accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml --num_processes=7 scripts/remote_grpo.py \
    --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_deepscaler.yaml \
    --learning_rate 5.0e-7 \
    --hub_model_id open-r1/DeepSeek-R1-Distill-Qwen-1.5B-DeepScaler \
    --hub_model_revision v01.00 \
    --output_dir data/DeepSeek-R1-Distill-Qwen-1.5B-DeepScaler-v01.00 \
    --run_name deepscaler-remote-v01.00 \
    --wandb_entity huggingface \
    --wandb_project open-r1