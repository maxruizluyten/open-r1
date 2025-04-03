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

accelerate launch --config_file=recipes/SmolLM2-1.7B-Instruct/sft_agent/config.yaml src/open_r1/sft.py