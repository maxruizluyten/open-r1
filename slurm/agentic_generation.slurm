#!/bin/bash
#SBATCH --job-name=agentic-r1
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --output=./logs/%x_%j_%n.out
#SBATCH --error=./logs/%x_%j_%n.err
#SBATCH --time=7-00:00:00

set -exuo pipefail

source ~/venvs/gaia/bin/activate

python scripts/generate_agent_traces.py \
    --output-file "codeforces_agentic_generations.jsonl" \
    --prompt-column "prompt" \
    --uuid-column "contestId" \
    --api-addr "10.53.83.199" \
    --num-generations 5 \
    --max-tokens 32768 \
    --max-concurrent 32