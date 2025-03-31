export UV_LINK_MODE=copy 
uv venv /fsx/baptiste_colle/open-r1-venv --python 3.10 
uv pip install -r baptiste/requirements.txt


uv venv /fsx/baptiste_colle/open-r1-test-venv --python 3.10 

sbatch slurm/agentic_generation.slurm

sbatch slurm/test_generate.slurm


squeue -u $USER

scontrol show job 15678390


sbatch slurm/serve_r1.slurm -m "/fsx/deepseek-r1-checkpoint" -e "sglang124"


curl http://10.53.86.164:39876/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "What is the capital of France?"}], "max_tokens": 32}'


sbatch slurm/serve_router.slurm

http://10.53.95.152:39876/v1/chat/completions


ROUTER_ADDRESS="10.53.86.164:39876"
FIRST_NODE_IP="26.0.174.186"
SERVER_PORT="39877"

curl -X POST "http://${ROUTER_ADDRESS}/add_worker?url=http://${FIRST_NODE_IP}:${SERVER_PORT}"




sbatch slurm/serve_router.slurm
sbatch slurm/serve_r1.slurm # do not forget to add the router address

sbatch slurm/agentic_generation.slurm


cp codeforces_agentic_generations.jsonl codeforces_agentic_generations_backup_$(date +%Y%m%d_%H%M%S).jsonl
 