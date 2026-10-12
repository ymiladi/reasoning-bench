#!/usr/bin/env bash
# Start both vLLM servers on one 8-GPU node:
#   GPUs 0-3 -> K2 Think V2   (73B, bf16, tensor-parallel 4)       port 8002
#   GPUs 4-7 -> Falcon-H1R-7B (data-parallel 4 replicas, faster)    port 8001
# On a 4-GPU node: run K2 alone first, then Falcon (set the GPU lists below accordingly).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs

K2_GPUS=${K2_GPUS:-0,1,2,3}
FALCON_GPUS=${FALCON_GPUS:-4,5,6,7}
MAX_LEN=${MAX_LEN:-72000}             # 64k output budget + prompt headroom

CUDA_VISIBLE_DEVICES=$K2_GPUS nohup vllm serve IFM/K2-Think-V2 \
  --port 8002 --dtype bfloat16 \
  --tensor-parallel-size "$(echo "$K2_GPUS" | tr ',' '\n' | wc -l)" \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization 0.92 \
  > logs/vllm_k2.log 2>&1 &
echo "K2 Think V2 starting on GPUs $K2_GPUS (log: logs/vllm_k2.log)"

CUDA_VISIBLE_DEVICES=$FALCON_GPUS nohup vllm serve tiiuae/Falcon-H1R-7B \
  --port 8001 --reasoning-parser deepseek_r1 \
  --data-parallel-size "$(echo "$FALCON_GPUS" | tr ',' '\n' | wc -l)" \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization 0.90 \
  > logs/vllm_falcon.log 2>&1 &
echo "Falcon-H1R-7B starting on GPUs $FALCON_GPUS (log: logs/vllm_falcon.log)"

for port in 8001 8002; do
  echo -n "waiting for :$port "
  until curl -sf "http://localhost:$port/v1/models" >/dev/null; do echo -n "."; sleep 15; done
  echo " up"
done
