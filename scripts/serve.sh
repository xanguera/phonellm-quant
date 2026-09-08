#!/usr/bin/env bash
# Serve the INT8 PhoneLLM checkpoint via vLLM's OpenAI-compatible API.
#
# Consumers: aicompanion_pro (LLM_PROVIDER=vllm, VLLM_BASE_URL=http://localhost:8000/v1) or any
# other local client that speaks the OpenAI chat-completions API.
#
# Runs on GPUs 1+2 only — GPU 0 is reserved for aicompanion_pro's own ASR/TTS. Foreground script
# (Ctrl-C stops it), matching this project's own run.sh convention. Use stop.sh to kill a
# backgrounded instance.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv-serve/bin/activate
export HF_HOME=/data/scratch/cache/hf

PORT="${VLLM_PORT:-8000}"
MODEL_PATH="${VLLM_MODEL_PATH:-/data/fast/checkpoints/phonellm-alpha-1-int8}"
SERVED_NAME="${VLLM_SERVED_MODEL_NAME:-phonellm-alpha-1-int8}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

# The checkpoint bundles a custom reasoning parser (nano_v3_reasoning_parser.py) that correctly
# splits <think>...</think> out of the response instead of leaking it into the visible content —
# vLLM doesn't auto-discover this, it has to be loaded explicitly as a plugin.
REASONING_PARSER_PLUGIN="$MODEL_PATH/nano_v3_reasoning_parser.py"

echo "Serving $MODEL_PATH as '$SERVED_NAME' on :$PORT (GPUs 1,2, TP=2, max_model_len=$MAX_MODEL_LEN)"
exec env CUDA_VISIBLE_DEVICES=1,2 vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --trust-remote-code \
  --reasoning-parser-plugin "$REASONING_PARSER_PLUGIN" \
  --reasoning-parser nano_v3
