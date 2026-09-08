#!/bin/bash
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/data/scratch/cache/hf
export UV_CACHE_DIR=/data/scratch/cache/uv
WORK=/data/fast/work/phonellm-quant
cd "$HOME/projects/phonellm-quant"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "=== Step 0: setting up two isolated venvs (quant, serve) ==="
rm -rf .venv-quant .venv-serve
uv venv --python 3.12 .venv-quant >> "$WORK/00_env.log" 2>&1
uv venv --python 3.12 .venv-serve >> "$WORK/00_env.log" 2>&1
source .venv-quant/bin/activate
uv pip install --upgrade pip >> "$WORK/00_env.log" 2>&1
uv pip install transformers accelerate "huggingface_hub[cli]" datasets llmcompressor >> "$WORK/00_env.log" 2>&1
RC=$?
deactivate
if [ $RC -ne 0 ]; then
  log "QUANT VENV INSTALL FAILED rc=$RC"; echo "PIPELINE_DONE status=1 stage=install_quant"; exit 1
fi
source .venv-serve/bin/activate
uv pip install --upgrade pip >> "$WORK/00_env.log" 2>&1
uv pip install vllm >> "$WORK/00_env.log" 2>&1
RC=$?
deactivate
if [ $RC -ne 0 ]; then
  log "SERVE VENV INSTALL FAILED rc=$RC"; echo "PIPELINE_DONE status=1 stage=install_serve"; exit 1
fi
log "Both venvs installed OK."

source .venv-quant/bin/activate

log "=== Step 2: downloading BF16 weights ==="
hf download pipecat-ai/phonellm-alpha-1 --local-dir /data/fast/checkpoints/phonellm-alpha-1-bf16 > "$WORK/02_download.log" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
  log "DOWNLOAD FAILED rc=$RC"; echo "PIPELINE_DONE status=1 stage=download"; exit 1
fi
log "Download OK. Size: $(du -sh /data/fast/checkpoints/phonellm-alpha-1-bf16 | cut -f1)"

log "=== Step 3: inspecting modules to build ignore list ==="
python3 scripts/inspect_modules.py > "$WORK/03_inspect.log" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
  log "INSPECT FAILED rc=$RC"; echo "PIPELINE_DONE status=1 stage=inspect"; exit 1
fi
log "Inspect OK."

log "=== Step 4: quantizing (GPUs 1,2 only, .venv-quant) ==="
CUDA_VISIBLE_DEVICES=1,2 python3 scripts/quantize.py > "$WORK/04_quantize.log" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
  log "QUANTIZE FAILED rc=$RC"; echo "PIPELINE_DONE status=1 stage=quantize"; exit 1
fi
log "Quantize OK. Output size: $(du -sh /data/fast/checkpoints/phonellm-alpha-1-int8 2>/dev/null | cut -f1)"
deactivate

source .venv-serve/bin/activate
log "=== Step 5: validating with vLLM (GPUs 1,2 only, .venv-serve) ==="
CUDA_VISIBLE_DEVICES=1,2 python3 scripts/validate_vllm.py > "$WORK/05_validate.log" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
  log "VALIDATE FAILED rc=$RC"; echo "PIPELINE_DONE status=1 stage=validate"; exit 1
fi
log "Validate OK."
deactivate

log "=== Step 6: post-validation tiering ==="
mkdir -p /data/warm/models
mv /data/fast/checkpoints/phonellm-alpha-1-bf16 /data/warm/models/phonellm-alpha-1-bf16
log "Moved BF16 source to /data/warm/models/."

log "=== ALL DONE ==="
echo "PIPELINE_DONE status=0 stage=complete"
