# phonellm-quant

INT8 quantization of [`pipecat-ai/phonellm-alpha-1`](https://huggingface.co/pipecat-ai/phonellm-alpha-1)
(Nemotron 3 Nano 30B-A3B, a hybrid Mamba-Transformer MoE) for GPUs with no native FP8/NVFP4 support
— built for 3× RTX 3090 (Ampere), but the recipe applies to any Ampere-class card. Includes the
resulting vLLM serving setup.

## Why

Neither official release fits Ampere hardware: BF16 is ~59GB (too tight for KV cache across
3×24GB cards) and the official NVFP4 build requires Blackwell tensor cores this hardware doesn't
have. This repo quantizes the BF16 checkpoint to INT8 instead, using `llm-compressor`, on the
officially-recommended serving stack (vLLM) rather than an unverified third-party GGUF conversion.

## What's quantized

- **INT8**: all MoE expert `up_proj`/`down_proj` (routed + shared experts) — the bulk of the model.
- **BF16 (protected)**: attention Q/K/V/O projections, Mamba `in_proj`/`out_proj`, `lm_head` —
  matching NVIDIA's own precedent for this exact architecture (their FP8 release keeps the same
  layers unquantized).
- **Untouched**: the MoE router/gate (not an `nn.Linear`, `targets="Linear"` never touches it).

**Quantization method actually used: RTN (round-to-nearest, weight-only), not calibrated GPTQ.**
This model's own MoE forward code (`self.moe()`) loops over all 128 experts inside one Python
function that PyTorch's FX tracer auto-wraps as a single opaque node — so GPTQ's hook-based
Hessian calibration structurally cannot observe per-expert activations for this architecture,
regardless of `sequential_targets` settings. `scripts/quantize.py` patches `quantize_weight()` to
detect modules with zero calibration signal and seed them from raw weights (equivalent to RTN)
instead of crashing or producing garbage from a degenerate Hessian. At INT8 (vs. INT4), RTN is a
reasonable, commonly-used method — validated empirically here (see `scripts/validate_*.py`), not
just assumed.

## Requirements

- Ampere-or-newer NVIDIA GPU(s) with enough combined VRAM for INT8 (~32GB weights) plus headroom
  for calibration — this was built and tested on 3× RTX 3090 (72GB total)
- CUDA toolkit matching your installed `torch` build's CUDA version (needed to compile
  `mamba-ssm`/`causal-conv1d` from source — see below)
- [`uv`](https://docs.astral.sh/uv/) for Python environment management
- ~100GB free disk (BF16 source ~59GB + INT8 output ~32GB, plus working space)

## Setup

Two **separate** venvs — `llm-compressor` and `vllm` pull in incompatible `torch` versions when
installed together:

```bash
uv venv --python 3.12 .venv-quant
source .venv-quant/bin/activate
uv pip install torch transformers accelerate "huggingface_hub[cli]" datasets llmcompressor
uv pip install mamba-ssm causal-conv1d   # needs a matching CUDA toolkit + nvcc; see gotchas below
deactivate

uv venv --python 3.12 .venv-serve
source .venv-serve/bin/activate
uv pip install torch vllm
deactivate
```

## Usage

```bash
# 1. Inspect the model's module structure and build the ignore list (protects attention/Mamba/router)
.venv-quant/bin/python scripts/inspect_modules.py

# 2. Quantize (long-running — expect a couple of hours; GPU count/paths are hardcoded, edit to taste)
.venv-quant/bin/python scripts/quantize.py

# 3. Validate before trusting the output
.venv-serve/bin/python scripts/validate_vllm.py       # coherence + basic tool-call check
.venv-serve/bin/python scripts/validate_saydo.py       # decisive two-turn say/do tool-call test

# 4. Serve
scripts/serve.sh     # -> http://localhost:8000/v1 (OpenAI-compatible)
scripts/stop.sh
```

`scripts/pipeline.sh` chains the download/inspect/quantize steps end-to-end for a from-scratch run.

## Gotchas (found the hard way — see below before you "fix" something that isn't broken)

- **Ignore-list regex escaping.** `inspect_modules.py` builds regex patterns like
  `re:.*module\.name$`. Using `r'\\.'` instead of `r'\.'` double-escapes the dot into "match a
  literal backslash then any character" — which matches nothing, so the intended protections
  silently vanish. Verify with `re.search()` against a real module name before trusting a rebuilt
  ignore list, and spot-check actual output tensor dtypes after quantizing — a "SUCCESS" status
  line does not mean the ignore list worked.
- **`sequential_targets` alone doesn't achieve true per-expert memory isolation** for this
  architecture, because the MoE forward loop is one un-traceable Python function (see above). The
  actual OOM fix is `max_memory` headroom (cap well below each GPU's physical limit so
  `device_map="auto"` offloads the rest to CPU RAM), not subgraph splitting.
- **`transformers`'s `_tied_weights_keys`**: this checkpoint's bundled code declares it as an
  old-style list; newer `transformers` expects a dict and crashes in `save_pretrained`. Since
  `config.json` has `tie_word_embeddings: false`, there's no real tied-weight relationship to
  preserve — patch `model._tied_weights_keys = {}` before saving.
- **`oneshot()`'s internal auto-save doesn't forward `save_original_format=False`** (or any extra
  save kwarg — there's a literal `# TODO` in the library for this). Set `output_dir=None` in the
  `oneshot()` call to skip its internal save and do the save explicitly yourself, where you do
  have full kwarg control.
- **vLLM needs the bundled `nano_v3_reasoning_parser.py` loaded explicitly** at serve time
  (`--reasoning-parser-plugin <path> --reasoning-parser nano_v3`, already in `serve.sh`) or the
  model's `<think>...</think>` reasoning leaks straight into the visible response — you'd see a
  literal `</think>` tag mid-sentence. `enable_thinking: false` alone is not sufficient.
- **vLLM's offline batch API and its served HTTP API are different code paths.** Validating one
  does not validate the other — the reasoning-parser bug above only showed up when testing the
  actual served endpoint, not `scripts/validate_vllm.py`'s offline `LLM.generate()` calls.

## Layout

```
scripts/
  inspect_modules.py       # build the ignore list from the model's safetensors index
  quantize.py               # the actual llm-compressor GPTQModifier W8A8 recipe
  validate_vllm.py          # coherence + basic tool-call smoke test (offline batch API)
  validate_saydo.py         # decisive two-turn say/do tool-call test (offline batch API)
  validate_vllm_bf16.py     # BF16 baseline comparison harness
  serve.sh / stop.sh        # persistent vLLM OpenAI-compatible server
  pipeline.sh               # end-to-end orchestration (download → inspect → quantize)
```

Model checkpoints (BF16 source and INT8 output) are **not** in this repo — they're data artifacts,
kept on fast local storage outside of git.
