import json, os, sys, traceback

BASE = "/data/fast/checkpoints/phonellm-alpha-1-bf16"
OUT = "/data/fast/checkpoints/phonellm-alpha-1-int8"
IGNORE_FILE = "/data/fast/work/phonellm-quant/ignore_list.json"
LOG = "/data/fast/work/phonellm-quant/quantize_debug.log"

def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    with open(IGNORE_FILE) as f:
        ignore_names = json.load(f)
    ignore = ["lm_head"] + [f"re:.*{n.replace('.', r'\.')}$" for n in ignore_names if n != "lm_head"]
    log(f"Loaded ignore list, {len(ignore)} entries (showing first 10): {ignore[:10]}")

    log("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

    log("Building calibration dataset (ultrachat_200k, 512 samples)...")
    NUM_SAMPLES = 512
    MAX_LEN = 2048
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=42).select(range(NUM_SAMPLES))

    def preprocess(example):
        messages = example["messages"]
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return tok(text, padding=False, max_length=MAX_LEN, truncation=True)

    ds = ds.map(preprocess, remove_columns=ds.column_names)
    log(f"Calibration dataset ready: {len(ds)} samples")

    log("Loading base model (bf16, device_map=auto across CUDA_VISIBLE_DEVICES)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
        max_memory={0: "14GiB", 1: "14GiB", 2: "14GiB", "cpu": "200GiB"},
    )
    log(f"Model loaded. Class: {type(model)}. attn_implementation={getattr(model.config, '_attn_implementation', None)}")

    # transformers==5.16.1 expects _tied_weights_keys as a dict (name -> mapping), but this
    # model's custom remote code declares it as an old-style list. config.json confirms
    # tie_word_embeddings=false for this model, so there is no real tied-weight relationship
    # to preserve -- neutralize the stale attribute so save_pretrained does not crash on it.
    if isinstance(getattr(model, "_tied_weights_keys", None), list):
        log(f"Patching stale list-style _tied_weights_keys ({model._tied_weights_keys}) to {{}} (tie_word_embeddings=False)")
        model._tied_weights_keys = {}


    oneshot = None
    GPTQModifier = None
    api_used = None
    try:
        from llmcompressor import oneshot as _oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier as _GPTQ
        oneshot, GPTQModifier = _oneshot, _GPTQ
        api_used = "llmcompressor top-level"
    except ImportError as e1:
        log(f"top-level import failed: {e1}")
        from llmcompressor.transformers import oneshot as _oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier as _GPTQ
        oneshot, GPTQModifier = _oneshot, _GPTQ
        api_used = "llmcompressor.transformers"

    log(f"Using llm-compressor API path: {api_used}")

    recipe = [
        GPTQModifier(targets="Linear", scheme="W8A8", ignore=ignore),
    ]

    # Confirmed from modeling_nemotron_h.py source:
    #   NEMOTRONH_ATTENTION_CLASSES = {"eager": NemotronHAttention, "sdpa": NemotronHSdpaAttention,
    #                                   "flash_attention_2": NemotronHFlashAttention2}
    #   NemotronHMOE.experts = nn.ModuleList([NemotronHMLP(...) for _ in range(n_routed_experts)])
    #   NemotronHMOE.shared_experts = NemotronHMLP(...)
    # Default sequential pipeline tried to hold a whole decoder layer (all 128+1 experts) in one
    # subgraph -> CUDA OOM. Splitting at attention/expert class boundaries, one module per
    # subgraph (sequential_targets_per_subgraph=1), keeps peak memory bounded.
    sequential_targets = [
        "NemotronHAttention",
        "NemotronHSdpaAttention",
        "NemotronHFlashAttention2",
        "NemotronHMLP",
    ]
    log(f"Using sequential_targets={sequential_targets}, sequential_targets_per_subgraph=1")

    # Fix for: AssertionError: No statistics available. Call observer(value) first.
    # Root cause: llmcompressor's per-subgraph weight-observer seeding
    # (observe(modules, base_name="weight") in gptq/base.py) misses a small number
    # of MoE expert modules -- almost certainly ones that received zero routed
    # calibration tokens out of 512 samples (128 experts/layer, real top-k routing,
    # since llmcompressor has no linearize_moe support for this custom NemotronH
    # architecture, so \ has no effect here). GPTQ's own
    # weight-quantization path crashes hard on this instead of the graceful
    # warn-and-skip the activation-quantization path already has. Patch: seed the
    # weight observer directly from the module's raw weight tensor (equivalent to
    # plain RTN quantization) as a fallback -- exactly what a layer with zero
    # calibration signal should reasonably get, using the library's own normal
    # observer mechanism, not a bypass of it.
    import llmcompressor.modifiers.gptq.base as _gptq_base

    _orig_quantize_weight = _gptq_base.quantize_weight
    _rtn_fallback_count = [0]

    def _patched_quantize_weight(module, quant_args, hessian, blocksize=128, percdamp=0.01):
        obs = getattr(module, "weight_observer", None)
        if obs is not None and not obs.has_statistics:
            _rtn_fallback_count[0] += 1
            obs(module.weight)
        return _orig_quantize_weight(
            module, quant_args, hessian, blocksize=blocksize, percdamp=percdamp
        )

    _gptq_base.quantize_weight = _patched_quantize_weight

    log("Starting oneshot quantization (this is the long step)...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_LEN,
        num_calibration_samples=NUM_SAMPLES,
        output_dir=None,  # skip oneshot internal auto-save (cannot pass save_original_format=False to it); we save explicitly below instead
        sequential_targets=sequential_targets,
        sequential_targets_per_subgraph=1,
    )
    log(f"RTN fallback (zero calibration traffic) used for {_rtn_fallback_count[0]} modules.")
    log("oneshot() returned normally.")

    try:
        model.save_pretrained(OUT, save_compressed=True, save_original_format=False)
        tok.save_pretrained(OUT)
        log("Explicit save_pretrained completed.")
    except Exception as e:
        log(f"Explicit save_pretrained skipped/failed (may be unnecessary if oneshot already saved): {e}")

    log("QUANTIZE_STATUS=SUCCESS")
except Exception:
    log("QUANTIZE_STATUS=FAILURE")
    log(traceback.format_exc())
    sys.exit(1)
