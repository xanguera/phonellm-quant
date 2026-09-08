import json, sys, traceback

OUT = "/data/fast/checkpoints/phonellm-alpha-1-bf16"
RESULTS = "/data/fast/work/phonellm-quant/validation_results_bf16.json"
LOG = "/data/fast/work/phonellm-quant/validate_debug_bf16.log"

def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)

results = {"prompts": [], "status": "FAILURE"}

try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    log("Loading quantized model with vLLM (TP=2)...")
    tok = AutoTokenizer.from_pretrained(OUT, trust_remote_code=True)
    llm = LLM(model=OUT, tensor_parallel_size=2, trust_remote_code=True, dtype="auto", gpu_memory_utilization=0.90, cpu_offload_gb=24)
    sp = SamplingParams(temperature=0, max_tokens=300)

    tests = []

    # 1. Basic coherence
    msgs1 = [{"role": "user", "content": "In one or two sentences, what is the capital of France and what is it known for?"}]
    tests.append(("coherence", msgs1, None))

    # 2. Say/do tool-calling scenario
    tools = [{
        "type": "function",
        "function": {
            "name": "book_table",
            "description": "Book a restaurant table",
            "parameters": {
                "type": "object",
                "properties": {
                    "party_size": {"type": "integer"},
                    "time": {"type": "string"},
                },
                "required": ["party_size", "time"],
            },
        },
    }]
    msgs2 = [{"role": "user", "content": "Hi, can you book me a table for 4 people at 7pm tonight?"}]
    tests.append(("say_do_tool_call", msgs2, tools))

    # 3. Multi-turn coherence
    msgs3 = [
        {"role": "user", "content": "I need to reschedule my appointment."},
        {"role": "assistant", "content": "Sure, what's your current appointment date and what would you like to change it to?"},
        {"role": "user", "content": "It's currently Thursday at 2pm, can we move it to Friday at 10am?"},
    ]
    tests.append(("multiturn", msgs3, None))

    for name, msgs, tools_arg in tests:
        try:
            kwargs = dict(tokenize=False, add_generation_prompt=True)
            try:
                kwargs["enable_thinking"] = False
            except Exception:
                pass
            if tools_arg is not None:
                prompt = tok.apply_chat_template(msgs, tools=tools_arg, **kwargs)
            else:
                prompt = tok.apply_chat_template(msgs, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            if tools_arg is not None:
                prompt = tok.apply_chat_template(msgs, tools=tools_arg, **kwargs)
            else:
                prompt = tok.apply_chat_template(msgs, **kwargs)

        out = llm.generate([prompt], sp)
        text = out[0].outputs[0].text
        log(f"=== {name} ===\nPROMPT:\n{prompt}\nOUTPUT:\n{text}\n")
        results["prompts"].append({"name": name, "output": text})

    results["status"] = "SUCCESS"
except Exception:
    log("VALIDATE_STATUS=FAILURE")
    log(traceback.format_exc())
    results["status"] = "FAILURE"
    results["error"] = traceback.format_exc()

with open(RESULTS, "w") as f:
    json.dump(results, f, indent=2)

log(f"VALIDATE_STATUS={results['status']}")
if results["status"] != "SUCCESS":
    sys.exit(1)
