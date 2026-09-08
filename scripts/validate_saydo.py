import json, sys, traceback

OUT = "/data/fast/checkpoints/phonellm-alpha-1-int8"
RESULTS = "/data/fast/work/phonellm-quant/validation_saydo_results.json"
LOG = "/data/fast/work/phonellm-quant/validate_saydo.log"

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
    llm = LLM(model=OUT, tensor_parallel_size=2, trust_remote_code=True, dtype="auto", gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0, max_tokens=300)

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

    tests = []

    # Test A: user has already confirmed everything explicitly up front -- no ambiguity, no missing params
    msgs_a = [{"role": "user", "content": "Book me a table for 4 people at 7pm tonight. Please go ahead and confirm it now, no need to ask me anything else."}]
    tests.append(("explicit_no_confirm_needed", msgs_a))

    # Test B: two-turn -- model asked to confirm (from the earlier real test), user now confirms
    msgs_b = [
        {"role": "user", "content": "Hi, can you book me a table for 4 people at 7pm tonight?"},
        {"role": "assistant", "content": "I can help with that. Just to confirm, you would like a table for 4 people at 7:00 PM tonight?"},
        {"role": "user", "content": "Yes, that is correct, please go ahead and book it."},
    ]
    tests.append(("post_confirmation_second_turn", msgs_b))

    for name, msgs in tests:
        kwargs = dict(tokenize=False, add_generation_prompt=True, tools=tools)
        try:
            prompt = tok.apply_chat_template(msgs, enable_thinking=False, **kwargs)
        except TypeError:
            prompt = tok.apply_chat_template(msgs, **kwargs)

        out = llm.generate([prompt], sp)
        text = out[0].outputs[0].text
        has_tool_call = "<tool_call>" in text and "<function=" in text
        log(f"=== {name} ===\nPROMPT:\n{prompt}\nOUTPUT:\n{text}\nHAS_TOOL_CALL={has_tool_call}\n")
        results["prompts"].append({"name": name, "output": text, "has_tool_call": has_tool_call})

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
