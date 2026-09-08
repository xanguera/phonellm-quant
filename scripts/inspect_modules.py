import json

idx_path = "/data/fast/checkpoints/phonellm-alpha-1-bf16/model.safetensors.index.json"
with open(idx_path) as f:
    idx = json.load(f)

names = sorted(set(k.rsplit(".", 1)[0] for k in idx["weight_map"].keys() if k.endswith(".weight")))
with open("/data/fast/work/phonellm-quant/module_names.txt", "w") as f:
    f.write("\n".join(names))

ignore = set()
for n in names:
    base = n.rsplit(".", 1)[-1]
    if n == "lm_head" or n.endswith(".lm_head"):
        ignore.add(n)
    elif ".experts." in n or ".shared_experts." in n:
        continue  # MoE FFN weights — must be quantized, never ignore
    elif base in ("q_proj", "k_proj", "v_proj", "o_proj"):
        ignore.add(n)  # attention projections
    elif base in ("router", "gate"):
        ignore.add(n)  # MoE routing gate
    elif base in ("in_proj", "out_proj", "conv1d", "norm"):
        ignore.add(n)  # Mamba SSM mixer components

with open("/data/fast/work/phonellm-quant/ignore_list.json", "w") as f:
    json.dump(sorted(ignore), f, indent=2)

expert_leaks = [n for n in ignore if ".experts." in n or ".shared_experts." in n]
print(f"total_named_modules={len(names)} ignored={len(ignore)} expert_leaks={len(expert_leaks)}")
print("SAMPLE_IGNORED (first 40):")
for n in sorted(ignore)[:40]:
    print(" ", n)
