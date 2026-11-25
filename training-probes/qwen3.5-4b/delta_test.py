#!/usr/bin/env python3
"""Follow-up: Qwen3.5-4B DeltaNet LoRA. The main probe's LoRA stage 1 (all
extras incl. conv1d) failed atomically on the torchao 0.10.0 ImportError when
PEFT tried to build a Conv1d LoRA layer, so the DeltaNet projections were
never attached. With torchao uninstalled first, retry the full target list
and report per-suffix grads after one step. Writes /content/delta_test.json.
"""
import json, time, traceback
import torch

OUT = "/content/delta_test.json"
R = {"model": "Qwen/Qwen3.5-4B", "phases": {}}

def flush():
    with open(OUT, "w") as f:
        json.dump(R, f, indent=2, default=str)

def vram():
    return {"alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 3),
            "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 3),
            "max_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)}

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-4B", quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16)
    R["load"] = {"secs": round(time.time() - t0, 1), "vram": vram()}
    names = [n for n, _ in model.named_modules()]
    R["conv1d_modules"] = [n for n in names if n.endswith("conv1d")][:4]
    R["conv1d_count"] = sum(1 for n in names if n.endswith("conv1d"))
    R["deltanet_suffixes"] = sorted({n.split(".")[-1] for n in names
                                     if any(s in n for s in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"))})
    print("LOAD OK", R["load"], "conv1d:", R["conv1d_count"], flush=True)
    R["phases"]["load"] = {"status": "ok"}
    flush()
except Exception as e:
    R["phases"]["load"] = {"status": "FAILED", "error": f"{type(e).__name__}: {e}",
                           "traceback": traceback.format_exc()[-2000:]}
    flush()
    raise SystemExit(2)

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
           "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]  # conv1d excluded: depthwise groups=8192 blocks PEFT LoRA (rank must divide groups); see delta_test_conv1d.json
try:
    from peft import LoraConfig, get_peft_model, TaskType
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                     target_modules=TARGETS, task_type=TaskType.CAUSAL_LM)
    m = get_peft_model(model, cfg)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    by_suffix = {}
    for n, p in m.named_parameters():
        if p.requires_grad:
            s = n.split(".")[-3]
            by_suffix[s] = by_suffix.get(s, 0) + p.numel()
    R["lora_full"] = {"trainable_params": int(n_train), "by_suffix": by_suffix, "vram": vram()}
    print("LORA(full) OK, trainable:", n_train, "by_suffix:", by_suffix, flush=True)
    R["phases"]["lora"] = {"status": "ok"}
    flush()
except Exception as e:
    R["phases"]["lora"] = {"status": "FAILED", "error": f"{type(e).__name__}: {e}",
                           "traceback": traceback.format_exc()[-2000:]}
    flush()
    print("LORA(full) FAILED:", e, flush=True)
    raise SystemExit(3)

try:
    m.gradient_checkpointing_enable()
    m.enable_input_require_grads()
    try:
        m.config.use_cache = False
    except Exception:
        pass
    m.train()
    opt = torch.optim.AdamW((p for p in m.parameters() if p.requires_grad), lr=2e-4)
    text = ("The quick brown fox jumps over the lazy dog. " * 12).strip()
    ids = tok(text, return_tensors="pt").input_ids[:, :120].to("cuda")
    labels = ids.clone()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = m(input_ids=ids, labels=labels, use_cache=False)
    loss = out.loss
    loss.backward()
    grads = {}
    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None and float(p.grad.norm()) > 0:
            key = n.split(".")[-3] + "/" + ("A" if "lora_A" in n else "B" if "lora_B" in n else "?")
            grads[key] = grads.get(key, 0) + 1
    opt.step(); opt.zero_grad()
    R["step"] = {"secs": round(time.time() - t0, 1), "loss": float(loss.item()),
                 "loss_finite": bool(torch.isfinite(loss).item()),
                 "nonzero_grad_params_by_suffix_AB": grads,
                 "peak_vram": vram()}
    R["phases"]["step"] = {"status": "ok"}
    print("STEP OK:", R["step"], flush=True)
    flush()
except Exception as e:
    R["phases"]["step"] = {"status": "FAILED", "error": f"{type(e).__name__}: {e}",
                           "traceback": traceback.format_exc()[-2000:]}
    flush()
    print("STEP FAILED", flush=True)
    raise SystemExit(4)

R["verdict"] = "DELTANET LORA WORKS" if R["step"].get("nonzero_grad_params_by_suffix_AB") else "NO GRADS"
flush()
print("DELTA TEST DONE:", R["verdict"], flush=True)
