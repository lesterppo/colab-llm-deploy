#!/usr/bin/env python3
"""Follow-up: can PEFT LoRA wrap LFM3's Conv1d kernels themselves?
The main probe showed in_proj/out_proj (the conv path's Linear projections)
take LoRA and get nonzero grads. The 13 Conv1d kernels are named `...conv.conv`
(suffix 'conv', class Conv1d) and were NOT targeted. Try target_modules=['conv'].
Writes /content/conv_test.json. Exits 0 if LoRA attaches AND a step runs.
"""
import json, time, traceback
import torch

OUT = "/content/conv_test.json"
R = {"model": "LiquidAI/LFM3-1.7B", "phases": {}}

def flush():
    with open(OUT, "w") as f:
        json.dump(R, f, indent=2, default=str)

def vram():
    return {"alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 3),
            "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 3),
            "max_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)}

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained("LiquidAI/LFM3-1.7B", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "LiquidAI/LFM3-1.7B", quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True)
    R["load"] = {"secs": round(time.time() - t0, 1), "vram": vram()}
    conv_names = [n for n, m in model.named_modules() if m.__class__.__name__ == "Conv1d"]
    R["conv_modules"] = {"count": len(conv_names), "sample": conv_names[:3]}
    print("LOAD OK, conv modules:", len(conv_names), flush=True)
    R["phases"]["load"] = {"status": "ok"}
    flush()
except Exception as e:
    R["phases"]["load"] = {"status": "FAILED", "error": f"{type(e).__name__}: {e}",
                           "traceback": traceback.format_exc()[-2000:]}
    flush()
    raise SystemExit(2)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                     target_modules=["conv.conv"], task_type=TaskType.CAUSAL_LM)
    m = get_peft_model(model, cfg)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    R["lora_conv_only"] = {"trainable_params": int(n_train), "vram": vram()}
    print("LORA(conv) OK, trainable:", n_train, flush=True)
    R["phases"]["lora"] = {"status": "ok"}
    flush()
except Exception as e:
    R["phases"]["lora"] = {"status": "FAILED", "error": f"{type(e).__name__}: {e}",
                           "traceback": traceback.format_exc()[-2000:]}
    flush()
    print("LORA(conv) FAILED", flush=True)
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
            grads[n] = round(float(p.grad.norm().item()), 6)
    opt.step(); opt.zero_grad()
    R["step"] = {"secs": round(time.time() - t0, 1), "loss": float(loss.item()),
                 "loss_finite": bool(torch.isfinite(loss).item()),
                 "nonzero_grad_params": len(grads), "grads_sample": dict(list(grads.items())[:5]),
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

R["verdict"] = "CONV-KERNEL LORA WORKS" if R["step"].get("nonzero_grad_params", 0) > 0 else "NO GRADS"
flush()
print("CONV TEST DONE:", R["verdict"], flush=True)
