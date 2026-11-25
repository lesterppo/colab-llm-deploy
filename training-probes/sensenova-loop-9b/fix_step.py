#!/usr/bin/env python3
"""SenseNova retry fix: reload NF4, LoRA, one training step with use_cache=False
passed explicitly (the custom sensenova_loop forward reads use_cache from its
own inner config, so the top-level config flag never reached it).
Writes /content/sensenova_step_fix.json. Fast: load + LoRA + step only."""
import json, time, traceback
import torch

OUT = "/content/sensenova_step_fix.json"
R = {"model": "sensenova/SenseNova-Loop-9B", "fix": "use_cache=False kwarg + nested config flags"}

def flush():
    json.dump(R, open(OUT, "w"), indent=2, default=str)

def vram():
    return {"alloc_gb": round(torch.cuda.memory_allocated()/2**30, 3),
            "reserved_gb": round(torch.cuda.memory_reserved()/2**30, 3),
            "max_alloc_gb": round(torch.cuda.max_memory_allocated()/2**30, 3)}

t0 = time.time()
from transformers import AutoTokenizer, BitsAndBytesConfig
import transformers
tok = AutoTokenizer.from_pretrained("sensenova/SenseNova-Loop-9B", trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
torch.cuda.reset_peak_memory_stats()
model = transformers.AutoModelForImageTextToText.from_pretrained(
    "sensenova/SenseNova-Loop-9B", quantization_config=bnb, device_map="auto",
    torch_dtype=torch.bfloat16, trust_remote_code=True)
R["load"] = {"secs": round(time.time()-t0, 1), "vram_after_load": vram()}
print("LOAD OK:", R["load"], flush=True); flush()

# set use_cache=False on EVERY config object we can reach
def kill_cache(cfg, path=""):
    try:
        if hasattr(cfg, "use_cache"):
            cfg.use_cache = False
    except Exception:
        pass
    for attr in ("text_config", "vision_config", "language_model", "model"):
        try:
            sub = getattr(cfg, attr, None)
            if sub is not None and sub is not cfg:
                kill_cache(sub, path + "." + attr)
        except Exception:
            pass
kill_cache(model.config)
try:
    model.generation_config.use_cache = False
except Exception:
    pass
for m in model.modules():
    try:
        if hasattr(m, "config"):
            kill_cache(m.config)
    except Exception:
        pass
try:
    model.config.use_cache = False
except Exception:
    pass
print("use_cache flags set", flush=True)

from peft import LoraConfig, get_peft_model, TaskType
targets = [c for c in ["q_proj", "k_proj", "v_proj", "o_proj"]
           if any(c in n for n, _ in model.named_modules())]
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                 target_modules=targets, task_type=TaskType.CAUSAL_LM)
peft_model = get_peft_model(model, cfg)
n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
R["lora"] = {"targets": targets, "trainable_params_m": round(n_train/1e6, 2),
             "vram_after_lora": vram()}
print("LORA OK:", R["lora"], flush=True); flush()

peft_model.gradient_checkpointing_enable()
peft_model.enable_input_require_grads()
peft_model.train()
import bitsandbytes as bnb_mod
opt = bnb_mod.optim.PagedAdamW8bit(
    (p for p in peft_model.parameters() if p.requires_grad), lr=2e-4)
text = ("The quick brown fox jumps over the lazy dog. " * 26).strip()
ids = tok(text, return_tensors="pt").input_ids[:, :256].to("cuda")
labels = ids.clone()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
try:
    out = peft_model(input_ids=ids, labels=labels, use_cache=False)
except TypeError as e:
    print("use_cache kwarg rejected, retrying without:", e, flush=True)
    out = peft_model(input_ids=ids, labels=labels)
loss = out.loss
loss.backward()
trainable = [p for p in peft_model.parameters() if p.requires_grad]
total_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).item())
gn = None
for n, p in peft_model.named_parameters():
    if p.requires_grad and p.grad is not None:
        gn = float(p.grad.norm().item()); R["grad_check_param"] = n; break
opt.step(); opt.zero_grad()
R["step"] = {"secs": round(time.time()-t0, 1), "loss": float(loss.item()),
             "loss_finite": bool(torch.isfinite(loss).item()),
             "grad_norm_first_lora_param": gn,
             "total_grad_norm_clipped": total_norm,
             "grads_flow": gn is not None and gn > 0,
             "peak_vram": vram(), "batch": 1, "seq": int(ids.shape[1])}
R["verdict"] = "GREEN" if (R["step"]["loss_finite"] and total_norm > 0) else "PARTIAL"
print("STEP:", R["step"], flush=True)
flush()
print("FIX PROBE DONE:", R["verdict"], flush=True)
