#!/usr/bin/env python3
"""Path B (fallback): Unsloth FastLanguageModel 4-bit load + LoRA + one step.
Parameterized version of the Inkling probe's probe_unsloth.py.
Usage: probe_unsloth2.py --model <HF id> --slug <slug> --out /content/<slug>_unsloth.json
Same measurements as Path A. Two-stage LoRA targets: full list, then q/k/v/o only.
Writes JSON incrementally; exits non-zero on failed phase.
NOTE: Unsloth downgrades transformers (requires <=5.17.0 in the 2026.11.11
release) and only supports a fixed set of architectures. For LFM3/Dory it is
expected to fail with 'not supported' — that failure IS the recorded result.
"""
import argparse, json, time, traceback
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--slug", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

OUT = a.out
R = {"path": "B (unsloth)", "model": a.model, "slug": a.slug, "phases": {}}

def flush():
    with open(OUT, "w") as f:
        json.dump(R, f, indent=2, default=str)

def ram():
    m = dict(l.split(":") for l in open("/proc/meminfo").read().splitlines())
    t = int(m["MemTotal"].split()[0]) / 2**20
    av = int(m["MemAvailable"].split()[0]) / 2**20
    return {"total_gb": round(t, 2), "used_gb": round(t - av, 2)}

def vram():
    return {"alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 3),
            "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 3),
            "max_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)}

def phase(name, fn):
    t0 = time.time()
    try:
        fn()
        R["phases"][name] = {"status": "ok", "secs": round(time.time() - t0, 1)}
    except Exception as e:
        R["phases"][name] = {"status": "FAILED", "secs": round(time.time() - t0, 1),
                             "error": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()[-2500:],
                             "ram": ram(), "vram": vram()}
        flush()
        raise
    flush()

import unsloth
R["unsloth_version"] = unsloth.__version__
try:
    import transformers
    R["transformers_version"] = transformers.__version__
except Exception as e:
    R["transformers_version"] = f"unknown ({e})"
R["ram_before"] = ram()
R["vram_before"] = vram()
flush()

def load():
    from unsloth import FastLanguageModel
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    model, tok = FastLanguageModel.from_pretrained(
        model_name=a.model,
        max_seq_length=512,
        load_in_4bit=True,
    )
    R["load"] = {"secs": round(time.time() - t0, 1), "vram": vram(), "ram": ram(),
                 "total_params_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 2)}
    print("LOAD OK", R["load"], flush=True)
    R["_model"] = model
    R["_tok"] = tok

try:
    phase("load", load)
except Exception:
    print("UNSLOTH LOAD FAILED", flush=True)
    raise SystemExit(2)

model = R.pop("_model"); tok = R.pop("_tok")

def lora():
    from unsloth import FastLanguageModel
    last_err = None
    for targets in (["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    ["q_proj", "k_proj", "v_proj", "o_proj"]):
        try:
            t0 = time.time()
            m = FastLanguageModel.get_peft_model(
                model, r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                target_modules=targets, use_gradient_checkpointing="unsloth",
            )
            R["lora"] = {"secs": round(time.time() - t0, 1), "targets_used": targets,
                         "trainable_params_m": round(sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6, 2),
                         "vram": vram()}
            print("LORA OK", R["lora"], flush=True)
            R["_peft"] = m
            return
        except Exception as e:
            last_err = f"{targets}: {type(e).__name__}: {e}"
            print("UNSLOTH LORA ATTEMPT FAILED:", last_err, flush=True)
            R.setdefault("lora_attempts_failed", []).append(last_err); flush()
    raise RuntimeError(f"all Unsloth LoRA attempts failed; last: {last_err}")

try:
    phase("lora", lora)
except Exception:
    print("UNSLOTH LORA FAILED", flush=True)
    raise SystemExit(3)

peft_model = R.pop("_peft")

def step():
    peft_model.train()
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW8bit(
            (p for p in peft_model.parameters() if p.requires_grad), lr=2e-4)
        R["optimizer"] = "paged_adamw_8bit"
    except Exception as e:
        opt = torch.optim.AdamW(
            (p for p in peft_model.parameters() if p.requires_grad), lr=2e-4, fused=True)
        R["optimizer"] = f"adamw_torch_fused (paged failed: {e})"
    text = ("The quick brown fox jumps over the lazy dog. " * 12).strip()
    ids = tok(text, return_tensors="pt").input_ids[:, :512].to("cuda")
    labels = ids.clone()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = peft_model(input_ids=ids, labels=labels)
    loss = out.loss
    loss.backward()
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    total_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).item())
    opt.step(); opt.zero_grad()
    R["step"] = {"secs": round(time.time() - t0, 1),
                 "loss": float(loss.item()),
                 "loss_finite": bool(torch.isfinite(loss).item()),
                 "total_grad_norm": total_norm,
                 "grads_flow": total_norm > 0,
                 "peak_vram": vram(), "ram": ram(),
                 "batch": 1, "seq": int(ids.shape[1])}
    print("STEP OK", R["step"], flush=True)

try:
    phase("step", step)
except Exception:
    print("UNSLOTH STEP FAILED", flush=True)
    raise SystemExit(4)

R["verdict"] = "GREEN (path B)"
flush()
print("PROBE COMPLETE — GREEN (path B)", flush=True)
