#!/usr/bin/env python3
"""VM-side wrapper for Qwen3.5-4B probe. Single attempt expected to work
(Colab ships transformers 5.22.1, Qwen3.5 is native there); logs everything."""
import subprocess, sys

MODEL = "Qwen/Qwen3.5-4B"
SLUG = "qwen3.5-4b"
# DeltaNet layers: in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj (+conv1d);
# full-attn layers: q/k/v/o_proj; MLP: gate/up/down_proj.
EXTRA = "gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,conv1d"

def run(cmd, log):
    log.write(f"\n$ {' '.join(cmd)}\n"); log.flush()
    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    log.write(f"\nEXIT={r.returncode}\n"); log.flush()
    return r.returncode

with open("/content/probe_run.log", "w") as log:
    rc = run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes==0.50.2"], log)
    log.write(f"pip bitsandbytes rc={rc}\n"); log.flush()
    rc = run([sys.executable, "/content/probe2.py", "--model", MODEL, "--slug", SLUG,
              "--extra-lora-targets", EXTRA,
              "--out", f"/content/{SLUG}_results.json"], log)
    print(f"probe done rc={rc}", flush=True)
