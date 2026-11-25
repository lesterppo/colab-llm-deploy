#!/usr/bin/env python3
"""VM-side wrapper for MiniCPM5-2B probe. Single attempt expected to work
(standard LlamaForCausalLM, Apache 2.0); logs everything."""
import subprocess, sys

MODEL = "openbmb/MiniCPM5-2B"
SLUG = "minicpm5-2b"
EXTRA = "gate_proj,up_proj,down_proj"

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
