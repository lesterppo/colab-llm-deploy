#!/usr/bin/env python3
"""VM-side wrapper: install bitsandbytes, run probe.py, log everything."""
import os, subprocess, sys

MODEL = "sensenova/SenseNova-Loop-9B"
SLUG = "sensenova-loop-9b"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def run(cmd, log):
    log.write(f"\n$ {' '.join(cmd)}\n"); log.flush()
    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    log.write(f"\nEXIT={r.returncode}\n"); log.flush()
    return r.returncode

with open("/content/probe_run.log", "w") as log:
    rc = run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes==0.50.2"], log)
    log.write(f"pip bitsandbytes rc={rc}\n"); log.flush()
    rc = run([sys.executable, "/content/probe.py", "--model", MODEL, "--slug", SLUG,
              "--trust-remote-code", "--multimodal",
              "--out", f"/content/{SLUG}_results.json"], log)
    print(f"probe done rc={rc}", flush=True)
