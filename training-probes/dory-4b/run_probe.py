#!/usr/bin/env python3
"""VM-side wrapper for Dory 1.0 4B probe.
Native Dory support shipped in transformers 5.24.0 (2026-11-21); Colab ships
5.22.1, so attempt 1 is EXPECTED to fail fast on unrecognized architecture
(config download only, no 8GB weight download before the failure).
  1. native load on shipped transformers (records the failure)
  2. pip install transformers from git main, then native load again
Stops at the first attempt whose probe exits 0. Logs everything.
Default loop count is 2 (checkpoint default); probe records the full config.
"""
import subprocess, sys

MODEL = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
SLUG = "dory-4b"
EXTRA = "gate_proj,up_proj,down_proj"

def run(cmd, log):
    log.write(f"\n$ {' '.join(cmd)}\n"); log.flush()
    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    log.write(f"\nEXIT={r.returncode}\n"); log.flush()
    return r.returncode

with open("/content/probe_run.log", "w") as log:
    rc = run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes==0.50.2"], log)
    log.write(f"pip bitsandbytes rc={rc}\n"); log.flush()
    base = [sys.executable, "/content/probe2.py", "--model", MODEL, "--slug", SLUG,
            "--extra-lora-targets", EXTRA]
    rc = run(base + ["--out", f"/content/{SLUG}_results_a1.json"], log)
    if rc != 0:
        log.write("attempt 1 failed (expected: Dory needs transformers >= 5.24), upgrading to git main\n"); log.flush()
        rc2 = run([sys.executable, "-m", "pip", "install", "-q",
                   "git+https://github.com/huggingface/transformers"], log)
        log.write(f"pip transformers-main rc={rc2}\n"); log.flush()
        rc = run(base + ["--out", f"/content/{SLUG}_results_a2.json"], log)
    print(f"probe done rc={rc}", flush=True)
