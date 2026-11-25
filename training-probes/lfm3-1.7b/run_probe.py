#!/usr/bin/env python3
"""VM-side wrapper for LFM3-1.7B probe.
Attempts (weights are cached on the VM after attempt 1, so retries are cheap):
  1. native transformers load (LFM3 contributed to transformers 2026-11-05;
     Colab ships 5.22.1, which may or may not include it)
  2. trust_remote_code=True (repo's own modeling code)
  3. transformers from git main, then native load again
Stops at the first attempt whose probe exits 0. Logs everything.
"""
import subprocess, sys

MODEL = "LiquidAI/LFM3-1.7B"
SLUG = "lfm3-1.7b"
EXTRA = "gate_proj,up_proj,down_proj,in_proj,out_proj,conv1d,conv2d"

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
        log.write("attempt 1 (native) failed, retrying with trust_remote_code\n"); log.flush()
        rc = run(base + ["--trust-remote-code", "--out", f"/content/{SLUG}_results_a2.json"], log)
    if rc != 0:
        log.write("attempt 2 (trust_remote_code) failed, upgrading transformers to git main\n"); log.flush()
        rc2 = run([sys.executable, "-m", "pip", "install", "-q",
                   "git+https://github.com/huggingface/transformers"], log)
        log.write(f"pip transformers-main rc={rc2}\n"); log.flush()
        rc = run(base + ["--out", f"/content/{SLUG}_results_a3.json"], log)
    print(f"probe done rc={rc}", flush=True)
