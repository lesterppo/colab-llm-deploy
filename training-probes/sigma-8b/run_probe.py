#!/usr/bin/env python3
"""VM-side wrapper for the Sigma 8B-instruct probe (REVISED per the 3B finding).

3B finding (measured 2026-11-25): the model is INCOHERENT under Colab's
transformers 5.22.1 and needs transformers 4.55.x + the schedule fix
(config.schedule_parameterization='projected_bernstein', the checkpoint's
own lineage) to produce coherent text. This wrapper pins transformers
4.55.x first, then runs the full battery with --schedule-fix.
Attempt 1: bf16 load (EXPECTED TO OOM: ~17.0 GB weights vs 14.56 GiB T4).
If the LOAD phase fails, attempt 2: NF4 load.
"""
import json, os, subprocess, sys

MODEL = "nvidia/Nemotron-Labs-Sigma-8B-instruct"
SLUG = "sigma-8b"

def run(cmd, log):
    log.write(f"\n$ {' '.join(cmd)}\n"); log.flush()
    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    log.write(f"\nEXIT={r.returncode}\n"); log.flush()
    return r.returncode

def load_failed(path):
    if not os.path.exists(path):
        return True
    try:
        d = json.load(open(path))
        return d.get("phases", {}).get("load", {}).get("status") != "ok"
    except Exception:
        return True

with open("/content/probe_run.log", "w") as log:
    rc = run([sys.executable, "-m", "pip", "install", "-q",
              "transformers>=4.55,<4.56"], log)
    log.write(f"pip transformers-4.55 rc={rc}\n"); log.flush()
    rc = run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes==0.50.2"], log)
    log.write(f"pip bitsandbytes rc={rc}\n"); log.flush()
    rc = run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], log)
    log.write(f"pip uninstall torchao rc={rc}\n"); log.flush()
    base = [sys.executable, "/content/sigma_probe.py", "--model", MODEL, "--slug", SLUG,
            "--schedule-fix"]
    out1 = f"/content/{SLUG}_results_bf16.json"
    rc = run(base + ["--load", "bf16", "--out", out1], log)
    if load_failed(out1):
        log.write("bf16 LOAD failed, retrying with NF4\n"); log.flush()
        out2 = f"/content/{SLUG}_results_nf4.json"
        rc = run(base + ["--load", "nf4", "--out", out2], log)
    print(f"probe done rc={rc}", flush=True)
