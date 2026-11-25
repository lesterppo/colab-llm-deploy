#!/usr/bin/env python3
"""VM-side wrapper: install bitsandbytes, run probe.py, log everything."""
import subprocess, sys

MODEL = "169Pi/Alpie_seed"
SLUG = "alpie-seed-9b"

def run(cmd, log):
    log.write(f"\n$ {' '.join(cmd)}\n"); log.flush()
    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    log.write(f"\nEXIT={r.returncode}\n"); log.flush()
    return r.returncode

with open("/content/probe_run.log", "w") as log:
    rc = run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes==0.50.2"], log)
    log.write(f"pip bitsandbytes rc={rc}\n"); log.flush()
    rc = run([sys.executable, "/content/probe.py", "--model", MODEL, "--slug", SLUG,
              "--out", f"/content/{SLUG}_results.json"], log)
    print(f"probe done rc={rc}", flush=True)
