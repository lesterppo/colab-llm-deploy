#!/usr/bin/env python3
"""VM-side wrapper: run fix_step.py, log everything."""
import subprocess, sys

def run(cmd, log):
    log.write(f"\n$ {' '.join(cmd)}\n"); log.flush()
    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    log.write(f"\nEXIT={r.returncode}\n"); log.flush()
    return r.returncode

with open("/content/fix_run.log", "w") as log:
    rc = run([sys.executable, "/content/fix_step.py"], log)
    print(f"fix done rc={rc}", flush=True)
