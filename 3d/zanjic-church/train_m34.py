"""Brush training, round 4 — full-length run.

Round 3 proved the pipeline end to end in 44s for 4,000 steps, but
the eval renders are undertrained: lighthouse tower/roof/lamp are
recognisable, the rest is smeared, mean val PSNR ~16.7 dB. Brush's
default is 30,000 steps, and 4,000 steps is only 13% of that, so run
the full 30,000 on the same session (est. ~6 min at round-3 speed),
exporting every 5,000 steps so an early reclamation still leaves a
usable intermediate PLY.
"""
import json, os, subprocess, sys, time

LOG = open("/content/brush_train_m34.log", "ab", buffering=0)
T0 = time.time()

def log(msg):
    line = f"[{time.time()-T0:8.1f}s] {msg}\n"
    LOG.write(line.encode()); print(line, end="", flush=True)

def status(**kw):
    d = {"elapsed_s": round(time.time() - T0, 1)}
    d.update(kw)
    with open("/content/brush_status_m34.json", "w") as f:
        json.dump(d, f, indent=2)

def run(cmd):
    log("$ " + " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log(p.stdout[-2000:] if p.stdout else "")
    return p

EXE = "/content/brush/brush-app-x86_64-unknown-linux-gnu/brush_app"
assert os.path.exists(EXE) and os.path.isdir("/content/dataset")
os.makedirs("/content/dataset_exports_m34", exist_ok=True)
status(stage="training", exe=EXE)
# M34 run (2026-11-05, Peter picked intermediate dose): --scale-loss-weight
# 3e-4, between 1e-4 (82,385 pts, 13.60 dB) and 1e-3 (26,447 pts, 16.96 dB
# GREEN but visibly blurry on the phone). Only change vs the clean run.
# Question: does the middle dose keep the fog cleared AND keep enough
# splats for sharper renders? Judged on counts + PSNR + visuals + the
# giant-splat fraction (max-scale > 0.5 was ~1% at 1e-3).
cmd = [EXE, "/content/dataset",
       "--total-steps", "30000",
       "--scale-loss-weight", "3e-4",
       "--export-every", "5000",
       "--eval-every", "5000",
       "--eval-save-to-disk",
       "--export-path", "dataset_exports_m34"]
log("command: " + " ".join(cmd))
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, cwd="/content")
last = time.time()
with open("/content/brush_train_out_m3434.log", "w") as out:
    for line in proc.stdout:
        out.write(line); out.flush()
        if time.time() - last > 60:
            last = time.time()
            run(["bash", "-c", "ls -lh /content/dataset_exports_m34/ 2>/dev/null; nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null"])
            status(stage="training", note="still running")
rc = proc.wait()
log(f"brush exited rc={rc}")
run(["bash", "-c", "ls -lh /content/dataset_exports_m34/ 2>/dev/null; tail -60 /content/brush_train_out_m3434.log"])
status(stage="done" if rc == 0 else "failed", rc=rc)
log("DONE" if rc == 0 else "FAILED")
sys.exit(rc)
