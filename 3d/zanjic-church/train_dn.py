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

LOG = open("/content/brush_train_dn.log", "ab", buffering=0)
T0 = time.time()

def log(msg):
    line = f"[{time.time()-T0:8.1f}s] {msg}\n"
    LOG.write(line.encode()); print(line, end="", flush=True)

def status(**kw):
    d = {"elapsed_s": round(time.time() - T0, 1)}
    d.update(kw)
    with open("/content/brush_status_dn.json", "w") as f:
        json.dump(d, f, indent=2)

def run(cmd):
    log("$ " + " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log(p.stdout[-2000:] if p.stdout else "")
    return p

EXE = "/content/brush/brush-app-x86_64-unknown-linux-gnu/brush_app"
assert os.path.exists(EXE) and os.path.isdir("/content/dataset")
os.makedirs("/content/dataset_exports_dn", exist_ok=True)
status(stage="training", exe=EXE)
# DN run (2026-11-05, Peter: densification tuning at 1e-3): clean scene,
# GREEN dose --scale-loss-weight 1e-3, PLUS pro-growth refine settings
# as one package (recorded exactly so a follow-up can ablate):
#   --growth-grad-threshold 0.00004 -> 0.00001 (4x easier growth trigger)
#   --growth-select-fraction 0.1 -> 0.3 (3x more flagged splats grow)
#   --growth-stop-iter 15000 -> 30000 (growth runs the full training,
#     instead of stopping halfway -- counts at 1e-3 froze at 15k)
# Question: does more growth restore density (>>26,447) and sharpness
# while scale loss still holds the fog down, or does it re-fog?
cmd = [EXE, "/content/dataset",
       "--total-steps", "30000",
       "--scale-loss-weight", "1e-3",
       "--growth-grad-threshold", "0.00001",
       "--growth-select-fraction", "0.3",
       "--growth-stop-iter", "30000",
       "--export-every", "5000",
       "--eval-every", "5000",
       "--eval-save-to-disk",
       "--export-path", "dataset_exports_dn"]
log("command: " + " ".join(cmd))
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, cwd="/content")
last = time.time()
with open("/content/brush_train_out_dnn.log", "w") as out:
    for line in proc.stdout:
        out.write(line); out.flush()
        if time.time() - last > 60:
            last = time.time()
            run(["bash", "-c", "ls -lh /content/dataset_exports_dn/ 2>/dev/null; nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null"])
            status(stage="training", note="still running")
rc = proc.wait()
log(f"brush exited rc={rc}")
run(["bash", "-c", "ls -lh /content/dataset_exports_dn/ 2>/dev/null; tail -60 /content/brush_train_out_dnn.log"])
status(stage="done" if rc == 0 else "failed", rc=rc)
log("DONE" if rc == 0 else "FAILED")
sys.exit(rc)
