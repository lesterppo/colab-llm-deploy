#!/usr/bin/env python3
"""Brush lighthouse: VM training (run detached on a Colab T4, after
setup.py). Faithful to the proven 2026-11-04 run (30,000 steps, 320.6 s
on a T4, rc 0, mean val PSNR 18.59 dB), with two robustness fixes vs
that run's driver:

* The dataset is GENERATED on the VM by build_scene.py (numpy + Pillow,
  both stock on Colab) instead of being uploaded as a tarball — one
  less host-side artifact, same pixels. The PNGs are saved
  uncompressed on purpose (see below).
* Status is polled from the brush process (not from its stdout lines):
  Brush's progress bars may not emit newlines for minutes, so a
  stdout-line loop would never refresh the status file.

Pipeline: build dataset -> train -> eval -> PSNR -> base64 package.
Writes /content/brush_train_status.json in the repo's status shape,
plus /content/brush_result.b64 (a base64 tar of the final PLY + eval
renders). The base64 step is deliberate: some Colab download paths
fail on raw binary files, while a plain-text .b64 file downloads
cleanly; the host script decodes and verifies it.

CRITICAL (Brush 0.3.0 bug, scene.rs get_image_data): the image-header
reader does read_exact over a 16,387-byte buffer, and tokio's
read_exact fails with "early eof" when the file is smaller than the
buffer. Flat-colour renders compress to ~14 KB, so compressed PNGs
kill the whole dataset load in ~1 s. build_scene.py therefore saves
every view uncompressed (~480 KB each, pixels identical). Do not
"optimise" this back to compressed PNGs.

Usage: python3 train.py [--steps 30000]  (steps override is for a
quick smoke test only; the proven recipe is 30000.)
"""
import argparse, base64, glob, json, math, os, subprocess, sys, tarfile, time
import numpy as np
from PIL import Image

STATUS = "/content/brush_train_status.json"
LOG = open("/content/brush_train.log", "ab", buffering=0)
T0 = time.time()


def log(msg):
    line = f"[{time.time() - T0:8.1f}s] {msg}\n"
    LOG.write(line.encode()); print(line, end="", flush=True)


def status(stage, ready=False, error="", **kw):
    d = {"stage": stage, "ready": ready, "error": error, "tail": "",
         "elapsed_s": round(time.time() - T0, 1)}
    d.update(kw)
    with open(STATUS, "w") as f:
        json.dump(d, f, indent=2)


def fatal(msg):
    log("FATAL: " + msg)
    status("fatal", ready=False, error=msg)
    sys.exit(2)


def run(cmd, timeout=1800):
    log("$ " + " ".join(str(c) for c in cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=timeout)
    log(p.stdout[-2000:] if p.stdout else "")
    return p


a = argparse.ArgumentParser()
a.add_argument("--steps", type=int, default=30000)
args = a.parse_args()
# Host-side step override travels through /content/brush_config.json
# (exec_detach runs this file with no argv, so a config file is the
# only race-free channel — same pattern as deploy_media's
# gen_config.json). Config wins over the default, argv wins over both.
if os.path.exists("/content/brush_config.json"):
    try:
        with open("/content/brush_config.json") as f:
            args.steps = int(json.load(f).get("steps", args.steps))
    except Exception as e:
        log(f"ignoring unreadable brush_config.json: {e}")

exe = glob.glob("/content/brush/**/brush_app", recursive=True)
if not exe:
    fatal("Brush binary missing — run setup.py first")
exe = exe[0]

# --- dataset ----------------------------------------------------------
if not os.path.isdir("/content/dataset"):
    status("build_dataset")
    p = run([sys.executable, "/content/build_scene.py"], timeout=900)
    if p.returncode != 0 or not os.path.isdir("/content/dataset"):
        fatal("build_scene.py failed to produce /content/dataset")
p = run(["bash", "-c",
         "ls /content/dataset/images | wc -l; "
         "ls -lh /content/dataset/images/train_000.png"])
small = [f for f in glob.glob("/content/dataset/images/*.png")
         if os.path.getsize(f) <= 16387]
if small:
    fatal(f"{len(small)} dataset PNGs are <= 16387 bytes — Brush 0.3.0 "
          "will fail with 'early eof' (see module docstring)")

# --- train ------------------------------------------------------------
os.makedirs("/content/dataset_exports", exist_ok=True)
status("training", steps=args.steps)
export_every = 5000 if args.steps >= 5000 else args.steps
cmd = [exe, "/content/dataset",
       "--total-steps", str(args.steps),
       "--export-every", str(export_every),
       "--eval-every", str(export_every),
       "--eval-save-to-disk",
       "--export-path", "dataset_exports"]
log("command: " + " ".join(cmd))
with open("/content/brush_train_out.log", "w") as out:
    proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                            cwd="/content")
    while proc.poll() is None:
        time.sleep(10)
        exports = sorted(os.path.basename(p) for p in
                         glob.glob("/content/dataset_exports/export_*.ply"))
        status("training", steps=args.steps, exports=exports)
rc = proc.returncode
log(f"brush exited rc={rc}")
if rc != 0:
    run(["bash", "-c", "tail -60 /content/brush_train_out.log"])
    fatal(f"brush exited rc={rc}; see /content/brush_train_out.log")
# Export naming pads to the width of --total-steps (export_30000.ply
# at 30000 steps, export_1000.ply at 4000 steps), so never guess the
# name: take the newest export by mtime.
cands = glob.glob("/content/dataset_exports/export_*.ply")
if not cands:
    fatal("no export PLY produced")
final_ply = max(cands, key=os.path.getmtime)

# --- PSNR + package ---------------------------------------------------
status("package", ply=final_ply)
eval_dir = f"/content/dataset_exports/eval_{args.steps}"
psnrs = {}
if os.path.isdir(eval_dir):
    for fn in sorted(os.listdir(eval_dir)):
        gt_path = f"/content/dataset/images/{fn}"
        if not os.path.exists(gt_path):
            continue
        gt = np.array(Image.open(gt_path).convert("RGB"), dtype=np.float64)
        ev = np.array(Image.open(os.path.join(eval_dir, fn)).convert("RGB"),
                      dtype=np.float64)
        if gt.shape != ev.shape:
            continue
        mse = float(np.mean((gt - ev) ** 2))
        psnrs[fn] = round(10 * math.log10(255 ** 2 / mse), 2) if mse > 0 else 99.9
mean_psnr = round(sum(psnrs.values()) / len(psnrs), 2) if psnrs else None
log(f"mean val PSNR: {mean_psnr} dB over {len(psnrs)} views: {psnrs}")

with tarfile.open("/content/brush_result.tar.gz", "w:gz") as tf:
    tf.add(final_ply, arcname=os.path.basename(final_ply))
    if os.path.isdir(eval_dir):
        for fn in sorted(os.listdir(eval_dir)):
            tf.add(os.path.join(eval_dir, fn), arcname=f"eval/{fn}")
with open("/content/brush_result.tar.gz", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
with open("/content/brush_result.b64", "w") as f:
    f.write(b64)
log(f"packaged {os.path.getsize('/content/brush_result.tar.gz')} bytes "
    f"-> {len(b64)} base64 chars")

status("ready", ready=True, ply=final_ply, mean_psnr_db=mean_psnr,
       psnr_db=psnrs, b64="/content/brush_result.b64")
log("TRAIN DONE")
