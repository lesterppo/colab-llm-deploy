#!/usr/bin/env python3
"""One-kick Brush lighthouse deployment (host side).

Proven deployment it reproduces: 2026-11-04, free Colab T4, Brush
0.3.0, 30,000 steps in 320.6 s, mean val PSNR 18.59 dB, 123,025-point
PLY. The VM scripts are the proven run's scripts, with two documented
deltas: the dataset is generated on the VM (build_scene.py) instead of
uploaded as a tarball, and train.py polls the brush process instead of
its stdout lines (Brush's progress bars emit no newlines for minutes).

Usage (from anywhere; paths resolve relative to this file):
  python3 deploy_brush.py [--session brush-lighthouse] [--gpu T4]
      [--steps 30000] [--out DIR] [--keep]
  # --steps 4000 is a ~1 min smoke test (mean PSNR ~16.7 dB, smeared
  # renders); the real recipe is the default 30000.

Flow: new session -> upload setup.py + train.py + build_scene.py ->
exec_detach setup -> poll brush_setup_status.json -> exec_detach
train -> poll brush_train_status.json -> download brush_result.b64
(a plain-text file, because some Colab download paths fail on raw
binary) -> decode, extract, and VERIFY the PLY locally (header,
point count, x/y/z read by name, no NaNs) -> stop the session unless
--keep. Output lands in --out (default ./outputs):
  <name>_trained.ply, eval/*.png (Brush's held-out val renders),
  brush_psnr.json.

Prerequisites: lesterppo/hermes-colab-cli installed and authenticated
(`python3 colab.py whoami` works); override its path with COLAB_PY.
Quota: free tier is ~3-8 GPU sessions/day/account, one GPU session at
a time. If `new` fails with 503 / outcome:2, the account's daily
quota is exhausted — switch account or wait, do not retry-loop.
"""
import argparse, base64, glob, json, os, subprocess, sys, tarfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
COLAB_PY = os.environ.get(
    "COLAB_PY", os.path.expanduser("~/workspace/hermes-colab-cli/colab.py"))


def sh(*args, timeout=300):
    return subprocess.run(list(args), capture_output=True, text=True,
                          timeout=timeout)


def colab(*args, timeout=300):
    r = sh(sys.executable, COLAB_PY, *args, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"colab.py {' '.join(args)} failed:\n"
                           f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout.strip()


def read_status(session, path, timeout=180):
    out = colab("exec", "-s", session, "--code",
                f"print(open({path!r}).read())", timeout=timeout)
    return json.loads(out[out.index("{"): ])


def wait_status(session, path, timeout_min, label):
    deadline = time.time() + timeout_min * 60
    status = {}
    while time.time() < deadline:
        time.sleep(60)
        try:
            status = read_status(session, path)
        except Exception:
            print(f"  {label}: status file not written yet")
            continue
        print(f"  {label}: stage={status.get('stage')} "
              f"ready={status.get('ready')}")
        if status.get("stage") == "fatal" or status.get("ready"):
            break
    else:
        raise SystemExit(f"Timed out waiting for {label}. Check logs.")
    if status.get("stage") == "fatal" or not status.get("ready"):
        raise SystemExit(f"{label} failed at stage {status.get('stage')}: "
                         f"{status.get('error')}")
    return status


def verify_ply(path):
    """Re-read a binary PLY the way a splat viewer does: properties by
    name, not by position (Brush writes them alphabetically, x/y/z
    last). Returns (points, bounds dict). Raises on any defect."""
    import numpy as np
    with open(path, "rb") as f:
        hdr = []
        while True:
            line = f.readline().decode(errors="ignore").strip()
            hdr.append(line)
            if line == "end_header":
                break
        if not hdr or hdr[0] != "ply":
            raise SystemExit(f"{path}: not a PLY file")
        n = int([x for x in hdr if x.startswith("element vertex")][0].split()[-1])
        props = [x.split()[-1] for x in hdr if x.startswith("property")]
        data = np.fromfile(f, dtype=np.float32)
    if len(data) != n * len(props):
        raise SystemExit(f"{path}: truncated PLY "
                         f"({len(data)} floats for {n}x{len(props)})")
    a = data.reshape(n, -1)
    idx = {p: i for i, p in enumerate(props)}
    for p in ("x", "y", "z"):
        if p not in idx:
            raise SystemExit(f"{path}: PLY has no {p} property")
    if bool(np.isnan(a).any()):
        raise SystemExit(f"{path}: PLY contains NaNs")
    bounds = {p: [round(float(a[:, idx[p]].min()), 2),
                  round(float(a[:, idx[p]].max()), 2)] for p in ("x", "y", "z")}
    return n, bounds


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--session", default="brush-lighthouse")
    p.add_argument("--gpu", default="T4")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--out")
    p.add_argument("--keep", action="store_true")
    a = p.parse_args()
    session, out_dir = a.session, os.path.abspath(a.out or os.path.join(HERE, "outputs"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/6] Creating session '{session}' on {a.gpu}...")
    colab("new", "-s", session, "--gpu", a.gpu, timeout=300)

    print("[2/6] Uploading setup + train + scene scripts...")
    for local, remote in (("setup.py", "/content/setup.py"),
                          ("train.py", "/content/train.py"),
                          ("build_scene.py", "/content/build_scene.py")):
        colab("upload", "-s", session, os.path.join(HERE, local), remote)

    print("[3/6] Running setup (Vulkan ICD + Brush 0.3.0, detached)...")
    colab("exec_detach", "-s", session, "-f", "/content/setup.py",
          "--log", "/content/setup.log")
    wait_status(session, "/content/brush_setup_status.json", 30, "setup")

    print("[4/6] Running training (detached)...")
    # Step count travels through brush_config.json (exec_detach runs the
    # file with no argv — same pattern as deploy_media's gen_config.json).
    cfg = json.dumps({"steps": a.steps})
    colab("exec", "-s", session, "--code",
          f"open('/content/brush_config.json','w').write({cfg!r})")
    colab("exec_detach", "-s", session, "-f", "/content/train.py",
          "--log", "/content/train.log")
    tstatus = wait_status(session, "/content/brush_train_status.json", 90, "train")

    print("[5/6] Downloading + verifying result...")
    b64_local = os.path.join(out_dir, "brush_result.b64")
    colab("download", "-s", session, "/content/brush_result.b64", b64_local,
          timeout=600)
    tar_local = os.path.join(out_dir, "brush_result.tar.gz")
    with open(tar_local, "wb") as f:
        f.write(base64.b64decode(open(b64_local).read()))
    with tarfile.open(tar_local) as tf:
        tf.extractall(out_dir)
    ply = glob.glob(os.path.join(out_dir, "export_*.ply"))
    if not ply:
        raise SystemExit("no PLY in downloaded result")
    ply = ply[0]
    n, bounds = verify_ply(ply)
    trained = os.path.join(out_dir, "lighthouse_trained.ply")
    os.replace(ply, trained)
    with open(os.path.join(out_dir, "brush_psnr.json"), "w") as f:
        json.dump({"mean_psnr_db": tstatus.get("mean_psnr_db"),
                   "psnr_db": tstatus.get("psnr_db"),
                   "points": n, "bounds": bounds}, f, indent=2)
    print(f"  verified: {trained} ({n} points, bounds {bounds})")
    print(f"  mean val PSNR: {tstatus.get('mean_psnr_db')} dB")

    if not a.keep:
        print("[6/6] Stopping session (billing halted)...")
        colab("stop", "-s", session, timeout=120)
    else:
        print(f"[6/6] Session '{session}' kept alive (--keep).")
    print(f"\nDONE. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
