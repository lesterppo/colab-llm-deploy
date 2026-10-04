#!/usr/bin/env python3
"""One-shot image/video generation orchestrator (host side).

Drives lesterppo/hermes-colab-cli end to end:
  new -> upload setup+gen scripts+config -> exec_detach setup ->
  poll setup status -> exec_detach gen -> poll gen status -> download output.

Usage:
  deploy_media.py generate --kind image --model flux2-klein --prompt "..."
      [--negative "..."] [--seed 42] [--width 1024] [--height 1024]
      [--steps 4] [--frames 49] [--fps 24] [--clips 5]
      [--gpu T4] [--session NAME] [--out DIR] [--keep]
  deploy_media.py status   --session NAME
  deploy_media.py undeploy --session NAME

Models:
  image/flux2-klein   FLUX.2-klein-4B (Q4_K_M GGUF), 1024px, 4 steps, ~70s
  image/hidream-i1    HiDream-I1 17B (Q4_K_M GGUF), 1024px, 24 steps
  video/ltx23         LTX-2.3 22B distilled (Q3_K_M GGUF), 576x320 49f, ~7min
  video/ltx23-chain   LTX-2.3 chained clips (first-frame conditioning)

gen_config.json is written to the VM so gen scripts never guess the prompt.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
COLAB_PY = os.environ.get(
    "COLAB_PY", os.path.expanduser("~/workspace/hermes-colab-cli/colab.py"))

MEDIA_MODELS = {
    "flux2-klein": {
        "kind": "image", "dir": "image/flux2-klein",
        "setup_script": "setup.py", "gen_script": "gen.py",
        "extra": ["merge_qwen3.py"],
        "setup_status": "/content/comfy_setup_status.json",
        "gen_status": "/content/gen_status.json", "output_key": "png",
        "setup_timeout_min": 60, "gen_timeout_min": 30,
        "defaults": {"width": 1024, "height": 1024, "steps": 4,
                     "cfg": 4.0}},
    "hidream-i1": {
        "kind": "image", "dir": "image/hidream-i1",
        "setup_script": "setup.py", "gen_script": "gen.py",
        "extra": [],
        "setup_status": "/content/hidream_setup_status.json",
        "gen_status": "/content/gen_status.json", "output_key": "png",
        "setup_timeout_min": 90, "gen_timeout_min": 60,
        "defaults": {"width": 1024, "height": 1024, "steps": 24,
                     "cfg": 5.0}},
    "ltx23": {
        "kind": "video", "dir": "video/ltx23",
        "setup_script": "setup.py", "gen_script": "gen.py",
        "extra": [],
        "setup_status": "/content/ltx23_setup_status.json",
        "gen_status": "/content/ltx23_gen_status.json",
        "output_key": "output",
        "setup_timeout_min": 90, "gen_timeout_min": 90,
        "defaults": {"width": 576, "height": 320, "frames": 49,
                     "fps": 24, "steps": 8}},
    "ltx23-chain": {
        "kind": "video", "dir": "video/ltx23",
        "setup_script": "setup.py", "gen_script": "gen_chain.py",
        "extra": [],
        "setup_status": "/content/ltx23_setup_status.json",
        "gen_status": "/content/ltx23_chain_status.json",
        "output_key": "final",
        "setup_timeout_min": 90, "gen_timeout_min": 180,
        "defaults": {"width": 576, "height": 320, "frames": 49,
                     "fps": 24, "steps": 8, "clips": 5}},
}


def sh(*args, timeout=120):
    return subprocess.run(list(args), capture_output=True, text=True,
                          timeout=timeout)


def colab(*args, timeout=180):
    r = sh(sys.executable, COLAB_PY, *args, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"colab.py {' '.join(args)} failed:\n"
                           f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout.strip()


def read_status(session, path, timeout=120):
    out = colab("exec", "-s", session, "--code",
                f"print(open({path!r}).read())", timeout=timeout)
    return json.loads(out[out.index("{"):])


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
        stage = status.get("stage")
        ready = status.get("ready")
        print(f"  {label}: stage={stage} ready={ready}")
        if stage == "fatal" or ready:
            break
    else:
        raise SystemExit(f"Timed out waiting for {label}. Check logs.")
    if status.get("stage") == "fatal" or not status.get("ready"):
        raise SystemExit(f"{label} failed at stage {status.get('stage')}: "
                         f"{status.get('error')}\n"
                         f"tail: {status.get('tail', '')[-1500:]}")
    return status


def cmd_generate(a):
    if a.model not in MEDIA_MODELS:
        raise SystemExit(f"unknown model '{a.model}'. Choices: "
                         + ", ".join(sorted(MEDIA_MODELS)))
    spec = MEDIA_MODELS[a.model]
    model_dir = os.path.join(HERE, spec["dir"])
    session = a.session or f"media-{a.model}"
    out_dir = os.path.abspath(a.out or os.path.join(HERE, "outputs"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/6] Creating session '{session}' on {a.gpu}...")
    colab("new", "-s", session, "--gpu", a.gpu, timeout=300)

    print("[2/6] Uploading setup + gen scripts + config...")
    colab("upload", "-s", session,
          os.path.join(model_dir, spec["setup_script"]), "/content/setup.py")
    colab("upload", "-s", session,
          os.path.join(model_dir, spec["gen_script"]), "/content/gen.py")
    for extra in spec["extra"]:
        colab("upload", "-s", session, os.path.join(model_dir, extra),
              f"/content/{extra}")
    cfg = {"prompt": a.prompt}
    if a.negative:
        cfg["negative"] = a.negative
    cfg.update(spec["defaults"])
    if a.seed is not None:
        cfg["seed"] = a.seed
    if a.width:
        cfg["width"] = a.width
    if a.height:
        cfg["height"] = a.height
    if a.steps:
        cfg["steps"] = a.steps
    if a.frames:
        cfg["frames"] = a.frames
    if a.fps:
        cfg["fps"] = a.fps
    if a.clips:
        cfg["clips"] = a.clips
    cfg["out_name"] = f"{a.model.replace('-', '_')}_out"
    colab("exec", "-s", session, "--code",
          f"open('/content/gen_config.json','w').write({json.dumps(cfg)!r})")

    print("[3/6] Running setup (downloads, detached)...")
    colab("exec_detach", "-s", session, "-f", "/content/setup.py",
          "--log", "/content/setup.log")
    wait_status(session, spec["setup_status"], spec["setup_timeout_min"],
                "setup")

    print("[4/6] Running generation (detached)...")
    colab("exec_detach", "-s", session, "-f", "/content/gen.py",
          "--log", "/content/gen.log")
    status = wait_status(session, spec["gen_status"], spec["gen_timeout_min"],
                         "gen")

    print("[5/6] Downloading output...")
    remote = status.get(spec["output_key"])
    if not remote:
        raise SystemExit(f"No {spec['output_key']} in gen status: "
                         f"{json.dumps(status)[:500]}")
    ext = ".mp4" if spec["kind"] == "video" else ".png"
    local = os.path.join(out_dir, os.path.basename(remote))
    if not local.endswith(ext):
        local += ext
    colab("download", "-s", session, remote, local, timeout=600)
    print(f"  saved: {local} ({os.path.getsize(local)} bytes)")

    if not a.keep:
        print("[6/6] Stopping session (billing halted)...")
        colab("stop", "-s", session, timeout=120)
    else:
        print(f"[6/6] Session '{session}' kept alive (--keep).")
    print(f"\nDONE. Output: {local}")


def cmd_status(a):
    for path in ("/content/gen_status.json", "/content/ltx23_gen_status.json",
                 "/content/ltx23_chain_status.json",
                 "/content/comfy_setup_status.json",
                 "/content/hidream_setup_status.json",
                 "/content/ltx23_setup_status.json"):
        try:
            s = read_status(a.session, path)
            print(f"{path}: stage={s.get('stage')} ready={s.get('ready')}")
        except Exception:
            pass


def cmd_undeploy(a):
    print(f"Tearing down session '{a.session}'...")
    kill_code = (
        "import subprocess\n"
        "for pat in ('ComfyUI/main.py',):\n"
        "    subprocess.run(['pkill', '-f', pat])\n"
        "print('procs killed')"
    )
    colab("exec", "-s", a.session, "--code", kill_code, timeout=120)
    print(colab("stop", "-s", a.session, timeout=120))
    print("Session stopped; compute-unit billing halted.")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--kind", choices=["image", "video"], required=True)
    g.add_argument("--model", required=True,
                   help="flux2-klein | hidream-i1 | ltx23 | ltx23-chain")
    g.add_argument("--prompt", required=True)
    g.add_argument("--negative")
    g.add_argument("--seed", type=int)
    g.add_argument("--width", type=int)
    g.add_argument("--height", type=int)
    g.add_argument("--steps", type=int)
    g.add_argument("--frames", type=int)
    g.add_argument("--fps", type=int)
    g.add_argument("--clips", type=int)
    g.add_argument("--gpu", default="T4")
    g.add_argument("--session")
    g.add_argument("--out")
    g.add_argument("--keep", action="store_true")

    s = sub.add_parser("status")
    s.add_argument("--session", required=True)

    u = sub.add_parser("undeploy")
    u.add_argument("--session", required=True)

    a = p.parse_args()
    try:
        {"generate": cmd_generate, "status": cmd_status,
         "undeploy": cmd_undeploy}[a.cmd](a)
    except RuntimeError as e:
        raise SystemExit(f"ERROR: {e}")


if __name__ == "__main__":
    main()
