#!/usr/bin/env python3
"""One-shot image/video/music generation orchestrator (host side).

Drives lesterppo/hermes-colab-cli end to end:
  new -> upload setup+gen scripts+config -> exec_detach setup ->
  poll setup status -> exec_detach gen -> poll gen status -> download output.

Usage:
  deploy_media.py generate --kind image --model flux2-klein --prompt "..."
      [--negative "..."] [--seed 42] [--width 1024] [--height 1024]
      [--steps 4] [--out DIR] [--keep]
  deploy_media.py generate --kind video --model ltx23 --prompt "..."
      [--frames 49] [--fps 24] [--steps 8] [--out DIR] [--keep]
  deploy_media.py generate --kind music --model yue2-3b
      --lyrics "[Verse] ..." --style "..." [--max-seconds 40]
      [--out DIR] [--keep]
  deploy_media.py status   --session NAME
  deploy_media.py undeploy --session NAME

Models:
  image/flux2-klein   FLUX.2-klein-4B (Q4_K_M GGUF), 1024px, 4 steps, ~70s
  image/hidream-i1    HiDream-I1 17B (Q4_K_M GGUF), 1024px, 24 steps
  video/ltx23         LTX-2.3 22B distilled (Q3_K_M GGUF), 576x320 49f, ~7min
  video/ltx23-chain   LTX-2.3 chained clips (first-frame conditioning)
  music/yue2-3b       YuE2-3B lyrics-to-song, ~40s song in ~250s
  music/minimax-music3  MiniMax Music 3 INT8 (11.9GB), ~30s song in ~300s

gen_config.json is written to the VM so gen scripts never guess the prompt.
For music, lyrics/style/caption go through the same config file (never
pasted through exec --code — multi-line lyrics break there).
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
    "yue2-3b": {
        "kind": "music", "dir": "music/yue2-3b",
        "setup_script": "setup.py", "gen_script": "gen.py",
        "extra": [],
        "setup_status": "/content/yue2_setup_status.json",
        "gen_status": "/content/yue2_gen_status.json",
        "output_key": "audio",
        "setup_timeout_min": 90, "gen_timeout_min": 90,
        "defaults": {"max_seconds": 40}},
    "minimax-music3": {
        "kind": "music", "dir": "music/minimax-music3",
        "setup_script": "setup.py", "gen_script": "gen.py",
        "extra": [],
        "setup_status": "/content/music3_setup_status.json",
        "gen_status": "/content/music3_gen_status.json",
        "output_key": "audio",
        "setup_timeout_min": 90, "gen_timeout_min": 90,
        "defaults": {"duration": 30}},
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
    cfg = {}
    if spec["kind"] == "music":
        # lyrics/style travel through gen_config.json — never exec --code
        # (multi-line lyrics break shell quoting).
        if a.lyrics:
            cfg["lyrics"] = a.lyrics
        if a.style:
            cfg["style"] = a.style
        if a.caption:
            cfg["caption"] = a.caption
    else:
        if not a.prompt:
            raise SystemExit("--prompt is required for image/video")
        cfg["prompt"] = a.prompt
        if a.negative:
            cfg["negative"] = a.negative
        if a.width:
            cfg["width"] = a.width
        if a.height:
            cfg["height"] = a.height
        if a.frames:
            cfg["frames"] = a.frames
        if a.fps:
            cfg["fps"] = a.fps
        if a.clips:
            cfg["clips"] = a.clips
    cfg.update(spec["defaults"])
    # CLI values override recipe defaults.
    if a.duration:
        cfg["duration"] = a.duration
    if a.max_seconds:
        cfg["max_seconds"] = a.max_seconds
    if a.steps:
        cfg["steps"] = a.steps
    if a.seed is not None:
        cfg["seed"] = a.seed
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
    if spec["kind"] == "music":
        # gen status carries the real audio path with its extension
        local = os.path.join(out_dir, os.path.basename(remote))
    else:
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
                 "/content/ltx23_setup_status.json",
                 "/content/yue2_setup_status.json",
                 "/content/yue2_gen_status.json",
                 "/content/music3_setup_status.json",
                 "/content/music3_gen_status.json"):
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
    g.add_argument("--kind", choices=["image", "video", "music"], required=True)
    g.add_argument("--model", required=True,
                   help=("flux2-klein | hidream-i1 | ltx23 | ltx23-chain | "
                         "yue2-3b | minimax-music3"))
    g.add_argument("--prompt",
                   help="image/video prompt (required for those kinds)")
    g.add_argument("--negative")
    g.add_argument("--lyrics", help="music: song lyrics (multi-line OK)")
    g.add_argument("--style", help="music: yue2-3b style description")
    g.add_argument("--caption", help="music: minimax-music3 caption")
    g.add_argument("--duration", type=int,
                   help="music: song length in seconds (minimax-music3)")
    g.add_argument("--max-seconds", type=int,
                   help="music: song length in seconds (yue2-3b)")
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
