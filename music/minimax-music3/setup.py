"""MiniMax Music 3 setup on a free Colab T4.

Installs ComfyUI (latest) + requirements, downloads the INT8 low-VRAM model
set from Comfy-Org/MiniMax-Music-3, launches ComfyUI headless on :8188.
Stage-gated, writes /content/music3_setup_status.json. Run via exec_detach.
(The cloudflared tunnel + watchdog from the original deploy flow live in
server.py for optional remote access; the one-shot deploy does not need them.)

Model files (INT8 set, ~11.9 GB total):
  diffusion_models/minimax_music3_dit_int8_convrot.safetensors          (2.5 GB)
  text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors (9.2 GB)
  vae/minimax_music3_dav.safetensors                                    (217 MB)
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

STATUS = "/content/music3_setup_status.json"
CU = "/content/ComfyUI"
PORT = 8188


def st(stage, **kw):
    d = {"stage": stage, **kw}
    open(STATUS, "w").write(json.dumps(d))
    print(json.dumps(d), flush=True)


def run(cmd, timeout=1800):
    print("+ " + cmd, flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=timeout)
    tail = (r.stdout or r.stderr or "").strip().splitlines()
    if tail:
        print("  -> " + tail[-1][:300], flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed rc={r.returncode}: {cmd}\n"
                           f"{(r.stderr or '')[-1500:]}")
    return r


def comfy_ready():
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/system_stats",
                timeout=10) as r:
            return '"devices"' in r.read().decode()
    except Exception:
        return False


def start_comfyui():
    if comfy_ready():
        print("ComfyUI already responding — skip launch", flush=True)
        return
    log = open("/content/comfyui.log", "a")
    subprocess.Popen(
        [sys.executable, os.path.join(CU, "main.py"),
         "--listen", "127.0.0.1", "--port", str(PORT),
         "--cache-none", "--disable-auto-launch"],
        stdout=log, stderr=subprocess.STDOUT, cwd=CU,
        start_new_session=True)
    print("ComfyUI launched", flush=True)


try:
    st("deps", msg="installing hf_transfer + huggingface_hub")
    run("pip install -q --no-input hf_transfer huggingface_hub", timeout=300)

    st("clone", msg="cloning ComfyUI")
    if not os.path.isdir(os.path.join(CU, ".git")):
        run(f"rm -rf {CU} && git clone --depth 1 "
            f"https://github.com/comfyanonymous/ComfyUI.git {CU}",
            timeout=600)
    else:
        run(f"cd {CU} && git pull --ff-only", timeout=300)

    st("pip", msg="installing ComfyUI requirements")
    run(f"pip install -q --no-input -r {CU}/requirements.txt", timeout=1500)

    st("dl", msg="downloading MiniMax-Music-3 INT8 set (~11.9GB)")
    models = [
        "diffusion_models/minimax_music3_dit_int8_convrot.safetensors",
        "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
        "vae/minimax_music3_dav.safetensors",
    ]
    for sub in ("diffusion_models", "text_encoders", "vae"):
        os.makedirs(f"{CU}/models/{sub}", exist_ok=True)
    for rel in models:
        dest = f"{CU}/models/{rel}"
        if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
            print(f"skip (exists): {rel}", flush=True)
            continue
        run(f"HF_HUB_ENABLE_HF_TRANSFER=1 hf download "
            f"Comfy-Org/MiniMax-Music-3 {rel} --local-dir {CU}/models",
            timeout=3600)
    missing = [r for r in models
               if not os.path.exists(f"{CU}/models/{r}")]
    if missing:
        raise RuntimeError(f"missing model files: {missing}")
    sizes = {r: round(os.path.getsize(f"{CU}/models/{r}") / 1e9, 3)
             for r in models}

    st("launch", msg="starting ComfyUI")
    start_comfyui()
    deadline = time.time() + 1800
    while time.time() < deadline:
        if comfy_ready():
            break
        time.sleep(5)
    else:
        raise RuntimeError("ComfyUI never came up; check /content/comfyui.log")

    st("done", ready=True, sizes_gb=sizes,
       total_gb=round(sum(sizes.values()), 2))
except Exception as e:
    import traceback
    st("fatal", ready=False, error=f"{type(e).__name__}: {e}",
       tail=traceback.format_exc()[-2000:])
    sys.exit(1)
