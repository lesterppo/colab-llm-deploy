"""MiniMax Music 3 setup on Colab T4.
ComfyUI + pip + INT8 low-VRAM model files from Comfy-Org/MiniMax-Music-3.
Stage-gated, writes /content/music3_setup_status.json. Run via exec_detach.

INT8 `convrot` repack (22GB full precision -> ~11.9GB):
  diffusion_models/minimax_music3_dit_int8_convrot.safetensors          (2.5GB)
  text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors (9.2GB)
  vae/minimax_music3_dav.safetensors                                     (217MB)

Adapted from lesterppo/minimax-music3-colab's deploy.py (tunnel/watchdog
removed — generation runs through the local ComfyUI API here).
"""
import json
import os
import subprocess
import sys
import time

STATUS = "/content/music3_setup_status.json"
CU = "/content/ComfyUI"
MODELS = [
    "diffusion_models/minimax_music3_dit_int8_convrot.safetensors",
    "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
    "vae/minimax_music3_dav.safetensors",
]


def st(stage, **kw):
    d = {"stage": stage, "t": time.time(), **kw}
    open(STATUS, "w").write(json.dumps(d))
    print(json.dumps(d), flush=True)


def run(cmd, timeout=1800, **kw):
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, timeout=timeout, **kw)


try:
    st("clone", msg="cloning ComfyUI")
    if not os.path.isdir(CU):
        try:
            run(["git", "clone", "--depth", "1", "--branch", "v0.38.0",
                 "https://github.com/comfyanonymous/ComfyUI", CU],
                timeout=600)
        except subprocess.CalledProcessError:
            run(["git", "clone", "--depth", "1",
                 "https://github.com/comfyanonymous/ComfyUI", CU],
                timeout=600)

    st("pip", msg="installing requirements + hf CLI")
    run([sys.executable, "-m", "pip", "install", "-q",
         "-r", os.path.join(CU, "requirements.txt")], timeout=1500)
    run([sys.executable, "-m", "pip", "install", "-q",
         "huggingface_hub[cli]"], timeout=900)

    st("dl_models", msg="downloading INT8 repack (~11.9GB)")
    for sub in ["diffusion_models", "text_encoders", "vae"]:
        os.makedirs(os.path.join(CU, "models", sub), exist_ok=True)
    for rel in MODELS:
        dest = os.path.join(CU, "models", rel)
        if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
            print(f"skip (exists): {rel}", flush=True)
            continue
        print(f"downloading {rel} ...", flush=True)
        run(["hf", "download", "Comfy-Org/MiniMax-Music-3", rel,
             "--local-dir", os.path.join(CU, "models")], timeout=3600)

    st("verify", msg="checking model files")
    missing = [r for r in MODELS
               if not os.path.exists(os.path.join(CU, "models", r))]
    if missing:
        raise RuntimeError(f"missing model files: {missing}")
    sizes = {r: round(os.path.getsize(os.path.join(CU, "models", r)) / 1e9, 2)
             for r in MODELS}
    st("done", ready=True, sizes_gb=sizes,
       total_gb=round(sum(sizes.values()), 2))
except Exception as e:
    import traceback
    st("fatal", ready=False, error=f"{type(e).__name__}: {e}",
       tail=traceback.format_exc()[-2000:])
    sys.exit(1)
