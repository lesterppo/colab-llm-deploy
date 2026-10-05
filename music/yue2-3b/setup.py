"""YuE2-3B (lyrics-to-song) setup on Colab T4.
Stack: ComfyUI v0.38.0 portable + pytraveler/YuE2-ComfyUI + tiktoken.
Weights via `hf` CLI (never deprecated huggingface-cli) into the
`original` layout the pack reads as-is:
  models/YuE2/YuE2-3B/model.safetensors + qwen.tiktoken  (m-a-p/YuE2-3B)
  models/YuE2/YuE2-Vae/model.safetensors                 (m-a-p/YuE2-Vae)
Stage-gated, writes /content/yue2_setup_status.json. Run via exec_detach.
"""
import json
import os
import subprocess
import sys

STATUS = "/content/yue2_setup_status.json"
CU = "/content/ComfyUI"


def st(stage, **kw):
    d = {"stage": stage, **kw}
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

    st("pip", msg="installing ComfyUI requirements + tiktoken")
    run([sys.executable, "-m", "pip", "install", "-q",
         "-r", os.path.join(CU, "requirements.txt")], timeout=1500)
    run([sys.executable, "-m", "pip", "install", "-q",
         "tiktoken", "huggingface_hub[cli]", "safetensors"], timeout=900)

    st("node", msg="cloning YuE2-ComfyUI pack")
    node = os.path.join(CU, "custom_nodes", "YuE2-ComfyUI")
    if not os.path.isdir(node):
        run(["git", "clone", "--depth", "1",
             "https://github.com/pytraveler/YuE2-ComfyUI", node],
            timeout=600)
    req = os.path.join(node, "requirements.txt")
    if os.path.exists(req):
        run([sys.executable, "-m", "pip", "install", "-q",
             "-r", req], timeout=900)

    st("dl_lm", msg="downloading m-a-p/YuE2-3B (6.76GB)")
    lm_dir = os.path.join(CU, "models", "YuE2", "YuE2-3B")
    os.makedirs(lm_dir, exist_ok=True)
    lm_path = os.path.join(lm_dir, "model.safetensors")
    if not (os.path.exists(lm_path) and os.path.getsize(lm_path) > 6e9):
        run(["hf", "download", "m-a-p/YuE2-3B",
             "--include", "model.safetensors",
             "--include", "qwen.tiktoken",
             "--include", "weights_manifest.json",
             "--local-dir", lm_dir], timeout=5400)
    else:
        print("YuE2-3B already present, skipping", flush=True)

    st("dl_vae", msg="downloading m-a-p/YuE2-Vae (0.49GB)")
    vae_dir = os.path.join(CU, "models", "YuE2", "YuE2-Vae")
    os.makedirs(vae_dir, exist_ok=True)
    vae_path = os.path.join(vae_dir, "model.safetensors")
    if not (os.path.exists(vae_path) and os.path.getsize(vae_path) > 4e8):
        run(["hf", "download", "m-a-p/YuE2-Vae",
             "--include", "model.safetensors",
             "--local-dir", vae_dir], timeout=1800)
    else:
        print("YuE2-Vae already present, skipping", flush=True)

    st("verify", msg="checking weight files")
    files = {
        "lm": lm_path,
        "merges": os.path.join(lm_dir, "qwen.tiktoken"),
        "vae": vae_path,
    }
    missing = [k for k, p in files.items() if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"missing weight files: {missing}")
    sizes = {k: round(os.path.getsize(p) / 1e9, 3) for k, p in files.items()}
    st("done", ready=True, files=files, sizes_gb=sizes,
       total_gb=round(sum(sizes.values()), 2))
except Exception as e:
    import traceback
    st("fatal", ready=False, error=f"{type(e).__name__}: {e}",
       tail=traceback.format_exc()[-2000:])
    sys.exit(1)
