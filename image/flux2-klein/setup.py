"""ComfyUI 0.38.0 + ComfyUI-GGUF setup for FLUX.2-klein-4B on Colab T4.
Stage-gated, writes /content/comfy_setup_status.json. Run via exec_detach."""
import json
import os
import subprocess
import sys

STATUS = "/content/comfy_setup_status.json"
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

    st("pip", msg="installing requirements")
    run([sys.executable, "-m", "pip", "install", "-q",
         "-r", os.path.join(CU, "requirements.txt")], timeout=1500)
    run([sys.executable, "-m", "pip", "install", "-q",
         "gguf", "huggingface_hub[cli]", "safetensors"], timeout=900)

    st("node", msg="installing ComfyUI-GGUF node")
    node = os.path.join(CU, "custom_nodes", "ComfyUI-GGUF")
    if not os.path.isdir(node):
        run(["git", "clone", "--depth", "1",
             "https://github.com/city96/ComfyUI-GGUF", node], timeout=600)

    st("dl_transformer", msg="downloading Q4_K_M GGUF (~2.5GB)")
    dm = os.path.join(CU, "models", "diffusion_models")
    os.makedirs(dm, exist_ok=True)
    run(["hf", "download", "unsloth/FLUX.2-klein-4B-GGUF",
         "flux-2-klein-4b-Q4_K_M.gguf", "--local-dir", dm], timeout=1800)

    st("dl_vae", msg="downloading flux2 VAE")
    vae_dir = os.path.join(CU, "models", "vae")
    os.makedirs(vae_dir, exist_ok=True)
    run(["hf", "download", "Comfy-Org/flux2-dev",
         "split_files/vae/flux2-vae.safetensors",
         "--local-dir", "/content/dl_vae"], timeout=900)
    import shutil
    shutil.move("/content/dl_vae/split_files/vae/flux2-vae.safetensors",
                os.path.join(vae_dir, "flux2-vae.safetensors"))

    st("dl_encoder", msg="downloading Qwen3 text encoder shards (~7.6GB)")
    run(["hf", "download", "black-forest-labs/FLUX.2-klein-4B",
         "--include", "text_encoder/*"], timeout=2400)

    st("merge_encoder", msg="merging text encoder to single safetensors")
    run([sys.executable, "/content/merge_qwen3.py"], timeout=1200)

    st("verify", msg="checking model files")
    files = {
        "transformer": os.path.join(dm, "flux-2-klein-4b-Q4_K_M.gguf"),
        "encoder": os.path.join(CU, "models", "text_encoders",
                                "qwen_3_4b.safetensors"),
        "vae": os.path.join(vae_dir, "flux2-vae.safetensors"),
    }
    missing = [k for k, p in files.items() if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"missing model files: {missing}")
    sizes = {k: round(os.path.getsize(p) / 1e9, 2)
             for k, p in files.items()}
    st("done", ready=True, files=files, sizes_gb=sizes)
except Exception as e:
    import traceback
    st("fatal", ready=False, error=f"{type(e).__name__}: {e}",
       tail=traceback.format_exc()[-2000:])
    sys.exit(1)
