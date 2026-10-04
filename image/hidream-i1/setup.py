"""ComfyUI + ComfyUI-GGUF + ComfyUI-HiDream-Sampler setup for HiDream-I1 Full Q4_K_M on Colab T4.
Proven pattern from flux2-klein-test (plain pip, git-cloned nodes, no venv).
Stage-gated, writes /content/hidream_setup_status.json. Run via exec_detach.

VRAM plan (15GB T4): DiT Q4_K_M (11.48GB) on GPU; the 4 text encoders
(clip_l+clip_g fp16, t5xxl+llama fp8 = 15.9GB) stay offloaded via --lowvram
(mmap'd safetensors, not anon RAM).
"""
import json
import os
import subprocess
import sys

STATUS = "/content/hidream_setup_status.json"
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

    st("nodes", msg="installing ComfyUI-GGUF + ComfyUI-HiDream-Sampler")
    cn = os.path.join(CU, "custom_nodes")
    os.makedirs(cn, exist_ok=True)
    # remove the abandoned lum3on sampler node if a previous run cloned it
    import shutil as _sh
    _sh.rmtree(os.path.join(cn, "comfyui_HiDream-Sampler"), ignore_errors=True)
    for name, repo in [
        ("ComfyUI-GGUF", "https://github.com/city96/ComfyUI-GGUF"),
    ]:
        node = os.path.join(cn, name)
        if not os.path.isdir(node):
            run(["git", "clone", "--depth", "1", repo, node], timeout=600)
        req = os.path.join(node, "requirements.txt")
        if os.path.exists(req):
            run([sys.executable, "-m", "pip", "install", "-q",
                 "-r", req], timeout=900)
        inst = os.path.join(node, "install.py")
        if os.path.exists(inst):
            run([sys.executable, inst], timeout=900)

    st("dl_dit", msg="downloading HiDream-I1 Full Q4_K_M (11.48GB)")
    dm = os.path.join(CU, "models", "diffusion_models")
    os.makedirs(dm, exist_ok=True)
    dit_path = os.path.join(dm, "hidream-i1-full-Q4_K_M.gguf")
    if not (os.path.exists(dit_path) and os.path.getsize(dit_path) > 10e9):
        run(["hf", "download", "city96/HiDream-I1-Full-gguf",
             "hidream-i1-full-Q4_K_M.gguf", "--local-dir", dm], timeout=5400)
    else:
        print("dit already present, skipping", flush=True)

    st("dl_encoders", msg="downloading 4 text encoders (15.9GB)")
    te = os.path.join(CU, "models", "text_encoders")
    os.makedirs(te, exist_ok=True)
    for f in ["clip_l_hidream.safetensors", "clip_g_hidream.safetensors",
              "t5xxl_fp8_e4m3fn_scaled.safetensors",
              "llama_3.1_8b_instruct_fp8_scaled.safetensors"]:
        dst = os.path.join(te, f)
        if os.path.exists(dst) and os.path.getsize(dst) > 1e6:
            print(f, "already present, skipping", flush=True)
            continue
        run(["hf", "download", "Comfy-Org/HiDream-I1_ComfyUI",
             f"split_files/text_encoders/{f}",
             "--local-dir", "/content/dl_te"], timeout=5400)
        import shutil
        shutil.move(f"/content/dl_te/split_files/text_encoders/{f}",
                    os.path.join(te, f))

    st("dl_vae", msg="downloading HiDream VAE")
    vae_dir = os.path.join(CU, "models", "vae")
    os.makedirs(vae_dir, exist_ok=True)
    vae_path = os.path.join(vae_dir, "ae.safetensors")
    if not (os.path.exists(vae_path) and os.path.getsize(vae_path) > 1e6):
        run(["hf", "download", "Comfy-Org/HiDream-I1_ComfyUI",
             "split_files/vae/ae.safetensors",
             "--local-dir", "/content/dl_vae"], timeout=900)
    import shutil
    vae_tmp = "/content/dl_vae/split_files/vae/ae.safetensors"
    if os.path.exists(vae_tmp):
        shutil.move(vae_tmp, vae_path)

    st("verify", msg="checking model files")
    files = {
        "dit": os.path.join(dm, "hidream-i1-full-Q4_K_M.gguf"),
        "clip_l": os.path.join(te, "clip_l_hidream.safetensors"),
        "clip_g": os.path.join(te, "clip_g_hidream.safetensors"),
        "t5xxl": os.path.join(te, "t5xxl_fp8_e4m3fn_scaled.safetensors"),
        "llama": os.path.join(te, "llama_3.1_8b_instruct_fp8_scaled.safetensors"),
        "vae": os.path.join(vae_dir, "ae.safetensors"),
    }
    missing = [k for k, p in files.items() if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"missing model files: {missing}")
    sizes = {k: round(os.path.getsize(p) / 1e9, 2) for k, p in files.items()}
    st("done", ready=True, files=files, sizes_gb=sizes,
       total_gb=round(sum(sizes.values()), 2))
except Exception as e:
    import traceback
    st("fatal", ready=False, error=f"{type(e).__name__}: {e}",
       tail=traceback.format_exc()[-2000:])
    sys.exit(1)
