"""LTX-2.3 (22B distilled Q3_K_M) setup on Colab T4 for text-to-video.
Stack: ComfyUI portable + ComfyUI-GGUF + KJNodes + VideoHelperSuite + LTXVideo.
Downloads via `hf` CLI (never deprecated huggingface-cli). Stage-gated,
writes /content/ltx23_setup_status.json. Run via exec_detach.

VRAM plan (15GB T4): DiT Q3_K_M (10.63GB) windowed via --lowvram;
gemma-3-12b Q3_K_M GGUF (5.7GB) freed post-encode via --cache-none.
"""
import json
import os
import subprocess
import sys

STATUS = "/content/ltx23_setup_status.json"
CU = "/content/ComfyUI"


def st(stage, **kw):
    d = {"stage": stage, **kw}
    open(STATUS, "w").write(json.dumps(d))
    print(json.dumps(d), flush=True)


def run(cmd, timeout=1800, **kw):
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, timeout=timeout, **kw)


def flatten(dst):
    """hf CLI preserves repo subpaths on download; move the file flat."""
    import glob
    if os.path.exists(dst):
        return
    cands = glob.glob(os.path.join(os.path.dirname(dst), "**",
                                   os.path.basename(dst)), recursive=True)
    cands = [c for c in cands if c != dst and os.path.isfile(c)]
    if cands:
        print(f"flattening {cands[0]} -> {dst}", flush=True)
        os.replace(cands[0], dst)


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

    st("pip", msg="installing ComfyUI requirements")
    run([sys.executable, "-m", "pip", "install", "-q",
         "-r", os.path.join(CU, "requirements.txt")], timeout=1500)
    run([sys.executable, "-m", "pip", "install", "-q",
         "gguf", "huggingface_hub[cli]", "safetensors"], timeout=900)

    st("nodes", msg="installing custom nodes")
    cn = os.path.join(CU, "custom_nodes")
    os.makedirs(cn, exist_ok=True)
    for name, repo in [
        ("ComfyUI-GGUF", "https://github.com/city96/ComfyUI-GGUF"),
        ("ComfyUI-KJNodes", "https://github.com/kijai/ComfyUI-KJNodes"),
        ("ComfyUI-VideoHelperSuite",
         "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"),
        ("ComfyUI-LTXVideo", "https://github.com/Lightricks/ComfyUI-LTXVideo"),
    ]:
        node = os.path.join(cn, name)
        if not os.path.isdir(node):
            run(["git", "clone", "--depth", "1", repo, node], timeout=600)
        req = os.path.join(node, "requirements.txt")
        if os.path.exists(req):
            try:
                run([sys.executable, "-m", "pip", "install", "-q",
                     "-r", req], timeout=900)
            except subprocess.CalledProcessError:
                print(f"node requirements failed for {name}, continuing",
                      flush=True)
        inst = os.path.join(node, "install.py")
        if os.path.exists(inst):
            run([sys.executable, inst], timeout=900)

    st("dl_dit", msg="downloading LTX-2.3 distilled Q3_K_M (10.63GB)")
    dm = os.path.join(CU, "models", "unet")
    os.makedirs(dm, exist_ok=True)
    dit_path = os.path.join(dm, "ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")
    if not (os.path.exists(dit_path) and os.path.getsize(dit_path) > 10e9):
        run(["hf", "download", "unsloth/LTX-2.3-GGUF",
             "distilled-1.1/ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf",
             "--local-dir", dm], timeout=5400)
    else:
        print("dit already present, skipping", flush=True)
    flatten(dit_path)

    st("dl_encoder", msg="downloading gemma-3-12b Q3_K_M (5.7GB)")
    te = os.path.join(CU, "models", "text_encoders")
    os.makedirs(te, exist_ok=True)
    enc_path = os.path.join(te, "gemma-3-12b-it-heretic-v2-Q3_K_M.gguf")
    if not (os.path.exists(enc_path) and os.path.getsize(enc_path) > 5e9):
        run(["hf", "download", "DreamFast/gemma-3-12b-it-heretic-v2",
             "gguf/gemma-3-12b-it-heretic-v2-Q3_K_M.gguf",
             "--local-dir", te], timeout=5400)
    else:
        print("encoder already present, skipping", flush=True)
    flatten(enc_path)

    st("dl_support", msg="downloading projection + VAEs from Kijai/LTX2.3_comfy")
    proj_path = os.path.join(te, "ltx-2.3_text_projection_bf16.safetensors")
    if not (os.path.exists(proj_path) and os.path.getsize(proj_path) > 2e9):
        run(["hf", "download", "Kijai/LTX2.3_comfy",
             "text_encoders/ltx-2.3_text_projection_bf16.safetensors",
             "--local-dir", "/content/dl_proj"], timeout=1800)
        os.replace("/content/dl_proj/text_encoders/ltx-2.3_text_projection_bf16.safetensors",
                   proj_path)
    vae_dir = os.path.join(CU, "models", "vae")
    os.makedirs(vae_dir, exist_ok=True)
    for f in ["LTX23_video_vae_bf16.safetensors",
              "LTX23_audio_vae_bf16.safetensors"]:
        dst = os.path.join(vae_dir, f)
        if os.path.exists(dst) and os.path.getsize(dst) > 3e8:
            print(f, "already present, skipping", flush=True)
            continue
        run(["hf", "download", "Kijai/LTX2.3_comfy", f"vae/{f}",
             "--local-dir", "/content/dl_vae"], timeout=1800)
        os.replace(f"/content/dl_vae/vae/{f}", dst)

    # LTXVAudioVAELoader scans models/checkpoints/, NOT models/vae/
    ckpt_dir = os.path.join(CU, "models", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    import shutil as _sh
    _sh.copy(os.path.join(vae_dir, "LTX23_audio_vae_bf16.safetensors"),
             os.path.join(ckpt_dir, "LTX23_audio_vae_bf16.safetensors"))
    st("verify", msg="checking model files")
    files = {
        "dit": dit_path,
        "gemma": enc_path,
        "projection": proj_path,
        "video_vae": os.path.join(vae_dir, "LTX23_video_vae_bf16.safetensors"),
        "audio_vae": os.path.join(vae_dir, "LTX23_audio_vae_bf16.safetensors"),
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
