"""Wan 2.2 TI2V-5B fp16 text-to-video on Colab T4 — setup.

Stack: ComfyUI (latest; Wan 2.2 native nodes in core) + core
CreateVideo/SaveVideo. No custom nodes needed for the T2V path.
Downloads via `hf` CLI. Stage-gated, writes
/content/wan22_setup_status.json. Run via exec_detach.

Model: Wan 2.2 TI2V-5B (Alibaba Tongyi Lab, Apache 2.0) — the latest Wan
family with public weights. Wan 2.5/2.6/2.7/3.0 are API-only (no public
weights, checked 2026-11-04); Wan 2.2 A14B is a MoE pair with an 80GB
official floor. The 5B is a dense hybrid T2V+I2V model, 720p @24fps.

Files (Comfy-Org/Wan-2.2, canonical repo — renamed from
Wan_2.2_ComfyUI_Repackaged; sizes verified via HF API 2026-11-12):
  diffusion  split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors  10.0GB
  encoder    split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors  6.74GB
  VAE        split_files/vae/wan2.2_vae.safetensors  1.41GB
Total ~18.2GB. The 2.2 VAE is the 16x16x4 high-compression VAE, ONLY for
the 5B model — the 14B models need the Wan 2.1 VAE; do NOT mix them up.
No native audio in Wan 2.2 — the output MP4 is SILENT by design.

Reference (8GB card, ComfyUI, no launch flags, alexanderharte.com):
fp16, 20 steps, sampling 60s, wall-clock 73s, peak VRAM 7,686 MiB
(832x480 clip); Q4_K_M GGUF 121 frames in 9m14s, peak 7,798 MiB.
fp16 chosen over Q4_K_M GGUF: same per-step speed, Q4 gave a visibly
different composition while Q8/fp16 matched.

T4 risk (Wan 2.1 lesson, AGENTS.md): torch 2.11+cu130 has NO
flash/mem-efficient SDPA kernel for sm_75; SDPA silently takes the math
path and materialises the full attention-score tensor -> deterministic
VRAM OOM at sampling step 0. ComfyUI's native Wan nodes + --lowvram
offloading handled the H3 run (peak 14,619MB, no SDPA OOM); if Wan
sampling OOMs at step 0, the fix is chunked/optimized attention
(attention_mode="comfy" equivalent) / blocks_to_swap — record it.
"""
import json
import os
import shutil
import subprocess
import sys

STATUS = "/content/wan22_setup_status.json"
CU = "/content/ComfyUI"

# (repo, remote path) candidates in order; HF renames mean the old
# Wan_2.2_ComfyUI_Repackaged id may still resolve, canonical is Wan-2.2.
REPO = "Comfy-Org/Wan-2.2"
REPO_ALT = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
FILES = [
    ("diffusion", "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
     "diffusion_models", 9.0),
    ("encoder", "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
     "text_encoders", 6.0),
    ("vae", "split_files/vae/wan2.2_vae.safetensors",
     "vae", 1.0),
]


def st(stage, **kw):
    d = {"stage": stage, **kw}
    open(STATUS, "w").write(json.dumps(d))
    print(json.dumps(d), flush=True)


def run(cmd, timeout=1800, **kw):
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, timeout=timeout, **kw)


def flatten(dst):
    import glob
    if os.path.exists(dst):
        return
    cands = glob.glob(os.path.join(os.path.dirname(dst), "**",
                                   os.path.basename(dst)), recursive=True)
    cands = [c for c in cands if c != dst and os.path.isfile(c)]
    if cands:
        print(f"flattening {cands[0]} -> {dst}", flush=True)
        os.replace(cands[0], dst)


def dl(repo, remote, dst, min_gb, timeout=5400):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > min_gb * 1e9:
        print(f"{os.path.basename(dst)} already present, skipping", flush=True)
        return
    last = None
    for r in [repo, REPO_ALT]:
        try:
            run(["hf", "download", r, remote, "--local-dir",
                 os.path.dirname(dst)], timeout=timeout)
            flatten(dst)
            if os.path.exists(dst) and os.path.getsize(dst) > min_gb * 1e9:
                print(f"downloaded {os.path.basename(dst)} from {r}", flush=True)
                return
        except Exception as e:
            last = e
            print(f"download from {r} failed: {e}", flush=True)
    raise RuntimeError(f"download failed for {dst}: {last}")


def env_probe():
    info = {}
    for cmd, key in [(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader"], "gpu"),
                     (["python3", "-c",
                       "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"],
                      "torch")]:
        try:
            info[key] = subprocess.run(cmd, capture_output=True, text=True,
                                       timeout=60).stdout.strip()
        except Exception as e:
            info[key] = f"probe failed: {e}"
    try:
        info["ram_total_gb"] = round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except Exception:
        pass
    du = shutil.disk_usage("/content")
    info["disk_free_gb"] = round(du.free / 1e9, 1)
    info["disk_total_gb"] = round(du.total / 1e9, 1)
    return info


try:
    env = env_probe()
    st("env", msg="VM probe", **env)

    st("clone", msg="cloning ComfyUI (latest; Wan 2.2 nodes are core)")
    if not os.path.isdir(CU):
        run(["git", "clone", "--depth", "1",
             "https://github.com/comfyanonymous/ComfyUI", CU], timeout=600)

    st("pip", msg="installing ComfyUI requirements")
    run([sys.executable, "-m", "pip", "install", "-q",
         "-r", os.path.join(CU, "requirements.txt")], timeout=1500)
    run([sys.executable, "-m", "pip", "install", "-q",
         "gguf", "huggingface_hub[cli]", "safetensors", "einops"],
        timeout=900)

    paths = {}
    for key, remote, subdir, min_gb in FILES:
        st(f"dl_{key}", msg=f"downloading {os.path.basename(remote)}")
        dst = os.path.join(CU, "models", subdir, os.path.basename(remote))
        dl(REPO, remote, dst, min_gb)
        paths[key] = dst

    st("verify", msg="checking model files")
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"missing model files: {missing}")
    sizes = {k: round(os.path.getsize(p) / 1e9, 2) for k, p in paths.items()}
    st("done", ready=True, files=paths, sizes_gb=sizes,
       total_gb=round(sum(sizes.values()), 2), repo=REPO, **env)
except Exception as e:
    import traceback
    st("fatal", ready=False, error=f"{type(e).__name__}: {e}",
       tail=traceback.format_exc()[-2000:])
    sys.exit(1)
