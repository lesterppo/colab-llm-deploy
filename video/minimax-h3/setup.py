"""MiniMax H3 (Hailuo 03) 4-step INT8 route on Colab T4 — setup.

Stack: ComfyUI (latest, H3 native nodes in core since 0.30) + core
CreateVideo/SaveVideo. No custom nodes needed for the T2V path.
Downloads via `hf` CLI. Stage-gated, writes
/content/minimax_h3_setup_status.json. Run via exec_detach.

Primary route ("int8"): official Comfy-Org/MiniMax-H3 pack, as used by the
official video_minimax_h3_t2v.json template + Local Lab 4-step recipe
(2026-10-22, RTX A4500 20GB, 5s 0.6MP in ~1 min, peak ~13GB VRAM):
  diffusion  minimax_h3_fl2va_pruned_int8_convrot.safetensors  ~19.5GB
  encoder    qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors       ~14.6GB
  video VAE  minimax_h3_video_vae_int8_convrot.safetensors      ~2.6GB
  audio VAE  minimax_h3_audio_vae_fp32.safetensors              ~0.6GB
  turbo LoRA minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors ~1.8GB
  (from Comfy-Org/MiniMax-H3 loras/, NOT lightx2v's ref2v file)
Total ~39GB. If free disk < 50GB, diffusion switches to the W6A8 file
(minimax_h3_fl2va_pruned_w6a8.safetensors, ~16GB) and records it.

Known T4 risks (recorded, not hidden): the 32B NVFP4 encoder is 15.7GB on
disk and exceeds Colab's ~12GB system RAM when loaded; ComfyUI --lowvram
must offload it. A 16GB swapfile is attempted (non-fatal if it fails).
NVFP4/AWQ does not require a Blackwell GPU per community notes, but if
CLIPLoader type=minimax fails on the T4, the fallback is route "int4"
(see ROUTES below) — a different setup run, not a silent substitution.

Fallback route ("int4", only on explicit --route int4):
  diffusion  FL2VA pruned W4A8 12.5GB (Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI)
  encoder    Qwen3-VL-4B INT4 2.81GB (Merserk/qwen3vl-4b-int4-convrot)
  clip_proj  mmh3-4b-ClipProj-v3.1-mlp.safetensors 503MB (NicoLab28/ClipProj-MiniMax-H3)
  turbo LoRA minimax_h3_turbo_4step_ema_ckpt500.safetensors 780MB
             (Larryvrh/ComfyUI-MiniMax-H3-Turbo, node MiniMaxH3TurboLoRA)
  VAEs as primary. Total ~17GB.
"""
import json
import os
import shutil
import subprocess
import sys

STATUS = "/content/minimax_h3_setup_status.json"
CU = "/content/ComfyUI"
ROUTE = "int8"
if "--route" in sys.argv:
    ROUTE = sys.argv[sys.argv.index("--route") + 1]
elif os.path.exists("/content/minimax_h3_route.txt"):
    ROUTE = open("/content/minimax_h3_route.txt").read().strip() or "int8"

ROUTES = {
    "int8": {
        "diffusion_repo": "Comfy-Org/MiniMax-H3",
        "diffusion_file": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "diffusion_small_file": "diffusion_models/minimax_h3_fl2va_pruned_w6a8.safetensors",
        "encoder_repo": "Comfy-Org/MiniMax-H3",
        "encoder_file": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "turbo_repo": "Comfy-Org/MiniMax-H3",
        "turbo_file": "loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
    },
    "int4": {
        "diffusion_repo": "Winnougan/MiniMax-H3-INT4_Convrot_ComfyUI",
        "diffusion_file": "minimax_h3_fl2va_pruned_w4a8.safetensors",
        "diffusion_small_file": None,
        "encoder_repo": "Merserk/qwen3vl-4b-int4-convrot",
        "encoder_file": "qwen3vl_4b_int4_convrot.safetensors",
        "turbo_repo": "Larryvrh/ComfyUI-MiniMax-H3-Turbo",
        "turbo_file": "minimax_h3_turbo_4step_ema_ckpt500.safetensors",
        "clip_proj_repo": "NicoLab28/ClipProj-MiniMax-H3",
        "clip_proj_file": "mmh3-4b-ClipProj-v3.1-mlp.safetensors",
    },
}


def st(stage, **kw):
    d = {"stage": stage, "route": ROUTE, **kw}
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
    run(["hf", "download", repo, remote, "--local-dir",
         os.path.dirname(dst)], timeout=timeout)
    flatten(dst)
    if not (os.path.exists(dst) and os.path.getsize(dst) > min_gb * 1e9):
        raise RuntimeError(f"downloaded file too small/missing: {dst}")


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
        import resource
        info["ram_total_gb"] = round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except Exception:
        pass
    du = shutil.disk_usage("/content")
    info["disk_free_gb"] = round(du.free / 1e9, 1)
    info["disk_total_gb"] = round(du.total / 1e9, 1)
    return info


try:
    if ROUTE not in ROUTES:
        raise RuntimeError(f"unknown route {ROUTE!r}, want int8|int4")
    R = ROUTES[ROUTE]
    env = env_probe()
    st("env", msg="VM probe", **env)

    # Swap insurance for the 15.7GB encoder on ~12GB RAM (best effort).
    swap_ok = False
    if ROUTE == "int8" and env.get("disk_free_gb", 0) > 60:
        try:
            if not os.path.exists("/content/swapfile"):
                run(["fallocate", "-l", "16G", "/content/swapfile"], timeout=300)
                run(["chmod", "600", "/content/swapfile"], timeout=60)
                run(["mkswap", "/content/swapfile"], timeout=120)
            run(["swapon", "/content/swapfile"], timeout=120)
            swap_ok = True
        except Exception as e:
            print(f"swap setup failed (non-fatal): {e}", flush=True)
    st("swap", swap_ok=swap_ok)

    st("clone", msg="cloning ComfyUI (latest, H3 nodes are core since 0.30)")
    if not os.path.isdir(CU):
        run(["git", "clone", "--depth", "1",
             "https://github.com/comfyanonymous/ComfyUI", CU], timeout=600)

    st("pip", msg="installing ComfyUI requirements")
    run([sys.executable, "-m", "pip", "install", "-q",
         "-r", os.path.join(CU, "requirements.txt")], timeout=1500)
    run([sys.executable, "-m", "pip", "install", "-q",
         "gguf", "huggingface_hub[cli]", "safetensors", "einops"],
        timeout=900)

    # Disk gate: <50GB free -> smaller diffusion file (int8 route only).
    env = env_probe()
    diff_file = R["diffusion_file"]
    if ROUTE == "int8" and env.get("disk_free_gb", 0) < 50 and R.get("diffusion_small_file"):
        diff_file = R["diffusion_small_file"]
        print(f"disk free {env['disk_free_gb']}GB < 50GB, using {diff_file}",
              flush=True)
    st("disk_gate", diffusion_file=diff_file, **env)

    dm = os.path.join(CU, "models", "diffusion_models")
    st("dl_dit", msg=f"downloading diffusion {os.path.basename(diff_file)}")
    dl(R["diffusion_repo"], diff_file,
       os.path.join(dm, os.path.basename(diff_file)), 10)

    te = os.path.join(CU, "models", "text_encoders")
    st("dl_encoder", msg="downloading text encoder")
    dl(R["encoder_repo"], R["encoder_file"],
       os.path.join(te, os.path.basename(R["encoder_file"])),
       2 if ROUTE == "int4" else 10)

    if ROUTE == "int4":
        st("dl_clip_proj", msg="downloading 4B clip projection")
        dl(R["clip_proj_repo"], R["clip_proj_file"],
           os.path.join(te, os.path.basename(R["clip_proj_file"])), 0.3)

    lo = os.path.join(CU, "models", "loras")
    st("dl_turbo", msg="downloading 4-step turbo LoRA")
    dl(R["turbo_repo"], R["turbo_file"],
       os.path.join(lo, os.path.basename(R["turbo_file"])), 0.5)

    vd = os.path.join(CU, "models", "vae")
    st("dl_vae", msg="downloading video + audio VAEs")
    dl("Comfy-Org/MiniMax-H3",
       "vae/minimax_h3_video_vae_int8_convrot.safetensors",
       os.path.join(vd, "minimax_h3_video_vae_int8_convrot.safetensors"), 2)
    dl("Comfy-Org/MiniMax-H3",
       "vae/minimax_h3_audio_vae_fp32.safetensors",
       os.path.join(vd, "minimax_h3_audio_vae_fp32.safetensors"), 0.3)

    st("verify", msg="checking model files")
    files = {
        "diffusion": os.path.join(dm, os.path.basename(diff_file)),
        "encoder": os.path.join(te, os.path.basename(R["encoder_file"])),
        "turbo_lora": os.path.join(lo, os.path.basename(R["turbo_file"])),
        "video_vae": os.path.join(vd, "minimax_h3_video_vae_int8_convrot.safetensors"),
        "audio_vae": os.path.join(vd, "minimax_h3_audio_vae_fp32.safetensors"),
    }
    if ROUTE == "int4":
        files["clip_proj"] = os.path.join(
            te, os.path.basename(R["clip_proj_file"]))
    missing = [k for k, p in files.items() if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"missing model files: {missing}")
    sizes = {k: round(os.path.getsize(p) / 1e9, 2) for k, p in files.items()}
    st("done", ready=True, files=files, sizes_gb=sizes,
       total_gb=round(sum(sizes.values()), 2),
       diffusion_file=os.path.basename(diff_file), swap_ok=swap_ok, **env)
except Exception as e:
    import traceback
    st("fatal", ready=False, error=f"{type(e).__name__}: {e}",
       tail=traceback.format_exc()[-2000:])
    sys.exit(1)
