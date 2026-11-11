"""MiniMax H3 4-step INT8 text-to-video on Colab T4.

Run via exec_detach AFTER video/minimax-h3/setup.py completes.
Reads /content/gen_config.json:
  {"prompt": str, "seed": int (opt), "width": int (opt, 640),
   "height": int (opt, 960), "frames": int (opt, 124 = 17*7+5, ~5.2s),
   "fps": int (opt, 24), "steps": int (opt, 4),
   "out_name": str (opt), "route": "int8" (opt)}
Writes /content/minimax_h3_gen_status.json; MP4 -> /content/<out_name>.mp4.
Launches ComfyUI headless (--lowvram --cache-none --fast-disk) itself.

Workflow (official video_minimax_h3_t2v.json template, API form):
  UNETLoader (fl2va pruned int8) -> LoraLoaderModelOnly (fl2v turbo 4-step)
  -> MiniMaxH3SigmaShift (video 6, audio 3) -> [optional
  MiniMaxH3MemoryEfficientSageAttentionPatch: T4 sm_75 has no fast SDPA
  kernel, SDPA math path materialises the full attention matrix — the Wan
  2.1 OOM lesson] -> BasicGuider; CLIPLoader type=minimax (NVFP4 encoder)
  + VAELoader x2 -> MiniMaxH3ImageToVideo (prompt string, no CLIPTextEncode)
  -> SamplerCustomAdvanced (er_sde + sgm_uniform, 4 steps, cfg 1.0)
  -> VAEDecode + VAEDecodeAudio -> CreateVideo -> SaveVideo.

Proven reference (NOT yet on T4): RTX A4500 20GB, 5s 0.6MP, ~1 min
sampling, peak ~13GB VRAM (Local Lab Digest 2026-10-22).
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

STATUS = "/content/minimax_h3_gen_status.json"
CONFIG = "/content/gen_config.json"
CU = "/content/ComfyUI"
BASE = "http://127.0.0.1:8188"

FILES = {
    "int8": {
        "diffusion": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "diffusion_small": "minimax_h3_fl2va_pruned_w6a8.safetensors",
        "encoder": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "turbo": "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
    },
    "int4": {
        "diffusion": "minimax_h3_fl2va_pruned_w4a8.safetensors",
        "encoder": "qwen3vl_4b_int4_convrot.safetensors",
        "turbo": "minimax_h3_turbo_4step_ema_ckpt500.safetensors",
    },
}


def st(stage, **kw):
    d = {"stage": stage, **kw}
    open(STATUS, "w").write(json.dumps(d))
    print(json.dumps(d), flush=True)


def api(method, path, payload=None, timeout=600):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        method=method,
        headers={"Content-Type": "application/json"} if payload else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:2000]
        raise RuntimeError(f"HTTP {e.code} on {method} {path}: {body}")


def vram_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out.split()[0])
    except Exception:
        return -1


def build_workflow(cfg, info, files):
    p, nid = {}, [0]

    def add(class_type, inputs):
        nid[0] += 1
        p[str(nid[0])] = {"class_type": class_type, "inputs": inputs}
        return str(nid[0])

    unet = add("UNETLoader",
               {"unet_name": files["diffusion"], "weight_dtype": "default"})
    model = unet
    # Turbo LoRA: LoraLoaderModelOnly in the int8 route; int4 uses a
    # dedicated MiniMaxH3TurboLoRA node if present.
    if cfg.get("route", "int8") == "int4" and "MiniMaxH3TurboLoRA" in info:
        model = add("MiniMaxH3TurboLoRA",
                    {"model": [model, 0], "lora_name": files["turbo"],
                     "strength_model": 1.0})
    else:
        model = add("LoraLoaderModelOnly",
                    {"model": [model, 0], "lora_name": files["turbo"],
                     "strength_model": 1.0})
    model = add("MiniMaxH3SigmaShift",
                {"model": [model, 0], "shift_video": 6.0, "shift_audio": 3.0})
    sage = "MiniMaxH3MemoryEfficientSageAttentionPatch" in info
    if sage:
        model = add("MiniMaxH3MemoryEfficientSageAttentionPatch",
                    {"model": [model, 0]})
    clip = add("CLIPLoader",
               {"clip_name": files["encoder"], "type": "minimax"})
    vae = add("VAELoader",
              {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"})
    avae = add("VAELoader",
               {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"})
    # MiniMaxH3ImageToVideo inputs vary by ComfyUI version; build from
    # the live /object_info schema so a missing audio_vae input is not
    # a 400. Prompt is a plain STRING on this node (no CLIPTextEncode).
    h3_inputs = {"clip": [clip, 0], "vae": [vae, 0],
                 "prompt": cfg["prompt"],
                 "width": cfg.get("width", 640),
                 "height": cfg.get("height", 960),
                 "length": cfg.get("frames", 124)}
    h3_schema = info.get("MiniMaxH3ImageToVideo", {}).get("input", {})
    h3_req = h3_schema.get("required", {})
    if "audio_vae" in h3_req:
        h3_inputs["audio_vae"] = [avae, 0]
    cond = add("MiniMaxH3ImageToVideo", h3_inputs)
    noise = add("RandomNoise",
                {"noise_seed": cfg.get("seed", 42), "noise_mode": "fixed"})
    guider = add("BasicGuider", {"model": [model, 0],
                                 "conditioning": [cond, 0]})
    ksamp = add("KSamplerSelect", {"sampler_name": "er_sde"})
    sigmas = add("BasicScheduler", {"model": [model, 0],
                                    "scheduler": "sgm_uniform",
                                    "steps": cfg.get("steps", 4),
                                    "denoise": 1.0})
    sampled = add("SamplerCustomAdvanced",
                  {"noise": [noise, 0], "guider": [guider, 0],
                   "sampler": [ksamp, 0], "sigmas": [sigmas, 0],
                   "latent_image": [cond, 1]})
    dec_v = add("VAEDecode", {"samples": [sampled, 0], "vae": [vae, 0]})
    # ComfyUI 0.45: VAEDecodeAudio takes "vae" (not "audio_vae"), and
    # CreateVideo takes "fps" (not "frame_rate") — both caught by the
    # server's prompt validation on the first submit (HTTP 400).
    dec_a = add("VAEDecodeAudio", {"samples": [sampled, 0],
                                   "vae": [avae, 0]})
    video = add("CreateVideo", {"images": [dec_v, 0], "audio": [dec_a, 0],
                                "fps": float(cfg.get("fps", 24))})
    # ComfyUI 0.45 SaveVideo: format/codec are COMFY_DYNAMICCOMBO_V3
    # keys ("auto"/"h264"/"av1"), NOT "video/h264-mp4" — the latter
    # passes validation but execute() then gets no format at all
    # (TypeError, prompt done, nothing saved). Template uses auto/auto.
    add("SaveVideo", {"video": [video, 0],
                      "filename_prefix": cfg.get("out_name", "minimax_h3_out"),
                      "format": "auto", "codec": "auto"})
    return p, sage


try:
    cfg = json.load(open(CONFIG))
    if not cfg.get("prompt"):
        raise RuntimeError("gen_config.json missing 'prompt'")
    route = cfg.get("route", "int8")
    files = dict(FILES[route])
    # Resolve the diffusion file actually present (W6A8 fallback).
    dm = os.path.join(CU, "models", "diffusion_models")
    if not os.path.exists(os.path.join(dm, files["diffusion"])):
        alt = files.get("diffusion_small")
        if alt and os.path.exists(os.path.join(dm, alt)):
            files["diffusion"] = alt
    # "diffusion_small" is only an alternative name for the diffusion
    # file (resolved above); it is not a file that must exist.
    check = {"diffusion": (dm, files["diffusion"]),
             "encoder": (os.path.join(CU, "models", "text_encoders"),
                         files["encoder"]),
             "turbo": (os.path.join(CU, "models", "loras"), files["turbo"])}
    missing_local = [k for k, (d, f) in check.items()
                     if not os.path.exists(os.path.join(d, f))]
    if missing_local:
        raise RuntimeError(f"model files missing on disk: {missing_local}")

    st("launch", msg="starting ComfyUI --lowvram --cache-none --fast-disk",
       route=route, files=files)
    # Reuse a ComfyUI already running on :8188 (e.g. from a previous gen
    # attempt in this session) instead of launching a second one.
    already_up = False
    try:
        info0 = api("GET", "/object_info", timeout=10)
        already_up = "UNETLoader" in info0
    except Exception:
        pass
    if not already_up:
        log = open("/content/comfy.log", "a")
        subprocess.Popen(
            [sys.executable, os.path.join(CU, "main.py"),
             "--listen", "127.0.0.1", "--port", "8188",
             "--disable-auto-launch", "--lowvram", "--cache-none", "--fast-disk"],
            stdout=log, stderr=subprocess.STDOUT, cwd=CU,
            start_new_session=True)
    else:
        print("ComfyUI already up on :8188, reusing it", flush=True)
    deadline = time.time() + 900
    info = {}
    while time.time() < deadline:
        try:
            info = api("GET", "/object_info", timeout=10)
            if "MiniMaxH3ImageToVideo" in info and "UNETLoader" in info:
                break
        except Exception:
            pass
        time.sleep(5)
    info = api("GET", "/object_info", timeout=30)
    req = ["UNETLoader", "LoraLoaderModelOnly", "MiniMaxH3SigmaShift",
           "CLIPLoader", "VAELoader", "MiniMaxH3ImageToVideo", "RandomNoise",
           "BasicGuider", "KSamplerSelect", "BasicScheduler",
           "SamplerCustomAdvanced", "VAEDecode", "VAEDecodeAudio",
           "CreateVideo", "SaveVideo"]
    missing = [c for c in req if c not in info]
    h3_nodes = sorted(n for n in info if "MiniMaxH3" in n or "MiniMax" in n)
    if missing:
        raise RuntimeError("missing ComfyUI nodes: " + ", ".join(missing) +
                           f"; H3 nodes present: {h3_nodes}")

    st("queue", msg="submitting workflow", vram_mb=vram_mb(), route=route)
    wf, sage = build_workflow(cfg, info, files)
    st("queue", msg="workflow built", vram_mb=vram_mb(), sage_patch=sage)
    resp = api("POST", "/prompt", {"prompt": wf}, timeout=120)
    pid = resp["prompt_id"]
    t0 = time.time()
    out_dir = os.path.join(CU, "output")
    prefix = cfg.get("out_name", "minimax_h3_out")
    deadline = t0 + 3600  # 60 min hard cap
    last_log = 0
    peak_vram = vram_mb()
    while time.time() < deadline:
        v = vram_mb()
        peak_vram = max(peak_vram, v)
        mp4s = sorted(f for f in os.listdir(out_dir)
                      if f.startswith(prefix) and f.endswith(".mp4")) \
            if os.path.isdir(out_dir) else []
        now = time.time()
        if now - last_log > 60:
            last_log = now
            st("sampling", elapsed_s=int(now - t0), vram_mb=v,
               peak_vram_mb=peak_vram, outputs=mp4s)
        done = False
        try:
            h = api("GET", f"/history/{pid}", timeout=20)
            if pid in h:
                done = True
        except Exception:
            pass
        if done and mp4s:
            path = os.path.join(out_dir, mp4s[-1])
            s1 = os.path.getsize(path)
            time.sleep(10)
            s2 = os.path.getsize(path)
            stable = 0
            while s1 != s2 and stable < 4:
                stable += 1
                time.sleep(10)
                s1, s2 = s2, os.path.getsize(path)
            if s1 == s2 and s1 > 100_000:
                dst = f"/content/{prefix}.mp4"
                import shutil
                shutil.copy(path, dst)
                st("done", ready=True, output=dst,
                   size_mb=round(s1 / 1048576, 2),
                   elapsed_s=int(time.time() - t0),
                   peak_vram_mb=peak_vram, sage_patch=sage)
                sys.exit(0)
        if done and not mp4s:
            raise RuntimeError(
                f"history shows done but no MP4 under {prefix}*.mp4; "
                f"history={json.dumps(api('GET', f'/history/{pid}', timeout=20))[:1500]}")
        time.sleep(15)
    raise RuntimeError("generation timed out after 60 min")
except Exception as e:
    import traceback
    st("fatal", ready=False, error=f"{type(e).__name__}: {e}",
       tail=traceback.format_exc()[-2000:])
    sys.exit(1)
