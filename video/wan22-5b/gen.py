"""Wan 2.2 TI2V-5B fp16 text-to-video on Colab T4.

Run via exec_detach AFTER video/wan22-5b/setup.py completes.
Reads /content/gen_config.json:
  {"prompt": str, "negative": str (opt), "seed": int (opt, 42),
   "width": int (opt, 832), "height": int (opt, 480),
   "frames": int (opt, 81 = 4*20+1, ~3.4s @24fps), "fps": int (opt, 24),
   "steps": int (opt, 20), "cfg": float (opt, 5.0),
   "out_name": str (opt)}
Writes /content/wan22_gen_status.json; MP4 -> /content/<out_name>.mp4.
Launches ComfyUI headless (--lowvram --cache-none --fast-disk) itself.
Output is SILENT (no native audio in Wan 2.2) — that is correct.

Workflow (official video_wan2_2_5B_ti2v.json template, API form):
  UNETLoader (wan2.2_ti2v_5B_fp16) -> ModelSamplingSD3 (shift 8.0)
  -> KSampler (uni_pc + simple, 20 steps, cfg 5.0, seed fixed)
  CLIPLoader (umt5_xxl_fp8_e4m3fn_scaled, type wan) -> CLIPTextEncode x2
  VAELoader (wan2.2_vae) + Wan22ImageToVideoLatent (no start_image:
  start_image is optional in the node — None means pure T2V; the
  template's LoadImage node is bypassed, mode=4)
  -> VAEDecode -> CreateVideo (fps 24) -> SaveVideo (auto/auto).

Template defaults: 1280x704, length 121, 20 steps, cfg 5, uni_pc/simple,
shift 8. First T4 target is smaller: 832x480, 81 frames, same sampler.
Frames must be 4n+1 (VAE 4x temporal compression).
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

STATUS = "/content/wan22_gen_status.json"
CONFIG = "/content/gen_config.json"
CU = "/content/ComfyUI"
BASE = "http://127.0.0.1:8188"

DIFFUSION = "wan2.2_ti2v_5B_fp16.safetensors"
ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE = "wan2.2_vae.safetensors"


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


def build_workflow(cfg):
    p, nid = {}, [0]

    def add(class_type, inputs):
        nid[0] += 1
        p[str(nid[0])] = {"class_type": class_type, "inputs": inputs}
        return str(nid[0])

    unet = add("UNETLoader",
               {"unet_name": DIFFUSION, "weight_dtype": "default"})
    model = add("ModelSamplingSD3",
                {"model": [unet, 0], "shift": 8.0})
    clip = add("CLIPLoader",
               {"clip_name": ENCODER, "type": "wan", "device": "default"})
    vae = add("VAELoader", {"vae_name": VAE})
    pos = add("CLIPTextEncode",
              {"text": cfg["prompt"], "clip": [clip, 0]})
    neg = add("CLIPTextEncode",
              {"text": cfg.get("negative", "low quality, blurry, static, "
                                "deformed, watermark, text, subtitles, "
                                "worst quality, low-res, noisy"),
               "clip": [clip, 0]})
    latent_inputs = {"vae": [vae, 0],
                     "width": cfg.get("width", 832),
                     "height": cfg.get("height", 480),
                     "length": cfg.get("frames", 81),
                     "batch_size": 1}
    # I2V leg: cfg["image"] is a filename in ComfyUI/input/. The node
    # treats it as a flexible reference, not a rigid first-frame lock
    # (strict first-frame I2V is the A14B I2V model, which does not fit
    # a T4) — expect loose composition following, not pixel-perfect.
    if cfg.get("image"):
        img = add("LoadImage", {"image": os.path.basename(cfg["image"])})
        latent_inputs["start_image"] = [img, 0]
    latent = add("Wan22ImageToVideoLatent", latent_inputs)
    sampled = add("KSampler",
                  {"model": [model, 0], "positive": [pos, 0],
                   "negative": [neg, 0], "latent_image": [latent, 0],
                   "seed": cfg.get("seed", 42),
                   "steps": cfg.get("steps", 20),
                   "cfg": cfg.get("cfg", 5.0),
                   "sampler_name": "uni_pc", "scheduler": "simple",
                   "denoise": 1.0})
    dec = add("VAEDecode", {"samples": [sampled, 0], "vae": [vae, 0]})
    video = add("CreateVideo", {"images": [dec, 0],
                                "fps": float(cfg.get("fps", 24))})
    add("SaveVideo", {"video": [video, 0],
                      "filename_prefix": cfg.get("out_name", "wan22_5b_out"),
                      "format": "auto", "codec": "auto"})
    return p


try:
    cfg = json.load(open(CONFIG))
    if not cfg.get("prompt"):
        raise RuntimeError("gen_config.json missing 'prompt'")
    check = {"diffusion": (os.path.join(CU, "models", "diffusion_models"),
                           DIFFUSION),
             "encoder": (os.path.join(CU, "models", "text_encoders"),
                         ENCODER),
             "vae": (os.path.join(CU, "models", "vae"), VAE)}
    missing_local = [k for k, (d, f) in check.items()
                     if not os.path.exists(os.path.join(d, f))]
    if missing_local:
        raise RuntimeError(f"model files missing on disk: {missing_local}")

    st("launch", msg="starting ComfyUI --lowvram --cache-none --fast-disk")
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
            if "Wan22ImageToVideoLatent" in info and "UNETLoader" in info:
                break
        except Exception:
            pass
        time.sleep(5)
    info = api("GET", "/object_info", timeout=30)
    req = ["UNETLoader", "CLIPLoader", "VAELoader",
           "Wan22ImageToVideoLatent", "CLIPTextEncode", "ModelSamplingSD3",
           "KSampler", "VAEDecode", "CreateVideo", "SaveVideo"]
    missing = [c for c in req if c not in info]
    if missing:
        raise RuntimeError("missing ComfyUI nodes: " + ", ".join(missing))

    st("queue", msg="submitting workflow", vram_mb=vram_mb())
    wf = build_workflow(cfg)
    resp = api("POST", "/prompt", {"prompt": wf}, timeout=120)
    pid = resp["prompt_id"]
    t0 = time.time()
    out_dir = os.path.join(CU, "output")
    prefix = cfg.get("out_name", "wan22_5b_out")
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
                   peak_vram_mb=peak_vram)
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
