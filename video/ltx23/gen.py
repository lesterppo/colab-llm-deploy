"""LTX-2.3 (22B distilled Q3_K_M) text-to-video on Colab T4.

Run via exec_detach AFTER video/ltx23/setup.py completes.
Reads /content/gen_config.json:
  {"prompt": str, "negative": str (opt), "seed": int (opt),
   "width": int (opt), "height": int (opt), "frames": int (opt),
   "fps": int (opt), "steps": int (opt), "out_name": str (opt)}
Writes /content/ltx23_gen_status.json; MP4 -> /content/<out_name>.mp4.
Launches ComfyUI headless (--lowvram --cache-none) itself.

Proven: 576x320, 49 frames @24fps, 8 steps in ~430s on T4, peak ~12.4GB.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

STATUS = "/content/ltx23_gen_status.json"
CONFIG = "/content/gen_config.json"
CU = "/content/ComfyUI"
BASE = "http://127.0.0.1:8188"


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
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


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

    unet = add("UnetLoaderGGUF",
               {"unet_name": "ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf"})
    model = add("LTXVChunkFeedForward", {"model": [unet, 0],
                                        "chunks": 2, "dim_threshold": 4096})
    clip = add("DualCLIPLoaderGGUF",
               {"clip_name1": "gemma-3-12b-it-heretic-v2-Q3_K_M.gguf",
                "clip_name2": "ltx-2.3_text_projection_bf16.safetensors",
                "type": "ltxv"})
    pos = add("CLIPTextEncode", {"text": cfg["prompt"], "clip": [clip, 0]})
    nneg = add("CLIPTextEncode",
               {"text": cfg.get("negative", ""), "clip": [clip, 0]})
    vae_node = add("VAELoader", {"vae_name": "LTX23_video_vae_bf16.safetensors"})
    cond = add("LTXVConditioning", {"positive": [pos, 0], "negative": [nneg, 0],
                                   "frame_rate": float(cfg.get("fps", 24))})
    frames = cfg.get("frames", 49)
    latent = add("EmptyLTXVLatentVideo",
                 {"width": cfg.get("width", 576),
                  "height": cfg.get("height", 320),
                  "length": frames, "batch_size": 1})
    audio_vae = add("LTXVAudioVAELoader",
                    {"ckpt_name": "LTX23_audio_vae_bf16.safetensors"})
    audio_lat = add("LTXVEmptyLatentAudio",
                    {"frames_number": frames,
                     "frame_rate": float(cfg.get("fps", 24)),
                     "batch_size": 1, "audio_vae": [audio_vae, 0]})
    concat = add("LTXVConcatAVLatent",
                 {"video_latent": [latent, 0], "audio_latent": [audio_lat, 0]})
    noise = add("RandomNoise",
                {"noise_seed": cfg.get("seed", 42), "noise_mode": "fixed"})
    guider = add("CFGGuider", {"model": [model, 0], "positive": [cond, 0],
                               "negative": [cond, 1], "cfg": 1.0})
    ksamp = add("KSamplerSelect", {"sampler_name": "euler_cfg_pp"})
    sigmas = add("LTXVScheduler", {"steps": cfg.get("steps", 8),
                                  "max_shift": 2.05, "base_shift": 0.95,
                                  "stretch": True, "terminal": 0.1})
    sampled = add("SamplerCustomAdvanced",
                  {"noise": [noise, 0], "guider": [guider, 0],
                   "sampler": [ksamp, 0], "sigmas": [sigmas, 0],
                   "latent_image": [concat, 0]})
    sep = add("LTXVSeparateAVLatent", {"av_latent": [sampled, 0]})
    dec_v = add("LTXVTiledVAEDecode", {"vae": [vae_node, 0],
                                      "latents": [sep, 0],
                                      "horizontal_tiles": 2,
                                      "vertical_tiles": 2,
                                      "overlap": 6,
                                      "last_frame_fix": True})
    dec_a = add("LTXVAudioVAEDecode", {"samples": [sep, 1],
                                      "audio_vae": [audio_vae, 0]})
    add("VHS_VideoCombine",
        {"images": [dec_v, 0], "frame_rate": cfg.get("fps", 24),
         "loop_count": 0,
         "filename_prefix": cfg.get("out_name", "ltx23_out"),
         "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 16,
         "save_metadata": False, "trim_to_audio": True, "pingpong": False,
         "save_output": True, "audio": [dec_a, 0]})
    return p


try:
    cfg = json.load(open(CONFIG))
    if not cfg.get("prompt"):
        raise RuntimeError("gen_config.json missing 'prompt'")

    st("launch", msg="starting ComfyUI --lowvram --cache-none")
    log = open("/content/comfy.log", "a")
    subprocess.Popen(
        [sys.executable, os.path.join(CU, "main.py"),
         "--listen", "127.0.0.1", "--port", "8188",
         "--disable-auto-launch", "--lowvram", "--cache-none"],
        stdout=log, stderr=subprocess.STDOUT, cwd=CU,
        start_new_session=True)
    deadline = time.time() + 600
    while time.time() < deadline:
        try:
            info = api("GET", "/object_info", timeout=10)
            if "UnetLoaderGGUF" in info:
                break
        except Exception:
            pass
        time.sleep(5)
    info = api("GET", "/object_info", timeout=30)
    req = ["UnetLoaderGGUF", "DualCLIPLoaderGGUF", "CLIPTextEncode",
           "LTXVConditioning", "EmptyLTXVLatentVideo", "RandomNoise",
           "CFGGuider", "KSamplerSelect", "LTXVScheduler",
           "SamplerCustomAdvanced", "LTXVSeparateAVLatent", "VAELoader",
           "LTXVAudioVAELoader", "LTXVEmptyLatentAudio", "LTXVConcatAVLatent",
           "LTXVAudioVAEDecode", "LTXVTiledVAEDecode", "VHS_VideoCombine",
           "LTXVChunkFeedForward"]
    missing = [c for c in req if c not in info]
    if missing:
        raise RuntimeError("missing ComfyUI nodes: " + ", ".join(missing))

    st("queue", msg="submitting workflow", vram_mb=vram_mb())
    wf = build_workflow(cfg)
    resp = api("POST", "/prompt", {"prompt": wf}, timeout=120)
    pid = resp["prompt_id"]
    t0 = time.time()
    out_dir = os.path.join(CU, "output")
    prefix = cfg.get("out_name", "ltx23_out")
    deadline = t0 + 3600  # 60 min hard cap
    last_log = 0
    while time.time() < deadline:
        mp4s = sorted(f for f in os.listdir(out_dir)
                      if f.startswith(prefix) and f.endswith(".mp4"))
        now = time.time()
        if now - last_log > 120:
            last_log = now
            st("sampling", elapsed_s=int(now - t0), vram_mb=vram_mb(),
               outputs=mp4s)
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
                   elapsed_s=int(time.time() - t0))
                sys.exit(0)
        time.sleep(15)
    raise RuntimeError("generation timed out after 60 min")
except Exception as e:
    import traceback
    st("fatal", ready=False, error=f"{type(e).__name__}: {e}",
       tail=traceback.format_exc()[-2000:])
    sys.exit(1)
