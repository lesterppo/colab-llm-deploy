"""LTX-2.3 chained multi-clip generation on Colab T4.

Run via exec_detach AFTER video/ltx23/setup.py completes.
Reads /content/gen_config.json:
  {"prompt": str, "negative": str (opt), "seed": int (opt),
   "width": int (opt), "height": int (opt), "frames": int (opt),
   "fps": int (opt), "steps": int (opt), "clips": int (opt),
   "out_name": str (opt)}
Clip 1: text-to-video. Clips 2..N: first-frame conditioned on the last
frame of the previous clip (LTXVImgToVideoConditionOnly), with automatic
prompt progression suffixes. Concatenated with ffmpeg -c copy.
Writes /content/ltx23_chain_status.json; final -> /content/<out_name>.mp4.

Proven: 5 clips x (576x320, 49f @24fps, 8 steps) = ~10.2s in ~38 min on T4.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

STATUS = "/content/ltx23_chain_status.json"
CONFIG = "/content/gen_config.json"
CU = "/content/ComfyUI"
BASE = "http://127.0.0.1:8188"
OUT = os.path.join(CU, "output")
CLIPDIR = "/content/ltx23/clips"

PROGRESSION = [
    "",
    ", camera slowly drifting right",
    ", waves swelling higher against the rocks",
    ", camera slowly drifting left, clouds moving",
    ", golden light intensifying, spray rising",
    ", camera pushing in slowly",
    ", gentle pan across the scene",
    ", light shifting, shadows lengthening",
]


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


def sh(cmd, timeout=600):
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, timeout=timeout,
                          capture_output=True, text=True)


def build_base(cfg, seed):
    """Shared loader/conditioning/sampler graph. Returns (p, add, refs)."""
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
    audio_vae = add("LTXVAudioVAELoader",
                    {"ckpt_name": "LTX23_audio_vae_bf16.safetensors"})
    audio_lat = add("LTXVEmptyLatentAudio",
                    {"frames_number": frames,
                     "frame_rate": float(cfg.get("fps", 24)),
                     "batch_size": 1, "audio_vae": [audio_vae, 0]})
    noise = add("RandomNoise", {"noise_seed": seed, "noise_mode": "fixed"})
    guider = add("CFGGuider", {"model": [model, 0], "positive": [cond, 0],
                               "negative": [cond, 1], "cfg": 1.0})
    ksamp = add("KSamplerSelect", {"sampler_name": "euler_cfg_pp"})
    sigmas = add("LTXVScheduler", {"steps": cfg.get("steps", 8),
                                  "max_shift": 2.05, "base_shift": 0.95,
                                  "stretch": True, "terminal": 0.1})
    refs = {"add": add, "model": model, "guider": guider, "sampler": ksamp,
            "sigmas": sigmas, "noise": noise, "vae_node": vae_node,
            "audio_vae": audio_vae, "audio_lat": audio_lat, "frames": frames}
    return p, refs


def finalize(p, refs, cfg, latent_ref, prefix):
    add = refs["add"]
    concat = add("LTXVConcatAVLatent",
                 {"video_latent": latent_ref,
                  "audio_latent": [refs["audio_lat"], 0]})
    sampled = add("SamplerCustomAdvanced",
                  {"noise": [refs["noise"], 0], "guider": [refs["guider"], 0],
                   "sampler": [refs["sampler"], 0],
                   "sigmas": [refs["sigmas"], 0],
                   "latent_image": [concat, 0]})
    sep = add("LTXVSeparateAVLatent", {"av_latent": [sampled, 0]})
    dec_v = add("LTXVTiledVAEDecode", {"vae": [refs["vae_node"], 0],
                                      "latents": [sep, 0],
                                      "horizontal_tiles": 2,
                                      "vertical_tiles": 2,
                                      "overlap": 6,
                                      "last_frame_fix": True})
    dec_a = add("LTXVAudioVAEDecode", {"samples": [sep, 1],
                                      "audio_vae": [refs["audio_vae"], 0]})
    add("VHS_VideoCombine",
        {"images": [dec_v, 0], "frame_rate": cfg.get("fps", 24),
         "loop_count": 0, "filename_prefix": prefix,
         "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 16,
         "save_metadata": False, "trim_to_audio": True, "pingpong": False,
         "save_output": True, "audio": [dec_a, 0]})
    return p


def build_t2v(cfg, prompt, seed, prefix):
    p, refs = build_base(cfg, seed)
    add = refs["add"]
    latent = add("EmptyLTXVLatentVideo",
                 {"width": cfg.get("width", 576),
                  "height": cfg.get("height", 320),
                  "length": refs["frames"], "batch_size": 1})
    return finalize(p, refs, cfg, [latent, 0], prefix)


def build_i2v(cfg, prompt, seed, prefix, frame_png):
    p, refs = build_base(cfg, seed)
    add = refs["add"]
    loadimg = add("LoadImage", {"image": os.path.basename(frame_png)})
    empty = add("EmptyLTXVLatentVideo",
                {"width": cfg.get("width", 576),
                 "height": cfg.get("height", 320),
                 "length": refs["frames"], "batch_size": 1})
    latent = add("LTXVImgToVideoConditionOnly",
                 {"vae": [refs["vae_node"], 0], "image": [loadimg, 0],
                  "latent": [empty, 0], "strength": 1.0})
    return finalize(p, refs, cfg, [latent, 0], prefix)


def wait_for_clip(prefix, timeout_s=2400):
    t0 = time.time()
    last_log = 0
    while time.time() - t0 < timeout_s:
        cands = sorted(f for f in os.listdir(OUT)
                       if f.startswith(prefix) and f.endswith(".mp4"))
        now = time.time()
        if now - last_log > 180:
            last_log = now
            st("sampling", clip=prefix, elapsed_s=int(now - t0),
               outputs=cands)
        if cands:
            path = os.path.join(OUT, cands[-1])
            s1 = os.path.getsize(path)
            time.sleep(15)
            s2 = os.path.getsize(path)
            stable = 0
            while s1 != s2 and stable < 6:
                stable += 1
                time.sleep(15)
                s1, s2 = s2, os.path.getsize(path)
            if s1 == s2 and s1 > 100_000:
                return path, int(time.time() - t0)
        time.sleep(20)
    raise RuntimeError(f"clip {prefix} timed out")


def extract_last_frame(mp4, png):
    sh(["ffmpeg", "-y", "-sseof", "-0.5", "-i", mp4,
        "-vframes", "1", png], timeout=120)


try:
    cfg = json.load(open(CONFIG))
    if not cfg.get("prompt"):
        raise RuntimeError("gen_config.json missing 'prompt'")
    n_clips = max(1, int(cfg.get("clips", 5)))
    base_prompt = cfg["prompt"]
    seed0 = cfg.get("seed", 42)
    out_name = cfg.get("out_name", "ltx23_chain")
    os.makedirs(CLIPDIR, exist_ok=True)

    try:
        sh(["ffmpeg", "-version"], timeout=30)
    except Exception:
        st("ffmpeg_install", msg="installing ffmpeg via apt")
        sh(["apt-get", "update", "-qq"], timeout=600)
        sh(["apt-get", "install", "-y", "-qq", "ffmpeg"], timeout=900)

    st("launch", msg="starting ComfyUI --lowvram --cache-none")
    comfy_up = False
    try:
        info = api("GET", "/object_info", timeout=10)
        comfy_up = "UnetLoaderGGUF" in info
    except Exception:
        pass
    if not comfy_up:
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
                    comfy_up = True
                    break
            except Exception:
                pass
            time.sleep(5)
    if not comfy_up:
        raise RuntimeError("ComfyUI did not come up")

    clip_paths, clip_times = [], []
    for i in range(n_clips):
        suffix = PROGRESSION[i] if i < len(PROGRESSION) else ""
        prompt = base_prompt + suffix
        seed = seed0 + i * 131
        prefix = f"{out_name}_clip{i+1:02d}"
        st("clip_start", clip=i + 1, total=n_clips, seed=seed)
        if i > 0:
            wf = build_i2v(cfg, prompt, seed, prefix,
                           os.path.join(CU, "input",
                                        f"chain_frame{i:02d}.png"))
        else:
            wf = build_t2v(cfg, prompt, seed, prefix)
        t0 = time.time()
        api("POST", "/prompt", {"prompt": wf}, timeout=120)
        path, _ = wait_for_clip(prefix)
        dst = os.path.join(CLIPDIR, f"clip_{i+1:02d}.mp4")
        os.replace(path, dst)
        clip_paths.append(dst)
        clip_times.append(int(time.time() - t0))
        st("clip_done", clip=i + 1, path=dst, elapsed_s=clip_times[-1])
        if i < n_clips - 1:
            frame_dst = os.path.join(CU, "input", f"chain_frame{i+1:02d}.png")
            extract_last_frame(dst, frame_dst)

    st("concat", msg=f"ffmpeg concat of {n_clips} clips")
    lst = os.path.join(CLIPDIR, "concat.txt")
    with open(lst, "w") as f:
        for cp in clip_paths:
            f.write(f"file '{cp}'\n")
    final = os.path.join(OUT, f"{out_name}.mp4")
    sh(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
        "-c", "copy", final], timeout=600)
    r = sh(["ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "csv=p=0", final], timeout=60)
    dur = float(r.stdout.strip())
    dst_final = f"/content/{out_name}.mp4"
    import shutil
    shutil.copy(final, dst_final)
    st("done", ready=True, final=dst_final,
       duration_s=round(dur, 2), clip_times_s=clip_times,
       clips=clip_paths)
except Exception as e:
    import traceback
    st("fatal", ready=False, error=f"{type(e).__name__}: {e}",
       tail=traceback.format_exc()[-2000:])
    sys.exit(1)
