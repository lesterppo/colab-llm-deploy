"""HiDream-I1 Full Q4_K_M text-to-image on Colab T4 via ComfyUI + ComfyUI-GGUF.

Run via exec_detach AFTER image/hidream-i1/setup.py completes.
Reads /content/gen_config.json:
  {"prompt": str, "negative": str (opt), "seed": int (opt),
   "width": int (opt), "height": int (opt), "steps": int (opt),
   "cfg": float (opt), "out_name": str (opt)}
Writes /content/gen_status.json; PNG -> /content/<out_name>.png.
Ensures ComfyUI is running (launches it detached if needed).

Proven: 1024x1024, 24 steps, cfg 5.0 on T4 (tight but workable).
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8188"
CU = "/content/ComfyUI"
STATUS = "/content/gen_status.json"
CONFIG = "/content/gen_config.json"


def st(**kw):
    open(STATUS, "w").write(json.dumps({"t": time.time(), **kw}))
    print(json.dumps(kw), flush=True)


def api(path, data=None, timeout=120):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def ensure_server():
    try:
        info = api("/object_info", timeout=10)
        if "UnetLoaderGGUF" in info:
            return info
    except Exception:
        pass
    st(stage="launch", msg="starting ComfyUI --lowvram")
    log = open("/content/comfy.log", "a")
    subprocess.Popen(
        [sys.executable, os.path.join(CU, "main.py"),
         "--listen", "127.0.0.1", "--port", "8188",
         "--disable-auto-launch", "--lowvram"],
        stdout=log, stderr=subprocess.STDOUT, cwd=CU,
        start_new_session=True)
    deadline = time.time() + 600
    while time.time() < deadline:
        try:
            info = api("/object_info", timeout=10)
            if "UnetLoaderGGUF" in info:
                return info
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError("ComfyUI did not come up")


def build_workflow(cfg):
    prompt = cfg["prompt"]
    p, nid = {}, [0]

    def add(class_type, inputs):
        nid[0] += 1
        p[str(nid[0])] = {"class_type": class_type, "inputs": inputs}
        return str(nid[0])

    unet = add("UnetLoaderGGUF",
               {"unet_name": "hidream-i1-full-Q4_K_M.gguf"})
    clip = add("QuadrupleCLIPLoader",
               {"clip_name1": "clip_l_hidream.safetensors",
                "clip_name2": "clip_g_hidream.safetensors",
                "clip_name3": "t5xxl_fp8_e4m3fn_scaled.safetensors",
                "clip_name4": "llama_3.1_8b_instruct_fp8_scaled.safetensors"})
    pos = add("CLIPTextEncodeHiDream",
              {"clip": [clip, 0], "clip_l": prompt, "clip_g": prompt,
               "t5xxl": prompt, "llama": prompt})
    neg = add("CLIPTextEncodeHiDream",
              {"clip": [clip, 0], "clip_l": "", "clip_g": "",
               "t5xxl": "", "llama": ""})
    samp_prep = add("ModelSamplingSD3", {"model": [unet, 0], "shift": 3.0})
    latent = add("EmptySD3LatentImage",
                 {"width": cfg.get("width", 1024),
                  "height": cfg.get("height", 1024), "batch_size": 1})
    samp = add("KSampler",
               {"model": [samp_prep, 0], "positive": [pos, 0],
                "negative": [neg, 0], "latent_image": [latent, 0],
                "seed": cfg.get("seed", 42), "steps": cfg.get("steps", 24),
                "cfg": cfg.get("cfg", 5.0), "sampler_name": "euler",
                "scheduler": "normal", "denoise": 1.0})
    vae = add("VAELoader", {"vae_name": "ae.safetensors"})
    dec = add("VAEDecode", {"samples": [samp, 0], "vae": [vae, 0]})
    add("SaveImage", {"images": [dec, 0],
                      "filename_prefix": cfg.get("out_name", "result")})
    return p


try:
    cfg = json.load(open(CONFIG))
    if not cfg.get("prompt"):
        raise RuntimeError("gen_config.json missing 'prompt'")
    st(stage="check", msg="ensuring ComfyUI is up")
    ensure_server()

    st(stage="submit")
    wf = build_workflow(cfg)
    resp = api("/prompt", {"prompt": wf})
    pid = resp["prompt_id"]
    t0 = time.time()
    st(stage="generating", prompt_id=pid)
    deadline = time.time() + 5400
    out = None
    while time.time() < deadline:
        time.sleep(20)
        try:
            h = api("/history/" + pid)
        except Exception as e:
            st(stage="poll_error", error=str(e)[:200])
            continue
        if pid in h:
            out = h[pid]
            break
    if out is None:
        raise RuntimeError("timed out waiting for generation")
    png_rel = None
    for node_id, node_out in out.items():
        for img in node_out.get("images", []):
            if img["filename"].endswith(".png"):
                png_rel = (img.get("subfolder", ""), img["filename"])
    if not png_rel:
        raise RuntimeError("no png in outputs")
    sub, fn = png_rel
    src = os.path.join(CU, "output", sub, fn)
    dst = f"/content/{cfg.get('out_name', 'result')}.png"
    import shutil
    shutil.copy(src, dst)
    st(stage="done", ready=True, png=dst, size_b=os.path.getsize(dst),
       gen_seconds=round(time.time() - t0))
except Exception as e:
    import traceback
    st(stage="fatal", ready=False, error=f"{type(e).__name__}: {e}",
       tail=traceback.format_exc()[-2000:])
    raise
