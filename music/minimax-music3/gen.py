"""MiniMax Music 3 song generation via local ComfyUI API.
Reads /content/gen_config.json (caption, lyrics, seed, duration, out_name)
written by deploy_media.py; defaults reproduce the tested lighthouse song.
Node graph from lesterppo/minimax-music3-colab AGENTS.md (verified on 0.31.0),
with probe-and-fallback for the SaveAudio node.
Writes /content/music3_gen_status.json; audio -> /content/<out_name>.mp3.
Run via exec_detach AFTER setup.py has ComfyUI up.
"""
import json
import os
import shutil
import subprocess
import time
import urllib.request

BASE = "http://127.0.0.1:8188"
CU = "/content/ComfyUI"
STATUS = "/content/music3_gen_status.json"
CONFIG = "/content/gen_config.json"

DEFAULT_CAPTION = ("Genre: cinematic folk. BPM: 90. Warm male vocal, "
                   "acoustic guitar, soft strings, gentle percussion. "
                   "Intimate and hopeful.")
DEFAULT_LYRICS = """[Verse]
Golden light on the water, waves are rolling low
The lighthouse keeps its watch where the sea winds blow
[Chorus]
Shine on, shine on through the night
Guide me home with your light"""


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


def vram_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out.split()[0])
    except Exception:
        return -1


def build_workflow(cfg, save_node):
    cap = cfg.get("caption", DEFAULT_CAPTION)
    lyr = cfg.get("lyrics", DEFAULT_LYRICS)
    seed = cfg.get("seed", 7)
    dur = cfg.get("duration", 30)
    p, nid = {}, [0]

    def add(class_type, inputs):
        nid[0] += 1
        p[str(nid[0])] = {"class_type": class_type, "inputs": inputs}
        return str(nid[0])

    clip = add("CLIPLoader",
               {"clip_name":
                "minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
                "type": "minimax"})
    unet = add("UNETLoader",
               {"unet_name": "minimax_music3_dit_int8_convrot.safetensors",
                "weight_dtype": "default"})
    vae = add("VAELoader", {"vae_name": "minimax_music3_dav.safetensors"})
    enc = add("MiniMaxMusic3TextEncode",
              {"clip": [clip, 0], "caption": cap, "lyrics": lyr,
               "seed": seed, "max_duration": dur, "cfg_scale": 1.7,
               "top_k": 50})
    zero = add("ConditioningZeroOut", {"conditioning": [enc, 0]})
    latent = add("EmptyMiniMaxMusic3LatentAudio",
                 {"seconds": [enc, 1], "batch_size": 1})
    samp = add("KSampler",
               {"model": [unet, 0], "positive": [enc, 0],
                "negative": [zero, 0], "latent_image": [latent, 0],
                "seed": seed, "steps": 30, "cfg": 1.7,
                "sampler_name": "euler", "scheduler": "simple",
                "denoise": 1.0})
    dec = add("VAEDecodeAudio", {"samples": [samp, 0], "vae": [vae, 0]})
    if save_node == "SaveAudioAdvanced":
        add("SaveAudioAdvanced",
            {"audio": [dec, 0],
             "filename_prefix": "audio/" + cfg.get("out_name", "music3_out"),
             "format": "mp3", "format.quality": "V0"})
    else:
        add("SaveAudio",
            {"audio": [dec, 0],
             "filename_prefix": cfg.get("out_name", "music3_out")})
    return p


try:
    cfg = {}
    if os.path.exists(CONFIG):
        cfg = json.load(open(CONFIG))

    st(stage="wait_comfy", msg="waiting for ComfyUI /system_stats")
    deadline = time.time() + 1800
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(BASE + "/system_stats",
                                        timeout=10) as r:
                if '"devices"' in r.read().decode():
                    break
        except Exception:
            pass
        time.sleep(10)
    else:
        raise RuntimeError("ComfyUI never came up")

    info = api("/object_info", timeout=60)
    for need in ("CLIPLoader", "UNETLoader", "VAELoader",
                 "MiniMaxMusic3TextEncode", "ConditioningZeroOut",
                 "EmptyMiniMaxMusic3LatentAudio", "KSampler",
                 "VAEDecodeAudio"):
        if need not in info:
            raise RuntimeError(f"missing node: {need}")
    save_node = ("SaveAudioAdvanced" if "SaveAudioAdvanced" in info
                 else "SaveAudio")
    st(stage="nodes", save_node=save_node)

    st(stage="submit", vram_mb=vram_mb())
    wf = build_workflow(cfg, save_node)
    resp = api("/prompt", {"prompt": wf}, timeout=120)
    pid = resp["prompt_id"]
    t0 = time.time()
    st(stage="sampling", prompt_id=pid)
    deadline = t0 + 5400  # 90 min hard cap
    peak, last_log, out = -1, 0, None
    while time.time() < deadline:
        peak = max(peak, vram_mb())
        now = time.time()
        if now - last_log > 300:
            last_log = now
            st(stage="sampling", elapsed_s=int(now - t0), vram_peak_mb=peak)
        try:
            h = api("/history/" + pid, timeout=30)
        except Exception as e:
            st(stage="poll_error", error=str(e)[:200])
            time.sleep(30)
            continue
        if pid in h:
            if h[pid].get("status", {}).get("status_str") == "error":
                raise RuntimeError(
                    f"prompt errored: {json.dumps(h[pid])[:800]}")
            out = h[pid].get("outputs")
            if out:
                break
        time.sleep(30)
    if not out:
        raise RuntimeError("timed out waiting for generation")
    audio_rel = None
    for node_id, node_out in out.items():
        for a in node_out.get("audio", []):
            if a["filename"].endswith((".mp3", ".wav", ".flac", ".ogg")):
                audio_rel = (a.get("subfolder", ""), a["filename"])
    if not audio_rel:
        raise RuntimeError(f"no audio in outputs: {json.dumps(out)[:500]}")
    sub, fn = audio_rel
    src = os.path.join(CU, "output", sub, fn)
    dst = f"/content/{cfg.get('out_name', 'music3_out')}{os.path.splitext(fn)[1]}"
    shutil.copy(src, dst)
    st(stage="done", ready=True, audio=dst, size_b=os.path.getsize(dst),
       gen_seconds=round(time.time() - t0), vram_peak_mb=peak)
except Exception as e:
    import traceback
    st(stage="fatal", ready=False, error=f"{type(e).__name__}: {e}",
       tail=traceback.format_exc()[-2000:])
    raise
