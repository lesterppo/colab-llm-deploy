"""YuE2-3B song generation on the VM via ComfyUI API.
Run via exec_detach AFTER setup_yue2.py completes.
Workflow: YuE2Options(max_seconds=40, download=off) -> YuE2GenerateSong
(style/lyrics/seed) -> SaveAudio.
Reads /content/gen_config.json (style, lyrics, seed, max_seconds, out_name).
Writes /content/yue2_gen_status.json; audio -> /content/<out_name>.<ext>.
Launches ComfyUI itself if not running.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

STATUS = "/content/yue2_gen_status.json"
CONFIG = "/content/gen_config.json"
CU = "/content/ComfyUI"
BASE = "http://127.0.0.1:8188"
OUT = os.path.join(CU, "output")

DEFAULT_STYLE = ("cinematic folk, acoustic guitar and strings, "
                 "warm male vocal, 90 bpm")
DEFAULT_LYRICS = """[Verse]
Golden light on the water, waves are rolling in
The lighthouse stands above it all, where the evening begins
[Chorus]
Shine on, shine on through the night
Guide me home with your light"""


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

    opts = add("YuE2Options",
               {"cot": "full",
                "max_seconds": float(cfg.get("max_seconds", 40)),
                "keep_model_loaded": False,
                "download": "off",
                "attention_backend": "sdpa"})
    song = add("YuE2GenerateSong",
               {"style": cfg.get("style", DEFAULT_STYLE),
                "lyrics": cfg.get("lyrics", DEFAULT_LYRICS),
                "seed": cfg.get("seed", 42),
                "options": [opts, 0]})
    add("SaveAudio",
        {"audio": [song, 0],
         "filename_prefix": cfg.get("out_name", "yue2_song")})
    return p


try:
    cfg = {}
    if os.path.exists(CONFIG):
        cfg = json.load(open(CONFIG))

    st("launch", msg="starting ComfyUI")
    comfy_up = False
    try:
        info = api("GET", "/object_info", timeout=10)
        comfy_up = "YuE2GenerateSong" in info
    except Exception:
        pass
    if not comfy_up:
        log = open("/content/comfy.log", "a")
        subprocess.Popen(
            [sys.executable, os.path.join(CU, "main.py"),
             "--listen", "127.0.0.1", "--port", "8188",
             "--disable-auto-launch"],
            stdout=log, stderr=subprocess.STDOUT, cwd=CU,
            start_new_session=True)
        deadline = time.time() + 900
        while time.time() < deadline:
            try:
                info = api("GET", "/object_info", timeout=10)
                if "YuE2GenerateSong" in info:
                    comfy_up = True
                    break
            except Exception:
                pass
            time.sleep(5)
    if not comfy_up:
        raise RuntimeError("ComfyUI did not come up with YuE2 nodes")

    st("queue", msg="submitting song workflow", vram_mb=vram_mb())
    wf = build_workflow(cfg)
    resp = api("POST", "/prompt", {"prompt": wf}, timeout=120)
    pid = resp["prompt_id"]
    t0 = time.time()
    prefix = cfg.get("out_name", "yue2_song")
    deadline = t0 + 5400  # 90 min hard cap (T4 has no BF16; acoustic stage slower)
    last_log = 0
    while time.time() < deadline:
        auds = sorted(f for f in os.listdir(OUT)
                      if f.startswith(prefix)
                      and (f.endswith(".wav") or f.endswith(".mp3")
                           or f.endswith(".flac") or f.endswith(".ogg")))
        now = time.time()
        if now - last_log > 180:
            last_log = now
            st("singing", elapsed_s=int(now - t0), vram_mb=vram_mb(),
               outputs=auds)
        done = False
        try:
            h = api("GET", f"/history/{pid}", timeout=20)
            if pid in h:
                done = True
        except Exception:
            pass
        if done and auds:
            path = os.path.join(OUT, auds[-1])
            s1 = os.path.getsize(path)
            time.sleep(10)
            s2 = os.path.getsize(path)
            stable = 0
            while s1 != s2 and stable < 4:
                stable += 1
                time.sleep(10)
                s1, s2 = s2, os.path.getsize(path)
            if s1 == s2 and s1 > 10_000:
                ext = os.path.splitext(path)[1]
                dst = f"/content/{prefix}{ext}"
                shutil.copy(path, dst)
                el = int(time.time() - t0)
                st("done", ready=True, audio=dst,
                   size_mb=round(s1 / 1048576, 2), elapsed_s=el,
                   vram_mb=vram_mb())
                sys.exit(0)
        time.sleep(20)
    raise RuntimeError("song generation timed out after 90 min")
except Exception as e:
    import traceback
    st("fatal", ready=False, error=f"{type(e).__name__}: {e}",
       tail=traceback.format_exc()[-2000:])
    sys.exit(1)
