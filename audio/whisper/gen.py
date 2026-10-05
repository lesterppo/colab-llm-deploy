"""Whisper large-v3-turbo transcription on Colab T4.
Run via exec_detach AFTER setup.py completes.
Reads /content/gen_config.json:
  {"audio_path": str, "out_name": str (opt)}
Writes /content/stt_gen_status.json; text -> /content/<out_name>.txt.
"""
import json
import os
import subprocess
import sys
import time

STATUS = "/content/stt_gen_status.json"
CONFIG = "/content/gen_config.json"
MODEL = "openai/whisper-large-v3-turbo"


def st(stage, **kw):
    d = {"stage": stage, **kw}
    open(STATUS, "w").write(json.dumps(d))
    print(json.dumps(d), flush=True)


def vram_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out.split()[0])
    except Exception:
        return -1


try:
    cfg = json.load(open(CONFIG))
    audio_path = cfg.get("audio_path", "/content/ref.wav")
    out_name = cfg.get("out_name", "transcription")
    if not os.path.exists(audio_path):
        raise RuntimeError(f"audio not found: {audio_path}")

    st("load", msg=f"loading {MODEL} on cuda fp16")
    import torch
    from transformers import pipeline
    t0 = time.time()
    pipe = pipeline("automatic-speech-recognition", model=MODEL,
                    device="cuda", torch_dtype=torch.float16)
    load_s = round(time.time() - t0, 1)
    st("loaded", load_s=load_s, vram_mb=vram_mb())

    st("transcribe", msg=f"transcribing {audio_path}")
    t0 = time.time()
    result = pipe(audio_path)
    text = result["text"].strip()
    gen_s = round(time.time() - t0, 1)

    dst = f"/content/{out_name}.txt"
    open(dst, "w").write(text + "\n")
    st("done", ready=True, transcription=text,
       word_count=len(text.split()), gen_s=gen_s,
       vram_mb=vram_mb(), txt=dst)
except Exception as e:
    import traceback
    st("fatal", ready=False, error=f"{type(e).__name__}: {e}",
       tail=traceback.format_exc()[-2000:])
    sys.exit(1)
