"""Whisper large-v3-turbo setup on Colab T4 for speech-to-text.
Stage-gated, writes /content/stt_setup_status.json. Run via exec_detach.

VRAM plan (15GB T4): whisper-large-v3-turbo is 809M params -> ~2GB in fp16.
torch is preinstalled on Colab; pin transformers<5.0 (5.x breaks things).
"""
import json
import os
import subprocess
import sys

STATUS = "/content/stt_setup_status.json"
MODEL = "openai/whisper-large-v3-turbo"


def st(stage, **kw):
    d = {"stage": stage, **kw}
    open(STATUS, "w").write(json.dumps(d))
    print(json.dumps(d), flush=True)


def run(cmd, timeout=1800, **kw):
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, timeout=timeout, **kw)


try:
    st("pip", msg="installing transformers + accelerate + hf CLI")
    run([sys.executable, "-m", "pip", "install", "-q",
         "transformers==4.48.0", "accelerate", "soundfile", "librosa",
         "huggingface_hub[cli]"], timeout=1500)

    st("dl_model", msg=f"downloading {MODEL}")
    out = run(["hf", "download", MODEL], timeout=1800,
              capture_output=True, text=True)
    cache_dir = out.stdout.strip().splitlines()[-1].strip()
    st("dl_model_done", cache_dir=cache_dir)

    st("verify", msg="import check")
    run([sys.executable, "-c",
         "from transformers import pipeline; print('transformers OK')"],
        timeout=300)

    st("done", ready=True, model=MODEL, cache_dir=cache_dir)
except Exception as e:
    import traceback
    st("fatal", ready=False, error=f"{type(e).__name__}: {e}",
       tail=traceback.format_exc()[-2000:])
    sys.exit(1)
