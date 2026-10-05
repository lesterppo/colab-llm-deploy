"""Chatterbox (ResembleAI TTS 0.5B) setup on Colab T4.
pip install + env fixes + weight download. Stage-gated, writes
/content/tts_setup_status.json. Run via exec_detach.

Fixes (learned live 2026-10-05):
- pip can pull torchvision newer than Colab's torch (torchvision::nms
  missing -> transformers LlamaModel import fails): pin
  torchvision==0.21.0 to match torch 2.6.0.
- Colab ships torchao 0.10; peft needs >0.16 -> uninstall torchao.
"""
import json
import subprocess
import sys
import time

STATUS = "/content/tts_setup_status.json"


def st(stage, **kw):
    d = {"stage": stage, "t": time.time(), **kw}
    open(STATUS, "w").write(json.dumps(d))
    print(json.dumps(d), flush=True)


def run(cmd, timeout=1800, **kw):
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, timeout=timeout, **kw)


try:
    st("install", msg="pip install chatterbox-tts + hf CLI")
    run([sys.executable, "-m", "pip", "install", "-q",
         "chatterbox-tts", "huggingface_hub[cli]"], timeout=1800)

    st("fixenv", msg="align torchvision with Colab torch, drop torchao")
    run([sys.executable, "-m", "pip", "install", "-q",
         "torchvision==0.21.0"], timeout=900)
    run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
        timeout=300)

    st("weights", msg="hf download ResembleAI/chatterbox")
    r = subprocess.run(
        ["hf", "download", "ResembleAI/chatterbox"],
        capture_output=True, text=True, timeout=3600)
    print(r.stdout[-2000:], flush=True)
    if r.returncode != 0:
        err = (r.stderr or "")[-1500:]
        if "401" in err or "Unauthorized" in err or "gated" in err.lower():
            st("blocked", ready=False,
               error="HF 401/gated on ResembleAI/chatterbox; need HF token")
            sys.exit(2)
        raise RuntimeError(f"hf download failed: {err}")

    st("done", ready=True)
except SystemExit:
    raise
except Exception as e:
    import traceback
    st("fatal", ready=False, error=f"{type(e).__name__}: {e}",
       tail=traceback.format_exc()[-2000:])
    sys.exit(1)
