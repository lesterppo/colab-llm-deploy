"""Launch ComfyUI server detached on the Colab VM (HiDream run).
--lowvram keeps the 4 text encoders (~15.9GB) on CPU (mmap'd) instead of
trying to fit them in 15GB VRAM alongside the 9.4GB DiT."""
import subprocess
import sys

log = open("/content/comfy.log", "a")
p = subprocess.Popen(
    [sys.executable, "/content/ComfyUI/main.py",
     "--listen", "127.0.0.1", "--port", "8188",
     "--disable-auto-launch", "--lowvram"],
    stdout=log, stderr=subprocess.STDOUT,
    cwd="/content/ComfyUI", start_new_session=True)
print(f"comfyui pid={p.pid}", flush=True)
