"""Launch ComfyUI server detached on the Colab VM."""
import subprocess
import sys

log = open("/content/comfy.log", "a")
p = subprocess.Popen(
    [sys.executable, "/content/ComfyUI/main.py",
     "--listen", "127.0.0.1", "--port", "8188",
     "--disable-auto-launch"],
    stdout=log, stderr=subprocess.STDOUT,
    cwd="/content/ComfyUI", start_new_session=True)
print(f"comfyui pid={p.pid}", flush=True)
