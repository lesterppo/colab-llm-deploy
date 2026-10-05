"""Optional: expose a running ComfyUI (music pipelines) via cloudflared.

One-shot deploys do not need this — deploy_media.py downloads the finished
audio directly. Run this detached only if you want a public URL to drive the
ComfyUI API from outside (mirrors the tunnel pattern in deploy_llm.py).
Writes the tunnel URL to /content/music_url.txt.
"""
import re
import subprocess
import sys
import time

PORT = 8188
LOG = "/content/cloudflared.log"


def running():
    r = subprocess.run("pgrep -f '[c]loudflared'", shell=True,
                       capture_output=True, text=True)
    return r.returncode == 0


def main():
    cf = "/usr/local/bin/cloudflared"
    if not (subprocess.run(["test", "-x", cf]).returncode == 0):
        subprocess.run(
            f"wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O {cf} && chmod +x {cf}",
            shell=True, check=True, timeout=300)
    if not running():
        subprocess.Popen(
            [cf, "tunnel", "--url", f"http://127.0.0.1:{PORT}",
             "--no-autoupdate"],
            stdout=open(LOG, "w"), stderr=subprocess.STDOUT,
            start_new_session=True)
        print("cloudflared launched", flush=True)
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            txt = open(LOG).read()
        except FileNotFoundError:
            txt = ""
        m = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', txt)
        if m:
            url = m.group(0)
            open("/content/music_url.txt", "w").write(url)
            print(f"TUNNEL: {url}", flush=True)
            return
        time.sleep(2)
    sys.exit("cloudflared never handed out a URL; check /content/cloudflared.log")


if __name__ == "__main__":
    main()
