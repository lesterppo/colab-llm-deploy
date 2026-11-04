#!/usr/bin/env python3
"""Brush lighthouse: VM setup (run detached on a Colab T4).

Proven 2026-11-04 on a free Colab T4 (session brush-lighthouse). Two
things a stock Colab VM lacks, both fixed here, in order:

1. A Vulkan ICD for the GPU. The VM has the CUDA driver (nvidia-smi
   works) but only the Vulkan loader — no ICD file, so Brush's wgpu
   backend panics with "No possible adapter available" before step 0.
   Fix: install the libnvidia-gl package matching the LOADED driver
   series (read from nvidia-smi, never guessed) plus vulkan-tools, then
   verify an ICD file exists and vulkaninfo reports the T4. This does
   NOT touch the kernel driver.
2. The Brush 0.3.0 prebuilt Linux binary, fetched via the GitHub
   releases API (asset picked by name, never a guessed URL), size
   verified against the API record, extracted, and smoke-checked with
   --help. Do NOT build from source on Colab: main-branch deps need
   rustc >= 1.95 and rustup on Colab gives 1.88 (proven dead end).

Writes /content/brush_setup_status.json in the repo's status shape:
{"stage": ..., "ready": bool, "error": ..., "tail": ..., "exe": ...}.
Idempotent: if the ICD and the binary are already present, it verifies
them and exits ready without reinstalling.
"""
import glob, json, os, subprocess, sys, tarfile, time, urllib.request

STATUS = "/content/brush_setup_status.json"
LOG = open("/content/brush_setup.log", "ab", buffering=0)
T0 = time.time()


def log(msg):
    line = f"[{time.time() - T0:8.1f}s] {msg}\n"
    LOG.write(line.encode()); print(line, end="", flush=True)


def status(stage, ready=False, error="", **kw):
    d = {"stage": stage, "ready": ready, "error": error,
         "tail": "", "elapsed_s": round(time.time() - T0, 1)}
    d.update(kw)
    with open(STATUS, "w") as f:
        json.dump(d, f, indent=2)


def run(cmd, timeout=1200):
    log("$ " + " ".join(str(c) for c in cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=timeout)
    log(p.stdout[-2000:] if p.stdout else "")
    return p


def fatal(msg):
    log("FATAL: " + msg)
    status("fatal", ready=False, error=msg)
    sys.exit(2)


def find_exe():
    hits = glob.glob("/content/brush/**/brush_app", recursive=True)
    return hits[0] if hits else None


def icd_ok():
    for d in ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d"):
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*nvidia*icd*.json")):
            return True
    return False


status("preflight")
p = run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
if p.returncode != 0:
    fatal("nvidia-smi failed — no GPU on this session")
gpu, driver = [x.strip() for x in p.stdout.strip().split(",")]
series = driver.split(".")[0]
log(f"GPU {gpu}, driver {driver}, series {series}")

# --- Vulkan ICD -------------------------------------------------------
if not icd_ok():
    status("vulkan")
    run(["bash", "-c", "apt-get update 2>&1 | tail -3"], timeout=900)
    p = run(["bash", "-c",
             f"apt-cache policy libnvidia-gl-{series} 2>&1 | head -20"])
    if series not in p.stdout:
        fatal(f"no libnvidia-gl-{series} package available for driver "
              f"{driver}; refusing to install a mismatched series")
    p = run(["bash", "-c",
             "DEBIAN_FRONTEND=noninteractive apt-get install -y "
             f"--no-install-recommends libnvidia-gl-{series} vulkan-tools "
             "2>&1 | tail -20"], timeout=1200)
    if p.returncode != 0 or not icd_ok():
        fatal("Vulkan ICD install failed or no ICD file appeared")
    run(["bash", "-c",
         "vulkaninfo 2>&1 | grep -E 'GPU|deviceName|driverVersion' | head -10"])
else:
    log("Vulkan ICD already present, skipping apt install")

# --- Brush 0.3.0 binary ----------------------------------------------
exe = find_exe()
if not exe:
    status("download_brush")
    api = "https://api.github.com/repos/ArthurBrussee/brush/releases/tags/v0.3.0"
    try:
        req = urllib.request.Request(api, headers={"User-Agent": "colab-llm-deploy"})
        rel = json.load(urllib.request.urlopen(req, timeout=60))
        asset = next(a for a in rel["assets"]
                     if "x86_64-unknown-linux-gnu" in a["name"])
        url, name, size = (asset["browser_download_url"], asset["name"],
                           asset["size"])
    except Exception as e:
        fatal(f"could not resolve Brush 0.3.0 release asset: {e}")
    log(f"asset: {name} size={size}")
    urllib.request.urlretrieve(url, f"/content/{name}")
    if os.path.getsize(f"/content/{name}") != size:
        fatal("Brush download size mismatch — refusing a partial binary")
    os.makedirs("/content/brush", exist_ok=True)
    with tarfile.open(f"/content/{name}") as tf:
        tf.extractall("/content/brush")
    exe = find_exe()
    if not exe:
        fatal("Brush tarball extracted but no brush_app binary found")
    os.chmod(exe, 0o755)
else:
    log(f"Brush binary already present: {exe}")

p = run([exe, "--help"])
if "--total-steps" not in p.stdout:
    fatal("Brush --help does not show --total-steps; wrong binary version "
          "(this recipe is pinned to the 0.3.0 prebuilt app)")
log("setup ready")
status("ready", ready=True, exe=exe)
