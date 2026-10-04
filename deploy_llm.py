#!/usr/bin/env python3 -u
"""colab-llm-deploy VM driver — one-shot LLM serving on a Colab GPU.

Driven headlessly by lesterppo/hermes-colab-cli via `exec_detach`
(upload + run detached, returns immediately). This script runs the full
lifecycle on the VM: stage-gated install -> model acquire -> serve ->
Cloudflare tunnel -> watchdog.

Stage machine-readable status is written to /content/deploy_status.json at
EVERY stage (exec_detach gives no failure signal, so the orchestrator polls
this file). Tunnel URL goes to /content/api_url.txt and /content/tunnel.log
(both covered by hermes-colab-cli's tunnel_discover path list).

Config: /content/deploy_config.json
  {"backend": "ollama"|"vllm", "model": "<catalog key in models.json>",
   "hf_token": "<optional, file only>", "port": 8000, "max_model_len": 4096}
Catalog: /content/models.json (uploaded alongside by the orchestrator).

Stdlib only. No secrets are ever printed.
"""
import subprocess, sys, os, time, re, json, shutil, urllib.request

PORT_DEFAULT = 8000
CONFIG_FILE = "/content/deploy_config.json"
MODELS_FILE = "/content/models.json"
STATUS_FILE = "/content/deploy_status.json"
TUNNEL_LOG = "/content/tunnel.log"
API_URL_FILE = "/content/api_url.txt"
DEPLOY_LOG = "/content/deploy_llm.log"

# Pinned versions (mirror versions.env)
OLLAMA_VER = "0.34.4"
CLOUDFLARED_VER = "2026.9.3"
VLLM_VER = "0.30.0"
OLLAMA_BIN = "/content/ollama"
CLOUDFLARED_BIN = "/content/cloudflared"

cfg = {}
catalog = {}


def write_status(stage, detail="", ok=True, extra=None):
    try:
        payload = {"stage": stage, "detail": detail, "ok": ok,
                   "backend": cfg.get("backend"), "model": cfg.get("model"),
                   "port": cfg.get("port", PORT_DEFAULT),
                   "pid": os.getpid(), "tunnel_url": cfg.get("tunnel_url"),
                   "ready": stage in ("ready", "running"), "error": "" if ok else detail,
                   "ts": time.time()}
        if extra:
            payload.update(extra)
        with open(STATUS_FILE, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass
    sys.stdout.flush()


def log_tail(n=40):
    try:
        with open(DEPLOY_LOG) as f:
            return "".join(f.readlines()[-n:])[-4000:]
    except Exception:
        return ""


def die(msg):
    write_status("fatal", msg, ok=False, extra={"log_tail": log_tail()})
    print(f"FATAL: {msg}", flush=True)
    sys.exit(1)


def run(cmd, timeout=300, check=True, **kw):
    return subprocess.run(cmd, timeout=timeout, check=check, **kw)


def gpu_vram_gb():
    """Total VRAM of GPU 0 in GB; 0 if no GPU."""
    try:
        out = run(["nvidia-smi", "--query-gpu=memory.total",
                   "--format=csv,noheader,nounits"], timeout=15,
                  capture_output=True, text=True).stdout
        return float(out.strip().split()[0]) / 1024
    except Exception:
        return 0.0


def http_ok(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def wait_healthy(url, proc, timeout_s, label):
    """Bounded health poll + process-liveness; dies with log tail on failure."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not alive(proc.pid):
            die(f"{label} process died during startup. Log tail:\n{log_tail()}")
        if http_ok(url):
            return True
        time.sleep(5)
    if not alive(proc.pid):
        die(f"{label} process died (timeout path). Log tail:\n{log_tail()}")
    die(f"{label} not healthy after {timeout_s}s at {url}. Log tail:\n{log_tail()}")


# ── Stage 0: config + tier check ───────────────────────────────────
write_status("init", "reading config")
try:
    cfg = json.load(open(CONFIG_FILE))
    cfg.setdefault("port", PORT_DEFAULT)
    if os.path.exists(MODELS_FILE):
        catalog = json.load(open(MODELS_FILE)).get("models", {})
    recipe = catalog.get(cfg.get("model", ""), {})
    # Direct recipe override allowed: config["recipe"] = {...}
    if cfg.get("recipe"):
        recipe = cfg["recipe"]
    if not recipe:
        die(f"Unknown model key '{cfg.get('model')}' and no models.json recipe. "
            f"Upload models.json or pass config['recipe'].")
    cfg["recipe"] = recipe
    backend = cfg.get("backend", recipe.get("backend", "ollama"))
    cfg["backend"] = backend
    port = int(cfg["port"])
except SystemExit:
    raise
except Exception as e:
    die(f"Config load failed: {e}")

vram = gpu_vram_gb()
print(f"GPU VRAM: {vram:.1f} GB | backend={backend} model={recipe.get('model_ref')}", flush=True)

# Tier gate: refuse unquantized large models on small GPUs (no silent OOM)
min_vram = float(recipe.get("min_vram_gb", 0))
params_b = float(recipe.get("params_b", 0))
quant = recipe.get("quant", "")
if vram and vram < 30 and params_b >= 27 and quant in ("", "bf16", "fp16", "fp32"):
    die(f"Refusing unquantized {params_b}B model on {vram:.1f}GB GPU "
        f"(<30GB tier). Pick a quantized recipe.")
if vram and min_vram and vram < min_vram:
    die(f"Recipe needs >= {min_vram}GB VRAM, GPU has {vram:.1f}GB. "
        f"Pick a smaller recipe or a bigger GPU.")
write_status("tier_check", f"{vram:.1f}GB VRAM, tier ok for {cfg['model']}")

# ── Stage 1: idempotent version-locked installs ─────────────────────
write_status("install", "installing pinned components")
print("[1/6] Installing components...", flush=True)
try:
    # Ollama binary (pinned)
    need_ollama = backend == "ollama" and not (
        os.path.exists(OLLAMA_BIN) and OLLAMA_VER in
        run([OLLAMA_BIN, "--version"], capture_output=True, text=True, timeout=15).stdout)
    if need_ollama:
        print(f"  Installing ollama {OLLAMA_VER}...", flush=True)
        url = (f"https://github.com/ollama/ollama/releases/download/v{OLLAMA_VER}/"
               f"ollama-linux-amd64.tgz")
        run(["curl", "-sL", "-o", "/tmp/ollama.tgz", url], timeout=300)
        run(["tar", "xzf", "/tmp/ollama.tgz", "-C", "/tmp"], timeout=60)
        shutil.move("/tmp/bin/ollama", OLLAMA_BIN)
        os.chmod(OLLAMA_BIN, 0o755)

    # cloudflared binary (pinned)
    need_cf = not (
        os.path.exists(CLOUDFLARED_BIN) and CLOUDFLARED_VER in
        run([CLOUDFLARED_BIN, "--version"], capture_output=True, text=True, timeout=15).stdout)
    if need_cf:
        print(f"  Installing cloudflared {CLOUDFLARED_VER}...", flush=True)
        url = (f"https://github.com/cloudflare/cloudflared/releases/download/"
               f"{CLOUDFLARED_VER}/cloudflared-linux-amd64")
        run(["curl", "-sL", "-o", CLOUDFLARED_BIN, url], timeout=120)
        os.chmod(CLOUDFLARED_BIN, 0o755)

    # vLLM (pinned) — only for the vllm backend
    if backend == "vllm":
        print(f"  Installing vllm {VLLM_VER} (this takes a while)...", flush=True)
        run([sys.executable, "-m", "pip", "install", "-q",
             f"vllm=={VLLM_VER}", "transformers>=4.46,<5.0",
             "huggingface_hub>=0.26"], timeout=1800)

    # torchao pitfall: Colab ships torchao 0.10, peft needs >0.16 -> ImportError.
    # Uninstall right after pip installs; peft falls back to its default path.
    run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
        timeout=120, check=False,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("  Components ready.", flush=True)
    write_status("install", "components ready")
except Exception as e:
    die(f"Install failed: {e}")

# ── Stage 2: tunnel egress preflight (BEFORE any weight download) ───
write_status("preflight", "testing tunnel egress")
print("[2/6] Tunnel egress preflight (30s)...", flush=True)
try:
    probe = subprocess.Popen(
        [CLOUDFLARED_BIN, "tunnel", "--url", "http://127.0.0.1:19999",
         "--no-autoupdate", "--metrics", "127.0.0.1:0"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    ok = False
    deadline = time.time() + 30
    while time.time() < deadline:
        line = probe.stdout.readline()
        if not line:
            time.sleep(1)
            continue
        if re.search(r"https://[^\s]*trycloudflare\.com", line):
            ok = True
            break
    probe.terminate()
    try:
        probe.wait(timeout=10)
    except Exception:
        probe.kill()
    if not ok:
        die("Tunnel egress preflight failed: no trycloudflare URL in 30s. "
            "Egress is likely blocked; aborting before weight download.")
    print("  Egress OK.", flush=True)
    write_status("preflight", "egress ok")
except SystemExit:
    raise
except Exception as e:
    die(f"Preflight failed: {e}")

# ── Stage 3: acquire model ─────────────────────────────────────────
write_status("acquire_model", f"pulling {recipe.get('model_ref')}")
print(f"[3/6] Acquiring model {recipe.get('model_ref')}...", flush=True)
model_ref = recipe["model_ref"]
try:
    if backend == "ollama":
        env = dict(os.environ, OLLAMA_HOST="0.0.0.0:11434",
                   OLLAMA_KEEP_ALIVE="-1")
        # Start server first so `ollama pull` has somewhere to go
        ollama_log = open("/content/ollama.log", "a")
        srv = subprocess.Popen([OLLAMA_BIN, "serve"], env=env,
                               stdout=ollama_log, stderr=subprocess.STDOUT)
        cfg["server_proc"] = srv
        wait_healthy("http://127.0.0.1:11434/api/version", srv, 120, "ollama")
        print("  Pulling model (may take several minutes)...", flush=True)
        run([OLLAMA_BIN, "pull", model_ref], timeout=3600, env=env)
        print("  Model pulled.", flush=True)
    else:  # vllm: warm the HF cache now so serve doesn't download cold
        hf_token = cfg.get("hf_token", "")
        if hf_token:
            print("  Warming HF cache with authenticated download...", flush=True)
            env = dict(os.environ, HF_HUB_ENABLE_HF_TRANSFER="1")
            code = (
                "from huggingface_hub import snapshot_download; "
                f"snapshot_download({model_ref!r}, resume_download=True, "
                f"token={hf_token!r})"
            )
            run([sys.executable, "-c", code], timeout=3600, env=env)
            print("  Cache warm.", flush=True)
        else:
            print("  No hf_token: vLLM will download on serve (public repos only).",
                  flush=True)
    write_status("acquire_model", "model ready")
except SystemExit:
    raise
except Exception as e:
    die(f"Model acquisition failed: {e}")

# ── Stage 4: serve (health-gated) ──────────────────────────────────
write_status("server_start", f"starting {backend}")
print(f"[4/6] Starting {backend} server...", flush=True)
try:
    if backend == "ollama":
        srv = cfg.get("server_proc")
        if srv is None or not alive(srv.pid):
            env = dict(os.environ, OLLAMA_HOST="0.0.0.0:11434",
                       OLLAMA_KEEP_ALIVE="-1")
            ollama_log = open("/content/ollama.log", "a")
            srv = subprocess.Popen([OLLAMA_BIN, "serve"], env=env,
                                   stdout=ollama_log, stderr=subprocess.STDOUT)
            cfg["server_proc"] = srv
            wait_healthy("http://127.0.0.1:11434/api/version", srv, 120, "ollama")
        # OpenAI-compat gate
        wait_healthy("http://127.0.0.1:11434/v1/models", srv, 60, "ollama /v1")
        cfg["server_pid"] = srv.pid
        print("  Ollama serving (OpenAI-compatible /v1).", flush=True)
    else:
        max_len = int(cfg.get("max_model_len") or recipe.get("ctx", 4096))
        # Cap context by VRAM tier: activations OOM, not weights
        if vram and vram < 30:
            max_len = min(max_len, 4096)
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", model_ref, "--host", "127.0.0.1", "--port", str(port),
               "--gpu-memory-utilization", "0.90",
               "--max-model-len", str(max_len),
               "--dtype", "bfloat16",          # bf16 on T4, NOT fp16
               "--trust-remote-code"]
        quant_cfg = recipe.get("quant", "")
        if quant_cfg == "AWQ":
            cmd += ["--quantization", "awq"]
        vllm_log = open("/content/vllm.log", "a")
        srv = subprocess.Popen(cmd, stdout=vllm_log, stderr=subprocess.STDOUT)
        cfg["server_proc"] = srv
        cfg["server_pid"] = srv.pid
        wait_healthy(f"http://127.0.0.1:{port}/v1/models", srv, 900, "vllm")
        print(f"  vLLM serving on :{port} (bf16, max_len={max_len}).", flush=True)
    write_status("server_start", "server healthy",
                 extra={"server_pid": cfg.get("server_pid")})
except SystemExit:
    raise
except Exception as e:
    die(f"Server start failed: {e}")

# ── Stage 5: cloudflared tunnel ────────────────────────────────────
write_status("tunnel_start", "starting tunnel")
print("[5/6] Starting Cloudflare tunnel...", flush=True)
try:
    serve_port = 11434 if backend == "ollama" else port
    run("pkill -f 'cloudflared tunnel' 2>/dev/null", shell=True, check=False)

    def start_tunnel():
        return subprocess.Popen(
            [CLOUDFLARED_BIN, "tunnel", "--url", f"http://127.0.0.1:{serve_port}",
             "--no-autoupdate", "--metrics", "0.0.0.0:0"],
            stdout=open(TUNNEL_LOG, "w"), stderr=subprocess.STDOUT)

    def extract_url():
        try:
            found = re.findall(r"https://[^\s]*trycloudflare\.com",
                               open(TUNNEL_LOG).read())
            return found[0] if found else None
        except Exception:
            return None

    tp = start_tunnel()
    url = None
    deadline = time.time() + 60
    while time.time() < deadline and not url:
        time.sleep(2)
        url = extract_url()
        if tp.poll() is not None:
            break
    if not url:
        die(f"No tunnel URL in {TUNNEL_LOG} after 60s. Tail:\n"
            f"{open(TUNNEL_LOG).read()[-2000:] if os.path.exists(TUNNEL_LOG) else ''}")

    cfg["tunnel_url"] = url
    with open(API_URL_FILE, "w") as f:
        f.write(url)
    cfg["tunnel_proc"] = tp
    write_status("ready", url, extra={"tunnel_url": url,
                                      "server_pid": cfg.get("server_pid")})
    print(f"\n{'=' * 60}\nDEPLOYED: {url}\nbackend={backend} model={cfg['model']}\n{'=' * 60}",
          flush=True)
except SystemExit:
    raise
except Exception as e:
    die(f"Tunnel setup failed: {e}")

# ── Stage 6: watchdog (runs forever; exec_detach owns this process) ─
print("[6/6] Watchdog running.", flush=True)
restarts = 0
try:
    while True:
        time.sleep(30)
        tp = cfg.get("tunnel_proc")
        srv = cfg.get("server_proc")
        if srv is not None and not alive(srv.pid):
            # Server death is fatal: orchestrator must redeploy, not limp on
            die(f"Backend server (pid {srv.pid}) died. Redeploy required. "
                f"Tail:\n{log_tail()}")
        if tp is not None and tp.poll() is not None:
            restarts += 1
            print(f"[WATCHDOG] Tunnel died (restart #{restarts}). Restarting...",
                  flush=True)
            tp = start_tunnel()
            cfg["tunnel_proc"] = tp
            time.sleep(10)
            new_url = extract_url()
            if new_url and new_url != cfg.get("tunnel_url"):
                cfg["tunnel_url"] = new_url
                with open(API_URL_FILE, "w") as f:
                    f.write(new_url)
                print(f"[WATCHDOG] New URL: {new_url}", flush=True)
        write_status("running", cfg.get("tunnel_url", ""),
                     extra={"tunnel_url": cfg.get("tunnel_url"),
                            "tunnel_restarts": restarts,
                            "server_pid": cfg.get("server_pid")})
except SystemExit:
    raise
except Exception as e:
    die(f"Watchdog failed: {e}")
