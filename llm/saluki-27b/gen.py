#!/usr/bin/env python3
"""Launch llama-server for Saluki 27B on a Colab T4 and run the smoke battery.

Reads /content/saluki_config.json (optional):
  { "ctx": 32768, "prompt": "..." }
Writes /content/saluki_results.json.

Server: /content/llama.cpp/bin/llama-server
  -m /content/models/Underdog-Saluki-27B-1.0-IQ2-mix.gguf
  --jinja -ngl 99 -fa on -c <ctx> --port 18080

Requires setup.py to have run first (model + llama.cpp + cudart bundle).

PITFALL: the cudart tarball nests libcudart.so* in a subdirectory, so
LD_LIBRARY_PATH must point at the dir that actually contains it
(find_cudart_libdir), not /content/cudart. After load, nvidia-smi must show
>1GB used — otherwise the server silently fell back to CPU: abort.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

SERVER = "/content/llama.cpp/bin/llama-server"
GGUF = "/content/models/Underdog-Saluki-27B-1.0-IQ2-mix.gguf"
CUDART_DIR = "/content/cudart"
CFG_PATH = "/content/saluki_config.json"
RES_PATH = "/content/saluki_results.json"
PORT = 18080
LOG = "/content/saluki_server.log"


def api(path, payload=None, timeout=300):
    url = f"http://127.0.0.1:{PORT}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def wait_health(tries=90):
    for _ in range(tries):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health",
                                   timeout=5)
            return True
        except Exception:
            time.sleep(5)
    return False


def vram_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    return int(out.stdout.strip().split()[0])


def find_cudart_libdir():
    """Locate the dir actually containing libcudart*.so* (tarball nests it)."""
    for root, _dirs, files in os.walk(CUDART_DIR):
        if any(f.startswith("libcudart.so") for f in files):
            return root
    return None


def main():
    cfg = {"ctx": 32768,
           "prompt": "Explain why the sky is blue in one paragraph."}
    if os.path.exists(CFG_PATH):
        with open(CFG_PATH) as f:
            cfg.update(json.load(f))
    res = {"config": cfg, "tests": {}}

    libdir = find_cudart_libdir()
    if not libdir:
        sys.exit("cudart bundle missing: run setup.py first")
    print("cudart libdir:", libdir, flush=True)

    # stop any existing server
    subprocess.run(["pkill", "-f", "llama-server"], check=False)
    time.sleep(3)

    cmd = [SERVER, "-m", GGUF, "--jinja", "-ngl", "99",
           "-fa", "on", "-c", str(cfg["ctx"]), "--port", str(PORT)]
    print("+", " ".join(cmd), flush=True)
    env = dict(os.environ,
               LD_LIBRARY_PATH=libdir + ":" +
               os.environ.get("LD_LIBRARY_PATH", ""))
    with open(LOG, "w") as log:
        subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    if not wait_health():
        res["error"] = "server did not become healthy"
        with open(LOG) as f:
            res["server_log_tail"] = f.read()[-4000:]
        json.dump(res, open(RES_PATH, "w"), indent=2)
        print(json.dumps(res, indent=2))
        sys.exit(1)

    time.sleep(5)
    used = vram_mib()
    res["vram_used_MiB_after_load"] = used
    if used < 1024:
        res["error"] = ("silent CPU fallback suspected: only "
                        f"{used} MiB VRAM after load")
        json.dump(res, open(RES_PATH, "w"), indent=2)
        print(json.dumps(res, indent=2))
        sys.exit(1)

    # Test 1: chat smoke (thinking off, temp 0)
    t0 = time.time()
    r1 = api("/v1/chat/completions", {
        "messages": [{"role": "user", "content": cfg["prompt"]}],
        "temperature": 0.0, "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    dt = time.time() - t0
    msg = r1["choices"][0]["message"]["content"]
    usage = r1.get("usage", {})
    res["tests"]["chat_smoke"] = {
        "content_excerpt": msg[:600],
        "usage": usage,
        "elapsed_s": round(dt, 1),
        "tok_per_s": round(usage.get("completion_tokens", 0) / max(dt, 0.01), 1),
    }
    res["vram_used_MiB_after_chat"] = vram_mib()

    # Test 2: tool calling (thinking off, temp 0) — the headline claim
    t0 = time.time()
    r2 = api("/v1/chat/completions", {
        "messages": [{"role": "user",
                      "content": "What is the weather like in Hong Kong right now?"}],
        "temperature": 0.0, "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string",
                                 "description": "City name, e.g. Hong Kong"}
                    },
                    "required": ["city"],
                },
            },
        }],
        "tool_choice": "auto",
    })
    dt = time.time() - t0
    m2 = r2["choices"][0]["message"]
    tool_calls = m2.get("tool_calls") or []
    res["tests"]["tool_call"] = {
        "elapsed_s": round(dt, 1),
        "tool_calls": tool_calls,
        "content": (m2.get("content") or "")[:300],
        "pass": bool(tool_calls) and tool_calls[0].get("type") == "function"
                and tool_calls[0]["function"]["name"] == "get_weather",
    }

    # Test 3: reasoning with thinking on
    t0 = time.time()
    r3 = api("/v1/chat/completions", {
        "messages": [{"role": "user",
                      "content": ("A farmer has 17 sheep. All but 9 run away. "
                                  "How many sheep are left? Think step by step, "
                                  "then give the final answer.")}],
        "temperature": 0.6, "top_p": 0.95, "top_k": 20,
        "max_tokens": 1024,
    }, timeout=600)
    dt = time.time() - t0
    msg3 = r3["choices"][0]["message"]["content"]
    usage3 = r3.get("usage", {})
    res["tests"]["reasoning"] = {
        "elapsed_s": round(dt, 1),
        "tok_per_s": round(usage3.get("completion_tokens", 0) / max(dt, 0.01), 1),
        "usage": usage3,
        "content_excerpt": msg3[:800],
        "final_answer_present": "9" in msg3[-200:],
    }

    res["vram_used_MiB_final"] = vram_mib()
    json.dump(res, open(RES_PATH, "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
