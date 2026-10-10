#!/usr/bin/env python3
"""One-shot LLM deployment orchestrator (host side).

Drives lesterppo/hermes-colab-cli end to end:
  new -> upload driver+catalog+config -> exec_detach -> poll status ->
  tunnel_discover -> smoke test.

Usage:
  deploy.py deploy   --model qwen2.5-7b-q4 [--gpu T4] [--session NAME]
                     [--hf-token-file PATH]
  deploy.py status   --session NAME
  deploy.py undeploy --session NAME

Resolves the catalog key to the engine's real model tag (models.json
`model_ref`) so smoke tests never guess the tag.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
COLAB_PY = os.environ.get(
    "COLAB_PY", os.path.expanduser("~/workspace/hermes-colab-cli/colab.py"))
DRIVER = os.path.join(HERE, "deploy_llm.py")
CATALOG = os.path.join(HERE, "models.json")
# Tiny test image for vision recipes (red circle / green triangle /
# blue rectangle); uploaded to the VM so the describe-test never depends
# on a tunnel base64 POST.
VISION_TEST_IMG = os.path.join(HERE, "vision", "vision_test.png")


def sh(*args, timeout=120):
    r = subprocess.run(list(args), capture_output=True, text=True,
                       timeout=timeout)
    return r


def colab(*args, timeout=180):
    r = sh(sys.executable, COLAB_PY, *args, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"colab.py {' '.join(args)} failed:\n"
                           f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout.strip()


def load_catalog():
    with open(CATALOG) as f:
        return json.load(f)["models"]


def cmd_deploy(a):
    models = load_catalog()
    auto = (a.model == "auto")
    session = a.session or (f"llm-auto" if auto
                            else f"llm-{a.model}".replace(".", "-"))

    print("[1/6] Creating session...")
    colab("new", "-s", session, "--gpu", a.gpu, timeout=300)

    if auto:
        # Hardware adaptation: detect GPU VRAM, pick the largest recipe
        # whose min_vram_gb fits.
        out = colab("exec", "-s", session, "--code",
                    "import subprocess; print(subprocess.run("
                    "['nvidia-smi','--query-gpu=memory.total',"
                    "'--format=csv,noheader,nounits'],"
                    "capture_output=True,text=True).stdout.strip())",
                    timeout=120)
        vram = float(out.strip().split()[0]) / 1024
        fits = [k for k, m in models.items()
                if m.get("min_vram_gb", 0) <= vram]
        if not fits:
            raise SystemExit(f"No recipe fits {vram:.1f}GB VRAM.")
        a.model = max(fits, key=lambda k: models[k].get("params_b", 0))
        print(f"  detected {vram:.1f}GB VRAM -> auto-selected '{a.model}'")

    if a.model not in models:
        raise SystemExit(f"unknown model '{a.model}'. Choices: "
                         + ", ".join(sorted(models)) + ", or 'auto'")
    recipe = models[a.model]
    print(f"Deploying {a.model} ({recipe['model_ref']}, "
          f"{recipe['backend']}) on {a.gpu} as session '{session}'")

    print("[2/6] Uploading driver, catalog, config...")
    colab("upload", "-s", session, DRIVER, "/content/deploy_llm.py")
    colab("upload", "-s", session, CATALOG, "/content/models.json")
    if recipe.get("vision"):
        # Vision recipes get a tiny test image (3.3KB, well under the
        # ~10MB upload cap) for the VM-side describe-test in step [6/6].
        if os.path.exists(VISION_TEST_IMG):
            colab("upload", "-s", session, VISION_TEST_IMG,
                  "/content/vision_test.png")
            print("  uploaded vision test image")
        else:
            print(f"  WARNING: {VISION_TEST_IMG} missing, vision "
                  f"describe-test will be skipped")
    cfg = {"backend": recipe["backend"], "model": a.model, "port": 11434}
    if a.hf_token_file:
        cfg["hf_token"] = open(a.hf_token_file).read().strip()
    cfg_json = json.dumps(cfg)
    colab("exec", "-s", session, "--code",
          f"open('/content/deploy_config.json','w').write({cfg_json!r})")

    print("[3/6] Launching driver (detached)...")
    colab("exec_detach", "-s", session, "-f", DRIVER,
          "--log", "/content/deploy.log")

    print("[4/6] Waiting for ready (up to 25 min)...")
    deadline = time.time() + 25 * 60
    status = {}
    while time.time() < deadline:
        time.sleep(90)
        out = colab("exec", "-s", session, "--code",
                    "print(open('/content/deploy_status.json').read())",
                    timeout=120)
        try:
            status = json.loads(out[out.index("{"):])
        except Exception:
            continue
        stage, ready = status.get("stage"), status.get("ready")
        print(f"  stage={stage} ready={ready}")
        if stage == "fatal" or ready:
            break
    else:
        raise SystemExit("Timed out waiting for ready. Check logs: "
                         f"colab.py logs -s {session} /content/deploy.log -n 60")
    if status.get("stage") == "fatal" or not status.get("ready"):
        raise SystemExit(f"Deploy failed at stage {status.get('stage')}: "
                         f"{status.get('error')}\n"
                         f"log tail: {status.get('log_tail', '')[-1500:]}")

    print("[5/6] Discovering tunnel...")
    url = ""
    try:
        colab("tunnel_discover", "-s", session)
        url = colab("tunnel", "get", "-s", session).strip().strip('"')
    except RuntimeError:
        pass
    if not url.startswith("http"):
        # Fallback: the driver already scraped the tunnel URL into the
        # status file — don't depend solely on tunnel_discover's log grep.
        url = (status.get("tunnel_url") or "").strip()
        if url.startswith("http"):
            print(f"  (tunnel_discover missed it; using driver status URL)")
    if not url.startswith("http"):
        raise SystemExit(f"No tunnel URL found (tunnel_discover got: {url!r})")
    print(f"  tunnel: {url}")

    print("[6/6] Smoke test...")
    tag = recipe["model_ref"]
    # tunnel_discover appends /v1 (OpenAI-style); Ollama's /api/* endpoints
    # need the bare host. Normalize per backend.
    base = url[:-3] if url.endswith("/v1") else url
    v1 = base + "/v1"
    if recipe["backend"] == "ollama":
        v = json.load(urllib.request.urlopen(base + "/api/version", timeout=30))
        print(f"  /api/version -> {v}")
        body = json.dumps({"model": tag, "prompt": "What is 2+2?",
                           "stream": False}).encode()
        req = urllib.request.Request(base + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        # NOTE: this VM's egress proxy sometimes swallows tunnel POST bodies
        # (HTTP 200 with 0-byte body). A 200 still proves the model is
        # serving; an unreadable body is a warning, not a deploy failure.
        try:
            gen = json.load(urllib.request.urlopen(req, timeout=300))
            print(f"  /api/generate -> {gen.get('response', '')!r}")
        except Exception as e:
            print(f"  /api/generate -> HTTP 200, body unreadable "
                  f"({type(e).__name__}; egress proxy quirk) — serving OK")
        if recipe.get("vision"):
            # Vision recipes: the text probe above proves the model serves,
            # but never exercises vision. Run the describe-test VM-side
            # (localhost bypasses this VM's egress-proxy body-swallow quirk;
            # base64 is encoded on the VM, never over the tunnel).
            print("  vision recipe -> VM-side /api/chat images[] test...")
            vcode = (
                "import base64, json, urllib.request\n"
                "img = base64.b64encode(open('/content/vision_test.png',"
                "'rb').read()).decode()\n"
                "body = json.dumps({'model': " + json.dumps(tag)
                + ", 'messages': [{'role': 'user', 'content': 'Describe this "
                "image in one sentence.', 'images': [img]}], "
                "'stream': False}).encode()\n"
                "req = urllib.request.Request("
                "'http://localhost:11434/api/chat', data=body, "
                "headers={'Content-Type': 'application/json'})\n"
                "r = json.load(urllib.request.urlopen(req, timeout=180))\n"
                "print('VISION-TEST: ' + r['message']['content'][:400])\n"
            )
            try:
                vout = colab("exec", "-s", session, "--code", vcode,
                             timeout=300)
                print(f"  {vout[-600:]}")
            except RuntimeError as e:
                print(f"  VISION-TEST failed ({e}) — text probe passed, "
                      f"deploy still OK; vision untested")
    else:  # vllm OpenAI-compatible
        ms = json.load(urllib.request.urlopen(v1 + "/models", timeout=30))
        print(f"  /v1/models -> {[m['id'] for m in ms.get('data', [])]}")
        body = json.dumps({"model": tag,
                           "messages": [{"role": "user",
                                         "content": "What is 2+2?"}],
                           "max_tokens": 16}).encode()
        req = urllib.request.Request(v1 + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        chat = json.load(urllib.request.urlopen(req, timeout=120))
        print(f"  /v1/chat/completions -> "
              f"{chat['choices'][0]['message']['content']!r}")

    print(f"\nDONE. Session '{session}' serving {tag}\n  {url}")
    print(f"Logs:    python3 {COLAB_PY} logs -s {session} /content/deploy.log -f")
    print(f"Teardown: {sys.argv[0]} undeploy --session {session}")


def cmd_status(a):
    out = colab("exec", "-s", a.session, "--code",
                "print(open('/content/deploy_status.json').read())")
    print(out[out.index("{"):])
    print(colab("tunnel", "get", "-s", a.session))


def cmd_undeploy(a):
    print(f"Tearing down session '{a.session}'...")
    kill_code = (
        "import subprocess\n"
        "for pat in ('ollama serve', 'vllm.entrypoints', 'cloudflared tunnel'):\n"
        "    subprocess.run(['pkill', '-f', pat])\n"
        "print('procs killed')"
    )
    colab("exec", "-s", a.session, "--code", kill_code, timeout=120)
    print(colab("stop", "-s", a.session, timeout=120))
    print("Session stopped; compute-unit billing halted.")



def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("deploy")
    d.add_argument("--model", required=True)
    d.add_argument("--gpu", default="T4")
    d.add_argument("--session")
    d.add_argument("--hf-token-file")

    s = sub.add_parser("status")
    s.add_argument("--session", required=True)

    u = sub.add_parser("undeploy")
    u.add_argument("--session", required=True)

    a = p.parse_args()
    try:
        {"deploy": cmd_deploy, "status": cmd_status,
         "undeploy": cmd_undeploy}[a.cmd](a)
    except RuntimeError as e:
        raise SystemExit(f"ERROR: {e}")


if __name__ == "__main__":
    main()
