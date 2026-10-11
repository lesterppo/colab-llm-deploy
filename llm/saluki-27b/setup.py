#!/usr/bin/env python3
"""Setup for Saluki 27B (ConwayResearch/Underdog-Saluki-27B-1.0, IQ2-mix GGUF)
on a Colab T4. Stage-gated to /content/saluki_setup_status.json.

Stages: download_model -> download_cudart -> download_llamacpp -> ready

PITFALL (caught live 2026-10-11): the llama.cpp CUDA tarball ships NO
libcudart — without the matching cudart bundle in LD_LIBRARY_PATH,
libggml-cuda.so fails to load and llama-server SILENTLY falls back to CPU
(0 MiB VRAM, ~0.5 tok/s, nothing in the log). Ground truth is nvidia-smi.
"""
import json
import os
import subprocess
import sys

STATUS = "/content/saluki_setup_status.json"
LLAMA_BUILD = "b11552"
LLAMA_URL = ("https://github.com/ggml-org/llama.cpp/releases/download/"
             f"{LLAMA_BUILD}/llama-{LLAMA_BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz")
CUDART_URL = ("https://github.com/ggml-org/llama.cpp/releases/download/"
              f"{LLAMA_BUILD}/cudart-llama-{LLAMA_BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz")
CUDART_DIR = "/content/cudart"
MODEL_DIR = "/content/models"
GGUF = "Underdog-Saluki-27B-1.0-IQ2-mix.gguf"
HF_REPO = "ConwayResearch/Underdog-Saluki-27B-1.0"


def status():
    if os.path.exists(STATUS):
        with open(STATUS) as f:
            return json.load(f)
    return {"stage": "start"}


def save(st):
    with open(STATUS, "w") as f:
        json.dump(st, f)


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def cudart_present():
    for _root, _dirs, files in os.walk(CUDART_DIR):
        if any(f.startswith("libcudart.so") for f in files):
            return True
    return False


def main():
    st = status()
    print("stage:", st["stage"], flush=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    if st["stage"] in ("start", "download_model"):
        st["stage"] = "download_model"
        save(st)
        env = dict(os.environ, HF_HUB_ENABLE_HF_TRANSFER="1")
        run([sys.executable, "-m", "pip", "install", "-q", "hf_transfer",
             "huggingface_hub"], env=env)
        run([sys.executable, "-c", (
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download(repo_id='{HF_REPO}', "
            f"allow_patterns=['{GGUF}'], local_dir='{MODEL_DIR}')")], env=env)
        sz = os.path.getsize(os.path.join(MODEL_DIR, GGUF))
        print(f"gguf bytes: {sz}", flush=True)
        assert sz > 7_000_000_000, f"gguf too small: {sz}"
        st["stage"] = "download_cudart"
        save(st)

    if st["stage"] == "download_cudart":
        if not cudart_present():
            os.makedirs(CUDART_DIR, exist_ok=True)
            run(["bash", "-c",
                 f"cd {CUDART_DIR} && curl -sL -o cudart.tgz '{CUDART_URL}' "
                 "&& tar xzf cudart.tgz && rm cudart.tgz"])
        assert cudart_present(), "cudart bundle missing libcudart.so*"
        print("cudart present", flush=True)
        st["stage"] = "download_llamacpp"
        save(st)

    if st["stage"] == "download_llamacpp":
        run(["bash", "-c",
             "cd /content && rm -rf llama.cpp && mkdir llama.cpp && "
             f"curl -sL -o /content/llama.tar.gz '{LLAMA_URL}' && "
             "tar xzf /content/llama.tar.gz -C /content/llama.cpp && "
             "rm /content/llama.tar.gz"])
        r = subprocess.run(["find", "/content/llama.cpp", "-name",
                            "llama-server", "-type", "f"],
                           capture_output=True, text=True)
        srv = r.stdout.strip().splitlines()
        assert srv, "llama-server not found in tarball"
        os.makedirs("/content/llama.cpp/bin", exist_ok=True)
        tgt = "/content/llama.cpp/bin/llama-server"
        if not os.path.exists(tgt):
            os.symlink(srv[0], tgt)
        run([tgt, "--version"])
        st["stage"] = "ready"
        save(st)

    print("setup complete: ready", flush=True)


if __name__ == "__main__":
    main()
