# AGENTS.md — colab-llm-deploy

Instructions for AI agents deploying LLMs on Google Colab GPUs with this repo.
One command does everything; read this file first, then run it.

## The one command

```bash
python3 deploy.py deploy --model <catalog-key> [--gpu T4] [--session NAME]
python3 deploy.py status --session NAME      # poll stage / tunnel URL
python3 deploy.py undeploy --session NAME    # kill procs + stop session (halts billing)
```

`--model auto` detects GPU VRAM via nvidia-smi and picks the largest recipe
whose `min_vram_gb` fits. `--model` also accepts any key in `models.json`.

## Prerequisites

- `lesterppo/hermes-colab-cli` installed and authenticated
  (`python3 colab.py whoami` works). `deploy.py` shells out to it.
- Override the CLI path with `COLAB_PY=/path/to/colab.py` if needed.
- A Colab account with GPU quota. Free tier: ~3-8 GPU sessions/day/account,
  **one GPU session at a time**. If `new` fails with 503/`outcome:2`, the
  account's daily quota is exhausted — switch account
  (`colab-use-account <name>`) or wait, do not retry-loop.

## How it works (so you can debug it)

1. `deploy.py` (host): `colab.py new` → uploads `deploy_llm.py` +
   `models.json` + generated `/content/deploy_config.json` → `exec_detach`
   → polls `/content/deploy_status.json` until `ready:true`/`fatal` →
   `tunnel_discover` → smoke test.
2. `deploy_llm.py` (VM driver, stdlib-only): 6 stage-gated stages, each
   writing `/content/deploy_status.json`
   (`stage, ready, backend, model, port, tunnel_url, error, log_tail`):
   tier_check → install (pinned versions) → tunnel egress preflight (30s,
   BEFORE any download) → acquire_model → serve (health-gated) →
   cloudflared tunnel + watchdog. Tunnel URL also lands in
   `/content/api_url.txt` and `/content/tunnel.log`.
3. `models.json`: recipe catalog. `model_ref` is the engine's real tag —
   always use it for smoke tests, never the catalog key.

## Proven on T4 (live-tested 2026-10-04, kyu008008)

| Recipe | Model | Result |
|---|---|---|
| qwen2.5-7b-q4 | qwen2.5:7b-instruct-q4_K_M | GREEN, ~6 min end-to-end |
| qwen2.5-14b-q4 | qwen2.5:14b-instruct-q4_K_M | GREEN via `--model auto` (15.0GB detected) |
| gemma4-12b-q4 | gemma4:12b | GREEN, ~6 min |
| qwen3.5-9b-q4 | qwen3.5:9b | GREEN, ~10 min |
| mimo-v2.6-9b-q4 | hf.co/bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF:Q4_K_M | GREEN — Ollama pulls `hf.co/` GGUF refs directly |

Queued (recipes in catalog, tags verified, not yet live-tested):
qwen3.5-4b-q4, qwen3.5-27b-q4 (≥24GB), qwen2.5-7b-awq (vLLM, T4-tight),
phi4-14b-q4 (≥24GB), gemma4-e4b-q4, gpt-oss-20b (T4-tight, 14GB),
deepseek-coder-6.7b-q4, llama3.1-8b-q4.

## Pitfalls (learned live)

1. **`tunnel_discover` appends `/v1`** to scraped URLs (OpenAI convention).
   Ollama's `/api/*` endpoints need the bare host — `deploy.py` strips a
   trailing `/v1` per backend. If you hand-roll a smoke test, normalize first.
2. **Egress proxy quirk (this VM)**: tunnel POSTs sometimes return HTTP 200
   with a 0-byte body, and the first tunnel POST can hang ~120s on a
   user-confirmation gate. Treat 200-with-unreadable-body as serving-OK
   (warning, not failure); use generous POST timeouts (300s).
3. **Upload limits**: `colab.py upload` has a ~60s timeout and ~10MB cap;
   parent dirs must exist on the VM. Heavy downloads happen on the VM.
4. **`exec_detach` has no failure signal** — the driver writes status at
   every stage; "status file missing after N minutes" is itself the error.
5. **Ollama `hf.co/` refs work**: any quality GGUF on HuggingFace is
   deployable as `hf.co/<user>/<repo>:<quant>` with zero conversion.
6. **VRAM math for sizing**: Q4_K_M ≈ 0.5 bytes/param. 15GB T4 fits up to
   ~20B dense at Q4; text encoders + VAE add ~2–8GB on top of the DiT.
7. **Do not use BF16 + `enable_model_cpu_offload`** for image DiTs on Colab:
   ~15GB of weights parked in the 12GB system-RAM cgroup → kernel OOM-kill
   (learned on FLUX.2 klein). GGUF quants are the way.
8. **Colab 90-min idle timeout, 24h max.** Free sessions get reclaimed
   without warning (observed 30–60 min lifetimes under load). Keep deploys
   idempotent so a rerun converges fast.
9. **Torchao**: Colab ships torchao 0.10; peft needs >0.16 → ImportError at
   `get_peft_model`. The driver uninstalls torchao after pip installs.
10. **transformers pin**: keep `transformers>=4.46,<5.0` on Colab; 5.x breaks
    things (and Colab's preinstalled deps conflict with tight pins — the
    driver avoids over-pinning).

## Teardown discipline

Always `undeploy` when done: kills server/tunnel procs, then `stop`
halts compute-unit billing immediately. Never leave sessions running —
free-tier quota is per-day and shared across the account.
