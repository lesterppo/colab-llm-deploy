# AGENTS.md — colab-llm-deploy

Instructions for AI agents deploying models on Google Colab GPUs with this
repo. Each modality has a one-command entry point; read the modality
README first, then run it.

## The commands

```bash
# LLM -> public OpenAI-compatible endpoint
python3 deploy.py deploy --model <catalog-key> [--gpu T4] [--session NAME]
python3 deploy.py status --session NAME
python3 deploy.py undeploy --session NAME        # kills procs + stops session

# Image / video / music -> local file
python3 deploy_media.py generate --kind image --model flux2-klein --prompt "..." [--out DIR]
python3 deploy_media.py generate --kind video --model ltx23 --prompt "..." [--out DIR]
python3 deploy_media.py generate --kind music --model yue2-3b --lyrics "..." [--out DIR]
python3 deploy_media.py status   --session NAME
python3 deploy_media.py undeploy --session NAME
```

`deploy.py --model auto` detects GPU VRAM via nvidia-smi and picks the
largest fitting recipe. `deploy_media.py --model` picks the pipeline:
`flux2-klein | hidream-i1 | ltx23 | ltx23-chain | yue2-3b | minimax-music3`;
prompts, sizes, and lyrics go
through `/content/gen_config.json` so gen scripts never guess.

Modality guides: [image/README.md](image/README.md),
[video/README.md](video/README.md),
[music/README.md](music/README.md). LLM details below.

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

## Media pitfalls (image/video/music, learned live 2026-10-04/05)

1. **ComfyUI must be >= 0.38** for the `flux2` CLIPLoader type (FLUX.2
   klein); older tags 400 the workflow.
2. **`hf` CLI preserves repo subpaths** on download
   (`unet/distilled-1.1/x.gguf` lands nested) — flatten to the model dir
   before the workflow runs (done in `video/ltx23/setup.py`).
3. **`LTXVAudioVAELoader` scans `models/checkpoints/`**, not `models/vae/`
   — copy the audio VAE there during setup.
4. **Never BF16 + full CPU offload for DiTs on Colab**: ~15GB of weights
   parked in the 12GB system-RAM cgroup → kernel OOM-kill. GGUF quants
   are the way for everything here.
5. **Q3 GGUF text encoder is mandatory for LTX-2.3** — the fp8 encoder
   OOMs. Same class of lesson as (4): size the encoder, not just the DiT.
6. **ComfyUI-GGUF node**: `UnetLoaderGGUF` + per-model CLIP loaders
   (`CLIPLoader` type `flux2`, `QuadrupleCLIPLoader`,
   `DualCLIPLoaderGGUF` type `ltxv`) — the loader must match the model.
7. Long generations: use `exec_detach` + status-file polling (status
   files are written at every stage); a 5-clip chain takes ~38 min —
   well inside session lifetimes, but don't run it in a foreground
   `exec` with a short timeout.
8. **Music lyrics must reach the VM as a file.** `deploy_media.py` writes
   them into `/content/gen_config.json` — never paste multi-line lyrics
   through `exec --code` (shell quoting breaks). Both music `gen.py`
   scripts reproduce the tested lighthouse song from built-in defaults
   when no `--lyrics` is given.
9. **MiniMax Music 3 INT8 repack**: 22GB → 11.9GB (DiT 2.5GB + pruned
   INT8 text encoder 9.2GB + DAV VAE). Probe `SaveAudioAdvanced` and
   fall back to core `SaveAudio` on older ComfyUI builds.
10. **YuE2 on T4**: ~7.25GB weights, no quantization needed; T4 has no
    BF16 so the acoustic stage runs slower than on A100-class GPUs. The
    pack reads weights in the `original` layout (`models/YuE2/YuE2-3B/`,
    `models/YuE2/YuE2-Vae`) — download with `--local-dir` directly there.

## Teardown discipline

Always `undeploy` when done: kills server/tunnel procs, then `stop`
halts compute-unit billing immediately. Never leave sessions running —
free-tier quota is per-day and shared across the account.
