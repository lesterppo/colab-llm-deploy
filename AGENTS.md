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

# Image / video / audio -> local file
python3 deploy_media.py generate --kind image --model flux2-klein --prompt "..." [--out DIR]
python3 deploy_media.py generate --kind video --model ltx23 --prompt "..." [--out DIR]
python3 deploy_media.py generate --kind audio --model chatterbox --text1 "..." --text2 "..." [--out DIR]
python3 deploy_media.py generate --kind audio --model yue2 --lyrics "..." --style "..." [--out DIR]
python3 deploy_media.py status   --session NAME
python3 deploy_media.py undeploy --session NAME

# 3D -> trained splat file (Brush Gaussian splatting, not a served model)
python3 3d/brush-lighthouse/deploy_brush.py [--session brush-lighthouse] [--steps 30000] [--out DIR]
```

`deploy.py --model auto` detects GPU VRAM via nvidia-smi and picks the
largest fitting recipe. `deploy_media.py --model` picks the pipeline:
`flux2-klein | hidream-i1 | ltx23 | ltx23-chain | minimax-h3 | wan22-5b |
chatterbox | yue2 | minimax-music3`; prompts, sizes, lyrics, and TTS text go
through `/content/gen_config.json` so gen scripts never guess.

Modality guides: [image/README.md](image/README.md),
[video/README.md](video/README.md),
[audio/README.md](audio/README.md),
[3d/README.md](3d/README.md) (lighthouse training, Zanjic church,
and the standalone viewer builder).
LLM details below.

## Proven on T4 — 3D (live-tested 2026-11-04)

| Recipe | What it does | Result |
|---|---|---|
| brush-lighthouse | Brush 0.3.0 trains a synthetic lighthouse-island splat (48 train + 8 val views, generated on the VM) | GREEN at 30,000 steps: 320.6 s train, 123,025 points, mean val PSNR 18.59 dB. One-kick wrapper's VM scripts re-tested the same evening at 4,000 steps: setup ready in 51 s, train ready, mean PSNR 16.42 dB, PLY verified. See [3d/brush-lighthouse/README.md](3d/brush-lighthouse/README.md) for the two Brush pitfalls that must not be re-learned (small-PNG "early eof", missing Vulkan ICD) and for the wrapper's live-test status. |
| zanjic-church | Same pipeline on the Žanjic island monastery | NOT GREEN: 30,000 steps rc=0 in ~8 min, 266,426 points, but mean val PSNR 12.98 dB, foggy eval renders — a low subject over a huge flat sea disc. Shipped viewer uses the procedural reconstruction (up to 327,377 points). See [3d/zanjic-church/README.md](3d/zanjic-church/README.md). |
| viewer (standalone) | Phone-proof splat viewer HTML for any scene | GREEN on the user's phone 2026-11-04 after four CDN versions failed: three.js + Spark embedded as base64 blob modules (with the three.core.js → three-core rewrite), classic-script error handling naming the failing step, splat URL load with embedded fallback. Build with `python3 3d/viewer/build_standalone.py --ply scene.ply --out scene.html`. Single-file ceiling ~100 MB (115 MB rejected, HTTP 413); rapid uploads hit HTTP 429. See [3d/viewer/README.md](3d/viewer/README.md). |

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
| qwen3-vl-8b-q4 | qwen3-vl:8b-instruct | GREEN 2026-10-09 — first vision-language recipe; vision describe-test 12s, no reasoning leak, 11.5/15.4GB VRAM (kyu008008) |
| saluki-27b-iq2 | ConwayResearch/Underdog-Saluki-27B-1.0 (IQ2-mix) | GREEN 2026-10-11 — **new `llamacpp` backend** (llama.cpp b11552 + cudart bundle; VRAM 8.6–8.9/15.4GB); tool calling verified (well-formed `tool_calls`, the post's headline claim), reasoning 8.2 tok/s (cyc236hk). Deep test same day: ctx 32768 fits (10.4/15.4GB), tool suite 5/6, reasoning 11/12, needle 3/3 @21.6k tok — see [docs/saluki-27b-deep-test.md](docs/saluki-27b-deep-test.md) |
| inkling-2-mini-iq3 | thinkingmachines/Inkling-2-Mini (UD-IQ3_XXS, 11.66GB) | GREEN 2026-11-24 — **llama.cpp bumped b11552 → b12669** (inkling2_loop needs ≥b12600; b11552 fails `unknown model architecture: 'inkling2_loop'`). Deep test @ctx16384: tool 5/6, reasoning **12/12**, needle 2/2 @10.3k/13.1k, ZH 3/3, decode 23.6 tok/s, VRAM 14.0/15.4GB; vision describe-test correct via mmproj F16 (manual `--mmproj`, driver has no mmproj support). See [docs/inkling-2-mini-deep-test.md](docs/inkling-2-mini-deep-test.md). **Q3_K_XL (13.63GB) FAILED on T4**: CUDA OOM crash mid-test (~475 MiB headroom). **ctx 32768 OOMs at load** — 16384 is the T4 max for this file. Agentic test same day (kyu009009): **13/14 corrected** — multi-step tool loops 3/3, error recovery 2/2, code agent 2/3 (bug-fix task failed twice, systematic — see doc), planning 2/2, memory 1/1, tool discipline 2/2, belief update 1/1; harness `inkling-2-mini-agentic-test.py`. See [docs/inkling-2-mini-agentic-test.md](docs/inkling-2-mini-agentic-test.md). Extensive agentic test same day (cyc236es): **39/41 corrected** (36/41 as-run) — tool loops 7/7, error recovery 4/5, code 9/10 (state-the-fix variant fixes the probe bug; remaining fails are loop-termination, not fixing), planning 6/6, memory 3/3, discipline 4/4, belief 3/3, long-context 3/3; harness `inkling-2-mini-agentic-test-ext.py`. See [docs/inkling-2-mini-agentic-test-ext.md](docs/inkling-2-mini-agentic-test-ext.md). |
| nemotron-3.5-bolt-e4b-q4 | nemotron-3.5-bolt:e4b-q4_K_M | VM-side GREEN 2026-10-25, tunnel RED — NVIDIA Bolt E4B (Gemma 4 E4B base, 8B/4B eff, 7.9GB Ollama build, ctx8192 variant, VRAM 7,861/15,360 MiB, "What is 2+2?" → "The answer is four."). **Requires Ollama ≥0.40.10** (driver bumped 0.35.1→0.40.11; 0.35.1 fails the pull with HTTP 412). Two fresh trycloudflare subdomains both returned Cloudflare 403 "Your request was blocked" from this VM's egress while the tunnel process was healthy — so this recipe is NOT marked `proven_t4` until a tunnel smoke test passes; if you see the same 403, test the URL from a non-datacenter network before blaming the model. |

## Training-track probes (2026-11-24/25)

These seven models were probed for **training** (transformers NF4 QLoRA,
one real training step on a free Colab T4), **not** served via
`deploy_llm.py` — they are not deployable recipes and none is marked
`proven_t4` for inference/serving. They live in `models.json` under the
separate top-level `training_probes` key so `deploy.py` (which reads only
`models["models"]`) is untouched. The catalog's `qwen3.5-4b-q4` Ollama
entry above is a different thing — a queued, unproven *serving* recipe —
and is left untouched. Default stays **Qwen3.5-9B**. Full evidence:
[docs/small-model-training-probes-2026-11-25.md](docs/small-model-training-probes-2026-11-25.md).

| Model | Training role | Peak VRAM at step | Capability headline | Key recipe note |
|---|---|---|---|---|
| Alpie Seed 9B (`169Pi/Alpie_seed`, 9.41B, Apache 2.0) | First alternate | 7.57 GB (load 7.12, step 3.0 s, loss 0.2303) | Reasoning 8/8, tool-JSON 2/3, needle 4k PASS | Prefer for reasoning/code papers; keep Qwen3.5-9B default until a head-to-head training run exists |
| TwIL-8B (`webAI-Official/TwIL-8B`, 8.79B) | Reasoning specialist | 5.85 GB (load 5.36, step 6.7 s, loss 0.1178) | Reasoning 7/8, Chinese 3/3, needle 4k PASS | webAI Non-Commercial 1.0 (research/education only), never the general default; verbose thinking traces need generous budgets |
| SenseNova Loop-9B (`sensenova/SenseNova-Loop-9B`, 9.66B, Apache 2.0) | Out of rotation — research only | 8.42 GB (load 7.35, step 15.2 s, loss 0.1754) | Reasoning 4/8, tool-JSON 3/3, 1.7 tok/s | Looped-architecture papers only; budget 2–6× a Qwen3.5-class run; pass `use_cache=False` explicitly on all nested configs |
| Qwen3.5-4B (`Qwen/Qwen3.5-4B`, 4.7B, Apache 2.0) | 4B general ablation arm | 3.33–3.58 GB (load 2.91, loss 0.1832) | 18/21, tool-JSON 3/3, reasoning 5/8, needle 8k PASS | DeltaNet LoRA proven (all 248 LoRA B grads nonzero); `pip uninstall -y torchao` or drop `conv1d` from targets |
| LFM3-1.7B (`LiquidAI/LFM3-1.7B`, 1.73B, LFM Open License v1.0) | Size/agent arm | 1.44 GB (load 1.20, step 0.9 s, loss 0.3153) | Reasoning 7/8, needle 8k PASS, 18.0 tok/s | Needs `trust_remote_code`; score tools in native `<tool_call>` format, not raw JSON; 13 depthwise conv kernels stay frozen |
| Dory 4B (`nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`, OpenMDW-1.1) | No rotation place | 3.24 GB (load 2.84, step 2.8 s, loss 0.1286) | Tool-JSON 3/3, reasoning 4/8, needle 8k PASS | Needs transformers git main (≥5.24); shipped 5.22.1 fails at load |
| MiniCPM5-2B (`openbmb/MiniCPM5-2B`, 2.52B, Apache 2.0) | Fallback only | 2.28 GB (load 1.95, step 1.2 s, loss 0.3134) | 15/21 (lowest), tool-JSON 2/3, reasoning 5/8 | Cleanest port (native 5.22.1, standard Llama); fallback only if both Qwen3.5-4B and LFM3 are unavailable |

Cross-cutting findings:
- **Needle 8k via transformers generate OOMs on a T4 for every 8–9B model**
  (SDPA math path on sm_75 materializes the [heads, seq, seq] score tensor)
  — long-context eval for 8–9B needs llama.cpp/vLLM or chunked attention —
  while **all four smaller models passed 8k**.
- **Shipped torchao 0.10.0 breaks any LoRA target list containing a Conv1d**
  — uninstall it (`pip uninstall -y torchao`) or drop the conv target.
  Depthwise conv kernels (groups 2048/8192) cannot take PEFT LoRA at r=16
  in either case; they stay frozen and are a negligible param count.

## The llamacpp backend

`deploy_llm.py` supports `backend: "llamacpp"` (added 2026-10-11 for
qwen3.5-arch models Ollama can't run). Stage 1 installs the pinned
llama.cpp CUDA tarball **plus the separate cudart tarball** and exposes
`bin/llama-server` at a stable path; stage 3 downloads the single GGUF
(`recipe["gguf_file"]` from the HF repo) via huggingface_hub; stage 4
serves `llama-server -m <gguf> --jinja -ngl 99 -fa on -c <ctx> --port <port>`
with `LD_LIBRARY_PATH` pointed at the cudart bundle. Smoke test is the
OpenAI `/v1/chat/completions` probe (same as vllm). Critical pitfall: the
CUDA tarball ships **no** libcudart — without the cudart bundle
libggml-cuda.so fails to load and llama-server silently runs on CPU
(0 MiB VRAM, ~0.5 tok/s); the driver dies loudly if libcudart is missing.

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
11. **qwen3-vl: use the `-instruct` tag, not the bare size tag.** The bare
    `qwen3-vl:8b` (and `:30b`) tags are thinking builds; Ollama ignores
    `think=false` on them, so every request burns hidden CoT tokens before
    answering (community-measured 8–70s on long inputs). For predictable
    latency always pin `:8b-instruct` / `:30b-a3b-instruct`. (Verified
    2026-10-09: bare :8b digest 901cae732162 vs :8b-instruct 0533d74300e4.)
12. **Multi-image same-size merge (ollama/ollama#17321).** Two same-size
    images in one /api/chat message are silently merged into video frames —
    the model reports seeing only one image. For multi-image prompts, vary
    each image's resolution (or use explicit [img-N] placeholders).
13. **`deploy.py`'s built-in smoke test used to be text-only.** `/api/generate
    "What is 2+2?"` proves the model serves but never exercised vision.
    Vision recipes (`"vision": true`) now get an extra VM-side `/api/chat`
    images[] describe-test during deploy — a green deploy of a vision recipe
    implies the describe-test passed, not just the text probe.
14. **Recipe `ctx` is now wired through to Ollama (2026-10-15).** Ollama
    serves at `num_ctx=4096` by default; the driver used to ignore the
    catalog's `ctx` for the ollama backend, so every recipe claiming 8192+
    was silently truncated at serve time (caught on the CLM run, whose
    harness peaks ~8k tokens). The driver now builds a
    `<catalog-key>-ctx<N>` Modelfile variant after pull and records it as
    `served_model` in `/content/deploy_status.json`; `deploy.py` probes the
    variant in the smoke test. >8k claims are capped to 8192 on <30GB GPUs
    (KV-cache OOM guard); `clef-flash` 262144 / `qwen3-vl` 32768 need ≥24GB
    to serve their full claims.
15. **Ollama is not on PATH in `colab.py console` shells.** The binary lives
    at `/content/ollama-dist/bin/ollama` (`which ollama` returns nothing) —
    use the full path or `export PATH=/content/ollama-dist/bin:$PATH` in
    hand-rolled console commands (driver itself always uses the full path).

## Media pitfalls (image/video/audio, learned live 2026-10-04/05)

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
11. **Chatterbox env fix**: pip can pull torchvision newer than Colab's
    torch 2.6.0 (`torchvision::nms` missing → LlamaModel import fails) —
    pin `torchvision==0.21.0` and `pip uninstall -y torchao` after
    installing `chatterbox-tts` (done in `audio/chatterbox/setup.py`).
12. **T4 (sm_75) has NO fast attention kernel** — torch 2.11+cu130's
    flash/mem-efficient SDPA both fail with "No available kernel", so
    SDPA silently falls back to the math path and materializes full
    [B, heads, seq, seq] score tensors. Wan 2.1 14B OOM'd at sampling
    step 0 (15.26GB requested at 848x480x33, 9.23GB at 25 frames — both
    over the 15GB card). Workaround: chunked attention
    (`attention_mode="comfy"` on WanVideoModelLoader → ComfyUI
    `optimized_attention`, O(chunk×seq) memory). Developed 2026-10-07
    for Wan 2.1, **not yet live-tested** — do not treat as proven until
    a run completes.
13. **`colab.py exec_detach -f` takes a LOCAL path** (hermes-colab-cli
    v3.3): the wrapper resolves it on the host and dies with
    `exec-detach-no-file` if it does not exist locally. Passing a
    remote path such as `-f /content/setup.py` always dies — pass the
    local script path instead; the wrapper uploads it to
    `/content/<basename>` itself. Bit `deploy_media.py` steps [3/6]/[4/6]
    and the MiniMax H3 test driver on 2026-11-11 (fixed the same day).
    Same class of silent failure: ComfyUI 0.45 `SaveVideo` needs
    `format: "auto", codec: "auto"` — `video/h264-mp4` passes prompt
    validation then writes no file (cost one full 32-min H3 run).

## Supervisor (surviving longer than one session)

Free Colab sessions are reclaimed server-side; nothing here prevents
that. `supervise.py` is the honest mitigation: a 5-min real heartbeat
exec (resets the 90-min idle timer), death detection, and rate-limited
auto-redeploy (3 attempts, 60/120/240s backoff, quota failures count,
account rotation opt-in only, never cyc236de/cyc236hk). It keeps a
tunnel when deploy.py's smoke test 403s but the VM reports running,
and — learned 2026-10-25 — a single heartbeat timeout is treated as
transient (2 consecutive transient failures before redeploy), never
as a crash: an uncaught `TimeoutExpired` once killed the supervisor
while the Colab session was still alive.

```bash
python3 supervise.py --model nemotron-3.5-bolt-e4b-q4 --gpu T4 \
  --check-interval 300 --max-restarts 3
# latest tunnel: current-tunnel.txt ; full state: supervisor-state.json
```

**Restart-proofing (2026-10-25):** `supervise_watchdog.sh` + a 10-minute
cron (`supervisor-watchdog`) restarts the supervisor if its process is
dead, without starting a duplicate and without redeploying — a restarted
supervisor first checks `session_alive()` and picks up the live session.
Tested live: killed the supervisor, watchdog restarted it, next heartbeat
was alive with the tunnel unchanged. The cron only reports on a restart,
a >20-min heartbeat gap, or a watchdog failure; routine checks are silent.

**Mini-benchmark (2026-10-25, `mini_bench.py`):** 28 locally-scored checks
on Bolt E4B (temp 0, think off except reasoning): tool 6/6, hierarchy 4/5,
constraints 6/6, multi-turn 3/3, grounding 2/2, reasoning 6/6 — **27/28**.
The one fail: system told it to end every reply with END, user said don't,
it answered "4" with no END. Think-on 9×8 = 72 correct; think-off 9×8 =
"Ninety-two" — do not trust think-off arithmetic on this model. This is a
mini-suite inspired by NVIDIA's axes (BFCL, IHEval, IFBench, MultiChallenge,
RULER), not their official harness.

**Soup + Gemma 4 on T4 (2026-10-25, NEGATIVE for Soup as-is):** Bolt itself
cannot be Soup-trained — NVIDIA's repo is GGUF-only (no safetensors, no
config). Its parent `google/gemma-4-E4B-it` (ungated safetensors, 15.99GB,
model_type `gemma4`) hits four stacked blockers in Soup 0.72.4:
(1) transformers 4.57.6 GemmaFast crashes on the tokenizer's
`extra_special_tokens: ['<|video|>']` list (expects a dict) — workaround:
patch a local copy's tokenizer_config to drop that key;
(2) transformers 4.57.6 does not recognise `gemma4` at all — workaround:
transformers from source (5.21.0.dev0) recognises `Gemma4Config`;
(3) layer streaming explicitly rejects gemma4 (supported up to
gemma3_text);
(4) non-streaming NF4 QLoRA loads 100% of weights but OOMs at the first
training step, even with the full GPU free, and even at the minimum
(batch 1, seq 128, gradient_checkpointing on). So Soup + Gemma 4 needs a
≥24GB GPU, Soup adding gemma4 to streaming, or a different trainer.
Configs + dataset + tokenizer patch for the attempt live in
`soup_gemma4*.yaml`, `soup_data.jsonl`, `patch_tokenizer.py`. Note:
`ollama stop` hung on a second attempt while `pkill -f llama-server`
freed VRAM instantly (Ollama serve stays up, a chat request respawns the
model in ~40s); foreground `colab.py exec` also hung on that step while
`exec_bg` worked — prefer `exec_bg` for stop/train launches.

## Teardown discipline

Always `undeploy` when done: kills server/tunnel procs, then `stop`
halts compute-unit billing immediately. Never leave sessions running —
free-tier quota is per-day and shared across the account.
