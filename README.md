# colab-llm-deploy

One-shot model deployment on Google Colab GPUs. Give it a GPU session, get
back a result — an OpenAI-compatible LLM endpoint, a generated image, or a
generated video — then tear it down when you're done.

| Modality | Entry point | What it does |
|---|---|---|
| LLM | `python3 deploy.py deploy --model auto` | Serves an Ollama/vLLM model behind a public Cloudflare-tunnel URL |
| Image | `python3 deploy_media.py generate --kind image --model flux2-klein --prompt "..."` | Text-to-image PNG (FLUX.2-klein-4B, HiDream-I1 17B) |
| Video | `python3 deploy_media.py generate --kind video --model ltx23 --prompt "..."` | Text-to-video MP4 (LTX-2.3 22B); `--model ltx23-chain` for longer chained videos |
| Audio | `python3 deploy_media.py generate --kind audio --model chatterbox --text1 "..."` | TTS + voice cloning (Chatterbox), lyrics-to-song (YuE2-3B, MiniMax Music 3) |
| Music | `python3 deploy_media.py generate --kind music --model yue2-3b --lyrics "..."` | Lyrics-to-song audio (YuE2-3B, MiniMax Music 3 INT8) |

> **Status: live-tested on free Colab T4s.**
> LLMs (2026-10-04): Qwen2.5-7B/14B, Gemma 4 12B, Qwen3.5 9B, MiMo-V2.6 9B —
> all deployed, served through tunnels, smoke-tested, torn down.
> Images (2026-10-04/05): FLUX.2-klein-4B (~70s per 1024px image),
> HiDream-I1 17B Q4 (tight but workable).
> Video (2026-10-05): LTX-2.3 22B Q3 (576x320 49f in ~7 min; 5-clip chain =
> ~10.2s in ~38 min).
> Audio (2026-10-05): Chatterbox TTS + zero-shot voice cloning, YuE2-3B
> lyrics-to-song (40s in ~4 min), MiniMax Music 3 INT8 repack (30s in ~5 min).
> See [AGENTS.md](AGENTS.md), [image/](image/), [video/](video/),
> [audio/](audio/) for the full test matrix and pitfalls.

## Architecture

Two halves, split at the Colab session boundary:

```
Host (your machine / agent)                 Colab VM (T4/L4/A100)
┌─────────────────────────┐                 ┌──────────────────────────────┐
│ hermes-colab-cli        │                 │ deploy_llm.py (this repo)    │
│  new -s llm --gpu T4    │── upload ──▶    │  1. tier check (nvidia-smi)    │
│  upload deploy_config   │                 │  2. pinned installs           │
│  exec_detach -f         │── run ────▶     │  3. tunnel egress preflight    │
│    deploy_llm.py        │                 │  4. model acquire (ollama      │
│  poll deploy_status     │◀── status ──    │     pull / HF cache warm)      │
│    .json                │                 │  5. serve (health-gated)      │
│  tunnel_discover        │◀── URL ────     │  6. cloudflared + watchdog     │
└─────────────────────────┘                 └──────────────────────────────┘
```

The host side is orchestration only — it uses
[lesterppo/hermes-colab-cli](https://github.com/lesterppo/hermes-colab-cli)
primitives (`new`, `exec_detach`, `tunnel_discover`) and holds no deployment
logic of its own. All VM-side logic lives in `deploy_llm.py` (stdlib only,
single file).

Patterns here are reimplemented from lessons learned, informed by
[yixuanw99/colab-llm-station](https://github.com/yixuanw99/colab-llm-station)
(dual-engine/tunnel matrix, idempotent version-locked bootstrap),
[enescingoz/colab-llm](https://github.com/enescingoz/colab-llm) (minimal
Ollama + Cloudflare flow), and
[galinilin/docgemma-app](https://github.com/galinilin/docgemma-app) (tunnel
egress preflight, health-gated readiness). No code was copied from them.

## Quickstart

One command does the whole flow (provision → upload → launch → poll →
tunnel → smoke test):

```bash
python3 deploy.py deploy --model qwen2.5-7b-q4
python3 deploy.py status --session llm-qwen2-5-7b-q4
python3 deploy.py undeploy --session llm-qwen2-5-7b-q4
```

Manual step-by-step (same thing `deploy.py` automates):

```bash
# 1. Provision a T4 (hermes-colab-cli must be installed + authed)
python3 colab.py new -s llmdeploy --gpu T4

# 2. Write the deploy config
cat > /tmp/deploy_config.json <<'EOF'
{"backend": "ollama", "model": "qwen2.5-7b-q4", "port": 8000}
EOF
python3 colab.py upload -s llmdeploy /tmp/deploy_config.json /content/deploy_config.json
python3 colab.py upload -s llmdeploy models.json /content/models.json

# 3. Launch the driver (detached — returns immediately)
python3 colab.py exec_detach -s llmdeploy -f deploy_llm.py --log /content/deploy_llm.log

# 4. Poll stage status (machine-readable; exec_detach has no failure signal)
python3 colab.py exec -s llmdeploy --code \
  "import json; print(json.load(open('/content/deploy_status.json'))['stage'])"

# 5. Grab the public URL (auto-discovered from the VM)
python3 colab.py tunnel_discover -s llmdeploy

# 6. Use it like any OpenAI-compatible endpoint
curl $(python3 colab.py tunnel get -s llmdeploy)/v1/models

# 7. Tear down explicitly (halts compute-unit billing immediately)
python3 colab.py exec -s llmdeploy --code \
  "import google.colab.runtime as r; r.unassign()"   # then: colab.py stop -s llmdeploy
```

`hf_token` goes in `deploy_config.json` only (never a CLI arg, never logged).
It warms the HF cache for gated/rate-limited repos before serving.

## Model catalog (`models.json`)

| key | backend | params | quant | VRAM | tier |
|---|---|---|---|---|---|
| `qwen2.5-7b-q4` | Ollama | 7B | GGUF Q4_K_M | ~6 GB | T4-ok (default recipe) |
| `llama3.1-8b-q4` | Ollama | 8B | GGUF Q4_K_M | ~7 GB | T4-ok |
| `deepseek-coder-6.7b-q4` | Ollama | 6.7B | GGUF Q4_K_M | ~5.5 GB | T4-ok |
| `qwen2.5-7b-awq` | vLLM | 7B | AWQ | ~9 GB | T4-tight (A100 preferred) |
| `phi4-14b-q4` | Ollama | 14B | GGUF Q4_K_M | ~12 GB | ≥24 GB (L4/A10G — not T4) |

The driver refuses to load an unquantized 27B+ model on a <30GB GPU instead of
OOMing silently, and caps vLLM `--max-model-len` at 4096 on the T4 tier
(activations OOM, not weights). vLLM runs `--gpu-memory-utilization 0.90`
with **bf16** compute — fp16 crashes `clip_grad_norm` on T4.

## Colab constraints (design around them, not against them)

- **~90-minute idle timeout, 24h max.** Free sessions get reclaimed (often in
  30–60 min in the evening). The driver is idempotent: every run
  verify-or-installs pinned versions, so a fresh VM converges cheaply.
- **Tunnel egress can be blocked.** A 30-second dummy-port tunnel test runs
  *before* any weight download — cheap insurance against a 10-minute
  download to nowhere.
- **Tunnel URL changes on restart.** The watchdog restarts dead cloudflared
  and rewrites `/content/api_url.txt`; re-run `tunnel_discover`.
- **Server death is fatal.** If the backend process dies, the driver writes a
  `fatal` stage with a log tail and exits — the orchestrator redeploys from
  scratch rather than limping on.
- **Proxy/session tokens expire (~1h).** For deploys that must survive hours,
  run a host-side keepalive alongside (see hermes-colab-cli examples).

## Colab T4 pitfalls baked in

From the hermes-colab-cli playbook: `pip uninstall -y torchao` after installs
(Colab's torchao 0.10 breaks peft), `cloudflared --metrics 0.0.0.0:0`
(multi-instance port conflict), `OLLAMA_KEEP_ALIVE=-1` (weights stay in
VRAM), transformers `<5.0`, and the HTTP 411 empty-body POST fix (host side,
in hermes-colab-cli's patch script).

## Files

- `deploy_llm.py` — the VM driver (stdlib only).
- `models.json` — recipe catalog: backend, quant, VRAM, context, notes.
- `versions.env` — pinned component versions.
- `LICENSE`, `.gitignore`.

## License

MIT — see LICENSE.
