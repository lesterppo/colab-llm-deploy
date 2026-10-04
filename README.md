# colab-llm-deploy

One-shot LLM deployment on Google Colab GPUs. Give it a GPU session, get back
an OpenAI-compatible endpoint on a public HTTPS URL — then tear it down when
you're done.

> **Status: live-tested.** End-to-end deploy verified 2026-10-04 on a free
> Colab T4 (session `llmtest2`): all 6 driver stages clean, Ollama 0.34.4
> serving `qwen2.5:7b-instruct-q4_K_M` (4.68GB GGUF Q4_K_M), public inference
> confirmed through the Cloudflare tunnel (`GET /api/version` → 200,
> `POST /api/generate` "What is 2+2?" → "4" in 3.4s), session stopped after.
> An earlier run on a different account failed at stage 1 (install) with an
> unverified cause — likely a transient VM network failure; the current
> driver is proven working.

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
