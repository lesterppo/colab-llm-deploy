# Training probes — small models, Nov 24–25 2026

The NF4 QLoRA training-gate + capability probe harness used for the
Nov 24–25 2026 probe rounds (7 causal-LM models + NVIDIA
Nemotron-Labs-Sigma 3B/8B), run on free Colab T4s (14.56 GiB) for the
arXiv digest training track. These scripts are **training probes, not
serving recipes**: none of them is a `deploy_llm.py` recipe and none of
the models here is marked `proven_t4` for inference/serving.

Measured numbers and the full write-up live in
[docs/small-model-training-probes-2026-11-25.md](../docs/small-model-training-probes-2026-11-25.md)
(7 causal models) and the Sigma verdicts below; the machine-readable
standings are the `training_probes` entries in `models.json`. Vendor
benchmark claims were **not** re-verified and are not used as evidence
anywhere in this folder.

## How to run a probe

One live Colab T4 session at a time, via the `hermes-colab-cli`
wrapper (`colab.py` — never the raw `google-colab-cli` package), with
an isolated session registry per model:

```bash
colab.py new -s <slug>-probe --gpu T4 --registry-file <slug>/registry.json
colab.py upload probe.py run_probe.py          # from this folder / the model's folder
colab.py exec_detach -f run_probe.py
# poll /content/<slug>_results*.json + /content/probe_run.log via logs/download
# verify kernel liveness (not just status timestamps), then:
colab.py stop -s <slug>-probe
```

**Training gate (same for every causal model):** NF4 load + LoRA
r=16/α=32 attach + one real training step complete, peak VRAM < 14 GB,
loss finite, grads flow. **Probe env (unless noted):** torch
2.14.1+cu130, transformers 5.22.1 (shipped on Colab), peft 0.21.2,
accelerate 1.15.0, bitsandbytes 0.50.2, paged_adamw_8bit, batch 1.
Capability battery: tool-JSON, reasoning math/code, instruction
following, needle 4k/8k, Chinese, temp-0 consistency, decode tok/s.

## Files

- `probe.py` / `probe2.py` — shared harness (gate + capability
  battery, results JSON). `probe_slim.py` — slim variant used for the
  SenseNova retry after a reclaim. `probe_unsloth2.py` — Unsloth 4-bit
  fallback path (not needed in these rounds; the transformers path
  succeeded everywhere it applied).
- `<model-slug>/run_probe.py` — per-model runner (load, LoRA attach,
  training step, battery). Extra per-model files are follow-ups
  (e.g. `delta_test.py`, `conv_test.py`, `sigma_sweep.py`).

## Per-model verdicts (measured, 2026-11-24/25)

Default stays **Qwen3.5-9B**.

| Model | Verdict | Key numbers | Recipe note |
|---|---|---|---|
| Alpie Seed 9B (`169Pi/Alpie_seed`) | GREEN, first alternate | Peak 7.57 GB, loss 0.2303, reasoning 8/8, tool 2/3, 9.5 tok/s | Prefer for reasoning/code papers |
| TwIL-8B (`webAI-Official/TwIL-8B`) | GREEN, reasoning specialist only | Peak 5.85 GB, loss 0.1178, reasoning 7/8, tool 1/3 strict, 9.8 tok/s | webAI Non-Commercial 1.0 — research/education only |
| SenseNova Loop-9B (`sensenova/SenseNova-Loop-9B`) | Gate GREEN, out of rotation | Peak 8.42 GB, loss 0.1754, step 15.2 s, reasoning 4/8, tool 3/3, 1.7 tok/s | Pass `use_cache=False` explicitly (custom forward reads it from its own config); budget 2–6× a Qwen3.5-class run |
| Qwen3.5-4B (`Qwen/Qwen3.5-4B`) | GREEN, 4B general ablation arm | Peak 3.58 GB, loss 0.1832, capability 18/21, tool 3/3, reasoning 5/8, needle 8k PASS | `pip uninstall -y torchao` first, or drop `conv1d` (depthwise, groups=8192, cannot take PEFT LoRA); DeltaNet LoRA proven, all 248 B grads nonzero |
| LFM3-1.7B (`LiquidAI/LFM3-1.7B`) | GREEN, size/agent arm | Peak 1.44 GB, loss 0.3153, capability 17/21, reasoning 7/8, needle 8k PASS, 18.0 tok/s | `trust_remote_code=True`; uninstall torchao; score tools in native `<tool_call>` format (raw-JSON tool score 1/3); 13 depthwise conv kernels stay frozen |
| Dory 4B (`nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`) | Gate GREEN, no rotation place | Peak 3.24 GB, loss 0.1286, capability 17/21, reasoning 4/8, tool 3/3, 5.1 tok/s | Needs transformers git main ≥5.24 (shipped 5.22.1 fails at load) |
| MiniCPM5-2B (`openbmb/MiniCPM5-2B`) | Gate GREEN, no place | Peak 2.28 GB, loss 0.3134, capability 15/21, reasoning 5/8, tool 2/3, 11.8 tok/s | Cleanest port (native 5.22.1, no workarounds); fallback only |
| Sigma 3B-instruct (`nvidia/Nemotron-Labs-Sigma-3B-instruct`) | Research-only agent / **RED training** | Capability 17/21 **only** on transformers 4.55.4 + `schedule_parameterization="projected_bernstein"`; **2/21 as shipped on 5.22.1** (degenerate text); load 27.1 s, 7.14 GB, 2.1 tok/s effective; tool 3/3, reasoning 6/8, needle 4k PASS / 8k OOM | Continuous diffusion LM, not autoregressive: `forward()` raises `NotImplementedError`, no causal loss, no `lm_head`, no shipped training code; frozen scorer impossible (no logprobs). `sigma_probe.py` is the diffusion-aware harness — the causal `probe.py` does not apply |
| Sigma 8B-instruct (`nvidia/Nemotron-Labs-Sigma-8B-instruct`) | **RED, both tracks — no working T4 config** | No capability score (meaningful 0) | bf16 pure-GPU OOMs (16.99 GB weights); CPU-offload bf16 degenerate at 0.3 tok/s; NF4 loads (4.83 GB) but `generate` crashes on step 1 — the repo's `_denoise_block` reads `.weight.float()` on a `Params4bit`. **The `sigma-8b/` scripts document a RED path, not a working recipe.** Re-probe only if NVIDIA ships fixed config/code |

## Cross-cutting findings

- Needle 8k via transformers `generate` OOMs on a T4 for every 8–9B
  model (SDPA math path on sm_75 materializes the [heads, seq, seq]
  score tensor); all four smaller causal models passed 8k. Sigma 3B
  also OOMs at 8k.
- Shipped torchao 0.10.0 breaks any LoRA target list containing a
  Conv1d — uninstall it or drop the conv target. Depthwise conv
  kernels (groups 2048 / 8192) cannot take PEFT LoRA at r=16; they
  stay frozen and are a negligible param count.
- Harness artifact, recorded not hidden: the `grads_flow: False` flag
  in the results JSONs reads the first LoRA param (`lora_A`, grad 0 at
  init because B=0); total grad norm and nonzero `lora_B` grads are
  the real evidence.
