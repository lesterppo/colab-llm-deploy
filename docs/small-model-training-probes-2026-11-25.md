# Small-model training probes — 2026-11-24/25

Consolidated evidence for the arXiv digest training track. Seven models
were probed on free Colab T4s (14.56 GiB) for **training** — transformers
NF4 QLoRA, one real training step — not for serving. None of these is a
`deploy_llm.py` recipe and none is marked `proven_t4` for
inference/serving; the catalog entries live under the separate
`training_probes` key in `models.json`.

**Training gate (same for all seven):** NF4 load + LoRA r=16/α=32 attach +
one real training step complete, peak VRAM < 14 GB, loss finite, grads
flow. All seven passed the gate.

**Probe env (unless noted):** torch 2.14.1+cu130, transformers 5.22.1
(shipped on Colab), peft 0.21.2, accelerate 1.15.0, bitsandbytes 0.50.2,
paged_adamw_8bit, batch 1. Capability battery is the Inkling deep-test
axes (tool-JSON proxy, reasoning math/code, needle, Chinese, temp-0
consistency, decode tok/s), scored by the probe harness — vendor
benchmark claims were **not** re-verified and are not used as evidence
anywhere below.

**Default stays Qwen3.5-9B.** Full per-model verdicts (results JSON,
probe logs, run scripts) live in the probe workspace
`~/workspace/small-model-probes/<slug>/` and are **not** in this repo.

## 1. Alpie Seed 9B — GREEN, first alternate

- Model: `169Pi/Alpie_seed`, 9.41B, Apache 2.0, safetensors FP16
  (single 18.8 GB file), Qwen3.5-9B architecture, SFT+GRPO, 262K ctx.
  Session `alpie-probe`, cyc236ha.
- Training: NF4 load 610 s (mostly the unauthenticated 18.8 GB
  download), VRAM after load 7.12 GB; LoRA r16, 3.93M trainable; step
  seq 120, 3.0 s, loss 0.2303 finite, total grad norm 0.2548;
  **peak step VRAM 7.57 GB**.
- Capability (15/16 completed): reasoning **8/8**; tool-JSON 2/3
  (parallel-3 naming slip `get_stock`/`symbol` vs
  `get_stock_price`/`ticker`); instruction 2/2; needle 4k PASS,
  **8k OOM** (see cross-cutting); decode 9.5 tok/s. Chinese and temp-0
  consistency not measured — battery aborted at the 8k OOM. Vision
  tower not separately verified.
- Recommendation: adopt as alternate training-track base alongside
  Qwen3.5-9B; prefer for reasoning/code papers; keep Qwen3.5-9B as
  default until a head-to-head training run exists.

## 2. TwIL-8B — GREEN, reasoning specialist only

- Model: `webAI-Official/TwIL-8B`, 8.79B, base
  `ibm-granite/granite-4.2-8b`, dense `GraniteForCausalLM`, 131K ctx,
  safetensors balanced variant. Session `twil-probe`, ppoppo205.
- Training: load 197 s (fastest of the 8–9B round), VRAM after load
  5.36 GB; LoRA r16, 17.04M trainable; step seq 256, 6.7 s, loss
  0.1178 finite, total grad norm 0.4760; **peak step VRAM 5.85 GB** —
  lightest of the three 8–9B probes.
- Capability 17/21 strict (18/21 effective): reasoning 7/8
  (combinatorics 54 failed — verbose thinking trace did not finish
  within 1024 tokens); tool-JSON 1/3 strict / 2/3 effective
  (single-tool output was correct JSON but ```json-fenced, a harness
  parser artifact; same parallel naming slip as Alpie); instruction
  2/2; Chinese 3/3 (substring check only — zh_qa hallucinated "香港的首
  都是香港市"); consistency 1/1; needle 4k PASS, **8k OOM**; decode
  9.8 tok/s.
- **License: webAI Non-Commercial 1.0** (base Apache 2.0) —
  research/education only, cannot go into products. This is the main
  reason it ranks behind Alpie as a general default.
- Recommendation: admit as the reasoning-paper specialist; do NOT make
  it the general default.

## 3. SenseNova Loop-9B — gate GREEN, OUT of rotation

- Model: `sensenova/SenseNova-Loop-9B`, 9.66B, Apache 2.0, safetensors
  BF16 (12 shards, ~19 GB), 512K ctx, looped (32 layers, middle 24 run
  twice = 56 executions, ~14.6B inference-equivalent),
  `trust_remote_code` custom code. Attempt 1 (`sensenova-probe`,
  cyc236es) **reclaimed after ~55 min** mid-capability; retry
  (`sensenova-retry`, kyu008008) completed with a slim probe.
- Training: load 424 s (attempt 1) / 781 s (retry), VRAM after load
  7.35 GB; LoRA r16, 3.93M trainable. First step attempt **failed in
  0.0 s** — `ValueError: Disable use_cache when training with gradient
  checkpointing` from custom `modeling_sensenova_loop.py` (custom
  forward reads `use_cache` from its own inner config; **harness bug,
  not a model defect**), fixed in-session by passing `use_cache=False`
  explicitly. Fixed step: seq 256, **15.2 s**, loss 0.1754 finite,
  total grad norm 5.7689, **peak 8.42 GB**.
- Capability 15/18 measured: tool-JSON **3/3** (only full marks of the
  three 8–9B models), instruction 2/2, Chinese **3/3** (best — correctly
  explains Hong Kong has no capital), consistency 1/1, reasoning
  **4/8 (weakest)**, decode **1.7 tok/s (5–6× slower)**. Needle not
  measured.
- Recommendation: **do NOT admit to the daily rotation.**
  Research-only for papers specifically about looped/weight-shared
  architectures; budget 2–6× the time of a Qwen3.5-class run.

## 4. Qwen3.5-4B — GREEN, 4B general ablation arm

- Model: `Qwen/Qwen3.5-4B`, 4.7B total (4.21B text-only), Qwen3.5
  hybrid Gated DeltaNet, Apache 2.0, safetensors, 262K ctx. 32 layers:
  24 DeltaNet + 8 full-attention. Session `qwen35-4b-probe`, ppoppo205.
- Training: load 129.9 s, VRAM after load 2.91 GB. LoRA stage 1 (full
  target list incl. conv1d) **failed atomically** on torchao 0.10.0
  (PEFT's Conv1d LoRA path imports torchao; shipped 0.10.0 < 0.16.0).
  Stage 2 base-only OK (3.15M trainable), step 1.8 s, loss 0.1832,
  peak 3.33 GB. **DeltaNet follow-up: the 9B QLoRA recipe ports.**
  With torchao uninstalled: full list *with* conv1d still fails (the
  DeltaNet `conv1d` is depthwise, groups=8192, PEFT requires rank
  divisible by groups); with conv1d dropped: **32.46M trainable**,
  step 3.2 s, loss 0.1832 finite, peak **3.58 GB**, and **all 248 LoRA
  B params have nonzero grads** — 248 is exactly the target-module
  count, so all 120 DeltaNet projection B's
  (in_proj_qkv/z/b/a + out_proj) train. No OOM, no silent no-op.
  Reference-kernel warnings only (`causal_conv1d` /
  `flash-linear-attention` not installed).
- Capability **18/21**: tool-JSON **3/3** (best of the smaller models),
  reasoning 5/8 (weakest axis), instruction 2/2, **needle 4k+8k
  PASS**, Chinese 3/3 genuinely correct ("香港没有首都，因为香港是中国
  的一个特别行政区，不是国家"), temp-0 1/1, decode 10.8 tok/s.
- Recipe note: `pip uninstall -y torchao` first, or drop `conv1d` from
  targets — either way the DeltaNet conv kernels stay frozen.
- Recommendation: earns a place as the **4B general ablation arm** —
  same family as the Qwen3.5-9B default, DeltaNet LoRA proven, tool
  3/3. Keep the 9B default for general papers (reasoning 5/8 here).

## 5. LFM3-1.7B — GREEN, size/agent arm

- Model: `LiquidAI/LFM3-1.7B`, 1.73B tri-hybrid (13 short-conv + 5 SWA
  + 6 global attn, 24 layers), 256K ctx, LFM Open License v1.0,
  safetensors BF16 3.46 GB. Session `lfm3-probe`, cyc236ha.
- Training: load attempt 1 native **FAILED** — shipped transformers
  5.22.1 still requires `trust_remote_code=True` for LFM3's custom
  modeling code. Attempt 2 with it: load 59.5 s, **VRAM after load
  1.20 GB**; LoRA r16 (q/k/v/o + gate_proj + in_proj + out_proj),
  5.26M trainable; step 0.9 s, loss 0.3153 finite, total grad norm
  1.86, **peak step VRAM 1.44 GB**.
- Conv question — answered: the conv *path* trains (LoRA B grads
  0.15–0.47 on in_proj/out_proj), but the 13 depthwise Conv1d kernels
  (~186k params) **cannot** take PEFT LoRA — target `conv` matched the
  `Lfm3ShortConv` container (unsupported), target `conv.conv` hit
  PEFT's "rank must be divisible by groups (2048)" refusal; also
  needed torchao uninstalled (shipped 0.10.0 < 0.16.0). Frozen kernels
  are immaterial.
- Capability 17/21: reasoning 7/8, tool-JSON 1/3 (single PASS;
  parallel FAIL — emitted native `<tool_call>` format; no-tool FAIL —
  invented a call), instruction 2/2, **needle 4k+8k PASS**, Chinese
  3/3 (substring artifact; zh_qa hallucinated a capital), temp-0 1/1,
  smoke FAIL ("2"), decode **18.0 tok/s** (fastest probed).
- Recommendation: earns a place as the **size-ablation / agent arm** —
  1.2 GB load, 1.4 GB training peak, needle-8k pass, reasoning 7/8.
  Score tools in its native `<tool_call>` format, not raw JSON. Not
  the general default.

## 6. Dory 4B — gate GREEN, NO rotation place

- Model: `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`, 4B looped
  latent-recurrent nGPT (8+20+8 unique layers, 1–4 loops), 256K ctx,
  OpenMDW-1.1. Session `dory-4b-probe`, cyc236es.
- Training: attempt 1 on shipped 5.22.1 **FAILED** fast
  (`layer_types` validation — no Dory modeling code); attempt 2 on
  **transformers git main** loaded in 211.8 s, **2.84 GB** after load;
  step 2.8 s, loss 0.1286 finite, total grad norm 0.57, **peak step
  VRAM 3.24 GB**. LoRA stage 1 OK first try: q/k/v/o + gate/up/down,
  **32.44M trainable** (36 unique decoder layers; config's 72 counts
  loop executions). Shared-recurrent LoRA trains stably — one step,
  default 2 loops, finite loss, no explosion.
- Capability 17/21: tool-JSON 3/3, reasoning **4/8 (weakest in the
  whole 7-model series)**, instruction 2/2, **needle 4k+8k PASS**,
  Chinese 3/3 genuinely correct, temp-0 1/1, decode **5.1 tok/s**
  (slowest — looped execution costs).
- Recommendation: GREEN on the gate, **no place in the rotation** —
  strictly dominated by Qwen3.5-4B (tools equal, reasoning better, 2×
  decode) and needs a git-main pin until Colab ships 5.24. Revisit
  only for a code/agent paper testing its vendor SWE-bench edge, after
  probing loop count 4 and multi-step stability.

## 7. MiniCPM5-2B — gate GREEN, NO place

- Model: `openbmb/MiniCPM5-2B`, 2.52B, standard `LlamaForCausalLM`,
  Apache 2.0, 131K ctx, 42 layers. Session `minicpm5-2b-probe`,
  kyu008008.
- Training: cleanest port in the series — single attempt, native
  5.22.1, standard Llama, no workarounds. Load 61.0 s, **1.95 GB**
  after load; step 1.2 s, loss 0.3134 finite, total grad norm 1.70,
  **peak step VRAM 2.28 GB**. LoRA stage 1 OK first try, **25.12M
  trainable**.
- Capability **15/21 (lowest of the four smaller models)**: smoke FAIL
  ("2"), tool-JSON 2/3 (parallel returned bare function names),
  reasoning 5/8, instruction 1/2 (3 lines, zero bullet markers),
  **needle 4k+8k PASS**, Chinese 3/3 genuinely correct, temp-0 1/1,
  decode 11.8 tok/s.
- Recommendation: GREEN on the gate, **no place in the rotation** —
  the easiest port doesn't compensate for the weakest capability;
  Qwen3.5-4B beats it on every axis at ~1 GB more VRAM. Fallback only
  if Qwen3.5-4B and LFM3 are both unavailable.

## Cross-cutting findings

- **Needle 8k via transformers generate OOMs on a T4 for every 8–9B
  model** (SDPA math path on sm_75 materializes the [heads, seq, seq]
  score tensor). Long-context eval on T4 needs llama.cpp/vLLM or
  chunked attention. **All four smaller models passed needle-8k**
  (7,111–7,117 tokens) — small hidden sizes keep the SDPA score tensor
  affordable on the T4.
- **Shipped torchao 0.10.0 breaks any LoRA target list containing a
  Conv1d** (LFM3, Qwen3.5-4B) — uninstall torchao or drop the conv
  target. Depthwise conv kernels (groups 2048 / 8192) cannot take PEFT
  LoRA at r=16 in either model; they stay frozen and are a negligible
  param count.
- Harness artifacts, recorded not hidden: the `grads_flow: False` flag
  in every results JSON reads the first LoRA param (`lora_A`, grad 0 at
  init because B=0); total grad norm (0.25–5.77 across the seven) and
  nonzero `lora_B` grads are the real evidence. `total_params_b` in the
  JSONs is roughly half the card value — bnb `Params4bit.numel()`
  counts packed storage.

## Rotation recommendation (2026-11-25)

- Default stays **Qwen3.5-9B**.
- **Alpie Seed 9B** — first alternate, prefer for reasoning/code
  papers.
- **TwIL-8B** — reasoning-paper specialist only (non-commercial
  license).
- **SenseNova Loop-9B** — out of rotation, research-only for
  looped-architecture papers, budget 2–6× the time.
- **Qwen3.5-4B** — 4B general ablation arm.
- **LFM3-1.7B** — size-ablation / agent arm.
- **Dory 4B** — no rotation place.
- **MiniCPM5-2B** — fallback only if Qwen3.5-4B and LFM3 are both
  unavailable.
- Neither smaller arm replaces the 8–9B default for general papers —
  reasoning is the binding constraint (LFM3 7/8, Qwen3.5-4B 5/8,
  Dory 4/8, MiniCPM5 5/8 vs Alpie 8/8).

Inkling 2 Mini stays **excluded from training** (RED probe: transformers
dequantizes its NVFP4/MXFP8 weights to bf16 at load) and is already an
optional agent-track subject in the arxiv-lab WORKFLOW §4.4 — it is not
part of these standings.
