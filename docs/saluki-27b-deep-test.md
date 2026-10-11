# Saluki 27B — deep test on Colab T4 (2026-10-11)

Follow-up to the morning smoke test (GREEN). Model: `ConwayResearch/Underdog-Saluki-27B-1.0`,
file `Underdog-Saluki-27B-1.0-IQ2-mix.gguf` (7.89 GB, 2-bit quant of Qwen3.8-27B).

**Setup:** account `kyu008008`, T4 15.4 GB, llama.cpp **b11552** CUDA + matching cudart bundle
(`LD_LIBRARY_PATH` — see pitfall below), `llama-server --jinja -ngl 99 -fa on -c 32768 --port 18080`.
This run used the model card's recommended **32k context** (morning ran 8192).
Sampling per model card: reasoning = thinking on, temp 0.6 / top_p 0.95 / top_k 20;
tool calls = thinking off, temp 0. Test script: `~/workspace/saluki-test/deep_test.py`;
raw results: `~/workspace/saluki-test/saluki_deep_results.json`.

## Results summary

| # | Area | Result |
|---|------|--------|
| A | Throughput | decode **9.4 tok/s**, prefill 46 tok/s (short) / **235–246 tok/s** (long prompts), TTFT 0.62 s |
| B | Tool calling (6 scenarios) | **5/6** — single ✓, parallel-3 ✓, multi-turn loop ✓, nested+enum schema ✓, no-tool-needed ✓, forced `tool_choice` ✗ |
| C | Reasoning (12 problems) | **11/12** — only miss: 3-digit combinatorics (expected 54) |
| D | Instruction following | **3/3** — strict JSON ✓, pirate persona 3 turns ✓, exactly-3-bullets ✓ |
| E | Needle-in-haystack | **3/3 @ 11.6k tokens**, **3/3 @ 21.6k tokens** |
| F | Multilingual (ZH) | fluent ZH Q&A, correct ZH→EN and EN→ZH |
| G | Consistency (temp 0) | identical outputs ✓ |
| H | VRAM | 10,385 MiB after load @32k · peak 10,429 MiB · ~5 GB headroom |

## Verdict vs the post's claims

1. **Tool calling intact (88/120 vs 84/120 full model) — SUPPORTED.** 5/6 scenarios pass,
   including 3 parallel tool calls in a single turn (`get_weather` Hong Kong +
   `get_time` Asia/Tokyo + `get_stock_price` AAPL, all correct args) and a two-turn
   tool loop (Hong Kong → tool result fed back → Tokyo). Nested object args + enum
   (`book_flight`: passenger.name/passport, cabin=economy) parsed perfectly.
2. **Parallel tool calls (42/100 vs 35/100) — SUPPORTED** on the tested shape (B2).
3. **96% average over 9 benchmarks — NOT VERIFIABLE here.** That needs Underdog's own
   harness; this battery probes the same dimensions but is not their benchmark.
4. **Math/reasoning is the weak spot (~82–85%) — PLAUSIBLY CONSISTENT.** 11/12 looks
   strong, but the single failure was the hardest problem (combinatorial counting:
   3-digit numbers with digit-sum 10 = 54 — model thought 125 s and emitted no answer),
   while all routine word problems, logic puzzles (knights/knaves, portrait, look-and-say,
   Monty Hall) passed. So: solid reasoning, combinatorics is where it cracks.

## Caveats

- **B4 forced `tool_choice`:** with `tool_choice: {"type":"function","function":{"name":"get_weather"}}`
  and prompt "Tell me a joke", the model answered the joke directly with **zero tool calls**
  (verified on re-run). The dict-form forced choice is not honored in this
  llama-server + jinja template setup — a harness/template behavior note, not proof the
  model can't do forced calls. `tool_choice: "auto"` works fine.
- **Cold start:** first request after load took ~159 s this morning (steady-state after that).
- **Needle token counts:** filler targeted 16k/30k tokens but landed at 11.6k/21.6k actual
  prompt tokens (char→token estimate was off) — still a genuine 21.6k-token retrieval test, 3/3.
- Only 32k context tested for KV fit; the card's MTP speculative-decoding variant was not tested.

## Operational pitfall (caught live this run)

The cudart tarball extracts to a **nested subdirectory**
(`cudart-llama-b11552-bin-ubuntu-cuda-12.8-x64/libcudart.so.12`), so
`LD_LIBRARY_PATH=/content/cudart` does **not** find it → llama-server starts with
**zero CUDA mentions in the log, 0 MiB VRAM, 0.51 tok/s** (silent CPU fallback).
Fix: point `LD_LIBRARY_PATH` at the nested dir containing `libcudart.so*`
(`deep_test.py: find_cudart_libdir()`). Ground truth is `nvidia-smi`, not the log —
the script now aborts the battery if VRAM < 1 GB after load.
