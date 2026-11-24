# Inkling 2 Mini — deep test on Colab T4 (2026-11-24)

Thinking Machines Lab Inkling 2 Mini (26.4B total / 3.4B active looped MoE,
Apache 2.0, released 2026-11-20). Served file:
`unsloth/Inkling-2-Mini-GGUF`, `Mainline_Support/Inkling-2-Mini-UD-IQ3_XXS.gguf_file`
(11.66 GB, 94.7% top-1 per Unsloth table). **Note the literal `.gguf_file`
suffix on HF** — the first deploy verified the name against the repo tree;
llama.cpp loads by magic, not extension.

**Setup:** account `cyc236ha` (kyu009009 failed first with capacity
"Backend busy"), T4 15,360 MiB, llama.cpp **b12669** CUDA + matching cudart
bundle, `llama-server --jinja -ngl 99 -fa on -c 16384 --port 11434`.
Test script: `~/workspace/inkling-2-mini/deep_test.py`; raw results:
`~/workspace/inkling-2-mini/inkling_deep_results_iq3_ctx16k.json`
(ctx-8k run also saved as `inkling_deep_results_iq3_ctx8k.json`).
Sampling temp 0 throughout (vendor default is temp 1.0, noted where it
matters); reasoning left at the server default — the trace counts against
`max_tokens`, so budgets here are generous (256–2048).

## Results summary (ctx 16384 run)

| # | Area | Result |
|---|------|--------|
| A | Throughput | decode **23.6 tok/s**, TTFT **0.41 s**, prefill **131 tok/s** @10,824 prompt tokens (82.6 s wall) |
| B | Tool calling (6 scenarios) | **5/6** — single ✓, parallel-3 ✓, multi-turn loop ✓, nested+enum schema ✓, no-tool-needed ✓, forced `tool_choice` ✗ |
| C | Reasoning (12 problems) | **12/12** — incl. combinatorics (3-digit digit-sum-10 = 54) and look-and-say |
| D | Instruction following | **3/3** — strict JSON ✓, pirate persona 3 turns ✓, exactly-3-bullets ✓ |
| E | Needle-in-haystack | **2/2** — @10,340 tokens (BLUE-FALCON-7) and @13,148 tokens (RED-TIGER-42) |
| F | Multilingual (ZH) | **3/3 on inspection** — fluent ZH Q&A, ZH→EN and EN→ZH correct (scorer logged 2/3: see caveat) |
| G | Consistency (temp 0) | identical outputs ✓ ("Canberra" twice) |
| H | VRAM | **14,025 MiB after load @16k** · end 14,217 MiB · ~1.3 GB headroom |
| V | Vision (stretch) | ✓ describe-test correct in 9.4 s (red circle / green triangle / blue rectangle) with mmproj F16; VRAM 14,735 MiB |

## Verdict vs the vendor's claims

1. **Reasoning is strong — SUPPORTED on this battery.** 12/12, including
   the combinatorics problem Saluki failed (54, correct here) and
   look-and-say (312211). No reasoning failure in 12 problems.
2. **Tool calling intact — SUPPORTED.** 5/6, including 3 parallel calls in
   one turn (Hong Kong weather + Tokyo time + AAPL, all correct args),
   a two-turn tool loop, and nested object args + enum (`book_flight`:
   passenger.name/passport, cabin=economy, all exact).
3. **Vendor benchmark claims (AA Index v5 20.1, SWE-bench Verified 79.2%,
   FrontierCode v2.0 15.1%, Terminal-Bench v4.0 5.6%, MMMU Pro 74.9%,
   BrowseComp 79.0%) — NOT VERIFIABLE here.** Those need the vendors'
   harnesses; this battery probes the same dimensions but is not their
   benchmark.
4. **Coding — SUPPORTED on a spot check.** Iterative Fibonacci, correct,
   11.5 s wall (255 completion tokens).

## Caveats

- **B4 forced `tool_choice`:** same as Saluki — with
  `tool_choice: {"type":"function","function":{"name":"get_weather"}}` and
  prompt "Tell me a joke", the model answered the joke directly with zero
  tool calls. Dict-form forced choice is not honored in this llama-server +
  jinja setup — a harness/template behavior note, not proof the model
  can't do forced calls. `tool_choice: "auto"` works fine.
- **F zh_to_en scorer artifact:** the model answered "Artificial
  intelligence is changing the healthcare industry." — a correct
  translation. The scorer required the substring "medical" and logged a
  fail. The committed script now accepts medical/healthcare/medicine.
  True F result is 3/3.
- **Cold start:** first completion after load took ~24 s wall (Q3 run,
  server log: prompt eval 19.4 s for 34 tokens — first-request graph
  warmup — then 126 tokens @25.3 tok/s). Steady-state TTFT is 0.41 s.
- **Long-prompt prefill is slow:** 10,824 tokens took 82.6 s wall
  (131 tok/s). Decode stays ~24 tok/s. Budget accordingly for long
  contexts.
- **Tunnel:** `https://communicate-importantly-endif-found.trycloudflare.com`
  was live on the VM, but every request from this VM's egress got
  Cloudflare 403 (datacenter block — same as the Nemotron case in
  AGENTS.md). All live/deep-test traffic here ran VM-side over localhost;
  the tunnel was NOT verified from a residential network.

## The Q3_K_XL failure (why the recipe is IQ3, not Q3)

The primary `UD-Q3_K_XL` (13.63 GB, 96.3% top-1) **loaded and answered
correctly** (2+2 = "4", decode ~25 tok/s, VRAM 14,885/15,360 MiB) but
**crashed mid-deep-test with CUDA OOM** in `cudaGraphInstantiate`
(`ggml_cuda_graph_launch_chunk`) — only ~475 MiB VRAM headroom after load,
and a new CUDA graph allocation tipped it over. The recipe is recorded in
`models.json` as failed on T4; do not retry it blindly.

Two knock-on findings, both worth keeping:

- **ctx 32768 OOMs at load** on this file: compute-buffer allocation of
  3,365 MiB failed. **16384 is the T4-feasible max** for IQ3_XXS
  (14,025 MiB after load). A 21k-token needle therefore cannot run here;
  the long needle is 14k instead (13,148 actual prompt tokens).
- **Driver watchdog is zombie-blind:** after llama-server died, its pid
  stayed a zombie, `alive(pid)` kept returning true, and the watchdog
  kept writing `stage: running` instead of dying. The in-place fallback
  (kill driver, serve IQ3_XXS on the same port/tunnel) worked around it.
  A future driver fix: reap the server proc (`poll()`) instead of
  `alive()` in the watchdog.

## Operational notes (caught live this run)

- `colab.py exec` has a ~120 s timeout: three sequential chat requests
  (~65 s of model time) timed out at the CLI layer while the server was
  healthy — run long batteries via `exec_detach -f` with a Python wrapper
  that writes results to a file, and poll the file. (`exec_bg --code`
  silently did not start the process in this session; `exec_detach` with
  a `.py` file is the reliable path. Note `exec_detach` executes the file
  as **Python** — a `.sh` file fails with SyntaxError.)
- The reasoning trace counts against `max_tokens`: a 16-token budget on
  "What is 2+2?" returns empty content (trace eats the budget). Budgets
  of 256+ for short answers, 512 for ZH (reasoning ~200 tokens), 2048
  for hard reasoning.
- The deploy's host-side smoke test 403'd (see Tunnel caveat) even though
  the driver reached stage 6 `ready:true` — the VM-side localhost probes
  are the ground truth for "serving".
