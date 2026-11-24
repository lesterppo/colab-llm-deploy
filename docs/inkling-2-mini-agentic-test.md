# Inkling 2 Mini — agentic-capabilities test on Colab T4 (2026-11-24)

Follow-up to the morning deep test (GREEN: tool 5/6, reasoning 12/12).
Same model, same recipe — `inkling-2-mini-iq3`, Unsloth UD-IQ3_XXS
(11.66 GB) via llama.cpp b12669, ctx 16384 — but now testing **agentic**
behavior: multi-step tool loops, error recovery, terminal-bench-style
code-agent work (the harness actually executes the model's scripts),
planning under constraints, long-horizon memory, tool-choice discipline,
and belief update. The harness mediates ALL tool execution; the model
never gets raw shell access. Code tasks run in `/content/agent_sandbox`.

**Setup:** account `kyu009009`, T4 15,360 MiB, `llama-server --jinja
-ngl 99 -fa on -c 16384 --port 11434`. Test script: `inkling-2-mini-agentic-test.py` (repo root; local copy
`~/workspace/inkling-2-mini/agentic_test.py`); raw results:
`~/workspace/inkling-2-mini/agentic_results.json` (as-run 12/14) and
`agentic_results_rerun.json` (confirmation re-run of 2 scenarios).
Sampling temp 0, reasoning at server default; max_tokens 2048 (4096 for
code tasks) — the reasoning trace counts against the budget. Max 10 tool
turns per task (8 in practice); 50-min overall deadline, results written
incrementally after every scenario.

## Results summary (as-run 12/14; corrected 13/14 — see caveats)

| # | Area | Result |
|---|------|--------|
| 1 | Multi-step tool loops (3) | **3/3** — trip planning ✓ (2 turns, 37 s), data task ✓ (exact JSON totals), multi-hop QA ✓ (author → birth year, 1920) |
| 2 | Error recovery (2) | **2/2** — bad argument ✓ (Atlantis error → retried Hong Kong), empty result ✓ (no flights Dec 25 → searched other dates, never invented a flight) |
| 3 | Code agent (3) | **2/3** — CSV stats ✓ (revenue $900.00, banana), hidden fizzbuzz tests ✓ (4/4), fix-the-buggy-script ✗ **twice** |
| 4 | Planning + constraints (2) | **2/2 corrected** — budget plan ✓ (TOTAL $448.00, exact, ≤$500), exact-format ✓ (re-scored; see caveats) |
| 5 | Long-horizon memory (1) | **1/1** — code name BLUE-FALCON + date 2027-03-15 recalled after 4 distractor turns |
| 6 | Tool-choice discipline (2) | **2/2** — picked only `get_stock_price` among 6 tools ✓; zero tool calls when none needed ✓ (2+2, capital of France) |
| 7 | Belief update (1) | **1/1** — noticed a stale $145.00 quote, re-checked, reported the corrected $152.75 |

Per-scenario cost (turns / tool calls / wall): trip 2/3/37 s, data 2/3/16 s,
multihop 3/2/12 s, bad-arg 3/2/18 s, empty-result 4/7/57 s, csv-stats 8/7/54 s,
fizzbuzz 6/5/44 s, fix-buggy 8/9/33 s, budget 4/13/62 s, exact-format 2/2/13 s,
memory 6/0/4 s, distractors 2/1/9 s, no-tool 2/0, belief 3/2/19 s.
Whole battery: ~7 min wall. VRAM 14,119 → 14,123 MiB.

## Verdict vs the deep test and the vendor's claims

1. **Single-turn tool calling holds up multi-turn — SUPPORTED.**
   The deep test's 5/6 extends cleanly: 3/3 multi-step loops, 2/2 error
   recovery, 2/2 tool-choice discipline. No wrong-tool or hallucinated-
   tool-result failure in 14 scenarios.
2. **Error recovery and honesty are the strongest axis.** On the empty
   flight search it searched 7 times across dates and explicitly refused
   to invent a flight; on the stale quote it re-checked unprompted and
   flagged the correction in its answer.
3. **Code agent: writes and verifies well, but cannot fix — MIXED.**
   It wrote correct scripts from scratch twice (CSV stats, fizzbuzz 4/4
   hidden tests) and self-verified by running them. But it failed the
   one-line bug fix **twice in a row** (see below). As a coding agent it
   is a good generator, an unreliable debugger.
4. **Vendor agentic claims (Terminal-Bench v2.1 71.6% / v4.0 5.6%,
   SWE-bench Verified 79.2%, BrowseComp 79.0%, MCP Atlas 82.7%,
   AutomationBench 41.8%) — NOT VERIFIABLE here.** Those need the
   vendors' harnesses; this battery probes the same dimensions at small
   scale (n=14) and is not their benchmark.

## The fix-buggy failure (systematic, reproduced twice)

Task: `/content/agent_sandbox/buggy.py` crashes with
`TypeError: unsupported operand type(s) for /: 'int' and 'str'`
(`count = "3"` on line 3). Fix it, re-run, report the result (20).

Attempt 1 (8 turns, 9 calls, 33 s) and attempt 2 (8 turns, 8 calls,
29 s) both ended `[MAX TURNS REACHED]` with the file unchanged
(`rerun_rc=1`, same TypeError). The transcript shows the same pattern
both times: the model runs the script, sees the TypeError, reads the
file — and then **never calls `write_file`**. Instead it burns the
remaining turns re-running leftover scripts from earlier scenarios
(`stats.py`, the fizzbuzz checks, `sales.csv`):

> `run_script buggy.py` → `TypeError: ... 'int' and 'str'`
> `run_script _cat.py` → prints the buggy file back, unchanged
> `run_script check_import.py` → `import and assertions OK` (fizzbuzz)
> `run_script ...` → `Total revenue: 900.00 ... banana (200 units)`
> `run_script buggy.py` → same TypeError — max turns

Notable: its turns are fast (~4 s each — near-empty replies + a tool
call), so it exhausts 8 turns in ~30 s without ever attempting the
edit. It knows how to use `write_file` (it did so correctly in the two
passing code tasks); on a *fix* task it drifts into irrelevant tool
calls instead. Practical consequence: supervise edit tasks — ask it to
state the fix before applying it, or it will wander.

## Most impressive success (belief update)

> User: "What is the current price of AAPL?"
> Tool (1st call): `AAPL: $145.00 (WARNING: delayed quote from 09:00 —
> the market has moved since then; consider re-checking ...)`
> Tool (2nd call, unprompted): `AAPL: $152.75 (live quote as of 15:30).`
> Model: "AAPL is currently trading at **$152.75** (live quote as of
> 15:30). Note: an earlier delayed quote showed $145.00, but the live
> quote is $152.75."

It detected the staleness signal in the tool output, re-checked without
being asked, and reported the corrected value while explicitly flagging
the superseded one. That is the behavior you want in an agent.

## Caveats

- **Two scorer bugs, both caught live and fixed — the model was right
  both times.** (1) Trip planning first logged FAIL because the model
  wrote "$500 USD = **77,600 JPY**" and the scorer wanted the substring
  "77600". The script now strips thousands separators before numeric
  checks; the re-run passed. (2) Exact-format first logged FAIL because
  the scorer regex demanded an extra comma-separated field the format
  does not have; the model's two lines were exactly right
  (`Hong Kong: 24C, sunny` / `Tokyo: 18C, sunny`). Regex fixed, live
  re-run passed (2 turns, 11 s). Corrected tally is 13/14; the raw
  as-run file (12/14) is kept alongside the rerun file.
- **Attempt 1 stands as the scored run.** The confirmation re-run exists
  only to test whether failures reproduce: exact-format passed on
  re-run (scorer bug), fix-buggy failed again (systematic).
- **Session recovery:** the host-side deploy process died in a runtime
  restart ~4 min into the deploy (its log file is 0 bytes). The VM-side
  driver, launched detached, continued on its own and reached
  `tunnel_start`; the session was recovered by polling
  `/content/deploy_status.json` and probing `127.0.0.1:11434/health`
  (200) + a 2+2 completion ("4", VRAM 14,107 MiB) before testing.
- **Tunnel:** live on the VM, but every request from VM egress got
  Cloudflare 403 as before; all test traffic ran VM-side over localhost.
- **Small-n warning:** 14 scenarios, temp 0, single run (plus one
  confirmation re-run of 2 scenarios). Treat the per-axis fractions as
  directional, not as benchmark scores.
