# Inkling 2 Mini — extensive agentic-capabilities test on Colab T4 (2026-11-24)

Second, much bigger battery after the 14-scenario probe earlier today
(12/14 as-run, 13/14 corrected, MIXED — see
[docs/inkling-2-mini-agentic-test.md](inkling-2-mini-agentic-test.md)).
Same model, same recipe — `inkling-2-mini-iq3`, Unsloth UD-IQ3_XXS
(11.66 GB) via llama.cpp b12669, ctx 16384 — but now **41 scenarios in
3 chunks** (12 + 16 + 13), tiered easy/medium/hard, plus a 3-scenario rerun
of the cases that needed a second look. The harness mediates ALL tool
execution; the model never gets raw shell access. Code tasks run in
`/content/agent_sandbox`.

**Setup:** account `cyc236es` (see deploy incidents below), T4 15,360 MiB, `llama-server --jinja -ngl 99 -fa on -c 16384
--port 11434`. Test script: `inkling-2-mini-agentic-test-ext.py` (repo
root; local copy `~/workspace/inkling-2-mini/agentic_test_ext.py`),
rerun driver `inkling-2-mini-agentic-test-ext-rerun.py` (repo root; local
`~/workspace/inkling-2-mini/rerun_ext.py`). Raw results:
`~/workspace/inkling-2-mini/agentic_results_ext.json` (merged, all chunks
+ rerun) and the per-chunk files `agentic_results_ext_chunk{1,2,3}.json`.
Sampling temp 0, reasoning at server default; max_tokens 2048 (4096 for
code tasks) — the reasoning trace counts against the budget. Max 10 tool
turns per task (12 for the longest chains). Results written incrementally
after every scenario. Chunk wall: ~8 min (ch. 1), ~11 min (ch. 2),
~8 min (ch. 3), rerun ~4 min.

## Results summary

**As-run 36/41; corrected 39/41** (three as-run FAILs were re-scored on
transcript inspection — see corrections below; the rerun confirmed the two
genuine failures are deterministic).

| # | Axis | As-run | Corrected | Notes |
|---|------|--------|-----------|-------|
| 1 | Multi-step tool loops (7) | **7/7** | 7/7 | 5–6 turn chains, branching, parallel+sequential, chained args |
| 2 | Error recovery (5) | 3/5 | **4/5** | 1 scorer false-negative corrected; 1 genuine spiral (see below) |
| 3 | Code agent (10) | **9/10** | 9/10 | The weak axis from the probe, now mostly fixed (see §) |
| 4 | Planning + constraints (6) | 4/6 | **6/6** | 1 harness-design non-test re-tested and passed; 1 re-scored on inspection |
| 5 | Long-horizon memory (3) | **3/3** | 3/3 | 8 distractor turns, two facts, mid-conversation update |
| 6 | Tool discipline (4) | **4/4** | 4/4 | Incl. 2 clarification cases — asked instead of guessing |
| 7 | Belief update (3) | **3/3** | 3/3 | Incl. unprompted disagreement detection between two feeds |
| 8 | Long-context agent (3) | **3/3** | 3/3 | NEW: ~8k-token doc retrieval combined with tool use |

## The bug-fix question: prompt-sensitive, not absolute

The probe's headline weakness was the `count = "3"` bug-fix task failing
twice (model reads the TypeError, never calls `write_file`, burns turns
re-running old scripts). This battery re-tested it with one change:
**"state the fix in words before applying it"** — and it PASSED
(`s3_fix_state_first`, 9 turns, `write_file` called, fix stated correctly:
"count was assigned the string \"3\" instead of an integer", rerun
prints 20.0). The other 8 code tasks also passed: off-by-one (55),
wrong dict key (42), missing import, logic error (min→max, 25),
multi-file helper+main (42), TDD grade() 10/10 hidden tests via the
`run_tests` tool, extend-a-file (median 15.5), refactor (TAX_RATE,
output unchanged 120.0). Two setup differences from the probe are noted
honestly: this battery adds a `read_file` tool (the probe had none), and
the state-the-fix prompt. The weakness is therefore prompt-sensitive and
partly setup-sensitive — not an inability to fix code.

## The two genuine failures (both reproduced identically on rerun, temp 0)

1. **Open-ended "cheapest" search spirals forever** (`s2_empty_then_cheapest`,
   axis 2). Asked for the cheapest HKG→Tokyo flight around Christmas, the
   model searched **33 dates** (12-25 outward to 11-30, 4 searches per turn)
   across all 10 turns and never answered — even though it found CX520
   $320 on 12-26 at turn 2. Bit-identical on rerun (33 searches, 160 s).
   No stopping criterion on open-ended optimization. Contrast:
   `s4_cheapest_first` (same "cheapest" ask, plus a hotel step after) DID
   stop after 20 searches and answered correctly — so the loop is
   framing-sensitive, not inevitable.
2. **Fix succeeds but the loop never stops** (`s3_traceback_only`, axis 3).
   With no `read_file` tool, the model debugged resourcefully: re-ran the
   crashing script, then wrote a `show.py` helper to print the file,
   fixed `area(width)` → `area(width, height)` at turn 9, verified it at
   turn 10 (exit 0, output 20) — and hit max turns before producing a
   final answer. The file state was correct both times (rerun rc 0,
   out '20'); the failure is loop termination at the finish line, after
   the fix consumed the turn budget.

## Corrections (as-run → corrected, all from transcript inspection)

- `s2_cascading_errors` (axis 2): scorer false-negative. The model DID
  recover (Atlantis → Paris → Hong Kong, reported HK sunny 24°C); the
  scorer failed it only because the final text mentions "Atlantis" while
  explaining the recovery.
- `s4_constraint_change` (axis 4): harness-design non-test. The chunk-2
  inject-after-turn-2 never fired — the model answered the $500 plan in
  2 turns, before the injection point. Re-tested as
  `s4_constraint_change_v2` (plan to completion, then budget cut to $350
  as a new user turn): PASSED — phase 1 $418, phase 2 $343 after dropping
  Disney, change stated.
- `s4_impossible_budget` (axis 4): re-scored PASS on inspection. The model
  explicitly flagged the 3-night hotel ($420) as over the $300 budget,
  then transparently proposed a re-scoped 3-day/2-night plan at $280 using
  real tool prices — no invented price, arithmetic exact. The scorer
  demanded no TOTAL line at all, which was too rigid; the flagged +
  disclosed re-scope is correct agent behavior.

## Hardest tasks passed

- `s3_tdd_grade`: implemented `grade()` to 10/10 hidden boundary tests
  using the `run_tests` feedback loop.
- `s4_constraint_change_v2`: mid-task budget cut $500 → $350, adapted
  correctly ($418 → $343) and stated what changed.
- `s8_doc_conflicting_codes`: ~8k-token doc with two codes; picked SAVE20
  (20% off, final $200) and explicitly flagged WELCOME10 as expired.
- `s7_disagree_unprompted`: two stock feeds disagreed ($150.25 vs $148.90)
  with NO prompt to check both — the model checked both feeds on its own
  and flagged the mismatch.
- `s6_clarify_flight` / `s6_clarify_ticker`: underspecified requests —
  asked for the missing parameters with zero tool calls instead of guessing.

## Failure taxonomy (5 as-run FAILs)

| Class | Count | Scenarios |
|---|---|---|
| Max-turns loop, no stopping criterion | 1 | s2_empty_then_cheapest (33 searches, ×2 deterministic) |
| Max-turns after successful fix | 1 | s3_traceback_only (fix verified, no final answer, ×2 deterministic) |
| Scorer false-negative | 1 | s2_cascading_errors (corrected PASS) |
| Harness-design non-test | 1 | s4_constraint_change (v2 re-test PASSED) |
| Re-scored on inspection | 1 | s4_impossible_budget (flagged + disclosed re-scope, corrected PASS) |

## Comparison vs the 14-scenario probe

| Axis | Probe | Extensive (corrected) |
|---|---|---|
| Multi-step tool loops | 3/3 | 7/7 (longer chains, branching) |
| Error recovery | 2/2 | 4/5 (harder: cascading, contradictory tools) |
| Code agent | 2/3 | 9/10 (10 bug types + TDD + multi-file + refactor) |
| Planning + constraints | 2/2 | 6/6 (tighter budgets, trade-offs, mid-task change, impossible budget) |
| Long-horizon memory | 1/1 | 3/3 (8 distractors, two facts, mid-update) |
| Tool discipline | 2/2 | 4/4 (incl. 2 clarification cases, new) |
| Belief update | 1/1 | 3/3 (stale→corrected, unprompted disagreement, retraction) |
| Long-context agent | — | 3/3 (new axis) |
| **Total** | **13/14 corrected** | **39/41 corrected** |

## Verdict vs the probe

**STRONG as an agent, with two systematic loop-termination weaknesses.**
Single- and multi-turn tool use is now clean across 41 scenarios: no
wrong-tool or hallucinated-result failure anywhere, clarification asked
instead of guessed (2/2), stale data re-checked unprompted, conflicts
flagged rather than silently picked. The probe's debugger weakness is
largely a prompt/setup artifact — 9/10 code tasks pass, and the exact
probe bug passes with a "state the fix first" instruction. What remains
is termination: an open-ended optimization ("cheapest") can spiral through
33 searches without answering, and a fix that consumes the turn budget
leaves no turn for the final answer. Practical meaning: safe for
supervised gather → compute → report workflows and for code tasks with a
stated fix step; put a turn cap + a "report what you have" instruction on
open-ended searches, and a "state the fix before applying it" instruction
on debug tasks. Vendor agentic claims (Terminal-Bench, SWE-bench 79.2%,
etc.) remain **NOT VERIFIABLE** here — this is a 41-scenario probe, not
their benchmark.

## Deploy incidents (same session served all 3 chunks + rerun)

- Session creation failed on **three accounts in a row** (kyu009009,
  cyc236ha, ppoppo205): `colab.py new` timed out after 300–370 s each.
  The colab log shows the pattern: GET `/tun/m/assign` returns 200 with
  a T4 assignment, then POST `/tun/m/assign` hangs with no response
  (~60 s), the CLI checks assignments (empty) and retries with a fresh
  notebook hash — repeating until the wrapper timeout. The 411 patch
  (`data="{}"`) IS applied in the installed client, so this is not the
  known 411 issue; it looks like a Colab-side provisioning stall that
  evening (the agentic probe redeployed fine on kyu009009 ~1 h earlier).
  Fourth account `cyc236es` created the session in 134 s.
- `deploy.py` was bypassed for the create step (its 300 s `new` timeout
  is shorter than the wrapper's own retry budget): session created with
  `colab.py new` directly, then driver + catalog uploaded and the driver
  launched with `exec_detach` by hand; driver reached `ready:true` in
  ~5 min. Tunnel `https://pendant-quotes-enough-requesting.trycloudflare.com`
  (403 from VM egress, as before — all test traffic VM-side).
- The session survived all 3 chunks + rerun (~50 min of testing) with no
  reclamation. VRAM 14,107–14,221 MiB throughout; health ok at every
  chunk boundary.

## Caveats

- Temp 0 makes reruns deterministic: both genuine failures reproduced
  bit-identically (same turn/call counts), which is evidence they are
  systematic, not flaky.
- The code-axis improvement over the probe is partly the added `read_file`
  tool and partly the state-the-fix prompt — the probe's no-read-file
  setup was harsher.
- The `s4_constraint_change` chunk-2 result is a harness artifact, not a
  model result; the v2 re-test is the valid measurement.
- Tunnel was never verified from a residential network (403 from VM egress).
