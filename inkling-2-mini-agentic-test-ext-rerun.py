#!/usr/bin/env python3
"""Rerun driver for the extensive agentic test: re-tests the three scenarios
whose chunk results need a second look, plus a FIXED constraint-change design.

- s2_empty_then_cheapest: chunk 1 spiralled (33 searches, max turns) where the
  probe had passed — flaky or systematic?
- s3_traceback_only: chunk 2 fixed the file (rerun rc 0, out '20') but hit max
  turns without a final answer — repeats?
- s4_constraint_change_v2: chunk 1's inject-after-turn-2 design never fired —
  the model answered the $500 plan in 2 turns before the injection point.
  This v2 runs the plan to completion, THEN sends the budget cut as a new
  user turn and continues the conversation.

Usage: python3 rerun_ext.py [--out FILE]
"""
import json, re, sys, time
sys.path.insert(0, ".")
import agentic_test as B
import agentic_test_ext as E

OUT = "/content/agentic_results_ext_rerun.json"

def s4_constraint_change_v2():
    st = B.ToolState(); ex = E.make_ext_tools(st)
    tools = E.tools_for(["get_hotel_price", "get_attraction_price", "get_weather"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": (
                "Plan a 2-day trip to Tokyo with a total budget of $500. Check the hotel price "
                "for Tokyo for 2 nights and the ticket prices of teamLab, Skytree and Disney. "
                "Present the plan as exactly 2 sections headed 'Day 1' and 'Day 2', and end "
                "with a line 'TOTAL: $X' (hotel + attractions). The total must not exceed "
                "the budget.")}]
    final1, turns1, ncalls1, tr1, wall1 = B.run_agent(msgs, tools, ex, max_turns=10)
    E.transcripts["s4_constraint_change_v2_phase1"] = tr1
    total1 = E._plan_total(final1)
    msgs2 = msgs + [{"role": "assistant", "content": final1},
                    {"role": "user", "content": (
                        "Budget update: the budget is now $350, not $500. Please adjust the "
                        "plan so the new TOTAL is $350 or less, and say what you changed.")}]
    final2, turns2, ncalls2, tr2, wall2 = B.run_agent(msgs2, tools, ex, max_turns=10)
    E.transcripts["s4_constraint_change_v2"] = tr1 + tr2
    total2 = E._plan_total(final2)
    exp2 = 280 + sum(v for k, v in E.PRICES.items() if k in E.norm(final2))
    ok = (total1 is not None and total1 <= 500
          and total2 is not None and total2 <= 350 and abs(total2 - exp2) < 0.01)
    E.check("4", "s4_constraint_change_v2", ok,
            f"phase1_total={total1} phase2_total={total2} expected2={exp2} "
            f"turns={turns1}+{turns2} calls={ncalls1}+{ncalls2} wall={wall1+wall2:.0f}s | {final2[:220]}",
            {"turns": turns1 + turns2, "tool_calls": ncalls1 + ncalls2,
             "wall_s": round(wall1 + wall2, 1)})

def main():
    E.OUT_PATH = OUT
    E.T0 = time.time()
    print("Rerun: s2_empty_then_cheapest, s3_traceback_only, s4_constraint_change_v2",
          flush=True)
    for fn in (E.s2_empty_then_cheapest, E.s3_traceback_only, s4_constraint_change_v2):
        try:
            fn()
        except Exception as e:
            print(f"ERROR in rerun {fn.__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
    summary = {}
    for r in E.results:
        summary.setdefault(r["axis"], [0, 0])
        summary[r["axis"]][1] += 1
        if r["pass"]:
            summary[r["axis"]][0] += 1
    total_pass = sum(v[0] for v in summary.values())
    total_n = sum(v[1] for v in summary.values())
    print(f"\nRERUN SUMMARY: {total_pass}/{total_n} per-axis={json.dumps(summary)}", flush=True)
    json.dump({"results": E.results, "transcripts": E.transcripts,
               "summary": summary, "total_pass": total_pass, "total_n": total_n,
               "elapsed_s": round(time.time() - E.T0, 1)},
              open(OUT, "w"), indent=2)
    print(f"Wrote {OUT}", flush=True)

if __name__ == "__main__":
    main()
