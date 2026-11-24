#!/usr/bin/env python3
"""Extended agentic-capability test for Inkling 2 Mini on Colab T4 (b12669).

Second, much bigger battery after the 14-scenario probe (12/14 as-run,
13/14 corrected, MIXED). 41 scenarios in 3 chunks of 12-16, tiered
easy/medium/hard, temp 0, reasoning at server default. Runs VM-side against
http://127.0.0.1:11434/v1 (tunnel 403s from VM egress).

Reuses agentic_test.py's client, agent loop and deterministic tools
(imported as B); this file only adds new tools and new scenarios.
The harness mediates ALL tool execution; the model never gets raw shell.
Code tasks run in /content/agent_sandbox only.

Chunking: --chunk 1 (axes 1+2, 12 scenarios), --chunk 2 (axes 3+4, 16),
--chunk 3 (axes 5+6+7+8, 13). Each chunk writes its own results JSON
incrementally after every scenario (reclaim-safe) and has a 40-min deadline.

Usage: python3 agentic_test_ext.py --chunk N --out FILE
"""
import argparse, importlib.util, json, os, re, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agentic_test as B

BASE = B.BASE
MODEL = B.MODEL
SANDBOX = B.SANDBOX
T0 = time.time()
DEADLINE_S = 2400  # 40 min per chunk

results = []
transcripts = {}
OUT_PATH = "agentic_results_ext_chunk1.json"

def save_partial():
    try:
        json.dump({"results": results, "transcripts": transcripts,
                   "elapsed_s": round(time.time() - T0, 1)},
                  open(OUT_PATH, "w"), indent=2)
    except Exception:
        pass

def check(axis, name, passed, detail, extra=None):
    rec = {"axis": axis, "name": name, "pass": bool(passed),
           "detail": str(detail)[:600]}
    if extra:
        rec.update(extra)
    results.append(rec)
    print(f"{'PASS' if passed else 'FAIL'} [{axis}] {name}: {str(detail)[:220]}", flush=True)
    save_partial()
    return bool(passed)

def norm(s):
    return B.norm(s)

def digits(s):
    return B.digits(s)

def vram_mib():
    return B.vram_mib()

def write_file_called(transcript):
    return "write_file" in B.names_used(transcript)

# ---------------- extra tools (deterministic, harness-side) ----------------
def make_ext_tools(state):
    ex = B.make_tools(state)
    # --- stale TSLA quote (axis 7.1): first call delayed, second live ---
    base_stock = ex["get_stock_price"]
    stock_calls = {"TSLA": 0}
    def get_stock_price(args):
        tick = (args.get("ticker", "") or "").upper().strip()
        if getattr(state, "stock_mode", "") == "stale2" and tick == "TSLA":
            stock_calls["TSLA"] += 1
            if stock_calls["TSLA"] == 1:
                return ("TSLA: $240.00 (WARNING: delayed quote from 09:00 — the market "
                        "has moved since then; consider re-checking for the current price.)")
            return "TSLA: $248.10 (live quote as of 15:30)."
        return base_stock(args)
    ex["get_stock_price"] = get_stock_price

    def get_stock_price_live(args):
        # A second, independent live feed that disagrees with the fixed feed.
        tick = (args.get("ticker", "") or "").upper().strip()
        if tick == "AAPL":
            return json.dumps({"ticker": "AAPL", "price_usd": 148.90,
                               "source": "live-feed"})
        return "ERROR: unknown ticker '%s'." % tick

    def get_author_nationality(args):
        a = norm(args.get("author", ""))
        if "frank herbert" in a:
            return "Frank Herbert was American."
        if "george orwell" in a:
            return "George Orwell was British."
        return "ERROR: unknown author '%s'." % args.get("author", "")

    def get_product_price(args):
        name = norm(args.get("product", ""))
        data = {"widget": 250, "gadget": 99, "headphones": 149}
        for k, v in data.items():
            if k in name:
                return json.dumps({"product": k, "price_usd": v})
        return "ERROR: unknown product '%s'." % args.get("product", "")

    def get_inventory(args):
        name = norm(args.get("product", ""))
        data = {"widget": 120, "gadget": 0, "headphones": 45}
        for k, v in data.items():
            if k in name:
                return json.dumps({"product": k, "in_stock": v})
        return "ERROR: unknown product '%s'." % args.get("product", "")

    def read_file(args):
        p = args.get("path", "")
        if not p.startswith("/content/agent_sandbox/"):
            p = os.path.join(SANDBOX, os.path.basename(p))
        try:
            with open(p) as f:
                return f.read()[:4000]
        except Exception as e:
            return "ERROR reading file: %s" % e

    def get_document(args):
        doc_id = norm(args.get("doc_id", ""))
        filler = ("The history of computing spans decades of steady progress, from room-sized "
                  "machines to devices that fit in a pocket, each generation building on the "
                  "last with faster processors, larger memories, and new ways to connect. ")
        if doc_id == "discount":
            needle = ("The current promotion is discount code SAVE20, which gives 20% off "
                      "all widgets. ")
        elif doc_id == "warehouse":
            needle = ("The East region warehouse currently holds the largest stock of any "
                      "region. ")
        elif doc_id == "codes":
            needle = ("Use code WELCOME10 for 10% off (expired 2026-01-01). The current valid "
                      "code is SAVE20 for 20% off. ")
        else:
            return "ERROR: unknown document '%s'." % args.get("doc_id", "")
        reps = 380
        doc = (filler * reps)[:33000]
        mid = len(doc) // 2
        return doc[:mid] + " " + needle + " " + doc[mid:]

    # Hidden-test runner for TDD tasks: TESTS keyed by file basename.
    # Each case: (function_name, [args], expected). Returns per-case results
    # WITH expected values (feedback is the point of the task).
    TESTS = {
        "grade.py": [("grade", [95], "A"), ("grade", [85], "B"), ("grade", [75], "C"),
                     ("grade", [65], "D"), ("grade", [55], "F"), ("grade", [90], "A"),
                     ("grade", [80], "B"), ("grade", [70], "C"), ("grade", [60], "D"),
                     ("grade", [59], "F")],
    }
    def run_tests(args):
        p = args.get("path", "")
        if not p.startswith("/content/agent_sandbox/"):
            p = os.path.join(SANDBOX, os.path.basename(p))
        cases = TESTS.get(os.path.basename(p))
        if cases is None:
            return "ERROR: no hidden tests registered for %s." % os.path.basename(p)
        try:
            spec = importlib.util.spec_from_file_location("ut_" + os.path.basename(p)[:-3], p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            return "ERROR: module failed to load: %s" % e
        lines = []
        n_pass = 0
        for fn, fargs, exp in cases:
            try:
                got = getattr(mod, fn)(*fargs)
                ok = got == exp
            except Exception as e:
                got = "ERROR: %s" % e
                ok = False
            n_pass += ok
            lines.append("%s(%s): got %r, expected %r -> %s"
                         % (fn, ", ".join(map(str, fargs)), got, exp,
                            "PASS" if ok else "FAIL"))
        return "hidden tests: %d/%d passed\n%s" % (n_pass, len(cases), "\n".join(lines))

    ex.update({"get_stock_price_live": get_stock_price_live,
               "get_author_nationality": get_author_nationality,
               "get_product_price": get_product_price,
               "get_inventory": get_inventory,
               "read_file": read_file,
               "get_document": get_document,
               "run_tests": run_tests})
    return ex

_EXT_SCHEMAS = {
    "get_stock_price_live": ("Live stock price from a second feed", {"ticker": "string"}, ["ticker"]),
    "get_author_nationality": ("Nationality of an author", {"author": "string"}, ["author"]),
    "get_product_price": ("Price in USD of a product", {"product": "string"}, ["product"]),
    "get_inventory": ("Units in stock of a product", {"product": "string"}, ["product"]),
    "read_file": ("Read a file in /content/agent_sandbox", {"path": "string"}, ["path"]),
    "get_document": ("Fetch a document by id: discount, warehouse, codes", {"doc_id": "string"}, ["doc_id"]),
    "run_tests": ("Run hidden tests on a Python file, returns per-case pass/fail", {"path": "string"}, ["path"]),
}

def ext_tool_schema(name):
    if name in _EXT_SCHEMAS:
        desc, props, req = _EXT_SCHEMAS[name]
        return {"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object",
                           "properties": {k: {"type": "string"} for k in props},
                           "required": req}}}
    return B.tool_schema(name)

def tools_for(names):
    return [ext_tool_schema(n) for n in names]

def reset_sandbox(seed=None):
    """Clear the sandbox, optionally seeding {filename: content}."""
    os.makedirs(SANDBOX, exist_ok=True)
    for f in os.listdir(SANDBOX):
        try:
            os.remove(os.path.join(SANDBOX, f))
        except Exception:
            pass
    if seed:
        for name, content in seed.items():
            with open(os.path.join(SANDBOX, name), "w") as f:
                f.write(content)

def rerun_file(name):
    p = os.path.join(SANDBOX, name)
    if not os.path.exists(p):
        return -1, "file missing"
    r = subprocess.run([sys.executable, p], cwd=SANDBOX, capture_output=True,
                       text=True, timeout=60)
    return r.returncode, (r.stdout + r.stderr).strip()

# ---------------- Axis 1: multi-step tool loops (7) ----------------
def s1_long_chain():
    """5-turn chain: weather + time + currency + hotel, then total cost."""
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_weather", "get_time", "get_currency_rate",
                       "get_hotel_price", "get_random_joke"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": (
                "Plan a 2-night trip to Tokyo. Check the weather in Tokyo (celsius), the "
                "current time in Asia/Tokyo, the USD to JPY exchange rate, and the hotel "
                "price in Tokyo for 2 nights. Then tell me: the weather, the Tokyo time, "
                "how many yen $500 USD converts to, and the total trip cost in USD if the "
                "hotel is the only other expense (hotel + a $350 flight).")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=12)
    transcripts["s1_long_chain"] = tr
    used = B.names_used(tr)
    ok = ("18" in final and "sunny" in norm(final) and "14:30" in final
          and "77600" in digits(final) and "630" in digits(final)
          and all(n in used for n in ["get_weather", "get_time",
                                      "get_currency_rate", "get_hotel_price"]))
    check("1", "s1_long_chain", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s tools={used} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def s1_branching():
    """Branch on a tool result: sunny Tokyo -> teamlab; report hotel + attraction total."""
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_weather", "get_attraction_price", "get_hotel_price",
                       "get_random_joke"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": (
                "Check the weather in Tokyo. If it is sunny, plan to visit teamLab and check "
                "its ticket price; if it is rainy, plan to visit the Ueno museum instead and "
                "check its price. Also check the hotel price in Tokyo for 2 nights. Then "
                "report the weather, the attraction you chose and its price, and the total "
                "cost (hotel + attraction).")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=12)
    transcripts["s1_branching"] = tr
    used = B.names_used(tr)
    # Tokyo sunny -> teamlab $35; hotel 2 nights $280; total $315.
    ok = ("sunny" in norm(final) and "teamlab" in norm(final) and "35" in final
          and "315" in digits(final) and "get_attraction_price" in used
          and "get_hotel_price" in used and "ueno" not in norm(final))
    check("1", "s1_branching", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s tools={used} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def s1_parallel_then_seq():
    """Parallel sales fetch for 4 regions, then sequential currency conversion of the best."""
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_sales", "get_currency_rate", "get_weather"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": (
                "Get the sales figures for the North, South, East and West regions. Find the "
                "best region and the total sales across all four. Then check the USD to JPY "
                "exchange rate and tell me the best region's sales converted to yen.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=12)
    transcripts["s1_parallel_then_seq"] = tr
    used = B.names_used(tr)
    # total 49000; best East 15600; 15600 * 155.20 = 2,421,120 yen.
    ok = ("east" in norm(final) and "49000" in digits(final)
          and "2421120" in digits(final) and used.count("get_sales") >= 4
          and "get_currency_rate" in used)
    check("1", "s1_parallel_then_seq", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s sales_calls={used.count('get_sales')} tools={used} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def s1_chained_args():
    """Tool A's result feeds tool B's arguments: stock price -> tax -> yen."""
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_sales", "get_currency_rate", "get_weather"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": (
                "Get the sales figure for the East region. A 10% tax applies to those sales. "
                "Compute the tax amount in USD, then check the USD to JPY exchange rate and "
                "tell me the tax amount in yen.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=12)
    transcripts["s1_chained_args"] = tr
    used = B.names_used(tr)
    # East 15600; tax 1560 USD; 1560 * 155.20 = 242,112 yen.
    ok = ("1560" in digits(final) and "242112" in digits(final)
          and "get_sales" in used and "get_currency_rate" in used)
    check("1", "s1_chained_args", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s tools={used} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def s1_six_turn_trip():
    """6-turn chain: weather HK + Tokyo, time Tokyo, flight, hotel, total."""
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_weather", "get_time", "search_flights",
                       "get_hotel_price", "get_random_joke"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": (
                "Plan my Hong Kong to Tokyo trip. Check the weather in Hong Kong and in Tokyo, "
                "the time in Asia/Tokyo, search for a flight from Hong Kong to Tokyo on "
                "2026-12-26, and check the hotel price in Tokyo for 2 nights. Then give me a "
                "summary with both weather reports, the Tokyo time, the flight number and "
                "price, the hotel total, and the grand total (flight + hotel).")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=12)
    transcripts["s1_six_turn_trip"] = tr
    used = B.names_used(tr)
    # CX520 $320 + hotel 2 nights $280 = $600.
    ok = ("24" in final and "18" in final and "14:30" in final and "CX520" in final
          and "320" in final and "280" in final and "600" in digits(final)
          and used.count("get_weather") >= 2 and "search_flights" in used
          and "get_hotel_price" in used)
    check("1", "s1_six_turn_trip", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s tools={used} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def s1_three_hop():
    """3-hop: book -> author -> birth year -> nationality."""
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_author", "get_author_birth_year",
                       "get_author_nationality", "get_weather"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": (
                "Who wrote the book 1984, in what year was that author born, and what was "
                "their nationality? Look each fact up with the tools.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=12)
    transcripts["s1_three_hop"] = tr
    used = B.names_used(tr)
    ok = ("orwell" in norm(final) and "1903" in final and "british" in norm(final)
          and all(n in used for n in ["get_author", "get_author_birth_year",
                                      "get_author_nationality"]))
    check("1", "s1_three_hop", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s tools={used} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def s1_warmest_city():
    """Parallel weather for 3 cities, pick the warmest, report it."""
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_weather", "get_time"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": (
                "Check the weather in Hong Kong, Tokyo and London (celsius). Tell me which "
                "city is warmest and what its temperature is.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=12)
    transcripts["s1_warmest_city"] = tr
    used = B.names_used(tr)
    ok = ("hong kong" in norm(final) and "24" in final
          and used.count("get_weather") >= 3)
    check("1", "s1_warmest_city", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s weather_calls={used.count('get_weather')} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

# ---------------- Axis 2: error recovery (5) ----------------
def s2_cascading_errors():
    """Two unknown-city errors in a row, then success."""
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_weather", "get_time"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": (
                "What is the weather in Atlantis? If the tool says the city is unknown, try "
                "Paris. If that also fails, try Hong Kong. Report the weather you finally get.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10)
    transcripts["s2_cascading_errors"] = tr
    wx = [t for t in tr if t.get("role") == "tool" and t.get("name") == "get_weather"]
    ok = (len(wx) >= 3 and "24" in final and "sunny" in norm(final)
          and "atlantis" not in norm(final).replace("atlantis isn't", ""))
    check("2", "s2_cascading_errors", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s weather_calls={len(wx)} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def s2_wrong_format_arg():
    """nights='three' errors; model must retry with a number."""
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_hotel_price", "get_weather"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": (
                "How much is a hotel in Tokyo for three nights? (Hint: the tool needs nights "
                "as a number.) Report the total price.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10)
    transcripts["s2_wrong_format_arg"] = tr
    hp = [t for t in tr if t.get("role") == "tool" and t.get("name") == "get_hotel_price"]
    ok = (len(hp) >= 1 and "420" in final and "ERROR" not in final)
    check("2", "s2_wrong_format_arg", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s hotel_calls={len(hp)} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def s2_empty_then_cheapest():
    """Empty Christmas Day, must try other dates and report the cheapest found."""
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["search_flights", "get_weather"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": (
                "Find me the cheapest flight from Hong Kong to Tokyo around Christmas. Start "
                "with 2026-12-25; if there are no flights that day, try nearby dates. Report "
                "the cheapest flight you actually find — do not invent one.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10)
    transcripts["s2_empty_then_cheapest"] = tr
    sfs = [t for t in tr if t.get("role") == "tool" and t.get("name") == "search_flights"]
    ok = len(sfs) >= 2 and ("CX520" in final and "320" in final)
    check("2", "s2_empty_then_cheapest", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s flight_searches={len(sfs)} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def s2_contradictory_tools():
    """Two stock feeds disagree (150.25 vs 148.90): must flag the conflict."""
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_stock_price", "get_stock_price_live", "get_weather"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": (
                "What is the price of AAPL? Check both stock price tools and tell me what "
                "each one says.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10)
    transcripts["s2_contradictory_tools"] = tr
    used = B.names_used(tr)
    ok = ("150.25" in final and "148.9" in final
          and any(w in norm(final) for w in ["differ", "disagree", "conflict",
                                             "not the same", "two different", "mismatch"])
          and "get_stock_price" in used and "get_stock_price_live" in used)
    check("2", "s2_contradictory_tools", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s tools={used} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def s2_unknown_book():
    """Unknown book errors; must say unknown, not invent an author."""
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_author", "get_weather"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": "Who wrote the book 'The Quantum Thief's Apprentice'?"}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10)
    transcripts["s2_unknown_book"] = tr
    ok = (any(w in norm(final) for w in ["unknown", "don't know", "do not know",
                                         "cannot find", "no record", "not found"])
          and "get_author" in B.names_used(tr))
    check("2", "s2_unknown_book", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

# ---------------- Axis 3: code agent (10) — the weak axis, probed hardest ----------------
CODE_TOOLS = ["write_file", "run_script", "list_files"]
CODE_TOOLS_READ = ["write_file", "run_script", "list_files", "read_file"]

def _code_result(name, final, turns, ncalls, tr, wall, ok, detail_extra=""):
    transcripts[name] = tr
    check("3", name, ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s write_file_called={write_file_called(tr)} "
          f"{detail_extra} | {final[:200]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1),
           "write_file_called": write_file_called(tr)})

def s3_fix_state_first():
    """SAME count='3' bug as the probe, but instructed to state the fix in words first.
    Exact probe setup (no read_file) to isolate the prompt effect."""
    reset_sandbox({"buggy.py": "nums = [10, 20, 30]\ntotal = sum(nums)\ncount = \"3\"\n"
                               "print(total / count)\n"})
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(CODE_TOOLS)
    sys_code = (B.SYS_CODE + " Before you change anything, state in words what the bug "
                "is and what you will change. Then fix the file and run it.")
    msgs = [{"role": "system", "content": sys_code},
            {"role": "user", "content": (
                "The file /content/agent_sandbox/buggy.py crashes when run. First tell me in "
                "words what the bug is and what you will change. Then fix the bug, run the "
                "script again, and report the printed result.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10,
                                                 max_tokens=4096)
    rc, out = rerun_file("buggy.py")
    stated = any(w in norm(final) for w in ["string", "str ", "count is"])
    ok = (rc == 0 and "20" in out and "20" in final)
    _code_result("s3_fix_state_first", final, turns, ncalls, tr, wall, ok,
                 f"rerun_rc={rc} out={out[:120]!r} stated_fix_in_words={stated}")

def s3_off_by_one():
    reset_sandbox({"sum_range.py": "total = sum(range(1, 10))\nprint(total)\n"})
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(CODE_TOOLS_READ)
    msgs = [{"role": "system", "content": B.SYS_CODE},
            {"role": "user", "content": (
                "The file /content/agent_sandbox/sum_range.py is supposed to print the sum of "
                "the integers from 1 to 10, but it prints the wrong number. Read the file, fix "
                "the bug, run it, and report the correct result.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10,
                                                 max_tokens=4096)
    rc, out = rerun_file("sum_range.py")
    ok = (rc == 0 and out.strip() == "55")
    _code_result("s3_off_by_one", final, turns, ncalls, tr, wall, ok,
                 f"rerun_rc={rc} out={out[:120]!r}")

def s3_wrong_key():
    reset_sandbox({"price.py": "data = {\"price\": 42}\nprint(data[\"prce\"])\n"})
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(CODE_TOOLS_READ)
    msgs = [{"role": "system", "content": B.SYS_CODE},
            {"role": "user", "content": (
                "The file /content/agent_sandbox/price.py crashes with a KeyError. Read the "
                "file, fix the bug, run it, and report the printed value.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10,
                                                 max_tokens=4096)
    rc, out = rerun_file("price.py")
    ok = (rc == 0 and out.strip() == "42")
    _code_result("s3_wrong_key", final, turns, ncalls, tr, wall, ok,
                 f"rerun_rc={rc} out={out[:120]!r}")

def s3_missing_import():
    reset_sandbox({"dump.py": "print(json.dumps({\"a\": 1}))\n"})
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(CODE_TOOLS_READ)
    msgs = [{"role": "system", "content": B.SYS_CODE},
            {"role": "user", "content": (
                "The file /content/agent_sandbox/dump.py crashes with a NameError. Read the "
                "file, fix the bug, run it, and report the printed output.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10,
                                                 max_tokens=4096)
    rc, out = rerun_file("dump.py")
    ok = (rc == 0 and '"a": 1' in out)
    _code_result("s3_missing_import", final, turns, ncalls, tr, wall, ok,
                 f"rerun_rc={rc} out={out[:120]!r}")

def s3_logic_error():
    reset_sandbox({"hottest.py": "temps = [21, 25, 19, 23]\nprint(min(temps))\n"})
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(CODE_TOOLS_READ)
    msgs = [{"role": "system", "content": B.SYS_CODE},
            {"role": "user", "content": (
                "The file /content/agent_sandbox/hottest.py is supposed to print the highest "
                "temperature in the list, but it prints the wrong value (it does not crash). "
                "Read the file, fix the logic error, run it, and report the correct value.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10,
                                                 max_tokens=4096)
    rc, out = rerun_file("hottest.py")
    ok = (rc == 0 and out.strip() == "25" and "25" in final)
    _code_result("s3_logic_error", final, turns, ncalls, tr, wall, ok,
                 f"rerun_rc={rc} out={out[:120]!r}")

def s3_multi_file():
    reset_sandbox()
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(CODE_TOOLS_READ)
    msgs = [{"role": "system", "content": B.SYS_CODE},
            {"role": "user", "content": (
                "Write a helper module /content/agent_sandbox/helpers.py defining a function "
                "double(n) that returns n * 2. Then write /content/agent_sandbox/main.py "
                "that imports double from helpers and prints double(21). Run main.py and "
                "report the printed result.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10,
                                                 max_tokens=4096)
    rc, out = rerun_file("main.py")
    helpers_ok = os.path.exists(os.path.join(SANDBOX, "helpers.py"))
    ok = (rc == 0 and out.strip() == "42" and helpers_ok)
    _code_result("s3_multi_file", final, turns, ncalls, tr, wall, ok,
                 f"rerun_rc={rc} out={out[:120]!r} helpers_exists={helpers_ok}")

def s3_tdd_grade():
    reset_sandbox()
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["write_file", "run_script", "run_tests", "list_files"])
    sys_code = (B.SYS_CODE + " You also have run_tests(path), which runs hidden tests on a "
                "file and reports per-case pass/fail — use it to check your work and fix "
                "any failures it reports.")
    msgs = [{"role": "system", "content": sys_code},
            {"role": "user", "content": (
                "Write /content/agent_sandbox/grade.py defining a function grade(score) that "
                "returns 'A' for scores >= 90, 'B' for >= 80, 'C' for >= 70, 'D' for >= 60, "
                "and 'F' otherwise. Use run_tests to check it against the hidden tests, fix "
                "any failures, and report the final test result.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10,
                                                 max_tokens=4096)
    # Verify directly with the same cases run_tests uses.
    cases = [(95, "A"), (85, "B"), (75, "C"), (65, "D"), (55, "F"), (90, "A"),
             (80, "B"), (70, "C"), (60, "D"), (59, "F")]
    passed = 0; detail = "file missing"
    p = os.path.join(SANDBOX, "grade.py")
    if os.path.exists(p):
        try:
            spec = importlib.util.spec_from_file_location("grade_mod", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            passed = sum(1 for n, e in cases if mod.grade(n) == e)
            detail = f"hidden {passed}/{len(cases)}"
        except Exception as e:
            detail = f"import/call failed: {e}"
    ok = passed == len(cases)
    _code_result("s3_tdd_grade", final, turns, ncalls, tr, wall, ok,
                 f"{detail} run_tests_used={'run_tests' in B.names_used(tr)}")

def s3_extend_file():
    reset_sandbox({"numbers.py": "nums = [4, 8, 15, 16, 23, 42]\n\n\n"
                                "def mean(xs):\n    return sum(xs) / len(xs)\n\n\n"
                                "print(mean(nums))\n"})
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(CODE_TOOLS_READ)
    msgs = [{"role": "system", "content": B.SYS_CODE},
            {"role": "user", "content": (
                "The file /content/agent_sandbox/numbers.py already defines mean() and prints "
                "the mean of nums. Read it, add a median(xs) function to the same file, and "
                "make the script also print the median. Run it and report both values.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10,
                                                 max_tokens=4096)
    rc, out = rerun_file("numbers.py")
    with open(os.path.join(SANDBOX, "numbers.py")) as f:
        src = f.read()
    ok = (rc == 0 and "15.5" in out and "def median" in src and "15.5" in final)
    _code_result("s3_extend_file", final, turns, ncalls, tr, wall, ok,
                 f"rerun_rc={rc} out={out[:150]!r} has_median_def={'def median' in src}")

def s3_refactor():
    reset_sandbox({"tax.py": "price = 100\nprint(price * 1.2)\n"})
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(CODE_TOOLS_READ)
    msgs = [{"role": "system", "content": B.SYS_CODE},
            {"role": "user", "content": (
                "Refactor /content/agent_sandbox/tax.py so the 0.2 tax rate is a named "
                "constant TAX_RATE instead of a magic number. The printed output must stay "
                "exactly the same. Run it to confirm and report the output.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10,
                                                 max_tokens=4096)
    rc, out = rerun_file("tax.py")
    with open(os.path.join(SANDBOX, "tax.py")) as f:
        src = f.read()
    ok = (rc == 0 and out.strip() in ("120.0", "120") and "TAX_RATE" in src)
    _code_result("s3_refactor", final, turns, ncalls, tr, wall, ok,
                 f"rerun_rc={rc} out={out[:120]!r} has_TAX_RATE={'TAX_RATE' in src}")

def s3_traceback_only():
    """Debug from the traceback alone (no read_file): wrong arg count."""
    reset_sandbox({"mystery.py": "def area(w, h):\n    return w * h\n\n\n"
                                "width = 5\nheight = 4\nprint(area(width))\n"})
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(CODE_TOOLS)  # no read_file: traceback only
    msgs = [{"role": "system", "content": B.SYS_CODE},
            {"role": "user", "content": (
                "The file /content/agent_sandbox/mystery.py crashes when run. Use run_script "
                "to see the error, fix the bug, run it again, and report the printed result.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10,
                                                 max_tokens=4096)
    rc, out = rerun_file("mystery.py")
    ok = (rc == 0 and out.strip() == "20" and "20" in final)
    _code_result("s3_traceback_only", final, turns, ncalls, tr, wall, ok,
                 f"rerun_rc={rc} out={out[:120]!r}")

# ---------------- Axis 4: planning + constraints (6) ----------------
PRICES = {"teamlab": 35, "skytree": 28, "ueno": 0, "meiji": 0, "disney": 75}

def _plan_total(final):
    m = re.search(r"total\D*\$?\s*(\d[\d,]*(?:\.\d+)?)", norm(final))
    return float(m.group(1).replace(",", "")) if m else None

def s4_tight_budget():
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_hotel_price", "get_attraction_price", "get_weather"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": (
                "Plan a 2-day trip to Tokyo with a total budget of $300. Check the hotel price "
                "for Tokyo for 2 nights and the ticket prices of the attractions you want to "
                "visit. Present the plan as exactly 2 sections headed 'Day 1' and 'Day 2', "
                "each with one attraction and its cost, and end with a line 'TOTAL: $X' where "
                "X is the sum of the hotel and the two attraction prices. The total must not "
                "exceed $300.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10)
    transcripts["s4_tight_budget"] = tr
    total = _plan_total(final)
    exp = 280 + sum(v for k, v in PRICES.items() if k in norm(final))
    ok = (total is not None and total <= 300 and abs(total - exp) < 0.01
          and "get_hotel_price" in B.names_used(tr)
          and "get_attraction_price" in B.names_used(tr))
    check("4", "s4_tight_budget", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s total={total} expected={exp} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def s4_tradeoff():
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_hotel_price", "get_attraction_price", "get_weather"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": (
                "Plan a 3-day trip to Tokyo with a total budget of $500. Check the hotel price "
                "for Tokyo for 3 nights. You want to visit as many attractions as possible "
                "without exceeding the budget — check the ticket prices of the attractions "
                "you are considering. Present the plan as exactly 3 sections headed 'Day 1', "
                "'Day 2', 'Day 3', each with one attraction and its cost, and end with a line "
                "'TOTAL: $X' (hotel + three attractions). The total must not exceed $500.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10)
    transcripts["s4_tradeoff"] = tr
    total = _plan_total(final)
    exp = 420 + sum(v for k, v in PRICES.items() if k in norm(final))
    n_attr = sum(1 for k in PRICES if k in norm(final))
    ok = (total is not None and total <= 500 and abs(total - exp) < 0.01
          and n_attr >= 2 and "get_hotel_price" in B.names_used(tr))
    check("4", "s4_tradeoff", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s total={total} expected={exp} attractions={n_attr} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def s4_multi_constraint():
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_hotel_price", "get_attraction_price", "get_time",
                       "get_weather"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": (
                "Plan a 3-day trip to Tokyo with a total budget of $500. Check the hotel price "
                "for 3 nights, the ticket prices of three attractions, and the current time in "
                "Asia/Tokyo. Present the plan as exactly 3 sections headed 'Day 1', 'Day 2', "
                "'Day 3', each with one attraction and its cost, include the Tokyo time "
                "somewhere in the plan, and end with a line 'TOTAL: $X' (hotel + attractions). "
                "The total must not exceed $500.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10)
    transcripts["s4_multi_constraint"] = tr
    total = _plan_total(final)
    exp = 420 + sum(v for k, v in PRICES.items() if k in norm(final))
    days = len(re.findall(r"day\s*[123]", norm(final)))
    ok = (days == 3 and "14:30" in final and total is not None and total <= 500
          and abs(total - exp) < 0.01 and "get_time" in B.names_used(tr))
    check("4", "s4_multi_constraint", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s days={days} total={total} expected={exp} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def run_agent_inject(messages, tools, executor, inject_at_turn, inject_msg,
                     max_turns=10, max_tokens=2048):
    """Agent loop with a mid-task user message injected after inject_at_turn
    assistant turns."""
    msgs = list(messages)
    transcript = []
    n_calls = 0
    t0 = time.time()
    injected = False
    for turn in range(max_turns):
        msg, usage, dt, _ = B.chat(msgs, tools=tools, max_tokens=max_tokens, timeout=600)
        calls = B.tcalls(msg)
        transcript.append({"turn": turn + 1, "role": "assistant",
                           "content": (B.content(msg) or "")[:800],
                           "tool_calls": [(c["name"], c["args"]) for c in calls],
                           "secs": round(dt, 1)})
        if not calls:
            return (B.content(msg) or ""), turn + 1, n_calls, transcript, time.time() - t0
        n_calls += len(calls)
        msgs.append({"role": "assistant", "content": B.content(msg) or "",
                     "tool_calls": msg.get("tool_calls")})
        for c in calls:
            fn = executor.get(c["name"])
            res = fn(c["args"]) if fn else "ERROR: unknown tool '%s'." % c["name"]
            transcript.append({"turn": turn + 1, "role": "tool", "name": c["name"],
                               "content": str(res)[:600]})
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": str(res)})
        if not injected and (turn + 1) >= inject_at_turn:
            msgs.append({"role": "user", "content": inject_msg})
            transcript.append({"turn": turn + 1, "role": "user",
                               "content": inject_msg[:300]})
            injected = True
    return "[MAX TURNS REACHED]", max_turns, n_calls, transcript, time.time() - t0

def s4_constraint_change():
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_hotel_price", "get_attraction_price", "get_weather"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": (
                "Plan a 2-day trip to Tokyo with a total budget of $500. Check the hotel price "
                "for Tokyo for 2 nights and the ticket prices of teamLab, Skytree and Disney. "
                "Present the plan as exactly 2 sections headed 'Day 1' and 'Day 2', and end "
                "with a line 'TOTAL: $X' (hotel + attractions). The total must not exceed "
                "the budget.")}]
    final, turns, ncalls, tr, wall = run_agent_inject(
        msgs, tools, ex, inject_at_turn=2,
        inject_msg=("Budget update: the budget is now $350, not $500. Please adjust the plan "
                    "so the new TOTAL is $350 or less, and say what you changed."),
        max_turns=10)
    transcripts["s4_constraint_change"] = tr
    total = _plan_total(final)
    exp = 280 + sum(v for k, v in PRICES.items() if k in norm(final))
    ok = (total is not None and total <= 350 and abs(total - exp) < 0.01)
    check("4", "s4_constraint_change", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s total={total} expected={exp} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def s4_impossible_budget():
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_hotel_price", "get_attraction_price", "get_weather"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": (
                "Plan a 3-day trip to Tokyo with a total budget of $300. Check the hotel price "
                "for Tokyo for 3 nights first. Then present the plan with a 'TOTAL: $X' line. "
                "The total must not exceed $300.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10)
    transcripts["s4_impossible_budget"] = tr
    total = _plan_total(final)
    imposs = any(w in norm(final) for w in ["impossible", "cannot", "not possible",
                                            "exceed", "over budget", "not enough",
                                            "insufficient"])
    ok = (imposs and "420" in final
          and (total is None or total > 300)
          and "get_hotel_price" in B.names_used(tr))
    check("4", "s4_impossible_budget", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s total={total} flagged_impossible={imposs} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def s4_cheapest_first():
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["search_flights", "get_hotel_price", "get_weather"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": (
                "Find the cheapest flight from Hong Kong to Tokyo around Christmas (start with "
                "2026-12-25, try other dates if needed). Then check the hotel price in Tokyo "
                "for 2 nights. Report the cheapest flight (date, flight number, price), the "
                "hotel total, and the grand total (flight + hotel).")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10)
    transcripts["s4_cheapest_first"] = tr
    sfs = [t for t in tr if t.get("role") == "tool" and t.get("name") == "search_flights"]
    ok = (len(sfs) >= 2 and "CX520" in final and "320" in final and "280" in final
          and "600" in digits(final))
    check("4", "s4_cheapest_first", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s flight_searches={len(sfs)} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

# ---------------- Axis 5: long-horizon memory (3) ----------------
def _memory_run(facts_user, distractors, question, expect_present, expect_absent=()):
    msgs = [{"role": "system", "content": "You are a helpful assistant. Remember facts the user tells you."},
            {"role": "user", "content": facts_user}]
    m, _, _, _ = B.chat(msgs, max_tokens=512)
    msgs.append({"role": "assistant", "content": B.content(m)})
    for q in distractors:
        msgs.append({"role": "user", "content": q})
        m, _, _, _ = B.chat(msgs, max_tokens=512)
        msgs.append({"role": "assistant", "content": B.content(m)})
    msgs.append({"role": "user", "content": question})
    t0 = time.time()
    m, _, _, _ = B.chat(msgs, max_tokens=512)
    wall = time.time() - t0
    final = B.content(m) or ""
    return final, wall, msgs

def s5_long_distractors():
    distractors = ["What is the weather like in general in spring?",
                   "What is 17 + 25?",
                   "Tell me a one-sentence joke about cats.",
                   "What is the capital of France?",
                   "What is 9 x 6?",
                   "Name one planet in our solar system.",
                   "What is the capital of Japan?",
                   "Tell me a one-sentence joke about dogs."]
    final, wall, _ = _memory_run(
        "My project code name is BLUE-FALCON and the launch date is 2027-03-15. Please remember this.",
        distractors, "What is my project code name and launch date?",
        ["BLUE-FALCON", "2027-03-15"])
    transcripts["s5_long_distractors"] = [{"note": final[:400]}]
    ok = "BLUE-FALCON" in final and "2027-03-15" in final
    check("5", "s5_long_distractors", ok,
          f"distractor_turns=8 wall={wall:.0f}s | {final[:220]}",
          {"turns": 10, "tool_calls": 0, "wall_s": round(wall, 1)})

def s5_two_facts():
    distractors = ["What is the capital of France?",
                   "What is 12 x 12?",
                   "Tell me a one-sentence joke about cats.",
                   "What is the largest ocean on Earth?",
                   "What is 100 - 37?",
                   "Name one primary color."]
    final, wall, _ = _memory_run(
        "My project code name is BLUE-FALCON, the launch date is 2027-03-15, and the "
        "project budget is $12,000. Please remember all three facts.",
        distractors,
        "What is my project code name, launch date, and budget?",
        ["BLUE-FALCON", "2027-03-15", "12,000"])
    transcripts["s5_two_facts"] = [{"note": final[:400]}]
    ok = ("BLUE-FALCON" in final and "2027-03-15" in final
          and ("12,000" in final or "12000" in final))
    check("5", "s5_two_facts", ok,
          f"distractor_turns=6 wall={wall:.0f}s | {final[:220]}",
          {"turns": 8, "tool_calls": 0, "wall_s": round(wall, 1)})

def s5_fact_updated():
    msgs = [{"role": "system", "content": "You are a helpful assistant. Remember facts the user tells you."},
            {"role": "user", "content": "My project launch date is 2027-03-15. Please remember this."}]
    m, _, _, _ = B.chat(msgs, max_tokens=512)
    msgs.append({"role": "assistant", "content": B.content(m)})
    for q in ["What is the capital of France?", "What is 8 x 7?",
              "Tell me a one-sentence joke about cats."]:
        msgs.append({"role": "user", "content": q})
        m, _, _, _ = B.chat(msgs, max_tokens=512)
        msgs.append({"role": "assistant", "content": B.content(m)})
    msgs.append({"role": "user", "content": "Update: the launch date has moved to 2027-06-01. Please remember the new date."})
    m, _, _, _ = B.chat(msgs, max_tokens=512)
    msgs.append({"role": "assistant", "content": B.content(m)})
    for q in ["What is 15 + 27?", "Name one planet in our solar system.",
              "What is the capital of Japan?"]:
        msgs.append({"role": "user", "content": q})
        m, _, _, _ = B.chat(msgs, max_tokens=512)
        msgs.append({"role": "assistant", "content": B.content(m)})
    msgs.append({"role": "user", "content": "What is my project launch date?"})
    t0 = time.time()
    m, _, _, _ = B.chat(msgs, max_tokens=512)
    wall = time.time() - t0
    final = B.content(m) or ""
    transcripts["s5_fact_updated"] = [{"note": final[:400]}]
    ok = "2027-06-01" in final and "2027-03-15" not in final
    check("5", "s5_fact_updated", ok,
          f"wall={wall:.0f}s uses_new={('2027-06-01' in final)} old_date_present={('2027-03-15' in final)} | {final[:220]}",
          {"turns": 9, "tool_calls": 0, "wall_s": round(wall, 1)})

# ---------------- Axis 6: tool discipline (4) ----------------
def s6_many_distractors():
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_stock_price", "get_weather", "get_time", "get_random_joke",
                       "translate", "get_sales", "get_product_price", "get_inventory"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": "What is the stock price of AAPL?"}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=6)
    transcripts["s6_many_distractors"] = tr
    used = B.names_used(tr)
    ok = used == ["get_stock_price"] and "150.25" in final
    check("6", "s6_many_distractors", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s tools={used} | {final[:150]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def s6_selective_tool():
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_time", "get_weather", "get_random_joke", "translate"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": "Is now a good time to call Tokyo?"}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=6)
    transcripts["s6_selective_tool"] = tr
    used = B.names_used(tr)
    ok = "get_time" in used and "get_weather" not in used and "14:30" in final
    check("6", "s6_selective_tool", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s tools={used} | {final[:200]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def s6_clarify_flight():
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["search_flights", "get_weather", "get_time"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": "Book a flight for me."}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=6,
                                                 max_tokens=512)
    transcripts["s6_clarify_flight"] = tr
    used = B.names_used(tr)
    ok = (ncalls == 0 and "?" in final
          and any(w in norm(final) for w in ["origin", "destination", "where",
                                             "when", "date", "which flight"]))
    check("6", "s6_clarify_flight", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s tools={used} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def s6_clarify_ticker():
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_stock_price", "get_weather", "get_time"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": "What is the stock price?"}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=6,
                                                 max_tokens=512)
    transcripts["s6_clarify_ticker"] = tr
    used = B.names_used(tr)
    ok = (ncalls == 0 and "?" in final
          and any(w in norm(final) for w in ["ticker", "which stock", "symbol",
                                             "which company", "which share"]))
    check("6", "s6_clarify_ticker", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s tools={used} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

# ---------------- Axis 7: belief update (3) ----------------
def s7_stale_tsla():
    st = B.ToolState(); st.stock_mode = "stale2"; ex = make_ext_tools(st)
    tools = tools_for(["get_stock_price", "get_weather"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": "What is the current price of TSLA?"}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=8)
    transcripts["s7_stale_tsla"] = tr
    n_stock = B.names_used(tr).count("get_stock_price")
    ok = n_stock >= 2 and "248.10" in final
    check("7", "s7_stale_tsla", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s stock_calls={n_stock} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def s7_disagree_unprompted():
    """Two feeds disagree, but this time the user does NOT ask to check both."""
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_stock_price", "get_stock_price_live", "get_weather"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": "What is the price of AAPL?"}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=8)
    transcripts["s7_disagree_unprompted"] = tr
    used = B.names_used(tr)
    ok = ("150.25" in final and "148.9" in final
          and any(w in norm(final) for w in ["differ", "disagree", "conflict",
                                             "not the same", "two different", "mismatch"])
          and "get_stock_price" in used and "get_stock_price_live" in used)
    check("7", "s7_disagree_unprompted", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s tools={used} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def s7_retraction():
    msgs = [{"role": "system", "content": "You are a helpful assistant. Remember facts the user tells you."},
            {"role": "user", "content": "The team meeting is on Friday."}]
    m, _, _, _ = B.chat(msgs, max_tokens=512)
    msgs.append({"role": "assistant", "content": B.content(m)})
    msgs.append({"role": "user", "content": "Correction — ignore Friday. The meeting is on Thursday."})
    m, _, _, _ = B.chat(msgs, max_tokens=512)
    msgs.append({"role": "assistant", "content": B.content(m)})
    msgs.append({"role": "user", "content": "When is the team meeting?"})
    t0 = time.time()
    m, _, _, _ = B.chat(msgs, max_tokens=512)
    wall = time.time() - t0
    final = B.content(m) or ""
    transcripts["s7_retraction"] = [{"note": final[:400]}]
    ok = "thursday" in norm(final) and "friday" not in norm(final)
    check("7", "s7_retraction", ok,
          f"wall={wall:.0f}s | {final[:220]}",
          {"turns": 3, "tool_calls": 0, "wall_s": round(wall, 1)})

# ---------------- Axis 8: long-context agent (3) ----------------
def s8_doc_discount():
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_document", "get_product_price", "get_weather"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": (
                "Read the promotions document (doc_id 'discount'), find the current discount "
                "code, then check the price of a widget and tell me the final price after "
                "applying the discount.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10)
    transcripts["s8_doc_discount"] = tr
    used = B.names_used(tr)
    ok = ("SAVE20" in final and "200" in digits(final)
          and "get_document" in used and "get_product_price" in used)
    check("8", "s8_doc_discount", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s tools={used} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def s8_doc_warehouse():
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_document", "get_sales", "get_weather"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": (
                "Read the warehouse document (doc_id 'warehouse') to find which region holds "
                "the largest stock, then get that region's sales figure and report it.")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10)
    transcripts["s8_doc_warehouse"] = tr
    used = B.names_used(tr)
    ok = ("east" in norm(final) and "15600" in digits(final)
          and "get_document" in used and "get_sales" in used)
    check("8", "s8_doc_warehouse", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s tools={used} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def s8_doc_conflicting_codes():
    st = B.ToolState(); ex = make_ext_tools(st)
    tools = tools_for(["get_document", "get_product_price", "get_weather"])
    msgs = [{"role": "system", "content": B.SYS_TOOL},
            {"role": "user", "content": (
                "Read the promotions document (doc_id 'codes'). There is more than one code "
                "in it. Which discount code should I use for a widget, and what is the final "
                "price after the discount?")}]
    final, turns, ncalls, tr, wall = B.run_agent(msgs, tools, ex, max_turns=10)
    transcripts["s8_doc_conflicting_codes"] = tr
    used = B.names_used(tr)
    ok = ("SAVE20" in final and "200" in digits(final)
          and ("expired" in norm(final) or "WELCOME10" in final)
          and "get_document" in used and "get_product_price" in used)
    check("8", "s8_doc_conflicting_codes", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s tools={used} | {final[:220]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

# ---------------- chunk registry + main ----------------
CHUNKS = {
    1: [("1", "s1_long_chain", s1_long_chain),
        ("1", "s1_branching", s1_branching),
        ("1", "s1_parallel_then_seq", s1_parallel_then_seq),
        ("1", "s1_chained_args", s1_chained_args),
        ("1", "s1_six_turn_trip", s1_six_turn_trip),
        ("1", "s1_three_hop", s1_three_hop),
        ("1", "s1_warmest_city", s1_warmest_city),
        ("2", "s2_cascading_errors", s2_cascading_errors),
        ("2", "s2_wrong_format_arg", s2_wrong_format_arg),
        ("2", "s2_empty_then_cheapest", s2_empty_then_cheapest),
        ("2", "s2_contradictory_tools", s2_contradictory_tools),
        ("2", "s2_unknown_book", s2_unknown_book)],
    2: [("3", "s3_fix_state_first", s3_fix_state_first),
        ("3", "s3_off_by_one", s3_off_by_one),
        ("3", "s3_wrong_key", s3_wrong_key),
        ("3", "s3_missing_import", s3_missing_import),
        ("3", "s3_logic_error", s3_logic_error),
        ("3", "s3_multi_file", s3_multi_file),
        ("3", "s3_tdd_grade", s3_tdd_grade),
        ("3", "s3_extend_file", s3_extend_file),
        ("3", "s3_refactor", s3_refactor),
        ("3", "s3_traceback_only", s3_traceback_only),
        ("4", "s4_tight_budget", s4_tight_budget),
        ("4", "s4_tradeoff", s4_tradeoff),
        ("4", "s4_multi_constraint", s4_multi_constraint),
        ("4", "s4_constraint_change", s4_constraint_change),
        ("4", "s4_impossible_budget", s4_impossible_budget),
        ("4", "s4_cheapest_first", s4_cheapest_first)],
    3: [("5", "s5_long_distractors", s5_long_distractors),
        ("5", "s5_two_facts", s5_two_facts),
        ("5", "s5_fact_updated", s5_fact_updated),
        ("6", "s6_many_distractors", s6_many_distractors),
        ("6", "s6_selective_tool", s6_selective_tool),
        ("6", "s6_clarify_flight", s6_clarify_flight),
        ("6", "s6_clarify_ticker", s6_clarify_ticker),
        ("7", "s7_stale_tsla", s7_stale_tsla),
        ("7", "s7_disagree_unprompted", s7_disagree_unprompted),
        ("7", "s7_retraction", s7_retraction),
        ("8", "s8_doc_discount", s8_doc_discount),
        ("8", "s8_doc_warehouse", s8_doc_warehouse),
        ("8", "s8_doc_conflicting_codes", s8_doc_conflicting_codes)],
}

def main():
    global BASE, OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default="agentic_results_ext.json")
    a = ap.parse_args()
    BASE = a.base.rstrip("/")
    B.BASE = BASE
    OUT_PATH = a.out
    os.makedirs(SANDBOX, exist_ok=True)

    vram = vram_mib()
    print(f"VRAM after load: {vram} MiB", flush=True)
    if vram < 1024:
        print("ABORT: VRAM < 1GB — server is on CPU (silent fallback).", flush=True)
        json.dump({"fatal": "cpu_fallback", "vram_mib": vram}, open(a.out, "w"))
        sys.exit(2)

    scenarios = CHUNKS[a.chunk]
    print(f"Chunk {a.chunk}: {len(scenarios)} scenarios", flush=True)
    for axis, name, fn in scenarios:
        if time.time() - T0 > DEADLINE_S:
            print(f"DEADLINE reached — marking {name} not_run.", flush=True)
            results.append({"axis": axis, "name": name, "pass": False,
                            "detail": "not_run: 40-min chunk deadline", "not_run": True})
            save_partial()
            continue
        try:
            fn()
        except Exception as e:
            print(f"ERROR in {name}: {e}", flush=True)
            import traceback; traceback.print_exc()
            check(axis, name, False, f"harness_error: {e}")

    summary = {}
    for r in results:
        if r.get("not_run"):
            continue
        summary.setdefault(r["axis"], [0, 0])
        summary[r["axis"]][1] += 1
        if r["pass"]:
            summary[r["axis"]][0] += 1
    total_pass = sum(v[0] for v in summary.values())
    total_n = sum(v[1] for v in summary.values())
    print(f"\nCHUNK {a.chunk} SUMMARY: {total_pass}/{total_n}  per-axis={json.dumps(summary)}", flush=True)
    json.dump({"results": results, "transcripts": transcripts, "summary": summary,
               "chunk": a.chunk, "total_pass": total_pass, "total_n": total_n,
               "vram_after_load_mib": vram, "vram_end_mib": vram_mib(),
               "elapsed_s": round(time.time() - T0, 1)},
              open(a.out, "w"), indent=2)
    print(f"Wrote {a.out}", flush=True)

if __name__ == "__main__":
    main()
