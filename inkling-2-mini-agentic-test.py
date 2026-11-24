#!/usr/bin/env python3
"""Agentic-capability test for Inkling 2 Mini on Colab T4 via llama.cpp (b12669).

Runs VM-side against http://127.0.0.1:11434/v1 (localhost; the tunnel 403s
from VM egress). Goes beyond the single-turn tool calling of deep_test.py:
multi-step tool loops with state, error recovery, terminal-bench-style code
agent work (harness actually executes the model's scripts), planning with
constraints, long-horizon memory, tool-choice discipline, belief update.

The harness mediates ALL tool execution; the model never gets raw shell
access. Code tasks run in /content/agent_sandbox only.

Sampling: temp 0, reasoning_effort left at server default (same as the
deep test). Reasoning traces count against max_tokens, so budgets are
generous (2048 default, 4096 for code tasks).

Usage: python3 agentic_test.py [--base URL] [--out FILE]
Writes results JSON incrementally after every scenario (reclaim-safe).
Aborts if VRAM used < 1GB after load (silent CPU fallback).
Overall deadline: 50 min from start; remaining scenarios marked not_run.
"""
import argparse, importlib.util, json, os, re, subprocess, sys, time, urllib.request

BASE = "http://127.0.0.1:11434/v1"
MODEL = "inkling-2-mini-q3"
SANDBOX = "/content/agent_sandbox"
DEADLINE_S = 3000  # 50 min
T0 = time.time()

results = []
transcripts = {}

def check(axis, name, passed, detail, extra=None):
    rec = {"axis": axis, "name": name, "pass": bool(passed),
           "detail": str(detail)[:500]}
    if extra:
        rec.update(extra)
    results.append(rec)
    print(f"{'PASS' if passed else 'FAIL'} [{axis}] {name}: {str(detail)[:220]}", flush=True)
    save_partial()
    return bool(passed)

OUT_PATH = "agentic_results.json"
def save_partial():
    try:
        json.dump({"results": results, "transcripts": transcripts,
                   "elapsed_s": round(time.time() - T0, 1)},
                  open(OUT_PATH, "w"), indent=2)
    except Exception:
        pass

def chat(messages, tools=None, tool_choice=None, temperature=0.0,
         max_tokens=2048, timeout=600):
    body = {"model": MODEL, "messages": messages, "stream": False,
            "temperature": temperature, "max_tokens": max_tokens}
    if tools is not None:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    req = urllib.request.Request(BASE + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        dt = time.time() - t0
        msg = d["choices"][0]["message"]
        msg["_reasoning_chars"] = len(msg.get("reasoning_content") or "")
        msg["_finish"] = d["choices"][0].get("finish_reason")
        return msg, d.get("usage", {}), dt, d
    except Exception as e:
        return {"content": f"[ERROR {e}]", "tool_calls": None}, {}, time.time() - t0, {}

def content(msg):
    c = msg.get("content")
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""

def tcalls(msg):
    out = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        out.append({"id": tc.get("id", "call_1"), "name": fn.get("name"), "args": args})
    return out

def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def digits(s):
    """Strip thousands separators so '77,600' matches 77600."""
    return re.sub(r"(?<=\d),(?=\d)", "", s or "")

def vram_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return int(out.split()[0])
    except Exception:
        return -1

# ---------------- tool implementations (harness-side, deterministic) -----
class ToolState:
    def __init__(self, stock_mode="fixed"):
        self.stock_mode = stock_mode
        self.stock_calls = {}

def make_tools(state):
    def get_weather(args):
        city = norm(args.get("city", ""))
        data = {"hong kong": (24, "sunny"), "tokyo": (18, "sunny"), "london": (11, "rainy")}
        for k, (t, c) in data.items():
            if k in city:
                return json.dumps({"city": k.title(), "temp_c": t, "condition": c})
        return ("ERROR: unknown city '%s'. Known cities: Hong Kong, Tokyo, London."
                % args.get("city", ""))
    def get_time(args):
        tz = norm(args.get("timezone", ""))
        data = {"tokyo": "14:30", "hong": "15:30", "london": "07:30", "new york": "02:30"}
        for k, v in data.items():
            if k in tz:
                return json.dumps({"timezone": args.get("timezone"), "time": v})
        return "ERROR: unknown timezone '%s'." % args.get("timezone", "")
    def get_stock_price(args):
        tick = (args.get("ticker", "") or "").upper().strip()
        if state.stock_mode == "stale" and tick == "AAPL":
            n = state.stock_calls.get("AAPL", 0) + 1
            state.stock_calls["AAPL"] = n
            if n == 1:
                return ("AAPL: $145.00 (WARNING: delayed quote from 09:00 — the market "
                        "has moved since then; consider re-checking for the current price.)")
            return "AAPL: $152.75 (live quote as of 15:30)."
        fixed = {"AAPL": 150.25, "TSLA": 248.10, "NVDA": 182.40}
        if tick in fixed:
            return json.dumps({"ticker": tick, "price_usd": fixed[tick]})
        return "ERROR: unknown ticker '%s'." % tick
    def get_currency_rate(args):
        frm = norm(args.get("from", "")); to = norm(args.get("to", ""))
        rates = {("usd", "jpy"): 155.20, ("usd", "hkd"): 7.82, ("eur", "usd"): 1.08}
        if (frm, to) in rates:
            return json.dumps({"from": frm.upper(), "to": to.upper(), "rate": rates[(frm, to)]})
        return "ERROR: no rate for %s->%s." % (frm, to)
    def get_sales(args):
        region = norm(args.get("region", ""))
        data = {"north": 12400, "south": 9800, "east": 15600, "west": 11200}
        for k, v in data.items():
            if k in region:
                return json.dumps({"region": k.title(), "sales_usd": v})
        return "ERROR: unknown region '%s'." % args.get("region", "")
    def get_author(args):
        book = norm(args.get("book", ""))
        if "dune" in book:
            return "The author of Dune is Frank Herbert."
        if "1984" in book:
            return "The author of 1984 is George Orwell."
        return "ERROR: unknown book '%s'." % args.get("book", "")
    def get_author_birth_year(args):
        a = norm(args.get("author", ""))
        if "frank herbert" in a:
            return "Frank Herbert was born in 1920."
        if "george orwell" in a:
            return "George Orwell was born in 1903."
        return "ERROR: unknown author '%s'." % args.get("author", "")
    def search_flights(args):
        o = norm(args.get("origin", "")); d = norm(args.get("destination", ""))
        date = (args.get("date", "") or "").strip()
        ok_o = ("hkg" in o or "hong kong" in o)
        ok_d = ("tokyo" in d or "tyo" in d or "nrt" in d or "hnd" in d)
        if not (ok_o and ok_d):
            return "ERROR: no routes for %s -> %s." % (o, d)
        if date == "2026-12-25":
            return json.dumps({"flights": []})
        if date == "2026-12-26":
            return json.dumps({"flights": [{"flight": "CX520", "depart": "09:15",
                                            "price_usd": 320}]})
        return json.dumps({"flights": [{"flight": "CX500", "depart": "08:00",
                                        "price_usd": 350}]})
    def get_hotel_price(args):
        city = norm(args.get("city", "")); nights = args.get("nights", 3)
        try:
            nights = int(nights)
        except Exception:
            return "ERROR: nights must be a number."
        if "tokyo" in city:
            return json.dumps({"city": "Tokyo", "nights": nights,
                               "total_usd": 140 * nights})
        return "ERROR: no hotels for '%s'." % args.get("city", "")
    def get_attraction_price(args):
        name = norm(args.get("name", ""))
        data = {"teamlab": 35, "skytree": 28, "ueno": 0, "meiji": 0, "disney": 75}
        for k, v in data.items():
            if k in name:
                return json.dumps({"attraction": args.get("name"), "price_usd": v})
        return "ERROR: unknown attraction '%s'." % args.get("name", "")
    def get_random_joke(args):
        return "Why did the scarecrow win an award? Because he was outstanding in his field."
    def translate(args):
        return "[Translated to %s]: %s" % (args.get("target_lang", "?"), args.get("text", ""))
    def write_file(args):
        p = args.get("path", "")
        if not p.startswith("/content/agent_sandbox/"):
            p = os.path.join(SANDBOX, os.path.basename(p))
        try:
            with open(p, "w") as f:
                f.write(args.get("content", ""))
            return "Wrote %d bytes to %s." % (len(args.get("content", "")), p)
        except Exception as e:
            return "ERROR writing file: %s" % e
    def run_script(args):
        p = args.get("path", "")
        if not p.startswith("/content/agent_sandbox/"):
            p = os.path.join(SANDBOX, os.path.basename(p))
        try:
            r = subprocess.run([sys.executable, p], cwd=SANDBOX,
                               capture_output=True, text=True, timeout=60)
            out = (r.stdout or "")[:2000]; err = (r.stderr or "")[:2000]
            return "exit_code=%d\nstdout:\n%s\nstderr:\n%s" % (r.returncode, out, err)
        except subprocess.TimeoutExpired:
            return "ERROR: script timed out after 60s."
        except Exception as e:
            return "ERROR running script: %s" % e
    def list_files(args):
        try:
            return json.dumps(sorted(os.listdir(SANDBOX)))
        except Exception as e:
            return "ERROR: %s" % e
    return {"get_weather": get_weather, "get_time": get_time,
            "get_stock_price": get_stock_price, "get_currency_rate": get_currency_rate,
            "get_sales": get_sales, "get_author": get_author,
            "get_author_birth_year": get_author_birth_year,
            "search_flights": search_flights, "get_hotel_price": get_hotel_price,
            "get_attraction_price": get_attraction_price,
            "get_random_joke": get_random_joke, "translate": translate,
            "write_file": write_file, "run_script": run_script, "list_files": list_files}

def tool_schema(name):
    S = {
      "get_weather": ("Weather for a city", {"city": "string", "unit": "string"}, ["city"]),
      "get_time": ("Current time in a timezone", {"timezone": "string"}, ["timezone"]),
      "get_stock_price": ("Stock price for a ticker", {"ticker": "string"}, ["ticker"]),
      "get_currency_rate": ("Exchange rate between two currencies", {"from": "string", "to": "string"}, ["from", "to"]),
      "get_sales": ("Sales in USD for a region", {"region": "string"}, ["region"]),
      "get_author": ("Author of a book", {"book": "string"}, ["book"]),
      "get_author_birth_year": ("Birth year of an author", {"author": "string"}, ["author"]),
      "search_flights": ("Search flights", {"origin": "string", "destination": "string", "date": "string"}, ["origin", "destination", "date"]),
      "get_hotel_price": ("Total hotel price for a city and nights", {"city": "string", "nights": "integer"}, ["city", "nights"]),
      "get_attraction_price": ("Ticket price of an attraction", {"name": "string"}, ["name"]),
      "get_random_joke": ("A random joke", {}, []),
      "translate": ("Translate text", {"text": "string", "target_lang": "string"}, ["text", "target_lang"]),
      "write_file": ("Write content to a file in /content/agent_sandbox", {"path": "string", "content": "string"}, ["path", "content"]),
      "run_script": ("Run a Python file in the sandbox, returns stdout/stderr", {"path": "string"}, ["path"]),
      "list_files": ("List files in the sandbox", {}, []),
    }
    desc, props, req = S[name]
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object",
                       "properties": {k: {"type": "string"} for k in props},
                       "required": req}}}

# ---------------- agent loop ----------------
def run_agent(messages, tools, executor, max_turns=8, max_tokens=2048, timeout=600):
    """messages: list (system+user). Returns (final_text, turns_used, n_tool_calls, transcript)."""
    msgs = list(messages)
    transcript = []
    n_calls = 0
    t0 = time.time()
    for turn in range(max_turns):
        msg, usage, dt, _ = chat(msgs, tools=tools, max_tokens=max_tokens, timeout=timeout)
        calls = tcalls(msg)
        transcript.append({"turn": turn + 1, "role": "assistant",
                           "content": (content(msg) or "")[:800],
                           "reasoning_chars": msg.get("_reasoning_chars", 0),
                           "tool_calls": [(c["name"], c["args"]) for c in calls],
                           "secs": round(dt, 1)})
        if not calls:
            return (content(msg) or ""), turn + 1, n_calls, transcript, time.time() - t0
        n_calls += len(calls)
        msgs.append({"role": "assistant", "content": content(msg) or "",
                     "tool_calls": msg.get("tool_calls")})
        for c in calls:
            fn = executor.get(c["name"])
            res = fn(c["args"]) if fn else "ERROR: unknown tool '%s'." % c["name"]
            transcript.append({"turn": turn + 1, "role": "tool", "name": c["name"],
                               "content": str(res)[:600]})
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": str(res)})
    return "[MAX TURNS REACHED]", max_turns, n_calls, transcript, time.time() - t0

SYS_TOOL = ("You are a helpful assistant with access to tools. Use the tools to gather "
            "facts before answering. Only use values the tools return — never invent tool "
            "results. When you have enough information, answer directly without more tool calls.")
SYS_CODE = ("You are a coding agent with tools: write_file(path, content) to write a file, "
            "run_script(path) to execute a Python file and see stdout/stderr, list_files() to "
            "list the sandbox. All files live in /content/agent_sandbox. Write the requested "
            "script, run it, and if it fails, read the error, fix the file, and run it again. "
            "When it works, report the result.")

def names_used(transcript):
    return [t["name"] for t in transcript if t.get("role") == "tool"]

# ---------------- Axis 1: multi-step tool loops ----------------
def sc_trip():
    st = ToolState(); ex = make_tools(st)
    tools = [tool_schema(n) for n in
             ["get_weather", "get_time", "get_currency_rate", "get_random_joke", "translate"]]
    msgs = [{"role": "system", "content": SYS_TOOL},
            {"role": "user", "content": (
                "I am planning a day trip to Tokyo. Please check the weather in Tokyo (celsius), "
                "the current time in Asia/Tokyo, and the USD to JPY exchange rate. Then give me a "
                "short itinerary that includes the weather, the Tokyo time, and how many yen "
                "I get for $500 USD at that rate.")}]
    final, turns, ncalls, tr, wall = run_agent(msgs, tools, ex, max_turns=8)
    transcripts["trip_planning"] = tr
    used = names_used(tr)
    ok = ("18" in final and "sunny" in norm(final) and "14:30" in final and "77600" in digits(final)
          and all(n in used for n in ["get_weather", "get_time", "get_currency_rate"]))
    check("1", "trip_planning", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s tools={used} | {final[:200]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def sc_data_task():
    st = ToolState(); ex = make_tools(st)
    tools = [tool_schema(n) for n in
             ["get_sales", "get_weather", "get_stock_price", "get_random_joke"]]
    msgs = [{"role": "system", "content": SYS_TOOL},
            {"role": "user", "content": (
                "Get the sales figures for the North, South and East regions. Then compute the "
                "total sales and the average sales across the three regions, and identify the "
                "best region. Reply with a JSON object with keys total, average, best_region.")}]
    final, turns, ncalls, tr, wall = run_agent(msgs, tools, ex, max_turns=8)
    transcripts["data_task"] = tr
    used = names_used(tr)
    m = re.search(r"\{[^}]*\}", final, re.S)
    ok = False; detail = f"turns={turns} calls={ncalls} wall={wall:.0f}s tools={used} | {final[:200]}"
    if m:
        try:
            obj = json.loads(m.group(0))
            ok = (obj.get("total") in (37800, 37800.0) and obj.get("average") in (12600, 12600.0)
                  and norm(str(obj.get("best_region", ""))) == "east")
            detail = f"parsed={obj} " + detail
        except Exception as e:
            detail = f"json_parse_fail={e} " + detail
    check("1", "data_task", ok, detail,
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def sc_multihop():
    st = ToolState(); ex = make_tools(st)
    tools = [tool_schema(n) for n in
             ["get_author", "get_author_birth_year", "get_weather", "translate"]]
    msgs = [{"role": "system", "content": SYS_TOOL},
            {"role": "user", "content": "In what year was the author of Dune born?"}]
    final, turns, ncalls, tr, wall = run_agent(msgs, tools, ex, max_turns=8)
    transcripts["multihop_qa"] = tr
    used = names_used(tr)
    ok = ("1920" in final and "get_author" in used and "get_author_birth_year" in used)
    check("1", "multihop_qa", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s tools={used} | {final[:200]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

# ---------------- Axis 2: error recovery ----------------
def sc_bad_argument():
    st = ToolState(); ex = make_tools(st)
    tools = [tool_schema(n) for n in ["get_weather", "get_time"]]
    msgs = [{"role": "system", "content": SYS_TOOL},
            {"role": "user", "content": (
                "What is the weather in Atlantis? If the tool says the city is unknown, "
                "it will list the known cities — then check the weather in Hong Kong instead "
                "and report that.")}]
    final, turns, ncalls, tr, wall = run_agent(msgs, tools, ex, max_turns=8)
    transcripts["error_bad_argument"] = tr
    calls = [(t["name"], t["content"]) for t in tr if t.get("role") == "tool"]
    wx = [t for t in tr if t.get("role") == "tool" and t.get("name") == "get_weather"]
    ok = (len(wx) >= 2 and "24" in final and "sunny" in norm(final)
          and "ERROR" in str(calls[0][1]) if calls else False)
    check("2", "error_bad_argument", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s weather_calls={len(wx)} | {final[:200]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def sc_empty_result():
    st = ToolState(); ex = make_tools(st)
    tools = [tool_schema(n) for n in ["search_flights", "get_weather"]]
    msgs = [{"role": "system", "content": SYS_TOOL},
            {"role": "user", "content": (
                "I need a flight from Hong Kong to Tokyo on 2026-12-25. Check if any flight "
                "is available that day. If there are none, try another nearby date and tell me "
                "what you find — do not invent a flight.")}]
    final, turns, ncalls, tr, wall = run_agent(msgs, tools, ex, max_turns=8)
    transcripts["error_empty_result"] = tr
    sfs = [t for t in tr if t.get("role") == "tool" and t.get("name") == "search_flights"]
    # pass = searched twice (did not invent a flight from the empty result)
    # and reported a real flight returned by a later search.
    ok = len(sfs) >= 2 and ("CX520" in final or "CX500" in final)
    check("2", "error_empty_result", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s flight_searches={len(sfs)} | {final[:200]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

# ---------------- Axis 3: code agent (harness executes) ----------------
def sc_csv_stats():
    os.makedirs(SANDBOX, exist_ok=True)
    with open(os.path.join(SANDBOX, "sales.csv"), "w") as f:
        f.write("product,units,price\napple,120,2.5\nbanana,200,1.2\ncherry,90,4.0\n")
    st = ToolState(); ex = make_tools(st)
    tools = [tool_schema(n) for n in ["write_file", "run_script", "list_files"]]
    msgs = [{"role": "system", "content": SYS_CODE},
            {"role": "user", "content": (
                "Write a Python script to /content/agent_sandbox/stats.py that reads "
                "/content/agent_sandbox/sales.csv and prints the total revenue and the "
                "best-selling product by units. Run the script and report the two numbers.")}]
    final, turns, ncalls, tr, wall = run_agent(msgs, tools, ex, max_turns=8,
                                               max_tokens=4096)
    transcripts["code_csv_stats"] = tr
    p = os.path.join(SANDBOX, "stats.py")
    out = ""
    if os.path.exists(p):
        r = subprocess.run([sys.executable, p], cwd=SANDBOX, capture_output=True,
                           text=True, timeout=60)
        out = r.stdout + r.stderr
    ok = ("900" in out and "banana" in norm(out) and "900" in final
          and "banana" in norm(final))
    check("3", "code_csv_stats", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s rerun_out={out.strip()[:150]!r} | {final[:200]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def sc_hidden_tests():
    os.makedirs(SANDBOX, exist_ok=True)
    st = ToolState(); ex = make_tools(st)
    tools = [tool_schema(n) for n in ["write_file", "run_script", "list_files"]]
    msgs = [{"role": "system", "content": SYS_CODE},
            {"role": "user", "content": (
                "Write a Python script to /content/agent_sandbox/fizzbuzz.py defining a "
                "function fizzbuzz(n): return 'FizzBuzz' if n is divisible by 15, 'Fizz' if "
                "divisible by 3, 'Buzz' if divisible by 5, else str(n). Run the script to "
                "check it loads, then report that it is done.")}]
    final, turns, ncalls, tr, wall = run_agent(msgs, tools, ex, max_turns=8,
                                               max_tokens=4096)
    transcripts["code_hidden_tests"] = tr
    p = os.path.join(SANDBOX, "fizzbuzz.py")
    passed = 0; total = 4; detail = "file missing"
    if os.path.exists(p):
        try:
            spec = importlib.util.spec_from_file_location("fb", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cases = [(3, "Fizz"), (5, "Buzz"), (15, "FizzBuzz"), (7, "7")]
            passed = sum(1 for n, e in cases if mod.fizzbuzz(n) == e)
            detail = f"hidden {passed}/{total}"
        except Exception as e:
            detail = f"import/call failed: {e}"
    ok = passed == total
    check("3", "code_hidden_tests", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s {detail} | {final[:150]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def sc_fix_buggy():
    os.makedirs(SANDBOX, exist_ok=True)
    with open(os.path.join(SANDBOX, "buggy.py"), "w") as f:
        f.write("nums = [10, 20, 30]\ntotal = sum(nums)\ncount = \"3\"\n"
                "print(total / count)\n")
    st = ToolState(); ex = make_tools(st)
    tools = [tool_schema(n) for n in ["write_file", "run_script", "list_files"]]
    msgs = [{"role": "system", "content": SYS_CODE},
            {"role": "user", "content": (
                "The file /content/agent_sandbox/buggy.py crashes when run. Read it (use "
                "run_script to see the error), fix the bug, run it again, and report the "
                "printed result.")}]
    final, turns, ncalls, tr, wall = run_agent(msgs, tools, ex, max_turns=8,
                                               max_tokens=4096)
    transcripts["code_fix_buggy"] = tr
    p = os.path.join(SANDBOX, "buggy.py")
    out = ""; rc = -1
    if os.path.exists(p):
        r = subprocess.run([sys.executable, p], cwd=SANDBOX, capture_output=True,
                           text=True, timeout=60)
        out = (r.stdout + r.stderr).strip(); rc = r.returncode
    ok = (rc == 0 and "20" in out and "20" in final)
    check("3", "code_fix_buggy", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s rerun_rc={rc} out={out[:120]!r} | {final[:200]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

# ---------------- Axis 4: planning + constraints ----------------
def sc_budget_plan():
    st = ToolState(); ex = make_tools(st)
    tools = [tool_schema(n) for n in
             ["get_hotel_price", "get_attraction_price", "get_weather", "translate"]]
    msgs = [{"role": "system", "content": SYS_TOOL},
            {"role": "user", "content": (
                "Plan a 3-day trip to Tokyo with a total budget of $500. First check the hotel "
                "price for Tokyo for 3 nights, and the ticket prices of the attractions you "
                "want to visit. Then present the plan as exactly 3 sections headed 'Day 1', "
                "'Day 2', 'Day 3', each with one attraction and its cost, and end with a line "
                "'TOTAL: $X' where X is the sum of the hotel and the three attraction prices. "
                "The total must not exceed $500.")}]
    final, turns, ncalls, tr, wall = run_agent(msgs, tools, ex, max_turns=8)
    transcripts["plan_budget"] = tr
    used = names_used(tr)
    days = len(re.findall(r"day\s*[123]", norm(final)))
    m = re.search(r"total\D*\$?\s*(\d[\d,]*(?:\.\d+)?)", norm(final))
    total = float(m.group(1).replace(",", "")) if m else None
    # expected total from tools: hotel 420 + prices of attractions named in final
    prices = {"teamlab": 35, "skytree": 28, "ueno": 0, "meiji": 0, "disney": 75}
    exp = 420 + sum(v for k, v in prices.items() if k in norm(final))
    ok = (days == 3 and total is not None and total <= 500
          and abs(total - exp) < 0.01
          and "get_hotel_price" in used and "get_attraction_price" in used)
    check("4", "plan_budget", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s days={days} total={total} expected={exp} tools={used} | {final[:200]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def sc_exact_format():
    st = ToolState(); ex = make_tools(st)
    tools = [tool_schema(n) for n in ["get_weather", "get_time", "get_random_joke"]]
    msgs = [{"role": "system", "content": SYS_TOOL},
            {"role": "user", "content": (
                "Check the weather in Hong Kong and in Tokyo. Then reply with EXACTLY two "
                "lines, one per city, in this format: 'Hong Kong: 24C, sunny' — i.e. "
                "'<city>: <temp>C, <condition>'. No intro, no extra text, no bullet points.")}]
    final, turns, ncalls, tr, wall = run_agent(msgs, tools, ex, max_turns=8)
    transcripts["plan_exact_format"] = tr
    lines = [l for l in (final or "").strip().splitlines() if l.strip()]
    pat = re.compile(r"^.+: \d+C, [a-z]+$", re.I)
    ok = (len(lines) == 2 and all(pat.match(l.strip()) for l in lines)
          and "24" in final and "18" in final)
    check("4", "plan_exact_format", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s lines={len(lines)} | {final[:200]!r}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

# ---------------- Axis 5: long-horizon memory ----------------
def sc_memory():
    msgs = [{"role": "system", "content": "You are a helpful assistant. Remember facts the user tells you."},
            {"role": "user", "content": "My project code name is BLUE-FALCON and the launch date is 2027-03-15. Please remember this."}]
    m, _, _, _ = chat(msgs, max_tokens=512)
    msgs.append({"role": "assistant", "content": content(m)})
    distractors = ["What is the weather like in general in spring?",
                   "What is 17 + 25?",
                   "Tell me a one-sentence joke about cats.",
                   "What is the capital of France?"]
    for q in distractors:
        msgs.append({"role": "user", "content": q})
        m, _, _, _ = chat(msgs, max_tokens=512)
        msgs.append({"role": "assistant", "content": content(m)})
    msgs.append({"role": "user", "content": "What is my project code name and launch date?"})
    t0 = time.time()
    m, _, _, _ = chat(msgs, max_tokens=512)
    wall = time.time() - t0
    final = content(m) or ""
    transcripts["memory_long_horizon"] = [
        {"turn": i + 1, "role": mm["role"], "content": (mm.get("content") or "")[:300]}
        for i, mm in enumerate(msgs[1:])] + [
        {"turn": 99, "role": "assistant", "content": final[:400]}]
    ok = "BLUE-FALCON" in final and "2027-03-15" in final
    check("5", "memory_long_horizon", ok,
          f"distractor_turns=4 wall={wall:.0f}s | {final[:200]}",
          {"turns": 6, "tool_calls": 0, "wall_s": round(wall, 1)})

# ---------------- Axis 6: tool-choice discipline ----------------
def sc_distractor_tools():
    st = ToolState(); ex = make_tools(st)
    tools = [tool_schema(n) for n in
             ["get_stock_price", "get_weather", "get_time", "get_random_joke", "translate",
              "get_sales"]]
    msgs = [{"role": "system", "content": SYS_TOOL},
            {"role": "user", "content": "What is the stock price of AAPL?"}]
    final, turns, ncalls, tr, wall = run_agent(msgs, tools, ex, max_turns=6)
    transcripts["discipline_distractors"] = tr
    used = names_used(tr)
    ok = used == ["get_stock_price"] and "150.25" in final
    check("6", "discipline_distractors", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s tools={used} | {final[:150]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

def sc_no_tool_needed():
    st = ToolState(); ex = make_tools(st)
    tools = [tool_schema(n) for n in
             ["get_stock_price", "get_weather", "get_time", "get_random_joke", "translate"]]
    ok_all = True; detail_parts = []
    for q, expect in [("What is 2+2? Answer with just the number.", "4"),
                      ("What is the capital of France? Answer with just the city name.", "paris")]:
        msgs = [{"role": "system", "content": SYS_TOOL}, {"role": "user", "content": q}]
        final, turns, ncalls, tr, wall = run_agent(msgs, tools, ex, max_turns=4,
                                                   max_tokens=512)
        used = names_used(tr)
        good = (ncalls == 0 and expect in norm(final))
        ok_all = ok_all and good
        detail_parts.append(f"{q[:20]}: calls={ncalls} ok={good} ans={final[:60]!r}")
    transcripts["discipline_no_tool"] = [{"note": "; ".join(detail_parts)}]
    check("6", "discipline_no_tool", ok_all, " | ".join(detail_parts),
          {"turns": 2, "tool_calls": 0, "wall_s": 0})

# ---------------- Axis 7: belief update ----------------
def sc_belief_update():
    st = ToolState(stock_mode="stale"); ex = make_tools(st)
    tools = [tool_schema(n) for n in ["get_stock_price", "get_weather"]]
    msgs = [{"role": "system", "content": SYS_TOOL},
            {"role": "user", "content": "What is the current price of AAPL?"}]
    final, turns, ncalls, tr, wall = run_agent(msgs, tools, ex, max_turns=8)
    transcripts["belief_update"] = tr
    n_stock = names_used(tr).count("get_stock_price")
    ok = n_stock >= 2 and "152.75" in final
    check("7", "belief_update", ok,
          f"turns={turns} calls={ncalls} wall={wall:.0f}s stock_calls={n_stock} | {final[:200]}",
          {"turns": turns, "tool_calls": ncalls, "wall_s": round(wall, 1)})

# ---------------- main ----------------
SCENARIOS = [
    ("1", "trip_planning", sc_trip),
    ("1", "data_task", sc_data_task),
    ("1", "multihop_qa", sc_multihop),
    ("2", "error_bad_argument", sc_bad_argument),
    ("2", "error_empty_result", sc_empty_result),
    ("3", "code_csv_stats", sc_csv_stats),
    ("3", "code_hidden_tests", sc_hidden_tests),
    ("3", "code_fix_buggy", sc_fix_buggy),
    ("4", "plan_budget", sc_budget_plan),
    ("4", "plan_exact_format", sc_exact_format),
    ("5", "memory_long_horizon", sc_memory),
    ("6", "discipline_distractors", sc_distractor_tools),
    ("6", "discipline_no_tool", sc_no_tool_needed),
    ("7", "belief_update", sc_belief_update),
]

def main():
    global BASE, OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default="agentic_results.json")
    a = ap.parse_args()
    BASE = a.base.rstrip("/")
    OUT_PATH = a.out
    os.makedirs(SANDBOX, exist_ok=True)

    vram = vram_mib()
    print(f"VRAM after load: {vram} MiB", flush=True)
    if vram < 1024:
        print("ABORT: VRAM < 1GB — server is on CPU (silent fallback).", flush=True)
        json.dump({"fatal": "cpu_fallback", "vram_mib": vram}, open(a.out, "w"))
        sys.exit(2)

    done = {r["name"] for r in results}
    for axis, name, fn in SCENARIOS:
        if name in done:
            continue
        if time.time() - T0 > DEADLINE_S:
            print(f"DEADLINE reached — marking {name} and the rest not_run.", flush=True)
            results.append({"axis": axis, "name": name, "pass": False,
                            "detail": "not_run: 50-min deadline", "not_run": True})
            save_partial()
            continue
        try:
            fn()
        except Exception as e:
            print(f"ERROR in {name}: {e}", flush=True)
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
    print(f"\nSUMMARY: {total_pass}/{total_n}  per-axis={json.dumps(summary)}", flush=True)
    json.dump({"results": results, "transcripts": transcripts, "summary": summary,
               "total_pass": total_pass, "total_n": total_n,
               "vram_after_load_mib": vram, "vram_end_mib": vram_mib(),
               "elapsed_s": round(time.time() - T0, 1)},
              open(a.out, "w"), indent=2)
    print(f"Wrote {a.out}", flush=True)

if __name__ == "__main__":
    main()
