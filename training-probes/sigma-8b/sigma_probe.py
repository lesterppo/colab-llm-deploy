#!/usr/bin/env python3
"""Sigma-specific probe for the arXiv digest tracks.

Nemotron-Labs-Sigma is a CONTINUOUS DIFFUSION LM, not a causal LM:
  - forward() raises NotImplementedError ("use model.generate(...)")
  - no lm_head, no causal loss, no token logprobs
  - generation is block-parallel diffusion (DDIM/DDPM) via the native
    model.generate(input_ids, sampler, time_schedule, steps, block_length,
    guidance_scale, score_temp) -> (out_ids, nfe)
Do NOT force the causal-LM probe (probe.py/probe2.py) onto it.

Phases (results JSON flushed after each):
  env -> hf_info -> load (bf16 | nf4) -> capability battery (native sampler)
  -> lora_probe (one standard QLoRA attach attempt) -> step_probe (one
  standard causal-LM training-step attempt) -> logprob_probe (frozen scorer?)
Usage: python3 sigma_probe.py --model <HF id> --slug <slug>
       --load {bf16,nf4} --out /content/<slug>_results_<load>.json
Exits 0 if load+capability completed (lora/step failures are recorded
findings, not script failures). Exits 2 if load failed.
Capability battery reuses probe.py's prompts: reasoning 8, tool-JSON 3,
instruction 2, needle 4k/8k, ZH 3, temp-0/seed determinism, throughput.
Sampler (README recommended, both sizes): DDIM-32, log_snr, block 32,
guidance 3.0, score_temp 0.3. Needle uses DDIM-16 / 0.4 to bound time.
Effective tok/s = new_tokens / wall seconds of a parallel block decode;
it is NOT autoregressive tok/s and is labelled as such everywhere.
"""
import argparse, json, os, re, sys, time, traceback
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--slug", required=True)
ap.add_argument("--load", choices=["bf16", "nf4"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--schedule-fix", action="store_true",
                help="set config.schedule_parameterization='projected_bernstein' "
                     "(the checkpoint's own lineage) before loading, so the "
                     "checkpoint's Bernstein increments + gamma endpoints "
                     "[-5.415, 4.855] are used instead of legacy defaults (-3, 6)")
a = ap.parse_args()

RESULTS = {"model": a.model, "slug": a.slug, "load_mode": a.load,
           "architecture": "continuous-diffusion-lm (NOT causal LM)",
           "phases": {}, "checks": []}

def flush():
    with open(a.out, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)

def vram():
    if torch.cuda.is_available():
        return {"alloc_gb": round(torch.cuda.memory_allocated()/2**30, 3),
                "reserved_gb": round(torch.cuda.memory_reserved()/2**30, 3),
                "max_alloc_gb": round(torch.cuda.max_memory_allocated()/2**30, 3)}
    return {}

def ram():
    try:
        m = dict(l.split(":") for l in open("/proc/meminfo").read().splitlines())
        total = int(m["MemTotal"].split()[0])/2**20
        avail = int(m["MemAvailable"].split()[0])/2**20
        return {"total_gb": round(total,2), "avail_gb": round(avail,2),
                "used_gb": round(total-avail,2)}
    except Exception as e:
        return {"error": str(e)}

def phase(name, fn):
    t0 = time.time()
    try:
        fn()
        RESULTS["phases"][name] = {"status": "ok", "secs": round(time.time()-t0,1)}
    except Exception as e:
        RESULTS["phases"][name] = {"status": "FAILED", "secs": round(time.time()-t0,1),
            "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:],
            "vram_at_failure": vram(), "ram_at_failure": ram()}
        flush(); raise
    flush()

def check(axis, name, passed, detail):
    RESULTS["checks"].append({"axis": axis, "name": name, "pass": bool(passed),
                              "detail": str(detail)[:400]})
    print(f"{'PASS' if passed else 'FAIL'} [{axis}] {name}: {str(detail)[:200]}", flush=True)
    flush()

def strip_think(s):
    if not s: return ""
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    return s.strip()

def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())

# ---------------- env ----------------
def env():
    import platform
    try:
        import transformers, peft, accelerate
        try:
            import bitsandbytes as bnb; bnb_v = bnb.__version__
        except Exception as e:
            bnb_v = f"NOT INSTALLED ({e})"
        RESULTS["versions"] = {"python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": transformers.__version__, "peft": peft.__version__,
            "accelerate": accelerate.__version__, "bitsandbytes": bnb_v,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory/2**30,2)
                if torch.cuda.is_available() else None}
    except Exception as e:
        RESULTS["versions"] = {"error": f"{type(e).__name__}: {e}"}
    RESULTS["ram_before"] = ram(); RESULTS["vram_before"] = vram()
    print("ENV:", json.dumps(RESULTS["versions"]), flush=True)
phase("env", env)

# ---------------- hf_info ----------------
def hf_info():
    from huggingface_hub import hf_hub_download, list_repo_files
    files = list_repo_files(a.model, repo_type="model")
    RESULTS["hf_files"] = sorted(files)
    cfg_path = hf_hub_download(a.model, "config.json")
    cfg = json.load(open(cfg_path))
    RESULTS["model_config"] = {k: v for k, v in cfg.items() if k != "tokenizer_config"}
    idx_path = hf_hub_download(a.model, "model.safetensors.index.json")
    idx = json.load(open(idx_path))
    RESULTS["total_size_gb"] = round(idx["metadata"]["total_size"]/2**30, 2)
    RESULTS["shards"] = sorted(set(idx["weight_map"].values()))
    RESULTS["has_chat_template_file"] = "chat_template.jinja" in files
    RESULTS["has_chat_template_in_tokenizer_config"] = bool(
        (cfg.get("tokenizer_config") or {}).get("chat_template"))
    train_files = [f for f in files if re.search(r"train|loss|trainer|sft|dpo|grpo",
                                                  f, re.I)]
    RESULTS["training_files_shipped"] = train_files
    RESULTS["custom_code_files"] = [f for f in files if f.endswith(".py")]
    try:
        rd = hf_hub_download(a.model, "README.md")
        text = open(rd, encoding="utf-8", errors="ignore").read()
        RESULTS["readme_license_mentions"] = {
            "Nemotron Open Model License": "Nemotron Open Model License" in text,
            "max_context_8192_claim": "8,192" in text or "8192" in text,
        }
        m = re.search(r"Maximum context length of ([\d,]+) tokens", text)
        RESULTS["readme_max_context"] = m.group(1) if m else None
    except Exception as e:
        RESULTS["readme_license_mentions"] = {"error": str(e)}
    print("HF INFO:", json.dumps({k: RESULTS[k] for k in
        ["model_config", "total_size_gb", "shards", "has_chat_template_file",
         "training_files_shipped"]}, default=str)[:1500], flush=True)
phase("hf_info", hf_info)

# ---------------- load ----------------
def load():
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    RESULTS["tokenizer"] = {"vocab_size": tok.vocab_size, "eos": tok.eos_token,
                            "pad": tok.pad_token, "chat_template_present":
                            tok.chat_template is not None}
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg_override = None
    if a.schedule_fix:
        cfg_override = AutoConfig.from_pretrained(a.model, trust_remote_code=True)
        cfg_override.schedule_parameterization = "projected_bernstein"
        cfg_override.schedule_degree = 32
        RESULTS["schedule_fix"] = {
            "applied": True,
            "schedule_parameterization": "projected_bernstein",
            "reason": "shipped config.json omits the flag (defaults to legacy) "
                      "but the checkpoint stores Bernstein increments + "
                      "gamma_bounds.endpoint_values=[-5.415, 4.855]; legacy "
                      "defaults (-3, 6) denoise over the wrong log-SNR range"}
    attempts = []
    model = None
    if a.load == "nf4":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        kw = dict(quantization_config=bnb, device_map="auto",
                  torch_dtype=torch.bfloat16, trust_remote_code=True)
        tries = [("nf4/device_map=auto", kw)]
    else:
        kw_auto = dict(torch_dtype=torch.bfloat16, device_map="auto",
                       trust_remote_code=True)
        kw_cuda = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
        tries = [("bf16/device_map=auto", kw_auto),
                 ("bf16/README-pattern cpu-then-cuda", kw_cuda)]
    last_err = None
    for label, kw in tries:
        try:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time(); ram0 = ram()
            kw2 = dict(kw)
            if cfg_override is not None:
                kw2["config"] = cfg_override
            m = AutoModel.from_pretrained(a.model, **kw2)
            if "cpu-then-cuda" in label:
                m = m.cuda().to(torch.bfloat16)
            m = m.eval()
            RESULTS["load"] = {"secs": round(time.time()-t0,1), "attempt": label,
                "vram_after_load": vram(), "ram_before_load": ram0,
                "ram_after_load": ram(),
                "total_params_b": round(sum(p.numel() for p in m.parameters())/1e9, 2),
                "model_class": type(m).__name__}
            print("LOAD OK:", RESULTS["load"], flush=True)
            model = m; break
        except Exception as e:
            last_err = f"{label}: {type(e).__name__}: {e}"
            print("LOAD ATTEMPT FAILED:", last_err[:500], flush=True)
            attempts.append(last_err); flush()
            try: torch.cuda.empty_cache()
            except Exception: pass
    RESULTS["load_attempts"] = attempts
    if model is None:
        raise RuntimeError(f"all load attempts failed; last: {last_err}")
    # module inventory
    from collections import Counter
    suffixes = Counter()
    n_linear = 0
    for name, mod in model.named_modules():
        suffixes[type(mod).__name__] += 1
        if isinstance(mod, torch.nn.Linear):
            n_linear += 1
    RESULTS["module_inventory"] = {"by_class": dict(suffixes),
        "n_linear": n_linear,
        "has_lm_head": hasattr(model, "lm_head"),
        "has_diffusion_head": hasattr(model, "diffusion_head"),
        "has_ar_head": hasattr(model, "ar_head"),
        "noise_schedule_class": type(model.noise_schedule).__name__,
        "gamma_bounds_loaded": [float(x) for x in model.gamma_bounds()],
        "forward_raises": None}
    try:
        model(input_ids=torch.ones(1, 2, dtype=torch.long, device="cuda"))
        RESULTS["module_inventory"]["forward_raises"] = False
    except Exception as e:
        RESULTS["module_inventory"]["forward_raises"] = f"{type(e).__name__}: {e}"
    print("MODULES:", json.dumps(RESULTS["module_inventory"], default=str)[:800], flush=True)
    RESULTS["_model"] = model; RESULTS["_tok"] = tok

try:
    phase("load", load)
except Exception:
    print("LOAD FAILED", flush=True); sys.exit(2)
model = RESULTS.pop("_model"); tok = RESULTS.pop("_tok")
RESULTS["vram_after_load_gb"] = vram()

# ---------------- native generate helper ----------------
GEN_DEFAULTS = dict(sampler="ddim", time_schedule="log_snr", steps=32,
                    block_length=32, guidance_scale=3.0, score_temp=0.3)
EOS_IDS = {11, 2}  # <|im_end|> (instruct EOS), </s>

def gen(messages, max_new=128, gen_kwargs=None, generator=None):
    """messages: [{role, content}]. Returns dict(text, prompt_tokens,
    new_tokens, secs, nfe, eff_tok_s, eos_hit, settings)."""
    kw = dict(GEN_DEFAULTS); kw.update(gen_kwargs or {})
    try:
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
    except Exception:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
    ids = tok(text, return_tensors="pt").to("cuda")
    ptok = int(ids["input_ids"].shape[1])
    t0 = time.time()
    with torch.no_grad():
        out_ids, nfe = model.generate(ids["input_ids"], max_new_tokens=max_new,
                                      generator=generator, **kw)
    secs = time.time() - t0
    new_ids = out_ids[0][ptok:]
    new_list = new_ids.tolist()
    raw_new = len(new_list)
    eos_hit = False
    for i, t in enumerate(new_list):
        if t in EOS_IDS:
            new_list = new_list[:i]; eos_hit = True; break
    txt = tok.decode(new_list, skip_special_tokens=True)
    return {"text": txt, "prompt_tokens": ptok, "new_tokens": len(new_list),
            "raw_new_tokens": raw_new,
            "secs": round(secs, 1), "nfe": int(nfe),
            "eff_tok_s": round(raw_new/max(secs, 0.01), 1),
            "eos_hit": eos_hit, "settings": kw,
            "raw_tail": tok.decode(out_ids[0][ptok:][:80].tolist(),
                                   skip_special_tokens=False)[-200:]}

def chat_gen(user_text, max_new=128, system=None, gen_kwargs=None, generator=None):
    msgs = []
    if system: msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_text})
    return gen(msgs, max_new=max_new, gen_kwargs=gen_kwargs, generator=generator)

# ---------------- capability battery ----------------
def capability():
    # A. smoke + throughput
    r = chat_gen("What is 2+2? Answer with just the number.", max_new=64)
    RESULTS["smoke_raw"] = r
    check("A", "smoke_2plus2", "4" in strip_think(r["text"]), r["text"][:150])
    r2 = chat_gen("Write a 150-word paragraph about why the sky is blue.", max_new=200)
    check("A", "throughput", r2["raw_new_tokens"] > 20 and r2["eff_tok_s"] > 0,
          f"raw_new_tokens={r2['raw_new_tokens']} (trunc={r2['new_tokens']}) wall={r2['secs']}s nfe={r2['nfe']} "
          f"effective~{r2['eff_tok_s']} tok/s (parallel block decode, NOT autoregressive)")
    RESULTS["throughput"] = {"effective_tok_s": r2["eff_tok_s"], "nfe": r2["nfe"],
        "secs": r2["secs"], "new_tokens": r2["new_tokens"],
        "note": "diffusion block-parallel decode; tok/s is wall-clock "
                "throughput, not per-token autoregressive decode"}
    # B. tool-JSON proxy (prompt-based, not native tool_calls)
    tool_sys = ("You are a helpful assistant with access to functions. When the user asks for "
        "weather/time/stock, reply with ONLY a JSON object like "
        '{"function": "get_weather", "arguments": {"city": "Hong Kong", "unit": "celsius"}}. '
        "No other text.")
    r = chat_gen("What is the weather in Hong Kong in celsius?",
                 max_new=256, system=tool_sys)
    ok = False; detail = r["text"][:200]
    try:
        j = json.loads(strip_think(r["text"]).strip().strip("`"))
        ok = j.get("function") == "get_weather" and "Hong Kong" in str(j.get("arguments", {}))
        detail = str(j)
    except Exception: pass
    check("B", "tool_json_single", ok, detail)
    r = chat_gen("Get the weather in Hong Kong (celsius), the time in Asia/Tokyo, and the stock price of AAPL. Reply with ONLY a JSON array of function calls, no other text.",
                 max_new=512, system=tool_sys)
    ok = False
    try:
        j = json.loads(strip_think(r["text"]).strip().strip("`"))
        names = {c.get("function") for c in j} if isinstance(j, list) else set()
        ok = names == {"get_weather", "get_time", "get_stock_price"}
        detail = str(j)[:300]
    except Exception: detail = r["text"][:200]
    check("B", "tool_json_parallel3", ok, detail)
    r = chat_gen("Tell me a one-sentence joke about cats.", max_new=256, system=tool_sys)
    check("B", "tool_json_no_tool", "function" not in strip_think(r["text"]).lower() or "get_" not in r["text"],
          r["text"][:150])
    # C. reasoning (8) — same prompts as probe.py
    problems = [
        ("arith1", "A train travels 60 km in 45 minutes. At the same speed, how far does it travel in 2 hours? Give just the number of km.", "160"),
        ("arith2", "What is 17 x 19? Give just the number.", "323"),
        ("arith4", "What is 9 x 8? Give just the number.", "72"),
        ("combinatorics", "How many 3-digit numbers have digits that sum to 10? Give just the number.", "54"),
        ("wordprob", "Maria has twice as many apples as Tom. Together they have 36 apples. How many does Maria have? Give just the number.", "24"),
        ("seq", "What is the next number in the sequence 2, 6, 12, 20, 30? Give just the number.", "42"),
        ("age", "A father is 3 times as old as his son. In 12 years he will be twice as old. How old is the son now? Give just the number.", "12"),
        ("logic_knights", "On an island, knights always tell the truth and knaves always lie. A says: 'B is a knave.' B says: 'A and I are the same type.' Who is the knave? Answer with just A or B.", "B"),
    ]
    for name, q, expect in problems:
        r = chat_gen(q, max_new=128)
        t = strip_think(r["text"]); tn = norm(t)
        if expect.isdigit():
            nums = re.findall(r"\d+", tn)
            passed = bool(nums) and nums[-1] == expect
            detail = f"expect={expect} last_num={nums[-1] if nums else None} eos_hit={r['eos_hit']} | {t[:150]}"
        else:
            passed = expect.lower() in tn
            detail = f"expect={expect} eos_hit={r['eos_hit']} | {t[:150]}"
        check("C", name, passed, detail)
    # D. instruction following (2)
    r = chat_gen('Return ONLY this JSON object, no other text: {"name": "Ada", "age": 36, "city": "London"}.', max_new=256)
    try:
        ok = json.loads(strip_think(r["text"]).strip()) == {"name": "Ada", "age": 36, "city": "London"}
    except Exception: ok = False
    check("D", "strict_json", ok, strip_think(r["text"])[:200])
    r = chat_gen("List exactly 3 benefits of drinking water, as exactly 3 bullet points, one per line, no other text.", max_new=256)
    lines = [l for l in strip_think(r["text"]).strip().splitlines() if l.strip()]
    bullets = [l for l in lines if l.strip().startswith(("-", "*", "•")) or re.match(r"\d+[.)]", l.strip())]
    check("D", "exact_3_bullets", len(bullets) == 3 and len(lines) == 3,
          f"lines={len(lines)} bullets={len(bullets)} | {strip_think(r['text'])[:150]}")
    # F. Chinese (3)
    r = chat_gen("用中文回答：香港的首都是哪里？请用一句话回答。", max_new=256)
    check("F", "zh_qa", "香港" in (r["text"] or ""), strip_think(r["text"])[:150])
    r = chat_gen("Translate to English: 人工智能正在改变医疗行业。", max_new=256)
    check("F", "zh_to_en", "artificial intelligence" in norm(strip_think(r["text"])) and any(w in norm(strip_think(r["text"])) for w in ["medical", "healthcare", "medicine"]), strip_think(r["text"])[:150])
    r = chat_gen("Translate to Chinese: The weather in Tokyo is sunny today.", max_new=256)
    check("F", "en_to_zh", "东京" in (r["text"] or "") and "晴" in (r["text"] or ""), strip_think(r["text"])[:150])
    # G. determinism: seeded (same torch.Generator seed twice) + unseeded (twice)
    q = "What is the capital of Australia? Answer with just the city name."
    def seeded(seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        return chat_gen(q, max_new=64, generator=g)
    s1 = seeded(20261125); s2 = seeded(20261125)
    u1 = chat_gen(q, max_new=64); u2 = chat_gen(q, max_new=64)
    RESULTS["determinism"] = {
        "seeded_run1": s1["text"], "seeded_run2": s2["text"],
        "seeded_identical": s1["text"] == s2["text"],
        "unseeded_run1": u1["text"], "unseeded_run2": u2["text"],
        "unseeded_identical": u1["text"] == u2["text"]}
    check("G", "temp0_consistency", s1["text"] == s2["text"] and "canberra" in norm(strip_think(s1["text"])),
          f"seeded_identical={s1['text']==s2['text']} unseeded_identical={u1['text']==u2['text']} "
          f"seeded={strip_think(s1['text'])!r} unseeded1={strip_think(u1['text'])!r}")
    # E. needle 4k / 8k (DDIM-16 to bound time; sanctioned setting)
    filler = "The history of the Roman Empire spans many centuries of trade, war, and law. "
    def needle(target_tok, code):
        n_chars = target_tok * 4
        hay = (filler * (n_chars // len(filler) + 1))[:n_chars]
        mid = len(hay)//2
        doc = hay[:mid] + f" The secret code is {code}. " + hay[mid:]
        q = f"Read the following text and answer the question.\n\n{doc}\n\nQuestion: What is the secret code? Answer with just the requested value."
        return chat_gen(q, max_new=64,
                        gen_kwargs=dict(steps=16, score_temp=0.4))
    for target_tok, code, name in [(4000, "BLUE-FALCON-7", "needle_4k"),
                                   (8000, "RED-TIGER-42", "needle_8k")]:
        try:
            r = needle(target_tok, code)
            check("E", name, code in (r["text"] or ""),
                  f"prompt_tokens={r['prompt_tokens']} nfe={r['nfe']} secs={r['secs']} | {strip_think(r['text'])[:120]}")
        except Exception as e:
            check("E", name, False, f"FAILED: {type(e).__name__}: {str(e)[:200]}")
            try: torch.cuda.empty_cache()
            except Exception: pass
    RESULTS["vram_after_capability"] = vram()
    summary = {}
    for r in RESULTS["checks"]:
        summary.setdefault(r["axis"], [0, 0])
        summary[r["axis"]][1] += 1
        if r["pass"]: summary[r["axis"]][0] += 1
    RESULTS["capability_summary"] = summary
    print("CAPABILITY SUMMARY:", json.dumps(summary), flush=True)

try:
    phase("capability", capability)
except Exception:
    print("CAPABILITY FAILED — continuing to LoRA/step probes if model usable", flush=True)
    RESULTS["capability_failed"] = True; flush()

# ---------------- QLoRA attach attempt ----------------
def lora_probe():
    from peft import LoraConfig, get_peft_model, TaskType
    targets = [c for c in ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"]
               if any(c in n for n, _ in model.named_modules())]
    RESULTS["lora_probe"] = {"matched_targets": targets}
    try:
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                         target_modules=targets, task_type=TaskType.CAUSAL_LM)
        m = get_peft_model(model, cfg)
        n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
        RESULTS["lora_probe"]["attach"] = "OK"
        RESULTS["lora_probe"]["trainable_params_m"] = round(n_train/1e6, 2)
        RESULTS["lora_probe"]["vram_after_attach"] = vram()
        RESULTS["_peft_model"] = m
        print("LORA ATTACH OK:", RESULTS["lora_probe"], flush=True)
    except Exception as e:
        RESULTS["lora_probe"]["attach"] = "FAILED"
        RESULTS["lora_probe"]["error"] = f"{type(e).__name__}: {e}"
        RESULTS["lora_probe"]["traceback"] = traceback.format_exc()[-2000:]
        print("LORA ATTACH FAILED:", RESULTS["lora_probe"]["error"], flush=True)
try:
    phase("lora_probe", lora_probe)
except Exception:
    print("LORA PROBE PHASE FAILED", flush=True)
peft_model = RESULTS.pop("_peft_model", None)

# ---------------- one standard causal-LM training step attempt ----------------
def step_probe():
    text = ("The quick brown fox jumps over the lazy dog. " * 26).strip()
    ids = tok(text, return_tensors="pt").input_ids[:, :256].to("cuda")
    labels = ids.clone()
    # (a) base model with labels — the track's standard step
    try:
        out = model(input_ids=ids, labels=labels)
        RESULTS["step_probe"] = {"base_forward_with_labels": "OK (unexpected)",
                                 "loss": float(out.loss.item())}
    except Exception as e:
        RESULTS["step_probe"] = {"base_forward_with_labels": "FAILED",
            "error": f"{type(e).__name__}: {e}"}
    # (b) PEFT-wrapped model, if attach succeeded
    if peft_model is not None:
        try:
            peft_model.train()
            out = peft_model(input_ids=ids, labels=labels)
            RESULTS["step_probe"]["peft_forward_with_labels"] = "OK (unexpected)"
            RESULTS["step_probe"]["peft_loss"] = float(out.loss.item())
        except Exception as e:
            RESULTS["step_probe"]["peft_forward_with_labels"] = "FAILED"
            RESULTS["step_probe"]["peft_error"] = f"{type(e).__name__}: {e}"
    print("STEP PROBE:", json.dumps(RESULTS["step_probe"], default=str), flush=True)
try:
    phase("step_probe", step_probe)
except Exception:
    print("STEP PROBE PHASE FAILED", flush=True)

# ---------------- frozen-scorer / logprob probe ----------------
def logprob_probe():
    RESULTS["logprob_probe"] = {
        "generate_returns": "tuple (token_ids, nfe) only — no scores/logits",
        "has_lm_head": hasattr(model, "lm_head"),
        "has_get_output_embeddings": False,
        "forward_usable": False,
        "note": "diffusion_head maps hidden->vocab inside _denoise_block for "
                "the denoising readout; no public API returns token logprobs. "
                "A causal-LM frozen scorer needs per-token logprobs, which "
                "this model does not expose.",
    }
    try:
        oe = model.get_output_embeddings()
        RESULTS["logprob_probe"]["has_get_output_embeddings"] = oe is not None
        RESULTS["logprob_probe"]["output_embeddings"] = str(type(oe))
    except Exception as e:
        RESULTS["logprob_probe"]["get_output_embeddings_error"] = f"{type(e).__name__}: {e}"
    try:
        model(input_ids=torch.ones(1, 2, dtype=torch.long, device="cuda"))
        RESULTS["logprob_probe"]["forward_usable"] = True
    except Exception as e:
        RESULTS["logprob_probe"]["forward_error"] = f"{type(e).__name__}: {e}"
    print("LOGPROB PROBE:", json.dumps(RESULTS["logprob_probe"], default=str), flush=True)
try:
    phase("logprob_probe", logprob_probe)
except Exception:
    print("LOGPROB PROBE PHASE FAILED", flush=True)

# summary
total = sum(1 for r in RESULTS["checks"] if r["pass"])
RESULTS["capability_score"] = f"{total}/{len(RESULTS['checks'])}"
RESULTS["verdict"] = "PROBE-COMPLETE (see VERDICT.md for track verdicts)"
flush()
print(f"SIGMA PROBE COMPLETE — capability {RESULTS['capability_score']}", flush=True)
