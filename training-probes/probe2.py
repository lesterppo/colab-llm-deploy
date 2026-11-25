#!/usr/bin/env python3
"""Generic small-model probe for the arXiv digest training track.

Phases (results JSON flushed after each):
  env -> load (NF4) -> capability battery (generate, temp 0) -> lora -> step
Usage: python3 probe.py --model <HF id> --slug <slug> --out /content/<slug>_results.json
       [--trust-remote-code] [--multimodal]
Writes JSON incrementally; exits non-zero on failed phase (results file keeps
the failure record). Capability battery is a probe, not the full deep test:
reasoning 8, tool-JSON proxy 3, instruction 2, needle 4k/8k, ZH 3,
temp-0 consistency, throughput.
"""
import argparse, json, os, re, sys, time, traceback
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--slug", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--trust-remote-code", action="store_true")
ap.add_argument("--multimodal", action="store_true")
ap.add_argument("--extra-lora-targets", default="",
                help="comma-separated extra LoRA target suffixes, tried if present in the model")
a = ap.parse_args()

RESULTS = {"model": a.model, "slug": a.slug, "phases": {}, "checks": []}

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

# ---------------- load ----------------
def load():
    from huggingface_hub import hf_hub_download
    cfg_path = hf_hub_download(a.model, "config.json")
    cfg = json.load(open(cfg_path))
    RESULTS["model_config"] = {"model_type": cfg.get("model_type"),
        "architectures": cfg.get("architectures"),
        "torch_dtype": cfg.get("torch_dtype") or cfg.get("dtype"),
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "text_config_layers": (cfg.get("text_config") or {}).get("num_hidden_layers"),
        "native_quantization_config": cfg.get("quantization_config")}
    print("MODEL CONFIG:", RESULTS["model_config"], flush=True); flush()
    from transformers import AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    RESULTS["tokenizer"] = {"vocab_size": tok.vocab_size, "eos": tok.eos_token,
                            "pad": tok.pad_token}
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    import transformers
    cls_names = []
    if a.multimodal:
        cls_names = ["AutoModelForImageTextToText", "AutoModelForCausalLM"]
    else:
        cls_names = ["AutoModelForCausalLM"]
    last_err = None
    model = None
    for cls_name in cls_names:
        cls = getattr(transformers, cls_name)
        for dm, dm_label in [("auto", "auto"), ({"": 0}, "forced-gpu")]:
            try:
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time(); ram0 = ram()
                model = cls.from_pretrained(a.model, quantization_config=bnb,
                    device_map=dm, torch_dtype=torch.bfloat16,
                    trust_remote_code=a.trust_remote_code)
                RESULTS["load"] = {"secs": round(time.time()-t0,1),
                    "class": cls_name, "device_map": dm_label,
                    "vram_after_load": vram(), "ram_before_load": ram0,
                    "ram_after_load": ram(),
                    "total_params_b": round(sum(p.numel() for p in model.parameters())/1e9, 2)}
                print("LOAD OK:", RESULTS["load"], flush=True)
                try: model.config.use_cache = False
                except Exception: pass
                # ---- module inventory: which layer types exist, and where LoRA can attach ----
                inv = {}
                for n, m in model.named_modules():
                    suf = n.split(".")[-1]
                    cls = m.__class__.__name__
                    key = f"{suf}:{cls}"
                    inv[key] = inv.get(key, 0) + 1
                RESULTS["module_inventory"] = dict(sorted(inv.items()))
                lin = sum(v for k, v in inv.items() if k.endswith(":Linear") or ":Linear4bit" in k or k.endswith(":Linear8bitLt"))
                conv = sum(v for k, v in inv.items() if "Conv1d" in k or "Conv2d" in k)
                RESULTS["module_counts"] = {"linear_modules": lin, "conv_modules": conv,
                    "total_modules": sum(inv.values())}
                try:
                    RESULTS["model_config_full"] = {k: str(v)[:300] for k, v in vars(model.config).items()}
                except Exception as e:
                    RESULTS["model_config_full"] = {"error": str(e)}
                print("MODULE INVENTORY:", RESULTS["module_counts"], flush=True)
                print("MODULE SUFFIXES:", RESULTS["module_inventory"], flush=True)
                RESULTS["_model"] = model; RESULTS["_tok"] = tok
                return
            except Exception as e:
                last_err = f"{cls_name}/{dm_label}: {type(e).__name__}: {e}"
                print("LOAD ATTEMPT FAILED:", last_err, flush=True)
                RESULTS.setdefault("load_attempts", []).append(last_err); flush()
    raise RuntimeError(f"all load attempts failed; last: {last_err}")

try:
    phase("load", load)
except Exception:
    print("LOAD FAILED", flush=True); sys.exit(2)
model = RESULTS.pop("_model"); tok = RESULTS.pop("_tok")
RESULTS["vram_after_load_gb"] = vram()

# ---------------- generate helper ----------------
def gen(messages, max_new=512, tools=None):
    """messages: list of {role, content} dicts. Returns (text, prompt_tokens, new_tokens, secs, tok_s)."""
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # fall back to plain concatenation
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
    ids = tok(text, return_tensors="pt").to("cuda")
    ptok = int(ids["input_ids"].shape[1])
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new,
                             do_sample=False, pad_token_id=tok.pad_token_id)
    secs = time.time() - t0
    new_ids = out[0][ptok:]
    txt = tok.decode(new_ids, skip_special_tokens=True)
    return txt, ptok, int(new_ids.shape[0]), secs, (len(new_ids)/max(secs, 0.01))

def chat_gen(user_text, max_new=512, system=None):
    msgs = []
    if system: msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_text})
    return gen(msgs, max_new=max_new)

# ---------------- capability battery ----------------
def capability():
    # A. throughput + smoke
    txt, pt, nt, secs, tps = chat_gen("What is 2+2? Answer with just the number.", max_new=64)
    check("A", "smoke_2plus2", "4" in strip_think(txt), txt[:150])
    txt2, pt2, nt2, secs2, tps2 = chat_gen(
        "Write a 150-word paragraph about why the sky is blue.", max_new=200)
    check("A", "throughput", nt2 > 20 and tps2 > 0,
          f"new_tokens={nt2} wall={secs2:.1f}s decode~{tps2:.1f} tok/s")
    RESULTS["throughput"] = {"decode_tok_s": round(tps2,1), "ttft_note": "generate-based, no stream TTFT"}
    # B. tool-JSON proxy (prompt-based, not native tool_calls)
    tool_sys = ("You are a helpful assistant with access to functions. When the user asks for "
        "weather/time/stock, reply with ONLY a JSON object like "
        '{"function": "get_weather", "arguments": {"city": "Hong Kong", "unit": "celsius"}}. '
        "No other text.")
    txt, _, _, _, _ = chat_gen("What is the weather in Hong Kong in celsius?",
                               max_new=256, system=tool_sys)
    ok = False; detail = txt[:200]
    try:
        j = json.loads(strip_think(txt).strip().strip("`"))
        ok = j.get("function") == "get_weather" and "Hong Kong" in str(j.get("arguments", {}))
        detail = str(j)
    except Exception: pass
    check("B", "tool_json_single", ok, detail)
    txt, _, _, _, _ = chat_gen("Get the weather in Hong Kong (celsius), the time in Asia/Tokyo, and the stock price of AAPL. Reply with ONLY a JSON array of function calls, no other text.",
                               max_new=512, system=tool_sys)
    ok = False
    try:
        j = json.loads(strip_think(txt).strip().strip("`"))
        names = {c.get("function") for c in j} if isinstance(j, list) else set()
        ok = names == {"get_weather", "get_time", "get_stock_price"}
        detail = str(j)[:300]
    except Exception: detail = txt[:200]
    check("B", "tool_json_parallel3", ok, detail)
    txt, _, _, _, _ = chat_gen("Tell me a one-sentence joke about cats.", max_new=256, system=tool_sys)
    check("B", "tool_json_no_tool", "function" not in strip_think(txt).lower() or "get_" not in txt,
          txt[:150])
    # C. reasoning (8)
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
        txt, _, _, _, _ = chat_gen(q, max_new=1024)
        t = strip_think(txt); tn = norm(t)
        if expect.isdigit():
            nums = re.findall(r"\d+", tn)
            passed = bool(nums) and nums[-1] == expect
            detail = f"expect={expect} last_num={nums[-1] if nums else None} | {t[:150]}"
        else:
            passed = expect.lower() in tn
            detail = f"expect={expect} | {t[:150]}"
        check("C", name, passed, detail)
    # D. instruction following (2)
    txt, _, _, _, _ = chat_gen('Return ONLY this JSON object, no other text: {"name": "Ada", "age": 36, "city": "London"}.', max_new=256)
    try:
        ok = json.loads(strip_think(txt).strip()) == {"name": "Ada", "age": 36, "city": "London"}
    except Exception: ok = False
    check("D", "strict_json", ok, strip_think(txt)[:200])
    txt, _, _, _, _ = chat_gen("List exactly 3 benefits of drinking water, as exactly 3 bullet points, one per line, no other text.", max_new=256)
    lines = [l for l in strip_think(txt).strip().splitlines() if l.strip()]
    bullets = [l for l in lines if l.strip().startswith(("-", "*", "•")) or re.match(r"\d+[.)]", l.strip())]
    check("D", "exact_3_bullets", len(bullets) == 3 and len(lines) == 3,
          f"lines={len(lines)} bullets={len(bullets)} | {strip_think(txt)[:150]}")
    # E. needle 4k / 8k (token-counted via tokenizer)
    filler = "The history of the Roman Empire spans many centuries of trade, war, and law. "
    def needle(target_tok, code):
        n_chars = target_tok * 4
        hay = (filler * (n_chars // len(filler) + 1))[:n_chars]
        mid = len(hay)//2
        doc = hay[:mid] + f" The secret code is {code}. " + hay[mid:]
        q = f"Read the following text and answer the question.\n\n{doc}\n\nQuestion: What is the secret code? Answer with just the requested value."
        txt, ptok, _, _, _ = chat_gen(q, max_new=128)
        return txt, ptok
    txt, ptok = needle(4000, "BLUE-FALCON-7")
    check("E", "needle_4k", "BLUE-FALCON-7" in (txt or ""), f"prompt_tokens={ptok} | {strip_think(txt)[:120]}")
    txt, ptok = needle(8000, "RED-TIGER-42")
    check("E", "needle_8k", "RED-TIGER-42" in (txt or ""), f"prompt_tokens={ptok} | {strip_think(txt)[:120]}")
    # F. Chinese (3)
    txt, _, _, _, _ = chat_gen("用中文回答：香港的首都是哪里？请用一句话回答。", max_new=256)
    check("F", "zh_qa", "香港" in (txt or ""), strip_think(txt)[:150])
    txt, _, _, _, _ = chat_gen("Translate to English: 人工智能正在改变医疗行业。", max_new=256)
    check("F", "zh_to_en", "artificial intelligence" in norm(strip_think(txt)) and any(w in norm(strip_think(txt)) for w in ["medical", "healthcare", "medicine"]), strip_think(txt)[:150])
    txt, _, _, _, _ = chat_gen("Translate to Chinese: The weather in Tokyo is sunny today.", max_new=256)
    check("F", "en_to_zh", "东京" in (txt or "") and "晴" in (txt or ""), strip_think(txt)[:150])
    # G. temp-0 consistency
    q = "What is the capital of Australia? Answer with just the city name."
    t1, _, _, _, _ = chat_gen(q, max_new=64)
    t2, _, _, _, _ = chat_gen(q, max_new=64)
    check("G", "temp0_consistency", norm(strip_think(t1)) == norm(strip_think(t2)) and "canberra" in norm(strip_think(t1)),
          f"run1={strip_think(t1)!r} run2={strip_think(t2)!r}")
    RESULTS["vram_after_capability"] = vram()
    # summary
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
    print("CAPABILITY FAILED — continuing to LoRA/step if model usable", flush=True)
    RESULTS["capability_failed"] = True; flush()

# ---------------- LoRA ----------------
def lora():
    from peft import LoraConfig, get_peft_model, TaskType
    all_names = [n for n, _ in model.named_modules()]
    base = [c for c in ["q_proj", "k_proj", "v_proj", "o_proj"]
            if any(c in n for n in all_names)]
    extras = [e.strip() for e in (a.extra_lora_targets or "").split(",") if e.strip()]
    extras = [e for e in extras if e not in base and any(e in n for n in all_names)]
    RESULTS["lora"] = {"base_targets": base, "extra_targets_requested": extras,
                       "extra_targets_present": extras}
    # Stage 1: base + extras (answers e.g. whether conv/DeltaNet layers take LoRA).
    # Stage 2 (fallback): base only, so a weird extra cannot block the step measurement.
    last_err = None
    for stage, targets in [(1, base + extras), (2, base)]:
        if stage == 2 and not extras:
            break
        try:
            cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                             target_modules=targets, task_type=TaskType.CAUSAL_LM)
            m = get_peft_model(model, cfg)
            n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
            # which module classes actually hold trainable params?
            trainable_by_class = {}
            for n, p in m.named_parameters():
                if p.requires_grad:
                    mod = n.split(".")[-2] if "." in n else n
                    trainable_by_class[mod] = trainable_by_class.get(mod, 0) + 1
            RESULTS["lora"].update({"stage_succeeded": stage, "targets_used": targets,
                "trainable_params_m": round(n_train/1e6, 2),
                "trainable_param_names_by_suffix": trainable_by_class,
                "vram_after_lora": vram()})
            RESULTS["_peft_model"] = m
            print(f"LORA OK stage {stage}:", RESULTS["lora"], flush=True)
            return
        except Exception as e:
            last_err = f"stage {stage} {targets}: {type(e).__name__}: {e}"
            print("LORA ATTEMPT FAILED:", last_err, flush=True)
            RESULTS["lora"].setdefault("attempts_failed", []).append(last_err); flush()
    raise RuntimeError(f"all LoRA attempts failed; last: {last_err}")
try:
    phase("lora", lora)
except Exception:
    print("LORA FAILED", flush=True); sys.exit(3)
peft_model = RESULTS.pop("_peft_model")

# ---------------- one training step ----------------
def step():
    peft_model.gradient_checkpointing_enable()
    peft_model.enable_input_require_grads()
    # SenseNova lesson (2026-11-25): custom-code models may read use_cache from
    # nested configs, so the top-level model.config.use_cache=False never reaches
    # them and the step dies with "Disable use_cache when training with gradient
    # checkpointing". Set it everywhere AND pass it explicitly in the forward call.
    try:
        peft_model.config.use_cache = False
    except Exception:
        pass
    for _m in peft_model.modules():
        try:
            _c = getattr(_m, "config", None)
            if _c is not None and hasattr(_c, "use_cache"):
                _c.use_cache = False
        except Exception:
            pass
    peft_model.train()
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW8bit(
            (p for p in peft_model.parameters() if p.requires_grad), lr=2e-4)
        RESULTS["optimizer"] = "paged_adamw_8bit"
    except Exception as e:
        opt = torch.optim.AdamW(
            (p for p in peft_model.parameters() if p.requires_grad), lr=2e-4, fused=True)
        RESULTS["optimizer"] = f"adamw_torch_fused (paged failed: {e})"
    text = ("The quick brown fox jumps over the lazy dog. " * 12).strip()
    ids = tok(text, return_tensors="pt").input_ids[:, :256].to("cuda")
    labels = ids.clone()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = peft_model(input_ids=ids, labels=labels, use_cache=False)
    loss = out.loss
    loss.backward()
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    total_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).item())
    # grad norm per LoRA target suffix + whether any conv module trains
    grad_by_suffix = {}
    for n, p in peft_model.named_parameters():
        if p.requires_grad and p.grad is not None:
            for suf in (RESULTS["lora"].get("targets_used") or []):
                if n.endswith(suf + ".lora_A.default.weight") or n.endswith(suf + ".lora_B.default.weight") or f".{suf}." in n:
                    if suf not in grad_by_suffix:
                        grad_by_suffix[suf] = {"param": n, "grad_norm": round(float(p.grad.norm().item()), 6)}
                    break
    conv_trainable = []
    for n, p in peft_model.named_parameters():
        if p.requires_grad and ("conv" in n.lower()):
            conv_trainable.append({"param": n,
                "grad_norm": round(float(p.grad.norm().item()), 6) if p.grad is not None else None})
    gn = None
    for n, p in peft_model.named_parameters():
        if p.requires_grad and p.grad is not None:
            gn = float(p.grad.norm().item()); RESULTS["grad_check_param"] = n; break
    RESULTS["step_grad_by_suffix"] = grad_by_suffix
    RESULTS["step_conv_trainable"] = conv_trainable[:10]
    RESULTS["step_conv_trainable_count"] = len(conv_trainable)
    opt.step(); opt.zero_grad()
    RESULTS["step"] = {"secs": round(time.time()-t0, 1), "loss": float(loss.item()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "grad_norm_first_lora_param": gn,
        "total_grad_norm_clipped": total_norm,
        "grads_flow": gn is not None and gn > 0,
        "peak_vram": vram(), "ram_after_step": ram(),
        "batch": 1, "seq": int(ids.shape[1])}
    print("STEP OK:", RESULTS["step"], flush=True)
try:
    phase("step", step)
except Exception:
    print("STEP FAILED", flush=True); sys.exit(4)

RESULTS["verdict"] = "GREEN"
flush()
print("PROBE COMPLETE — GREEN", flush=True)
