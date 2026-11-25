#!/usr/bin/env python3
"""SenseNova capability top-up (retry session, weights cached): load NF4 from
cache, run the capability checks attempt 1 never reached: D instruction (2),
F Chinese (3), G consistency (1), C remaining reasoning (4 @256 tokens).
Writes /content/sensenova_cap2.json."""
import json, re, time
import torch

OUT = "/content/sensenova_cap2.json"
R = {"model": "sensenova/SenseNova-Loop-9B", "checks": []}

def flush():
    json.dump(R, open(OUT, "w"), indent=2, default=str)

def strip_think(s):
    return re.sub(r"<think>.*?</think>", "", s or "", flags=re.S).strip()

def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def check(axis, name, passed, detail):
    R["checks"].append({"axis": axis, "name": name, "pass": bool(passed),
                        "detail": str(detail)[:400]})
    print(f"{'PASS' if passed else 'FAIL'} [{axis}] {name}: {str(detail)[:150]}", flush=True)
    flush()

from transformers import AutoTokenizer
import transformers
tok = AutoTokenizer.from_pretrained("sensenova/SenseNova-Loop-9B", trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
from transformers import BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
t0 = time.time()
model = transformers.AutoModelForImageTextToText.from_pretrained(
    "sensenova/SenseNova-Loop-9B", quantization_config=bnb, device_map="auto",
    torch_dtype=torch.bfloat16, trust_remote_code=True)
R["load_secs"] = round(time.time()-t0, 1)
R["vram_after_load_gb"] = round(torch.cuda.memory_allocated()/2**30, 2)
print("LOAD OK:", R["load_secs"], "s", R["vram_after_load_gb"], "GB", flush=True)
flush()

def gen(messages, max_new=256):
    try:
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to("cuda")
    ptok = int(ids["input_ids"].shape[1])
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    secs = time.time() - t0
    new_ids = out[0][ptok:]
    return tok.decode(new_ids, skip_special_tokens=True), len(new_ids), secs

def chat_gen(user_text, max_new=256, system=None):
    msgs = []
    if system: msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_text})
    return gen(msgs, max_new=max_new)

# C remaining reasoning (4 @256)
for name, q, expect in [
    ("combinatorics", "How many 3-digit numbers have digits that sum to 10? Give just the number.", "54"),
    ("wordprob", "Maria has twice as many apples as Tom. Together they have 36 apples. How many does Maria have? Give just the number.", "24"),
    ("seq", "What is the next number in the sequence 2, 6, 12, 20, 30? Give just the number.", "42"),
    ("age", "A father is 3 times as old as his son. In 12 years he will be twice as old. How old is the son now? Give just the number.", "12"),
]:
    txt, nt, secs = chat_gen(q, max_new=256)
    t = strip_think(txt); tn = norm(t)
    nums = re.findall(r"\d+", tn)
    passed = bool(nums) and nums[-1] == expect
    check("C", name, passed, f"expect={expect} last_num={nums[-1] if nums else None} ntok={nt} {secs:.0f}s | {t[:120]}")
# D instruction (2)
txt, _, _ = chat_gen('Return ONLY this JSON object, no other text: {"name": "Ada", "age": 36, "city": "London"}.', max_new=256)
try:
    ok = json.loads(strip_think(txt).strip()) == {"name": "Ada", "age": 36, "city": "London"}
except Exception:
    ok = False
check("D", "strict_json", ok, strip_think(txt)[:200])
txt, _, _ = chat_gen("List exactly 3 benefits of drinking water, as exactly 3 bullet points, one per line, no other text.", max_new=256)
lines = [l for l in strip_think(txt).strip().splitlines() if l.strip()]
bullets = [l for l in lines if l.strip().startswith(("-", "*", "•")) or re.match(r"\d+[.)]", l.strip())]
check("D", "exact_3_bullets", len(bullets) == 3 and len(lines) == 3,
      f"lines={len(lines)} bullets={len(bullets)} | {strip_think(txt)[:120]}")
# F Chinese (3)
txt, _, _ = chat_gen("用中文回答：香港的首都是哪里？请用一句话回答。", max_new=256)
check("F", "zh_qa", "香港" in (txt or ""), strip_think(txt)[:150])
txt, _, _ = chat_gen("Translate to English: 人工智能正在改变医疗行业。", max_new=256)
check("F", "zh_to_en", "artificial intelligence" in norm(strip_think(txt)) and any(w in norm(strip_think(txt)) for w in ["medical", "healthcare", "medicine"]), strip_think(txt)[:150])
txt, _, _ = chat_gen("Translate to Chinese: The weather in Tokyo is sunny today.", max_new=256)
check("F", "en_to_zh", "东京" in (txt or "") and "晴" in (txt or ""), strip_think(txt)[:150])
# G consistency
q = "What is the capital of Australia? Answer with just the city name."
t1, _, _ = chat_gen(q, max_new=64)
t2, _, _ = chat_gen(q, max_new=64)
check("G", "temp0_consistency", norm(strip_think(t1)) == norm(strip_think(t2)) and "canberra" in norm(strip_think(t1)),
      f"run1={strip_think(t1)!r} run2={strip_think(t2)!r}")
summary = {}
for r in R["checks"]:
    summary.setdefault(r["axis"], [0, 0])
    summary[r["axis"]][1] += 1
    if r["pass"]: summary[r["axis"]][0] += 1
R["summary"] = summary
print("TOP-UP SUMMARY:", json.dumps(summary), flush=True)
flush()
print("CAP TOP-UP DONE", flush=True)
