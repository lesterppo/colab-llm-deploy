#!/usr/bin/env python3
"""Diagnostic: load Sigma-3B-instruct with the schedule parameterization
corrected to the checkpoint's own lineage (projected_bernstein) and test
whether the README recipe then produces coherent text.

Evidence for the fix (measured 2026-11-25 on the VM):
  - config.json ships NO schedule_parameterization -> modeling code defaults
    to "legacy" -> instantiates NLCDNoiseSchedule (random W1/W2/b1) +
    NLCDGammaBounds (gamma_0=-3.0, gamma_1=6.0 defaults).
  - The checkpoint stores noise_schedule.increment_values (trained 32-dim
    Bernstein increments) + gamma_bounds.endpoint_values = [-5.415, 4.855],
    which from_pretrained then reports UNEXPECTED and drops.
  - generate() builds its gamma grid from gamma_bounds() only, so the
    as-shipped load denoises over the WRONG log-SNR range (6.0..-3.0
    instead of 4.85..-5.42) -> degenerate output (measured: smoke FAIL,
    reasoning 0/4 so far, repeated "the/seems/It" tokens).
The fix uses the modeling code's own documented flag; nothing else changes.
"""
import json, sys, time, torch

MODEL = "nvidia/Nemotron-Labs-Sigma-3B-instruct"
OUT = "/content/sigma_fix_test.json"
RES = {"model": MODEL, "fix": "schedule_parameterization=projected_bernstein",
       "checks": []}

def gen(model, tok, messages, max_new=128, **kw):
    d = dict(sampler="ddim", time_schedule="log_snr", steps=32,
             block_length=32, guidance_scale=3.0, score_temp=0.3)
    d.update(kw)
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)
    ids = tok(text, return_tensors="pt").to("cuda")
    ptok = int(ids["input_ids"].shape[1])
    t0 = time.time()
    with torch.no_grad():
        out_ids, nfe = model.generate(ids["input_ids"], max_new_tokens=max_new, **d)
    secs = time.time() - t0
    new = out_ids[0][ptok:].tolist()
    for i, t in enumerate(new):
        if t in (11, 2):
            new = new[:i]; break
    txt = tok.decode(new, skip_special_tokens=True)
    return {"text": txt, "prompt_tokens": ptok, "raw_new_tokens": int(out_ids.shape[1]-ptok),
            "trunc_new_tokens": len(new), "secs": round(secs, 1), "nfe": int(nfe),
            "raw_tok_s": round(int(out_ids.shape[1]-ptok)/max(secs, 0.01), 1)}

def main():
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    cfg.schedule_parameterization = "projected_bernstein"
    cfg.schedule_degree = 32
    t0 = time.time()
    model = AutoModel.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16,
                                      device_map="auto", trust_remote_code=True).eval()
    RES["load_secs"] = round(time.time()-t0, 1)
    RES["vram_after_load_gb"] = round(torch.cuda.memory_allocated()/2**30, 2)
    RES["gamma_bounds_loaded"] = [float(x) for x in model.gamma_bounds()]
    RES["noise_schedule_increments_sum"] = float(
        model.noise_schedule.increments().sum().item())
    RES["schedule_class"] = type(model.noise_schedule).__name__
    print("LOAD OK", RES["load_secs"], "s VRAM", RES["vram_after_load_gb"],
          "gamma_bounds", RES["gamma_bounds_loaded"], flush=True)

    def chat(user, max_new=128, system=None):
        msgs = []
        if system: msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        return gen(model, tok, msgs, max_new=max_new)

    # README example verbatim (system "" + "What is 15% of 240?")
    r = gen(model, tok, [{"role": "system", "content": ""},
                         {"role": "user", "content": "What is 15% of 240?"}],
            max_new=256)
    RES["checks"].append({"name": "readme_example_15pct240", **r})
    print("README-EXAMPLE:", r["text"][:300], "| nfe", r["nfe"], "secs", r["secs"], flush=True)
    r = chat("What is 2+2? Answer with just the number.", max_new=64)
    RES["checks"].append({"name": "smoke_2plus2", **r})
    print("SMOKE:", repr(r["text"][:200]), flush=True)
    r = chat("What is 17 x 19? Give just the number.", max_new=128)
    RES["checks"].append({"name": "arith2_17x19", **r})
    print("ARITH2:", repr(r["text"][:200]), flush=True)
    r = chat("How many 3-digit numbers have digits that sum to 10? Give just the number.", max_new=128)
    RES["checks"].append({"name": "combinatorics", **r})
    print("COMB:", repr(r["text"][:200]), flush=True)
    tool_sys = ("You are a helpful assistant with access to functions. When the user asks for "
        "weather/time/stock, reply with ONLY a JSON object like "
        '{"function": "get_weather", "arguments": {"city": "Hong Kong", "unit": "celsius"}}. '
        "No other text.")
    r = chat("What is the weather in Hong Kong in celsius?", max_new=256, system=tool_sys)
    RES["checks"].append({"name": "tool_json_single", **r})
    print("TOOL:", repr(r["text"][:300]), flush=True)
    json.dump(RES, open(OUT, "w"), indent=2, default=str)
    print("FIX TEST DONE", flush=True)

main()
