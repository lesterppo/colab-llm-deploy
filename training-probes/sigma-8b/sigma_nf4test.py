#!/usr/bin/env python3
"""NF4 coherence test for Sigma-8B-instruct.

The bf16 path "loaded" only via CPU offload (device_map=auto: 11.97 GB GPU
+ ~5 GB CPU) and produced degenerate output ("real real real" + <unk>,
smoke FAIL) at 0.3 tok/s (1.4 s/NFE). This tests whether NF4 — fully
GPU-resident (~4.3 GB) — yields coherent text under the same working stack
(transformers 4.55.4 + schedule_parameterization=projected_bernstein).
Four short generations: README example, smoke, arith2, tool-JSON single.
"""
import json, time, torch

MODEL = "nvidia/Nemotron-Labs-Sigma-8B-instruct"
OUT = "/content/sigma_8b_nf4test.json"
RES = {"model": MODEL, "load": "nf4", "fix": "projected_bernstein",
       "transformers": None, "checks": []}

def main():
    import transformers
    RES["transformers"] = transformers.__version__
    from transformers import AutoTokenizer, AutoModel, AutoConfig, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    cfg.schedule_parameterization = "projected_bernstein"
    cfg.schedule_degree = 32
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    t0 = time.time()
    model = AutoModel.from_pretrained(MODEL, config=cfg, quantization_config=bnb,
                                      device_map="auto", torch_dtype=torch.bfloat16,
                                      trust_remote_code=True).eval()
    RES["load_secs"] = round(time.time()-t0, 1)
    RES["vram_after_load_gb"] = round(torch.cuda.memory_allocated()/2**30, 2)
    RES["gamma_bounds_loaded"] = [float(x) for x in model.gamma_bounds()]
    print(f"LOAD OK {RES['load_secs']}s VRAM {RES['vram_after_load_gb']} "
          f"gamma {RES['gamma_bounds_loaded']}", flush=True)

    def gen(messages, max_new=128, thinking=False):
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=thinking)
        ids = tok(text, return_tensors="pt").to("cuda")
        ptok = int(ids["input_ids"].shape[1])
        t0 = time.time()
        with torch.no_grad():
            out_ids, nfe = model.generate(
                ids["input_ids"], max_new_tokens=max_new, sampler="ddim",
                time_schedule="log_snr", steps=32, block_length=32,
                guidance_scale=3.0, score_temp=0.3)
        secs = time.time() - t0
        new = out_ids[0][ptok:].tolist()
        for i, t in enumerate(new):
            if t in (11, 2):
                new = new[:i]; break
        return {"text": tok.decode(new, skip_special_tokens=True),
                "raw_new_tokens": int(out_ids.shape[1]-ptok),
                "trunc_new_tokens": len(new), "secs": round(secs, 1),
                "nfe": int(nfe),
                "raw_tok_s": round(int(out_ids.shape[1]-ptok)/max(secs, 0.01), 1)}

    def chat(user, max_new=128, system=None):
        msgs = []
        if system: msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        return gen(msgs, max_new=max_new)

    r = gen([{"role": "system", "content": ""},
             {"role": "user", "content": "What is 15% of 240?"}], max_new=128)
    RES["checks"].append({"name": "readme_example_15pct240", **r})
    print("README-EXAMPLE:", repr(r["text"][:250]), f"(nfe={r['nfe']} secs={r['secs']})", flush=True)
    r = chat("What is 2+2? Answer with just the number.", max_new=64)
    RES["checks"].append({"name": "smoke_2plus2", **r})
    print("SMOKE:", repr(r["text"][:200]), flush=True)
    r = chat("What is 17 x 19? Give just the number.", max_new=128)
    RES["checks"].append({"name": "arith2_17x19", **r})
    print("ARITH2:", repr(r["text"][:200]), flush=True)
    tool_sys = ("You are a helpful assistant with access to functions. When the user asks for "
        "weather/time/stock, reply with ONLY a JSON object like "
        '{"function": "get_weather", "arguments": {"city": "Hong Kong", "unit": "celsius"}}. '
        "No other text.")
    r = chat("What is the weather in Hong Kong in celsius?", max_new=256, system=tool_sys)
    RES["checks"].append({"name": "tool_json_single", **r})
    print("TOOL:", repr(r["text"][:300]), flush=True)
    json.dump(RES, open(OUT, "w"), indent=2, default=str)
    print("NF4 TEST DONE", flush=True)

main()
