#!/usr/bin/env python3
"""Last principled fallback for Sigma-3B-instruct: transformers 4.55.x
(the card requires >=4.55; Colab ships 5.22.1). Same schedule fix +
README-pattern load (.cuda().to(bf16)) + README example verbatim.
If this is still garbage, the checkpoint/code pair is broken as shipped
and no transformers version is the explanation.
"""
import json, time, torch
import transformers
print("transformers:", transformers.__version__, flush=True)

MODEL = "nvidia/Nemotron-Labs-Sigma-3B-instruct"
OUT = "/content/sigma_tf455.json"
RES = {"model": MODEL, "transformers": transformers.__version__,
       "fix": "projected_bernstein + README-pattern load", "checks": []}

from transformers import AutoTokenizer, AutoModel, AutoConfig
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
cfg.schedule_parameterization = "projected_bernstein"
cfg.schedule_degree = 32
t0 = time.time()
model = AutoModel.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16,
                                  device_map="auto", trust_remote_code=True).eval()
RES["load_secs"] = round(time.time()-t0, 1)
RES["vram_gb"] = round(torch.cuda.memory_allocated()/2**30, 2)
RES["gamma_bounds"] = [float(x) for x in model.gamma_bounds()]
print(f"LOAD OK {RES['load_secs']}s VRAM {RES['vram_gb']} gamma {RES['gamma_bounds']}", flush=True)

def run(name, messages, max_new=128, thinking=False):
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
    txt = tok.decode(new, skip_special_tokens=True)
    RES["checks"].append({"name": name, "nfe": int(nfe), "secs": round(secs, 1),
                          "text": txt})
    print(f"--- {name}: {txt[:250]!r} (nfe={nfe} secs={secs:.0f})", flush=True)

run("readme_example_15pct240",
    [{"role": "system", "content": ""},
     {"role": "user", "content": "What is 15% of 240?"}], max_new=256)
run("smoke_2plus2",
    [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
    max_new=64)
json.dump(RES, open(OUT, "w"), indent=2, default=str)
print("TF455 TEST DONE", flush=True)
