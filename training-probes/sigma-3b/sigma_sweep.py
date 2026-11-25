#!/usr/bin/env python3
"""Sampler/prompt sweep for Sigma-3B-instruct (after the schedule fix did
NOT restore coherence). One load, five short generations on 'What is 2+2?':
  1. thinking=True (template default; model may need a think trace)
  2. guidance off + score_temp 1.0 (no guidance/temp shaping)
  3. DDIM-64 (more denoising steps)
  4. DDPM-32 / edm_sigma (the generate() defaults)
  5. plain-text prompt, no chat template
All with the projected_bernstein schedule fix applied.
"""
import json, time, torch

MODEL = "nvidia/Nemotron-Labs-Sigma-3B-instruct"
OUT = "/content/sigma_sweep.json"
RES = {"model": MODEL, "fix": "projected_bernstein", "variants": []}

def main():
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    cfg.schedule_parameterization = "projected_bernstein"
    cfg.schedule_degree = 32
    t0 = time.time()
    model = AutoModel.from_pretrained(MODEL, config=cfg, torch_dtype=torch.bfloat16,
                                      device_map="auto", trust_remote_code=True).eval()
    print(f"LOAD OK {time.time()-t0:.1f}s gamma={model.gamma_bounds()}", flush=True)

    def run(name, prompt_text=None, messages=None, max_new=64, thinking=False, **kw):
        d = dict(sampler="ddim", time_schedule="log_snr", steps=32,
                 block_length=32, guidance_scale=3.0, score_temp=0.3)
        d.update(kw)
        if messages is not None:
            text = tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=thinking)
        else:
            text = prompt_text
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
        r = {"name": name, "settings": d, "thinking": thinking,
             "prompt_tokens": ptok, "secs": round(secs, 1), "nfe": int(nfe),
             "text": txt}
        RES["variants"].append(r)
        print(f"--- {name}: {txt[:200]!r} (nfe={nfe} secs={secs:.0f})", flush=True)

    q = "What is 2+2? Answer with just the number."
    msgs = [{"role": "user", "content": q}]
    run("1_thinking_true", messages=msgs, max_new=256, thinking=True)
    run("2_no_guidance_no_temp", messages=msgs, guidance_scale=1.0, score_temp=1.0)
    run("3_ddim64", messages=msgs, steps=64, score_temp=0.2)
    run("4_ddpm_edm", messages=msgs, sampler="ddpm", time_schedule="edm_sigma")
    run("5_plain_text", prompt_text=q + " ")
    json.dump(RES, open(OUT, "w"), indent=2, default=str)
    print("SWEEP DONE", flush=True)

main()
