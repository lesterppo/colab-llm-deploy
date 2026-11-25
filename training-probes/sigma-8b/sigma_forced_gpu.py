#!/usr/bin/env python3
"""Forced-GPU bf16 load test for Sigma-8B-instruct.

The device_map='auto' bf16 load "succeeded" only by offloading ~5 GB to CPU
(11.97 GB GPU vs 16.99 GB weights) and then generated degenerate output.
This test pins device_map={'': 0} (pure GPU, no offload) to measure whether
a pure-GPU bf16 load fits the 14.56 GiB T4 at all.
"""
import json, time, torch

MODEL = "nvidia/Nemotron-Labs-Sigma-8B-instruct"
OUT = "/content/sigma_8b_forced_gpu.json"
RES = {"model": MODEL, "test": "bf16 device_map={'': 0} (pure GPU, no offload)"}

def main():
    import transformers
    RES["transformers"] = transformers.__version__
    from transformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    cfg.schedule_parameterization = "projected_bernstein"
    cfg.schedule_degree = 32
    t0 = time.time()
    try:
        model = AutoModel.from_pretrained(
            MODEL, config=cfg, torch_dtype=torch.bfloat16,
            device_map={"": 0}, trust_remote_code=True).eval()
        RES["outcome"] = "LOADED (unexpected)"
        RES["secs"] = round(time.time()-t0, 1)
        RES["vram_gb"] = round(torch.cuda.memory_allocated()/2**30, 2)
    except Exception as e:
        import traceback
        RES["outcome"] = "FAILED"
        RES["secs"] = round(time.time()-t0, 1)
        RES["error"] = f"{type(e).__name__}: {e}"
        RES["traceback_tail"] = traceback.format_exc()[-1500:]
        RES["vram_at_failure_gb"] = round(torch.cuda.memory_allocated()/2**30, 2)
    print(json.dumps(RES, indent=2, default=str), flush=True)
    json.dump(RES, open(OUT, "w"), indent=2, default=str)
    print("FORCED-GPU TEST DONE", flush=True)

main()
