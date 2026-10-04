# Image generation on Colab

One-command text-to-image on free Colab GPUs (T4, 15GB VRAM) via ComfyUI +
ComfyUI-GGUF, with quantized (GGUF Q4_K_M) transformers. All models fit
comfortably; the encoders are the real constraint — see each model's notes.

## One command

```bash
# From the repo root:
python3 deploy_media.py generate --kind image --model flux2-klein \
    --prompt "A serene mountain lake at sunrise, photorealistic" \
    --out ./outputs

python3 deploy_media.py status   --session media-flux2-klein
python3 deploy_media.py undeploy --session media-flux2-klein
```

`deploy_media.py` (repo root) does everything: creates the session, uploads
the setup + gen scripts, runs setup detached (pip + model downloads via the
`hf` CLI), polls the stage-gated status file, runs generation detached,
polls again, downloads the PNG, and stops the session. No manual steps.

## Models

### flux2-klein — FLUX.2 [klein] 4B (Apache 2.0)
- Files: `flux2-klein/setup.py`, `gen.py`, `server.py`, `merge_qwen3.py`
- Stack: ComfyUI 0.38.0 + ComfyUI-GGUF. Transformer `unsloth/FLUX.2-klein-4B-GGUF`
  Q4_K_M (2.5GB) + **Qwen3-4B text encoder** (merged from the diffusers-format
  shards via `merge_qwen3.py`, ~7.6GB) + flux2 VAE. CLIPLoader type `flux2`.
- Proven: 1024x1024, 4 steps, cfg 4.0 in ~70s on T4 (2026-10-04).
- Pitfall: full-precision diffusers + CPU offload OOM-dies on Colab's 12GB
  system RAM — GGUF is mandatory. ComfyUI must be >= 0.38 for the `flux2`
  CLIP type.

### hidream-i1 — HiDream-I1 17B (custom license)
- Files: `hidream-i1/setup.py`, `gen.py`, `server.py`
- Stack: ComfyUI 0.38.0 + ComfyUI-GGUF. DiT `city96/HiDream-I1-Full-gguf`
  Q4_K_M (11.48GB) on GPU; 4 text encoders (clip_l+clip_g fp16, t5xxl+llama
  fp8, ~15.9GB) offloaded via `--lowvram`. QuadrupleCLIPLoader +
  CLIPTextEncodeHiDream, ModelSamplingSD3 shift 3.0.
- Proven: 1024x1024, 24 steps, cfg 5.0 on T4 — tight but workable (2026-10-05).
  Took ~2.5h end to end (27GB of downloads) on the first run.

## For agents

Each model dir is self-contained: `setup.py` (VM-side, stage-gated, writes
`/content/<name>_setup_status.json`) and `gen.py` (VM-side, reads
`/content/gen_config.json`, writes `/content/gen_status.json`). You can
also drive them manually with hermes-colab-cli (`new` → `upload` →
`exec_detach` → poll status), following `deploy_media.py` as the reference.
