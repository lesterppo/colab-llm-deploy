# Video generation on Colab

One-command text-to-video on free Colab GPUs (T4, 15GB VRAM) via ComfyUI +
ComfyUI-GGUF, with a quantized (GGUF Q3_K_M) 22B transformer. The largest
model proven working on a free T4 in this repo.

## One command

```bash
# From the repo root — single 2s clip:
python3 deploy_media.py generate --kind video --model ltx23 \
    --prompt "A lighthouse on rugged cliffs at golden hour, cinematic" \
    --out ./outputs

# Chained ~10s video (5 clips, first-frame conditioning):
python3 deploy_media.py generate --kind video --model ltx23-chain \
    --prompt "A lighthouse on rugged cliffs at golden hour, cinematic" \
    --clips 5 --out ./outputs

python3 deploy_media.py status   --session media-ltx23
python3 deploy_media.py undeploy --session media-ltx23
```

`deploy_media.py` (repo root) does everything: session → upload → detached
setup (pip + ~20.8GB of model downloads via the `hf` CLI) → poll → detached
generation → poll → download MP4 → stop session.

## Model: minimax-h3 — MiniMax H3 33B 4-step INT8 (GREEN 2026-11-11)

- Files: `minimax-h3/setup.py`, `gen.py`
- Stack: ComfyUI 0.45.0 (H3 nodes are core since 0.30), no custom nodes
  for the T2V path, flags `--lowvram --cache-none --fast-disk`.
- Weights (Comfy-Org/MiniMax-H3, 42.04GB total): FL2VA pruned INT8
  diffusion 20.97GB + Qwen3-VL-32B NVFP4 encoder 15.69GB + 4-step turbo
  LoRA 1.96GB + int8 video VAE 2.81GB + fp32 audio VAE 0.61GB.
- Recipe: `er_sde` + `sgm_uniform`, 4 steps, Sigma Shift video 6 / audio 3.
- Proven (2026-11-11, free T4, kyu009009, session `minimax-h3-4step`,
  ~1h33m, not reclaimed): 640x960, 124 frames @24fps (~5.2s) in 1,979s,
  peak VRAM 14,619/15,360 MB. MP4 verified on this VM (5.17s, AAC 32kHz
  stereo, 6,017,812 bytes). Honest limits: ~700MB VRAM headroom; 1MP /
  longer clips untested; int4 fallback staged but untested.
- Pitfall (fixed in `gen.py`): ComfyUI 0.45 `SaveVideo` needs
  `format: "auto", codec: "auto"` — `video/h264-mp4` passes validation
  then writes no file (cost one full 32-min run).
- License: MiniMax H3 Community License — excludes US, EU, UK, South Korea.

```bash
# From the repo root — 5s, 640x960, with native audio:
python3 deploy_media.py generate --kind video --model minimax-h3 \
    --prompt "A lighthouse on rugged cliffs at golden hour, waves crashing below, cinematic, with the sound of waves and wind" \
    --out ./outputs
```

## Model: ltx23 — LTX-2.3 22B distilled (verify license for non-test use)

- Files: `ltx23/setup.py`, `gen.py` (single clip), `gen_chain.py` (chained)
- Stack: ComfyUI 0.38.0 + ComfyUI-GGUF + KJNodes + VideoHelperSuite +
  ComfyUI-LTXVideo, flags `--lowvram --cache-none`.
- Weights: DiT `unsloth/LTX-2.3-GGUF` `distilled-1.1/...Q3_K_M.gguf` (10.63GB)
  + Gemma-3-12B Q3_K_M encoder (6GB, **Q3 GGUF mandatory — fp8 OOMs**) +
  text projection (2.3GB) + video/audio VAEs.
- Recipe: LTXVChunkFeedForward (chunks=2), euler_cfg_pp + LTXVScheduler
  (8 steps), tiled VAE decode (`last_frame_fix`), joint AV latent, crf 16.
- Proven (2026-10-05, T4): 576x320, 49 frames @24fps in ~430s, peak VRAM
  ~12.4GB. Chain: 5 clips = ~10.2s in ~38 min; clips pinned to the previous
  clip's last frame via `LTXVImgToVideoConditionOnly`, concatenated with
  `ffmpeg -c copy`.
- Pitfalls (all fixed in the scripts):
  1. The `hf` CLI preserves repo subpaths on download — `setup.py` flattens
     `unet/distilled-1.1/*.gguf` → `unet/` and `text_encoders/gguf/*.gguf` →
     `text_encoders/` automatically.
  2. `LTXVAudioVAELoader` scans `models/checkpoints/`, NOT `models/vae/` —
     `setup.py` copies the audio VAE there (first run 400'd without it).

## For agents

`setup.py` is stage-gated (`/content/ltx23_setup_status.json`); `gen.py`
writes `/content/ltx23_gen_status.json`, `gen_chain.py` writes
`/content/ltx23_chain_status.json`. Both read `/content/gen_config.json`
(prompt/seed/size/frames/fps/steps/clips/out_name). Drive them manually
with hermes-colab-cli if you need custom workflows — `deploy_media.py`
is the reference implementation.
