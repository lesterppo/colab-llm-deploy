# Music generation on Colab

One-command lyrics-to-song on free Colab GPUs (T4, 15GB VRAM) via ComfyUI,
with quantized/compact models. Both pipelines render through a ComfyUI API
workflow and download the finished audio file — no chat endpoint involved.

## One command

```bash
# YuE2-3B (lyrics-to-song, 40s):
python3 deploy_media.py generate --kind music --model yue2-3b \
    --lyrics "[Verse] Golden light on the water, waves are rolling in..." \
    --style "cinematic folk, acoustic guitar, warm male vocal" \
    --out ./outputs

# MiniMax Music 3 (INT8 repack, 30s):
python3 deploy_media.py generate --kind music --model minimax-music3 \
    --caption "Genre: cinematic folk. BPM: 90. Warm male vocal." \
    --lyrics "..." \
    --out ./outputs

python3 deploy_media.py status   --session media-yue2-3b
python3 deploy_media.py undeploy --session media-yue2-3b
```

`deploy_media.py` (repo root) does everything: session → upload → detached
setup (pip + model downloads via the `hf` CLI) → poll → detached
generation → poll → download audio → stop session. Lyrics/style go through
`/content/gen_config.json`; run with no `--lyrics` to reproduce the tested
lighthouse song from the built-in defaults.

## Model: yue2-3b — YuE2-3B (weights CC BY-NC 4.0, test use only)

- Files: `yue2-3b/setup.py`, `gen.py`
- Stack: ComfyUI 0.38.0 + pytraveler/YuE2-ComfyUI + tiktoken, sdpa
  attention backend.
- Weights (7.25GB): `m-a-p/YuE2-3B` (model.safetensors + qwen.tiktoken)
  + `m-a-p/YuE2-Vae`, downloaded into the `original` layout the pack
  reads as-is (`models/YuE2/YuE2-3B/`, `models/YuE2/YuE2-Vae/`).
- Workflow: `YuE2Options(cot=full, max_seconds=40, download=off)` →
  `YuE2GenerateSong(style/lyrics/seed)` → `SaveAudio`.
- Proven (2026-10-05, T4, kyu008008): 40s cinematic-folk lighthouse song
  in ~250s, peak VRAM ~6.4GB — no quantization needed.
- Pitfalls (fixed in the scripts): multi-line lyrics must reach the VM as a
  file (`gen_config.json` upload) — pasting them through `exec --code`
  breaks on newlines. T4 has no BF16, so the acoustic stage is slower than
  on A100-class GPUs.

## Model: minimax-music3 — MiniMax Music 3 (verify license for non-test use)

- Files: `minimax-music3/setup.py`, `gen.py`, `server.py` (optional tunnel)
- Stack: ComfyUI (latest) + custom MiniMaxMusic3 nodes from Comfy-Org pack.
- Weights (11.9GB): Peter's INT8 convrot repack recipe (22GB → 11.9GB):
  `diffusion_models/minimax_music3_dit_int8_convrot.safetensors` (2.5GB) +
  `text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors`
  (9.2GB) + `vae/minimax_music3_dav.safetensors` (217MB), from
  `Comfy-Org/MiniMax-Music-3`.
- Workflow: `CLIPLoader(type=minimax)` → `MiniMaxMusic3TextEncode`
  (caption/lyrics/seed/max_duration/cfg 1.7) → `ConditioningZeroOut` →
  `EmptyMiniMaxMusic3LatentAudio` → `KSampler` (euler, 30 steps) →
  `VAEDecodeAudio` → `SaveAudioAdvanced` (probe-and-fallback to core
  `SaveAudio` on older ComfyUI builds).
- Proven (2026-10-05, T4, kyu008008): 30s cinematic-folk lighthouse song in
  ~301s, peak VRAM 11.9GB (3.4GB headroom).

## For agents

`setup.py` is stage-gated (`/content/yue2_setup_status.json`,
`/content/music3_setup_status.json`); `gen.py` writes
`/content/yue2_gen_status.json` / `/content/music3_gen_status.json`. Both
read `/content/gen_config.json` (style/caption/lyrics/seed/max_seconds/
duration/out_name). Drive them manually with hermes-colab-cli if you need
custom workflows — `deploy_media.py` is the reference implementation.
