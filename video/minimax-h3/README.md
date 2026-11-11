# MiniMax H3 (Hailuo 03) — 4-step INT8 text-to-video on free Colab T4

**Status: GREEN 2026-11-11.** Live-tested on a free T4 (account kyu009009,
session `minimax-h3-4step`, ~1h33m total, NOT reclaimed): setup ~15 min
(42.04GB download), generation 1,979s (~33 min), peak VRAM 14,619/15,360 MB.
Output verified on this VM: 5.17s, 640x960 @24fps, AAC 32kHz stereo,
6,017,812 bytes (`minimax_h3_out.mp4`); frame at 2.5s matches the lighthouse
prompt (golden-hour cliffs/waves). Stack: ComfyUI 0.45.0, torch 2.14.1+cu130,
driver 580.82.07.

## What this is

MiniMax H3, 33B omni-modal video model (open weights 2026-08-03), run in
ComfyUI via the official `video_minimax_h3_t2v.json` template recipe:
FL2VA pruned INT8 diffusion + NVFP4 Qwen3-VL-32B encoder + int8 video VAE
+ fp32 audio VAE + lightx2v 4-step turbo LoRA, sampler `er_sde` +
scheduler `sgm_uniform`, 4 steps, Sigma Shift video 6 / audio 3.

Reference (Local Lab Digest 2026-10-22, RTX A4500 20GB): 5s, 640x960
0.6MP, ~1 min sampling, peak ~13GB VRAM. The T4 has 15GB VRAM but only
~12GB system RAM — the 15.7GB encoder is the risk, not the diffusion
model. A 16GB swapfile is attempted during setup (best effort).

## Files

- `setup.py` — stage-gated, writes `/content/minimax_h3_setup_status.json`
- `gen.py` — reads `/content/gen_config.json`, writes
  `/content/minimax_h3_gen_status.json`, MP4 to `/content/minimax_h3_out.mp4`
- `gen_config.json` (written by the driver): prompt, seed 42, 640x960,
  124 frames (17*7+5 grid, ~5.2s @24fps), 4 steps

## Model files (Comfy-Org/MiniMax-H3, 42.04GB total, verified on the run)

| File | Size (verified) |
|---|---|
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20.97GB |
| `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15.69GB |
| `vae/minimax_h3_video_vae_int8_convrot.safetensors` | 2.81GB |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 0.61GB |
| `loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors` | 1.96GB |

If free disk < 50GB, setup switches diffusion to W6A8
(`minimax_h3_fl2va_pruned_w6a8.safetensors`, ~16GB) and records it.

## Honest limits (2026-11-11 run)

- Only ~700MB VRAM headroom (14,619/15,360 MB peak). 1MP and longer clips
  are UNTESTED — the GREEN result is 0.6MP, 124 frames (~5.2s @24fps), 4 steps.
- The int4 fallback route (below) is staged but UNTESTED.
- VSA / Semantic Bridge skipped; the SageAttention patch node was not
  present in ComfyUI 0.45.0 (`sage_patch: false` in the gen status) — the
  run still completed in 1,979s without it.
- Swapfile attempt failed (`swap_ok: false`) and was non-fatal.
- Evening sessions get reclaimed; this run survived ~1h33m on kyu009009.
  A morning window is the safer bet for a repeat.

## Pitfall fixed in gen.py (cost one full 32-min sampling run)

ComfyUI 0.45 `SaveVideo`: `format`/`codec` must be the dynamic-combo keys
`auto`/`auto`, NOT `video/h264-mp4`. The latter passes prompt validation,
then `execute()` gets no format at all (TypeError, prompt "done", no file
written). `gen.py` now uses `format: "auto", codec: "auto"` (template default).

## Fallback route (int4)

If the 32B NVFP4 encoder fails on the T4, one fallback in the same run:
FL2VA pruned W4A8 12.5GB + Qwen3-VL-4B INT4 2.81GB + ClipProj 503MB +
4-step turbo LoRA 780MB, 864x480, 6 steps. See `setup.py` ROUTES.
Total ~17GB. Not yet attempted.

## License

MiniMax H3 Community License — excludes the US, EU, UK and South Korea.
Verify before non-test use.

## Run

One command from the repo root (wired into `deploy_media.py`):

```bash
python3 deploy_media.py generate --kind video --model minimax-h3 \
    --prompt "A lighthouse on rugged cliffs at golden hour, waves crashing below, cinematic, with the sound of waves and wind" \
    --out ./outputs
```

Host driver used for the GREEN run: `~/workspace/minimax-h3-test/run.py`
(isolated registry, session `minimax-h3-4step`, account kyu009009).
