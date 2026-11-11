# Wan 2.2 TI2V-5B fp16 — text-to-video on free Colab T4

**Status: GREEN 2026-11-12.** Live-tested on a free T4 (account kyu009009,
session `wan22-5b`, ~21 min total, NOT reclaimed): setup ~7 min
(18.15GB download), then TWO clips in the same session, both verified
on this VM. Stack: ComfyUI 0.45.0 (latest), torch 2.14.1+cu130,
driver 580.82.07, T4 15,360 MiB, RAM 13.6GB.

Order actually run: **I2V first, then T2V** (Peter reordered mid-test;
T2V had not started yet when the reorder arrived, so the host driver
was stopped before gen and the I2V leg ran first).

## Leg 1 — I2V (Peter's own photo)

- Input: Peter's photo `1000088579_1_ev0c.jpg` (2048x1536, 4:3) —
  woman with glasses and orange scarf at a stone railing viewpoint
  on Huangshan, granite peaks, pines, layered blue mountains.
  Resized to 768x576 (4:3 preserved, 32-multiples, no crop, no
  stretch) as `peter_huangshan_768x576.png`; original kept at
  `~/workspace/wan22-5b-test/outputs/peter_huangshan_source.jpg`.
- Prompt (person kept STILL): "Slow cinematic camera push-in across
  a Huangshan mountain viewpoint, clouds drifting slowly between
  distant peaks, pine branches swaying gently, the woman standing
  still and smiling, her orange scarf fluttering very slightly in
  the breeze". Negative adds: person walking/turning/talking,
  face changing, morphing.
- Output verified: 3.375s, 768x576 @24fps, h264, 1,958,898 bytes
  (`wan22_5b_i2v_out.mp4`); generation 415s, peak VRAM 13,679MB.
  Mid-clip frame matches the photo's composition closely (loose
  reference, as TI2V is — not a pixel-perfect first-frame lock;
  the strict first-frame I2V is the A14B model, which does not fit
  a T4). SILENT (no native audio in Wan 2.2).

## Leg 2 — T2V (Huangshan drone)

- Prompt: "Drone aerial view flying over Huangshan (Yellow Mountain)
  at sunrise, sea of clouds drifting between granite peaks and pine
  trees, golden light, cinematic slow drone movement". Seed 42,
  832x480, 81 frames, 20 steps, cfg 5.0, uni_pc/simple, shift 8.0.
- Output verified: 3.375s, 832x480 @24fps, h264, 287,951 bytes
  (`wan22_5b_out.mp4`); generation 370s, peak VRAM 14,011MB.
  Mid-clip frame: golden sunrise, sea of clouds between granite
  peaks — matches the prompt. SILENT.

## Notes

- cyc236es and kyu008008 were GPU-unavailable that night; kyu009009
  got the T4 (one attempt each, no retry loop).
- A VM-side service restart killed the host driver mid-I2V-gen; the
  VM-side gen kept running and the driver re-attached with
  `--watch-only` (same pattern as the H3 run).
- Sampling is slow on T4 (~7 min per 3.4s clip vs ~60s on an 8GB
  desktop card in the reference) — sm_75 math-path attention +
  offloading. Peak VRAM 13.7–14.0GB leaves ~1.3GB headroom at
  832x480; 720p/121 frames untested on T4.
- A Wikimedia Huangshan photo was found first (CC BY-SA 4.0,
  "Huangshan mist 20141109.jpg", author 颐园居) but NOT used —
  Peter sent his own photo instead.

## What this is

Wan 2.2 TI2V-5B (Alibaba Tongyi Lab, Apache 2.0) in ComfyUI via the
official `video_wan2_2_5B_ti2v.json` template: fp16 diffusion + fp8
UMT5-XXL encoder + 2.2 high-compression VAE, sampler `uni_pc` +
scheduler `simple`, 20 steps, cfg 5.0, ModelSamplingSD3 shift 8.0.
The 5B is a dense hybrid T2V+I2V model, 720p @24fps — the latest Wan
family with public weights (2.5/2.6/2.7/3.0 are API-only).

Reference (8GB card, ComfyUI, no launch flags): fp16, 20 steps,
sampling 60s, wall-clock 73s, peak VRAM 7,686 MiB (832x480 clip).
First T4 target is the same size: 832x480, 81 frames (~3.4s @24fps).

## Files

- `setup.py` — stage-gated, writes `/content/wan22_setup_status.json`
- `gen.py` — reads `/content/gen_config.json`, writes
  `/content/wan22_gen_status.json`, MP4 to `/content/wan22_5b_out.mp4`
- `gen_config.json` (written by the driver): prompt, seed 42, 832x480,
  81 frames, 20 steps, cfg 5.0

## Model files (Comfy-Org/Wan-2.2, ~18.2GB total, verified via HF API)

| File | Size |
|---|---|
| `diffusion_models/wan2.2_ti2v_5B_fp16.safetensors` | 10.0GB |
| `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6.74GB |
| `vae/wan2.2_vae.safetensors` | 1.41GB |

The 2.2 VAE is the 16x16x4 high-compression VAE, ONLY for the 5B
model — the 14B models need the Wan 2.1 VAE. No native audio in
Wan 2.2: the output MP4 is SILENT by design, not a failure.

## T4 risks

- sm_75 SDPA: torch 2.11+cu130 has no fast SDPA kernel on T4; the
  math path materialises the full attention matrix (Wan 2.1 OOM
  lesson). ComfyUI native Wan nodes + `--lowvram` handled H3 fine;
  if sampling OOMs at step 0, chunked attention / blocks_to_swap
  is the fix to try before declaring failure.
- Frames must be 4n+1 (VAE 4x temporal compression).

## Run

One-command T2V (wired into `deploy_media.py`):

```bash
python3 deploy_media.py generate --kind video --model wan22-5b \
    --prompt "Drone aerial view flying over Huangshan (Yellow Mountain) at sunrise, sea of clouds drifting between granite peaks and pine trees, golden light, cinematic slow drone movement" \
    --out ./outputs
```

Host driver for manual runs: `~/workspace/wan22-5b-test/run.py`
(isolated registry, session `wan22-5b`, account cyc236es first).

## I2V (manual driver run — no `--image` flag in deploy_media.py)

`gen.py` reads `cfg["image"]` as a filename in `ComfyUI/input/` and
feeds it to `Wan22ImageToVideoLatent` as `start_image`. The one-command
path is T2V only; for I2V, after `setup.py` completes, upload the image
to `/content/ComfyUI/input/<name>.png` and write
`/content/gen_config.json` with `"image": "<name>.png"`,
`"out_name": "wan22_5b_i2v_out"`, and width/height matching the photo's
aspect (the tested leg used 768x576 for a 4:3 photo — do NOT stretch
to 16:9), then run `gen.py` via `exec_detach` as `deploy_media.py`
does for T2V. Keep the person STILL in the prompt (camera and
scenery move, not the person) to avoid face/body morphing.
