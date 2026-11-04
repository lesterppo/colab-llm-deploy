# Zanjic Church — island monastery reconstruction

A 3D reconstruction of the Žanjic island monastery: the Church of the
Presentation of the Virgin (Vavedenje) on tiny Mala Gospa island at the
entrance to Boka Bay, Montenegro (first mentioned 1508 as
Sv. Maria de Sagnic). Stone single-nave church with a west bell tower
and cross, red-tiled roofs, semicircular east apse, fortified perimeter
wall with a jetty gap, cypress and olive trees, on a rocky island in
the Adriatic — plus, in the lively edition, wave caps, flower bushes,
birds, people, a jetty flag and a second boat.

## Files

- `build_zanjic.py` — the scene. Same two-output structure as the
  lighthouse script: a ray-traced dataset (`dataset/`, 48 train + 8 val
  views, 400×400, uncompressed PNGs — see the small-PNG warning below)
  and a procedural PLY sampled from the same primitives
  (`zanjic_procedural.ply`, isotropic Gaussians, SH degree 0).
  Run `python3 build_zanjic.py` (needs numpy + Pillow; ~80 s for the
  dataset). The script as shipped uses the lively + sharpest scales:
  it regenerates the 327,377-point PLY. Coarser scales for smaller
  files are documented in the SCALE comment history inside the file.
- `../viewer/` — the standalone viewer builder and template. This is
  the recommended way to present the result (see that README for the
  full phone-proven design):
  ```bash
  python3 ../viewer/build_standalone.py --ply zanjic_procedural.ply \
      --out zanjic.html --title "Zanjic Church" --spin-on
  ```
  If the single file exceeds ~100 MB, use `--ply-url` + `--fallback-ply`
  instead (the lively edition shipped that way: 6.7 MB HTML + 81 MB
  PLY, because a 115 MB single file was rejected with HTTP 413).

## Brush training result — honest verdict, read before re-running

The scene was Brush-trained on a free Colab T4 exactly like the
lighthouse (session `brush-zanjic`, 30,000 steps, rc=0 in ~8 min,
266,426 points, 62,878,087 bytes, SH degree 3), and it UNDERPERFORMED:
mean val PSNR **12.98 dB** (best view 16.15, worst 10.05) against the
lighthouse's 18.59 dB. The eval renders are foggy: large translucent
ground/sea splats swallow the low, wide church, while the tower and
red roofs stay recognisable. Suspected cause: a low subject over a
huge flat sea disc (r=8) — the tall lighthouse stayed crisp above the
same kind of haze. The shipped phone viewer therefore uses the crisp
procedural reconstruction, not the trained splat. Do NOT present the
trained result as the good version. A retrain worth trying (not yet
done): shrink the sea disc so the ground plane dominates less of the
frame, and compare val PSNR before shipping any trained version.

## Brush pitfalls (shared with the lighthouse — do not re-learn)

- **Small-PNG "early eof".** Brush 0.3.0's image-header reader does
  `read_exact` over a 16,387-byte buffer; a PNG smaller than that
  fails the whole dataset load. Flat-colour renders compress to
  ~14 KB, so this script saves every PNG uncompressed (~480 KB each).
- **Missing Vulkan ICD.** A stock Colab VM has the CUDA driver but no
  Vulkan ICD, so Brush's wgpu backend panics with "No possible adapter
  available". Install the `libnvidia-gl-<series>` package matching the
  LOADED driver series (read from `nvidia-smi`, never guessed) plus
  `vulkan-tools`. See `../brush-lighthouse/setup.py`, which is
  reusable as-is.
- **CLI flags.** The prebuilt 0.3.0 binary uses `--total-steps`
  (default 30000), `--export-every`, `--eval-every`,
  `--eval-save-to-disk`, `--export-path`; main-branch source uses
  `--total-train-iters`. Trust the 0.3.0 `--help`. Do NOT build Brush
  from source on Colab: main-branch deps need rustc ≥ 1.95, rustup
  gives 1.88.
- **Downloads.** `colab-rs download` fails on raw binary files, so
  ship results as base64-encoded tars and decode locally.
- **Account hygiene (standing rule).** Never switch the VM-global
  active Colab account and never touch the arxiv training rotation
  for this work; use a token-file + isolated registry for the session
  account (this run used cyc236es that way, session stopped afterwards,
  global account and rotation untouched).

## Version history (2026-11-04 evening)

- v1: 20,217 points, single-file standalone (13.4 MB).
- High resolution: scales halved → 75,102 points, single file (31.5 MB).
- Lively + sharpest: lively prims added, scales halved again →
  327,377 points, 81.2 MB PLY (shipped as 6.7 MB HTML + hosted PLY,
  with an 87,528-point embedded medium lively fallback after the full
  PLY fetch failed on the phone with "Failed to fetch").
