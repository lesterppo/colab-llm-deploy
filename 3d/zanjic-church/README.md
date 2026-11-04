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

## GREEN attempt that FAILED (2026-11-04, do not repeat blindly)

On the order "make the training GREEN", one retrain tried the obvious
framing fix: orbit R 5.5 → 4.2, field of view 0.8727 → 0.70 rad, so the
church covers ~1.7× more width and ~3× the pixels (dataset 48 train +
8 val, 400×400, same Brush 0.3.0 defaults, 30,000 steps, T4, rc=0 in
402 s). Result: **NOT GREEN — worse than the original.** Mean val
PSNR 10.89 dB at 30,000 steps (per view 12.90, 14.19, 9.48, 8.35,
14.30, 8.11, 11.72, 8.06), against the original's 12.98 dB; the
15,000-step export was better (12.51 dB) than the final one, so longer
training degraded it, and the eval renders are heavily fog-washed —
the church is barely readable under translucent smears. Two confounds,
so the framing hypothesis itself is NOT cleanly refuted: (1) the
retrain's dataset was built from the LIVELY scene (wave caps, birds,
flowers, second boat), not the clean pre-lively scene the original
trained on — tiny view-inconsistent clutter like birds is exactly what
a splat optimizer smears; (2) closer views see more of the flat sea
plane at grazing angles. Next hypotheses, in order: (a) rebuild the
dataset from the pre-lively scene at the original framing to isolate
the clutter effect; (b) Brush optimizer knobs — `--scale-loss-weight`
to penalise giant translucent splats, or a `--max-splats` cap; (c)
higher render resolution (640 px) so thin structures resolve; (d)
early stopping near 15,000 steps, since 15k beat 30k here. The green
script and eval renders are kept locally in the author's workspace,
not pushed, because the run is a negative result; this note exists so
the next agent does not re-run the same framing fix expecting a
different outcome.

## Clean-scene rerun (2026-11-04, the delegated pick after the above)

The pick was hypothesis (a) in its sharpest form: keep the GREEN
framing (R=4.2, FOV 0.70, identical cameras — val_007's camera matches
the failed run exactly) but remove ALL lively clutter and restore the
original colours, i.e. the exact pre-lively scene the original 12.98 dB
run trained on. Same Brush 0.3.0 defaults, 30,000 steps, T4, rc=0 in
410 s. Result: **still NOT GREEN, but the experiment answered its
question.** Mean val PSNR 12.44 dB at 15k, 12.85 dB at 20k, 12.36 dB at
30k — the late-training collapse is GONE (failed run degraded
12.51 → 10.89) and 30k improved by +1.47 dB over the failed run, so
lively clutter WAS a real poison. But the ceiling is unchanged:
~13 dB, same as the original wide clean run (12.98 dB), and the eval
renders are still hazy — red roofs and tower readable, but translucent
smears over the lower half and soft edges. So framing gives nothing
and clutter removal only returns to the ~13 dB ceiling: the remaining
fog is structural to Brush's defaults on this flat-textured synthetic
scene, not framing or clutter. Next pick, named but NOT run: Brush
optimizer knobs — `--scale-loss-weight` to penalise the giant
translucent splats directly (the observed failure mode), since
framing, clutter, and step count are now all eliminated as causes.
