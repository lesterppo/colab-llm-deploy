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

## Scale-loss knob run (2026-11-04, the delegated pick after the above)

The pick was `--scale-loss-weight 1e-2` (Brush 0.3.0 default 1e-8),
the ONLY change vs the clean run: same clean dataset, cameras,
30,000 steps, T4, rc=0 in 256 s. Result: **model collapse — NOT
GREEN, and a metric warning.** Splat counts: 19,274 at 5k, 19,874 at
10k, 2,434 at 15k, **4 points at 20k/25k/30k** — the weight was orders
of magnitude too strong and the MCMC pruning annihilated the model.
The trap: mean val PSNR at 30k read **14.64 dB**, higher than the
clean run's 12.85, which a numbers-only report would have called an
improvement. It is not: the 30k eval renders are a FLAT solid colour
(no sky gradient, no sea, no church at all) and the 15k renders are
nearly black. A flat average-colour image scores ~14.6 on this
dataset because most pixels are background — so **mean PSNR alone
is nearly meaningless here**; every future verdict needs the visual
check (done for this run) or a foreground-masked PSNR. What the run
did prove: the scale-loss term DOES bite hard on splat survival, so
the mechanism is real but the dose was wrong. Next probe, named but
NOT run: `--scale-loss-weight 1e-5` (log-middle between the 1e-8
default and the lethal 1e-2), evaluated with masked PSNR + visuals.

## Scale-loss 1e-5 run (2026-11-04, "Further develop")

The log-middle dose, the ONLY change vs the clean run: same clean
dataset, cameras, 30,000 steps, T4, rc=0 in 371 s. Result: **still
NOT GREEN, but a real, small, dose-response signal.** Substance guard
passed: 154,958 points at 15k/20k/25k/30k (vs ~198k clean, vs 4 points
at 1e-2 — the model survives). Mean val PSNR 13.22 dB at 15k, 13.17 at
20k, 12.88 at 30k, vs clean 12.44 / 12.85 / 12.36: +0.78 / +0.32 /
+0.52, best overall 13.22 vs 12.85. Eval renders: church readable
(red roofs, tower, nave) but still hazy, translucent smears over the
lower half — fog reduced marginally, not eliminated. Dose-response
so far: 1e-8 → 12.85 best, 1e-5 → 13.22 best, 1e-2 → collapse. The
curve is shallow (+0.37 dB over three orders of magnitude), so a
sweet spot between 1e-5 and 1e-2 (e.g. 1e-4, 1e-3) is unlikely to
close the remaining ~2.8 dB to the 16 dB bar on its own. The remaining
gap looks structural: Brush MCMC on perfectly flat synthetic sea/sky
at 400 px. Next ideas, NOT run: (a) texture the scene — subtle
procedural variation in sea/sky (clouds, sea mottling) so flat
regions stop being flat, which attacks the root cause rather than
the optimizer; (b) 640 px renders; (c) a 1e-4 / 1e-3 dose probe.
Method note from this run: a foreground-masked PSNR was attempted
and abandoned after verification — at close framing the island +
church fill ~71% of the frame (confirmed two ways), so mask and
all-pixel PSNR nearly coincide; the operative guards remain splat
count (>=100k) + visual check, which is what caught the 1e-2 trap.

## Parallel sweep, 2026-11-04 ~23:40 HKT (Peter: "Further develop",
10 Colab accounts now available) — GREEN

Four variants trained in parallel, one T4 session per account, all
30,000 steps, rc=0, all sessions stopped + registries deleted after:
- **Textured scene + 1e-5** (cyc236es, 367 s): sea modulated +/-8%
  by hit position, sky/sea_bg by soft directional cloud variation,
  same geometry/cameras. 153,095 points; mean val PSNR 14.13 (15k) /
  14.30 (20k) / 13.37 (30k). Texture helps (+1.08 over 1e-5 alone)
  but does NOT reach GREEN on its own.
- **Dose 1e-4** (ppoppo2052, 291 s): clean scene. 82,385 points;
  13.34 / 13.51 / 13.60. Slight gain over 1e-5, still far from GREEN.
- **Dose 1e-3** (5stardse, 281 s): clean scene. 26,447 points;
  12.66 / 13.92 / **16.96**. Per-view at 30k: 19.21, 20.04, 15.57,
  9.69, 17.84, 20.22, 19.51, 13.59. **GREEN on the declared bar:**
  mean 16.96 dB >= 16, substance guard passed (26k points, not a
  collapse), and the eval renders are visibly crisp on 6/8 views
  (sharp red roofs, clean tower, structured sea) on HELD-OUT val
  views, so this is genuine generalisation, not memorisation.
  Qualified: val_003 remains hazy (9.69 dB) — one structurally hard
  angle (worst in every run so far), and val_007 is middling
  (13.59). The dose curve is now: 1e-8 -> 12.85, 1e-5 -> 13.22,
  1e-4 -> 13.60, 1e-3 -> 16.96, 1e-2 -> collapse. The 1e-4 -> 1e-3
  jump (+3.36) is a regime change, not a trend: aggressive pruning
  to 26k well-placed splats beats 150k+ foggy ones. Trained PLY
  (export_30000, 26,447 points, 6.0 MB) kept locally as
  zanjic_trained_1e-3_30000.ply; it is now the candidate to ship
  in the phone viewer in place of the procedural reconstruction.
- **640 px renders** (kyu110110, 573 s): clean scene, defaults.
  196,804 points; 12.67 / 12.37 / 11.95 — WORSE than 400 px.
  Resolution alone is eliminated.

Lesson overturned: the 1e-5 report's "too shallow to reach 16 dB"
read was wrong — the curve is flat until 1e-3, then jumps. The
standing GREEN recipe for this scene is: clean scene, close
framing (R 4.2, FOV 0.70), `--scale-loss-weight 1e-3`, 30,000
steps. Untested combo that might fix the weak val_003 angle:
textured scene + 1e-3 (see below: the intermediate dose did not
displace it). Arxiv rotation untouched; global active
account was observed as kyu009009 at teardown (switched by another
agent during the evening, not by this work — all sweep sessions
used --token-file, never the global active).

## Intermediate dose 3e-4 (2026-11-05, Peter's pick after the phone
demo looked blurry)

Question: does a dose between 1e-4 and 1e-3 keep the fog cleared AND
keep enough splats for sharper renders? Only change vs the clean run,
30,000 steps, T4, rc=0 in 285 s. Account note: cyc236es refused
creation with 503, so one switch to ppoppo2052 (verified idle first);
kyu008008/kyu009009 (VEC4 gate), cyc236ha (unknown T4), kyu110110
(possibly weight-space) and ppoppo205 were deliberately avoided, and
5stardse's token was found dead (refresh invalid-grant, 401) during
the idle check. Result: **does NOT fix the blur.** 41,557 points
(monotonic in dose, as expected); mean val PSNR 13.56 (15k) / 13.98
(20k) / 14.34 (30k) — above 1e-4's 13.60 but far below 1e-3's 16.96.
Visuals: val_000 softer than 1e-3's (more smear, softer roof edges),
val_003 marginally better than 1e-3's (10.57 vs 9.69) but still hazy.
Giant-splat fraction: 98 splats with max-scale > 0.5 (0.24%, vs 1e-3's
~264 / 1.0%) but a larger single maximum (38 units vs 10) — so giant
count alone does not explain the blur; the softness tracks the
dose regime, not just the tail. Dose curve now complete: 1e-8 ->
12.85, 1e-5 -> 13.22, 1e-4 -> 13.60, 3e-4 -> 14.34, 1e-3 -> 16.96,
1e-2 -> collapse. The GREEN jump lives strictly between 3e-4 and
1e-3. The 1e-3 model remains the best demo model. Remaining blur
fixes, still not run: giant-splat stripping (cheap post-process,
weakened as a sole explanation by this run) and textured scene +
1e-3. Trained PLY kept locally as zanjic_trained_3e-4_30000.ply
(41,557 points, 9.4 MB). Session stopped, registry deleted, arxiv
rotation untouched.

## Both remaining blur approaches tested (2026-11-05, Peter: "Test
both remaining approaches")

### Approach 1: giant-splat stripping (post-process, no retrain)

Removed all splats with max exp(scale) > 0.5 from the GREEN 1e-3
model: 163 giants removed (0.62%, max-scale up to 10.01), 26,284
splats left. Correction to the earlier report: the giant count is
163, not ~264 (that was a 2-decimal rounding of 0.0062 to 0.01).
Test method, stated honestly: Brush 0.3.0 has NO render-a-PLY mode
(its CLI only trains from a dataset), so the stripped model was
tested two ways: (a) a purpose-built CPU Gaussian renderer
(standard 3DGS EWA splatting, SH degree 3) rendering both models at
the 8 val cameras, and (b) a phone demo built with the stripped PLY
for judgment in Spark, the real viewer. The CPU renderer's absolute
numbers are NOT comparable to Brush's (its renders come out much
darker: mean ~31 vs Brush's ~111, i.e. it under-paints small splats
relative to Brush/Spark), but the stripped-vs-original comparison
under identical rendering is controlled. Result in the CPU
renderer: stripping HURTS on all 8 views — mean PSNR vs GT 8.41
stripped vs 9.54 original. In that renderer the giants are
load-bearing: they paint the large sea/sky regions the sub-pixel
dust cannot cover. Caveat: because the CPU renderer under-paints
small splats, giants matter more there than they may in Spark, so
the phone verdict on the stripped demo rules for the viewer
experience, not this renderer. Stripped demo:
zanjic_green_demo_stripped.html (15.0 MB, JS syntax-checked,
26,284 points). Verdict: stripping is NOT validated as a blur fix
by the renderer test; it stands or falls on the phone comparison.

### Approach 2: textured scene + 1e-3 (Colab retrain)

Textured dataset (the one that gave 14.30 at 1e-5) trained at the
GREEN dose 1e-3, 30,000 steps, T4 ppoppo2052, rc=0 in 278 s.
Result: **clear failure.** 26,912 points; mean val PSNR 11.85
(15k) / 11.47 (20k) / 11.67 (30k) vs 1e-3-clean's 16.96, and the
val_000 render is a smeary wash with the church barely
recognisable. Mechanism (consistent with both texture runs):
texture helps only when splats are plentiful (tex+1e-5: 14.30 >
13.22); at the 1e-3 pruning regime (~27k splats) the model cannot
afford the extra sea/sky detail, so texture hurts badly. Giants:
81 (0.30%), max 6.31 — fewer giants than clean 1e-3, yet far worse,
confirming again that giant count does not drive quality here.
Trained PLY kept locally as zanjic_trained_tex_1e-3_30000.ply.
Conclusion: the plain clean-scene 1e-3 model remains the best
model for the demo. Both remaining blur approaches are now closed
unless the phone comparison of the stripped demo says otherwise.
Session stopped, registry deleted, arxiv rotation untouched.

### Phone verdict on stripping (2026-11-05 07:03 HKT) — stripping CLOSED

Peter compared the stripped demo with the original GREEN demo and
the procedural lively v2 viewer on his phone: the stripped demo
looks SIMILARLY BLURRY, and the procedural lively v2 looks HIGHER
DEFINITION than any trained demo. Renderer A/B and phone now agree
stripping does not help, so giant-splat stripping is closed as a
blur fix. The procedural sharpness gap is expected, not a mystery:
procedural is ray-traced geometry sampled at 87,528–327,377 points
with tiny splats and exact colours, while the best trained model
has 26,447 pruned splats. Procedural wins on sharpness; trained wins
on being a real learned reconstruction.

### Densification tuning at 1e-3 (2026-11-05 07:08–07:18 HKT)

Last untried mechanism (not another dose): Brush refine knobs at
the GREEN dose, as one pro-growth package — `--growth-grad-
threshold 0.00004 -> 0.00001` (4x easier growth trigger),
`--growth-select-fraction 0.1 -> 0.3` (3x more flagged splats
grow), `--growth-stop-iter 15000 -> 30000` (growth runs the full
training instead of stopping halfway). Clean scene, 30,000 steps,
T4 ppoppo2052, session brush-zanjic-dn, rc=0 in 267 s.
Counts trajectory: 68,798 (5k) -> 5,956 (10k) -> 4,244 (15k) ->
4,250 (20k/25k) -> 18,844 (30k). Net density was NOT restored:
18,844 < clean 1e-3's 26,447 — aggressive growth plus 1e-3 scale
loss prunes harder mid-run, and the 30k rebound does not recover
the deficit. Giants: 1,295 (6.87%), max 5.75 — far more giants
proportionally than clean 1e-3 (163, 0.62%), yet PSNR still rose,
confirming again that giant count does not drive quality.
Mean val PSNR (30k, same local computation as all runs): 18.65 dB
vs clean 1e-3's 16.96 (+1.69). Per view: 22.10, 19.48, 20.21,
11.42, 23.29, 22.19, 15.64, 14.86 — beats clean 1e-3 on 6/8 views,
loses on val_001 (19.48 vs 20.04) and val_006 (15.64 vs 19.51).
Visuals: val_000 roofs look fairly clean-edged, walls/ground
still soft; val_003 improved (11.42 vs 9.69) but still hazy and
washed out (render mean 151 vs GT 96). Verdict: NUANCED, not a
sharpness breakthrough — PSNR improved through better placement,
not restored density, and the phone/Spark sharpness question is
still open. Demo built for the phone comparison:
zanjic_green_demo_densify.html (12.6 MB, JS syntax-checked,
18,844 points). Trained PLY kept locally as
zanjic_trained_dn_1e-3_30000.ply. Session stopped, registry
deleted, arxiv rotation untouched.

## Handoff for another agent (2026-11-05)

Everything needed to reproduce or continue this work is now in this
directory. The elimination record above is the starting point: do NOT
re-run framing, clutter, 640 px, tex+1e-3, or the mapped doses
(1e-8/1e-5/1e-4/3e-4/1e-3/1e-2) expecting different outcomes without
a genuinely new hypothesis.

### File map

- `build_zanjic.py` — lively procedural scene + medium lively PLY
  (the sharp phone demo baseline; procedural wins on sharpness).
- `build_zanjic_clean.py` — the CLEAN scene (pre-lively colours, no
  lively prims). This is the dataset every training run used.
  Run it, then `tar czf dataset.tar.gz dataset dataset_val` — PNGs
  MUST be saved uncompressed (see the 16,387-byte Brush bug above).
- `build_zanjic_tex.py` — textured variant (sea ±8% sinusoidal,
  soft directional cloud variation on sky/sea_bg).
- `setup.py` — Colab VM setup (Vulkan ICD via libnvidia-gl-<series>
  from `nvidia-smi`, Brush 0.3.0 binary). Identical for all runs.
- `train_clean.py` — baseline, default scale loss (1e-8).
- `train_d4.py` — dose 1e-4. `train_d3.py` — dose 1e-3, THE GREEN
  RUN. `train_m34.py` — dose 3e-4. `train_tex.py` — textured + 1e-5.
  `train_tx3.py` — textured + 1e-3. `train_dn.py` — 1e-3 +
  pro-growth densification (threshold 1e-5, fraction 0.3, stop 30k).
  Each script trains 30,000 steps, evals every 5,000 steps, exports
  to its own `dataset_exports_*` dir, and writes a status JSON.
- `render_ply.py` — purpose-built CPU 3DGS EWA renderer (SH degree
  3, val cameras R=4.2/elev 20°/offset 3.75°/FOV 0.70/400×400).
  Brush cannot render a standalone PLY, so this is how the
  stripping A/B was done. Caveat: it under-paints small splats
  (renders darker than Brush/Spark); only same-renderer A/B
  comparisons are evidence, never its absolute PSNR.
- `strip_giants.py` — removes max-exp(scale) > 0.5 splats, rewrites
  the vertex count. Verified: reproduces the shipped stripped PLY
  byte-identically. Stripping itself is CLOSED (renderer + phone
  agree it does not help); kept for reproducibility only.
- `trained/zanjic_trained_1e-3_30000.ply` — GREEN model, 26,447
  points, mean val PSNR 16.96 dB. Current demo default.
- `trained/zanjic_trained_dn_1e-3_30000.ply` — densified model,
  18,844 points, mean val PSNR 18.65 dB. Higher PSNR, lower density.
- `demos/zanjic_green_demo.html` — GREEN demo (15.0 MB, spin-on).
- `demos/zanjic_green_demo_densify.html` — densified demo
  (12.6 MB, spin-on).

### Reproduce a training run (Colab T4, colab-rs)

1. Build the dataset with `build_zanjic_clean.py` (or `_tex.py`).
2. Create a session on a VERIFIED-IDLE account only, with a
   per-account `--token-file` and an isolated `--config` registry.
   NEVER switch the VM-global active account; NEVER touch the
   arxiv training rotation. Recently used idle accounts:
   ppoppo2052, cyc236es. AVOID: kyu008008/kyu009009 (other chats),
   cyc236ha, kyu110110 (possibly claimed), supervisor-only
   cyc236de/cyc236hk, ppoppo205 (other chat), 5stardse (token
   dead since 2026-11-05 06:34 HKT — needs a fresh sign-in).
3. Upload `dataset.tar.gz`, `setup.py`, `train_*.py`; run setup
   detached, wait for ready, run train detached (~270–410 s).
4. `colab-rs download` FAILS on raw binaries: download exports
   and eval renders as base64 and decode locally. Compute val PSNR
   locally against `dataset_val` (same computation as all numbers
   in this README). Judge sharpness visually, never by mean PSNR
   alone (the 1e-2 flat-colour trap: 14.64 dB from 4 splats).
5. Stop the session and delete the registry when done.

### Build a phone demo

`../viewer/build_standalone.py --ply trained/<model>.ply --out
<name>.html --spin-on` (title/subtitle flags available). The repo
builder fetches three/Spark from CDN at build time; on a
CDN-blocked VM, replace the four lib URLs with local file paths
(three.module.js, three.core.js, Pass.js, spark.module.js) and
keep the three-core rewrite (three.module.js wrapper import
rewritten to bare specifier `three-core`, mapped in the importmap
— relative imports cannot resolve from blob: URLs). Check: no
unfilled template tokens, JS syntax OK, embedded PLY byte-identical
to the trained file. The phone's status-line text / visual verdict
is ground truth; this VM cannot reliably verify these pages.

### Open threads (2026-11-05 07:28 HKT)

- IMMEDIATE: phone comparison of `demos/zanjic_green_demo_densify
  .html` vs `demos/zanjic_green_demo.html`. If densified looks
  sharper, it becomes the demo default; if similarly blurry,
  densification joins the closed list and trained-sharpness work
  is out of knobs (only genuinely new hypotheses remain).
- The lively-v2 procedural phone confirmation (full 327k vs
  embedded 87.5k fallback) from 2026-11-04 was never confirmed and
  is overtaken by the GREEN demo thread.
- Trained PLYs for the failed variants (3e-4, tex+1e-3, tex+1e-5,
  1e-4, clean baseline) were NOT pushed; their scripts and numbers
  above reproduce them.
