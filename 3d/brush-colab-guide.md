# Brush on Google Colab — general agent guide

How to train a 3D Gaussian-splatting scene with Brush on a free Colab
T4, evaluate it honestly, and ship a phone-viewable result — for ANY
scene, not just the two examples in this repo. Written from runs that
actually completed on free T4s (2026-11-04/05). If a step below
contradicts your memory of Brush, trust this file: it was verified
against the 0.3.0 binary's own `--help` and against real runs.

Scene-specific companions: `brush-lighthouse/` (one-kick synthetic
scene, `python3 3d/brush-lighthouse/deploy_brush.py`) and
`zanjic-church/` (a full training campaign with an elimination
record and an agent handoff section). This guide is the general
layer above both. Numbers quoted from those scenes are examples,
NOT defaults — the one number you must not cargo-cult is Zanjic's
`--scale-loss-weight 1e-3`, which is scene-specific (see Tuning).

## 1. What Brush is

Brush (Arthur Brussee, Apache-2.0, Rust + WebGPU + Burn) trains a
3D Gaussian-splatting model from photographs and renders it in
real time. The model is NOT a mesh: it is a cloud of 3D Gaussians,
each with position, covariance (shape/scale), opacity, and
view-dependent colour (spherical harmonics). Nobody models anything;
the optimiser fits the blobs to the photos by differentiable
rendering. Consequence: photoreal capture comes free, but there is
no solid interior, no reliable dimension, no editable "wall" object,
and no mesh output. If you need CAD precision, use Blender/FreeCAD,
not Brush.

**Version pin: use the prebuilt 0.3.0 Linux binary.** Do NOT build
from source on Colab: main-branch deps need rustc >= 1.95 and
Colab's rustup gives 1.88 (proven dead end). Fetch it via the
GitHub releases API, picking the asset by name and verifying its
size against the API record — never a guessed URL. The binary's
`--help` is the truth for flags: main-branch source uses
`--total-train-iters`, but the 0.3.0 binary uses `--total-steps`.
`3d/brush-lighthouse/setup.py` does the whole fetch + verify and
is reusable as-is.

## 2. The dataset

Brush takes a COLMAP / Nerfstudio-style dataset: a directory of
images plus `transforms.json` (camera poses + intrinsics,
Nerfstudio format), passed as the positional `PATH_OR_URL`
argument. Practical rules:

- **Train/eval split is mandatory for honest numbers.** Hold out
  ~15% of views as eval (e.g. 48 train + 8 val). Or let Brush make
  one with `--eval-split-every <n>` (every nth image becomes eval).
- **Resolution:** 400×400 views trained fine and fast (~270–410 s
  for 30,000 steps on a T4). 1920 is the `--max-resolution` default;
  higher resolution does NOT fix a foggy model (on Zanjic, 640 px
  scored worse than 400 px) — it only costs time.
- **CRITICAL BUG (Brush 0.3.0, `get_image_data`): the image-header
  reader does `read_exact` over a 16,387-byte buffer, and it fails
  with "early eof" on any file smaller than that.** Flat-colour
  renders compress to ~14 KB and kill the whole dataset load in
  ~1 s. Fix: save every PNG uncompressed (~480 KB each at 400×400,
  pixels identical). Before uploading, check your SMALLEST image
  file is over 16,387 bytes. Do not "optimise" this back.
- **Coverage matters more than count.** Orbit the subject with
  overlapping views; a view angle missing from training stays bad
  in eval (on Zanjic, one val angle was worst in EVERY run).

## 3. Colab setup (the two stock-VM gaps)

A stock Colab T4 VM lacks two things Brush needs. `setup.py` in
`brush-lighthouse/` fixes both, idempotently, in ~50 s:

1. **A Vulkan ICD.** The VM has the CUDA driver (`nvidia-smi`
   works) but only the Vulkan loader — no ICD file — so Brush's
   wgpu backend panics with "No possible adapter available"
   before step 0. Fix: read the driver series from `nvidia-smi`
   (never guess it), `apt install libnvidia-gl-<series>` plus
   `vulkan-tools`, then verify an ICD file exists and `vulkaninfo`
   reports the T4. This does NOT touch the kernel driver.
2. **The Brush binary** (see §1).

**Classic mistake: creating a CPU session.** A session created
without an explicit GPU request gets accelerator NONE. Always
request the T4 explicitly and verify with `nvidia-smi` (expect a
Tesla T4) before training.

## 4. The training command

```bash
brush_app /content/dataset \
  --total-steps 30000 \
  --eval-every 5000 \
  --eval-save-to-disk \
  --export-every 5000 \
  --export-path dataset_exports
```

Flags that matter, with 0.3.0 defaults (full list in `--help`):

| Flag | Default | Notes |
|---|---|---|
| `--total-steps` | 30000 | Proven recipe for 400 px scenes. |
| `--eval-every` | 1000 | Eval renders at these steps. |
| `--eval-save-to-disk` | off | Must be ON to keep eval renders. |
| `--export-every` | 5000 | Exports `export_{iter}.ply` (see `--export-name`). |
| `--export-path` | `.` | Where exports + eval renders go. |
| `--scale-loss-weight` | 1e-8 | The fog knob. See §6. (Help text calls it "opacity loss"; that is a copy-paste error.) |
| `--opac-loss-weight` | 1e-9 | Opacity loss. Leave alone first. |
| `--ssim-weight` | 0.2 | SSIM vs L1 balance. |
| `--lr-mean` / `--lr-mean-end` | 2e-5 / 1e-6 | Mean LR schedule. |
| `--lr-scale` / `--lr-scale-end` | 1e-2 / 6e-3 | Scale LR schedule. |
| `--lr-opac` | 1e-2 | Opacity LR. |
| `--lr-coeffs-dc` | 2e-3 | Base SH (RGB) LR; higher orders divided by `--lr-coeffs-sh-scale` (20.0). |
| `--lr-rotation` | 1e-3 | Rotation LR. |
| `--sh-degree` | 3 | SH degree of splats. |
| `--max-splats` | 10,000,000 | Upper bound only; real runs land far below it. |
| `--refine-every` | 200 | Refinement frequency; roughly the image count that "covers" the scene. |
| `--growth-grad-threshold` | 0.00004 | Lower = faster growth. |
| `--growth-select-fraction` | 0.1 | Fraction of flagged splats that actually grow; higher = more aggressive. |
| `--growth-stop-iter` | 15000 | Growth stops after this step. |
| `--match-alpha-weight` | 0.1 | L1 on alpha for transparent inputs. |
| `--lpips-loss-weight` | 0.0 | Off by default. |
| `--max-frames` / `--max-resolution` | — / 1920 | Dataset caps. |
| `--subsample-frames` / `--subsample-points` | — | Load every nth frame / SfM point. |
| `--seed` | 42 | |
| `--start-iter` | 0 | Resume iteration (Brush resumes TRAINING, it does not render a PLY). |
| `--with-viewer` | off | Spawns a viewer; useless on headless Colab. |

Run training **detached** (it takes minutes and its progress bars
may not emit newlines for minutes — poll the process or a status
file, never a stdout-line loop, or you will think it hung).
Expect rc=0 and a `DONE`-style completion marker from your driver.

## 5. Getting results off the VM

- **Downloads of raw binaries can fail** on some Colab download
  paths. Ship results as base64 (`.b64`) and decode on the host.
  This is not optional paranoia; it is the observed failure mode.
- **Download incrementally.** Free-tier sessions live ~30–60 min
  and die without warning (reclamation). A 30k-step run fits
  comfortably, but never chain runs back-to-back on one session,
  and never wait for a final file to preserve a long run — pull
  exports at each eval step.
- **VM clock is UTC.** File timestamps read ~8 h behind HKT; this
  has caused false "stale file" alarms. Convert before concluding
  a file is old.

## 6. Evaluation: how to judge a run honestly

Compute val PSNR locally: render-vs-GT per held-out view, then the
mean — and report the per-view table, not just the mean. Check
three things before declaring any verdict:

1. **Substance.** Read the exported PLY's `element vertex` count.
   A run with a handful of splats is a collapse, whatever its
   PSNR says (see trap 1).
2. **Visuals.** Look at the eval renders, especially the worst
   view. PSNR is a pixel-error average, not uniform sharpness.
3. **The traps, all observed:**
   - **Flat-colour trap.** A model that renders a flat average
     colour can score ~14.6 dB on a background-dominated dataset
     — higher than a real but foggy model. Mean PSNR alone is
     NEVER a verdict here.
   - **Mask trap.** A foreground-masked PSNR sounds rigorous, but
     if the subject fills most of the frame, masked ≈ unmasked
     and even an empty model scores well on the mask. Check the
     foreground fraction before trusting a masked number.
   - **Worst-angle trap.** One bad angle can hide inside a good
     mean. Always name the worst view and its score.

## 7. Tuning ladder (general — doses are scene-dependent)

Start from defaults. Change ONE thing at a time, and record the
exact flags, counts, per-view PSNR, and visuals for every run, or
the campaign becomes uninterpretable.

1. **Foggy / washed-out renders** (subject recognisable but veiled
   by giant translucent sea/ground splats — the classic 3DGS fog):
   raise `--scale-loss-weight` in decade steps from 1e-8 (1e-5,
   1e-4, 1e-3, …). It prunes large splats aggressively. Watch the
   point count at every step: a crash to near-zero is collapse
   (too far), not a result. The response is NOT linear — on one
   scene the jump from 3e-4 to 1e-3 was a regime change (+3.4 dB),
   on others it would collapse. There is no universal dose.
2. **Too sparse after pruning:** the refine knobs —
   `--growth-grad-threshold` down, `--growth-select-fraction` up,
   `--growth-stop-iter` later (growth runs the full training
   instead of stopping halfway). Warning from observation: at a
   high scale-loss dose, aggressive growth can REDUCE net density
   (growth + pruning fight, pruning wins mid-run) while still
   improving PSNR through better placement. Density and PSNR are
   not the same axis; report both.
3. **Extra scene detail (texture, clutter) costs splats.** It
   helps only when splats are plentiful; at a heavily pruned
   regime it hurts badly, because the model cannot afford the
   extra detail. If a textured scene underperforms the clean one,
   simplify the scene before touching the optimiser.
4. **Resolution, framing, and clutter are scene decisions, not
   optimiser settings.** Test them as separate runs with the same
   flags, or you cannot attribute anything.
5. **Giant-splat count does not drive quality.** Runs with far
   more giants have beaten runs with far fewer, and stripping
   giants post-hoc (see `zanjic-church/strip_giants.py`) hurt in a
   renderer A/B and changed nothing on the phone. Do not chase
   this number.

Stop conditions: when the dose curve, texture, resolution, and
growth variants are all mapped and the remaining gap is a single
bad angle or uniform softness, you are at the method's ceiling for
that scene — say so, and ship the best model rather than re-running
the same points on the curve.

## 8. The exported PLY

Standard 3DGS PLY, binary little-endian, SH degree 3. Two parsing
gotchas, both observed:

- Brush writes properties **alphabetically with x/y/z last**.
  Loaders must read properties BY NAME, never by position.
- Scale is stored log-scale; opacity is a logit (sigmoid-sane
  range roughly [-6, +9] on real runs). A parser that reads the
  header wrong produces NaNs and absurd giant counts — if your
  giant fraction looks insane, your parser is misaligned (the
  classic cause: stopping the header read at `end_header` without
  consuming its trailing newline, shifting all data by one byte).

Brush 0.3.0 has **no render-a-PLY mode**: its CLI only trains from
a dataset (`--start-iter` resumes training). To inspect a finished
PLY you need a viewer (see §9) or your own renderer — the CPU EWA
renderer in `zanjic-church/render_ply.py` is a starting point, with
the caveat that it under-paints small splats, so only same-renderer
A/B comparisons are evidence from it.

## 9. Shipping a phone viewer

`viewer/` builds a standalone HTML viewer (three.js 0.180.0 +
Spark 2.4.0) that runs on a phone with no CDN and no install:
touch orbit, Reset + Spin buttons, a PLY file picker, and a
`?ply=<url>` override. Build it with:

```bash
python3 viewer/build_standalone.py --ply <model>.ply \
  --out <name>.html --spin-on
```

Lessons that cost real debugging time:

- **CDN-free is not optional for some phones.** If the target
  phone cannot reach jsDelivr/unpkg/cdnjs, embed the libraries as
  base64 → `blob:` module URLs via an injected importmap.
- **The error handler must be a classic script registered BEFORE
  the module script**, or a failed import shows eternal "Loading".
- **three 0.180.0's `three.module.js` is only a wrapper** importing
  `./three.core.js`. Relative imports cannot resolve from `blob:`
  URLs, so at build time rewrite that import to the bare specifier
  `three-core`, embed `three.core.js`, and map it in the importmap.
- **Never host the HTML on jsDelivr** — it serves `.html` as plain
  text, so the page never runs (binary PLYs are fine there).
- **Verification from a VM is unreliable** (org policy, localhost
  rejection, no CDN egress). The phone's status-line text and
  visual verdict are ground truth.
- If you build on a CDN-blocked VM, point the builder's four lib
  URLs at local files instead of the CDN (the Zanjic demos were
  built this way). Before shipping, check: no unfilled template
  tokens, JS syntax OK, embedded PLY byte-identical to the trained
  file.

## 10. Session and account hygiene (this environment)

Colab work here shares accounts with other agents. These rules
are not optional:

- Use a per-account token file plus an isolated registry file for
  every session. NEVER switch the VM-global active account, and
  NEVER touch the arxiv training rotation.
- Create a session ONLY on an account verified idle immediately
  beforehand (another chat may be using an account right now).
- On HTTP 503 / gpu-unavailable, switch AT MOST ONCE to another
  verified-idle account. No retry loops.
- Stop the session and delete its registry at teardown.
- Known-bad accounts (do not use): 5stardse (token refresh dead
  since 2026-11-05, needs a fresh sign-in); supervisor-only
  accounts are never training accounts.
- Expect reclamation; frame every run by cost and risk, never by
  promised stability.

## 11. Minimal end-to-end checklist

1. Dataset: images + `transforms.json`, held-out val split,
   smallest PNG > 16,387 bytes (else uncompressed PNGs).
2. Session: T4, verified-idle account, isolated registry.
3. Setup: Vulkan ICD (driver series from `nvidia-smi`) + 0.3.0
   binary + `--help` smoke test.
4. Train: 30,000 steps, eval + export every 5,000, detached.
5. Download exports + eval renders as base64, incrementally.
6. Evaluate: per-view PSNR vs GT, vertex count, visuals incl.
   worst view. Declare a verdict only on all three.
7. Tune per §7, one variable at a time, recording everything.
8. Ship: `viewer/build_standalone.py` demo, phone-verified.

If you follow this guide and still get a foggy model at defaults,
start at §7 step 1 — and read `zanjic-church/README.md` for a
worked example of the whole ladder, including every dead end, so
you do not re-walk it.
