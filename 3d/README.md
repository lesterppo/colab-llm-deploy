# 3D

Training a 3D scene on Colab, rather than serving a model or generating
a media file. Two scenes, plus the reusable phone viewer:

- [brush-lighthouse/](brush-lighthouse/) — train a lighthouse-island
  Gaussian splatting scene with Brush 0.3.0 on a free T4, one command:
  `python3 3d/brush-lighthouse/deploy_brush.py`
  (30,000 steps, ~6 min train, mean val PSNR 18.59 dB, proven 2026-11-04).
  Its viewer is the offline embedded-libraries build (see `viewer/`),
  which is the version that actually works on a phone.
- [zanjic-church/](zanjic-church/) — the same pipeline applied to the
  Žanjic island monastery (Mala Gospa, Boka Bay, Montenegro): scene
  scripts, the full Brush training campaign (GREEN at scale-loss
  1e-3: 26,447 points, mean val PSNR 16.96 dB; densified variant
  18,844 points, 18.65 dB — read that README's elimination record
  before re-running anything), trained PLYs, phone demos, and a
  handoff section for another agent. The procedural reconstruction
  (up to 327,377 points) still wins on outright sharpness.
- [viewer/](viewer/) — build a standalone splat-viewer HTML that works
  on a phone with no CDN access: three.js + Spark embedded as base64
  blob modules (including the three.core.js rewrite), a classic-script
  error handler that names the failing step, and a splat load that
  falls back to an embedded version instead of failing silently.
  `python3 3d/viewer/build_standalone.py --ply scene.ply --out scene.html`
  Start here for any new scene's presentation; do not re-derive the
  CDN viewer that failed four times on a real phone.
