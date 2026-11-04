# 3D

Training a 3D scene on Colab, rather than serving a model or generating
a media file. One recipe so far:

- [brush-lighthouse/](brush-lighthouse/) — train a lighthouse-island
  Gaussian splatting scene with Brush 0.3.0 on a free T4, one command:
  `python3 3d/brush-lighthouse/deploy_brush.py`
  (30,000 steps, ~6 min train, mean val PSNR 18.59 dB, proven 2026-11-04).
