#!/usr/bin/env python3
"""Strip giant splats from a Brush-trained PLY (post-process, no retrain).

Removes every splat whose max exp(scale) exceeds THRESH (default 0.5),
rewrites the PLY header vertex count, keeps everything else byte-identical
in layout. Used 2026-11-05 on the GREEN 1e-3 model: removed 163 giants
(max-scale range 0.51-10.01), 26,447 -> 26,284 splats.

Verdict (do not re-run expecting a different outcome): stripping HURT in
the CPU renderer A/B (mean PSNR vs GT 8.41 stripped vs 9.54 original, all
8 views worse) and Peter's phone comparison found the stripped demo
SIMILARLY BLURRY to the original GREEN demo. Stripping is CLOSED as a
blur fix; this script is kept only so the experiment is reproducible.

Usage: python3 strip_giants.py in.ply out.ply [--thresh 0.5]
"""
import sys, numpy as np

def main():
    inp, outp = sys.argv[1], sys.argv[2]
    thresh = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    with open(inp, 'rb') as fh:
        hdr = b''
        while not hdr.endswith(b'end_header\n'):
            hdr += fh.read(1)
        lines = hdr.decode().splitlines()
        props = [l.split()[2] for l in lines if l.startswith('property float')]
        n = int([l for l in lines if l.startswith('element vertex')][0].split()[2])
        data = np.fromfile(fh, dtype=np.dtype([(p, '<f4') for p in props]), count=n)
    s = np.stack([data['scale_0'], data['scale_1'], data['scale_2']], axis=1)
    keep = np.exp(s).max(axis=1) <= thresh
    new_hdr = '\n'.join(
        l if not l.startswith('element vertex') else f'element vertex {int(keep.sum())}'
        for l in lines) + '\n'
    with open(outp, 'wb') as fh:
        fh.write(new_hdr.encode())
        data[keep].tofile(fh)
    print(f'{inp}: {n} -> {int(keep.sum())} splats, removed {int((~keep).sum())} giants (thresh {thresh})')

if __name__ == '__main__':
    main()
