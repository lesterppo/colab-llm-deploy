#!/usr/bin/env python3
"""CPU Gaussian-splat renderer for A/B testing PLY files (e.g. stripped
vs original). Standard 3DGS EWA splatting: project, 2D covariance,
SH degree-3 colour, front-to-back compositing over black. It will NOT
match Brush's own rasterizer pixel-for-pixel, so PSNRs here are NOT
comparable to Brush eval PSNRs -- but stripped-vs-original rendered
with the SAME code, cameras and GT is a controlled comparison.

Usage: python3 render_ply.py <ply> <outdir>  (renders 8 val views of the
clean Zanjic scene: R=4.2, elev 20deg, offset 3.75deg, FOV 0.70, 400x400)
"""
import sys, math, struct
import numpy as np
from PIL import Image

W = H = 400
FOVX = 0.70
FX = (W / 2) / math.tan(FOVX / 2)
FY = (H / 2) / math.tan(FOVX / 2)
CX, CY = W / 2, H / 2

def load_ply(path):
    with open(path, 'rb') as f:
        data = f.read()
    end = data.index(b'end_header') + len(b'end_header') + 1
    header = data[:end].decode('ascii')
    props = [l.split()[2] for l in header.splitlines() if l.startswith('property float')]
    count = int([l for l in header.splitlines() if l.startswith('element vertex')][0].split()[2])
    vals = np.frombuffer(data[end:], dtype='<f4').reshape(count, len(props)).astype(np.float64)
    return props, vals

def quat_to_R(q):
    # 3DGS convention: rot_0 = w, rot_1..3 = x, y, z
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

SH_C0 = 0.28209479177387814
def sh_color(f_dc, f_rest, dirs):
    # dirs: (N,3) unit vectors from splat toward camera. Returns (N,3).
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    # l=1 (3), l=2 (5), l=3 (7) basis, same order as 3DGS sh.py.
    # 3DGS PLY f_rest layout is BASIS-major: f_rest[k*3 + c] = basis k,
    # channel c (the save/load pair transposes (N,3,15) <-> (N,15,3)).
    basis = [
        -0.4886025119029199 * y,
        0.4886025119029199 * z,
        -0.4886025119029199 * x,
        1.0925484305920792 * xy,
        -1.0925484305920792 * yz,
        0.3153916535399475 * (2 * zz - xx - yy),
        -1.0925484305920792 * xz,
        0.5462742152960396 * (xx - yy),
        -0.5900435899266435 * y * (3 * xx - yy),
        2.890611442640554 * xy * z,
        -0.4570457994644658 * y * (4 * zz - xx - yy),
        0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy),
        -0.4570457994644658 * x * (4 * zz - xx - yy),
        1.445305721320277 * z * (xx - yy),
        -0.5900435899266435 * x * (xx - 3 * yy),
    ]
    col = SH_C0 * f_dc.copy()
    for c in range(3):
        for k in range(15):
            col[:, c] += f_rest[:, k * 3 + c] * basis[k]
    return col

def render_view(props, vals, cam, look):
    ix = {p: i for i, p in enumerate(props)}
    xyz = vals[:, [ix['x'], ix['y'], ix['z']]]
    op = 1 / (1 + np.exp(-vals[:, ix['opacity']]))
    scale = np.exp(vals[:, [ix['scale_0'], ix['scale_1'], ix['scale_2']]])
    R = quat_to_R(vals[:, [ix['rot_0'], ix['rot_1'], ix['rot_2'], ix['rot_3']]])
    f_dc = vals[:, [ix['f_dc_0'], ix['f_dc_1'], ix['f_dc_2']]]
    f_rest = vals[:, [ix[f'f_rest_{k}'] for k in range(45)]] if 'f_rest_0' in ix else np.zeros((len(vals), 45))
    # view matrix (camera looks down -Z in cam space, y up)
    cam = np.array(cam, float)
    look = np.array(look, float)
    Z = cam - look
    Z /= np.linalg.norm(Z)
    X = np.cross([0, 1, 0], Z)
    X /= np.linalg.norm(X)
    Y = np.cross(Z, X)
    Wmat = np.stack([X, Y, Z])  # rows: cam axes in world coords
    rel = xyz - cam
    cam_xyz = rel @ Wmat.T
    depth = -cam_xyz[:, 2]
    valid = depth > 0.1
    # SH colour needs world-space dir from splat to camera
    dirs = cam - xyz
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    col = np.clip(sh_color(f_dc, f_rest, dirs), 0, 1)
    # 3D covariance
    S = scale[:, :, None] * np.eye(3)[None, :, :]
    cov3 = R @ S @ np.transpose(S, (0, 2, 1)) @ np.transpose(R, (0, 2, 1))
    # project
    px = FX * cam_xyz[:, 0] / depth + CX
    # Image rows increase downward while cam-space +Y is up, and the
    # ray tracer's vs has row 0 at +Y: flip Y so the render matches.
    py = CY - FY * cam_xyz[:, 1] / depth
    # Jacobian of perspective projection
    J = np.zeros((len(vals), 2, 3))
    J[:, 0, 0] = FX / depth
    J[:, 0, 2] = -FX * cam_xyz[:, 0] / depth ** 2
    J[:, 1, 1] = FY / depth
    J[:, 1, 2] = -FY * cam_xyz[:, 1] / depth ** 2
    cov2 = J @ (Wmat @ cov3 @ Wmat.T) @ np.transpose(J, (0, 2, 1))
    cov2[:, 0, 0] += 0.3
    cov2[:, 1, 1] += 0.3
    det = cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] * cov2[:, 1, 0]
    conic = np.empty_like(cov2)
    conic[:, 0, 0] = cov2[:, 1, 1] / det
    conic[:, 0, 1] = -cov2[:, 0, 1] / det
    conic[:, 1, 0] = -cov2[:, 1, 0] / det
    conic[:, 1, 1] = cov2[:, 0, 0] / det
    eig = np.linalg.eigvalsh(cov2)
    radius = np.ceil(3 * np.sqrt(np.maximum(eig[:, -1], 0))).astype(int)
    # Front-to-back compositing: nearest splat first, so T attenuates
    # correctly. (Far-to-near here inverts occlusion and darkens the
    # render with background splats — caught by the smoke test.)
    order = np.argsort(depth, kind='stable')
    img = np.zeros((H, W, 3))
    T = np.ones((H, W))
    for idx in order:
        if not valid[idx] or op[idx] < 0.003:
            continue
        r = radius[idx]
        if r <= 0 or r > 400:
            r = min(max(r, 1), 400)
        x0, x1 = max(0, int(px[idx]) - r), min(W, int(px[idx]) + r + 1)
        y0, y1 = max(0, int(py[idx]) - r), min(H, int(py[idx]) + r + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        xs = np.arange(x0, x1)[None, :] - px[idx]
        ys = np.arange(y0, y1)[:, None] - py[idx]
        c = conic[idx]
        maha = c[0, 0] * xs ** 2 + (c[0, 1] + c[1, 0]) * xs * ys + c[1, 1] * ys ** 2
        a = op[idx] * np.exp(-0.5 * maha)
        a[maha > 9] = 0
        if not np.any(a > 0.001):
            continue
        t = T[y0:y1, x0:x1]
        w = t * a
        img[y0:y1, x0:x1] += w[:, :, None] * col[idx][None, None, :]
        T[y0:y1, x0:x1] = t * (1 - a)
    return np.clip(img, 0, 1)

def main():
    ply_path, outdir = sys.argv[1], sys.argv[2]
    import os
    os.makedirs(outdir, exist_ok=True)
    props, vals = load_ply(ply_path)
    print(f'loaded {len(vals)} splats from {ply_path}')
    for i in range(8):
        az = math.radians(3.75 + i * 45.0)
        el = math.radians(20)
        Rr = 4.2
        y = 0.9 + Rr * math.sin(el)
        r = Rr * math.cos(el)
        cam = (r * math.cos(az), y, r * math.sin(az))
        img = render_view(props, vals, cam, (0, 0.9, 0))
        Image.fromarray((img * 255).astype(np.uint8)).save(f'{outdir}/val_{i:03d}.png')
        print(f'  val_{i:03d} done', flush=True)

if __name__ == '__main__':
    main()
