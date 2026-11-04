"""Build one 3D scenario: a lighthouse island at dusk.

Two outputs from the SAME scene description, so they can be checked
against each other:
  1. dataset/  - 48 ray-traced views + transforms.json in nerfstudio
     format, ready for Brush (`brush <dataset> --export-path ...`).
  2. lighthouse_procedural.ply - a standard 3DGS PLY sampled from the
     same primitives (isotropic Gaussians, so rotation is irrelevant),
     loadable directly in the Brush viewer.

Scene (world units, Y up, ground y=0):
  sea disc, sandy island, grass, a striped lighthouse, 3 pine trees,
  rocks, a wooden pier and a small sailboat. Dusk sun from the west.
"""
import json, math, struct
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20261104)
SUN = np.array([0.50, 0.80, 0.30], float); SUN /= np.linalg.norm(SUN)

# ---------------------------------------------------------------- primitives
# Each prim is a dict; ray tracer and sampler both understand these types.
def disc(c, r, color, y=None, name=""):
    return dict(kind="disc", c=np.array(c, float), r=float(r),
                color=np.array(color, float), name=name)
def sphere(c, r, color, name="", emissive=False):
    return dict(kind="sphere", c=np.array(c, float), r=float(r),
                color=np.array(color, float), name=name, emissive=emissive)
def cyl(c, r, h, color, name="", emissive=False):  # c = base centre
    return dict(kind="cyl", c=np.array(c, float), r=float(r), h=float(h),
                color=np.array(color, float), name=name, emissive=emissive)
def cone(c, r, h, color, name=""):  # c = base centre, apex up
    return dict(kind="cone", c=np.array(c, float), r=float(r), h=float(h),
                color=np.array(color, float), name=name)
def box(c, hx, hy, hz, color, rot_y=0.0, name=""):
    return dict(kind="box", c=np.array(c, float), h=np.array([hx, hy, hz], float),
                color=np.array(color, float), rot_y=float(rot_y), name=name)
def tri(a, b, cc, color, name=""):
    return dict(kind="tri", a=np.array(a, float), b=np.array(b, float),
                c=np.array(cc, float), color=np.array(color, float), name=name)

STONE = (0.78, 0.75, 0.68); STONE_DK = (0.62, 0.58, 0.52)
TERRA = (0.62, 0.25, 0.18); DARK = (0.12, 0.10, 0.10)
PRIMS = []
# Adriatic sea + tiny rocky monastery island (Mala Gospa, ~700 m2 in life)
PRIMS.append(disc((0, 0.00, 0), 8.0, (0.13, 0.48, 0.60), name="sea"))
PRIMS.append(disc((0, 0.10, 0), 2.60, STONE_DK, name="island"))
PRIMS.append(disc((0, 0.18, 0), 2.10, STONE, name="court"))
# rocks around the island rim
for ang, rr, rad in [(0.5, 2.35, 0.38), (2.2, 2.40, 0.30), (3.9, 2.30, 0.42),
                     (5.3, 2.45, 0.28), (1.4, 2.55, 0.33)]:
    PRIMS.append(sphere((rr * math.cos(ang), 0.10, rr * math.sin(ang)), rad,
                        (0.45, 0.45, 0.48), name="rock"))
# fortified perimeter wall: 12 tangential segments, gap on the south-west
# for the jetty. Box local x maps to world angle -rot_y, so rot_y=-a-pi/2
# puts the long axis along the tangent at angle a.
for k in range(12):
    a = k * 2 * math.pi / 12
    if k == 8:  # jetty gap
        continue
    PRIMS.append(box((2.35 * math.cos(a), 0.18 + 0.275, 2.35 * math.sin(a)),
                     0.65, 0.275, 0.09, STONE_DK, rot_y=-a - math.pi / 2, name="wall"))
# church: single nave along x (west tower, east apse), stone walls
PRIMS.append(box((0, 0.18 + 0.55, 0), 1.10, 0.55, 0.65, STONE, name="nave"))
PRIMS.append(box((0, 0.18 + 1.10 + 0.06, 0), 1.18, 0.06, 0.78, TERRA, name="roof"))
PRIMS.append(box((0, 0.18 + 1.22, 0), 1.05, 0.05, 0.10, (0.45, 0.18, 0.13), name="ridge"))
# gable triangles at the west/east ends of the nave
PRIMS.append(tri((1.10, 1.28, -0.65), (1.10, 1.28, 0.65), (1.10, 1.68, 0), STONE, name="gable_e"))
PRIMS.append(tri((-1.10, 1.28, -0.65), (-1.10, 1.28, 0.65), (-1.10, 1.68, 0), STONE, name="gable_w"))
# semicircular east apse (cylinder half-embedded in the nave) + its roof
PRIMS.append(cyl((1.10, 0.18, 0), 0.55, 1.00, STONE, name="apse"))
PRIMS.append(cone((1.10, 1.18, 0), 0.60, 0.35, TERRA, name="apse_roof"))
# west bell tower + belfry + pyramid-ish roof + cross
PRIMS.append(box((-1.55, 0.18 + 0.85, 0), 0.38, 0.85, 0.38, STONE, name="tower"))
PRIMS.append(box((-1.55, 0.18 + 1.70 + 0.175, 0), 0.42, 0.175, 0.42, STONE, name="belfry"))
PRIMS.append(box((-1.55, 0.18 + 1.70 + 0.175, 0.43), 0.20, 0.20, 0.02, DARK, name="belfry_open"))
PRIMS.append(cone((-1.55, 0.18 + 2.05, 0), 0.55, 0.55, TERRA, name="tower_roof"))
PRIMS.append(box((-1.55, 0.18 + 2.60 + 0.11, 0), 0.03, 0.11, 0.03, (0.35, 0.28, 0.12), name="cross_v"))
PRIMS.append(box((-1.55, 0.18 + 2.60 + 0.13, 0), 0.14, 0.03, 0.03, (0.35, 0.28, 0.12), name="cross_h"))
# door (south face) + windows (dark rectangles on both long faces)
PRIMS.append(box((0.20, 0.18 + 0.35, 0.66), 0.18, 0.35, 0.02, DARK, name="door"))
for wx in (-0.60, 0.0, 0.60):
    PRIMS.append(box((wx, 0.18 + 0.70, 0.66), 0.10, 0.22, 0.02, DARK, name="window"))
    PRIMS.append(box((wx, 0.18 + 0.70, -0.66), 0.10, 0.22, 0.02, DARK, name="window"))
# cypress + olive trees on the island
for tx, tz in [(-1.95, 1.15), (1.85, -1.20), (1.95, 1.05)]:
    PRIMS.append(cyl((tx, 0.18, tz), 0.07, 0.30, (0.35, 0.22, 0.12), name="trunk"))
    PRIMS.append(cone((tx, 0.45, tz), 0.22, 1.10, (0.10, 0.32, 0.16), name="cypress"))
for tx, tz in [(-2.00, -0.90), (0.90, 1.70)]:
    PRIMS.append(cyl((tx, 0.18, tz), 0.07, 0.35, (0.35, 0.22, 0.12), name="trunk"))
    PRIMS.append(sphere((tx, 0.75, tz), 0.35, (0.45, 0.55, 0.35), name="olive"))
# stone jetty through the wall gap, heading south-west
PRIMS.append(box((-2.85, 0.12, -1.35), 0.90, 0.08, 0.30, STONE_DK, rot_y=0.6, name="jetty"))
# small white motorboat out on the bay
bx, bz, th = 5.5, 2.0, 0.4
def boat_pt(x, y, z):
    ct, st = math.cos(th), math.sin(th)
    return (bx + x * ct + z * st, y, bz - x * st + z * ct)
PRIMS.append(box(boat_pt(0, 0.12, 0), 0.50, 0.14, 0.22, (0.95, 0.95, 0.92),
                  rot_y=th, name="hull"))
PRIMS.append(box(boat_pt(0, 0.26, 0), 0.20, 0.10, 0.16, (0.15, 0.15, 0.18),
                  rot_y=th, name="cabin"))
def shade(color, normal, emissive=False):
    if emissive:
        return np.clip(color, 0, 1)
    d = max(0.0, float(np.dot(normal, SUN)))
    return np.clip(color * (0.45 + 0.55 * d), 0, 1)

# ---------------------------------------------------------------- ray tracer
def render_view(cam, look, fovx=0.70, W=400, H=400):
    cam = np.array(cam, float); look = np.array(look, float)
    Z = cam - look; Z /= np.linalg.norm(Z)
    X = np.cross([0, 1, 0], Z); X /= np.linalg.norm(X)
    Y = np.cross(Z, X)
    us = (np.arange(W)[None, :] + 0.5) / W * 2 - 1
    vs = 1 - (np.arange(H)[:, None] + 0.5) / H * 2
    tan = math.tan(fovx / 2)
    D = X[None, None, :] * (us[..., None] * tan) + Y[None, None, :] * (vs[..., None] * tan) - Z
    D /= np.linalg.norm(D, axis=-1, keepdims=True)
    O = np.broadcast_to(cam, D.shape)
    INF = 1e9
    best_t = np.full(D.shape[:2], INF); best_col = np.zeros(D.shape); best_n = np.zeros(D.shape)
    Ox, Oy, Oz = O[..., 0], O[..., 1], O[..., 2]
    Dx, Dy, Dz = D[..., 0], D[..., 1], D[..., 2]

    def consider(t, n, col, mask):
        nonlocal best_t, best_col, best_n
        hit = mask & (t > 1e-3) & (t < best_t)
        if np.any(hit):
            best_t = np.where(hit, t, best_t)
            best_col = np.where(hit[..., None], col, best_col)
            best_n = np.where(hit[..., None], n, best_n)

    with np.errstate(all="ignore"):
        for p in PRIMS:
            col, em = p["color"], p.get("emissive", False)
            if p["kind"] == "disc":
                y = p["c"][1]
                t = np.where(np.abs(Dy) > 1e-9, (y - Oy) / np.where(np.abs(Dy) > 1e-9, Dy, 1), INF)
                px, pz = Ox + t * Dx, Oz + t * Dz
                m = (px - p["c"][0]) ** 2 + (pz - p["c"][2]) ** 2 <= p["r"] ** 2
                consider(t, np.broadcast_to([0, 1, 0], D.shape),
                         np.broadcast_to(shade(col, [0, 1, 0], em), D.shape), m)
            elif p["kind"] == "sphere":
                oc = O - p["c"]; b = np.sum(oc * D, axis=-1)
                c2 = np.sum(oc * oc, axis=-1) - p["r"] ** 2
                discr = b * b - c2
                t = np.where(discr > 0, -b - np.sqrt(np.maximum(discr, 0)), INF)
                P = O + t[..., None] * D
                n = (P - p["c"]) / p["r"]
                consider(t, n, shade(col, n, em) if False else
                         np.clip(col * (0.45 + 0.55 * np.maximum(0, np.sum(n * SUN, axis=-1)))[..., None] if not em else col, 0, 1),
                         discr > 0)
            elif p["kind"] == "cyl":
                cx, cy, cz, r, h = p["c"][0], p["c"][1], p["c"][2], p["r"], p["h"]
                # side
                a = Dx ** 2 + Dz ** 2
                b = 2 * ((Ox - cx) * Dx + (Oz - cz) * Dz)
                c2 = (Ox - cx) ** 2 + (Oz - cz) ** 2 - r ** 2
                discr = b * b - 4 * a * c2
                t = np.where((discr > 0) & (a > 1e-12),
                             (-b - np.sqrt(np.maximum(discr, 0))) / np.where(a > 1e-12, 2 * a, 1), INF)
                Py = Oy + t * Dy
                m = (discr > 0) & (Py >= cy) & (Py <= cy + h)
                P = O + t[..., None] * D
                n = np.stack([(P[..., 0] - cx) / r, np.zeros_like(P[..., 0]), (P[..., 2] - cz) / r], axis=-1)
                consider(t, n, np.clip(col * (0.45 + 0.55 * np.maximum(0, np.sum(n * SUN, axis=-1)))[..., None], 0, 1), m)
                # caps
                for cap_y, cap_n in [(cy, [0, -1, 0]), (cy + h, [0, 1, 0])]:
                    t2 = np.where(np.abs(Dy) > 1e-9, (cap_y - Oy) / np.where(np.abs(Dy) > 1e-9, Dy, 1), INF)
                    px, pz = Ox + t2 * Dx, Oz + t2 * Dz
                    m2 = (px - cx) ** 2 + (pz - cz) ** 2 <= r ** 2
                    consider(t2, np.broadcast_to(cap_n, D.shape),
                             np.broadcast_to(shade(col, cap_n, em), D.shape), m2)
            elif p["kind"] == "cone":
                cx, cy, cz, r, h = p["c"][0], p["c"][1], p["c"][2], p["r"], p["h"]
                s = r / h; K = r - s * (Oy - cy)
                A = Dx ** 2 + Dz ** 2 - s ** 2 * Dy ** 2
                B = 2 * ((Ox - cx) * Dx + (Oz - cz) * Dz + K * s * Dy)
                C = (Ox - cx) ** 2 + (Oz - cz) ** 2 - K ** 2
                discr = B * B - 4 * A * C
                t = np.where((discr > 0) & (np.abs(A) > 1e-12),
                             (-B - np.sqrt(np.maximum(discr, 0))) / np.where(np.abs(A) > 1e-12, 2 * A, 1), INF)
                Py = Oy + t * Dy
                m = (discr > 0) & (Py >= cy) & (Py <= cy + h)
                P = O + t[..., None] * D
                rad = np.sqrt(np.maximum((P[..., 0] - cx) ** 2 + (P[..., 2] - cz) ** 2, 1e-12))
                n = np.stack([(P[..., 0] - cx) / rad, np.full_like(rad, s), (P[..., 2] - cz) / rad], axis=-1)
                n /= np.linalg.norm(n, axis=-1, keepdims=True)
                consider(t, n, np.clip(col * (0.45 + 0.55 * np.maximum(0, np.sum(n * SUN, axis=-1)))[..., None], 0, 1), m)
                t2 = np.where(np.abs(Dy) > 1e-9, (cy - Oy) / np.where(np.abs(Dy) > 1e-9, Dy, 1), INF)
                px, pz = Ox + t2 * Dx, Oz + t2 * Dz
                consider(t2, np.broadcast_to([0, -1, 0], D.shape),
                         np.broadcast_to(shade(col, [0, -1, 0], em), D.shape),
                         (px - cx) ** 2 + (pz - cz) ** 2 <= r ** 2)
            elif p["kind"] == "box":
                th2 = p["rot_y"]; ct, st = math.cos(-th2), math.sin(-th2)
                lx, lz = (Ox - p["c"][0]) * ct - (Oz - p["c"][2]) * st, (Ox - p["c"][0]) * st + (Oz - p["c"][2]) * ct
                lDx, lDz = Dx * ct - Dz * st, Dx * st + Dz * ct
                lo = np.stack([lx, Oy - p["c"][1], lz], axis=-1)
                ld = np.stack([lDx, Dy, lDz], axis=-1)
                tmin, tmax = np.full(D.shape[:2], -INF), np.full(D.shape[:2], INF)
                nmin = np.zeros(D.shape)
                for ax in range(3):
                    inv = np.where(np.abs(ld[..., ax]) > 1e-9, 1 / np.where(np.abs(ld[..., ax]) > 1e-9, ld[..., ax], 1), INF)
                    t1, t2 = (-p["h"][ax] - lo[..., ax]) * inv, (p["h"][ax] - lo[..., ax]) * inv
                    swap = t1 > t2; t1, t2 = np.where(swap, t2, t1), np.where(swap, t1, t2)
                    ax_n = np.zeros(D.shape); ax_n[..., ax] = np.where(ld[..., ax] > 0, -1, 1)
                    upd = t1 > tmin
                    tmin = np.where(upd, t1, tmin); nmin = np.where(upd[..., None], ax_n, nmin)
                    tmax = np.minimum(tmax, t2)
                hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 1e-3)
                # rotate normal back by +th
                ct2, st2 = math.cos(th2), math.sin(th2)
                nw = np.stack([nmin[..., 0] * ct2 - nmin[..., 2] * st2, nmin[..., 1],
                               nmin[..., 0] * st2 + nmin[..., 2] * ct2], axis=-1)
                consider(np.where(hit, tmin, INF), nw,
                         np.clip(col * (0.45 + 0.55 * np.maximum(0, np.sum(nw * SUN, axis=-1)))[..., None], 0, 1), hit)
            elif p["kind"] == "tri":
                e1, e2 = p["b"] - p["a"], p["c"] - p["a"]
                pv = np.cross(D, e2); det = np.sum(pv * e1, axis=-1)
                inv = np.where(np.abs(det) > 1e-9, 1 / np.where(np.abs(det) > 1e-9, det, 1), 0)
                tv = O - p["a"]; u = np.sum(tv * pv, axis=-1) * inv
                qv = np.cross(tv, e1); v = np.sum(D * qv, axis=-1) * inv
                t = np.sum(qv * e2, axis=-1) * inv
                nrm = np.cross(e1, e2); nrm /= np.linalg.norm(nrm)
                if nrm[2] < 0: nrm = -nrm
                consider(t, np.broadcast_to(nrm, D.shape),
                         np.broadcast_to(shade(col, nrm, em), D.shape),
                         (np.abs(det) > 1e-9) & (u >= 0) & (v >= 0) & (u + v <= 1))
    # background: dusk sky above, dark sea below
    sky_t = np.clip(D[..., 1] * 1.6, 0, 1)[..., None]
    sky = (1 - sky_t) * np.array([0.78, 0.87, 0.96]) + sky_t * np.array([0.25, 0.55, 0.85])
    sea_bg = np.array([0.10, 0.38, 0.50])
    bg = np.where((D[..., 1] > 0)[..., None], sky, sea_bg)
    img = np.where((best_t < INF)[..., None], best_col, bg)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def cam_pose(cam, look):
    cam = np.array(cam, float); look = np.array(look, float)
    Z = cam - look; Z /= np.linalg.norm(Z)
    X = np.cross([0, 1, 0], Z); X /= np.linalg.norm(X)
    Y = np.cross(Z, X)
    m = [[X[0], Y[0], Z[0], cam[0]], [X[1], Y[1], Z[1], cam[1]],
         [X[2], Y[2], Z[2], cam[2]], [0, 0, 0, 1]]
    return m

# CLEAN variant (2026-11-04, Peter delegated the pick after the GREEN framing
# retrain failed at 10.89 dB): the GREEN framing (R=4.2, FOV 0.70) kept, but the
# LIVELY prims (wave caps, flowers, extra trees, birds, people, flag, second
# boat) and lively colours are REMOVED, restoring the exact pre-lively scene
# the original 12.98 dB run trained on. Vs the failed green run this isolates
# exactly one bundle -- lively clutter -- with identical cameras, so the PSNR
# comparison is apples-to-apples. Vs the original run it isolates framing.
# GREEN variant (2026-11-04, on Peter's "Make the training GREEN"):
# the original dataset (R=5.5, FOV 0.8727 rad) left the church small in
# frame and most pixels flat sea/sky, and Brush answered with giant
# translucent sea/ground splats that fogged the church (mean val PSNR
# 12.98 dB). This variant moves the orbit in to R=4.2 and narrows the
# FOV to 0.70 rad, so the church covers ~1.7x more width and ~3x the
# pixels, and flat sea/sky shrink as a frame fraction. Everything else
# (scene prims, 48 train + 8 val, 400x400, uncompressed PNGs) is
# unchanged, so the PSNR delta is attributable to framing.
def build_dataset():
    look = (0, 0.9, 0); R = 4.2
    def views(n, offset_deg, elev_fn, prefix):
        out = []
        for i in range(n):
            az = math.radians(offset_deg + i * 360.0 / n)
            el = math.radians(elev_fn(i, n))
            y = 0.9 + R * math.sin(el)
            r = R * math.cos(el)
            cam = (r * math.cos(az), y, r * math.sin(az))
            fn = f"images/{prefix}_{i:03d}.png"
            img = render_view(cam, look)
            # Brush 0.3.0's image-header reader (scene.rs get_image_data)
            # does read_exact over a 16,387-byte buffer first: a PNG smaller
            # than that fails the whole dataset load with "early eof".
            # Flat-colour renders compress to ~14 KB, so save uncompressed
            # (~480 KB each) — pixel data identical, header parse succeeds.
            Image.fromarray(img).save(
                ROOT / "dataset" / fn if prefix == "train" else ROOT / "dataset_val" / fn,
                compress_level=0)
            out.append({"file_path": fn, "transform_matrix": cam_pose(cam, look)})
            print(f"  {fn} cam={tuple(round(v, 2) for v in cam)}", flush=True)
        return out
    print("train views:")
    train = views(48, 0, lambda i, n: 18 + 8 * math.sin(i * 2 * math.pi / n * 2), "train")
    print("val views:")
    val = views(8, 3.75, lambda i, n: 20, "val")
    for name, frames, folder in [("transforms.json", train, ROOT / "dataset"),
                                 ("transforms_val.json", val, ROOT / "dataset_val")]:
        # Brush reads the json in the same folder as the images it references;
        # keep a single dataset dir: copy val frames into dataset/transforms_val.json
        pass
    base = {"camera_angle_x": 0.70, "frames": train}
    (ROOT / "dataset" / "transforms.json").write_text(json.dumps(base, indent=2))
    (ROOT / "dataset" / "transforms_val.json").write_text(
        json.dumps({"camera_angle_x": 0.70,
                    "frames": [{"file_path": f"images/val_{i:03d}.png",
                                "transform_matrix": f["transform_matrix"]} for i, f in enumerate(val)]}, indent=2))
    # move val images into dataset/images/
    for i in range(len(val)):
        src = ROOT / "dataset_val" / f"images/val_{i:03d}.png"
        src.replace(ROOT / "dataset" / f"images/val_{i:03d}.png")
    print(f"dataset: {len(train)} train + {len(val)} val, 400x400")

# ---------------------------------------------------------------- PLY sampler
def sample_prim(p, scale):
    """Return (points, normals) sampled on the primitive surface."""
    if p["kind"] == "disc":
        col_area = math.pi * p["r"] ** 2
    elif p["kind"] == "sphere":
        col_area = 4 * math.pi * p["r"] ** 2
    elif p["kind"] == "cyl":
        col_area = 2 * math.pi * p["r"] * p["h"] + 2 * math.pi * p["r"] ** 2
    elif p["kind"] == "cone":
        col_area = math.pi * p["r"] * math.hypot(p["r"], p["h"]) + math.pi * p["r"] ** 2
    elif p["kind"] == "box":
        hh = p["h"]
        col_area = 8 * (hh[0] * hh[1] + hh[0] * hh[2] + hh[1] * hh[2])
    else:
        col_area = 0.5 * np.linalg.norm(np.cross(p["b"] - p["a"], p["c"] - p["a"]))
    n = int(max(120, col_area / (scale * 1.3) ** 2))
    pts, nrms = [], []
    if p["kind"] == "disc":
        a = RNG.random(n) * 2 * math.pi; rr = p["r"] * np.sqrt(RNG.random(n))
        pts = np.stack([p["c"][0] + rr * np.cos(a), np.full(n, p["c"][1]), p["c"][2] + rr * np.sin(a)], axis=1)
        nrms = np.tile([0, 1, 0], (n, 1))
    elif p["kind"] == "sphere":
        d = RNG.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
        pts, nrms = p["c"] + d * p["r"], d
    elif p["kind"] == "cyl":
        side_frac = (2 * math.pi * p["r"] * p["h"]) / col_area
        ns = int(n * side_frac)
        a = RNG.random(ns) * 2 * math.pi; y = p["c"][1] + RNG.random(ns) * p["h"]
        side = np.stack([p["c"][0] + p["r"] * np.cos(a), y, p["c"][2] + p["r"] * np.sin(a)], axis=1)
        sn = np.stack([np.cos(a), np.zeros(ns), np.sin(a)], axis=1)
        nc = n - ns; a2 = RNG.random(nc) * 2 * math.pi; rr = p["r"] * np.sqrt(RNG.random(nc))
        top = RNG.random(nc) < 0.5
        capy = np.where(top, p["c"][1] + p["h"], p["c"][1])
        capn = np.where(top[:, None], [0, 1, 0], [0, -1, 0])
        caps = np.stack([p["c"][0] + rr * np.cos(a2), capy, p["c"][2] + rr * np.sin(a2)], axis=1)
        pts, nrms = np.vstack([side, caps]), np.vstack([sn, capn])
    elif p["kind"] == "cone":
        side_frac = (math.pi * p["r"] * math.hypot(p["r"], p["h"])) / col_area
        ns = int(n * side_frac)
        u = RNG.random(ns); s = 1 - np.sqrt(1 - u)  # height fraction, area-weighted
        a = RNG.random(ns) * 2 * math.pi; rr = p["r"] * (1 - s)
        side = np.stack([p["c"][0] + rr * np.cos(a), p["c"][1] + s * p["h"], p["c"][2] + rr * np.sin(a)], axis=1)
        sn = np.stack([np.cos(a), np.full(ns, p["r"] / p["h"]), np.sin(a)], axis=1)
        sn /= np.linalg.norm(sn, axis=1, keepdims=True)
        nc = n - ns; a2 = RNG.random(nc) * 2 * math.pi; rr2 = p["r"] * np.sqrt(RNG.random(nc))
        caps = np.stack([p["c"][0] + rr2 * np.cos(a2), np.full(nc, p["c"][1]), p["c"][2] + rr2 * np.sin(a2)], axis=1)
        pts, nrms = np.vstack([side, caps]), np.vstack([sn, np.tile([0, -1, 0], (nc, 1))])
    elif p["kind"] == "box":
        faces = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]
        areas = [4 * p["h"][(ax + 1) % 3] * p["h"][(ax + 2) % 3] for ax, _ in faces]
        counts = np.maximum(20, (np.array(areas) / sum(areas) * n).astype(int))
        th = p["rot_y"]; ct, st = math.cos(th), math.sin(th)
        allp, alln = [], []
        for (ax, sgn), cnt in zip(faces, counts):
            u_ax, v_ax = (ax + 1) % 3, (ax + 2) % 3
            local = np.zeros((cnt, 3))
            local[:, ax] = sgn * p["h"][ax]
            local[:, u_ax] = (RNG.random(cnt) * 2 - 1) * p["h"][u_ax]
            local[:, v_ax] = (RNG.random(cnt) * 2 - 1) * p["h"][v_ax]
            w = np.stack([local[:, 0] * ct + local[:, 2] * st, local[:, 1],
                          -local[:, 0] * st + local[:, 2] * ct], axis=1) + p["c"]
            ln = np.zeros((cnt, 3)); ln[:, ax] = sgn
            wn = np.stack([ln[:, 0] * ct + ln[:, 2] * st, ln[:, 1],
                           -ln[:, 0] * st + ln[:, 2] * ct], axis=1)
            allp.append(w); alln.append(wn)
        pts, nrms = np.vstack(allp), np.vstack(alln)
    elif p["kind"] == "tri":
        r1, r2 = RNG.random(n), RNG.random(n)
        sq = np.sqrt(r1); w0, w1, w2 = 1 - sq, sq * (1 - r2), sq * r2
        pts = w0[:, None] * p["a"] + w1[:, None] * p["b"] + w2[:, None] * p["c"]
        nn = np.cross(p["b"] - p["a"], p["c"] - p["a"]); nn /= np.linalg.norm(nn)
        nrms = np.tile(nn, (n, 1))
    return pts, nrms

# Lively + sharper pass (2026-11-04, on Peter's "more lively and sharper"):
# scales halved AGAIN vs the 75k version, so density ~4x more (~300k
# points) and splat radius ~half again. New lively prims use the same
# default scale (0.035) unless named here.
SCALE = {"sea": 0.055, "island": 0.0325, "court": 0.0325, "rock": 0.0175, "wall": 0.015,
         "nave": 0.015, "roof": 0.015, "ridge": 0.0125, "gable_e": 0.0125, "gable_w": 0.0125,
         "apse": 0.015, "apse_roof": 0.0125, "tower": 0.015, "belfry": 0.0125,
         "belfry_open": 0.01, "tower_roof": 0.0125, "cross_v": 0.01, "cross_h": 0.01,
         "door": 0.01, "window": 0.01, "trunk": 0.0125, "cypress": 0.015,
         "olive": 0.0175, "jetty": 0.0175, "hull": 0.015, "cabin": 0.0125,
         "wave": 0.02, "flower": 0.02, "bird": 0.02, "person": 0.015, "head": 0.015,
         "flagpole": 0.0125, "flag": 0.015, "hull2": 0.015, "wake": 0.02}

def build_ply():
    SH_C0 = 0.28209479177387814
    all_pts, all_col, all_scale = [], [], []
    for p in PRIMS:
        if p["name"].startswith("lh_"):
            sc = 0.05
        elif p["name"] in ("pine1", "pine2"):
            sc = 0.06
        else:
            sc = SCALE.get(p["name"], 0.07)
        pts, nrms = sample_prim(p, sc)
        cols = np.array([shade(p["color"], nn, p.get("emissive", False)) for nn in nrms])
        all_pts.append(pts); all_col.append(cols); all_scale.append(np.full(len(pts), sc))
    P = np.vstack(all_pts); C = np.vstack(all_col); S = np.concatenate(all_scale)
    print(f"PLY points: {len(P)}  bounds x[{P[:,0].min():.2f},{P[:,0].max():.2f}] "
          f"y[{P[:,1].min():.2f},{P[:,1].max():.2f}] z[{P[:,2].min():.2f},{P[:,2].max():.2f}]")
    N = len(P)
    props = [("x", "float"), ("y", "float"), ("z", "float"),
             ("nx", "float"), ("ny", "float"), ("nz", "float"),
             ("f_dc_0", "float"), ("f_dc_1", "float"), ("f_dc_2", "float")]
    props += [(f"f_rest_{i}", "float") for i in range(45)]
    props += [("opacity", "float"), ("scale_0", "float"), ("scale_1", "float"), ("scale_2", "float"),
              ("rot_0", "float"), ("rot_1", "float"), ("rot_2", "float"), ("rot_3", "float")]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {N}"]
    header += [f"property {t} {n}" for n, t in props]
    header += ["end_header"]
    arr = np.zeros(N, dtype=[(n, "<f4") for n, _ in props])
    arr["x"], arr["y"], arr["z"] = P[:, 0], P[:, 1], P[:, 2]
    # normals are not needed by splat renderers; write the sampled normal anyway
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = ((C - 0.5) / SH_C0).T
    arr["opacity"] = np.log(0.92 / 0.08)
    ls = np.log(S)
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = ls, ls, ls
    arr["rot_0"], arr["rot_3"] = 1.0, 0.0
    out = ROOT / "zanjic_procedural.ply"
    with open(out, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        arr.tofile(f)
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
    (ROOT / "manifest.json").write_text(json.dumps({
        "scene": "Zanjic (Mala Gospa) island monastery, Montenegro (synthetic, Y-up, sea y=0, church at origin)",
        "points": int(N), "bounds": {"x": [float(P[:, 0].min()), float(P[:, 0].max())],
        "y": [float(P[:, 1].min()), float(P[:, 1].max())],
        "z": [float(P[:, 2].min()), float(P[:, 2].max())]},
        "dataset": "dataset/transforms.json (48 train + 8 val, 400x400, camera_angle_x 0.70, GREEN variant R=4.2)",
        "ply": "zanjic_procedural.ply (standard 3DGS PLY, isotropic Gaussians, SH degree 0)",
    }, indent=2))
    return P, C

if __name__ == "__main__":
    build_dataset()
    build_ply()
