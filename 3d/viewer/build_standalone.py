#!/usr/bin/env python3
"""Build a standalone Gaussian-splat viewer HTML from viewer_template.html.

This is the phone-proven viewer design from the lighthouse (2026-11-04)
and Zanjic Church (2026-11-04) work, packaged so another agent can reuse
it for a new scene without re-deriving the pitfalls:

  * NO CDN at runtime. three.js + Spark are embedded in the file as
    base64 and turned into blob: module URLs via an injected importmap.
    A phone that cannot reach jsDelivr / unpkg / cdnjs (proven failure
    mode: "FAILED at three.js: Failed to fetch dynamically imported
    module") still opens this file, because nothing is fetched from a
    CDN. The libraries are downloaded here, on the BUILD machine, where
    CDN access works.
  * three 0.180.0's three.module.js is only a wrapper importing
    './three.core.js', and a relative import CANNOT resolve from a blob:
    URL ("Failed to resolve module specifier './three.core.js'").
    Fix, applied here at build time: rewrite that import to the bare
    specifier 'three-core' and map it in the importmap to the embedded
    three.core.js blob. Spark itself imports only 'three' and
    'three/addons/postprocessing/Pass.js' (verified by grep), so those
    are the only other importmap entries needed.
  * The splat is either embedded (single file, but watch the size) or
    loaded from a URL with an embedded fallback: the viewer fetches the
    URL itself, and on ANY failure (cross-origin when the HTML was
    opened as a downloaded attachment, a stalled mobile download, a
    storage rate limit) it loads the embedded fallback splat and says
    which version loaded in the status line. Never let the loader fail
    silently — the original bug was an eternal "Loading…" with no error.
  * Size limits observed (Muse storage, 2026-11-04): a 115 MB single
    file is rejected (HTTP 413), an 81 MB file uploads fine, and rapid
    successive uploads hit HTTP 429 — wait a minute and retry. If your
    single-file build exceeds ~100 MB, ship libs-embedded HTML + a
    separately hosted PLY URL instead (see --ply-url), with an embedded
    medium-density fallback.

Usage:
  python3 build_standalone.py --ply scene.ply --out scene.html \\
      --title "My Scene" --subtitle "Gaussian splat, standalone"
  python3 build_standalone.py --ply-url https://example.com/scene.ply \\
      --fallback-ply scene_medium.ply --out scene.html --title "My Scene"
  python3 build_standalone.py --ply scene.ply --out scene.html --spin-on

--ply and --ply-url are mutually exclusive as the PRIMARY splat; use
--fallback-ply with --ply-url to bake in the fallback. At least one
splat source is required. Camera home/target are "x, y, z" strings.
The template lives next to this script (viewer_template.html).
"""
import argparse, base64, json, re, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
THREE_VER = "0.180.0"
SPARK_VER = "2.4.0"
LIB_URLS = {
    "three": f"https://cdn.jsdelivr.net/npm/three@{THREE_VER}/build/three.module.js",
    "three_core": f"https://cdn.jsdelivr.net/npm/three@{THREE_VER}/build/three.core.js",
    "pass": f"https://cdn.jsdelivr.net/npm/three@{THREE_VER}/examples/jsm/postprocessing/Pass.js",
    "spark": f"https://sparkjs.dev/releases/spark/{SPARK_VER}/spark.module.js",
}

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "colab-llm-deploy-viewer-builder"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if len(data) < 1000:
        raise RuntimeError(f"suspiciously small download from {url} ({len(data)} bytes)")
    return data

def b64(data):
    return base64.b64encode(data).decode()

def ply_points(path):
    """Point count from a PLY header (element vertex), or None."""
    with open(path, "rb") as f:
        head = f.read(4096).decode("ascii", "ignore")
    m = re.search(r"element vertex (\d+)", head)
    return int(m.group(1)) if m else None

def main():
    ap = argparse.ArgumentParser(description="Build a standalone splat viewer HTML.")
    ap.add_argument("--template", default=str(ROOT / "viewer_template.html"))
    ap.add_argument("--ply", help="primary PLY file to embed (omit with --ply-url)")
    ap.add_argument("--ply-url", default="", help="primary PLY URL (omit with --ply)")
    ap.add_argument("--fallback-ply", default="", help="fallback PLY file to embed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="3D Scene")
    ap.add_argument("--subtitle", default="Gaussian splat · standalone, no CDN needed")
    ap.add_argument("--cam-pos", default="5.5, 3.0, 5.5")
    ap.add_argument("--cam-tgt", default="0, 0.9, 0")
    ap.add_argument("--spin-on", action="store_true", help="start auto-rotating")
    args = ap.parse_args()

    if bool(args.ply) == bool(args.ply_url):
        ap.error("exactly one of --ply / --ply-url is required as the primary splat")
    if not args.ply and not args.fallback_ply and not args.ply_url:
        ap.error("no splat source at all")

    print("downloading viewer libraries (build machine only)…", flush=True)
    three = fetch(LIB_URLS["three"]).decode()
    if "./three.core.js" not in three:
        raise RuntimeError("three.module.js no longer imports ./three.core.js — "
                           "the three-core rewrite below must be revisited")
    three_fixed = three.replace("from './three.core.js'", "from 'three-core'")
    libs = {
        "THREE_B64": b64(three_fixed.encode()),
        "THREE_CORE_B64": b64(fetch(LIB_URLS["three_core"])),
        "PASS_B64": b64(fetch(LIB_URLS["pass"])),
        "SPARK_B64": b64(fetch(LIB_URLS["spark"])),
    }
    for k, v in libs.items():
        print(f"  {k}: {len(v)} base64 chars", flush=True)

    template = Path(args.template).read_text()
    tokens = {
        "TITLE": args.title,
        "SUBTITLE": args.subtitle,
        "CAM_POS": args.cam_pos,
        "CAM_TGT": args.cam_tgt,
        "PLY_URL": args.ply_url,
        "PRIMARY_PLY_B64": b64(Path(args.ply).read_bytes()) if args.ply else "",
        "FALLBACK_PLY_B64": b64(Path(args.fallback_ply).read_bytes()) if args.fallback_ply else "",
    }
    tokens.update(libs)
    html = template
    for key, val in tokens.items():
        html = html.replace("{{" + key + "}}", val)
    left = sorted(set(re.findall(r"{{[A-Za-z0-9_]+}}", html)))
    if left:
        raise RuntimeError(f"unfilled template tokens remain: {left}")
    if args.spin_on:
        html = html.replace('<button id="spin">Spin: off</button>',
                            '<button id="spin">Spin: on</button>')
        html = html.replace("let mesh = null, spinning = false;",
                            "let mesh = null, spinning = true;")
    out = Path(args.out)
    out.write_text(html)
    n = ply_points(args.ply) if args.ply else None
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)"
          + (f" · primary PLY points: {n}" if n else ""), flush=True)
    print("NOTE: syntax-check the embedded script before shipping "
          "(extract <script>…</script> and run: node --check it).", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
