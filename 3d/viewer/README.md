# Standalone splat viewer builder

Builds a phone-proof Gaussian-splat viewer HTML from `viewer_template.html`.
This is the design that survived real phone testing (2026-11-04, after
four CDN-based versions failed on the user's phone in turn). Read this
before changing the template — every odd-looking choice below is a
bug that was actually hit.

## Quick use

```bash
# Single file: libraries + splat embedded (watch the ~100 MB host limit)
python3 3d/viewer/build_standalone.py --ply scene.ply --out scene.html \
    --title "My Scene" --spin-on

# Two parts: libraries embedded, splat from a URL, embedded fallback
python3 3d/viewer/build_standalone.py \
    --ply-url https://example.com/scene.ply \
    --fallback-ply scene_medium.ply --out scene.html --title "My Scene"
```

Open the output in Chrome on a phone. Touch: 1 finger to look around,
2 fingers to move. Buttons: Reset view, Spin toggle, "Open a PLY file"
picker for any local `.ply` / `.spz` / `.splat` / `.ksplat`. A `?ply=<url>`
query parameter overrides the built-in splat.

## Why it looks like this (all proven failure modes)

1. **No CDN at runtime.** The first viewer used an importmap pointing
   at jsDelivr (three) and sparkjs.dev (Spark). On the phone it failed
   with `FAILED at three.js: Failed to fetch dynamically imported
   module`, and a multi-host probe then failed on jsDelivr, unpkg AND
   cdnjs. The phone's network simply does not reach those CDNs.
   So `build_standalone.py` downloads three.js + Spark on the BUILD
   machine (where CDN access works) and embeds them as base64, turned
   into `blob:` module URLs through an injected importmap. The phone
   fetches nothing from a CDN.
2. **The three.core.js rewrite.** three 0.180.0's `three.module.js` is
   only a wrapper importing `./three.core.js`, and a relative import
   cannot resolve from a `blob:` URL (`Failed to resolve module
   specifier "./three.core.js". Invalid relative url or base scheme
   isn't hierarchical`). The builder rewrites that import to the bare
   specifier `three-core` and maps it in the importmap to the embedded
   `three.core.js` blob. Spark imports only `three` and
   `three/addons/postprocessing/Pass.js` (verified by grep), so the
   importmap needs exactly three entries plus `three-core`.
3. **The error handler lives in a classic script, not the module.**
   The first viewer put its error handler inside the module script, so
   a failed import meant the handler never registered and the page sat
   on "Loading…" forever with no error. The template's script is a
   classic script that loads the libraries with dynamic `import()`,
   one at a time, and names the failing step in the status line.
4. **The splat load never fails silently.** The viewer fetches a URL
   splat itself; on ANY failure (cross-origin when the HTML was opened
   as a downloaded attachment, a stalled mobile download, a storage
   rate limit — all seen as `Failed to fetch`) it loads the embedded
   fallback splat and says which version loaded. A URL-only splat with
   no fallback is how the lively edition first broke on the phone.
5. **jsDelivr never serves the HTML.** jsDelivr serves `.html` as plain
   text, so a jsDelivr viewer link shows source code and never runs.
   Host the HTML on a host that serves it as a page; jsDelivr is fine
   for the binary PLY only (if the phone can reach it at all).

## Size limits observed (Muse storage, 2026-11-04)

- A 115 MB single-file HTML is rejected (HTTP 413); an 81 MB file
  uploads fine. Treat ~100 MB as the single-file ceiling.
- Rapid successive uploads hit HTTP 429 — wait a minute and retry.
- If the single-file build exceeds the ceiling, ship libs-embedded
  HTML + a separately hosted PLY URL (the `--ply-url` mode) with an
  embedded medium-density fallback, exactly as the Zanjic lively
  edition does. All such links expire — the template carries no
  permanence claim; re-upload or host the PLY somewhere permanent
  before the expiry.

## Verification before shipping

- `node --check` the template's embedded script (the builder prints
  this reminder; it does not run node itself).
- Confirm no `{{TOKEN}}` placeholders remain in the output.
- Confirm the embedded libraries decode to the source sizes
  (three 603,103 / three.core 1,403,455 / Pass 4,218 / Spark 2,997,944
  bytes for the pinned versions) and that the three-core rewrite is
  present in the decoded three module.
- The only reliable rendering check is the target phone itself:
  live-browser verification from a VM is blocked or unrepresentative
  (org policy blocks the file host, localhost is rejected, and the
  VM's egress proxy cannot reach the CDNs). Ship, then ask for the
  exact status-line text.
