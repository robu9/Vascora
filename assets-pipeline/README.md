# Heart asset pipeline

The delivered `heart.blend` is an editable mesh project. The shipped GLB is ready
to use; running the app does not require Blender. `export_heart.py` preserves the
original Bezier control points and handles before tessellation; editable source
curves can be regenerated from the pinned archive.

## Reproduce from the repository root

Requires `uv`, Node 22.12+ and npm. The isolated `bpy` package supplies Blender
5.0.1 without installing the desktop application. A compatible Blender 5.0.1
desktop installation can alternatively execute the Python scripts in background mode.

```sh
python3 assets-pipeline/fetch_source.py
uv run --python 3.11 --with bpy==5.0.1 python assets-pipeline/inspect_source.py
uv run --python 3.11 --with bpy==5.0.1 python assets-pipeline/export_heart.py
npm --prefix frontend ci
npm --prefix frontend run asset:optimize
npm --prefix frontend run asset:validate
```

The source archive (~83 MB) and extracted atlas (~307 MB) are ignored by Git.
The archive hash in `provenance.json` pins its contents; download verification
fails if the upstream `master` archive changes. No source add-on scripts are executed.

## Anatomy and geometry

- External left/right ventricular and atrial surfaces, pulmonary trunk, ascending
  aorta, SVC root and left main artery come from the source.
- Native LAD/LCX/RCA curves are split at shared control points. Adjacent segments
  retain the same endpoint and Bezier handles. Side branches remain grouped with
  their parent vessel; no separate predictions are invented.
- D1 uses source LAD splines 2–3; OM1 uses LCX spline 3. These are representative
  branch labels, not clinically validated numbering.
- No explicitly identified PDA exists in the source collection. The new PDA is
  an authored, surface-fitted right-dominant approximation starting at the RCA's
  posterior endpoint. The native right inferolateral branches remain in RCA_dist.
  Source donor dominance cannot be established from naming alone.
- Proximal/mid/distal names identify display partitions, not clinical lesion segments.
- Internal valve/papillary structures and torso anatomy are excluded. Open vessel
  roots are intentional; the surface asset has no cutaway functionality.
- glTF axes: +X patient left, +Y superior, +Z anterior. All structures share one
  normalization transform and the same runtime heartbeat transform.
- Runtime materials use no external texture or environment downloads. The model
  supplies geometry and base colors; lighting/material response is illustrative.

`provenance.json` records exact source spline ranges and normalization.
`validation.json` records mesh counts, triangles and compressed byte size.
The validator requires every registry mesh exactly once, independent artery
materials, spatial alignment with the heart surfaces, meshopt compression,
<150,000 triangles and <8 MiB. Source transforms are baked in the original atlas
before extracting the library, avoiding parent-dependent transforms on reload.

## Verification on 13 October 2026

- 20 exported meshes, all 12 required artery meshes; 126,630 triangles.
- Compressed model: 870,960 bytes (about 851 KiB), no texture downloads.
- Four camera views visually inspected in Chromium, including vessel attachment
  and the shared heartbeat transform. This is visual engineering QA, not clinical review.
- 8 frontend unit tests and 12 backend tests passed; backend lint and frontend
  TypeScript/production build passed.
- Browser coverage includes all sample predictions, vessel selection, view
  presets, retry, missing GLB, invalid mapping, zero probability, unavailable
  WebGL, narrow layout and reduced motion.
- The four-second orbit/heartbeat sample averaged **60.3 FPS** at 1440×900 on
  Apple M1, macOS arm64, Chromium 156.0.8078.4 using ANGLE Metal. See
  `performance.json`. This is a headless hardware-backed test of animation-frame
  cadence, not a guarantee across laptops or a GPU benchmark.

## Licensing

See `LICENSE-source.txt` for the unchanged Z-Anatomy source notice and
`../frontend/public/models/ATTRIBUTION.md` for complete attribution, modifications,
and license links. The adapted coronary file retains CC BY-SA 4.0 licensing.
The viewer draws neshallads' Realistic Human Heart (`realistic_human_heart.glb`,
CC BY 4.0) instead of these meshes.
