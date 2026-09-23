# Scenes

Mitsuba XML scenes live here.

| Path | Role |
| --- | --- |
| `cornell-box/` | Smoke-test scene from OctaOctree (Bitterli / CC0) |
| `c3vd/<seq>/` | One scene per official C3VD sequence (see below) |

## C3VD

`scripts/build_c3vd_scenes.py` writes `scenes/c3vd/<seq>/scene.xml` for every sequence that has both `pose.txt` and `coverage_mesh.obj`.

Each scene:

- **Geometry**: that sequence's `coverage_mesh.obj` (same world as `pose.txt`). Path is relative through `data/c3vd`.
- **Camera**: first `pose.txt` row, column-major C3VD c2w, converted OpenCV → Mitsuba with `diag(1, -1, 1, 1)` so +Z still looks down the lumen and film Y is up.
- **Light**: colocated `spot` with the same `to_world`, ~80° cutoff (endoscope-like).
- **Material**: `roughplastic` GGX, `alpha=0.05`, `int_ior=1.38`, constant flesh albedo `(0.78, 0.42, 0.36)`. No textures.

Pinhole FOV is the Scaramuzza linear term `a0=769.24` at 1350×1080 (`fov_x ≈ 82.56°`). Units stay millimeters.

Rebuild after adding sequences:

```bash
python scripts/build_c3vd_scenes.py
```

Interactive view (path tracer, no neural cache):

```bash
source scripts/active_mitsuba.sh
python render.py -c null -s c3vd/cecum_t1_a
```
