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
- **Camera**: `c3vd` sensor, the CF-HQ190L Scaramuzza model (`cx`, `cy`, `a0`–`a4`, stretch `e,f,g`). First `pose.txt` row is the Mitsuba `to_world`: column-major OpenCV c2w times `diag(-1, -1, 1, 1)`, because Mitsuba's camera is X left / Y up / Z forward.
- **Light**: colocated `spot` with the same `to_world`, ~80° cutoff (endoscope-like).
- **Material**: `roughplastic` GGX, `alpha=0.05`, `int_ior=1.38`, constant flesh albedo `(0.78, 0.42, 0.36)`. No textures.

The plugin is registered by `src/sensor/c3vd.py`, which `render.py` and `train.py` import before loading a scene. Units stay millimeters.

Rebuild after adding sequences:

```bash
python scripts/build_c3vd_scenes.py
```

Interactive view (path tracer, no neural cache):

```bash
source scripts/active_mitsuba.sh
python render.py -c null -s c3vd/cecum_t1_a
```
