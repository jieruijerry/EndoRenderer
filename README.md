# EndoRenderer

Near-field glossy neural radiosity for known-anatomy virtual endoscopy.

This repository starts from the [OctaOctree](https://github.com/jieruijerry/OctaOctree) training / Mitsuba 3 / interactive viewer stack (path tracing, custom CUDA kernels, tiny-cuda-nn). The target application is **forward rendering** on known colon meshes, not video reconstruction.

## Baseline

Copied from `~/Documents/Research/OctaOctree`:

- Customized Mitsuba 3.5.2 (`cuda_rgb`) path tracer and G-buffer integrators
- Interactive GLFW / ImGui viewer with GPU texture display
- OctaOctree / NCR-style neural radiosity training (`train.py`)
- Custom CUDA extensions: `src/module/hash_grid_cuda`, `src/module/octa_octree_cuda`
- tiny-cuda-nn encodings via `tinycudann`

Smoke-test scene: `scenes/cornell-box` (small Bitterli Cornell box). C3VD lives on disk and is symlinked at `data/c3vd`.

## Installation

Same environment as OctaOctree:

```bash
conda activate mi3
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install ocnn
```

Mitsuba 3.5.2 is the patched tree next to this repo:

```text
../official-submodules/mitsuba3-old
```

```bash
source scripts/active_mitsuba.sh
```

## Usage

Interactive path tracing (no trained cache needed):

```bash
source scripts/active_mitsuba.sh
python render.py -c null -s cornell-box
```

Train OctaOctree on a Mitsuba scene:

```bash
python train.py -c octaoctree -s cornell-box
```

Offline image:

```bash
python scripts/render_image.py -c null -s cornell-box -r PT -o cornell_pt
```

## Data

C3VD lives on the HDD and is linked at `data/c3vd`. Layout and inventory: `data/README.md`.

```text
data/c3vd/sequences/<seq>   # official RGB, depth, normals, pose, coverage_mesh.obj
data/c3vd/models            # lumen OBJ
```

17 of 22 registered sequences are ready. To resume missing official zips after Drive quota cools down:

```bash
python scripts/download_c3vd_official.py --no-screening
```

C3VD Mitsuba scenes (one per sequence, first-frame camera + colocated spot):

```bash
python scripts/build_c3vd_scenes.py
python render.py -c null -s c3vd/cecum_t1_a
```

See `scenes/README.md`.

## Next

1. Wrap geometry with an adaptive SDF shell; shade with NCR / OctaOctree
2. Compare against high-SPP path tracing

## License

Code is MIT, derived from OctaOctree. C3VD is CC BY-NC-SA 4.0 (Durr Lab, JHU). `scenes/cornell-box` retains its original Bitterli / CC0 license.
