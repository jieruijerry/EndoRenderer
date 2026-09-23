# Datasets

Physical copies live on HDD2. Repo paths under `data/` are symlinks.

| Dataset | Physical path | Repo link |
| --- | --- | --- |
| C3VD | `/media/jerry/HDD2/datasets/c3vd` | `data/c3vd` |
| RealSynCol | `/media/jerry/HDD2/datasets/realsyncol` | `data/realsyncol` |

---

# C3VD

Physical copy: `/media/jerry/HDD2/datasets/c3vd`  
Repo link: `data/c3vd`

Source: [durrlab.github.io/C3VD](https://durrlab.github.io/C3VD/) (CC BY-NC-SA 4.0).  
Bobrow et al., *Medical Image Analysis*, 2023.

```text
data/c3vd/
├── sequences/     # official registered videos (use these)
├── models/        # lumen OBJ meshes
├── calibration/   # Olympus CF-HQ190L
├── zips/          # official archives
└── extras/        # Hugging Face leftovers, not for main experiments
```

Each official sequence has RGB, depth, normals, flow, occlusion, `pose.txt`, and `coverage_mesh.obj`.

On-disk `pose.txt` is one 4×4 camera-to-world matrix per RGB frame (16 comma-separated floats, translation in millimeters). Each sequence’s poses and `coverage_mesh.obj` share a **per-sequence** registered frame — not one shared phantom frame and not `models/*.obj` / `full_model.obj` coordinates. Same-anatomy coverage meshes are the same vertex set under a small-to-large rigid offset (see analysis notes).

## Official sequences (17 / 22)

| Sequence | Frames | Status |
| --- | ---: | --- |
| cecum_t1_a | 276 | ready |
| cecum_t1_b | 765 | missing (Drive quota) |
| cecum_t2_a | 370 | ready |
| cecum_t2_b | 1142 | missing (Drive quota) |
| cecum_t2_c | 595 | missing (Drive quota) |
| cecum_t3_a | 730 | missing (Drive quota) |
| cecum_t4_a | 465 | ready |
| cecum_t4_b | 425 | ready |
| desc_t4_a | 148 | ready |
| sigmoid_t1_a | 700 | missing (Drive quota) |
| sigmoid_t2_a | 514 | ready |
| sigmoid_t3_a | 613 | ready |
| sigmoid_t3_b | 536 | ready |
| trans_t1_a | 61 | ready |
| trans_t1_b | 700 | ready |
| trans_t2_a | 194 | ready |
| trans_t2_b | 103 | ready |
| trans_t2_c | 235 | ready |
| trans_t3_a | 250 | ready |
| trans_t3_b | 214 | ready |
| trans_t4_a | 382 | ready |
| trans_t4_b | 597 | ready |

`extras/hf_cecum_t1_b` has color, depth, pose, and mesh only (no normals / flow). Not a substitute for the official zip.

To add a missing sequence: download the zip from the project page into `data/c3vd/zips/`, then:

```bash
python scripts/download_c3vd_official.py --no-screening
```

---

# RealSynCol

Physical copy: `/media/jerry/HDD2/datasets/realsyncol`  
Repo link: `data/realsyncol`

Source: [Zenodo 10.5281/zenodo.19705803](https://doi.org/10.5281/zenodo.19705803) (CC BY 4.0).  
Lena et al., arXiv:2602.08397 / *Scientific Data*, 2026.  
Code: [github.com/ChiaraLena/RealSynCol](https://github.com/ChiaraLena/RealSynCol)

20 sequences (10 CT-derived colon models × forward/backward), 28,130 frames at 1024×1024. Official release includes RGB, depth (EXR), optical flow (EXR), normals (EXR), `Trajectory.txt` (camera-to-world), `Intrinsic.txt`, and `Colon_XX.ply` meshes.

Verified on the first extracted sequence (`Colon_6_forward`, 561 frames): `Frame/`, `Depth/`, `Flow/`, `Normal/`, `Trajectory.txt` (561 c2w lines), `Intrinsic.txt` (fx=fy=610.18, cx=cy=511.5), `Colon_6.ply`.

```text
data/realsyncol/
├── zips/          # official Zenodo archives + MANIFEST.md5
├── sequences/     # extracted sequences (after unzip)
├── logs/          # wget / queue logs
└── README.md
```

Resume an interrupted download (do not start a second wget on the same partial zip):

```bash
python /media/jerry/HDD2/datasets/realsyncol/queue_download.py
```

Or per file:

```bash
wget -c --tries=0 -O /media/jerry/HDD2/datasets/realsyncol/zips/Colon_1_forward.zip \
  'https://zenodo.org/records/19705803/files/Colon_1_forward.zip?download=1'
```
