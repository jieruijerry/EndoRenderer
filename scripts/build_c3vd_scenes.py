#!/usr/bin/env python3
"""Build one Mitsuba scene per official C3VD sequence.

Each scene uses that sequence's coverage_mesh.obj and the first-frame
camera-to-world pose. The endoscope light is a colocated spotlight.
Material is constant-albedo roughplastic (no textures).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQ_ROOT = REPO_ROOT / "data" / "c3vd" / "sequences"
DEFAULT_SCENE_ROOT = REPO_ROOT / "scenes" / "c3vd"

# Scaramuzza intrinsics for the CF-HQ190L, from the C3VD calibration file.
# e, f, g are the stretch matrix [[e, f], [g, 1]] in Bobrow et al. Eq. 3.
CALIB_PATH = (
    REPO_ROOT
    / "data"
    / "c3vd"
    / "calibration"
    / "cfhq190l_omnidirectional_params"
    / "cfhq190l_omnidirectional_values.txt"
)


def load_intrinsics() -> dict:
    vals = np.loadtxt(CALIB_PATH)
    width, height, cx, cy, a0, a2, a3, a4, stretch_e, stretch_f, stretch_g = vals.tolist()
    return {
        "width": int(width),
        "height": int(height),
        "cx": cx,
        "cy": cy,
        "a0": a0,
        "a2": a2,
        "a3": a3,
        "a4": a4,
        "e": stretch_e,
        "f": stretch_f,
        "g": stretch_g,
    }


INTRINSICS = load_intrinsics()
C3VD_WIDTH = INTRINSICS["width"]
C3VD_HEIGHT = INTRINSICS["height"]

# OpenCV camera (X right, Y down, Z forward) -> Mitsuba camera
# (X left, Y up, Z forward). Both are right-handed.
OPENCV_TO_MITSUBA = np.diag([-1.0, -1.0, 1.0, 1.0])

# Mucosa-like constants. alpha is GGX roughness; IOR ~ colorectal tissue.
ROUGHNESS_ALPHA = 0.05
INT_IOR = 1.38
ALBEDO = (0.78, 0.42, 0.36)
SPOT_INTENSITY = (9000.0, 8200.0, 7400.0)
SPOT_CUTOFF_DEG = 80.0
SPOT_BEAM_DEG = 60.0


def parse_c2w(line: str) -> np.ndarray:
    """Load one C3VD pose.txt row as a 4x4 OpenCV camera-to-world matrix."""
    vals = np.fromstring(line.strip(), sep=",", dtype=np.float64)
    if vals.size != 16:
        raise ValueError(f"expected 16 pose values, got {vals.size}")
    # Official dump is column-major [R t; 0 0 0 1].
    return vals.reshape(4, 4).T


def first_pose(pose_path: Path) -> np.ndarray:
    with pose_path.open() as f:
        for line in f:
            if line.strip():
                return parse_c2w(line)
    raise ValueError(f"empty pose file: {pose_path}")


def matrix_attr(m: np.ndarray) -> str:
    return " ".join(f"{x:.8g}" for x in m.reshape(-1))


def scene_xml(seq_name: str, c2w_mi: np.ndarray, mesh_rel: str) -> str:
    mat = matrix_attr(c2w_mi)
    cam = INTRINSICS
    ar, ag, ab = ALBEDO
    ir, ig, ib = SPOT_INTENSITY
    return f"""<scene version="3.0.0">
	<default name="integrator" value="path" />
	<default name="spp" value="64" />
	<default name="resx" value="{C3VD_WIDTH}" />
	<default name="resy" value="{C3VD_HEIGHT}" />
	<default name="max_depth" value="8" />

	<integrator type="$integrator">
		<integer name="max_depth" value="$max_depth" />
	</integrator>

	<bsdf type="roughplastic" id="tissue">
		<string name="distribution" value="ggx" />
		<float name="alpha" value="{ROUGHNESS_ALPHA}" />
		<float name="int_ior" value="{INT_IOR}" />
		<float name="ext_ior" value="1.0" />
		<boolean name="nonlinear" value="true" />
		<rgb name="diffuse_reflectance" value="{ar}, {ag}, {ab}" />
	</bsdf>

	<sensor type="c3vd">
		<float name="cx" value="{cam['cx']:.12g}" />
		<float name="cy" value="{cam['cy']:.12g}" />
		<float name="a0" value="{cam['a0']:.12g}" />
		<float name="a2" value="{cam['a2']:.12g}" />
		<float name="a3" value="{cam['a3']:.12g}" />
		<float name="a4" value="{cam['a4']:.12g}" />
		<float name="e" value="{cam['e']:.12g}" />
		<float name="f" value="{cam['f']:.12g}" />
		<float name="g" value="{cam['g']:.12g}" />
		<float name="near_clip" value="0.1" />
		<float name="far_clip" value="200" />
		<transform name="to_world">
			<matrix value="{mat}" />
		</transform>
		<sampler type="independent">
			<integer name="sample_count" value="$spp" />
		</sampler>
		<film type="hdrfilm">
			<integer name="width" value="$resx" />
			<integer name="height" value="$resy" />
			<string name="file_format" value="openexr" />
			<string name="pixel_format" value="rgb" />
			<rfilter type="box" />
		</film>
	</sensor>

	<emitter type="spot">
		<rgb name="intensity" value="{ir}, {ig}, {ib}" />
		<float name="cutoff_angle" value="{SPOT_CUTOFF_DEG}" />
		<float name="beam_width" value="{SPOT_BEAM_DEG}" />
		<transform name="to_world">
			<matrix value="{mat}" />
		</transform>
	</emitter>

	<shape type="obj" id="{seq_name}_mesh">
		<string name="filename" value="{mesh_rel}" />
		<boolean name="face_normals" value="false" />
		<ref id="tissue" />
	</shape>
</scene>
"""


def build_one(seq_dir: Path, scene_root: Path) -> Path:
    seq_name = seq_dir.name
    pose_path = seq_dir / "pose.txt"
    mesh_path = seq_dir / "coverage_mesh.obj"
    if not pose_path.is_file() or not mesh_path.is_file():
        raise FileNotFoundError(f"{seq_name}: missing pose.txt or coverage_mesh.obj")

    c2w_cv = first_pose(pose_path)
    c2w_mi = c2w_cv @ OPENCV_TO_MITSUBA

    out_dir = scene_root / seq_name
    out_dir.mkdir(parents=True, exist_ok=True)
    # abspath keeps the data/c3vd symlink; resolve() would bake in the HDD path.
    mesh_rel = Path(os.path.relpath(os.path.abspath(mesh_path), os.path.abspath(out_dir))).as_posix()
    xml_path = out_dir / "scene.xml"
    xml_path.write_text(scene_xml(seq_name, c2w_mi, mesh_rel), encoding="utf-8")
    return xml_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-root", type=Path, default=DEFAULT_SEQ_ROOT)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--only", nargs="*", default=None, help="Optional sequence names")
    args = parser.parse_args()

    seq_root = args.seq_root
    if not seq_root.is_dir():
        raise SystemExit(f"sequence root not found: {seq_root}")

    names = args.only if args.only else sorted(p.name for p in seq_root.iterdir() if p.is_dir())
    written = []
    for name in names:
        seq_dir = seq_root / name
        if not seq_dir.is_dir():
            print(f"skip missing {name}")
            continue
        xml_path = build_one(seq_dir, args.scene_root)
        written.append(xml_path)
        print(f"wrote {xml_path.relative_to(REPO_ROOT)}")

    print(f"built {len(written)} C3VD Mitsuba scenes")


if __name__ == "__main__":
    main()
