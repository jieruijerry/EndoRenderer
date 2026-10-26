#!/usr/bin/env python3
"""Compare our C3VD camera against the dataset depth and normal maps.

Renders one sequence with the scene sensor at dataset poses. Depth is the
camera-frame Z in millimeters (C3VD clamps this to 0–100 mm). Normals are
converted from Mitsuba's camera (X left, Y up) to OpenCV (X right, Y down).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import drjit as dr
import mitsuba as mi

mi.set_variant("cuda_rgb")

from src.dataset.c3vd_color import parse_mitsuba_c2w
from src.sensor.c3vd import register_c3vd_sensor, sensor_prefix

register_c3vd_sensor()

DEPTH_MAX_MM = 100.0
RAY_BATCH = 262144


def load_poses(seq_dir: Path) -> np.ndarray:
    lines = [line for line in (seq_dir / "pose.txt").read_text().splitlines() if line.strip()]
    return np.stack([parse_mitsuba_c2w(line) for line in lines], axis=0)


def load_depth_mm(path: Path) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    return raw.astype(np.float32) / np.float32(65535.0) * DEPTH_MAX_MM, raw > 0


def load_normal_cv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV reads the TIFF as BGR. C3VD stores X/Y/Z in R/G/B."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    encoded = raw.astype(np.float32) / np.float32(65535.0) * 2.0 - 1.0
    normal = np.stack([encoded[:, :, 2], encoded[:, :, 1], encoded[:, :, 0]], axis=-1)
    valid = raw.max(axis=-1) > 0
    return normal, valid


def pixel_centers(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    xs = (np.arange(width, dtype=np.float32) + 0.5) / width
    ys = (np.arange(height, dtype=np.float32) + 0.5) / height
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    return grid_x.reshape(-1), grid_y.reshape(-1)


def render_geometry(scene: mi.Scene, c2w: np.ndarray, batch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = mi.traverse(scene)
    prefix = sensor_prefix(params)
    mat = mi.Matrix4f(c2w[None, ...])
    params[f"{prefix}.to_world"] = mat
    params.update()

    sensor = scene.sensors()[0]
    film = sensor.film().size()
    width, height = int(film[0]), int(film[1])
    sample_x, sample_y = pixel_centers(width, height)
    count = sample_x.shape[0]

    rotation = c2w[:3, :3].astype(np.float32)
    origin = c2w[:3, 3].astype(np.float32)
    depth = np.zeros(count, dtype=np.float32)
    normal = np.zeros((count, 3), dtype=np.float32)
    valid = np.zeros(count, dtype=bool)

    for start in range(0, count, batch):
        stop = min(start + batch, count)
        position = mi.Point2f(sample_x[start:stop], sample_y[start:stop])
        ray, _ = sensor.sample_ray(
            mi.Float(0.0),
            mi.Float(0.5),
            position,
            mi.Point2f(0.5, 0.5),
            True,
        )
        si = scene.ray_intersect(ray)
        hit = si.is_valid().torch().cpu().numpy().astype(bool)
        point = si.p.torch().cpu().numpy().astype(np.float32)
        shading = si.sh_frame.n.torch().cpu().numpy().astype(np.float32)
        camera_point = (point - origin) @ rotation
        camera_normal = shading @ rotation
        camera_normal *= np.array([-1.0, -1.0, 1.0], dtype=np.float32)
        depth[start:stop] = camera_point[:, 2]
        normal[start:stop] = camera_normal
        valid[start:stop] = hit

    return (
        depth.reshape(height, width),
        normal.reshape(height, width, 3),
        valid.reshape(height, width),
    )


def dataset_corner_mask(seq_dir: Path) -> np.ndarray:
    """Pixels C3VD leaves empty. The mask is constant across a sequence."""
    raw = cv2.imread(str(seq_dir / "0000_depth.tiff"), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(seq_dir / "0000_depth.tiff")
    return raw > 0


def apply_dataset_mask(depth: np.ndarray, normal: np.ndarray, valid: np.ndarray, inside: np.ndarray) -> np.ndarray:
    depth[~inside] = 0
    normal[~inside] = 0
    return valid & inside


def summarize(name: str, error: np.ndarray) -> str:
    if error.size == 0:
        return f"{name}: no overlapping pixels"
    return (
        f"{name}: n={error.size}  mse={np.mean(error ** 2):.6g}  "
        f"rmse={np.sqrt(np.mean(error ** 2)):.6g}  mae={np.mean(np.abs(error)):.6g}  "
        f"median={np.median(np.abs(error)):.6g}"
    )


def compare_frame(seq_dir: Path, frame: int, depth: np.ndarray, normal: np.ndarray, valid: np.ndarray) -> None:
    gt_depth, gt_depth_valid = load_depth_mm(seq_dir / f"{frame:04d}_depth.tiff")
    gt_normal, gt_normal_valid = load_normal_cv(seq_dir / f"{frame:04d}_normals.tiff")
    if gt_depth.shape != depth.shape or gt_normal.shape != normal.shape:
        raise RuntimeError(
            f"frame {frame}: rendered {depth.shape} vs dataset depth {gt_depth.shape}"
        )

    depth_mask = valid & gt_depth_valid & (gt_depth < DEPTH_MAX_MM)
    depth_error = depth[depth_mask] - gt_depth[depth_mask]
    saturated = int((valid & (gt_depth >= DEPTH_MAX_MM)).sum())

    pred_normal = normal.copy()
    norm = np.linalg.norm(pred_normal, axis=-1, keepdims=True)
    pred_normal = pred_normal / np.clip(norm, 1e-8, None)
    gt_unit = gt_normal / np.clip(np.linalg.norm(gt_normal, axis=-1, keepdims=True), 1e-8, None)
    normal_mask = valid & gt_normal_valid
    cosine = np.sum(pred_normal[normal_mask] * gt_unit[normal_mask], axis=-1)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    print(f"frame {frame}")
    print(" ", summarize("depth mm", depth_error))
    print(f"  depth saturated at {DEPTH_MAX_MM:.0f} mm (excluded): {saturated}")
    print(" ", summarize("normal deg", angle), f"mean cosine {float(cosine.mean()) if cosine.size else float('nan'):.4f}")


def save_frame(out_dir: Path, frame: int, depth: np.ndarray, normal: np.ndarray, valid: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_u16 = np.clip(depth, 0.0, DEPTH_MAX_MM) / DEPTH_MAX_MM * 65535.0
    depth_u16[~valid] = 0
    cv2.imwrite(str(out_dir / f"{frame:04d}_depth.tiff"), depth_u16.astype(np.uint16))

    encoded = np.clip(normal, -1.0, 1.0)
    # Store OpenCV X/Y/Z in R/G/B, and let OpenCV write BGR.
    rgb = (encoded + 1.0) * 0.5 * 65535.0
    bgr = np.stack([rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0]], axis=-1)
    bgr[~valid] = 0
    cv2.imwrite(str(out_dir / f"{frame:04d}_normals.tiff"), bgr.astype(np.uint16))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", help="sequence name, e.g. cecum_t1_a, or a scene.xml path")
    parser.add_argument("--frames", type=int, nargs="+", default=[0], help="pose indices to render")
    parser.add_argument("--output", type=Path, default=None, help="directory for rendered depth and normal TIFFs")
    parser.add_argument("--batch", type=int, default=RAY_BATCH)
    args = parser.parse_args()

    scene_path = Path(args.sequence)
    if scene_path.suffix == ".xml":
        seq_name = scene_path.parent.name
    else:
        seq_name = scene_path.name
        scene_path = REPO_ROOT / "scenes" / "c3vd" / seq_name / "scene.xml"
    seq_dir = REPO_ROOT / "data" / "c3vd" / "sequences" / seq_name
    poses = load_poses(seq_dir)
    inside = dataset_corner_mask(seq_dir)
    scene = mi.load_file(str(scene_path))

    for frame in args.frames:
        if frame < 0 or frame >= len(poses):
            raise IndexError(f"frame {frame} outside 0..{len(poses) - 1}")
        depth, normal, valid = render_geometry(scene, poses[frame], args.batch)
        valid = apply_dataset_mask(depth, normal, valid, inside)
        compare_frame(seq_dir, frame, depth, normal, valid)
        if args.output is not None:
            save_frame(args.output, frame, depth, normal, valid)
            print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
