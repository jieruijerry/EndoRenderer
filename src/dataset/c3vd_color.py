import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# OpenCV camera (X right, Y down, Z forward) -> Mitsuba camera
# (X left, Y up, Z forward). Both are right-handed.
OPENCV_TO_MITSUBA = np.diag([-1.0, -1.0, 1.0, 1.0])

# CF-HQ190L principal point, and the image-circle radius in pixels.
# Past this radius the frame is the black vignette in the corners.
C3VD_CX = 679.544839263292
C3VD_CY = 543.975887548343
C3VD_IMAGE_RADIUS = 700.0


def srgb_to_linear(color: torch.Tensor) -> torch.Tensor:
    """Convert sRGB in [0, 1] to linear RGB."""
    return torch.where(
        color <= 0.04045,
        color / 12.92,
        ((color + 0.055) / 1.055) ** 2.4,
    )


def image_circle_pixels(width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Integer pixel coordinates inside the endoscope image circle."""
    ys, xs = torch.meshgrid(
        torch.arange(height), torch.arange(width), indexing="ij"
    )
    radius2 = (xs.float() + 0.5 - C3VD_CX) ** 2 + (ys.float() + 0.5 - C3VD_CY) ** 2
    inside = radius2 < C3VD_IMAGE_RADIUS ** 2
    return xs[inside].to(torch.int16), ys[inside].to(torch.int16)


def parse_mitsuba_c2w(line: str) -> np.ndarray:
    """One C3VD pose.txt row -> Mitsuba camera-to-world."""
    vals = np.fromstring(line.strip(), sep=",", dtype=np.float64)
    if vals.size != 16:
        raise ValueError(f"expected 16 pose values, got {vals.size}")
    # Official dump is column-major [R t; 0 0 0 1].
    c2w_cv = vals.reshape(4, 4).T
    return (c2w_cv @ OPENCV_TO_MITSUBA).astype(np.float32)


class C3VDColorDataset(Dataset):
    """
    One training batch per C3VD frame.

    Each item samples ``n_rays`` pixel centers from ``{frame}_color.png``
    and pairs them with that frame's Mitsuba camera pose. Pixels outside
    the endoscope image circle are not sampled. Colors are linear RGB.
    Images are cached as uint8 on CPU.
    """

    def __init__(self, seq_dir: str, n_rays: int):
        if not os.path.isdir(seq_dir):
            raise FileNotFoundError(f"C3VD sequence not found: {seq_dir}")
        if n_rays < 1:
            raise ValueError(f"n_rays must be positive, got {n_rays}")

        self.seq_dir = seq_dir
        self.n_rays = n_rays

        pose_path = os.path.join(seq_dir, "pose.txt")
        with open(pose_path) as f:
            pose_lines = [line for line in f if line.strip()]
        c2w = [parse_mitsuba_c2w(line) for line in pose_lines]

        color_paths = {}
        for name in os.listdir(seq_dir):
            if not name.endswith("_color.png"):
                continue
            frame = int(name.split("_", 1)[0])
            color_paths[frame] = os.path.join(seq_dir, name)

        n_frames = len(c2w)
        missing = [i for i in range(n_frames) if i not in color_paths]
        if missing or len(color_paths) != n_frames:
            raise RuntimeError(
                f"{seq_dir}: {n_frames} poses but color frames {sorted(color_paths)} "
                f"(missing {missing[:8]})"
            )

        first = cv2.imread(color_paths[0], cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"failed to read {color_paths[0]}")
        height, width = first.shape[:2]
        images = torch.empty((n_frames, height, width, 3), dtype=torch.uint8)
        images[0] = torch.from_numpy(np.ascontiguousarray(first[:, :, ::-1]))

        print(f"Loading {n_frames} C3VD color frames from {seq_dir}")
        for frame in range(1, n_frames):
            bgr = cv2.imread(color_paths[frame], cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[0] != height or bgr.shape[1] != width:
                raise RuntimeError(f"bad color frame {color_paths[frame]}")
            images[frame] = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1]))

        self.images = images
        self.c2w = torch.from_numpy(np.stack(c2w, axis=0))
        self.height = height
        self.width = width
        self.valid_xs, self.valid_ys = image_circle_pixels(width, height)
        print(
            f"Cached {n_frames} frames at {width}x{height}, "
            f"{self.valid_xs.shape[0]} pixels inside the image circle, "
            f"{n_rays} rays/batch"
        )

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        image = self.images[idx]
        choice = torch.randint(0, self.valid_xs.shape[0], (self.n_rays,))
        xs = self.valid_xs[choice].long()
        ys = self.valid_ys[choice].long()
        color = image[ys, xs].float() / 255.0
        pixel = torch.stack(
            ((xs.float() + 0.5) / self.width, (ys.float() + 0.5) / self.height),
            dim=-1,
        )
        return {
            "c2w": self.c2w[idx],
            "pixel": pixel,
            "color": srgb_to_linear(color),
        }


def discover_c3vd_sequences(
    data_root: str = os.path.join("data", "c3vd", "sequences"),
    scene_root: str = os.path.join("scenes", "c3vd"),
) -> list[str]:
    """Sequence names that have both a Mitsuba scene and a color sequence."""
    if not os.path.isdir(scene_root):
        raise FileNotFoundError(f"C3VD scenes not found: {scene_root}")
    names = []
    for name in sorted(os.listdir(scene_root)):
        scene_xml = os.path.join(scene_root, name, "scene.xml")
        pose = os.path.join(data_root, name, "pose.txt")
        if os.path.isfile(scene_xml) and os.path.isfile(pose):
            names.append(name)
    if not names:
        raise RuntimeError(f"no C3VD sequences under {data_root} and {scene_root}")
    return names


def _frame_table(seq_dir: str) -> tuple[list[np.ndarray], dict[int, str]]:
    """Camera matrices and ``{frame}_color.png`` paths, checked against each other."""
    pose_path = os.path.join(seq_dir, "pose.txt")
    with open(pose_path) as f:
        c2w = [parse_mitsuba_c2w(line) for line in f if line.strip()]
    color_paths = {}
    for name in os.listdir(seq_dir):
        if not name.endswith("_color.png"):
            continue
        frame = int(name.split("_", 1)[0])
        color_paths[frame] = os.path.join(seq_dir, name)
    missing = [i for i in range(len(c2w)) if i not in color_paths]
    if missing or len(color_paths) != len(c2w):
        raise RuntimeError(
            f"{seq_dir}: {len(c2w)} poses but color frames {sorted(color_paths)} "
            f"(missing {missing[:8]})"
        )
    return c2w, color_paths


def _frame_circle_mask(width: int, height: int, scale: int) -> torch.Tensor:
    """Circle mask at the network resolution. Pixel centers stay on the full film."""
    out_w = width // scale
    out_h = height // scale
    ys = torch.arange(out_h)
    xs = torch.arange(out_w)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    origin_x = (grid_x.float() + 0.5) * scale
    origin_y = (grid_y.float() + 0.5) * scale
    radius2 = (origin_x - C3VD_CX) ** 2 + (origin_y - C3VD_CY) ** 2
    return radius2 < C3VD_IMAGE_RADIUS ** 2


class C3VDScreenDataset(Dataset):
    """
    Full endoscope frames for a screen-space network.

    Each item is the whole image, optionally shrunk by an integer ``scale``
    that keeps the fisheye centered. A crop would move the distorted rim
    into the middle of the tensor. ``split`` is ``train`` (held-in
    sequences, frames not divisible by ``holdout_modulo``), ``view`` (those
    held-out frames), or ``sequence`` (every frame of the held-out sequence).
    """

    def __init__(
        self,
        samples: list[dict],
        scale: int,
        uniform_sequences: bool = False,
    ) -> None:
        if scale < 1:
            raise ValueError(f"scale must be a positive integer, got {scale}")
        if not samples:
            raise RuntimeError("C3VD screen dataset has no frames")
        self.samples = samples
        self.scale = int(scale)
        self.by_sequence: dict[str, list[dict]] = {}
        for sample in samples:
            self.by_sequence.setdefault(sample["sequence"], []).append(sample)
        self.order = sorted(self.by_sequence)
        # Training draws one sequence per step so a long video does not
        # crowd a short one out of the shared UNet.
        self.uniform_sequences = bool(uniform_sequences)

    def __len__(self) -> int:
        if self.uniform_sequences:
            return len(self.order)
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if self.uniform_sequences:
            sample = random.choice(self.by_sequence[self.order[idx]])
        else:
            sample = self.samples[idx]
        bgr = cv2.imread(sample["path"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to read {sample['path']}")
        height, width = bgr.shape[:2]
        out_w = width // self.scale
        out_h = height // self.scale
        if out_w < 1 or out_h < 1:
            raise ValueError(f"scale {self.scale} does not fit in {width}x{height}")
        if self.scale > 1:
            bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        rgb = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).float() / 255.0
        return {
            "sequence": sample["sequence"],
            "frame": sample["frame"],
            "c2w": sample["c2w"],
            "color": srgb_to_linear(rgb),
            "mask": _frame_circle_mask(width, height, self.scale),
            "full_size": (width, height),
            "scale": self.scale,
        }


def resolve_screen_sequences(config: dict) -> tuple[list[str], str | None]:
    """
    Training sequence names, and the one sequence withheld from joint training.

    An explicit ``holdout_sequence`` of null keeps every discovered sequence
    in the training set. Otherwise the named sequence, or the last discovered
    name when more than one sequence is available, is withheld.
    """
    listed = config["model"].get("sequences")
    names = list(listed) if listed else discover_c3vd_sequences()
    if not names:
        raise RuntimeError("screen training has no C3VD sequences")
    if "holdout_sequence" in config["model"] and config["model"]["holdout_sequence"] is None:
        return names, None
    holdout = config["model"].get("holdout_sequence")
    if holdout is None:
        holdout = names[-1] if len(names) > 1 else None
    if holdout is not None and holdout not in names:
        raise RuntimeError(
            f"holdout sequence {holdout} is not available. Found {names}"
        )
    return names, holdout


def _screen_samples(
    names: list[str],
    split: str,
    holdout: str | None,
    modulo: int,
    data_root: str,
) -> list[dict]:
    samples = []
    for name in names:
        if split == "sequence":
            if name != holdout:
                continue
        elif holdout is not None and name == holdout:
            continue
        c2w, paths = _frame_table(os.path.join(data_root, name))
        kept = 0
        for frame, path in sorted(paths.items()):
            held_view = frame % modulo == 0
            if split == "train" and held_view:
                continue
            if split == "view" and not held_view:
                continue
            samples.append({
                "sequence": name,
                "frame": frame,
                "path": path,
                "c2w": torch.from_numpy(c2w[frame]),
            })
            kept += 1
        print(f"{split} {name}: {kept} frames")
    return samples


def build_screen_dataset(config: dict, split: str = "train") -> tuple[C3VDScreenDataset, list[str], str | None]:
    """
    Build a full-frame dataset.

    Returns the dataset, the sequences that receive a feature grid, and the
    held-out sequence name. ``split`` selects training frames, held-out
    views of those sequences, or the held-out sequence.
    """
    if split not in ("train", "view", "sequence"):
        raise ValueError(f"unknown screen split {split}")
    names, holdout = resolve_screen_sequences(config)
    modulo = int(config["model"].get("holdout_modulo", 8))
    if modulo < 2:
        raise ValueError(f"holdout_modulo must be at least 2, got {modulo}")
    train_names = [name for name in names if name != holdout]
    if split == "sequence" and holdout is None:
        raise RuntimeError("sequence split needs a holdout sequence")
    if split != "sequence" and not train_names:
        raise RuntimeError("every sequence was held out, so there is nothing to train")
    data_root = os.path.join("data", "c3vd", "sequences")
    if holdout is not None:
        _frame_table(os.path.join(data_root, holdout))
    source = [holdout] if split == "sequence" else train_names
    samples = _screen_samples(source, split, holdout, modulo, data_root)
    dataset = C3VDScreenDataset(
        samples,
        scale=int(config["model"].get("scale", 1)),
        uniform_sequences=(split == "train"),
    )
    print(
        f"Screen dataset split={split} scale={dataset.scale} "
        f"frames={len(samples)} steps_per_epoch={len(dataset)} "
        f"holdout={holdout} modulo={modulo}"
    )
    return dataset, train_names, holdout
