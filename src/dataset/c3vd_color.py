import os

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
