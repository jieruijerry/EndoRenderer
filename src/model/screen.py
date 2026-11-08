"""
Screen-space C3VD renderer.

Each sequence owns a K-Planes grid on (x, y, z, disparity). The grid stores
surface appearance only. Normals, view direction, disparity, n·v, and the
endoscope disk radius stay out of the grid, because a highlight is the image
of that disk and has to slide when the camera moves. One UNet, shared by
every training sequence, reads the feature image and those buffers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from src.model.nerf import _disk_light
from src.module.k_planes import KPlanes
from src.sample.util import extract_input, get_model_bbox, guard_query_pos, normalize_pos

import drjit as dr
import mitsuba as mi

from src.sensor.c3vd import sensor_prefix
from src.dataset.c3vd_color import C3VD_CX, C3VD_CY, C3VD_IMAGE_RADIUS


# normal 3, view 3, disparity, n·v, disk radius, pixel offset from the principal point
GEOM_CHANNELS = 11


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = _groups(out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ScreenUNet(nn.Module):
    """
    UNet that halves resolution at every encoder stage and doubles it back.

    ``channels`` lists the width of each encoder resolution, then the
    bottleneck. The first two entries are the full-resolution and
    half-resolution stages, and stay narrow so a 1350×1080 frame fits.
    Spatial size need not be divisible by a power of two: the input is
    padded and the pad is cropped off the output.
    """

    def __init__(self, in_channels: int, channels: list[int]) -> None:
        super().__init__()
        if len(channels) < 3:
            raise ValueError(f"need a bottleneck and at least two encoder stages, got {channels}")
        self.in_channels = int(in_channels)
        self.channels = [int(ch) for ch in channels]
        encoders = []
        ch_in = self.in_channels
        for ch in self.channels[:-1]:
            encoders.append(_ConvBlock(ch_in, ch))
            ch_in = ch
        self.encoders = nn.ModuleList(encoders)
        self.bot = _ConvBlock(ch_in, self.channels[-1])
        decoders = []
        prev = self.channels[-1]
        for skip in reversed(self.channels[:-1]):
            decoders.append(_ConvBlock(prev + skip, skip))
            prev = skip
        self.decoders = nn.ModuleList(decoders)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.head = nn.Conv2d(self.channels[0], 3, 1)
        # Near 0.5 at init, but not exactly independent of the feature image,
        # so the first step can update the grids.
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        factor = 2 ** (len(self.encoders))
        pad_h = (factor - height % factor) % factor
        pad_w = (factor - width % factor) % factor
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        skips = []
        hidden = x
        for encoder in self.encoders:
            hidden = encoder(hidden)
            skips.append(hidden)
            hidden = self.pool(hidden)
        hidden = self.bot(hidden)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            hidden = self.up(hidden)
            if hidden.shape[-2:] != skip.shape[-2:]:
                hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            hidden = decoder(torch.cat([hidden, skip], dim=1))
        color = torch.sigmoid(self.head(hidden))
        return color[..., :height, :width]


def _ray_config(pipeline_config: dict) -> dict:
    config = dict(pipeline_config["model"]["ray"])
    config["name"] = "KPlanes"
    config["n_dims_to_encode"] = 4
    return config


class ScreenSpace(L.LightningModule):
    """
    Per-sequence K-Planes plus one UNet trained on every C3VD sequence.

    ``scenes`` maps a sequence name to its Mitsuba scene. Only names in that
    map get a grid. The held-out sequence is absent on purpose.
    """

    def __init__(self, pipeline_config: dict, scenes: dict) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["scenes"])
        if not scenes:
            raise RuntimeError("ScreenSpace needs at least one training scene")
        self.config = pipeline_config
        names = sorted(scenes.keys())
        self.sequence_names = names
        self.scenes = {name: scenes[name] for name in names}
        self.sequence_index = {name: i for i, name in enumerate(names)}

        ray_config = _ray_config(pipeline_config)
        self.grids = nn.ModuleDict({name: KPlanes(ray_config) for name in names})
        feature_dim = self.grids[names[0]].output_dim
        channels = pipeline_config["model"].get("channels", [8, 16, 32, 64, 128, 256])
        self.unet = ScreenUNet(feature_dim + GEOM_CHANNELS, channels)

        bboxes = []
        near_clips = []
        radii = []
        fallback = float(pipeline_config["model"].get("light_radius", 1.0))
        for name in names:
            scene = self.scenes[name]
            bboxes.append(get_model_bbox(scene, device="cpu"))
            sensor = scene.sensors()[0]
            near_clips.append(float(getattr(sensor, "near_clip", 0.1)))
            radius, _ = _disk_light(scene)
            radii.append(fallback if radius is None else radius)
        self.register_buffer("bboxes", torch.stack(bboxes))
        self.register_buffer("near_clips", torch.tensor(near_clips, dtype=torch.float32))
        self.register_buffer("light_radii", torch.tensor(radii, dtype=torch.float32))
        self._checked_films = set()

    def configure_optimizers(self):
        train = self.config["train"]
        grid_lr = float(train.get("grid_learning_rate", train["learning_rate"]))
        return torch.optim.Adam([
            {"params": self.grids.parameters(), "lr": grid_lr},
            {"params": self.unet.parameters(), "lr": float(train["learning_rate"])},
        ])

    def _set_camera(self, scene: mi.Scene, c2w: torch.Tensor) -> None:
        params = mi.traverse(scene)
        pose = c2w.detach().float().cpu().numpy()
        params[f"{sensor_prefix(params)}.to_world"] = mi.Matrix4f(pose[None, ...])
        params.update()

    def _check_film(self, sequence: str, scene: mi.Scene, width: int, height: int) -> None:
        if sequence in self._checked_films:
            return
        params = mi.traverse(scene)
        film = params[f"{sensor_prefix(params)}.film.size"].numpy()
        film_w, film_h = int(film[0]), int(film[1])
        if film_w != width or film_h != height:
            raise RuntimeError(
                f"{sequence} image is {width}x{height} but the sensor film is {film_w}x{film_h}"
            )
        self._checked_films.add(sequence)

    def feature_image(
        self,
        sequence: str,
        c2w: torch.Tensor,
        full_size: tuple[int, int],
        out_size: tuple[int, int],
        scale: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Raycast one full frame and pack the UNet input.

        ``out_size`` is the network resolution. Pixel centers are spaced by
        ``scale`` in the original film, so a coarser frame is a uniform
        shrink of the whole fisheye, not a window. Returns ``(1, C, H, W)``
        and a ``(H, W)`` hit mask. Misses are zeros and are not written into
        the feature grid.
        """
        scene = self.scenes[sequence]
        width, height = int(full_size[0]), int(full_size[1])
        out_h, out_w = int(out_size[0]), int(out_size[1])
        self._check_film(sequence, scene, width, height)
        self._set_camera(scene, c2w)

        # cuda_rgb Point2f is a GPU array, so the pixel coordinates have to live there too.
        device = self.device
        xs = (torch.arange(out_w, device=device, dtype=torch.float32) + 0.5) * scale
        ys = (torch.arange(out_h, device=device, dtype=torch.float32) + 0.5) * scale
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        position = mi.Point2f(
            (grid_x / width).reshape(-1),
            (grid_y / height).reshape(-1),
        )
        sensor = scene.sensors()[0]
        ray, _ = sensor.sample_ray(
            mi.Float(0.0),
            mi.Float(0.5),
            position,
            mi.Point2f(0.5, 0.5),
            True,
        )
        si = scene.ray_intersect(ray, True)
        dr.eval(si)

        image, keep = self._channels(sequence, si, grid_x, grid_y)
        return image, keep.view(out_h, out_w)

    def _sequence_of(self, scene: mi.Scene) -> str:
        """Which grid to use for this Mitsuba scene."""
        chosen = getattr(self, "render_sequence", None)
        if chosen in self.grids:
            return chosen
        for name, owned in self.scenes.items():
            if owned is scene and name in self.grids:
                return name
        for shape in scene.shapes():
            sid = shape.id()
            if sid.endswith("_mesh"):
                name = sid[: -len("_mesh")]
                if name in self.grids:
                    return name
        raise RuntimeError(
            f"no ScreenSpace grid for this scene. Available: {list(self.grids.keys())}"
        )

    def _channels(
        self,
        sequence: str,
        si: mi.SurfaceInteraction3f,
        grid_x: torch.Tensor,
        grid_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pack one ray per pixel into a ``(1, C, H, W)`` UNet image.

        ``grid_x`` and ``grid_y`` are original-film pixel centers, shaped
        ``(H, W)`` in row-major order, matching the ray batch.
        """
        pos, normal, view, _, _, valid = extract_input(
            si, flip_dir=False, device=self.device, dtype=torch.float32
        )
        index = self.sequence_index[sequence]
        pos = normalize_pos(pos, self.bboxes[index])
        pos, keep = guard_query_pos(pos, valid)
        height, width = grid_y.shape
        if pos.shape[0] != height * width:
            raise RuntimeError(
                f"ScreenSpace expected {height}*{width}={height * width} rays, got {pos.shape[0]}"
            )

        near = self.near_clips[index]
        depth = si.t.torch().to(device=pos.device, dtype=torch.float32).unsqueeze(-1)
        depth = torch.nan_to_num(depth + near, nan=near, posinf=near).clamp(min=near)
        disparity = (near / depth).clamp(0.0, 1.0 - 1e-4)
        disparity = torch.where(keep[:, None], disparity, torch.full_like(disparity, 0.5))

        feature_dim = self.grids[sequence].output_dim
        features = torch.zeros(pos.shape[0], feature_dim, device=pos.device, dtype=torch.float32)
        if bool(keep.any()):
            xyzt = torch.cat([pos[keep], disparity[keep]], dim=-1)
            with torch.autocast(device_type=pos.device.type, enabled=False):
                queried = self.grids[sequence](xyzt)
            # index_add keeps the grid in the graph. A slice write into a
            # constant tensor would drop it.
            rows = torch.nonzero(keep, as_tuple=False).squeeze(1)
            features = features.index_add(0, rows, queried)

        normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        view = view / view.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        ndotv = (normal * view).sum(dim=-1, keepdim=True)
        radius = self.light_radii[index].to(dtype=features.dtype).reshape(1, 1).expand(pos.shape[0], 1)
        # Offset from the principal point, in units of the image-circle radius.
        # Kept even on ray misses, so the black rim is not confused with the center.
        coord_u = ((grid_x - C3VD_CX) / C3VD_IMAGE_RADIUS).reshape(-1, 1)
        coord_v = ((grid_y - C3VD_CY) / C3VD_IMAGE_RADIUS).reshape(-1, 1)
        surface = torch.cat([features, normal, view, disparity, ndotv, radius], dim=-1)
        surface = torch.where(keep[:, None], surface, torch.zeros_like(surface))
        channels = torch.cat([surface, coord_u, coord_v], dim=-1)
        if channels.shape[-1] != self.unet.in_channels:
            raise RuntimeError(
                f"feature image has {channels.shape[-1]} channels, UNet expects {self.unet.in_channels}"
            )
        image = channels.contiguous().view(height, width, -1).permute(2, 0, 1).unsqueeze(0)
        return image, keep

    def render(self, scene, sampler, ray, active=True, precision=torch.float32):
        """
        Shade one ray per pixel, like the other pipelines.

        The ray batch has to be a full film in row-major order, so its width
        equals ``H * W``. Those rays are reshaped into an image and passed
        through the UNet. Returns ``(color, valid, None)`` with color zero
        on misses.
        """
        del sampler
        si = scene.ray_intersect(ray, active)
        dr.eval(si)
        params = mi.traverse(scene)
        film = params[f"{sensor_prefix(params)}.film.size"].numpy()
        width, height = int(film[0]), int(film[1])
        n = int(dr.width(ray.o.x))
        if n != height * width:
            raise RuntimeError(
                f"ScreenSpace.render expected {height}*{width}={height * width} rays, got {n}"
            )
        device = self.device
        xs = torch.arange(width, device=device, dtype=torch.float32) + 0.5
        ys = torch.arange(height, device=device, dtype=torch.float32) + 0.5
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        image, keep = self._channels(self._sequence_of(scene), si, grid_x, grid_y)
        pred = self.unet(image)
        color = pred.permute(0, 2, 3, 1).reshape(n, 3)
        color = torch.nan_to_num(color, nan=0.0, posinf=0.0, neginf=0.0)
        valid = keep.reshape(n)
        color = torch.where(valid[:, None], color, torch.zeros_like(color))
        return color.to(dtype=precision), valid, None

    def training_step(self, batch, batch_idx):
        sequence = batch["sequence"]
        color = batch["color"]
        image, hit = self.feature_image(
            sequence,
            batch["c2w"],
            batch["full_size"],
            (int(color.shape[0]), int(color.shape[1])),
            int(batch["scale"]),
        )
        pred = self.unet(image)
        target = batch["color"].to(device=pred.device, dtype=pred.dtype).permute(2, 0, 1).unsqueeze(0)
        circle = batch["mask"].to(device=hit.device)
        loss_mask = circle & hit
        if bool(loss_mask.any()):
            per_pixel = (pred - target).abs().mean(dim=1)
            weight = loss_mask.to(dtype=per_pixel.dtype)
            loss = (per_pixel * weight).sum() / weight.sum().clamp(min=1.0)
        else:
            loss = pred.sum() * 0.0
        self.log("loss", loss, prog_bar=True)
        self.log("hit_fraction", loss_mask.to(dtype=loss.dtype).mean(), prog_bar=False)
        return loss
