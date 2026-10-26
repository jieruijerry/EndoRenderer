import math
import time

import numpy as np
import torch
import torch.nn as nn
import lightning as L
import tinycudann as tcnn

from src.module.basic import ShallowMLP
from src.module.hash_grid import HashGrid
from src.module.k_planes import KPlanes
from src.sample.util import extract_input, get_model_bbox, guard_query_pos, normalize_pos

import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")

from src.sensor.c3vd import register_c3vd_sensor, sensor_prefix

register_c3vd_sensor()


def get_time():
    dr.sync_device()
    torch.cuda.synchronize()
    return time.time()


def _disk_light(scene: mi.Scene):
    """Radius and radiance * radius^2 for the endoscope disk, if the scene has one."""
    params = mi.traverse(scene)
    pose_key = "EndoscopeLight.to_world"
    radiance_key = "EndoscopeLight.emitter.radiance.value"
    if pose_key not in params or radiance_key not in params:
        return None, None
    matrix = np.array(params[pose_key].matrix, dtype=np.float64).reshape(-1, 4, 4)[0]
    radius = float(np.linalg.norm(matrix[:3, 0]))
    radiance = np.array(params[radiance_key], dtype=np.float64).reshape(-1)[:3]
    return radius, radiance * radius ** 2


class GuiControl:
    """One scalar the viewer can edit."""

    def __init__(self, key: str, label: str, value: float, minimum: float, maximum: float) -> None:
        self.key = key
        self.label = label
        self.value = float(value)
        self.minimum = float(minimum)
        self.maximum = float(maximum)


class ImagePipeline(L.LightningModule):
    """
    Image-supervised pipeline. Subclasses implement ``query_model``.
    """

    def __init__(self, pipeline_config: dict, scene: mi.Scene) -> None:
        super().__init__()
        self.config = pipeline_config
        self.scene = scene
        self.register_buffer("bbox", get_model_bbox(scene))
        # L * r^2 stays constant, so changing the radius keeps the on-axis intensity.
        self.light_radius, self.light_radiance_ref = _disk_light(scene)

    def gui_controls(self) -> list:
        """Sliders the viewer should show. Empty unless a subclass opts in."""
        if self.light_radius is None:
            return []
        return [GuiControl("light_radius", "Light radius", self.light_radius, 0.05, 10.0)]

    def set_gui_control(self, key: str, value: float) -> None:
        """Apply one value reported by the viewer."""
        if key == "light_radius" and self.light_radius is not None:
            self.light_radius = float(value)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config["train"]["learning_rate"])

    def query_model(self, si: mi.SurfaceInteraction3f, precision=torch.float32):
        raise NotImplementedError("ImagePipeline is an abstract class, please implement query_model.")

    def render(self, scene, sampler, ray, active=True, precision=torch.float32):
        """
        Intersect ``ray`` and query outgoing color at the hit.
        Returns ``(color, valid, reg)`` with color zero on misses.
        ``reg`` is the optional scalar from ``query_model``, or ``None``.
        """
        del sampler
        si = scene.ray_intersect(ray, active)
        color, reg = self.query_model(si, precision=precision)
        valid = si.is_valid().torch().bool()
        color = torch.nan_to_num(color, nan=0.0, posinf=0.0, neginf=0.0)
        color = torch.where(valid.unsqueeze(-1), color, torch.zeros_like(color))
        return color, valid, reg

    def training_step(self, batch, batch_idx):
        c2w = batch["c2w"].detach().float().cpu().numpy()
        params = mi.traverse(self.scene)
        params[f"{sensor_prefix(params)}.to_world"] = mi.Matrix4f(c2w[None, ...])
        params.update()

        pixel = batch["pixel"]
        position = mi.Point2f(pixel[:, 0], pixel[:, 1])
        sensor = self.scene.sensors()[0]
        ray, _ = sensor.sample_ray(
            mi.Float(0.0),
            mi.Float(0.5),
            position,
            mi.Point2f(0.5, 0.5),
            True,
        )

        pred, valid, reg = self.render(self.scene, None, ray, True)
        color = batch["color"].to(device=pred.device, dtype=pred.dtype)
        if bool(valid.any()):
            loss = torch.mean((pred[valid] - color[valid]) ** 2)
        else:
            loss = pred.sum() * 0.0
        if torch.is_tensor(reg):
            # print(f"loss: {loss.item():.6f}, reg: {reg.detach().item():.6f}")
            loss = loss + reg
            self.log("normal_reg", reg.detach().item(), prog_bar=True)

        self.log("loss", loss.detach().item(), prog_bar=True)
        self.log("real_step", self.global_step, prog_bar=True)
        return loss


class NeRF(ImagePipeline):
    """
    Surface color model supervised by dataset pixel colors.
    The feature grid is K-Planes on (x, y, z, t). t is the disparity
    near / d, where d is the distance from the colocated light to the
    shading point, so nearby surfaces take more of the grid.
    """

    def __init__(self, pipeline_config: dict, scene: mi.Scene) -> None:
        super().__init__(pipeline_config, scene)

        config = pipeline_config["model"]["ray"]
        config["name"] = "KPlanes"
        config["n_dims_to_encode"] = 4
        self.encoding = KPlanes(config)

        sensor = scene.sensors()[0]
        self.near_clip = float(getattr(sensor, "near_clip", 0.1))

        if config["use_tcnn"]:
            network_config = {
                "otype": "FullyFusedMLP" if config["n_hidden_dims"] <= 128 else "CutlassMLP",
                "n_hidden_layers": config["n_hidden_layers"],
                "n_neurons": config["n_hidden_dims"],
                "activation": "ReLU",
                "output_activation": config["output_activation"],
            }
            self.mlp = tcnn.Network(
                n_input_dims=self.encoding.output_dim + config["n_dims_to_encode"] + 6,
                n_output_dims=3,
                network_config=network_config,
            )
        else:
            self.mlp = ShallowMLP(
                # encoding + xyzt + direction + normal
                in_channels=self.encoding.output_dim + config["n_dims_to_encode"] + 6,
                out_channels=3,
                hidden_layers=config["n_hidden_layers"],
                hidden_channels=config["n_hidden_dims"],
                activation=nn.ReLU(),
                output_activation=nn.Identity() if config["output_activation"] == "None" else nn.Softplus(),
            )

    def train(self, mode: bool = True):
        """
        Set the model to training mode
        """
        super().train(mode)
        if not mode:
            self.encoding.load_kernel("NeRF_hash_grid_cuda")
        else:
            self.encoding.use_kernel = False
        return self

    def query_model(self, si: mi.SurfaceInteraction3f, precision=torch.float32):
        dr.eval(si)
        pos, normal, direction, _, _, valid = extract_input(
            si, flip_dir=False, device=self.device, dtype=precision
        )

        pos = normalize_pos(pos, self.bbox)
        pos, keep = guard_query_pos(pos, valid)
        # Ray origin sits near_clip in front of the light, and the direction is unit length.
        depth = si.t.torch().to(device=pos.device, dtype=precision).unsqueeze(-1)
        depth = depth + self.near_clip
        disparity = self.near_clip / depth.clamp(min=self.near_clip)
        disparity = torch.nan_to_num(disparity, nan=0.5, posinf=0.5, neginf=0.5)
        disparity = disparity.clamp(0.0, 1.0 - 1e-4)
        disparity = torch.where(keep[:, None], disparity, torch.full_like(disparity, 0.5))
        xyzt = torch.cat([pos, disparity], dim=-1)
        t0 = get_time()
        enc = self.encoding(xyzt)
        t1 = get_time()
        enc = torch.cat([enc, xyzt, direction, normal], dim=-1)
        color = torch.abs(self.mlp(enc)).to(dtype=precision)
        t2 = get_time()
        # print(f"Encoding time: {t1 - t0:.6f}s, MLP time: {t2 - t1:.6f}s")
        color = torch.where(keep[:, None], color, torch.zeros_like(color))
        return color, None


def _mlp(
    config: dict,
    in_channels: int,
    out_channels: int,
    output_activation: str = "Sigmoid",
) -> nn.Module:
    activations = {
        "Sigmoid": nn.Sigmoid,
        "Tanh": nn.Tanh,
    }
    if output_activation not in activations:
        raise ValueError(f"Unsupported MLP output activation: {output_activation}")
    if config["use_tcnn"]:
        return tcnn.Network(
            n_input_dims=in_channels,
            n_output_dims=out_channels,
            network_config={
                "otype": "FullyFusedMLP" if config["n_hidden_dims"] <= 128 else "CutlassMLP",
                "n_hidden_layers": config["n_hidden_layers"],
                "n_neurons": config["n_hidden_dims"],
                "activation": "ReLU",
                "output_activation": output_activation,
            },
        )
    return ShallowMLP(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_layers=config["n_hidden_layers"],
        hidden_channels=config["n_hidden_dims"],
        activation=nn.ReLU(),
        output_activation=activations[output_activation](),
    )


def disney_colocated(albedo, roughness, metallic, cos, incident):
    """
    Simplified Disney BRDF for a light sitting on the camera, so wi = wo.
    The half vector is wo and Fresnel collapses to F0.
    ``cos`` is n·wo. ``incident`` is L0 / d^2.
    """
    cos = cos.clamp(1e-4, 1.0)
    alpha = roughness.clamp(1e-3, 1.0).square()
    alpha2 = alpha.square()
    cos2 = cos.square()

    ggx = cos2 * (alpha2 - 1.0) + 1.0
    distribution = alpha2 / (math.pi * ggx.square())
    smith = 2.0 * cos / (cos + torch.sqrt(alpha2 + (1.0 - alpha2) * cos2))
    geometry = smith.square()
    f0 = 0.04 * (1.0 - metallic) + albedo * metallic

    diffuse = (1.0 - metallic) / math.pi * albedo * cos
    specular = distribution * geometry / cos * f0
    return (diffuse + specular) * incident


class EndoPBR(ImagePipeline):
    """
    Colocated point light and a spatial Disney BRDF.

    The light is the camera, so the rendering equation is one direction:
    L0 is a global positive intensity and d is the light-to-point distance.
    Radiance is then passed through 1 - exp(-C) so the loss compares it
    with linear RGB in [0, 1). Albedo comes from the hash grid. Roughness
    and metallic are either one global value each, or a second MLP.
    """

    def __init__(self, pipeline_config: dict, scene: mi.Scene) -> None:
        super().__init__(pipeline_config, scene)

        config = dict(pipeline_config["model"]["ray"])
        config["name"] = "HashGrid"
        config["n_dims_to_encode"] = 3
        self.encoding = HashGrid(config)

        sensor = scene.sensors()[0]
        self.near_clip = float(getattr(sensor, "near_clip", 0.1))
        # exp(5) / (20 mm)^2 is order-1 radiance for a nearby wall.
        self.log_light = nn.Parameter(torch.tensor(5.0))

        in_channels = self.encoding.output_dim
        self.albedo_mlp = _mlp(config, in_channels, 3)
        self.global_rm = bool(pipeline_config["model"].get("global_rm", True))
        # if self.global_rm:
        #     # sigmoid(0) = 0.5. sigmoid(-2.944) ≈ 0.05, nearly dielectric.
        #     self.roughness_logit = nn.Parameter(torch.tensor(0.0))
        #     self.metallic_logit = nn.Parameter(torch.tensor(-2.944438879))
        # else:
        #     self.rm_mlp = _mlp(config, in_channels, 2)

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            self.encoding.load_kernel("EndoPBR_hash_grid_cuda")
        else:
            self.encoding.use_kernel = False
        return self

    def query_model(self, si: mi.SurfaceInteraction3f, precision=torch.float32):
        dr.eval(si)
        pos, normal, view, _, _, valid = extract_input(
            si, flip_dir=False, device=self.device, dtype=precision
        )
        pos = normalize_pos(pos, self.bbox)
        pos, keep = guard_query_pos(pos, valid)

        depth = si.t.torch().to(device=pos.device, dtype=precision).unsqueeze(-1)
        depth = torch.nan_to_num(depth + self.near_clip, nan=self.near_clip, posinf=self.near_clip)
        depth = depth.clamp(min=self.near_clip)
        incident = torch.exp(self.log_light) / depth.square()

        encoded = self.encoding(pos)
        albedo = self.albedo_mlp(encoded).to(dtype=precision)
        # if self.global_rm:
        #     roughness = torch.sigmoid(self.roughness_logit).expand(albedo.shape[0], 1)
        #     metallic = torch.sigmoid(self.metallic_logit).expand(albedo.shape[0], 1)
        #     print(f"Global roughness: {roughness.mean().item():.4f}, metallic: {metallic.mean().item():.4f}")
        # else:
        #     rm = self.rm_mlp(encoded).to(dtype=precision)
        #     roughness = rm[:, 0:1]
        #     metallic = rm[:, 1:2]
        # roughness = roughness.to(dtype=precision)
        # metallic = metallic.to(dtype=precision)

        roughness = torch.tensor(0.1)
        metallic = torch.tensor(0.001)

        view = view / view.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cos = (normal * view).sum(dim=-1, keepdim=True)
        radiance = disney_colocated(albedo, roughness, metallic, cos, incident)
        color = 1.0 - torch.exp(-radiance.clamp(min=0.0))
        color = torch.where(keep[:, None], color, torch.zeros_like(color))
        return color.to(dtype=precision), None


def _frame_axis(axis, device, dtype) -> torch.Tensor:
    values = axis.torch().to(device=device, dtype=dtype)
    return torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


class EndoRen(ImagePipeline):
    """
    Colocated Disney BRDF with a global material and local residuals.

    Albedo is one learnable color plus a small hash-grid offset. Roughness
    and metallic are global. The shading normal is the mesh normal plus a
    tangent-space bump from the same grid. Lighting matches EndoPBR: L0 / d^2
    and 1 - exp(-C).
    """

    def __init__(self, pipeline_config: dict, scene: mi.Scene) -> None:
        super().__init__(pipeline_config, scene)

        config = dict(pipeline_config["model"]["ray"])
        config["name"] = "HashGrid"
        config["n_dims_to_encode"] = 3
        self.encoding = HashGrid(config)

        sensor = scene.sensors()[0]
        self.near_clip = float(getattr(sensor, "near_clip", 0.1))
        # exp(5) / (20 mm)^2 is order-1 radiance for a nearby wall.
        self.log_light = nn.Parameter(torch.tensor(5.0))

        model_config = pipeline_config["model"]
        self.normal_offset_scale = float(model_config.get("normal_offset_scale", 5.0))
        self.albedo_offset_scale = float(model_config.get("albedo_offset_scale", 0.3))

        in_channels = self.encoding.output_dim
        self.local_mlp = _mlp(config, in_channels, 5, output_activation="Tanh")
        if isinstance(self.local_mlp, ShallowMLP) and len(self.local_mlp.layers) > 0:
            last = self.local_mlp.layers[-1]
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

        # Viewer slider. Not a checkpoint parameter.
        self.roughness = 0.1

        # sigmoid ≈ (0.75, 0.35, 0.30), a reddish mucosa color.
        self.albedo_logit = nn.Parameter(torch.tensor([
            math.log(0.75 / 0.25),
            math.log(0.35 / 0.65),
            math.log(0.30 / 0.70),
        ]))
        # # sigmoid = 0.1, matching the EndoPBR constant.
        # self.roughness_logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        # # sigmoid = 0.05. A logit near 0.001 has almost no gradient.
        # self.metallic_logit = nn.Parameter(torch.tensor(math.log(0.05 / 0.95)))

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            self.encoding.load_kernel("EndoRen_hash_grid_cuda")
        else:
            self.encoding.use_kernel = False
        return self

    def gui_controls(self) -> list:
        controls = super().gui_controls()
        controls.append(GuiControl("roughness", "Roughness", self.roughness, 1e-3, 1.0))
        return controls

    def set_gui_control(self, key: str, value: float) -> None:
        if key == "roughness":
            self.roughness = float(value)
            return
        super().set_gui_control(key, value)

    def query_model(self, si: mi.SurfaceInteraction3f, precision=torch.float32):
        dr.eval(si)
        pos, _, view, _, _, valid = extract_input(
            si, flip_dir=False, device=self.device, dtype=precision
        )
        pos = normalize_pos(pos, self.bbox)
        pos, keep = guard_query_pos(pos, valid)

        geo_n = _frame_axis(si.sh_frame.n, pos.device, precision)
        tangent = _frame_axis(si.sh_frame.s, pos.device, precision)
        bitangent = _frame_axis(si.sh_frame.t, pos.device, precision)

        depth = si.t.torch().to(device=pos.device, dtype=precision).unsqueeze(-1)
        depth = torch.nan_to_num(depth + self.near_clip, nan=self.near_clip, posinf=self.near_clip)
        depth = depth.clamp(min=self.near_clip)
        incident = torch.exp(self.log_light) / depth.square()

        encoded = self.encoding(pos)
        residual = self.local_mlp(encoded).to(dtype=precision)
        normal_offset = residual[:, 0:2] * self.normal_offset_scale
        albedo_offset = residual[:, 2:5] * self.albedo_offset_scale
        weight = keep.to(dtype=normal_offset.dtype).unsqueeze(-1)
        mean_offset = (normal_offset * weight).sum(dim=0) / weight.sum().clamp(min=1.0)
        normal_reg = mean_offset.abs().sum()

        albedo = torch.sigmoid(self.albedo_logit).to(dtype=precision) + albedo_offset
        albedo = albedo.clamp(0.0, 1.0)
        shading_n = geo_n + normal_offset[:, 0:1] * tangent + normal_offset[:, 1:2] * bitangent
        shading_n = shading_n / shading_n.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # shading_n = (shading_n + 1) * 0.5
        # return shading_n, normal_reg
        # return albedo, normal_reg

        n = albedo.shape[0]
        # roughness = torch.sigmoid(self.roughness_logit).to(dtype=precision).expand(n, 1)
        # metallic = torch.sigmoid(self.metallic_logit).to(dtype=precision).expand(n, 1)

        roughness = torch.tensor(self.roughness, device=pos.device, dtype=precision)
        metallic = torch.tensor(0.001, device=pos.device, dtype=precision)

        view = view / view.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cos = (shading_n * view).sum(dim=-1, keepdim=True)
        radiance = disney_colocated(albedo, roughness, metallic, cos, incident)
        color = 1.0 - torch.exp(-radiance.clamp(min=0.0))
        color = torch.where(keep[:, None], color, torch.zeros_like(color))
        return color.to(dtype=precision), 0.25 * normal_reg