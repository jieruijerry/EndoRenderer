import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import tinycudann as tcnn

from src.module.basic import ShallowMLP
from src.module.hash_grid import HashGrid
from src.module.k_planes import KPlanes, PVPlanes
from src.module.octa_grid import OctaGrid
from src.module.octree import OctreeEncoding
from src.module.octa_octree import OctaOctreeEncoding
from src.sample.lhs_rhs import LHSRHS, extract_input, get_wr_itsc
from src.sample.util import *

import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")


def get_time():
    dr.sync_device()
    torch.cuda.synchronize()
    return time.time()


class RadiosityPipeline(L.LightningModule):
    """
    Radiosity pipeline
    """
    def __init__(self, config: dict, scene: mi.Scene) -> None:
        super(RadiosityPipeline, self).__init__()
        self.config = config
        self.scene = scene
        self.register_buffer("bbox", get_model_bbox(scene))

    def training_step(self, batch, batch_idx):
        """
        Training step for the model
        """
        seed = self.global_step * self.trainer.world_size + self.global_rank

        # Adaptive RHS and epsilon
        ad_ratio = 2 ** int(4 * (self.global_step / self.trainer.max_steps))
        point_num = self.config["sample"]["n_points"] // ad_ratio
        dirs_per_point = self.config["sample"]["n_dirs_per_point"] * ad_ratio
        eps = 1e-1 / ad_ratio

        # Sample
        lhs_rhs = LHSRHS(
            scene=self.scene,
            point_num=point_num,
            dirs_per_point=dirs_per_point,
        )
        if self.config["sample"]["from_poses"]:
            lhs_rhs.sample(seed=seed, pose=batch)
        else:
            lhs_rhs.sample(seed=seed)
        lhs_rhs.to(device=self.device)

        # Forward pass
        result = self(lhs_rhs)
        lhs_color = result["lhs"]
        rhs_color = result["rhs"].detach()

        # Compute loss
        nr_norm = (rhs_color + lhs_color).detach() / 2 + eps
        loss = torch.mean(((rhs_color - lhs_color) / nr_norm) ** 2) / ad_ratio

        if result["reg"] is not None:
            reg_loss = torch.mean(result["reg"])
            loss += self.config["train"]["reg_lambda"] * reg_loss
            self.log("reg_loss", reg_loss.item(), prog_bar=True)

        # Logging
        self.log("loss", loss.item(), prog_bar=True)
        self.log("real_step", self.global_step, prog_bar=True)

        if (self.global_step + 1) % 200 == 0 and self.global_rank == 0:
            print(f"##### Step {self.global_step + 1} #####")
            print("lhs color", lhs_color[:3])
            print("rhs color", rhs_color[:3])

        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config["train"]["learning_rate"])
    
    def forward(self, lhs_rhs: LHSRHS) -> torch.Tensor:
        """
        Query the model with lhs and rhs interactions
        """
        si_lhs = lhs_rhs.si_lhs
        si_rhs = lhs_rhs.si_bsdf

        lhs_color, reg = self.query_model(si_lhs)
        with torch.no_grad():
            rhs_color, _ = self.query_model(si_rhs)

        # Render rhs color
        rhs_color = rhs_color.reshape(-1, lhs_rhs.dirs_per_point, 3)
        rhs_color = lhs_rhs.render(rhs_color, None)

        return {
            "lhs": lhs_color,
            "rhs": rhs_color,
            "reg": reg,
        }
    
    def query_model(self, si: mi.SurfaceInteraction3f, precision=torch.float32) -> torch.Tensor:
        """
        Query the model with surface interaction
        """
        raise NotImplementedError("RadiosityPipeline is an abstract class, please implement query_model method in the subclass.")

    def render(self, scene, sampler, ray, active=True, precision=torch.float32):
        """
        Intersect ``ray`` and query outgoing color at the hit.
        Returns ``(color, valid)`` with color zero on misses.
        """
        del sampler
        si = scene.ray_intersect(ray, active)
        color, _ = self.query_model(si, precision=precision)
        valid = si.is_valid().torch().bool()
        color = torch.nan_to_num(color, nan=0.0, posinf=0.0, neginf=0.0)
        color = torch.where(valid.unsqueeze(-1), color, torch.zeros_like(color))
        return color, valid
    

class NeuralRadiosity(RadiosityPipeline):
    """
    Neural Radiosity model
    """
    def __init__(self, pipeline_config: dict, scene: mi.Scene) -> None:
        super(NeuralRadiosity, self).__init__(pipeline_config, scene)

        config = pipeline_config["model"]["ray"]

        if config["name"] == "HashGrid":
            self.encoding = HashGrid(config)
        elif config["name"] == "KPlanes":
            self.encoding = KPlanes(config)
        elif config["name"] == "CFKGrid":
            self.encoding = OctaGrid(config)
        elif config["name"] == "Octree":
            # Sample points from the scene, normalize to [-1, 1]
            points = sample_si(scene, 2 ** 26).p.torch()
            points = normalize_pos(points, self.bbox.to(device=points.device)) * 2 - 1
            # Initialize octree encoding
            self.encoding = OctreeEncoding(config, points)
        else:
            raise NotImplementedError()

        if config["use_tcnn"]:
            network_config = {
                "otype": "FullyFusedMLP" if config["n_hidden_dims"] <= 128 else "CutlassMLP",
                "n_hidden_layers": config["n_hidden_layers"],
                "n_neurons": config["n_hidden_dims"],
                "activation": "ReLU",
                "output_activation": config["output_activation"],
            }
            self.mlp = tcnn.Network(
                n_input_dims=self.encoding.output_dim + 3 * 3 + 1 + config["n_dims_to_encode"],
                n_output_dims=3,
                network_config=network_config
            )
        else:
            self.mlp = ShallowMLP(
                # encoding + pos + normal + wr + albedo + roughness
                in_channels=self.encoding.output_dim + 3 * 3 + 1 + config["n_dims_to_encode"],
                out_channels=3,
                hidden_layers=config["n_hidden_layers"],
                hidden_channels=config["n_hidden_dims"],
                activation=nn.ReLU(),
                output_activation=nn.Identity() if config["output_activation"] == "None" else nn.Softplus()
            )

    def train(self, mode: bool = True):
        """
        Set the model to training mode
        """
        super().train(mode)
        # if not mode:
        #     self.hash_grid.load_kernel("NR_hash_grid_cuda")
        # else:
        #     self.hash_grid.use_kernel = False
        return self

    def query_model(self, si: mi.SurfaceInteraction3f, precision=torch.float32) -> torch.Tensor:
        """
        Query the model with surface interaction
        """
        dr.eval(si)
        pos, normal, dir, albedo, roughness, valid = extract_input(si, device=self.device, dtype=precision)

        # Hash grid encoding
        pos = normalize_pos(pos, self.bbox)
        pos, keep = guard_query_pos(pos, valid)
        enc = self.encoding(pos)

        # Concatenate encoding with wr_direction and roughness
        enc = torch.cat([enc, pos, dir, normal, albedo, roughness], dim=-1)

        # Pass through MLP
        color = torch.abs(self.mlp(enc)).to(dtype=precision)
        color = torch.where(keep[:, None], color, torch.zeros_like(color))

        return color, None
