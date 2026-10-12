import torch
from typing import Union

from src.model.radiosity import RadiosityPipeline
from src.model.nerf import NeRF
from src.sample.lhs_rhs import first_smooth

import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")


class RadiosityIntegrator(mi.SamplingIntegrator):
    def __init__(
        self,
        model: Union[RadiosityPipeline, NeRF],
        precision=torch.float32
    ) -> None:
        props = mi.Properties()
        # Set ray direction to pixel center (Implemented in Mitsuba3)
        props["pixel_center"] = True

        super().__init__(props)

        self.model = model
        self.precision = precision

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True,
    ) -> tuple[mi.Color3f, bool, list[float]]:

        with torch.no_grad():
            color, valid = self.model.render(
                scene, sampler, ray, active, precision=self.precision
            )

        result = mi.Color3f(color.to(torch.float32))
        mask = mi.Float(valid.to(torch.float32)) > 0

        dr.sync_device()
        torch.cuda.synchronize()

        return result, mask, []
