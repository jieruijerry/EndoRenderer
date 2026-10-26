# Basics
import os
import json
import argparse
import imgui
import time

# Computational
import numpy as np
import torch

# Custom
from src.util.progress_bar import find_best_ckpt
from src.model.radiosity import *
from src.model.nerf import EndoPBR, EndoRen, NeRF
from src.integrator.neural import *
from src.integrator.path import *
from src.integrator.g_buffer import *
from src.integrator.ao import *
from src.viewer.camera import FPSCamera
from src.viewer.ui import UI
# Mitsuba
import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")

from src.dataset.c3vd_color import parse_mitsuba_c2w
from src.sensor.c3vd import register_c3vd_sensor, sensor_prefix

register_c3vd_sensor()


def _light_key(params) -> str | None:
    """Scene parameter for the endoscope light, disk preferred over spotlight."""
    if "EndoscopeLight.to_world" in params:
        return "EndoscopeLight.to_world"
    if "SpotLight.to_world" in params:
        return "SpotLight.to_world"
    return None


def _pose_with_radius(pose: np.ndarray, radius: float) -> np.ndarray:
    """Keep a pose's rotation and translation, and set its uniform scale."""
    pose = np.array(pose, dtype=np.float64).reshape(-1, 4, 4)[0].copy()
    current = float(np.linalg.norm(pose[:3, 0]))
    pose[:3, :3] *= radius / max(current, 1e-8)
    return pose


def update_scene(params, camera, model, follow_light: bool) -> None:
    """
    Write the viewer state into the Mitsuba scene.

    The camera, the endoscope light, and the tissue roughness are separate
    blocks. ``params.update()`` runs once after all of them.
    """
    # Camera. The sensor pose is the viewer camera. x_fov exists only on
    # perspective sensors; the C3VD sensor keeps its own intrinsics.
    cam_to_world = camera.get_transform()
    prefix = sensor_prefix(params)
    params[f"{prefix}.to_world"] = mi.Matrix4f(cam_to_world[None, ...])
    fov_key = f"{prefix}.x_fov"
    if fov_key in params:
        params[fov_key] = mi.Float32(camera.get_x_fov()[None, ...])

    # Light. Follow the camera when requested. A disk also uses the
    # pipeline's light radius; radiance is reference / r^2 so the on-axis
    # intensity stays constant as the disk changes size.
    light_key = _light_key(params)
    if light_key == "EndoscopeLight.to_world":
        radius = getattr(model, "light_radius", None)
        if radius is None:
            radius = float(np.linalg.norm(
                np.array(params[light_key].matrix, dtype=np.float64).reshape(-1, 4, 4)[0, :3, 0]
            ))
        radius = float(radius)
        if follow_light:
            pose = cam_to_world.copy()
            pose[:3, :3] *= radius
        else:
            pose = _pose_with_radius(params[light_key].matrix, radius)
        params[light_key] = mi.Matrix4f(pose[None, ...])
        ref = getattr(model, "light_radiance_ref", None)
        if ref is not None:
            radiance = np.asarray(ref, dtype=np.float64) / radius ** 2
            params["EndoscopeLight.emitter.radiance.value"] = mi.Color3f(
                float(radiance[0]), float(radiance[1]), float(radiance[2])
            )
    elif light_key is not None and follow_light:
        params[light_key] = mi.Matrix4f(cam_to_world[None, ...])

    # Material. The slider is Disney roughness. disney_colocated squares it
    # to get the GGX alpha, which is what roughplastic stores as ``alpha``.
    roughness = getattr(model, "roughness", None)
    if roughness is not None and "tissue.alpha" in params:
        params["tissue.alpha"] = mi.Float(float(roughness) ** 2)

    params.update()


def load_dataset_poses(scene_name: str):
    """C3VD pose.txt rows as Mitsuba camera matrices, or None."""
    pose_path = os.path.join(
        "data", "c3vd", "sequences", os.path.basename(scene_name), "pose.txt"
    )
    if not os.path.isfile(pose_path):
        return None
    with open(pose_path) as f:
        return [parse_mitsuba_c2w(line) for line in f if line.strip()]


def load_render_vars(config: dict, args: argparse.Namespace):
    """
    Load render variables
    """
    # Load scene
    manager = None
    scene = mi.load_file(os.path.join("scenes", args.scene, "scene.xml"))
    params = mi.traverse(scene)

    # Load model
    ckpt_dir = os.path.join("out", args.scene, "checkpoints", config["model"]["name"])
    if args.model_ckpt is not None:
        ckpt_path = os.path.join(ckpt_dir, args.model_ckpt + ".ckpt")
    elif config["model"]["name"] == "NULL":
        ckpt_path = None
    else:
        ckpt_path, _ = find_best_ckpt(ckpt_dir, metric="loss")

    model_name = config["model"]["name"]
    if model_name.startswith("EndoPBR"):
        model = EndoPBR.load_from_checkpoint(
            ckpt_path,
            pipeline_config=config,
            scene=scene
        )
    elif model_name.startswith("EndoRen"):
        model = EndoRen.load_from_checkpoint(
            ckpt_path,
            pipeline_config=config,
            scene=scene
        )
    elif model_name.startswith("NeRF"):
        model = NeRF.load_from_checkpoint(
            ckpt_path,
            pipeline_config=config,
            scene=scene
        )
    elif model_name[:2] == "NR":
        model = NeuralRadiosity.load_from_checkpoint(
            ckpt_path,
            pipeline_config=config,
            scene=scene
        )
    elif model_name == "NULL":
        model = torch.nn.Module()
    print("Restoring model from", ckpt_path)
    if args.half_precision:
        model = model.half().cuda()
    model.eval()

    # Initialize integrator
    nr_integrator = RadiosityIntegrator(
        model=model,
        precision=torch.float16 if args.half_precision else torch.float32,
    )
    path_integrator = mi.load_dict({
        "type": "pt",
        "max_depth": 16,
    })
    depth_integrator = mi.load_dict({
        "type": "depth"
    })
    albedo_integrator = mi.load_dict({
        "type": "albedo"
    })
    normal_integrator = mi.load_dict({
        "type": "normal"
    })
    ao_integrator = mi.load_dict({
        "type": "ao"
    })

    prefix = sensor_prefix(params)
    width, height = params[f"{prefix}.film.size"].numpy()
    fov_key = f"{prefix}.x_fov"
    # The endoscope intrinsics are fixed. x_fov is only the viewer dolly state.
    x_fov = float(params[fov_key].numpy()[0]) if fov_key in params else 126.0
    extrinsic = params[f"{prefix}.to_world"].matrix.numpy()[0]
    camera = FPSCamera({
        "width": width,
        "height": height,
        "x_fov": x_fov,
    }, extrinsic, 0.2)

    return {
        "scene": scene,
        "manager": manager,
        "integrators": {
            args.config: nr_integrator,
            "path": path_integrator,
            "depth": depth_integrator,
            "albedo": albedo_integrator,
            "normal": normal_integrator,
            "ao": ao_integrator,
        },
        "camera": camera,
    }

def render(config: dict, args: argparse.Namespace):
    """
    Render scene
    """
    # Load render variables
    render_vars = load_render_vars(config, args)
    scene: mi.Scene = render_vars["scene"]
    params = mi.traverse(scene)
    nr_integrator: RadiosityIntegrator = render_vars["integrators"][args.config]
    path_integrator = render_vars["integrators"]["path"]
    depth_integrator = render_vars["integrators"]["depth"]
    albedo_integrator = render_vars["integrators"]["albedo"]
    normal_integrator = render_vars["integrators"]["normal"]
    ao_integrator = render_vars["integrators"]["ao"]
    camera: FPSCamera = render_vars["camera"]
    width, height = camera.width, camera.height

    # Initialize UI
    ui = UI(width, height, camera, bbox=scene.bbox())

    # UI variables
    int_type = 0
    slider_spp = 1
    spp = 1
    use_antialiasing = False
    exposure = 1.0
    save_img = False
    light_key = _light_key(params)
    follow_light = True
    pose_index = 0
    dataset_poses = load_dataset_poses(args.scene)

    while not ui.should_close():
        ui.begin_frame()

        if imgui.tree_node("Render Options", imgui.TREE_NODE_DEFAULT_OPEN):

            _, int_type = imgui.combo("Integrator", int_type, [
                                    "Path", "Neural", "Depth", "Albedo", "Normal", "AO"])
            _, slider_spp = imgui.slider_int("SPP", slider_spp, 1, 32)

            if int_type == 0:
                integrator = path_integrator
                spp = slider_spp
            elif int_type == 1:
                integrator = nr_integrator
                spp = 1
            elif int_type == 2:
                integrator = depth_integrator
                if use_antialiasing:
                    depth_integrator.ray_type = "secondary"
                else:
                    depth_integrator.ray_type = "primary"
                spp = slider_spp
            elif int_type == 3:
                integrator = albedo_integrator
                integrator.specular = use_antialiasing
                spp = slider_spp
            elif int_type == 4:
                integrator = normal_integrator
                spp = slider_spp
            elif int_type == 5:
                integrator = ao_integrator
                spp = slider_spp

            _, use_antialiasing = imgui.checkbox(
                "Anti-aliasing", use_antialiasing)
            _, exposure = imgui.slider_float("Exposure", exposure, 0.1, 5)
            _, save_img = imgui.checkbox("Save image", save_img)

            imgui.tree_pop()
        
        if imgui.tree_node("Camera", imgui.TREE_NODE_DEFAULT_OPEN):
            if light_key:
                _, follow_light = imgui.checkbox("Light follows camera", follow_light)
            if dataset_poses:
                changed, pose_index = imgui.input_int(
                    "Dataset pose", pose_index, step=1, step_fast=10
                )
                pose_index = int(np.clip(pose_index, 0, len(dataset_poses) - 1))
                imgui.text(f"0 .. {len(dataset_poses) - 1}")
                if changed:
                    camera.set_transform(dataset_poses[pose_index])
                    # print(f"Loaded dataset pose {pose_index}")
            imgui.tree_pop()

        model = nr_integrator.model
        controls = model.gui_controls() if hasattr(model, "gui_controls") else []
        if controls and imgui.tree_node("Model", imgui.TREE_NODE_DEFAULT_OPEN):
            for control in controls:
                changed, value = imgui.slider_float(
                    control.label, control.value, control.minimum, control.maximum
                )
                if changed:
                    model.set_gui_control(control.key, value)
            imgui.tree_pop()

        # Camera, light, and tissue roughness, after the sliders have run.
        update_scene(params, camera, model, follow_light)

        seed = int(ui.duration * 1000)
        img = mi.render(scene, integrator=integrator, seed=seed, spp=spp)

        img = img.torch()

        if save_img:
            dr.sync_device()
            torch.cuda.synchronize()
            out_dir = os.path.join("out", args.scene, args.config + ".exr")
            mi.util.write_bitmap(out_dir, img)
            print("Image saved to", out_dir)
            save_img = False

        ui.end_frame()
        exposure = 1
        img = torch.log1p(torch.abs(exposure * img))  # tone mapping
        img = img ** (1 / 2.2)  # gamma correction
        # dr.sync_device()
        # torch.cuda.synchronize()
        # dr.flush_malloc_cache()
        # torch.cuda.empty_cache()
        # ui.write_texture_cpu(img.cpu().numpy())
        ui.write_texture_gpu(img)
    ui.close()


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="null")
    parser.add_argument("-s", "--scene", type=str, default="cornell-box")
    parser.add_argument("-m", "--model_ckpt", type=str, default=None)
    parser.add_argument("-o", "--output", type=str, default="./out/test.exr")
    parser.add_argument("-H", "--half_precision", type=bool, default=False)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Load config file
    with open(os.path.join("configs", args.config + ".json"), "r") as f:
        config = json.load(f)
    
    render(config, args)