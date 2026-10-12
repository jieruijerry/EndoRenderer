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
from src.model.nerf import EndoPBR, NeRF
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

from src.sensor.c3vd import register_c3vd_sensor, sensor_prefix

register_c3vd_sensor()


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
    save_camera = False
    load_camera = False
    has_spot = "SpotLight.to_world" in params
    follow_light = True

    camera_id = 0
    scene_idx = 0

    while not ui.should_close():
        ui.begin_frame()
        
        cam_to_world = mi.Matrix4f(camera.get_transform()[None, ...])
        prefix = sensor_prefix(params)
        params[f"{prefix}.to_world"] = cam_to_world
        fov_key = f"{prefix}.x_fov"
        if fov_key in params:
            params[fov_key] = mi.Float32(camera.get_x_fov()[None, ...])
        if has_spot and follow_light:
            params['SpotLight.to_world'] = cam_to_world
        params.update()

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
            if has_spot:
                _, follow_light = imgui.checkbox("Light follows camera", follow_light)
            _, save_camera = imgui.checkbox("Save camera config", save_camera)
            if save_camera:
                x_fov = camera.get_x_fov()
                extrinsics = camera.get_transform()
                intrinsics = {
                    "width": width,
                    "height": height,
                    "x_fov": x_fov,
                }
                print(extrinsics)
                np.savez(
                    "./out/poses/{:04d}.npz".format(camera_id),
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                )
                print("Camera config saved to ./out/poses/{:04d}.npz".format(camera_id))
                camera_id += 1
                save_camera = False

            _, load_camera = imgui.checkbox("Load camera config", load_camera)
            if load_camera:
                with np.load(os.path.join("out", args.scene, "camera.npz"), allow_pickle=True) as data:
                    extrinsics = data["extrinsics"]
                    intrinsics = data["intrinsics"].item()
                x_fov = intrinsics["x_fov"]
                camera.set_transform(extrinsics)
                camera.set_x_fov(x_fov)
                print("Camera config loaded from", os.path.join("out", args.scene, "camera.npz"))
                load_camera = False
                
            imgui.tree_pop()

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