# Basics
import os
import json
import argparse
import types

# Computational
import numpy as np
import torch
from torch.utils.data import DataLoader

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import BaseFinetuning, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# Mitsuba
import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_rgb")

# Custom
from src.model.radiosity import *
from src.model.nerf import EndoPBR, EndoRen, NeRF
from src.model.screen import ScreenSpace
from src.dataset.camera import *
from src.dataset.c3vd_color import C3VDColorDataset, build_screen_dataset
from src.sensor.c3vd import register_c3vd_sensor

register_c3vd_sensor()
from src.util.progress_bar import StepRichProgressBar, find_best_ckpt


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", 1))


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))


def _trainer(config: dict, output_dir: str, strategy: str = "ddp") -> tuple[Trainer, str | None]:
    """Lightning trainer and the best checkpoint under ``output_dir``, if any."""
    logger = TensorBoardLogger(
        os.path.join(output_dir, "tb_logs"),
        name=config["model"]["name"],
    )
    ckpt_dir = os.path.join(output_dir, "checkpoints", config["model"]["name"])
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path, _ = find_best_ckpt(ckpt_dir, metric="loss")
    checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        mode="min",
        save_top_k=3,
        save_last=False,
        every_n_train_steps=config["train"]["save_every"],
        dirpath=ckpt_dir,
        filename="{step}_{loss:.4f}",
    )
    max_steps = int(np.ceil(config["train"]["epochs"] / get_world_size()))
    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=-1,
        max_steps=max_steps,
        logger=logger,
        callbacks=[checkpoint_callback, StepRichProgressBar(total_steps=max_steps)],
        log_every_n_steps=1,
    )
    return trainer, ckpt_path


def train_screen(config: dict) -> None:
    """
    Jointly train one UNet and one K-Planes grid per C3VD sequence.

    Each step draws one full frame from a random training sequence. Frames
    divisible by ``holdout_modulo`` and the held-out sequence are not in
    this dataset.
    """
    dataset, train_names, holdout = build_screen_dataset(config, split="train")
    print(f"Screen grids: {train_names}. Held-out sequence: {holdout}")
    scenes = {
        name: mi.load_file(os.path.join("scenes", "c3vd", name, "scene.xml"))
        for name in train_names
    }
    # One step touches only the sampled sequence's grid. DDP treats the
    # other grids as unused parameters and makes the backward path stall.
    trainer, ckpt_path = _trainer(
        config,
        os.path.join("out", "c3vd_screen"),
        strategy="auto",
    )
    data_loader = DataLoader(
        dataset,
        collate_fn=lambda x: x[0],
        batch_size=1,
        shuffle=True,
    )
    if ckpt_path is None:
        model = ScreenSpace(config, scenes)
    else:
        model = ScreenSpace.load_from_checkpoint(
            ckpt_path,
            pipeline_config=config,
            scenes=scenes,
        )
    model.train()
    trainer.fit(model, train_dataloaders=data_loader, ckpt_path=ckpt_path)


def train(config: dict, args: argparse.Namespace):
    """
    Train model
    """
    config = json.loads(json.dumps(config))
    if config["model"]["name"] == "ScreenSpace":
        train_screen(config)
        return

    # Load scene
    manager = None
    scene = mi.load_file(os.path.join("scenes", args.scene, "scene.xml"))

    trainer, ckpt_path = _trainer(config, os.path.join("out", args.scene))
    
    # Train
    model_name = config["model"]["name"]
    if model_name.startswith(("NeRF", "EndoPBR", "EndoRen")):
        seq_dir = os.path.join("data", "c3vd", "sequences", os.path.basename(args.scene))
        dataset = C3VDColorDataset(seq_dir, n_rays=config["sample"]["n_rays"])
    elif os.path.exists(os.path.join("scenes", args.scene, "camera_poses")):
        dataset = CameraDataset(os.path.join("scenes", args.scene, "camera_poses"))
    elif manager is not None:
        dataset = manager.get_dataset()
    else:
        if config["sample"].get("from_poses", False):
            print("No camera poses found, falling back to DummyDataset and disabling pose-based sampling.")
            config["sample"]["from_poses"] = False
        dataset = DummyDataset(1 if manager is None else len(manager))
    
    data_loader = DataLoader(
        dataset,
        collate_fn=lambda x: x[0],
        batch_size=1,
    )

    if ckpt_path is None:
        # Load model
        if model_name.startswith("EndoPBR"):
            model = EndoPBR(config, scene)
        elif model_name.startswith("EndoRen"):
            model = EndoRen(config, scene)
        elif model_name.startswith("NeRF"):
            model = NeRF(config, scene)
        elif model_name[:2] == "NR":
            model = NeuralRadiosity(config, scene)
        model.train()

        trainer.fit(model, train_dataloaders=data_loader)
    else:
        # Train from the chosen checkpoint
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
        model.train()
        
        trainer.fit(model, ckpt_path=ckpt_path, train_dataloaders=data_loader)
    

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="nr")
    parser.add_argument("-s", "--scene", type=str, default="cornell-box")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Load config file
    with open(os.path.join("configs", args.config + ".json"), "r") as f:
        config = json.load(f)
    
    # Train model
    train(config, args)
