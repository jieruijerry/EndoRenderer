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
from src.dataset.camera import *
from src.dataset.c3vd_color import C3VDColorDataset
from src.sensor.c3vd import register_c3vd_sensor

register_c3vd_sensor()
from src.util.progress_bar import StepRichProgressBar, find_best_ckpt


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", 1))


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))


def train(config: dict, args: argparse.Namespace):
    """
    Train model
    """
    config = json.loads(json.dumps(config))

    # Load scene
    manager = None
    scene = mi.load_file(os.path.join("scenes", args.scene, "scene.xml"))

    # Tensorboard logger
    logger = TensorBoardLogger(
        os.path.join("out", args.scene, "tb_logs"),
        name=config["model"]["name"]
    )

    # Set up training directory or load from checkpoint
    if not os.path.exists(os.path.join("out", args.scene, "checkpoints", config["model"]["name"])):
        os.makedirs(os.path.join("out", args.scene, "checkpoints", config["model"]["name"]))

    # Load checkpoint files, choose the best one, set corresponding step
    ckpt_dir = os.path.join("out", args.scene, "checkpoints", config["model"]["name"])
    ckpt_path, ckpt_step = find_best_ckpt(ckpt_dir, metric="loss")

    checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        mode="min",
        save_top_k=3,
        save_last=False,
        every_n_train_steps=config["train"]["save_every"],
        dirpath=ckpt_dir,
        filename="{step}_{loss:.4f}"
    )

    # Lightning trainer
    max_steps = int(np.ceil(config["train"]["epochs"] / get_world_size()))
    callbacks = [checkpoint_callback, StepRichProgressBar(total_steps=max_steps)]

    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        strategy="ddp",
        precision="bf16-mixed",  # mixed precision
        max_epochs=-1,
        max_steps=max_steps,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
    )
    
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
