from omegaconf import DictConfig, OmegaConf, open_dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
from dgl.dataloading import GraphDataLoader
import hydra
from omegaconf import DictConfig
from lightning_module import PIBNetLightning

from data import PIBNetDataset, PIBNetDatasetTest
from utils.config_utils import set_run_name, etypes_levels

# Main training script with Hydra
@hydra.main(version_base=None, config_path="../../configs/pibnet", config_name="training")
def main(cfg: DictConfig):
    set_run_name(cfg)
    etypes_levels(cfg)
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)
    
    # Initialize dataset and dataloader
    train_dataset = PIBNetDataset(os.path.join(cfg.dataset_root, cfg.dataset_train), cfg)
    train_sampler = SubsetRandomSampler(torch.arange(len(train_dataset)))
    train_dataloader = GraphDataLoader(train_dataset, sampler=train_sampler, batch_size=cfg.training_batch_size, drop_last=True, num_workers=16, timeout=10)
    val_dataset = PIBNetDataset(os.path.join(cfg.dataset_root, cfg.dataset_val), cfg)
    val_dataloader = GraphDataLoader(val_dataset, batch_size=cfg.val_batch_size, drop_last=False, num_workers=16, timeout=100)
    
    lightning_module = PIBNetLightning(cfg)

    # Initialize the Wandb logger
    wandb_logger = WandbLogger(
        log_model=True,
        project=cfg.project,
        name=cfg.run_name,
        save_dir=os.path.join("checkpoints")
    )

    # Initialize LR monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=cfg.max_epochs,
        gradient_clip_val=cfg.max_grad_norm,
        callbacks=[lr_monitor],
        accumulate_grad_batches=cfg.accumulate_grad_batches,
    )

    # Train the model
    trainer.fit(lightning_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    os.makedirs(os.path.join("evaluations", cfg.run_name), exist_ok=True)
    os.makedirs(os.path.join("evaluations", cfg.run_name, "boundary_sol"), exist_ok=True)
    os.makedirs(os.path.join("evaluations", cfg.run_name, "full_sol"), exist_ok=True)
    
    # Initialize test dataset and dataloader
    test_dataset = PIBNetDatasetTest(os.path.join(cfg.dataset_root, cfg.dataset_test), cfg)
    test_dataloader = GraphDataLoader(test_dataset, batch_size=1, drop_last=False, num_workers=8)

    # Test the model
    trainer.test(lightning_module, test_dataloader)

if __name__ == "__main__":
    main()