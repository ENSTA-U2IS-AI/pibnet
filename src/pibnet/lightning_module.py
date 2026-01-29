from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
from multiprocessing import cpu_count
import json
import bempp_cl.api
from copy import deepcopy

from torch.optim.lr_scheduler import CosineAnnealingLR

from gnn import *
from utils.metrics import get_reduced_metric_dict

from data.data_utils import *


class RescalingInverse:
    """
    Inverse and rescale

    Parameters
    ----------
    min_x: float
        Minimum
    max_x: float
        Maximum
    """
    def __init__(self, min_x, max_x):
        self.max_inv = 1 / min_x
        self.min_inv = 1 / max_x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (1 / x - self.min_inv) / (self.max_inv - self.min_inv) * 2 -1
    

class PositionalEncoding:
    """
    Unidimensional sinusoidale position encoding

    Parameters
    ----------
    num_channels: int
        Number of channels
    min_dist: float
        Minimum distance
    max_dist: float
        Maximum distance
    device: str
        Device, by default 'cuda'
    """
    def __init__(self, num_channels, min_dist, max_dist, device='cuda'):
        omega_min = torch.pi / (2 * max_dist)
        omega_max = torch.pi / (2 * min_dist)
        self.div_term = (torch.exp(torch.linspace(0, 1, num_channels // 2) * np.log(omega_min / omega_max)) * omega_max).unsqueeze(0).to(device)

    def __call__(self, dist):
        return torch.cat([torch.sin(dist * self.div_term), -torch.cos(dist * self.div_term)], dim=-1)


def normalize_with_inv_dist(feature, dist=None):
    if dist is not None:
        return feature * dist
    return feature


def denormalize_with_inv_dist(feature, dist=None):
    if dist is not None:
        return feature / dist
    return feature


class PIBNetLightning(pl.LightningModule):
    """
    PIBNet LightningModule

    Parameters
    ----------
    cfg: config
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        print(OmegaConf.to_yaml(self.cfg))

        dim_obs = sum(cfg.data.node_features.values())
        dim_edges = sum(cfg.data.edge_features.values())

        if cfg.data.equation == "helmholtz":
            num_out = 2
        elif cfg.data.equation == "laplace":
            num_out = 1
        else:
            raise ValueError(f"Wrong equation name '{cfg.data.equation}'")
            
        # Initialize model and Lightning module        
        self.model = PIBNet(
            num_levels=cfg.data.num_levels,
            input_dim_nodes=dim_obs,
            input_dim_edges={etype: dim_edges for etype in cfg.data.etypes},
            output_dim=num_out,
            first_top_processor_size=cfg.model.first_top_processor_size,
            last_top_processor_size=cfg.model.last_top_processor_size,
            bottom_processor_size=cfg.model.bottom_processor_size,
            distant_edge_sample_period=2, #cfg.model.distant_edge_sample_period,
            hidden_dim_processor=cfg.model.hidden_dim,
            hidden_dim_node_encoder=cfg.model.hidden_dim,
            hidden_dim_edge_encoder=cfg.model.hidden_dim,
            hidden_dim_node_decoder=cfg.model.hidden_dim,
            hidden_dim_scaling=cfg.model.hidden_dim_scaling,
            mlp_activation_fn='silu'
        )
        # print(self.model)

        max_dist = cfg.data.environment.size * np.sqrt(2) # np.sqrt(2) because of square env
        self.pe_dist = PositionalEncoding(cfg.pe.dist.dim, cfg.data.h, max_dist)
        
        self.wl_pe = RescalingInverse(cfg.data.source.wavelength.min, cfg.data.source.wavelength.max)
        
        self.loss_fn = hydra.utils.instantiate(cfg.loss)
        
        self.save_hyperparameters()


    def configure_optimizers(self):
        if self.cfg.lr_scheduler:
            # optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr_max)
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr_max, weight_decay=1e-5)
            scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.max_epochs, eta_min=self.cfg.lr_min)

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch', # Call scheduler.step() every epoch
                    'frequency': 1,
                },
            }
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr_max)


    def forward_step(self, batched_graph):

        if 'source_dist' in batched_graph.ndata.keys():
            batched_graph.ndata['source_dist_pe'] = self.pe_dist(batched_graph.ndata['source_dist'])
        if 'center_dist' in batched_graph.ndata.keys():
            batched_graph.ndata['center_dist_pe'] = self.pe_dist(batched_graph.ndata['center_dist'])
        if 'wavelength' in batched_graph.ndata.keys():
            batched_graph.ndata['wavelength_pe'] = self.wl_pe(batched_graph.ndata['wavelength'])
        
        node_features = torch.cat([
                batched_graph.ndata[key] for key in self.cfg.data.node_features
            ], dim=-1)

        edge_features = {}
        for etype in self.cfg.data.etypes:
            dist = batched_graph.edges[etype].data['dist']
            batched_graph.edges[etype].data['dist_pe'] = self.pe_dist(dist)
            if 'wavelength' in batched_graph.edges[etype].data.keys():
                wavelength = batched_graph.edges[etype].data['wavelength']
                batched_graph.edges[etype].data['wavelength_pe'] = self.wl_pe(wavelength)
            
            edge_features[etype] = torch.cat([
                batched_graph.edges[etype].data[key] for key in self.cfg.data.edge_features
            ], dim=-1)

        return self.model(node_features, edge_features, batched_graph)


    def training_step(self, batched_graph, batch_idx):

        preds = self.forward_step(batched_graph)
        labels = batched_graph.ndata['labels']

        if 'source_dist' in batched_graph.ndata.keys():
            source_dist = batched_graph.ndata['source_dist']
        else:
            source_dist = None
        
        # Get normalized and denormalized preds
        if self.cfg.predict_normalized:
            preds_normalized = preds
            preds = denormalize_with_inv_dist(preds, source_dist)
        else:
            preds_normalized = normalize_with_inv_dist(preds, source_dist)

        loss = self.loss_fn(preds, labels)
        self.log("train/loss", loss, on_epoch=True, batch_size=self.cfg.training_batch_size)
        
        if 'labels_normalized' in batched_graph.ndata.keys():
            labels_normalized = batched_graph.ndata['labels_normalized']
            loss_normalized = self.loss_fn(preds_normalized, labels_normalized)
            self.log("train/loss_norm", loss_normalized, on_epoch=True, batch_size=self.cfg.training_batch_size)

        metrics = get_reduced_metric_dict(preds, labels)
        for metric, value in metrics.items():
            self.log(f"train/{metric}", value, on_epoch=True, batch_size=self.cfg.training_batch_size)

        if self.cfg.compare_normalized:
            return loss_normalized
        return loss


    def validation_step(self, batched_graph, batch_idx):

        preds = self.forward_step(batched_graph)
        labels = batched_graph.ndata['labels']
        
        if 'source_dist' in batched_graph.ndata.keys():
            source_dist = batched_graph.ndata['source_dist']
        else:
            source_dist = None
        
        # Get normalized and denormalized preds
        if self.cfg.predict_normalized:
            preds_normalized = preds
            preds = denormalize_with_inv_dist(preds, source_dist)
        else:
            preds_normalized = normalize_with_inv_dist(preds, source_dist)

        loss = self.loss_fn(preds, labels)
        self.log("val/loss", loss, on_epoch=True, batch_size=self.cfg.val_batch_size)

        if 'labels_normalized' in batched_graph.ndata.keys():
            labels_normalized = batched_graph.ndata['labels_normalized']
            loss_normalized = self.loss_fn(preds_normalized, labels_normalized)
            self.log("val/loss_norm", loss_normalized, on_epoch=True, batch_size=self.cfg.val_batch_size)

        metrics = get_reduced_metric_dict(preds, labels)
        for metric, value in metrics.items():
            self.log(f"val/{metric}", value, on_epoch=True, batch_size=self.cfg.val_batch_size)


    def test_step(self, batched_data, batch_idx):

        batched_graph = batched_data

        preds = self.forward_step(batched_graph)

        labels = batched_graph.ndata['labels']
        points = batched_graph.ndata['position']

        if 'source_dist' in batched_graph.ndata.keys():
            source_dist = batched_graph.ndata['source_dist']
        else:
            source_dist = None
        
        # Get normalized and denormalized preds
        if self.cfg.predict_normalized:
            preds = denormalize_with_inv_dist(preds, source_dist)

        metrics = get_reduced_metric_dict(preds, labels)
        for metric, value in metrics.items():
            self.log(f"test/{metric}", value, on_epoch=True, batch_size=self.cfg.val_batch_size)
