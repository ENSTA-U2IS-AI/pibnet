import numpy as np
import torch
from typing import Union


def get_reduced_metric_dict(pred: torch.tensor, gt: torch.tensor) -> torch.tensor:
    """
    Returns a dictionary of metrics

    Parameters
    ----------
    pred: torch.tensor
        Network predictions
    gt: torch.tensor
        Ground truth
    """

    errors = {}
    errors["error"] = torch.abs(pred - gt)

    if pred.shape[-1] == 2:
        pred_real = pred[:, 0]
        pred_imag = pred[:, 1]
        pred_ampl = torch.sqrt(pred_real**2 + pred_imag**2)
        pred_angle = torch.arctan2(pred_imag, pred_real)

        gt_real = gt[:, 0]
        gt_imag = gt[:, 1]
        gt_ampl = torch.sqrt(gt_real**2 + gt_imag**2)
        gt_angle = torch.arctan2(gt_imag, gt_real)

        errors["error_ampl"] = torch.abs(pred_ampl - gt_ampl)
        errors["error_rel_ampl"] = torch.abs(pred_ampl - gt_ampl) / gt_ampl
        errors["error_angle"] = torch.abs(torch.atan2(torch.sin(pred_angle - gt_angle), torch.cos(pred_angle - gt_angle)))

    elif pred.shape[-1] == 1:
        errors["error_rel"] = torch.abs(pred - gt) / torch.abs(gt).mean(0, keepdims=True)
    
    else:
        raise ValueError(f"pred.shape[-1] must be one of 1 or 2, not f{pred.shape[-1]}")

    return {key: torch.mean(value) for key, value in errors.items()}
