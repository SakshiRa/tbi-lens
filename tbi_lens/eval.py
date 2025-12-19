"""Evaluation utilities for TBI-LENS inference outputs."""

from __future__ import annotations

from typing import Optional, Dict

import numpy as np
import nibabel as nib


def _compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute the Dice score for binary masks."""
    pred_bin = pred > 0
    gt_bin = gt > 0
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    if denom == 0:
        return 1.0
    return 2.0 * intersection / denom


def _volume_ml(mask: np.ndarray, affine: np.ndarray) -> float:
    """Compute volume in milliliters from a binary mask and affine."""
    voxel_volume_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
    volume_mm3 = (mask > 0).sum() * voxel_volume_mm3
    return volume_mm3 / 1000.0


def evaluate_prediction(
    pred_path: str, gt_path: Optional[str] = None
) -> Dict[str, Optional[float]]:
    """Evaluate a prediction mask against an optional ground truth mask."""
    pred_img = nib.load(pred_path)
    pred_data = pred_img.get_fdata()
    pred_volume_ml = _volume_ml(pred_data, pred_img.affine)

    dice: Optional[float] = None
    gt_volume_ml: Optional[float] = None
    if gt_path:
        gt_img = nib.load(gt_path)
        gt_data = gt_img.get_fdata()
        dice = _compute_dice(pred_data, gt_data)
        gt_volume_ml = _volume_ml(gt_data, gt_img.affine)

    return {
        "dice": dice,
        "gt_volume_ml": gt_volume_ml,
        "pred_volume_ml": pred_volume_ml,
    }
