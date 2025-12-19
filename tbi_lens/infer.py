"""Inference pipeline for TBI-LENS."""

from __future__ import annotations

import csv
import os
from typing import Any, Dict

import numpy as np
import torch
import yaml
import nibabel as nib
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    Orientation,
    ScaleIntensityRange,
    Spacing,
)
from monai.networks.nets import UNet, SegResNet

from tbi_lens.eval import evaluate_prediction


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _select_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key[len(prefix):]: value for key, value in state_dict.items()}


def build_model(model_cfg: Dict[str, Any]) -> torch.nn.Module:
    """Build a MONAI-compatible model from config."""
    name = model_cfg.get("name")
    params = model_cfg.get("params", {})
    if name == "UNet":
        return UNet(**params)
    if name == "SegResNet":
        return SegResNet(**params)
    raise ValueError(f"Unsupported model name: {name}")


def build_pre_transforms(cfg: Dict[str, Any]) -> Compose:
    """Build preprocessing transforms from config."""
    tcfg = cfg["transforms"]
    intensity = tcfg["intensity"]
    return Compose(
        [
            EnsureChannelFirst(),
            Orientation(axcodes=tcfg["orientation"]),
            Spacing(pixdim=tcfg["spacing"], mode=tcfg.get("spacing_mode", "bilinear")),
            ScaleIntensityRange(
                a_min=intensity["a_min"],
                a_max=intensity["a_max"],
                b_min=intensity["b_min"],
                b_max=intensity["b_max"],
                clip=bool(intensity.get("clip", True)),
            ),
        ]
    )


def _post_process(logits: torch.Tensor, cfg: Dict[str, Any]) -> np.ndarray:
    activation = cfg.get("activation", "softmax")
    if activation == "sigmoid":
        probs = torch.sigmoid(logits)
        threshold = cfg.get("threshold", 0.5)
        mask = (probs > threshold).squeeze(0).squeeze(0)
    elif activation == "softmax":
        probs = torch.softmax(logits, dim=1)
        if cfg.get("argmax", True):
            mask = torch.argmax(probs, dim=1).squeeze(0)
        else:
            threshold = cfg.get("threshold", 0.5)
            mask = (probs[:, 1, ...] > threshold).squeeze(0)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    return mask.detach().cpu().numpy().astype(np.uint8)


def _case_id_from_path(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".nii.gz"):
        return base[:-7]
    return os.path.splitext(base)[0]


def _write_metrics_csv(csv_path: str, case_id: str, metrics: Dict[str, Any]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["case_id", "dice", "gt_volume_ml", "pred_volume_ml"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": case_id,
                "dice": "" if metrics["dice"] is None else metrics["dice"],
                "gt_volume_ml": ""
                if metrics["gt_volume_ml"] is None
                else metrics["gt_volume_ml"],
                "pred_volume_ml": metrics["pred_volume_ml"],
            }
        )


def run_inference(config_path: str, image_path: str, ckpt_path: str, outdir: str) -> Dict[str, Any]:
    """Run inference and evaluation for a single input image."""
    cfg = load_config(config_path)
    os.makedirs(outdir, exist_ok=True)

    device = _select_device(cfg["inference"].get("device", "cpu"))

    loader = LoadImage(image_only=True)
    image = loader(image_path)
    pre = build_pre_transforms(cfg)
    image = pre(image)
    image = image.unsqueeze(0).to(device)

    model = build_model(cfg["model"])
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = _strip_prefix(state_dict, "module.")
    model.load_state_dict(state_dict, strict=bool(cfg["model"].get("strict", True)))
    model.to(device)
    model.eval()

    infer_cfg = cfg["inference"]
    with torch.no_grad():
        logits = sliding_window_inference(
            image,
            roi_size=infer_cfg["roi_size"],
            sw_batch_size=infer_cfg["sw_batch_size"],
            predictor=model,
            overlap=infer_cfg["overlap"],
        )
    pred = _post_process(logits, cfg["post"])

    case_id = _case_id_from_path(image_path)
    pred_path = os.path.join(outdir, f"{case_id}_pred.nii.gz")

    affine = image.meta.get("affine")
    if affine is None:
        affine = np.eye(4, dtype=float)
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), affine), pred_path)

    gt_path = cfg.get("ground_truth")
    metrics = evaluate_prediction(pred_path, gt_path)
    csv_path = os.path.join(outdir, f"{case_id}_metrics.csv")
    _write_metrics_csv(csv_path, case_id, metrics)
    return metrics
