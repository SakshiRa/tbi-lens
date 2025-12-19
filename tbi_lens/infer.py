"""Inference pipeline for TBI-LENS using Auto3DSeg bundles."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from monai.apps.auto3dseg import AlgoEnsembleBestN, AlgoEnsembleBuilder, import_bundle_algo_history
from monai.utils.enums import AlgoKeys

from tbi_lens.utils import DEFAULT_HF_REPO, ensure_weights


def _case_id_from_path(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".nii.gz"):
        return base[:-7]
    return os.path.splitext(base)[0]


def _validate_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu instead.")
    return device


def _load_bundle_history(bundle_dir: Path) -> list[dict]:
    parent_dir = bundle_dir.parent
    bundle_id = bundle_dir.name
    history = import_bundle_algo_history(str(parent_dir))
    if not history:
        raise ValueError(f"No Auto3DSeg bundle history found in: {parent_dir}")
    filtered = [algo for algo in history if algo.get(AlgoKeys.ID) == bundle_id]
    if not filtered:
        available = ", ".join(sorted({algo.get(AlgoKeys.ID) for algo in history}))
        raise ValueError(
            f"Bundle identifier '{bundle_id}' not found in {parent_dir}. "
            f"Available bundles: {available}"
        )
    return filtered


def run_inference_segresnet(
    bundle_dir: str,
    image_path: str,
    output_dir: str,
    device: str = "cuda",
) -> str:
    """Run Auto3DSeg SegResNet bundle inference for a single input image."""
    bundle_path = Path(bundle_dir)
    if not bundle_path.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

    os.makedirs(output_dir, exist_ok=True)
    device = _validate_device(device)

    repo_id = os.environ.get("TBI_LENS_HF_REPO", DEFAULT_HF_REPO)
    filename = os.environ.get("TBI_LENS_HF_FILENAME", "model.pt")
    ensure_weights(repo_id, filename, bundle_path / "model")

    history = _load_bundle_history(bundle_path)
    builder = AlgoEnsembleBuilder(history, data_src_cfg_name=None)
    builder.set_ensemble_method(AlgoEnsembleBestN(n_best=1))
    ensemble = builder.get_ensemble()

    ensemble.set_infer_files(
        dataroot="/",
        data_list_or_path=[{"image": image_path}],
    )

    pred_params = {
        "mode": "mean",
        "image_save_func": {
            "_target_": "SaveImage",
            "output_dir": output_dir,
            "output_postfix": "pred",
            "output_dtype": "$np.uint8",
            "resample": False,
            "separate_folder": False,
            "print_log": False,
        },
        "device": device,
    }
    _ = ensemble(pred_params)

    case_id = _case_id_from_path(image_path)
    pred_path = os.path.join(output_dir, f"{case_id}_pred.nii.gz")
    if not os.path.exists(pred_path):
        raise RuntimeError(f"Expected prediction not found at: {pred_path}")
    return pred_path
