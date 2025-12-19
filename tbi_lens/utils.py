"""Utility helpers for TBI-LENS."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_HF_REPO = "SakshiRa/tbi-lens-segresnet"


def ensure_weights(repo_id: str, filename: str, target_dir: Path) -> Path:
    """Ensure model weights are present locally, downloading from Hugging Face if needed."""
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    if target_path.exists():
        return target_path

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to download weights from {repo_id}/{filename}") from exc

    downloaded = Path(downloaded_path)
    if not downloaded.exists():
        raise RuntimeError(f"Downloaded weights missing at: {downloaded_path}")
    return downloaded
