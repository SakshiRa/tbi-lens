# TBI-LENS

TBI-LENS (Traumatic Brain Injury Lesion Segmentation) provides a clean,
reproducible inference and evaluation pipeline for 3D NIfTI scans using MONAI.

## Quick start

Install dependencies (example with pip):

```bash
pip install -e .
```

Run inference on a single image:

```bash
tbi-lens infer \
  --config configs/infer.yaml \
  --image /path/to/image.nii.gz \
  --ckpt /path/to/model.pt \
  --outdir /path/to/output
```

Outputs:
- `*_pred.nii.gz` prediction mask
- `*_metrics.csv` with `case_id,dice,gt_volume_ml,pred_volume_ml`

Optional: set `ground_truth` in `configs/infer.yaml` to compute Dice and GT volume.
