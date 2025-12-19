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
  --image /path/to/image.nii.gz \
  --bundle /path/to/bundle \
  --outdir /path/to/output
```

Outputs:
- `*_pred.nii.gz` prediction mask

Pretrained weights are auto-downloaded from Hugging Face into the bundle
`model/` directory when missing. Set `TBI_LENS_HF_REPO` (required) and
`TBI_LENS_HF_FILENAME` (optional, default `model.pt`) before running.
