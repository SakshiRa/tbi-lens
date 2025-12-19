"""Command-line interface for TBI-LENS."""

from __future__ import annotations

import argparse
import warnings
from typing import Sequence

from tbi_lens.infer import run_inference_segresnet


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tbi-lens", description="TBI-LENS CLI")
    subparsers = parser.add_subparsers(dest="command")

    infer_parser = subparsers.add_parser("infer", help="Run inference on a single image")
    infer_parser.add_argument("--bundle", required=True, help="Path to Auto3DSeg bundle directory")
    infer_parser.add_argument("--image", required=True, help="Path to input NIfTI image")
    infer_parser.add_argument("--outdir", required=True, help="Output directory")
    infer_parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"], help="Device for inference"
    )
    infer_parser.add_argument(
        "--ckpt",
        required=False,
        default=None,
        help="Deprecated and ignored (weights are pulled from Hugging Face).",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "infer":
        if args.ckpt:
            warnings.warn("--ckpt is ignored; using Auto3DSeg bundle weights instead.", stacklevel=2)
        run_inference_segresnet(args.bundle, args.image, args.outdir, device=args.device)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
