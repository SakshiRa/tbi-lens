"""Command-line interface for TBI-LENS."""

from __future__ import annotations

import argparse
from typing import Sequence

from tbi_lens.infer import run_inference


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tbi-lens", description="TBI-LENS CLI")
    subparsers = parser.add_subparsers(dest="command")

    infer_parser = subparsers.add_parser("infer", help="Run inference on a single image")
    infer_parser.add_argument("--config", required=True, help="Path to inference YAML config")
    infer_parser.add_argument("--image", required=True, help="Path to input NIfTI image")
    infer_parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    infer_parser.add_argument("--outdir", required=True, help="Output directory")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "infer":
        run_inference(args.config, args.image, args.ckpt, args.outdir)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
