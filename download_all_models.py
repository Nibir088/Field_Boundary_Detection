#!/usr/bin/env python3
"""Download all current, non-legacy FTW models."""

import argparse
import os
from pathlib import Path
from urllib.parse import urlparse

import torch

from ftw_tools.inference.model_registry import MODEL_REGISTRY


def parse_args() -> argparse.Namespace:
    default_output = Path("/scratch") / os.environ.get("USER", "unknown") / "ftw_models"
    parser = argparse.ArgumentParser(
        description="Download all current, non-legacy FTW models."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Model destination directory (default: {default_output})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        name: spec for name, spec in MODEL_REGISTRY.items() if not spec.legacy
    }

    print(f"Downloading {len(models)} current models into {output_dir}")

    for index, (name, spec) in enumerate(models.items(), start=1):
        extension = Path(urlparse(spec.url).path).suffix
        destination = output_dir / f"{name}{extension}"

        if destination.exists() and destination.stat().st_size > 0:
            print(f"[{index}/{len(models)}] Already exists: {destination.name}")
            continue

        print(f"[{index}/{len(models)}] Downloading: {name}")
        print(f"URL: {spec.url}")
        torch.hub.download_url_to_file(spec.url, str(destination), progress=True)

    print("All current models downloaded.")


if __name__ == "__main__":
    main()
