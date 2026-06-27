#!/usr/bin/env python3
"""Validate the packaged English GT-construction figure assets."""

from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "figures"
REQUIRED_ASSETS = (
    OUTPUT_DIR / "gt_data_construction_pipeline_en.png",
    OUTPUT_DIR / "gt_data_construction_pipeline_en.svg",
)


def main() -> None:
    missing = [path for path in REQUIRED_ASSETS if not path.exists()]
    if missing:
        missing_text = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            "Missing packaged GT-construction assets: "
            f"{missing_text}. Keep the English figures under analysis/figures/."
        )

    for path in REQUIRED_ASSETS:
        print(path)


if __name__ == "__main__":
    main()
