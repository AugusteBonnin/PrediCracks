#!/usr/bin/env python3
"""Convert the 256x256 float32 images stored in .data files to grayscale PNG."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SHAPE = (256, 256)
EXPECTED_FLOATS = 1 + IMAGE_SHAPE[0] * IMAGE_SHAPE[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data_png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.data_dir.glob("*.data"))
    if not files:
        raise ValueError("No .data files found in {}".format(args.data_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in files:
        data = np.fromfile(str(path), dtype=np.float32)
        if data.size != EXPECTED_FLOATS:
            raise ValueError(
                "{}: expected {} float32 values, got {}".format(
                    path.name, EXPECTED_FLOATS, data.size
                )
            )
        time = float(data[0])
        image = data[1:].reshape(IMAGE_SHAPE)
        pixels = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        output_path = args.output_dir / (path.stem + ".png")
        Image.fromarray(pixels, mode="L").save(str(output_path))
        rows.append((path.name, output_path.name, time, float(image.min()),
                     float(image.max())))

    csv_path = args.output_dir / "index.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("data_file", "png_file", "normalized_time",
                         "raw_min", "raw_max"))
        writer.writerows(rows)
    print("Converted {} files into {}".format(len(files), args.output_dir))


if __name__ == "__main__":
    main()
