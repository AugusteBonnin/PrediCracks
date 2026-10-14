#!/usr/bin/env python3
"""Generate future crack images autoregressively with a trained ConvLSTM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import plaidml.keras

    plaidml.keras.install_backend()
    import keras
except ImportError as exc:
    raise SystemExit(
        "PlaidML/Keras is required. Activate the environment containing "
        "plaidml-keras before launching this script."
    ) from exc

from train_convlstm import load_dataset, make_crack_strength


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("best_convlstm.h5"))
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("convlstm_predictions"))
    parser.add_argument("--start-index", type=int, default=0,
                        help="First real frame used to seed the temporal window")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--delta-t", type=float, default=None,
                        help="Future time step (default: median dataset time step)")
    parser.add_argument(
        "--noise-black-fraction", type=float, default=None,
        help="Seed with binary noise containing this fraction of black pixels "
             "instead of real dataset frames (example: 0.125)",
    )
    parser.add_argument(
        "--seed-image", type=Path, default=None,
        help="Seed from a grayscale image instead of real dataset frames. The "
             "image must be 256x256 and is repeated across the initial window.",
    )
    parser.add_argument("--noise-seed", type=int, default=279)
    parser.add_argument(
        "--remove-global-drift", action="store_true",
        help="Remove the spatial median from every predicted delta. This prevents "
             "a small brightness bias from accumulating autoregressively.",
    )
    parser.add_argument(
        "--monotonic-darkening", action="store_true",
        help="Do not allow an inferred crack pixel to become brighter than the "
             "previous frame.",
    )
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=1.0)
    return parser.parse_args()


def save_png(image: np.ndarray, path: Path) -> None:
    pixels = (np.clip(image[..., 0], 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path)


def main() -> None:
    args = parse_args()
    if args.clip_min >= args.clip_max:
        raise ValueError("--clip-min must be smaller than --clip-max")
    # Keras 2.2.x does not accept pathlib.Path objects in its HDF5 loader.
    # Do not restore the optimizer/training graph during inference. Besides being
    # unnecessary, Adam's extra tensors can exhaust PlaidML SVM memory.
    model = keras.models.load_model(str(args.model), compile=False)
    times, images = load_dataset(args.data_dir)
    metadata_path = args.model.with_suffix(".json")
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        sequence_length = int(metadata["sequence_length"])
    else:
        shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
        sequence_length = int(shape[1])
    input_channels = int(metadata.get("input_channels", 2))
    output_mode = metadata.get("output_mode", "image")
    delta_scale = float(metadata.get("delta_scale", 1.0))
    if output_mode == "crack_strength":
        images = make_crack_strength(
            images, float(metadata.get("crack_contrast", 0.25))
        )

    delta_t = args.delta_t or float(np.median(np.diff(times)))
    if args.noise_black_fraction is not None and args.seed_image is not None:
        raise ValueError("--noise-black-fraction and --seed-image are mutually exclusive")
    if args.seed_image is not None:
        seed = np.asarray(Image.open(args.seed_image).convert("L"), dtype=np.float32)
        if seed.shape != (256, 256):
            raise ValueError("--seed-image must be exactly 256x256 pixels")
        seed = (seed / 255.0)[..., None]
        if output_mode == "crack_strength":
            seed = 1.0 - seed
        frame_window = [seed.copy() for _ in range(sequence_length)]
        time_window = [index * delta_t for index in range(sequence_length)]
    elif args.noise_black_fraction is not None:
        if not 0.0 <= args.noise_black_fraction <= 1.0:
            raise ValueError("--noise-black-fraction must be between 0 and 1")
        rng = np.random.RandomState(args.noise_seed)
        noise = np.ones((256, 256, 1), dtype=np.float32)
        black_count = int(round(args.noise_black_fraction * 256 * 256))
        black_indices = rng.choice(256 * 256, size=black_count, replace=False)
        noise.reshape(-1)[black_indices] = 0.0
        frame_window = [noise.copy() for _ in range(sequence_length)]
        time_window = [index * delta_t for index in range(sequence_length)]
    else:
        stop = args.start_index + sequence_length
        if args.start_index < 0 or stop > len(images):
            raise ValueError("The requested seed window lies outside the dataset")
        frame_window = [frame.copy() for frame in images[args.start_index:stop]]
        time_window = list(map(float, times[args.start_index:stop]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    display_seed = 1.0 - frame_window[-1] if output_mode == "crack_strength" else frame_window[-1]
    save_png(display_seed, args.output_dir / "seed.png")
    np.save(args.output_dir / "seed.npy", frame_window[-1])
    generated_times = []
    for step in range(args.steps):
        next_time = time_window[-1] + delta_t
        sequence = np.empty(
            (1, sequence_length, 256, 256, input_channels), dtype=np.float32
        )
        frames = np.stack(frame_window)
        time_maps = np.broadcast_to(
            np.asarray(time_window, dtype=np.float32)[:, None, None, None],
            (*frames.shape[:-1], 1),
        )
        channels = [frames, time_maps]
        if input_channels == 3:
            channels.append(np.full_like(time_maps, next_time))
        sequence[0] = np.concatenate(channels, axis=-1)
        if output_mode == "scaled_delta":
            predicted_delta = model.predict({"sequence": sequence}, verbose=0)[0]
            if args.remove_global_drift:
                # A tiny spatially uniform bias is harmless for one-step metrics,
                # but dominates after hundreds of autoregressive iterations.
                predicted_delta = predicted_delta - np.median(predicted_delta)
            prediction = frames[-1] + delta_scale * predicted_delta
        elif output_mode == "crack_strength":
            prediction = model.predict({"sequence": sequence}, verbose=0)[0]
            if args.monotonic_darkening:
                prediction = np.maximum(prediction, frames[-1])
        else:
            prediction = model.predict(
                {"sequence": sequence, "last_image": frames[-1][None]}, verbose=0
            )[0]
        if args.monotonic_darkening and output_mode != "crack_strength":
            prediction = np.minimum(prediction, frames[-1])
        prediction = np.clip(prediction, args.clip_min, args.clip_max)
        display_prediction = 1.0 - prediction if output_mode == "crack_strength" else prediction
        save_png(display_prediction, args.output_dir / f"prediction_{step:04d}.png")
        np.save(args.output_dir / f"prediction_{step:04d}.npy", prediction)
        generated_times.append(next_time)
        frame_window = frame_window[1:] + [prediction]
        time_window = time_window[1:] + [next_time]

    np.savetxt(
        args.output_dir / "times.csv", np.asarray(generated_times),
        delimiter=",", header="normalized_time", comments="",
    )
    print(f"Saved {args.steps} predictions in {args.output_dir}")


if __name__ == "__main__":
    main()
