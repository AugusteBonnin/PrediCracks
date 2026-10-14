#!/usr/bin/env python3
"""Train a ConvLSTM to predict the next crack image in a time series."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

try:
    # This must happen before importing standalone Keras.
    import plaidml.keras

    plaidml.keras.install_backend()
    import keras
    from keras import backend as K
    from keras import layers
except ImportError as exc:
    raise SystemExit(
        "PlaidML/Keras is required. Activate the environment containing "
        "plaidml-keras before launching this script."
    ) from exc


IMAGE_SHAPE = (256, 256)


def load_dataset(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(directory.glob("*.data"))
    if not files:
        raise ValueError(f"No .data files found in {directory}")

    times = np.empty(len(files), dtype=np.float32)
    images = np.empty((len(files), *IMAGE_SHAPE, 1), dtype=np.float32)
    expected_size = 1 + np.prod(IMAGE_SHAPE)
    for index, path in enumerate(files):
        data = np.fromfile(path, dtype=np.float32)
        if data.size != expected_size:
            raise ValueError(
                f"{path.name}: expected {expected_size} float32 values, got {data.size}"
            )
        times[index] = data[0]
        images[index, ..., 0] = data[1:].reshape(IMAGE_SHAPE)

    order = np.argsort(times, kind="stable")
    times, images = times[order], images[order]
    if np.any(np.diff(times) <= 0):
        raise ValueError("Times must be strictly increasing")
    return times, images


def transform_spatial(array: np.ndarray, code: int) -> np.ndarray:
    """Apply one of the eight square symmetries to all frames identically."""
    rotation, flip = code % 4, code // 4
    result = np.rot90(array, k=rotation, axes=(-3, -2))
    if flip:
        result = np.flip(result, axis=-2)
    return np.ascontiguousarray(result)


def make_crack_strength(images: np.ndarray, contrast: float = 0.25) -> np.ndarray:
    """Convert grayscale frames to temporally denoised crack-strength maps."""
    backgrounds = np.percentile(images[..., 0], 85, axis=(1, 2))
    strength = np.clip(
        (backgrounds[:, None, None] - images[..., 0]) / contrast, 0.0, 1.0
    ).astype(np.float32)
    denoised = np.empty_like(strength)
    for index in range(len(strength)):
        start, stop = max(0, index - 2), min(len(strength), index + 3)
        denoised[index] = np.median(strength[start:stop], axis=0)
    return denoised[..., None]


class WindowSequence(keras.utils.Sequence):
    def __init__(
        self,
        times: np.ndarray,
        images: np.ndarray,
        samples: np.ndarray,
        sequence_length: int,
        batch_size: int,
        delta_scale: float,
        augment: bool,
        shuffle: bool,
        seed: int,
        target_mode: str = "scaled_delta",
    ):
        self.times = times
        self.images = images
        # Each sample is (last input frame index, target frame index).
        self.samples = np.asarray(samples, dtype=np.int64)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.delta_scale = delta_scale
        self.augment = augment
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.target_mode = target_mode
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, batch_index: int):
        selected = self.samples[
            batch_index * self.batch_size : (batch_index + 1) * self.batch_size
        ]
        batch_size = len(selected)
        sequence = np.empty(
            (batch_size, self.sequence_length, *IMAGE_SHAPE, 3), dtype=np.float32
        )
        target_delta = np.empty((batch_size, *IMAGE_SHAPE, 1), dtype=np.float32)

        for row, (input_end, target_index) in enumerate(selected):
            start = input_end - self.sequence_length + 1
            frames = self.images[start:input_end + 1]
            output = self.images[target_index]
            code = int(self.rng.integers(8)) if self.augment else 0
            frames = transform_spatial(frames, code)
            output = transform_spatial(output, code)

            # A time map lets every ConvLSTM cell see the absolute normalized time.
            time_maps = np.broadcast_to(
                self.times[start:input_end + 1, None, None, None],
                (*frames.shape[:-1], 1),
            )
            target_time_maps = np.full(
                (*frames.shape[:-1], 1), self.times[target_index], dtype=np.float32
            )
            sequence[row] = np.concatenate(
                (frames, time_maps, target_time_maps), axis=-1
            )
            if self.target_mode == "crack_strength":
                target_delta[row] = output
            else:
                target_delta[row] = (output - frames[-1]) / self.delta_scale

        return {"sequence": sequence}, target_delta

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.samples)


def make_samples(first_target: int, last_target: int, sequence_length: int,
                 max_horizon: int) -> np.ndarray:
    samples = []
    for target_index in range(first_target, last_target):
        for horizon in range(1, max_horizon + 1):
            input_end = target_index - horizon
            if input_end - sequence_length + 1 >= 0:
                samples.append((input_end, target_index))
    return np.asarray(samples, dtype=np.int64)


def weighted_delta_loss(y_true, y_pred):
    """Favor localized crack growth instead of the dominant static background."""
    magnitude = K.minimum(K.abs(y_true), 2.0)
    darkening = K.minimum(K.relu(-y_true), 2.0)
    pixel_loss = (1.0 + 25.0 * magnitude + 10.0 * darkening) * K.square(
        y_pred - y_true
    )

    # Matching spatial derivatives makes thin crack fronts matter even though
    # they occupy only a small fraction of the image.
    true_dx = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
    pred_dx = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
    true_dy = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
    pred_dy = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
    gradient_loss = K.mean(K.abs(pred_dx - true_dx)) + K.mean(
        K.abs(pred_dy - true_dy)
    )
    return K.mean(pixel_loss) + 0.5 * gradient_loss


def soft_erode(image):
    """Differentiable approximation of binary erosion for thin structures."""
    vertical = -K.pool2d(
        -image, (3, 1), strides=(1, 1), padding="same", pool_mode="max"
    )
    horizontal = -K.pool2d(
        -image, (1, 3), strides=(1, 1), padding="same", pool_mode="max"
    )
    return K.minimum(vertical, horizontal)


def soft_dilate(image):
    return K.pool2d(
        image, (3, 3), strides=(1, 1), padding="same", pool_mode="max"
    )


def soft_skeleton(image, iterations: int = 3):
    """Extract a differentiable centerline with iterative morphology."""
    opened = soft_dilate(soft_erode(image))
    skeleton = K.relu(image - opened)
    for _ in range(iterations):
        image = soft_erode(image)
        opened = soft_dilate(soft_erode(image))
        delta = K.relu(image - opened)
        skeleton = skeleton + K.relu(delta - skeleton * delta)
    return skeleton


def crack_region_losses(y_true, y_pred):
    """Return weighted pixel BCE and Dice overlap losses."""
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

    pixel_weights = 1.0 + 6.0 * y_true
    binary_crossentropy = -K.mean(
        pixel_weights * (
            y_true * K.log(y_pred)
            + (1.0 - y_true) * K.log(1.0 - y_pred)
        )
    )

    intersection = K.sum(y_true * y_pred)
    dice = (2.0 * intersection + epsilon) / (
        K.sum(y_true) + K.sum(y_pred) + epsilon
    )
    dice_loss = 1.0 - dice
    return binary_crossentropy, dice_loss, y_pred


def crack_structure_loss_without_cldice(y_true, y_pred):
    """GPU-friendly weighted BCE and Dice objective."""
    binary_crossentropy, dice_loss, _ = crack_region_losses(y_true, y_pred)
    return (2.0 / 3.0) * binary_crossentropy + (1.0 / 3.0) * dice_loss


def crack_structure_loss(y_true, y_pred):
    """Combine pixel balance, overlap, and centerline connectivity."""
    epsilon = K.epsilon()
    binary_crossentropy, dice_loss, y_pred = crack_region_losses(y_true, y_pred)

    true_skeleton = soft_skeleton(y_true)
    predicted_skeleton = soft_skeleton(y_pred)
    topology_precision = (
        K.sum(predicted_skeleton * y_true) + epsilon
    ) / (K.sum(predicted_skeleton) + epsilon)
    topology_sensitivity = (
        K.sum(true_skeleton * y_pred) + epsilon
    ) / (K.sum(true_skeleton) + epsilon)
    cldice = (
        2.0 * topology_precision * topology_sensitivity + epsilon
    ) / (topology_precision + topology_sensitivity + epsilon)
    cldice_loss = 1.0 - cldice

    return (
        0.5 * binary_crossentropy
        + 0.25 * dice_loss
        + 0.25 * cldice_loss
    )


def residual_block(x, filters: int, time_distributed: bool = False):
    """Two convolutions with an identity or projected residual shortcut."""
    def convolution(kernel_size: int, activation=None):
        layer = layers.Conv2D(
            filters, kernel_size, padding="same", activation=activation
        )
        return layers.TimeDistributed(layer) if time_distributed else layer

    shortcut = x
    x = convolution(3, activation="relu")(x)
    x = convolution(3)(x)
    if K.int_shape(shortcut)[-1] != filters:
        shortcut = convolution(1)(shortcut)
    return layers.Activation("relu")(layers.Add()([x, shortcut]))


def dilated_spatial_pyramid(x, filters: int):
    """Aggregate spatial context over four receptive-field sizes."""
    branch_filters = max(filters // 4, 1)
    branches = [
        layers.Conv2D(
            branch_filters, 3, padding="same", dilation_rate=rate,
            activation="relu",
        )(x)
        for rate in (1, 2, 4, 8)
    ]
    x = layers.Concatenate(name="dilated_context")(branches)
    return layers.Conv2D(
        filters, 1, padding="same", activation="relu",
        name="context_projection",
    )(x)


def last_time_step(x, name: str):
    """Select encoder features aligned with the most recent input frame."""
    return layers.Lambda(
        lambda tensor: tensor[:, -1],
        output_shape=lambda input_shape: (input_shape[0],) + input_shape[2:],
        name=name,
    )(x)


def build_model(sequence_length: int, learning_rate: float,
                target_mode: str = "scaled_delta", base_filters: int = 16,
                topology_loss: bool = True) -> keras.Model:
    sequence = layers.Input((sequence_length, *IMAGE_SHAPE, 3), name="sequence")

    x = residual_block(sequence, base_filters, time_distributed=True)
    skip_256 = last_time_step(x, "skip_256")
    x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)
    x = residual_block(x, 2 * base_filters, time_distributed=True)
    skip_128 = last_time_step(x, "skip_128")
    x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)
    x = residual_block(x, 4 * base_filters, time_distributed=True)
    skip_64 = last_time_step(x, "skip_64")
    x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)
    x = layers.ConvLSTM2D(
        8 * base_filters, 3, padding="same", return_sequences=False
    )(x)
    x = dilated_spatial_pyramid(x, 8 * base_filters)
    x = residual_block(x, 8 * base_filters)
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate(name="decoder_skip_64")([x, skip_64])
    x = residual_block(x, 4 * base_filters)
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate(name="decoder_skip_128")([x, skip_128])
    x = residual_block(x, 2 * base_filters)
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate(name="decoder_skip_256")([x, skip_256])
    x = residual_block(x, base_filters)
    if target_mode == "crack_strength":
        growth = layers.Conv2D(
            1, 3, padding="same", activation="sigmoid",
            name="growth_probability",
        )(x)
        last_crack = layers.Lambda(
            lambda tensor: tensor[:, -1, :, :, :1],
            output_shape=lambda input_shape: (
                input_shape[0], input_shape[2], input_shape[3], 1
            ),
            name="last_crack_strength",
        )(sequence)
        available = layers.Lambda(
            lambda tensor: 1.0 - tensor,
            output_shape=lambda input_shape: input_shape,
            name="available_surface",
        )(last_crack)
        growth = layers.Multiply(name="new_crack_growth")([growth, available])
        prediction = layers.Add(name="crack_prediction")([last_crack, growth])
    else:
        prediction = layers.Conv2D(
            1, 3, padding="same", name="delta_prediction"
        )(x)

    model = keras.models.Model(sequence, prediction, name="crack_delta_convlstm")
    crack_loss = (
        crack_structure_loss if topology_loss
        else crack_structure_loss_without_cldice
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss=crack_loss if target_mode == "crack_strength" else weighted_delta_loss,
        metrics=["mean_absolute_error"],
    )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument("--output", type=Path,
                        default=Path("best_convlstm_delta.h5"))
    parser.add_argument(
        "--initial-weights", type=Path, default=None,
        help="Resume from weights saved by an earlier training process.",
    )
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--initial-epoch", type=int, default=0,
        help="Number of already completed epochs (used for resumed run labels).",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--max-horizon", type=int, default=5)
    parser.add_argument("--delta-scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=279)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--target-mode", choices=("scaled_delta", "crack_strength"),
                        default="scaled_delta")
    parser.add_argument("--base-filters", type=int, default=8)
    parser.add_argument("--crack-contrast", type=float, default=0.25)
    parser.add_argument(
        "--no-topology-loss", action="store_true",
        help="Use weighted BCE + Dice without clDice for older GPU backends.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sequence_length < 2:
        raise ValueError("--sequence-length must be at least 2")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be between 0 and 0.5")
    if args.max_horizon < 1 or args.delta_scale <= 0.0:
        raise ValueError("--max-horizon and --delta-scale must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    times, images = load_dataset(args.data_dir)
    if args.target_mode == "crack_strength":
        images = make_crack_strength(images, args.crack_contrast)

    # Split chronologically by target frame. Validation may use preceding training
    # frames as context, but no validation target can leak into training.
    split = int(len(images) * (1.0 - args.validation_fraction))
    train_samples = make_samples(
        args.sequence_length, split, args.sequence_length, args.max_horizon
    )
    val_samples = make_samples(
        split, len(images), args.sequence_length, args.max_horizon
    )
    if not len(train_samples) or not len(val_samples):
        raise ValueError("Dataset is too small for the requested split/window")

    train_data = WindowSequence(
        times, images, train_samples, args.sequence_length, args.batch_size,
        args.delta_scale, not args.no_augmentation, True, args.seed,
        args.target_mode,
    )
    val_data = WindowSequence(
        times, images, val_samples, args.sequence_length, args.batch_size,
        args.delta_scale, False, False, args.seed,
        args.target_mode,
    )
    model = build_model(
        args.sequence_length, args.learning_rate, args.target_mode,
        args.base_filters, not args.no_topology_loss,
    )
    if args.initial_weights is not None:
        if not args.initial_weights.exists():
            raise ValueError(f"Initial weights not found: {args.initial_weights}")
        # Loading weights instead of the complete compiled model avoids
        # deserializing the custom loss and stale optimizer/GPU state.
        model.load_weights(str(args.initial_weights))
        print(f"Loaded initial weights from {args.initial_weights}")
    model.summary()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps({
        "sequence_length": args.sequence_length,
        "image_shape": list(IMAGE_SHAPE),
        "training_frames": int(split),
        "validation_frames": int(len(images) - split),
        "training_samples": int(len(train_samples)),
        "validation_samples": int(len(val_samples)),
        "max_horizon": args.max_horizon,
        "output_mode": args.target_mode,
        "growth_mode": (
            "additive_monotonic"
            if args.target_mode == "crack_strength" else "none"
        ),
        "topology_loss": not args.no_topology_loss,
        "base_filters": args.base_filters,
        "crack_contrast": args.crack_contrast,
        "delta_scale": args.delta_scale,
        "input_channels": 3,
        "median_delta_t": float(np.median(np.diff(times))),
        "pixel_min": float(images.min()),
        "pixel_max": float(images.max()),
    }, indent=2) + "\n")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(args.output), monitor="val_loss", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(3, args.patience // 3),
            min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.CSVLogger(str(args.output.with_suffix(".csv"))),
    ]
    model.fit_generator(
        train_data,
        validation_data=val_data,
        initial_epoch=args.initial_epoch,
        epochs=args.initial_epoch + args.epochs,
        callbacks=callbacks,
        workers=1,
        use_multiprocessing=False,
    )


if __name__ == "__main__":
    main()
