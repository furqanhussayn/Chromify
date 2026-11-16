"""Reusable image utilities for CHROMIFY.

These helpers keep preprocessing and training code consistent across the project.
The LAB normalization uses the OpenCV Lab convention, where a and b channels are
stored in the range [-128, 127]. We convert them to normalized values with the
following convention:

    L  -> [0.0, 1.0] by dividing by 255
    a  -> (a - 128) / 110
    b  -> (b - 128) / 110

This keeps the channel range roughly centered around zero, which is convenient
for model training and matches the common colorization setup used with Lab data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
from PIL import Image

PathLike = Union[str, Path]


def load_rgb_image(path: PathLike) -> np.ndarray:
    """Load an image from disk and convert it to an RGB NumPy array."""
    image_path = Path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_grayscale(rgb_image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to conventional luminance grayscale."""
    return cv2.cvtColor(np.asarray(rgb_image), cv2.COLOR_RGB2GRAY)


def rgb_to_lab(rgb_image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to the CIELAB color space using OpenCV.

    OpenCV stores Lab values in the following channel ranges:
    For 8-bit images, OpenCV stores L as 0..255 and offsets a and b from
    their natural -128..127 ranges by 128, so all three channels are stored
    as unsigned bytes.
    """
    rgb_array = np.asarray(rgb_image)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2Lab)


def lab_l_channel(lab_image: np.ndarray) -> np.ndarray:
    """Extract the OpenCV CIELAB L channel as an 8-bit grayscale image."""
    lab_array = np.asarray(lab_image)
    if lab_array.ndim != 3 or lab_array.shape[2] != 3:
        raise ValueError("Lab image must have shape (height, width, 3).")
    return lab_array[..., 0]


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize an image with high-quality interpolation while preserving colors."""
    image_array = np.asarray(image)
    pillow_image = Image.fromarray(image_array)
    resized = pillow_image.resize(size, resample=Image.Resampling.LANCZOS)
    return np.asarray(resized)


def crop_to_4_5(
    image: np.ndarray,
    target_width: int | None = None,
    target_height: int | None = None,
) -> np.ndarray:
    """Crop an image to a 4:5 portrait aspect ratio without stretching it.

    The function keeps the center region of the image. It is designed for portrait
    crops such as passport-style headshots.
    """
    image_array = np.asarray(image)
    if image_array.ndim == 2:
        image_array = np.repeat(image_array[:, :, None], 3, axis=2)

    height, width = image_array.shape[:2]

    if width == 0 or height == 0:
        raise ValueError("Image dimensions must be greater than zero.")

    desired_width = width
    desired_height = int(round(width * 5 / 4))

    if desired_height > height:
        desired_height = height
        desired_width = int(round(height * 4 / 5))

    if desired_width > width:
        desired_width = width

    start_y = max(0, (height - desired_height) // 2)
    start_x = max(0, (width - desired_width) // 2)
    end_y = start_y + desired_height
    end_x = start_x + desired_width

    cropped = image_array[start_y:end_y, start_x:end_x]

    if cropped.shape[:2] != (desired_height, desired_width):
        cropped = np.pad(
            cropped,
            ((0, max(0, desired_height - cropped.shape[0])), (0, max(0, desired_width - cropped.shape[1])), (0, 0)),
            mode="constant",
        )

    if target_width is not None and target_height is not None:
        if cropped.shape[1] != target_width or cropped.shape[0] != target_height:
            return resize_image(cropped, (target_width, target_height))

    return cropped


def normalize_lab_channels(lab_image: np.ndarray) -> np.ndarray:
    """Normalize OpenCV Lab channels to a model-friendly floating-point range.

    The conversion follows the OpenCV Lab ranges exactly:
    - L: 0..255 -> 0..1
    - a: -128..127 -> normalized around zero using /110
    - b: -128..127 -> normalized around zero using /110
    """
    lab = np.asarray(lab_image, dtype=np.float32)
    if lab.ndim != 3 or lab.shape[2] != 3:
        raise ValueError("Lab image must have shape (height, width, 3).")

    normalized = np.empty_like(lab, dtype=np.float32)
    normalized[..., 0] = lab[..., 0] / 255.0
    normalized[..., 1] = (lab[..., 1] - 128.0) / 110.0
    normalized[..., 2] = (lab[..., 2] - 128.0) / 110.0
    return normalized


def lab_to_rgb(lab_image: np.ndarray) -> np.ndarray:
    """Convert normalized Lab channels back into an RGB array.

    The input is expected to follow the same convention used by
    `normalize_lab_channels`, meaning L is in [0, 1] and the a/b channels are
    centered around zero using the 110 scale factor.
    """
    lab = np.asarray(lab_image, dtype=np.float32).copy()
    if lab.ndim != 3 or lab.shape[2] != 3:
        raise ValueError("Lab image must have shape (height, width, 3).")

    lab[..., 0] = np.clip(lab[..., 0] * 255.0, 0, 255)
    lab[..., 1] = np.clip(lab[..., 1] * 110.0 + 128.0, 0, 255)
    lab[..., 2] = np.clip(lab[..., 2] * 110.0 + 128.0, 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2RGB)
