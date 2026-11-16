"""Shared image-level evaluation metrics for CHROMIFY colorization models."""

from __future__ import annotations

from typing import Dict

import numpy as np

try:
    from skimage.metrics import structural_similarity
except ImportError:
    structural_similarity = None

from src.preprocessing.image_utils import lab_to_rgb


def reconstruct_rgb_from_prediction(
    grayscale_image: np.ndarray,
    predicted_ab: np.ndarray,
) -> np.ndarray:
    """Reconstruct RGB from grayscale L input and normalized predicted a/b."""
    grayscale = np.asarray(grayscale_image, dtype=np.float32)
    ab = np.asarray(predicted_ab, dtype=np.float32)
    if grayscale.ndim != 2 or ab.shape != (*grayscale.shape, 2):
        raise ValueError("Expected grayscale shape (height, width) and a/b shape (height, width, 2).")

    normalized_lab = np.empty((*grayscale.shape, 3), dtype=np.float32)
    normalized_lab[..., 0] = grayscale / 255.0
    normalized_lab[..., 1:3] = ab
    return lab_to_rgb(normalized_lab)


def calculate_rgb_metrics(predicted_rgb: np.ndarray, target_rgb: np.ndarray) -> Dict[str, float]:
    """Calculate MAE, MSE, PSNR, and SSIM on RGB images normalized to [0, 1]."""
    prediction = np.asarray(predicted_rgb, dtype=np.float32) / 255.0
    target = np.asarray(target_rgb, dtype=np.float32) / 255.0
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction and target shapes differ: {prediction.shape} vs {target.shape}")

    difference = prediction - target
    mse = float(np.mean(np.square(difference)))
    mae = float(np.mean(np.abs(difference)))
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))
    if structural_similarity is not None:
        ssim = float(
            structural_similarity(
                prediction,
                target,
                channel_axis=2,
                data_range=1.0,
            )
        )
    else:
        ssim = _calculate_rgb_ssim(prediction, target)
    return {"mae": mae, "mse": mse, "psnr": psnr, "ssim": ssim}


def _calculate_rgb_ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    """Calculate channel-wise SSIM when scikit-image is unavailable."""
    import cv2

    c1 = 0.01**2
    c2 = 0.03**2
    scores = []
    for channel in range(prediction.shape[2]):
        predicted_channel = prediction[..., channel]
        target_channel = target[..., channel]
        mean_prediction = cv2.GaussianBlur(predicted_channel, (11, 11), 1.5)
        mean_target = cv2.GaussianBlur(target_channel, (11, 11), 1.5)
        variance_prediction = cv2.GaussianBlur(
            predicted_channel * predicted_channel, (11, 11), 1.5
        ) - mean_prediction * mean_prediction
        variance_target = cv2.GaussianBlur(
            target_channel * target_channel, (11, 11), 1.5
        ) - mean_target * mean_target
        covariance = cv2.GaussianBlur(
            predicted_channel * target_channel, (11, 11), 1.5
        ) - mean_prediction * mean_target
        score = (
            (2 * mean_prediction * mean_target + c1)
            * (2 * covariance + c2)
            / ((mean_prediction**2 + mean_target**2 + c1)
               * (variance_prediction + variance_target + c2))
        )
        scores.append(float(np.mean(score)))
    return float(np.mean(scores))


def average_metrics(metrics: list[Dict[str, float]]) -> Dict[str, float]:
    """Average per-image metric dictionaries."""
    if not metrics:
        raise ValueError("At least one metric result is required.")
    return {
        key: float(np.mean([metric[key] for metric in metrics]))
        for key in metrics[0]
    }
