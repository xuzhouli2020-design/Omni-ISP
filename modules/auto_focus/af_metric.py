"""
File: af_metric.py
Description: Focus sharpness metrics for contrast-detect auto-focus (CDAF).

Two metrics are provided:

  Tenengrad  — mean squared Sobel gradient magnitude in the ROI.
               High SNR robustness. Industry standard for CDAF.
               Higher value = sharper focus.

  LaplacianVar — variance of the Laplacian in the ROI.
                 Simpler, slightly more noise-sensitive.
                 Higher value = sharper focus.

All metrics operate on a normalised float32 greyscale (G sub-channel) image.

Phase 4.6
"""

import numpy as np
from util.bayer_utils import sobel_gradients, laplacian


def tenengrad(img: np.ndarray) -> float:
    """
    Tenengrad focus metric — mean of squared Sobel gradient magnitudes.

    Parameters
    ----------
    img : np.ndarray  2-D float32, any value range (typically [0, 1]).

    Returns
    -------
    float  Focus score. Higher = sharper.
           Returns 0.0 if image is too small to compute gradients.
    """
    if img.size < 9 or min(img.shape) < 3:
        return 0.0
    gx, gy = sobel_gradients(img)
    return float(np.mean(gx ** 2 + gy ** 2))


def laplacian_var(img: np.ndarray) -> float:
    """
    Laplacian variance focus metric.

    Parameters
    ----------
    img : np.ndarray  2-D float32.

    Returns
    -------
    float  Focus score. Higher = sharper.
    """
    if img.size < 9 or min(img.shape) < 3:
        return 0.0
    lap = laplacian(img)
    return float(np.var(lap))


def compute_focus_metric(img: np.ndarray, metric: str = "tenengrad") -> float:
    """
    Dispatch to the requested focus metric.

    Parameters
    ----------
    img    : np.ndarray  2-D float32 greyscale.
    metric : str         "tenengrad" | "laplacian_var"

    Returns
    -------
    float  Focus score (higher = sharper).
    """
    if metric == "tenengrad":
        return tenengrad(img)
    elif metric == "laplacian_var":
        return laplacian_var(img)
    else:
        raise ValueError(
            f"Unknown focus metric '{metric}'. "
            "Valid: 'tenengrad', 'laplacian_var'"
        )
