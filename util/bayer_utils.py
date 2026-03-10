"""
File: bayer_utils.py
Description: Shared Bayer-pattern channel extraction utilities used by
             3A stats, AWB, AF and other modules that need per-channel
             sub-images without duplicating pattern logic everywhere.

Supported CFA patterns: rggb, bggr, grbg, gbrg
"""

import numpy as np


# ---------------------------------------------------------------------------
# Pattern → channel offsets
# ---------------------------------------------------------------------------

# Maps pattern name → {"r":(row,col), "gr":(r,c), "gb":(r,c), "b":(r,c)}
_OFFSETS = {
    "rggb": {"r": (0, 0), "gr": (0, 1), "gb": (1, 0), "b": (1, 1)},
    "bggr": {"b": (0, 0), "gb": (0, 1), "gr": (1, 0), "r": (1, 1)},
    "grbg": {"gr": (0, 0), "r": (0, 1), "b": (1, 0), "gb": (1, 1)},
    "gbrg": {"gb": (0, 0), "b": (0, 1), "r": (1, 0), "gr": (1, 1)},
}


def get_channel_offsets(bayer_pattern: str) -> dict:
    """
    Return {channel_name: (row_offset, col_offset)} for a CFA pattern.
    """
    pattern = bayer_pattern.lower()
    if pattern not in _OFFSETS:
        raise ValueError(
            f"Unknown Bayer pattern '{bayer_pattern}'. "
            f"Supported: {list(_OFFSETS.keys())}"
        )
    return _OFFSETS[pattern]


def extract_channels(raw: np.ndarray, bayer_pattern: str) -> dict:
    """
    Extract the four Bayer sub-images (R, Gr, Gb, B) from a full-resolution
    raw Bayer frame by slicing at stride 2.

    Parameters
    ----------
    raw           : np.ndarray  2-D Bayer image, any integer dtype.
    bayer_pattern : str         CFA layout string ("rggb", "bggr", etc.)

    Returns
    -------
    dict with keys "r", "gr", "gb", "b" — each a 2-D ndarray of shape
    (H//2, W//2), same dtype as input.
    """
    offsets = get_channel_offsets(bayer_pattern)
    return {
        name: raw[ro::2, co::2]
        for name, (ro, co) in offsets.items()
    }


def green_channel(raw: np.ndarray, bayer_pattern: str) -> np.ndarray:
    """
    Return the average of Gr and Gb sub-images (shape H//2 × W//2).
    Used as a luminance proxy in 3A statistics and AF metrics.
    """
    chs = extract_channels(raw, bayer_pattern)
    return (chs["gr"].astype(np.float32) + chs["gb"].astype(np.float32)) / 2.0


def reconstruct_bayer(channels: dict, bayer_pattern: str,
                       dtype=None) -> np.ndarray:
    """
    Reconstruct a full-resolution Bayer image from four sub-channel arrays.

    Parameters
    ----------
    channels      : dict  {"r", "gr", "gb", "b"} → 2-D arrays (H//2, W//2)
    bayer_pattern : str   CFA layout
    dtype         : optional output dtype (defaults to input dtype)

    Returns
    -------
    np.ndarray  shape (H, W) full Bayer image
    """
    offsets = get_channel_offsets(bayer_pattern)
    sample = next(iter(channels.values()))
    h2, w2 = sample.shape
    out = np.zeros((h2 * 2, w2 * 2),
                   dtype=dtype or sample.dtype)
    for name, (ro, co) in offsets.items():
        out[ro::2, co::2] = channels[name]
    return out


def normalise(raw: np.ndarray, bit_depth: int) -> np.ndarray:
    """Scale a raw integer image to float32 [0, 1]."""
    return raw.astype(np.float32) / float((1 << bit_depth) - 1)


def denormalise(img: np.ndarray, bit_depth: int,
                dtype=np.uint16) -> np.ndarray:
    """Scale a float32 [0, 1] image back to integer raw."""
    maxval = float((1 << bit_depth) - 1)
    return np.clip(np.round(img * maxval), 0, maxval).astype(dtype)


# ---------------------------------------------------------------------------
# Gradient utilities (shared by Gray Edge AWB and AF metrics)
# ---------------------------------------------------------------------------

def sobel_gradients(img: np.ndarray) -> tuple:
    """
    Compute horizontal and vertical Sobel gradient magnitudes via central
    differences (no scipy/OpenCV dependency).

    Returns (gx, gy) each shape (H-2, W-2) float32.
    """
    f = img.astype(np.float32)
    # Sobel-like central differences (separable 3×1 × 1×3)
    # gx: horizontal gradient
    gx = (f[1:-1, 2:] - f[1:-1, :-2]) * 0.5
    # gy: vertical gradient
    gy = (f[2:, 1:-1] - f[:-2, 1:-1]) * 0.5
    return gx, gy


def laplacian(img: np.ndarray) -> np.ndarray:
    """
    4-connected discrete Laplacian, shape (H-2, W-2) float32.
    """
    f = img.astype(np.float32)
    return (f[1:-1, 2:] + f[1:-1, :-2]
            + f[2:, 1:-1] + f[:-2, 1:-1]
            - 4.0 * f[1:-1, 1:-1])
