"""
File: adaptive_sharpen.py
Description: Edge-adaptive unsharp masking — sharpening gain gated by local edge
             strength so flat / noisy regions are not amplified.

Algorithm
---------
1. Gaussian blur of Y channel  → unsharp mask  = Y - blur(Y)
2. Halo suppression           → mask  = clip(mask, -halo_limit, +halo_limit)
3. Sobel gradient magnitude   → edge_str  (normalised to [0, 1])
4. Gain ramp                  → gain_map = gain_max × ramp(edge_str, noise_floor, edge_max)
5. Sharpen                    → Y_sharp  = Y + gain_map × mask

Pure NumPy — no scipy dependency.

Config keys used (all optional, with sensible defaults):
  sharpen_sigma     float  1.5    Gaussian blur radius for unsharp mask
  gain              float  0.8    Maximum sharpening gain (at strong edges)
  noise_floor       float  0.02   Normalised edge strength below which gain = 0
  edge_max          float  0.15   Normalised edge strength at which gain = gain_max
  halo_limit        float  0.1    Mask clip value for halo suppression (normalised)

Author: Omni-ISP / 10xEngineers Pvt Ltd
"""
import numpy as np


def _gaussian_kernel_1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    """Return a normalised 1-D Gaussian kernel."""
    radius = max(1, int(truncate * sigma + 0.5))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


def _gaussian_blur_2d(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable 2-D Gaussian blur via row-then-column 1-D sliding sums.

    Uses reflect-padding to avoid border artefacts.  Pure NumPy — no scipy.
    """
    k = _gaussian_kernel_1d(sigma)
    radius = len(k) // 2
    h, w = img.shape

    # --- horizontal pass ---
    pad_h = np.pad(img, [(0, 0), (radius, radius)], mode="reflect").astype(np.float32)
    blurred_h = np.zeros((h, w), dtype=np.float32)
    for i, ki in enumerate(k):
        blurred_h += ki * pad_h[:, i : i + w]

    # --- vertical pass ---
    pad_v = np.pad(blurred_h, [(radius, radius), (0, 0)], mode="reflect")
    blurred = np.zeros((h, w), dtype=np.float32)
    for i, ki in enumerate(k):
        blurred += ki * pad_v[i : i + h, :]

    return blurred


def _sobel_magnitude(img: np.ndarray) -> np.ndarray:
    """Return Sobel gradient magnitude of a 2-D float image.

    Implements:
        Gx = [-1  0  1; -2  0  2; -1  0  1] / 8
        Gy = Gx.T
        magnitude = sqrt(Gx² + Gy²)

    Uses reflect-padding for a full-size output.
    """
    kx = np.array(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float32
    ) / 8.0
    ky = kx.T

    h, w = img.shape
    padded = np.pad(img, 1, mode="reflect").astype(np.float32)

    gx = np.zeros((h, w), dtype=np.float32)
    gy = np.zeros((h, w), dtype=np.float32)
    for r in range(3):
        for c in range(3):
            tile = padded[r : r + h, c : c + w]
            gx += kx[r, c] * tile
            gy += ky[r, c] * tile

    return np.sqrt(gx * gx + gy * gy)


class AdaptiveSharpen:
    """Edge-adaptive sharpening: gain is proportional to local edge strength.

    Flat / noisy regions receive zero gain → noise not amplified.
    Strong true edges receive full gain_max → maximum enhancement.
    Halo limiting clips the unsharp mask before gain application.
    """

    def __init__(self, img: np.ndarray, parm_sha: dict):
        self.img = img
        self.sigma = float(parm_sha.get("sharpen_sigma", 1.5))
        self.gain_max = float(parm_sha.get("gain", 0.8))
        self.noise_floor = float(parm_sha.get("noise_floor", 0.02))
        self.edge_max = float(parm_sha.get("edge_max", 0.15))
        self.halo_limit = float(parm_sha.get("halo_limit", 0.1))

    def apply_sharpen(self) -> np.ndarray:
        """Apply edge-adaptive sharpening to Y channel (channel 0) in-place."""
        luma = np.float32(self.img[:, :, 0])

        # Determine the operating range and normalise to [0, 1] for edge
        # detection so that noise_floor / edge_max are sensor-agnostic.
        luma_max = float(luma.max())
        scale = luma_max if luma_max > 1.0 else 1.0
        luma_norm = luma / scale  # [0, 1]

        # 1. Unsharp mask in normalised domain
        blurred = _gaussian_blur_2d(luma_norm, self.sigma)
        mask = luma_norm - blurred  # high-pass residual

        # 2. Halo suppression — clip mask to ±halo_limit
        halo_limit_scaled = self.halo_limit
        mask = np.clip(mask, -halo_limit_scaled, halo_limit_scaled)

        # 3. Sobel edge strength
        edge_str = _sobel_magnitude(luma_norm)

        # 4. Gain ramp: 0 at noise_floor → gain_max at edge_max
        denom = max(self.edge_max - self.noise_floor, 1e-8)
        gain_map = self.gain_max * np.clip(
            (edge_str - self.noise_floor) / denom, 0.0, 1.0
        ).astype(np.float32)

        # 5. Apply gain-modulated mask (still in normalised space)
        sharpened_norm = luma_norm + gain_map * mask

        # Rescale back and write to channel 0
        sharpened = sharpened_norm * scale

        if self.img.dtype == np.float32 or str(self.img.dtype).startswith("float"):
            self.img[:, :, 0] = np.clip(sharpened, 0.0, scale)
        else:
            self.img[:, :, 0] = np.uint8(np.clip(sharpened, 0, 255))

        return self.img
