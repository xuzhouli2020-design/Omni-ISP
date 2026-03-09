"""
File: bilateral.py
Description: Bilateral filter for Y-channel denoising — edge-preserving spatial
             smoother with separate spatial (geometric) and intensity (photometric)
             Gaussian kernels.

Algorithm
---------
  BF[p] = (1/K) × Σ_q  f(||p−q||) × g(|I[p] − I[q]|) × I[q]

where f = spatial Gaussian, g = intensity Gaussian, K = normalisation sum.

Implementation: inner loop iterates over window offsets (window_size²) rather
than over pixels, allowing full NumPy vectorisation over the spatial dimension.
Complexity: O(H × W × window_size²).

Pure NumPy — no scipy dependency.

Config keys
-----------
  y_spatial_sigma   float  3.0   Spatial Gaussian radius (in pixels)
  y_intensity_sigma float  0.05  Range Gaussian sigma (normalised to [0,1])
  window_size       int    7     Search window side length (odd)

Author: EdgeISP / 10xEngineers Pvt Ltd
"""
import numpy as np


class BilateralFilter:
    """Bilateral filter applied to channel 0 (Y / luma) of a 3-channel image.

    Chroma channels (1 and 2) are passed through unchanged — use the 2D NR
    chroma_sigma path for chroma smoothing.
    """

    def __init__(self, img: np.ndarray, parm_2dnr: dict):
        self.img = img
        self.spatial_sigma = float(parm_2dnr.get("y_spatial_sigma", 3.0))
        self.intensity_sigma = float(parm_2dnr.get("y_intensity_sigma", 0.05))
        # window_size can come from new or legacy key
        self.window_size = int(parm_2dnr.get("window_size", 7))
        if self.window_size % 2 == 0:
            self.window_size += 1  # force odd

    def apply_bilateral(self) -> np.ndarray:
        """Apply bilateral filter and return the processed image."""
        in_image = self.img
        luma = in_image[:, :, 0].astype(np.float32)

        # Normalise to [0, 1] for intensity kernel so sigma is sensor-agnostic
        luma_max = float(luma.max())
        scale = luma_max if luma_max > 1.0 else 1.0
        luma_n = luma / scale

        h, w = luma_n.shape
        pad = self.window_size // 2

        # Reflect-pad the normalised luma
        padded = np.pad(luma_n, pad, mode="reflect")

        # Precompute spatial Gaussian weights for each offset (r, c)
        # shape: (window_size, window_size)
        half = self.window_size // 2
        ys, xs = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float32)
        spatial_w = np.exp(-(xs ** 2 + ys ** 2) / (2.0 * self.spatial_sigma ** 2))

        # Accumulators
        weighted_sum = np.zeros((h, w), dtype=np.float64)
        weight_sum = np.zeros((h, w), dtype=np.float64)

        for r in range(self.window_size):
            for c in range(self.window_size):
                # Tile of neighbour pixels aligned with luma_n
                neighbour = padded[r : r + h, c : c + w]

                # Intensity (range) Gaussian weight
                diff = (luma_n - neighbour) ** 2
                intensity_w = np.exp(
                    -diff / (2.0 * self.intensity_sigma ** 2)
                ).astype(np.float64)

                w_combined = spatial_w[r, c] * intensity_w

                weighted_sum += w_combined * neighbour
                weight_sum += w_combined

        # Normalise
        filtered = (weighted_sum / np.maximum(weight_sum, 1e-8)).astype(np.float32)

        # Rescale back to original range
        filtered_out = filtered * scale

        # Build output — only Y channel is modified
        out = in_image.copy()
        if in_image.dtype == np.float32 or str(in_image.dtype).startswith("float"):
            out[:, :, 0] = np.clip(filtered_out, 0.0, scale)
        else:
            out[:, :, 0] = np.uint8(np.clip(filtered_out, 0, 255))

        return out
