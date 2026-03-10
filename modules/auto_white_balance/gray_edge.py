"""
File: gray_edge.py
Description: Gray Edge illuminant estimation for Auto White Balance.

Algorithm
─────────
van de Weijer et al. (2007) "Edge-Based Color Constancy", IEEE TIP.

  The Gray World hypothesis assumes the average color of a scene is achromatic.
  Gray Edge generalises this to image *derivatives* of order n:

    E[ |∂ⁿI/∂xⁿ|^p ]^(1/p)  ∝  illuminant color

  Key insight: edges in natural images are more uniformly distributed in color
  than flat regions, making the gradient domain more robust against scene
  content with a dominant hue (e.g. a green forest, a blue ocean).

Implementation
──────────────
  n=1, p=1  (first-order, L1):
    For each channel c ∈ {R, G, B}:
      mean_grad_c = mean( |∂I_c/∂x| + |∂I_c/∂y| )   [Sobel central-diff]
    r_gain = mean_grad_G / mean_grad_R
    b_gain = mean_grad_G / mean_grad_B

  n=2, p=1  (Laplacian variant):
    mean_grad_c = mean( |∇²I_c| )

  Minkowski p-norm generalisation:
    mean_grad_c = ( mean( (|grad_c|)^p ) )^(1/p)

  Robust thresholding:
    Pixels where any channel is saturated (> sat_threshold × max) or very dark
    (< dark_threshold × max) are excluded from gradient accumulation.

Parameters (from configs.yml awb section)
──────────────────────────────────────────
  grad_order      : int    1 = Sobel, 2 = Laplacian           [1]
  minkowski_p     : float  p-norm exponent                    [1.0]
  sat_threshold   : float  Saturation exclusion fraction      [0.95]
  dark_threshold  : float  Dark-pixel exclusion fraction      [0.05]
  min_gain        : float  Clamp gains below this to 1.0      [1.0]
  max_gain        : float  Clamp gains above this             [4.0]

Phase 4.4
"""

import numpy as np
from util.bayer_utils import sobel_gradients, laplacian


class GrayEdge:
    """
    Gray Edge illuminant estimator operating on a Bayer-domain raw image.

    Parameters
    ----------
    channels   : dict   {"r","gr","gb","b"} → (H//2,W//2) uint16 sub-images
    bit_depth  : int    Sensor bit depth
    grad_order : int    1 = Sobel, 2 = Laplacian
    p          : float  Minkowski p-norm (1 = L1 mean, 2 = RMS, …)
    sat_thresh : float  Saturation threshold fraction [0, 1]
    dark_thresh: float  Dark threshold fraction [0, 1]
    min_gain   : float  Minimum gain (gains < this → 1.0)
    max_gain   : float  Maximum gain clamp
    """

    def __init__(self,
                 channels: dict,
                 bit_depth: int,
                 grad_order: int  = 1,
                 p: float         = 1.0,
                 sat_thresh: float  = 0.95,
                 dark_thresh: float = 0.05,
                 min_gain: float    = 1.0,
                 max_gain: float    = 4.0):
        self.channels   = channels
        self.bit_depth  = bit_depth
        self.grad_order = grad_order
        self.p          = p
        self.sat_thresh  = sat_thresh
        self.dark_thresh = dark_thresh
        self.min_gain    = min_gain
        self.max_gain    = max_gain

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def calculate_gains(self) -> tuple:
        """
        Estimate illuminant and return (r_gain, b_gain).

        Returns
        -------
        (r_gain, b_gain) : tuple of float
            Gains to apply to R and B channels so they match G.
        """
        maxval = float((1 << self.bit_depth) - 1)

        r  = self.channels["r"].astype(np.float32)  / maxval
        gr = self.channels["gr"].astype(np.float32) / maxval
        gb = self.channels["gb"].astype(np.float32) / maxval
        b  = self.channels["b"].astype(np.float32)  / maxval
        g  = (gr + gb) * 0.5

        # Exclusion mask (built on the same half-res grid as the channels)
        sat  = self.sat_thresh
        dark = self.dark_thresh
        valid = (
            (r  > dark) & (r  < sat) &
            (g  > dark) & (g  < sat) &
            (b  > dark) & (b  < sat)
        )

        gm_r = self._channel_grad_mean(r, valid)
        gm_g = self._channel_grad_mean(g, valid)
        gm_b = self._channel_grad_mean(b, valid)

        r_gain = self._safe_gain(gm_g, gm_r)
        b_gain = self._safe_gain(gm_g, gm_b)

        # Clamp gains
        r_gain = float(np.clip(r_gain, self.min_gain, self.max_gain))
        b_gain = float(np.clip(b_gain, self.min_gain, self.max_gain))
        return r_gain, b_gain

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _channel_grad_mean(self, ch: np.ndarray, valid: np.ndarray) -> float:
        """
        Compute the Minkowski-p gradient mean for a single channel,
        restricted to valid (non-saturated, non-dark) pixels.

        Gradient is computed on the full channel; the valid mask is then
        applied to the interior region (which loses 1-pixel border due to
        finite-difference kernels).
        """
        if self.grad_order == 1:
            gx, gy = sobel_gradients(ch)
            grad = np.abs(gx) + np.abs(gy)
        else:
            grad = np.abs(laplacian(ch))

        # Trim valid mask to interior (gradient arrays lose 1-pixel border)
        valid_inner = valid[1:-1, 1:-1]

        if not np.any(valid_inner):
            # Fallback to channel mean if no valid pixels after masking
            return float(ch.mean()) + 1e-8

        masked = grad[valid_inner]

        p = self.p
        if p == 1.0:
            return float(masked.mean()) + 1e-8
        else:
            return float(np.mean(masked ** p) ** (1.0 / p)) + 1e-8

    def _safe_gain(self, numerator: float, denominator: float) -> float:
        """Compute gain ratio, clamped to [min_gain, max_gain]."""
        if denominator < 1e-8:
            return self.min_gain
        ratio = numerator / denominator
        # If the ratio falls below 1.0, the channel is already brighter
        # than green — we should still not boost (keep gain ≥ min_gain).
        return max(self.min_gain, ratio)
