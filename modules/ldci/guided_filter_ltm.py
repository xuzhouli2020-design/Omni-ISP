"""
File: guided_filter_ltm.py
Description: Guided Filter Local Tone Mapping for LDCI.
             Implements edge-preserving base/detail decomposition using the
             He et al. (IEEE TPAMI 2013) guided filter — self-guided variant.
             Pure NumPy, O(N) via integral-image box filtering.
Author: EdgeISP (10xEngineers fork)
------------------------------------------------------------
Algorithm
---------
1.  Extract Y channel from YUV (H×W×3) image.
2.  Normalise Y to [0, 1] float domain.
3.  Self-guided filter: Y_base = guided_filter(Y, Y, r=guided_radius, eps=eps)
4.  Y_detail = Y − Y_base   (local contrast residual)
5.  Compress base:  Y_base_c = Y_base ^ base_compression  (gamma-like curve)
6.  Reconstruct:   Y_out = Y_base_c + detail_gain × Y_detail
7.  Clamp to [0, 1], scale back, write channel 0; Cb/Cr unchanged.

Key tunables
------------
guided_radius      : smoothing radius (pixels).  Larger → stronger LTM.
guided_eps         : edge sensitivity (0–1 normalised).
                     Small eps → more edge-preserving (sharper base).
detail_gain        : multiplier for detail layer (1.0 = no enhancement).
base_compression   : exponent for tone curve (< 1.0 compresses highlights).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Box filter via 2D integral image — O(N) independent of radius
# ---------------------------------------------------------------------------

def _box_filter(img: np.ndarray, r: int) -> np.ndarray:
    """
    2D uniform box filter with radius r (window = (2r+1)×(2r+1)).
    Uses reflect padding at borders and integral-image (prefix-sum) method.

    Parameters
    ----------
    img : np.ndarray  shape (H, W), float32 or float64
    r   : int         filter half-width

    Returns
    -------
    np.ndarray  shape (H, W), float32 — averaged output
    """
    H, W = img.shape
    # Reflect-pad to handle borders naturally
    padded = np.pad(img.astype(np.float64), r, mode="reflect")

    # 2D prefix sum (integral image)
    cs = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    # Prepend a zero row and column so indexing is offset-free
    cs = np.pad(cs, ((1, 0), (1, 0)), mode="constant")

    # Vectorised lookup for all (i,j) simultaneously via broadcasting
    r0 = np.arange(H)[:, None]   # shape (H, 1)
    c0 = np.arange(W)[None, :]   # shape (1, W)
    r1 = r0 + 2 * r              # shape (H, 1) — last row of window
    c1 = c0 + 2 * r              # shape (1, W) — last col of window

    # Integral image rectangle sum: A[r0:r1+1, c0:c1+1]
    box = (
        cs[r1 + 1, c1 + 1]
        - cs[r0,   c1 + 1]
        - cs[r1 + 1, c0  ]
        + cs[r0,   c0    ]
    )
    area = float((2 * r + 1) ** 2)
    return (box / area).astype(np.float32)


# ---------------------------------------------------------------------------
# Self-guided filter
# ---------------------------------------------------------------------------

def _guided_filter(I: np.ndarray, r: int, eps: float) -> np.ndarray:
    """
    Self-guided (edge-preserving) filter.

    Computes the LMMSE-optimal linear filter in each local window:
        q = a_k * I + b_k
    where a_k = var_I / (var_I + eps) and b_k = (1 - a_k) * mean_I.

    Parameters
    ----------
    I   : np.ndarray  shape (H, W), float32 in [0, 1]
    r   : int         filter radius (half-window)
    eps : float       regularisation — controls edge sensitivity
                      (0.01 on [0,1]-normalised signal ≈ moderate edge-awareness)

    Returns
    -------
    np.ndarray  shape (H, W), float32 — smoothed base layer
    """
    mean_I  = _box_filter(I,     r)
    mean_II = _box_filter(I * I, r)

    var_I = mean_II - mean_I * mean_I        # local variance
    var_I = np.maximum(var_I, 0.0)           # numerical safety (clip negatives)

    # Local filter coefficients
    a = var_I / (var_I + eps)                # ∈ [0, 1]: 0 at flat regions, 1 at edges
    b = mean_I * (1.0 - a)                   # = (1 - a) * mean_I

    # Average coefficients over local windows (the He et al. two-pass approach)
    mean_a = _box_filter(a, r)
    mean_b = _box_filter(b, r)

    return mean_a * I + mean_b


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class GuidedFilterLTM:
    """
    Local Tone Mapping via edge-preserving Guided Filter.

    Parameters
    ----------
    yuv            : np.ndarray  shape (H, W, 3) — YUV image
                     dtype float32 ([0,1] range) or uint8 ([0,255] range)
    parm_ldci      : dict with keys:
                       guided_radius      (int,   default 64)
                       guided_eps         (float, default 0.01)
                       detail_gain        (float, default 1.5)
                       base_compression   (float, default 0.7)
    """

    def __init__(self, yuv: np.ndarray, parm_ldci: dict):
        self.yuv = yuv
        self.guided_radius    = int(parm_ldci.get("guided_radius",    64))
        self.guided_eps       = float(parm_ldci.get("guided_eps",     0.01))
        self.detail_gain      = float(parm_ldci.get("detail_gain",    1.5))
        self.base_compression = float(parm_ldci.get("base_compression", 0.7))

    def apply_guided_ltm(self) -> np.ndarray:
        """
        Apply Guided Filter LTM to the Y channel.

        Returns
        -------
        np.ndarray  same shape and dtype as self.yuv
            Cb/Cr channels are copied unchanged; only Y is modified.
        """
        yuv = self.yuv
        is_float = yuv.dtype == np.float32

        # ---------- Extract Y, normalise to [0, 1] float32 ----------
        if is_float:
            Y = yuv[:, :, 0].astype(np.float32)           # already [0, 1]
        else:
            Y = yuv[:, :, 0].astype(np.float32) / 255.0   # uint8 → [0, 1]

        # ---------- Guided filter: smooth base layer ----------
        eps_scaled = self.guided_eps  # eps on [0,1]^2 normalised signal
        Y_base   = _guided_filter(Y, self.guided_radius, eps_scaled)

        # ---------- Detail layer: Y − Y_base ----------
        Y_detail = Y - Y_base

        # ---------- Compress base with power-law tone curve ----------
        # Y_base in [0,1]; raise to base_compression < 1 to lift shadows,
        # compress highlights.
        Y_base_safe = np.clip(Y_base, 0.0, 1.0)
        Y_base_c = np.power(Y_base_safe, self.base_compression)

        # ---------- Reconstruct Y ----------
        Y_out = Y_base_c + self.detail_gain * Y_detail
        Y_out = np.clip(Y_out, 0.0, 1.0)

        # ---------- Write back ----------
        out = yuv.copy()
        if is_float:
            out[:, :, 0] = Y_out
        else:
            out[:, :, 0] = np.round(Y_out * 255.0).astype(np.uint8)

        # Cb / Cr channels untouched (copied via yuv.copy())
        return out
