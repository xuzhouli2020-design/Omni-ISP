"""
File: lmmse.py
Description: LMMSE (Linear Minimum Mean Square Error) demosaicing.
             Implements the directional G-interpolation + colour-difference bilinear
             approach based on Zhang & Wu, "Color demosaicking via directional linear
             minimum mean square-error estimation," IEEE TIP, 2005.
             Pure NumPy — no scipy required.
Author: EdgeISP (10xEngineers fork)
------------------------------------------------------------
Algorithm outline
-----------------
1. Estimate G at every non-G site using two directional (H and V) first-pass
   estimates, then fuse them with activity-based (2nd-derivative) weighting —
   this is the LMMSE Wiener-inspired step that gives noise robustness.

2. Compute colour-difference planes R−G and B−G at their known sample sites.

3. Bilinear-interpolate R−G and B−G to every pixel using the regular 2×2
   sampling grid of each colour plane.

4. Reconstruct R = G + (R−G) and B = G + (B−G) everywhere.

Works for any standard Bayer pattern (RGGB, BGGR, GRBG, GBRG) because in
every pattern a non-G pixel always has G at its immediate H and V neighbours.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _interpolate_G(raw: np.ndarray,
                   mask_g: np.ndarray,
                   mask_r: np.ndarray,
                   mask_b: np.ndarray) -> np.ndarray:
    """
    Step 1 — Interpolate the G channel at all R and B sites.

    For each non-G pixel at (i, j), two gradient-corrected bilinear estimates
    are formed:
        GH = (raw[i,j-1] + raw[i,j+1]) / 2
             + (2·raw[i,j] − raw[i,j-2] − raw[i,j+2]) / 4
        GV = (raw[i-1,j] + raw[i+1,j]) / 2
             + (2·raw[i,j] − raw[i-2,j] − raw[i+2,j]) / 4

    These are then fused by inverse-activity weighting:
        DH = |raw[i,j-2] − 2·raw[i,j] + raw[i,j+2]| / 2
        DV = |raw[i-2,j] − 2·raw[i,j] + raw[i+2,j]| / 2
        G  = (DV·GH + DH·GV) / (DH + DV + ε)

    (Low DH means the signal is smooth horizontally → trust GH more.)
    """
    f = raw.astype(np.float32)
    # Reflect-pad by 2 so all shifted slices are the same shape as f
    p = np.pad(f, 2, mode="reflect")

    c  = p[2:-2, 2:-2]   # center (= f itself)
    hl = p[2:-2, 1:-3]   # left  1
    hr = p[2:-2, 3:-1]   # right 1
    hl2 = p[2:-2, 0:-4]  # left  2
    hr2 = p[2:-2, 4:]    # right 2
    vu  = p[1:-3, 2:-2]  # up    1
    vd  = p[3:-1, 2:-2]  # down  1
    vu2 = p[0:-4, 2:-2]  # up    2
    vd2 = p[4:,   2:-2]  # down  2

    # Directional G estimates
    GH = (hl + hr) / 2.0 + (2.0 * c - hl2 - hr2) / 4.0
    GV = (vu + vd) / 2.0 + (2.0 * c - vu2 - vd2) / 4.0

    # Activity (2nd-order gradient magnitude)
    DH = np.abs(hl2 - 2.0 * c + hr2) * 0.5
    DV = np.abs(vu2 - 2.0 * c + vd2) * 0.5

    # LMMSE-inspired fusion: lower activity direction gets higher weight.
    # Adding eps to EACH weight avoids 0/0 when both DH=DV=0 (uniform regions)
    # and ensures (GH+GV)/2 in the flat case.
    eps = 1e-6
    w_H = DV + eps   # weight for GH: larger when V activity is high → trust H
    w_V = DH + eps   # weight for GV: larger when H activity is high → trust V
    G_interp = (w_H * GH + w_V * GV) / (w_H + w_V)

    # Keep known G samples; fill R and B sites with the fused estimate
    G = np.where(mask_g, f, G_interp)
    return G


def _interpolate_color_diff(diff_known: np.ndarray,
                             mask_known: np.ndarray) -> np.ndarray:
    """
    Step 3 — Bilinear interpolation of a colour-difference plane.

    diff_known : the colour-difference array (e.g. R − G), non-zero only at
                 known-colour sites.
    mask_known : boolean mask — True where diff_known is valid.

    Strategy: for each pixel that is NOT a known site,
      - use the average of its 2 horizontal same-colour neighbours (when they exist)
      - else use the average of its 2 vertical same-colour neighbours
      - else use the average of its 4 diagonal same-colour neighbours

    This exactly implements bilinear interpolation on the regular 2×2 colour
    grid without requiring any sparse matrix solver.
    """
    known = mask_known.astype(np.float32)
    diff  = diff_known.astype(np.float32)

    # Reflect-pad diff by 1; zero-pad mask by 1 (no phantom known pixels at border)
    pd = np.pad(diff,  1, mode="reflect")
    pm = np.pad(known, 1, mode="constant", constant_values=0.0)

    # Horizontal neighbours (left and right by 1)
    h_sum = pd[1:-1, :-2] + pd[1:-1, 2:]
    h_cnt = pm[1:-1, :-2] + pm[1:-1, 2:]

    # Vertical neighbours (up and down by 1)
    v_sum = pd[:-2, 1:-1] + pd[2:, 1:-1]
    v_cnt = pm[:-2, 1:-1] + pm[2:, 1:-1]

    # Diagonal neighbours (four corners)
    d_sum = pd[:-2, :-2] + pd[:-2, 2:] + pd[2:, :-2] + pd[2:, 2:]
    d_cnt = pm[:-2, :-2] + pm[:-2, 2:] + pm[2:, :-2] + pm[2:, 2:]

    eps = 1e-8
    h_est = h_sum / (h_cnt + eps)
    v_est = v_sum / (v_cnt + eps)
    d_est = d_sum / (d_cnt + eps)

    # Priority: known > H > V > diagonal
    result = np.where(
        mask_known,
        diff,
        np.where(
            h_cnt > 0,
            h_est,
            np.where(
                v_cnt > 0,
                v_est,
                d_est,
            ),
        ),
    )
    return result


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class LMMSE:
    """
    LMMSE CFA demosaicing.

    Parameters
    ----------
    raw_in : np.ndarray  shape (H, W), dtype uint16 (or any integer)
        Single-channel Bayer mosaic.
    masks  : tuple of three bool arrays (mask_r, mask_g, mask_b)
        True where the respective channel is sampled — as produced by
        Demosaic.masks_cfa_bayer().
    """

    def __init__(self, raw_in: np.ndarray, masks):
        self.img    = raw_in
        self.mask_r, self.mask_g, self.mask_b = masks

    def apply_lmmse(self) -> np.ndarray:
        """
        Run LMMSE demosaicing.

        Returns
        -------
        np.ndarray  shape (H, W, 3), dtype float32
            Demosaiced RGB image (values may exceed input range slightly;
            caller should clip to [0, 2**bit_depth − 1] and cast to uint16).
        """
        raw = self.img.astype(np.float32)
        H, W = raw.shape

        # ------------------------------------------------------------------
        # Step 1: interpolate G everywhere
        # ------------------------------------------------------------------
        G = _interpolate_G(raw, self.mask_g, self.mask_r, self.mask_b)

        # ------------------------------------------------------------------
        # Step 2: colour-difference planes at known sites
        # ------------------------------------------------------------------
        R_diff_known = np.where(self.mask_r, raw - G, 0.0).astype(np.float32)
        B_diff_known = np.where(self.mask_b, raw - G, 0.0).astype(np.float32)

        # ------------------------------------------------------------------
        # Step 3: bilinear interpolation of colour differences
        # ------------------------------------------------------------------
        R_diff = _interpolate_color_diff(R_diff_known, self.mask_r)
        B_diff = _interpolate_color_diff(B_diff_known, self.mask_b)

        # ------------------------------------------------------------------
        # Step 4: reconstruct R, B from G + colour differences
        # ------------------------------------------------------------------
        R = G + R_diff
        B = G + B_diff

        demos_out = np.stack([R, G, B], axis=-1)   # (H, W, 3)
        return demos_out.astype(np.float32)
