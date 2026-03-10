"""
File: purple_fringe_removal.py
Description: Remove purple/magenta halos that appear at blown highlight edges
             due to longitudinal chromatic aberration.

Algorithm
─────────
1. Detect "purple" pixels by hue: H ∈ [hue_center ± hue_half_width] degrees
   AND saturation S > sat_threshold.
2. Detect blown highlights: luminance Y > highlight_threshold.
3. Dilate the highlight mask by `desaturation_radius` pixels — the fringe
   occurs at the *edge* of highlights, not inside them.
4. Fringe mask = purple_mask AND dilated_highlight_mask (but NOT inside blown core).
5. Desaturate fringe pixels: blend chroma toward gray (reduce S by `strength`).

Pipeline position: RGB domain, after Demosaic, before CCM.

Input/output dtype
──────────────────
Accepts uint8, uint16, or float32 RGB.  Normalises internally to float32 [0,1],
processes, then returns in the same dtype/range as input.

Author: Omni-ISP contributors
"""

import numpy as np


class PurpleFringeRemoval:
    """
    Selective desaturation of purple/magenta chromatic fringe near highlights.
    """

    def __init__(self, img: np.ndarray, platform: dict,
                 sensor_info: dict, parm_pfr: dict):
        self.img      = img
        self.platform = platform
        self.enable   = bool(parm_pfr.get("is_enable", False))

        # Detection parameters
        self.hue_center      = float(parm_pfr.get("hue_center",       290.0))
        self.hue_half_width  = float(parm_pfr.get("hue_half_width",    30.0))
        self.sat_threshold   = float(parm_pfr.get("sat_threshold",      0.35))
        self.highlight_thr   = float(parm_pfr.get("highlight_threshold", 0.85))
        self.desat_radius    = int(parm_pfr.get("desaturation_radius",   3))
        self.strength        = float(parm_pfr.get("strength",            1.0))

    # ── Public API ────────────────────────────────────────────────────────────

    def execute(self) -> np.ndarray:
        """
        Detect and remove purple fringe.  Returns corrected RGB image in
        the same dtype and range as the input.
        """
        if not self.enable:
            return self.img

        img_in = self.img
        # Normalise to float32 [0, 1]
        if img_in.dtype == np.uint8:
            img_f = img_in.astype(np.float32) / 255.0
            out_uint8 = True
        elif img_in.dtype == np.uint16:
            img_f = img_in.astype(np.float32) / 65535.0
            out_uint8 = False
        else:
            img_f = img_in.astype(np.float32)
            img_f = np.clip(img_f, 0.0, 1.0)
            out_uint8 = None   # float passthrough

        corrected_f = self._remove_fringe(img_f)

        # Return in original dtype
        if out_uint8 is True:
            return (corrected_f * 255.0).clip(0, 255).astype(np.uint8)
        elif out_uint8 is False:
            return (corrected_f * 65535.0).clip(0, 65535).astype(np.uint16)
        else:
            return corrected_f.astype(img_in.dtype)

    # ── Implementation ────────────────────────────────────────────────────────

    def _remove_fringe(self, img: np.ndarray) -> np.ndarray:
        """Core processing on float32 [0,1] RGB."""
        R, G, B = img[..., 0], img[..., 1], img[..., 2]

        # ── HSV conversion ────────────────────────────────────────────────────
        H, S, V = self._rgb_to_hsv(R, G, B)

        # ── Purple hue mask ───────────────────────────────────────────────────
        hue_lo = (self.hue_center - self.hue_half_width) % 360.0
        hue_hi = (self.hue_center + self.hue_half_width) % 360.0

        if hue_lo < hue_hi:
            in_hue_range = (H >= hue_lo) & (H <= hue_hi)
        else:
            # Wraps around 0°/360°
            in_hue_range = (H >= hue_lo) | (H <= hue_hi)

        purple_mask = in_hue_range & (S > self.sat_threshold)

        # ── Highlight proximity mask ──────────────────────────────────────────
        lum = 0.2126 * R + 0.7152 * G + 0.0722 * B
        highlight_core = lum > self.highlight_thr

        # Dilate highlight mask to find the fringe zone around highlights
        fringe_zone = self._dilate(highlight_core, self.desat_radius)

        # Fringe: purple AND near highlight AND not inside the blown core itself
        fringe_mask = purple_mask & fringe_zone & (~highlight_core)

        if not np.any(fringe_mask):
            return img

        # ── Desaturation of fringe pixels ────────────────────────────────────
        # Reduce saturation toward zero (gray) by `strength`, preserving hue.
        S_new = S * (1.0 - self.strength * fringe_mask.astype(np.float32))

        # Reconstruct RGB from adjusted HSV
        R_out, G_out, B_out = self._hsv_to_rgb(H, S_new, V)

        return np.stack([R_out, G_out, B_out], axis=2)

    # ── HSV helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _rgb_to_hsv(R: np.ndarray, G: np.ndarray,
                    B: np.ndarray) -> tuple:
        """
        Vectorised RGB [0,1] → HSV.
        H in [0, 360), S in [0, 1], V in [0, 1].
        """
        Cmax = np.maximum(np.maximum(R, G), B)
        Cmin = np.minimum(np.minimum(R, G), B)
        delta = Cmax - Cmin

        V = Cmax

        safe_max = np.where(Cmax > 1e-8, Cmax, 1.0)
        S = np.where(Cmax > 1e-8, delta / safe_max, 0.0)

        # Hue
        H = np.zeros_like(R)
        mask_r = (Cmax == R) & (delta > 1e-8)
        mask_g = (Cmax == G) & (delta > 1e-8)
        mask_b = (Cmax == B) & (delta > 1e-8)

        H[mask_r] = (60.0 * ((G[mask_r] - B[mask_r]) / delta[mask_r])) % 360.0
        H[mask_g] = (60.0 * ((B[mask_g] - R[mask_g]) / delta[mask_g]) + 120.0) % 360.0
        H[mask_b] = (60.0 * ((R[mask_b] - G[mask_b]) / delta[mask_b]) + 240.0) % 360.0

        return H, S, V

    @staticmethod
    def _hsv_to_rgb(H: np.ndarray, S: np.ndarray,
                    V: np.ndarray) -> tuple:
        """Vectorised HSV → RGB [0,1]."""
        h = H / 60.0
        i = np.floor(h).astype(np.int32) % 6
        f = h - np.floor(h)

        p = V * (1.0 - S)
        q = V * (1.0 - S * f)
        t = V * (1.0 - S * (1.0 - f))

        R = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                      [V, q, p, p, t, V])
        G = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                      [t, V, V, q, p, p])
        B = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                      [p, p, t, V, V, q])
        return R, G, B

    @staticmethod
    def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
        """
        Binary dilation by `radius` pixels using repeated max-pooling.
        Pure NumPy — no scipy dependency.
        """
        if radius <= 0:
            return mask
        out = mask.copy().astype(np.float32)
        for _ in range(radius):
            # 4-connected max spread
            out_new = out.copy()
            out_new[1:,  :]  = np.maximum(out_new[1:,  :],  out[:-1, :])   # down
            out_new[:-1, :]  = np.maximum(out_new[:-1, :],  out[1:,  :])   # up
            out_new[:,  1:]  = np.maximum(out_new[:,  1:],  out[:,  :-1])  # right
            out_new[:,  :-1] = np.maximum(out_new[:,  :-1], out[:,  1:])   # left
            out = out_new
        return out > 0.5
