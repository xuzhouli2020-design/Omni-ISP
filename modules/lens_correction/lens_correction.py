"""
File: lens_correction.py
Description: Unified Chromatic Aberration Correction (CAC) + Lens Distortion
             Correction (LDC) applied as a single warp in the Bayer domain.

Why a single Bayer-domain warp?
────────────────────────────────
Both CAC and LDC are coordinate-remapping operations (warpers):
  • LDC  — Brown-Conrady radial+tangential, same coefficients for all channels
  • CAC  — per-channel radial scale differential (R and B shift relative to G)

Composing them analytically means the demosaicker always sees a geometrically
correct AND chromatically aligned Bayer image.  Two separate passes would
require two bilinear resamples (quality loss) and would leave the demosaicker
working on geometrically distorted pixels.

Pipeline position: after Digital Gain, before LSC.

Usage
─────
  lc = LensCorrection(dga_raw, platform, sensor_info, parm_lens)
  lc_raw = lc.execute()

Author: Omni-ISP contributors
"""

import numpy as np
from util.bayer_utils import get_channel_offsets, reconstruct_bayer
from modules.lens_correction.warp_field import build_warp_field, remap_channel


class LensCorrection:
    """
    Unified CAC + LDC warp in the Bayer domain.

    Both corrections are composed into a single per-channel source-coordinate
    map that is evaluated with one bilinear resample per sub-channel.
    """

    def __init__(self, img: np.ndarray, platform: dict,
                 sensor_info: dict, parm_lens: dict):
        self.img         = img
        self.platform    = platform
        self.sensor_info = sensor_info
        self.parm        = parm_lens

        self.enable      = bool(parm_lens.get("is_enable", False))
        self.ldc_enable  = bool(parm_lens.get("ldc_enable", False))
        self.cac_enable  = bool(parm_lens.get("cac_enable", False))

        # LDC (Brown-Conrady) — achromatic, applied equally to all channels
        self.ldc_params = {
            "k1":              float(parm_lens.get("k1",              0.0)),
            "k2":              float(parm_lens.get("k2",              0.0)),
            "k3":              float(parm_lens.get("k3",              0.0)),
            "p1":              float(parm_lens.get("p1",              0.0)),
            "p2":              float(parm_lens.get("p2",              0.0)),
            "focal_length_px": float(parm_lens.get("focal_length_px", 0.0)),
            "center":          parm_lens.get("center", "auto"),
        }

        # CAC — per-channel differential radial scale polynomial [k1, k2, k3]
        r_ca_cfg = parm_lens.get("r_ca", [0.0, 0.0, 0.0])
        b_ca_cfg = parm_lens.get("b_ca", [0.0, 0.0, 0.0])
        self.r_ca = tuple(float(v) for v in r_ca_cfg)
        self.b_ca = tuple(float(v) for v in b_ca_cfg)
        self.g_ca = (0.0, 0.0, 0.0)   # G is the alignment reference

        self.bayer_pattern = sensor_info.get("bayer_pattern", "rggb").lower()
        self.bit_depth     = int(sensor_info.get("bit_depth", 12))

    # ── Public API ────────────────────────────────────────────────────────────

    def execute(self) -> np.ndarray:
        """
        Apply the unified CAC+LDC warp and return corrected Bayer uint16.
        Returns the input unchanged if is_enable is False.
        """
        if not self.enable:
            return self.img

        # If both sub-corrections are disabled, nothing to do
        if not self.ldc_enable and not self.cac_enable:
            return self.img

        # Effective LDC params: zero them out if ldc_enable is False
        ldc = self.ldc_params if self.ldc_enable else {
            "k1": 0.0, "k2": 0.0, "k3": 0.0,
            "p1": 0.0, "p2": 0.0,
            "focal_length_px": self.ldc_params["focal_length_px"],
            "center":          self.ldc_params["center"],
        }
        r_ca = self.r_ca if self.cac_enable else self.g_ca
        b_ca = self.b_ca if self.cac_enable else self.g_ca

        return self._apply_warp(ldc, r_ca, b_ca)

    # ── Implementation ────────────────────────────────────────────────────────

    def _apply_warp(self, ldc_params: dict,
                    r_ca: tuple, b_ca: tuple) -> np.ndarray:
        """Extract sub-channels, build per-channel warp, remap, reconstruct."""

        H, W = self.img.shape
        H_sub, W_sub = H // 2, W // 2

        offsets = get_channel_offsets(self.bayer_pattern)

        # Map each Bayer position to its CA delta: R, G (Gr/Gb), or B
        ca_map = {}
        for name, (ro, co) in offsets.items():
            if name == "r":
                ca_map[(ro, co)] = r_ca
            elif name == "b":
                ca_map[(ro, co)] = b_ca
            else:   # "gr" or "gb"
                ca_map[(ro, co)] = self.g_ca

        # Process each of the four sub-channels
        warped = {}
        for name, (ro, co) in offsets.items():
            sub_ch = self.img[ro::2, co::2].astype(np.float32)
            ca_delta = ca_map[(ro, co)]

            rows_src, cols_src = build_warp_field(
                H_sub, W_sub, ldc_params, ca_delta
            )
            warped[name] = remap_channel(sub_ch, rows_src, cols_src)

        # Clip to valid sensor range and reconstruct Bayer
        max_val = float(2 ** self.bit_depth - 1)
        for name in warped:
            warped[name] = np.clip(warped[name], 0.0, max_val)

        return reconstruct_bayer(warped, self.bayer_pattern,
                                 dtype=np.uint16)
