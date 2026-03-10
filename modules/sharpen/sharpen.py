"""
File: sharpen.py
Description: Implements sharpening for Omni-ISP.
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd

Modes
-----
  "usm"      — classic uniform unsharp masking (default, fast, backward compat)
  "adaptive" — edge-adaptive gain: noise-gated ramp so flat / noisy areas are not
               amplified; uses pure-NumPy Gaussian blur + Sobel edge detection
               (no scipy dependency).  Ref: docs/dev_notes.md "Sharpening — edge-
               adaptive gain + pipeline order fix".
"""

import time
from modules.sharpen.unsharp_masking import UnsharpMasking as USM
from modules.sharpen.adaptive_sharpen import AdaptiveSharpen

from util.utils import save_output_array_yuv


class Sharpening:
    """
    Sharpening — supports two modes:
      "usm"      Classical unsharp masking (default).
      "adaptive" Edge-strength-gated unsharp masking (higher quality, no noise
                 amplification, no halos at aggressive settings).
    """

    def __init__(self, img, platform, sensor_info, parm_sha, conv_std):
        self.img = img
        self.enable = parm_sha["is_enable"]
        self.sensor_info = sensor_info
        self.parm_sha = parm_sha
        self.mode = parm_sha.get("mode", "usm")  # "usm" | "adaptive"
        self.is_save = parm_sha["is_save"]
        self.platform = platform
        self.conv_std = conv_std

    def apply_unsharp_masking(self):
        """Apply classic uniform unsharp masking (USM mode)."""
        usm = USM(
            self.img, self.parm_sha["sharpen_sigma"], self.parm_sha["sharpen_strength"]
        )
        return usm.apply_sharpen()

    def apply_adaptive_sharpening(self):
        """Apply edge-adaptive sharpening (adaptive mode)."""
        ada = AdaptiveSharpen(self.img, self.parm_sha)
        return ada.apply_sharpen()

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array_yuv(
                self.platform["in_file"],
                self.img,
                "Out_Shapening_",
                self.platform,
                self.conv_std,
            )

    def execute(self):
        """
        Applying sharpening to input image
        """
        print("Sharpening = " + str(self.enable) + "  mode=" + self.mode)

        if self.enable is True:
            start = time.time()
            if self.mode == "adaptive":
                s_out = self.apply_adaptive_sharpening()
            else:
                # "usm" or any unrecognised value → fall back to classic USM
                s_out = self.apply_unsharp_masking()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = s_out

        self.save()
        return self.img
