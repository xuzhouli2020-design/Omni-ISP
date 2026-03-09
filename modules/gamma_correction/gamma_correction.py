"""
File: gamma_correction.py
Description: Implements the gamma look up table provided in the config file.
             Also supports analytical EOTF modes via the eotf module.
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------

Modes (parm_gmm["eotf"], default "lut"):
  "lut"     — per-bit-depth LUT from config (original behaviour, backward compat)
  "srgb"    — IEC 61966-2-1 sRGB piecewise EOTF
  "rec709"  — ITU-R BT.709 broadcast SDR transfer function
  "linear"  — γ = 1.0 passthrough (for scientific output / ML pipelines)
  "pq"      — SMPTE ST.2084 PQ for HDR10 (input normalised 0–10 000 nits → [0,1])
  "hlg"     — ARIB STD-B67 HLG for HDR broadcast (scene-referred)

Ref: docs/dev_notes.md "CCM target primaries + Gamma EOTF — multi-format output"
"""
import time
import numpy as np

from util.utils import save_output_array
from modules.gamma_correction.eotf import apply_eotf


class GammaCorrection:
    """
    Gamma Correction — supports LUT-based and analytical EOTF modes.
    """

    def __init__(self, img, platform, sensor_info, parm_gmm):
        self.img = img
        self.enable = parm_gmm["is_enable"]
        self.sensor_info = sensor_info
        self.bit_depth = sensor_info["bit_depth"]
        self.parm_gmm = parm_gmm
        self.eotf = parm_gmm.get("eotf", "lut")  # "lut"|"srgb"|"rec709"|"linear"|"pq"|"hlg"
        self.is_save = parm_gmm["is_save"]
        self.platform = platform

    def generate_gamma_8bit_lut(self):
        """
        Generates Gamma LUT for 8bit Image
        """
        lut = np.linspace(0, 255, 256)
        lut = np.uint8(np.round(255 * ((lut / 255) ** (1 / 2.2))))
        return lut

    def apply_gamma_lut(self):
        """Apply per-bit-depth LUT from config (original algorithm)."""
        if self.bit_depth == 8:
            lut = np.uint16(np.array(self.parm_gmm["gamma_lut_8"]))
        elif self.bit_depth == 10:
            lut = np.uint16(np.array(self.parm_gmm["gamma_lut_10"]))
        elif self.bit_depth == 12:
            lut = np.uint16(np.array(self.parm_gmm["gamma_lut_12"]))
        elif self.bit_depth == 14:
            lut = np.uint16(np.array(self.parm_gmm["gamma_lut_14"]))
        else:
            print("LUT is not available for the given bit depth.")
            return self.img
        return lut[self.img]

    def apply_gamma(self):
        """Dispatch to LUT or analytical EOTF based on config."""
        if self.eotf == "lut":
            return self.apply_gamma_lut()
        else:
            # Analytical EOTF: normalise input → apply function → uint8 [0,255]
            return apply_eotf(self.img, self.eotf, self.bit_depth)

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_gamma_correction_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """
        Execute Gamma Correction
        """
        print("Gamma Correction = " + str(self.enable) + "  eotf=" + self.eotf)
        if self.enable is True:
            start = time.time()
            gc_out = self.apply_gamma()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = gc_out

        self.save()
        return self.img
