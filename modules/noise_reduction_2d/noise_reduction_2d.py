"""
File: noise_reduction_2d.py
Description: Apply denoising algorithms on luminance channel (and optionally chroma).
Author: 10xEngineers
------------------------------------------------------------

Modes (controlled by parm_2dnr["mode"], default "nlm"):
  "nlm"         — Non-Local Means on Y channel (original, default, high quality)
  "bilateral"   — Bilateral filter on Y channel (pure NumPy, edge-preserving)
  "chroma_only" — Gaussian smooth on Cb/Cr only; Y left untouched.
                  Lightweight: useful when Y is already clean (e.g. DL joint model).
  "off"         — pass-through regardless of is_enable flag

Ref: docs/dev_notes.md "2D NR — dual-role design"
"""
import time
import numpy as np
from util.utils import save_output_array_yuv
from modules.noise_reduction_2d.non_local_means import NLM
from modules.noise_reduction_2d.bilateral import BilateralFilter


def _gaussian_blur_2d(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable 2-D Gaussian blur (pure NumPy). See adaptive_sharpen for details."""
    radius = max(1, int(3 * sigma + 0.5))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k = (k / k.sum()).astype(np.float32)
    h, w = img.shape
    # Horizontal pass
    pad_h = np.pad(img.astype(np.float32), [(0, 0), (radius, radius)], mode="reflect")
    blurred_h = np.zeros((h, w), dtype=np.float32)
    for i, ki in enumerate(k):
        blurred_h += ki * pad_h[:, i : i + w]
    # Vertical pass
    pad_v = np.pad(blurred_h, [(radius, radius), (0, 0)], mode="reflect")
    blurred = np.zeros((h, w), dtype=np.float32)
    for i, ki in enumerate(k):
        blurred += ki * pad_v[i : i + h, :]
    return blurred


class NoiseReduction2d:
    """
    2D Noise Reduction — supports four modes:
      "nlm"         Classical NLM on Y (default, high quality).
      "bilateral"   Edge-preserving bilateral filter on Y (pure NumPy, faster).
      "chroma_only" Gaussian smoothing of Cb/Cr only; Y is passed through.
      "off"         Full pass-through (overrides is_enable).
    """

    def __init__(self, img, sensor_info, parm_2dnr, platform, conv_std):
        self.img = img
        self.enable = parm_2dnr["is_enable"]
        self.mode = parm_2dnr.get("mode", "nlm")  # "nlm"|"bilateral"|"chroma_only"|"off"
        self.sensor_info = sensor_info
        self.parm_2dnr = parm_2dnr
        self.conv_std = conv_std
        self.is_progress = platform["disable_progress_bar"]
        self.is_leave = platform["leave_pbar_string"]
        self.is_save = parm_2dnr["is_save"]
        self.platform = platform

    def apply_nlm(self):
        """Apply Non-Local Means to Y channel (original algorithm)."""
        nlm = NLM(self.img, self.sensor_info, self.parm_2dnr, self.platform)
        return nlm.apply_nlm()

    def apply_bilateral(self):
        """Apply bilateral filter to Y channel (pure NumPy)."""
        bf = BilateralFilter(self.img, self.parm_2dnr)
        return bf.apply_bilateral()

    def apply_chroma_only(self):
        """Gaussian-smooth Cb and Cr; leave Y (channel 0) untouched.

        chroma_sigma controls the blur radius. Aggressive blurring is appropriate
        because the human eye is insensitive to chroma resolution.
        """
        chroma_sigma = float(self.parm_2dnr.get("chroma_sigma", 6.0))
        out = self.img.copy()
        # Smooth Cb and Cr independently
        for ch in (1, 2):
            plane = self.img[:, :, ch].astype(np.float32)
            blurred = _gaussian_blur_2d(plane, chroma_sigma)
            if out.dtype == np.float32 or str(out.dtype).startswith("float"):
                out[:, :, ch] = np.clip(blurred, 0.0, plane.max() if plane.max() > 1.0 else 1.0)
            else:
                out[:, :, ch] = np.uint8(np.clip(blurred, 0, 255))
        return out

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array_yuv(
                self.platform["in_file"],
                self.img,
                "Out_2d_noise_reduction_",
                self.platform,
                self.conv_std,
            )

    def execute(self):
        """
        Executing 2D noise reduction module
        """
        print("Noise Reduction 2d = " + str(self.enable) + "  mode=" + self.mode)

        if self.mode == "off":
            # Hard bypass regardless of is_enable
            self.save()
            return self.img

        if self.enable is True:
            start = time.time()
            if self.mode == "bilateral":
                s_out = self.apply_bilateral()
            elif self.mode == "chroma_only":
                s_out = self.apply_chroma_only()
            else:
                # "nlm" or any unrecognised value → NLM (default, original behaviour)
                s_out = self.apply_nlm()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = s_out

        self.save()
        return self.img
