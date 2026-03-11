"""
modules/hdr_tone_mapping/hdr_tone_mapping.py
=============================================
HDRToneMapping — global tone mapping for correct PQ / HLG output.

Pipeline position: between 2D Noise Reduction and Gamma EOTF.

  ... → 2D NR (linear) → [HDRToneMapping] → Gamma EOTF → CSC → ...

Purpose
-------
The Gamma module already supports PQ (ST.2084) and HLG EOTF encoding.
However, a correct HDR output requires that the *input* to those EOTFs is
in the right luminance domain:

  PQ  — expects absolute scene luminance in [0, 10000] nits, normalised to [0, 1].
          A pixel value of 1.0 → 10000 nits on a BT.2100 display.
  HLG — scene-referred; expects normalised linear signal [0, 1] that is
          relative to display peak.  A light highlight rolloff handles
          slight over-exposures.

When the output profile is sRGB / Rec.709 (SDR), tone mapping is optional
but useful for scenes with high dynamic range (e.g. outdoor direct sunlight).

Modes
-----
  "reinhard"     — global Reinhard (2002), log-average key adaptation
  "reinhard_ext" — extended Reinhard with automatic scene-white estimation
  "aces"         — ACES RRT/ODT approximation (Narkowicz 2015 fit)
  "hable"        — Hable / Uncharted 2 filmic tone mapping curve
  "hlg_rolloff"  — gentle highlight knee only (for HLG output, no recompression)

Absolute Luminance (optional)
------------------------------
Set `absolute_luminance: true` in config plus ISO / shutter / aperture to
convert the normalised linear values to scene nits before tone mapping.
The formula follows ISO 12232:
  L_nits = pixel_value × (calibration_k × aperture_f²) / (iso × shutter_sec)
where calibration_k ≈ 12.5 cd·s/(m²·ISO) for reflected light.

Config
------
  hdr_tone_mapping:
    is_enable: false
    mode: "reinhard_ext"       # see Modes above
    key: 0.18                  # Reinhard key (only used in reinhard / reinhard_ext)
    white_percentile: 99.0     # scene-white percentile (reinhard_ext)
    exposure_bias: 2.0         # Hable pre-exposure multiplier
    peak_nits: 1000.0          # display peak luminance for absolute-nit normalisation
    absolute_luminance: false  # enable ISO/shutter/aperture → nit conversion
    iso: 0                     # 0 = read from sensor_info["iso"] (if present)
    shutter_ms: 0              # 0 = read from sensor_info["shutter_ms"] (if present)
    aperture: 0.0              # 0 = read from sensor_info["aperture"] (if present)
    calibration_k: 12.5        # ISO 12232 camera calibration constant

Author: Omni-ISP contributors
"""

import warnings
import numpy as np

from modules.hdr_tone_mapping.operators import (
    reinhard,
    reinhard_ext,
    aces,
    hable,
    highlight_rolloff,
    scene_to_absolute_nits,
)


class HDRToneMapping:
    """
    HDR Tone Mapping module.

    Parameters
    ----------
    img         : np.ndarray  (H, W, 3) float32 linear RGB, normalised [0, 1]
                  (output of 2D NR / DL denoising stage)
    platform    : dict — pipeline platform config
    sensor_info : dict — sensor metadata (iso, shutter_ms, aperture, bit_depth)
    parm_hdrtm  : dict — hdr_tone_mapping section from configs.yml
    """

    def __init__(self, img, platform, sensor_info, parm_hdrtm):
        self.img         = img
        self.platform    = platform
        self.sensor_info = sensor_info

        self.enable            = parm_hdrtm.get("is_enable", False)
        self.mode              = parm_hdrtm.get("mode", "reinhard_ext")
        self.key               = float(parm_hdrtm.get("key", 0.18))
        self.white_percentile  = float(parm_hdrtm.get("white_percentile", 99.0))
        self.exposure_bias     = float(parm_hdrtm.get("exposure_bias", 2.0))
        self.peak_nits         = float(parm_hdrtm.get("peak_nits", 1000.0))
        self.abs_lum_enable    = parm_hdrtm.get("absolute_luminance", False)
        self.calibration_k     = float(parm_hdrtm.get("calibration_k", 12.5))

        # Exposure parameters — prefer explicit config, fall back to sensor_info
        def _resolve(key_cfg, key_si, scale=1.0):
            v = float(parm_hdrtm.get(key_cfg, 0))
            if v <= 0:
                v = float(sensor_info.get(key_si, 0)) * scale
            return v

        self.iso         = _resolve("iso",         "iso",        1.0)
        self.shutter_sec = _resolve("shutter_ms",  "shutter_ms", 1e-3)  # ms → s
        self.aperture_f  = _resolve("aperture",    "aperture",   1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self) -> np.ndarray:
        """
        Apply tone mapping.

        Returns
        -------
        np.ndarray  (H, W, 3) float32 in [0, 1]
        If disabled, returns img unchanged.
        """
        if not self.enable:
            return self.img

        img = self.img.astype(np.float32)

        # ------------------------------------------------------------------
        # Optional: absolute luminance scaling
        # Converts normalised [0,1] → scene nits, then normalises to
        # [0, 1] relative to peak display luminance so tone mapping
        # operators see properly scaled scene content.
        # ------------------------------------------------------------------
        if self.abs_lum_enable:
            img = self._apply_absolute_luminance(img)

        # ------------------------------------------------------------------
        # Tone mapping dispatch
        # ------------------------------------------------------------------
        mode = self.mode.lower()
        if mode == "reinhard":
            result = reinhard(
                img,
                key=self.key,
                white_percentile=self.white_percentile,
            )
        elif mode == "reinhard_ext":
            result = reinhard_ext(
                img,
                key=self.key,
                white_percentile=self.white_percentile,
            )
        elif mode == "aces":
            result = aces(img)
        elif mode == "hable":
            result = hable(img, exposure_bias=self.exposure_bias)
        elif mode == "hlg_rolloff":
            result = highlight_rolloff(img)
        else:
            warnings.warn(
                f"HDRToneMapping: unknown mode '{self.mode}'. Passing through.",
                RuntimeWarning,
                stacklevel=2,
            )
            result = np.clip(img, 0.0, 1.0)

        return result.astype(np.float32)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_absolute_luminance(self, img: np.ndarray) -> np.ndarray:
        """
        Convert normalised linear RGB to scene nits, then normalise by
        peak display luminance so the tone mapper receives [0, ≥1] input
        where 1.0 = display peak.

        Falls back gracefully if exposure metadata is missing.
        """
        if self.iso <= 0 or self.shutter_sec <= 0 or self.aperture_f <= 0:
            warnings.warn(
                "HDRToneMapping: absolute_luminance=true but exposure metadata "
                f"(iso={self.iso}, shutter_sec={self.shutter_sec:.4f}, "
                f"aperture_f={self.aperture_f}) is not fully specified. "
                "Skipping absolute luminance conversion.",
                RuntimeWarning,
                stacklevel=3,
            )
            return img

        # Convert to scene nits
        img_nits = scene_to_absolute_nits(
            img,
            iso=self.iso,
            shutter_sec=self.shutter_sec,
            aperture_f=self.aperture_f,
            calibration_k=self.calibration_k,
        )

        # Normalise so that peak_nits → 1.0
        return img_nits / self.peak_nits
