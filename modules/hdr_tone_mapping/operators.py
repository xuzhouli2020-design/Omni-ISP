"""
modules/hdr_tone_mapping/operators.py
======================================
Global tone mapping operators for HDR → display mapping.

All operators accept a float32 linear RGB image (H, W, 3) with values that
may exceed 1.0 (scene-linear, possibly in absolute nit units after absolute
luminance scaling).  They return float32 linear RGB in [0, 1].

Operators
---------
  reinhard(rgb)           — global Reinhard (2002), log-average key adaptation
  reinhard_ext(rgb)       — extended Reinhard, preserves specular highlights
  aces(rgb)               — ACES RRT/ODT approximation (Narkowicz 2015)
  hable(rgb)              — Hable "Uncharted 2" filmic (2010)
  highlight_rolloff(rgb)  — soft knee for SDR highlight protection (HLG mode)

References
----------
  Reinhard et al. 2002 — "Photographic Tone Reproduction for Digital Images"
  Narkowicz 2015 — "ACES Filmic Tone Mapping Curve"
  Hable 2010 — "Filmic Tonemapping for Real-Time Rendering" (GDC slides)

Author: Omni-ISP contributors
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _luminance(rgb: np.ndarray) -> np.ndarray:
    """BT.709 luminance from (H, W, 3) float32 linear RGB."""
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _scale_by_luminance_ratio(rgb: np.ndarray, L_in: np.ndarray,
                               L_out: np.ndarray) -> np.ndarray:
    """
    Scale RGB channels by L_out / L_in, preserving chrominance.
    Avoids artefacts from per-channel tone mapping.
    """
    ratio = L_out / np.maximum(L_in, 1e-8)
    return np.clip(rgb * ratio[..., None], 0.0, 1.0)


def _log_average_luminance(L: np.ndarray, delta: float = 1e-4) -> float:
    """Geometric mean of scene luminance (Reinhard 2002 Eq. 1)."""
    return float(np.exp(np.mean(np.log(L + delta))))


# ---------------------------------------------------------------------------
# Reinhard 2002 — global operator
# ---------------------------------------------------------------------------

def reinhard(
    rgb: np.ndarray,
    key: float = 0.18,
    white_percentile: float = 99.0,
) -> np.ndarray:
    """
    Global Reinhard tone mapping (luminance domain).

    Steps
    -----
    1. Compute log-average scene luminance L_avg (Eq. 1).
    2. Scale scene luminance so key value maps to 0.18 (Eq. 2).
    3. Apply L_d = L_s / (1 + L_s)  (simple Reinhard, Eq. 4).
    4. Recover RGB by scaling channels proportionally.

    Parameters
    ----------
    rgb             : float32 (H, W, 3), linear, may exceed 1.0
    key             : scene key — 0.18 (mid-tone), lower = darker, higher = brighter
    white_percentile: auto-determine scene white from this percentile of luminance

    Returns
    -------
    float32 (H, W, 3) in [0, 1]
    """
    L_in  = _luminance(rgb)
    L_avg = _log_average_luminance(L_in)

    # Map scene luminance so that the "key" grey tone lands at key value
    L_s = (key / L_avg) * L_in

    # Simple Reinhard without white-point clipping
    L_d = L_s / (1.0 + L_s)

    return _scale_by_luminance_ratio(rgb, L_in, L_d)


# ---------------------------------------------------------------------------
# Extended Reinhard — specular highlight preservation
# ---------------------------------------------------------------------------

def reinhard_ext(
    rgb: np.ndarray,
    key: float = 0.18,
    white_percentile: float = 99.0,
) -> np.ndarray:
    """
    Extended Reinhard tone mapping (Reinhard 2002, Eq. 4 with white point).

      L_d = L_s * (1 + L_s / L_w²) / (1 + L_s)

    where L_w = scene white (the brightest reproducible scene luminance).
    Values at or above L_w map to 1.0.  Below L_w the curve is slightly
    brighter than plain Reinhard.

    Parameters
    ----------
    rgb             : float32 (H, W, 3)
    key             : scene key value (default 0.18)
    white_percentile: percentile of scaled luminance used to set L_w
    """
    L_in  = _luminance(rgb)
    L_avg = _log_average_luminance(L_in)
    L_s   = (key / L_avg) * L_in

    # Determine scene white from percentile of scaled luminance
    L_w = float(np.percentile(L_s, white_percentile))
    L_w = max(L_w, 1e-3)   # guard against pathological inputs

    # Extended Reinhard
    L_d = (L_s * (1.0 + L_s / (L_w ** 2))) / (1.0 + L_s)

    return _scale_by_luminance_ratio(rgb, L_in, L_d)


# ---------------------------------------------------------------------------
# ACES RRT/ODT approximation — Narkowicz 2015
# ---------------------------------------------------------------------------

def aces(rgb: np.ndarray) -> np.ndarray:
    """
    ACES filmic tone mapping curve (Narkowicz 2015 fit to ACES RRT+ODT).

      out = (x * (2.51x + 0.03)) / (x * (2.43x + 0.59) + 0.14)

    Applied per-channel (the original curve is designed for scene-linear [0,1]
    but extends gracefully above 1.0).  No pre-exposure is applied here; if
    the input is in absolute nit units, normalise first with
    `rgb / peak_nits` before calling.

    Reference: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    """
    x = rgb.astype(np.float32)
    result = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Hable / Uncharted 2 filmic — John Hable 2010
# ---------------------------------------------------------------------------

# Uncharted 2 curve constants
_U2_A = 0.15   # Shoulder strength
_U2_B = 0.50   # Linear strength
_U2_C = 0.10   # Linear angle
_U2_D = 0.20   # Toe strength
_U2_E = 0.02   # Toe numerator
_U2_F = 0.30   # Toe denominator
_U2_W = 11.2   # Linear white point


def _u2_fn(x: np.ndarray) -> np.ndarray:
    """Uncharted 2 curve function (applied to a single exposure-biased array)."""
    return (
        (x * (_U2_A * x + _U2_C * _U2_B) + _U2_D * _U2_E)
        / (x * (_U2_A * x + _U2_B) + _U2_D * _U2_F)
    ) - _U2_E / _U2_F


def hable(rgb: np.ndarray, exposure_bias: float = 2.0) -> np.ndarray:
    """
    Hable / Uncharted 2 filmic tone mapping.

    Produces a classic "filmic look": crushed blacks, lifted shadows, strong
    but smooth highlight rolloff, and a slight contrast S-curve.

    Parameters
    ----------
    rgb           : float32 (H, W, 3), linear, may exceed 1.0
    exposure_bias : pre-exposure multiplier (default 2.0 as per original GDC slides)

    Returns
    -------
    float32 (H, W, 3) in [0, 1]
    """
    x = rgb.astype(np.float32)
    biased  = _u2_fn(x * exposure_bias)
    white_s = float(_u2_fn(np.array([_U2_W], dtype=np.float32))[0])
    result  = biased / max(white_s, 1e-8)
    return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Highlight rolloff — gentle knee for HLG / SDR highlight protection
# ---------------------------------------------------------------------------

def highlight_rolloff(
    rgb: np.ndarray,
    knee_start: float = 0.85,
    knee_end: float   = 1.10,
) -> np.ndarray:
    """
    Soft highlight rolloff knee (no tone compression below knee_start).

    Values below knee_start pass through unchanged.
    Values between knee_start and knee_end are compressed with a cubic ease.
    Values above knee_end are clamped to 1.0.

    This is not a full tone mapper — it is a light-handed highlight limiter
    suitable for use in HLG mode, where the scene-referred signal may have
    small excursions above 1.0 that need to be brought in gracefully.

    Parameters
    ----------
    rgb        : float32 (H, W, 3)
    knee_start : fraction where rolloff begins (must be < knee_end)
    knee_end   : fraction where signal is fully limited to 1.0
    """
    out = rgb.astype(np.float32).copy()
    knee_range = knee_end - knee_start

    if knee_range <= 0:
        return np.clip(out, 0.0, 1.0)

    # Mask of pixels that need rolloff (any channel above knee_start)
    L = _luminance(out)
    above = L > knee_start

    if not np.any(above):
        return np.clip(out, 0.0, 1.0)

    # t ∈ [0, 1] over the knee region
    t = np.clip((L - knee_start) / knee_range, 0.0, 1.0)
    # Smooth cubic ease-in-out: 3t² - 2t³
    t_smooth = t * t * (3.0 - 2.0 * t)
    # Blended luminance: linear below knee, compressed above
    L_knee = knee_start + (1.0 - knee_start) * (1.0 - (1.0 - t_smooth) ** 0.5)
    L_out = np.where(above, L_knee, L)
    L_out = np.clip(L_out, 0.0, 1.0)

    return _scale_by_luminance_ratio(rgb, L, L_out)


# ---------------------------------------------------------------------------
# Absolute luminance scaling
# ---------------------------------------------------------------------------

def scene_to_absolute_nits(
    rgb_norm: np.ndarray,
    iso: float,
    shutter_sec: float,
    aperture_f: float,
    calibration_k: float = 12.5,
) -> np.ndarray:
    """
    Convert normalised linear RGB [0, 1] to scene luminance in nits (cd/m²).

    Formula (ISO 12232 exposure equation):
      L_nits = (pixel_value × calibration_k) / (iso × shutter_sec / aperture_f²)

    Equivalently:
      L_nits = pixel_value × (calibration_k × aperture_f²) / (iso × shutter_sec)

    Parameters
    ----------
    rgb_norm      : float32 (H, W, 3), linear, normalised [0, 1]
    iso           : ISO speed rating (e.g. 100, 400, 3200)
    shutter_sec   : exposure time in seconds (e.g. 1/100 → 0.01)
    aperture_f    : f-number (e.g. f/2.8 → 2.8)
    calibration_k : camera calibration constant, default 12.5 cd·s/(m²·ISO)

    Returns
    -------
    float32 (H, W, 3), values in nits (cd/m²), unbounded above
    """
    if iso <= 0 or shutter_sec <= 0 or aperture_f <= 0:
        raise ValueError(
            f"Absolute luminance requires iso>0, shutter_sec>0, aperture_f>0. "
            f"Got iso={iso}, shutter_sec={shutter_sec}, aperture_f={aperture_f}"
        )
    scale = (calibration_k * aperture_f ** 2) / (iso * shutter_sec)
    return rgb_norm.astype(np.float32) * scale
