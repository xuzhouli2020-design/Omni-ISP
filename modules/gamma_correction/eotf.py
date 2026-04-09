"""
File: eotf.py
Description: Electro-Optical Transfer Functions (gamma / tone curves) for the
             Gamma Correction module.

Supported EOTFs
---------------
  "lut"     — Use per-bit-depth LUT from config (original behaviour, default).
  "srgb"    — IEC 61966-2-1 sRGB piecewise transfer function (SDR, ~100 nit displays).
  "rec709"  — ITU-R BT.709 broadcast SDR transfer function.
  "linear"  — γ = 1.0, pass-through (for scientific output or ML pipelines).
  "pq"      — SMPTE ST.2084 / PQ for HDR10 (absolute, 0–10 000 nit displays).
  "hlg"     — ARIB STD-B67 / HLG for HDR broadcast (scene-referred, backward compat).

All functions take a float32 array normalised to [0, 1] (linear light) and return
a float32 array in [0, 1] (encoded signal).  Clipping to [0, 1] is the caller's
responsibility.

Ref: docs/dev_notes.md "CCM target primaries + Gamma EOTF — multi-format output"

Author: Omni-ISP / 10xEngineers Pvt Ltd
"""
import numpy as np


def apply_srgb(L: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 sRGB piecewise EOTF (forward, linear → encoded).

    Exact piecewise formula (more accurate than simple γ = 2.2 power law):
      V = 12.92 × L              for L ≤ 0.0031308
      V = 1.055 × L^(1/2.4) - 0.055  for L >  0.0031308
    """
    L = np.asarray(L, dtype=np.float64)
    out = np.where(
        L <= 0.0031308,
        12.92 * L,
        1.055 * np.power(np.maximum(L, 0.0), 1.0 / 2.4) - 0.055,
    )
    return np.float32(out)


def apply_rec709(L: np.ndarray) -> np.ndarray:
    """ITU-R BT.709 transfer function (SDR broadcast, forward).

      V = 4.5 × L                for L < 0.018
      V = 1.099 × L^0.45 - 0.099  for L ≥ 0.018
    """
    L = np.asarray(L, dtype=np.float64)
    out = np.where(
        L < 0.018,
        4.5 * L,
        1.099 * np.power(np.maximum(L, 0.0), 0.45) - 0.099,
    )
    return np.float32(out)


def apply_linear(L: np.ndarray) -> np.ndarray:
    """γ = 1.0 — linear light passthrough.  For scientific output or ML input."""
    return np.float32(L)


def apply_pq(L: np.ndarray) -> np.ndarray:
    """SMPTE ST.2084 / Perceptual Quantiser (PQ) HDR EOTF (forward, absolute nits).

    Input L is normalised to [0, 1] relative to 10 000 nit peak (so L=1.0 = 10 000 nits).
    Output is the PQ-encoded signal V in [0, 1].

    Constants from SMPTE ST.2084-2014:
      m1 = 2610 / 4096 / 4  ≈ 0.1593017578125
      m2 = 2523 / 4096 × 128 ≈ 78.84375
      c1 = 3424 / 4096       ≈ 0.8359375
      c2 = 2413 / 4096 × 32  ≈ 18.8515625
      c3 = 2392 / 4096 × 32  ≈ 18.6875
    """
    L = np.asarray(L, dtype=np.float64)
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875
    Lp = np.power(np.maximum(L, 0.0), m1)
    V = np.power((c1 + c2 * Lp) / (1.0 + c3 * Lp), m2)
    return np.float32(V)


def apply_hlg(L: np.ndarray) -> np.ndarray:
    """ARIB STD-B67 / Hybrid Log-Gamma (HLG) HDR EOTF (forward, scene-referred).

    Input L is normalised linear scene luminance [0, 1].
    Output is the HLG-encoded signal E in [0, 1].

      E = sqrt(3L)             for  L ≤ 1/12  (SDR-compatible square-root portion)
      E = a × ln(12L - b) + c  for  L >  1/12  (log portion)

    Constants (ARIB STD-B67):
      a = 0.17883277, b = 0.28466892, c = 0.55991073
    """
    L = np.asarray(L, dtype=np.float64)
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073
    # Ensure no negative log argument
    safe_L = np.maximum(L, 0.0)
    out = np.where(
        L <= 1.0 / 12.0,
        np.sqrt(3.0 * safe_L),
        a * np.log(np.maximum(12.0 * safe_L - b, 1e-10)) + c,
    )
    return np.float32(out)


# Registry: map eotf name → function(L: ndarray[float32, 0-1]) → ndarray[float32, 0-1]
EOTF_REGISTRY = {
    "srgb": apply_srgb,
    "rec709": apply_rec709,
    "linear": apply_linear,
    "pq": apply_pq,
    "hlg": apply_hlg,
}


def apply_eotf(img: np.ndarray, eotf_name: str, bit_depth: int) -> np.ndarray:
    """Apply the selected EOTF to an image and return uint8 [0,255] output.

    Parameters
    ----------
    img        Input image array — any numeric dtype; values are normalised
               to [0, 1] relative to the full-scale of the array before the
               transfer function is applied.
    eotf_name  One of "srgb" | "rec709" | "linear" | "pq" | "hlg".
    bit_depth  Sensor bit depth — used for normalising integer inputs.

    Returns
    -------
    uint8 array in [0, 255].  HDR EOTFs (pq, hlg) still return uint8 here —
    full 10/12-bit HDR output encoding is deferred to a future output module.
    """
    fn = EOTF_REGISTRY.get(eotf_name)
    if fn is None:
        raise ValueError(
            f"Unknown EOTF '{eotf_name}'. "
            f"Choose one of: {sorted(EOTF_REGISTRY)}"
        )

    # Normalise input to [0, 1] using full-scale bit depth value
    img_f = img.astype(np.float64)
    full_scale = float((1 << bit_depth) - 1)
    if full_scale <= 0.0:
        return np.zeros_like(img, dtype=np.uint8)

    img_norm = np.clip(img_f / full_scale, 0.0, 1.0).astype(np.float32)

    # Apply per-channel EOTF
    out_norm = fn(img_norm)
    out_norm = np.clip(out_norm, 0.0, 1.0)

    # Scale to [0, 255] for uint8 output
    return np.uint8(np.round(out_norm * 255.0))
