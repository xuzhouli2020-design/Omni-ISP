"""
File: warp_field.py
Description: Per-channel warp field computation for unified CAC+LDC correction.

Both Chromatic Aberration Correction (CAC) and Lens Distortion Correction (LDC)
are coordinate-remapping operations (warpers).  Because they are both warps they
can be composed analytically into a single per-channel source-coordinate map,
requiring only ONE bilinear resample per sub-channel.

This avoids the quality loss of cascaded resampling (LDC then CAC separately
would require two bilinear passes, each introducing blur).

Warp model per sub-channel
─────────────────────────
For each output pixel at sub-channel position (r_out, c_out):

  Step 1 — LDC (Brown-Conrady undistortion):
    Normalize:  xn = (c_out − cx) / f
                yn = (r_out − cy) / f
    r² = xn² + yn²
    radial  = 1 + k1·r² + k2·r⁴ + k3·r⁶
    xd = xn·radial + 2·p1·xn·yn + p2·(r² + 2·xn²)
    yd = yn·radial + p1·(r² + 2·yn²) + 2·p2·xn·yn
    → source position after LDC: (ys_ldc, xs_ldc) = (yd·f+cy, xd·f+cx)

  Step 2 — CAC (radial scale differential, applied at the LDC position):
    dx = xs_ldc − cx,   dy = ys_ldc − cy
    r_ca = √(dx²+dy²) / R_norm          (normalised by half-diagonal)
    ca_scale = ca_k1·r_ca + ca_k2·r_ca² + ca_k3·r_ca³
    xs_final = cx + dx·(1 + ca_scale)
    ys_final = cy + dy·(1 + ca_scale)

  For G sub-channels (Gr, Gb): ca_scale = 0  →  xs_final = xs_ldc
  For R and B:                  ca_scale per-channel from config

The final (ys_final, xs_final) is the source coordinate to bilinear-sample in
the corresponding distorted input sub-channel.

Normalisation convention
────────────────────────
  f        = focal_length_px / 2  (scaled to sub-channel resolution)
             If focal_length_px ≤ 0: use image diagonal/2 of the sub-channel.
  R_norm   = half-diagonal of the sub-channel  (independent of f choice)
             Ensures the CA polynomial coefficients are always in [0, 1] range
             and not tied to the LDC focal length.

Author: Omni-ISP contributors
"""

import numpy as np

try:
    from scipy.ndimage import map_coordinates as _scipy_map
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _half_diagonal(H_sub: int, W_sub: int) -> float:
    """Half-diagonal of a sub-channel image — used as normalisation radius."""
    return float(np.sqrt((W_sub * 0.5) ** 2 + (H_sub * 0.5) ** 2))


def _focal_length(ldc_params: dict, H_sub: int, W_sub: int) -> float:
    """
    Resolve effective focal length in sub-channel pixels.
    If the config value is zero / absent, fall back to half-diagonal.
    """
    f_cfg = float(ldc_params.get("focal_length_px", 0.0))
    # Config stores focal_length_px in full-resolution pixels; halve for sub-channel
    f_sub = f_cfg / 2.0 if f_cfg > 0 else None
    return f_sub if (f_sub and f_sub > 0) else _half_diagonal(H_sub, W_sub)


def _principal_point(cfg_center, H_sub: int, W_sub: int) -> tuple:
    """
    Resolve principal point in sub-channel pixel coordinates.
    'auto' → image centre.  [cx, cy] list → cx/2, cy/2.
    """
    if cfg_center == "auto" or cfg_center is None:
        return (W_sub - 1) * 0.5, (H_sub - 1) * 0.5   # (cx, cy)
    # User supplied [cx_full, cy_full] in full-resolution pixels
    cx_full, cy_full = float(cfg_center[0]), float(cfg_center[1])
    return cx_full / 2.0, cy_full / 2.0


# ---------------------------------------------------------------------------
# Core: build per-channel warp field
# ---------------------------------------------------------------------------

def build_warp_field(
    H_sub: int,
    W_sub: int,
    ldc_params: dict,
    ca_delta: tuple,          # (ca_k1, ca_k2, ca_k3) for this channel; (0,0,0) for G
) -> tuple:
    """
    Compute source coordinate arrays for one sub-channel.

    Parameters
    ----------
    H_sub, W_sub  : sub-channel shape (height, width) — i.e. full_H//2, full_W//2
    ldc_params    : dict with keys k1, k2, k3, p1, p2, focal_length_px, center
    ca_delta      : (ca_k1, ca_k2, ca_k3) — CA radial scale polynomial for this
                    channel.  Pass (0, 0, 0) for Gr / Gb.

    Returns
    -------
    rows_src, cols_src : np.ndarray of shape (H_sub, W_sub), dtype float32
        For each output pixel (r, c), sample the input at (rows_src[r,c], cols_src[r,c]).
    """
    # ── Resolve geometry ──────────────────────────────────────────────────────
    cx, cy = _principal_point(ldc_params.get("center", "auto"), H_sub, W_sub)
    f       = _focal_length(ldc_params, H_sub, W_sub)
    R_norm  = _half_diagonal(H_sub, W_sub)   # CA normalisation radius

    # ── Output grid (all sub-channel pixel centres) ────────────────────────
    cols = np.arange(W_sub, dtype=np.float64)
    rows = np.arange(H_sub, dtype=np.float64)
    col_grid, row_grid = np.meshgrid(cols, rows)   # (H_sub, W_sub)

    # ── LDC: Brown-Conrady undistortion ────────────────────────────────────
    k1 = float(ldc_params.get("k1", 0.0))
    k2 = float(ldc_params.get("k2", 0.0))
    k3 = float(ldc_params.get("k3", 0.0))
    p1 = float(ldc_params.get("p1", 0.0))
    p2 = float(ldc_params.get("p2", 0.0))

    xn = (col_grid - cx) / f
    yn = (row_grid - cy) / f
    r2 = xn * xn + yn * yn

    if k1 != 0 or k2 != 0 or k3 != 0 or p1 != 0 or p2 != 0:
        r4      = r2 * r2
        r6      = r4 * r2
        radial  = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
    else:
        # LDC disabled — identity
        xd, yd = xn, yn

    # Back to sub-channel pixel coordinates
    xs_ldc = xd * f + cx
    ys_ldc = yd * f + cy

    # ── CAC: differential radial scale applied at LDC position ─────────────
    ca_k1, ca_k2, ca_k3 = float(ca_delta[0]), float(ca_delta[1]), float(ca_delta[2])

    if ca_k1 != 0 or ca_k2 != 0 or ca_k3 != 0:
        dx     = xs_ldc - cx
        dy     = ys_ldc - cy
        r_ca   = np.sqrt(dx * dx + dy * dy) / R_norm    # normalised to [0, ~1]
        ca_sc  = ca_k1 * r_ca + ca_k2 * (r_ca * r_ca) + ca_k3 * (r_ca ** 3)
        xs_out = cx + dx * (1.0 + ca_sc)
        ys_out = cy + dy * (1.0 + ca_sc)
    else:
        xs_out = xs_ldc
        ys_out = ys_ldc

    return ys_out.astype(np.float32), xs_out.astype(np.float32)


# ---------------------------------------------------------------------------
# Bilinear remap
# ---------------------------------------------------------------------------

def remap_channel(
    sub_ch: np.ndarray,
    rows_src: np.ndarray,
    cols_src: np.ndarray,
) -> np.ndarray:
    """
    Bilinear remap of a 2-D sub-channel image.

    Parameters
    ----------
    sub_ch            : (H, W) float32 sub-channel image
    rows_src, cols_src: (H, W) float32 source coordinates

    Returns
    -------
    (H, W) float32 remapped image, same range as input.
    Border pixels clamped (mode='nearest') — no black edge artifacts.
    """
    if _HAVE_SCIPY:
        coords = np.stack([rows_src, cols_src], axis=0).astype(np.float64)
        return _scipy_map(
            sub_ch.astype(np.float64),
            coords,
            order=1,          # bilinear
            mode="nearest",   # clamp at borders
            prefilter=False,
        ).astype(np.float32)

    # Pure-NumPy bilinear fallback (no scipy)
    return _numpy_bilinear(sub_ch, rows_src, cols_src)


def _numpy_bilinear(img: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """
    Pure-NumPy bilinear interpolation.
    img : (H, W) float32
    ys, xs : (H, W) float32 source coordinates
    """
    H, W = img.shape

    # Clamp coordinates to valid range
    xs_c = np.clip(xs, 0, W - 1)
    ys_c = np.clip(ys, 0, H - 1)

    x0 = np.floor(xs_c).astype(np.int32)
    y0 = np.floor(ys_c).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)

    wx = (xs_c - x0).astype(np.float32)   # fractional x weight
    wy = (ys_c - y0).astype(np.float32)   # fractional y weight

    # Four corner samples
    v00 = img[y0, x0]
    v10 = img[y0, x1]
    v01 = img[y1, x0]
    v11 = img[y1, x1]

    # Bilinear blend
    return (v00 * (1 - wx) * (1 - wy)
            + v10 * wx       * (1 - wy)
            + v01 * (1 - wx) * wy
            + v11 * wx       * wy)
