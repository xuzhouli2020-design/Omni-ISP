"""
File: burst_registration.py
Description: Phase-correlation frame registration + Bayer-aware sub-pixel shift.

Algorithm
─────────
1. Extract the green channel (Gr or average of Gr+Gb) from each Bayer frame.
2. Compute FFT cross-power spectrum between reference green and target green.
3. Peak of the inverse FFT gives integer pixel translation.
4. Parabolic sub-pixel refinement around the peak for sub-pixel accuracy.
5. Apply the estimated (dy, dx) shift to each Bayer sub-channel (R, Gr, Gb, B)
   at half the full-resolution shift (sub-images are already 2× downsampled).

Why Bayer-aware shifting?
   A full-frame 2-pixel shift moves the CFA by 1 Bayer period — sub-channels
   must each shift by shift/2.  Non-integer shifts use scipy.ndimage.shift
   if available, otherwise a custom bilinear shift via numpy.

Limitations
───────────
- Translation only (no rotation/perspective). Adequate for wearable handheld
  burst of 4–8 frames at short exposures with small hand motion.
- Future upgrade: homography (8-DOF) using feature matching on the G channel.

Phase 5.2
"""

import numpy as np
from util.bayer_utils import extract_channels, reconstruct_bayer


# ---------------------------------------------------------------------------
# Phase correlation helpers
# ---------------------------------------------------------------------------

def _green_channel(raw: np.ndarray, bayer_pattern: str) -> np.ndarray:
    """Extract and average the two green sub-channels."""
    chs = extract_channels(raw, bayer_pattern)
    return (chs["gr"].astype(np.float32) + chs["gb"].astype(np.float32)) * 0.5


def _parabolic_peak(cc: np.ndarray, peak: tuple) -> tuple:
    """
    Sub-pixel refinement of the cross-correlation peak using a 1-D parabolic
    fit along each axis independently.

    Returns (dy, dx) as floats.
    """
    H, W = cc.shape
    y0, x0 = peak

    def _refine_1d(vals_prev, val_peak, vals_next):
        """Fit parabola to 3 points, return offset from centre."""
        denom = 2.0 * (vals_prev - 2.0 * val_peak + vals_next)
        if abs(denom) < 1e-12:
            return 0.0
        return (vals_prev - vals_next) / denom

    # Wrap-around neighbours (cross-correlation is circular)
    y_prev = cc[(y0 - 1) % H, x0]
    y_next = cc[(y0 + 1) % H, x0]
    x_prev = cc[y0, (x0 - 1) % W]
    x_next = cc[y0, (x0 + 1) % W]

    dy_sub = _refine_1d(y_prev, cc[y0, x0], y_next)
    dx_sub = _refine_1d(x_prev, cc[y0, x0], x_next)
    return (y0 + dy_sub, x0 + dx_sub)


def phase_correlate(ref: np.ndarray, target: np.ndarray,
                    upsample: int = 1) -> tuple:
    """
    Estimate the (dy, dx) translation that maps ``target`` onto ``ref``.

    Parameters
    ----------
    ref, target : (H, W) float32  Green channel sub-images (half-res).
    upsample    : int              Reserved for future DFT up-sampling refinement.

    Returns
    -------
    (dy, dx) : tuple of float  Shift of target relative to ref (pixels in
                               the half-resolution G channel space).
    """
    eps = 1e-8
    H, W = ref.shape

    F_ref = np.fft.fft2(ref)
    F_tar = np.fft.fft2(target)

    cross     = F_ref * np.conj(F_tar)
    magnitude = np.abs(cross) + eps
    norm_cross = cross / magnitude

    cc = np.real(np.fft.ifft2(norm_cross))

    # Integer peak
    peak_flat = int(np.argmax(cc))
    y0 = peak_flat // W
    x0 = peak_flat % W

    # Sub-pixel via parabolic fit
    dy_sub, dx_sub = _parabolic_peak(cc, (y0, x0))

    # Map shift to [-H/2, H/2) convention (unwrap circular)
    dy = dy_sub if dy_sub <= H / 2 else dy_sub - H
    dx = dx_sub if dx_sub <= W / 2 else dx_sub - W

    return float(dy), float(dx)


# ---------------------------------------------------------------------------
# Bayer-aware shift
# ---------------------------------------------------------------------------

def _shift_channel(ch: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """
    Shift a 2D float32 array by (dy, dx) pixels using bilinear interpolation.

    Uses scipy.ndimage.shift if available (accurate, anti-aliased).
    Falls back to a pure NumPy integer-rounded shift otherwise.
    """
    try:
        from scipy.ndimage import shift as ndimage_shift
        shifted = ndimage_shift(
            ch.astype(np.float64),
            shift=(dy, dx),
            mode="nearest",
            order=1,
        )
        return shifted.astype(np.float32)
    except ImportError:
        # Integer fallback — crops and zero-pads
        return _shift_integer(ch, int(round(dy)), int(round(dx)))


def _shift_integer(ch: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Simple integer-pixel shift with edge replication."""
    H, W = ch.shape
    out = np.empty_like(ch)
    # Source window in `ch` and destination window in `out`
    src_y0 = max(0,  dy);  src_y1 = min(H, H + dy)
    src_x0 = max(0,  dx);  src_x1 = min(W, W + dx)
    dst_y0 = max(0, -dy);  dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = max(0, -dx);  dst_x1 = dst_x0 + (src_x1 - src_x0)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = ch[src_y0:src_y1, src_x0:src_x1]
    # Fill borders via edge replication
    if dst_y0 > 0:
        out[:dst_y0, :]  = out[dst_y0, :]
    if dst_y1 < H:
        out[dst_y1:, :]  = out[dst_y1-1, :]
    if dst_x0 > 0:
        out[:, :dst_x0]  = out[:, dst_x0:dst_x0+1]
    if dst_x1 < W:
        out[:, dst_x1:]  = out[:, dst_x1-1:dst_x1]
    return out


def apply_shift_bayer(frame: np.ndarray, dy: float, dx: float,
                      bayer_pattern: str) -> np.ndarray:
    """
    Apply a full-frame (dy, dx) translation to a Bayer image.

    Because each Bayer sub-channel (R, Gr, Gb, B) is half-resolution,
    the shift in sub-channel space is (dy/2, dx/2).

    Parameters
    ----------
    frame          : (H, W) uint16  Bayer raw frame.
    dy, dx         : float          Full-frame pixel shift.
    bayer_pattern  : str            Bayer pattern string.

    Returns
    -------
    (H, W) uint16  Shifted Bayer frame (same dtype as input).
    """
    channels = extract_channels(frame, bayer_pattern)

    # Sub-channel shift = half of full-frame shift
    dy_sub = dy / 2.0
    dx_sub = dx / 2.0

    shifted = {}
    for name, ch in channels.items():
        ch_f = ch.astype(np.float32)
        shifted[name] = _shift_channel(ch_f, dy_sub, dx_sub)

    # Reconstruct — clip and round back to uint16
    maxval = float((1 << 16) - 1)
    for k in shifted:
        shifted[k] = np.clip(np.round(shifted[k]), 0, maxval).astype(np.uint16)

    return reconstruct_bayer(shifted, bayer_pattern)


# ---------------------------------------------------------------------------
# High-level registration function
# ---------------------------------------------------------------------------

def register_stack(stack: np.ndarray, bayer_pattern: str,
                   ref_idx: int = 0) -> tuple:
    """
    Register all frames in a burst stack to the reference frame.

    Parameters
    ----------
    stack         : (N, H, W) uint16  Raw Bayer burst stack (BLC+OECF applied).
    bayer_pattern : str
    ref_idx       : int                Index of the reference frame (default 0).

    Returns
    -------
    registered    : (N, H, W) uint16  All frames aligned to reference.
    shifts        : list of (dy, dx)  Estimated shifts per frame.
    """
    N = stack.shape[0]
    registered = stack.copy()
    shifts = [(0.0, 0.0)] * N

    # Extract reference green (half-res)
    ref_g = _green_channel(stack[ref_idx], bayer_pattern)

    for i in range(N):
        if i == ref_idx:
            continue
        target_g = _green_channel(stack[i], bayer_pattern)
        dy, dx   = phase_correlate(ref_g, target_g)

        # Convert sub-channel shift back to full-frame shift for apply_shift_bayer
        dy_full = dy * 2.0
        dx_full = dx * 2.0

        shifts[i] = (dy_full, dx_full)

        if abs(dy_full) > 0.01 or abs(dx_full) > 0.01:
            registered[i] = apply_shift_bayer(stack[i], dy_full, dx_full,
                                               bayer_pattern)
            print(f"  [BurstReg] frame {i}: shift=({dy_full:+.2f}, {dx_full:+.2f}) px")
        else:
            print(f"  [BurstReg] frame {i}: no shift needed")

    return registered, shifts
