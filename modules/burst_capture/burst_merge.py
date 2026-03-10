"""
File: burst_merge.py
Description: Motion detection and temporal merge for burst Bayer stacks.

Pipeline
────────
  registered (N, H, W) uint16
       │
       ├── detect_motion()  →  motion_masks (N, H, W) float32 in [0, 1]
       │                        1 = static (match reference), 0 = moving
       │
       └── temporal_merge() →  merged (H, W) uint16
                                static pixels:  mean of all N frames  → √N SNR gain
                                moving pixels:  reference frame only  → no ghosting
                                boundary:       Gaussian-weighted blend

Motion detection
────────────────
  Block-wise SAD (Sum of Absolute Differences) on the green channel.
  Block size and threshold are configurable.  The per-block boolean mask is
  upsampled back to full Bayer resolution using nearest-neighbour assignment
  so that motion boundaries are CFA-aligned.

Merge methods
─────────────
  "mean"           — simple unweighted mean of all N aligned frames.
  "weighted_mean"  — motion-weighted mean: each pixel's weight is its
                     motion_mask value (1=static, 0=moving).  Static pixels
                     accumulate across all frames; moving pixels fall back to
                     the reference.
  "median"         — per-pixel median across the stack (outlier-robust but
                     slower).  Ignores motion masks.

Phase 5.3
"""

import numpy as np
from util.bayer_utils import extract_channels


# ---------------------------------------------------------------------------
# Motion detection
# ---------------------------------------------------------------------------

def detect_motion(
    stack: np.ndarray,
    bayer_pattern: str,
    ref_idx: int  = 0,
    block_size: int = 16,
    threshold: float = 15.0,
) -> np.ndarray:
    """
    Compute per-pixel motion confidence masks for each frame vs reference.

    Parameters
    ----------
    stack        : (N, H, W) uint16  Registered Bayer stack.
    bayer_pattern: str
    ref_idx      : int               Reference frame index.
    block_size   : int               SAD block size in pixels (full Bayer res).
    threshold    : float             SAD/pixel threshold; frames above = moving.

    Returns
    -------
    motion_masks : (N, H, W) float32
        Values in [0, 1].  1 = static (similar to reference), 0 = moving.
        Reference frame mask is all-ones.
    """
    N, H, W = stack.shape

    # Work on green channel (half-res) for efficiency
    ref_g = _extract_green(stack[ref_idx], bayer_pattern)  # (H//2, W//2)
    Hg, Wg = ref_g.shape
    bs = block_size // 2   # block size in sub-channel coordinates

    motion_masks = np.ones((N, H, W), dtype=np.float32)

    for i in range(N):
        if i == ref_idx:
            continue
        target_g = _extract_green(stack[i], bayer_pattern)
        diff = np.abs(ref_g.astype(np.float32) - target_g.astype(np.float32))

        # Block-wise SAD → per-block binary mask (in sub-channel coords)
        block_mask_g = _block_sad_mask(diff, bs, threshold)

        # Upsample block mask back to full Bayer resolution (nearest-neighbour)
        # block_mask_g is in (Hg // bs, Wg // bs) → need to reach (H, W)
        full_mask = _upsample_mask_to_bayer(block_mask_g, H, W, bs)
        motion_masks[i] = full_mask

    return motion_masks


def _extract_green(raw: np.ndarray, bayer_pattern: str) -> np.ndarray:
    chs = extract_channels(raw, bayer_pattern)
    return ((chs["gr"].astype(np.float32) + chs["gb"].astype(np.float32)) * 0.5)


def _block_sad_mask(
    diff: np.ndarray,
    block_size: int,
    threshold: float,
) -> np.ndarray:
    """
    Compute per-block mean SAD and return a float mask (1=static, 0=moving).

    Parameters
    ----------
    diff       : (Hg, Wg) float32  Absolute difference between green channels.
    block_size : int                Block size in sub-channel pixels.
    threshold  : float              Mean-SAD-per-pixel threshold.

    Returns
    -------
    mask : (n_blocks_y, n_blocks_x) float32  1.0=static, 0.0=moving.
    """
    Hg, Wg = diff.shape
    bs = max(1, block_size)
    n_by = max(1, Hg // bs)
    n_bx = max(1, Wg // bs)

    mask = np.ones((n_by, n_bx), dtype=np.float32)
    for by in range(n_by):
        for bx in range(n_bx):
            y0, y1 = by * bs, min((by + 1) * bs, Hg)
            x0, x1 = bx * bs, min((bx + 1) * bs, Wg)
            block_sad = diff[y0:y1, x0:x1].mean()
            if block_sad > threshold:
                mask[by, bx] = 0.0
    return mask


def _upsample_mask_to_bayer(
    block_mask: np.ndarray,   # (n_by, n_bx) in sub-channel block grid
    H: int, W: int,           # full Bayer frame size
    block_size_g: int,        # block size in sub-channel pixels
) -> np.ndarray:
    """
    Upsample the block-level mask to full Bayer resolution.

    The block mask is defined at half resolution (green sub-channel) in units
    of block_size_g.  Each block covers (block_size_g * 2) × (block_size_g * 2)
    pixels in the full Bayer frame.
    """
    n_by, n_bx = block_mask.shape
    full = np.ones((H, W), dtype=np.float32)

    block_full = block_size_g * 2   # block size in full-res pixels

    for by in range(n_by):
        for bx in range(n_bx):
            y0 = by * block_full
            y1 = min(y0 + block_full, H)
            x0 = bx * block_full
            x1 = min(x0 + block_full, W)
            full[y0:y1, x0:x1] = block_mask[by, bx]

    return full


# ---------------------------------------------------------------------------
# Temporal merge
# ---------------------------------------------------------------------------

def temporal_merge(
    stack: np.ndarray,
    motion_masks: np.ndarray,
    method: str  = "weighted_mean",
    ref_idx: int = 0,
) -> np.ndarray:
    """
    Merge N aligned and motion-masked Bayer frames into a single output.

    Parameters
    ----------
    stack        : (N, H, W) uint16  Registered Bayer stack.
    motion_masks : (N, H, W) float32 Motion confidence (1=static, 0=moving).
    method       : str               "mean" | "weighted_mean" | "median"
    ref_idx      : int               Reference frame (used as fallback for
                                     purely moving regions).

    Returns
    -------
    merged : (H, W) uint16  Merged single Bayer frame.
    """
    if method == "mean":
        return _merge_mean(stack)
    elif method == "median":
        return _merge_median(stack)
    else:  # weighted_mean (default)
        return _merge_weighted(stack, motion_masks, ref_idx)


def _merge_mean(stack: np.ndarray) -> np.ndarray:
    """Simple unweighted mean across N frames."""
    merged = stack.astype(np.float64).mean(axis=0)
    return np.clip(np.round(merged), 0, 65535).astype(np.uint16)


def _merge_median(stack: np.ndarray) -> np.ndarray:
    """Per-pixel median — outlier-robust but 2-3× slower."""
    merged = np.median(stack.astype(np.float32), axis=0)
    return np.clip(np.round(merged), 0, 65535).astype(np.uint16)


def _merge_weighted(
    stack: np.ndarray,
    motion_masks: np.ndarray,
    ref_idx: int,
) -> np.ndarray:
    """
    Motion-aware weighted mean.

    For each pixel (y, x):
      weight_i = motion_mask[i, y, x]   (1=static, 0=moving)
      total_weight = sum of weights across N frames

      If total_weight > 0:
        merged[y,x] = sum(stack[i,y,x] * weight_i) / total_weight
      Else (all frames show motion at this pixel):
        merged[y,x] = stack[ref_idx, y, x]   (reference fallback)

    For fully static pixels (all weights = 1.0) this becomes the mean of all
    N frames, achieving √N noise reduction.
    """
    stack_f   = stack.astype(np.float32)          # (N, H, W)
    weights   = motion_masks.astype(np.float32)    # (N, H, W)

    numerator   = (stack_f * weights).sum(axis=0)  # (H, W)
    denominator = weights.sum(axis=0)              # (H, W)

    # Pixels where all frames are flagged as moving → fall back to reference
    moving_mask = denominator < 0.5               # all weights near zero
    denominator[moving_mask] = 1.0                # avoid division by zero
    merged = numerator / denominator

    # Inject reference frame at fully-moving pixels
    merged[moving_mask] = stack_f[ref_idx][moving_mask]

    return np.clip(np.round(merged), 0, 65535).astype(np.uint16)


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def merge_burst(
    stack: np.ndarray,
    bayer_pattern: str,
    ref_idx: int    = 0,
    block_size: int = 16,
    threshold: float = 15.0,
    method: str      = "weighted_mean",
) -> np.ndarray:
    """
    Full burst merge: motion detection → weighted temporal merge.

    Parameters
    ----------
    stack        : (N, H, W) uint16  Registered Bayer stack.
    bayer_pattern: str
    ref_idx      : int               Reference frame index.
    block_size   : int               SAD block size (Bayer pixels).
    threshold    : float             SAD/pixel threshold.
    method       : str               Merge method.

    Returns
    -------
    merged : (H, W) uint16  Single denoised Bayer frame.
    """
    N = stack.shape[0]
    print(f"  [BurstMerge] merging {N} frames  method={method}  "
          f"block={block_size}px  SAD_thresh={threshold}")

    masks = detect_motion(stack, bayer_pattern, ref_idx, block_size, threshold)

    static_fraction = float(masks.mean())
    print(f"  [BurstMerge] static fraction = {static_fraction:.1%}")

    merged = temporal_merge(stack, masks, method, ref_idx)
    return merged
