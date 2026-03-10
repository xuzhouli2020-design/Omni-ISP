"""
File: temporal_nr.py
Description: Temporal Noise Reduction (TNR) — IIR per-pixel low-pass filter
             with motion masking for continuous video capture.

Design
──────
For each incoming Bayer frame, TNR blends it with the previous *filtered*
output using an exponential moving average (EMA):

    static pixels:   out[y,x] = (1-α) * current[y,x] + α * prev[y,x]
    moving pixels:   out[y,x] = current[y,x]

where α = `alpha` (0 = no smoothing, 1 = freeze on previous frame).

This is an IIR temporal low-pass filter.  Static regions accumulate signal
over time (equivalent SNR gain ≈ 1/√(1-α²)); moving regions bypass the
filter entirely to avoid ghosting / motion blur.

Motion detection
────────────────
Frame difference on a downsampled green channel:
    motion[y,x] = 1  if  |current_g[y,x] - prev_g[y,x]| > motion_threshold
                = 0  otherwise

The downsampling factor (`downsample`) reduces computation.  The binary mask
is upsampled back to full resolution via nearest-neighbour before blending.

The motion mask is *dilated* by one block to soften moving edges — this
prevents ghosting halos at object boundaries.

Pipeline position
─────────────────
Bayer domain, **after BNR, before Stats3A / AWB**.  Must be in Bayer domain
so demosaic sees temporally-smoothed data and chroma artifacts are minimised.

Persistence
───────────
TNR is stateful: `self.prev_filtered` and `self.prev_g` must be stored on
`OmniISP` across `execute()` calls, then passed back via `tnr_state`.

Parameters (configs.yml `temporal_nr` section)
────────────────────────────────────────────────
  is_enable        : bool   Enable TNR                          [false]
  alpha            : float  IIR blend weight (0=off, 0.7=strong)[0.7]
  motion_threshold : float  Green-channel diff threshold (ADU)  [8.0]
  downsample       : int    Motion estimation downsample factor  [2]

Phase 5.5
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from util.bayer_utils import extract_channels


@dataclass
class TNRState:
    """Persistent state carried between video frames."""
    prev_filtered: np.ndarray   # (H, W) uint16 — previous TNR output
    prev_g:        np.ndarray   # (H//2, W//2) float32 — previous green channel


class TemporalNR:
    """
    Temporal Noise Reduction module (IIR EMA filter with motion masking).

    Parameters
    ----------
    img          : (H, W) uint16  Current Bayer frame (after BNR).
    sensor_info  : dict           Sensor metadata.
    parm_tnr     : dict           TNR config section from configs.yml.
    tnr_state    : TNRState|None  Persistent state from the previous frame.
                                  None on the first frame — pass-through only.
    """

    def __init__(self,
                 img:        np.ndarray,
                 sensor_info: dict,
                 parm_tnr:   dict,
                 tnr_state:  Optional[TNRState] = None):
        self.img         = img
        self.sensor_info = sensor_info
        self.enable      = bool(parm_tnr.get("is_enable", False))
        self.alpha       = float(parm_tnr.get("alpha",            0.7))
        self.threshold   = float(parm_tnr.get("motion_threshold", 8.0))
        self.downsample  = int(parm_tnr.get("downsample",         2))
        self.bayer       = sensor_info.get("bayer_pattern", "rggb")
        self.bit_depth   = sensor_info.get("bit_depth",     10)

        self.tnr_state   = tnr_state   # may be None on first frame

        # Will be populated by execute()
        self.new_state: Optional[TNRState] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(self) -> np.ndarray:
        """
        Apply one TNR step.

        Returns
        -------
        np.ndarray  (H, W) uint16 — temporally filtered Bayer frame.
                    (On first frame / disabled: returns input unchanged.)
        """
        if not self.enable:
            self._record_state(self.img)
            return self.img

        if self.tnr_state is None:
            # First frame — no history yet; record state for next frame
            self._record_state(self.img)
            return self.img

        # Compute motion mask
        current_g  = self._green(self.img)
        motion_mask = self._motion_mask(current_g, self.tnr_state.prev_g)

        # IIR blend
        filtered = self._blend(
            self.img,
            self.tnr_state.prev_filtered,
            motion_mask,
        )

        self._record_state(filtered)
        return filtered

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _green(self, raw: np.ndarray) -> np.ndarray:
        """Extract half-resolution green channel (average of Gr + Gb)."""
        chs = extract_channels(raw, self.bayer)
        return (chs["gr"].astype(np.float32) + chs["gb"].astype(np.float32)) * 0.5

    def _motion_mask(self,
                     current_g: np.ndarray,
                     prev_g:    np.ndarray) -> np.ndarray:
        """
        Compute a full-Bayer-resolution motion mask.

        Returns (H, W) float32:  1.0 = static,  0.0 = moving.
        """
        ds = max(1, self.downsample)
        Hg, Wg = current_g.shape

        # Optional downsampling for speed
        cur_ds  = current_g[::ds, ::ds]
        prev_ds = prev_g[::ds, ::ds]

        diff = np.abs(cur_ds - prev_ds)

        # Per-pixel binary mask at downsampled resolution (1=static)
        static_ds = (diff <= self.threshold).astype(np.float32)

        # Dilate moving regions by one block (soften ghosting at edges)
        static_ds = self._erode_static(static_ds)

        # Upsample back to green sub-channel resolution
        static_g = np.kron(static_ds, np.ones((ds, ds), dtype=np.float32))
        # Trim / pad to match original green shape
        static_g = static_g[:Hg, :Wg]
        if static_g.shape[0] < Hg or static_g.shape[1] < Wg:
            pad_y = Hg - static_g.shape[0]
            pad_x = Wg - static_g.shape[1]
            static_g = np.pad(static_g, ((0, pad_y), (0, pad_x)), mode="edge")

        # Expand green mask to full Bayer resolution (2× in each dimension)
        H, W = self.img.shape
        full_mask = np.kron(static_g, np.ones((2, 2), dtype=np.float32))
        full_mask = full_mask[:H, :W]
        if full_mask.shape[0] < H or full_mask.shape[1] < W:
            pad_y = H - full_mask.shape[0]
            pad_x = W - full_mask.shape[1]
            full_mask = np.pad(full_mask, ((0, pad_y), (0, pad_x)), mode="edge")

        return full_mask

    @staticmethod
    def _erode_static(static: np.ndarray) -> np.ndarray:
        """
        Morphological erosion of the static mask by 1 pixel.
        A static pixel is considered static only if all 4 neighbours are also
        static — this dilates moving regions and shrinks static ones, preventing
        ghosting halos at boundaries.
        Pure NumPy (no scipy needed).
        """
        eroded = static.copy()
        # Erode using minimum of 4-connected neighbours
        H, W = static.shape
        eroded[1:,  :]  = np.minimum(eroded[1:,  :],  static[:-1, :])
        eroded[:-1, :]  = np.minimum(eroded[:-1, :],  static[1:,  :])
        eroded[:,  1:]  = np.minimum(eroded[:,  1:],  static[:,  :-1])
        eroded[:, :-1]  = np.minimum(eroded[:, :-1],  static[:,  1:])
        return eroded

    def _blend(self,
               current:  np.ndarray,
               previous: np.ndarray,
               mask:     np.ndarray) -> np.ndarray:
        """
        IIR EMA blend:
            out = (1-α)*current + α*prev     where mask=1 (static)
            out = current                    where mask=0 (moving)

        Both current and previous are uint16; blend is done in float32.
        """
        α = self.alpha
        cur_f  = current.astype(np.float32)
        prev_f = previous.astype(np.float32)

        blended = (1.0 - α) * cur_f + α * prev_f
        # Use current frame for moving pixels
        out_f = mask * blended + (1.0 - mask) * cur_f

        maxval = float((1 << self.bit_depth) - 1)
        return np.clip(np.round(out_f), 0, maxval).astype(np.uint16)

    def _record_state(self, filtered: np.ndarray) -> None:
        """Store current filtered frame and green channel for next frame."""
        self.new_state = TNRState(
            prev_filtered = filtered.copy(),
            prev_g        = self._green(filtered),
        )
