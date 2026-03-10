"""
File: ae_metering.py
Description: AE zone metering — converts Stats3A zone luminance data into a
             single scene-brightness estimate (0–1) used by the AE controller.

Metering modes
──────────────
  "center_weighted"  — Gaussian-weighted average of zone_means, centred on
                       frame.  Classic camera metering; good general-purpose.

  "spot"             — Average of the centre N×N zones only (configurable
                       fraction of frame).  For backlit subjects, concerts, etc.

  "matrix"           — Evaluative / multi-zone metering.  Weighted by zone
                       saliency (contrast-relative weight) and highlight
                       penalty, then averaged.  Adapts to complex lighting.

Highlight detection
───────────────────
  The AE controller also receives a highlight fraction (from Stats3A) so it can
  apply a clipping penalty independently of the metering mode.

Parameters (from configs.yml ae section)
──────────────────────────────────────────
  metering_mode    : str    "center_weighted" | "spot" | "matrix"  ["center_weighted"]
  spot_fraction    : float  Fraction of frame edge used for spot zone [0.25]
  highlight_weight : float  Penalty for highlight-blown zones in matrix [2.0]

Phase 4.2
"""

import numpy as np


class AEMetering:
    """
    Compute a scalar scene-brightness estimate from Stats3A zone data.

    Parameters
    ----------
    stats          : Stats3A   Per-frame 3A statistics.
    metering_mode  : str       Metering mode (see module docstring).
    spot_fraction  : float     Fraction of linear frame dimension for spot zone.
    highlight_weight: float    Down-weight factor for blown zones in matrix mode.
    """

    def __init__(self,
                 stats,
                 metering_mode: str   = "center_weighted",
                 spot_fraction: float  = 0.25,
                 highlight_weight: float = 2.0):
        self.stats            = stats
        self.metering_mode    = metering_mode
        self.spot_fraction    = float(spot_fraction)
        self.highlight_weight = float(highlight_weight)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def measure(self) -> float:
        """
        Compute scene brightness in [0, 1].

        Returns
        -------
        float   Metered luminance estimate (0 = black, 1 = full scale).
        """
        zone_means = self.stats.zone_means      # (Nz, Nz) float32
        Nz = zone_means.shape[0]

        if self.metering_mode == "spot":
            return self._spot(zone_means, Nz)
        elif self.metering_mode == "matrix":
            return self._matrix(zone_means, Nz)
        else:  # center_weighted (default)
            return self._center_weighted(zone_means, Nz)

    # ------------------------------------------------------------------
    # Metering implementations
    # ------------------------------------------------------------------

    def _center_weighted(self, zone_means: np.ndarray, Nz: int) -> float:
        """
        Gaussian-weighted mean centred on the frame.

        The Gaussian sigma is set to Nz/3, giving heavy weight to the
        central third of the frame and soft falloff to the edges.
        """
        sigma = Nz / 3.0
        cy = cx = (Nz - 1) / 2.0
        ys = np.arange(Nz, dtype=np.float32)
        xs = np.arange(Nz, dtype=np.float32)
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        weights = np.exp(-0.5 * ((Y - cy) ** 2 + (X - cx) ** 2) / sigma ** 2)
        weights /= weights.sum()
        return float((zone_means * weights).sum())

    def _spot(self, zone_means: np.ndarray, Nz: int) -> float:
        """
        Average of the central (spot_fraction × Nz)² zones.
        """
        half = max(1, int(round(Nz * self.spot_fraction / 2)))
        cy = Nz // 2
        cx = Nz // 2
        r0 = max(0, cy - half)
        r1 = min(Nz, cy + half)
        c0 = max(0, cx - half)
        c1 = min(Nz, cx + half)
        patch = zone_means[r0:r1, c0:c1]
        if patch.size == 0:
            return float(zone_means.mean())
        return float(patch.mean())

    def _matrix(self, zone_means: np.ndarray, Nz: int) -> float:
        """
        Evaluative multi-zone metering.

        Each zone gets a weight based on:
          - Local contrast (deviation from frame mean) — higher contrast zones
            are more informative and receive more weight.
          - Highlight penalty — zones close to saturation are down-weighted
            to prevent the metering from chasing highlights.

        The final brightness is the weighted mean of zone luminances.
        """
        frame_mean = float(zone_means.mean())

        # Saliency weight: zone deviation from frame mean (normalised)
        deviation = np.abs(zone_means - frame_mean)
        saliency = deviation / (deviation.max() + 1e-6) + 1.0   # [1, 2]

        # Highlight penalty: zones very close to saturation (> 0.9) carry
        # less weight so the controller does not over-expose to fix them.
        highlight_mask = zone_means > 0.9
        highlight_penalty = np.where(
            highlight_mask,
            1.0 / self.highlight_weight,
            1.0,
        )

        weights = saliency * highlight_penalty
        weights /= weights.sum()
        return float((zone_means * weights).sum())
