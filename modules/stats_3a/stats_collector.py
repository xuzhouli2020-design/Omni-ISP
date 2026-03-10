"""
File: stats_collector.py
Description: Shared 3A statistics collector — computes all per-frame
             statistics needed by AE, AWB and AF in a single Bayer-domain
             pass, avoiding redundant computation.

Runs: after BNR (clean signal), before AWB in the pipeline.

Statistics collected
────────────────────
  Zone luminance  — NxN grid of G-channel means and histograms → AE
  Channel means   — per-channel (R, Gr, Gb, B) mean pixel value → AWB
  Channel grads   — per-channel Minkowski-p gradient mean → Gray Edge AWB
  Focus metric    — Tenengrad / Laplacian variance in AF ROI → AF

Phase 4.1
"""

import time
from dataclasses import dataclass, field
import numpy as np

from util.bayer_utils import (
    extract_channels, green_channel, normalise,
    sobel_gradients, laplacian,
)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class Stats3A:
    """
    All per-frame statistics produced by the Stats3ACollector.
    Consumed by AE, AWB (Gray Edge), and AF modules.
    """
    # ---- Luminance / AE ----
    zone_means: np.ndarray          # (Nz, Nz) float32, G-channel mean per zone [0,1]
    zone_histograms: np.ndarray     # (Nz, Nz, bins) uint32, per-zone histograms
    global_histogram: np.ndarray    # (bins,) uint32, full-frame G histogram
    frame_mean: float               # overall G-channel mean [0,1]
    highlight_fraction: float       # fraction of G pixels above 0.95 full scale

    # ---- Colour / AWB ----
    channel_means: dict             # {"r","gr","gb","b"} → float [0,1]
    channel_grad_means: dict        # {"r","gr","gb","b"} → float (L1 gradient mean)

    # ---- Focus / AF ----
    focus_metric: float             # Tenengrad score in AF ROI (higher = sharper)
    focus_roi_g: np.ndarray         # normalised G sub-image cropped to AF ROI

    # ---- Metadata ----
    bayer_pattern: str
    bit_depth: int
    zone_grid: int
    hist_bins: int


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class Stats3ACollector:
    """
    Compute Stats3A from a Bayer-domain image.

    Parameters
    ----------
    img          : np.ndarray  Full-resolution Bayer raw (uint16)
    sensor_info  : dict        Must contain "bit_depth" and "bayer_pattern"
    parm_3a      : dict        3a_stats config section from configs.yml
    """

    def __init__(self, img: np.ndarray, sensor_info: dict, parm_3a: dict):
        self.img         = img
        self.bit_depth   = sensor_info["bit_depth"]
        self.bayer_pat   = sensor_info["bayer_pattern"]
        self.enable      = bool(parm_3a.get("is_enable", True))
        self.zone_grid   = int(parm_3a.get("zone_grid", 16))
        self.hist_bins   = int(parm_3a.get("hist_bins", 256))
        self.af_roi      = list(parm_3a.get("af_roi", [0.33, 0.33, 0.67, 0.67]))
        # Gradient order + norm for channel_grad_means
        self.grad_order  = int(parm_3a.get("grad_order", 1))   # 1=Sobel, 2=Laplacian
        self.grad_p      = float(parm_3a.get("minkowski_norm", 1))  # p-norm

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _zone_stats(self, g_norm: np.ndarray) -> tuple:
        """
        Divide the G sub-image into a zone_grid × zone_grid grid and compute
        per-zone mean luminance and histogram.

        Parameters
        ----------
        g_norm : float32 array (Hg, Wg) in [0, 1]

        Returns
        -------
        zone_means      : (Nz, Nz) float32
        zone_histograms : (Nz, Nz, bins) uint32
        global_histogram: (bins,) uint32
        """
        Hg, Wg = g_norm.shape
        Nz = self.zone_grid
        bins = self.hist_bins

        zone_means = np.zeros((Nz, Nz), dtype=np.float32)
        zone_hists = np.zeros((Nz, Nz, bins), dtype=np.uint32)

        h_step = Hg / Nz
        w_step = Wg / Nz

        for i in range(Nz):
            r0 = int(round(i * h_step))
            r1 = int(round((i + 1) * h_step))
            for j in range(Nz):
                c0 = int(round(j * w_step))
                c1 = int(round((j + 1) * w_step))
                patch = g_norm[r0:r1, c0:c1]
                if patch.size == 0:
                    continue
                zone_means[i, j] = float(patch.mean())
                hist, _ = np.histogram(patch, bins=bins, range=(0.0, 1.0))
                zone_hists[i, j] = hist.astype(np.uint32)

        global_hist, _ = np.histogram(g_norm, bins=bins, range=(0.0, 1.0))
        return zone_means, zone_hists, global_hist.astype(np.uint32)

    def _channel_stats(self, channels: dict) -> tuple:
        """
        Compute per-channel (R, Gr, Gb, B) mean and gradient mean.

        Returns
        -------
        means      : {"r","gr","gb","b"} → float [0,1]
        grad_means : {"r","gr","gb","b"} → float (p-norm gradient mean)
        """
        maxval = float((1 << self.bit_depth) - 1)
        means = {}
        grad_means = {}

        for name, ch in channels.items():
            ch_f = ch.astype(np.float32) / maxval
            means[name] = float(ch_f.mean())

            # Gradient for Gray Edge AWB
            if self.grad_order == 1:
                gx, gy = sobel_gradients(ch_f)
                grad = np.abs(gx) + np.abs(gy)
            else:
                grad = np.abs(laplacian(ch_f))

            p = self.grad_p
            if p == 1:
                grad_means[name] = float(grad.mean())
            else:
                grad_means[name] = float(np.mean(grad ** p) ** (1.0 / p))

        return means, grad_means

    def _focus_metric(self, g_norm: np.ndarray) -> tuple:
        """
        Tenengrad focus metric in the configured AF ROI.

        Returns
        -------
        metric  : float  (sum of squared Sobel gradients, higher = sharper)
        roi_g   : ndarray  normalised G sub-image in ROI
        """
        Hg, Wg = g_norm.shape
        x0, y0, x1, y1 = self.af_roi
        r0 = int(round(y0 * Hg));  r1 = int(round(y1 * Hg))
        c0 = int(round(x0 * Wg));  c1 = int(round(x1 * Wg))
        r0, r1 = max(0, r0), min(Hg, r1)
        c0, c1 = max(0, c0), min(Wg, c1)
        roi = g_norm[r0:r1, c0:c1]
        if roi.size < 9:
            return 0.0, roi
        gx, gy = sobel_gradients(roi)
        metric = float(np.mean(gx ** 2 + gy ** 2))
        return metric, roi

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def execute(self) -> Stats3A:
        """Collect all 3A statistics and return a Stats3A object."""
        start = time.time()

        # Extract Bayer sub-channels
        channels = extract_channels(self.img, self.bayer_pat)
        maxval = float((1 << self.bit_depth) - 1)

        # G sub-image (luminance proxy): average Gr + Gb
        g_norm = (channels["gr"].astype(np.float32)
                  + channels["gb"].astype(np.float32)) / (2.0 * maxval)

        # Zone luminance stats (AE)
        zone_means, zone_hists, global_hist = self._zone_stats(g_norm)
        frame_mean = float(g_norm.mean())
        highlight_fraction = float(np.mean(g_norm > 0.95))

        # Channel stats (AWB)
        channel_means, channel_grad_means = self._channel_stats(channels)

        # Focus metric (AF)
        focus_metric, focus_roi_g = self._focus_metric(g_norm)

        elapsed = time.time() - start
        print(f"  3A stats collected in {elapsed:.3f}s  "
              f"(frame_mean={frame_mean:.3f}, "
              f"highlight={highlight_fraction*100:.1f}%, "
              f"focus={focus_metric:.4f})")

        return Stats3A(
            zone_means=zone_means,
            zone_histograms=zone_hists,
            global_histogram=global_hist,
            frame_mean=frame_mean,
            highlight_fraction=highlight_fraction,
            channel_means=channel_means,
            channel_grad_means=channel_grad_means,
            focus_metric=focus_metric,
            focus_roi_g=focus_roi_g,
            bayer_pattern=self.bayer_pat,
            bit_depth=self.bit_depth,
            zone_grid=self.zone_grid,
            hist_bins=self.hist_bins,
        )
