"""
File: auto_white_balance.py
Description: 3A - AWB Runs the AWB algorithm based on selection from config file.
             Extended with Gray Edge algorithm (Phase 4.4) and temporal gain
             damping (Phase 4.5) for stable, flicker-free white balance in video.

Code / Paper  Reference:
  - Gray World: https://www.sciencedirect.com/science/article/abs/pii/0016003280900587
  - Norm-2 GW : https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/cic/12/1/art00008
  - PCA       : https://opg.optica.org/josaa/viewmedia.cfm?uri=josaa-31-5-1049&seq=0
  - Gray Edge : van de Weijer et al., "Edge-Based Color Constancy", IEEE TIP 2007.

Author: 10xEngineers Pvt Ltd (original); Omni-ISP extensions — Phase 4.4/4.5
------------------------------------------------------------
"""
import time
import numpy as np
from modules.auto_white_balance.gray_world import GrayWorld as GW
from modules.auto_white_balance.norm_gray_world import NormGrayWorld as NGW
from modules.auto_white_balance.pca import PCAIlluminEstimation as PCA
from modules.auto_white_balance.gray_edge import GrayEdge
from util.bayer_utils import extract_channels


class AutoWhiteBalance:
    """
    Auto White Balance Module.

    Algorithms selectable via ``parm_awb["algorithm"]``:
      "gray_world"  — classic Gray World (default)
      "norm_2"      — Norm-2 Gray World
      "pca"         — PCA illuminant estimation
      "gray_edge"   — Gray Edge (gradient-domain, more robust) [Phase 4.4]

    Temporal damping (Phase 4.5):
      Enabled when ``parm_awb["temporal_damping"] > 0``.  New gains are
      blended with the previous frame's gains using an IIR filter:
        gain_out = alpha * gain_new + (1 - alpha) * gain_prev
      where alpha = 1 - temporal_damping  (0 = freeze, 1 = instant).

    Parameters
    ----------
    raw         : np.ndarray        Raw Bayer image.
    sensor_info : dict              Sensor metadata.
    parm_awb    : dict              AWB config section from configs.yml.
    stats       : Stats3A | None    Pre-computed 3A stats (used by gray_edge
                                    for the pre-computed channel_grad_means).
    prev_gains  : np.ndarray | None Previous frame's [r_gain, b_gain] for
                                    temporal damping.  None = first frame.
    """

    def __init__(self, raw, sensor_info, parm_awb,
                 stats=None, prev_gains=None):

        self.raw = raw
        self.sensor_info = sensor_info
        self.parm_awb = parm_awb
        self.enable = parm_awb["is_enable"]
        self.bit_depth = sensor_info["bit_depth"]
        self.is_debug = parm_awb.get("is_debug", False)
        self.underexposed_percentage = parm_awb["underexposed_percentage"]
        self.overexposed_percentage  = parm_awb["overexposed_percentage"]
        self.flatten_img = None
        self.bayer = self.sensor_info["bayer_pattern"]
        self.algorithm = parm_awb["algorithm"]

        # Stats3A — optional fast path for gray_edge
        self.stats = stats

        # Temporal damping state (Phase 4.5)
        self.temporal_damping = float(parm_awb.get("temporal_damping", 0.0))
        self.prev_gains = prev_gains  # np.array([r, b]) or None

    # ------------------------------------------------------------------
    # Main orchestration
    # ------------------------------------------------------------------

    def determine_white_balance_gain(self):
        """
        Determine white balance gains via the selected algorithm.
        """

        if self.algorithm == "gray_edge":
            rgain, bgain = self.apply_gray_edge()
        else:
            # Legacy algorithms — need flattened RGB triplets
            self._prepare_flatten_img()
            if self.algorithm == "norm_2":
                rgain, bgain = self.apply_norm_gray_world()
            elif self.algorithm == "pca":
                rgain, bgain = self.apply_pca_illuminant_estimation()
            else:
                rgain, bgain = self.apply_gray_world()

        # Legacy clamp: gain must be >= 1 (green is reference channel)
        rgain = max(1.0, rgain)
        bgain = max(1.0, bgain)

        if self.is_debug:
            print("   - AWB Actual Gains: ")
            print("   - AWB - RGain = ", rgain)
            print("   - AWB - Bgain = ", bgain)

        return rgain, bgain

    def _prepare_flatten_img(self):
        """
        Build the flattened (N, 3) array of valid (R, G, B) pixel triplets
        used by the legacy gray-world / PCA algorithms.
        """
        max_pixel_value  = 2 ** self.bit_depth
        approx_percentage = max_pixel_value / 100
        overexposed_limit  = (
            max_pixel_value - self.overexposed_percentage  * approx_percentage
        )
        underexposed_limit = self.underexposed_percentage * approx_percentage

        if self.is_debug:
            print("   - AWB - Underexposed Pixel Limit = ", underexposed_limit)
            print("   - AWB - Overexposed Pixel Limit  = ", overexposed_limit)

        channels = extract_channels(self.raw, self.bayer)
        r_channel  = channels["r"]
        gr_channel = channels["gr"]
        gb_channel = channels["gb"]
        b_channel  = channels["b"]

        g_channel    = (gr_channel + gb_channel) / 2
        bayer_channels = np.dstack((r_channel, g_channel, b_channel))

        bad_pixels = np.sum(
            np.where(
                (bayer_channels < underexposed_limit)
                | (bayer_channels > overexposed_limit),
                1, 0,
            ),
            axis=2,
        )
        self.flatten_img = bayer_channels[bad_pixels == 0]

    # ------------------------------------------------------------------
    # Algorithm implementations
    # ------------------------------------------------------------------

    def apply_gray_world(self):
        """
        Gray World White Balance: gains = G_mean / C_mean per channel.
        """
        gwa = GW(self.flatten_img)
        return gwa.calculate_gains()

    def apply_norm_gray_world(self):
        """
        Norm-2 Gray World: uses L2 norm instead of mean.
        """
        ngw = NGW(self.flatten_img)
        return ngw.calculate_gains()

    def apply_pca_illuminant_estimation(self):
        """
        PCA Illuminant Estimation — bright + dark pixel projection.
        """
        pixel_percentage = self.parm_awb["percentage"]
        pca = PCA(self.flatten_img, pixel_percentage)
        return pca.calculate_gains()

    def apply_gray_edge(self):
        """
        Gray Edge (Phase 4.4): gradient-domain illuminant estimation.

        If Stats3A channel_grad_means are already available they are used
        directly (fast path); otherwise GrayEdge recomputes from the raw
        Bayer image.
        """
        parm = self.parm_awb

        # Fast path: use pre-computed gradient means from Stats3A
        if self.stats is not None and hasattr(self.stats, "channel_grad_means"):
            gm = self.stats.channel_grad_means
            gm_r  = float(gm.get("r",  1e-8))
            gm_gr = float(gm.get("gr", 1e-8))
            gm_gb = float(gm.get("gb", 1e-8))
            gm_b  = float(gm.get("b",  1e-8))
            gm_g  = (gm_gr + gm_gb) * 0.5 + 1e-8

            min_gain = float(parm.get("min_gain", 1.0))
            max_gain = float(parm.get("max_gain", 4.0))

            r_gain = float(np.clip(gm_g / (gm_r + 1e-8), min_gain, max_gain))
            b_gain = float(np.clip(gm_g / (gm_b + 1e-8), min_gain, max_gain))
            return r_gain, b_gain

        # Slow path: compute from raw Bayer image
        channels = extract_channels(self.raw, self.bayer)
        ge = GrayEdge(
            channels    = channels,
            bit_depth   = self.bit_depth,
            grad_order  = int(parm.get("grad_order",    1)),
            p           = float(parm.get("minkowski_p", 1.0)),
            sat_thresh  = float(parm.get("sat_threshold",  0.95)),
            dark_thresh = float(parm.get("dark_threshold", 0.05)),
            min_gain    = float(parm.get("min_gain", 1.0)),
            max_gain    = float(parm.get("max_gain", 4.0)),
        )
        return ge.calculate_gains()

    # ------------------------------------------------------------------
    # Temporal damping (Phase 4.5)
    # ------------------------------------------------------------------

    def _apply_temporal_damping(self, rgain: float, bgain: float):
        """
        Blend current gains with previous frame gains (IIR low-pass).

        alpha = 1 - temporal_damping
          alpha=1.0 → instant update (no damping)
          alpha=0.0 → freeze gains (max damping)
        """
        if self.temporal_damping <= 0 or self.prev_gains is None:
            return rgain, bgain

        alpha = 1.0 - float(np.clip(self.temporal_damping, 0.0, 1.0))
        r_prev, b_prev = float(self.prev_gains[0]), float(self.prev_gains[1])
        rgain = alpha * rgain + (1.0 - alpha) * r_prev
        bgain = alpha * bgain + (1.0 - alpha) * b_prev
        return rgain, bgain

    # ------------------------------------------------------------------
    # Public execute
    # ------------------------------------------------------------------

    def execute(self):
        """
        Execute Auto White Balance Module.

        Returns
        -------
        np.ndarray([r_gain, b_gain])  or  None if disabled.
        """
        print("Auto White balancing = " + str(self.enable))

        if self.enable is True:
            start = time.time()
            rgain, bgain = self.determine_white_balance_gain()
            rgain, bgain = self._apply_temporal_damping(rgain, bgain)
            print(f"  Execution time: {time.time() - start:.3f}s")
            return np.array([rgain, bgain])
        return None
