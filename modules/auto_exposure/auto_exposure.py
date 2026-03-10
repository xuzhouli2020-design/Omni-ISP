"""
File: auto_exposure.py
Description: 3A-AE Runs the Auto exposure algorithm in a loop.
             Extended with industry-grade zone metering (Phase 4.2) and
             an exposure-triangle P-gain convergence controller (Phase 4.3).

Original algorithm (legacy mode):
  Skewness-based histogram AE — returns ±1/0 digital-gain feedback.

Industry-grade mode ("zone_metering"):
  1. AEMetering — computes metered scene brightness from Stats3A zone_means.
  2. AEController — allocates EV correction across (shutter_us, analog_gain_db,
     digital_gain) using a proportional controller, returns ExposureParams.

Mode selection:
  parm_ae["mode"]:
    "legacy"        — original skewness histogram feedback (default, backward compat)
    "zone_metering" — new zone metering + exposure triangle solver

Code / Paper Reference:
  - Original: https://www.atlantis-press.com/article/25875811.pdf
  - AE zone metering: standard camera AE textbooks
Author: 10xEngineers Pvt Ltd (original); Omni-ISP Phase 4.2/4.3 extensions
------------------------------------------------------------
"""
import time
import numpy as np

from modules.auto_exposure.ae_metering   import AEMetering
from modules.auto_exposure.ae_controller import AEController, ExposureParams


class AutoExposure:
    """
    Auto Exposure Module.

    Parameters
    ----------
    img         : np.ndarray            Post-gamma RGB image (for legacy mode).
    sensor_info : dict                  Sensor metadata.
    parm_ae     : dict                  AE config section from configs.yml.
    stats       : Stats3A | None        Pre-computed 3A stats (zone_metering mode).
    prev_params : ExposureParams | None Previous frame's exposure params (warm-start).
    """

    def __init__(self, img, sensor_info, parm_ae,
                 stats=None, prev_params=None):
        self.img        = img
        self.enable     = parm_ae["is_enable"]
        self.is_debug   = parm_ae.get("is_debug", False)
        self.sensor_info = sensor_info
        self.parm_ae    = parm_ae
        self.bit_depth  = sensor_info["bit_depth"]
        self.mode       = str(parm_ae.get("mode", "legacy"))

        # Legacy parameters (always parsed for backward compat)
        self.center_illuminance       = parm_ae.get("center_illuminance", 128)
        self.histogram_skewness_range = parm_ae.get("histogram_skewness", 0.5)

        # Phase 4.2/4.3 state
        self.stats       = stats
        self.prev_params = prev_params   # ExposureParams or None

    # ------------------------------------------------------------------
    # Public execute
    # ------------------------------------------------------------------

    def execute(self):
        """
        Execute Auto Exposure.

        Returns
        -------
        In legacy mode : int  (+1 too dark / 0 OK / -1 too bright)
        In zone_metering mode : ExposureParams dataclass with full metadata.
        Returns None when disabled.
        """
        print("Auto Exposure= " + str(self.enable))

        if not self.enable:
            return None

        start = time.time()

        if self.mode == "zone_metering" and self.stats is not None:
            result = self._zone_metering_step()
        else:
            result = self._legacy_step()

        print(f"  Execution time: {time.time() - start:.3f}s")
        return result

    # ------------------------------------------------------------------
    # Zone metering path (Phase 4.2 / 4.3)
    # ------------------------------------------------------------------

    def _zone_metering_step(self) -> ExposureParams:
        """Industry-grade zone metering + exposure triangle convergence."""
        parm = self.parm_ae

        # Step 1: measure scene brightness
        metering = AEMetering(
            stats           = self.stats,
            metering_mode   = str(parm.get("metering_mode",   "center_weighted")),
            spot_fraction   = float(parm.get("spot_fraction",   0.25)),
            highlight_weight= float(parm.get("highlight_weight", 2.0)),
        )
        measured = metering.measure()

        if self.is_debug:
            print(f"   - AE zone_metering: measured={measured:.4f}")

        # Step 2: solve exposure triangle
        controller = AEController(parm_ae=parm, prev_params=self.prev_params)
        result = controller.step(
            measured_brightness = measured,
            highlight_fraction  = float(self.stats.highlight_fraction),
        )

        if self.is_debug:
            print(f"   - AE result: shutter={result.shutter_us:.0f}µs  "
                  f"analog={result.analog_gain_db:.1f}dB  "
                  f"digital={result.digital_gain:.3f}  "
                  f"error_ev={result.error_ev:.3f}  "
                  f"converged={result.converged}")

        return result

    # ------------------------------------------------------------------
    # Legacy path (original Omni-ISP algorithm)
    # ------------------------------------------------------------------

    def _legacy_step(self) -> int:
        """Original skewness-histogram AE feedback (returns -1/0/+1)."""
        # Convert Image into 8-bit for AE Calculation
        img = self.img >> (self.bit_depth - 8)
        grey_img, avg_lum = self._get_greyscale_image(img)
        print("Average luminance is = ", avg_lum)
        skewness = self._get_luminance_histogram_skewness(grey_img)
        upper_limit =  self.histogram_skewness_range
        lower_limit = -self.histogram_skewness_range
        if self.is_debug:
            print("   - AE - Histogram Skewness Range = ", upper_limit)
        if skewness < lower_limit:
            return -1
        elif skewness > upper_limit:
            return 1
        return 0

    def _get_greyscale_image(self, img):
        grey_img = np.clip(
            np.dot(img[..., :3], [0.299, 0.587, 0.144]), 0, (2 ** 8)
        ).astype(np.uint16)
        return grey_img, np.average(grey_img, axis=(0, 1))

    def _get_luminance_histogram_skewness(self, img):
        img = img.astype(np.float64) - self.center_illuminance
        img_size = img.size
        m_2 = np.sum(np.power(img, 2)) / img_size
        m_3 = np.sum(np.power(img, 3)) / img_size
        g_1 = np.sqrt(img_size * (img_size - 1)) / (img_size - 2)
        skewness = np.nan_to_num((m_3 / abs(m_2) ** (3 / 2)) * g_1)
        if self.is_debug:
            print("   - AE - Histogram Skewness = ", skewness)
        return skewness
