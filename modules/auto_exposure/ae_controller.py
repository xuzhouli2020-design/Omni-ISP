"""
File: ae_controller.py
Description: Industry-grade AE exposure-triangle solver with P-gain convergence
             controller and flicker-avoidance shutter quantisation.

Design
──────
The controller manages three gain degrees of freedom in priority order:

  1. Shutter speed (shutter_us)    — cleanest: no noise penalty
  2. Analog gain   (analog_gain_db)— moderate noise but before ADC
  3. Digital gain  (digital_gain)  — last resort: multiplies quantisation noise

Exposure value (EV) arithmetic is done in log₂ space so that equal steps
correspond to equal perceptual doubling of brightness.

Convergence (P-controller)
──────────────────────────
  error_ev = log2(target / measured)   [measured is the metered brightness]
  delta_ev  = Kp × error_ev            [Kp = proportional gain, typically 0.5]

  If |error_ev| < tolerance_ev the controller holds current settings.

Flicker avoidance
─────────────────
  Under mains-lit scenes (50 Hz or 60 Hz fluorescent), shutter must be a
  multiple of 1/(2×f) to prevent banding.  The controller quantises the
  computed shutter to the nearest allowed step.

  flicker_avoidance: "none" | "50hz" | "60hz"   (from config)

  50 Hz → allowed shutters: multiples of 10 ms  (1/100 s)
  60 Hz → allowed shutters: multiples of 8.33 ms (1/120 s)

Exposure priority modes
───────────────────────
  "auto"    — full triangle; shutter → analog → digital
  "shutter" — shutter priority (fixed by config); apply analog+digital
  "iso"     — ISO/analog priority (fixed); apply shutter+digital

Parameters (from configs.yml ae section)
──────────────────────────────────────────
  target_brightness     : float  Desired scene brightness [0,1]         [0.18]
  tolerance_ev          : float  Dead-band ± EV before correction starts [0.15]
  kp                    : float  Proportional gain                       [0.5]
  shutter_min_us        : float  Minimum shutter in microseconds         [100]
  shutter_max_us        : float  Maximum shutter                         [33333]
  analog_gain_min_db    : float  Minimum analog gain (0 dB = ×1)         [0.0]
  analog_gain_max_db    : float  Maximum analog gain                      [24.0]
  digital_gain_max      : float  Maximum digital gain multiplier          [4.0]
  flicker_avoidance     : str    "none" | "50hz" | "60hz"                ["none"]
  exposure_priority     : str    "auto" | "shutter" | "iso"              ["auto"]
  shutter_fixed_us      : float  Used when priority="shutter"            [10000]
  analog_gain_fixed_db  : float  Used when priority="iso"                [0.0]

Phase 4.3
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExposureParams:
    """Recommended exposure settings for the next frame."""
    shutter_us:      float   # Integration time in microseconds
    analog_gain_db:  float   # Analog (sensor) gain in dB  (0 dB = ×1)
    digital_gain:    float   # Digital gain multiplier (1.0 = unity)
    # Diagnostics
    measured_brightness: float   # Scene brightness used for this decision
    error_ev:        float        # Signed EV error (log2 scale)
    converged:       bool         # True if |error_ev| < tolerance_ev


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _db_to_linear(db: float) -> float:
    """Convert dB gain to linear multiplier (power/voltage convention)."""
    return 10.0 ** (db / 20.0)


def _linear_to_db(linear: float) -> float:
    """Convert linear gain multiplier to dB."""
    return 20.0 * np.log10(max(linear, 1e-6))


def _ev_error(measured: float, target: float) -> float:
    """Signed EV error in log₂ space.  Positive = too dark, negative = too bright."""
    if measured < 1e-6:
        return 0.0
    return float(np.log2(target / measured))


def _apply_flicker(shutter_us: float, mode: str) -> float:
    """Quantise shutter to nearest flicker-safe multiple."""
    if mode == "50hz":
        step_us = 10_000.0        # 1/(2×50) = 10 ms
    elif mode == "60hz":
        step_us = 1_000_000.0 / 120.0   # 8.333 ms
    else:
        return shutter_us
    if step_us <= 0:
        return shutter_us
    quantised = round(shutter_us / step_us) * step_us
    return max(step_us, quantised)   # must be at least one step


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class AEController:
    """
    Proportional AE controller.  Stateless — call step() once per frame.

    Parameters
    ----------
    parm_ae     : dict                AE config section from configs.yml.
    prev_params : ExposureParams | None  Previous frame's exposure (warm-start).
    """

    # Default safe starting exposure (dim indoor scene)
    _DEFAULT_SHUTTER_US     = 10_000.0   # 10 ms
    _DEFAULT_ANALOG_GAIN_DB = 0.0        # ×1
    _DEFAULT_DIGITAL_GAIN   = 1.0

    def __init__(self, parm_ae: dict, prev_params: Optional[ExposureParams] = None):
        self.target_brightness   = float(parm_ae.get("target_brightness",   0.18))
        self.tolerance_ev        = float(parm_ae.get("tolerance_ev",         0.15))
        self.kp                  = float(parm_ae.get("kp",                   0.50))
        self.shutter_min_us      = float(parm_ae.get("shutter_min_us",      100.0))
        self.shutter_max_us      = float(parm_ae.get("shutter_max_us",    33333.0))
        self.analog_gain_min_db  = float(parm_ae.get("analog_gain_min_db",   0.0))
        self.analog_gain_max_db  = float(parm_ae.get("analog_gain_max_db",  24.0))
        self.digital_gain_max    = float(parm_ae.get("digital_gain_max",     4.0))
        self.flicker_mode        = str(parm_ae.get("flicker_avoidance",    "none"))
        self.priority            = str(parm_ae.get("exposure_priority",    "auto"))
        self.shutter_fixed_us    = float(parm_ae.get("shutter_fixed_us",  10000.0))
        self.analog_gain_fixed_db= float(parm_ae.get("analog_gain_fixed_db",  0.0))

        # Restore or initialise state
        if prev_params is not None:
            self.prev = prev_params
        else:
            self.prev = ExposureParams(
                shutter_us          = self._DEFAULT_SHUTTER_US,
                analog_gain_db      = self._DEFAULT_ANALOG_GAIN_DB,
                digital_gain        = self._DEFAULT_DIGITAL_GAIN,
                measured_brightness = self.target_brightness,
                error_ev            = 0.0,
                converged           = False,
            )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def step(self, measured_brightness: float,
             highlight_fraction: float = 0.0) -> ExposureParams:
        """
        Compute new exposure parameters for the next frame.

        Parameters
        ----------
        measured_brightness : float  Metered scene brightness [0, 1].
        highlight_fraction  : float  Fraction of blown highlights [0, 1].
                                     When high, forces underexposure correction.

        Returns
        -------
        ExposureParams  — updated exposure triangle + diagnostics.
        """
        # Effective target: lower slightly when highlights are clipping
        effective_target = self.target_brightness * (1.0 - 0.5 * highlight_fraction)

        error_ev  = _ev_error(measured_brightness, effective_target)
        converged = abs(error_ev) < self.tolerance_ev

        if converged:
            # Hold current exposure — just update diagnostics
            return ExposureParams(
                shutter_us          = self.prev.shutter_us,
                analog_gain_db      = self.prev.analog_gain_db,
                digital_gain        = self.prev.digital_gain,
                measured_brightness = measured_brightness,
                error_ev            = error_ev,
                converged           = True,
            )

        # Proportional correction step
        delta_ev = self.kp * error_ev
        new_params = self._allocate_ev(delta_ev)

        result = ExposureParams(
            shutter_us          = new_params[0],
            analog_gain_db      = new_params[1],
            digital_gain        = new_params[2],
            measured_brightness = measured_brightness,
            error_ev            = error_ev,
            converged           = False,
        )
        self.prev = result
        return result

    # ------------------------------------------------------------------
    # EV allocation across the exposure triangle
    # ------------------------------------------------------------------

    def _allocate_ev(self, delta_ev: float):
        """
        Allocate delta_ev (signed) across the exposure triangle in priority order:
          1. Shutter (no noise penalty)
          2. Analog gain (moderate noise)
          3. Digital gain (last resort)

        Priority modes:
          "auto"    — full priority cascade
          "shutter" — shutter is fixed; distribute to analog+digital only
          "iso"     — analog gain is fixed; distribute to shutter+digital only

        Returns (shutter_us, analog_gain_db, digital_gain).
        """
        shutter_us      = self.prev.shutter_us
        analog_gain_db  = self.prev.analog_gain_db
        digital_gain    = self.prev.digital_gain

        remaining_ev = delta_ev

        # ---- Shutter ----
        if self.priority == "shutter":
            shutter_us = self.shutter_fixed_us
        else:
            shutter_us, remaining_ev = self._adjust_shutter(shutter_us, remaining_ev)
            shutter_us = _apply_flicker(shutter_us, self.flicker_mode)

        # ---- Analog gain ----
        if self.priority == "iso":
            analog_gain_db = self.analog_gain_fixed_db
        else:
            analog_gain_db, remaining_ev = self._adjust_analog(analog_gain_db, remaining_ev)

        # ---- Digital gain (last resort) ----
        digital_gain = self._adjust_digital(digital_gain, remaining_ev)

        return shutter_us, analog_gain_db, digital_gain

    def _adjust_shutter(self, current_us: float, delta_ev: float):
        """Apply delta_ev to shutter; return (new_shutter, leftover_ev)."""
        new_us = current_us * (2.0 ** delta_ev)
        new_us = float(np.clip(new_us, self.shutter_min_us, self.shutter_max_us))
        consumed_ev = np.log2(new_us / current_us) if current_us > 0 else 0.0
        leftover_ev = delta_ev - consumed_ev
        return new_us, float(leftover_ev)

    def _adjust_analog(self, current_db: float, delta_ev: float):
        """Apply delta_ev to analog gain; return (new_db, leftover_ev)."""
        # delta_ev stops → delta_db = delta_ev × 20*log10(2) ≈ 6.02 dB/stop
        db_per_ev    = 20.0 * np.log10(2.0)   # ~6.02 dB
        new_db       = current_db + delta_ev * db_per_ev
        new_db       = float(np.clip(new_db,
                                     self.analog_gain_min_db,
                                     self.analog_gain_max_db))
        consumed_ev  = (new_db - current_db) / db_per_ev
        leftover_ev  = delta_ev - consumed_ev
        return new_db, float(leftover_ev)

    def _adjust_digital(self, current_gain: float, delta_ev: float) -> float:
        """Apply remaining delta_ev to digital gain (clamped)."""
        new_gain = current_gain * (2.0 ** delta_ev)
        new_gain = float(np.clip(new_gain, 1.0, self.digital_gain_max))
        return new_gain
