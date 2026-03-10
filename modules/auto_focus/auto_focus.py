"""
File: auto_focus.py
Description: Auto-Focus pipeline module — wraps AFSearchMachine and focus-metric
             computation into the standard Omni-ISP module interface.

Role in pipeline
────────────────
AF is a *feedback* module: it does not modify the image.  Instead, it:

  1. Computes or re-uses the Tenengrad focus metric from Stats3A (if provided).
  2. Updates the stateful AFSearchMachine with that metric.
  3. Exposes the recommended lens position / direction as metadata.

The host application (or simulator) is responsible for physically moving the
lens to the recommended position before the next frame is captured.

Persistence
───────────
``AFSearchMachine`` must survive across pipeline execute() calls.  The caller
should store the ``AutoFocus`` instance (or its ``af_machine`` attribute) and
pass it back via the ``af_state`` constructor argument on subsequent frames.

  first_frame:
      af_module = AutoFocus(img, sensor_info, parm_af, stats=stats3a)
      result = af_module.execute()
      machine = af_module.af_machine       # persist this

  next_frame:
      af_module = AutoFocus(img, sensor_info, parm_af, stats=stats3a,
                            af_state=machine)
      result = af_module.execute()

Parameters (from configs.yml `auto_focus` section)
────────────────────────────────────────────────────
  is_enable        : bool   Enable AF (default False — passive ISP mode)
  lens_range       : int    Total actuator steps (0=∞, max=macro) [100]
  coarse_steps     : int    Positions in coarse sweep [10]
  fine_steps       : int    Half-steps around coarse peak [5]
  tracking_threshold: float Relative metric drop to re-trigger [0.05]
  metric           : str    Focus metric: "tenengrad" | "laplacian_var" ["tenengrad"]
  trigger_on_start : bool   Auto-trigger a search when AF is first enabled [True]

Phase 4.8
"""

from typing import Optional

from modules.auto_focus.af_metric import compute_focus_metric
from modules.auto_focus.af_search import AFSearchMachine, AFResult, AFState


class AutoFocus:
    """
    Stateless pipeline wrapper around the stateful AFSearchMachine.

    The image is returned unchanged; AF output is in ``self.result``.

    Parameters
    ----------
    img         : np.ndarray  Current frame (passed through unchanged).
    sensor_info : dict        Sensor metadata (bit_depth, bayer_pattern, etc.)
    parm_af     : dict        ``auto_focus`` config section from configs.yml.
    stats       : Stats3A | None
                  Pre-computed 3A statistics.  If provided and the Stats3A
                  ``focus_metric`` is non-zero, it is used directly (avoids
                  re-computing Tenengrad).  If None, the module re-computes
                  the metric from the full frame green channel (less ideal but
                  still functional).
    af_state    : AFSearchMachine | None
                  Persisted state machine from the previous frame.  Pass None
                  on the first frame — the module will create a new machine.
    """

    def __init__(self,
                 img,
                 sensor_info: dict,
                 parm_af: dict,
                 stats=None,
                 af_state: Optional[AFSearchMachine] = None):
        self.img         = img
        self.sensor_info = sensor_info
        self.enable      = bool(parm_af.get("is_enable", False))

        # AF parameters
        self.lens_range          = int(parm_af.get("lens_range",          100))
        self.coarse_steps        = int(parm_af.get("coarse_steps",         10))
        self.fine_steps          = int(parm_af.get("fine_steps",            5))
        self.tracking_threshold  = float(parm_af.get("tracking_threshold", 0.05))
        self.metric_name         = str(parm_af.get("metric",        "tenengrad"))
        self.trigger_on_start    = bool(parm_af.get("trigger_on_start",   True))

        # Stats3A (optional fast path)
        self.stats = stats

        # Restore or create the state machine
        if af_state is not None:
            self.af_machine: AFSearchMachine = af_state
        else:
            self.af_machine = AFSearchMachine(
                lens_range          = self.lens_range,
                coarse_steps        = self.coarse_steps,
                fine_steps          = self.fine_steps,
                tracking_threshold  = self.tracking_threshold,
            )
            if self.enable and self.trigger_on_start:
                self.af_machine.trigger()

        # Output — populated by execute()
        self.result: Optional[AFResult] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(self):
        """
        Run one AF step for the current frame.

        Returns
        -------
        np.ndarray  The input image, unchanged.
        """
        if not self.enable:
            return self.img

        metric = self._get_focus_metric()
        self.result = self.af_machine.update(metric)

        self._log_result()
        return self.img

    def trigger(self) -> None:
        """
        Manually trigger a new focus search (e.g. on scene change detected
        externally, or on user request).
        """
        self.af_machine.trigger()

    @property
    def recommended_position(self) -> int:
        """Current recommended lens position (actuator steps)."""
        return self.af_machine.position

    @property
    def converged(self) -> bool:
        """True once fine search has completed and lens is at best position."""
        if self.result is None:
            return False
        return self.result.converged

    @property
    def af_state_name(self) -> str:
        """Current AF state as a string (idle/coarse/fine/tracking)."""
        return self.af_machine.state.value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_focus_metric(self) -> float:
        """
        Return focus metric for this frame.

        Priority:
          1. stats.focus_metric if stats provided and metric > 0
          2. Recompute from stats.focus_roi_g if available
          3. Recompute from full green channel (fallback)
        """
        if self.stats is not None:
            if self.stats.focus_metric > 0:
                return float(self.stats.focus_metric)
            # Stats were computed but metric = 0 — try using the ROI image
            if self.stats.focus_roi_g is not None and self.stats.focus_roi_g.size >= 9:
                return compute_focus_metric(self.stats.focus_roi_g, self.metric_name)

        # Fallback: compute from the raw Bayer image green channel
        # This is less efficient but keeps AF functional without Stats3A
        return self._compute_from_raw()

    def _compute_from_raw(self) -> float:
        """Compute focus metric from raw Bayer frame (fallback path)."""
        import numpy as np
        from util.bayer_utils import extract_channels

        bayer_pat = self.sensor_info.get("bayer_pattern", "rggb")
        bit_depth  = self.sensor_info.get("bit_depth",      10)
        maxval     = float((1 << bit_depth) - 1)

        try:
            channels = extract_channels(self.img, bayer_pat)
            g_norm   = (channels["gr"].astype(np.float32)
                        + channels["gb"].astype(np.float32)) / (2.0 * maxval)
            return compute_focus_metric(g_norm, self.metric_name)
        except Exception:
            return 0.0

    def _log_result(self) -> None:
        if self.result is None:
            return
        r = self.result
        print(
            f"  AF [{r.state:8s}]  metric={r.focus_metric:.4f}  "
            f"best={r.best_metric:.4f}  pos={r.recommended_position:3d}  "
            f"dir={r.direction:+d}  converged={r.converged}"
        )
