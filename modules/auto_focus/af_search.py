"""
File: af_search.py
Description: Contrast-detect AF search strategy — stateful machine that
             decides lens movement across successive frames.

State machine
─────────────
  IDLE         → waiting for trigger (scene change or user request)
  COARSE_SWEEP → step through full lens range, record focus metric at each stop
  FINE_SEARCH  → hill-climb around the coarse peak with half-steps
  TRACKING     → monitor; re-trigger if metric drops by tracking_threshold

The caller is responsible for:
  1. Moving the physical lens to the recommended position after each frame.
  2. Calling update() with the focus metric measured from the NEW frame.
  3. Reading direction / recommended_position to command the actuator.

Lens position convention: integer steps, 0 = infinity, lens_range = macro.

Phase 4.7
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import numpy as np


class AFState(Enum):
    IDLE         = "idle"
    COARSE_SWEEP = "coarse"
    FINE_SEARCH  = "fine"
    TRACKING     = "tracking"


@dataclass
class AFResult:
    """Output produced each frame by the AF state machine."""
    focus_metric: float           # metric this frame
    best_metric: float            # best metric seen so far in this search
    recommended_position: int     # lens step to move to (host responsibility)
    direction: int                # -1=near, 0=hold, +1=far
    converged: bool
    state: str                    # current state name


class AFSearchMachine:
    """
    Stateful contrast-detect AF search machine.

    Instantiate once and call update() for every new frame.
    Persist this object across pipeline execute() calls in InfiniteISP.

    Parameters
    ----------
    lens_range          : int    Total actuator steps (0 = ∞, max = macro).
    coarse_steps        : int    Number of positions in coarse sweep.
    fine_steps          : int    Half-steps around coarse peak for fine search.
    tracking_threshold  : float  Relative metric drop to re-trigger search.
    """

    def __init__(self,
                 lens_range: int          = 100,
                 coarse_steps: int        = 10,
                 fine_steps: int          = 5,
                 tracking_threshold: float = 0.05):
        self.lens_range           = int(lens_range)
        self.coarse_steps         = int(coarse_steps)
        self.fine_steps           = int(fine_steps)
        self.tracking_threshold   = float(tracking_threshold)

        # State
        self._state               = AFState.IDLE
        self._position            = 0               # current lens position
        self._best_metric         = 0.0
        self._best_position       = 0
        self._sweep_positions: List[int] = []       # planned positions
        self._sweep_metrics: List[float] = []       # collected metrics
        self._sweep_idx           = 0               # next position to visit
        self._tracking_ref        = 0.0             # metric when tracking started
        self._converged           = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def state(self) -> AFState:
        return self._state

    @property
    def position(self) -> int:
        return self._position

    def trigger(self) -> None:
        """Start a new focus search (e.g. on scene change or user request)."""
        self._start_coarse()

    def update(self, focus_metric: float) -> AFResult:
        """
        Process one new frame's focus metric and decide next action.

        Parameters
        ----------
        focus_metric : float  Metric computed from the frame at the lens
                              position previously recommended.

        Returns
        -------
        AFResult — read recommended_position and direction to command lens.
        """
        if self._state == AFState.IDLE:
            return self._idle_result(focus_metric)
        elif self._state == AFState.COARSE_SWEEP:
            return self._step_coarse(focus_metric)
        elif self._state == AFState.FINE_SEARCH:
            return self._step_fine(focus_metric)
        elif self._state == AFState.TRACKING:
            return self._step_tracking(focus_metric)
        else:
            return self._idle_result(focus_metric)

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _start_coarse(self) -> None:
        step = max(1, self.lens_range // self.coarse_steps)
        self._sweep_positions = list(
            range(0, self.lens_range + 1, step)
        )
        if self._sweep_positions[-1] != self.lens_range:
            self._sweep_positions.append(self.lens_range)
        self._sweep_metrics  = []
        self._sweep_idx      = 0
        self._best_metric    = 0.0
        self._best_position  = self._sweep_positions[0]
        self._converged      = False
        self._state          = AFState.COARSE_SWEEP
        # Move to first position immediately
        self._position = self._sweep_positions[0]

    def _step_coarse(self, metric: float) -> AFResult:
        # Record metric for the position we just visited
        self._sweep_metrics.append(metric)
        if metric > self._best_metric:
            self._best_metric   = metric
            self._best_position = self._position

        self._sweep_idx += 1
        if self._sweep_idx < len(self._sweep_positions):
            # Move to next coarse position
            next_pos = self._sweep_positions[self._sweep_idx]
            direction = 1 if next_pos > self._position else -1
            self._position = next_pos
            return AFResult(
                focus_metric=metric,
                best_metric=self._best_metric,
                recommended_position=self._position,
                direction=direction,
                converged=False,
                state=self._state.value,
            )
        else:
            # Coarse sweep complete → start fine search around best position
            return self._start_fine(metric)

    def _start_fine(self, last_metric: float) -> AFResult:
        """Transition to fine search around best coarse position."""
        half_step = max(
            1,
            (self.lens_range // self.coarse_steps) // 2
        )
        center = self._best_position
        self._sweep_positions = []
        for k in range(-self.fine_steps, self.fine_steps + 1):
            pos = int(np.clip(center + k * half_step, 0, self.lens_range))
            if not self._sweep_positions or pos != self._sweep_positions[-1]:
                self._sweep_positions.append(pos)
        self._sweep_metrics  = []
        self._sweep_idx      = 0
        self._state          = AFState.FINE_SEARCH
        self._position       = self._sweep_positions[0]
        return AFResult(
            focus_metric=last_metric,
            best_metric=self._best_metric,
            recommended_position=self._position,
            direction=0,
            converged=False,
            state=self._state.value,
        )

    def _step_fine(self, metric: float) -> AFResult:
        self._sweep_metrics.append(metric)
        if metric > self._best_metric:
            self._best_metric   = metric
            self._best_position = self._position

        self._sweep_idx += 1
        if self._sweep_idx < len(self._sweep_positions):
            next_pos = self._sweep_positions[self._sweep_idx]
            direction = 1 if next_pos > self._position else -1
            self._position = next_pos
            return AFResult(
                focus_metric=metric,
                best_metric=self._best_metric,
                recommended_position=self._position,
                direction=direction,
                converged=False,
                state=self._state.value,
            )
        else:
            # Fine search complete → move to best and start tracking
            self._position = self._best_position
            self._tracking_ref = self._best_metric
            self._converged    = True
            self._state        = AFState.TRACKING
            return AFResult(
                focus_metric=metric,
                best_metric=self._best_metric,
                recommended_position=self._position,
                direction=0,
                converged=True,
                state=self._state.value,
            )

    def _step_tracking(self, metric: float) -> AFResult:
        if self._tracking_ref > 0:
            relative_drop = (self._tracking_ref - metric) / self._tracking_ref
        else:
            relative_drop = 0.0

        if relative_drop > self.tracking_threshold:
            # Focus lost — re-trigger coarse sweep
            self._start_coarse()
            return AFResult(
                focus_metric=metric,
                best_metric=self._best_metric,
                recommended_position=self._position,
                direction=0,
                converged=False,
                state=self._state.value,
            )
        else:
            # Still in focus — hold position
            return AFResult(
                focus_metric=metric,
                best_metric=self._best_metric,
                recommended_position=self._position,
                direction=0,
                converged=True,
                state=self._state.value,
            )

    def _idle_result(self, metric: float) -> AFResult:
        return AFResult(
            focus_metric=metric,
            best_metric=self._best_metric,
            recommended_position=self._position,
            direction=0,
            converged=self._converged,
            state=AFState.IDLE.value,
        )
