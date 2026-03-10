"""
EdgeISP — Phase 4: 3A Algorithm Tests
=======================================
Covers:
  4.1  Stats3ACollector
  4.2  AE zone metering (AEMetering)
  4.3  AE controller / exposure triangle (AEController)
  4.4  Gray Edge AWB
  4.5  AWB temporal damping
  4.6  AF focus metrics
  4.7  AF search state machine (AFSearchMachine)
  4.8  AutoFocus module wrapper

Run from the project root:
    python -m unittest tests.test_3a -v
    python tests/test_3a.py
"""

import sys
import os
import unittest
from unittest.mock import MagicMock
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub heavy/unavailable deps so tests run offline
for _mod in ("scipy", "scipy.signal", "rawpy", "tqdm", "cv2"):
    sys.modules.setdefault(_mod, MagicMock())
sys.modules.setdefault("util.utils", MagicMock())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw(H=64, W=64, bit_depth=10, fill=None):
    rng = np.random.default_rng(42)
    maxval = (1 << bit_depth) - 1
    if fill is not None:
        return np.full((H, W), int(fill * maxval), dtype=np.uint16)
    return rng.integers(0, maxval, size=(H, W), dtype=np.uint16)


def _sensor_info(bit_depth=10, pattern="rggb"):
    return {"bit_depth": bit_depth, "bayer_pattern": pattern, "width": 64, "height": 64}


def _parm_3a():
    return {
        "is_enable": True,
        "zone_grid": 8,
        "hist_bins": 64,
        "af_roi": [0.25, 0.25, 0.75, 0.75],
        "grad_order": 1,
        "minkowski_norm": 1,
    }


# ---------------------------------------------------------------------------
# Fake Stats3A for modules that accept one
# ---------------------------------------------------------------------------
class _FakeStats:
    def __init__(self, brightness=0.5, Nz=8, highlight_fraction=0.0,
                 focus_metric=0.0, channel_grad_means=None):
        self.zone_means = np.full((Nz, Nz), brightness, dtype=np.float32)
        self.highlight_fraction = highlight_fraction
        self.focus_metric = focus_metric
        self.focus_roi_g = np.full((16, 16), brightness, dtype=np.float32)
        self.channel_grad_means = channel_grad_means or {
            "r": 0.01, "gr": 0.01, "gb": 0.01, "b": 0.01
        }


# ===========================================================================
# 4.1  Stats3ACollector
# ===========================================================================

class TestStats3ACollector(unittest.TestCase):

    def _make_collector(self, fill=None):
        from modules.stats_3a.stats_collector import Stats3ACollector
        raw = _make_raw(fill=fill)
        return Stats3ACollector(raw, _sensor_info(), _parm_3a())

    def test_execute_returns_stats3a(self):
        from modules.stats_3a.stats_collector import Stats3ACollector, Stats3A
        col = self._make_collector()
        stats = col.execute()
        self.assertIsInstance(stats, Stats3A)

    def test_zone_means_shape(self):
        col = self._make_collector()
        stats = col.execute()
        self.assertEqual(stats.zone_means.shape, (8, 8))

    def test_zone_means_range(self):
        col = self._make_collector()
        stats = col.execute()
        self.assertGreaterEqual(float(stats.zone_means.min()), 0.0)
        self.assertLessEqual(float(stats.zone_means.max()), 1.0)

    def test_channel_means_keys(self):
        col = self._make_collector()
        stats = col.execute()
        self.assertEqual(set(stats.channel_means.keys()), {"r", "gr", "gb", "b"})

    def test_channel_grad_means_keys(self):
        col = self._make_collector()
        stats = col.execute()
        self.assertEqual(set(stats.channel_grad_means.keys()), {"r", "gr", "gb", "b"})

    def test_focus_metric_positive_for_random_image(self):
        col = self._make_collector()
        stats = col.execute()
        self.assertGreater(stats.focus_metric, 0)

    def test_focus_metric_zero_for_flat_image(self):
        from modules.stats_3a.stats_collector import Stats3ACollector
        raw = np.full((64, 64), 512, dtype=np.uint16)
        col = Stats3ACollector(raw, _sensor_info(), _parm_3a())
        stats = col.execute()
        self.assertAlmostEqual(stats.focus_metric, 0.0, places=5)

    def test_highlight_fraction_bright_image(self):
        col = self._make_collector(fill=1.0)
        stats = col.execute()
        self.assertGreater(stats.highlight_fraction, 0.9)

    def test_highlight_fraction_dark_image(self):
        col = self._make_collector(fill=0.0)
        stats = col.execute()
        self.assertAlmostEqual(stats.highlight_fraction, 0.0, places=5)

    def test_global_histogram_shape(self):
        col = self._make_collector()
        stats = col.execute()
        self.assertEqual(stats.global_histogram.shape, (64,))

    def test_disabled_still_returns_stats(self):
        from modules.stats_3a.stats_collector import Stats3ACollector
        parm = _parm_3a()
        parm["is_enable"] = False
        raw = _make_raw()
        col = Stats3ACollector(raw, _sensor_info(), parm)
        stats = col.execute()
        self.assertIsNotNone(stats)


# ===========================================================================
# 4.2  AE Metering
# ===========================================================================

class TestAEMetering(unittest.TestCase):

    def test_center_weighted_uniform(self):
        from modules.auto_exposure.ae_metering import AEMetering
        m = AEMetering(_FakeStats(0.4), metering_mode="center_weighted")
        self.assertAlmostEqual(m.measure(), 0.4, delta=0.02)

    def test_spot_uniform(self):
        from modules.auto_exposure.ae_metering import AEMetering
        m = AEMetering(_FakeStats(0.3), metering_mode="spot")
        self.assertAlmostEqual(m.measure(), 0.3, delta=0.02)

    def test_matrix_returns_in_range(self):
        from modules.auto_exposure.ae_metering import AEMetering
        m = AEMetering(_FakeStats(0.6), metering_mode="matrix")
        result = m.measure()
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_center_weighted_brighter_center(self):
        from modules.auto_exposure.ae_metering import AEMetering
        stats = _FakeStats(0.1)
        stats.zone_means[3:5, 3:5] = 0.9
        m = AEMetering(stats, metering_mode="center_weighted")
        # Should be pulled up by bright center
        self.assertGreater(m.measure(), 0.1)

    def test_all_modes_return_unit_range(self):
        from modules.auto_exposure.ae_metering import AEMetering
        for mode in ("center_weighted", "spot", "matrix"):
            m = AEMetering(_FakeStats(0.5), metering_mode=mode)
            result = m.measure()
            self.assertGreaterEqual(result, 0.0, msg=f"{mode} below 0")
            self.assertLessEqual(result, 1.0, msg=f"{mode} above 1")


# ===========================================================================
# 4.3  AE Controller
# ===========================================================================

def _ae_parm(**overrides):
    p = {
        "target_brightness": 0.18,
        "tolerance_ev": 0.15,
        "kp": 0.5,
        "shutter_min_us": 100.0,
        "shutter_max_us": 33333.0,
        "analog_gain_min_db": 0.0,
        "analog_gain_max_db": 24.0,
        "digital_gain_max": 4.0,
        "flicker_avoidance": "none",
        "exposure_priority": "auto",
        "shutter_fixed_us": 10000.0,
        "analog_gain_fixed_db": 0.0,
    }
    p.update(overrides)
    return p


class TestAEController(unittest.TestCase):

    def test_returns_exposure_params(self):
        from modules.auto_exposure.ae_controller import AEController, ExposureParams
        ctrl = AEController(_ae_parm())
        result = ctrl.step(0.10)
        self.assertIsInstance(result, ExposureParams)

    def test_converged_on_target(self):
        from modules.auto_exposure.ae_controller import AEController
        ctrl = AEController(_ae_parm(tolerance_ev=0.5))
        result = ctrl.step(0.18)
        self.assertTrue(result.converged)

    def test_not_converged_far_from_target(self):
        from modules.auto_exposure.ae_controller import AEController
        ctrl = AEController(_ae_parm())
        result = ctrl.step(0.02)
        self.assertFalse(result.converged)

    def test_positive_error_when_dark(self):
        from modules.auto_exposure.ae_controller import AEController
        ctrl = AEController(_ae_parm())
        result = ctrl.step(0.05)
        self.assertGreater(result.error_ev, 0)

    def test_negative_error_when_bright(self):
        from modules.auto_exposure.ae_controller import AEController
        ctrl = AEController(_ae_parm())
        result = ctrl.step(0.90)
        self.assertLess(result.error_ev, 0)

    def test_shutter_increases_for_dark_scene(self):
        from modules.auto_exposure.ae_controller import AEController
        ctrl = AEController(_ae_parm())
        default_shutter = ctrl.prev.shutter_us
        result = ctrl.step(0.05)
        self.assertGreaterEqual(result.shutter_us, default_shutter)

    def test_shutter_clamped_to_min(self):
        from modules.auto_exposure.ae_controller import AEController
        ctrl = AEController(_ae_parm(shutter_min_us=500.0))
        result = ctrl.step(0.99)
        self.assertGreaterEqual(result.shutter_us, 500.0)

    def test_flicker_50hz_quantises_shutter(self):
        from modules.auto_exposure.ae_controller import AEController
        ctrl = AEController(_ae_parm(flicker_avoidance="50hz"))
        result = ctrl.step(0.05)
        remainder = result.shutter_us % 10000.0
        self.assertAlmostEqual(remainder, 0.0, delta=1.0)

    def test_digital_gain_minimum_unity(self):
        from modules.auto_exposure.ae_controller import AEController
        ctrl = AEController(_ae_parm())
        result = ctrl.step(0.18)
        self.assertGreaterEqual(result.digital_gain, 1.0)

    def test_analog_gain_within_bounds(self):
        from modules.auto_exposure.ae_controller import AEController
        ctrl = AEController(_ae_parm())
        result = ctrl.step(0.001)
        self.assertGreaterEqual(result.analog_gain_db, 0.0)
        self.assertLessEqual(result.analog_gain_db, 24.0)

    def test_highlight_penalty_shifts_error(self):
        from modules.auto_exposure.ae_controller import AEController
        parm = _ae_parm()
        r_no = AEController(parm).step(0.18, highlight_fraction=0.0)
        r_hi = AEController(parm).step(0.18, highlight_fraction=0.8)
        self.assertLess(r_hi.error_ev, r_no.error_ev)


# ===========================================================================
# 4.4  Gray Edge AWB
# ===========================================================================

def _make_channels(r_val=0.5, g_val=0.5, b_val=0.5, H=32, W=32, bit_depth=10):
    rng = np.random.default_rng(17)
    maxval = (1 << bit_depth) - 1
    def _ch(v):
        base = np.full((H, W), int(v * maxval), dtype=np.uint16)
        noise = rng.integers(-10, 10, size=(H, W))
        return np.clip(base.astype(np.int32) + noise, 0, maxval).astype(np.uint16)
    return {"r": _ch(r_val), "gr": _ch(g_val), "gb": _ch(g_val), "b": _ch(b_val)}


class TestGrayEdge(unittest.TestCase):

    def test_neutral_gains_near_unity(self):
        from modules.auto_white_balance.gray_edge import GrayEdge
        chs = _make_channels(r_val=0.5, g_val=0.5, b_val=0.5)
        r_gain, b_gain = GrayEdge(chs, bit_depth=10).calculate_gains()
        self.assertAlmostEqual(r_gain, 1.0, delta=0.2)
        self.assertAlmostEqual(b_gain, 1.0, delta=0.2)

    def test_gains_within_bounds(self):
        from modules.auto_white_balance.gray_edge import GrayEdge
        chs = _make_channels()
        r_gain, b_gain = GrayEdge(chs, bit_depth=10,
                                   min_gain=1.0, max_gain=4.0).calculate_gains()
        self.assertGreaterEqual(r_gain, 1.0)
        self.assertLessEqual(r_gain, 4.0)
        self.assertGreaterEqual(b_gain, 1.0)
        self.assertLessEqual(b_gain, 4.0)

    def test_laplacian_order_runs(self):
        from modules.auto_white_balance.gray_edge import GrayEdge
        chs = _make_channels()
        r_gain, b_gain = GrayEdge(chs, bit_depth=10, grad_order=2).calculate_gains()
        self.assertGreaterEqual(r_gain, 1.0)
        self.assertGreaterEqual(b_gain, 1.0)

    def test_minkowski_p2_runs(self):
        from modules.auto_white_balance.gray_edge import GrayEdge
        chs = _make_channels()
        r_gain, b_gain = GrayEdge(chs, bit_depth=10, p=2.0).calculate_gains()
        self.assertTrue(math.isfinite(r_gain))
        self.assertTrue(math.isfinite(b_gain))

    def test_textured_g_flat_b_gives_high_b_gain(self):
        """When G has much higher gradient than B, b_gain should be > 1."""
        from modules.auto_white_balance.gray_edge import GrayEdge
        H, W, bd = 32, 32, 10
        maxval = (1 << bd) - 1
        # Create textured G channels (sinusoidal) and flat B
        xs = np.linspace(0, 4 * np.pi, W)
        ys = np.linspace(0, 4 * np.pi, H)
        XX, YY = np.meshgrid(xs, ys)
        texture = ((np.sin(XX) + np.sin(YY)) * 0.2 + 0.5)
        g_ch = np.clip(texture * maxval, 0, maxval).astype(np.uint16)
        b_ch = np.full((H, W), int(0.3 * maxval), dtype=np.uint16)
        r_ch = g_ch.copy()
        chs = {"r": r_ch, "gr": g_ch, "gb": g_ch, "b": b_ch}
        _, b_gain = GrayEdge(chs, bit_depth=bd, max_gain=8.0).calculate_gains()
        self.assertGreater(b_gain, 1.0)


# ===========================================================================
# 4.5  AWB Temporal Damping
# ===========================================================================

class TestAWBTemporalDamping(unittest.TestCase):

    def _parm(self, damping=0.5, algo="gray_world"):
        return {
            "is_enable": True,
            "is_debug": False,
            "underexposed_percentage": 5,
            "overexposed_percentage": 5,
            "algorithm": algo,
            "percentage": 3.5,
            "temporal_damping": damping,
        }

    def test_no_damping_returns_gains(self):
        from modules.auto_white_balance.auto_white_balance import AutoWhiteBalance
        raw = _make_raw()
        awb = AutoWhiteBalance(raw, _sensor_info(), self._parm(damping=0.0))
        gains = awb.execute()
        self.assertIsNotNone(gains)

    def test_heavy_damping_stays_near_prev(self):
        from modules.auto_white_balance.auto_white_balance import AutoWhiteBalance
        raw = _make_raw()
        prev = np.array([2.5, 3.0])
        awb = AutoWhiteBalance(raw, _sensor_info(), self._parm(damping=0.9),
                               prev_gains=prev)
        gains = awb.execute()
        self.assertIsNotNone(gains)
        self.assertLess(abs(gains[0] - prev[0]), 0.5)
        self.assertLess(abs(gains[1] - prev[1]), 0.5)

    def test_disabled_returns_none(self):
        from modules.auto_white_balance.auto_white_balance import AutoWhiteBalance
        raw = _make_raw()
        parm = self._parm()
        parm["is_enable"] = False
        awb = AutoWhiteBalance(raw, _sensor_info(), parm)
        self.assertIsNone(awb.execute())

    def test_gray_edge_algorithm_runs(self):
        from modules.auto_white_balance.auto_white_balance import AutoWhiteBalance
        raw = _make_raw()
        parm = self._parm(algo="gray_edge")
        awb = AutoWhiteBalance(raw, _sensor_info(), parm)
        gains = awb.execute()
        self.assertIsNotNone(gains)

    def test_gray_edge_fast_path_with_stats(self):
        """Gray Edge with Stats3A uses pre-computed channel_grad_means."""
        from modules.auto_white_balance.auto_white_balance import AutoWhiteBalance
        raw = _make_raw()
        parm = self._parm(algo="gray_edge")
        stats = _FakeStats(channel_grad_means={
            "r": 0.02, "gr": 0.015, "gb": 0.015, "b": 0.008
        })
        awb = AutoWhiteBalance(raw, _sensor_info(), parm, stats=stats)
        gains = awb.execute()
        self.assertIsNotNone(gains)


# ===========================================================================
# 4.6  AF Focus Metrics
# ===========================================================================

class TestAFFocusMetrics(unittest.TestCase):

    def test_tenengrad_flat_zero(self):
        from modules.auto_focus.af_metric import tenengrad
        flat = np.full((32, 32), 0.5, dtype=np.float32)
        self.assertAlmostEqual(tenengrad(flat), 0.0, places=5)

    def test_tenengrad_random_positive(self):
        from modules.auto_focus.af_metric import tenengrad
        img = np.random.default_rng(0).random((32, 32)).astype(np.float32)
        self.assertGreater(tenengrad(img), 0)

    def test_laplacian_var_flat_zero(self):
        from modules.auto_focus.af_metric import laplacian_var
        flat = np.full((32, 32), 0.5, dtype=np.float32)
        self.assertAlmostEqual(laplacian_var(flat), 0.0, places=5)

    def test_laplacian_var_random_positive(self):
        from modules.auto_focus.af_metric import laplacian_var
        img = np.random.default_rng(1).random((32, 32)).astype(np.float32)
        self.assertGreater(laplacian_var(img), 0)

    def test_compute_focus_metric_tenengrad(self):
        from modules.auto_focus.af_metric import compute_focus_metric
        img = np.random.default_rng(2).random((32, 32)).astype(np.float32)
        self.assertGreater(compute_focus_metric(img, "tenengrad"), 0)

    def test_compute_focus_metric_laplacian_var(self):
        from modules.auto_focus.af_metric import compute_focus_metric
        img = np.random.default_rng(3).random((32, 32)).astype(np.float32)
        self.assertGreater(compute_focus_metric(img, "laplacian_var"), 0)

    def test_sharper_image_higher_tenengrad(self):
        from modules.auto_focus.af_metric import tenengrad
        flat  = np.full((32, 32), 0.5, dtype=np.float32)
        sharp = np.random.default_rng(5).random((32, 32)).astype(np.float32)
        self.assertGreater(tenengrad(sharp), tenengrad(flat))

    def test_tiny_image_returns_zero(self):
        from modules.auto_focus.af_metric import compute_focus_metric
        tiny = np.ones((2, 2), dtype=np.float32)
        self.assertAlmostEqual(compute_focus_metric(tiny), 0.0, places=5)


# ===========================================================================
# 4.7  AF Search State Machine
# ===========================================================================

class TestAFSearchMachine(unittest.TestCase):

    def test_initial_state_idle(self):
        from modules.auto_focus.af_search import AFSearchMachine, AFState
        m = AFSearchMachine()
        self.assertEqual(m.state, AFState.IDLE)

    def test_trigger_enters_coarse(self):
        from modules.auto_focus.af_search import AFSearchMachine, AFState
        m = AFSearchMachine()
        m.trigger()
        self.assertEqual(m.state, AFState.COARSE_SWEEP)

    def test_idle_update_returns_idle(self):
        from modules.auto_focus.af_search import AFSearchMachine
        m = AFSearchMachine()
        result = m.update(0.5)
        self.assertEqual(result.state, "idle")

    def test_coarse_sweep_progresses(self):
        from modules.auto_focus.af_search import AFSearchMachine, AFState
        m = AFSearchMachine(lens_range=10, coarse_steps=5)
        m.trigger()
        for _ in range(20):
            m.update(0.5)
            if m.state != AFState.COARSE_SWEEP:
                break
        self.assertIn(m.state, (AFState.FINE_SEARCH, AFState.TRACKING))

    def test_full_sweep_converges(self):
        from modules.auto_focus.af_search import AFSearchMachine, AFState
        m = AFSearchMachine(lens_range=100, coarse_steps=10, fine_steps=5)
        m.trigger()
        best_pos = 50.0
        result = None
        for _ in range(300):
            pos = float(m.position)
            metric = math.exp(-0.5 * ((pos - best_pos) / 10.0) ** 2)
            result = m.update(metric)
            if result.converged:
                break
        self.assertTrue(result.converged)
        self.assertEqual(m.state, AFState.TRACKING)
        self.assertLess(abs(m.position - best_pos), 15)

    def test_tracking_holds_when_stable(self):
        from modules.auto_focus.af_search import AFSearchMachine, AFState
        m = AFSearchMachine(lens_range=100, coarse_steps=5, fine_steps=3)
        m.trigger()
        for _ in range(300):
            m.update(0.9)
            if m.state == AFState.TRACKING:
                break
        for _ in range(5):
            result = m.update(0.9)
        self.assertEqual(m.state, AFState.TRACKING)
        self.assertTrue(result.converged)

    def test_tracking_retriggers_on_drop(self):
        from modules.auto_focus.af_search import AFSearchMachine, AFState
        m = AFSearchMachine(lens_range=100, coarse_steps=5, fine_steps=3,
                            tracking_threshold=0.05)
        m.trigger()
        for _ in range(300):
            m.update(0.9)
            if m.state == AFState.TRACKING:
                break
        m.update(0.5)
        self.assertEqual(m.state, AFState.COARSE_SWEEP)

    def test_af_result_has_required_fields(self):
        from modules.auto_focus.af_search import AFSearchMachine
        m = AFSearchMachine()
        m.trigger()
        result = m.update(0.7)
        for field in ("focus_metric", "best_metric", "recommended_position",
                      "direction", "converged", "state"):
            self.assertTrue(hasattr(result, field), f"Missing field: {field}")

    def test_direction_in_valid_set(self):
        from modules.auto_focus.af_search import AFSearchMachine
        m = AFSearchMachine(lens_range=100, coarse_steps=10)
        m.trigger()
        result = m.update(0.5)
        self.assertIn(result.direction, (-1, 0, 1))


# ===========================================================================
# 4.8  AutoFocus module wrapper
# ===========================================================================

class TestAutoFocusModule(unittest.TestCase):

    def _parm(self, enabled=True):
        return {
            "is_enable": enabled,
            "lens_range": 100,
            "coarse_steps": 10,
            "fine_steps": 5,
            "tracking_threshold": 0.05,
            "metric": "tenengrad",
            "trigger_on_start": True,
        }

    def test_disabled_returns_same_image(self):
        from modules.auto_focus.auto_focus import AutoFocus
        raw = _make_raw()
        af = AutoFocus(raw, _sensor_info(), self._parm(enabled=False))
        out = af.execute()
        self.assertIs(out, raw)

    def test_enabled_returns_image(self):
        from modules.auto_focus.auto_focus import AutoFocus
        raw = _make_raw()
        af = AutoFocus(raw, _sensor_info(), self._parm())
        out = af.execute()
        self.assertIsNotNone(out)

    def test_state_persists_across_instances(self):
        from modules.auto_focus.auto_focus import AutoFocus
        from modules.auto_focus.af_search import AFState
        raw = _make_raw()
        parm = self._parm()
        af1 = AutoFocus(raw, _sensor_info(), parm)
        af1.execute()
        machine = af1.af_machine

        af2 = AutoFocus(raw, _sensor_info(), parm, af_state=machine)
        af2.execute()
        self.assertNotEqual(af2.af_machine.state, AFState.IDLE)

    def test_stats_fast_path(self):
        from modules.auto_focus.auto_focus import AutoFocus
        raw = _make_raw()
        stats = _FakeStats(focus_metric=0.05)
        af = AutoFocus(raw, _sensor_info(), self._parm(), stats=stats)
        out = af.execute()
        self.assertIsNotNone(out)

    def test_fallback_without_stats(self):
        from modules.auto_focus.auto_focus import AutoFocus
        raw = _make_raw()
        af = AutoFocus(raw, _sensor_info(), self._parm(), stats=None)
        out = af.execute()
        self.assertIsNotNone(out)

    def test_trigger_method(self):
        from modules.auto_focus.auto_focus import AutoFocus
        from modules.auto_focus.af_search import AFState
        raw = _make_raw()
        af = AutoFocus(raw, _sensor_info(), self._parm())
        af.execute()
        # After first frame, machine may be in coarse/fine/tracking
        af.trigger()
        self.assertEqual(af.af_machine.state, AFState.COARSE_SWEEP)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
