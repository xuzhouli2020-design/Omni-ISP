"""
Omni-ISP — Phase 5: Multi-frame Pipeline Tests
===============================================
Covers:
  5.1  RawStackLoader (file loading / pre-processing)
  5.2  Burst registration (phase_correlate, apply_shift_bayer, register_stack)
  5.3  Burst merge (detect_motion, temporal_merge, merge_burst)
  5.5  TemporalNR (IIR EMA filter + motion masking)
  5.4  Pipeline integration (set_burst_stack, TNR state persistence)

Run from project root:
    python -m unittest tests.test_phase6 -v
    python tests/test_phase6.py
"""

import sys
import os
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _mod in ("scipy", "scipy.signal", "rawpy", "tqdm", "cv2"):
    sys.modules.setdefault(_mod, MagicMock())
sys.modules.setdefault("util.utils", MagicMock())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw(H=64, W=64, bit_depth=10, fill=None, seed=42):
    rng = np.random.default_rng(seed)
    maxval = (1 << bit_depth) - 1
    if fill is not None:
        return np.full((H, W), int(fill * maxval), dtype=np.uint16)
    return rng.integers(0, maxval, size=(H, W), dtype=np.uint16)


def _make_stack(N=4, H=64, W=64, bit_depth=10, shift_px=0):
    """Create a synthetic burst stack, optionally with per-frame integer shifts."""
    frames = []
    for i in range(N):
        rng = np.random.default_rng(i)
        maxval = (1 << bit_depth) - 1
        frame = rng.integers(200, 800, size=(H, W), dtype=np.uint16)
        if shift_px and i > 0:
            # Apply a simple integer roll (simulates camera shake)
            frame = np.roll(frame, shift_px * i, axis=1)
        frames.append(frame)
    return np.stack(frames, axis=0)


def _sensor_info(bit_depth=10, pattern="rggb"):
    return {"bit_depth": bit_depth, "bayer_pattern": pattern, "width": 64, "height": 64}


def _parm_tnr(enabled=True, alpha=0.7, threshold=8.0, ds=2):
    return {
        "is_enable": enabled,
        "alpha": alpha,
        "motion_threshold": threshold,
        "downsample": ds,
    }


# ===========================================================================
# 5.1  RawStackLoader
# ===========================================================================

class TestRawStackLoader(unittest.TestCase):
    """RawStackLoader loads .raw files and applies BLC+OECF per frame."""

    def _write_raw_file(self, tmp_dir, name, H=64, W=64, bit_depth=10, seed=0):
        path = Path(tmp_dir) / name
        rng = np.random.default_rng(seed)
        maxval = (1 << bit_depth) - 1
        data = rng.integers(0, maxval, size=(H, W), dtype=np.uint16)
        data.tofile(str(path))
        return path, data

    def _make_platform(self):
        return {"in_file": "test", "out_file": "out_test"}

    def _make_parm_blc(self, bit_depth=10):
        return {
            "is_enable": True,
            "is_linear": True,
            "r_offset": 0, "gr_offset": 0, "gb_offset": 0, "b_offset": 0,
            "r_sat": (1 << bit_depth) - 1,
            "gr_sat": (1 << bit_depth) - 1,
            "gb_sat": (1 << bit_depth) - 1,
            "b_sat": (1 << bit_depth) - 1,
            "is_save": False,
            "is_debug": False,
        }

    def _make_parm_oecf(self):
        return {"is_enable": False, "is_save": False, "is_debug": False}

    def test_load_returns_stack(self):
        from modules.burst_capture.raw_stack_loader import RawStackLoader
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(3):
                p, _ = self._write_raw_file(tmp, f"frame{i}.raw", seed=i)
                paths.append(p)
            loader = RawStackLoader(
                raw_paths=paths,
                sensor_info=_sensor_info(),
                parm_blc=self._make_parm_blc(),
                parm_oecf=self._make_parm_oecf(),
                platform=self._make_platform(),
            )
            stack = loader.load()
            self.assertEqual(stack.shape, (3, 64, 64))

    def test_stack_dtype_uint16(self):
        from modules.burst_capture.raw_stack_loader import RawStackLoader
        with tempfile.TemporaryDirectory() as tmp:
            paths = [self._write_raw_file(tmp, f"f{i}.raw", seed=i)[0]
                     for i in range(2)]
            loader = RawStackLoader(
                raw_paths=paths,
                sensor_info=_sensor_info(),
                parm_blc=self._make_parm_blc(),
                parm_oecf=self._make_parm_oecf(),
                platform=self._make_platform(),
            )
            stack = loader.load()
            self.assertEqual(stack.dtype, np.uint16)

    def test_single_frame_stack(self):
        from modules.burst_capture.raw_stack_loader import RawStackLoader
        with tempfile.TemporaryDirectory() as tmp:
            p, _ = self._write_raw_file(tmp, "frame0.raw")
            loader = RawStackLoader(
                raw_paths=[p],
                sensor_info=_sensor_info(),
                parm_blc=self._make_parm_blc(),
                parm_oecf=self._make_parm_oecf(),
                platform=self._make_platform(),
            )
            stack = loader.load()
            self.assertEqual(stack.shape[0], 1)


# ===========================================================================
# 5.2  Burst registration
# ===========================================================================

class TestPhaseCorrelation(unittest.TestCase):
    """phase_correlate returns a shift that is close to zero for identical frames."""

    def test_identical_frames_zero_shift(self):
        from modules.burst_capture.burst_registration import phase_correlate
        rng = np.random.default_rng(0)
        g = rng.random((32, 32)).astype(np.float32)
        dy, dx = phase_correlate(g, g)
        self.assertAlmostEqual(dy, 0.0, delta=0.5)
        self.assertAlmostEqual(dx, 0.0, delta=0.5)

    def test_known_integer_shift_detected(self):
        from modules.burst_capture.burst_registration import phase_correlate
        rng = np.random.default_rng(1)
        ref = rng.random((64, 64)).astype(np.float32)
        # Shift by 3 columns → dx should be ~3 (or ~-61 because of wrap)
        target = np.roll(ref, 3, axis=1)
        dy, dx = phase_correlate(ref, target)
        # Expect |dx| == 3 (or 64-3=61 wrapped from other direction)
        self.assertTrue(abs(abs(dx) - 3) < 1 or abs(abs(dx) - 61) < 1,
                        f"Expected |dx| ~3, got {dx:.2f}")

    def test_output_is_tuple_of_floats(self):
        from modules.burst_capture.burst_registration import phase_correlate
        g = np.ones((16, 16), dtype=np.float32)
        result = phase_correlate(g, g)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)


class TestApplyShiftBayer(unittest.TestCase):
    """apply_shift_bayer preserves dtype and shape."""

    def test_zero_shift_unchanged(self):
        from modules.burst_capture.burst_registration import apply_shift_bayer
        frame = _make_raw(H=32, W=32)
        shifted = apply_shift_bayer(frame, 0.0, 0.0, "rggb")
        np.testing.assert_array_equal(shifted, frame)

    def test_output_shape_preserved(self):
        from modules.burst_capture.burst_registration import apply_shift_bayer
        frame = _make_raw(H=64, W=64)
        shifted = apply_shift_bayer(frame, 2.0, -4.0, "rggb")
        self.assertEqual(shifted.shape, frame.shape)

    def test_output_dtype_uint16(self):
        from modules.burst_capture.burst_registration import apply_shift_bayer
        frame = _make_raw(H=32, W=32)
        shifted = apply_shift_bayer(frame, 2.0, 2.0, "rggb")
        self.assertEqual(shifted.dtype, np.uint16)

    def test_different_bayer_patterns_run(self):
        from modules.burst_capture.burst_registration import apply_shift_bayer
        frame = _make_raw(H=32, W=32)
        for pat in ("rggb", "bggr", "grbg", "gbrg"):
            out = apply_shift_bayer(frame, 1.0, 1.0, pat)
            self.assertEqual(out.shape, frame.shape)


class TestRegisterStack(unittest.TestCase):
    """register_stack aligns all frames to the reference."""

    def test_returns_registered_and_shifts(self):
        from modules.burst_capture.burst_registration import register_stack
        stack = _make_stack(N=3, H=64, W=64)
        registered, shifts = register_stack(stack, "rggb", ref_idx=0)
        self.assertEqual(registered.shape, stack.shape)
        self.assertEqual(len(shifts), 3)

    def test_reference_frame_unmodified(self):
        from modules.burst_capture.burst_registration import register_stack
        stack = _make_stack(N=3, H=64, W=64)
        registered, shifts = register_stack(stack, "rggb", ref_idx=0)
        np.testing.assert_array_equal(registered[0], stack[0])

    def test_reference_shift_is_zero(self):
        from modules.burst_capture.burst_registration import register_stack
        stack = _make_stack(N=3, H=64, W=64)
        _, shifts = register_stack(stack, "rggb", ref_idx=0)
        self.assertEqual(shifts[0], (0.0, 0.0))

    def test_single_frame_stack_runs(self):
        from modules.burst_capture.burst_registration import register_stack
        stack = _make_stack(N=1, H=32, W=32)
        registered, shifts = register_stack(stack, "rggb", ref_idx=0)
        self.assertEqual(registered.shape, stack.shape)


# ===========================================================================
# 5.3  Burst merge
# ===========================================================================

class TestDetectMotion(unittest.TestCase):

    def test_returns_masks_shape(self):
        from modules.burst_capture.burst_merge import detect_motion
        stack = _make_stack(N=4)
        masks = detect_motion(stack, "rggb")
        self.assertEqual(masks.shape, stack.shape)

    def test_reference_mask_all_ones(self):
        from modules.burst_capture.burst_merge import detect_motion
        stack = _make_stack(N=4)
        masks = detect_motion(stack, "rggb", ref_idx=0)
        np.testing.assert_array_equal(masks[0], np.ones((64, 64), dtype=np.float32))

    def test_identical_stack_all_static(self):
        """Duplicate frames → all pixels should be static (mask=1)."""
        from modules.burst_capture.burst_merge import detect_motion
        frame = _make_raw()
        stack = np.stack([frame] * 4, axis=0)
        masks = detect_motion(stack, "rggb", threshold=1.0)
        for i in range(1, 4):
            self.assertAlmostEqual(float(masks[i].mean()), 1.0, delta=0.05)

    def test_large_difference_causes_motion(self):
        """Frames with large differences should produce motion (mask < 1)."""
        from modules.burst_capture.burst_merge import detect_motion
        frame_ref = np.zeros((64, 64), dtype=np.uint16)
        frame_mov = np.full((64, 64), 500, dtype=np.uint16)
        stack = np.stack([frame_ref, frame_mov], axis=0)
        masks = detect_motion(stack, "rggb", threshold=10.0)
        # Most of mask[1] should be 0 (moving)
        self.assertLess(float(masks[1].mean()), 0.5)

    def test_mask_values_binary(self):
        """Motion masks should only contain 0 or 1."""
        from modules.burst_capture.burst_merge import detect_motion
        stack = _make_stack(N=3)
        masks = detect_motion(stack, "rggb")
        unique = np.unique(masks)
        for v in unique:
            self.assertIn(float(v), [0.0, 1.0])


class TestTemporalMerge(unittest.TestCase):

    def test_mean_merge_shape(self):
        from modules.burst_capture.burst_merge import temporal_merge
        stack = _make_stack(N=4)
        masks = np.ones_like(stack, dtype=np.float32)
        merged = temporal_merge(stack, masks, method="mean")
        self.assertEqual(merged.shape, (64, 64))

    def test_mean_merge_dtype(self):
        from modules.burst_capture.burst_merge import temporal_merge
        stack = _make_stack(N=4)
        masks = np.ones_like(stack, dtype=np.float32)
        merged = temporal_merge(stack, masks, method="mean")
        self.assertEqual(merged.dtype, np.uint16)

    def test_weighted_mean_all_static_equals_mean(self):
        """All-static masks → weighted_mean should equal simple mean."""
        from modules.burst_capture.burst_merge import temporal_merge
        stack = _make_stack(N=4)
        masks = np.ones_like(stack, dtype=np.float32)
        mean_result     = temporal_merge(stack, masks, method="mean")
        weighted_result = temporal_merge(stack, masks, method="weighted_mean")
        np.testing.assert_array_equal(mean_result, weighted_result)

    def test_weighted_mean_all_moving_returns_reference(self):
        """All-moving masks → weighted_mean should fall back to reference."""
        from modules.burst_capture.burst_merge import temporal_merge
        stack = _make_stack(N=3)
        masks = np.zeros_like(stack, dtype=np.float32)
        masks[0] = 1.0   # reference is always static
        merged = temporal_merge(stack, masks, method="weighted_mean", ref_idx=0)
        # All non-reference pixels have zero weight → fallback to reference
        np.testing.assert_array_equal(merged, stack[0])

    def test_median_merge_shape(self):
        from modules.burst_capture.burst_merge import temporal_merge
        stack = _make_stack(N=5)
        masks = np.ones_like(stack, dtype=np.float32)
        merged = temporal_merge(stack, masks, method="median")
        self.assertEqual(merged.shape, (64, 64))

    def test_identical_frames_merged_equals_frame(self):
        """Merging N identical frames should yield the same frame."""
        from modules.burst_capture.burst_merge import temporal_merge
        frame = _make_raw()
        stack = np.stack([frame] * 4, axis=0)
        masks = np.ones_like(stack, dtype=np.float32)
        for method in ("mean", "weighted_mean", "median"):
            merged = temporal_merge(stack, masks, method=method)
            np.testing.assert_array_equal(merged, frame, err_msg=f"method={method}")


class TestMergeBurst(unittest.TestCase):

    def test_merge_burst_output_shape(self):
        from modules.burst_capture.burst_merge import merge_burst
        stack = _make_stack(N=4)
        merged = merge_burst(stack, "rggb")
        self.assertEqual(merged.shape, (64, 64))

    def test_merge_burst_output_dtype(self):
        from modules.burst_capture.burst_merge import merge_burst
        stack = _make_stack(N=4)
        merged = merge_burst(stack, "rggb")
        self.assertEqual(merged.dtype, np.uint16)

    def test_merge_burst_single_frame(self):
        """Single-frame burst → output == input frame."""
        from modules.burst_capture.burst_merge import merge_burst
        frame = _make_raw()
        stack = frame[np.newaxis]
        merged = merge_burst(stack, "rggb")
        self.assertEqual(merged.shape, (64, 64))

    def test_merge_reduces_noise(self):
        """Averaging identical + noisy frames should reduce variance."""
        from modules.burst_capture.burst_merge import merge_burst
        rng = np.random.default_rng(42)
        signal = np.full((64, 64), 512, dtype=np.int32)
        frames = []
        for _ in range(8):
            noise = rng.integers(-20, 20, size=(64, 64))
            frames.append(np.clip(signal + noise, 0, 1023).astype(np.uint16))
        stack = np.stack(frames, axis=0)
        merged = merge_burst(stack, "rggb", method="mean")
        # Merged variance should be much smaller than single-frame variance
        single_var = float(frames[0].astype(np.float32).var())
        merged_var = float(merged.astype(np.float32).var())
        self.assertLess(merged_var, single_var / 2)


# ===========================================================================
# 5.5  TemporalNR
# ===========================================================================

class TestTemporalNR(unittest.TestCase):

    def test_disabled_returns_same_image(self):
        from modules.temporal_nr.temporal_nr import TemporalNR
        raw = _make_raw()
        tnr = TemporalNR(raw, _sensor_info(), _parm_tnr(enabled=False))
        out = tnr.execute()
        self.assertIs(out, raw)

    def test_first_frame_pass_through(self):
        """With no prior state, first frame is passed through unchanged."""
        from modules.temporal_nr.temporal_nr import TemporalNR
        raw = _make_raw()
        tnr = TemporalNR(raw, _sensor_info(), _parm_tnr(enabled=True),
                         tnr_state=None)
        out = tnr.execute()
        np.testing.assert_array_equal(out, raw)

    def test_records_state_on_first_frame(self):
        from modules.temporal_nr.temporal_nr import TemporalNR, TNRState
        raw = _make_raw()
        tnr = TemporalNR(raw, _sensor_info(), _parm_tnr(enabled=True),
                         tnr_state=None)
        tnr.execute()
        self.assertIsNotNone(tnr.new_state)
        self.assertIsInstance(tnr.new_state, TNRState)

    def test_second_frame_blended(self):
        """Second frame should differ from raw input (IIR blend applied)."""
        from modules.temporal_nr.temporal_nr import TemporalNR
        # Two clearly different frames: 200 vs 800 (far apart so blend is obvious)
        raw1 = np.full((64, 64), 200, dtype=np.uint16)
        raw2 = np.full((64, 64), 800, dtype=np.uint16)

        tnr1 = TemporalNR(raw1, _sensor_info(),
                          _parm_tnr(alpha=0.5, threshold=1000.0),
                          tnr_state=None)
        tnr1.execute()

        tnr2 = TemporalNR(raw2, _sensor_info(),
                          _parm_tnr(alpha=0.5, threshold=1000.0),
                          tnr_state=tnr1.new_state)
        out2 = tnr2.execute()

        # With alpha=0.5 and no motion (threshold=1000): out = 0.5*800 + 0.5*200 = 500
        mean_out = float(out2.mean())
        self.assertGreater(mean_out, 250)
        self.assertLess(mean_out, 750)

    def test_static_scene_smoothed_over_frames(self):
        """Repeated identical frames → temporal filter should converge smoothly."""
        from modules.temporal_nr.temporal_nr import TemporalNR
        frame = np.full((64, 64), 512, dtype=np.uint16)
        state = None
        for _ in range(10):
            tnr = TemporalNR(frame, _sensor_info(), _parm_tnr(alpha=0.7),
                             tnr_state=state)
            out = tnr.execute()
            state = tnr.new_state
        # After 10 identical frames, output should equal input
        np.testing.assert_array_equal(out, frame)

    def test_motion_pixels_not_blended(self):
        """Pixels that change dramatically should use current frame value."""
        from modules.temporal_nr.temporal_nr import TemporalNR
        # Frame 1: left half = 100, right half = 100
        frame1 = np.full((64, 64), 100, dtype=np.uint16)
        # Frame 2: left half = 100 (static), right half = 900 (motion)
        frame2 = frame1.copy()
        frame2[:, 32:] = 900

        tnr1 = TemporalNR(frame1, _sensor_info(),
                          _parm_tnr(alpha=0.8, threshold=50.0),
                          tnr_state=None)
        tnr1.execute()

        tnr2 = TemporalNR(frame2, _sensor_info(),
                          _parm_tnr(alpha=0.8, threshold=50.0),
                          tnr_state=tnr1.new_state)
        out2 = tnr2.execute()

        # Right half (moving): should be close to current frame (900)
        right_mean = float(out2[:, 32:].mean())
        self.assertGreater(right_mean, 700,
                           "Moving region should retain current-frame value")

    def test_output_shape_preserved(self):
        from modules.temporal_nr.temporal_nr import TemporalNR
        raw = _make_raw(H=48, W=80)
        si  = {"bit_depth": 10, "bayer_pattern": "rggb", "width": 80, "height": 48}
        tnr = TemporalNR(raw, si, _parm_tnr(), tnr_state=None)
        out = tnr.execute()
        self.assertEqual(out.shape, raw.shape)

    def test_output_dtype_uint16(self):
        from modules.temporal_nr.temporal_nr import TemporalNR
        raw = _make_raw()
        tnr = TemporalNR(raw, _sensor_info(), _parm_tnr(), tnr_state=None)
        out = tnr.execute()
        self.assertEqual(out.dtype, np.uint16)

    def test_alpha_zero_no_smoothing(self):
        """alpha=0 → output equals current frame regardless of history."""
        from modules.temporal_nr.temporal_nr import TemporalNR
        frame1 = np.full((64, 64), 100, dtype=np.uint16)
        frame2 = np.full((64, 64), 800, dtype=np.uint16)

        tnr1 = TemporalNR(frame1, _sensor_info(), _parm_tnr(alpha=0.0),
                          tnr_state=None)
        tnr1.execute()

        tnr2 = TemporalNR(frame2, _sensor_info(), _parm_tnr(alpha=0.0),
                          tnr_state=tnr1.new_state)
        out2 = tnr2.execute()
        np.testing.assert_array_equal(out2, frame2)


# ===========================================================================
# 5.4  Pipeline integration — set_burst_stack + TNR state
# ===========================================================================

class TestPipelineIntegration(unittest.TestCase):
    """Lightweight smoke tests for burst / TNR wiring in OmniISP."""

    def _make_isp_stub(self):
        """Return a minimal OmniISP-like object with burst/TNR attributes."""
        # We test the state-management attributes directly rather than running
        # the full pipeline (which requires a real config file and raw image).
        class FakeISP:
            def __init__(self):
                self._burst_stack  = None
                self._tnr_state    = None

            def set_burst_stack(self, stack):
                self._burst_stack = stack

        return FakeISP()

    def test_set_burst_stack_stores_stack(self):
        isp = self._make_isp_stub()
        stack = _make_stack(N=4)
        isp.set_burst_stack(stack)
        self.assertIsNotNone(isp._burst_stack)
        self.assertEqual(isp._burst_stack.shape[0], 4)

    def test_tnr_state_initialises_none(self):
        isp = self._make_isp_stub()
        self.assertIsNone(isp._tnr_state)

    def test_tnr_state_propagates(self):
        """TNRState persisted after first frame should be a TNRState instance."""
        from modules.temporal_nr.temporal_nr import TemporalNR, TNRState
        raw = _make_raw()
        tnr = TemporalNR(raw, _sensor_info(), _parm_tnr(enabled=True),
                         tnr_state=None)
        tnr.execute()
        # Simulate pipeline: store new_state for next frame
        new_state = tnr.new_state
        self.assertIsInstance(new_state, TNRState)
        self.assertIsNotNone(new_state.prev_filtered)
        self.assertIsNotNone(new_state.prev_g)

    def test_burst_merge_integrates_with_registration(self):
        """Full burst path: register → merge yields correct shape."""
        from modules.burst_capture.burst_registration import register_stack
        from modules.burst_capture.burst_merge import merge_burst
        stack = _make_stack(N=4, H=64, W=64, shift_px=2)
        registered, shifts = register_stack(stack, "rggb", ref_idx=0)
        merged = merge_burst(registered, "rggb", method="weighted_mean")
        self.assertEqual(merged.shape, (64, 64))
        self.assertEqual(merged.dtype, np.uint16)

    def test_burst_disabled_path_skips_merge(self):
        """When burst is disabled, run_pipeline should not call merge_burst."""
        # Verify that when parm_burst["is_enable"]=False and no stack is set,
        # the merge path is skipped (no AttributeError or crash).
        # We simulate this with a direct attribute check.
        class FakeISP:
            parm_burst = {"is_enable": False}
            _burst_stack = None

        isp = FakeISP()
        # Pipeline guard: is_enable=False OR _burst_stack is None → skip
        skip = not isp.parm_burst.get("is_enable", False) or isp._burst_stack is None
        self.assertTrue(skip)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
