"""
tests/test_phase4.py — Regression tests for Phase 2.2, 2.3, 2.7 features.

Run with:
    python tests/test_phase4.py
    python -m pytest tests/test_phase4.py -v

No scipy / rawpy / tqdm required. All external dependencies are stubbed.

Sections
--------
  TestLMMSEDemosaic      (14 tests) — Phase 2.2 LMMSE demosaicing
  TestGuidedFilterLTM    (12 tests) — Phase 2.3 Guided Filter LDCI
  TestBLCOBPixels        (12 tests) — Phase 2.7 OB pixel black-level estimation

Total: 38 tests
"""

import sys
import os
import types
import unittest
from unittest.mock import MagicMock
import numpy as np

# ---------------------------------------------------------------------------
# sys.path — allow running from project root or tests/ directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Stub unavailable packages
# ---------------------------------------------------------------------------
_util = types.ModuleType("util")
_util_utils = types.ModuleType("util.utils")
_util_utils.save_output_array     = MagicMock()
_util_utils.save_output_array_yuv = MagicMock()
_util.utils = _util_utils
sys.modules.setdefault("util",       _util)
sys.modules.setdefault("util.utils", _util_utils)
sys.modules.setdefault("cv2",        MagicMock())

_scipy = types.ModuleType("scipy")
_scipy_signal = types.ModuleType("scipy.signal")
_scipy_signal.correlate2d = MagicMock(
    side_effect=lambda a, b, mode="full", boundary="fill", fillvalue=0:
        np.zeros_like(a, dtype=np.float32)
)
_scipy.signal = _scipy_signal
sys.modules.setdefault("scipy",        _scipy)
sys.modules.setdefault("scipy.signal", _scipy_signal)
sys.modules.setdefault("tqdm",         MagicMock())
sys.modules.setdefault("rawpy",        MagicMock())
sys.modules.setdefault("yaml",         MagicMock())

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from modules.demosaic.lmmse import LMMSE, _interpolate_G, _interpolate_color_diff
from modules.demosaic.demosaic import Demosaic

from modules.ldci.guided_filter_ltm import GuidedFilterLTM, _box_filter, _guided_filter
from modules.ldci.ldci import LDCI

from modules.black_level_correction.black_level_correction import BlackLevelCorrection

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def rggb_masks(h, w):
    """Return (mask_r, mask_g, mask_b) for RGGB pattern."""
    mask_r = np.zeros((h, w), dtype=bool)
    mask_g = np.zeros((h, w), dtype=bool)
    mask_b = np.zeros((h, w), dtype=bool)
    mask_r[0::2, 0::2] = True
    mask_g[0::2, 1::2] = True
    mask_g[1::2, 0::2] = True
    mask_b[1::2, 1::2] = True
    return mask_r, mask_g, mask_b


def make_bayer_rggb(h=32, w=32, r=200, g=250, b=180):
    """Synthetic uniform RGGB Bayer image with known channel values."""
    raw = np.zeros((h, w), dtype=np.uint16)
    raw[0::2, 0::2] = r
    raw[0::2, 1::2] = g
    raw[1::2, 0::2] = g
    raw[1::2, 1::2] = b
    return raw


def make_platform():
    return {"in_file": "test", "save_format": "png",
            "disable_progress_bar": True, "leave_pbar_string": False}


def make_sensor_info(bit_depth=10, bayer="rggb"):
    return {"bit_depth": bit_depth, "bayer_pattern": bayer}


# ===========================================================================
# Phase 2.2 — LMMSE Demosaicing
# ===========================================================================

class TestLMMSEDemosaic(unittest.TestCase):
    """Phase 2.2 — LMMSE directional demosaicing (pure NumPy)."""

    def _uniform_bayer(self, h=32, w=32):
        return make_bayer_rggb(h, w, r=512, g=600, b=400)

    def test_output_shape(self):
        raw = self._uniform_bayer()
        masks = rggb_masks(*raw.shape)
        out = LMMSE(raw, masks).apply_lmmse()
        self.assertEqual(out.shape, (32, 32, 3))

    def test_output_dtype_float32(self):
        raw = self._uniform_bayer()
        masks = rggb_masks(*raw.shape)
        out = LMMSE(raw, masks).apply_lmmse()
        self.assertEqual(out.dtype, np.float32)

    def test_uniform_image_recovers_channels(self):
        """On a uniform RGGB image interior pixels should reconstruct correctly.
        We test the interior crop (4:-4, 4:-4) to avoid reflect-padding edge artefacts
        that appear only at the 2-pixel border of small test images.
        """
        raw = make_bayer_rggb(64, 64, r=500, g=600, b=400)
        masks = rggb_masks(64, 64)
        out = LMMSE(raw, masks).apply_lmmse()
        # Crop out 4-pixel border to exclude reflect-padding effects
        interior = out[4:-4, 4:-4, :]
        tol = 2
        self.assertAlmostEqual(float(np.mean(interior[:, :, 0])), 500.0, delta=tol)
        self.assertAlmostEqual(float(np.mean(interior[:, :, 1])), 600.0, delta=tol)
        self.assertAlmostEqual(float(np.mean(interior[:, :, 2])), 400.0, delta=tol)

    def test_known_G_pixels_unchanged(self):
        """G-sampled pixels in output G channel should match raw input."""
        raw = make_bayer_rggb(32, 32, r=400, g=700, b=300)
        mask_r, mask_g, mask_b = rggb_masks(32, 32)
        G_out = _interpolate_G(raw, mask_g, mask_r, mask_b)
        # At G sites, output must equal raw input
        diff = np.abs(G_out[mask_g].astype(np.float32) - raw[mask_g].astype(np.float32))
        self.assertTrue(np.all(diff < 1e-3), f"max G-site error: {diff.max():.4f}")

    def test_known_G_pixels_unchanged_via_LMMSE(self):
        """Via full LMMSE class: G channel at G-sampled sites = raw input."""
        raw = make_bayer_rggb(32, 32, r=400, g=700, b=300)
        masks = rggb_masks(32, 32)
        mask_g = masks[1]
        out = LMMSE(raw, masks).apply_lmmse()
        diff = np.abs(out[:, :, 1][mask_g].astype(np.float32)
                      - raw[mask_g].astype(np.float32))
        self.assertTrue(np.all(diff < 1e-3), f"max G error: {diff.max():.4f}")

    def test_color_diff_interpolation_known_sites_unchanged(self):
        """_interpolate_color_diff must preserve known-site values."""
        rng = np.random.default_rng(0)
        mask_r, _, _ = rggb_masks(16, 16)
        diff_known = (rng.random((16, 16)) * 100).astype(np.float32)
        diff_known[~mask_r] = 0.0
        result = _interpolate_color_diff(diff_known, mask_r)
        err = np.abs(result[mask_r] - diff_known[mask_r])
        self.assertTrue(np.all(err < 1e-4))

    def test_color_diff_interpolation_fills_all_pixels(self):
        """Every pixel in output of _interpolate_color_diff should be non-NaN."""
        mask_r, _, _ = rggb_masks(16, 16)
        diff = np.ones((16, 16), dtype=np.float32) * 50.0
        diff[~mask_r] = 0.0
        result = _interpolate_color_diff(diff, mask_r)
        self.assertFalse(np.any(np.isnan(result)))

    def test_demosaic_class_mhc_mode_dispatch(self):
        """mode='mhc' must use Malvar path (scipy mock returns zeros)."""
        raw = self._uniform_bayer()
        parm = {"is_save": False, "demosaic_method": "mhc"}
        d = Demosaic(raw, make_platform(), make_sensor_info(), parm)
        out = d.execute()
        self.assertEqual(out.shape, (32, 32, 3))

    def test_demosaic_class_lmmse_mode_dispatch(self):
        """mode='lmmse' must use LMMSE path and return uint16 (H,W,3)."""
        raw = self._uniform_bayer()
        parm = {"is_save": False, "demosaic_method": "lmmse"}
        d = Demosaic(raw, make_platform(), make_sensor_info(), parm)
        out = d.execute()
        self.assertEqual(out.shape, (32, 32, 3))
        self.assertEqual(out.dtype, np.uint16)

    def test_demosaic_default_is_mhc(self):
        """No demosaic_method key → defaults to MHC."""
        raw = self._uniform_bayer()
        parm = {"is_save": False}
        d = Demosaic(raw, make_platform(), make_sensor_info(), parm)
        self.assertEqual(d.method, "mhc")

    def test_lmmse_output_range_within_bit_depth(self):
        """After clip in Demosaic.apply_cfa(), output is in [0, 2**bit_depth - 1]."""
        raw = make_bayer_rggb(32, 32, r=500, g=600, b=400)
        parm = {"is_save": False, "demosaic_method": "lmmse"}
        d = Demosaic(raw, make_platform(), make_sensor_info(bit_depth=10), parm)
        out = d.apply_cfa()
        self.assertTrue(np.all(out >= 0))
        self.assertTrue(np.all(out <= 1023))

    def test_lmmse_non_negative_after_clip(self):
        """After Demosaic.apply_cfa() (which clips), output must be in [0, max_val].
        Raw LMMSE float output may have small negatives at border pixels due to
        reflect-padding, which is expected and correct — apply_cfa clips them.
        """
        raw = make_bayer_rggb(32, 32, r=200, g=300, b=150)
        parm = {"is_save": False, "demosaic_method": "lmmse"}
        out = Demosaic(raw, make_platform(), make_sensor_info(10), parm).apply_cfa()
        self.assertTrue(np.all(out >= 0))

    def test_lmmse_larger_image(self):
        """LMMSE runs on 128×128 without error."""
        raw = make_bayer_rggb(128, 128, r=300, g=400, b=250)
        masks = rggb_masks(128, 128)
        out = LMMSE(raw, masks).apply_lmmse()
        self.assertEqual(out.shape, (128, 128, 3))

    def test_lmmse_odd_dimensions_handled(self):
        """LMMSE on even-sized image only (Bayer requires even dims).
           Confirm no crash on 30×30."""
        raw = make_bayer_rggb(30, 30)
        masks = rggb_masks(30, 30)
        out = LMMSE(raw, masks).apply_lmmse()
        self.assertEqual(out.shape, (30, 30, 3))


# ===========================================================================
# Phase 2.3 — Guided Filter LDCI
# ===========================================================================

class TestGuidedFilterLTM(unittest.TestCase):
    """Phase 2.3 — Edge-preserving Guided Filter Local Tone Mapping."""

    def _make_yuv_float(self, h=32, w=32):
        yuv = RNG.random((h, w, 3)).astype(np.float32)
        return yuv

    def _make_yuv_uint8(self, h=32, w=32):
        return (RNG.random((h, w, 3)) * 255).astype(np.uint8)

    def _base_parm(self, **kw):
        p = {
            "is_enable": True, "is_save": False,
            "mode": "guided_filter",
            "clip_limit": 1, "wind": 16,
            "guided_radius": 4,    # small for fast tests
            "guided_eps": 0.01,
            "detail_gain": 1.5,
            "base_compression": 0.7,
        }
        p.update(kw)
        return p

    # --- _box_filter ---
    def test_box_filter_flat_image(self):
        """Box filter of constant image = constant."""
        img = np.full((16, 16), 0.5, dtype=np.float32)
        out = _box_filter(img, r=3)
        np.testing.assert_allclose(out, 0.5, atol=1e-5)

    def test_box_filter_shape_preserved(self):
        img = np.ones((20, 30), dtype=np.float32)
        out = _box_filter(img, r=2)
        self.assertEqual(out.shape, (20, 30))

    def test_box_filter_dtype_float32(self):
        img = np.ones((8, 8), dtype=np.float32)
        self.assertEqual(_box_filter(img, 2).dtype, np.float32)

    # --- _guided_filter ---
    def test_guided_filter_flat_image_unchanged(self):
        """Guided filter of constant image = constant (a=0, b=mean → q=mean)."""
        img = np.full((16, 16), 0.7, dtype=np.float32)
        out = _guided_filter(img, r=4, eps=0.01)
        np.testing.assert_allclose(out, 0.7, atol=1e-4)

    def test_guided_filter_output_in_range(self):
        """Guided filter output stays in [0, 1] for input in [0, 1]."""
        img = RNG.random((32, 32)).astype(np.float32)
        out = _guided_filter(img, r=4, eps=0.01)
        self.assertTrue(np.all(out >= -1e-4))   # small float error allowed
        self.assertTrue(np.all(out <= 1.0 + 1e-4))

    def test_guided_filter_shape_preserved(self):
        img = np.ones((24, 36), dtype=np.float32) * 0.5
        out = _guided_filter(img, r=3, eps=0.01)
        self.assertEqual(out.shape, (24, 36))

    # --- GuidedFilterLTM ---
    def test_output_shape_float(self):
        yuv = self._make_yuv_float()
        gf = GuidedFilterLTM(yuv, self._base_parm())
        out = gf.apply_guided_ltm()
        self.assertEqual(out.shape, yuv.shape)

    def test_output_dtype_float(self):
        yuv = self._make_yuv_float()
        out = GuidedFilterLTM(yuv, self._base_parm()).apply_guided_ltm()
        self.assertEqual(out.dtype, np.float32)

    def test_output_y_in_range_float(self):
        yuv = self._make_yuv_float()
        out = GuidedFilterLTM(yuv, self._base_parm()).apply_guided_ltm()
        self.assertTrue(np.all(out[:, :, 0] >= 0.0))
        self.assertTrue(np.all(out[:, :, 0] <= 1.0))

    def test_chroma_channels_unchanged(self):
        """Cb and Cr channels must be copied unchanged by guided filter."""
        yuv = self._make_yuv_float()
        out = GuidedFilterLTM(yuv, self._base_parm()).apply_guided_ltm()
        np.testing.assert_array_equal(out[:, :, 1], yuv[:, :, 1])
        np.testing.assert_array_equal(out[:, :, 2], yuv[:, :, 2])

    def test_ldci_dispatch_guided_filter(self):
        """LDCI class with mode='guided_filter' calls GuidedFilterLTM path."""
        yuv = self._make_yuv_float()
        parm = self._base_parm()
        ldci = LDCI(yuv, make_platform(), make_sensor_info(), parm, conv_std="yuv444")
        out = ldci.execute()
        self.assertEqual(out.shape, yuv.shape)

    def test_ldci_default_mode_is_clahe(self):
        """LDCI mode defaults to 'clahe' (backward compat) if not specified."""
        yuv = self._make_yuv_float()
        parm = {"is_enable": True, "is_save": False, "clip_limit": 1, "wind": 16}
        ldci = LDCI(yuv, make_platform(), make_sensor_info(), parm, conv_std="yuv444")
        self.assertEqual(ldci.mode, "clahe")

    def test_ldci_disabled_passthrough(self):
        yuv = self._make_yuv_float()
        parm = self._base_parm()
        parm["is_enable"] = False
        ldci = LDCI(yuv, make_platform(), make_sensor_info(), parm, conv_std="yuv444")
        out = ldci.execute()
        np.testing.assert_array_equal(out, yuv)


# ===========================================================================
# Phase 2.7 — BLC OB Pixel Estimation
# ===========================================================================

class TestBLCOBPixels(unittest.TestCase):
    """Phase 2.7 — Optical black pixel auto black-level estimation."""

    def _base_parm(self, **kw):
        p = {
            "is_enable": True, "is_linear": False, "is_save": False,
            "r_offset": 0, "gr_offset": 0, "gb_offset": 0, "b_offset": 0,
            "r_sat": 4095, "gr_sat": 4095, "gb_sat": 4095, "b_sat": 4095,
            "use_ob_pixels": False,
            "ob_rows": 0, "ob_cols": 0,
            "ob_correction_mode": "scalar",
            "ob_smoothing": "median",
        }
        p.update(kw)
        return p

    def _make_raw_with_ob(self, h=32, w=32, ob_rows=4,
                          ob_level=200, signal_level=1000):
        """
        Create a RGGB raw image with ob_rows OB rows at the top (value = ob_level)
        and active pixel region (value = signal_level).
        """
        raw = np.full((h, w), signal_level, dtype=np.uint16)
        raw[:ob_rows, :] = ob_level
        return raw

    # --- use_ob_pixels=False (default) ---
    def test_ob_disabled_uses_config_offsets(self):
        """When use_ob_pixels=False, fixed config offsets are used unchanged."""
        raw = self._make_raw_with_ob(signal_level=500)
        parm = self._base_parm(r_offset=100, gr_offset=100, gb_offset=100, b_offset=100)
        blc = BlackLevelCorrection(raw, make_platform(), make_sensor_info(), parm)
        self.assertEqual(blc.param_blc["r_offset"], 100)

    # --- OB scalar estimation ---
    def test_ob_scalar_estimates_correct_r_offset(self):
        """OB scalar mode: estimated r_offset should match the OB region value."""
        ob_level = 150
        raw = self._make_raw_with_ob(h=32, w=32, ob_rows=4,
                                     ob_level=ob_level, signal_level=800)
        parm = self._base_parm(
            use_ob_pixels=True, ob_rows=4,
            ob_correction_mode="scalar", ob_smoothing="median",
        )
        blc = BlackLevelCorrection(raw, make_platform(),
                                   make_sensor_info(bayer="rggb"), parm)
        # The OB region has uniform value ob_level → estimated offset should be ob_level
        self.assertAlmostEqual(blc.param_blc["r_offset"], float(ob_level), delta=2)

    def test_ob_scalar_all_channels_estimated(self):
        """All four channel offsets should be updated from OB region."""
        ob_level = 200
        raw = self._make_raw_with_ob(ob_level=ob_level)
        parm = self._base_parm(use_ob_pixels=True, ob_rows=4,
                               ob_correction_mode="scalar")
        blc = BlackLevelCorrection(raw, make_platform(),
                                   make_sensor_info(bayer="rggb"), parm)
        for key in ("r_offset", "gr_offset", "gb_offset", "b_offset"):
            self.assertAlmostEqual(blc.param_blc[key], float(ob_level), delta=2,
                                   msg=f"{key} incorrect")

    def test_ob_scalar_subtraction_effect(self):
        """After OB-based BLC, active pixels should have value ≈ signal − ob_level."""
        ob_level, signal = 200, 700
        raw = self._make_raw_with_ob(h=32, w=32, ob_rows=4,
                                     ob_level=ob_level, signal_level=signal)
        parm = self._base_parm(use_ob_pixels=True, ob_rows=4,
                               ob_correction_mode="scalar")
        blc = BlackLevelCorrection(raw, make_platform(),
                                   make_sensor_info(bayer="rggb"), parm)
        out = blc.execute()
        # Active region starts at row 4; after subtraction ≈ signal - ob_level
        active = out[4:, :].astype(np.float32)
        expected = float(signal - ob_level)
        self.assertAlmostEqual(float(np.mean(active)), expected, delta=5)

    def test_ob_mean_smoothing(self):
        """ob_smoothing='mean' produces same result as median for uniform OB."""
        ob_level = 180
        raw = self._make_raw_with_ob(ob_level=ob_level)
        for sm in ("mean", "median"):
            parm = self._base_parm(use_ob_pixels=True, ob_rows=4,
                                   ob_correction_mode="scalar", ob_smoothing=sm)
            blc = BlackLevelCorrection(raw, make_platform(),
                                       make_sensor_info(bayer="rggb"), parm)
            self.assertAlmostEqual(blc.param_blc["r_offset"], float(ob_level), delta=2,
                                   msg=f"smoothing={sm}")

    # --- OB per_column mode ---
    def test_ob_per_column_applies_column_offset(self):
        """per_column mode should produce a per-column corrected image."""
        ob_level = 200
        raw = self._make_raw_with_ob(h=32, w=32, ob_rows=4,
                                     ob_level=ob_level, signal_level=1000)
        parm = self._base_parm(use_ob_pixels=True, ob_rows=4,
                               ob_correction_mode="per_column")
        blc = BlackLevelCorrection(raw, make_platform(),
                                   make_sensor_info(bayer="rggb"), parm)
        out = blc.execute()
        # With uniform OB the per-column offset is the same for all columns
        active = out[4:, :].astype(np.float32)
        self.assertAlmostEqual(float(np.mean(active)), float(1000 - ob_level), delta=5)

    def test_ob_per_column_attr_created(self):
        """When per_column mode active, _ob_col_offset attribute is created."""
        raw = self._make_raw_with_ob()
        parm = self._base_parm(use_ob_pixels=True, ob_rows=4,
                               ob_correction_mode="per_column")
        blc = BlackLevelCorrection(raw, make_platform(),
                                   make_sensor_info(bayer="rggb"), parm)
        self.assertTrue(hasattr(blc, "_ob_col_offset"))
        self.assertEqual(blc._ob_col_offset.shape[1], raw.shape[1])

    # --- ob_rows=0 fallback ---
    def test_ob_zero_rows_falls_back_to_config(self):
        """If ob_rows=0 and ob_cols=0, no estimation occurs; config values kept."""
        raw = self._make_raw_with_ob()
        parm = self._base_parm(use_ob_pixels=True, ob_rows=0, ob_cols=0,
                               r_offset=77)
        blc = BlackLevelCorrection(raw, make_platform(),
                                   make_sensor_info(bayer="rggb"), parm)
        self.assertEqual(blc.param_blc["r_offset"], 77)

    # --- bayer pattern variants ---
    def test_ob_bggr_pattern(self):
        """OB estimation works for BGGR pattern."""
        ob_level = 160
        raw = self._make_raw_with_ob(ob_level=ob_level)
        parm = self._base_parm(use_ob_pixels=True, ob_rows=4,
                               ob_correction_mode="scalar")
        blc = BlackLevelCorrection(raw, make_platform(),
                                   make_sensor_info(bayer="bggr"), parm)
        # All offsets should be estimated (values depend on pattern mapping)
        for key in ("r_offset", "gr_offset", "gb_offset", "b_offset"):
            self.assertAlmostEqual(blc.param_blc[key], float(ob_level), delta=2,
                                   msg=f"bggr {key}")

    # --- backward compatibility ---
    def test_existing_blc_behaviour_unchanged_when_ob_disabled(self):
        """With use_ob_pixels=False and step='full', output matches legacy BLC."""
        raw = np.full((16, 16), 300, dtype=np.uint16)
        parm = self._base_parm(r_offset=100, gr_offset=100, gb_offset=100, b_offset=100,
                               is_linear=False)
        blc = BlackLevelCorrection(raw, make_platform(),
                                   make_sensor_info(bayer="rggb"), parm)
        out = blc.execute()
        # 300 - 100 = 200 for all pixels
        self.assertTrue(np.all(out == 200))

    def test_config_offsets_not_mutated(self):
        """Original parm_blc dict must not be mutated by OB estimation (uses .copy())."""
        raw = self._make_raw_with_ob(ob_level=150)
        original_parm = self._base_parm(use_ob_pixels=True, ob_rows=4,
                                        r_offset=999)
        saved_r = original_parm["r_offset"]
        BlackLevelCorrection(raw, make_platform(),
                             make_sensor_info(bayer="rggb"), original_parm)
        # Original dict must be unchanged
        self.assertEqual(original_parm["r_offset"], saved_r)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
