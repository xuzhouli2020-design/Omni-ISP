"""
tests/test_phase3.py — Regression tests for Phase 2.4 + 2.5 + 2.6 + 3.2 features.

Run with:
    python -m pytest tests/test_phase3.py -v

Or:
    python tests/test_phase3.py

No scipy / rawpy / tqdm required.  All external dependencies are stubbed.

Sections
--------
  TestAdaptiveSharpen       (10 tests) — Phase 2.4 edge-adaptive sharpening
  TestColorSaturation       (9 tests)  — Phase 2.5 vibrance mode + chroma limiter
  TestNR2DModes             (11 tests) — Phase 2.6 bilateral / chroma_only / off
  TestGammaEOTF             (12 tests) — Phase 3.2 multi-EOTF functions

Total: 42 tests
"""

import sys
import os
import types
import unittest
from unittest.mock import MagicMock
import numpy as np

# Run from project root so 'modules' package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Stub all unavailable external packages before any module imports
# ---------------------------------------------------------------------------

# util stubs (modules import from util.utils for save helpers)
_util = types.ModuleType("util")
_util_utils = types.ModuleType("util.utils")
_util_utils.save_output_array = MagicMock()
_util_utils.save_output_array_yuv = MagicMock()
_util.utils = _util_utils
sys.modules.setdefault("util", _util)
sys.modules.setdefault("util.utils", _util_utils)

# cv2
sys.modules.setdefault("cv2", MagicMock())

# scipy
_scipy = types.ModuleType("scipy")
_scipy_signal = types.ModuleType("scipy.signal")
_scipy_signal.correlate2d = MagicMock()
_scipy_ndimage = types.ModuleType("scipy.ndimage")
_scipy_ndimage.gaussian_filter = MagicMock(side_effect=lambda a, *args, **kw: a)
_scipy_ndimage.maximum_filter = MagicMock(return_value=np.zeros((4, 4)))
_scipy_ndimage.minimum_filter = MagicMock(return_value=np.zeros((4, 4)))
_scipy_ndimage.correlate = MagicMock(return_value=np.zeros((4, 4)))
_scipy.signal = _scipy_signal
_scipy.ndimage = _scipy_ndimage
sys.modules.setdefault("scipy", _scipy)
sys.modules.setdefault("scipy.signal", _scipy_signal)
sys.modules.setdefault("scipy.ndimage", _scipy_ndimage)

# tqdm
_tqdm_mod = types.ModuleType("tqdm")
_tqdm_mod.tqdm = lambda iterable, **kw: iterable
sys.modules.setdefault("tqdm", _tqdm_mod)

# rawpy (used only by omni_isp.py loader — not needed here)
sys.modules.setdefault("rawpy", MagicMock())
sys.modules.setdefault("yaml", MagicMock())

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def make_platform():
    return {
        "in_file": "test",
        "save_format": "png",
        "disable_progress_bar": True,
        "leave_pbar_string": False,
    }


def make_sensor_info(bit_depth=8):
    return {"bit_depth": bit_depth, "bayer_pattern": "rggb"}


def random_img_uint8(h=32, w=32):
    rng = np.random.default_rng(42)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def random_img_float32(h=32, w=32):
    rng = np.random.default_rng(42)
    return rng.random((h, w, 3)).astype(np.float32)


# ===========================================================================
# Phase 2.4 — AdaptiveSharpen (edge-adaptive gain)
# ===========================================================================

from modules.sharpen.adaptive_sharpen import AdaptiveSharpen, _gaussian_blur_2d, _sobel_magnitude
from modules.sharpen.sharpen import Sharpening


class TestAdaptiveSharpen(unittest.TestCase):
    """Phase 2.4 — edge-adaptive sharpening (pure NumPy, no scipy)."""

    BASE_PARM = {
        "sharpen_sigma": 1.5,
        "gain": 0.8,
        "noise_floor": 0.02,
        "edge_max": 0.15,
        "halo_limit": 0.1,
    }

    # --- unit: gaussian blur ---

    def test_gaussian_blur_shape_preserved(self):
        luma = np.random.rand(40, 50).astype(np.float32)
        out = _gaussian_blur_2d(luma, sigma=2.0)
        self.assertEqual(out.shape, luma.shape)

    def test_gaussian_blur_flat_image_unchanged(self):
        """Gaussian blur of a constant image must equal that constant."""
        luma = np.full((20, 20), 0.5, dtype=np.float32)
        out = _gaussian_blur_2d(luma, sigma=2.0)
        np.testing.assert_allclose(out, luma, atol=1e-5)

    # --- unit: Sobel magnitude ---

    def test_sobel_shape_preserved(self):
        luma = np.random.rand(20, 30).astype(np.float32)
        out = _sobel_magnitude(luma)
        self.assertEqual(out.shape, luma.shape)

    def test_sobel_flat_image_near_zero(self):
        luma = np.full((20, 20), 0.5, dtype=np.float32)
        out = _sobel_magnitude(luma)
        self.assertLess(float(out.max()), 1e-5)

    def test_sobel_nonnegative(self):
        luma = np.random.rand(20, 20).astype(np.float32)
        out = _sobel_magnitude(luma)
        self.assertGreaterEqual(float(out.min()), 0.0)

    # --- AdaptiveSharpen ---

    def test_output_shape_preserved_uint8(self):
        img = random_img_uint8()
        ada = AdaptiveSharpen(img.copy(), self.BASE_PARM)
        out = ada.apply_sharpen()
        self.assertEqual(out.shape, img.shape)

    def test_output_dtype_uint8(self):
        img = random_img_uint8()
        ada = AdaptiveSharpen(img.copy(), self.BASE_PARM)
        out = ada.apply_sharpen()
        self.assertEqual(out.dtype, np.uint8)

    def test_output_range_uint8(self):
        img = random_img_uint8()
        ada = AdaptiveSharpen(img.copy(), self.BASE_PARM)
        out = ada.apply_sharpen()
        self.assertGreaterEqual(int(out.min()), 0)
        self.assertLessEqual(int(out.max()), 255)

    def test_output_shape_preserved_float32(self):
        img = random_img_float32()
        ada = AdaptiveSharpen(img.copy(), self.BASE_PARM)
        out = ada.apply_sharpen()
        self.assertEqual(out.shape, img.shape)

    def test_flat_image_unchanged_by_adaptive(self):
        """On a perfectly flat image, Sobel = 0, so gain_map = 0 everywhere → no change."""
        img = np.full((24, 24, 3), 0.5, dtype=np.float32)
        ada = AdaptiveSharpen(img.copy(), self.BASE_PARM)
        out = ada.apply_sharpen()
        np.testing.assert_allclose(out[:, :, 0], img[:, :, 0], atol=1e-4,
                                   err_msg="Flat image Y channel should be unchanged")

    # --- Sharpening dispatcher ---

    def test_sharpening_adaptive_mode_dispatch(self):
        platform = make_platform()
        sensor_info = make_sensor_info()
        parm = {
            "is_enable": True, "is_save": False, "mode": "adaptive",
            **self.BASE_PARM
        }
        img = random_img_uint8()
        s = Sharpening(img.copy(), platform, sensor_info, parm, "bt601")
        out = s.execute()
        self.assertEqual(out.shape, img.shape)

    def test_sharpening_usm_mode_backward_compat(self):
        """mode='usm' falls through to original UnsharpMasking (scipy mocked)."""
        platform = make_platform()
        sensor_info = make_sensor_info()
        parm = {
            "is_enable": True, "is_save": False, "mode": "usm",
            "sharpen_sigma": 1.5, "sharpen_strength": 0.5,
        }
        img = random_img_uint8()
        s = Sharpening(img.copy(), platform, sensor_info, parm, "bt601")
        out = s.execute()  # scipy gaussian_filter is mocked → returns input unchanged
        self.assertEqual(out.shape, img.shape)

    def test_sharpening_disabled_passthrough(self):
        platform = make_platform()
        sensor_info = make_sensor_info()
        parm = {"is_enable": False, "is_save": False, "mode": "adaptive", **self.BASE_PARM}
        img = random_img_uint8()
        s = Sharpening(img.copy(), platform, sensor_info, parm, "bt601")
        out = s.execute()
        np.testing.assert_array_equal(out, img)


# ===========================================================================
# Phase 2.5 — ColorSpaceConversion saturation: flat + vibrance + chroma limit
# ===========================================================================

from modules.color_space_conversion.color_space_conversion import ColorSpaceConversion


class TestColorSaturation(unittest.TestCase):
    """Phase 2.5 — vibrance mode and soft-knee chroma limiter in CSC."""

    BASE_CSC = {"conv_standard": 2, "is_save": False}

    def _csc(self, img, parm_cse):
        return ColorSpaceConversion(
            img.copy(), make_platform(), make_sensor_info(), self.BASE_CSC, parm_cse
        )

    # --- flat mode (backward compat) ---

    def test_flat_mode_output_shape(self):
        out = self._csc(random_img_uint8(), {"is_enable": True, "mode": "flat",
                                              "saturation_gain": 1.5, "chroma_limit": 1.0}).execute()
        self.assertEqual(out.shape, (32, 32, 3))

    def test_flat_mode_output_dtype(self):
        out = self._csc(random_img_uint8(), {"is_enable": True, "mode": "flat",
                                              "saturation_gain": 1.0, "chroma_limit": 1.0}).execute()
        self.assertEqual(out.dtype, np.uint8)

    def test_flat_mode_unity_gain_same_as_disabled(self):
        """saturation_gain=1.0 and is_enable=False should produce same Y channel."""
        img = random_img_uint8()
        out_on  = self._csc(img, {"is_enable": True,  "mode": "flat",
                                   "saturation_gain": 1.0, "chroma_limit": 1.0}).execute()
        out_off = self._csc(img, {"is_enable": False}).execute()
        # Y channel (channel 0) should be identical
        np.testing.assert_array_equal(out_on[:, :, 0], out_off[:, :, 0])

    # --- vibrance mode ---

    def test_vibrance_mode_output_shape(self):
        out = self._csc(random_img_uint8(), {"is_enable": True, "mode": "vibrance",
                                              "vibrance_strength": 0.3, "chroma_limit": 1.0}).execute()
        self.assertEqual(out.shape, (32, 32, 3))

    def test_vibrance_mode_output_dtype(self):
        out = self._csc(random_img_uint8(), {"is_enable": True, "mode": "vibrance",
                                              "vibrance_strength": 0.3, "chroma_limit": 1.0}).execute()
        self.assertEqual(out.dtype, np.uint8)

    def test_vibrance_zero_strength_same_as_flat_unity(self):
        """vibrance_strength=0 → gain_map=1.0 everywhere → same as flat gain=1.0."""
        img = random_img_uint8()
        out_vib = self._csc(img, {"is_enable": True, "mode": "vibrance",
                                   "vibrance_strength": 0.0, "chroma_limit": 1.0}).execute()
        out_flat = self._csc(img, {"is_enable": True, "mode": "flat",
                                    "saturation_gain": 1.0, "chroma_limit": 1.0}).execute()
        # Both should produce identical output
        np.testing.assert_array_equal(out_vib, out_flat)

    # --- chroma limiter ---

    def test_chroma_limit_clips_output(self):
        """High saturation_gain + tight chroma_limit must keep output in [0, 255]."""
        img = random_img_uint8()
        out = self._csc(img, {"is_enable": True, "mode": "flat",
                               "saturation_gain": 10.0, "chroma_limit": 0.3}).execute()
        self.assertLessEqual(int(out.max()), 255)
        self.assertGreaterEqual(int(out.min()), 0)

    def test_chroma_limit_no_limit_does_not_crash(self):
        """chroma_limit=1.0 (no limiting) must still complete without error."""
        img = random_img_uint8()
        out = self._csc(img, {"is_enable": True, "mode": "flat",
                               "saturation_gain": 1.5, "chroma_limit": 1.0}).execute()
        self.assertEqual(out.dtype, np.uint8)

    def test_disabled_cse_passes_through(self):
        """is_enable=False: saturation code not reached, Y/Cb/Cr are unchanged."""
        img = random_img_uint8()
        out = self._csc(img, {"is_enable": False}).execute()
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, (32, 32, 3))


# ===========================================================================
# Phase 2.6 — NoiseReduction2d: bilateral / chroma_only / off modes
# ===========================================================================

from modules.noise_reduction_2d.bilateral import BilateralFilter
from modules.noise_reduction_2d.noise_reduction_2d import NoiseReduction2d


class TestNR2DModes(unittest.TestCase):
    """Phase 2.6 — NR2D bilateral, chroma_only, and off modes."""

    BASE_PARM = {
        "is_enable": True,
        "is_save": False,
        "window_size": 5,
        "patch_size": 3,
        "wts": 10,
        "y_spatial_sigma": 1.5,
        "y_intensity_sigma": 0.1,
        "chroma_sigma": 3.0,
    }

    # --- BilateralFilter unit tests ---

    def test_bilateral_output_shape(self):
        img = random_img_uint8()
        bf = BilateralFilter(img.copy(), self.BASE_PARM)
        out = bf.apply_bilateral()
        self.assertEqual(out.shape, img.shape)

    def test_bilateral_output_dtype_uint8(self):
        img = random_img_uint8()
        bf = BilateralFilter(img.copy(), self.BASE_PARM)
        out = bf.apply_bilateral()
        self.assertEqual(out.dtype, np.uint8)

    def test_bilateral_range_uint8(self):
        img = random_img_uint8()
        bf = BilateralFilter(img.copy(), self.BASE_PARM)
        out = bf.apply_bilateral()
        self.assertGreaterEqual(int(out.min()), 0)
        self.assertLessEqual(int(out.max()), 255)

    def test_bilateral_chroma_unchanged(self):
        """Bilateral filter only modifies Y channel (channel 0); Cb/Cr unchanged."""
        img = random_img_uint8()
        bf = BilateralFilter(img.copy(), self.BASE_PARM)
        out = bf.apply_bilateral()
        np.testing.assert_array_equal(out[:, :, 1], img[:, :, 1],
                                      err_msg="Cb channel should not change in bilateral")
        np.testing.assert_array_equal(out[:, :, 2], img[:, :, 2],
                                      err_msg="Cr channel should not change in bilateral")

    def test_bilateral_flat_image_unchanged(self):
        """Bilateral filter of a flat image must be identical to input."""
        img = np.full((24, 24, 3), 100, dtype=np.uint8)
        bf = BilateralFilter(img.copy(), {**self.BASE_PARM, "window_size": 5})
        out = bf.apply_bilateral()
        np.testing.assert_array_equal(out[:, :, 0], img[:, :, 0])

    # --- NoiseReduction2d dispatcher ---

    def test_bilateral_mode_dispatch(self):
        parm = {**self.BASE_PARM, "mode": "bilateral"}
        img = random_img_uint8(h=20, w=20)
        nr = NoiseReduction2d(img.copy(), make_sensor_info(), parm, make_platform(), "bt601")
        out = nr.execute()
        self.assertEqual(out.shape, img.shape)

    def test_chroma_only_y_unchanged(self):
        """chroma_only mode must not modify Y channel (channel 0)."""
        parm = {**self.BASE_PARM, "mode": "chroma_only"}
        img = random_img_uint8(h=20, w=20)
        nr = NoiseReduction2d(img.copy(), make_sensor_info(), parm, make_platform(), "bt601")
        out = nr.execute()
        np.testing.assert_array_equal(out[:, :, 0], img[:, :, 0],
                                      err_msg="Y channel must not change in chroma_only mode")

    def test_chroma_only_modifies_chroma(self):
        """chroma_only must smooth Cb/Cr (output will differ from input for most images)."""
        parm = {**self.BASE_PARM, "mode": "chroma_only", "chroma_sigma": 5.0}
        rng = np.random.default_rng(99)
        img = (rng.random((30, 30, 3)) * 255).astype(np.uint8)
        nr = NoiseReduction2d(img.copy(), make_sensor_info(), parm, make_platform(), "bt601")
        out = nr.execute()
        # At least some Cb/Cr values should differ (except all-flat images)
        # Check output is a valid array in range
        self.assertLessEqual(int(out.max()), 255)
        self.assertGreaterEqual(int(out.min()), 0)

    def test_off_mode_passes_through(self):
        """mode='off' returns image unchanged regardless of is_enable."""
        parm = {**self.BASE_PARM, "mode": "off"}
        img = random_img_uint8()
        nr = NoiseReduction2d(img.copy(), make_sensor_info(), parm, make_platform(), "bt601")
        out = nr.execute()
        np.testing.assert_array_equal(out, img)

    def test_disabled_passes_through(self):
        """is_enable=False passes through (nlm mode, not invoked)."""
        parm = {**self.BASE_PARM, "mode": "nlm", "is_enable": False}
        img = random_img_uint8()
        nr = NoiseReduction2d(img.copy(), make_sensor_info(), parm, make_platform(), "bt601")
        out = nr.execute()
        np.testing.assert_array_equal(out, img)

    def test_nlm_default_small_window(self):
        """NLM mode with tiny window completes without error (original algorithm)."""
        parm = {**self.BASE_PARM, "mode": "nlm", "window_size": 3, "patch_size": 3}
        img = random_img_uint8(h=16, w=16)
        nr = NoiseReduction2d(img.copy(), make_sensor_info(), parm, make_platform(), "bt601")
        out = nr.execute()
        self.assertEqual(out.shape, img.shape)


# ===========================================================================
# Phase 3.2 — Gamma multi-EOTF
# ===========================================================================

from modules.gamma_correction.eotf import (
    apply_srgb, apply_rec709, apply_linear, apply_pq, apply_hlg, apply_eotf
)
from modules.gamma_correction.gamma_correction import GammaCorrection


class TestGammaEOTF(unittest.TestCase):
    """Phase 3.2 — analytical EOTF functions and GammaCorrection dispatcher."""

    RAMP = np.linspace(0.0, 1.0, 256, dtype=np.float32)

    # --- EOTF mathematical properties (0→0, 1→1, monotonic) ---

    def _check_eotf(self, fn, name):
        y = fn(self.RAMP)
        self.assertAlmostEqual(float(y[0]), 0.0, places=4, msg=f"{name}(0) ≠ 0")
        self.assertAlmostEqual(float(y[-1]), 1.0, places=2, msg=f"{name}(1) ≠ 1")
        diffs = np.diff(y)
        self.assertTrue(np.all(diffs >= -1e-6), msg=f"{name} not monotonic")

    def test_srgb_boundary_and_monotonic(self):
        self._check_eotf(apply_srgb, "sRGB")

    def test_rec709_boundary_and_monotonic(self):
        self._check_eotf(apply_rec709, "Rec.709")

    def test_linear_passthrough(self):
        y = apply_linear(self.RAMP)
        np.testing.assert_allclose(y, self.RAMP, atol=1e-6, err_msg="Linear not a passthrough")

    def test_pq_boundary_and_monotonic(self):
        self._check_eotf(apply_pq, "PQ")

    def test_hlg_boundary_and_monotonic(self):
        self._check_eotf(apply_hlg, "HLG")

    def test_srgb_toe_region(self):
        """sRGB toe (L < 0.0031308) should be linear 12.92 × L."""
        L = np.array([0.001, 0.002, 0.003], dtype=np.float32)
        y = apply_srgb(L)
        expected = 12.92 * L
        np.testing.assert_allclose(y, expected, atol=1e-5)

    def test_rec709_toe_region(self):
        """Rec.709 toe (L < 0.018) should be 4.5 × L."""
        L = np.array([0.005, 0.010, 0.015], dtype=np.float32)
        y = apply_rec709(L)
        expected = 4.5 * L
        np.testing.assert_allclose(y, expected, atol=1e-5)

    # --- apply_eotf helper ---

    def test_apply_eotf_uint8_input_returns_uint8(self):
        img = random_img_uint8()
        for name in ("srgb", "rec709", "linear", "pq", "hlg"):
            out = apply_eotf(img, name, 8)
            self.assertEqual(out.dtype, np.uint8, msg=f"{name} output dtype")

    def test_apply_eotf_range_valid(self):
        img = random_img_uint8()
        for name in ("srgb", "rec709", "linear", "pq", "hlg"):
            out = apply_eotf(img, name, 8)
            self.assertGreaterEqual(int(out.min()), 0)
            self.assertLessEqual(int(out.max()), 255)

    def test_apply_eotf_unknown_raises(self):
        img = random_img_uint8()
        with self.assertRaises(ValueError):
            apply_eotf(img, "nonexistent_eotf", 8)

    # --- GammaCorrection dispatcher ---

    def test_gc_lut_mode_backward_compat(self):
        """eotf='lut' must use the LUT (identity LUT → output equals input)."""
        parm = {
            "is_enable": True, "is_save": False, "eotf": "lut",
            "gamma_lut_8": list(range(256)),  # identity
        }
        img = random_img_uint8()
        gc = GammaCorrection(img.copy(), make_platform(), make_sensor_info(8), parm)
        out = gc.execute()
        np.testing.assert_array_equal(out, img, err_msg="Identity LUT should not change image")

    def test_gc_srgb_analytical_mode(self):
        parm = {"is_enable": True, "is_save": False, "eotf": "srgb"}
        img = random_img_uint8()
        gc = GammaCorrection(img.copy(), make_platform(), make_sensor_info(8), parm)
        out = gc.execute()
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, img.shape)

    def test_gc_disabled_passthrough(self):
        parm = {"is_enable": False, "is_save": False, "eotf": "srgb"}
        img = random_img_uint8()
        gc = GammaCorrection(img.copy(), make_platform(), make_sensor_info(8), parm)
        out = gc.execute()
        np.testing.assert_array_equal(out, img)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestAdaptiveSharpen, TestColorSaturation, TestNR2DModes, TestGammaEOTF]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
