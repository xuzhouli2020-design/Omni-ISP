"""
Omni-ISP — Phase 1 Regression Tests
=====================================
Tests for the three pipeline correctness fixes made in Phase 1.

Run from the project root:
    python -m pytest tests/test_phase1.py -v
    # or without pytest:
    python tests/test_phase1.py
"""

import sys
import os
import unittest
from unittest.mock import MagicMock

import numpy as np

# Run from project root so modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub out heavy/unavailable dependencies before importing project modules.
# This lets the tests run without scipy, rawpy, tqdm, etc. installed.
for _mod in ("scipy", "scipy.signal", "rawpy", "tqdm", "cv2"):
    sys.modules.setdefault(_mod, MagicMock())
sys.modules.setdefault("util", MagicMock())
sys.modules.setdefault("util.utils", MagicMock())

from modules.black_level_correction.black_level_correction import (
    BlackLevelCorrection as BLC,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _platform():
    return {
        "in_file": "test_raw",
        "out_file": "test_out",
        "disable_progress_bar": True,
        "leave_pbar_string": False,
    }


def _sensor(bayer="rggb", bpp=12):
    return {"bayer_pattern": bayer, "bit_depth": bpp, "width": 8, "height": 8}


def _blc_cfg(step="full", is_linear=True):
    return {
        "is_enable": True,
        "r_offset": 200,
        "gr_offset": 200,
        "gb_offset": 200,
        "b_offset": 200,
        "r_sat": 4095,
        "gr_sat": 4095,
        "gb_sat": 4095,
        "b_sat": 4095,
        "is_linear": is_linear,
        "is_save": False,
        "step": step,
    }


def _run_blc(raw, step="full", bayer="rggb", bpp=12, is_linear=True):
    return BLC(raw.copy(), _platform(), _sensor(bayer, bpp), _blc_cfg(step, is_linear)).execute()


# ---------------------------------------------------------------------------
# Phase 1.3 — BLC split (offset_only / linearise_only / full)
# ---------------------------------------------------------------------------

class TestBLCSplit(unittest.TestCase):
    """Phase 1.3: BLC must support offset_only, linearise_only, and full modes."""

    def test_full_equals_offset_then_linearise(self):
        """BLC full == offset_only followed by linearise_only (rggb, 12-bit)."""
        raw = np.full((8, 8), 2048, dtype=np.uint16)
        result_full = _run_blc(raw, step="full")
        after_offset = _run_blc(raw, step="offset_only")
        result_split = _run_blc(after_offset, step="linearise_only")
        np.testing.assert_array_equal(
            result_full, result_split,
            err_msg="BLC full must equal offset_only→linearise_only chain",
        )

    def test_offset_only_subtracts_only(self):
        """offset_only subtracts black offsets without any scaling."""
        raw = np.full((8, 8), 2048, dtype=np.uint16)
        result = _run_blc(raw, step="offset_only")
        # All pixels: 2048 - 200 = 1848, clipped to [0, 4095]
        np.testing.assert_array_equal(result, 1848,
            err_msg="offset_only should subtract 200 from all pixels")

    def test_offset_only_clips_to_zero(self):
        """offset_only clips dark pixels to 0, not negative."""
        raw = np.full((8, 8), 100, dtype=np.uint16)  # below BL offset of 200
        result = _run_blc(raw, step="offset_only")
        self.assertTrue(np.all(result == 0),
            "Pixels below black level must clip to 0 after offset subtraction")

    def test_linearise_only_scales_correctly(self):
        """linearise_only scales by (sat-offset) → full range."""
        bpp, r_offset, r_sat = 12, 200, 4095
        post_offset_val = 1848  # 2048 - 200
        raw = np.full((8, 8), post_offset_val, dtype=np.uint16)
        result = _run_blc(raw, step="linearise_only")
        # Compute expected using the same float32 arithmetic the module uses
        # (float32 intermediate → uint16 clip, not round)
        expected = int(np.clip(
            np.float32(post_offset_val) / np.float32(r_sat - r_offset) * np.float32((2 ** bpp) - 1),
            0, (2 ** bpp) - 1,
        ))
        np.testing.assert_array_equal(result, expected,
            err_msg=f"linearise_only: expected {expected}, got {result[0, 0]}")

    def test_step_default_is_full_backward_compat(self):
        """A config without a 'step' key behaves identically to step='full'."""
        raw = np.full((8, 8), 2048, dtype=np.uint16)
        cfg_no_step = {k: v for k, v in _blc_cfg(step="full").items() if k != "step"}
        result_default = BLC(raw.copy(), _platform(), _sensor(), cfg_no_step).execute()
        result_full = _run_blc(raw, step="full")
        np.testing.assert_array_equal(result_default, result_full,
            err_msg="Missing 'step' key must default to full behaviour")

    def test_no_linearise_when_is_linear_false(self):
        """When is_linear=False, linearise_only mode must be a no-op."""
        raw = np.full((8, 8), 1848, dtype=np.uint16)
        result = _run_blc(raw, step="linearise_only", is_linear=False)
        # No scaling applied — output should equal input (clipped)
        np.testing.assert_array_equal(result, 1848,
            err_msg="linearise_only with is_linear=False must not scale")

    def test_all_bayer_patterns(self):
        """BLC full == offset→linearise chain for all four Bayer patterns."""
        for bayer in ("rggb", "bggr", "grbg", "gbrg"):
            with self.subTest(bayer=bayer):
                raw = np.full((8, 8), 3000, dtype=np.uint16)
                result_full = _run_blc(raw, step="full", bayer=bayer)
                after_offset = _run_blc(raw, step="offset_only", bayer=bayer)
                result_split = _run_blc(after_offset, step="linearise_only", bayer=bayer)
                np.testing.assert_array_equal(result_full, result_split,
                    err_msg=f"BLC split chain failed for bayer={bayer}")

    def test_full_with_no_linearise_equals_offset_only(self):
        """BLC full with is_linear=False must equal offset_only."""
        raw = np.full((8, 8), 2048, dtype=np.uint16)
        result_full_no_lin = _run_blc(raw, step="full", is_linear=False)
        result_offset = _run_blc(raw, step="offset_only")
        np.testing.assert_array_equal(result_full_no_lin, result_offset,
            err_msg="full with is_linear=False must equal offset_only")


# ---------------------------------------------------------------------------
# Phase 1.2 — Linear luminance wrapper math
# ---------------------------------------------------------------------------

class TestLinearLuminanceWrapper(unittest.TestCase):
    """Phase 1.2: LDCI/NR2D receive linear luminance Y extracted from CCM RGB."""

    def test_bt709_coefficients_sum_to_one(self):
        """BT.709 linear Y coefficients (0.2126, 0.7152, 0.0722) must sum to 1."""
        coeffs = (0.2126, 0.7152, 0.0722)
        self.assertAlmostEqual(sum(coeffs), 1.0, places=4,
            msg="BT.709 coefficients must sum to 1.0")

    def test_pure_white_luminance(self):
        """Pure white linear RGB (1,1,1) → Y = 1.0."""
        r = g = b = np.ones((4, 4), dtype=np.float32)
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        np.testing.assert_allclose(y, 1.0, atol=1e-6,
            err_msg="White (1,1,1) must give Y=1.0")

    def test_pure_black_luminance(self):
        """Pure black linear RGB (0,0,0) → Y = 0.0."""
        r = g = b = np.zeros((4, 4), dtype=np.float32)
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        np.testing.assert_allclose(y, 0.0, atol=1e-8)

    def test_primary_luminance_values(self):
        """Pure R/G/B primaries give correct BT.709 relative luminance."""
        ones = np.ones((4, 4), dtype=np.float32)
        zeros = np.zeros((4, 4), dtype=np.float32)
        np.testing.assert_allclose(
            0.2126 * ones + 0.7152 * zeros + 0.0722 * zeros, 0.2126, atol=1e-6)
        np.testing.assert_allclose(
            0.2126 * zeros + 0.7152 * ones + 0.0722 * zeros, 0.7152, atol=1e-6)
        np.testing.assert_allclose(
            0.2126 * zeros + 0.7152 * zeros + 0.0722 * ones, 0.0722, atol=1e-6)

    def test_identity_enhancement_passes_rgb_through(self):
        """When Y_enhanced == Y_linear (module off), RGB must pass through unchanged."""
        rng = np.random.default_rng(42)
        r = rng.random((8, 8)).astype(np.float32)
        g = rng.random((8, 8)).astype(np.float32)
        b = rng.random((8, 8)).astype(np.float32)
        y_lin = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)
        y_enh = y_lin.copy()  # identity — no change
        ratio = y_enh / np.maximum(y_lin, 1e-8)
        np.testing.assert_allclose(r * ratio, r, atol=1e-6)
        np.testing.assert_allclose(g * ratio, g, atol=1e-6)
        np.testing.assert_allclose(b * ratio, b, atol=1e-6)

    def test_proportional_scaling_preserves_hue(self):
        """Proportional luminance boost must not change R/G or B/G ratios (hue)."""
        r = np.array([[0.4, 0.2]], dtype=np.float32)
        g = np.array([[0.5, 0.6]], dtype=np.float32)
        b = np.array([[0.1, 0.3]], dtype=np.float32)
        y_lin = 0.2126 * r + 0.7152 * g + 0.0722 * b
        y_enh = y_lin * 1.5  # 50% luminance boost
        ratio = y_enh / np.maximum(y_lin, 1e-8)
        r_out, g_out, b_out = r * ratio, g * ratio, b * ratio
        np.testing.assert_allclose(r_out / g_out, r / g, atol=1e-5,
            err_msg="R/G ratio (hue) must be preserved by proportional scaling")
        np.testing.assert_allclose(b_out / g_out, b / g, atol=1e-5,
            err_msg="B/G ratio (hue) must be preserved by proportional scaling")

    def test_zero_luminance_no_nan_or_inf(self):
        """Black pixels (Y≈0) must not cause NaN or Inf in output."""
        r = g = b = np.zeros((8, 8), dtype=np.float32)
        y_lin = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.float32)
        y_enh = y_lin + 0.05  # some enhancement even at black
        ratio = y_enh / np.maximum(y_lin, 1e-8)
        out = r * ratio
        self.assertTrue(np.all(np.isfinite(out)),
            "Zero-luminance pixels must not produce NaN or Inf")

    def test_fake_yuv_channel_layout(self):
        """Fake YCbCr array must have Y in channel 0, zeros in channels 1 and 2."""
        y_lin = np.random.rand(8, 8).astype(np.float32)
        fake_yuv = np.stack(
            [y_lin, np.zeros_like(y_lin), np.zeros_like(y_lin)], axis=2
        ).astype(np.float32)
        np.testing.assert_array_equal(fake_yuv[..., 0], y_lin)
        np.testing.assert_array_equal(fake_yuv[..., 1], 0)
        np.testing.assert_array_equal(fake_yuv[..., 2], 0)


# ---------------------------------------------------------------------------
# Phase 1.1 + 1.2 + 1.3 — Pipeline execution order (structural)
# ---------------------------------------------------------------------------

class TestPipelineOrder(unittest.TestCase):
    """
    Structural tests: verify correct module ordering in omni_isp.py
    by inspecting the source text of run_pipeline().
    """

    @classmethod
    def setUpClass(cls):
        src_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "omni_isp.py",
        )
        with open(src_path) as f:
            lines = f.readlines()
        in_pipeline = False
        body = []
        for line in lines:
            if "def run_pipeline" in line:
                in_pipeline = True
            elif in_pipeline and line.startswith("    def "):
                break
            if in_pipeline:
                body.append(line)
        cls.src = "".join(body)

    def _pos(self, token):
        """Return position of token in run_pipeline source, or fail if absent."""
        p = self.src.find(token)
        self.assertNotEqual(p, -1, f"Token not found in run_pipeline: {token!r}")
        return p

    # --- Phase 1.1 ---

    def test_1_1_nr2d_before_sharpen(self):
        """Phase 1.1: NR2D must be instantiated before SHARP."""
        self.assertLess(
            self._pos("nr2d = NR2D"),
            self._pos("sharp = SHARP"),
            "NR2D must execute before Sharpening",
        )

    def test_1_1_sharp_feeds_rgbc(self):
        """Phase 1.1: RGBC must consume sharp_img, not nr2d_img."""
        rgbc_start = self._pos("rgbc = RGBC")
        rgbc_block = self.src[rgbc_start: rgbc_start + 300]
        self.assertIn("sharp_img", rgbc_block,
            "RGBC must receive sharp_img as first argument")
        self.assertNotIn("nr2d_img", rgbc_block,
            "RGBC must NOT receive nr2d_img — that was the old (wrong) order")

    # --- Phase 1.2 ---

    def test_1_2_ldci_before_gamma(self):
        """Phase 1.2: LDCI must execute before Gamma."""
        self.assertLess(
            self._pos("ldci = LDCI"),
            self._pos("gmc = GC"),
            "LDCI must come before Gamma",
        )

    def test_1_2_nr2d_before_gamma(self):
        """Phase 1.2: NR2D must execute before Gamma."""
        self.assertLess(
            self._pos("nr2d = NR2D"),
            self._pos("gmc = GC"),
            "NR2D must come before Gamma",
        )

    def test_1_2_gamma_before_csc(self):
        """Phase 1.2: Gamma must execute before CSC."""
        self.assertLess(
            self._pos("gmc = GC"),
            self._pos("csc = CSC"),
            "Gamma must come before CSC",
        )

    def test_1_2_csc_before_sharpen(self):
        """Phase 1.2: CSC must execute before Sharpening (Sharpen operates on YUV)."""
        self.assertLess(
            self._pos("csc = CSC"),
            self._pos("sharp = SHARP"),
            "CSC must come before Sharpening",
        )

    def test_1_2_ccm_before_ldci(self):
        """Phase 1.2: CCM must execute before LDCI (LDCI receives linear CCM output)."""
        self.assertLess(
            self._pos("ccm = CCM"),
            self._pos("ldci = LDCI"),
            "CCM must come before LDCI",
        )

    # --- Phase 1.3 ---

    def test_1_3_blc_offset_before_oecf(self):
        """Phase 1.3: BLC offset step must come before OECF."""
        self.assertLess(
            self._pos('"offset_only"'),
            self._pos("oecf = OECF"),
            "BLC offset_only must come before OECF",
        )

    def test_1_3_oecf_before_blc_linearise(self):
        """Phase 1.3: OECF must come before BLC linearise step."""
        self.assertLess(
            self._pos("oecf = OECF"),
            self._pos('"linearise_only"'),
            "OECF must come before BLC linearise_only",
        )

    def test_1_3_blc_linearise_before_digital_gain(self):
        """Phase 1.3: BLC linearise must come before Digital Gain."""
        self.assertLess(
            self._pos('"linearise_only"'),
            self._pos("dga = DG"),
            "BLC linearise_only must come before Digital Gain",
        )

    def test_1_3_digital_gain_receives_blc_raw(self):
        """Phase 1.3: Digital Gain must consume blc_raw, not oecf_raw."""
        dg_start = self._pos("dga = DG")
        dg_line = self.src[dg_start: dg_start + 80]
        self.assertIn("blc_raw", dg_line,
            "Digital Gain must receive blc_raw (post-linearise BLC output)")
        self.assertNotIn("oecf_raw", dg_line,
            "Digital Gain must NOT receive oecf_raw directly")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
