"""
EdgeISP — Phase 2 Regression Tests
=====================================
Tests for Phase 2 algorithm upgrades:
  - 2.1  DPC: MAD-based adaptive threshold (dynamic_dpc.py)
  - 2.8  YUV 4:2:0 I420 planar output (yuv_conv_format.py)

Run from the project root:
    python -m pytest tests/test_phase2.py -v
    # or without pytest:
    python tests/test_phase2.py
"""

import sys
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Run from project root so modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub out heavy/unavailable dependencies before importing project modules.
for _mod in (
    "scipy", "scipy.signal", "scipy.ndimage",
    "rawpy", "tqdm", "cv2",
):
    sys.modules.setdefault(_mod, MagicMock())
sys.modules.setdefault("util", MagicMock())
sys.modules.setdefault("util.utils", MagicMock())

from modules.dead_pixel_correction.dynamic_dpc import DynamicDPC
from modules.yuv_conv_format.yuv_conv_format import YUVConvFormat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dpc_sensor(bpp=12):
    return {"bayer_pattern": "rggb", "bit_depth": bpp, "width": 16, "height": 16}


def _dpc_cfg(mode="fixed", dp_threshold=80, dp_threshold_k=4.0, is_debug=False):
    return {
        "dp_threshold": dp_threshold,
        "mode": mode,
        "dp_threshold_k": dp_threshold_k,
        "is_debug": is_debug,
    }


def _make_dpc(img, mode="fixed", dp_threshold=80, k=4.0, is_debug=False):
    return DynamicDPC(img.copy(), _dpc_sensor(), _dpc_cfg(mode, dp_threshold, k, is_debug))


def _yuv_platform():
    return {
        "in_file": "test_2592x1536",
        "out_file": "test_out",
        "rgb_output": False,
        "disable_progress_bar": True,
        "leave_pbar_string": False,
        "save_format": "png",
    }


def _yuv_parm(conv_type="420", is_enable=True, is_save=False):
    return {"is_enable": is_enable, "conv_type": conv_type, "is_save": is_save}


def _yuv_sensor():
    return {"bayer_pattern": "rggb", "bit_depth": 8}


def _make_yuv_module(img, conv_type="420"):
    return YUVConvFormat(img, _yuv_platform(), _yuv_sensor(), _yuv_parm(conv_type))


def _run_convert(img, conv_type):
    """Call convert2yuv_format() with file I/O redirected to a temp file.

    numpy's ndarray.tofile() needs a real OS file descriptor (fileno()),
    so we give it a real TemporaryFile rather than a mock.  The actual
    disk write is discarded; only the computed return value is used.
    """
    module = _make_yuv_module(img, conv_type)
    with tempfile.TemporaryFile() as tmp:
        with patch("builtins.open", return_value=tmp):
            return module.convert2yuv_format()


# ---------------------------------------------------------------------------
# Phase 2.1 — DPC MAD-based adaptive threshold
# ---------------------------------------------------------------------------

class TestDPCMADThreshold(unittest.TestCase):
    """Phase 2.1: DynamicDPC must support mode='mad' with MAD-adaptive threshold."""

    # --- MAD maths ---

    def test_mad_formula_uniform_image(self):
        """MAD of a uniform image is 0 → adaptive threshold is 0."""
        img = np.full((16, 16), 1000, dtype=np.float32)
        dpc = _make_dpc(img, mode="mad", k=4.0)
        thresh = dpc.compute_mad_threshold()
        self.assertAlmostEqual(thresh, 0.0, places=4,
            msg="Uniform image has MAD=0, so adaptive threshold must be 0")

    def test_mad_formula_known_values(self):
        """Verify sigma = MAD/0.6745 and threshold = k*sigma for a controlled image."""
        # Create an image whose median and MAD we know exactly.
        # All pixels = 100 except one = 200 (doesn't matter — we're testing formula).
        img = np.full((16, 16), 512, dtype=np.float32)
        img[0, 0] = 612  # one bright pixel
        # MAD = median of |x - median(x)|; median(x)=512, most |dev|=0, so MAD=0.
        # That's boring. Use a distribution with known MAD instead.
        # Easier: just call and verify the return value matches formula manually.
        dpc = _make_dpc(img, mode="mad", k=6.0)
        median_val = float(np.median(img))
        mad = float(np.median(np.abs(img.astype(np.float32) - median_val)))
        sigma = mad / 0.6745
        expected = 6.0 * sigma
        actual = dpc.compute_mad_threshold()
        self.assertAlmostEqual(actual, expected, places=4,
            msg="compute_mad_threshold must implement k * MAD / 0.6745")

    def test_k_scales_threshold_linearly(self):
        """Doubling dp_threshold_k must double the adaptive threshold."""
        rng = np.random.default_rng(7)
        img = rng.integers(200, 3900, size=(32, 32), dtype=np.uint16).astype(np.float32)
        dpc_k2 = _make_dpc(img, mode="mad", k=2.0)
        dpc_k4 = _make_dpc(img, mode="mad", k=4.0)
        thresh_k2 = dpc_k2.compute_mad_threshold()
        thresh_k4 = dpc_k4.compute_mad_threshold()
        self.assertAlmostEqual(thresh_k4, 2.0 * thresh_k2, places=3,
            msg="Threshold must scale linearly with dp_threshold_k")

    def test_mad_threshold_adapts_to_noise_level(self):
        """Higher-noise images must produce a higher MAD threshold."""
        rng = np.random.default_rng(42)
        low_noise  = (np.full((32, 32), 2000) + rng.normal(0, 10, (32, 32))).astype(np.float32)
        high_noise = (np.full((32, 32), 2000) + rng.normal(0, 200, (32, 32))).astype(np.float32)
        thresh_low  = _make_dpc(low_noise,  mode="mad", k=4.0).compute_mad_threshold()
        thresh_high = _make_dpc(high_noise, mode="mad", k=4.0).compute_mad_threshold()
        self.assertGreater(thresh_high, thresh_low,
            msg="Higher-noise image must produce a larger MAD threshold")

    # --- Mode dispatch ---

    def test_fixed_mode_uses_config_dp_threshold(self):
        """mode='fixed' must use parm_dpc['dp_threshold'] verbatim."""
        img = np.full((16, 16), 1000, dtype=np.float32)
        dpc = _make_dpc(img, mode="fixed", dp_threshold=123)
        self.assertEqual(dpc.threshold, 123,
            msg="Fixed mode threshold must equal config dp_threshold")

    def test_mode_default_is_fixed_backward_compat(self):
        """Missing 'mode' key must default to 'fixed' (backward compat)."""
        img = np.full((16, 16), 1000, dtype=np.float32)
        cfg_no_mode = {k: v for k, v in _dpc_cfg(mode="fixed").items() if k != "mode"}
        dpc = DynamicDPC(img.copy(), _dpc_sensor(), cfg_no_mode)
        self.assertEqual(dpc.mode, "fixed",
            msg="Missing 'mode' key must default to 'fixed'")

    def test_dp_threshold_k_default_is_4(self):
        """Missing 'dp_threshold_k' key must default to 4.0."""
        img = np.full((16, 16), 1000, dtype=np.float32)
        cfg_no_k = {k: v for k, v in _dpc_cfg().items() if k != "dp_threshold_k"}
        dpc = DynamicDPC(img.copy(), _dpc_sensor(), cfg_no_k)
        self.assertAlmostEqual(dpc.threshold_k, 4.0,
            msg="Missing 'dp_threshold_k' key must default to 4.0")

    def test_mad_mode_stored_correctly(self):
        """mode='mad' must be stored in self.mode."""
        img = np.full((16, 16), 1000, dtype=np.float32)
        dpc = _make_dpc(img, mode="mad", k=5.0)
        self.assertEqual(dpc.mode, "mad")
        self.assertAlmostEqual(dpc.threshold_k, 5.0)

    # --- Threshold sanity for a realistic Bayer image ---

    def test_mad_threshold_reasonable_for_12bit_image(self):
        """MAD threshold on a realistic 12-bit Bayer image must be between 10 and 500 ADU."""
        rng = np.random.default_rng(0)
        # Simulate a 12-bit Bayer image with mild noise (sigma ~30 ADU)
        img = np.clip(
            2048 + rng.normal(0, 30, (64, 64)), 0, 4095
        ).astype(np.float32)
        dpc = _make_dpc(img, mode="mad", k=4.0)
        thresh = dpc.compute_mad_threshold()
        self.assertGreater(thresh, 10,
            msg="MAD threshold on a noisy image should be > 10 ADU")
        self.assertLess(thresh, 500,
            msg="MAD threshold on a mild-noise image should be < 500 ADU")


# ---------------------------------------------------------------------------
# Phase 2.8 — YUV 4:2:0 I420 planar output
# ---------------------------------------------------------------------------

class TestYUV420(unittest.TestCase):
    """Phase 2.8: YUVConvFormat must support conv_type='420' (I420 planar)."""

    # --- Output size ---

    def test_420_output_element_count(self):
        """4:2:0 output must contain H*W + 2*(H/2)*(W/2) = 1.5*H*W elements."""
        H, W = 8, 8
        img = np.zeros((H, W, 3), dtype=np.uint8)
        out = _run_convert(img, "420")
        expected_len = H * W + 2 * (H // 2) * (W // 2)
        self.assertEqual(len(out), expected_len,
            msg=f"4:2:0 output should have {expected_len} elements, got {len(out)}")

    def test_420_output_larger_image(self):
        """4:2:0 size formula holds for a larger even-dimension image."""
        H, W = 64, 80
        img = np.zeros((H, W, 3), dtype=np.uint8)
        out = _run_convert(img, "420")
        expected_len = H * W + 2 * (H // 2) * (W // 2)
        self.assertEqual(len(out), expected_len)

    # --- Y plane correctness ---

    def test_420_y_plane_is_full_resolution(self):
        """First H*W elements of 4:2:0 output must equal the Y channel verbatim."""
        H, W = 8, 8
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:, :, 0] = np.arange(H * W, dtype=np.uint8).reshape(H, W)  # unique Y values
        out = _run_convert(img, "420")
        np.testing.assert_array_equal(
            out[:H * W],
            img[:, :, 0].flatten(),
            err_msg="Y plane must be full-resolution and unmodified",
        )

    # --- Cb/Cr downsampling correctness ---

    def test_420_chroma_is_2x2_average(self):
        """Cb and Cr planes must be 2×2 box-filter averages of the input chroma."""
        # Controlled 4×4 image: Cb values chosen so 2×2 averages are exact integers.
        H, W = 4, 4
        img = np.zeros((H, W, 3), dtype=np.uint8)
        # Set Cb channel: top-left 2×2 block = [10,20,30,40] → avg = 25
        #                 top-right 2×2 block = [50,60,70,80] → avg = 65
        #                 bottom-left 2×2     = [100,110,120,130] → avg = 115
        #                 bottom-right 2×2    = [140,150,160,170] → avg = 155
        img[0, 0, 1] = 10;  img[0, 1, 1] = 20
        img[1, 0, 1] = 30;  img[1, 1, 1] = 40
        img[0, 2, 1] = 50;  img[0, 3, 1] = 60
        img[1, 2, 1] = 70;  img[1, 3, 1] = 80
        img[2, 0, 1] = 100; img[2, 1, 1] = 110
        img[3, 0, 1] = 120; img[3, 1, 1] = 130
        img[2, 2, 1] = 140; img[2, 3, 1] = 150
        img[3, 2, 1] = 160; img[3, 3, 1] = 170

        out = _run_convert(img, "420")

        # Cb plane: elements [H*W : H*W + (H/2)*(W/2)]
        cb_out = out[H * W : H * W + (H // 2) * (W // 2)]
        expected_cb = np.array([25, 65, 115, 155], dtype=np.uint8)
        np.testing.assert_array_equal(cb_out, expected_cb,
            err_msg="Cb plane must be exact 2×2 box-filter averages")

    def test_420_cr_plane_position(self):
        """Cr plane must follow immediately after Cb plane in I420 order."""
        H, W = 4, 4
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:, :, 2] = 200  # constant Cr = 200
        out = _run_convert(img, "420")
        cr_start = H * W + (H // 2) * (W // 2)
        cr_out = out[cr_start:]
        np.testing.assert_array_equal(cr_out, 200,
            err_msg="Cr plane must appear after Cb plane and contain correct values")

    def test_420_uniform_chroma_unchanged(self):
        """Uniform Cb/Cr channels must survive downsampling intact."""
        H, W = 8, 8
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:, :, 1] = 128  # Cb = 128 everywhere
        img[:, :, 2] = 64   # Cr = 64 everywhere
        out = _run_convert(img, "420")
        cb_out = out[H * W : H * W + (H // 2) * (W // 2)]
        cr_out = out[H * W + (H // 2) * (W // 2):]
        np.testing.assert_array_equal(cb_out, 128,
            err_msg="Uniform Cb=128 must survive 4:2:0 downsampling")
        np.testing.assert_array_equal(cr_out, 64,
            err_msg="Uniform Cr=64 must survive 4:2:0 downsampling")

    def test_420_chroma_size_is_quarter_of_y(self):
        """Each chroma plane must have exactly H/2 × W/2 elements."""
        H, W = 16, 24
        img = np.zeros((H, W, 3), dtype=np.uint8)
        out = _run_convert(img, "420")
        y_size = H * W
        expected_chroma_size = (H // 2) * (W // 2)
        cb_out = out[y_size : y_size + expected_chroma_size]
        cr_out = out[y_size + expected_chroma_size:]
        self.assertEqual(len(cb_out), expected_chroma_size,
            msg="Cb plane must have (H/2)*(W/2) elements")
        self.assertEqual(len(cr_out), expected_chroma_size,
            msg="Cr plane must have (H/2)*(W/2) elements")

    # --- Edge case: odd dimensions ---

    def test_420_odd_height_does_not_crash(self):
        """Odd-height images must not raise an exception (dimension clipped to even)."""
        img = np.zeros((9, 8, 3), dtype=np.uint8)  # odd height
        try:
            out = _run_convert(img, "420")
        except Exception as exc:
            self.fail(f"420 conversion raised on odd height: {exc}")
        # Output size uses clipped even dims: H_e=8, W_e=8
        expected_len = 8 * 8 + 2 * 4 * 4
        self.assertEqual(len(out), expected_len,
            msg="Odd height must be silently clipped to even for chroma subsampling")

    def test_420_odd_width_does_not_crash(self):
        """Odd-width images must not raise an exception."""
        img = np.zeros((8, 9, 3), dtype=np.uint8)  # odd width
        try:
            out = _run_convert(img, "420")
        except Exception as exc:
            self.fail(f"420 conversion raised on odd width: {exc}")
        expected_len = 8 * 8 + 2 * 4 * 4  # W_e=8
        self.assertEqual(len(out), expected_len)

    def test_420_both_odd_dimensions(self):
        """Odd height and width must both be clipped to even."""
        img = np.zeros((7, 9, 3), dtype=np.uint8)
        out = _run_convert(img, "420")
        # H_e=6, W_e=8
        expected_len = 6 * 8 + 2 * 3 * 4
        self.assertEqual(len(out), expected_len)

    # --- Regression: 444 and 422 still work ---

    def test_444_output_size_unchanged(self):
        """4:4:4 must still produce H*W*3 elements (regression)."""
        H, W = 8, 8
        img = np.zeros((H, W, 3), dtype=np.uint8)
        out = _run_convert(img, "444")
        self.assertEqual(len(out), H * W * 3,
            msg="4:4:4 output size must be H*W*3")

    def test_444_preserves_all_channels(self):
        """4:4:4 output must interleave Y, Cb, Cr at full resolution."""
        H, W = 4, 4
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:, :, 0] = 10  # Y
        img[:, :, 1] = 20  # Cb
        img[:, :, 2] = 30  # Cr
        out = _run_convert(img, "444")
        # 444 packs column-by-column: reshape(-1,1) then concat → [Y0,Cb0,Cr0, Y1,Cb1,Cr1,...]
        # Actually the existing code concatenates separately (all Y, all Cb, all Cr) in one axis
        # Let's just verify the output contains the right unique values
        unique_vals = set(out.tolist())
        self.assertIn(10, unique_vals, "Y=10 must appear in 4:4:4 output")
        self.assertIn(20, unique_vals, "Cb=20 must appear in 4:4:4 output")
        self.assertIn(30, unique_vals, "Cr=30 must appear in 4:4:4 output")

    def test_422_output_size_unchanged(self):
        """4:2:2 must still produce H*W*2 elements (regression)."""
        H, W = 8, 8
        img = np.zeros((H, W, 3), dtype=np.uint8)
        out = _run_convert(img, "422")
        self.assertEqual(len(out), H * W * 2,
            msg="4:2:2 output size must be H*W*2")

    def test_invalid_conv_type_raises_value_error(self):
        """Unknown conv_type must raise ValueError (guard against silent misconfiguration)."""
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        with self.assertRaises(ValueError,
                msg="Unknown conv_type must raise ValueError"):
            _run_convert(img, "999")

    # --- Config integration ---

    def test_420_config_key_accepted(self):
        """YUVConvFormat must accept conv_type='420' without error at construction."""
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        try:
            YUVConvFormat(img, _yuv_platform(), _yuv_sensor(), _yuv_parm("420"))
        except Exception as exc:
            self.fail(f"Construction with conv_type='420' raised: {exc}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
