"""
test_phase5.py — Tests for Phase 3 features
=============================================
Covers:
  Phase 3.1 — CCM XYZ intermediate architecture (xyz_matrices.py + CCM class)
  Phase 3.3 — Output profile system (output_profile.py)
  Phase 3.4 — Blue noise dithering (dither.py)
  Phase 3.5 — JPEG output encoding (save_pipeline_output)

Run:
    python tests/test_phase5.py
    python -m pytest tests/test_phase5.py -v
"""

import sys
import os
import unittest
import types
from pathlib import Path

# Ensure the project root is on sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np

# ---------------------------------------------------------------------------
# Minimal stubs so tests don't require every heavy dependency
# ---------------------------------------------------------------------------

# scipy stub
if "scipy" not in sys.modules:
    scipy_stub  = types.ModuleType("scipy")
    signal_stub = types.ModuleType("scipy.signal")
    signal_stub.correlate2d = (
        lambda a, b, mode="full", boundary="fill", fillvalue=0:
        np.zeros_like(a, dtype=np.float32)
    )
    scipy_stub.signal = signal_stub
    sys.modules["scipy"]        = scipy_stub
    sys.modules["scipy.signal"] = signal_stub

# rawpy stub
if "rawpy" not in sys.modules:
    sys.modules["rawpy"] = types.ModuleType("rawpy")


# ---------------------------------------------------------------------------
# Phase 3.1 — xyz_matrices.py
# ---------------------------------------------------------------------------

class TestXYZMatrices(unittest.TestCase):
    """Tests for the standard CIE colour matrices module."""

    @classmethod
    def setUpClass(cls):
        from modules.color_correction_matrix.xyz_matrices import (
            get_xyz_to_target, derive_camera_to_xyz,
            M_XYZ_TO_SRGB, M_XYZ_TO_P3, M_XYZ_TO_BT2020,
            M_SRGB_TO_XYZ, M_XYZ_TO_XYZ,
        )
        cls.get_xyz_to_target   = staticmethod(get_xyz_to_target)
        cls.derive_camera_to_xyz = staticmethod(derive_camera_to_xyz)
        cls.M_XYZ_TO_SRGB  = M_XYZ_TO_SRGB
        cls.M_XYZ_TO_P3    = M_XYZ_TO_P3
        cls.M_XYZ_TO_BT2020 = M_XYZ_TO_BT2020
        cls.M_SRGB_TO_XYZ  = M_SRGB_TO_XYZ
        cls.M_XYZ_TO_XYZ   = M_XYZ_TO_XYZ

    def test_get_xyz_to_srgb_shape(self):
        m = self.get_xyz_to_target("srgb")
        self.assertEqual(m.shape, (3, 3))
        self.assertEqual(m.dtype, np.float64)

    def test_get_xyz_to_target_all_keys(self):
        for key in ("srgb", "display_p3", "bt2020", "xyz"):
            with self.subTest(key=key):
                m = self.get_xyz_to_target(key)
                self.assertEqual(m.shape, (3, 3))

    def test_get_xyz_to_target_case_insensitive(self):
        m1 = self.get_xyz_to_target("sRGB")
        m2 = self.get_xyz_to_target("SRGB")
        np.testing.assert_array_equal(m1, m2)

    def test_get_xyz_to_target_invalid_raises(self):
        with self.assertRaises(ValueError):
            self.get_xyz_to_target("fantasy_space")

    def test_xyz_passthrough_is_identity(self):
        m = self.get_xyz_to_target("xyz")
        np.testing.assert_array_almost_equal(m, np.eye(3))

    def test_srgb_xyz_round_trip(self):
        """M_XYZ_TO_SRGB @ M_SRGB_TO_XYZ ≈ I"""
        product = self.M_XYZ_TO_SRGB @ self.M_SRGB_TO_XYZ
        np.testing.assert_array_almost_equal(product, np.eye(3), decimal=5)

    def test_get_xyz_to_target_returns_independent_copy(self):
        m1 = self.get_xyz_to_target("srgb")
        m2 = self.get_xyz_to_target("srgb")
        m1[0, 0] = 999.0
        self.assertNotEqual(m2[0, 0], 999.0)

    def test_derive_camera_to_xyz_identity_ccm(self):
        """Identity CCM → result should equal M_SRGB_TO_XYZ."""
        out = self.derive_camera_to_xyz(np.eye(3, dtype=np.float32))
        np.testing.assert_array_almost_equal(out, self.M_SRGB_TO_XYZ, decimal=6)

    def test_derive_camera_to_xyz_shape_and_dtype(self):
        ccm = np.random.rand(3, 3).astype(np.float32)
        out = self.derive_camera_to_xyz(ccm)
        self.assertEqual(out.shape, (3, 3))
        self.assertEqual(out.dtype, np.float64)


# ---------------------------------------------------------------------------
# Phase 3.1 — ColorCorrectionMatrix XYZ mode
# ---------------------------------------------------------------------------

class TestCCMXYZMode(unittest.TestCase):
    """Tests for the XYZ intermediate architecture in the CCM class."""

    @classmethod
    def setUpClass(cls):
        from modules.color_correction_matrix.color_correction_matrix import (
            ColorCorrectionMatrix,
        )
        cls.CCM = ColorCorrectionMatrix

    def _make_ccm(self, mode="direct", target="srgb", bit_depth=8, extra=None):
        parm = {
            "is_enable": True,
            "is_save": False,
            "corrected_red":   [1.0, 0.0, 0.0],
            "corrected_green": [0.0, 1.0, 0.0],
            "corrected_blue":  [0.0, 0.0, 1.0],
            "mode": mode,
            "target": target,
        }
        if extra:
            parm.update(extra)
        sensor = {"bit_depth": bit_depth, "bayer_pattern": "rggb"}
        img    = np.ones((4, 4, 3), dtype=np.uint16) * 128
        return self.CCM(img, {}, sensor, parm)

    def test_direct_identity_preserves_values(self):
        ccm = self._make_ccm(mode="direct")
        out = ccm.apply_ccm()
        self.assertEqual(out.dtype, np.uint16)
        np.testing.assert_array_equal(out[0, 0], [128, 128, 128])

    def test_xyz_mode_produces_uint16(self):
        ccm = self._make_ccm(mode="xyz", target="srgb")
        out = ccm.apply_ccm()
        self.assertEqual(out.dtype, np.uint16)

    def test_xyz_mode_no_negatives(self):
        ccm = self._make_ccm(mode="xyz", target="srgb")
        out = ccm.apply_ccm()
        self.assertTrue((out >= 0).all())

    def test_xyz_mode_target_display_p3(self):
        ccm = self._make_ccm(mode="xyz", target="display_p3")
        out = ccm.apply_ccm()
        self.assertEqual(out.shape, (4, 4, 3))

    def test_xyz_mode_target_bt2020(self):
        ccm = self._make_ccm(mode="xyz", target="bt2020")
        out = ccm.apply_ccm()
        self.assertEqual(out.shape, (4, 4, 3))

    def test_xyz_mode_explicit_camera_to_xyz(self):
        from modules.color_correction_matrix.xyz_matrices import M_SRGB_TO_XYZ
        extra = {
            "camera_to_xyz_red":   list(M_SRGB_TO_XYZ[0]),
            "camera_to_xyz_green": list(M_SRGB_TO_XYZ[1]),
            "camera_to_xyz_blue":  list(M_SRGB_TO_XYZ[2]),
        }
        ccm = self._make_ccm(mode="xyz", target="srgb", extra=extra)
        out = ccm.apply_ccm()
        self.assertEqual(out.shape, (4, 4, 3))

    def test_default_mode_is_direct(self):
        """mode key absent → should default to 'direct'."""
        parm = {
            "is_enable": True, "is_save": False,
            "corrected_red":   [1.0, 0.0, 0.0],
            "corrected_green": [0.0, 1.0, 0.0],
            "corrected_blue":  [0.0, 0.0, 1.0],
        }
        sensor = {"bit_depth": 8, "bayer_pattern": "rggb"}
        img    = np.ones((4, 4, 3), dtype=np.uint16) * 64
        ccm    = self.CCM(img, {}, sensor, parm)
        self.assertEqual(ccm.mode, "direct")

    def test_xyz_black_stays_black(self):
        """Zero-valued pixels must remain zero after CCM."""
        for mode in ("direct", "xyz"):
            with self.subTest(mode=mode):
                parm = {
                    "is_enable": True, "is_save": False,
                    "corrected_red":   [1.0, 0.0, 0.0],
                    "corrected_green": [0.0, 1.0, 0.0],
                    "corrected_blue":  [0.0, 0.0, 1.0],
                    "mode": mode, "target": "srgb",
                }
                sensor = {"bit_depth": 8, "bayer_pattern": "rggb"}
                img    = np.zeros((4, 4, 3), dtype=np.uint16)
                ccm    = self.CCM(img, {}, sensor, parm)
                out    = ccm.apply_ccm()
                np.testing.assert_array_equal(out, 0)


# ---------------------------------------------------------------------------
# Phase 3.3 — Output profile system
# ---------------------------------------------------------------------------

class TestOutputProfile(unittest.TestCase):
    """Tests for util/output_profile.py."""

    @classmethod
    def setUpClass(cls):
        from util.output_profile import get_profile, apply_profile_to_params
        cls.get_profile           = staticmethod(get_profile)
        cls.apply_profile_to_params = staticmethod(apply_profile_to_params)

    def test_srgb_profile(self):
        result = self.get_profile("srgb")
        self.assertEqual(result, ("xyz", "srgb", "srgb"))

    def test_rec709_eotf(self):
        _, _, eotf = self.get_profile("rec709")
        self.assertEqual(eotf, "rec709")

    def test_display_p3_target(self):
        _, ccm_target, _ = self.get_profile("display_p3")
        self.assertEqual(ccm_target, "display_p3")

    def test_hdr10_target_and_eotf(self):
        _, ccm_target, eotf = self.get_profile("hdr10")
        self.assertEqual(ccm_target, "bt2020")
        self.assertEqual(eotf, "pq")

    def test_hlg_eotf(self):
        _, _, eotf = self.get_profile("hlg")
        self.assertEqual(eotf, "hlg")

    def test_linear_eotf(self):
        _, _, eotf = self.get_profile("linear")
        self.assertEqual(eotf, "linear")

    def test_custom_returns_none(self):
        self.assertIsNone(self.get_profile("custom"))

    def test_none_returns_none(self):
        self.assertIsNone(self.get_profile(None))

    def test_invalid_profile_raises(self):
        with self.assertRaises(ValueError):
            self.get_profile("fantasy_profile")

    def test_apply_profile_overrides_both_params(self):
        parm_ccm = {"mode": "direct", "target": "srgb"}
        parm_gmc = {"eotf": "lut"}
        self.apply_profile_to_params({"profile": "display_p3"}, parm_ccm, parm_gmc)
        self.assertEqual(parm_ccm["mode"],   "xyz")
        self.assertEqual(parm_ccm["target"], "display_p3")
        self.assertEqual(parm_gmc["eotf"],   "srgb")

    def test_apply_profile_custom_is_noop(self):
        parm_ccm = {"mode": "direct", "target": "srgb"}
        parm_gmc = {"eotf": "lut"}
        self.apply_profile_to_params({"profile": "custom"}, parm_ccm, parm_gmc)
        self.assertEqual(parm_ccm["mode"], "direct")  # unchanged
        self.assertEqual(parm_gmc["eotf"], "lut")     # unchanged

    def test_apply_profile_empty_config_is_noop(self):
        parm_ccm = {"mode": "direct"}
        parm_gmc = {"eotf": "lut"}
        self.apply_profile_to_params({}, parm_ccm, parm_gmc)
        self.assertEqual(parm_ccm["mode"], "direct")  # unchanged


# ---------------------------------------------------------------------------
# Phase 3.4 — Blue noise dithering
# ---------------------------------------------------------------------------

class TestDither(unittest.TestCase):
    """Tests for util/dither.py."""

    @classmethod
    def setUpClass(cls):
        from util.dither import apply_dither, encode_8bit, generate_void_and_cluster
        cls.apply_dither = staticmethod(apply_dither)
        cls.encode_8bit  = staticmethod(encode_8bit)
        cls.gen_mask     = staticmethod(generate_void_and_cluster)

    def test_none_mode_is_plain_round(self):
        img = np.full((8, 8, 3), 0.5, dtype=np.float32)
        out = self.apply_dither(img, mode="none")
        self.assertEqual(out.dtype, np.uint8)
        np.testing.assert_array_equal(out, 128)

    def test_all_modes_output_range(self):
        img = np.random.rand(32, 32, 3).astype(np.float32)
        for mode in ("none", "tpdf", "blue_noise"):
            with self.subTest(mode=mode):
                out = self.apply_dither(img, mode=mode)
                self.assertGreaterEqual(int(out.min()), 0)
                self.assertLessEqual(int(out.max()), 255)

    def test_all_modes_output_uint8(self):
        img = np.full((8, 8, 3), 0.3, dtype=np.float32)
        for mode in ("none", "tpdf", "blue_noise"):
            with self.subTest(mode=mode):
                out = self.apply_dither(img, mode=mode)
                self.assertEqual(out.dtype, np.uint8)

    def test_mean_preservation(self):
        """All modes should preserve spatial mean within 1.5 LSB."""
        rng = np.random.default_rng(42)
        img = rng.random((64, 64, 3)).astype(np.float32)
        for mode in ("none", "tpdf", "blue_noise"):
            with self.subTest(mode=mode):
                out = self.apply_dither(img, mode=mode)
                expected = img.mean() * 255.0
                actual   = float(out.astype(np.float32).mean())
                self.assertLess(abs(actual - expected), 1.5,
                                msg=f"mode={mode}: mean error too large")

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            self.apply_dither(np.zeros((4, 4, 3), dtype=np.float32),
                              mode="ordered_bayer")

    def test_encode_8bit_uint8_none(self):
        img = np.full((8, 8, 3), 200, dtype=np.uint8)
        out = self.encode_8bit(img, dither_mode="none")
        self.assertEqual(out.dtype, np.uint8)
        np.testing.assert_array_equal(out, img)

    def test_encode_8bit_float(self):
        img = np.full((8, 8, 3), 0.75, dtype=np.float32)
        out = self.encode_8bit(img, dither_mode="none")
        self.assertEqual(out.dtype, np.uint8)
        # 0.75 * 255 = 191.25 → round = 191
        np.testing.assert_array_equal(out, 191)

    def test_encode_8bit_uint16(self):
        # 32767 / 65535 * 255 ≈ 127.5 → 128
        img = np.full((4, 4, 3), 32767, dtype=np.uint16)
        out = self.encode_8bit(img, dither_mode="none")
        self.assertEqual(out.dtype, np.uint8)
        self.assertLessEqual(abs(int(out[0, 0, 0]) - 128), 1)

    def test_blue_noise_mask_statistics(self):
        """Blue noise mask should have near-uniform [0,1) distribution."""
        mask = self.gen_mask(size=32, rng_seed=13)
        self.assertEqual(mask.shape, (32, 32))
        self.assertGreaterEqual(float(mask.min()), 0.0)
        self.assertLessEqual(float(mask.max()), 1.0)
        # Mean of uniform [0,1) is 0.5; allow ±0.05 tolerance
        self.assertLess(abs(float(mask.mean()) - 0.5), 0.05)

    def test_blue_noise_mask_tileability(self):
        """The mask should tile without obvious seams (spatial wrap is smooth)."""
        from util.dither import _tile_mask
        mask = self.gen_mask(size=16, rng_seed=5)
        tiled = _tile_mask(mask, H=48, W=64)
        self.assertEqual(tiled.shape, (48, 64))


# ---------------------------------------------------------------------------
# Phase 3.5 — save_pipeline_output format / dither
# ---------------------------------------------------------------------------

class TestSavePipelineOutput(unittest.TestCase):
    """Integration tests for save_pipeline_output with format and dither options."""

    def setUp(self):
        import util.utils as utils_mod
        import matplotlib.pyplot as plt
        self._saved          = {}
        self._orig_imsave    = plt.imsave
        self._orig_outdir    = utils_mod.OUTPUT_DIR

        def _fake_imsave(path, img, **kwargs):
            self._saved[str(path)] = {"img": img.copy(), "kwargs": kwargs}

        plt.imsave         = _fake_imsave
        utils_mod.OUTPUT_DIR = "/tmp/"

    def tearDown(self):
        import util.utils as utils_mod
        import matplotlib.pyplot as plt
        plt.imsave           = self._orig_imsave
        utils_mod.OUTPUT_DIR = self._orig_outdir

    def _run(self, output_cfg, img=None):
        from util.utils import save_pipeline_output
        if img is None:
            img = np.full((16, 16, 3), 128, dtype=np.uint8)
        config = {"output": output_cfg} if output_cfg is not None else {}
        save_pipeline_output("test", img, config)

    def test_default_saves_png(self):
        self._run(None)
        png_paths = [p for p in self._saved if p.endswith(".png")]
        self.assertEqual(len(png_paths), 1)

    def test_explicit_png_format(self):
        self._run({"format": "png", "dither": "none"})
        png_paths = [p for p in self._saved if p.endswith(".png")]
        self.assertEqual(len(png_paths), 1)

    def test_jpeg_format_saves_jpg(self):
        self._run({"format": "jpeg", "dither": "none", "jpeg_quality": 90})
        jpg_paths = [p for p in self._saved if p.endswith(".jpg")]
        self.assertEqual(len(jpg_paths), 1)

    def test_dither_none_preserves_uint8_values(self):
        img = np.full((16, 16, 3), 200, dtype=np.uint8)
        self._run({"dither": "none", "format": "png"}, img=img)
        saved_img = list(self._saved.values())[0]["img"]
        np.testing.assert_array_equal(saved_img, img)

    def test_dither_blue_noise_output_is_uint8(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        self._run({"dither": "blue_noise", "format": "png"}, img=img)
        saved_img = list(self._saved.values())[0]["img"]
        self.assertEqual(saved_img.dtype, np.uint8)

    def test_dither_tpdf_output_is_uint8(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        self._run({"dither": "tpdf", "format": "png"}, img=img)
        saved_img = list(self._saved.values())[0]["img"]
        self.assertEqual(saved_img.dtype, np.uint8)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
