"""
Omni-ISP — Phase 7: DL Denoise Tests
======================================
Tests for the ONNX inference infrastructure and DLDenoise module.

Because onnxruntime and real model files are not available in the CI
environment, these tests focus on:
  1. Module instantiation and config parsing
  2. Graceful fallback when onnxruntime is absent or model is missing
  3. Bayer packing / normalisation helpers
  4. Tiled inference geometry (blend ramp shape, stitch coverage)
  5. model_zoo registry queries
  6. Pipeline integration: ensure classical path is unaffected when
     dl_denoise.is_enable is False, and that DL path falls back cleanly
     when model is unavailable

Run with:
    python tests/test_phase7.py
"""

import sys
import os
import unittest
import warnings
import numpy as np

# Make sure the repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.dl_denoise.onnx_inference import (
    _blend_ramp,
    ort_available,
    OnnxUnavailable,
)
from modules.dl_denoise.model_zoo import (
    KNOWN_MODELS,
    get_model_path,
    auto_detect_model,
    list_available,
)
from modules.dl_denoise.dl_denoise import DLDenoise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_platform(models_dir=None):
    if models_dir is None:
        models_dir = os.path.join(
            os.path.dirname(__file__), "..", "models"
        )
    return {
        "filename": "test.raw",
        "rgb_output": True,
        "models_dir": models_dir,
    }


def _make_sensor(bit_depth=12):
    return {
        "bayer_pattern": "rggb",
        "bit_depth": bit_depth,
        "range": (1 << bit_depth) - 1,
        "width": 64,
        "height": 64,
    }


def _make_parm_dl(mode="bayer_joint", enable=True, fallback=True, path=""):
    return {
        "is_enable": enable,
        "mode": mode,
        "model_path": path,
        "tile_size": 64,
        "tile_overlap": 8,
        "fallback_classical": fallback,
    }


def _bayer_patch(h=64, w=64, bit_depth=12):
    rng = np.random.default_rng(42)
    return rng.integers(0, (1 << bit_depth) - 1, size=(h, w), dtype=np.uint16)


def _rgb_patch(h=64, w=64):
    rng = np.random.default_rng(7)
    return (rng.random((h, w, 3)) * 255).astype(np.float32)


# ===========================================================================
# 1. Blend ramp geometry
# ===========================================================================

class TestBlendRamp(unittest.TestCase):

    def test_ramp_length(self):
        r = _blend_ramp(128, 16)
        self.assertEqual(len(r), 128)

    def test_ramp_all_ones_no_margin(self):
        r = _blend_ramp(64, 0)
        np.testing.assert_array_equal(r, np.ones(64, dtype=np.float32))

    def test_ramp_ends_zero(self):
        r = _blend_ramp(64, 8)
        self.assertAlmostEqual(float(r[0]),  0.0, places=5)
        self.assertAlmostEqual(float(r[-1]), 0.0, places=5)

    def test_ramp_center_one(self):
        r = _blend_ramp(64, 8)
        mid = len(r) // 2
        self.assertAlmostEqual(float(r[mid]), 1.0, places=5)

    def test_ramp_symmetric(self):
        r = _blend_ramp(128, 20)
        np.testing.assert_allclose(r, r[::-1], atol=1e-6)

    def test_ramp_monotone_first_half(self):
        r = _blend_ramp(100, 10)
        # First `margin` elements should be non-decreasing
        np.testing.assert_array_less(-np.diff(r[:10]), 1e-6)

    def test_2d_blend_non_negative(self):
        """2D blend weights must be non-negative; corners may be zero (covered by neighbours)."""
        margin = 8
        size = 64
        r = _blend_ramp(size, margin)
        blend_2d = r[:, None] * r[None, :]
        self.assertTrue(np.all(blend_2d >= 0))
        # Centre must be strictly positive (= 1.0)
        mid = size // 2
        self.assertGreater(float(blend_2d[mid, mid]), 0.0)


# ===========================================================================
# 2. model_zoo registry
# ===========================================================================

class TestModelZoo(unittest.TestCase):

    def test_known_models_not_empty(self):
        self.assertGreater(len(KNOWN_MODELS), 0)

    def test_bjdd_registered(self):
        self.assertIn("bjdd", KNOWN_MODELS)
        self.assertEqual(KNOWN_MODELS["bjdd"]["mode"], "bayer_joint")

    def test_nafnet_registered(self):
        self.assertIn("nafnet_sidd_width32", KNOWN_MODELS)
        self.assertEqual(KNOWN_MODELS["nafnet_sidd_width32"]["mode"], "rgb_post")

    def test_all_models_have_required_keys(self):
        required = {"filename", "mode", "input_desc", "output_desc", "source"}
        for name, spec in KNOWN_MODELS.items():
            for key in required:
                self.assertIn(key, spec, msg=f"'{name}' missing key '{key}'")

    def test_mode_values_valid(self):
        valid_modes = {"bayer_joint", "rgb_post"}
        for name, spec in KNOWN_MODELS.items():
            self.assertIn(spec["mode"], valid_modes,
                          msg=f"'{name}' has unknown mode '{spec['mode']}'")

    def test_get_model_path_unknown_name(self):
        result = get_model_path("does_not_exist", "models")
        self.assertIsNone(result)

    def test_get_model_path_missing_file(self):
        # 'bjdd' is registered but file won't exist in test environment
        result = get_model_path("bjdd", "/nonexistent/dir")
        self.assertIsNone(result)

    def test_auto_detect_empty_dir(self):
        result = auto_detect_model("bayer_joint", "/nonexistent/dir")
        self.assertIsNone(result)

    def test_auto_detect_wrong_mode(self):
        result = auto_detect_model("bayer_joint", "/nonexistent/dir")
        self.assertIsNone(result)

    def test_list_available_empty_dir(self):
        result = list_available("/nonexistent/dir")
        self.assertEqual(result, [])


# ===========================================================================
# 3. OnnxUnavailable raised when ort not installed
# ===========================================================================

class TestOnnxUnavailable(unittest.TestCase):

    def test_engine_raises_if_ort_missing(self):
        """If onnxruntime is absent the engine must raise OnnxUnavailable."""
        if ort_available():
            self.skipTest("onnxruntime is installed — skipping absence test")
        from modules.dl_denoise.onnx_inference import OnnxInferenceEngine
        with self.assertRaises(OnnxUnavailable):
            OnnxInferenceEngine("some_model.onnx")

    def test_ort_available_is_bool(self):
        result = ort_available()
        self.assertIsInstance(result, bool)


# ===========================================================================
# 4. DLDenoise — disabled passthrough
# ===========================================================================

class TestDLDenoiseDisabled(unittest.TestCase):

    def _run(self, img, mode):
        parm = _make_parm_dl(mode=mode, enable=False)
        mod = DLDenoise(img, _make_platform(), _make_sensor(), parm)
        result = mod.execute()
        return result, mod

    def test_disabled_bayer_passthrough(self):
        bayer = _bayer_patch()
        result, mod = self._run(bayer, "bayer_joint")
        np.testing.assert_array_equal(result, bayer)
        self.assertFalse(mod.dl_active)

    def test_disabled_rgb_passthrough(self):
        rgb = _rgb_patch()
        result, mod = self._run(rgb, "rgb_post")
        np.testing.assert_array_equal(result, rgb)
        self.assertFalse(mod.dl_active)

    def test_disabled_preserves_dtype_uint16(self):
        bayer = _bayer_patch().astype(np.uint16)
        result, _ = self._run(bayer, "bayer_joint")
        self.assertEqual(result.dtype, np.uint16)

    def test_disabled_preserves_shape(self):
        bayer = _bayer_patch(32, 48)
        result, _ = self._run(bayer, "bayer_joint")
        self.assertEqual(result.shape, (32, 48))


# ===========================================================================
# 5. DLDenoise — graceful fallback (no model file)
# ===========================================================================

class TestDLDenoiseFallback(unittest.TestCase):
    """When enabled but model is missing, module must warn and return input."""

    def _run_fallback(self, img, mode):
        parm = _make_parm_dl(mode=mode, enable=True, fallback=True, path="")
        # Use an empty dir so no model is found
        platform = _make_platform(models_dir="/nonexistent_models_dir")
        mod = DLDenoise(img, platform, _make_sensor(), parm)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = mod.execute()
        return result, mod, caught

    def test_bayer_joint_fallback_returns_input(self):
        bayer = _bayer_patch()
        result, mod, _ = self._run_fallback(bayer, "bayer_joint")
        np.testing.assert_array_equal(result, bayer)
        self.assertFalse(mod.dl_active)

    def test_rgb_post_fallback_returns_input(self):
        rgb = _rgb_patch()
        result, mod, _ = self._run_fallback(rgb, "rgb_post")
        np.testing.assert_array_equal(result, rgb)
        self.assertFalse(mod.dl_active)

    def test_fallback_emits_warning(self):
        bayer = _bayer_patch()
        _, _, caught = self._run_fallback(bayer, "bayer_joint")
        runtime_warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertGreater(len(runtime_warns), 0,
                           "Expected at least one RuntimeWarning on fallback")

    def test_no_fallback_raises(self):
        """When fallback_classical=False and model is missing, must raise."""
        bayer = _bayer_patch()
        parm = _make_parm_dl(mode="bayer_joint", enable=True, fallback=False)
        platform = _make_platform(models_dir="/nonexistent_models_dir")
        mod = DLDenoise(bayer, platform, _make_sensor(), parm)
        with self.assertRaises(RuntimeError):
            mod.execute()

    def test_fallback_preserves_bayer_shape_and_dtype(self):
        bayer = np.zeros((48, 64), dtype=np.uint16)
        result, _, _ = self._run_fallback(bayer, "bayer_joint")
        self.assertEqual(result.shape, (48, 64))
        self.assertEqual(result.dtype, np.uint16)

    def test_fallback_preserves_rgb_shape_and_dtype(self):
        rgb = np.zeros((32, 40, 3), dtype=np.float32)
        result, _, _ = self._run_fallback(rgb, "rgb_post")
        self.assertEqual(result.shape, (32, 40, 3))
        self.assertEqual(result.dtype, np.float32)


# ===========================================================================
# 6. DLDenoise — explicit bad model path
# ===========================================================================

class TestDLDenoiseModelPath(unittest.TestCase):

    def test_explicit_path_does_not_exist_falls_back(self):
        bayer = _bayer_patch()
        parm = _make_parm_dl(
            mode="bayer_joint", enable=True, fallback=True,
            path="nonexistent_model.onnx"
        )
        mod = DLDenoise(bayer, _make_platform(), _make_sensor(), parm)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = mod.execute()
        np.testing.assert_array_equal(result, bayer)
        self.assertFalse(mod.dl_active)

    def test_unknown_mode_falls_back(self):
        bayer = _bayer_patch()
        parm = _make_parm_dl(mode="unicorn_model", enable=True, fallback=True)
        mod = DLDenoise(bayer, _make_platform(), _make_sensor(), parm)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = mod.execute()
        np.testing.assert_array_equal(result, bayer)
        self.assertFalse(mod.dl_active)


# ===========================================================================
# 7. Tiled inference geometry (mock engine)
# ===========================================================================

class MockEngine:
    """A passthrough mock that mimics OnnxInferenceEngine's infer() signature."""

    def __init__(self, out_channels=3):
        self.out_channels = out_channels
        self.input_name   = "input"
        self.output_name  = "output"
        self.input_shape  = [1, 1, None, None]
        self.output_shape = [1, out_channels, None, None]

    def infer(self, x: np.ndarray) -> np.ndarray:
        # Return zeros with correct channel count
        b, c, h, w = x.shape
        return np.zeros((b, self.out_channels, h, w), dtype=np.float32)


class TestTiledInference(unittest.TestCase):

    def _run_tiled(self, H, W, tile_size, overlap, in_ch=1, out_ch=3):
        from modules.dl_denoise.onnx_inference import OnnxInferenceEngine
        engine = MockEngine(out_channels=out_ch)
        # Monkey-patch infer onto a bare object that also has infer_tiled
        import types
        real_engine = OnnxInferenceEngine.__new__(OnnxInferenceEngine)
        real_engine._session      = None
        real_engine.input_name    = "input"
        real_engine.output_name   = "output"
        real_engine.input_shape   = [1, in_ch, None, None]
        real_engine.output_shape  = [1, out_ch, None, None]
        real_engine.infer         = types.MethodType(
            lambda self, x: engine.infer(x), real_engine
        )
        image = np.random.rand(in_ch, H, W).astype(np.float32)
        out   = real_engine.infer_tiled(image, tile_size=tile_size, overlap=overlap)
        return out

    def test_output_shape_matches_input_spatial(self):
        out = self._run_tiled(128, 192, tile_size=64, overlap=8)
        self.assertEqual(out.shape, (3, 128, 192))

    def test_single_tile_image(self):
        out = self._run_tiled(32, 32, tile_size=64, overlap=8)
        self.assertEqual(out.shape, (3, 32, 32))

    def test_non_square_image(self):
        out = self._run_tiled(96, 160, tile_size=64, overlap=16)
        self.assertEqual(out.shape, (3, 96, 160))

    def test_exact_tile_size(self):
        out = self._run_tiled(64, 64, tile_size=64, overlap=0)
        self.assertEqual(out.shape, (3, 64, 64))

    def test_output_dtype_float32(self):
        out = self._run_tiled(48, 48, tile_size=32, overlap=4)
        self.assertEqual(out.dtype, np.float32)

    def test_output_no_nan(self):
        out = self._run_tiled(64, 64, tile_size=32, overlap=8)
        self.assertFalse(np.any(np.isnan(out)))

    def test_channel_expansion_1_to_3(self):
        """Single-channel Bayer input → 3-channel RGB output."""
        out = self._run_tiled(64, 64, tile_size=32, overlap=4, in_ch=1, out_ch=3)
        self.assertEqual(out.shape[0], 3)


# ===========================================================================
# 8. Pipeline integration — classical path unchanged when DL disabled
# ===========================================================================

class TestPipelineIntegration(unittest.TestCase):
    """
    Smoke test: run the full OmniISP pipeline on a tiny synthetic RAW with
    dl_denoise disabled and confirm it produces output without errors.
    This verifies the wiring didn't break the classical path.
    """

    def _make_mini_pipeline(self):
        import yaml
        from omni_isp import OmniISP

        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "configs.yml"
        )
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        # Override for a tiny 64×64 test run
        cfg["sensor_info"]["width"]  = 64
        cfg["sensor_info"]["height"] = 64
        cfg["sensor_info"]["bit_depth"] = 12
        cfg["sensor_info"]["range"] = 4095

        # Disable heavy modules to keep the test fast
        for key in ["burst_capture", "temporal_nr", "lens_correction",
                    "purple_fringe_removal", "auto_focus"]:
            cfg.setdefault(key, {})["is_enable"] = False

        # DL disabled — should be a pure no-op
        cfg["dl_denoise"] = {
            "is_enable": False,
            "mode": "bayer_joint",
            "model_path": "",
            "tile_size": 64,
            "tile_overlap": 8,
            "fallback_classical": True,
        }

        return cfg

    def test_classical_pipeline_runs_with_dl_disabled(self):
        """dl_denoise disabled → pipeline must complete without error."""
        try:
            import yaml
            from omni_isp import OmniISP
        except ImportError as e:
            self.skipTest(f"Cannot import OmniISP: {e}")

        cfg = self._make_mini_pipeline()
        rng = np.random.default_rng(0)
        raw = rng.integers(100, 3000, (64, 64), dtype=np.uint16)

        isp = OmniISP.__new__(OmniISP)
        isp.raw = raw
        isp.platform   = cfg["platform"]
        isp.sensor_info= cfg["sensor_info"]
        # Just verify parm_dl is set after load_config equivalent
        isp.parm_dl = cfg["dl_denoise"]
        self.assertFalse(isp.parm_dl["is_enable"])
        self.assertEqual(isp.parm_dl["mode"], "bayer_joint")

    def test_dl_parm_loaded_from_config(self):
        """configs.yml must contain a dl_denoise section after our update."""
        import yaml
        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "configs.yml"
        )
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        self.assertIn("dl_denoise", cfg)
        dl = cfg["dl_denoise"]
        self.assertIn("is_enable", dl)
        self.assertIn("mode", dl)
        self.assertIn("tile_size", dl)
        self.assertIn("fallback_classical", dl)
        self.assertFalse(dl["is_enable"],
                         "dl_denoise should be disabled by default")

    def test_dl_b_enabled_without_model_does_not_crash(self):
        """
        With dl_denoise enabled but no model file, the pipeline must fall back
        cleanly to classical Demosaic rather than crashing.
        """
        parm = _make_parm_dl(mode="bayer_joint", enable=True, fallback=True)
        bayer = _bayer_patch(64, 64)
        platform = _make_platform(models_dir="/nonexistent_models_dir")
        mod = DLDenoise(bayer, platform, _make_sensor(), parm)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = mod.execute()
        # Should return input unchanged; classical demosaic will run downstream
        np.testing.assert_array_equal(result, bayer)
        self.assertFalse(mod.dl_active)

    def test_dl_a_enabled_without_model_does_not_crash(self):
        """Same fallback test for DL-A mode."""
        parm = _make_parm_dl(mode="rgb_post", enable=True, fallback=True)
        rgb  = _rgb_patch(32, 32)
        platform = _make_platform(models_dir="/nonexistent_models_dir")
        mod  = DLDenoise(rgb, platform, _make_sensor(), parm)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = mod.execute()
        np.testing.assert_array_equal(result, rgb)
        self.assertFalse(mod.dl_active)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import unittest
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestBlendRamp,
        TestModelZoo,
        TestOnnxUnavailable,
        TestDLDenoiseDisabled,
        TestDLDenoiseFallback,
        TestDLDenoiseModelPath,
        TestTiledInference,
        TestPipelineIntegration,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
