"""
tests/test_phase8.py
====================
Phase 8 — HDR Pipeline test suite.

Covers:
  - Tone mapping operators (reinhard, reinhard_ext, aces, hable, highlight_rolloff)
  - Absolute luminance scaling (ISO/shutter/aperture → nits)
  - HDRToneMapping module (dispatch, unknown mode, passthrough, absolute lum)
  - HDRMerge module (Mertens fusion, Debevec, single frame, disabled)
  - Pipeline integration (config keys present, API surface)

Run from project root:
    python tests/test_phase8.py
    python -m pytest tests/test_phase8.py -v   # if pytest available
"""

import sys
import os
import warnings
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.hdr_tone_mapping.operators import (
    reinhard,
    reinhard_ext,
    aces,
    hable,
    highlight_rolloff,
    scene_to_absolute_nits,
    _luminance,
)
from modules.hdr_tone_mapping.hdr_tone_mapping import HDRToneMapping
from modules.hdr_merge.hdr_merge import HDRMerge


# ---------------------------------------------------------------------------
# Shared test data helpers
# ---------------------------------------------------------------------------

def _flat(value=0.18, shape=(16, 16, 3)):
    return np.full(shape, value, dtype=np.float32)


def _gradient(lo=0.0, hi=2.0, shape=(16, 32, 3)):
    H, W = shape[:2]
    x = np.linspace(lo, hi, W, dtype=np.float32)
    img = np.zeros(shape, dtype=np.float32)
    for c in range(3):
        img[:, :, c] = x[None, :] * (1.0 - 0.2 * c)
    return img


def _bracket_frames(H=16, W=16):
    """Three-exposure bracket [-2, 0, +2 EV], each (H, W, 3) float32."""
    base = np.tile(np.linspace(0.1, 0.9, W, dtype=np.float32)[None, :, None], (H, 1, 3))
    evs = [-2.0, 0.0, 2.0]
    return [np.clip(base * (2.0 ** ev), 0.0, 1.0) for ev in evs], evs


def _tm_parm(**overrides):
    d = {
        "is_enable": True,
        "mode": "reinhard_ext",
        "key": 0.18,
        "white_percentile": 99.0,
        "exposure_bias": 2.0,
        "peak_nits": 1000.0,
        "absolute_luminance": False,
        "iso": 0,
        "shutter_ms": 0,
        "aperture": 0.0,
        "calibration_k": 12.5,
    }
    d.update(overrides)
    return d


# ===========================================================================
# 1. Reinhard operator
# ===========================================================================

class TestReinhardOperator(unittest.TestCase):

    def test_output_range(self):
        out = reinhard(_gradient())
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-5)

    def test_output_dtype(self):
        self.assertEqual(reinhard(_flat()).dtype, np.float32)

    def test_monotone_luminance(self):
        """Higher input luminance should map to higher (or equal) output luminance."""
        img = _gradient(lo=0.01, hi=2.0, shape=(4, 32, 3))
        out = reinhard(img)
        col_L = _luminance(out).mean(axis=0)
        diffs = np.diff(col_L)
        self.assertTrue(np.all(diffs >= -1e-5),
                        f"Reinhard not monotone — min diff={diffs.min():.6f}")

    def test_flat_grey_in_reasonable_range(self):
        """Flat 0.18 image → L_s = 0.18, L_d = 0.18/1.18 ≈ 0.152."""
        out = reinhard(_flat(0.18))
        mean = float(out.mean())
        self.assertGreater(mean, 0.09)
        self.assertLess(mean, 0.22)

    def test_shape_preserved(self):
        img = _gradient()
        self.assertEqual(reinhard(img).shape, img.shape)


# ===========================================================================
# 2. Extended Reinhard operator
# ===========================================================================

class TestReinhardExtOperator(unittest.TestCase):

    def test_output_range(self):
        out = reinhard_ext(_gradient())
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-5)

    def test_brighter_than_plain_reinhard(self):
        """Extended Reinhard >= plain Reinhard: (1 + L/Lw²) ≥ 1 factor."""
        img = _gradient()
        self.assertGreaterEqual(float(reinhard_ext(img).mean()),
                                float(reinhard(img).mean()) - 1e-4)

    def test_output_dtype(self):
        self.assertEqual(reinhard_ext(_flat()).dtype, np.float32)


# ===========================================================================
# 3. ACES operator
# ===========================================================================

class TestACESOperator(unittest.TestCase):

    def test_output_range(self):
        out = aces(_gradient())
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-5)

    def test_zero_input(self):
        out = aces(np.zeros((4, 4, 3), dtype=np.float32))
        self.assertTrue(np.allclose(out, 0.0, atol=1e-5))

    def test_monotone_per_channel(self):
        x = np.linspace(0.0, 3.0, 100, dtype=np.float32)
        rgb = np.stack([x, x, x], axis=-1)[None]   # (1, 100, 3)
        out = aces(rgb)
        diffs = np.diff(out[0, :, 0])
        self.assertTrue(np.all(diffs >= -1e-6))

    def test_shape_preserved(self):
        img = _flat()
        self.assertEqual(aces(img).shape, img.shape)


# ===========================================================================
# 4. Hable operator
# ===========================================================================

class TestHableOperator(unittest.TestCase):

    def test_output_range(self):
        out = hable(_gradient())
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-5)

    def test_different_bias_gives_different_result(self):
        img = _flat()
        self.assertFalse(np.allclose(hable(img, 1.0), hable(img, 3.0)))

    def test_output_dtype(self):
        self.assertEqual(hable(_flat()).dtype, np.float32)


# ===========================================================================
# 5. Highlight rolloff
# ===========================================================================

class TestHighlightRolloff(unittest.TestCase):

    def test_below_knee_unchanged(self):
        img = _flat(0.5)
        out = highlight_rolloff(img, knee_start=0.8)
        self.assertTrue(np.allclose(out, 0.5, atol=1e-5))

    def test_above_knee_clamped(self):
        img = _flat(5.0)
        out = highlight_rolloff(img, knee_start=0.8, knee_end=1.1)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-5)

    def test_output_range(self):
        out = highlight_rolloff(_gradient())
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-5)


# ===========================================================================
# 6. Absolute luminance scaling
# ===========================================================================

class TestAbsoluteLuminance(unittest.TestCase):

    def test_known_value(self):
        """ISO 100, 10 ms, f/2.8, k=12.5 → scale = 12.5*7.84/(100*0.01) = 98."""
        rgb = _flat(0.18)
        out = scene_to_absolute_nits(rgb, iso=100, shutter_sec=0.01, aperture_f=2.8)
        expected = 0.18 * (12.5 * 2.8 ** 2) / (100 * 0.01)
        self.assertTrue(np.allclose(out, expected, rtol=1e-5))

    def test_proportional_to_pixel(self):
        o1 = scene_to_absolute_nits(_flat(0.1), iso=200, shutter_sec=0.02, aperture_f=4.0)
        o3 = scene_to_absolute_nits(_flat(0.3), iso=200, shutter_sec=0.02, aperture_f=4.0)
        self.assertTrue(np.allclose(o3 / o1, 3.0, rtol=1e-5))

    def test_zero_iso_raises(self):
        with self.assertRaises(ValueError):
            scene_to_absolute_nits(_flat(), iso=0, shutter_sec=0.01, aperture_f=2.8)

    def test_zero_shutter_raises(self):
        with self.assertRaises(ValueError):
            scene_to_absolute_nits(_flat(), iso=100, shutter_sec=0.0, aperture_f=2.8)


# ===========================================================================
# 7. HDRToneMapping module — disabled / passthrough
# ===========================================================================

class TestHDRToneMappingDisabled(unittest.TestCase):

    def test_passthrough_when_disabled(self):
        img = _flat()
        parm = {"is_enable": False}
        out = HDRToneMapping(img, {}, {}, parm).execute()
        self.assertTrue(np.allclose(out, img))

    def test_passthrough_preserves_shape(self):
        img = _gradient()
        parm = {"is_enable": False}
        out = HDRToneMapping(img, {}, {}, parm).execute()
        self.assertEqual(out.shape, img.shape)


# ===========================================================================
# 8. HDRToneMapping module — all modes
# ===========================================================================

class TestHDRToneMappingModes(unittest.TestCase):

    def _run(self, mode):
        img = _flat()
        parm = _tm_parm(mode=mode)
        out = HDRToneMapping(img, {}, {}, parm).execute()
        return out

    def test_reinhard(self):
        out = self._run("reinhard")
        self.assertEqual(out.dtype, np.float32)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-5)

    def test_reinhard_ext(self):
        out = self._run("reinhard_ext")
        self.assertEqual(out.dtype, np.float32)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-5)

    def test_aces(self):
        out = self._run("aces")
        self.assertEqual(out.dtype, np.float32)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-5)

    def test_hable(self):
        out = self._run("hable")
        self.assertEqual(out.dtype, np.float32)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-5)

    def test_hlg_rolloff(self):
        out = self._run("hlg_rolloff")
        self.assertEqual(out.dtype, np.float32)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-5)

    def test_unknown_mode_warns_and_clips(self):
        img = _flat()
        parm = _tm_parm(mode="nonexistent_operator")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = HDRToneMapping(img, {}, {}, parm).execute()
        self.assertTrue(any("unknown mode" in str(x.message).lower() for x in w))
        self.assertLessEqual(float(out.max()), 1.0 + 1e-5)

    def test_output_clips_to_unit_range(self):
        img = _gradient()   # values up to ~2.0
        parm = _tm_parm(mode="reinhard")
        out = HDRToneMapping(img, {}, {}, parm).execute()
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-5)


# ===========================================================================
# 9. HDRToneMapping — absolute luminance
# ===========================================================================

class TestHDRToneMappingAbsoluteLuminance(unittest.TestCase):

    def test_absolute_mode_valid_params(self):
        img = _flat()
        parm = _tm_parm(absolute_luminance=True, iso=400,
                        shutter_ms=10.0, aperture=2.8, peak_nits=1000.0)
        out = HDRToneMapping(img, {}, {}, parm).execute()
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.float32)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-5)

    def test_absolute_mode_warns_if_metadata_missing(self):
        img = _flat()
        parm = _tm_parm(absolute_luminance=True, iso=0, shutter_ms=0, aperture=0.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = HDRToneMapping(img, {}, {}, parm).execute()
        has_warn = any("absolute" in str(x.message).lower() for x in w)
        self.assertTrue(has_warn, "Should warn when absolute lum metadata missing")
        self.assertEqual(out.shape, img.shape)  # graceful fallback

    def test_sensor_info_provides_fallback(self):
        """When parm has iso=0, shutter_ms=0, use sensor_info values."""
        img = _flat()
        parm = _tm_parm(absolute_luminance=True, iso=0, shutter_ms=0, aperture=0.0,
                        peak_nits=1000.0, mode="reinhard")
        sensor_info = {"iso": 200, "shutter_ms": 8.0, "aperture": 4.0}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = HDRToneMapping(img, {}, sensor_info, parm).execute()
        # Should NOT warn about missing metadata
        missing_warn = [x for x in w if "not fully specified" in str(x.message)]
        self.assertEqual(len(missing_warn), 0)
        self.assertEqual(out.shape, img.shape)


# ===========================================================================
# 10. HDRMerge — disabled / single frame
# ===========================================================================

class TestHDRMergeDisabled(unittest.TestCase):

    def test_disabled_returns_first_frame(self):
        frames, evs = _bracket_frames()
        parm = {"is_enable": False, "mode": "mertens"}
        out = HDRMerge(frames, evs, parm).execute()
        self.assertTrue(np.allclose(out, frames[0]))

    def test_single_frame_returned_unchanged(self):
        img = _flat()
        out = HDRMerge([img], None, {"is_enable": True, "mode": "mertens"}).execute()
        self.assertTrue(np.allclose(out, img))


# ===========================================================================
# 11. HDRMerge — Mertens fusion
# ===========================================================================

class TestMertensFusion(unittest.TestCase):

    def test_output_shape(self):
        frames, evs = _bracket_frames()
        out = HDRMerge(frames, evs, {"is_enable": True, "mode": "mertens"}).execute()
        self.assertEqual(out.shape, frames[0].shape)

    def test_output_range(self):
        frames, evs = _bracket_frames()
        out = HDRMerge(frames, evs, {"is_enable": True, "mode": "mertens"}).execute()
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-5)

    def test_output_dtype(self):
        frames, evs = _bracket_frames()
        out = HDRMerge(frames, evs, {"is_enable": True, "mode": "mertens"}).execute()
        self.assertEqual(out.dtype, np.float32)

    def test_merged_differs_from_all_inputs(self):
        frames, evs = _bracket_frames()
        out = HDRMerge(frames, evs, {"is_enable": True, "mode": "mertens"}).execute()
        for i, f in enumerate(frames):
            self.assertFalse(np.allclose(out, f),
                             f"Merged should differ from bracket frame {i}")

    def test_shape_mismatch_raises(self):
        frames = [np.zeros((16, 16, 3), dtype=np.float32),
                  np.zeros((32, 32, 3), dtype=np.float32)]
        with self.assertRaises(ValueError):
            HDRMerge(frames, None, {"is_enable": True, "mode": "mertens"}).execute()

    def test_identical_frames_return_same_value(self):
        """Three identical frames → merged ≈ each frame (equal weights)."""
        frame = _flat(0.4)
        frames = [frame.copy(), frame.copy(), frame.copy()]
        out = HDRMerge(frames, None, {"is_enable": True, "mode": "mertens"}).execute()
        self.assertTrue(np.allclose(out, frame, atol=1e-4))

    def test_uniform_frames_average(self):
        """For zero-contrast frames, equal quality → simple average."""
        frames = [_flat(0.3), _flat(0.5), _flat(0.7)]
        out = HDRMerge(frames, None, {"is_enable": True, "mode": "mertens"}).execute()
        self.assertAlmostEqual(float(out.mean()), 0.5, places=3)

    def test_unknown_mode_warns_returns_first(self):
        frames, evs = _bracket_frames()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = HDRMerge(frames, evs,
                           {"is_enable": True, "mode": "magic_mode"}).execute()
        self.assertTrue(any("unknown mode" in str(x.message).lower() for x in w))
        self.assertTrue(np.allclose(out, frames[0]))


# ===========================================================================
# 12. HDRMerge — Debevec reconstruction
# ===========================================================================

class TestDebevecMerge(unittest.TestCase):

    def _parm(self):
        return {"is_enable": True, "mode": "debevec",
                "response_samples": 16, "lambda_smooth": 20.0}

    def test_output_shape(self):
        frames, evs = _bracket_frames(H=8, W=8)
        out = HDRMerge(frames, evs, self._parm()).execute()
        self.assertEqual(out.shape, frames[0].shape)

    def test_output_range(self):
        frames, evs = _bracket_frames(H=8, W=8)
        out = HDRMerge(frames, evs, self._parm()).execute()
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-5)

    def test_output_dtype(self):
        frames, evs = _bracket_frames(H=8, W=8)
        out = HDRMerge(frames, evs, self._parm()).execute()
        self.assertEqual(out.dtype, np.float32)

    def test_fallback_to_mertens_if_no_evs(self):
        frames, _ = _bracket_frames(H=8, W=8)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = HDRMerge(frames, None, self._parm()).execute()
        self.assertTrue(any("debevec" in str(x.message).lower() for x in w))
        self.assertEqual(out.shape, frames[0].shape)


# ===========================================================================
# 13. Pipeline integration
# ===========================================================================

class TestPipelineIntegration(unittest.TestCase):

    def _config_text(self):
        """Return raw configs.yml text (avoids any yaml module mock)."""
        cfg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "config", "configs.yml"
        )
        with open(cfg_path) as f:
            return f.read()

    def test_hdr_tone_mapping_in_config(self):
        text = self._config_text()
        self.assertIn("hdr_tone_mapping:", text)
        for key in ("is_enable", "mode", "peak_nits", "absolute_luminance"):
            self.assertIn(key + ":", text,
                          f"hdr_tone_mapping.{key} missing from configs.yml")

    def test_hdr_merge_in_config(self):
        text = self._config_text()
        self.assertIn("hdr_merge:", text)
        for key in ("is_enable", "mode", "weight_contrast", "weight_exposedness"):
            self.assertIn(key + ":", text,
                          f"hdr_merge.{key} missing from configs.yml")

    def test_disabled_hdrtm_passthrough(self):
        img = _flat()
        out = HDRToneMapping(img, {}, {}, {"is_enable": False}).execute()
        self.assertTrue(np.allclose(out, img))

    def test_merge_exposures_api(self):
        """OmniISP.merge_exposures delegates to HDRMerge correctly."""
        from omni_isp import OmniISP
        isp = OmniISP.__new__(OmniISP)
        isp.parm_hdrm = {"is_enable": True, "mode": "mertens"}
        frames, evs = _bracket_frames()
        out = isp.merge_exposures(frames, evs)
        self.assertEqual(out.shape, frames[0].shape)
        self.assertEqual(out.dtype, np.float32)
        self.assertGreaterEqual(float(out.mean()), 0.0)
        self.assertLessEqual(float(out.mean()), 1.0)

    def test_merge_exposures_disabled(self):
        from omni_isp import OmniISP
        isp = OmniISP.__new__(OmniISP)
        isp.parm_hdrm = {"is_enable": False, "mode": "mertens"}
        frames, evs = _bracket_frames()
        out = isp.merge_exposures(frames, evs)
        self.assertTrue(np.allclose(out, frames[0]))

    def test_hdrtm_parm_loaded_from_config(self):
        """Verify is_enable defaults to false in configs.yml."""
        text = self._config_text()
        # Find the hdr_tone_mapping section and check is_enable
        idx = text.find("hdr_tone_mapping:")
        self.assertGreater(idx, -1)
        snippet = text[idx: idx + 400]
        self.assertIn("is_enable: false", snippet,
                      "Default hdr_tone_mapping.is_enable should be false")

    def test_hdrm_parm_loaded_from_config(self):
        text = self._config_text()
        idx = text.find("hdr_merge:")
        self.assertGreater(idx, -1)
        snippet = text[idx: idx + 300]
        self.assertIn("is_enable: false", snippet,
                      "Default hdr_merge.is_enable should be false")


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
