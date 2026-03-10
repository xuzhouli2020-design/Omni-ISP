"""
Tests for Phase 6: Unified Lens Correction (CAC + LDC) and Purple Fringe Removal.

Test groups
───────────
TestWarpField          — build_warp_field geometry and composition math
TestLensCorrection     — module-level: passthrough, isolation, reconstruction
TestPurpleFringeRemoval — detection, desaturation, edge cases
TestPipelineIntegration — LensCorrection + PFR in minimal pipeline context
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import unittest

from modules.lens_correction.warp_field import build_warp_field, remap_channel
from modules.lens_correction.lens_correction import LensCorrection
from modules.purple_fringe_removal.purple_fringe_removal import PurpleFringeRemoval


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _sensor_info(bayer='rggb', bit_depth=12, h=64, w=64):
    return dict(bayer_pattern=bayer, bit_depth=bit_depth,
                width=w, height=h, range=4095)

def _platform():
    return dict(rgb_output=True)

def _parm_lens(**kwargs):
    base = dict(is_enable=True, ldc_enable=False,
                k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0,
                focal_length_px=0.0, center='auto',
                cac_enable=False,
                r_ca=[0.0, 0.0, 0.0], b_ca=[0.0, 0.0, 0.0])
    base.update(kwargs)
    return base

def _parm_pfr(**kwargs):
    base = dict(is_enable=True, hue_center=290.0, hue_half_width=30.0,
                sat_threshold=0.35, highlight_threshold=0.85,
                desaturation_radius=3, strength=1.0)
    base.update(kwargs)
    return base

def _rggb_bayer(h=64, w=64, bit_depth=12, seed=42):
    """Synthetic RGGB Bayer with per-channel distinct gradients."""
    rng = np.random.default_rng(seed)
    max_val = 2**bit_depth - 1
    raw = rng.integers(800, 2000, size=(h, w), dtype=np.uint16)
    # Give each channel a distinct DC offset for easier debugging
    raw[0::2, 0::2] = np.clip(raw[0::2, 0::2].astype(int) + 500, 0, max_val)  # R brighter
    raw[1::2, 1::2] = np.clip(raw[1::2, 1::2].astype(int) - 300, 0, max_val)  # B darker
    return raw


# ═══════════════════════════════════════════════════════════════════════════════
# TestWarpField
# ═══════════════════════════════════════════════════════════════════════════════

class TestWarpField(unittest.TestCase):

    def test_identity_all_zeros(self):
        """All-zero coefficients → warp maps each pixel to itself."""
        H, W = 32, 48
        ldc = dict(k1=0, k2=0, k3=0, p1=0, p2=0,
                   focal_length_px=0, center='auto')
        rows_src, cols_src = build_warp_field(H, W, ldc, (0, 0, 0))

        rows_ref = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))
        cols_ref = np.tile(np.arange(W, dtype=np.float32)[None, :], (H, 1))

        np.testing.assert_allclose(rows_src, rows_ref, atol=1e-4)
        np.testing.assert_allclose(cols_src, cols_ref, atol=1e-4)

    def test_ldc_barrel_moves_edges_inward(self):
        """k1 < 0 (barrel) moves corner sources toward centre."""
        H, W = 64, 64
        ldc = dict(k1=-0.2, k2=0, k3=0, p1=0, p2=0,
                   focal_length_px=0, center='auto')
        rows_src, cols_src = build_warp_field(H, W, ldc, (0, 0, 0))
        cx, cy = (W-1)/2.0, (H-1)/2.0

        # Top-left corner pixel (0,0): source should be closer to centre
        r0, c0 = rows_src[0, 0], cols_src[0, 0]
        self.assertGreater(c0, 0.0)   # shifted right (toward cx)
        self.assertGreater(r0, 0.0)   # shifted down  (toward cy)

    def test_ldc_pincushion_moves_edges_outward(self):
        """k1 > 0 (pincushion) moves corner sources away from centre."""
        H, W = 64, 64
        ldc = dict(k1=0.2, k2=0, k3=0, p1=0, p2=0,
                   focal_length_px=0, center='auto')
        rows_src, cols_src = build_warp_field(H, W, ldc, (0, 0, 0))

        r0, c0 = rows_src[0, 0], cols_src[0, 0]
        self.assertLess(c0, 0.0)   # shifted left  (away from cx)
        self.assertLess(r0, 0.0)   # shifted up    (away from cy)

    def test_ldc_centre_pixel_unchanged(self):
        """Centre pixel should be near-identity for any pure radial LDC.
        Use odd dimensions so the true centre falls exactly on an integer pixel."""
        H, W = 65, 65           # odd → centre at exactly (32, 32)
        cx, cy = (W-1)/2.0, (H-1)/2.0   # = 32.0, 32.0
        for k1 in [-0.3, 0.0, 0.3]:
            ldc = dict(k1=k1, k2=0, k3=0, p1=0, p2=0,
                       focal_length_px=0, center='auto')
            rows_src, cols_src = build_warp_field(H, W, ldc, (0, 0, 0))
            cr = int(round(cy)); cc = int(round(cx))
            self.assertAlmostEqual(float(rows_src[cr, cc]), cy, delta=0.05)
            self.assertAlmostEqual(float(cols_src[cr, cc]), cx, delta=0.05)

    def test_cac_zero_on_g_channel(self):
        """G channel with zero CA delta should be identical to pure LDC."""
        H, W = 32, 32
        ldc = dict(k1=-0.1, k2=0, k3=0, p1=0, p2=0,
                   focal_length_px=0, center='auto')
        rows_g, cols_g = build_warp_field(H, W, ldc, (0.0, 0.0, 0.0))
        rows_ldc_only, cols_ldc_only = build_warp_field(H, W, ldc, (0.0, 0.0, 0.0))
        np.testing.assert_array_equal(rows_g, rows_ldc_only)
        np.testing.assert_array_equal(cols_g, cols_ldc_only)

    def test_cac_r_differs_from_g(self):
        """R channel with non-zero CA delta should differ from G (LDC-only)."""
        H, W = 32, 32
        ldc = dict(k1=0, k2=0, k3=0, p1=0, p2=0,
                   focal_length_px=0, center='auto')
        rows_g, cols_g = build_warp_field(H, W, ldc, (0.0, 0.0, 0.0))
        rows_r, cols_r = build_warp_field(H, W, ldc, (0.05, 0.0, 0.0))

        # Centre pixel should be the same (r_ca=0 at centre)
        cy, cx = (H-1)//2, (W-1)//2
        self.assertAlmostEqual(float(rows_r[cy, cx]), float(rows_g[cy, cx]), delta=0.05)
        # Corner pixels should differ
        self.assertFalse(np.allclose(rows_r, rows_g))

    def test_cac_positive_scale_expands_corners(self):
        """Positive ca_k1 for R → corner source position further from centre."""
        H, W = 64, 64
        cx, cy = (W-1)/2.0, (H-1)/2.0
        ldc = dict(k1=0, k2=0, k3=0, p1=0, p2=0,
                   focal_length_px=0, center='auto')
        rows_g, cols_g = build_warp_field(H, W, ldc, (0.0, 0.0, 0.0))
        rows_r, cols_r = build_warp_field(H, W, ldc, (0.1, 0.0, 0.0))

        # Top-left corner: R source should be further from centre than G source
        dist_g = np.sqrt((cols_g[0,0]-cx)**2 + (rows_g[0,0]-cy)**2)
        dist_r = np.sqrt((cols_r[0,0]-cx)**2 + (rows_r[0,0]-cy)**2)
        self.assertGreater(dist_r, dist_g)

    def test_output_shape(self):
        """Warp field output shape matches (H_sub, W_sub)."""
        H, W = 40, 60
        ldc = dict(k1=0.1, k2=0, k3=0, p1=0.01, p2=0,
                   focal_length_px=0, center='auto')
        rows, cols = build_warp_field(H, W, ldc, (0.02, 0.0, 0.0))
        self.assertEqual(rows.shape, (H, W))
        self.assertEqual(cols.shape, (H, W))

    def test_custom_center(self):
        """Custom principal point shifts the symmetry axis of distortion."""
        H, W = 32, 32
        ldc_auto   = dict(k1=0.2, k2=0, k3=0, p1=0, p2=0,
                          focal_length_px=0, center='auto')
        ldc_offset = dict(k1=0.2, k2=0, k3=0, p1=0, p2=0,
                          focal_length_px=0, center=[20.0, 20.0])  # full-res → /2=10
        rows_auto, cols_auto = build_warp_field(H, W, ldc_auto,   (0, 0, 0))
        rows_off,  cols_off  = build_warp_field(H, W, ldc_offset, (0, 0, 0))
        # Different centre → different warp field
        self.assertFalse(np.allclose(rows_auto, rows_off))

    def test_remap_channel_identity_integer(self):
        """Identity warp (source = output position) returns the input unchanged."""
        H, W = 16, 16
        img = np.random.default_rng(0).random((H, W)).astype(np.float32)
        rows_ref = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))
        cols_ref = np.tile(np.arange(W, dtype=np.float32)[None, :], (H, 1))
        out = remap_channel(img, rows_ref, cols_ref)
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_remap_preserves_range(self):
        """Remap should not introduce values outside the input range."""
        H, W = 20, 20
        img = np.random.default_rng(1).random((H, W)).astype(np.float32) * 1000
        ldc = dict(k1=-0.2, k2=0, k3=0, p1=0, p2=0,
                   focal_length_px=0, center='auto')
        rows, cols = build_warp_field(H, W, ldc, (0, 0, 0))
        out = remap_channel(img, rows, cols)
        # With mode='nearest' clamping, out should stay within input min/max
        self.assertGreaterEqual(float(out.min()), float(img.min()) - 1.0)
        self.assertLessEqual(float(out.max()),    float(img.max()) + 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# TestLensCorrection
# ═══════════════════════════════════════════════════════════════════════════════

class TestLensCorrection(unittest.TestCase):

    def _make(self, bayer='rggb', h=64, w=64, bit_depth=12, **kwargs):
        raw = _rggb_bayer(h, w, bit_depth)
        si  = _sensor_info(bayer, bit_depth, h, w)
        p   = _parm_lens(**kwargs)
        return LensCorrection(raw, _platform(), si, p), raw

    def test_disabled_returns_input_unchanged(self):
        lc, raw = self._make(is_enable=False)
        out = lc.execute()
        np.testing.assert_array_equal(out, raw)

    def test_enabled_but_both_sub_disabled_passthrough(self):
        lc, raw = self._make(is_enable=True, ldc_enable=False, cac_enable=False)
        out = lc.execute()
        np.testing.assert_array_equal(out, raw)

    def test_output_shape_preserved(self):
        lc, raw = self._make(is_enable=True, ldc_enable=True, k1=-0.1)
        out = lc.execute()
        self.assertEqual(out.shape, raw.shape)

    def test_output_dtype_uint16(self):
        lc, raw = self._make(is_enable=True, ldc_enable=True, k1=0.1)
        out = lc.execute()
        self.assertEqual(out.dtype, np.uint16)

    def test_output_values_in_range(self):
        lc, raw = self._make(is_enable=True, ldc_enable=True, k1=-0.2)
        out = lc.execute()
        max_val = 2**12 - 1
        self.assertGreaterEqual(int(out.min()), 0)
        self.assertLessEqual(int(out.max()), max_val)

    def test_identity_ldc_unchanged(self):
        """All-zero LDC coefficients → output equals input."""
        lc, raw = self._make(is_enable=True, ldc_enable=True,
                             k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0)
        out = lc.execute()
        np.testing.assert_array_equal(out, raw)

    def test_identity_cac_unchanged(self):
        """All-zero CA deltas → output equals input."""
        lc, raw = self._make(is_enable=True, cac_enable=True,
                             r_ca=[0.0, 0.0, 0.0], b_ca=[0.0, 0.0, 0.0])
        out = lc.execute()
        np.testing.assert_array_equal(out, raw)

    def test_barrel_ldc_changes_output(self):
        """Non-zero LDC should change the image."""
        lc, raw = self._make(is_enable=True, ldc_enable=True, k1=-0.15)
        out = lc.execute()
        self.assertFalse(np.array_equal(out, raw))

    def test_cac_only_changes_r_and_b_subchannels(self):
        """CAC with non-zero r_ca/b_ca should change R and B but leave G unchanged."""
        from util.bayer_utils import extract_channels
        raw = _rggb_bayer(64, 64)
        si  = _sensor_info('rggb', 12, 64, 64)
        p   = _parm_lens(is_enable=True, cac_enable=True,
                         r_ca=[0.05, 0.0, 0.0], b_ca=[-0.05, 0.0, 0.0])
        lc  = LensCorrection(raw, _platform(), si, p)
        out = lc.execute()

        in_ch  = extract_channels(raw, 'rggb')
        out_ch = extract_channels(out, 'rggb')

        # G channels (Gr, Gb) should be identical — CA only touches R and B
        np.testing.assert_array_equal(out_ch['gr'], in_ch['gr'])
        np.testing.assert_array_equal(out_ch['gb'], in_ch['gb'])

        # R and B should have changed
        self.assertFalse(np.array_equal(out_ch['r'], in_ch['r']))
        self.assertFalse(np.array_equal(out_ch['b'], in_ch['b']))

    def test_cac_positive_r_shifts_outward(self):
        """Positive ca_k1 for R channel → R sub-channel source expanded toward edges."""
        from util.bayer_utils import extract_channels
        # Use a flat-gradient image so the shift is visible as a mean offset
        H, W = 64, 64
        raw = np.zeros((H, W), dtype=np.uint16)
        raw[0::2, 0::2] = np.tile(np.arange(W//2, dtype=np.uint16), (H//2, 1))  # R = col gradient
        raw[0::2, 1::2] = 512   # Gr flat
        raw[1::2, 0::2] = 512   # Gb flat
        raw[1::2, 1::2] = 512   # B flat

        si = _sensor_info('rggb', 12, H, W)
        # Positive CA: R source pulled outward → bright pixels near edges sampled
        p_pos = _parm_lens(is_enable=True, cac_enable=True, r_ca=[0.3, 0.0, 0.0])
        # Negative CA: R source pulled inward
        p_neg = _parm_lens(is_enable=True, cac_enable=True, r_ca=[-0.3, 0.0, 0.0])

        out_pos = LensCorrection(raw, _platform(), si, p_pos).execute()
        out_neg = LensCorrection(raw, _platform(), si, p_neg).execute()

        r_pos = extract_channels(out_pos, 'rggb')['r'].astype(float)
        r_neg = extract_channels(out_neg, 'rggb')['r'].astype(float)
        # Positive CA shifts source outward → higher column values for right half
        self.assertGreater(r_pos[:, W//4:].mean(), r_neg[:, W//4:].mean())

    def test_bayer_pattern_bggr(self):
        """Module handles BGGR pattern without error."""
        H, W = 32, 32
        raw = np.random.default_rng(5).integers(500, 3000, (H, W), dtype=np.uint16)
        si  = _sensor_info('bggr', 12, H, W)
        p   = _parm_lens(is_enable=True, cac_enable=True,
                         r_ca=[0.05, 0.0, 0.0], b_ca=[-0.03, 0.0, 0.0])
        lc  = LensCorrection(raw, _platform(), si, p)
        out = lc.execute()
        self.assertEqual(out.shape, (H, W))
        self.assertEqual(out.dtype, np.uint16)

    def test_combined_ldc_cac_differs_from_either_alone(self):
        """Combined LDC+CAC warp should differ from each applied alone."""
        raw = _rggb_bayer(64, 64)
        si  = _sensor_info('rggb', 12, 64, 64)

        ldc_only = LensCorrection(raw, _platform(), si,
            _parm_lens(is_enable=True, ldc_enable=True, k1=-0.1)).execute()
        cac_only = LensCorrection(raw, _platform(), si,
            _parm_lens(is_enable=True, cac_enable=True, r_ca=[0.05,0,0])).execute()
        both     = LensCorrection(raw, _platform(), si,
            _parm_lens(is_enable=True, ldc_enable=True, k1=-0.1,
                       cac_enable=True, r_ca=[0.05,0,0])).execute()

        self.assertFalse(np.array_equal(both, ldc_only))
        self.assertFalse(np.array_equal(both, cac_only))


# ═══════════════════════════════════════════════════════════════════════════════
# TestPurpleFringeRemoval
# ═══════════════════════════════════════════════════════════════════════════════

def _make_pfr_img(H=32, W=32):
    """Uniform grey image, float32 [0,1]."""
    return np.full((H, W, 3), 0.5, dtype=np.float32)


class TestPurpleFringeRemoval(unittest.TestCase):

    def _make(self, img, **kwargs):
        si = _sensor_info()
        p  = _parm_pfr(**kwargs)
        return PurpleFringeRemoval(img, _platform(), si, p)

    def test_disabled_passthrough(self):
        img = _make_pfr_img()
        pfr = PurpleFringeRemoval(img, _platform(), _sensor_info(),
                                  _parm_pfr(is_enable=False))
        out = pfr.execute()
        np.testing.assert_array_equal(out, img)

    def test_no_purple_pixels_unchanged(self):
        """Green/yellow scene with no purple → output identical to input."""
        H, W = 16, 16
        img = np.zeros((H, W, 3), dtype=np.float32)
        img[..., 1] = 0.9   # pure green
        pfr = self._make(img)
        out = pfr.execute()
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_purple_near_highlight_desaturated(self):
        """Purple pixel adjacent to a blown highlight should be desaturated."""
        H, W = 16, 16
        img = np.zeros((H, W, 3), dtype=np.float32)
        # Blown highlight at centre
        cy, cx = H//2, W//2
        img[cy, cx, :] = [1.0, 1.0, 1.0]
        # Purple pixel one row above centre — within default desaturation_radius
        img[cy-1, cx] = [0.6, 0.0, 0.9]   # hue ≈ 280°, S ≈ 1.0, V ≈ 0.9

        pfr  = self._make(img, desaturation_radius=3)
        out  = pfr.execute()

        # After desaturation, the purple pixel should be more grey
        # (saturation reduced → R, G, B closer together)
        r, g, b = out[cy-1, cx, 0], out[cy-1, cx, 1], out[cy-1, cx, 2]
        in_sat = np.max([0.6, 0.0, 0.9]) - np.min([0.6, 0.0, 0.9])
        out_sat = float(max(r,g,b)) - float(min(r,g,b))
        self.assertLess(out_sat, in_sat)

    def test_purple_far_from_highlight_unchanged(self):
        """Purple pixel far from any highlight should NOT be desaturated."""
        H, W = 32, 32
        img = np.zeros((H, W, 3), dtype=np.float32) + 0.1  # dark background
        # Highlight at top-left corner (0,0)
        img[0, 0] = [1.0, 1.0, 1.0]
        # Purple pixel at bottom-right — far from highlight
        img[H-1, W-1] = [0.6, 0.0, 0.9]

        pfr = self._make(img, desaturation_radius=2)  # small radius
        out = pfr.execute()

        np.testing.assert_allclose(out[H-1, W-1], img[H-1, W-1], atol=1e-5)

    def test_non_purple_hue_near_highlight_unchanged(self):
        """Orange pixel near a blown highlight should not be touched."""
        H, W = 16, 16
        img = np.zeros((H, W, 3), dtype=np.float32) + 0.05
        cy, cx = H//2, W//2
        img[cy, cx, :] = [1.0, 1.0, 1.0]   # highlight
        img[cy-1, cx]  = [0.9, 0.4, 0.0]   # orange, hue ≈ 27°

        pfr = self._make(img)
        out = pfr.execute()
        np.testing.assert_allclose(out[cy-1, cx], img[cy-1, cx], atol=1e-5)

    def test_strength_zero_no_change(self):
        """strength=0 → no desaturation even for valid fringe pixels."""
        H, W = 16, 16
        img = np.zeros((H, W, 3), dtype=np.float32) + 0.05
        cy, cx = H//2, W//2
        img[cy, cx, :] = [1.0, 1.0, 1.0]
        img[cy-1, cx]  = [0.6, 0.0, 0.9]

        pfr = self._make(img, strength=0.0)
        out = pfr.execute()
        np.testing.assert_allclose(out[cy-1, cx], img[cy-1, cx], atol=1e-5)

    def test_partial_strength(self):
        """strength=0.5 → partial desaturation (less than full strength=1)."""
        H, W = 16, 16
        img = np.zeros((H, W, 3), dtype=np.float32) + 0.05
        cy, cx = H//2, W//2
        img[cy, cx, :] = [1.0, 1.0, 1.0]
        img[cy-1, cx]  = [0.6, 0.0, 0.9]

        out_full = self._make(img, strength=1.0).execute()
        out_half = self._make(img, strength=0.5).execute()

        # Half-strength saturation should be between original and full
        sat_orig = float(max(0.6,0,0.9) - min(0.6,0,0.9))
        r,g,b    = out_half[cy-1,cx]
        sat_half = float(max(r,g,b) - min(r,g,b))
        r,g,b    = out_full[cy-1,cx]
        sat_full = float(max(r,g,b) - min(r,g,b))

        self.assertGreater(sat_half, sat_full - 1e-5)
        self.assertLess(sat_half,    sat_orig + 1e-5)

    def test_output_shape_preserved(self):
        img = _make_pfr_img(24, 36)
        out = self._make(img).execute()
        self.assertEqual(out.shape, img.shape)

    def test_uint8_input_output(self):
        """uint8 input should produce uint8 output in [0, 255]."""
        img = (np.random.default_rng(9).random((16, 16, 3)) * 255).astype(np.uint8)
        pfr = PurpleFringeRemoval(img, _platform(), _sensor_info(),
                                  _parm_pfr(is_enable=True))
        out = pfr.execute()
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreaterEqual(int(out.min()), 0)
        self.assertLessEqual(int(out.max()), 255)

    def test_float32_input_output_range(self):
        """float32 input should produce float32 output in [0, 1]."""
        img = np.random.default_rng(10).random((16, 16, 3)).astype(np.float32)
        pfr = PurpleFringeRemoval(img, _platform(), _sensor_info(),
                                  _parm_pfr(is_enable=True))
        out = pfr.execute()
        self.assertEqual(out.dtype, np.float32)
        self.assertGreaterEqual(float(out.min()), 0.0 - 1e-6)
        self.assertLessEqual(float(out.max()),    1.0 + 1e-6)

    def test_hue_wrap_around_detection(self):
        """Hue band that wraps around 0°/360° should be correctly detected."""
        H, W = 16, 16
        img = np.zeros((H, W, 3), dtype=np.float32) + 0.05
        cy, cx = H//2, W//2
        img[cy, cx, :] = [1.0, 1.0, 1.0]
        # A pixel with hue ≈ 5° (near-red with slight blue) — wraps if band is centered at 350°
        img[cy-1, cx] = [0.9, 0.0, 0.1]   # hue ≈ 7°

        # Band centred at 350° ± 20° → covers 330–370° wrapping to 0–10°
        pfr = self._make(img, hue_center=350.0, hue_half_width=20.0, strength=1.0)
        out = pfr.execute()

        # This pixel should be desaturated
        r, g, b   = out[cy-1, cx]
        in_sat    = max(0.9, 0.0, 0.1) - min(0.9, 0.0, 0.1)
        out_sat   = float(max(r,g,b) - min(r,g,b))
        self.assertLess(out_sat, in_sat)


# ═══════════════════════════════════════════════════════════════════════════════
# TestPipelineIntegration
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineIntegration(unittest.TestCase):
    """Lightweight pipeline integration tests: LensCorrection → Demosaic → PFR."""

    def _run_pipeline(self, raw, lc_params=None, pfr_params=None):
        """Minimal: LensCorrection → simple half-res demosaic → PFR."""
        si = _sensor_info('rggb', 12, raw.shape[0], raw.shape[1])
        pf = _platform()

        # LensCorrection
        lc  = LensCorrection(raw, pf, si, lc_params or _parm_lens(is_enable=False))
        lc_out = lc.execute()
        self.assertEqual(lc_out.dtype, np.uint16)

        # Simple half-res demosaic
        norm = lc_out.astype(np.float32) / 4095.0
        R  = norm[0::2, 0::2]
        Gr = norm[0::2, 1::2]
        Gb = norm[1::2, 0::2]
        B  = norm[1::2, 1::2]
        rgb = np.stack([R, (Gr+Gb)*0.5, B], axis=2)

        # PFR
        pfr = PurpleFringeRemoval(rgb, pf, si, pfr_params or _parm_pfr(is_enable=False))
        pfr_out = pfr.execute()
        return pfr_out

    def test_passthrough_pipeline(self):
        """All disabled: pipeline is lossless."""
        raw = _rggb_bayer(64, 64)
        out1 = self._run_pipeline(raw)
        out2 = self._run_pipeline(raw)
        np.testing.assert_array_equal(out1, out2)

    def test_lc_enabled_pipeline_runs(self):
        raw = _rggb_bayer(64, 64)
        out = self._run_pipeline(raw, lc_params=_parm_lens(
            is_enable=True, ldc_enable=True, k1=-0.1))
        self.assertEqual(out.shape, (32, 32, 3))

    def test_pfr_enabled_pipeline_runs(self):
        raw = _rggb_bayer(64, 64)
        out = self._run_pipeline(raw, pfr_params=_parm_pfr(is_enable=True))
        self.assertEqual(out.shape, (32, 32, 3))

    def test_both_enabled_pipeline_runs(self):
        raw = _rggb_bayer(64, 64)
        out = self._run_pipeline(
            raw,
            lc_params=_parm_lens(is_enable=True, cac_enable=True, r_ca=[0.05,0,0]),
            pfr_params=_parm_pfr(is_enable=True),
        )
        self.assertEqual(out.shape, (32, 32, 3))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_output_values_finite_no_nan(self):
        """No NaN or Inf should be produced regardless of config."""
        raw = _rggb_bayer(64, 64)
        for k1 in [-0.3, 0.0, 0.3]:
            out = self._run_pipeline(raw, lc_params=_parm_lens(
                is_enable=True, ldc_enable=True, k1=k1))
            self.assertTrue(np.all(np.isfinite(out)), f"NaN/Inf at k1={k1}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
