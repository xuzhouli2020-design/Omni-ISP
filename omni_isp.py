"""
File: isp_pipeline.py
Description: Executes the complete pipeline
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
from pathlib import Path
import numpy as np
import yaml
import rawpy

import util.utils as util
from util.output_profile import apply_profile_to_params

from modules.crop.crop import Crop
from modules.dead_pixel_correction.dead_pixel_correction import (
    DeadPixelCorrection as DPC,
)
from modules.black_level_correction.black_level_correction import (
    BlackLevelCorrection as BLC,
)
from modules.oecf.oecf import OECF
from modules.digital_gain.digital_gain import DigitalGain as DG
from modules.lens_shading_correction.lens_shading_correction import (
    LensShadingCorrection as LSC,
)
from modules.bayer_noise_reduction.bayer_noise_reduction import (
    BayerNoiseReduction as BNR,
)
from modules.stats_3a.stats_collector import Stats3ACollector
from modules.auto_white_balance.auto_white_balance import AutoWhiteBalance as AWB
from modules.white_balance.white_balance import WhiteBalance as WB
from modules.auto_focus.auto_focus import AutoFocus
from modules.demosaic.demosaic import Demosaic
from modules.color_correction_matrix.color_correction_matrix import (
    ColorCorrectionMatrix as CCM,
)
from modules.gamma_correction.gamma_correction import GammaCorrection as GC
from modules.auto_exposure.auto_exposure import AutoExposure as AE
from modules.color_space_conversion.color_space_conversion import (
    ColorSpaceConversion as CSC,
)
from modules.ldci.ldci import LDCI
from modules.sharpen.sharpen import Sharpening as SHARP
from modules.noise_reduction_2d.noise_reduction_2d import NoiseReduction2d as NR2D
from modules.rgb_conversion.rgb_conversion import RGBConversion as RGBC
from modules.scale.scale import Scale
from modules.yuv_conv_format.yuv_conv_format import YUVConvFormat as YUV_C
from modules.burst_capture.burst_registration import register_stack
from modules.burst_capture.burst_merge import merge_burst
from modules.temporal_nr.temporal_nr import TemporalNR, TNRState
from modules.lens_correction.lens_correction import LensCorrection
from modules.purple_fringe_removal.purple_fringe_removal import PurpleFringeRemoval
from modules.dl_denoise.dl_denoise import DLDenoise
from modules.hdr_tone_mapping.hdr_tone_mapping import HDRToneMapping
from modules.hdr_merge.hdr_merge import HDRMerge


class OmniISP:
    """
    Omni-ISP Pipeline
    """

    def __init__(self, data_path, config_path):
        """
        Constructor: Initialize with config and raw file path
        and Load configuration parameter from yaml file
        """
        self.data_path = data_path
        self.load_config(config_path)

    def load_config(self, config_path):
        """
        Load config information to respective module parameters
        """
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as file:
            c_yaml = yaml.safe_load(file)

            # Extract workspace info
            self.platform = c_yaml["platform"]
            self.raw_file = self.platform["filename"]
            self.render_3a = self.platform["render_3a"]

            # Extract basic sensor info
            self.sensor_info = c_yaml["sensor_info"]

            # Get isp module params
            self.parm_dpc = c_yaml["dead_pixel_correction"]
            self.parm_dga = c_yaml["digital_gain"]
            self.parm_lsc = c_yaml["lens_shading_correction"]
            self.parm_bnr = c_yaml["bayer_noise_reduction"]
            self.parm_blc = c_yaml["black_level_correction"]
            self.parm_oec = c_yaml["oecf"]
            self.parm_wbc = c_yaml["white_balance"]
            self.parm_awb = c_yaml["auto_white_balance"]
            self.parm_dem = c_yaml["demosaic"]
            self.parm_ae = c_yaml["auto_exposure"]
            self.parm_ccm = c_yaml["color_correction_matrix"]
            self.parm_gmc = c_yaml["gamma_correction"]
            self.parm_csc = c_yaml["color_space_conversion"]
            self.parm_cse = c_yaml["color_saturation_enhancement"]
            self.parm_ldci = c_yaml["ldci"]
            self.parm_sha = c_yaml["sharpen"]
            self.parm_2dn = c_yaml["2d_noise_reduction"]
            self.parm_rgb = c_yaml["rgb_conversion"]
            self.parm_sca = c_yaml["scale"]
            self.parm_cro = c_yaml["crop"]
            self.parm_yuv = c_yaml["yuv_conversion_format"]
            self.c_yaml = c_yaml

            # Phase 3.3: apply output profile overrides (CCM target + Gamma EOTF)
            # If output: section is absent or profile="custom", this is a no-op.
            output_cfg = c_yaml.get("output", {})
            apply_profile_to_params(output_cfg, self.parm_ccm, self.parm_gmc)

            self.platform["rgb_output"] = self.parm_rgb["is_enable"]

            # Phase 4 — 3A config sections (with safe defaults for missing keys)
            self.parm_3a  = c_yaml.get("3a_stats",    {"is_enable": True})
            self.parm_af  = c_yaml.get("auto_focus",  {"is_enable": False})

            # Phase 5 — multi-frame and TNR config sections
            self.parm_burst = c_yaml.get("burst_capture", {"is_enable": False})
            self.parm_tnr   = c_yaml.get("temporal_nr",   {"is_enable": False})

            # Phase 6 — lens correction config sections
            self.parm_lens  = c_yaml.get("lens_correction",       {"is_enable": False})
            self.parm_pfr   = c_yaml.get("purple_fringe_removal", {"is_enable": False})
            self.parm_dl    = c_yaml.get("dl_denoise",            {"is_enable": False})
            self.parm_hdrtm = c_yaml.get("hdr_tone_mapping",     {"is_enable": False})
            self.parm_hdrm  = c_yaml.get("hdr_merge",            {"is_enable": False})

        # Phase 4 — persistent 3A state across execute() calls
        self._af_machine      = None   # AFSearchMachine
        self._awb_prev_gains  = None   # np.array([r, b])
        self._ae_prev_params  = None   # ExposureParams

        # Phase 5 — persistent multi-frame state
        self._tnr_state       = None   # TNRState (prev frame for IIR filter)
        # Burst state: raw_stack is populated externally via set_burst_stack()
        # or by a burst_loader call inside run_pipeline() when is_enable=True.
        self._burst_stack     = None   # (N, H, W) uint16 — externally supplied

    def set_burst_stack(self, stack: np.ndarray) -> None:
        """
        Supply a pre-loaded (N, H, W) uint16 Bayer burst stack.
        When burst_capture.is_enable is True and this stack is set,
        run_pipeline() will register + merge the stack instead of using
        self.raw as the single input frame.

        The stack should already have BLC-offset + OECF applied per frame
        (use RawStackLoader for this).  Frame 0 is the reference.
        """
        self._burst_stack = stack

    def merge_exposures(
        self,
        linear_rgb_frames: list,
        exposure_evs: list = None,
    ) -> np.ndarray:
        """
        Phase 8.1 — Multi-exposure HDR merge (call before run_pipeline).

        Takes a list of linear RGB frames (already processed through CCM) from
        different exposure brackets and returns a single merged frame.

        Typical workflow
        ----------------
        1. Run execute() on each bracket to produce linear RGB through CCM.
        2. Collect the CCM-output frames in a list (capture them from a custom
           pipeline subclass or by calling run_pipeline with an early exit).
        3. Call merge_exposures(frames, evs) to get a merged float32 frame.
        4. Feed the merged frame into the tone mapping + gamma stages.

        For convenience, this can also be used standalone:
          merged = isp.merge_exposures(frames)  # Mertens (no EV values needed)
          merged = isp.merge_exposures(frames, [-2, 0, 2])  # Debevec

        Parameters
        ----------
        linear_rgb_frames : list of np.ndarray
            Each (H, W, 3) float32 linear RGB, normalised to [0, 1].
        exposure_evs      : list of float or None
            EV offsets for Debevec mode (e.g. [-2, 0, +2]).
            Ignored in Mertens mode.

        Returns
        -------
        np.ndarray  (H, W, 3) float32 in [0, 1]
        """
        merge = HDRMerge(
            images=linear_rgb_frames,
            exposure_evs=exposure_evs,
            parm_hdrm=self.parm_hdrm,
        )
        return merge.execute()

    def load_raw(self):
        """
        Load raw image from provided path
        """
        # Load raw image file information
        path_object = Path(self.data_path, self.raw_file)
        raw_path = str(path_object.resolve())
        self.in_file = path_object.stem
        self.out_file = "Out_" + self.in_file

        self.platform["in_file"] = self.in_file
        self.platform["out_file"] = self.out_file

        width = self.sensor_info["width"]
        height = self.sensor_info["height"]
        bit_depth = self.sensor_info["bit_depth"]

        # Load Raw
        if path_object.suffix == ".raw":
            if bit_depth > 8:
                self.raw = np.fromfile(raw_path, dtype=np.uint16).reshape(
                    (height, width)
                )
            else:
                self.raw = (
                    np.fromfile(raw_path, dtype=np.uint8)
                    .reshape((height, width))
                    .astype(np.uint16)
                )
        else:
            img = rawpy.imread(raw_path)
            self.raw = img.raw_image

    def run_pipeline(self, visualize_output=True):
        """
        Simulation of ISP-Pipeline
        """

        # =====================================================================
        # Cropping
        crop = Crop(self.raw, self.platform, self.sensor_info, self.parm_cro)
        cropped_img = crop.execute()

        # =====================================================================
        #  Dead pixels correction
        dpc = DPC(cropped_img, self.sensor_info, self.parm_dpc, self.platform)
        dpc_raw = dpc.execute()

        # =====================================================================
        # Black level correction — step 1: subtract per-channel offsets only
        # (see dev_notes.md "OECF before BLC linearization")
        blc_offset = BLC(
            dpc_raw,
            self.platform,
            self.sensor_info,
            {**self.parm_blc, "step": "offset_only"},
        )
        blc_offset_raw = blc_offset.execute()

        # =====================================================================
        # OECF — must index into BL-offset-corrected domain, before linearisation
        oecf = OECF(blc_offset_raw, self.platform, self.sensor_info, self.parm_oec)
        oecf_raw = oecf.execute()

        # =====================================================================
        # Burst capture merge (Phase 5.1–5.3)
        # When burst_capture.is_enable and a stack was supplied via
        # set_burst_stack(), register all N frames to frame 0 and merge them
        # into a single denoised Bayer image that replaces oecf_raw.
        # Position: after OECF (shared domain), before BLC linearise.
        burst_cfg = self.parm_burst
        if burst_cfg.get("is_enable", False) and self._burst_stack is not None:
            bayer_pat = self.sensor_info.get("bayer_pattern", "rggb")
            ref_idx   = int(burst_cfg.get("ref_idx",         0))
            reg_mode  = str(burst_cfg.get("registration",    "phase"))
            method    = str(burst_cfg.get("merge_method",    "weighted_mean"))
            block_sz  = int(burst_cfg.get("block_size",      16))
            sad_thr   = float(burst_cfg.get("motion_threshold", 15.0))

            print(f"  [Burst] {self._burst_stack.shape[0]} frames  "
                  f"reg={reg_mode}  merge={method}")

            if reg_mode == "phase":
                registered, shifts = register_stack(
                    self._burst_stack, bayer_pat, ref_idx
                )
            else:
                registered = self._burst_stack

            oecf_raw = merge_burst(
                registered, bayer_pat, ref_idx, block_sz, sad_thr, method
            )

        # =====================================================================
        # Black level correction — step 2: linearise (scale to full range)
        blc_lin = BLC(
            oecf_raw,
            self.platform,
            self.sensor_info,
            {**self.parm_blc, "step": "linearise_only"},
        )
        blc_raw = blc_lin.execute()

        # =====================================================================
        # Digital Gain
        dga = DG(blc_raw, self.platform, self.sensor_info, self.parm_dga)
        dga_raw, self.dga_current_gain = dga.execute()

        # =====================================================================
        # Lens Correction — unified CAC + LDC warp (Phase 6)
        # Composed into a single per-channel bilinear remap in Bayer domain.
        # Position: after Digital Gain (signal is linear), before LSC (which
        # assumes geometric correctness) and before Demosaic (channels must be
        # aligned before interpolation).
        lc = LensCorrection(dga_raw, self.platform, self.sensor_info, self.parm_lens)
        lc_raw = lc.execute()

        # =====================================================================
        # Lens shading correction
        lsc = LSC(lc_raw, self.platform, self.sensor_info, self.parm_lsc)
        lsc_raw = lsc.execute()

        # =====================================================================
        # DL-B: joint Bayer→RGB (Phase 7) — attempt before classical BNR/Demosaic
        # When active, the model receives WB-corrected Bayer and returns full-res
        # linear RGB in [0,1], replacing BNR + Demosaic in a single forward pass.
        # Falls back to classical path silently if model is missing or ort unavailable.
        _parm_dl   = self.parm_dl
        _use_dl_b  = (
            _parm_dl.get("is_enable", False)
            and _parm_dl.get("mode", "") == "bayer_joint"
        )
        _dl_b_active = False

        # =====================================================================
        # Bayer noise reduction — skipped when DL-B will handle it
        if _use_dl_b:
            # DL model takes over BNR's job; pass lsc_raw through for 3A/AWB
            bnr_raw = lsc_raw
        else:
            bnr = BNR(lsc_raw, self.sensor_info, self.parm_bnr, self.platform)
            bnr_raw = bnr.execute()

        # =====================================================================
        # Temporal Noise Reduction (Phase 5.5) — skipped in DL-B mode
        # IIR EMA filter with motion masking — video / continuous capture mode.
        # Runs after BNR (cleaner input) and before Stats3A so 3A sees
        # temporally-smoothed Bayer data.
        if not _use_dl_b:
            tnr_mod = TemporalNR(
                bnr_raw,
                self.sensor_info,
                self.parm_tnr,
                tnr_state = self._tnr_state,
            )
            bnr_raw = tnr_mod.execute()
            # Persist TNR state for next frame
            if tnr_mod.new_state is not None:
                self._tnr_state = tnr_mod.new_state

        # =====================================================================
        # 3A Statistics — shared Bayer-domain stats for AE, AWB, AF
        # (Phase 4.1) Runs after BNR (clean signal), before AWB.
        stats_col = Stats3ACollector(bnr_raw, self.sensor_info, self.parm_3a)
        self.stats_3a = stats_col.execute()

        # =====================================================================
        # Auto Focus — passive; does not modify image; updates lens metadata
        # (Phase 4.8) Runs in Bayer domain using Tenengrad from stats_3a.
        af_mod = AutoFocus(
            bnr_raw,
            self.sensor_info,
            self.parm_af,
            stats    = self.stats_3a,
            af_state = self._af_machine,
        )
        af_mod.execute()
        # Persist state machine for next frame
        self._af_machine   = af_mod.af_machine
        self.af_result     = af_mod.result

        # =====================================================================
        # Auto White Balance
        awb = AWB(
            bnr_raw,
            self.sensor_info,
            self.parm_awb,
            stats      = self.stats_3a,
            prev_gains = self._awb_prev_gains,
        )
        self.awb_gains = awb.execute()
        if self.awb_gains is not None:
            self._awb_prev_gains = self.awb_gains.copy()

        # =====================================================================
        # White balancing
        wbc = WB(bnr_raw, self.platform, self.sensor_info, self.parm_wbc)
        wb_raw = wbc.execute()

        # =====================================================================
        # DL-B joint inference: WB-corrected Bayer → linear RGB in one pass.
        # If the model is unavailable we fall back to classical Demosaic below.
        if _use_dl_b:
            _dl_b_mod    = DLDenoise(wb_raw, self.platform, self.sensor_info, _parm_dl)
            _dl_b_result = _dl_b_mod.execute()
            _dl_b_active = _dl_b_mod.dl_active

        # =====================================================================
        # CFA demosaicing — skipped when DL-B ran successfully
        if _dl_b_active:
            # DL result is float32 (H, W, 3) in [0, 1]; scale to sensor range
            # so downstream modules (CCM, LDCI, etc.) see familiar magnitude.
            _wl = float(self.sensor_info.get("range",
                        (1 << self.sensor_info.get("bit_depth", 12)) - 1))
            demos_img = np.clip(_dl_b_result * _wl, 0.0, _wl).astype(np.float32)
        else:
            cfa_inter = Demosaic(wb_raw, self.platform, self.sensor_info, self.parm_dem)
            demos_img = cfa_inter.execute()

        # =====================================================================
        # Purple Fringe Removal — RGB domain, after demosaic, before CCM (Phase 6)
        # Detects purple/magenta halos at blown highlight edges (longitudinal CA)
        # and selectively desaturates them.  Must run after demosaic (needs full
        # RGB hue info) and before CCM (avoids re-introducing hue error).
        pfr = PurpleFringeRemoval(demos_img, self.platform, self.sensor_info, self.parm_pfr)
        demos_img = pfr.execute()

        # =====================================================================
        # Color correction matrix
        ccm = CCM(demos_img, self.platform, self.sensor_info, self.parm_ccm)
        ccm_img = ccm.execute()

        # =====================================================================
        # Local Dynamic Contrast Improvement — linear domain
        # Must run before Gamma so it operates on linear light values.
        # (see dev_notes.md "Gamma ordering fix")
        #
        # LDCI/NR2D internally process channel 0 as luminance Y.
        # We extract linear Y from CCM's linear RGB, pass as channel 0 of a
        # fake YCbCr array, then recover RGB proportionally from enhanced Y.
        #
        # Normalisation: CCM outputs uint16 in [0, (2^bit_depth)-1].
        # LDCI, NR2D (NLM/bilateral), DL-A, and HDR TM all expect float32 in
        # [0, 1].  We normalise here and scale back just before Gamma so the
        # LUT (and analytical EOTF) receive values in the correct range.
        _white_level = float((1 << self.sensor_info.get("bit_depth", 12)) - 1)
        _r = ccm_img[..., 0].astype(np.float32) / _white_level
        _g = ccm_img[..., 1].astype(np.float32) / _white_level
        _b = ccm_img[..., 2].astype(np.float32) / _white_level
        _y_lin = (0.2126 * _r + 0.7152 * _g + 0.0722 * _b).astype(np.float32)
        _fake_yuv = np.stack(
            [_y_lin, np.zeros_like(_y_lin), np.zeros_like(_y_lin)], axis=2
        ).astype(np.float32)
        ldci = LDCI(
            _fake_yuv,
            self.platform,
            self.sensor_info,
            self.parm_ldci,
            self.parm_csc["conv_standard"],
        )
        _ldci_out = ldci.execute()
        _y_enh = _ldci_out[..., 0]
        _ratio = _y_enh / np.maximum(_y_lin, 1e-8)
        ldci_img = np.stack(
            [_r * _ratio, _g * _ratio, _b * _ratio], axis=2
        ).astype(np.float32)

        # =====================================================================
        # 2D Noise Reduction — linear domain, before Gamma and before Sharpen
        # (see dev_notes.md "Gamma ordering fix", "Sharpening — pipeline order fix")
        #
        # DL-A path ("rgb_post"): the DL model handles Y-channel denoising; we
        # downgrade 2D NR to chroma_only so Cb/Cr still get a light polish but
        # we avoid re-blurring luminance the model just cleaned.
        # DL-B path ("bayer_joint"): the joint model already denoised everything;
        # again we run 2D NR in chroma_only as a cheap safety valve.
        _use_dl_a = (
            _parm_dl.get("is_enable", False)
            and _parm_dl.get("mode", "") == "rgb_post"
        )

        # Effective 2D NR params: force chroma_only when any DL path is active
        _parm_2dn_eff = self.parm_2dn
        if _dl_b_active or _use_dl_a:
            _parm_2dn_eff = dict(self.parm_2dn)
            _parm_2dn_eff["mode"] = "chroma_only"

        # DL-A inference: operates on linear RGB output of LDCI
        if _use_dl_a:
            _dl_a_mod    = DLDenoise(ldci_img, self.platform, self.sensor_info, _parm_dl)
            _dl_a_result = _dl_a_mod.execute()
            if _dl_a_mod.dl_active:
                ldci_img = _dl_a_result   # denoised linear RGB replaces LDCI output

        _r2, _g2, _b2 = ldci_img[..., 0], ldci_img[..., 1], ldci_img[..., 2]
        _y_lin2 = (0.2126 * _r2 + 0.7152 * _g2 + 0.0722 * _b2).astype(np.float32)
        _fake_yuv2 = np.stack(
            [_y_lin2, np.zeros_like(_y_lin2), np.zeros_like(_y_lin2)], axis=2
        ).astype(np.float32)
        nr2d = NR2D(
            _fake_yuv2,
            self.sensor_info,
            _parm_2dn_eff,
            self.platform,
            self.parm_csc["conv_standard"],
        )
        _nr2d_out = nr2d.execute()
        _y_den = _nr2d_out[..., 0]
        _ratio2 = _y_den / np.maximum(_y_lin2, 1e-8)
        nr2d_img = np.stack(
            [_r2 * _ratio2, _g2 * _ratio2, _b2 * _ratio2], axis=2
        ).astype(np.float32)

        # =====================================================================
        # HDR Tone Mapping — Phase 8.3
        # Sits between 2D NR (last linear-domain step) and Gamma EOTF.
        # Required for correct PQ / HLG output:
        #   PQ  — maps scene luminance to [0, peak_nits], normalised to [0, 1]
        #   HLG — light highlight rolloff only (scene-referred encoding)
        #   SDR — optional filmic rolloff for high-dynamic-range scenes
        # Input : float32 linear RGB [0, ~1] (may exceed 1.0 in HDR scenes)
        # Output: float32 linear RGB [0, 1]   ready for Gamma EOTF
        hdrtm = HDRToneMapping(
            nr2d_img, self.platform, self.sensor_info, self.parm_hdrtm
        )
        tm_img = hdrtm.execute()

        # =====================================================================
        # Gamma — applied after all linear-domain processing (LDCI, 2D NR, TM)
        # (see dev_notes.md "Gamma ordering fix")
        #
        # Scale from normalised [0, 1] back to sensor bit-depth range so the
        # LUT gamma (and analytical EOTFs that normalise internally) receives
        # the expected value domain.  HDR TM output is clipped to [0, 1]
        # before scaling so that LUT indices stay within [0, (2^depth)-1].
        _tm_scaled = np.clip(tm_img, 0.0, 1.0) * _white_level
        gmc = GC(_tm_scaled, self.platform, self.sensor_info, self.parm_gmc)
        gamma_raw = gmc.execute()

        # =====================================================================
        # Auto-Exposure — runs post-gamma for luminance evaluation (legacy path)
        # or uses Stats3A zone_means in zone_metering mode (Phase 4.2/4.3).
        aef = AE(
            gamma_raw,
            self.sensor_info,
            self.parm_ae,
            stats       = self.stats_3a,
            prev_params = self._ae_prev_params,
        )
        self.ae_feedback = aef.execute()
        # Persist ExposureParams state for next frame (zone_metering mode)
        from modules.auto_exposure.ae_controller import ExposureParams
        if isinstance(self.ae_feedback, ExposureParams):
            self._ae_prev_params = self.ae_feedback

        # =====================================================================
        # Color space conversion
        csc = CSC(gamma_raw, self.platform, self.sensor_info, self.parm_csc, self.parm_cse)
        csc_img = csc.execute()

        # =====================================================================
        # Sharpening — after CSC (YUV domain), after 2D NR
        # (see dev_notes.md "Sharpening — pipeline order fix")
        sharp = SHARP(
            csc_img,
            self.platform,
            self.sensor_info,
            self.parm_sha,
            self.parm_csc["conv_standard"],
        )
        sharp_img = sharp.execute()

        # =====================================================================
        # RGB conversion
        rgbc = RGBC(
            sharp_img, self.platform, self.sensor_info, self.parm_rgb, self.parm_csc
        )
        rgbc_img = rgbc.execute()

        # =====================================================================
        # Scaling
        scale = Scale(
            rgbc_img,
            self.platform,
            self.sensor_info,
            self.parm_sca,
            self.parm_csc["conv_standard"],
        )
        scaled_img = scale.execute()

        # =====================================================================
        # YUV saving format 444, 422 etc
        yuv = YUV_C(scaled_img, self.platform, self.sensor_info, self.parm_yuv)
        yuv_conv = yuv.execute()

        # only to view image if csc is off it does nothing
        out_img = yuv_conv
        out_dim = scaled_img.shape  # dimensions of Output Image

        # Is not part of ISP-pipeline only assists in visualizing output results
        if visualize_output:

            # There can be two out_img formats depending upon which modules are
            # enabled 1. YUV    2. RGB

            if self.parm_yuv["is_enable"] is True:

                # YUV_C is enabled and RGB_C is disabled: Output is compressed YUV
                # To display : Need to decompress it and convert it to RGB.
                image_height, image_width, _ = out_dim
                yuv_custom_format = self.parm_yuv["conv_type"]

                yuv_conv = util.get_image_from_yuv_format_conversion(
                    yuv_conv, image_height, image_width, yuv_custom_format
                )

                rgbc.yuv_img = yuv_conv
                out_rgb = rgbc.yuv_to_rgb()

            elif self.parm_rgb["is_enable"] is False:

                # RGB_C is disabled: Output is 3D - YUV
                # To display : Only convert it to RGB
                rgbc.yuv_img = yuv_conv
                out_rgb = rgbc.yuv_to_rgb()

            else:
                # RGB_C is enabled: Output is RGB
                # no further processing is needed for display
                out_rgb = out_img

            # If both RGB_C and YUV_C are enabled. Omni-ISP will generate
            # an output but it will be an invalid image.

            util.save_pipeline_output(self.out_file, out_rgb, self.c_yaml)

    def execute(self, img_path=None):
        """
        Start execution of Omni-ISP
        """
        if img_path is not None:
            self.raw_file = img_path
            self.c_yaml["platform"]["filename"] = self.raw_file

        self.load_raw()

        # Print Logs to mark start of pipeline Execution
        print(50 * "-" + "\nLoading RAW Image Done......\n")
        print("Filename: ", self.in_file)

        # Note Initial Time for Pipeline Execution
        start = time.time()

        if not self.render_3a:
            # Run ISP-Pipeline once
            self.run_pipeline(visualize_output=True)
            # Display 3A Statistics
        else:
            # Run ISP-Pipeline till Correct Exposure with AWB gains
            self.execute_with_3a_statistics()

        util.display_ae_statistics(self.ae_feedback, self.awb_gains)

        # Print Logs to mark end of pipeline Execution
        print(50 * "-" + "\n")

        # Calculate pipeline execution time
        print(f"\nPipeline Elapsed Time: {time.time() - start:.3f}s")

    def load_3a_statistics(self, awb_on=True, ae_on=True):
        """
        Update 3A Stats into WB and DG modules parameters
        """
        # Update 3A in c_yaml too because it is output config
        if awb_on is True and self.parm_dga["is_auto"] and self.parm_awb["is_enable"]:
            self.parm_wbc["r_gain"] = self.c_yaml["white_balance"]["r_gain"] = float(
                self.awb_gains[0]
            )
            self.parm_wbc["b_gain"] = self.c_yaml["white_balance"]["b_gain"] = float(
                self.awb_gains[1]
            )
        if ae_on is True and self.parm_dga["is_auto"] and self.parm_ae["is_enable"]:
            # zone_metering mode: ExposureParams carries digital_gain directly
            from modules.auto_exposure.ae_controller import ExposureParams
            if isinstance(self.ae_feedback, ExposureParams):
                # Use the digital_gain component as the feedback integer equivalent
                # for the legacy DGA gain array logic.  Map to -1/0/+1.
                ep = self.ae_feedback
                if ep.converged:
                    fb = 0
                elif ep.error_ev > 0:
                    fb = 1    # too dark → increase gain
                else:
                    fb = -1   # too bright → decrease gain
                self.parm_dga["ae_feedback"] = self.c_yaml["digital_gain"][
                    "ae_feedback"
                ] = fb
            else:
                self.parm_dga["ae_feedback"] = self.c_yaml["digital_gain"][
                    "ae_feedback"
                ] = self.ae_feedback
            self.parm_dga["current_gain"] = self.c_yaml["digital_gain"][
                "current_gain"
            ] = self.dga_current_gain

    def execute_with_3a_statistics(self):
        """
        Execute Omni-ISP with AWB gains and correct exposure
        """

        # Maximum Iterations depend on total permissible gains
        max_dg = len(self.parm_dga["gain_array"])

        # Run ISP-Pipeline
        self.run_pipeline(visualize_output=False)
        self.load_3a_statistics()
        while not (
            (self.ae_feedback == 0)
            or (self.ae_feedback == -1 and self.dga_current_gain == max_dg)
            or (self.ae_feedback == 1 and self.dga_current_gain == 0)
            or self.ae_feedback is None
        ):
            self.run_pipeline(visualize_output=False)
            self.load_3a_statistics()

        self.run_pipeline(visualize_output=True)

    def update_sensor_info(self, sensor_info, update_blc_wb=False):
        """
        Update sensor_info in config files
        """
        self.sensor_info["width"] = self.c_yaml["sensor_info"]["width"] = sensor_info[0]

        self.sensor_info["height"] = self.c_yaml["sensor_info"]["height"] = sensor_info[
            1
        ]

        self.sensor_info["bit_depth"] = self.c_yaml["sensor_info"][
            "bit_depth"
        ] = sensor_info[2]

        self.sensor_info["bayer_pattern"] = self.c_yaml["sensor_info"][
            "bayer_pattern"
        ] = sensor_info[3]

        if update_blc_wb:
            self.parm_blc["r_offset"] = self.c_yaml["black_level_correction"][
                "r_offset"
            ] = sensor_info[4][0]
            self.parm_blc["gr_offset"] = self.c_yaml["black_level_correction"][
                "gr_offset"
            ] = sensor_info[4][1]
            self.parm_blc["gb_offset"] = self.c_yaml["black_level_correction"][
                "gb_offset"
            ] = sensor_info[4][2]
            self.parm_blc["b_offset"] = self.c_yaml["black_level_correction"][
                "b_offset"
            ] = sensor_info[4][3]

            self.parm_blc["r_sat"] = self.c_yaml["black_level_correction"][
                "r_sat"
            ] = sensor_info[5]
            self.parm_blc["gr_sat"] = self.c_yaml["black_level_correction"][
                "gr_sat"
            ] = sensor_info[5]
            self.parm_blc["gb_sat"] = self.c_yaml["black_level_correction"][
                "gb_sat"
            ] = sensor_info[5]
            self.parm_blc["b_sat"] = self.c_yaml["black_level_correction"][
                "b_sat"
            ] = sensor_info[5]

            self.parm_wbc["r_gain"] = self.c_yaml["white_balance"][
                "r_gain"
            ] = sensor_info[6][0]
            self.parm_wbc["b_gain"] = self.c_yaml["white_balance"][
                "b_gain"
            ] = sensor_info[6][2]

            # if sensor_info[7] is not None:
            #     self.parm_ccm["corrected_red"] = sensor_info[7][0,0:3]
            #     self.parm_ccm["corrected_green"] = sensor_info[7][1,0:3]
            #     self.parm_ccm["corrected_blue"] = sensor_info[7][2,0:3]
