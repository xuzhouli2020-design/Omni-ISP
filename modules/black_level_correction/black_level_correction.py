"""
File: black_level_correction.py
Description: Implements black level correction and image linearization based on config file params
Code / Paper  Reference:
Author: 10xEngineers Pvt Ltd
------------------------------------------------------------
"""
import time
import numpy as np

from util.utils import save_output_array


class BlackLevelCorrection:
    """
    Black Level Correction
    """

    def __init__(self, img, platform, sensor_info, parm_blc):
        self.img = img
        self.enable = parm_blc["is_enable"]
        self.sensor_info = sensor_info
        self.param_blc = parm_blc.copy()   # copy so OB estimation can update offsets
        self.is_linearize = self.param_blc["is_linear"]
        self.is_save = parm_blc["is_save"]
        self.platform = platform
        # step controls which part of BLC to apply:
        #   "full"           — offset subtract + linearise (original behaviour, default)
        #   "offset_only"    — subtract black level offsets, no scaling
        #   "linearise_only" — scale to full range only (assumes offsets already removed)
        self.step = parm_blc.get("step", "full")
        # OB pixel settings
        self.use_ob_pixels     = bool(parm_blc.get("use_ob_pixels",    False))
        self.ob_rows           = int(parm_blc.get("ob_rows",           8))
        self.ob_cols           = int(parm_blc.get("ob_cols",           0))
        self.ob_correction_mode = str(parm_blc.get("ob_correction_mode", "scalar"))
        self.ob_smoothing      = str(parm_blc.get("ob_smoothing",      "median"))

        # If OB pixel estimation is requested, override config offsets now
        if self.use_ob_pixels and self.step in ("full", "offset_only"):
            self._estimate_offsets_from_ob()

    # ------------------------------------------------------------------
    # OB pixel estimation helpers (Phase 2.7)
    # ------------------------------------------------------------------

    def _ob_stat(self, pixels: np.ndarray) -> float:
        """Compute scalar black level estimate from OB pixel array."""
        if self.ob_smoothing == "mean":
            return float(np.mean(pixels))
        return float(np.median(pixels))   # default: median (robust to hot pixels)

    def _estimate_offsets_from_ob(self):
        """
        Estimate per-channel black levels from the optical black pixel region
        and update self.param_blc offsets so apply_blc_parameters() uses them.

        Supports:
          ob_correction_mode="scalar"     — single float per channel (replaces config offsets)
          ob_correction_mode="per_column" — 1-D column offset (stored separately, handled in
                                            apply_blc_per_column())

        OB pixels: top ob_rows rows, left ob_cols columns (either/both may be 0).
        If neither ob_rows nor ob_cols is configured, falls back to config offsets silently.
        """
        raw   = np.float32(self.img)
        bayer = self.sensor_info["bayer_pattern"]

        # Gather OB region(s) into a single concatenated array per mode
        if self.ob_rows > 0 and self.ob_cols > 0:
            ob_h = np.vstack([raw[:self.ob_rows, :],
                               raw[:, :self.ob_cols].T])
        elif self.ob_rows > 0:
            ob_h = raw[:self.ob_rows, :]
        elif self.ob_cols > 0:
            ob_h = raw[:, :self.ob_cols]
        else:
            return   # no OB region configured — keep config values

        if self.ob_correction_mode == "scalar":
            # Extract per-channel OB pixels according to bayer pattern
            # Map: bayer string → (row_step, col_step) for each of r, gr, gb, b
            _MASKS = {
                "rggb":  [(0, 0), (0, 1), (1, 0), (1, 1)],  # r, gr, gb, b
                "bggr":  [(1, 1), (1, 0), (0, 1), (0, 0)],
                "grbg":  [(0, 1), (0, 0), (1, 1), (1, 0)],
                "gbrg":  [(1, 0), (1, 1), (0, 0), (0, 1)],
            }
            offsets_idx = _MASKS.get(bayer, _MASKS["rggb"])
            # offsets_idx: (r_row_parity, r_col_parity), (gr...), (gb...), (b...)
            (r_rp, r_cp), (gr_rp, gr_cp), (gb_rp, gb_cp), (b_rp, b_cp) = offsets_idx

            self.param_blc["r_offset"]  = self._ob_stat(ob_h[r_rp::2,  r_cp::2])
            self.param_blc["gr_offset"] = self._ob_stat(ob_h[gr_rp::2, gr_cp::2])
            self.param_blc["gb_offset"] = self._ob_stat(ob_h[gb_rp::2, gb_cp::2])
            self.param_blc["b_offset"]  = self._ob_stat(ob_h[b_rp::2,  b_cp::2])

        elif self.ob_correction_mode in ("per_column", "per_row"):
            # Store the column (or row) offsets for apply_blc_per_column
            if self.ob_rows > 0:
                ob_region = raw[:self.ob_rows, :]
            else:
                ob_region = raw[:, :self.ob_cols].T  # transpose → rows of pixels
            if self.ob_smoothing == "mean":
                self._ob_col_offset = np.mean(ob_region, axis=0, keepdims=True)
            else:
                self._ob_col_offset = np.median(ob_region, axis=0, keepdims=True)
        # else: unsupported mode, fall back to config values (do nothing)

    def apply_blc_per_column(self, raw: np.ndarray) -> np.ndarray:
        """
        Apply a per-column (or per-row) offset estimated from OB pixels.
        Called from apply_blc_parameters when ob_correction_mode is per_column.
        """
        if hasattr(self, "_ob_col_offset"):
            return raw - self._ob_col_offset   # broadcast: (1, W) subtracted from (H, W)
        return raw

    def apply_blc_parameters(self):
        """
        Apply BLC parameters provided in config file (or OB-estimated values).

        Behaviour is controlled by self.step:
          "full"           — subtract offsets then linearise (original behaviour)
          "offset_only"    — subtract black level offsets, no scaling
          "linearise_only" — scale to full range only (offsets assumed removed)

        If use_ob_pixels=True and ob_correction_mode="per_column", the per-column
        path is taken for the offset step, then linearisation proceeds normally.
        """
        bpp = self.sensor_info["bit_depth"]

        # Per-column OB correction path (offset step only)
        if (self.use_ob_pixels
                and self.ob_correction_mode in ("per_column", "per_row")
                and self.step in ("full", "offset_only")):
            raw = np.float32(self.img)
            raw = self.apply_blc_per_column(raw)
            # linearise using config sat values if needed
            if self.step == "full" and self.is_linearize:
                bayer = self.sensor_info["bayer_pattern"]
                r_sat  = self.param_blc["r_sat"]
                gr_sat = self.param_blc["gr_sat"]
                gb_sat = self.param_blc["gb_sat"]
                b_sat  = self.param_blc["b_sat"]
                # After per-column subtract, offsets are ~0; use sat values directly
                scale = (2**bpp - 1)
                if bayer == "rggb":
                    raw[0::2, 0::2] = raw[0::2, 0::2] / r_sat  * scale
                    raw[0::2, 1::2] = raw[0::2, 1::2] / gr_sat * scale
                    raw[1::2, 0::2] = raw[1::2, 0::2] / gb_sat * scale
                    raw[1::2, 1::2] = raw[1::2, 1::2] / b_sat  * scale
            return np.uint16(np.clip(raw, 0, (2**bpp) - 1))

        # Standard scalar-offset path (original + OB-scalar override)
        # get config parm
        bayer = self.sensor_info["bayer_pattern"]
        bpp = self.sensor_info["bit_depth"]
        r_offset = self.param_blc["r_offset"]
        gb_offset = self.param_blc["gb_offset"]
        gr_offset = self.param_blc["gr_offset"]
        b_offset = self.param_blc["b_offset"]

        r_sat = self.param_blc["r_sat"]
        gr_sat = self.param_blc["gr_sat"]
        gb_sat = self.param_blc["gb_sat"]
        b_sat = self.param_blc["b_sat"]

        raw = np.float32(self.img)

        do_offset = self.step in ("full", "offset_only")
        do_linearise = self.step in ("full", "linearise_only") and self.is_linearize

        if bayer == "rggb":
            if do_offset:
                raw[0::2, 0::2] = raw[0::2, 0::2] - r_offset
                raw[0::2, 1::2] = raw[0::2, 1::2] - gr_offset
                raw[1::2, 0::2] = raw[1::2, 0::2] - gb_offset
                raw[1::2, 1::2] = raw[1::2, 1::2] - b_offset
            if do_linearise:
                raw[0::2, 0::2] = (
                    raw[0::2, 0::2] / (r_sat - r_offset) * ((2**bpp) - 1)
                )
                raw[0::2, 1::2] = (
                    raw[0::2, 1::2] / (gr_sat - gr_offset) * ((2**bpp) - 1)
                )
                raw[1::2, 0::2] = (
                    raw[1::2, 0::2] / (gb_sat - gb_offset) * ((2**bpp) - 1)
                )
                raw[1::2, 1::2] = (
                    raw[1::2, 1::2] / (b_sat - b_offset) * ((2**bpp) - 1)
                )

        elif bayer == "bggr":
            if do_offset:
                raw[0::2, 0::2] = raw[0::2, 0::2] - b_offset
                raw[0::2, 1::2] = raw[0::2, 1::2] - gb_offset
                raw[1::2, 0::2] = raw[1::2, 0::2] - gr_offset
                raw[1::2, 1::2] = raw[1::2, 1::2] - r_offset
            if do_linearise:
                raw[0::2, 0::2] = (
                    raw[0::2, 0::2] / (b_sat - b_offset) * ((2**bpp) - 1)
                )
                raw[0::2, 1::2] = (
                    raw[0::2, 1::2] / (gb_sat - gb_offset) * ((2**bpp) - 1)
                )
                raw[1::2, 0::2] = (
                    raw[1::2, 0::2] / (gr_sat - gr_offset) * ((2**bpp) - 1)
                )
                raw[1::2, 1::2] = (
                    raw[1::2, 1::2] / (r_sat - r_offset) * ((2**bpp) - 1)
                )

        elif bayer == "grbg":
            if do_offset:
                raw[0::2, 0::2] = raw[0::2, 0::2] - gr_offset
                raw[0::2, 1::2] = raw[0::2, 1::2] - r_offset
                raw[1::2, 0::2] = raw[1::2, 0::2] - b_offset
                raw[1::2, 1::2] = raw[1::2, 1::2] - gb_offset
            if do_linearise:
                raw[0::2, 0::2] = (
                    raw[0::2, 0::2] / (gr_sat - gr_offset) * ((2**bpp) - 1)
                )
                raw[0::2, 1::2] = (
                    raw[0::2, 1::2] / (r_sat - r_offset) * ((2**bpp) - 1)
                )
                raw[1::2, 0::2] = (
                    raw[1::2, 0::2] / (b_sat - b_offset) * ((2**bpp) - 1)
                )
                raw[1::2, 1::2] = (
                    raw[1::2, 1::2] / (gb_sat - gb_offset) * ((2**bpp) - 1)
                )

        elif bayer == "gbrg":
            if do_offset:
                raw[0::2, 0::2] = raw[0::2, 0::2] - gb_offset
                raw[0::2, 1::2] = raw[0::2, 1::2] - b_offset
                raw[1::2, 0::2] = raw[1::2, 0::2] - r_offset
                raw[1::2, 1::2] = raw[1::2, 1::2] - gr_offset
            if do_linearise:
                raw[0::2, 0::2] = (
                    raw[0::2, 0::2] / (gb_sat - gb_offset) * ((2**bpp) - 1)
                )
                raw[0::2, 1::2] = (
                    raw[0::2, 1::2] / (b_sat - b_offset) * ((2**bpp) - 1)
                )
                raw[1::2, 0::2] = (
                    raw[1::2, 0::2] / (r_sat - r_offset) * ((2**bpp) - 1)
                )
                raw[1::2, 1::2] = (
                    raw[1::2, 1::2] / (gr_sat - gr_offset) * ((2**bpp) - 1)
                )

        raw_blc = np.uint16(np.clip(raw, 0, (2**bpp) - 1))
        return raw_blc

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_black_level_correction_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """
        Black Level Correction
        """
        print("Black Level Correction = " + str(self.enable))

        if self.enable:
            start = time.time()
            blc_out = self.apply_blc_parameters()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = blc_out
        self.save()
        return self.img
