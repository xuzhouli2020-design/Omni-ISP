"""
File: color_correction_matrix.py
Description: Applies the 3x3 correction matrix on the image.
             Supports two architecture modes:
               "direct"  — original single-step camera→sRGB matrix (imatest convention,
                           backward compatible default)
               "xyz"     — two-step camera→XYZ→target_primaries pipeline using standard
                           CIE matrices from xyz_matrices.py (Phase 3.1)
Code / Paper  Reference: https://www.imatest.com/docs/colormatrix/
Author: 10xEngineers Pvt Ltd  /  EdgeISP Phase 3.1 extension
------------------------------------------------------------

XYZ mode config keys (all optional, default = backward-compat):
  mode: "direct"          # "direct" | "xyz"
  target: "srgb"          # "srgb" | "display_p3" | "bt2020" | "xyz"
                          #   (only used when mode="xyz")
  camera_to_xyz_red:   [0.4124564, 0.3575761, 0.1804375]   # row-vector
  camera_to_xyz_green: [0.2126729, 0.7151522, 0.0721750]   # row-vector
  camera_to_xyz_blue:  [0.0193339, 0.1191920, 0.9503041]   # row-vector
    (if absent, derived automatically from direct CCM + sRGB→XYZ)
"""
import time
import numpy as np

from util.utils import save_output_array
from modules.color_correction_matrix.xyz_matrices import (
    get_xyz_to_target,
    derive_camera_to_xyz,
)


class ColorCorrectionMatrix:
    "Apply the color correction 3x3 matrix"

    def __init__(self, img, platform, sensor_info, parm_ccm):
        self.img = img
        self.enable = parm_ccm["is_enable"]
        self.sensor_info = sensor_info
        self.parm_ccm = parm_ccm
        self.bit_depthth = sensor_info["bit_depth"]
        self.ccm_mat = None
        self.is_save = parm_ccm["is_save"]
        self.platform = platform
        # Phase 3.1: architecture mode
        self.mode = parm_ccm.get("mode", "direct")   # "direct" | "xyz"

    # ------------------------------------------------------------------
    # Matrix builders
    # ------------------------------------------------------------------

    def _build_direct_matrix(self) -> np.ndarray:
        """
        Original imatest-convention camera→sRGB matrix from config rows.
        Returns float32 (3, 3).
        """
        r_1 = np.array(self.parm_ccm["corrected_red"],   dtype=np.float32)
        r_2 = np.array(self.parm_ccm["corrected_green"], dtype=np.float32)
        r_3 = np.array(self.parm_ccm["corrected_blue"],  dtype=np.float32)
        return np.float32([r_1, r_2, r_3])   # shape (3, 3)

    def _build_xyz_matrix(self) -> np.ndarray:
        """
        Build the combined camera→XYZ→target matrix for XYZ intermediate mode.

        Derivation:
          1. If camera_to_xyz_{red,green,blue} rows are in config, use them
             directly as the camera→XYZ matrix (M_cam_xyz, row-vector conv.).
          2. Otherwise fall back to: M_cam_xyz = direct_CCM @ M_sRGB→XYZ.
          3. Compose:  M_combined = M_cam_xyz @ M_xyz_to_target.T
             (row-vector convention: out_row = in_row @ M_combined.T)

        Returns float32 (3, 3).
        """
        # --- camera→XYZ ---
        if ("camera_to_xyz_red" in self.parm_ccm
                and "camera_to_xyz_green" in self.parm_ccm
                and "camera_to_xyz_blue" in self.parm_ccm):
            r_x = np.array(self.parm_ccm["camera_to_xyz_red"],   dtype=np.float64)
            r_y = np.array(self.parm_ccm["camera_to_xyz_green"], dtype=np.float64)
            r_z = np.array(self.parm_ccm["camera_to_xyz_blue"],  dtype=np.float64)
            M_cam_xyz = np.stack([r_x, r_y, r_z], axis=0)   # (3, 3) row-vectors
        else:
            # Derive from the direct CCM as a convenience fallback
            direct_mat = self._build_direct_matrix().astype(np.float64)
            M_cam_xyz = derive_camera_to_xyz(direct_mat)     # (3, 3)

        # --- XYZ→target ---
        target = self.parm_ccm.get("target", "srgb")
        M_xyz_target = get_xyz_to_target(target)   # (3, 3) col-vector conv.

        # --- combine: row-vector convention ---
        # col-vector:  y_col = M_xyz_target @ M_cam_xyz @ x_col
        # row-vector:  y_row = x_row @ (M_cam_xyz.T @ M_xyz_target.T)
        #            = x_row @ M_combined.T  where M_combined = M_xyz_target @ M_cam_xyz
        M_combined = (M_xyz_target @ M_cam_xyz).astype(np.float32)
        return M_combined   # shape (3, 3)

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    def apply_ccm(self):
        """
        Apply CCM — dispatches to direct or XYZ mode based on config.
        """
        if self.mode == "xyz":
            self.ccm_mat = self._build_xyz_matrix()
        else:
            # default: original direct (backward compatible)
            self.ccm_mat = self._build_direct_matrix()

        # normalize nbit to 0-1 img
        img_f = np.float32(self.img) / (2**self.bit_depthth - 1)

        # convert to Nx3 (imatest row-vector convention)
        img_flat = img_f.reshape(-1, 3)

        # O*A => out = img_flat @ ccm_mat.T
        out_flat = np.matmul(img_flat, self.ccm_mat.T.astype(np.float32))

        # clipping after CCM is mandatory — XYZ matrices can produce negatives
        out_flat = np.float32(np.clip(out_flat, 0, 1))

        # convert back
        out = out_flat.reshape(img_f.shape)
        out = np.uint16(out * (2**self.bit_depthth - 1))

        return out

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array(
                self.platform["in_file"],
                self.img,
                "Out_color_correction_matrix_",
                self.platform,
                self.sensor_info["bit_depth"],
                self.sensor_info["bayer_pattern"],
            )

    def execute(self):
        """Execute ccm if enabled."""
        print("Color Correction Matrix = " + str(self.enable)
              + "  mode=" + self.mode)

        if self.enable:
            start = time.time()
            ccm_out = self.apply_ccm()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = ccm_out

        self.save()
        return self.img
