"""
File: ldci.py
Description: Implements the contrast adjustment in the yuv domain
Author: 10xEngineers
------------------------------------------------------------
"""
import time
from util.utils import save_output_array_yuv
from modules.ldci.clahe import CLAHE
from modules.ldci.guided_filter_ltm import GuidedFilterLTM


class LDCI:
    """
    Local Dynamic Contrast Enhancement

    mode: "clahe"          — original CLAHE bilinear tile method (default)
    mode: "guided_filter"  — Guided Filter Local Tone Mapping (high quality,
                             edge-aware, no tile artefacts, pure NumPy)
    """

    def __init__(self, yuv, platform, sensor_info, parm_ldci, conv_std):
        self.yuv = yuv
        self.img = yuv
        self.enable = parm_ldci["is_enable"]
        self.sensor_info = sensor_info
        self.parm_ldci = parm_ldci
        self.is_save = parm_ldci["is_save"]
        self.platform = platform
        self.conv_std = conv_std
        # mode: "clahe" (default, backward compat) | "guided_filter"
        self.mode = parm_ldci.get("mode", "clahe")

    def apply_ldci(self):
        """
        Applying LDCI module using the configured algorithm.
        """
        if self.mode == "guided_filter":
            gf = GuidedFilterLTM(self.yuv, self.parm_ldci)
            return gf.apply_guided_ltm()
        else:
            # Default: original CLAHE path — fully backward compatible
            clahe = CLAHE(self.yuv, self.platform, self.sensor_info, self.parm_ldci)
            return clahe.apply_clahe()

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array_yuv(
                self.platform["in_file"],
                self.img,
                "Out_ldci_",
                self.platform,
                self.conv_std,
            )

    def execute(self):
        """
        Executing LDCI module according to user choice
        """
        print(f"LDCI = {self.enable}  mode={self.mode}")

        if self.enable is True:
            start = time.time()
            s_out = self.apply_ldci()
            print(f"  Execution time: {time.time() - start:.3f}s")
            self.img = s_out

        self.save()
        return self.img
