"""
File: color_space_conversion.py
Description: Converts RGB to YUV or YCbCr
Code / Paper  Reference: https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.709_conversion
                         https://www.itu.int/rec/R-REC-BT.601/
                         https://www.itu.int/rec/R-REC-BT.709-6-201506-I/en
                         https://learn.microsoft.com/en-us/windows/win32/medfound/recommended-
                         8-bit-yuv-formats-for-video-rendering
                         https://web.archive.org/web/20180423091842/http://www.equasys.de/
                         colorconversion.html
                         Author: 10xEngineers Pvt Ltd
------------------------------------------------------------------------------
"""

import time
import numpy as np

from util.utils import save_output_array_yuv


class ColorSpaceConversion:
    """
    Color Space Conversion
    """

    def __init__(self, img, platform, sensor_info, parm_csc, parm_cse):
        self.img = img.copy()
        self.is_save = parm_csc["is_save"]
        self.platform = platform
        self.sensor_info = sensor_info
        self.parm_csc = parm_csc
        self.bit_depth = sensor_info["bit_depth"]
        self.conv_std = self.parm_csc["conv_standard"]
        self.rgb2yuv_mat = None
        self.yuv_img = None
        self.parm_cse = parm_cse

    def rgb_to_yuv_8bit(self):
        """
        RGB-to-YUV Colorspace conversion 8bit
        """

        if self.conv_std == 1:
            # for BT. 709
            self.rgb2yuv_mat = np.array(
                [[47, 157, 16], [-26, -86, 112], [112, -102, -10]]
            )
        else:

            # for BT.601/407
            # conversion metrix with 8bit integer co-efficients - m=8
            self.rgb2yuv_mat = np.array(
                [[77, 150, 29], [131, -110, -21], [-44, -87, 138]]
            )

        # make nx3 2d matrix of image
        mat_2d = self.img.reshape((self.img.shape[0] * self.img.shape[1], 3))

        # convert to 3xn for matrix multiplication
        mat2d_t = mat_2d.transpose()

        # convert to YUV
        yuv_2d = np.matmul(self.rgb2yuv_mat, mat2d_t)

        # convert image with its provided bit_depth
        yuv_2d = np.float64(yuv_2d) / (2**8)
        yuv_2d = np.where(yuv_2d >= 0, np.floor(yuv_2d + 0.5), np.ceil(yuv_2d - 0.5))

        # color saturation enhancement block
        # Modes: "flat" (uniform gain — original, default) | "vibrance" (adaptive)
        # Both modes include a soft-knee chroma magnitude limiter.
        # Cb/Cr at this point are centred on 0, typical range ≈ [-127, +127].
        # Ref: dev_notes.md "Color Saturation Enhancement — dual-mode implementation plan"
        if self.parm_cse['is_enable']:
            mode_cse = self.parm_cse.get('mode', 'flat')
            cb = yuv_2d[1, :].astype(np.float64)
            cr = yuv_2d[2, :].astype(np.float64)
            chroma_mag = np.sqrt(cb * cb + cr * cr)  # magnitude, always >= 0

            if mode_cse == 'vibrance':
                # Vibrance: boost is inversely proportional to existing chroma
                # so muted colours gain more saturation than already-vivid ones.
                # chroma_max ≈ sqrt(127² + 127²) ≈ 180 for 8-bit Cb/Cr domain
                vibrance_strength = float(self.parm_cse.get('vibrance_strength', 0.3))
                chroma_max = 180.0
                chroma_norm = np.minimum(chroma_mag / chroma_max, 1.0)
                # gain_map: 1 + strength for silent colours → 1.0 for full vivid
                gain_map = 1.0 + vibrance_strength * (1.0 - chroma_norm)
                cb = cb * gain_map
                cr = cr * gain_map
            else:
                # "flat" mode — uniform gain (original behaviour, backward compat)
                gain = float(self.parm_cse.get('saturation_gain', 1.0))
                cb = cb * gain
                cr = cr * gain

            # Soft-knee chrominance limiter: scale Cb/Cr proportionally when the
            # chroma magnitude exceeds the limit, preserving hue angle.
            chroma_limit_norm = float(self.parm_cse.get('chroma_limit', 1.0))
            if chroma_limit_norm < 1.0:
                abs_limit = chroma_limit_norm * 128.0  # normalised → Cb/Cr domain
                chroma_out = np.sqrt(cb * cb + cr * cr)
                exceed = chroma_out > abs_limit
                if np.any(exceed):
                    scale = np.where(
                        exceed,
                        abs_limit / np.maximum(chroma_out, 1e-8),
                        1.0,
                    )
                    cb = cb * scale
                    cr = cr * scale

            yuv_2d[1, :] = cb
            yuv_2d[2, :] = cr




        # black-level/DC offset added to YUV values
        yuv_2d[0, :] = 2 ** (self.bit_depth - 4) + yuv_2d[0, :]
        yuv_2d[1, :] = 2 ** (self.bit_depth - 1) + yuv_2d[1, :]
        yuv_2d[2, :] = 2 ** (self.bit_depth - 1) + yuv_2d[2, :]

        # reshape the image back
        yuv2d_t = yuv_2d.transpose()

        yuv2d_t = np.clip(yuv2d_t, 0, (2**self.bit_depth) - 1)

        # Modules after CSC need 8-bit YUV so converting it into 8-bit after Normalizing.
        yuv2d_t = yuv2d_t / (2 ** (self.bit_depth - 8))
        yuv2d_t = np.where(
            yuv2d_t >= 0, np.floor(yuv2d_t + 0.5), np.ceil(yuv2d_t - 0.5)
        )

        yuv2d_t = np.clip(yuv2d_t, 0, 255)

        self.img = yuv2d_t.reshape(self.img.shape).astype(np.uint8)
        return self.img

    def save(self):
        """
        Function to save module output
        """
        if self.is_save:
            save_output_array_yuv(
                self.platform["in_file"],
                self.img,
                "Out_color_space_conversion_",
                self.platform,
                self.conv_std,
            )

    def execute(self):
        """
        Execute Color Space Conversion
        """
        print("Color Space Conversion (default) = True")

        start = time.time()
        csc_out = self.rgb_to_yuv_8bit()
        print(f"  Execution time: {time.time() - start:.3f}s")
        self.img = csc_out
        self.save()
        return self.img
