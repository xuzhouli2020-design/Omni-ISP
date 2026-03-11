"""
modules/dl_denoise/dl_denoise.py
=================================
DLDenoise — ONNX-based denoising / joint demosaic+denoise module.

Supports two pipeline modes:

  DL-A  ("rgb_post")    — post-demosaic RGB denoiser (NAFNet-SIDD)
      Replaces 2D NR. BNR + Demosaic still run classically.
      Input to module:  linear RGB (H, W, 3) or (3, H, W)
      Output:           denoised linear RGB, same shape

  DL-B  ("bayer_joint") — joint Bayer→RGB (BJDD or equivalent)
      Replaces BNR + Demosaic + 2D NR Y-channel in one pass.
      Input to module:  uint16 Bayer after Digital Gain + Lens Correction
      Output:           float32 linear RGB (H, W, 3)

Graceful fallback:
  If onnxruntime is not installed, or the model file is missing, and
  fallback_classical is True, the module returns its input unchanged and
  emits a warning. The pipeline then continues on the classical path.

  For DL-B fallback the pipeline must detect that the module was a no-op
  and still run the classical BNR / Demosaic / 2D NR chain. This is
  handled by omni_isp.py checking self._dl_active after calling execute().

Author: Omni-ISP contributors
"""

import os
import warnings
import numpy as np

from modules.dl_denoise.onnx_inference import (
    OnnxInferenceEngine,
    OnnxUnavailable,
    ort_available,
)
from modules.dl_denoise.model_zoo import auto_detect_model


class DLDenoise:
    """
    DL denoising / joint demosaic+denoise module.

    Parameters
    ----------
    img : np.ndarray
        DL-A mode : post-demosaic linear RGB, shape (H, W, 3), uint8 or float
        DL-B mode : pre-demosaic Bayer uint16, shape (H, W)
    platform    : dict  — platform config (used for models_dir)
    sensor_info : dict  — sensor metadata (bit_depth, range)
    parm_dl     : dict  — dl_denoise config section from configs.yml
    """

    def __init__(self, img, platform, sensor_info, parm_dl):
        self.img         = img
        self.platform    = platform
        self.sensor_info = sensor_info

        # Config
        self.enable            = parm_dl.get("is_enable", False)
        self.mode              = parm_dl.get("mode", "bayer_joint")
        self.model_path        = parm_dl.get("model_path", "")
        self.tile_size         = int(parm_dl.get("tile_size", 512))
        self.tile_overlap      = int(parm_dl.get("tile_overlap", 32))
        self.fallback_classical= parm_dl.get("fallback_classical", True)

        # Sensor range for Bayer normalisation
        self.bit_depth  = sensor_info.get("bit_depth", 12)
        self.white_level= sensor_info.get("range", (1 << self.bit_depth) - 1)

        # Resolve models directory: same folder as the config file, or "models/"
        raw_dir = platform.get("models_dir", "models")
        self.models_dir = raw_dir

        # Internal state — set by execute()
        self._dl_active = False   # True only if DL model actually ran

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dl_active(self) -> bool:
        """
        True after execute() if the DL model ran successfully.
        omni_isp.py checks this to decide whether to skip the classical
        BNR/Demosaic path (DL-B) or 2D NR (DL-A).
        """
        return self._dl_active

    def execute(self):
        """
        Run DL inference or pass through if disabled/unavailable.

        Returns
        -------
        np.ndarray
          DL-A: denoised linear RGB, shape (H, W, 3), same dtype as input
          DL-B: linear RGB, shape (H, W, 3), float32 in [0, 1]
          Fallback: original img unchanged
        """
        self._dl_active = False

        if not self.enable:
            return self.img

        # ------------------------------------------------------------------
        # Resolve model path
        # ------------------------------------------------------------------
        model_path = self._resolve_model_path()
        if model_path is None:
            return self._fallback(
                f"No ONNX model found for mode='{self.mode}' in '{self.models_dir}'. "
                "Run scripts/download_models.py to download pretrained models."
            )

        # ------------------------------------------------------------------
        # Load ONNX engine
        # ------------------------------------------------------------------
        try:
            engine = OnnxInferenceEngine(model_path)
        except OnnxUnavailable as e:
            return self._fallback(str(e))
        except Exception as e:
            return self._fallback(f"Failed to load ONNX model: {e}")

        # ------------------------------------------------------------------
        # Dispatch
        # ------------------------------------------------------------------
        if self.mode == "bayer_joint":
            return self._run_bayer_joint(engine)
        elif self.mode == "rgb_post":
            return self._run_rgb_post(engine)
        else:
            return self._fallback(f"Unknown dl_denoise mode: '{self.mode}'")

    # ------------------------------------------------------------------
    # DL-B: joint Bayer → RGB
    # ------------------------------------------------------------------

    def _run_bayer_joint(self, engine: OnnxInferenceEngine):
        """
        Input : uint16 Bayer (H, W)
        Output: float32 linear RGB (H, W, 3), values in [0, 1]
        """
        bayer = self.img  # (H, W) uint16

        # Normalise to [0, 1]
        norm = bayer.astype(np.float32) / float(self.white_level)
        norm = np.clip(norm, 0.0, 1.0)

        # Model expects (1, 1, H, W)
        inp = norm[None, None]   # (1, 1, H, W)

        try:
            out = engine.infer_tiled(
                inp[0],                 # (1, H, W) — tiler works on CHW
                tile_size=self.tile_size,
                overlap=self.tile_overlap,
            )
        except Exception as e:
            return self._fallback(f"ONNX inference failed: {e}")

        # out: (3, H, W) float32 in [0, 1]
        rgb = np.transpose(out, (1, 2, 0))   # (H, W, 3)
        rgb = np.clip(rgb, 0.0, 1.0)

        self._dl_active = True
        return rgb

    # ------------------------------------------------------------------
    # DL-A: post-demosaic RGB denoiser
    # ------------------------------------------------------------------

    def _run_rgb_post(self, engine: OnnxInferenceEngine):
        """
        Input : linear RGB (H, W, 3) — any uint8/uint16/float dtype
        Output: denoised linear RGB, same shape and dtype as input
        """
        img = self.img
        orig_dtype = img.dtype

        # Normalise to float32 [0, 1]
        if np.issubdtype(orig_dtype, np.integer):
            max_val = float(np.iinfo(orig_dtype).max)
            norm = img.astype(np.float32) / max_val
        else:
            norm = img.astype(np.float32)
        norm = np.clip(norm, 0.0, 1.0)

        # Model expects (1, 3, H, W)
        inp_chw = np.transpose(norm, (2, 0, 1))  # (3, H, W)

        try:
            out_chw = engine.infer_tiled(
                inp_chw,
                tile_size=self.tile_size,
                overlap=self.tile_overlap,
            )
        except Exception as e:
            return self._fallback(f"ONNX inference failed: {e}")

        # out_chw: (3, H, W)
        out_hwc = np.transpose(out_chw, (1, 2, 0))  # (H, W, 3)
        out_hwc = np.clip(out_hwc, 0.0, 1.0)

        # Restore original dtype
        if np.issubdtype(orig_dtype, np.integer):
            max_val = float(np.iinfo(orig_dtype).max)
            result = (out_hwc * max_val).round().astype(orig_dtype)
        else:
            result = out_hwc.astype(orig_dtype)

        self._dl_active = True
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_model_path(self):
        """
        Return a usable model path, or None.
        Priority: explicit config path → auto-detect from models_dir.
        """
        # 1. Explicit path in config
        if self.model_path:
            if os.path.isfile(self.model_path):
                return self.model_path
            # Treat as relative to models_dir
            candidate = os.path.join(self.models_dir, self.model_path)
            if os.path.isfile(candidate):
                return candidate

        # 2. Auto-detect: scan models_dir for a matching mode
        return auto_detect_model(self.mode, self.models_dir)

    def _fallback(self, reason: str):
        """Warn and return input unchanged (classical path continues)."""
        if not self.fallback_classical:
            raise RuntimeError(
                f"DLDenoise: {reason}\n"
                "Set fallback_classical: true to fall back to the classical path."
            )
        warnings.warn(
            f"DLDenoise: falling back to classical path. Reason: {reason}",
            RuntimeWarning,
            stacklevel=3,
        )
        self._dl_active = False
        return self.img
