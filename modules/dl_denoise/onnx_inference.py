"""
modules/dl_denoise/onnx_inference.py
=====================================
Generic ONNX Runtime inference engine for Omni-ISP.

Provides:
  - OnnxInferenceEngine  : loads a model, runs forward pass on arbitrary tensors
  - infer_tiled          : tile-and-stitch inference for large images with
                           linear-blend overlap so tile boundaries are invisible

No hard dependency on onnxruntime — if it is not installed the engine raises
OnnxUnavailable and the pipeline falls back to the classical path via
dl_denoise.py's graceful-fallback logic.

Author: Omni-ISP contributors
"""

import numpy as np

# ---------------------------------------------------------------------------
# Optional import — onnxruntime is a runtime dep, not a hard requirement
# ---------------------------------------------------------------------------
try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False


class OnnxUnavailable(RuntimeError):
    """Raised when onnxruntime is not installed."""
    pass


class OnnxInferenceEngine:
    """
    Thin wrapper around an ONNX Runtime InferenceSession.

    Parameters
    ----------
    model_path : str
        Path to the .onnx model file.
    providers : list[str] | None
        ONNX Runtime execution providers in priority order.
        Defaults to ["CPUExecutionProvider"].
        Pass ["CUDAExecutionProvider", "CPUExecutionProvider"] for GPU.

    Raises
    ------
    OnnxUnavailable
        If onnxruntime is not installed.
    FileNotFoundError
        If model_path does not exist.
    """

    def __init__(self, model_path: str, providers=None):
        if not _ORT_AVAILABLE:
            raise OnnxUnavailable(
                "onnxruntime is not installed. "
                "Run: pip install onnxruntime  (CPU) "
                "or:  pip install onnxruntime-gpu  (GPU/CUDA)"
            )

        import os
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}\n"
                "Run scripts/download_models.py to download pretrained models."
            )

        if providers is None:
            providers = ["CPUExecutionProvider"]

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_opts.inter_op_num_threads = 1   # safe default for embedded
        sess_opts.intra_op_num_threads = 4

        self._session = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=providers
        )

        # Cache input / output metadata
        inputs  = self._session.get_inputs()
        outputs = self._session.get_outputs()

        self.input_name   = inputs[0].name
        self.output_name  = outputs[0].name
        self.input_shape  = inputs[0].shape   # list, may contain None for dynamic dims
        self.output_shape = outputs[0].shape

    # ------------------------------------------------------------------
    # Direct inference
    # ------------------------------------------------------------------

    def infer(self, x: np.ndarray) -> np.ndarray:
        """
        Run a single forward pass.

        Parameters
        ----------
        x : np.ndarray, shape (1, C, H, W), dtype float32

        Returns
        -------
        np.ndarray, shape (1, C', H', W'), dtype float32
        """
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        result = self._session.run([self.output_name], {self.input_name: x})
        return result[0]

    # ------------------------------------------------------------------
    # Tiled inference for full-resolution frames
    # ------------------------------------------------------------------

    def infer_tiled(
        self,
        image: np.ndarray,
        tile_size: int = 512,
        overlap: int   = 32,
    ) -> np.ndarray:
        """
        Tile a large CHW image, run inference on each tile, and stitch with
        linear-blend overlap to hide tile boundaries.

        Parameters
        ----------
        image : np.ndarray
            Shape (C, H, W), dtype float32, values in [0, 1].
        tile_size : int
            Spatial size of each square tile (before overlap extension).
        overlap : int
            Pixels of overlap on each side. Must be < tile_size // 2.

        Returns
        -------
        np.ndarray
            Stitched output, shape (C_out, H, W), dtype float32.
            C_out may differ from C (e.g. 1-channel Bayer → 3-channel RGB).
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        C_in, H, W = image.shape

        # --- determine output channels from the model's declared output ---
        # output_shape is e.g. [1, 3, None, None] or [1, 3, 512, 512]
        C_out = self.output_shape[1] if (
            len(self.output_shape) >= 2
            and isinstance(self.output_shape[1], int)
            and self.output_shape[1] > 0
        ) else C_in

        output     = np.zeros((C_out, H, W), dtype=np.float32)
        weight_map = np.zeros((1,     H, W), dtype=np.float32)

        # Build 1D blend ramp: flat in the centre, linear fade at edges
        ramp = _blend_ramp(tile_size + 2 * overlap, overlap)  # length = tile+2*ov
        blend_2d = ramp[:, None] * ramp[None, :]               # (T+2ov, T+2ov)

        step = tile_size  # stride between tile origins

        ys = list(range(0, H, step))
        xs = list(range(0, W, step))

        for y0 in ys:
            for x0 in xs:
                # Extended tile boundaries (with overlap, clamped to image)
                y_ext0 = max(0, y0 - overlap)
                x_ext0 = max(0, x0 - overlap)
                y_ext1 = min(H, y0 + step + overlap)
                x_ext1 = min(W, x0 + step + overlap)

                patch = image[:, y_ext0:y_ext1, x_ext0:x_ext1]  # (C, ph, pw)
                ph, pw = patch.shape[1], patch.shape[2]

                # Pad to expected size if near right/bottom edge
                pad_b = max(0, (tile_size + 2 * overlap) - ph)
                pad_r = max(0, (tile_size + 2 * overlap) - pw)
                if pad_b > 0 or pad_r > 0:
                    patch = np.pad(
                        patch,
                        ((0, 0), (0, pad_b), (0, pad_r)),
                        mode="reflect",
                    )

                # Run model
                inp   = patch[None]                     # (1, C, ph+pad, pw+pad)
                out   = self.infer(inp)[0]              # (C_out, ph+pad, pw+pad)
                out   = out[:, :ph, :pw]                # trim padding

                # Blend weight for this tile extent
                bh = blend_2d[:ph, :pw].copy()

                output    [:, y_ext0:y_ext1, x_ext0:x_ext1] += out  * bh
                weight_map[0, y_ext0:y_ext1, x_ext0:x_ext1] += bh

        # Normalise by accumulated blend weights
        weight_map = np.maximum(weight_map, 1e-8)
        output /= weight_map

        return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blend_ramp(size: int, margin: int) -> np.ndarray:
    """
    1D blend ramp of length `size`.
    Values ramp from 0→1 over the first `margin` pixels,
    stay at 1 in the middle, and ramp from 1→0 over the last `margin` pixels.
    """
    ramp = np.ones(size, dtype=np.float32)
    if margin > 0:
        fade = np.linspace(0.0, 1.0, margin, dtype=np.float32)
        ramp[:margin]        = fade
        ramp[size - margin:] = fade[::-1]
    return ramp


def ort_available() -> bool:
    """Return True if onnxruntime is importable."""
    return _ORT_AVAILABLE
