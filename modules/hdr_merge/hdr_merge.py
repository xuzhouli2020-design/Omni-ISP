"""
modules/hdr_merge/hdr_merge.py
================================
HDRMerge — multi-exposure Mertens fusion and basic Debevec HDR reconstruction.

Pipeline position
-----------------
HDR merge operates on **linear RGB images** (post-CCM, pre-tone-mapping) from
multiple exposure brackets.  In the Omni-ISP workflow this means running the
pipeline through CCM on each bracketed frame, then merging the resulting
linear RGB stacks, and continuing with a single merged frame from tone mapping
onwards.

  [RAW brackets] → [ISP through CCM] × N → HDRMerge → HDRToneMapping → Gamma → ...

The class can also operate on post-gamma 8-bit / 16-bit images when used
in Mertens mode (which is camera-response-curve agnostic).

Modes
-----
  "mertens" — Exposure fusion (Mertens et al. 2007)
      Combines N LDR images into a visually pleasing HDR-like result
      without requiring absolute exposure values.  Does not produce an
      HDR image in the strict radiometric sense, but works well for
      display output.  Uses per-pixel quality weights: contrast,
      saturation, and well-exposedness.

  "debevec" — Debevec & Malik 1997 HDR radiance reconstruction
      Recovers scene radiance from N differently-exposed images.
      Requires exposure times (EV or shutter speeds) for each frame.
      Returns a linear HDR image in scene-relative units.

Config
------
  hdr_merge:
    is_enable: false
    mode: "mertens"            # "mertens" | "debevec"
    # ---- Mertens weights ----
    weight_contrast:    1.0    # Laplacian contrast quality weight
    weight_saturation:  1.0    # per-pixel saturation quality weight
    weight_exposedness: 1.0    # well-exposedness (Gaussian centred at 0.5) weight
    sigma_exposedness:  0.2    # Gaussian σ for well-exposedness measure
    # ---- Debevec ----
    lambda_smooth:      20.0   # smoothness weight for response curve fitting
    response_samples:   256    # number of pixels sampled for curve estimation

References
----------
  Mertens et al. 2007 — "Exposure Fusion"
  Debevec & Malik 1997 — "Recovering High Dynamic Range Radiance Maps from Photographs"

Author: Omni-ISP contributors
"""

import numpy as np
import warnings


class HDRMerge:
    """
    Multi-exposure HDR merge.

    Parameters
    ----------
    images       : list of np.ndarray
                   Each array is (H, W, 3), float32 [0, 1] linear RGB.
                   Must all have the same spatial dimensions.
    exposure_evs : list of float or None
                   Relative EV values for each frame (e.g. [-2, 0, +2]).
                   Required for Debevec mode; ignored in Mertens mode.
                   Positive EV = brighter (longer exposure).
    parm_hdrm    : dict — hdr_merge config section

    Usage
    -----
    merge = HDRMerge(images, exposure_evs, parm_hdrm)
    merged_rgb = merge.execute()   # returns (H, W, 3) float32
    """

    def __init__(self, images, exposure_evs=None, parm_hdrm=None):
        if parm_hdrm is None:
            parm_hdrm = {}

        self.images       = [img.astype(np.float32) for img in images]
        self.exposure_evs = exposure_evs

        self.enable               = parm_hdrm.get("is_enable", False)
        self.mode                 = parm_hdrm.get("mode", "mertens")
        self.w_contrast           = float(parm_hdrm.get("weight_contrast",    1.0))
        self.w_saturation         = float(parm_hdrm.get("weight_saturation",  1.0))
        self.w_exposedness        = float(parm_hdrm.get("weight_exposedness", 1.0))
        self.sigma_exposedness    = float(parm_hdrm.get("sigma_exposedness",  0.2))
        self.lambda_smooth        = float(parm_hdrm.get("lambda_smooth",     20.0))
        self.response_samples     = int(parm_hdrm.get("response_samples",   256))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self) -> np.ndarray:
        """
        Merge the exposure stack.

        Returns
        -------
        np.ndarray  (H, W, 3) float32
          Mertens: blended LDR-like result in [0, 1].
          Debevec:  HDR radiance map, normalised to [0, 1] for downstream.
          Single frame or disabled: first frame unchanged.
        """
        if not self.enable or len(self.images) < 2:
            return self.images[0]

        # Validate shape consistency
        H, W = self.images[0].shape[:2]
        for i, img in enumerate(self.images[1:], 1):
            if img.shape[:2] != (H, W):
                raise ValueError(
                    f"HDRMerge: image {i} shape {img.shape[:2]} != "
                    f"reference {(H, W)}"
                )

        mode = self.mode.lower()
        if mode == "mertens":
            return self._mertens_fusion()
        elif mode == "debevec":
            return self._debevec_merge()
        else:
            warnings.warn(
                f"HDRMerge: unknown mode '{self.mode}'. Returning first frame.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self.images[0]

    # ------------------------------------------------------------------
    # Mertens Exposure Fusion (2007)
    # ------------------------------------------------------------------

    def _mertens_fusion(self) -> np.ndarray:
        """
        Exposure fusion without absolute exposure values.

        For each image compute three quality measures per pixel:
          Contrast(p)    = |Laplacian(greyscale(p))|
          Saturation(p)  = std(R, G, B) across channels
          Exposedness(p) = prod_c exp(-(I_c - 0.5)² / (2σ²))

        Quality weight: W(p) = Contrast^w_c · Saturation^w_s · Exposedness^w_e
        Normalise across the stack: W_i(p) = W_i(p) / Σ_j W_j(p)
        Output: Σ_i W_i(p) · image_i(p)

        Returns float32 (H, W, 3) in [0, 1].
        """
        N = len(self.images)
        H, W = self.images[0].shape[:2]

        weights = np.zeros((N, H, W), dtype=np.float32)

        for i, img in enumerate(self.images):
            weights[i] = self._quality_weight(img)

        # Add small uniform floor before normalising.
        # This ensures that for completely flat / uniform frames (zero contrast,
        # zero saturation) the result degrades gracefully to a weighted average
        # rather than collapsing to zero due to float32 underflow.
        weights += 1e-6

        # Normalise so weights sum to 1 across the stack at each pixel
        weight_sum = weights.sum(axis=0, keepdims=True)   # (1, H, W)
        weights /= weight_sum                              # (N, H, W)

        # Weighted sum of images
        merged = np.zeros((H, W, 3), dtype=np.float32)
        for i, img in enumerate(self.images):
            merged += weights[i, :, :, None] * img

        return np.clip(merged, 0.0, 1.0)

    def _quality_weight(self, img: np.ndarray) -> np.ndarray:
        """Per-pixel quality weight for one exposure (H, W)."""
        w = np.ones(img.shape[:2], dtype=np.float32)

        # --- Contrast ---
        if self.w_contrast != 0:
            gray = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
            lap  = np.abs(self._laplacian(gray))
            w   *= (lap + 1e-8) ** self.w_contrast

        # --- Saturation ---
        if self.w_saturation != 0:
            sat  = img.std(axis=2)             # per-pixel std across R,G,B
            w   *= (sat + 1e-8) ** self.w_saturation

        # --- Well-exposedness ---
        if self.w_exposedness != 0:
            sigma2 = 2.0 * self.sigma_exposedness ** 2
            exp_w  = np.exp(-((img - 0.5) ** 2) / sigma2)   # (H, W, 3)
            exp_w  = exp_w.prod(axis=2)                      # product across channels
            w     *= (exp_w + 1e-8) ** self.w_exposedness

        return w

    @staticmethod
    def _laplacian(img_gray: np.ndarray) -> np.ndarray:
        """Simple 3×3 discrete Laplacian."""
        from numpy.lib.stride_tricks import as_strided
        # Pad with edge replication, then convolve
        padded = np.pad(img_gray, 1, mode="edge")
        kernel_sum = (
              padded[0:-2, 1:-1]   # top
            + padded[2:,   1:-1]   # bottom
            + padded[1:-1, 0:-2]   # left
            + padded[1:-1, 2:]     # right
            - 4 * padded[1:-1, 1:-1]   # centre
        )
        return kernel_sum

    # ------------------------------------------------------------------
    # Debevec & Malik 1997 HDR Reconstruction
    # ------------------------------------------------------------------

    def _debevec_merge(self) -> np.ndarray:
        """
        Recover HDR radiance map using Debevec & Malik 1997.

        Steps:
          1. Sample pixel values across the exposure stack at random locations.
          2. Solve for the camera response function g(z) via least-squares.
          3. Use g to convert pixel values back to linear log-radiance.
          4. Weight by the hat weighting function and merge.

        Returns float32 (H, W, 3), normalised to [0, 1] by max value.
        """
        if self.exposure_evs is None or len(self.exposure_evs) != len(self.images):
            warnings.warn(
                "HDRMerge (Debevec): exposure_evs not provided or length mismatch. "
                "Falling back to Mertens fusion.",
                RuntimeWarning,
                stacklevel=3,
            )
            return self._mertens_fusion()

        N = len(self.images)
        H, W = self.images[0].shape[:2]

        # Exposure times from EV offsets (relative, reference = EV 0)
        # t_i = 2^EV_i (relative units)
        log_t = np.array([ev * np.log(2.0) for ev in self.exposure_evs],
                         dtype=np.float32)

        # Process each channel independently
        merged_hdr = np.zeros((H, W, 3), dtype=np.float32)

        for c in range(3):
            channel_stack = np.stack(
                [np.clip(img[..., c] * 255.0, 0, 255).astype(np.uint8)
                 for img in self.images],
                axis=0,
            )  # (N, H, W) uint8
            g, _ = self._recover_response(channel_stack, log_t)
            merged_hdr[..., c] = self._reconstruct_radiance(
                channel_stack, g, log_t
            )

        # Normalise: map median-exposed pixel to ~0.18 (middle grey)
        # Use the merged image of the reference frame (EV=0 or closest)
        ref_idx = int(np.argmin(np.abs(np.array(self.exposure_evs))))
        ref_lum = (
            0.2126 * merged_hdr[..., 0]
            + 0.7152 * merged_hdr[..., 1]
            + 0.0722 * merged_hdr[..., 2]
        )
        scene_median = float(np.median(np.exp(ref_lum[ref_lum > np.log(1e-8)])))
        scale = 0.18 / max(scene_median, 1e-8)
        merged_hdr = np.exp(merged_hdr) * scale

        # Clip to reasonable range and normalise to [0, 1]
        p_max = float(np.percentile(merged_hdr, 99.9))
        if p_max > 0:
            merged_hdr = merged_hdr / p_max

        return np.clip(merged_hdr, 0.0, 1.0)

    def _hat_weight(self, z: np.ndarray, z_min: int = 0, z_max: int = 255) -> np.ndarray:
        """Hat weighting function (Debevec 1997): full weight in mid-range, zero at clipping."""
        z = z.astype(np.float32)
        mid = (z_min + z_max) / 2.0
        return np.where(z <= mid, z - z_min, z_max - z).astype(np.float32) + 1.0

    def _recover_response(self, stack: np.ndarray, log_t: np.ndarray):
        """
        Solve for camera response function g(z) as a 256-element array.
        stack: (N, H, W) uint8
        log_t: (N,) log exposure times
        Returns (g, lE) where g is shape (256,) float32.
        """
        N, H, W = stack.shape
        n_samples = min(self.response_samples, H * W)

        rng = np.random.default_rng(seed=42)
        pixel_idx = rng.choice(H * W, size=n_samples, replace=False)
        py, px = np.unravel_index(pixel_idx, (H, W))

        Z = stack[:, py, px].T   # (n_samples, N) integer pixel values

        n_eq   = n_samples * N
        n_unkn = 256 + n_samples
        A = np.zeros((n_eq + 255, n_unkn), dtype=np.float32)
        b = np.zeros((n_eq + 255,),        dtype=np.float32)

        k = 0
        for i in range(n_samples):
            for j in range(N):
                z_ij = int(Z[i, j])
                wij  = self._hat_weight(np.array([z_ij]))[0]
                A[k, z_ij]          =  wij
                A[k, 256 + i]       = -wij
                b[k]                =  wij * log_t[j]
                k += 1

        # Smoothness constraint on g
        lam = self.lambda_smooth
        for z in range(1, 255):
            w_z = self._hat_weight(np.array([z]))[0]
            A[k, z - 1] =  lam * w_z
            A[k, z]     = -2 * lam * w_z
            A[k, z + 1] =  lam * w_z
            k += 1

        # Fix middle grey (g(128) = 0 in log domain)
        mid_eq = n_eq + 127
        A[mid_eq, 128] = 1.0

        # Solve via least-squares
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        g  = x[:256].astype(np.float32)
        lE = x[256:].astype(np.float32)
        return g, lE

    def _reconstruct_radiance(
        self,
        stack: np.ndarray,
        g: np.ndarray,
        log_t: np.ndarray,
    ) -> np.ndarray:
        """
        Reconstruct log-radiance for each pixel using recovered g.
        Returns (H, W) float32 log-radiance.
        """
        N, H, W = stack.shape
        log_E   = np.zeros((H, W), dtype=np.float32)
        w_total = np.zeros((H, W), dtype=np.float32)

        for j in range(N):
            z_j  = stack[j].astype(np.int32)                   # (H, W)
            w_j  = self._hat_weight(stack[j]).astype(np.float32)
            log_E += w_j * (g[z_j] - log_t[j])
            w_total += w_j

        w_total = np.maximum(w_total, 1e-8)
        return log_E / w_total
