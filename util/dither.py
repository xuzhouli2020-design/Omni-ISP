"""
File: dither.py
Description: Blue noise and TPDF dithering for 8-bit pipeline output.

             Dithering converts high-precision float values to 8-bit integers
             while spatially encoding sub-LSB information — preventing banding
             artefacts in smooth gradients (sky, skin tone, etc.).

             Applied AFTER gamma (in perceptually-uniform [0,1] space),
             immediately before uint8 quantisation.

Modes (output.dither config key):
  "none"        — plain rounding (original behaviour)
  "tpdf"        — Triangular Probability Density Function: sum of two
                  independent uniform random noise samples. Simple, unbiased,
                  but white-noise texture.
  "blue_noise"  — Void-and-cluster spatial threshold mask. Noise energy is
                  concentrated at high spatial frequencies → film-grain look,
                  subjectively invisible at normal viewing distance.

References:
  Ulichney, R. (1993). Void-and-cluster method for dither array generation.
  SPIE Human Vision, Visual Processing, and Digital Display IV.

Phase 3.4.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Blue noise mask generation (void-and-cluster, Ulichney 1993)
# ---------------------------------------------------------------------------

def _gaussian_energy(binary: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Approximate Gaussian-weighted energy map via separable 1-D box filter
    (σ ≈ 1.5 pixels, 3 passes of 3-wide box ≈ Gaussian).  Pure NumPy.
    Returns a float32 energy map — high values where 1-bits cluster.
    """
    # Toroidal padding: reflect wraps to simulate tileability
    f = binary.astype(np.float32)
    H, W = f.shape
    # Use cumulative sum for fast box filtering (toroidal via np.roll)
    r = max(1, int(round(sigma * 2)))   # box half-width ≈ 2σ

    def _box1d_h(a, r):
        # horizontal box filter with periodic boundary (tile seam-free)
        out = np.zeros_like(a)
        for shift in range(-r, r + 1):
            out += np.roll(a, shift, axis=1)
        return out / (2 * r + 1)

    def _box1d_v(a, r):
        out = np.zeros_like(a)
        for shift in range(-r, r + 1):
            out += np.roll(a, shift, axis=0)
        return out / (2 * r + 1)

    # Separable: horizontal then vertical (3 passes each for better Gauss approx)
    e = f
    for _ in range(3):
        e = _box1d_h(e, r)
    for _ in range(3):
        e = _box1d_v(e, r)
    return e


def generate_void_and_cluster(size: int = 128,
                               initial_ones: float = 0.1,
                               rng_seed: int = 42) -> np.ndarray:
    """
    Generate a blue noise threshold mask using the void-and-cluster algorithm.

    Parameters
    ----------
    size        : int — width and height of the square tileable mask.
    initial_ones: float — fraction of initial 1-bits (typically 0.1).
    rng_seed    : int — RNG seed for reproducibility.

    Returns
    -------
    np.ndarray  shape (size, size), float32, values in [0, 1)
        Threshold mask: pixel (i,j) rounds up if fractional_part > mask[i,j].
    """
    rng = np.random.default_rng(rng_seed)
    N = size * size
    n_ones = max(1, int(round(N * initial_ones)))

    # -- Phase 1: build initial binary pattern with isolated 1-bits --
    pattern = np.zeros(N, dtype=np.uint8)
    pattern[:n_ones] = 1
    rng.shuffle(pattern)
    pattern = pattern.reshape(size, size)

    # Iteratively swap tightest cluster with largest void
    # (stop when pattern is maximally spread — fixed number of passes for speed)
    n_iter = N // 2   # enough iterations for a 128×128 mask
    for _ in range(n_iter):
        energy = _gaussian_energy(pattern)
        # Tightest cluster: highest energy among 1-pixels
        cluster_mask = (pattern == 1)
        if not cluster_mask.any():
            break
        cluster_energy = np.where(cluster_mask, energy, -np.inf)
        ci = np.unravel_index(np.argmax(cluster_energy), (size, size))
        # Largest void: lowest energy among 0-pixels
        void_mask = (pattern == 0)
        if not void_mask.any():
            break
        void_energy = np.where(void_mask, energy, np.inf)
        vi = np.unravel_index(np.argmin(void_energy), (size, size))
        if ci == vi:
            break
        pattern[ci] = 0
        pattern[vi] = 1

    # -- Phase 2: build rank array from the converged pattern --
    # Assign rank 0..n_ones-1 by progressively removing tightest clusters
    rank = np.full((size, size), -1, dtype=np.int32)
    work = pattern.copy()
    for rank_val in range(n_ones - 1, -1, -1):
        energy = _gaussian_energy(work)
        ones_mask = (work == 1)
        if not ones_mask.any():
            break
        e_masked = np.where(ones_mask, energy, -np.inf)
        idx = np.unravel_index(np.argmax(e_masked), (size, size))
        rank[idx] = rank_val
        work[idx] = 0

    # -- Phase 3: fill remaining ranks by progressively adding to largest voids --
    work = pattern.copy()
    for rank_val in range(n_ones, N):
        energy = _gaussian_energy(work)
        zeros_mask = (work == 0)
        if not zeros_mask.any():
            break
        e_masked = np.where(zeros_mask, energy, np.inf)
        idx = np.unravel_index(np.argmin(e_masked), (size, size))
        rank[idx] = rank_val
        work[idx] = 1

    # Normalise ranks to [0, 1)
    mask = rank.astype(np.float32) / float(N)
    return mask


# ---------------------------------------------------------------------------
# Cached mask (generated once on first use)
# ---------------------------------------------------------------------------

_BLUE_NOISE_MASK: np.ndarray | None = None
_BLUE_NOISE_SIZE: int = 64   # 64×64 for reasonable generation speed; tiles freely


def _get_blue_noise_mask(size: int = _BLUE_NOISE_SIZE) -> np.ndarray:
    """Return (and cache) the blue noise mask."""
    global _BLUE_NOISE_MASK
    if _BLUE_NOISE_MASK is None or _BLUE_NOISE_MASK.shape[0] != size:
        print(f"[Dither] Generating {size}×{size} blue noise mask "
              f"(void-and-cluster) …", flush=True)
        _BLUE_NOISE_MASK = generate_void_and_cluster(size)
    return _BLUE_NOISE_MASK


# ---------------------------------------------------------------------------
# Runtime dithering functions
# ---------------------------------------------------------------------------

def _tile_mask(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    """Tile a 2-D mask to cover (H, W)."""
    mH, mW = mask.shape
    t = np.tile(mask, (H // mH + 1, W // mW + 1))
    return t[:H, :W]


def apply_dither(img: np.ndarray,
                 mode: str = "blue_noise",
                 out_bits: int = 8) -> np.ndarray:
    """
    Dither `img` before quantising to `out_bits` output.

    Parameters
    ----------
    img      : np.ndarray  shape (..., H, W) or (H, W, C), values in [0, 1]
                           float32/float64.  Will be converted if needed.
    mode     : str         "none" | "tpdf" | "blue_noise"
    out_bits : int         Target bit depth (default 8).

    Returns
    -------
    np.ndarray  shape same as img, dtype uint8 (or uint16 for out_bits > 8),
        values clipped to [0, 2**out_bits - 1].
    """
    maxval = float(2**out_bits - 1)
    img_f = img.astype(np.float32)

    if mode == "none" or out_bits > 8:
        # Plain round — no dithering for HDR output
        result = np.clip(np.round(img_f * maxval), 0, maxval)
        return result.astype(np.uint8 if out_bits <= 8 else np.uint16)

    H = img_f.shape[0]
    W = img_f.shape[1]

    if mode == "tpdf":
        # Triangular PDF: add sum of two uniform ±0.5/maxval noises
        # Equivalent to a single uniform over ±1/maxval, but with a triangular
        # distribution that ensures statistical mean is preserved.
        noise = (np.random.random((H, W)).astype(np.float32)
                 + np.random.random((H, W)).astype(np.float32)
                 - 1.0) / maxval       # range: (-1/maxval, +1/maxval)
        if img_f.ndim == 3:
            noise = noise[:, :, None]  # broadcast over channels
        result = np.clip(np.round((img_f + noise) * maxval), 0, maxval)

    elif mode == "blue_noise":
        # Threshold dither: add (mask - 0.5) / maxval
        # round(v*maxval + mask - 0.5) = floor(v*maxval + mask)
        mask = _get_blue_noise_mask(_BLUE_NOISE_SIZE)
        tiled = _tile_mask(mask, H, W).astype(np.float32)
        # Threshold dither: floor(v*maxval + mask) where mask ∈ [0, 1)
        if img_f.ndim == 3:
            result = np.clip(np.floor(img_f * maxval + tiled[:, :, None]),
                             0, maxval)
        else:
            result = np.clip(np.floor(img_f * maxval + tiled),
                             0, maxval)
    else:
        raise ValueError(f"Unknown dither mode '{mode}'. "
                         f"Valid: 'none', 'tpdf', 'blue_noise'")

    return result.astype(np.uint8 if out_bits <= 8 else np.uint16)


def encode_8bit(img: np.ndarray,
                dither_mode: str = "blue_noise") -> np.ndarray:
    """
    Convert a pipeline image to uint8, applying dithering.

    Accepts:
      - float32/float64 in [0, 1]   → scaled to 0–255
      - uint8 already               → dither in float, re-quantise
      - uint16 in range [0, 65535]  → normalise, dither, 8-bit out

    Parameters
    ----------
    img          : np.ndarray  Pipeline output image.
    dither_mode  : str         "none" | "tpdf" | "blue_noise"

    Returns
    -------
    np.ndarray  dtype uint8, values [0, 255].
    """
    if img.dtype == np.uint8:
        if dither_mode == "none":
            return img
        # Re-process as float
        img_f = img.astype(np.float32) / 255.0
    elif np.issubdtype(img.dtype, np.integer):
        # uint16 or other integer — normalise by max representable value
        imax = float(np.iinfo(img.dtype).max)
        img_f = img.astype(np.float32) / imax
    else:
        # Already float — assume [0, 1]
        img_f = img.astype(np.float32)
        if img_f.max() > 1.0:
            # Might be float in [0, 255] range
            img_f = img_f / 255.0

    return apply_dither(img_f, mode=dither_mode, out_bits=8)
