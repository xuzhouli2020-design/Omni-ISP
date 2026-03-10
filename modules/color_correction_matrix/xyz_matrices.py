"""
File: xyz_matrices.py
Description: Standard CIE XYZ ↔ target colour-space transformation matrices.
             All matrices are for the D65 illuminant and assume linear light input.
             Used by the XYZ-intermediate CCM architecture (Phase 3.1).
Author: EdgeISP (10xEngineers fork)
------------------------------------------------------------
Conventions
-----------
- Matrices are 3×3 numpy float64.
- Applied as:  xyz_col_vec = M @ rgb_col_vec   (standard linear-algebra convention)
- The CCM class reshapes the image to (N, 3) and uses  out = in @ M.T  to match the
  existing imatest row-vector convention.
- XYZ values are normalised so that the D65 white point maps to (1, 1, 1) approximately.

Sources
-------
- sRGB / BT.709:   IEC 61966-2-1, ICC colour profiles
- Display P3:      DCI-P3 with D65 white point (Apple, SMPTE RP 431-2 derived)
- BT.2020:         ITU-R BT.2020 Table 2
- sRGB→XYZ used to derive the default camera_to_xyz placeholder from the existing CCM
"""

import numpy as np

# ---------------------------------------------------------------------------
# XYZ → target primaries
# ---------------------------------------------------------------------------

# XYZ → sRGB (BT.709, D65)  — IEC 61966-2-1
M_XYZ_TO_SRGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
], dtype=np.float64)

# XYZ → Display P3 (D65 white point)
M_XYZ_TO_P3 = np.array([
    [ 2.4934969, -0.9313837, -0.4027108],
    [-0.8294890,  1.7626641,  0.0236247],
    [ 0.0358458, -0.0761724,  0.9568845],
], dtype=np.float64)

# XYZ → BT.2020 (D65)  — ITU-R BT.2020
M_XYZ_TO_BT2020 = np.array([
    [ 1.7166512, -0.3556708, -0.2533663],
    [-0.6666844,  1.6164812,  0.0157685],
    [ 0.0176399, -0.0427706,  0.9421031],
], dtype=np.float64)

# XYZ → XYZ identity (passthrough — useful for debugging / scientific output)
M_XYZ_TO_XYZ = np.eye(3, dtype=np.float64)

# sRGB → XYZ (D65)  — inverse of M_XYZ_TO_SRGB, used to derive default placeholder
M_SRGB_TO_XYZ = np.array([
    [0.4124564,  0.3575761,  0.1804375],
    [0.2126729,  0.7151522,  0.0721750],
    [0.0193339,  0.1191920,  0.9503041],
], dtype=np.float64)

# Registry: config key → matrix
_TARGET_REGISTRY = {
    "srgb":       M_XYZ_TO_SRGB,
    "display_p3": M_XYZ_TO_P3,
    "bt2020":     M_XYZ_TO_BT2020,
    "xyz":        M_XYZ_TO_XYZ,       # passthrough: keeps CIE XYZ output
}


def get_xyz_to_target(target: str) -> np.ndarray:
    """
    Return the 3×3 XYZ→target matrix for the given colour-space name.

    Parameters
    ----------
    target : str
        One of "srgb", "display_p3", "bt2020", "xyz".

    Returns
    -------
    np.ndarray  shape (3, 3), float64
    """
    key = target.lower().replace("-", "_").replace(" ", "_")
    if key not in _TARGET_REGISTRY:
        raise ValueError(
            f"Unknown CCM target '{target}'. "
            f"Valid targets: {list(_TARGET_REGISTRY)}"
        )
    return _TARGET_REGISTRY[key].copy()


def derive_camera_to_xyz(direct_ccm: np.ndarray) -> np.ndarray:
    """
    Derive an approximate camera→XYZ matrix from an existing direct CCM
    (camera→sRGB) by composing with the sRGB→XYZ matrix.

    This is a convenience helper for bootstrapping the XYZ mode when a proper
    colorimetric camera→XYZ calibration has not yet been performed.

    Parameters
    ----------
    direct_ccm : np.ndarray  shape (3, 3)
        Existing camera→sRGB CCM in row-vector convention (imatest).

    Returns
    -------
    np.ndarray  shape (3, 3), float64  — approximate camera→XYZ matrix
    """
    return (direct_ccm.astype(np.float64) @ M_SRGB_TO_XYZ)
