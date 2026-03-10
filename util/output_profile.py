"""
File: output_profile.py
Description: Named output profile system — bundles CCM target primaries and
             Gamma EOTF together so they are always kept in sync.

             Selecting a profile overrides the per-module config keys:
               • color_correction_matrix: mode, target
               • gamma_correction:        eotf

             Use profile="custom" (or omit output: section) to control each
             module individually, preserving full backward compatibility.

Phase 3.3.
"""

# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------
# Each entry: (ccm_mode, ccm_target, gamma_eotf)
#   ccm_mode    — "direct" (config matrix) or "xyz" (CIE XYZ intermediate)
#   ccm_target  — XYZ→target key, used only when ccm_mode="xyz"
#   gamma_eotf  — EOTF selector passed to gamma_correction module

_PROFILE_REGISTRY = {
    # Standard SDR — default web/photo output
    "srgb": ("xyz", "srgb", "srgb"),
    # Broadcast SDR video (linear segment is slightly different from sRGB)
    "rec709": ("xyz", "srgb", "rec709"),  # BT.709 uses same primaries as sRGB
    # Wide-colour SDR — modern phones, Mac, iPad (Display P3 / D65)
    "display_p3": ("xyz", "display_p3", "srgb"),  # P3 uses sRGB EOTF per Apple spec
    # HDR10 — UHDTV / streaming HDR (ST.2084 PQ, BT.2020 primaries)
    "hdr10": ("xyz", "bt2020", "pq"),
    # HDR broadcast — HLG (ARIB STD-B67, BT.2020 primaries)
    "hlg": ("xyz", "bt2020", "hlg"),
    # Linear light — for ML pipelines, scientific output, further processing
    "linear": ("xyz", "srgb", "linear"),
}


def get_profile(name: str):
    """
    Return (ccm_mode, ccm_target, gamma_eotf) for the named profile.

    Parameters
    ----------
    name : str
        Profile name, case-insensitive. One of:
        "srgb", "rec709", "display_p3", "hdr10", "hlg", "linear".
        Pass "custom" or None to use per-module config (no override).

    Returns
    -------
    tuple (str, str, str) or None
        (ccm_mode, ccm_target, gamma_eotf), or None if no override.

    Raises
    ------
    ValueError if name is not recognised and not "custom".
    """
    if name is None or str(name).lower() in ("custom", ""):
        return None
    key = str(name).lower().replace("-", "_")
    if key not in _PROFILE_REGISTRY:
        valid = list(_PROFILE_REGISTRY.keys()) + ["custom"]
        raise ValueError(
            f"Unknown output profile '{name}'. Valid profiles: {valid}"
        )
    return _PROFILE_REGISTRY[key]


def apply_profile_to_params(output_cfg: dict,
                             parm_ccm: dict,
                             parm_gmc: dict) -> None:
    """
    Mutate parm_ccm and parm_gmc in-place to apply the output profile
    specified in output_cfg (the 'output:' section of configs.yml).

    If output_cfg is absent, empty, or profile="custom", this is a no-op.

    Parameters
    ----------
    output_cfg : dict   Contents of the 'output:' config section.
    parm_ccm   : dict   color_correction_matrix section (mutated in-place).
    parm_gmc   : dict   gamma_correction section (mutated in-place).
    """
    if not output_cfg:
        return
    profile_name = output_cfg.get("profile", "custom")
    result = get_profile(profile_name)
    if result is None:
        return   # "custom" — leave individual module config unchanged

    ccm_mode, ccm_target, gamma_eotf = result
    parm_ccm["mode"]   = ccm_mode
    parm_ccm["target"] = ccm_target
    parm_gmc["eotf"]   = gamma_eotf
    print(f"[OutputProfile] '{profile_name}' → "
          f"CCM mode={ccm_mode} target={ccm_target}, "
          f"Gamma eotf={gamma_eotf}")
