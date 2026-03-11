"""
modules/dl_denoise/model_zoo.py
================================
Registry of known pretrained models: names, expected file locations,
input/output specifications, and download instructions.

The actual .onnx files are NOT shipped in the repo (they are gitignored).
Run  scripts/download_models.py  to download and convert them once.

Author: Omni-ISP contributors
"""

import os

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry describes a model that can be used with the DLDenoise module.
#
# Keys:
#   filename     : expected .onnx filename under models/
#   mode         : "bayer_joint" (DL-B) or "rgb_post" (DL-A)
#   input_desc   : human-readable input tensor description
#   output_desc  : human-readable output tensor description
#   source       : where to obtain the weights (paper / repo)
#   notes        : usage notes
# ---------------------------------------------------------------------------

KNOWN_MODELS = {
    # ------------------------------------------------------------------
    # DL-B: joint Bayer → RGB
    # ------------------------------------------------------------------
    "bjdd": {
        "filename":    "bjdd_bayer_joint.onnx",
        "mode":        "bayer_joint",
        "input_desc":  "(1, 1, H, W) float32 [0,1] single-channel Bayer mosaic",
        "output_desc": "(1, 3, H, W) float32 [0,1] full-resolution RGB",
        "source":      "https://github.com/sharif-apu/BJDD_CVPR21",
        "notes": (
            "Beyond Joint Demosaicking and Denoising (CVPRW 2021). "
            "Pretrained on Bayer CFA. Download PyTorch weights and convert "
            "to ONNX via scripts/download_models.py."
        ),
    },

    # ------------------------------------------------------------------
    # DL-A: post-demosaic RGB denoiser
    # ------------------------------------------------------------------
    "nafnet_sidd_width32": {
        "filename":    "nafnet_sidd_width32.onnx",
        "mode":        "rgb_post",
        "input_desc":  "(1, 3, H, W) float32 [0,1] linear RGB",
        "output_desc": "(1, 3, H, W) float32 [0,1] denoised linear RGB",
        "source":      "https://github.com/megvii-research/NAFNet",
        "notes": (
            "NAFNet width-32 trained on SIDD. 40.30 dB PSNR. "
            "Pre-converted ONNX available at "
            "https://huggingface.co/mikestealth/nafnet-models — "
            "or convert from PyTorch via scripts/download_models.py."
        ),
    },

    "nafnet_sidd_width64": {
        "filename":    "nafnet_sidd_width64.onnx",
        "mode":        "rgb_post",
        "input_desc":  "(1, 3, H, W) float32 [0,1] linear RGB",
        "output_desc": "(1, 3, H, W) float32 [0,1] denoised linear RGB",
        "source":      "https://github.com/megvii-research/NAFNet",
        "notes": (
            "NAFNet width-64 trained on SIDD. Higher quality, larger model. "
            "Convert from PyTorch via scripts/download_models.py."
        ),
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_model_path(model_name: str, models_dir: str = "models") -> str | None:
    """
    Return the path to a model's .onnx file if it exists on disk.
    Returns None if the model is not in the registry or file not found.

    Parameters
    ----------
    model_name  : key in KNOWN_MODELS (e.g. "bjdd", "nafnet_sidd_width32")
    models_dir  : directory that contains .onnx files (default: "models/")
    """
    if model_name not in KNOWN_MODELS:
        return None
    filename = KNOWN_MODELS[model_name]["filename"]
    path = os.path.join(models_dir, filename)
    return path if os.path.isfile(path) else None


def auto_detect_model(mode: str, models_dir: str = "models") -> str | None:
    """
    Scan models_dir for the first known model matching the requested mode.
    Returns the full path, or None if nothing suitable is found.

    Parameters
    ----------
    mode       : "bayer_joint" or "rgb_post"
    models_dir : directory containing .onnx files
    """
    for name, spec in KNOWN_MODELS.items():
        if spec["mode"] == mode:
            path = os.path.join(models_dir, spec["filename"])
            if os.path.isfile(path):
                return path
    return None


def list_available(models_dir: str = "models") -> list:
    """Return list of (name, path) for every model present on disk."""
    found = []
    for name, spec in KNOWN_MODELS.items():
        path = os.path.join(models_dir, spec["filename"])
        if os.path.isfile(path):
            found.append((name, path))
    return found


def print_model_info():
    """Print the full registry to stdout (useful for debugging)."""
    for name, spec in KNOWN_MODELS.items():
        print(f"\n[{name}]")
        for k, v in spec.items():
            print(f"  {k}: {v}")
