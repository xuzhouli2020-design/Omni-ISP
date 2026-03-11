"""
scripts/download_models.py
===========================
One-time setup script: download pretrained model weights and convert to ONNX.

Run this on a machine that has internet access and PyTorch installed.
The resulting .onnx files are placed in  models/  and then used by the
Omni-ISP pipeline at inference time (no PyTorch needed at runtime).

Usage
-----
  python scripts/download_models.py --model all
  python scripts/download_models.py --model bjdd
  python scripts/download_models.py --model nafnet_sidd_width32

Requirements (download machine only, not needed at runtime)
-------------------------------------------------------------
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install nafnetlib requests tqdm

  For BJDD you also need:
  pip install gdown   # Google Drive downloader used by BJDD repo
"""

import argparse
import os
import sys

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model-specific download + export functions
# ---------------------------------------------------------------------------

def download_bjdd():
    """
    Download BJDD (Beyond Joint Demosaicking and Denoising, CVPRW 2021)
    pretrained weights and export to ONNX.

    Source: https://github.com/sharif-apu/BJDD_CVPR21
    Input:  (1, 1, H, W) float32 [0, 1] single-channel Bayer
    Output: (1, 3, H, W) float32 [0, 1] full-resolution RGB
    """
    print("\n=== BJDD: joint Bayer→RGB ===")
    _require_torch()

    import torch
    import torch.nn as nn
    import urllib.request

    out_path = os.path.join(MODELS_DIR, "bjdd_bayer_joint.onnx")
    if os.path.isfile(out_path):
        print(f"  Already exists: {out_path}  (skipping)")
        return

    # ---------------------------------------------------------------
    # Step 1: Clone BJDD repo and load model definition
    # ---------------------------------------------------------------
    import subprocess, tempfile, shutil

    tmpdir = tempfile.mkdtemp(prefix="bjdd_")
    try:
        print("  Cloning BJDD repo…")
        subprocess.check_call(
            ["git", "clone", "--depth", "1",
             "https://github.com/sharif-apu/BJDD_CVPR21.git", tmpdir],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        # ---------------------------------------------------------------
        # Step 2: Download pretrained weights
        # The BJDD repo hosts weights on Google Drive via gdown.
        # ---------------------------------------------------------------
        print("  Downloading pretrained weights (requires gdown)…")
        _require_package("gdown")
        import gdown

        # Weight file IDs from BJDD README (Bayer CFA model)
        weight_url = "https://github.com/sharif-apu/BJDD_CVPR21/releases/download/v1.0/bestModel.pth"
        weight_path = os.path.join(tmpdir, "bestModel.pth")

        try:
            print(f"  Downloading from GitHub releases…")
            urllib.request.urlretrieve(weight_url, weight_path)
        except Exception as e:
            print(f"  GitHub download failed ({e}).")
            print("  Please manually download bestModel.pth from:")
            print("  https://github.com/sharif-apu/BJDD_CVPR21")
            print(f"  and place it at: {weight_path}")
            return

        # ---------------------------------------------------------------
        # Step 3: Load model
        # ---------------------------------------------------------------
        sys.path.insert(0, tmpdir)
        try:
            # BJDD uses a module called 'model' or 'mainModel'
            # Adjust the import based on actual repo structure
            from model import BayerDemosaicDenoise as BJDDModel  # type: ignore
        except ImportError:
            try:
                from mainModel import GenerateNet as BJDDModel    # type: ignore
            except ImportError:
                print("  Could not import BJDD model class. Check repo structure.")
                print(f"  Repo cloned to: {tmpdir}")
                return

        net = BJDDModel()
        state = torch.load(weight_path, map_location="cpu")
        # Handle various checkpoint formats
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]
        net.load_state_dict(state, strict=False)
        net.eval()

        # ---------------------------------------------------------------
        # Step 4: Export to ONNX
        # ---------------------------------------------------------------
        print(f"  Exporting to ONNX: {out_path}")
        dummy = torch.zeros(1, 1, 512, 512)
        torch.onnx.export(
            net, dummy, out_path,
            input_names=["bayer"],
            output_names=["rgb"],
            dynamic_axes={"bayer": {2: "H", 3: "W"}, "rgb": {2: "H", 3: "W"}},
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"  ✓ Saved: {out_path}  ({os.path.getsize(out_path) // 1024} KB)")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        if tmpdir in sys.path:
            sys.path.remove(tmpdir)


def download_nafnet_sidd_width32():
    """
    Download NAFNet-SIDD-width32 ONNX model.

    Strategy A (preferred): pull pre-converted ONNX from HuggingFace.
    Strategy B (fallback):  download PyTorch weights via nafnetlib and export.

    Source: https://github.com/megvii-research/NAFNet
    Input:  (1, 3, H, W) float32 [0, 1] linear RGB
    Output: (1, 3, H, W) float32 [0, 1] denoised linear RGB
    """
    print("\n=== NAFNet-SIDD-width32: post-demosaic RGB denoiser ===")
    out_path = os.path.join(MODELS_DIR, "nafnet_sidd_width32.onnx")

    if os.path.isfile(out_path):
        print(f"  Already exists: {out_path}  (skipping)")
        return

    # ---------------------------------------------------------------
    # Strategy A: pre-converted ONNX from HuggingFace
    # ---------------------------------------------------------------
    print("  Trying HuggingFace (mikestealth/nafnet-models)…")
    hf_url = (
        "https://huggingface.co/mikestealth/nafnet-models/resolve/main/"
        "NAFNet-SIDD-width32.onnx"
    )
    try:
        import urllib.request
        urllib.request.urlretrieve(hf_url, out_path)
        print(f"  ✓ Downloaded ONNX directly: {out_path}  ({os.path.getsize(out_path) // 1024} KB)")
        return
    except Exception as e:
        print(f"  HuggingFace download failed: {e}")
        print("  Falling back to nafnetlib…")

    # ---------------------------------------------------------------
    # Strategy B: nafnetlib + PyTorch export
    # ---------------------------------------------------------------
    _require_torch()
    _require_package("nafnetlib", pip_name="nafnetlib")

    import torch
    from nafnetlib import DenoiseProcessor  # type: ignore

    print("  Downloading weights via nafnetlib (auto-cached)…")
    proc = DenoiseProcessor(arch="NAFNet-SIDD-width32")

    # Get underlying model
    net = proc.model
    net.eval()

    print(f"  Exporting to ONNX: {out_path}")
    dummy = torch.zeros(1, 3, 256, 256)
    torch.onnx.export(
        net, dummy, out_path,
        input_names=["rgb_noisy"],
        output_names=["rgb_clean"],
        dynamic_axes={
            "rgb_noisy": {2: "H", 3: "W"},
            "rgb_clean": {2: "H", 3: "W"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  ✓ Saved: {out_path}  ({os.path.getsize(out_path) // 1024} KB)")


def download_nafnet_sidd_width64():
    """NAFNet width-64 variant — same procedure as width-32."""
    print("\n=== NAFNet-SIDD-width64: post-demosaic RGB denoiser (large) ===")
    out_path = os.path.join(MODELS_DIR, "nafnet_sidd_width64.onnx")
    if os.path.isfile(out_path):
        print(f"  Already exists: {out_path}  (skipping)")
        return

    print("  Trying HuggingFace…")
    hf_url = (
        "https://huggingface.co/mikestealth/nafnet-models/resolve/main/"
        "NAFNet-SIDD-width64.onnx"
    )
    try:
        import urllib.request
        urllib.request.urlretrieve(hf_url, out_path)
        print(f"  ✓ Downloaded ONNX: {out_path}  ({os.path.getsize(out_path) // 1024} KB)")
        return
    except Exception as e:
        print(f"  HuggingFace failed: {e}. Try nafnetlib export manually.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import torch  # noqa
    except ImportError:
        print("\nPyTorch is required for weight download/conversion but is not installed.")
        print("Install it with:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        sys.exit(1)


def _require_package(name, pip_name=None):
    import importlib
    if importlib.util.find_spec(name) is None:
        pip_name = pip_name or name
        print(f"\n'{name}' is required but not installed. Run:")
        print(f"  pip install {pip_name}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MODELS = {
    "bjdd":                download_bjdd,
    "nafnet_sidd_width32": download_nafnet_sidd_width32,
    "nafnet_sidd_width64": download_nafnet_sidd_width64,
}


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert pretrained DL models for Omni-ISP."
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=list(MODELS.keys()) + ["all"],
        help="Which model to download (default: all)",
    )
    parser.add_argument(
        "--models-dir",
        default=MODELS_DIR,
        help=f"Output directory for .onnx files (default: {MODELS_DIR})",
    )
    args = parser.parse_args()

    global MODELS_DIR
    MODELS_DIR = os.path.abspath(args.models_dir)
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Models directory: {MODELS_DIR}")

    targets = list(MODELS.keys()) if args.model == "all" else [args.model]
    for name in targets:
        try:
            MODELS[name]()
        except Exception as e:
            print(f"  ERROR downloading {name}: {e}")

    print("\n--- Download complete ---")
    found = [
        f for f in os.listdir(MODELS_DIR) if f.endswith(".onnx")
    ]
    if found:
        print(f"Available models in {MODELS_DIR}:")
        for f in found:
            size_kb = os.path.getsize(os.path.join(MODELS_DIR, f)) // 1024
            print(f"  {f}  ({size_kb} KB)")
    else:
        print("No .onnx files found. Check error messages above.")


if __name__ == "__main__":
    main()
