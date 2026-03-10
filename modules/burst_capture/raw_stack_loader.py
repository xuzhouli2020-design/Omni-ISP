"""
File: raw_stack_loader.py
Description: Load and pre-process N raw Bayer frames into a (N, H, W) stack.

Each frame receives the same BLC offset + OECF treatment as the main pipeline
(Phases 1 BLC-offset and OECF steps) so all frames share the same linearised
domain before registration and merging.

Usage
─────
  loader = RawStackLoader(
      raw_paths   = ["frame0.raw", "frame1.raw", "frame2.raw"],
      sensor_info = sensor_info,
      parm_blc    = parm_blc,
      parm_oecf   = parm_oecf,
      data_root   = Path("in_frames"),
  )
  stack = loader.load()   # (N, H, W) uint16

Notes
─────
- Frames must all be the same sensor format (same width/height/bit_depth).
- .raw files: little-endian uint16 (>8-bit) or uint8 (≤8-bit).
- .dng / other libraw-supported formats: loaded via rawpy if available.
- If rawpy is not installed, only .raw files are supported.

Phase 5.1
"""

from pathlib import Path
from typing import List, Union

import numpy as np

from modules.black_level_correction.black_level_correction import (
    BlackLevelCorrection as BLC,
)
from modules.oecf.oecf import OECF


class RawStackLoader:
    """
    Load N raw Bayer frames and apply BLC-offset + OECF pre-processing.

    Parameters
    ----------
    raw_paths   : list of str or Path
                  Paths to raw frame files (any order; frame 0 is reference).
    sensor_info : dict
                  Sensor metadata (width, height, bit_depth, bayer_pattern).
    parm_blc    : dict
                  BLC config section from configs.yml.
    parm_oecf   : dict
                  OECF config section from configs.yml.
    platform    : dict
                  Platform dict (passed through to BLC/OECF).
    data_root   : Path | None
                  Optional root directory prepended to relative raw_paths.
    """

    def __init__(
        self,
        raw_paths:   List[Union[str, Path]],
        sensor_info: dict,
        parm_blc:    dict,
        parm_oecf:   dict,
        platform:    dict,
        data_root:   Union[str, Path, None] = None,
    ):
        self.raw_paths   = [Path(p) for p in raw_paths]
        self.sensor_info = sensor_info
        self.parm_blc    = parm_blc
        self.parm_oecf   = parm_oecf
        self.platform    = platform
        self.data_root   = Path(data_root) if data_root else None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def load(self) -> np.ndarray:
        """
        Load all frames and pre-process them.

        Returns
        -------
        np.ndarray  shape (N, H, W) dtype uint16
                    BLC-offset + OECF corrected Bayer stack.
        """
        frames = []
        for idx, path in enumerate(self.raw_paths):
            raw = self._load_single(path)
            preprocessed = self._preprocess(raw)
            frames.append(preprocessed)
            print(f"  [BurstLoader] frame {idx+1}/{len(self.raw_paths)}: {path.name}")

        stack = np.stack(frames, axis=0)   # (N, H, W)
        print(f"  [BurstLoader] loaded {len(frames)} frames → stack {stack.shape}")
        return stack

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        if self.data_root is not None:
            candidate = self.data_root / path
            if candidate.exists():
                return candidate
        return path

    def _load_single(self, path: Path) -> np.ndarray:
        """Load a single raw frame into a (H, W) uint16 array."""
        path = self._resolve(path)
        width     = self.sensor_info["width"]
        height    = self.sensor_info["height"]
        bit_depth = self.sensor_info["bit_depth"]

        suffix = path.suffix.lower()
        if suffix == ".raw":
            if bit_depth > 8:
                raw = np.fromfile(str(path), dtype=np.uint16).reshape((height, width))
            else:
                raw = (
                    np.fromfile(str(path), dtype=np.uint8)
                    .reshape((height, width))
                    .astype(np.uint16)
                )
            return raw
        else:
            # Try rawpy for DNG / CR2 / ARW etc.
            try:
                import rawpy
                img = rawpy.imread(str(path))
                return img.raw_image.copy()
            except ImportError:
                raise RuntimeError(
                    f"rawpy is not installed — cannot load '{path.suffix}' files. "
                    "Install rawpy or convert to .raw format."
                )

    def _preprocess(self, raw: np.ndarray) -> np.ndarray:
        """Apply BLC offset step + OECF to a single raw frame."""
        # BLC step 1: subtract per-channel offsets
        blc_offset = BLC(
            raw,
            self.platform,
            self.sensor_info,
            {**self.parm_blc, "step": "offset_only"},
        )
        blc_out = blc_offset.execute()

        # OECF: must index into BL-offset domain (before linearisation)
        oecf = OECF(blc_out, self.platform, self.sensor_info, self.parm_oecf)
        oecf_out = oecf.execute()

        return oecf_out
