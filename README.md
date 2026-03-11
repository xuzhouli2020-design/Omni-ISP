# Omni-ISP

**An open-source ISP pipeline for wearable and embedded cameras.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-419%20passing-brightgreen.svg)](tests/)

Built on [Infinite-ISP](https://github.com/10x-Engineers/Infinite-ISP) by 10xEngineers — with major algorithmic upgrades, a corrected pipeline, multi-frame denoising, unified lens correction, deep learning inference, and a full HDR pipeline.

---

## What is Omni-ISP?

Omni-ISP converts raw Bayer sensor data into display-ready images. It targets the hardest end of the ISP design space: **small-aperture wearable cameras** — where apertures are fixed, compute is constrained, noise is high, and optical imperfections can't be compensated with a bigger lens.

Three pillars drive every design decision:

- **Correct pipeline** — modules run in the right signal domain. Linear processing before gamma. NR before sharpening. BLC split around OECF. These aren't tweaks — they're correctness fixes that compound across the full pipeline.
- **Quality modes** — every module has a fast baseline (real-time embedded) and an opt-in high-quality mode (offline/research). Defaults are always backward-compatible with upstream.
- **Wearable orientation** — multi-frame burst denoising, unified lens correction, deep-learning denoising, and a full HDR path address the specific failure modes of small-sensor cameras.

---

## Pipeline

```
RAW Bayer
  │
  ├── Crop → DPC → BLC (offset) → OECF → BLC (linearise) → Digital Gain
  │           ↑ adaptive MAD         ↑ correct domain split
  │
  ├── Lens Correction  ← unified CAC + LDC single-pass Bayer warp  [Phase 6]
  │
  ├── LSC → BNR → AWB → WB → Demosaic
  │                            ↑ LMMSE high-quality mode
  │
  ├── Purple Fringe Removal  ← hue-selective desaturation  [Phase 6]
  │
  ├── CCM → LDCI → 2D NR  ← NLM / bilateral / chroma-only
  │          ↑ guided             ↑ [Phase 7] chroma-only mode when DL active
  │          filter LTM
  │
  ├── [DL-A: rgb_post]  ← ONNX model inference, tiled, CPU/GPU  [Phase 7]
  │
  ├── HDR Tone Mapping  ← Reinhard / ACES / Hable / HLG  [Phase 8]
  │
  ├── Gamma EOTF  ← sRGB / Rec.709 / PQ / HLG / linear
  │
  └── AE → CSC → Color Sat → Sharpen → RGB Conv → Scale → YUV → Dither → Output
                  ↑ vibrance   ↑ edge-adaptive                   ↑ blue noise
```

---

## Key improvements over Infinite-ISP

| Module / Area | Infinite-ISP | Omni-ISP |
|---|---|---|
| **Pipeline order** | Gamma before LDCI/NR; Sharpen before NR | Corrected linear-domain ordering throughout |
| **BLC + OECF** | Single BLC block before OECF | BLC split: offset → OECF → linearise (correct domain) |
| **DPC** | Fixed threshold | MAD-based adaptive threshold (scene-aware) |
| **Demosaic** | Malvar-He-Cutler only | + LMMSE high-quality mode |
| **LDCI** | CLAHE with bilinear tiles | + Guided filter LTM (no tile artefacts, edge-aware) |
| **Sharpen** | Uniform USM | + Edge-adaptive gain (noise floor suppressed) |
| **2D NR** | Single NLM mode | + Bilateral, chroma-only, off (adapts to DL path) |
| **Color Saturation** | Flat gain only | + Vibrance mode (protects already-saturated colours) |
| **CCM** | Direct camera → sRGB | XYZ intermediate → sRGB / Display P3 / BT.2020 |
| **Gamma** | LUT only | Analytical sRGB, Rec.709, PQ (HDR10), HLG, linear |
| **Output** | PNG only | + JPEG; 4:4:4 / 4:2:2 / 4:2:0 YUV |
| **Dithering** | None | Blue noise spatial dithering before 8-bit quantisation |
| **Lens correction** | None | **Unified CAC + LDC** — single Bayer-domain warp per channel |
| **Purple fringe** | None | HSV hue-band detection + highlight proximity mask |
| **Multi-frame** | None | Burst averaging with phase-correlation registration |
| **Temporal NR** | None | IIR temporal filter for video capture |
| **3A — AE** | Histogram only | + Zone metering, PID convergence, highlight protection |
| **3A — AWB** | Gray World | + Gray Edge (robust to non-neutral scenes) |
| **3A — AF** | None | Contrast-detect (Tenengrad + Laplacian), state machine |
| **BLC** | Fixed config values | + OB pixel estimation, per-column FPN correction |
| **DL inference** | None | **ONNX wrapper** — tiled, batched, graceful fallback to classical |
| **HDR tone mapping** | None | **5 operators** — Reinhard, Reinhard-ext, ACES, Hable, HLG knee |
| **Multi-exposure merge** | None | **Mertens 2007** exposure fusion + **Debevec & Malik 1997** HDR |
| **Absolute luminance** | None | ISO 12232 physical mapping — accurate nit-level HDR pipeline |

---

## Highlights

### Unified CAC + LDC — one warp, no quality loss

Chromatic Aberration Correction and Lens Distortion Correction are both coordinate-remapping operations. Omni-ISP composes them analytically into a **single per-channel warp field in the Bayer domain**, applied before demosaicing:

```
for each sub-channel (R, Gr, Gb, B):
    (xs, ys) = Brown-Conrady LDC undistortion
    (xs, ys) += per-channel CA differential radial scale
    output[y, x] = bilinear(input, ys, xs)
```

One bilinear resample per channel instead of two. No cascaded interpolation artefacts. The demosaicker always receives a geometrically and chromatically correct Bayer image.

---

### Burst denoising — validated on real hardware

Tested on Sony ZVE10 (24 MP, ISO 2000–4000, 8–10 frame bursts):

| Mode | Normalised noise σ | Gain vs single frame |
|---|---|---|
| Single frame, no NR | 0.00169 | 1.00× |
| Single frame + spatial NR | 0.00048 | 3.5× |
| **8-frame burst merge** | **0.00060** | **2.82×** |
| Burst + light spatial NR | 0.00028 | 6.0× |

Theoretical maximum for 8 frames: √8 = **2.83×**. Measured: **2.82×** — within 0.4% of the theoretical limit.

The pipeline uses phase-correlation registration on the green channel with sub-pixel parabolic refinement, per-block SAD motion detection, and weighted temporal merge (static regions: mean of all N frames; moving regions: reference only).

---

### Deep learning inference (Phase 7)

Omni-ISP includes a full **ONNX-based DL inference layer** that slots into the pipeline without breaking the classical path:

**DL-A: post-RGB denoising** inserts an ONNX model after 2D NR. In this mode, 2D NR automatically switches to chroma-only (spatial chroma smoothing only), and the DL model handles luma denoising. The entire tiling, padding, and batch-assembly logic is handled transparently:

```yaml
dl_denoise:
  is_enable: true
  mode: "rgb_post"           # post-demosaic ONNX inference
  model_path: "models/nafnet_sidd_width32.onnx"
  tile_size: 256             # input tile side length
  tile_overlap: 32           # overlap to suppress tile boundary artefacts
  batch_size: 4              # tiles per ONNX batch
  fallback_classical: true   # gracefully continue if model absent/invalid
```

**DL-B: joint Bayer model** (planned) will replace BNR + Demosaic + 2D NR luma in a single end-to-end pass.

**Graceful fallback**: if `model_path` is missing or the ONNX session fails to initialise, a `RuntimeWarning` is emitted and the pipeline continues with the classical 2D NR result — no crash, no silent corruption.

---

### HDR pipeline (Phase 8)

Omni-ISP includes a complete HDR path covering multi-exposure merge, physical luminance mapping, and display-referred tone mapping.

**Tone mapping** sits between 2D NR and Gamma EOTF — the only correct position to operate in linear light before encoding. All operators work in **luminance domain** (BT.709: 0.2126R + 0.7152G + 0.0722B), then scale RGB by the L_out/L_in ratio to preserve chrominance:

| Operator | Config `mode` | Best for |
|---|---|---|
| Global Reinhard | `reinhard` | Fast, smooth, natural |
| Extended Reinhard | `reinhard_ext` | White point control, prevents washed-out highlights |
| ACES (Narkowicz 2015) | `aces` | Cinematic S-curve, deep shadows, bright highlights |
| Hable / Uncharted 2 | `hable` | Film emulation, strong shadow lift |
| HLG highlight rolloff | `highlight_rolloff` | HLG scene-referred content only |

```yaml
hdr_tone_mapping:
  is_enable: true
  mode: "reinhard_ext"       # operator selection
  peak_nits: 1000.0          # display peak (nits) — used for absolute luminance normalisation
  key: 0.18                  # scene key value (log-average exposure anchor)
  white_percentile: 99.0     # percentile for extended Reinhard white point

  # Optional: absolute luminance mapping (ISO 12232)
  absolute_luminance: false
  iso: 100
  shutter_ms: 10.0
  aperture: 2.8
  calibration_k: 12.5        # ISO 12232 reflected-light calibration constant
```

**Multi-exposure merge** fuses a bracket of exposures before the single-image pipeline runs:

```python
# API: pass a list of linear RGB frames and their EV offsets
merged = isp.merge_exposures(frames, exposure_evs=[-1, 0, +1])
```

Two algorithms are available — `mertens` (Mertens 2007 exposure fusion, SDR output, no HDR headroom required) and `debevec` (Debevec & Malik 1997, recovers camera response curve, outputs true HDR radiance map for downstream tone mapping).

---

### Output profiles

Named profiles bundle CCM target primaries and gamma EOTF into a single config key:

```yaml
output:
  profile: "hdr10"    # srgb | rec709 | display_p3 | hdr10 | hlg | linear
```

This ensures CCM and gamma are always consistent — no more accidentally applying an sRGB gamma to a P3 CCM output.

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Run on the included ColorChecker sample
python isp_pipeline.py
```

Output is saved to `out_frames/`. All pipeline parameters are in `config/configs.yml` — every new feature is a config flag with a safe default.

### Running the test suite

```bash
python tests/test_phase1.py             # Pipeline correctness (27 tests)
python tests/test_phase2.py             # Module algorithm upgrades (24 tests)
python tests/test_phase3.py             # Output system (46 tests)
python tests/test_phase4.py             # BLC/2D NR/YUV extras (38 tests)
python tests/test_phase5.py             # CCM/gamma/output profiles (45 tests)
python tests/test_3a.py                 # 3A algorithms (60 tests)
python tests/test_phase6.py             # Burst + TNR pipeline (43 tests)
python tests/test_lens_correction.py    # Lens correction (39 tests)
python tests/test_phase7.py             # DL inference infrastructure (42 tests)
python tests/test_phase8.py             # HDR tone mapping + merge (55 tests)
```

All 419 tests pass on the current codebase.

---

## Enabling features

All new modes are off by default. Enable them individually in `config/configs.yml`:

```yaml
# High-quality demosaic
demosaic:
  is_enable: true
  demosaic_method: "lmmse"    # default: "mhc"

# Guided filter LDCI (no tile artefacts)
ldci:
  is_enable: true
  mode: "guided_filter"       # default: "clahe"

# Edge-adaptive sharpening
sharpen:
  is_enable: true
  mode: "adaptive"            # default: "usm"
  gain: 1.5
  noise_floor: 0.02
  edge_max: 0.15

# Burst denoising (requires pre-loaded stack)
burst_capture:
  is_enable: true
  n_frames: 8
  registration: "phase"
  merge_method: "weighted_mean"

# Lens distortion + chromatic aberration correction
lens_correction:
  is_enable: true
  ldc_enable: true
  k1: -0.12
  k2: 0.04
  cac_enable: true
  r_ca: [0.0008, 0.0, 0.0]
  b_ca: [-0.0010, 0.0, 0.0]

# DL-A post-RGB denoising (requires ONNX model)
dl_denoise:
  is_enable: true
  mode: "rgb_post"
  model_path: "models/nafnet_sidd_width32.onnx"
  tile_size: 256
  tile_overlap: 32
  batch_size: 4
  fallback_classical: true

# HDR tone mapping
hdr_tone_mapping:
  is_enable: true
  mode: "reinhard_ext"        # reinhard | reinhard_ext | aces | hable | highlight_rolloff
  peak_nits: 1000.0
  key: 0.18
  white_percentile: 99.0

# HDR10 output
output:
  profile: "hdr10"
  dither: "blue_noise"
```

---

## Project structure

```
omni_isp.py                   # Main pipeline class (OmniISP)
isp_pipeline.py               # CLI entry point
config/configs.yml            # All module parameters — single source of truth
modules/
  ├── dead_pixel_correction/  # DPC — MAD adaptive threshold
  ├── black_level_correction/ # BLC — OB pixel estimation, FPN correction
  ├── demosaic/               # MHC + LMMSE
  ├── ldci/                   # CLAHE + guided filter LTM
  ├── sharpen/                # USM + edge-adaptive
  ├── noise_reduction_2d/     # NLM + bilateral + chroma-only
  ├── color_correction_matrix/# Direct + XYZ intermediate
  ├── gamma_correction/       # LUT + sRGB/Rec.709/PQ/HLG/linear
  ├── lens_correction/        # Unified CAC + LDC Bayer warp      ← Phase 6
  ├── purple_fringe_removal/  # Hue-selective desaturation         ← Phase 6
  ├── burst_capture/          # Stack loader + registration + merge
  ├── temporal_nr/            # IIR temporal filter
  ├── dl_denoise/             # ONNX wrapper — tiled inference      ← Phase 7
  ├── hdr_tone_mapping/       # Tone mapping operators              ← Phase 8
  ├── hdr_merge/              # Mertens + Debevec merge             ← Phase 8
  ├── auto_exposure/          # Zone metering + PID
  ├── auto_white_balance/     # Gray World + Gray Edge
  └── auto_focus/             # Contrast-detect AF
util/
  ├── output_profile.py       # Named output profiles
  ├── dither.py               # Blue noise dithering
  └── bayer_utils.py          # Shared Bayer helpers
calibration/                  # Sensor calibration tools (in development)
docs/
  └── module_guide.md         # Per-module documentation
tests/                        # 419 tests across 10 suites
```

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 — Pipeline correctness | ✅ Complete | Ordering fixes, BLC/OECF domain split |
| 2 — Module upgrades | ✅ Complete | LMMSE, guided LDCI, adaptive sharpen, vibrance, NLM |
| 3 — Output system | ✅ Complete | XYZ CCM, multi-EOTF gamma, output profiles, blue noise dither |
| 4 — 3A algorithms | ✅ Complete | Zone AE, Gray Edge AWB, contrast-detect AF |
| 5 — Multi-frame | ✅ Complete | Burst denoising, phase-correlation registration, TNR |
| 6 — Lens corrections | ✅ Complete | Unified CAC + LDC warp, purple fringe removal |
| 7 — DL integration | ✅ Complete | ONNX inference layer, tiled DL-A (rgb_post), graceful fallback |
| 8 — HDR pipeline | ✅ Complete | 5 tone mapping operators, Mertens + Debevec merge, ISO 12232 luminance |
| 9 — DL-B joint model | 🔜 Planned | Joint Bayer → RGB model replacing BNR + Demosaic + NR luma |

---

## Documentation

- **[Module Guide](docs/module_guide.md)** — algorithm walkthrough, config reference, and tuning notes for every module
- **[Progress](PROGRESS.md)** — phased roadmap with detailed implementation notes per item

---

## License

Omni-ISP is licensed under the [Apache License 2.0](LICENSE).

Built on [Infinite-ISP](https://github.com/10x-Engineers/Infinite-ISP) — Copyright 2024, 10xEngineers. See [NOTICE](NOTICE) for upstream attribution.
