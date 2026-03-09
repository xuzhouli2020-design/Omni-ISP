# EdgeISP — Implementation Roadmap

Status: **Phase 2 mostly complete; Phase 3.2 (Gamma EOTF) done; Phase 3 in progress**
Last updated: 2026-03-08

---

## Phase 0 — Project Setup [DONE]

- [x] Complete pipeline review and design spec (`docs/dev_notes.md`)
- [x] Module-by-module documentation (`docs/module_guide.md`)
- [x] Create `CLAUDE.md` (AI development guide)
- [x] Create `PROGRESS.md` (this file)

---

## Phase 1 — Pipeline Correctness Fixes [DONE]

These are bugs in the current pipeline. Must be fixed before any feature work.

- [x] **1.1 Swap 2D NR ↔ Sharpen order** — run 2D NR before Sharpen in `infinite_isp.py`
  - Swapped NR2D and SHARP blocks; RGBC now takes sharp_img
  - Ref: dev_notes.md "Sharpening — edge-adaptive gain + pipeline order fix"

- [x] **1.2 Move Gamma after LDCI and 2D NR** — Gamma now executes after all linear-domain processing
  - LDCI and NR2D moved before Gamma using linear luminance wrapper
  - Linear Y = 0.2126R + 0.7152G + 0.0722B extracted from CCM linear RGB; RGB scaled proportionally by enhanced Y
  - Pipeline: CCM → LDCI(linear) → NR2D(linear) → Gamma → AE → CSC → Sharpen
  - Ref: dev_notes.md "Gamma ordering fix"

- [x] **1.3 Split BLC into offset + linearise, insert OECF between** — correct OECF domain
  - Added `step` parameter to `BlackLevelCorrection` ("full" / "offset_only" / "linearise_only")
  - Pipeline: DPC → BLC(offset_only) → OECF → BLC(linearise_only) → Digital Gain
  - Backward compat: step defaults to "full" (original behaviour unchanged)
  - Ref: dev_notes.md "OECF before BLC linearization"

---

## Phase 2 — Module Algorithm Upgrades [IN PROGRESS]

Each item adds a configurable high-quality mode alongside the existing baseline default. Prioritised by impact and independence (can be done in parallel).

### Tier A — High impact, independent

- [x] **2.1 DPC: MAD-based adaptive threshold**
  - Added `mode: "fixed" | "mad"` and `dp_threshold_k: 4.0` to config
  - `dynamic_dpc.py`: `compute_mad_threshold()` estimates sigma = MAD/0.6745, threshold = k×sigma
  - Backward compat: default mode is "fixed" (original behaviour unchanged)
  - Ref: dev_notes.md "DPC: Auto-threshold"

- [ ] **2.2 Demosaic: LMMSE mode**
  - Add `demosaic_method: "mhc" | "lmmse"` to config
  - Implement LMMSE in new file `modules/demosaic/lmmse.py`
  - Ref: dev_notes.md "Demosaicing — MHC recap + state-of-the-art landscape"

- [ ] **2.3 LDCI: Guided filter LTM mode**
  - Add `mode: "clahe" | "guided_filter"` to config
  - Implement guided filter base/detail decomposition on Y channel
  - Ref: dev_notes.md "LDCI — upgrade to Guided Filter LTM"

- [x] **2.4 Sharpen: Edge-adaptive gain mode**
  - Added `mode: "usm" | "adaptive"` to config; `gain`, `noise_floor`, `edge_max`, `halo_limit`
  - `adaptive_sharpen.py`: Sobel-gated USM — gain ramps from 0 (noise_floor) to gain_max (edge_max)
  - Halo suppression via mask clip; pure NumPy Gaussian blur (no scipy)
  - Backward compat: default mode is "usm" (original UnsharpMasking, unchanged)
  - Ref: dev_notes.md "Sharpening — edge-adaptive gain"

### Tier B — Medium impact

- [x] **2.5 Color saturation: Vibrance mode**
  - Added `mode: "flat" | "vibrance"`, `vibrance_strength`, `chroma_limit` to config
  - Extended `color_space_conversion.py` saturation block: vibrance gain = 1 + k × (1 − chroma_norm)
  - Soft-knee chroma magnitude limiter for both modes (preserves hue angle)
  - Backward compat: default mode is "flat" (original uniform gain, unchanged)
  - Ref: dev_notes.md "Color Saturation Enhancement"

- [x] **2.6 2D NR: NLM + chroma-only modes**
  - Added `mode: "nlm" | "bilateral" | "chroma_only" | "off"` to config
  - `bilateral.py`: edge-preserving bilateral filter on Y channel (pure NumPy, no scipy)
  - `chroma_only`: Gaussian smooth on Cb/Cr only; Y untouched (for DL pipeline path)
  - `off`: hard bypass regardless of is_enable flag
  - Backward compat: default mode is "nlm" (original NLM algorithm, unchanged)
  - Ref: dev_notes.md "2D NR — dual-role design"

- [ ] **2.7 BLC: OB pixel per-capture black level**
  - Add `ob_pixel_rows` config, auto-detect BL from optical black region
  - Ref: dev_notes.md "BLC OB pixels"

- [x] **2.8 YUV format: 4:2:0 output**
  - Added `conv_type: "420"` to YUV config (`"444" | "422" | "420"`)
  - `yuv_conv_format.py`: I420 planar format — Y plane (H×W), Cb plane (H/2×W/2), Cr plane (H/2×W/2)
  - 2×2 box-filter chroma downsampling; odd dimensions clipped to even before subsampling
  - Ref: dev_notes.md "YUV format — add 4:2:0"

---

## Phase 3 — New Capabilities [NOT STARTED]

### Output system

- [ ] **3.1 CCM: XYZ intermediate architecture**
  - Split CCM into camera→XYZ (calibrated) + XYZ→target (fixed matrix)
  - Add target matrices for sRGB, Display P3, BT.2020
  - Ref: dev_notes.md "CCM target primaries + Gamma EOTF"

- [x] **3.2 Gamma: multi-EOTF support**
  - Added `eotf: "lut" | "srgb" | "rec709" | "linear" | "pq" | "hlg"` to config
  - `eotf.py`: exact analytical implementations of all 5 transfer functions (pure NumPy)
    - sRGB: IEC 61966-2-1 piecewise; Rec.709: BT.709 broadcast; linear: γ=1 passthrough
    - PQ: SMPTE ST.2084 (exact constants); HLG: ARIB STD-B67 sqrt+log
  - All EOTFs: 0→0, 1→1, monotonic, verified
  - Backward compat: default `eotf: "lut"` uses original LUT-based path unchanged
  - Ref: dev_notes.md "CCM target primaries + Gamma EOTF"

- [ ] **3.3 Output profile system**
  - Add `output_profile: "srgb" | "display_p3" | "hdr10" | "hlg" | "linear"` to config
  - Profile bundles CCM target + EOTF + bit depth automatically
  - Ref: dev_notes.md "CCM target primaries + Gamma EOTF"

- [ ] **3.4 Blue noise dithering**
  - Generate 128×128 void-and-cluster mask (static asset)
  - Add dither step before uint8 quantisation
  - Add `dither: "none" | "tpdf" | "blue_noise"` to output config
  - Ref: dev_notes.md "Blue noise dithering for 8-bit output"

- [ ] **3.5 JPEG output encoding**
  - Add `format: "png" | "jpeg" | "exr"` with `jpeg_quality` parameter
  - Complete the raw→JPEG pipeline

### Calibration module

- [ ] **3.6 Calibration module framework**
  - Create `calibration/` directory structure
  - Implement `calibration_runner.py` + shared utilities
  - Ref: dev_notes.md "Calibration Module — Design Proposal"

- [ ] **3.7 BLC calibrator** — dark frame capture, per-channel offset estimation
- [ ] **3.8 DPC calibrator** — dead pixel map from dark/bright frames
- [ ] **3.9 OECF calibrator** — stepped exposure → LUT generation
- [ ] **3.10 LSC calibrator** — flat-field capture → per-channel gain map
- [ ] **3.11 WB + CCM calibrator** — ColorChecker capture → WB gains + 3×3 CCM to XYZ
- [ ] **3.12 LSC: dual-mode implementation** — calibrated gain map + lensfun database lookup
  - Ref: dev_notes.md "LSC design"

---

## Phase 4 — Multi-frame & Low-light [NOT STARTED]

- [ ] **4.1 Multi-frame capture pipeline** — N Bayer frames → register → average
  - Homography registration on G channel
  - Weighted mean merge with motion masking
  - Ref: dev_notes.md "BNR: Multi-Frame Denoising design"

- [ ] **4.2 Low-light capture mode** — config profile bundling multi-frame + appropriate NR + demosaic
  - Ref: dev_notes.md "Low-light capture mode — pipeline architecture decision"

- [ ] **4.3 Adaptive N frames** — SNR-based frame count selection
  - Ref: dev_notes.md "BNR: Multi-Frame Denoising design"

---

## Phase 5 — DL Integration [NOT STARTED — FUTURE]

- [ ] **5.1 Joint Bayer DL denoise + demosaic model**
  - Noisy Bayer → clean RGB, replaces BNR + Demosaic + 2D NR Y-channel
  - Training data via unprocessing technique
  - Ref: dev_notes.md "ML/DL Denoising — Bayer domain vs. post-demosaic RGB domain"

- [ ] **5.2 Burst DL for low-light**
  - N raw Bayer frames → clean RGB, no explicit registration
  - Ref: dev_notes.md "Burst DL (Phase 4)"

- [ ] **5.3 HDR absolute luminance mapping**
  - Exposure metadata → nit mapping before PQ encoding
  - Ref: dev_notes.md "HDR absolute luminance add-on"

---

## Implementation notes

- **Backward compatibility**: default config must always produce identical output to original Infinite-ISP
- **Config-driven**: every new feature is a config flag, never a hardcoded change
- **dev_notes.md is the spec**: every item above references a section in `docs/dev_notes.md` with full technical rationale, algorithm detail, and config YAML design
- **Testing**: after each change, run `python isp_pipeline.py` and verify output in `out_frames/`
