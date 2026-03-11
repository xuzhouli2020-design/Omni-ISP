# Omni-ISP — Implementation Roadmap

Status: **Phases 1–3 + Phase 5–8 complete (419 tests); Phase 7 model training in `omni-isp-train`; factory calibration deferred**
Last updated: 2026-03-10

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

- [x] **2.2 Demosaic: LMMSE mode**
  - Added `demosaic_method: "mhc" | "lmmse"` to config; default "mhc" (backward compat)
  - `lmmse.py`: directional G interpolation (activity-weighted H/V fusion, eps-regularised)
    + bilinear colour-difference interpolation (H → V → diagonal priority)
  - Pure NumPy — no scipy; O(H×W) apart from small constant per kernel offset
  - Backward compat: default "mhc" calls Malvar-He-Cutler unchanged
  - Ref: dev_notes.md "Demosaicing — MHC recap + state-of-the-art landscape"

- [x] **2.3 LDCI: Guided filter LTM mode**
  - Added `mode: "clahe" | "guided_filter"` to config; default "clahe" (backward compat)
  - `guided_filter_ltm.py`: O(N) integral-image box filter → self-guided filter →
    base/detail decomposition → power-law base compression → detail amplification
  - Config: `guided_radius`, `guided_eps`, `detail_gain`, `base_compression`
  - Handles both float32 [0,1] and uint8 [0,255] YUV input; Cb/Cr untouched
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

- [x] **2.7 BLC: OB pixel per-capture black level**
  - Added `use_ob_pixels`, `ob_rows`, `ob_cols`, `ob_correction_mode`, `ob_smoothing` to config
  - `_estimate_offsets_from_ob()`: estimates per-channel offsets from OB region via
    median/mean; scalar mode overrides config offsets; per_column stores column-wise offset
  - `apply_blc_per_column()`: subtracts per-column FPN correction (1D broadcast)
  - Original parm dict not mutated (`.copy()` in `__init__`); fallback to config if no OB
  - Backward compat: `use_ob_pixels: false` (default) uses fixed config values unchanged
  - Ref: dev_notes.md "BLC: Leverage optical black (OB) pixels for per-capture black level"

- [x] **2.8 YUV format: 4:2:0 output**
  - Added `conv_type: "420"` to YUV config (`"444" | "422" | "420"`)
  - `yuv_conv_format.py`: I420 planar format — Y plane (H×W), Cb plane (H/2×W/2), Cr plane (H/2×W/2)
  - 2×2 box-filter chroma downsampling; odd dimensions clipped to even before subsampling
  - Ref: dev_notes.md "YUV format — add 4:2:0"

---

## Phase 3 — New Capabilities [IN PROGRESS]

### Output system

- [x] **3.1 CCM: XYZ intermediate architecture**
  - `xyz_matrices.py`: M_XYZ_TO_SRGB, M_XYZ_TO_P3, M_XYZ_TO_BT2020, M_SRGB_TO_XYZ
  - `get_xyz_to_target(target)` + `derive_camera_to_xyz(direct_ccm)` helpers
  - `color_correction_matrix.py`: `mode: "direct" | "xyz"` dispatch
    - "direct" = original single-step camera→sRGB (backward compat default)
    - "xyz" = camera→XYZ→target with optional explicit `camera_to_xyz_*` rows
    - Falls back to derived matrix (direct_ccm @ M_SRGB_TO_XYZ) when rows absent
  - Config: `mode`, `target`, optional `camera_to_xyz_{red,green,blue}` in CCM section
  - Ref: dev_notes.md "CCM target primaries + Gamma EOTF"

- [x] **3.2 Gamma: multi-EOTF support**
  - Added `eotf: "lut" | "srgb" | "rec709" | "linear" | "pq" | "hlg"` to config
  - `eotf.py`: exact analytical implementations of all 5 transfer functions (pure NumPy)
    - sRGB: IEC 61966-2-1 piecewise; Rec.709: BT.709 broadcast; linear: γ=1 passthrough
    - PQ: SMPTE ST.2084 (exact constants); HLG: ARIB STD-B67 sqrt+log
  - All EOTFs: 0→0, 1→1, monotonic, verified
  - Backward compat: default `eotf: "lut"` uses original LUT-based path unchanged
  - Ref: dev_notes.md "CCM target primaries + Gamma EOTF"

- [x] **3.3 Output profile system**
  - `util/output_profile.py`: 6 named profiles (srgb, rec709, display_p3, hdr10, hlg, linear)
  - Each profile bundles (ccm_mode, ccm_target, gamma_eotf) — always consistent
  - `apply_profile_to_params()` mutates parm_ccm + parm_gmc before module execution
  - Wired into `OmniISP.__init__()` after config loading
  - `output.profile: "custom"` (default) is a no-op — backward compat preserved
  - Config: new `output:` section with `profile` key at end of configs.yml
  - Ref: dev_notes.md "CCM and EOTF are coupled — use output profiles"

- [x] **3.4 Blue noise dithering**
  - `util/dither.py`: void-and-cluster mask generation + runtime dithering
    - `generate_void_and_cluster(size)`: Ulichney 1993 algorithm, pure NumPy
    - `apply_dither(img, mode)`: "none" | "tpdf" | "blue_noise"
    - `encode_8bit(img, dither_mode)`: accepts uint8/uint16/float input
    - Blue noise mask cached on first use (64×64, generates in ~2s)
  - Applied in `save_pipeline_output()` before uint8 save
  - Config: `output.dither: "none"` (default, backward compat)
  - Ref: dev_notes.md "Blue noise dithering for 8-bit output"

- [x] **3.5 JPEG output encoding**
  - `save_pipeline_output()` now reads `output.format` ("png" | "jpeg")
  - JPEG path: `plt.imsave(..., format="jpeg", pil_kwargs={"quality": N})`
  - Config: `output.format: "png"` (default), `output.jpeg_quality: 95`
  - Backward compat: default PNG path unchanged

### Factory Calibration [DEFERRED]

Factory-level calibration (OECF, LSC, CCM, BLC, DPC) deferred until core pipeline and 3A are mature.
Design spec preserved in dev_notes.md "Calibration Module — Design Proposal" [2026-03-06].

---

## Phase 4 — 3A Algorithm Upgrades [NOT STARTED]

### 3A Stats Infrastructure

- [ ] **4.1 3A stats collector** — shared zone-based stats for AE+AWB+AF
  - `modules/3a_stats/stats_collector.py`: zone luminance histograms, channel means/gradients, focus metric
  - NxN grid (default 16×16) computed once per frame in Bayer domain after BNR
  - Output: `Stats3A` dataclass consumed by all three algorithms
  - Ref: dev_notes.md "Phase 4 — 3A Algorithm Upgrades"

### Auto-Exposure (industry-grade)

- [ ] **4.2 AE: zone metering + histogram analysis**
  - `ae_stats.py`: zone luminance extraction, histogram computation from Stats3A
  - `ae_metering.py`: center-weighted, spot, matrix metering modes with configurable weights
  - Highlight protection (keep 97th percentile below saturation)
  - Ref: dev_notes.md "4.2 — AE: Industry-grade Auto-Exposure"

- [ ] **4.3 AE: exposure triangle solver + PID convergence**
  - `ae_controller.py`: PID convergence with damping, hysteresis band
  - Exposure triangle: shutter_us → analog_gain_db → digital_gain (configurable priority)
  - Flicker avoidance (50Hz/60Hz quantised shutter)
  - Output: `AEResult` dataclass with full exposure metadata for host
  - EV compensation support
  - Ref: dev_notes.md "4.2 — AE: Industry-grade Auto-Exposure"

### Auto-White-Balance

- [ ] **4.4 AWB: Gray Edge algorithm**
  - `gray_edge.py`: gradient-based illuminant estimation (Sobel/Laplacian, Minkowski p-norm)
  - More robust than Gray World for non-neutral scenes (sunset, foliage, tungsten)
  - Configurable: `edge_order` (1=Sobel, 2=Laplacian), `minkowski_norm` (1=L1, 6≈max)
  - Ref: dev_notes.md "4.3 — AWB: Gray Edge + temporal damping"

- [ ] **4.5 AWB: temporal damping**
  - IIR filter on gains across frames: `gain_t = α × new + (1-α) × prev`
  - Prevents WB flicker in video/burst. α=0 for single-shot (backward compat).
  - Ref: dev_notes.md "4.3 — AWB: Gray Edge + temporal damping"

### Auto-Focus

- [ ] **4.6 AF: focus metrics** — Tenengrad (Sobel gradient energy) + Laplacian variance
  - `af_metric.py`: compute sharpness score in configurable ROI
  - Operates on G channel in Bayer domain (highest spatial density)
  - Ref: dev_notes.md "4.4 — AF: Contrast-Detect Auto-Focus"

- [ ] **4.7 AF: search strategy + state machine**
  - `af_search.py`: IDLE → COARSE_SWEEP → FINE_SEARCH → TRACKING state machine
  - Coarse sweep across full lens range, fine hill-climb around peak
  - Output: `AFResult` with recommended lens position, direction, converged flag
  - Stateful across frames (remembers search progress)
  - Ref: dev_notes.md "4.4 — AF: Contrast-Detect Auto-Focus"

- [ ] **4.8 AF: pipeline integration**
  - `auto_focus.py`: module class with `execute()` returning `AFResult`
  - Position: Bayer domain after BNR, alongside AWB
  - Config: `auto_focus.is_enable: false` (default off — not all systems have AF)
  - Ref: dev_notes.md "4.4 — AF: Contrast-Detect Auto-Focus"

---

## Phase 5 — Multi-frame Pipeline [COMPLETE — 2026-03-09]

### Burst capture denoising

- [x] **5.1 Raw stack loader** — `modules/burst_capture/raw_stack_loader.py`
  - `RawStackLoader.load()` → (N, H, W) uint16 Bayer stack
  - Per-frame BLC-offset + OECF preprocessing using existing modules
  - Supports `.raw` (uint8/uint16) and libraw formats (rawpy)

- [x] **5.2 Phase correlation registration** — `modules/burst_capture/burst_registration.py`
  - `phase_correlate(ref_g, target_g)` → (dy, dx) subpixel translation
  - Parabolic peak refinement for sub-pixel accuracy
  - `apply_shift_bayer()` — Bayer-aware shift (sub-channel shift = shift/2)
  - scipy.ndimage.shift if available; pure NumPy integer fallback
  - `register_stack()` — aligns all N frames to reference

- [x] **5.3 Motion detection + temporal merge** — `modules/burst_capture/burst_merge.py`
  - `detect_motion()` — per-block SAD on green channel → (N, H, W) float32 masks
  - `temporal_merge()` — weighted_mean / mean / median
  - `merge_burst()` — top-level convenience: detect → merge
  - Static regions: mean of all N frames (√N SNR gain)
  - Moving regions: reference frame only (no ghosting)

- [x] **5.4 Pipeline integration** — `infinite_isp.py`
  - `set_burst_stack(stack)` — supply pre-loaded stack before `execute()`
  - Burst merge positioned after OECF, before BLC linearise
  - `burst_capture` config section in `config/configs.yml`
  - TNR wired after BNR, before Stats3A; `_tnr_state` persisted between frames

### Temporal Noise Reduction (video)

- [x] **5.5 TNR: IIR temporal filter** — `modules/temporal_nr/temporal_nr.py`
  - `TemporalNR` class with `TNRState` persistence dataclass
  - Per-pixel IIR EMA: `(1-α)*current + α*prev` for static pixels
  - Motion detection: frame-difference on downsampled green channel
  - Morphological erosion of static mask (prevents ghosting at edges)
  - `temporal_nr` config section; off by default

---

## Phase 6 — Lens Corrections [COMPLETE — 2026-03-09]

**Key design decision**: CAC and LDC are both warpers (inverse coordinate remapping + bilinear interpolation). They are composed analytically into a **single per-channel warp in the Bayer domain**, requiring only one bilinear resample per sub-channel instead of two. This avoids cascaded resampling quality loss and ensures the demosaicker always sees a geometrically correct AND chromatically aligned Bayer image. See dev_notes.md "Phase 6 — Lens Corrections".

- [x] **6.1 + 6.2 Unified CAC + LDC warp** — `modules/lens_correction/`
  - `warp_field.py`: Brown-Conrady LDC warp composed with per-channel CAC radial scale into single `(rows_src, cols_src)` map per Bayer sub-channel
  - `lens_correction.py`: `LensCorrection` class, Bayer domain (after Digital Gain, before LSC)
  - Supports all four Bayer patterns (RGGB, BGGR, GRBG, GBRG)
  - Bilinear remap via `scipy.ndimage.map_coordinates` (pure NumPy fallback)
  - Config: `lens_correction:` section with `ldc_enable`, `cac_enable`, `k1/k2/k3/p1/p2`, `r_ca`, `b_ca`, `focal_length_px`, `center`
  - All disabled by default (backward compat)

- [x] **6.3 Purple Fringe Removal** — `modules/purple_fringe_removal/`
  - `purple_fringe_removal.py`: `PurpleFringeRemoval` class, RGB domain (after Demosaic, before CCM)
  - Hue-band detection (configurable center+width) + saturation threshold + highlight proximity (dilated mask)
  - Pure-NumPy 4-connected binary dilation, vectorised RGB↔HSV conversion
  - Config: `purple_fringe_removal:` section with `hue_center`, `hue_half_width`, `sat_threshold`, `highlight_threshold`, `desaturation_radius`, `strength`
  - Disabled by default (backward compat)

- [x] **6.4 Tests** — `tests/test_lens_correction.py` (39 tests)
  - `TestWarpField` (10), `TestLensCorrection` (12), `TestPurpleFringeRemoval` (11), `TestPipelineIntegration` (5+1)
  - All 39 passing; full suite: 322 tests across 8 suites, all passing

---

## Phase 7 — DL Integration [INFERENCE INFRASTRUCTURE COMPLETE — 2026-03-10]

**Repo boundary**: Omni-ISP is inference-only. Training infrastructure (unprocessing data generator, training scripts, fine-tuning) lives in a separate `omni-isp-train` repo. The `.onnx` model files are dropped into `models/` and used at inference time without any PyTorch dependency.

- [x] **7.2 ONNX inference wrapper** — `modules/dl_denoise/onnx_inference.py`
  - `OnnxInferenceEngine`: loads any ONNX model, runs forward pass
  - `infer_tiled()`: tile-and-stitch with linear-blend overlap — handles full-resolution frames transparently (configurable `tile_size`, `tile_overlap`)
  - Graceful `OnnxUnavailable` exception when onnxruntime is not installed
  - CPU-friendly (ONNX Runtime), supports GPU via execution provider list

- [x] **7.3 DL-A: post-demosaic RGB denoiser** — `mode: "rgb_post"`
  - Target model: **NAFNet-SIDD-width32** (40.30 dB PSNR on SIDD)
  - Input: `(1, 3, H, W)` linear RGB; output: denoised linear RGB
  - Replaces 2D NR Y-channel; 2D NR downgraded to `chroma_only` as safety valve
  - Download: `python scripts/download_models.py --model nafnet_sidd_width32`
  - Ref: dev_notes.md "Phase 7 — DL Integration"

- [x] **7.4 DL-B: joint Bayer→RGB** — `mode: "bayer_joint"`
  - Target model: **BJDD** (CVPRW 2021, Sharif et al.)
  - Input: `(1, 1, H, W)` normalised Bayer [0,1]; output: `(1, 3, H, W)` full-res RGB
  - Replaces BNR + Demosaic in one ONNX forward pass; TNR skipped in DL-B mode
  - 2D NR auto-downgraded to `chroma_only`; WB still applied before DL pass
  - Download + convert: `python scripts/download_models.py --model bjdd`
  - `module_zoo.py`: registry of known models with input/output specs
  - `fallback_classical: true` (default) → silent fallback if model missing
  - Ref: dev_notes.md "Phase 7 — DL Integration"

- [x] **7.5 Model download helper** — `scripts/download_models.py`
  - Downloads NAFNet ONNX directly from HuggingFace (mikestealth/nafnet-models)
  - Downloads BJDD PyTorch weights and exports to ONNX via `torch.onnx.export`
  - One-time setup on any machine with internet + PyTorch; runtime needs only onnxruntime

- [ ] **7.1 Unprocessing data generator** — `omni-isp-train` repo (separate project)
  - sRGB → degamma → inverse CCM → mosaic → add sensor noise
  - Uses actual Omni-ISP pipeline parameters (BLC, OECF, CCM, Gamma) as the camera model
  - Foundation for fine-tuning BJDD or training compact wearable model from scratch

- [ ] **7.6 Compact wearable model** — `omni-isp-train` repo (separate project)
  - 4-channel packed Bayer input `(1, 4, H/2, W/2)`, <5 MB INT8 target
  - Designed for NPU deployment (Snapdragon, Apple Neural Engine)
  - Train from scratch using unprocessing generator with Sony ZVE10 pipeline params

- [ ] **7.7 DL-C: burst DL pipeline** — N frames → clean RGB, no explicit registration
  - End-to-end learned alignment + merge + denoise + demosaic
  - `dl_denoise.mode: "burst"` (infrastructure placeholder exists)
  - Ref: dev_notes.md "Burst DL"

---

## Phase 8 — HDR Pipeline [COMPLETE — 2026-03-10]

55 tests. All operators, merge modes, absolute luminance, and pipeline integration verified.

### 8.3 HDR Tone Mapping — `modules/hdr_tone_mapping/`
- [x] **operators.py** — four global operators + highlight rolloff knee:
  - `reinhard`      — log-average key adaptation, L_d = L_s / (1 + L_s)
  - `reinhard_ext`  — extended Reinhard with auto scene-white percentile
  - `aces`          — ACES RRT/ODT approximation (Narkowicz 2015 fit)
  - `hable`         — Hable / Uncharted 2 filmic curve (configurable exposure_bias)
  - `highlight_rolloff` — gentle cubic knee for HLG mode (no full recompression)
- [x] **hdr_tone_mapping.py** — `HDRToneMapping` module class
  - Wired between 2D NR (last linear step) and Gamma EOTF
  - Dispatches to operator by `mode` key; warns + clips on unknown mode
  - Graceful fallback when `absolute_luminance=true` but metadata missing

### 8.2 Absolute Luminance Mapping — integrated into HDRToneMapping
- [x] `scene_to_absolute_nits(rgb, iso, shutter_sec, aperture_f, k=12.5)` in operators.py
  - Formula: `L_nits = pixel_value × (k × f²) / (iso × t_sec)`  (ISO 12232)
  - Normalises by `peak_nits` to feed tone mapper relative to display peak
  - Reads from `sensor_info` (iso / shutter_ms / aperture) when parm values = 0

### 8.1 HDR Merge — `modules/hdr_merge/`
- [x] **hdr_merge.py** — `HDRMerge` class with two modes:
  - `"mertens"` — Exposure fusion (Mertens 2007): quality weights = contrast ×
    saturation × well-exposedness; epsilon floor prevents underflow on flat scenes
  - `"debevec"` — Debevec & Malik 1997: recovers camera response curve via
    least-squares, reconstructs log-radiance per channel, normalised by median grey
  - `isp.merge_exposures(frames, evs)` public API on `OmniISP` for multi-exposure workflow
- [x] Config: `hdr_merge.is_enable: false` (safe default — single-frame path unchanged)

---

## Factory Calibration [DEFERRED]

- [ ] **F.1 Calibration framework** — `calibration/` directory, runner, shared utilities
- [ ] **F.2 BLC calibrator** — dark frames → per-channel offsets + saturation levels
- [ ] **F.3 DPC calibrator** — dark/bright frames → static defect map
- [ ] **F.4 OECF calibrator** — exposure series → linearisation LUT
- [ ] **F.5 LSC calibrator** — flat-field → per-channel gain map
- [ ] **F.6 WB + CCM calibrator** — ColorChecker → WB gains + 3×3 CCM
- [ ] **F.7 Lens calibration** — checkerboard → distortion coefficients + CA model

Design spec in dev_notes.md "Calibration Module — Design Proposal" [2026-03-06].

---

## Implementation notes

- **Backward compatibility**: default config must always produce identical output to original Infinite-ISP
- **Config-driven**: every new feature is a config flag, never a hardcoded change
- **dev_notes.md is the spec**: every item above references a section in `docs/dev_notes.md` with full technical rationale, algorithm detail, and config YAML design
- **Testing**: after each change, run `python isp_pipeline.py` and verify output in `out_frames/`
- **Test suites**: test_phase1.py (27), test_phase2.py (24), test_phase3.py (46), test_phase4.py (38), test_phase5.py (45), test_3a.py (60), test_phase6.py (43), test_lens_correction.py (39) — 322 total tests, all passing
