# Infinite-ISP — Development Notes

A running log of ideas, observations, and feasibility analyses for pipeline improvements.

---

## [2026-03-06] DPC: Auto-threshold and parameter-free detection

### Observation / Problem

The current DPC threshold (`dp_threshold = 80` by default for a 12-bit image) is a fixed raw pixel value and has two known weaknesses:

1. **Arbitrary and sensor-specific.** An offset of 80 out of 4095 (~2% of full scale) may be too aggressive on a noisy sensor (many false positives — good pixels flagged as dead) or too lenient on a very clean sensor (dead pixels missed).
2. **Breaks in dark / low-SNR scenes.** In dark regions, shot noise and read noise can easily exceed this threshold, causing normal noisy pixels to be incorrectly corrected.

### Current Algorithm Summary

Two conditions must both be true before a pixel is corrected:
- **Condition 1:** The pixel falls outside the min–max range of its 8 same-channel neighbours (local outlier check).
- **Condition 2:** ALL 8 directional differences between the pixel and its neighbours exceed `dp_threshold`.

The strict "ALL 8" requirement in Condition 2 already reduces false positives considerably, but the performance still degrades when the image noise floor approaches `dp_threshold`.

---

### Feasibility: Is auto-thresholding possible?

**Yes — several well-established approaches exist.** Detailed below, ordered from most to least practical for this pipeline.

---

#### Option 1: MAD-based Adaptive Threshold ✅ (Recommended)

**Idea:** Estimate the image noise standard deviation from the image itself using the **Median Absolute Deviation (MAD)**, then set the threshold as a multiple of that estimate.

**Formula:**
```
sigma_noise  = 1.4826 × median(|X - median(X)|)   -- robust noise estimator
dp_threshold = k × sigma_noise                      -- k ≈ 5–8 (5-sigma outlier rule)
```

The constant 1.4826 converts MAD to an equivalent Gaussian σ. The factor `k` is dimensionless and much more universal than a raw pixel threshold — `k = 5` works robustly across sensors, bit depths, and illumination levels.

**Why it works in dark scenes:** In a dark, noisy image, the noise floor is high, so MAD will be high, and the threshold automatically rises to match. The detector effectively asks "is this pixel an outlier relative to *this image's* noise level?" rather than a hardcoded absolute distance.

**Implementation sketch:**
```python
import numpy as np

def compute_adaptive_threshold(img, k=6):
    """Estimate noise-adaptive DPC threshold via MAD."""
    # Work on the gradient image (min directional difference per pixel)
    # to focus on spatial outliers rather than global brightness
    median_val = np.median(img)
    mad = np.median(np.abs(img - median_val))
    sigma_hat = 1.4826 * mad
    return max(k * sigma_hat, 1.0)   # floor at 1 to avoid divide-by-zero

# In DynamicDPC.dynamic_dpc(), replace:
#   threshold = self.threshold
# With:
#   threshold = compute_adaptive_threshold(self.img, k=6)
```

A more refined version estimates the threshold on the **minimum directional gradient** map (already computed inside the algorithm) rather than on raw pixel values — this makes it specifically sensitive to the spatial anomaly magnitude, not global scene brightness.

**Pros:** No calibration, single universal parameter `k`, works across illumination levels and sensors.
**Cons:** Adds a small computational overhead (one median operation); `k` still needs to be chosen once but is far less sensitive than a raw value.

---

#### Option 2: Bit-depth Normalised Fixed Threshold ✅ (Quick Win)

The current fixed threshold doesn't account for bit depth at all. A simple improvement:

```python
# Replace fixed threshold with a fraction of the full-scale range
THRESHOLD_FRACTION = 0.025   # ~2.5% of full scale
dp_threshold = THRESHOLD_FRACTION * (2**bpp - 1)
```

This ensures the same *relative* sensitivity regardless of whether the image is 8-bit, 10-bit, 12-bit, or 14-bit — meaning `dp_threshold` doesn't need to be retuned when bit depth changes.

**Pros:** One-line change, no calibration, backward compatible.
**Cons:** Still doesn't adapt to scene content or noise level.

---

#### Option 3: Local Statistics Threshold ✅ (Scene-Adaptive)

**Idea:** Instead of a global threshold, estimate the local noise level in a sliding window around each pixel.

```
local_mean   = mean(neighbours in window)
local_std    = std(neighbours in window)
dp_threshold = local_mean + k × local_std
```

A pixel is flagged dead only if it deviates from its local neighbourhood by more than `k` standard deviations.

**Pros:** Handles spatially non-uniform noise (e.g., gradient illumination, vignetting).
**Cons:** More compute-intensive; `k` still needs choosing; std dev is less robust than MAD in the presence of multiple dead pixels in the same neighbourhood.

---

#### Option 4: Dark Frame Calibration (Gold Standard) ✅ (Highest Accuracy)

**Idea:** Capture a **dark frame** (lens covered, same exposure/temperature as the target) and identify stuck pixels directly by their persistent deviation from the expected dark level.

```
dark_frame_mean = mean of N dark frames
defect_map = pixels where |dark_frame_mean - expected_dark| > k × noise_std
```

Store the `defect_map` as a binary mask and apply it to every subsequent frame instead of re-detecting every image.

**Pros:** Zero false positives; parameter-free at inference time; the map can be stored in camera firmware.
**Cons:** Requires a calibration step and a way to store/load the defect map; does not detect pixels that degrade over time unless recalibrated; not usable in pure single-image inference pipelines.

---

#### Option 5: Temporal Detection (Video/Burst Mode) ✅

**Idea:** A pixel that is persistently an outlier across multiple consecutive frames is definitively dead.

```
for each frame in burst:
    detect candidates using Condition 1 only (no threshold needed)

defect_pixel = pixels flagged in M out of N frames (e.g., 5/5)
```

**Pros:** Near-zero false positives; completely parameter-free.
**Cons:** Requires multiple frames; not applicable to single-frame still image processing; latency before detection on first boot.

---

### Comparison Summary

| Approach | Requires Calibration | Works Single-Frame | Adapts to Noise | Recommended Use |
|---|---|---|---|---|
| Fixed threshold (current) | No | Yes | No | Legacy / simple baseline |
| **MAD-adaptive** | **No** | **Yes** | **Yes** | **Best general option** |
| Bit-depth normalised | No | Yes | No | Quick fix for multi-bit-depth support |
| Local statistics | No | Yes | Partially | High-quality single-image |
| Dark frame map | Yes (offline) | Yes | Yes | Production cameras with calibration flow |
| Temporal | No | No (needs burst) | Yes | Video / burst pipelines |

---

### Recommendation

For Infinite-ISP as a software ISP reference model, the **MAD-based adaptive threshold (Option 1)** is the most practical improvement:

- Drop-in replacement for the fixed threshold
- No calibration required
- Removes sensitivity to bit depth, sensor, and illumination
- `k = 5` to `k = 8` is a well-established prior (Donoho's universal threshold principle)
- Can be made optional via a new config flag: `auto_threshold: true`

Proposed config change:
```yaml
dead_pixel_correction:
  is_enable: true
  is_debug: false
  auto_threshold: true      # NEW: use MAD-based auto threshold
  dp_threshold: 80          # used only when auto_threshold: false
  dp_threshold_k: 6.0       # NEW: sigma multiplier for auto mode
```

---

### References

- Donoho, D.L. (1995) — *De-noising by soft-thresholding* — IEEE Trans. Information Theory. (MAD-based noise estimation)
- Hampel, F.R. et al. (1986) — *Robust Statistics: The Approach Based on Influence Functions* (MAD as robust σ estimator)
- Yongji's et al. (2020) — *Dynamic Defective Pixel Correction for Image Sensor*, IEEE. (Current algorithm basis)
- ISO 17957:2015 — *Photography — Digital cameras — Methods for measuring opto-electronic conversion functions*

---

## [2026-03-06] DPC vs BLC ordering — which should go first?

### Observation / Problem

The current pipeline order is **DPC → BLC**. The question is whether **BLC → DPC** is physically more correct, with the argument that the optical input is spatially continuous (smooth scene brightness), whereas the *electrical* readout is what introduces abrupt defects — and correcting the systematic electrical offset (black level) first might give DPC a more meaningful signal to work with.

---

### Physical Argument: "Optical is continuous, electrical readout is abrupt"

This is a meaningful distinction. The sensor stack can be thought of in layers:

```
Scene (optical, smooth)
    → Photon absorption (spatially smooth)
    → Photo-electron generation (smooth + shot noise)
    → Integration + dark current (adds a systematic per-pixel offset)
    → ADC readout (adds fixed ADC offset = black level)
    → Electrical output (raw Bayer values)
```

The optical contribution is continuous and spatially correlated. The electrical contribution (black level, dark current) is a systematic bias added uniformly before the ADC output. A dead/hot pixel produces an "abrupt" anomaly that is independent of and on top of this electrical bias.

The argument for BLC first: removing the systematic electrical bias first puts DPC in a space that more closely represents the optical signal, making the "abruptness" of a defective pixel more physically meaningful against its local background.

---

### Critical Numerical Analysis — Where the Argument Breaks

This is where the ordering debate reveals a non-trivial failure mode. Consider a cold/dead pixel (stuck at 0 or below the black level) in a **dark scene**.

Let `BL = 200` (black level offset), `dp_threshold = 80`:

| Scenario | Raw pixel | Raw neighbors | DPC first diff | Pixel after BLC | Neighbors after BLC | BLC first diff |
|---|---|---|---|---|---|---|
| Hot pixel, any scene | 4095 | ~200 | 3895 ✅ | 3895 | ~0 | 3895 ✅ |
| Cold pixel, bright scene | 0 | ~2000 | 2000 ✅ | 0 (clipped from −200) | ~1800 | ~1800 ✅ |
| **Cold pixel, dark scene** | **0** | **~210** | **210 ✅** | **0 (clipped from −200)** | **~10** | **~10 ❌ MISSED** |
| **Sub-BL pixel (stuck=100)** | **100** | **~210** | **110 ✅** | **0 (clipped from −100)** | **~10** | **~10 ❌ MISSED** |

The critical failure: when BLC clips negative values to 0 (as the current implementation does with `np.clip(..., 0, ...)`), cold dead pixels in dark scenes become **indistinguishable from their neighbours** — both become ~0 after offset subtraction and clipping. DPC then sees no anomaly.

**BLC first → cold pixel detection degrades precisely in the hardest case: dark scenes and low-light captures.**

This is the opposite of the desired behaviour, since dark scenes are exactly where cold/stuck-low pixels are most visually harmful.

---

### Verdict on Ordering

There is no single universally correct ordering. The right answer depends on defect type:

| Defect type | DPC first (current) | BLC first (with clipping) |
|---|---|---|
| Hot pixel (stuck high) | ✅ Always detectable | ✅ Always detectable |
| Cold pixel, bright scene | ✅ | ✅ |
| Cold pixel, dark scene | ✅ | ❌ BLC clipping hides the defect |
| Sub-BL pixel | ✅ | ❌ BLC clipping hides the defect |

**Conclusion: The current DPC → BLC order is correct for cold/sub-BL pixel detection.**

However, the physical intuition opens an interesting middle path that resolves both:

---

### Recommended Resolution: Signed BLC → DPC → Clip

The root cause of the failure is **premature clipping** in BLC, not BLC itself. If BLC subtracts offsets in float without clipping, and DPC then operates on the signed (possibly negative) float data, the cold pixel is now a clear negative outlier — detectable against near-zero neighbours in a dark scene.

Proposed pipeline segment:
```
Raw Bayer (uint16)
  → BLC  : subtract per-channel offsets, output float32 (no clipping)
  → DPC  : operates on signed float — cold pixel = large negative outlier ✓
  → Clip : np.clip(result, 0, 2^bpp - 1) — first and only clipping point
  → OECF / linearisation / rest of pipeline
```

This approach:
- Preserves the physical argument (DPC works in offset-corrected space)
- Avoids hiding cold pixels through premature clipping
- Cold pixels in dark scenes appear as negative outliers → large |diff| → detectable
- Consistent with how professional hardware ISPs handle signed intermediate arithmetic

**Proposed config change** — add a `clip_after_blc` flag to defer clipping:

```yaml
black_level_correction:
  is_enable: true
  clip_after_blc: false   # NEW: defer clipping until after DPC stage
  ...
```

---

## [2026-03-06] BLC: Leverage optical black (OB) pixels for per-capture black level

### Observation / Problem

The current BLC reads fixed offsets from the config file (`r_offset`, `gr_offset`, `gb_offset`, `b_offset`). These are calibration values determined once and hard-coded. The problem is that black level is **not constant**:

- **Temperature dependence**: Dark current (thermally generated electrons) approximately doubles every 6–8°C. A camera warming up during a shoot will drift from its calibrated black level.
- **Exposure time dependence**: Dark current accumulates proportionally to integration time. The same offset calibrated at 1/100s will be wrong at 1s.
- **Inter-capture variation**: Subtle readout circuit variation causes the black level to fluctuate shot-to-shot.

A fixed config value is therefore a snapshot from one calibration condition and will be systematically wrong under any other condition.

---

### Optical Black (OB) Pixels — How They Work

Most image sensors dedicate a border region of actual photosites to **optically black pixels**: real photosensors permanently shielded from light by a metal layer. Because they receive zero photons, their output during a capture represents **only** dark current + read noise + ADC offset — i.e., the true black level for that exact capture, temperature, and exposure time.

```
Sensor layout (conceptual):
┌──────────────────────────────────────┐
│  OB rows (top, typically 8–16 rows)  │  ← metal-shielded, no light
├──────────────────────────────────────┤
│ OB  │  Active pixel array            │
│cols │  (actual image)                │  ← left columns, no light
│     │                                │
└──────────────────────────────────────┘
```

The number and location of OB pixels are specified in the sensor datasheet. Some raw file formats expose them directly:
- **DNG**: `BlackLevel`, `BlackLevelRepeatDim`, `ActiveArea` tags describe the OB region
- **Proprietary RAW** (NEF, CR2, ARW): vendor metadata includes OB pixel coordinates
- **Raw binary**: OB rows are often prepended to the active image region

---

### Implementation Sketch

```python
def estimate_black_level_from_ob(raw_img, bayer_pattern, ob_rows=8):
    """
    Estimate per-channel black level from optically black pixel rows.
    Uses median for robustness against hot pixels within the OB region itself.
    """
    ob_region = raw_img[:ob_rows, :]   # top OB rows

    if bayer_pattern == "rggb":
        r_bl  = np.median(ob_region[0::2, 0::2])
        gr_bl = np.median(ob_region[0::2, 1::2])
        gb_bl = np.median(ob_region[1::2, 0::2])
        b_bl  = np.median(ob_region[1::2, 1::2])

    return {"r": r_bl, "gr": gr_bl, "gb": gb_bl, "b": b_bl}


def apply_per_column_correction(active_img, ob_region):
    """
    Per-column BLC from OB rows — corrects vertical fixed-pattern noise (FPN).
    Each column gets its own offset, removing column-to-column readout variation.
    """
    col_offset = np.median(ob_region, axis=0, keepdims=True)  # shape: (1, W)
    return active_img - col_offset   # broadcasts across all rows
```

---

### Correction Levels — from Coarse to Fine

| Mode | What it corrects | When to use |
|---|---|---|
| **Global scalar** (current) | Same offset for entire channel | Calibration only; no thermal drift |
| **OB scalar** | Per-channel, per-capture offset | Default; handles temperature + exposure drift |
| **OB per-column** | Column-level FPN + drift | High ISO; visible vertical stripe artefacts |
| **OB per-row** | Row-level FPN | Some sensor architectures; horizontal banding |
| **OB 2D map** | Full spatial non-uniformity of dark current | Long-exposure and scientific imaging |

---

### Connection to the Ordering Note Above

If BLC uses live OB pixels (rather than fixed config values), the estimated black level is **accurate for this specific capture**. In a dark scene, true dark pixels will be close to the OB-estimated offset, and will correctly subtract to near zero. Combined with the **Signed BLC → DPC** order proposed above, cold dead pixels will still be detectable as negative outliers even in dark scenes — the two improvements directly reinforce each other.

---

### Proposed Config Extension

```yaml
black_level_correction:
  is_enable: true
  clip_after_blc: false          # defer clipping until after DPC (see ordering note)

  # Fixed offsets (fallback / used when OB pixels are unavailable)
  r_offset: 200
  gr_offset: 200
  gb_offset: 200
  b_offset: 200
  is_linear: true
  r_sat: 4095
  gr_sat: 4095
  gb_sat: 4095
  b_sat: 4095

  # Optical black pixel auto-estimation (NEW)
  use_ob_pixels: false           # set true when sensor exports OB pixel region
  ob_rows: 8                     # OB rows at top of raw frame (from sensor datasheet)
  ob_cols: 0                     # OB columns at left of raw frame
  ob_correction_mode: "scalar"   # "scalar" | "per_column" | "per_row"
  ob_smoothing: "median"         # "median" recommended; robust against OB hot pixels
```

---

### References

- Healey, G. & Kondepudy, R. (1994) — *Radiometric CCD camera calibration and noise estimation*, IEEE TPAMI (dark current temperature model)
- Nakamura, J. (2006) — *Image Sensors and Signal Processing for Digital Still Cameras*, CRC Press, Chapter 4 (OB pixel design)
- Adobe DNG Specification 1.6 — `ActiveArea`, `BlackLevel`, `BlackLevelRepeatDim` tags
- EMVA Standard 1288 — *Standard for Characterization of Image Sensors and Cameras* (formal dark current measurement procedures)

---

## [2026-03-06] LSC: Pipeline position, calibration approach, and standard lens profile loading

### Context

The current LSC module (`modules/lens_shading_correction/lens_shading_correction.py`) is a complete stub — `execute()` returns the image unchanged regardless of `is_enable`. The config carries only `is_enable` and `is_save`. This note captures the design brainstorm for a proper implementation.

---

### Pipeline Position

LSC is currently placed after Digital Gain:

```
... → BLC → OECF → Digital Gain → LSC → BNR → WB → Demosaic → ...
```

**LSC must run in the linear optical domain** — vignetting is a multiplicative attenuation of linear light, so the gain map correction (`pixel × gain`) is only physically correct in a linear signal space. After OECF + linearisation we satisfy this requirement, so the current placement is valid.

**However, LSC should arguably move to before Digital Gain:**

- Digital Gain is a spatially uniform amplifier. LSC is a spatially varying correction.
- If vignetting is corrected *after* DG, the corners (which are darker due to vignetting) have already been amplified by DG — their noise is amplified alongside the signal. Correcting LSC first then applying DG is slightly cleaner: fix the optical non-uniformity, then boost the entire corrected signal uniformly.
- Minor difference in practice for typical vignetting magnitudes, but conceptually correct.
- **Decision: move LSC to before Digital Gain.**

Proposed updated early pipeline:
```
DPC → BLC offset → OECF → BLC linearise → LSC → Digital Gain → BNR → WB → Demosaic → ...
```

---

### What LSC Actually Corrects — Vignetting and Color Shading

Lens shading has two distinct components that are often conflated:

**1. Luminance vignetting** — brightness fall-off from centre to corners. Caused by:
- Optical vignetting: the cos⁴(θ) geometric roll-off from lens physics (θ = angle from optical axis)
- Mechanical vignetting: lens barrel physically blocks oblique rays at wide apertures
- Natural vignetting: oblique angle of projection onto the sensor plane

**2. Color shading** — *different* vignetting magnitude for each colour channel. Caused by:
- Microlens arrays on CMOS sensors have angle-dependent spectral efficiency
- Marginal rays hitting corner pixels at steep angles couple into the microlenses differently for R, G, B wavelengths
- Wide-angle lenses show this most severely — the R channel can be 10–20% darker at corners than the G channel

**This is why LSC requires a separate gain map for each Bayer channel (R, Gr, Gb, B), not a single luminance map.** A luminance-only correction will fix the brightness fall-off but leave a residual colour cast at the corners — typically a green tint, since G is sampled twice and tends to fall off less.

---

### Approach 1: Flat-Field Calibration (Proper Measurement)

#### Equipment

| Equipment | Purpose | Accuracy |
|---|---|---|
| Integrating sphere | Perfectly uniform diffuse illumination | ±0.1% — gold standard |
| LED flat-field panel | Near-uniform backlit diffuser | ±1–3% — practical for most uses |
| Opal diffuser + collimated light | DIY flat-field | ±3–5% — budget option |

The light source must be **spectrally broad** (white) to illuminate all Bayer channels simultaneously, and its spatial uniformity must be verified without the lens (measure sensor + flat panel only, no lens, to establish the reference).

#### Capture Procedure

1. Mount camera at working conditions (intended aperture, focus at infinity or working distance).
2. Point at the uniform illumination source filling the full field of view.
3. Capture **N ≥ 20 frames** and **average** to reduce shot noise and read noise.
4. Capture a **dark frame** at the same settings (same exposure, same temperature, lens cap on) and subtract from the averaged flat-field.
5. Repeat at each aperture you want to calibrate — vignetting is strongly aperture-dependent (wide open = worst; stopped down = significantly reduced).
6. For zoom lenses: repeat at each focal length of interest.

#### Processing

```python
# 1. Average flat-field and subtract dark
flat = np.mean(flat_frames, axis=0) - dark_frame

# 2. Per-channel, find the maximum (near-centre peak)
#    Work on each Bayer channel separately
for channel in [R, Gr, Gb, B]:
    ch_flat = extract_bayer_channel(flat, channel)

    # 3. Smooth the flat (Gaussian or polynomial fit) to remove
    #    any remaining noise — the gain map must be smooth
    ch_smooth = gaussian_filter(ch_flat, sigma=10)

    # 4. Gain = max_value / local_value  (inverts the shading)
    ch_gain = ch_smooth.max() / ch_smooth

    # 5. Downsample to sparse grid (e.g., 17×13 control points)
    #    and store — runtime interpolation fills it back
    gain_map[channel] = downsample_to_grid(ch_gain, grid_size=(17, 13))
```

#### Gain Map Representation

Storing a full-resolution gain map is wasteful (2592×1536×4 floats = ~60 MB for a 12-bit sensor). Vignetting is smooth, so it compresses well. Standard representations:

| Representation | Size | Accuracy | Notes |
|---|---|---|---|
| Sparse grid (17×13 per channel) | ~3.5 KB | High — handles asymmetric lenses | Runtime bilinear or bicubic interpolation |
| 4th-order polynomial (radial) | 5 floats per channel | Moderate — assumes radial symmetry | Fast runtime; wrong for decentred lenses |
| Lensfun polynomial (`k0..k3`) | 4 floats per channel | Good for typical lenses | Standard open-source format |
| DNG grid (WarpRectilinear opcode) | variable | High | Embedded in DNG files |

**Sparse grid is recommended for Infinite-ISP**: accurate, compact enough for a YAML config, and handles real-world asymmetric or decentred lenses that polynomial models cannot capture.

---

### Approach 2: Standard Lens Profile Database

For production use or when a calibration rig is unavailable, established lens correction databases provide pre-measured gain maps indexed by lens make, model, focal length, and aperture.

#### Lensfun (Open Source — Recommended)

- Library: [github.com/lensfun/lensfun](https://github.com/lensfun/lensfun)
- License: LGPL — usable in open-source projects
- Coverage: >3,000 lenses from all major manufacturers
- Used by: darktable, RawTherapee, digiKam, GIMP, Hugin

Vignetting is stored as a 6th-order radial polynomial per aperture step:
```
v(r) = 1 + k1·r² + k2·r⁴ + k3·r⁶
```
where `r` is the normalised radius from image centre (0 = centre, 1 = corner). Multiple `(aperture, k1, k2, k3)` entries are stored and interpolated at runtime for the actual aperture.

Integration path for Infinite-ISP:
```python
import lensfun

# Look up lens from EXIF metadata (lens make, model, focal length, aperture)
db = lensfun.Database()
lens = db.find_lenses(camera, lens_make, lens_model)[0]

# Compute vignetting gain map at actual aperture
vignetting = lens.interpolate_vignetting(aperture=f_number)
gain_map = vignetting.apply_to_grid(image_width, image_height)
```

#### Adobe Lens Profile (LCP)

- Format: XML-based `.lcp` files
- Distribution: shipped with Lightroom/ACR; freely downloadable from Adobe
- Coverage: similar to lensfun but proprietary format
- Python parser: `lxml` can parse LCP XML directly
- Contains per-channel vignetting correction as a 2D polynomial

#### DNG Embedded Profiles

- Many cameras embed lens correction data in DNG metadata (`OpcodeList2` tag)
- `FixVignette` opcode stores a radial gain map directly applicable to the image
- `rawpy` exposes this; `exiftool` can extract it

#### EXIF-Based Auto-Lookup Flow

```python
# From raw file EXIF:
lens_make  = exif["LensMake"]       # e.g., "Canon"
lens_model = exif["LensModel"]      # e.g., "EF 24-70mm f/2.8L II USM"
focal_len  = exif["FocalLength"]    # e.g., 35.0
aperture   = exif["FNumber"]        # e.g., 2.8

# Look up in lensfun DB and generate gain map:
gain_map = lensfun_lookup(lens_make, lens_model, focal_len, aperture, img_shape)

# Fall back to config-provided gain map if lens not found:
if gain_map is None:
    gain_map = load_gain_map_from_config(parm_lsc)
```

---

### Comparison: Calibration vs. Database

| Aspect | Flat-Field Calibration | Lensfun / LCP Database |
|---|---|---|
| Accuracy | Highest — captures actual unit + lens combination | Good — averaged over many units of that lens model |
| Covers colour shading | Yes — per-channel (R, Gr, Gb, B) | Partially — lensfun vignetting is luminance-only |
| Equipment required | Integrating sphere or flat panel | None |
| Lens coverage | Any lens, including custom/prototype | Only catalogued commercial lenses |
| Aperture dependence | Must calibrate at each aperture | Built into lensfun (multiple entries per aperture) |
| Thermal/unit variation | Captured exactly | Not captured |
| Maintenance | Recalibrate if lens changed | Keep lensfun DB updated |
| Best for | Camera module manufacturers, lab use | General-purpose consumer ISP |

---

### Recommendation for Infinite-ISP

A **dual-mode implementation** best serves the reference model goal:

**Mode A — Config gain map** (for sensor/lens-specific calibration):
- User provides a sparse gain grid (e.g., 17×13 control points per channel) in `configs.yml`
- Runtime bilinear interpolation to full image size
- Handles colour shading (4 channels: R, Gr, Gb, B)
- Suitable for any lens including custom optics

**Mode B — Lensfun database lookup** (for general use):
- Read lens make/model/focal length/aperture from EXIF or config
- Query lensfun database for vignetting polynomial
- Convert to gain map and apply
- Falls back to Mode A if lens not in database

**Config extension:**
```yaml
lens_shading_correction:
  is_enable: true
  is_save: false

  # Mode A: provide gain map directly
  use_gain_map: true
  gain_map_r:  [[...17x13 float values...]]   # per-channel sparse grid
  gain_map_gr: [[...]]
  gain_map_gb: [[...]]
  gain_map_b:  [[...]]
  gain_map_interp: "bilinear"                 # "bilinear" | "bicubic"

  # Mode B: auto-lookup from lensfun (overrides gain_map if lens found)
  use_lensfun: false
  lens_make:   "Canon"
  lens_model:  "EF 24-70mm f/2.8L II USM"
  focal_length: 35.0
  aperture:    2.8
```

**Implementation sketch:**
```python
def apply_lsc(self, img, gain_map_r, gain_map_gr, gain_map_gb, gain_map_b):
    """Apply per-channel gain map to Bayer image."""
    H, W = img.shape
    bayer = self.sensor_info["bayer_pattern"]

    # Interpolate each sparse gain map to full image resolution
    # (half resolution for each Bayer channel)
    for ch_gain, (row_off, col_off) in zip(
        [gain_map_r, gain_map_gr, gain_map_gb, gain_map_b],
        bayer_offsets(bayer)         # channel pixel positions
    ):
        full_map = bilinear_upscale(ch_gain, target=(H//2, W//2))
        img[row_off::2, col_off::2] *= full_map

    return np.clip(img, 0, 2**bpp - 1)
```

---

### Open Question: Per-Channel vs. Luminance-Only Correction

Lensfun and most consumer databases only store **luminance vignetting** — a single polynomial applied equally to all channels. This corrects the brightness fall-off but leaves colour shading uncorrected. For high-quality results (especially with wide-angle lenses), per-channel correction from a measured flat-field is necessary.

For Infinite-ISP as a reference model, supporting per-channel gain maps in Mode A (even if lensfun Mode B is luminance-only) covers this case correctly and distinguishes the implementation from simpler open-source ISPs.

---

### References

- Kang, S.B. et al. (2000) — *Radiometric self-calibration with illumination-compensated image sequences* — IEEE CVPR
- lensfun project — [lensfun.github.io](https://lensfun.github.io) — database schema and vignetting polynomial definition
- Adobe DNG Specification 1.6 — `OpcodeList`, `FixVignette` opcode
- ISO 15739:2023 — *Photography — Electronic still-picture imaging — Noise measurements*

---

## [2026-03-06] Calibration Module — Design Proposal (separate from pipeline)

### Motivation

All the pipeline modules discussed in these notes — BLC, OECF, LSC, WB, CCM — require calibration parameters that are currently either hard-coded in `configs.yml` with placeholder values, or skipped entirely (LSC). These parameters must come from real measurements of the physical sensor + lens system, yet the project provides no tooling to produce them. Every user starting a new sensor port has to figure out calibration independently.

Additionally, the existing `Infinite-ISP_TuningTool` (separate repo) focuses on **subjective perceptual tuning** — adjusting noise reduction strength, sharpening, saturation for a desired look. That is a different activity from **objective measurement-based calibration**, which generates physically grounded constants for sensor characterisation. These two should be kept separate.

**The calibration module is a first-class addition to the algorithm design repo — not a separate repo.** It belongs here because calibration parameters feed directly into `configs.yml`, and the calibration procedures depend on the same raw loader and Bayer utilities already in the codebase.

---

### Fundamental Distinction: Calibration vs. Tuning vs. Runtime Pipeline

```
CALIBRATION (offline, setup-time, objective, measurement-based)
  → captures physical constants of a specific sensor + lens
  → output: calibrated configs.yml ready for the pipeline
  → done once per camera system (or when conditions change)

TUNING (offline, subjective, application-dependent)
  → adjusts perceptual quality parameters (sharpening, NR, saturation)
  → output: tuned configs.yml for a specific scene type or use case
  → done per use-case, by a human evaluating visual quality
  → scope of the existing Tuning Tool

PIPELINE (online, runtime, applies both to each frame)
  → consumes the outputs of calibration and tuning
  → scope of modules/ and infinite_isp.py
```

The calibration module fills the currently empty left column of this triangle.

---

### What Needs Calibrating — Mapping to Pipeline Parameters

| Calibration | Physical quantity measured | Config parameters produced | Capture requirements |
|---|---|---|---|
| **Black Level** | Dark current + ADC offset | `r_offset`, `gr_offset`, `gb_offset`, `b_offset`, `r_sat`…`b_sat` | Dark frames (lens cap, N≥20) |
| **Dead Pixel Map** | Stuck/defective pixels | Static defect map (future); `dp_threshold` recommendation | Same dark frames as BLC |
| **OECF** | Sensor opto-electronic response curve | `r_lut` (and per-channel LUTs) | Uniform flat-field at multiple exposures (or step wedge) |
| **LSC** | Per-channel vignetting / colour shading | `gain_map_r`, `gain_map_gr`, `gain_map_gb`, `gain_map_b` | Flat-field frame(s) at target aperture |
| **White Balance** | Sensor response to known neutral | `r_gain`, `b_gain` | Gray card or ColorChecker white patch under target illuminant |
| **CCM** | Camera-to-output colour space transform | `corrected_red`, `corrected_green`, `corrected_blue` | Macbeth ColorChecker under D65 |
| **Gamma LUT** | Target tone reproduction curve | `gamma_lut_8/10/12/14` | Derived from OECF, or from a specified target (e.g., sRGB) |

**Note:** The repo already contains ColorChecker RAW files (`in_frames/normal/ColorChecker*.raw`) that are currently unused. These are ready-made inputs for WB and CCM calibration demonstrations.

---

### Calibration Dependency Order

Calibrations are not independent — each one should be computed on data that has had earlier calibrations applied, so the measurements are clean:

```
Stage 1 — Sensor characterisation (raw domain, no prior corrections needed)
  ├── Black Level Calibration   (dark frames)
  └── Dead Pixel Mapping        (same dark frames)

Stage 2 — Opto-electronic correction (needs BLC applied first)
  └── OECF Calibration          (flat-field exposure series after BLC)

Stage 3 — Spatial corrections (needs BLC + OECF applied)
  └── LSC Calibration           (flat-field after BLC + OECF)

Stage 4 — Colour calibration (needs BLC + OECF + LSC applied, then demosaic)
  ├── White Balance Calibration  (gray card or neutral patch)
  └── CCM Calibration            (Macbeth ColorChecker, after WB)

Stage 5 — Tone reproduction (derived from OECF measurement or target spec)
  └── Gamma LUT generation
```

This dependency structure means the calibration tool should enforce or guide the order — you can't correctly calibrate CCM before you have a good LSC and WB.

---

### Proposed Directory Structure

```
calibration/
├── README.md                    # user-facing guide: what to shoot, in what order
├── calibration_runner.py        # CLI entry point: run one or all calibrations
│
├── utils/
│   ├── raw_loader.py            # load .raw files using sensor_info from config
│   ├── bayer_utils.py           # extract R/Gr/Gb/B channels, reconstruct
│   ├── chart_detector.py        # auto-detect and extract ColorChecker patches
│   ├── visualiser.py            # plot calibration curves, gain maps, CCM results
│   └── config_writer.py         # write calibrated params back into configs.yml
│
├── black_level/
│   ├── blc_calibrator.py        # dark frame stack → per-channel offsets + sat levels
│   └── README.md
│
├── dead_pixel/
│   ├── dp_calibrator.py         # dark frames → static defect map + threshold advice
│   └── README.md
│
├── oecf/
│   ├── oecf_calibrator.py       # exposure series → response curve → LUT
│   └── README.md
│
├── lsc/
│   ├── lsc_calibrator.py        # flat-field frames → per-channel sparse gain map
│   └── README.md
│
├── white_balance/
│   ├── wb_calibrator.py         # neutral reference → r_gain, b_gain
│   └── README.md
│
├── ccm/
│   ├── ccm_calibrator.py        # ColorChecker → least-squares 3×3 CCM
│   ├── colorchecker_refs.py     # D50/D65 reference XYZ and linearised sRGB per patch
│   └── README.md
│
└── gamma/
    ├── gamma_calibrator.py      # target curve spec or OECF-derived → gamma LUT
    └── README.md
```

---

### Key Design Principles

**1. Output-first design.** Every calibrator's primary output is a YAML snippet that drops directly into `configs.yml`. The tool's final step is always `config_writer.py` merging the result into the user's config. No calibration is "done" until the config is updated.

**2. Validate before computing.** Each calibrator checks data quality before running:
- BLC: Are the frames actually dark? (mean < 5% of full scale)
- OECF: Do the exposures span the full dynamic range? (no clipping at top/bottom)
- LSC: Is the flat-field actually uniform before the lens? (< 3% spatial variation in sensor-only reference)
- CCM: Is the ColorChecker recognisable? (patch contrast, exposure level)

**3. Visualise every result.** Every calibrator produces a plot the user can review before accepting:
- BLC: histogram of dark frame values per channel
- OECF: measured response curve vs. ideal linear
- LSC: 2D gain map heatmap per channel (should look like smooth bowl shape)
- CCM: colour error per patch before and after correction (ΔE plot)

**4. Incremental — calibrate one module at a time.** Users shouldn't have to redo everything. Each calibrator reads the current `configs.yml` (applying already-calibrated parameters) and appends/updates only its own section.

**5. Capture guide.** The `README.md` for each calibrator specifies exactly what to shoot, how many frames, at what settings, and why. This is as important as the code.

---

### Algorithm Sketches for Key Calibrators

#### BLC Calibrator

```python
# Input: N dark frames (uint16 Bayer RAW)
# Output: per-channel offsets and saturation levels

dark_stack = load_raw_stack(dark_frame_paths, sensor_info)
avg_dark   = np.median(dark_stack, axis=0)   # median across frames

for channel, (row, col) in bayer_channel_positions(bayer_pattern):
    ch = avg_dark[row::2, col::2]
    offset = np.percentile(ch, 50)   # median of dark = black level
    sat    = find_saturation_knee(ch) # where response flattens (sensor-specific)
    config[channel + "_offset"] = round(offset)
    config[channel + "_sat"]    = round(sat)
```

#### OECF Calibrator

```python
# Input: M flat-field frames at M different exposure times (after BLC)
# Output: LUT mapping raw code → linearised code

# For each exposure level, measure mean pixel value of flat-field region
# Fit a polynomial or spline to (mean_value, expected_linear_value) pairs
# expected_linear = exposure_time / reference_exposure_time * reference_code

measured_codes   = [mean_channel(frame) for frame in sorted_by_exposure]
expected_linear  = [t / t_ref * code_ref for t in exposure_times]

# Fit inverse: code → linear (this becomes the LUT)
from scipy.interpolate import interp1d
oecf_inverse = interp1d(measured_codes, expected_linear,
                        kind='cubic', fill_value='extrapolate')
lut = np.round(oecf_inverse(np.arange(2**bpp))).astype(np.uint16)
```

#### CCM Calibrator

```python
# Input: ColorChecker raw (after BLC + OECF + LSC + WB + demosaic)
# Output: 3×3 CCM

# Reference: D65 linearised sRGB values for all 24 Macbeth patches
reference_srgb = load_colorchecker_d65_reference()   # (24, 3) array

# Measured: mean RGB in each patch from the processed image
measured_rgb = extract_patch_means(processed_img, patch_locations)  # (24, 3)

# Solve least-squares: measured_rgb @ CCM.T ≈ reference_srgb
# Using numpy lstsq: minimise ||M @ CCM.T - reference||_F
CCM, residuals, _, _ = np.linalg.lstsq(measured_rgb, reference_srgb, rcond=None)

# CCM is (3,3); enforce rows sum to 1 (imatest convention) via post-normalisation
CCM = CCM / CCM.sum(axis=1, keepdims=True)
```

---

### CCM Note: ColorChecker Files Already in Repo

The repo contains three Macbeth ColorChecker RAW files at `in_frames/normal/`:
- `ColorChecker_2592x1536_12bits_RGGB.raw` — standard exposure
- `ColorCheckerRAW_ISO2500_2592x1536_12bit_RGGB.raw` — high ISO variant
- `ColorCheckerRaw_100DPs_ISO100_2592x1536_12bits_RGGB.raw` — clean ISO100 with dead pixels

These should become the **reference inputs for the CCM calibrator demo**, and also allow the CCM calibrator to serve as an integration test: run the full calibration pipeline → generate CCM → apply CCM → measure ΔE on the patches → assert ΔE < threshold.

---

### Integration with the Existing Pipeline

The calibration module is offline-only — it does not run inside `infinite_isp.py`. The integration point is `configs.yml`:

```
calibration/ → writes → configs.yml → read by → infinite_isp.py
```

The calibration runner should accept an existing `configs.yml` as both input (to read `sensor_info`, `bayer_pattern`, `bit_depth`, etc.) and output (to write calibrated params back). This avoids duplicating sensor definitions.

---

### Relationship to Existing Tuning Tool Repo

The `Infinite-ISP_TuningTool` (separate repo) handles subjective quality tuning. The calibration module is complementary, not competing:

```
Calibration module (this repo)     →  Tuning Tool (separate repo)
  Objective measurement                Subjective quality judgement
  Sensor/lens characterisation         Scene/application-specific tuning
  Dark frames, flat-fields, charts     Real-scene images
  Must run first                       Runs second, on calibrated system
  Output: physical constants           Output: perceptual parameters
```

Ideally the two tools share a `configs.yml` format and are documented as a two-step workflow: **calibrate first, then tune**.

---

### Open Questions for Future Discussion

1. **Patch detection automation**: Should `chart_detector.py` auto-locate ColorChecker patches (using homography), or require the user to manually specify patch coordinates? Auto-detection is better UX but adds a significant computer vision dependency (OpenCV).

2. **Multi-illuminant CCM**: A single D65 CCM is correct for daylight but wrong under tungsten (D30). Professional ISPs have multiple CCMs and interpolate between them based on the scene's colour temperature (estimated by AWB). Should the calibration tool support multi-illuminant CCM generation?

3. **Aperture series for LSC**: Vignetting changes significantly with aperture. Should the LSC calibrator require captures at each target aperture, or fit a model (e.g., polynomial in f-stop) to interpolate?

4. **Sensor-in-the-loop capture**: Can the calibration runner directly trigger capture via a camera API (e.g., gphoto2 for DSLR, libcamera for embedded sensors) for fully guided capture, rather than requiring the user to pre-capture and specify file paths?

---

### References

- Ilie, A. & Welch, G. (2005) — *Ensuring Color Consistency Across Multiple Cameras* — ICCV
- Karaimer, H.C. & Brown, M.S. (2016) — *A Software Platform for Manipulating the Camera Imaging Pipeline* — ECCV (CCM and pipeline calibration)
- X-Rite Macbeth ColorChecker — spectral data and reference values: [xrite.com](https://www.xrite.com)
- EMVA Standard 1288 — measurement methodology for all sensor characterisation
- Lebowsky, F. (ed., 2020) — *Energy Efficiency of Electronic Systems and Displays*, Chapter 6 (ISP calibration flow)

---

## [2026-03-06] BNR: Noise physics clarification + Multi-Frame Denoising design

### Noise Physics — Clarification

**Shot noise** is the natural randomness in photon arrival. Photons are discrete quantum events and arrive at a photosite following a Poisson process — even under perfectly stable illumination, the count varies frame to frame. Since Poisson variance = mean, a pixel receiving μ photons has noise σ = √μ and SNR = √μ. This is fundamental physics, not a sensor defect.

**Read noise** is the total noise floor added by the entire electronic readout chain. It is broader than just ADC quantisation precision loss, which is one component:

| Source | Mechanism | Dominant in |
|---|---|---|
| In-pixel source follower amplifier | Thermal (Johnson) noise — random voltage fluctuations in the readout transistor | Most modern CMOS sensors — typically the largest contributor |
| Reset noise (kTC noise) | Thermal noise from the reset transistor charging the sense node | Sensors without CDS (Correlated Double Sampling) |
| Column amplifier noise | Each column has a shared amplifier; its thermal noise affects all pixels in that column | Rolling shutter sensors |
| ADC quantisation | Finite precision of the analogue-to-digital converter | Budget/low-resolution ADCs; usually smaller than amplifier noise |

So read noise is best described as: *the noise floor of everything the electronics add before and during digitisation* — ADC precision is one piece, but in a well-designed sensor the in-pixel amplifier dominates.

**Why this matters for BNR placement:** Shot noise is signal-dependent and random (uncorrelated frame-to-frame). Read noise is approximately signal-independent and also random (uncorrelated frame-to-frame). Both can be reduced by multi-frame averaging. Fixed Pattern Noise (FPN — column stripes, PRNU) is systematic and does **not** average away with multiple frames of scene content; it requires a separate calibration correction (BLC offset for global FPN, per-column OB correction for column FPN).

---

### The Bilateral Filter — Clarification

A standard Gaussian blur computes a weighted average where weights depend only on spatial distance:

```
blurred(p) = Σ G_spatial(|p - q|) × pixel(q)  /  Σ G_spatial(|p - q|)
```

A bilateral filter adds a second weight factor based on **intensity similarity**:

```
bilateral(p) = Σ G_spatial(|p - q|) × G_range(|pixel(p) - pixel(q)|) × pixel(q)
               ─────────────────────────────────────────────────────────────────
               Σ G_spatial(|p - q|) × G_range(|pixel(p) - pixel(q)|)
```

`G_range` is a Gaussian over the intensity difference. Pixels with very different values (across an edge) get near-zero weight, so the edge is excluded from the average. Pixels with similar values (same flat region) get high weight and are smoothed together. The result: noise is averaged out within uniform regions, edges are preserved.

The **Joint** variant replaces `pixel(p) - pixel(q)` in the range kernel with the corresponding difference in a *guide* channel (green), so the edge preservation logic uses green's spatial structure even when filtering red or blue.

---

### Multi-Frame Denoising — Design Note

**Your understanding is correct**: the right sequence is capture N frames → register them to a common reference → average in the Bayer domain → then apply JBF on the averaged result.

This is the basis of modern computational photography night modes (Google Night Sight, Apple Night Mode, Samsung Expert RAW). The key insight is that shot noise is *statistically independent across frames* for the same pixel — it cancels when averaged. True scene signal is consistent and accumulates coherently. Averaging N frames gives a √N improvement in SNR.

#### Why Averaging Happens in the Bayer Domain (Before Demosaic)

Registration and averaging must be done **before demosaicing** for the same reason BNR is done before demosaicing — demosaicing spreads and cross-correlates noise spatially and chromatically. If you demosaic each frame first then average RGB images, any residual misalignment appears as colour fringing. In the Bayer domain, each pixel has a single colour measurement, misalignment artifacts stay within the same colour channel, and the averaged result feeds a single clean demosaic pass.

#### Where It Fits in the Pipeline

```
Capture N Bayer frames (same scene, same or similar settings)
  │
  ▼
BLC offset × N              (remove dark level per-capture, stay in float)
  │
  ▼
OECF × N                    (linearise sensor response per frame)
  │
  ▼
BLC linearise × N           (rescale to full bit depth per frame)
  │
  ▼
Registration                (align frames 2..N onto reference frame 1)
  │
  ▼
Temporal averaging          (merge N frames → one clean Bayer frame)
  │
  ▼
LSC                         (vignetting correction on merged frame)
  │
  ▼
BNR / JBF                   (handles residual read noise + FPN after averaging)
  │
  ▼
Digital Gain (AE)
  │
  ▼
White Balance → Demosaic → ... rest of pipeline
```

#### Registration — The Hard Part

Frame-to-frame misalignment sources: camera shake (dominant for handheld), subject motion, and rolling shutter wobble. Registration quality determines whether averaging produces a sharper result or a blurry ghost.

**Levels of registration, from coarse to fine:**

| Level | Method | Handles | Compute cost |
|---|---|---|---|
| Global translation | Phase correlation / cross-correlation on Bayer | Pure camera translation | Very fast |
| Homography (8-DOF) | Feature matching (ORB/FAST) + RANSAC | Camera rotation + translation | Moderate |
| Per-block motion vectors | Block matching (like video codec P-frames) | Local parallax, subject motion | Moderate-high |
| Dense optical flow | Lucas-Kanade or Farnebäck on green channel | Arbitrary per-pixel motion | High |

For a reference ISP, **global homography** is the right starting point — covers the dominant camera shake case, implementable without heavy dependencies, and gives correct results on static scenes.

**Critical**: registration must be computed on **linearised, BLC-corrected Bayer data**, not raw codes. Photometric differences between frames (slight exposure variation) should be normalised before alignment comparison, otherwise the motion estimator mistakes brightness variation for spatial motion.

#### Temporal Averaging — Simple and Robust Variants

**Simple mean** (best SNR gain, assumes all frames equally exposed, no motion):
```python
merged = np.mean(aligned_frames, axis=0)   # shape: (H, W), float32
# SNR gain: √N  (e.g., 4 frames → 2× SNR improvement)
```

**Weighted mean** (handles motion regions by reducing frame weights):
```python
# Per-pixel motion confidence: high confidence = low motion = use all frames
# Regions with high inter-frame difference get lower weight
motion_score = inter_frame_difference(aligned_frames)
weights = np.exp(-motion_score / motion_threshold)   # (N, H, W)
merged = np.sum(weights * aligned_frames, axis=0) / np.sum(weights, axis=0)
```

**Median** (robust to outlier frames, e.g., one blurry or flash-lit frame):
```python
merged = np.median(aligned_frames, axis=0)
# Note: median gives only ~√(π/2) ≈ 1.25× SNR gain vs. √N for mean
# Trade: robustness over maximum SNR
```

For a first implementation, weighted mean is the best balance: better noise reduction than median, handles motion unlike simple mean.

#### Role of JBF After Averaging

After multi-frame averaging, shot noise is substantially reduced (√N × improvement). The JBF's job changes:
- **Before multi-frame**: primary denoising — must handle both shot noise and read noise aggressively
- **After multi-frame**: cleanup pass — mainly handles residual read noise (which does not average away with only N=4–8 frames), FPN that wasn't removed by BLC, and any alignment artefacts at motion boundaries

This means the JBF can run with **weaker parameters** (lower `std_dev_s`, lower `std_dev_r`) when preceded by temporal averaging — less blur risk, better detail preservation.

#### Adaptive Frame Count

Rather than a fixed N, the ideal system decides N based on the estimated noise level of the scene:

```python
# Estimate shot noise from first frame's signal level
# Low-light scene → high noise → need more frames
# Bright scene → low noise → fewer frames needed, or skip entirely

signal_estimate = np.percentile(first_frame_green, 50)   # median green
noise_estimate  = estimate_noise_mad(first_frame_green)   # from Note 1

snr_current = signal_estimate / noise_estimate
snr_target  = 20.0   # dB target

N_needed = max(1, int((snr_target / snr_current) ** 2))
N_capture = min(N_needed, N_max)   # cap at sensor/memory budget
```

This is essentially what phone night modes do: in very dark scenes they capture more frames (up to ~20), in moderate light they capture fewer.

#### Connection to AE

Multi-frame denoising changes the optimal AE strategy. In single-frame mode, AE increases gain (ISO) to reach target brightness in dark scenes. In multi-frame mode, it is better to:
- Use **lower per-frame gain** (lower ISO → less noise amplification, lower read noise floor)
- Use **shorter per-frame exposure** (reduces motion blur per frame)
- Capture more frames to compensate for lower per-frame brightness

This means the AE module should be aware of whether multi-frame mode is active and adjust its gain/exposure tradeoff accordingly — a new interaction point between AE and the BNR/multi-frame subsystem.

---

### Proposed Config Extension

```yaml
bayer_noise_reduction:
  is_enable: true
  filt_window: 5
  r_std_dev_s: 2.0
  r_std_dev_r: 20.0
  g_std_dev_s: 2.0
  g_std_dev_r: 20.0
  b_std_dev_s: 2.0
  b_std_dev_r: 20.0

  # Multi-frame temporal denoising (NEW)
  multi_frame_enable: false      # enable temporal averaging before JBF
  n_frames: 4                    # fixed number of frames (used when adaptive=false)
  adaptive_n: true               # auto-decide N from noise estimate
  n_frames_max: 8                # cap on adaptive mode
  snr_target_db: 20.0            # target SNR for adaptive N decision
  registration_method: "homography"   # "translation" | "homography" | "optical_flow"
  merge_method: "weighted_mean"       # "mean" | "weighted_mean" | "median"
  motion_threshold: 0.05              # relative intensity difference for ghosting suppression
```

---

### References

- Buades, A. et al. (2011) — *Non-Local Means Denoising* — IPOL (spatial NLM, comparable benchmark)
- Tomasi & Manduchi (1998) — *Bilateral Filtering for Gray and Color Images* — ICCV (bilateral filter original)
- Wronski, B. et al. (2019) — *Handheld Multi-Frame Super-Resolution* — SIGGRAPH (Google Night Sight technical basis)
- Liba, O. et al. (2019) — *Handheld Mobile Photography in Very Low Light* — SIGGRAPH Asia (Google Night Sight detail)
- Liu, C. et al. (2014) — *Fast Burst Images Denoising* — SIGGRAPH Asia (efficient multi-frame denoising)

---

## [2026-03-06] DPC/BLC ordering — decision: keep DPC → BLC, signed approach not worth the complexity

### Decision

The signed intermediate arithmetic approach (Signed BLC → DPC → Clip) is theoretically correct but **not pursued** for now. Reasons:

- Cold/sub-BL dead pixels in dark scenes are an edge case — dead pixels are typically stuck high (hot) or stuck at intermediate values, not sub-BL.
- Introducing signed float arithmetic into the DPC stage adds meaningful implementation complexity: DPC currently works on `uint16` with `maximum_filter` / `minimum_filter` / `correlate` — all of which behave differently on signed vs. unsigned data, and `scipy.ndimage` filters assume non-negative inputs in several modes.
- The cold-pixel-in-dark-scene failure mode, while real, is unlikely to be visually significant: a cold pixel in a dark scene produces a very small absolute error (the pixel stays at 0, which is close to where it should be anyway).
- The adaptive threshold improvement (MAD-based, see Note 1) is a higher-priority and lower-cost improvement that already helps with the dark-scene sensitivity problem.

**Conclusion: retain current `DPC → BLC` order. Flag the cold-pixel-dark-scene limitation as a known edge case. Revisit the signed approach if empirical testing shows visible artefacts.**

---

## [2026-03-06] OECF should precede BLC linearization — correct sensor curve before stretching range

### Observation / Problem

The current pipeline applies modules in this order:

```
DPC → BLC (offset subtract + linearise) → OECF → Digital Gain → ...
```

The BLC module bundles two logically distinct operations:
1. **Offset subtraction**: `x' = x - BL` — removes the systematic dark level, maps [BL, sat] to [0, sat−BL]
2. **Linearisation**: `x'' = x' / (sat−BL) × (2^bpp − 1)` — rescales the usable range to fill the full bit depth

OECF then corrects the sensor's nonlinear opto-electronic response by applying a LUT.

The problem: OECF is applied *after* linearisation, but the OECF LUT is calibrated against the sensor's native output codes. If you stretch the code range first and then apply the LUT, the LUT indices no longer correspond to the codes it was calibrated for — you are correcting a nonlinearity in an already-rescaled domain, which is physically inconsistent.

---

### Physical Argument

The sensor's opto-electronic characteristic describes the fundamental nonlinear mapping from photons to ADC codes. This nonlinearity is a property of the sensor itself — it exists in the native code domain. The correct sequence is:

```
1. Remove the dark offset         (BLC offset subtract)
   → now codes represent only photon-generated signal, starting from 0

2. Correct the sensor's nonlinear response curve  (OECF)
   → applied to offset-subtracted codes, consistent with calibration domain
   → output: a signal that is now linear with respect to photon count

3. Rescale to full bit-depth working range  (BLC linearisation)
   → stretch the linearised, OECF-corrected signal to [0, 2^bpp − 1]
   → now every downstream module (WB, CCM, gamma) works on a properly linearised signal
```

If instead you linearise before OECF, you are stretching a nonlinearly-encoded signal and then trying to apply a correction LUT that was calibrated for the unstretched domain — the correction will be wrong unless the LUT was specifically recalibrated for the post-linearisation codes, which is not the standard calibration workflow.

---

### Proposed Pipeline Restructuring

Split BLC into two stages and insert OECF between them:

```
Current:
  DPC → [BLC: offset + linearise] → OECF → ...

Proposed:
  DPC → [BLC offset only] → OECF → [BLC linearise] → Digital Gain → ...
```

In practice this can be implemented without creating a new module — BLC gains a `linearise` flag that is split into a separate call, or OECF is reordered to run before the linearisation step inside BLC. The cleanest approach is to make OECF a separate, standalone step that runs on offset-subtracted-but-not-yet-linearised codes, and move linearisation to a dedicated step after OECF.

---

### Impact on OECF LUT

With the proposed order, the OECF LUT is applied to offset-subtracted codes in the range `[0, sat−BL]`. The LUT must therefore be indexed and calibrated in that domain — **not** the full `[0, 2^bpp−1]` range.

If the sensor's characteristic is measured from raw codes (common in calibration workflows using a step-wedge or integrating sphere), the LUT naturally lives in the pre-BLC-offset domain. After offset subtraction, it should be re-indexed to start from 0. This is a one-time calibration adjustment and does not affect runtime behaviour.

---

### Summary of Proposed Final Early-Pipeline Order

```
Raw Bayer
  → Crop
  → DPC              (defect detection/correction on raw codes)
  → BLC offset       (subtract per-channel dark level, output float32)
  → OECF             (apply sensor linearisation LUT on offset-subtracted codes)
  → BLC linearise    (rescale [0, sat−BL] → [0, 2^bpp−1])
  → Digital Gain
  → LSC
  → BNR
  → White Balance
  → Demosaic
  → ...
```

This order correctly reflects the physical meaning of each correction and ensures that all downstream modules after OECF+linearise are working on a signal that is both (a) free of the systematic dark offset, (b) correctly linearised for the sensor's actual response curve, and (c) scaled to the full available dynamic range.

---

## [2026-03-06] ML/DL Denoising — Bayer domain vs. post-demosaic RGB domain

### The question

> If we want to add a machine learning / deep learning denoising stage, should it operate on the **Bayer raw domain** (before demosaicing) or the **post-demosaic RGB / pixel domain**?

Both are valid and both are used in the literature. The right answer depends on what you want the model to do and where you want it to sit in the pipeline. Here is a structured comparison.

---

### Option A — Bayer domain DL denoising

The model receives a single-channel (or 4-channel RGGB packed) Bayer image, still in the linear sensor domain.

**Advantages:**

1. **Clean, well-characterised noise model.** Raw noise is the sum of shot noise (Poisson, signal-dependent) and read noise (approximately Gaussian). This is the textbook signal-dependent noise model and DL architectures that are aware of the noise level map (e.g. CBDNet, PMN, ELD) can exploit this directly. In the RGB domain the noise has been distorted by demosaicing interpolation — it is no longer independent per-pixel.

2. **No demosaic artefact leakage into the model.** Demosaicing is a reconstruction step that necessarily creates correlations. If you denoise post-demosaic you need the model to learn to separate real detail from demosaic ringing/zipper artefacts — a harder and less stable task.

3. **Joint denoise + demosaic is possible.** A single model can take noisy Bayer → clean RGB in one forward pass, effectively replacing both the BNR module and the Demosaic module. This is architecturally the most efficient choice and is the approach taken by several state-of-the-art methods (see examples below).

4. **Keeps the linear domain.** The signal before demosaicing is still linear in photon count (assuming OECF has been applied). Non-linearities (gamma, tone mapping) make DL denoising harder because noise amplitudes become scene-dependent in a more complex way.

5. **Metadata is straightforward.** ISO, exposure time, and black level — which inform the noise level — are still meaningful and unchanged at this point in the pipeline.

**Disadvantages:**

1. **Bayer-specific training data required.** Public datasets are dominated by sRGB image pairs. Collecting paired noisy/clean raw pairs requires careful hardware setup (long-exposure ground truth or synthetic noise injection on clean raws). This is solvable but non-trivial.

2. **Spatial resolution is half in each axis.** A 4000×3000 Bayer grid is really a 2000×1500 grid per colour channel, so the model needs to reason about sub-pixel colour from the sparse mosaic pattern.

3. **Bayer pattern must be known.** The model either needs to be pattern-agnostic or retrained per pattern (RGGB / BGGR / GRBG etc.).

---

### Option B — post-demosaic RGB domain DL denoising

The model receives a full-resolution 3-channel RGB image after demosaicing (and ideally after white balance but before gamma/tone mapping, to keep linearity).

**Advantages:**

1. **Massive training data availability.** Datasets like SIDD, DND, CBSD68, Kodak, and many others are in sRGB or linear RGB. Pre-trained models (DnCNN, FFDNet, RIDNet, NAFNet, Restormer) can be fine-tuned rather than trained from scratch.

2. **Full-resolution spatial reasoning.** Each pixel now has all three colour channels; the model can reason about colour edges and textures at native resolution.

3. **Simpler to reason about and visualise.** The input/output are standard RGB images, which makes debugging and loss design straightforward.

4. **Modular.** Drop-in replacement without touching demosaicing. Useful if you want to keep a classical demosaic and only upgrade denoising.

**Disadvantages:**

1. **Noise is no longer independent per-pixel.** Demosaicing (especially Malvar-He-Cutler or other frequency-domain methods) creates spatial correlations in the noise. The model must implicitly learn to account for this, making it harder to apply noise level estimation correctly.

2. **Cannot undo demosaic artefacts.** If the demosaicing step introduced false colour or zipper edges in high-noise regions, a post-demosaic model sees those as signal and cannot recover.

3. **Two separate modules to maintain.** If you later improve demosaicing you may need to retrain the denoising model.

---

### Option C — End-to-end Bayer → RGB (best of both)

A single deep model takes a **noisy Bayer image** as input and produces a **clean, demosaiced RGB image** as output, learning to perform denoising and demosaicing jointly in one step.

This is the most powerful architecture and represents the current state of the art for cameras with known noise characteristics:

| Paper / Method | Notes |
|---|---|
| **CycleISP** (Zamir et al., CVPR 2020) | Two-branch: raw domain and RGB domain, trained end-to-end with unpaired data |
| **DeepJoint** (Gharbi et al., SIGGRAPH 2016) | Joint demosaic+denoise from Bayer, seminal work |
| **Unprocessing** (Brooks et al., CVPR 2019) | Synthesises training pairs from sRGB datasets by "unprocessing" through camera model — solves the training data problem |
| **ELD / PMN** (Wei et al.) | Extreme low-light, explicitly models read noise + shot noise in raw |
| **Noise2Noise / Noise2Void** | Self-supervised — can be applied to raw, no clean target needed |

The **Unprocessing** technique is particularly relevant here: it uses a camera model (similar to what Infinite-ISP implements) to convert clean sRGB images back to synthetic noisy raws, creating unlimited paired training data without any additional data collection.

---

### Recommendation for Infinite-ISP

Given the existing pipeline architecture, the recommended path is:

**Primary: Bayer domain, joint denoise + demosaic model**

```
DPC → BLC offset → OECF → BLC linearise → Digital Gain → LSC
  ↓
[DL model: noisy Bayer → clean RGB]    ← replaces BNR + Demosaic
  ↓
CCM → Gamma → CSC → LDCI → Sharpen → 2D NR → ...
```

This approach:
- Keeps all the linear-domain processing (BLC, OECF, gain, LSC) as classical modules where they are well understood
- Lets the DL model work in the clean linear Bayer domain with a known noise model
- Removes the hard dependency between BNR quality and Demosaic quality (they are solved jointly)
- Allows fallback to classical BNR + Demosaic when DL is disabled

**Secondary / interim: post-demosaic RGB denoising (if leveraging pre-trained models)**

If the goal is to quickly add an ML denoising stage by fine-tuning an existing pre-trained model (e.g. NAFNet, Restormer), the path of least resistance is to insert it between Demosaic and CCM:

```
... → Demosaic → [DL denoiser: RGB → RGB] → CCM → ...
```

This is lower risk and faster to prototype. The quality ceiling is lower because demosaic artefacts are baked in, but it provides a working DL denoising block with far less engineering effort.

---

### Training data strategy for Bayer domain models

The biggest practical obstacle is paired noisy/clean Bayer data. Three approaches in order of increasing quality:

1. **Synthetic noise injection** (easiest): Take a clean long-exposure raw → pack to 4-channel → add synthetic Poisson + Gaussian noise with estimated camera parameters. Fully controllable but distribution gap from real sensor noise.

2. **Unprocessing from sRGB** (balanced): Use a camera model calibrated with Infinite-ISP's own BLC/OECF/CCM/Gamma parameters to "undo" the ISP on clean sRGB images, producing synthetic noisy raws. This is well-matched to the specific pipeline.

3. **Real paired captures** (best quality): For each scene, capture many frames at high ISO + aligned low-ISO ground truth. Requires stable tripod and static scenes but yields real read-noise distribution. The existing `in_frames/` RAW files in the repo can seed this collection.

---

### Config placeholder

A future DL denoising module config might look like:

```yaml
dl_denoiser:
  enable: false
  mode: "bayer_joint"        # "bayer_joint" | "rgb_postdemosaic"
  model_path: "models/denoiser_bayer_v1.onnx"
  noise_model: "poisson_gaussian"
  iso_conditioning: true     # pass ISO as noise level hint to model
  fallback_classical: true   # use BNR+Demosaic if model not found
```

---

### Summary table

| Criterion | Bayer domain | Post-demosaic RGB | End-to-end joint |
|---|---|---|---|
| Noise model cleanliness | ✅ Clean Poisson+Gaussian | ❌ Correlated, complex | ✅ Clean (input) |
| Can fix demosaic artefacts | ✅ Yes (joint model) | ❌ No | ✅ Yes |
| Training data availability | ⚠️ Limited, needs synthesis | ✅ Abundant | ⚠️ Needs synthesis |
| Pre-trained model availability | ⚠️ Few Bayer models | ✅ Many RGB models | ⚠️ Few |
| Pipeline integration | Replaces BNR + Demosaic | Inserts after Demosaic | Replaces BNR + Demosaic |
| Engineering complexity | Medium | Low | High |
| Quality ceiling | High | Medium | Highest |

**Bottom line**: Bayer domain is the architecturally correct choice for a high-quality DL denoising path in Infinite-ISP. Post-demosaic RGB is the pragmatic shortcut if you want to prototype quickly with existing models. The end-to-end joint model is the long-term target.

> **⚑ REVISIT LATER** — Joint Bayer-domain ML denoising + demosaicing is a planned future module. When picking this up, start from the end-to-end architecture (noisy Bayer → clean RGB), use the unprocessing technique for training data synthesis (leveraging Infinite-ISP's own BLC/OECF/CCM/Gamma parameters as the camera model), and target replacing both the BNR and Demosaic modules in one step. Key references to review: CycleISP (Zamir et al., CVPR 2020), DeepJoint (Gharbi et al., SIGGRAPH 2016), Unprocessing (Brooks et al., CVPR 2019), PMN/ELD for physics-based noise modelling.

---

## [2026-03-07] Demosaicing — MHC recap + state-of-the-art landscape

### MHC understanding summary

MHC (Malvar-He-Cutler 2004) is a **guided linear interpolation** method. The key ideas:

- The Bayer pattern requires: 4× reconstruction of R and B (sampled at 1/4 density), 2× reconstruction of G (sampled on a quincunx grid at 1/2 density).
- The core assumption is **chrominance smoothness**: the color difference $D_R = R - G$ is a low-frequency signal. Therefore interpolating $D_R$ is far more accurate than interpolating R directly.
- MHC implements this as: $\hat{R} = R_\text{bilinear} + \alpha \cdot \nabla^2 G$, where the Laplacian of the dense G channel corrects for the curvature error in the sparse R bilinear estimate. The coefficient $\alpha = 1/2$ is Wiener-optimal over natural images.
- In frequency terms: MHC = LPF(R mosaic) + (1/2)·HPF(G), recovering high-frequency R content guided by G edge structure.
- Fails at **colour edges** (where $\nabla^2 R \neq \nabla^2 G$) and high-chroma fine textures — the smoothness assumption breaks and produces colour fringing / zipper artefacts.

---

### State-of-the-art demosaicing — classical and DL

#### Tier 1 — Frequency-domain framing (conceptual foundation)

**Alleysson et al. 2005 — Luminance-Chrominance spectral model**

The most elegant signal processing view of the Bayer problem. The full Bayer mosaic can be written as a single modulated signal:

```
mosaic(x,y) = L(x,y) + C1(x,y)·cos(πx) + C2(x,y)·cos(πy) + C3(x,y)·cos(πx)cos(πy)
```

where L is luminance (≈ G), and C1, C2, C3 are chrominance modulations that have been frequency-shifted to the Nyquist corners of the spectrum by the Bayer modulation carrier. Demosaicing is then **demodulation** — separating L from C1/C2/C3 using lowpass/bandpass filters.

This framing makes it clear why aliasing occurs (C signals overlap L in frequency at edges), why G is the luminance proxy (it's the baseband signal), and why all classical demosaicers are ultimately frequency separators with different filter designs. MHC is a 5×5 Wiener approximation of this demodulation.

---

#### Tier 2 — Adaptive / edge-directed classical methods

**AHD — Adaptive Homogeneity-Directed Demosaicing (Hirakawa & Parks, 2005)**

Instead of applying a fixed filter everywhere, AHD:
1. Produces two candidate interpolations at each pixel: one favouring horizontal edges, one vertical.
2. Computes a **homogeneity map** in CIELab space — counts how many neighbouring pixels have similar colour in each direction.
3. Selects the interpolation direction with higher homogeneity.

The key improvement: at edges, it picks the direction parallel to the edge (not crossing it), avoiding the zipper artefact that MHC produces when it incorrectly borrows Laplacian information across a colour edge. Used in dcraw and RawTherapee. Roughly +0.5–1.0 dB PSNR over MHC on standard benchmarks.

**LMMSE — Linear Minimum Mean Square Error (Zhang & Wu, 2005)**

Treats demosaicing as a **statistical estimation problem** in the frequency domain. Assumes G, R-G, and B-G are stationary random fields with known power spectral densities (estimated from natural image statistics). The LMMSE Wiener filter is then derived analytically per frequency bin:

```
Ĝ(ω) = [S_GG(ω) / (S_GG(ω) + S_nn(ω))] · G_mosaic(ω)
```

where S_GG is the signal PSD and S_nn is the noise PSD. Key advantages: (1) naturally handles noise — the noise PSD term acts as regularisation, suppressing noise at frequencies where SNR is low; (2) theoretically optimal under the stationarity assumption. Particularly good at high ISO. Roughly +1.0–1.5 dB over MHC.

**RCDLR / RI — Residual Interpolation (Kiku et al., 2014–2016)**

The idea: after a first-pass interpolation (e.g. bilinear), compute the **residual** (error between interpolated and known pixels at sampled locations), interpolate the residual separately, and add it back. Iterating this converges to a more accurate estimate. The residual is smoother than the original signal so its interpolation is more accurate. Achieves very high PSNR (+2–3 dB over MHC) with pure classical methods. RCDLR (Residual interpolation using Constant-Difference at Low Resolution) is the refined version.

---

#### Tier 3 — Deep learning methods (non-joint)

**DMCNN / DJDD (Deep Joint Demosaicing and Denoising, Gharbi et al., 2016 / Kokkinos 2018)**

CNN-based approaches trained end-to-end on Bayer → RGB. Key architectural advances:
- Input is the 4-channel RGGB Bayer (packed to H/2 × W/2 × 4) rather than the sparse single-channel mosaic — avoids the network having to learn to skip over empty pixels.
- Multi-scale features capture both fine texture and coarse colour structure.
- Jointly trained for denoising + demosaicing → a single model replaces both BNR and Demosaic modules.
- +3–5 dB over MHC on clean images; +5–8 dB on noisy images where the combined objective matters most.

**TENet — Texture Enhancement Network (Liu et al., 2020)**

Introduces a **texture prior** branch: extracts high-frequency texture guidance from G channel (similar in spirit to MHC's Laplacian term, but learned) and injects it into the R/B reconstruction branch. Outperforms plain CNNs because the network explicitly reasons about the luminance/chrominance decomposition.

**RAFT-based / Transformer demosaicers (2022–2024)**

Recent transformer architectures with global self-attention can capture long-range dependencies — important for textures like fabric weave or foliage where the repeating pattern is a strong prior for reconstruction. On Kodak and McM benchmarks these push PSNR above 44 dB, compared to MHC's ~38 dB.

---

### Performance comparison (Kodak dataset, sRGB PSNR dB, higher = better)

| Method | Type | PSNR (dB) | Colour artefacts | Noise robustness |
|---|---|---|---|---|
| Bilinear | Classical | ~33 | Severe zipper | Poor |
| MHC | Classical linear | ~38 | Moderate at colour edges | Moderate |
| AHD | Classical adaptive | ~39 | Low | Moderate |
| LMMSE | Classical statistical | ~39.5 | Low | **Good** (noise-aware) |
| RCDLR | Classical iterative | ~41 | Very low | Moderate |
| DeepJoint (CNN) | DL joint | ~43 | Very low | **Good** (joint trained) |
| TENet | DL guided | ~43.5 | Very low | Good |
| Transformer | DL attention | ~44–45 | Minimal | Good |

---

### Recommendation for Infinite-ISP — phased roadmap

**Phase 1 (near-term, pure classical, drop-in replacement):**

→ **LMMSE** is the best single upgrade. It is: (1) pure numpy/scipy, no training required; (2) naturally noise-aware, which matters because Infinite-ISP targets embedded/mobile cameras with moderate ISO; (3) theoretically grounded and easy to explain; (4) ~1.5 dB gain over MHC with similar runtime. Implement as `demosaic_method: "lmmse"` in config alongside `"mhc"`.

**Phase 2 (medium-term, higher quality classical):**

→ **AHD** or **RCDLR** as an optional high-quality mode. RCDLR gives the best classical quality but is iterative (2–3× slower). AHD is a good balance of quality and speed.

**Phase 3 (long-term, connects to BNR DL plan):**

→ **Joint DL denoise + demosaic** (Bayer → RGB in one model, as planned in the BNR ML note above). This replaces both BNR and Demosaic modules and is the architecture ceiling. The Phase 1/2 classical methods remain as the fallback when DL is disabled.

> **⚑ NEXT IMPLEMENTATION** — Add LMMSE as a second demosaic option. Add `demosaic_method` key to `configs.yml`. Keep MHC as default for backward compatibility; LMMSE as `"high_quality"` mode. Key reference: Zhang & Wu, "Color demosaicking via directional linear minimum mean square-error estimation," IEEE Trans. Image Process., 2005.

---

## [2026-03-07] Low-light capture mode — pipeline architecture decision

### Proposed plan (confirmed)

For a dedicated **low-light capture mode**, the agreed pipeline is:

```
Capture N Bayer frames
  → BLC offset × N  →  OECF × N  →  BLC linearise × N
  → Registration (homography on G channel)
  → Temporal average (weighted mean, motion-masked)          ← √N noise reduction
  → [DL joint model: averaged noisy Bayer → clean RGB]       ← residual denoise + demosaic
  → CCM → Gamma → CSC → LDCI → Sharpen → ...
```

This is the right architecture. The reasoning is below.

---

### Why average FIRST, then DL (not DL per-frame then average)

Two orderings are possible:

**Option A (correct): Average → DL**
- Averaging N frames in the Bayer domain reduces shot noise by √N and read noise by √N *before* the DL model sees the data.
- The DL model receives a much cleaner input — it only needs to handle residual noise, which is a simpler task requiring less model capacity.
- After averaging, the noise is closer to Gaussian (by CLT) even though per-frame shot noise is Poisson — the DL noise model is cleaner.
- No demosaic artifacts are introduced before merging (averaging is done on raw Bayer).

**Option B (wrong): DL per frame → average in RGB**
- The DL model must handle the full raw noise on each frame independently — harder task.
- Averaging in RGB after demosaicing mixes demosaic interpolation errors across frames — you average artifacts, not just noise.
- Loses the √N benefit on the DL input; the model does more work for worse results.

**Option A is unambiguously better.**

---

### Why this plan is strong

1. **Complementary noise reduction**: Multi-frame averaging targets the dominant low-light noise (shot noise, proportional to √signal — worst in dark areas). DL handles the residual floor noise (read noise, fixed additive, not reduced well by classical BNR at very low signal levels).

2. **Correct domain for both steps**: Averaging in Bayer (linear, before any non-linear processing) is mathematically clean. DL operates on the averaged Bayer (still linear domain, known noise model) — the best input it can receive.

3. **Single model replaces two modules**: The DL joint model eliminates both the classical BNR module and the Demosaic module in this mode, simplifying the pipeline and eliminating the risk of BNR→Demosaic artefact propagation.

4. **Training data strategy works cleanly**: The "unprocessing" technique generates synthetic noisy Bayer pairs, and we can simulate multi-frame averaging during training by averaging N synthetic noisy frames as the model input, keeping the training distribution matched to the runtime input.

---

### The ceiling: end-to-end burst DL

The plan above uses explicit (classical) registration + averaging before the DL model. There is a more powerful alternative:

**End-to-end burst DL**: N raw Bayer frames → single clean RGB, with no explicit registration step. The model learns alignment, merging, denoising, and demosaicing jointly.

Key references:
- **KPN (Kernel Prediction Networks)** — Mildenhall et al., CVPR 2018. Predicts per-pixel blend kernels across frames; handles misalignment implicitly.
- **BPN (Burst Photography Networks)** — Xia et al. Extends KPN with feature-space alignment.
- **DBSR / DeepRaw** — newer burst denoising in raw domain.

This is architecturally superior (no registration error propagation, globally optimal alignment) but significantly harder to train and deploy. It is the Phase 4 / long-term ceiling for low-light mode.

---

### Capture mode config sketch

```yaml
capture_mode:
  mode: "normal"          # "normal" | "low_light"

low_light_mode:
  multi_frame_enable: true
  n_frames: 8
  adaptive_n: true
  snr_target_db: 20.0
  registration_method: "homography"
  merge_method: "weighted_mean"
  motion_threshold: 0.05
  demosaic_backend: "dl_joint"    # "mhc" | "lmmse" | "dl_joint"
  dl_model_path: "models/lowlight_bayer_joint_v1.onnx"
  fallback_demosaic: "lmmse"      # used if DL model not found
```

---

### Phased implementation roadmap (updated)

| Phase | What | Mode | Complexity |
|---|---|---|---|
| 1 | LMMSE demosaic | Normal + low-light | Low — pure numpy |
| 2 | Classical multi-frame avg + LMMSE | Low-light | Medium — add registration |
| 3 | Classical multi-frame avg + DL joint denoise+demosaic | Low-light | High — needs model |
| 4 | End-to-end burst DL (N frames → RGB) | Low-light | Very high — needs burst training |

Phase 2 already gets most of the low-light quality gain with no ML dependency. Phase 3 is the agreed target. Phase 4 is the research ceiling.

> **⚑ REVISIT LATER** — Low-light mode implementation. Start at Phase 2 (classical multi-frame + LMMSE), validate pipeline and registration quality, then layer in Phase 3 DL joint model once the BNR ML work is underway. The two efforts (BNR ML note + this note) converge at the same DL model.

> **⚑ REVISIT LATER — Burst DL (Phase 4)** — End-to-end burst DL is architecturally superior to Phase 3 because it eliminates explicit frame registration entirely — no homography estimator, no IMU data, no motion threshold tuning, no ghosting suppression heuristics. The model implicitly learns to align frames from the multi-frame input, so the entire low-light pipeline collapses to: capture N frames → stack into (N, H, W) Bayer tensor → single DL forward pass → clean RGB. This is simpler at runtime despite being harder to train. Key references: KPN (Mildenhall et al., CVPR 2018 — per-pixel blend kernels across frames), DBSR (Bhat et al., ICCV 2021 — deep burst super-resolution, adaptable to denoising), BPN (Xia et al.). Training datasets: SID (See-in-the-Dark), MIT-Adobe FiveK burst, or synthetic burst pairs via unprocessing.

---

## [2026-03-07] CCM target primaries + Gamma EOTF — multi-format output

### CCM: target depends on output colour space

The CCM maps from camera native linear RGB to a target display colour space. The correct target primary set depends on the intended output format:

| Output format | Primary standard | White point | Gamut |
|---|---|---|---|
| sRGB / Rec.709 | BT.709 | D65 | ~35% of visible gamut — consumer SDR |
| Display P3 | DCI-P3 / D65 | D65 | ~45% — modern phones, MacBooks, iPad Pro |
| HDR10 | BT.2020 | D65 | ~75% — wide gamut HDR displays |
| DCI cinema | DCI-P3 | DCI white (~6300K) | Theatrical projection |

**Recommended architecture — XYZ as intermediate:**

Rather than calibrating a separate CCM for each output target, calibrate once to CIE XYZ (device-independent), then apply a fixed analytical matrix from XYZ to the target primaries. The second matrix is always derivable from the standard's primary chromaticity coordinates — no hardware required.

```
Camera native → [CCM_to_XYZ]  →  CIE XYZ  →  [M_XYZ_to_target]  →  target linear RGB
               (calibrated,       (device-independent     (fixed constant per
                per sensor)        intermediate)           output standard)
```

This decouples calibration (hardware-dependent) from output format selection (software switch). Switching from sRGB to HDR10 output is just changing `M_XYZ_to_target` — no recalibration needed.

---

### Gamma / EOTF — one per output format

The EOTF (electro-optical transfer function) encodes the linear light signal for a specific display technology. Each output format has its own:

**sRGB** — piecewise, SDR consumer displays (~100 nits peak):
```python
def eotf_srgb(L):
    return np.where(L <= 0.0031308,
                    12.92 * L,
                    1.055 * L**(1/2.4) - 0.055)
```

**Rec.709** — broadcast SDR (slightly different toe from sRGB):
```python
def eotf_rec709(L):
    return np.where(L < 0.018,
                    4.5 * L,
                    1.099 * L**0.45 - 0.099)
```

**PQ / ST.2084** — HDR10, absolute luminance 0–10,000 nits. Designed from Barten's contrast sensitivity function:
```python
def eotf_pq(L):          # L in nits [0, 10000]
    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    Lp = (L / 10000) ** m1
    return ((c1 + c2 * Lp) / (1 + c3 * Lp)) ** m2
```
Key property: **absolute encoding** — a code value always maps to a fixed number of nits regardless of display peak brightness. This is what makes HDR10 static metadata (MaxCLL, MaxFALL) meaningful.

**HLG / Hybrid Log-Gamma** — BBC/NHK HDR broadcast, scene-referred (relative, not absolute):
```python
def eotf_hlg(E):         # E = normalised linear signal [0, 1]
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    return np.where(E <= 0.5,
                    (E ** 2) / 3,
                    (np.exp((E - c) / a) + b) / 12)
```
Key property: **backward compatible** — an SDR display ignores the log portion and renders the sqrt section, producing a usable SDR image with no metadata. Preferred for live broadcast where the display type is unknown.

**Linear (γ=1.0)** — for scientific output, intermediate processing, or passing to a downstream tone mapper.

---

### CCM and EOTF are coupled — use output profiles

CCM target primaries and EOTF must always change together — they jointly define the output colour space. Applying a BT.709 CCM with PQ gamma produces wrong primaries. The clean solution is a named **output profile** that bundles both:

| Profile | CCM target | EOTF | Use case |
|---|---|---|---|
| `srgb` | BT.709 | sRGB piecewise | Default — SDR web, JPEG, PNG |
| `rec709` | BT.709 | Rec.709 | Broadcast SDR video |
| `display_p3` | DCI-P3 / D65 | sRGB | Modern phones, Mac displays |
| `hdr10` | BT.2020 | PQ / ST.2084 | HDR10 video, 10-bit+ output |
| `hlg` | BT.2020 | HLG | HDR broadcast, live streaming |
| `linear` | BT.709 or XYZ | γ = 1.0 | Further processing / ML input |

Config design:
```yaml
output:
  profile: "srgb"       # "srgb" | "rec709" | "display_p3" | "hdr10" | "hlg" | "linear"
  bit_depth: 8          # 8 for SDR, 10/12 for HDR
```

Selecting a profile automatically sets: (1) the XYZ→target matrix applied after CCM_to_XYZ, and (2) the EOTF applied in the Gamma module. No per-module config needed.

---

### Additional note for HDR: tone mapping before EOTF

For HDR output (PQ / HLG), a **tone mapping** step is needed before the EOTF to map the captured scene dynamic range (potentially 14–16 stops of raw dynamic range) into the display's peak luminance. The current LDCI module handles local contrast enhancement but is not a full HDR tone mapper. Proper HDR support requires:

1. **Absolute luminance estimation** — map relative sensor signal to nit values using known exposure metadata (ISO, shutter, aperture, ND).
2. **Global tone mapping** — compress scene luminance range to display range (e.g. Reinhard, ACES RRT, or a DL-based tone mapper).
3. **PQ / HLG encoding** — apply EOTF to the tone-mapped absolute luminance values.

This is a larger scope item — note for future HDR mode design.

**ALS (Ambient Light Sensor) and HDR — clarification:**

ALS is relevant to HDR but primarily on the **display rendering side**, not the capture/encoding side.

- **Capture side — ALS is NOT the primary input.** Absolute luminance calibration (nits per sensor code value) requires exposure metadata: ISO, shutter speed, aperture, and ND factor. These four values let you compute scene luminance in nits at any pixel. ALS measures ambient light falling on the device, not the scene luminance captured through the lens — it cannot substitute for exposure metadata.

- **Display side — ALS IS used.** HDR display pipelines (Dolby Vision, HLG) use ALS to adapt tone mapping to the viewing environment. HLG explicitly bakes this into its standard: system gamma γ_system = γ × L_W^(1/4), where L_W (display white level) can be driven by ALS. A film graded at 1000 nits peak looks correct in a dark room but washed out in bright sunlight — ALS lets the display compensate.

- **ALS as an optional ISP hint.** ALS can optionally feed back into the ISP tone mapper — knowing the ambient lux level can inform decisions like highlight compression aggressiveness in a mixed-light scene. This is an enhancement, not a requirement.

Dependency summary:
```
Exposure metadata (ISO, shutter, aperture, ND)  →  absolute scene luminance  →  PQ encoding  [REQUIRED]
ALS                                              →  display OOTF adaptation                  [display-side]
ALS                                              →  ISP tone mapping hint                    [optional enhancement]
```

> **⚑ REVISIT LATER** — Multi-format output profile system. Priority order: (1) implement XYZ intermediate architecture in CCM (trivial — add one matrix multiply), (2) add EOTF selector to gamma module (sRGB already present, add Rec.709, PQ, HLG, linear), (3) add `output_profile` config key that sets both together, (4) full HDR mode with tone mapping is a separate larger effort.

> **⚑ REVISIT LATER — HDR absolute luminance add-on** — The fundamental requirement for correct HDR encoding (PQ/HLG) is an absolute mapping from capture conditions to scene luminance in nits. This requires: (1) reading exposure metadata (ISO, shutter speed, aperture, ND factor) from the capture context, (2) computing absolute scene luminance per pixel from those values, (3) feeding that into the tone mapper before EOTF encoding. Without this absolute relationship, PQ encoding is syntactically valid but tonally incorrect — the display cannot know what the brightest pixel actually represents. Implement as an add-on module (`absolute_luminance_mapping`) that sits before tone mapping, gated by `hdr_mode: true` in config.

---

## [2026-03-07] Color Saturation Enhancement — dual-mode implementation plan

### Agreed design: flat gain + vibrance, both configurable

The color saturation module (operating on Cb/Cr after CSC RGB→YCbCr) will support two modes, selectable via config:

**Mode 1 — Flat gain** (simple, predictable):
```python
Cb' = s × Cb
Cr' = s × Cr
```
`s > 1` boosts all colours uniformly. Fast, one multiply per channel. Good for scenes where the user wants consistent global saturation control. Risk: already-vivid colours clip more easily — requires soft-knee gamut limiting on Cb/Cr magnitude.

**Mode 2 — Vibrance** (adaptive, more natural):
```python
chroma = sqrt(Cb² + Cr²)
gain = 1 + vibrance_strength × (1 - chroma / chroma_max)
Cb' = gain × Cb
Cr' = gain × Cr
```
Muted/pastel colours receive a stronger boost; already-saturated colours are boosted less. Skin tones (naturally low Cb/Cr magnitude) are partially self-protecting. Produces a more natural look and is less prone to clipping than flat gain at equivalent strength settings.

Both modes should include soft-knee chrominance limiting to prevent out-of-gamut values after YCbCr→RGB conversion:
```python
chroma_out = sqrt(Cb'² + Cr'²)
if chroma_out > chroma_limit:
    scale = soft_clip(chroma_out, chroma_limit) / chroma_out
    Cb', Cr' = Cb' * scale, Cr' * scale
```

**Config design:**
```yaml
color_saturation:
  enable: true
  mode: "flat"              # "flat" | "vibrance"
  saturation_gain: 1.2      # used in flat mode (1.0 = no change)
  vibrance_strength: 0.3    # used in vibrance mode (0.0 = no change)
  chroma_limit: 0.5         # soft-knee clamp on Cb/Cr magnitude (normalised)
```

**Implementation note:** Keep flat gain as the default for backward compatibility with existing configs. Vibrance mode is opt-in. Both share the same soft-knee limiter.

> **⚑ NEXT IMPLEMENTATION** — Add `mode`, `vibrance_strength`, and `chroma_limit` parameters to the color saturation module. Flat gain path already exists (or is trivial); add vibrance path as a branch on `mode`. Add soft-knee chrominance limiter to both paths.

---

## [2026-03-07] LDCI — upgrade to Guided Filter LTM as configurable feature

### Current LDCI limitations (CLAHE / bilinear tile interpolation)

The existing LDCI module uses CLAHE-style tile-based local histogram equalization with bilinear interpolation between tile transfer functions. Two fundamental problems:

1. **Bilinear tile blending guarantees C0 continuity but not local consistency.** Each tile's transfer function is derived from that tile's pixel distribution. Blending two transfer functions from tiles with different local statistics produces a result that satisfies neither tile's equalization objective. Smooth across boundaries ≠ perceptually correct across boundaries.

2. **Not edge-aware.** Bilinear weights are purely positional — a pixel on the boundary between a bright sky tile and a dark shadow tile gets a 50/50 blend of two radically different transfer functions, regardless of whether there's a hard edge at that position. This creates halo-like gradients along high-contrast edges.

### Agreed upgrade: Guided Filter LTM

Guided filter local tone mapping addresses both problems and is the agreed implementation target. It operates entirely on the Y channel in the YUV domain, leaving Cb/Cr untouched.

**Core idea:** decompose Y into a base layer (large-scale illumination) and a detail layer (local contrast), compress the base layer, amplify the detail layer, recompose.

```python
# Guided filter with Y as both input and guide
Y_base   = guided_filter(Y, Y, radius=r, eps=epsilon)   # smooth illumination estimate
Y_detail = Y - Y_base                                    # local contrast residual

Y_out = tone_curve(Y_base) + detail_gain * Y_detail      # compress base, preserve detail
```

The guided filter is edge-preserving by construction — it respects luminance edges when computing the smooth base layer, so there are no halos or cross-edge blending artefacts. No tiles, no interpolation, no boundary consistency problem.

**Why guided filter over other approaches:**

| Approach | No tile artefacts | Edge-aware | Complexity | Fits YUV pipeline |
|---|---|---|---|---|
| CLAHE bilinear | ⚠️ Smooth only | ❌ | Low | ✅ |
| Laplacian pyramid | ✅ | Partial | Medium | ✅ |
| Bilateral grid | ✅ | ✅ | Medium-High | ✅ |
| **Guided filter LTM** | ✅ | ✅ | **Medium** | ✅ |

Guided filter is the best balance — edge-aware, no tile artefacts, pure numpy/scipy implementable (no GPU required), well-understood with clean reference implementations.

**Config design — both modes configurable:**
```yaml
ldci:
  enable: true
  mode: "clahe"             # "clahe" | "guided_filter" — keep clahe for backward compat
  # CLAHE parameters (existing)
  clahe_clip_limit: 2.0
  clahe_tile_grid: [8, 8]
  # Guided filter LTM parameters (new)
  guided_radius: 64         # spatial smoothing radius for base layer (pixels)
  guided_eps: 0.01          # edge threshold (larger = smoother base, more detail preserved)
  detail_gain: 1.5          # amplification of detail layer (1.0 = no enhancement)
  base_compression: 0.7     # tone curve compression on base layer (< 1.0 = compress)
```

**Implementation note:** Keep `clahe` as the default mode for backward compatibility. `guided_filter` is the opt-in high-quality mode. The guided filter radius and epsilon are the two key tuning knobs — larger radius gives more global base estimation (stronger local contrast effect), smaller epsilon makes the filter more edge-sensitive.

Key reference: He, Sun & Tang, "Guided Image Filtering," IEEE TPAMI 2013. Fast O(N) implementation available via `cv2.ximgproc.guidedFilter` or pure numpy box-filter approximation.

> **⚑ NEXT IMPLEMENTATION** — Add `mode: "guided_filter"` path to LDCI module. Implement guided filter base/detail decomposition on Y channel. Expose `guided_radius`, `guided_eps`, `detail_gain`, `base_compression` in config. Keep existing CLAHE path untouched as default.

---

## [2026-03-07] Sharpening — edge-adaptive gain + pipeline order fix

### Two agreed changes

**Change 1 (correctness — must fix): swap pipeline order 2D NR → Sharpen**

Current order: `... → Sharpen → 2D NR → ...`
Correct order: `... → 2D NR → Sharpen → ...`

The current order is logically backwards. Sharpening amplifies all high-frequency content including residual noise. Then 2D NR runs and tries to suppress the noise — but in doing so it also smooths the sharpened edges, partially undoing the sharpening. The two modules are working against each other.

The correct order is to denoise first (clean signal) and then sharpen (enhance edges on the clean signal). This gives maximum sharpening effectiveness with minimum noise amplification. **This is a pipeline-level fix, not optional — it should be corrected before any other sharpening work.**

**Change 2 (quality — configurable): edge-adaptive sharpening gain**

Current USM applies a uniform gain to the unsharp mask everywhere:
```python
Y_sharp = Y + gain × (Y - gaussian_blur(Y))
```

This amplifies edges and noise equally. The fix is to gate the gain on local edge strength:

```python
mask      = Y - gaussian_blur(Y, sigma=r)
edge_str  = gradient_magnitude(Y)            # Sobel or Scharr magnitude
gain_map  = gain_max × clip(
                (edge_str - noise_floor) / (edge_max - noise_floor),
                0, 1)                        # ramp: 0 in flat/noisy → gain_max at strong edges
Y_sharp   = Y + gain_map × mask
```

Behaviour:
- Flat / low-signal regions: `gain_map ≈ 0` → noise not amplified
- Weak texture: partial gain → gentle enhancement
- Strong true edges: full `gain_max` → maximum sharpening

Optional halo suppression — clip the mask before applying gain:
```python
mask = clip(mask, -halo_limit, halo_limit)   # prevents overshoot halos at strong edges
```

**Config design — both modes, speed/quality trade-off:**
```yaml
sharpen:
  enable: true
  mode: "usm"               # "usm" | "adaptive" — usm is fast, adaptive is higher quality
  # Shared parameters
  radius: 1.5               # Gaussian blur radius for unsharp mask
  gain: 0.8                 # gain in usm mode; gain_max in adaptive mode
  # Adaptive mode only
  noise_floor: 0.02         # edge strength below which gain is zero (noise gate)
  edge_max: 0.15            # edge strength at which full gain is applied
  halo_limit: 0.1           # mask clip threshold for halo suppression (normalised)
```

**Speed trade-off:**
- `usm`: one Gaussian blur + one multiply — fastest, good enough for real-time embedded
- `adaptive`: adds Sobel gradient + gain map computation — roughly 2–3× slower but no noise amplification and no halos at aggressive gain settings

**Implementation note:** Keep `usm` as default. `adaptive` is the opt-in high-quality mode. Both operate on Y channel only in YUV domain.

> **⚑ IMMEDIATE FIX** — Swap pipeline execution order: run 2D NR before Sharpen in `infinite_isp.py`. This is a one-line change in the pipeline runner and must be done before any sharpening quality work — the current order partially defeats both modules.

> **⚑ NEXT IMPLEMENTATION** — Add `mode: "adaptive"` path to Sharpen module with edge-strength-gated gain map and optional halo limiter. Expose `noise_floor`, `edge_max`, `halo_limit` in config. Keep `usm` as default.

---

## [2026-03-07] 2D NR — dual-role design: full denoiser in classical mode, chroma-only polish in DL mode

### Key insight — 2D NR role depends on pipeline mode

2D NR exists to handle: (1) residual raw noise that survived BNR, and (2) structured demosaic artefacts. In the classical pipeline both problems are real and require a full spatial denoiser. But in the DL joint pipeline (Bayer → clean RGB), the model handles both by definition — it was trained to output clean demosaiced RGB from noisy Bayer. **In the DL path, the Y-channel denoising role of 2D NR is therefore redundant.**

The only remaining job in the DL path is an optional lightweight **chroma polish** — a cheap Gaussian on Cb/Cr to catch any subtle chroma noise the model may not have fully suppressed in edge cases (extreme low light, out-of-distribution scenes). This is near-zero cost and acts as a safety valve, not a primary denoiser.

### Two-mode pipeline architecture

**Classical mode** (no DL model):
```
BNR (Bayer domain)
  → Demosaic
  → CCM → Gamma → CSC
  → 2D NR: full bilateral on Y + aggressive Gaussian on Cb/Cr
  → Sharpen
```
Three complementary denoising stages (BNR, Demosaic-aware, 2D NR), each doing meaningful work.

**DL mode** (joint Bayer → clean RGB model):
```
[DL model: noisy Bayer → clean RGB]    ← replaces BNR + Demosaic + Y denoising
  → CCM → Gamma → CSC
  → 2D NR: chroma_only mode (lightweight Gaussian on Cb/Cr, Y untouched)
  → Sharpen
```
One model replaces three stages. 2D NR degrades gracefully to a minimal chroma polish — or is disabled entirely if the model handles chroma noise well enough.

### 2D NR module config — dual mode support

```yaml
noise_reduction_2d:
  enable: true
  mode: "bilateral"       # "bilateral" | "nlm" | "chroma_only" | "off"
                          # classical pipeline → "bilateral" or "nlm"
                          # DL pipeline       → "chroma_only" or "off"
  # Y channel (bilateral / nlm modes)
  y_spatial_sigma: 3.0
  y_intensity_sigma: 0.05
  # Cb/Cr channel (all modes except "off")
  chroma_sigma: 6.0       # aggressive Gaussian — human eye insensitive to chroma resolution
  # NLM mode only
  nlm_search_window: 21
  nlm_patch_size: 7
  nlm_h: 0.08
```

The capture/pipeline mode config selects the appropriate 2D NR mode automatically. No manual override needed in normal operation.

### Approach trade-offs (for classical mode selection)

| Mode | Edge preservation | Texture retention | Speed | Best for |
|---|---|---|---|---|
| Gaussian | ❌ None | ❌ Poor | ✅ Fastest | Cb/Cr only |
| Bilateral | ✅ Good | ⚠️ Moderate | ⚠️ Moderate | Default Y denoising |
| NLM | ✅ Excellent | ✅ Excellent | ❌ Slow | Offline / high-quality mode |
| chroma_only | N/A (Y skipped) | N/A | ✅ Cheapest | DL pipeline |
| off | N/A | N/A | ✅ None | DL pipeline (model handles all) |

> **⚑ NEXT IMPLEMENTATION** — Add `mode: "nlm"` and `mode: "chroma_only"` paths to 2D NR module. Wire pipeline mode (classical vs. DL) to auto-select 2D NR mode in config. Bilateral remains default for classical mode.

---

## [2026-03-07] YUV format — add 4:2:0 as configurable output choice

### Chroma subsampling format notation (J:a:b)

The J:a:b notation describes chroma sampling over a 4-pixel wide, 2-row reference block:
- **J** = luma (Y) samples per row — always 4
- **a** = Cb and Cr samples in the first row
- **b** = Cb and Cr samples in the second row (0 = second row shares first row's samples)

| Format | Horizontal sub | Vertical sub | Storage | Use case |
|---|---|---|---|---|
| 4:4:4 | None | None | 3 bytes/px | Professional / RAW / ML input |
| 4:2:2 | 2× | None | 2 bytes/px | Broadcast production, ProRes |
| 4:2:0 | 2× | 2× | 1.5 bytes/px | H.264, H.265, JPEG, consumer video |

### Why 4:2:0 is the dominant consumer format

The human eye's spatial resolution for colour (chrominance) is approximately 2–4× lower than for luminance. This is the same physiological reason 2D NR can aggressively smooth Cb/Cr without visible quality loss. 4:2:0 exploits this directly — halving chroma resolution both horizontally and vertically (quarter chroma area) saves 50% bandwidth versus 4:4:4, with essentially imperceptible quality loss on natural content.

### Agreed: add chroma subsampling format as a configurable output choice

```yaml
yuv_format:
  enable: true
  format: "420"       # "444" | "422" | "420"
                      # 444 → no subsampling (full quality, ML/professional use)
                      # 422 → horizontal subsample only (broadcast)
                      # 420 → horizontal + vertical subsample (consumer video/streaming)
```

The subsampling step sits at the very end of the pipeline (YUV Format module) after all processing is done on full-resolution Cb/Cr. All upstream modules (LDCI, 2D NR, saturation) continue to operate at full 4:4:4 chroma resolution — subsampling is a final output formatting step only.

> **⚑ NEXT IMPLEMENTATION** — Add `format` key to YUV format module config. Implement 4:2:2 (drop every other Cb/Cr column) and 4:2:0 (drop every other Cb/Cr column and row) as output options alongside existing behaviour. Keep current format as default for backward compatibility.

---

## [2026-03-07] Gamma ordering fix + Blue noise dithering for 8-bit output

### Fix 1 — Gamma must move later in the pipeline

**Rule: all signal processing must happen in linear light domain. Gamma is the last transform before output encoding.**

Current order (wrong): `... → CCM → Gamma → CSC → LDCI → Sharpen → 2D NR → ...`

Correct order:
```
... → CCM
    → LDCI              (linear domain — operating on true luminance ratios)
    → 2D NR             (linear domain — clean Poisson+Gaussian noise model)
    → Gamma             ← move here
    → CSC (RGB → YUV)   (standard YCbCr is defined on gamma-encoded RGB)
    → Color Saturation  (Cb/Cr)
    → Sharpen           (Y channel)
    → Scale
    → YUV Format (4:2:0)
    → Dither + Quantise
    → Output encode
```

Why each module belongs in linear domain:
- **LDCI**: local histogram equalization on gamma-encoded values double-processes shadows (gamma already expanded them). Linear domain LDCI operates on true luminance ratios — physically meaningful.
- **2D NR**: noise is Poisson+Gaussian in linear domain — clean model, correct bilateral sigma. After gamma, noise is non-stationary (shadows amplified, highlights compressed) — bilateral sigma is wrong.

Sharpen stays in YUV domain (after CSC) because operating on Y alone separates luminance from chrominance cleanly — an acceptable and standard engineering trade-off.

> **⚑ IMMEDIATE FIX** — Move Gamma module to execute after 2D NR and before CSC in `infinite_isp.py`. LDCI and 2D NR should see linear float input, not gamma-encoded values.

---

### Fix 2 — Blue noise dithering for 8-bit output

#### The problem: quantisation banding

The ISP pipeline works in float or high bit depth (12–16 bit). Truncating to 8-bit for PNG/JPEG output causes visible **banding** in smooth gradients (sky, skin) because adjacent float values round to the same uint8 value.

#### The mechanism: threshold dithering with a spatial map

For each pixel, the float value has an integer part and a fractional (sub-LSB) part. Instead of always rounding to nearest (losing the fractional information), look up a pre-computed spatial map to decide round-up or round-down:

```python
integer_part = floor(v * 255)
frac_part    = (v * 255) - integer_part    # sub-LSB value in [0, 1)

output = integer_part + 1  if map[x,y] < frac_part  else integer_part
# Equivalently:
output = floor(v * 255 + map[x, y])
```

**Bit depth preservation**: in any local N×N neighbourhood, the fraction of pixels that round up equals `frac_part` (because map values are uniformly distributed). So the spatial average of the output equals the original float value — the sub-LSB information is encoded in the spatial pattern of round-up/round-down decisions:

```
Effective bit depth from spatial averaging:
  single pixel:    8-bit
  2×2 average:    10-bit effective
  4×4 average:    12-bit effective
  8×8 average:    14-bit effective
```

#### Why blue noise is the optimal map

The map distribution determines what the dithering looks like — the bit depth preservation holds for any map with uniform marginal distribution:

| Map | Distribution | Visual result |
|---|---|---|
| Bayer matrix | Regular ordered | Artificial dot pattern |
| Uniform random | White noise | Grain at all frequencies |
| **Blue noise** | **High-freq only** | **Film grain — barely visible** |

Blue noise concentrates map energy at high spatial frequencies where the human eye's contrast sensitivity is lowest. The round-up/round-down pattern is **maximally spread** — no clustering at any scale — so the grain is isotropic, fine, and looks like natural film grain rather than structured noise.

#### Void-and-cluster: how the map is generated (once, offline)

The void-and-cluster algorithm (Ulichney 1993) generates a map with blue noise statistics:
1. Start with a random binary pattern
2. Iteratively find the tightest cluster (1s) and largest void (0s), swap them
3. Repeat until the pattern is maximally uniform at every scale
4. Assign rank values to build the full [0,1) threshold mask

The result is a tileable mask (128×128 or 256×256). At runtime: just tile and add — **no random number generation, faster than TPDF**.

#### Runtime implementation

```python
# One-time: pre-compute or load blue noise mask
blue_noise_mask = generate_void_and_cluster(size=128)   # values in [0, 1)

# Per-frame: dither and quantise
def encode_8bit_blue_noise(img_float, mask):
    H, W = img_float.shape[:2]
    tiled = np.tile(mask, (H//128+1, W//128+1))[:H, :W]
    dithered = img_float + (tiled[..., None] - 0.5) / 255.0   # broadcast over channels
    return np.clip(np.round(dithered * 255), 0, 255).astype(np.uint8)
```

Applied **after gamma** (in perceptually uniform [0,1] float space), after all processing, just before file encoding.

**Config:**
```yaml
output:
  format: "png"             # "png" | "jpeg" | "exr"
  bit_depth: 8
  dither: "blue_noise"      # "none" | "tpdf" | "blue_noise"
  jpeg_quality: 95
```

> **⚑ NEXT IMPLEMENTATION** — (1) Generate and store 128×128 void-and-cluster blue noise mask as a static asset. (2) Add dither step after all processing, before uint8 cast. (3) Expose `dither` key in output config. Blue noise as default for 8-bit output; none for 16-bit/EXR.

---

## [2026-03-07] Project summary — improvements over Infinite-ISP + project positioning

### Project positioning

This project extends Infinite-ISP as an **open-source, wearable/embedded-oriented ISP** with three differentiating pillars:

1. **Better pipeline** — algorithmic upgrades to every module, configurable speed/quality trade-offs throughout, and correctness fixes to the pipeline execution order
2. **Calibration guidance** — a first-class standalone calibration module that helps users characterise their specific sensor and generate the config parameters the pipeline needs
3. **Wearable/embedded orientation** — every configurable choice has a fast (embedded/real-time) default and a high-quality (offline/research) opt-in mode; low-light multi-frame and DL paths target the challenging conditions common in wearable cameras (small aperture, no optical stabilisation, limited compute)

---

### Pipeline correctness fixes (immediate — not optional)

| Fix | What | Impact |
|---|---|---|
| Pipeline order: 2D NR before Sharpen | One-line change in `infinite_isp.py` | Sharpening no longer partially undone by downstream NR |
| Gamma placement: after LDCI + 2D NR, before CSC | Move Gamma module later | LDCI and 2D NR operate in linear domain — correct noise model, correct luminance ratios |
| OECF ordering: BLC offset → OECF → BLC linearise | Split BLC into two sub-steps | OECF indexes into the domain it was calibrated for |

---

### Module-level algorithm upgrades

| Module | Current (Infinite-ISP) | Upgrade | Mode |
|---|---|---|---|
| DPC | Fixed `dp_threshold=80` | MAD-based adaptive threshold | Auto / fallback |
| BLC | Fixed config offsets | OB pixel per-capture black level | Config flag |
| LSC | Complete stub — returns image unchanged | Dual-mode: calibrated gain map + lensfun DB | Config |
| BNR | Single-frame JBF | Multi-frame average + JBF | `multi_frame_enable` |
| Demosaic | MHC only | LMMSE (high quality), AHD/RCDLR (research) | `demosaic_method` |
| LDCI | CLAHE bilinear tiles | Guided filter LTM (edge-aware, no tile artefacts) | `mode` |
| Color saturation | Flat Cb/Cr gain | + Vibrance (adaptive, protects skin tones) | `mode` |
| Sharpen | Uniform USM | + Edge-adaptive gain (no noise amplification) | `mode` |
| 2D NR | Single-mode bilateral | + NLM (high quality) + chroma-only (DL path) + off | `mode` |
| CCM | Direct to sRGB | XYZ intermediate → any target primaries | Architecture |
| Gamma | sRGB only | + Rec.709, PQ, HLG, linear | `output_profile` |
| YUV format | Fixed format | 4:4:4 / 4:2:2 / 4:2:0 selectable | `format` |

---

### New capabilities not in Infinite-ISP

| Capability | Description |
|---|---|
| Output profile system | Named bundles of CCM target + EOTF (sRGB, Display P3, HDR10, HLG, linear) |
| JPEG output | Pipeline completes to raw→JPEG, not just raw→PNG |
| Blue noise dithering | Spatial threshold dithering preserving bit depth in local averages |
| Low-light capture mode | Multi-frame Bayer average + configurable DL or classical demosaic |
| Calibration module | Standalone `calibration/` framework: BLC, DPC, OECF, LSC, WB, CCM, Gamma calibrators |

---

### Future DL modules (flagged, not yet implemented)

| Module | What | Replaces |
|---|---|---|
| Joint Bayer DL denoise + demosaic | Noisy Bayer → clean RGB in one model | BNR + Demosaic + 2D NR Y-channel |
| Burst DL (low-light) | N raw Bayer frames → clean RGB, implicit alignment | Multi-frame avg + registration + DL model |
| HDR absolute luminance | Exposure metadata → nit mapping before PQ encoding | Add-on to HDR output path |

---

### Comparison to baseline Infinite-ISP

| Dimension | Infinite-ISP | This project |
|---|---|---|
| Pipeline correctness | Gamma before LDCI/NR; Sharpen before NR | Fixed ordering throughout |
| Algorithm quality | Baseline implementations | Configurable baseline + high-quality modes |
| LSC | Stub — no implementation | Dual-mode with calibration support |
| Output formats | PNG only | PNG, JPEG, EXR; 4:4:4/4:2:2/4:2:0 |
| Low-light | No multi-frame | Multi-frame + DL path |
| Calibration | None | First-class standalone calibration module |
| Colour output targets | sRGB only | sRGB, Display P3, HDR10, HLG, linear |
| Dithering | None | Blue noise spatial dithering |
| DL integration | None | Planned joint model replacing BNR+Demosaic+2D NR |
| Target use case | Research / education | Research + wearable/embedded + production output |

---
