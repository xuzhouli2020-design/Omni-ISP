# Infinite-ISP: Core Pipeline Module Guide

A detailed description and tutorial for each module in the Infinite-ISP image signal processing pipeline. This guide covers all core deterministic pipeline modules in the order they are applied, from raw Bayer input to final YUV output.

---

## Pipeline Overview

```
RAW Bayer Input
      │
      ▼
  1.  Crop
  2.  Dead Pixel Correction
  3.  Black Level Correction
  4.  OECF (Opto-Electronic Conversion Function)
  5.  Digital Gain
  6.  Lens Shading Correction  [placeholder]
  7.  Bayer Noise Reduction
  8.  White Balance
  9.  Demosaic (CFA Interpolation)
  10. Color Correction Matrix
  11. Gamma Correction
  12. Color Space Conversion (RGB → YUV)
  13. Color Saturation Enhancement
  14. LDCI (Local Dynamic Contrast Enhancement)
  15. Sharpening / Edge Enhancement
  16. 2D Noise Reduction
  17. RGB Conversion (YUV → RGB)
  18. Scale
  19. YUV Format Conversion
      │
      ▼
  Output Image (RGB or YUV)
```

All module parameters are loaded from `config/configs.yml`. Each module reads `is_enable` to determine whether it runs.

---

## 1. Crop

**Module path:** `modules/crop/crop.py`

### Description

The Crop module trims the input RAW Bayer image to a smaller size before any processing begins. The critical constraint is **CFA pattern safety**: the number of rows and columns to crop must each be a multiple of 4 so that the Bayer mosaic pattern (e.g., RGGB) is not disturbed. If this constraint is violated, the module prints a warning and skips the crop.

Cropping is symmetric — equal amounts are removed from both sides (top/bottom and left/right), keeping the image centred.

### Algorithm

1. Compute `crop_rows = old_height - new_height` and `crop_cols = old_width - new_width`.
2. Check that both values are multiples of 4 (required to preserve Bayer alignment).
3. Slice the 2D array symmetrically: `img[crop_rows//2 : -crop_rows//2, crop_cols//2 : -crop_cols//2]`.
4. Update the `sensor_info` dictionary so downstream modules use the new dimensions.

### Config Parameters

| Parameter    | Type    | Description |
|-------------|---------|-------------|
| `is_enable`  | bool    | Enable or disable this module |
| `is_debug`   | bool    | Print debug info (rows/cols cropped, output shape) |
| `new_width`  | int     | Target width after cropping |
| `new_height` | int     | Target height after cropping |

**Example config:**
```yaml
crop:
  is_enable: true
  is_debug: false
  new_width: 1280
  new_height: 720
```

### Code Walkthrough

```python
from modules.crop.crop import Crop

# Module is instantiated with the raw image, platform config, sensor info, and crop params
crop_module = Crop(img, platform, sensor_info, parm_cro)

# execute() checks is_enable and calls apply_cropping() if True
output_img = crop_module.execute()
```

Inside `apply_cropping()`:

```python
crop_rows = self.old_size[0] - self.new_size[0]
crop_cols = self.old_size[1] - self.new_size[1]

# Pattern-safe check: both must be multiples of 4
if rows_to_crop % 4 == 0 and cols_to_crop % 4 == 0:
    img = img[
        rows_to_crop // 2 : -rows_to_crop // 2,
        cols_to_crop // 2 : -cols_to_crop // 2,
    ]
```

### Key Behaviours

- If `old_size == new_size`, no operation is performed.
- If the new size is larger than the input, the module logs an error and returns the original image.
- If crop amounts are not multiples of 4, the module refuses to crop to avoid pattern corruption.

---

## 2. Dead Pixel Correction (DPC)

**Module path:** `modules/dead_pixel_correction/dead_pixel_correction.py`
**Algorithm file:** `modules/dead_pixel_correction/dynamic_dpc.py`
**Reference:** [Yongji et al., IEEE 2020](https://ieeexplore.ieee.org/document/9194921)

### Description

Camera sensors can have pixels that are permanently stuck at a very high (hot) or very low (dead) value due to manufacturing defects or radiation damage. These pixels produce visual artefacts, usually appearing as bright or dark spots in the image. The Dead Pixel Correction module detects and corrects these pixels in the RAW Bayer domain using a gradient-based neighbourhood analysis.

### Algorithm

The **Dynamic DPC** algorithm works as follows:

1. **Pad** the image with a 2-pixel mirror border to handle edges.
2. For each pixel, examine its 3×3 neighbourhood.
3. Compute **gradient differences** in four directions (horizontal, vertical, diagonal) between the pixel and its same-channel neighbours.
4. If the maximum gradient exceeds `dp_threshold`, the pixel is flagged as defective.
5. Replace the defective pixel with the **directional neighbour** whose gradient was smallest (i.e., the most similar neighbour).

The threshold `dp_threshold` is the key tuning parameter: lowering it increases sensitivity (more pixels corrected), raising it decreases sensitivity.

### Config Parameters

| Parameter      | Type  | Description |
|---------------|-------|-------------|
| `is_enable`    | bool  | Enable or disable DPC |
| `is_debug`     | bool  | Print debug logs |
| `dp_threshold` | int   | Detection threshold; lower = more aggressive correction (default: 80) |

**Example config:**
```yaml
dead_pixel_correction:
  is_enable: true
  dp_threshold: 80
  is_debug: false
```

### Code Walkthrough

```python
from modules.dead_pixel_correction.dead_pixel_correction import DeadPixelCorrection

dpc = DeadPixelCorrection(img, sensor_info, parm_dpc, platform)
corrected_img = dpc.execute()
```

Inside the module, the `DynamicDPC` class does the heavy lifting:

```python
# In dynamic_dpc.py — conceptual flow:
img_pad = np.pad(self.img, (2, 2), "reflect")   # mirror pad

for each pixel (r, c):
    neighbours = extract_3x3_neighbourhood(img_pad, r, c)
    gradients  = compute_directional_gradients(pixel, neighbours)

    if max(gradients) > dp_threshold:
        # Replace with the neighbour with minimum gradient
        pixel = neighbour_with_min_gradient
```

### Tuning Tips

- A `dp_threshold` of 80 works well for typical 12-bit sensors.
- If too many good pixels are being corrected (over-correction), increase the threshold.
- If dead pixels remain visible in the output, decrease the threshold.

---

## 3. Black Level Correction (BLC)

**Module path:** `modules/black_level_correction/black_level_correction.py`

### Description

Every image sensor has a non-zero output even when no light falls on it. This is called the **black level** or dark current offset. Without correction, the darkest areas of an image appear grey rather than black, reducing contrast and distorting colour. Black Level Correction subtracts a per-channel offset from the raw Bayer data to drive truly dark pixels toward zero.

An optional **linearisation** step can also rescale the data so that the black level maps to 0 and the sensor's saturation level maps to the full bit range `(2^bpp - 1)`.

### Algorithm

For each Bayer channel (R, Gr, Gb, B):

```
corrected = pixel - offset
```

If linearisation is enabled:

```
linearised = corrected / (sat_level - offset) × (2^bpp - 1)
```

The result is clipped to `[0, 2^bpp - 1]` and stored as `uint16`.

### Config Parameters

| Parameter    | Type  | Description |
|-------------|-------|-------------|
| `is_enable`  | bool  | Enable or disable BLC |
| `r_offset`   | int   | Red channel black level offset |
| `gr_offset`  | int   | Gr (green-red) channel offset |
| `gb_offset`  | int   | Gb (green-blue) channel offset |
| `b_offset`   | int   | Blue channel offset |
| `is_linear`  | bool  | Enable linearisation |
| `r_sat`      | int   | Red channel saturation level |
| `gr_sat`     | int   | Gr saturation level |
| `gb_sat`     | int   | Gb saturation level |
| `b_sat`      | int   | Blue saturation level |

**Example config:**
```yaml
black_level_correction:
  is_enable: true
  r_offset: 200
  gr_offset: 200
  gb_offset: 200
  b_offset: 200
  is_linear: true
  r_sat: 4095
  gr_sat: 4095
  gb_sat: 4095
  b_sat: 4095
```

### Code Walkthrough

```python
from modules.black_level_correction.black_level_correction import BlackLevelCorrection

blc = BlackLevelCorrection(img, platform, sensor_info, parm_blc)
corrected_img = blc.execute()
```

Inside `apply_blc_parameters()` for an RGGB Bayer pattern:

```python
raw = np.float32(self.img)

# Subtract per-channel offsets
raw[0::2, 0::2] -= r_offset   # R pixels
raw[0::2, 1::2] -= gr_offset  # Gr pixels
raw[1::2, 0::2] -= gb_offset  # Gb pixels
raw[1::2, 1::2] -= b_offset   # B pixels

# Optional linearisation
if is_linear:
    raw[0::2, 0::2] = raw[0::2, 0::2] / (r_sat - r_offset) * (2**bpp - 1)
    # ... (same for other channels)

raw_blc = np.uint16(np.clip(raw, 0, 2**bpp - 1))
```

### Key Behaviours

- Offsets are sensor-specific and should be determined by calibration (measuring sensor output in darkness).
- The module handles all four Bayer patterns: `rggb`, `bggr`, `grbg`, `gbrg`.
- Clipping to `[0, max]` prevents negative values from corrupting downstream modules.

---

## 4. Opto-Electronic Conversion Function (OECF)

**Module path:** `modules/oecf/oecf.py`

### Description

Real camera sensors do not respond linearly to light. The **Opto-Electronic Conversion Function (OECF)** describes the nonlinear relationship between the number of photons hitting a pixel and the digital value it produces. This module applies a per-channel **lookup table (LUT)** to correct that non-linearity, converting sensor-specific response curves into a known, predictable response.

The LUT is obtained through sensor calibration using standardised measurement procedures (e.g., shooting a step wedge chart under controlled lighting).

### Algorithm

The correction is a simple LUT indexing operation:

```
corrected_pixel = LUT[raw_pixel_value]
```

A separate LUT can be provided for each Bayer channel (R, Gr, Gb, B). Currently, the module duplicates the `r_lut` for all channels — separate LUTs per channel can be added in the config for finer control.

### Config Parameters

| Parameter  | Type        | Description |
|-----------|-------------|-------------|
| `is_enable`| bool        | Enable or disable OECF |
| `r_lut`    | list of int | LUT for the Red channel (length must equal `2^bit_depth`) |

**Example config (identity LUT for 8-bit — no correction):**
```yaml
oecf:
  is_enable: false
  r_lut: [0, 1, 2, 3, ..., 255]
```

**A real OECF LUT** would have non-linear values, e.g., applying a linearising curve so that sensor response maps to a linear light scale.

### Code Walkthrough

```python
from modules.oecf.oecf import OECF

oecf = OECF(img, platform, sensor_info, parm_oecf)
corrected_img = oecf.execute()
```

Inside `apply_oecf()`:

```python
rd_lut = np.uint16(np.array(self.parm_oecf["r_lut"]))
# (gr_lut, gb_lut, bl_lut are currently duplicated from r_lut)

# For RGGB pattern:
raw_oecf[0::2, 0::2] = rd_lut[raw[0::2, 0::2]]   # R
raw_oecf[0::2, 1::2] = gr_lut[raw[0::2, 1::2]]   # Gr
raw_oecf[1::2, 0::2] = gb_lut[raw[1::2, 0::2]]   # Gb
raw_oecf[1::2, 1::2] = bl_lut[raw[1::2, 1::2]]   # B
```

### Key Behaviours

- The LUT must have exactly `2^bit_depth` entries (e.g., 4096 entries for 12-bit images).
- If the sensor has a linear response, the identity LUT `[0, 1, 2, ..., 2^bpp-1]` (which does nothing) should be used.
- This module is typically disabled when `black_level_correction.is_linear = true` already linearises the data.

---

## 5. Digital Gain

**Module path:** `modules/digital_gain/digital_gain.py`

### Description

Digital Gain amplifies the sensor signal in software after digitisation. This is used to increase image brightness when the scene is dark, serving as the software complement to analogue gain in the sensor. It also interfaces with the **Auto Exposure (AE)** feedback loop: when AE determines the image is under- or over-exposed, it adjusts which gain from a predefined `gain_array` is applied.

Unlike analogue gain (applied in the sensor before ADC), digital gain amplifies noise along with signal, so it is best used sparingly.

### Algorithm

1. Read `current_gain` (index into `gain_array`) and optionally update it based on AE feedback.
2. Multiply all pixels by `gain_array[current_gain]`.
3. Clip to `[0, 2^bpp - 1]`.

AE feedback convention:
- `ae_feedback == 0`: use the default gain (no adjustment).
- `ae_feedback < 0`: image is underexposed → increment gain index (brighter).
- `ae_feedback > 0`: image is overexposed → decrement gain index (darker).

### Config Parameters

| Parameter      | Type         | Description |
|---------------|--------------|-------------|
| `is_debug`     | bool         | Print the applied gain value |
| `is_auto`      | bool         | Use AE feedback to adjust gain index |
| `gain_array`   | list of float| Ordered list of available gain multipliers |
| `current_gain` | int          | Starting index into `gain_array` |
| `ae_feedback`  | int          | AE correction signal (`-1`, `0`, or `1`) |

**Example config:**
```yaml
digital_gain:
  is_enable: true
  is_debug: false
  is_auto: false
  gain_array: [1.0, 1.5, 2.0, 3.0, 4.0]
  current_gain: 0
  ae_feedback: 0
```

### Code Walkthrough

```python
from modules.digital_gain.digital_gain import DigitalGain

dg = DigitalGain(img, platform, sensor_info, parm_dga)
output_img, current_gain = dg.execute()
```

Inside `apply_digital_gain()`:

```python
self.img = np.float32(self.img)

if self.is_auto:
    if self.ae_feedback < 0:   # underexposed
        self.current_gain = min(len(self.gains_array) - 1, self.current_gain + 1)
    elif self.ae_feedback > 0: # overexposed
        self.current_gain = max(0, self.current_gain - 1)

self.img = self.gains_array[self.current_gain] * self.img
self.img = np.uint16(np.clip(self.img, 0, (2**bpp) - 1))
```

### Key Behaviours

- This module is always executed (it cannot be disabled), because gain = 1.0 at `current_gain = 0` is the default neutral operation.
- The `execute()` method returns both the image and the final `current_gain` index (for AE state tracking).

---

## 6. Lens Shading Correction (LSC)

**Module path:** `modules/lens_shading_correction/lens_shading_correction.py`

### Description

Optical lenses cause a characteristic brightness fall-off from the centre of the image to the edges, known as **vignetting** or **lens shading**. This occurs because lens elements cannot deliver uniform light throughput across the full field of view. Lens Shading Correction compensates for this by applying gain that increases towards the image edges.

> **Status: This module is currently a placeholder in Infinite-ISP v1.1. The correction is not yet implemented.** When implemented, it will require a per-channel gain map obtained through calibration (shooting a flat, uniform field and measuring the brightness falloff pattern).

### Planned Config Parameters

| Parameter  | Type  | Description |
|-----------|-------|-------------|
| `is_enable`| bool  | Enable or disable LSC |

---

## 7. Bayer Noise Reduction (BNR)

**Module path:** `modules/bayer_noise_reduction/bayer_noise_reduction.py`
**Algorithm file:** `modules/bayer_noise_reduction/joint_bf.py`
**Reference:** [Tan et al. — Green Channel Guiding Denoising](https://www.researchgate.net/publication/261753644_Green_Channel_Guiding_Denoising_on_Bayer_Image)

### Description

Sensor noise is most effectively reduced **before demosaicing**, while the image is still in Bayer format. Noise reduction after demosaicing risks introducing colour artefacts, because the interpolation step mixes noise between channels.

The BNR module uses a **Joint Bilateral Filter (JBF)**, where the Green channel of the Bayer mosaic serves as the guidance signal for filtering R and B channels. The green channel is selected as a guide because it is sampled twice as densely as red and blue (in most Bayer patterns) and thus has higher spatial frequency fidelity.

### Algorithm — Joint Bilateral Filter

A bilateral filter is an edge-preserving smoothing filter. Instead of a simple Gaussian average, it weights each neighbour by **both** spatial proximity (Gaussian kernel) and **intensity similarity** (range kernel):

```
filtered_pixel = Σ G_spatial(dist) × G_range(|pixel - neighbour|) × neighbour
                 ─────────────────────────────────────────────────────────────
                          normalisation factor
```

In the joint variant, the range kernel uses the **green channel** intensities even when filtering R or B:

```
filtered_R_pixel = Σ G_spatial(dist) × G_range(|G_pixel - G_neighbour|) × R_neighbour
```

This preserves edges (where green intensity changes sharply) while smoothing noise within uniform regions.

### Config Parameters

| Parameter     | Type  | Description |
|--------------|-------|-------------|
| `is_enable`   | bool  | Enable or disable BNR |
| `filt_window` | int   | Filter kernel size (must be odd, e.g., 3, 5, 7) |
| `r_std_dev_s` | float | Red channel spatial kernel σ (larger = more blur) |
| `r_std_dev_r` | float | Red channel range kernel σ (larger = more edge preservation) |
| `g_std_dev_s` | float | Green channel spatial σ |
| `g_std_dev_r` | float | Green channel range σ |
| `b_std_dev_s` | float | Blue channel spatial σ |
| `b_std_dev_r` | float | Blue channel range σ |

**Example config:**
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
```

### Code Walkthrough

```python
from modules.bayer_noise_reduction.bayer_noise_reduction import BayerNoiseReduction

bnr = BayerNoiseReduction(img, sensor_info, parm_bnr, platform)
denoised_img = bnr.execute()
```

Inside `joint_bf.py`, the JBF logic extracts each Bayer channel, applies the bilateral filter guided by green, then reconstructs the Bayer image:

```python
# Conceptual flow
for each non-green pixel (R or B):
    guided_weight = gaussian_range(|green[r,c] - green[nr,nc]|, std_dev_r)
    spatial_weight = gaussian_spatial(distance, std_dev_s)
    weight = guided_weight * spatial_weight
    filtered_pixel = sum(weight * neighbour) / sum(weight)
```

### Tuning Tips

- `std_dev_s` controls the **spatial blur radius**. Increase for more smoothing; decrease to limit blur to a tighter neighbourhood.
- `std_dev_r` controls **edge sensitivity**. Larger values preserve fewer edges (more smoothing); smaller values make the filter more conservative near edges.
- For low-noise sensors, this module can be disabled without visible impact.

---

## 8. White Balance (WB)

**Module path:** `modules/white_balance/white_balance.py`

### Description

Colour temperature of light sources varies widely — tungsten lights appear orange, overcast skies appear blue. A camera sensor records this cast faithfully, but human vision adapts to perceive whites as white under any illuminant. White Balance corrects for this by applying per-channel gain multipliers to bring the image to a neutral, perceptually accurate white point.

In **manual** mode, the user provides fixed R and B gains (green is the reference, so its gain is implicitly 1). In **auto** mode (`is_auto: true`), the gains are computed by the Auto White Balance (AWB) module.

### Algorithm

For each channel in the Bayer mosaic, multiply by the corresponding gain:

```
R_corrected = R_raw × r_gain
B_corrected = B_raw × b_gain
G unchanged (gain = 1.0)
```

Result is clipped to `[0, 2^bpp - 1]`.

### Config Parameters

| Parameter  | Type  | Description |
|-----------|-------|-------------|
| `is_enable`| bool  | Enable or disable WB |
| `is_auto`  | bool  | Use AWB-computed gains (overrides manual gains) |
| `is_debug` | bool  | Print applied gains |
| `r_gain`   | float | Red channel gain multiplier |
| `b_gain`   | float | Blue channel gain multiplier |

**Example config:**
```yaml
white_balance:
  is_enable: true
  is_auto: false
  is_debug: false
  r_gain: 1.8
  b_gain: 1.5
```

### Code Walkthrough

```python
from modules.white_balance.white_balance import WhiteBalance

wb = WhiteBalance(img, platform, sensor_info, parm_wbc)
balanced_img = wb.execute()
```

Inside `apply_wb_parameters()` for RGGB:

```python
self.raw = np.float32(self.img)

self.raw[::2, ::2]  *= redgain   # R pixels
self.raw[1::2, 1::2] *= bluegain  # B pixels
# Gr and Gb pixels are left unchanged (green reference)

raw_whitebal = np.uint16(np.clip(self.raw, 0, (2**self.bpp) - 1))
```

### Key Behaviours

- All four Bayer patterns (RGGB, BGGR, GRBG, GBRG) are supported; the module selects the correct pixel positions automatically.
- When `is_auto = true`, the AWB module writes its computed gains back into `parm_wbc` before this module runs.
- Gains > 1 boost a channel; gains < 1 attenuate it.

---

## 9. Demosaic (CFA Interpolation)

**Module path:** `modules/demosaic/demosaic.py`
**Algorithm file:** `modules/demosaic/malvar_he_cutler.py`
**Reference:** [Malvar, He, Cutler (2004) — High-Quality Linear Interpolation for Demosaicing](https://www.ipol.im/pub/art/2011/g_mhcd/article.pdf)

### Description

A Bayer sensor captures only **one colour per pixel** in a mosaic pattern (typically RGGB). To reconstruct a full-colour RGB image, the missing two colour values at each pixel must be **interpolated** from neighbouring pixels. This process is called **demosaicing** or CFA (Color Filter Array) interpolation.

Infinite-ISP uses the **Malvar-He-Cutler (MHC)** algorithm, which achieves high quality through gradient-corrected bilinear interpolation using specialised 5×5 convolution kernels — one set for each of the five pixel type positions in the Bayer pattern.

### Algorithm — Malvar-He-Cutler

The MHC algorithm defines five distinct 5×5 linear filters corresponding to the five pixel positions in the Bayer 2×2 tile (R, G at R rows, G at B rows, B, and the border). For each colour channel at each pixel:

1. Identify the pixel's position in the Bayer mosaic.
2. Apply the appropriate 5×5 kernel to the neighbourhood.
3. Sum the weighted neighbours to obtain the interpolated channel value.

The gradient-correction terms in the kernels help preserve sharp edges and reduce the characteristic "zipper" artefacts of simpler bilinear demosaicing.

### Config Parameters

Demosaicing is an **essential module** and cannot be disabled.

| Parameter  | Type  | Description |
|-----------|-------|-------------|
| `is_save`  | bool  | Save intermediate output to file |

### Code Walkthrough

```python
from modules.demosaic.demosaic import Demosaic

demosaic = Demosaic(img, platform, sensor_info, parm_dga)
rgb_img = demosaic.execute()
```

Step 1 — Generate boolean CFA masks for each channel:

```python
def masks_cfa_bayer(self):
    # Creates a (H, W) boolean mask for R, G, and B
    # True where that channel is physically measured
    channels = {c: np.zeros(self.img.shape, dtype=bool) for c in "rgb"}
    for channel, (y, x) in zip(pattern, [(0,0), (0,1), (1,0), (1,1)]):
        channels[channel][y::2, x::2] = True
    return tuple(channels[c] for c in "rgb")
```

Step 2 — Apply MHC interpolation:

```python
# In malvar_he_cutler.py:
# Five 5x5 kernels are convolved with the single-channel Bayer image.
# Results are combined per-mask to reconstruct all three channels at every pixel.
demos_out = mal.apply_malvar()
demos_out = np.clip(demos_out, 0, 2**bit_depth - 1).astype(np.uint16)
```

### Output

The input is a 2D array (H × W) Bayer image. The output is a 3D array (H × W × 3) RGB image.

---

## 10. Color Correction Matrix (CCM)

**Module path:** `modules/color_correction_matrix/color_correction_matrix.py`
**Reference:** [Imatest Color Matrix Documentation](https://www.imatest.com/docs/colormatrix/)

### Description

Even after demosaicing and white balance, camera colours do not exactly match the colours a human observer would perceive. This is because the spectral sensitivities of camera sensors differ from human colour vision (the CIE XYZ colour matching functions). The **Color Correction Matrix (CCM)** is a 3×3 linear transformation that corrects these deviations, mapping camera RGB to a more perceptually accurate colour space.

The CCM is sensor-specific and must be determined by calibration, typically by photographing a standardised colour chart (e.g., a Macbeth ColorChecker) under controlled lighting and computing the least-squares best-fit matrix.

### Algorithm

```
[R_out]   [ccm11 ccm12 ccm13]   [R_in]
[G_out] = [ccm21 ccm22 ccm23] × [G_in]
[B_out]   [ccm31 ccm32 ccm33]   [B_in]
```

Following the **Imatest convention**, the rows of the matrix sum to 1 (maintaining white balance), and negative entries are expected and necessary for wide-gamut correction.

### Config Parameters

| Parameter         | Type          | Description |
|------------------|---------------|-------------|
| `is_enable`       | bool          | Enable or disable CCM |
| `corrected_red`   | list of 3 float | Row 1 of CCM (weights for R output) |
| `corrected_green` | list of 3 float | Row 2 of CCM (weights for G output) |
| `corrected_blue`  | list of 3 float | Row 3 of CCM (weights for B output) |

**Example config (identity matrix — no correction):**
```yaml
color_correction_matrix:
  is_enable: true
  corrected_red:   [1.0,  0.0,  0.0]
  corrected_green: [0.0,  1.0,  0.0]
  corrected_blue:  [0.0,  0.0,  1.0]
```

**Typical calibrated matrix:**
```yaml
  corrected_red:   [ 1.7,  -0.5, -0.2]
  corrected_green: [-0.15,  1.4, -0.25]
  corrected_blue:  [-0.05, -0.35, 1.4]
```

### Code Walkthrough

```python
from modules.color_correction_matrix.color_correction_matrix import ColorCorrectionMatrix

ccm = ColorCorrectionMatrix(img, platform, sensor_info, parm_ccm)
corrected_img = ccm.execute()
```

Inside `apply_ccm()`:

```python
ccm_mat = np.float32([corrected_red, corrected_green, corrected_blue])  # 3×3

# Normalise from bit-depth range to [0, 1]
img_norm = np.float32(self.img) / (2**bit_depth - 1)

# Reshape to (N, 3) for matrix multiplication
img_flat = img_norm.reshape((-1, 3))

# Apply: output = input × CCM^T  (Imatest convention: column-major)
out = np.matmul(img_flat, ccm_mat.T)

# Clip (negative values possible from CCM), reshape, rescale
out = np.clip(out, 0, 1).reshape(img_norm.shape)
out = np.uint16(out * (2**bit_depth - 1))
```

### Key Behaviours

- The matrix is applied in the **float domain** on normalised `[0, 1]` values to avoid integer overflow.
- Clipping after CCM is essential because negative off-diagonal weights can produce values outside `[0, 1]`.
- Row sums equal to 1 preserves the white point established by white balance.

---

## 11. Gamma Correction

**Module path:** `modules/gamma_correction/gamma_correction.py`

### Description

After sensor data processing, pixel values represent **linear light** — a value twice as large corresponds to twice as many photons. However, human vision is not linear: it is approximately logarithmic, more sensitive to changes in dark tones than bright ones. Displays also expect non-linear encoded signals.

**Gamma correction** applies a nonlinear tone curve to map linear light values to display-ready values. The standard monitor gamma is approximately 2.2, meaning the encoding gamma is approximately 1/2.2 ≈ 0.45:

```
output = input^(1/2.2)
```

In practice, Infinite-ISP uses a **lookup table (LUT)** for efficiency rather than computing the power function per pixel. The LUT is pre-computed and loaded from the config file, supporting 8, 10, 12, or 14-bit images.

### Config Parameters

| Parameter      | Type          | Description |
|---------------|---------------|-------------|
| `is_enable`    | bool          | Enable or disable gamma correction |
| `gamma_lut_8`  | list of int   | 256-entry LUT for 8-bit images |
| `gamma_lut_10` | list of int   | 1024-entry LUT for 10-bit images |
| `gamma_lut_12` | list of int   | 4096-entry LUT for 12-bit images |
| `gamma_lut_14` | list of int   | 16384-entry LUT for 14-bit images |

**Example config (standard gamma 2.2 LUT generation):**
```python
# To generate a standard gamma 2.2 LUT for 8-bit:
import numpy as np
lut = np.linspace(0, 255, 256)
lut = np.uint8(np.round(255 * (lut / 255) ** (1 / 2.2)))
```

### Code Walkthrough

```python
from modules.gamma_correction.gamma_correction import GammaCorrection

gc = GammaCorrection(img, platform, sensor_info, parm_gmm)
gamma_img = gc.execute()
```

Inside `apply_gamma()`:

```python
# Select the LUT matching the sensor bit depth
if self.bit_depth == 12:
    lut = np.uint16(np.array(self.parm_gmm["gamma_lut_12"]))

# Apply LUT — a single vectorised array indexing operation
gamma_img = lut[self.img]
```

### Key Behaviours

- The LUT indexing `lut[self.img]` is highly efficient and vectorised by NumPy.
- Any arbitrary tone curve (not just gamma 2.2) can be encoded in the LUT.
- The LUT must have exactly `2^bit_depth` entries.

---

## 12. Color Space Conversion (CSC)

**Module path:** `modules/color_space_conversion/color_space_conversion.py`
**References:** BT.709, BT.601/407 standards

### Description

After the RGB processing chain, the pipeline converts the image from **RGB colour space** to **YCbCr (YUV) colour space**. This separation of **luma (Y)** from **chroma (Cb, Cr)** is essential for video encoding and enables several subsequent processing steps (such as noise reduction and sharpening) to operate on the luma channel independently, without introducing colour artefacts.

Two ITU standards are supported:
- **BT.709** — used for HD video (1280×720 and above)
- **BT.601/407** — used for SD video and legacy formats

The module also incorporates **Color Saturation Enhancement (CSE)**, which can boost or attenuate the chroma channels to increase or decrease colour vividness.

### Algorithm

The conversion applies a fixed integer coefficient matrix to each pixel:

**BT.709:**
```
Y  = ( 47·R + 157·G +  16·B) / 256 + 16
Cb = (-26·R -  86·G + 112·B) / 256 + 128
Cr = (112·R - 102·G -  10·B) / 256 + 128
```

**BT.601:**
```
Y  = ( 77·R + 150·G +  29·B) / 256 + 16
Cb = (131·R - 110·G -  21·B) / 256 + 128   [note: sign convention may differ per implementation]
Cr = (-44·R -  87·G + 138·B) / 256 + 128
```

The output is an 8-bit YCbCr image with Y in `[16, 235]` and Cb/Cr in `[16, 240]` (standard video range).

### Config Parameters

**CSC:**

| Parameter       | Type | Description |
|----------------|------|-------------|
| `is_enable`     | bool | This is an essential module and cannot be disabled |
| `conv_standard` | int  | `1` = BT.709, `2` = BT.601/407 |

**CSE (Color Saturation Enhancement):**

| Parameter        | Type  | Description |
|-----------------|-------|-------------|
| `is_enable`      | bool  | Enable or disable saturation boost |
| `saturation_gain`| float | Multiplier applied to both Cb and Cr channels (>1 boosts, <1 reduces saturation) |

**Example config:**
```yaml
color_space_conversion:
  is_enable: true
  conv_standard: 1   # BT.709

color_saturation_enhancement:
  is_enable: true
  saturation_gain: 1.2
```

### Code Walkthrough

```python
from modules.color_space_conversion.color_space_conversion import ColorSpaceConversion

csc = ColorSpaceConversion(img, platform, sensor_info, parm_csc, parm_cse)
yuv_img = csc.execute()
```

Inside `rgb_to_yuv_8bit()`:

```python
# BT.709 coefficient matrix
rgb2yuv_mat = np.array([[47, 157, 16], [-26, -86, 112], [112, -102, -10]])

# Reshape image to (N, 3) for matrix multiplication
img_flat = img.reshape((-1, 3))

# Apply matrix: multiply each pixel triplet
yuv = np.matmul(img_flat, rgb2yuv_mat.T)

# Add offset (Y: 16, Cb/Cr: 128) and normalise by 256
# (integer arithmetic avoids floating point for hardware compatibility)
```

CSE is applied to the Cb and Cr channels:

```python
# After conversion to YCbCr:
yuv[:, :, 1] = np.clip(yuv[:, :, 1] * saturation_gain, 16, 240)  # Cb
yuv[:, :, 2] = np.clip(yuv[:, :, 2] * saturation_gain, 16, 240)  # Cr
```

---

## 13. Local Dynamic Contrast Enhancement (LDCI)

**Module path:** `modules/ldci/ldci.py`
**Algorithm file:** `modules/ldci/clahe.py`
**Reference:** [Modified CLAHE — arXiv 2021](https://arxiv.org/ftp/arxiv/papers/2108/2108.12818.pdf)

### Description

Standard histogram equalisation improves global contrast but can wash out local detail and produce unnatural-looking images. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** solves this by:

1. Dividing the image into small **tiles**.
2. Computing and equalising the histogram **within each tile** independently.
3. **Limiting the clip** on histogram amplification to prevent noise amplification and over-enhancement.
4. Blending tile results using **bilinear interpolation** at tile boundaries to avoid visible seams.

In Infinite-ISP, LDCI is applied to the **Y (luma) channel only** to enhance contrast without affecting colour.

### Config Parameters

| Parameter    | Type  | Description |
|-------------|-------|-------------|
| `is_enable`  | bool  | Enable or disable LDCI |
| `clip_limit` | float | Maximum histogram amplification (higher = more contrast, more noise amplification) |
| `wind`       | int   | Tile/window size for local histogram computation |

**Example config:**
```yaml
ldci:
  is_enable: true
  clip_limit: 2.0
  wind: 8
```

### Code Walkthrough

```python
from modules.ldci.ldci import LDCI

ldci = LDCI(img, platform, sensor_info, parm_ldci)
enhanced_img = ldci.execute()
```

Inside `clahe.py`, the CLAHE algorithm:

```python
# 1. Divide image into tiles of size (wind × wind)
# 2. For each tile:
#    a. Compute histogram
#    b. Clip histogram bins at clip_limit (redistribute excess to other bins)
#    c. Compute cumulative distribution function (CDF)
#    d. Map pixel values via CDF (local equalisation)
# 3. Bilinear interpolation between adjacent tile mappings
```

### Tuning Tips

- `clip_limit = 1.0` gives no amplification (equivalent to standard histogram equalisation).
- `clip_limit = 2.0–4.0` is typical for natural images; go higher for medical or night images.
- Smaller `wind` produces more localised contrast enhancement; larger values approach global equalisation.

---

## 14. Sharpening / Edge Enhancement

**Module path:** `modules/sharpen/sharpen.py`
**Algorithm file:** `modules/sharpen/unsharp_masking.py`

### Description

Camera optics and demosaicing can introduce slight blurring. Sharpening enhances fine detail by detecting and amplifying high-frequency components (edges and textures). Infinite-ISP uses **Unsharp Masking (USM)**, the most common sharpening technique in imaging pipelines.

Sharpening is applied to the **Y (luma) channel only** in YCbCr space, ensuring that colour (chroma) is not affected.

### Algorithm — Unsharp Masking

1. Apply a **Gaussian blur** to the image to create a blurred version (the "unsharp mask").
2. Subtract the blurred version from the original to extract the **high-frequency detail**.
3. Add a scaled version of this detail back to the original.

```
detail     = original - gaussian_blur(original, sigma)
sharpened  = original + strength × detail
```

### Config Parameters

| Parameter        | Type  | Description |
|-----------------|-------|-------------|
| `is_enable`      | bool  | Enable or disable sharpening |
| `sharpen_sigma`  | float | Standard deviation of the Gaussian blur (larger = stronger blur, wider sharpening halo) |
| `sharpen_strength`| float | Scale factor for the detail signal (larger = more aggressive sharpening) |

**Example config:**
```yaml
sharpen:
  is_enable: true
  sharpen_sigma: 1.0
  sharpen_strength: 1.5
```

### Code Walkthrough

```python
from modules.sharpen.sharpen import Sharpen

sharpen = Sharpen(img, platform, sensor_info, parm_sha)
sharpened_img = sharpen.execute()
```

Inside `unsharp_masking.py`:

```python
from scipy.ndimage import gaussian_filter

# Extract luma (Y) channel
Y = img[:, :, 0]

# Create unsharp mask: blur the Y channel
blurred = gaussian_filter(Y, sigma=sharpen_sigma)

# Compute high-frequency detail
detail = Y - blurred

# Add amplified detail back
Y_sharp = Y + sharpen_strength * detail

# Clip to valid range
Y_sharp = np.clip(Y_sharp, 0, 255)
img[:, :, 0] = Y_sharp
```

### Tuning Tips

- `sharpen_sigma = 0.5–1.0`: tight sharpening for fine detail.
- `sharpen_sigma = 1.5–3.0`: broader edge enhancement.
- `sharpen_strength = 0.5–1.0`: subtle; `1.5–3.0`: noticeable; >3.0: may introduce haloing artefacts.

---

## 15. 2D Noise Reduction

**Module path:** `modules/noise_reduction_2d/noise_reduction_2d.py`
**Algorithm file:** `modules/noise_reduction_2d/non_local_means.py`
**Reference:** [Buades, Coll, Morel — Non-Local Means (IPOL)](https://www.ipol.im/pub/art/2011/bcm_nlm/article.pdf)

### Description

After demosaicing and the full colour processing chain, residual noise may still be visible. The 2D Noise Reduction module provides post-processing denoising in the YCbCr domain, operating on the full RGB-converted image.

Two algorithms are available:
- **NLM (Non-Local Means)** — a patch-based algorithm that weights pixel values by the similarity of their surrounding neighbourhoods.
- **EBF (Entropy-Based Bilateral Filter)** — a variant of the bilateral filter using entropy to weight the range kernel.

### Algorithm — Non-Local Means (NLM)

NLM is based on the observation that natural images contain many similar patches across different spatial locations. For each pixel:

1. Find patches (small rectangular windows) centred at all other pixels within a search window.
2. Compute the **Euclidean distance** between the current patch and each candidate patch.
3. Use these distances to compute **weights** via a Gaussian function.
4. Replace the pixel with the **weighted average** of all candidate pixels.

```
NLM_pixel(p) = Σ w(p, q) × pixel(q) / Σ w(p, q)
where w(p, q) = exp(-||patch(p) - patch(q)||² / h²)
```

The parameter `h` (related to `wts`) controls the degree of smoothing.

### Config Parameters

| Parameter     | Type   | Description |
|--------------|--------|-------------|
| `is_enable`   | bool   | Enable or disable 2D NR |
| `algorithm`   | string | `"nlm"` or `"ebf"` |
| `window_size` | int    | Search window radius for NLM |
| `patch_size`  | int    | Patch size for NLM comparison |
| `wts`         | float  | Smoothing strength for NLM |
| `wind`        | int    | Window size for EBF |
| `sigma`       | float  | Range/spatial kernel σ for EBF |

**Example config:**
```yaml
noise_reduction_2d:
  is_enable: true
  algorithm: "nlm"
  window_size: 11
  patch_size: 3
  wts: 0.1
  wind: 5
  sigma: 15.0
```

### Code Walkthrough

```python
from modules.noise_reduction_2d.noise_reduction_2d import NoiseReduction2d

nr = NoiseReduction2d(img, platform, sensor_info, parm_2dnr)
denoised_img = nr.execute()
```

Inside `non_local_means.py` (conceptual):

```python
# Pre-compute weight LUT from Euclidean distances
weight_lut = np.exp(-distances**2 / (2 * wts**2))

for each pixel p:
    weights = []
    for each pixel q in search_window:
        d = euclidean_distance(patch_p, patch_q)
        weights.append(weight_lut[d])

    pixel_output[p] = weighted_average(search_pixels, weights)
```

### Tuning Tips

- NLM is computationally expensive. Use a smaller `window_size` for faster processing.
- Increasing `wts` produces more aggressive smoothing (may blur fine detail).
- EBF is faster than NLM but typically yields lower quality results.

---

## 16. RGB Conversion (YUV → RGB)

**Module path:** `modules/rgb_conversion/rgb_conversion.py`

### Description

When the pipeline is configured to output **RGB** (rather than YUV), the YCbCr image produced by the Color Space Conversion module must be converted back to RGB. The RGB Conversion module applies the **inverse** of the CSC matrix using the same ITU standard (BT.709 or BT.601).

This ensures a round-trip colour consistency: the YCbCr intermediate allows luma-domain processing (sharpening, noise reduction, contrast enhancement) without altering chroma, and the final RGB conversion restores the full-colour image.

### Algorithm

Inverse of the CSC matrix multiplication:

**BT.709 inverse:**
```
R = Y × (256/219) + Cr × (256/224) × 1.5748 − offset_R
G = Y × (256/219) − Cb × (256/224) × 0.1873 − Cr × (256/224) × 0.4681 − offset_G
B = Y × (256/219) + Cb × (256/224) × 1.8556 − offset_B
```

The same integer coefficient matrix approach used in CSC is applied here in reverse.

### Config Parameters

RGB Conversion is controlled by the platform parameter `rgb_output`:

```yaml
platform:
  rgb_output: true   # When true, YUV is converted back to RGB
```

The conversion standard (`conv_standard`) is inherited from the CSC config.

### Code Walkthrough

```python
from modules.rgb_conversion.rgb_conversion import RGBConversion

rgb_conv = RGBConversion(img, platform, sensor_info, parm_csc)
rgb_img = rgb_conv.execute()
```

Inside `rgb_conversion.py`, the inverse matrix is applied:

```python
if conv_std == 1:  # BT.709
    yuv2rgb_mat = np.array([...])   # Inverse BT.709 matrix

img_flat = img.reshape((-1, 3))
rgb = np.matmul(img_flat, yuv2rgb_mat.T) + offset
rgb = np.clip(rgb, 0, 255).reshape(img.shape).astype(np.uint8)
```

---

## 17. Scale

**Module path:** `modules/scale/scale.py`
**Algorithm files:** `modules/scale/bilinear_interpolation.py`, `modules/scale/nearest_neighbor.py`

### Description

The Scale module resizes the output image to a target resolution. It supports two modes:

- **Software mode** (`is_hardware = false`): Applies standard interpolation algorithms directly — Nearest Neighbour or Bilinear.
- **Hardware mode** (`is_hardware = true`): A three-step process optimised for specific input/output resolution pairs that can be efficiently implemented in hardware: integer downscale → crop → non-integer scale.

Scaling is applied **after** all image quality processing, so it does not affect the quality of the ISP output.

### Config Parameters

| Parameter          | Type   | Description |
|-------------------|--------|-------------|
| `is_enable`        | bool   | Enable or disable scaling |
| `is_debug`         | bool   | Print debug logs |
| `new_width`        | int    | Target output width |
| `new_height`       | int    | Target output height |
| `is_hardware`      | bool   | Use hardware-friendly 3-step scaling |
| `algorithm`        | string | Software mode: `"Nearest_Neighbor"` or `"Bilinear"` |
| `upscale_method`   | string | Hardware mode: upscaling algorithm |
| `downscale_method` | string | Hardware mode: downscaling algorithm |

**Supported hardware scaling pairs:**
- 2592×1944 → 1920×1080, 1280×960, 1280×720, 640×480, 640×360
- 2592×1536 → 1280×720, 640×480, 640×360
- 1920×1080 → 1280×720, 640×480, 640×360

**Example config:**
```yaml
scale:
  is_enable: true
  is_debug: false
  new_width: 1920
  new_height: 1080
  is_hardware: false
  algorithm: "Bilinear"
  upscale_method: "Nearest_Neighbor"
  downscale_method: "Bilinear"
```

### Code Walkthrough

```python
from modules.scale.scale import Scale

scale = Scale(img, platform, sensor_info, parm_sca)
scaled_img = scale.execute()
```

**Bilinear interpolation** (software mode):

```python
# For each output pixel (x_out, y_out):
x_in = x_out * (input_width  / output_width)
y_in = y_out * (input_height / output_height)

# Find the four surrounding input pixels and interpolate
x0, y0 = floor(x_in), floor(y_in)
x1, y1 = x0 + 1, y0 + 1

# Weighted average of the four corners
output = (1-fx)*(1-fy)*img[y0,x0] + fx*(1-fy)*img[y0,x1]
       + (1-fx)*fy*img[y1,x0]     + fx*fy*img[y1,x1]
```

**Nearest Neighbour** is faster but lower quality:

```python
x_in = round(x_out * (input_width  / output_width))
y_in = round(y_out * (input_height / output_height))
output = img[y_in, x_in]
```

### Tuning Tips

- Use **Bilinear** for downscaling to avoid aliasing artefacts.
- Use **Nearest Neighbour** when speed is critical or for integer downscaling (no quality loss for 2× downscale).
- The hardware mode is restricted to predefined resolution pairs for FPGA compatibility.

---

## 18. YUV Format Conversion

**Module path:** `modules/yuv_conv_format/yuv_conv_format.py`

### Description

Standard YCbCr `444` format stores Y, Cb, and Cr at full resolution — three values per pixel. Video encoding and transmission standards often use **chroma subsampling** to reduce bandwidth, based on the fact that human vision is more sensitive to luminance than colour. The YUV Format Conversion module converts between:

- **YUV 4:4:4** — Full chroma resolution (3 bytes per pixel)
- **YUV 4:2:2** — Horizontal chroma subsampled (2 bytes per pixel average). For every pair of horizontally adjacent pixels, one Cb and one Cr value is shared.

The output is serialised and saved as a raw `.yuv` binary file, suitable for use with standard video players and encoders.

### Config Parameters

| Parameter    | Type   | Description |
|-------------|--------|-------------|
| `is_enable`  | bool   | Enable or disable format conversion |
| `conv_type`  | string | `"444"` for full chroma, `"422"` for 2:1 horizontal subsampling |

**Example config:**
```yaml
yuv_conversion_format:
  is_enable: true
  conv_type: "422"
```

### Code Walkthrough

```python
from modules.yuv_conv_format.yuv_conv_format import YUVConvFormat

yuv_fmt = YUVConvFormat(img, platform, sensor_info, parm_yuv)
yuv_fmt.execute()
```

Inside `convert2yuv_format()`:

```python
if conv_type == "422":
    # For each pair of adjacent pixels, extract shared chroma
    y_0 = img[:, 0::2, 0]   # Y for even columns
    u   = img[:, 0::2, 1]   # Cb from even columns (shared)
    v   = img[:, 0::2, 2]   # Cr from even columns (shared)
    y_1 = img[:, 1::2, 0]   # Y for odd columns
    yuv = concatenate([y_0, u, y_1, v])   # YUYV interleave

elif conv_type == "444":
    y = img[:, :, 0]
    u = img[:, :, 1]
    v = img[:, :, 2]
    yuv = concatenate([y, u, v])

# Write raw binary to file
yuv.flatten().tofile("out_frames/out_<filename>.yuv")
```

### Key Behaviours

- This module requires a YCbCr (not RGB) input. If `platform.rgb_output = true`, the module refuses to run and prints an error.
- The output `.yuv` file is a flat binary — no header. When playing back, the resolution, chroma format, and bit depth must be specified explicitly to the player (e.g., VLC, FFplay).
- `conv_type = "422"` uses the **YUYV** byte order: Y₀ U Y₁ V for each two-pixel block.

---

## Quick Reference: Config YAML Template

```yaml
platform:
  filename: "your_image.raw"
  disable_progress_bar: false
  leave_pbar_string: false
  rgb_output: true

sensor_info:
  bayer_pattern: "rggb"    # rggb / bggr / grbg / gbrg
  bit_depth: 12
  width: 2592
  height: 1536

crop:
  is_enable: false
  new_width: 1920
  new_height: 1080

dead_pixel_correction:
  is_enable: true
  dp_threshold: 80

black_level_correction:
  is_enable: true
  r_offset: 200
  gr_offset: 200
  gb_offset: 200
  b_offset: 200
  is_linear: true
  r_sat: 4095
  gr_sat: 4095
  gb_sat: 4095
  b_sat: 4095

oecf:
  is_enable: false

digital_gain:
  is_enable: true
  gain_array: [1.0, 1.5, 2.0, 3.0]
  current_gain: 0
  ae_feedback: 0
  is_auto: false

bayer_noise_reduction:
  is_enable: true
  filt_window: 5
  r_std_dev_s: 2.0
  r_std_dev_r: 20.0
  g_std_dev_s: 2.0
  g_std_dev_r: 20.0
  b_std_dev_s: 2.0
  b_std_dev_r: 20.0

white_balance:
  is_enable: true
  is_auto: false
  r_gain: 1.8
  b_gain: 1.5

color_correction_matrix:
  is_enable: true
  corrected_red:   [1.0, 0.0, 0.0]
  corrected_green: [0.0, 1.0, 0.0]
  corrected_blue:  [0.0, 0.0, 1.0]

gamma_correction:
  is_enable: true
  # gamma_lut_12: [...]  (4096 values)

color_space_conversion:
  is_enable: true
  conv_standard: 1    # 1=BT.709, 2=BT.601

color_saturation_enhancement:
  is_enable: true
  saturation_gain: 1.2

ldci:
  is_enable: true
  clip_limit: 2.0
  wind: 8

sharpen:
  is_enable: true
  sharpen_sigma: 1.0
  sharpen_strength: 1.5

noise_reduction_2d:
  is_enable: true
  algorithm: "nlm"
  window_size: 11
  patch_size: 3
  wts: 0.1

scale:
  is_enable: false
  new_width: 1920
  new_height: 1080
  is_hardware: false
  algorithm: "Bilinear"

yuv_conversion_format:
  is_enable: false
  conv_type: "422"
```

---

*Generated from source code of Infinite-ISP v1.1 — [github.com/10xEngineers/Infinite-ISP](https://github.com/10xEngineers/Infinite-ISP)*
