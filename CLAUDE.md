# CLAUDE.md — Project Guide for AI-Assisted Development

## Project Identity

**EdgeISP** — an open-source Image Signal Processing pipeline for wearable and embedded cameras, built on [Infinite-ISP](https://github.com/10xEngineers/Infinite-ISP) by 10xEngineers. The project has three pillars:

1. **Better pipeline** — algorithmic upgrades with configurable speed/quality trade-offs
2. **Calibration guidance** — standalone calibration module for sensor characterisation
3. **Wearable orientation** — targeting small-aperture, compute-constrained, low-light scenarios

## First principle:
请使用第一性原理思考。你不能总是假设我非常清楚自己想要什么和该怎么得到。请保持审慎，从原始需求和问题出发，如果动机和目标不清晰，停下来和我讨论。如果目标清晰但是路径不是最短，告诉我，并且建议更好的办法.

## Repository Structure

```
├── infinite_isp.py           # Main pipeline class — module execution order defined here
├── isp_pipeline.py           # CLI entry point
├── config/
│   └── configs.yml           # All module parameters — single source of truth
├── modules/                  # One directory per pipeline module
│   ├── crop/
│   ├── dead_pixel_correction/
│   ├── black_level_correction/
│   ├── oecf/
│   ├── digital_gain/
│   ├── lens_shading_correction/
│   ├── bayer_noise_reduction/
│   ├── auto_white_balance/
│   ├── white_balance/
│   ├── demosaic/
│   ├── color_correction_matrix/
│   ├── gamma_correction/
│   ├── auto_exposure/
│   ├── color_space_conversion/
│   ├── ldci/
│   ├── sharpen/
│   ├── noise_reduction_2d/
│   ├── rgb_conversion/
│   ├── scale/
│   └── yuv_conv_format/
├── calibration/              # [NEW — to be created] Calibration module
├── docs/
│   ├── module_guide.md       # Detailed module-by-module documentation
│   └── dev_notes.md          # Design spec — all improvement decisions recorded here
├── in_frames/                # Input RAW files (including ColorChecker samples)
├── out_frames/               # Pipeline output images
├── util/                     # Shared utilities
├── CLAUDE.md                 # This file
└── PROGRESS.md               # Implementation roadmap and status tracking
```

## Pipeline Execution Order

### Current order (in `infinite_isp.py`):
```
Crop → DPC → BLC → OECF → Digital Gain → LSC → BNR → AWB → WB → Demosaic
→ CCM → Gamma → AE → CSC → LDCI → Sharpen → 2D NR → RGB Conv → Scale → YUV Format
```

### Correct order (to be implemented):
```
Crop → DPC → [BLC offset] → OECF → [BLC linearise] → Digital Gain → LSC → BNR
→ AWB → WB → Demosaic → CCM → LDCI → 2D NR → Gamma → AE → CSC
→ Color Saturation → Sharpen → RGB Conv → Scale → YUV Format → Dither → Output Encode
```

Key ordering fixes (see dev_notes.md for rationale):
- **OECF between BLC offset and BLC linearise** — OECF LUT must index into its calibrated domain
- **LDCI and 2D NR before Gamma** — must operate in linear light domain
- **2D NR before Sharpen** — current order defeats both modules
- **Dither after all processing** — blue noise spatial dithering before uint8 quantisation

## Design Specification

**`docs/dev_notes.md` is the authoritative design spec.** Every improvement decision, algorithm choice, config design, and implementation flag is recorded there. When implementing a feature:

1. Read the relevant section in dev_notes.md first
2. Follow the config design specified there (YAML key names, defaults, modes)
3. Maintain backward compatibility — new features are opt-in via config modes
4. Keep the classical/baseline path as default; new algorithms are alternative modes

## Implementation Conventions

### Code style
- Python 3.8+ compatible
- NumPy for all array operations — no unnecessary dependencies
- Each module is a class with an `execute()` method returning the processed image
- Module constructors take: `(image, platform, sensor_info, module_params, ...)`

### Config conventions
- All parameters in `config/configs.yml`
- Every module has `is_enable: true/false`
- New modes use a `mode` key: `mode: "default" | "high_quality" | ...`
- Default values always reproduce current Infinite-ISP behaviour (backward compat)

### Module implementation pattern
```python
class ModuleName:
    def __init__(self, img, platform, sensor_info, parm_mod):
        self.img = img
        self.platform = platform
        self.sensor_info = sensor_info
        self.enable = parm_mod["is_enable"]
        # ... extract parameters ...

    def execute(self):
        if not self.enable:
            return self.img
        # ... processing ...
        return result
```

### Adding a new mode to an existing module
1. Add `mode` parameter to config section in `configs.yml`
2. Branch on `self.mode` inside `execute()`
3. Keep original algorithm as the default mode
4. New algorithm in a separate method or file in the module directory
5. Update `docs/dev_notes.md` to mark the item as implemented

### Testing
- Run `python isp_pipeline.py` for single-image pipeline test
- Compare output in `out_frames/` against reference
- Verify config backward compatibility: default config must produce identical output

## Priority Tags in dev_notes.md

- **⚑ IMMEDIATE FIX** — correctness bugs, must fix before other work
- **⚑ NEXT IMPLEMENTATION** — ready to implement, clear spec in the note
- **⚑ REVISIT LATER** — design complete, implementation deferred

## Key Technical Decisions (quick reference)

- DPC before BLC (BLC-first clips cold dead pixels in dark scenes)
- Signed BLC rejected — complexity not worth the gain
- BLC split: offset step → OECF → linearise step
- LSC: dual-mode (config gain map + lensfun database), move before Digital Gain
- BNR: multi-frame averaging (Bayer domain) + JBF
- Demosaic: LMMSE as high-quality alternative to MHC
- CCM: XYZ intermediate → target primaries (decouples calibration from output format)
- Gamma: sRGB + Rec.709 + PQ + HLG + linear (bundled as output profiles)
- LDCI: guided filter LTM replaces CLAHE bilinear tiles
- Color saturation: flat gain + vibrance mode
- Sharpen: edge-adaptive gain (after 2D NR, not before)
- 2D NR: dual-role — full bilateral/NLM in classical mode, chroma-only in DL mode
- DL joint model: Bayer → clean RGB, replaces BNR + Demosaic + 2D NR Y-channel
- Burst DL: end-to-end N frames → RGB, no explicit registration
- Output: PNG (default) + JPEG + EXR; blue noise dithering for 8-bit
- YUV: 4:4:4 / 4:2:2 / 4:2:0 configurable
