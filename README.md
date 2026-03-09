# EdgeISP

An open-source Image Signal Processing pipeline for wearable and embedded cameras.

Built on [Infinite-ISP](https://github.com/10xEngineers/Infinite-ISP) by 10xEngineers (Apache 2.0).

---

## What is EdgeISP?

EdgeISP is a Python-based camera ISP pipeline that converts raw Bayer sensor data into display-ready images. It extends the Infinite-ISP algorithm design with three goals:

**Better pipeline** — algorithmic upgrades to every module with configurable speed/quality trade-offs. Each module offers a fast baseline (suitable for real-time embedded) and an optional high-quality mode (for offline or less compute-constrained use). Pipeline correctness fixes ensure modules operate in the right signal domain.

**Calibration guidance** — a standalone calibration module that helps users characterise their specific sensor. Capture a ColorChecker and a flat-field target, run the calibrators, and get the config parameters your pipeline needs — BLC offsets, OECF curve, LSC gain map, WB gains, and CCM.

**Wearable and embedded orientation** — designed for the constraints of wearable cameras: small apertures, no optical stabilisation, limited compute budgets, and challenging low-light conditions. Multi-frame averaging, adaptive noise reduction, and a planned DL inference path target these scenarios directly.

---

## Pipeline

```
RAW Bayer
  → Crop → DPC → BLC offset → OECF → BLC linearise → Digital Gain → LSC
  → BNR → AWB → WB → Demosaic → CCM → LDCI → 2D NR → Gamma → AE
  → CSC → Color Saturation → Sharpen → RGB Conv → Scale
  → YUV Format → Dither → Output Encode (PNG / JPEG)
```

---

## Key improvements over Infinite-ISP

| Area | Infinite-ISP | EdgeISP |
|---|---|---|
| Pipeline ordering | Gamma before LDCI/NR; Sharpen before NR | Corrected: linear-domain processing before Gamma; NR before Sharpen |
| DPC | Fixed threshold | MAD-based adaptive threshold |
| Demosaic | MHC only | + LMMSE high-quality mode |
| LDCI | CLAHE with bilinear tiles | + Guided filter LTM (edge-aware, no tile artefacts) |
| Sharpen | Uniform USM | + Edge-adaptive gain (no noise amplification) |
| 2D NR | Single mode | + NLM, chroma-only, off (adapts to pipeline mode) |
| Color saturation | Flat gain | + Vibrance mode |
| CCM | Direct to sRGB | XYZ intermediate → any target primaries |
| Gamma | sRGB only | + Rec.709, PQ (HDR10), HLG, linear |
| Output | PNG only | + JPEG, EXR; 4:4:4 / 4:2:2 / 4:2:0 |
| Dithering | None | Blue noise spatial dithering for 8-bit |
| Low-light | No multi-frame | Multi-frame Bayer averaging + DL path (planned) |
| Calibration | None | Standalone calibration module |
| LSC | Stub (returns image unchanged) | Dual-mode: calibrated gain map + lensfun database |

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline on a sample RAW image
python isp_pipeline.py
```

Output is saved to `out_frames/`. Pipeline parameters are configured in `config/configs.yml`.

---

## Project structure

```
├── infinite_isp.py           # Main pipeline class
├── isp_pipeline.py           # CLI entry point
├── config/configs.yml        # All module parameters
├── modules/                  # One directory per pipeline module
├── calibration/              # Sensor calibration tools (in development)
├── docs/
│   ├── module_guide.md       # Module-by-module documentation
│   └── dev_notes.md          # Design spec and technical decisions
├── in_frames/                # Input RAW files
├── out_frames/               # Pipeline output images
├── CLAUDE.md                 # AI development guide
└── PROGRESS.md               # Implementation roadmap
```

---

## Documentation

- **[Module Guide](docs/module_guide.md)** — detailed description, algorithm walkthrough, config parameters, and tuning tips for every pipeline module
- **[Dev Notes](docs/dev_notes.md)** — design spec recording every improvement decision, algorithm choice, and implementation plan
- **[Progress](PROGRESS.md)** — phased implementation roadmap with status tracking

---

## Roadmap

See [PROGRESS.md](PROGRESS.md) for the full roadmap. Summary:

- **Phase 1** — Pipeline correctness fixes (ordering, signal domain)
- **Phase 2** — Module algorithm upgrades (LMMSE, guided filter LDCI, adaptive sharpen, vibrance, NLM NR)
- **Phase 3** — New capabilities (output profiles, calibration module, blue noise dithering, JPEG output)
- **Phase 4** — Multi-frame and low-light modes
- **Phase 5** — DL integration (joint Bayer denoise + demosaic, burst DL)

---

## License

EdgeISP is licensed under the [Apache License 2.0](LICENSE).

Built on Infinite-ISP — Copyright 2024, 10xEngineers. See [NOTICE](NOTICE) for attribution.
