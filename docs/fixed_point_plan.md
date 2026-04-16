# Fixed-Point Model Conversion Plan — Omni-ISP

This document is a stage-by-stage plan to convert the floating-point ISP
pipeline in `omni_isp.py` to a bit-exact, hardware-friendly fixed-point
reference model. The goal of the fixed-point model is twofold:

1. Provide a Python reference that matches the bit-precision of the
   downstream RTL/HW implementation.
2. Keep the existing float pipeline intact so it can be used as the
   golden reference for PSNR / SSIM / per-pixel-error verification.

---

## 1. Guiding Principles

| # | Principle | Rationale |
|---|-----------|-----------|
| 1 | **Two pipelines side-by-side.** Add `omni_isp_fxp.py` next to `omni_isp.py`. | Lets us regression-test fxp vs. float every commit. |
| 2 | **One fxp module per float module.** `modules/<name>.py` ➜ `modules/<name>_fxp.py`. | Keeps blame, tests, and config local. |
| 3 | **Single `FxpConfig` controls all Q-formats.** | Sweeping bit-widths becomes a one-line change. |
| 4 | **No transcendental functions at runtime.** All `log/exp/pow/atan2/√` become LUTs or piece-wise polynomial approximations precomputed at init. | Matches what HW will actually do. |
| 5 | **Saturating arithmetic, explicit rounding.** Wrap each op in `sat()` / `round_half_up()` helpers; never rely on numpy's silent wrap-around. | Exposes overflow bugs early. |
| 6 | **No silent up-promotion to float64.** Use `np.int32` / `np.int64` accumulators with assertions on width. | A single `/`, `np.mean`, or `**` will demote to float without notice. |
| 7 | **Bit-exact target, not "close enough".** The fxp model must reproduce HW output exactly; the float model is the perceptual reference. | Two distinct quality bars. |

---

## 2. Q-Format Strategy

Default formats (overridable per module via `FxpConfig`):

| Domain | Format | Range | Notes |
|--------|--------|-------|-------|
| Bayer raw | `uint16` (UQ12.0 / UQ14.0) | sensor bitdepth | Native sensor format up through Demosaic. |
| WB / Digital gain | `UQ4.12` | [0, 16) | 12 frac bits ⇒ ULP ≈ 2.4e-4. |
| CCM coefficients | `SQ3.12` | [-8, 8) | Matrix entries are in [-2, 3]. |
| Normalized RGB after CCM | `UQ0.16` | [0, 1) | 16-bit unsigned in `uint16`; replaces `float32 / white_level`. |
| Tone-map / LDCI internal | `UQ4.20` (in `int32`) | [0, 16) | Headroom for HDR luma > 1.0. |
| Gamma input index | `UQ0.12` | LUT addressed by top 12 bits | Matches existing 4096-entry LUT. |
| Gamma output | `uint8` / `uint10` | [0, 255] / [0, 1023] | Set by output bit-depth. |
| YUV after CSC | `SQ8.8` (Y), `SQ7.8` (Cb/Cr) | Y∈[0,255], C∈[-128,127] | 8 frac bits keeps sub-pixel sharpening accurate. |
| Stats3A accumulators | `int32` / `int64` | sums over tile | Per-tile sums of 16-bit pixels need ≥ 32 bits. |

A small helper module `util/fxp.py` will provide:

```text
class Fxp:           # value, q_int, q_frac, signed
def to_fxp(x_float, q_int, q_frac, signed) -> ndarray[int]
def to_float(x_fxp,  q_int, q_frac, signed) -> ndarray[float]
def sat(x, q_int, q_frac, signed) -> ndarray[int]
def round_half_up(x_int, shift) -> ndarray[int]
def mul_q(a, b, qa, qb, q_out) -> ndarray[int]
def reciprocal_lut(n_entries, q_in, q_out) -> ndarray[int]
```

All fxp modules call into this helper — no ad-hoc shifts inline.

---

## 3. Module Conversion Tiers

Modules are split into four tiers by conversion difficulty. Each tier
becomes one PR / one milestone.

### Tier 1 — Linear & LUT-only (1–2 days)
Straight remap of float ops to integer arithmetic. No approximation
required.

| Module | Op | Conversion |
|--------|----|------------|
| `crop` | indexing | trivial |
| `dpc` (median + threshold) | compare + replace | trivial |
| `blc` (offset/linearise) | `sat(x − off) * gain >> N` | precompute `gain = white/(sat-off)` as UQ0.16 |
| `oecf` | LUT lookup | already integer |
| `digital_gain` | mul by Q4.12 gain | `mul_q + sat` |
| `lens_shading` | per-pixel UQ4.12 gain mul | bilinear-interp gain map in fxp |
| `csc` (RGB↔YUV) | 3×3 matrix | int matrix multiply + rounding shift |
| `scale` (bilinear) | α·p1 + (1−α)·p2 | α as UQ0.8 |
| `yuv_format` | subsample | trivial |

**Exit criteria:** per-pixel diff vs. float ≤ 1 LSB on `tests/test_phase1.py`
and `tests/test_phase4.py`.

### Tier 2 — Integer-friendly with care (3–5 days)
Need accumulator widths and reciprocal LUTs but no transcendentals.

| Module | Hard part | Approach |
|--------|-----------|----------|
| `awb` (gray-world) | per-channel mean, ratio | int64 sum, divide via reciprocal LUT keyed on shift-normalised G mean |
| `white_balance` | gain ≥ 1.0 | gain in UQ4.12, `mul_q` then `sat` to sensor bitdepth |
| `ccm` | 3×3 matmul + normalise | int32 accumulator, single shift at end; bias = 1<<(frac-1) for round-to-nearest |
| `demosaic_mhc` | Malvar-He-Cutler kernel | integer kernel ÷8 ⇒ shift right 3 |
| `temporal_nr` | EMA `(1-α)*prev + α*cur` | α as UQ0.8; one mul-add + shift |
| `bnr` (joint bilateral) | range weight | weight LUT keyed on │Δ│, normalised by precomputed reciprocal of weight sum |
| `stats_3a` | hist + sums | int32/int64; histogram is already integer |
| `sharpening_usm` | unsharp mask | Gaussian kernel as int Q0.8; high-pass = `(c − blur)`, scaled by UQ4.8 strength |

**Exit criteria:** PSNR ≥ 50 dB vs. float over the `in_frames/` corpus.

### Tier 3 — Needs precomputed LUTs / piecewise approximations (1 week)
Transcendentals or divisions on wide dynamic range. Each gets a small
LUT generator in `util/fxp_luts.py` so HW can reuse the same tables.

| Module | Transcendental | LUT design |
|--------|----------------|------------|
| `awb_gray_edge` | `log`, `√` | 256-entry log2 LUT + 5-bit Newton refinement; rsqrt LUT |
| `lens_correction` | bilinear remap with FP coords | warp coords as SQ12.4; 4-tap filter in int |
| `demosaic_lmmse` | weight = `dv/(dh+dv+ε)` | reciprocal LUT on `(dh+dv)`; clamp ε to 1 LSB |
| `nr2d_bilateral` | `exp(−x²/2σ²)` | 256-entry exp LUT, normalisation via reciprocal LUT |
| `nr2d_chroma` | Gaussian | precomputed integer kernel (separable) |
| `purple_fringe` | `atan2(Cr,Cb)` | 12-bit atan2 LUT or 8-iter CORDIC |
| `ldci` (CLAHE) | per-tile CDF | CDF in int; bilinear blend across tiles |
| `sharpening_adaptive` | edge-strength ramp | piecewise-linear 8-segment ramp |

**Exit criteria:** PSNR ≥ 45 dB and ΔE76 < 1.0 on the SDR test set.

### Tier 4 — Hardest (1–2 weeks)
Wide dynamic range + multiple transcendentals chained together. Each
deserves its own design doc and bit-width sweep.

| Module | Why hard | Plan |
|--------|----------|------|
| `hdr_tone_map` (Reinhard / ACES / Hable) | log-avg luma, `1/(1+x)`, `pow` | **2-stage**: (1) global log-luma in UQ4.20 via log2 LUT + linear interp; (2) per-pixel curve as 1024-entry LUT regenerated when scene key changes |
| `gamma` (sRGB / Rec.709 / PQ / HLG) | piecewise `pow`, nested log/root for PQ | Direct 4096-entry LUT in `uint10` per curve; PQ/HLG generated once per output-format change |
| `ldci` (guided filter variant) | covariance + division | Box-filter in int; ε guards division; reciprocal LUT on `(σ²+ε)` |

**Exit criteria:** ΔE76 < 2.0 vs. float on HDR test frames; bit-exact match against a hand-checked golden vector for 16 reference pixels.

---

## 4. Verification Plan

A new `tests/test_fxp_parity.py` runs each module in both float and fxp
modes on every image in `in_frames/` and asserts:

| Metric | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|--------|--------|--------|--------|--------|
| Max │Δ│ (LSB at output bitdepth) | ≤ 1 | ≤ 2 | ≤ 4 | ≤ 8 |
| Mean │Δ│ | ≤ 0.1 | ≤ 0.3 | ≤ 0.6 | ≤ 1.0 |
| PSNR vs. float | ≥ 60 dB | ≥ 50 dB | ≥ 45 dB | ≥ 40 dB |

Additional gates:

1. **Bit-width sweep CI job** that re-runs Tier 3/4 with frac bits ±2 to
   catch precision cliffs.
2. **Golden vectors**: 16 hand-picked pixels per stage stored as JSON;
   any change must update the golden file via an explicit reviewer step.
3. **Full-pipeline integration test**: SDR path (sRGB out) and HDR path
   (PQ out), each with a max-ΔE budget.

---

## 5. Config Surface

Extend `config/configs.yml` with a new top-level block:

```yaml
fxp:
  enabled: false                # master switch; CLI flag --fxp also flips it
  default_q:
    rgb_norm:   {int: 0,  frac: 16, signed: false}
    wb_gain:    {int: 4,  frac: 12, signed: false}
    ccm_coeff:  {int: 3,  frac: 12, signed: true}
    yuv:        {int: 8,  frac: 8,  signed: true}
  per_module:                   # overrides
    hdr_tone_map: {int: 4, frac: 20, signed: false}
    gamma_lut_bits: 12
  rounding: round_half_up       # or: truncate, round_half_even
  saturation: clip              # or: wrap (debug only)
```

`util/config_utils.py` gains an `FxpConfig` dataclass that validates the
block at startup and is threaded through `OmniISP.__init__`.

---

## 6. Milestones & Order of Work

| # | Milestone | Deliverable | Est. |
|---|-----------|-------------|------|
| M0 | Scaffolding | `util/fxp.py`, `util/fxp_luts.py`, `FxpConfig`, `omni_isp_fxp.py` skeleton, `tests/test_fxp_parity.py` harness with 1 dummy module | 1 day |
| M1 | Tier-1 modules | 9 modules + parity tests green | 2 days |
| M2 | Tier-2 modules | 8 modules + parity tests green | 1 week |
| M3 | Tier-3 modules | 8 modules + LUT generators + parity tests green | 1 week |
| M4 | Tier-4 modules | 3 modules + golden vectors | 2 weeks |
| M5 | End-to-end SDR & HDR ΔE budget met | Final report + bit-width sweep results | 3 days |

Total: **~5 weeks** of focused work.

---

## 7. Open Questions for Reviewer

1. What is the **target HW bit-width** for the normalized RGB datapath?
   Plan assumes `UQ0.16`; if HW is `UQ0.12` we need to revisit Tier 3 ΔE
   budgets.
2. Is **PQ/HLG** in scope for the first cut, or SDR-only?
3. Do we need a **C/RTL co-simulation hook** (writing fxp intermediate
   tensors to disk for HW comparison), or is Python-only verification
   sufficient for now?
4. Rounding mode: `round_half_up` is simplest; HW often does
   `round_half_even`. Which should be the default?

Resolving these unblocks M0.
