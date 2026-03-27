# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CyFFP is a Julia module for **cylindrical near-to-far-field propagation** using scalar cylindrical harmonics. It computes the point spread function (PSF) of rotationally symmetric optical systems at large oblique angles by combining FFTLog Hankel transforms with Graf's addition theorem for a purely modal-space basis shift (no interpolation or Cartesian regridding).

**Authors:** Arvin Keshvari (supervised by Dr. Zin Lin)

## Running Tests

```bash
# Step-by-step unit tests
julia test_step1.jl          # Angular decomposition
julia test_step2_fftlog.jl   # FFTLog Hankel transform
julia test_step3.jl          # Scalar propagation + symmetry
julia test_step4.jl          # Graf shift
julia test_step5.jl          # Inverse Hankel
julia test_step6.jl          # Angular synthesis + full pipeline Airy

# End-to-end validation
julia test_scalar_airy.jl         # Airy disk with small lens (R=3λ)
julia test_bruteforce_reference.jl # QuadGK cross-check

# Scaling validation (R=3λ..1000λ, up to 558 modes)
julia test_scaling.jl

# Production scale (needs ~37 GB for full M_max)
julia -t auto test_production_aperture.jl
```

Start Julia with `-t N` for N threads. The module uses `Threads.@threads` for shared-memory parallelism (no Distributed).

Tests use direct assertions and printed output.

Optional: `] add Plots` to generate PSF heatmap PNGs.

## Dependencies

- **FFTW** - Fast Fourier transforms
- **SpecialFunctions** - `loggamma` and `besselj`
- **QuadGK** - Numerical integration (tests only)
- **Plots** (optional) - PSF heatmap generation

Install via Julia package manager: `] add FFTW SpecialFunctions QuadGK`

No Project.toml exists; the module is loaded via `include("cyffp.jl")`.

## Architecture

Single-module design in `cyffp.jl`. The scalar pipeline propagates a single Cartesian component (e.g. E_y) which satisfies the scalar Helmholtz equation exactly.

### Scalar pipeline (6-step)

1. **Angular decomposition** (`angular_decompose`) — FFT over θ, extract m ≥ 0 only
2. **Scalar spectral coefficients** (`compute_scalar_coeffs`) — FFTLog at order m; ONE transform per mode: a_m(kr) = H_m[r u_m](kr); threaded over modes
3. **Propagation + symmetry** (`propagate_scalar`) — Apply exp(ikz f), reconstruct negative-m via scalar symmetry ã_{-m} = (-1)^m ã_m
4. **Graf shift** (`graf_shift`) — Mode convolution `B_l = Σ_m ã_{m+l} J_m(kr x₀)`; Miller backward recurrence for Bessel weights; @simd inner loop; threaded over kr
5. **Inverse Hankel** (`inverse_hankel`) — FFTLog in local basis; only 2L_max+1 calls (~27); output ρ-grid = reciprocal of kr-grid = original r-grid; threaded over modes
6. **Angular synthesis** (`angular_synthesis`) — IFFT over local modes; u(ρ,ψ) = Σ_l b_l(ρ) e^{ilψ}; take |u|² for PSF intensity

### Neumann shift-theorem fast path (`neumann_shift_coeffs`)

For LPA fields of the form t(r)·exp(ikₓ r cosθ), uses the Neumann addition formula to obtain ALL M_max modal coefficients from a **single** order-0 FFTLog call + interpolation + FFT per kr. Replaces `compute_scalar_coeffs` (Step 2) for this field type. Output feeds directly into `propagate_scalar` (Step 3).

**Limitation:** Linear interpolation of T₀ degrades at large M_max. Validated to <1% PSF accuracy up to M_max≈1,600 (R≤500λ, α≤30°). At production scale (M_max≈12,600), the PSF error reaches ~9% and there is no speed advantage. **Use the standard path for production.** The Neumann path is useful for smaller problems, coma physics studies, and cross-validation at moderate scales.

## Key Design Decisions

- **Scalar formulation**: Each Cartesian component of E satisfies the scalar Helmholtz equation (proven in the tex formulation). This gives ONE Hankel transform per mode at order m, replacing the 4-transform TE/TM approach. The TE/TM combination Ã = A^TE + A^TM was found to be incorrect — it conflates different Bessel structures and produces wrong PSFs.
- **Scalar symmetry**: For y-polarized illumination with tilt in xz-plane, E_y is even under y→-y, giving u_{-m} = u_m. Combined with J_{-m} = (-1)^m J_m, this gives ã_{-m} = (-1)^m ã_m. This halves the FFTLog calls.
- **Threading**: `Threads.@threads` for shared-memory parallelism (no serialization overhead). FFTW plan cache is warmed before threaded regions.
- **FFTLog for Hankel transforms**: Log-space FFT approach, stable for Bessel orders |m| ~ 600+. Filter kernel uses `loggamma` to avoid overflow. Key: |U(q)| = 1 for all q (complex conjugate Gamma pair).
- **All grids are log-spaced**: `r[j+1]/r[j] = const`. Output ρ-grid is the reciprocal of kr-grid. **Critical**: `r_min` must be ≤ `1/k = λ/(2π)` or propagating modes are lost.
- **Graf shift optimizations**: Bessel truncation at `|m| > kr·x₀ + 20`; Miller backward recurrence for J₀..J_{M_max} (~10× faster than independent calls); `@simd` inner loop; `J_{-m} = (-1)^m J_m` halves Bessel evaluations; self-normalizing identity `1 = J₀ + 2Σ J_{2k}` with overflow protection.
- **Normal incidence special case**: When α ≈ 0, x₀ ≈ 0, `J_m(0) = δ_{m,0}`, so the Graf shift becomes the identity.

## Formulation

- `cyFFP_formulation.tex` — Self-contained formulation document: scalar proof, FFTLog, Graf derivation, Neumann shift theorem, cost analysis. TE/TM retained in Appendix A for reference.
- `cyFFP0.pdf`, `cyFFP1.pdf`, `cyFFP2.pdf` — Historical derivation PDFs (superseded by tex document).

## Deleted Files (for historical reference)

- `dev/` — Old buggy code with wrong FFTLog kernel, wrong TE/TM combination. Deleted.
- `test_step2_te_tm.jl` — Tested the old compute_TE_TM_coeffs (no longer exists). Deleted.
