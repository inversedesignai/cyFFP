# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CyFFP is a Julia module for **cylindrical near-to-far-field propagation** using Vector Cylindrical Harmonics. It computes the point spread function (PSF) of optical systems at large oblique angles by combining FFTLog Hankel transforms with Graf's addition theorem for a purely modal-space basis shift (no interpolation or Cartesian regridding).

**Authors:** Arvin Keshvari (supervised by Dr. Zin Lin)

## Running Tests

```bash
julia test_cyfft_graf.jl
```

Tests use direct assertions and printed output (no test framework). All 7 tests must print "PASSED" and the script must end with "All tests passed."

## Dependencies

- **FFTW** - Fast Fourier transforms
- **SpecialFunctions** - `loggamma` and `besselj`

Install via Julia package manager: `] add FFTW SpecialFunctions`

No Project.toml exists; the module is loaded via `include("CyFFP_graf.jl")`.

## Architecture

Single-module design in `CyFFP_graf.jl`. The algorithm is a 6-step pipeline, each step implemented as a separate exported function:

1. **Angular decomposition** (`angular_decompose`) - FFT over azimuthal angle to extract modes m ≥ 0 only
2. **Forward Hankel transforms** (`compute_TE_TM_coeffs`) - FFTLog for TE/TM coefficients using orders m±1; 4 FFTLog calls per mode; only m ≥ 0 computed (half the work)
3. **Propagation + symmetry** (`propagate_and_symmetrize`) - Apply `exp(i*kz*f)` phase and reconstruct negative-m modes via `Ã_{-m} = (-1)^m (A^TM_m - A^TE_m) · prop`
4. **Graf shift** (`graf_shift_all_kr` calling `graf_shift_one_kr`) - Discrete convolution in mode index via Graf's addition theorem: `B_l(kr) = sum_m A_{m+l}(kr) * J_m(kr*x0)`
5. **Inverse Hankel transforms** (`local_hankel_inverse`) - FFTLog in local basis; only 2*L_max+1 calls (L_max << M_max)
6. **Angular synthesis** (`synthesize_local_psf`) - IFFT over local modes to get PSF(rho, psi)

Two top-level drivers chain all steps:
- `cyfft_farfield(Er, Etheta, r, k, alpha, f)` — from full 2D field arrays
- `cyfft_farfield_modal(Em_r_pos, Em_theta_pos, r, k, alpha, f)` — from pre-decomposed m ≥ 0 modes (no negative-m simulation needed)

## Key Design Decisions

- **Negative-m symmetry**: For φ=0 x-polarized illumination, `E_{-m,r} = E_{m,r}` and `E_{-m,θ} = -E_{m,θ}` (cyFFP0 Eqs. 65-66). This yields `A^TE_{-m} = (-1)^{m+1} A^TE_m` and `A^TM_{-m} = (-1)^m A^TM_m`, halving the FFTLog calls. FDTD/RCWA only needs to simulate m ≥ 0.
- **FFTLog for Hankel transforms**: Log-space FFT approach enables numerical stability for Bessel orders |m| ~ 600+. The filter kernel uses `loggamma` to avoid overflow.
- **All grids are log-spaced**: Radial input `r` must satisfy `r[j+1]/r[j] = const`. Output grids are reciprocal log-grids.
- **Bessel truncation in Graf shift**: `J_m(kr*x0)` is negligible for `|m| > kr*x0 + 20`, so the inner convolution sum is naturally bounded. Uses `@inbounds` direct indexing (not Dict) for speed.
- **Normal incidence special case**: When `alpha ~ 0`, `x0 ~ 0`, and `J_m(0) = delta_{m,0}`, so the Graf shift becomes the identity.
- **pmap parallelism**: Steps 2, 4, 5 use `pmap` for distributed parallelism. Falls back to serial `map` when no workers are added.
