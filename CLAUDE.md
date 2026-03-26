# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CyFFP is a Julia module for **cylindrical near-to-far-field propagation** using Vector Cylindrical Harmonics. It computes the point spread function (PSF) of optical systems at large oblique angles by combining FFTLog Hankel transforms with Graf's addition theorem for a purely modal-space basis shift (no interpolation or Cartesian regridding).

**Authors:** Arvin Keshvari (supervised by Dr. Zin Lin)

## Running Tests

```bash
# Unit tests (small arrays, quick)
julia test_cyfft_graf.jl

# Realistic workload (2mm lens, 500nm, NA=0.25, α=10°)
julia -t 300 test_real_workload.jl
```

Start Julia with `-t N` for N threads. The module uses `Threads.@threads` for shared-memory parallelism (no Distributed).

Tests use direct assertions and printed output. Both scripts must end with "All tests passed."

Optional: `] add Plots` to generate PSF heatmap PNGs.

## Dependencies

- **FFTW** - Fast Fourier transforms
- **SpecialFunctions** - `loggamma` and `besselj`
- **Plots** (optional) - PSF heatmap generation

Install via Julia package manager: `] add FFTW SpecialFunctions`

No Project.toml exists; the module is loaded via `include("CyFFP_graf.jl")`.

## Architecture

Single-module design in `CyFFP_graf.jl`. Three entry points:

### Standard path (6-step pipeline)

1. **Angular decomposition** (`angular_decompose`) — FFT over θ, extract m ≥ 0 only
2. **Forward Hankel transforms** (`compute_TE_TM_coeffs`) — FFTLog at orders m±1; 4 calls per mode (2 when E_θ=0); threaded over modes
3. **Propagation + symmetry** (`propagate_and_symmetrize`) — Apply exp(ikz f), reconstruct negative-m via `Ã_{-m} = (-1)^m (A^TM - A^TE) · prop`
4. **Graf shift** (`graf_shift_all_kr` → `graf_shift_one_kr`) — Mode convolution `B_l = Σ_m Ã_{m+l} J_m(kr x₀)`; Miller backward recurrence for Bessel weights; @simd inner loop; threaded over kr
5. **Inverse Hankel** (`local_hankel_inverse`) — FFTLog in local basis; only 2L_max+1 calls; threaded
6. **Angular synthesis** (`synthesize_local_psf`) — IFFT over local modes; threaded over ρ

Drivers:
- `cyfft_farfield(Er, Etheta, r, k, alpha, f)` — from full 2D field arrays
- `cyfft_farfield_modal(Em_r_pos, Em_theta_pos, r, k, alpha, f)` — from pre-decomposed m ≥ 0 modes (no negative-m FDTD simulation needed)

### Neumann shift-theorem fast path

- `cyfft_farfield_shift(t_r, r, k, alpha, f)` — for LPA fields of the form t(r)·exp(ikₓ r cosθ)·x̂

Uses the Neumann addition formula `J_m(ar)J_m(br) = (1/2π)∫ J₀(c(φ)r) e^{-imφ} dφ` to obtain ALL M_max modal coefficients from a **single** order-0 FFTLog call + interpolation + FFT per kr. Reduces Step 2 from O(M_max · Nr log Nr) to O(Nr log Nr + Nkr · Nphi log Nphi).

## Key Design Decisions

- **Threading**: `Threads.@threads` for shared-memory parallelism (no serialization overhead). FFTW plan cache is warmed before threaded regions.
- **Negative-m symmetry**: For φ=0 x-polarized illumination, `E_{-m,r} = E_{m,r}` and `E_{-m,θ} = -E_{m,θ}` (cyFFP0 Eqs. 65-66). This yields `A^TE_{-m} = (-1)^{m+1} A^TE_m` and `A^TM_{-m} = (-1)^m A^TM_m`, halving the FFTLog calls.
- **FFTLog for Hankel transforms**: Log-space FFT approach, stable for Bessel orders |m| ~ 600+. Filter kernel uses `loggamma` to avoid overflow.
- **All grids are log-spaced**: `r[j+1]/r[j] = const`. Output ρ-grid is the reciprocal of kr-grid. **Critical**: `r_min` must be ≤ `1/k = λ/(2π)` or propagating modes are lost (the code warns).
- **E_θ=0 fast path**: When E_θ is identically zero, only 2 FFTLog calls per mode instead of 4.
- **Graf shift optimizations**: Bessel truncation at `|m| > kr·x₀ + 20`; Miller backward recurrence for J₀..J_{M_max} (~10× faster than independent calls); `@simd` inner loop; `J_{-m} = (-1)^m J_m` halves Bessel evaluations.
- **Normal incidence special case**: When α ≈ 0, x₀ ≈ 0, `J_m(0) = δ_{m,0}`, so the Graf shift becomes the identity.

## Formulation PDFs

- `cyFFP0.pdf` — Original derivation: vector cylindrical harmonics, TE/TM projections, negative-m symmetry proof
- `cyFFP1.pdf` — First reformulation: FFTLog + polar-to-Cartesian regridding (superseded by Graf approach)
- `cyFFP2.pdf` — Current formulation: FFTLog + Graf's addition theorem for spectrally exact local-basis shift
