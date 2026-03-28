# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CyFFP is a Julia module for **cylindrical near-to-far-field propagation** using scalar cylindrical harmonics. It computes the point spread function (PSF) of rotationally symmetric optical systems at large oblique angles by combining FFTLog Hankel transforms with Graf's addition theorem for a purely modal-space basis shift (no interpolation or Cartesian regridding).

**Authors:** Arvin Keshvari (supervised by Dr. Zin Lin)

## Running Tests

```bash
# Step-by-step unit tests
julia test/test_step1.jl          # Angular decomposition
julia test/test_step2_fftlog.jl   # FFTLog Hankel transform
julia test/test_step3.jl          # Scalar propagation + symmetry
julia test/test_step4.jl          # Graf shift
julia test/test_step5.jl          # Inverse Hankel
julia test/test_step6.jl          # Angular synthesis + full pipeline Airy

# End-to-end validation
julia test/test_scalar_airy.jl         # Airy disk with small lens (R=3λ)
julia test/test_bruteforce_reference.jl # QuadGK cross-check

# Scaling validation (R=3λ..1000λ, up to 558 modes)
julia test/test_scaling.jl

# Neumann path validation
julia test/test_neumann.jl

# Maximum-scale tests (up to 1581 modes)
julia test/test_maxscale.jl

# Adjoint (gradient) validation
julia test/test_adjoint.jl              # FD checks at 4 loss functions
julia test/test_adjoint_scaling.jl      # Scaling study M=53..1112

# Production scale (needs ~37 GB for full M_max)
julia -t auto test/test_production_aperture.jl
julia -t auto test/test_production_psf.jl   # Ideal oblique + Neumann comparison
```

Start Julia with `-t N` for N threads. The module uses `Threads.@threads` for shared-memory parallelism (no Distributed).

Tests use direct assertions and printed output.

Optional: `] add Plots` to generate PSF heatmap PNGs.

## Dependencies

- **FFTW** - Fast Fourier transforms
- **SpecialFunctions** - `loggamma` and `besselj`
- **QuadGK** - Numerical integration (tests only)
- **Plots** (optional) - PSF heatmap generation
- **ChainRulesCore**, **Zygote** (optional) - Automatic differentiation

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

### User-facing LPA driver (`compute_psf`, `prepare_psf`, `execute_psf`)

For metalens PSF computation from a radial transmission profile t(r):

**Single evaluation:**
```julia
result = compute_psf(t_vals, pitch_um; lambda_um=0.5, alpha_deg=30.0, NA=0.4)
# result.I is the normalized 2D PSF on a uniform Cartesian grid
# result.x_um, result.y_um are the axes in μm
```

- `t_vals`: complex transmission per unit cell (length N_cells)
- `pitch_um`: unit cell pitch in μm
- Modes computed via Jacobi-Anger (i^m t(r) J_m(kₓr)), bypasses Step 1 entirely
- Output: 2D PSF on uniform (x,y) grid centered at focal spot

**Optimization loop** (precompute geometry once, reuse):
```julia
plan = prepare_psf(pitch_um, N_cells; lambda_um=0.5, alpha_deg=30.0, NA=0.4)  # ~15s
result = execute_psf(plan, t_vals)   # ~55s per design
```

`prepare_psf` precomputes: Bessel matrix J_m(kₓr), FFTLog kernels, FFTW plans, cell lookup table. Saves ~15s per iteration vs `compute_psf`.

### Neumann shift-theorem fast path (`neumann_shift_coeffs`)

For LPA fields of the form t(r)·exp(ikₓ r cosθ), uses the Neumann addition formula to obtain ALL M_max modal coefficients from a **single** order-0 FFTLog call + interpolation + FFT per kr. Replaces `compute_scalar_coeffs` (Step 2) for this field type. Output feeds directly into `propagate_scalar` (Step 3).

**Limitation:** Linear interpolation of T₀ degrades at large M_max. Validated to <1% PSF accuracy up to M_max≈1,600 (R≤500λ, α≤30°). At production scale (M_max≈12,600), the PSF error reaches ~9% and there is no speed advantage. **Use the standard path for production.** The Neumann path is useful for smaller problems, coma physics studies, and cross-validation at moderate scales.

### Adjoint / gradient computation (`psf_adjoint`)

Hand-derived reverse-mode differentiation of the full `execute_psf` pipeline. No AD framework needed — the adjoint of each step (Cartesian interp, |u|², angular synthesis, inverse Hankel, Graf shift, propagation, FFTLog, mode construction, resampling) is implemented explicitly using the self-adjoint property of the Hankel transform (adjoint FFTLog uses conjugated kernel). Full derivation in `cyFFP_formulation.tex` §Adjoint.

**Manual usage (no Zygote):**
```julia
plan = prepare_psf(pitch_um, N_cells; lambda_um=0.5, alpha_deg=30.0, NA=0.4)
result = execute_psf(plan, t_vals)
dL_dI = your_loss_gradient(result.I_raw)  # Matrix{Float64}, same size as I_raw
dL_dt = psf_adjoint(plan, t_vals, result, dL_dI)  # ~2.4-4× forward cost
```

FD-verified to 1e-7 at N_cells up to 1667. Steps 5+4 (Graf+propagation) and Steps 3+2 (FFTLog+modes) of the adjoint are each fused to avoid large intermediate arrays (53 GB and 26 GB respectively eliminated).

**Performance:** On a 2-socket 350-thread production machine (R=1000μm, α=30°, M=6303, Nr=65536):
- Forward: **16s** (modes 0.2s + FFTLog 11s + prop 4s + graf 0.8s)
- Adjoint: **~3s** (s5+4 ~1.6s + s3+2 ~0.6s)
- Total iteration (fwd+adj): **~19s**

At full production scale (R=2000μm, M=12587, Nr=131072):
- Forward: **~120s** (modes + FFTLog + prop + graf)
- Adjoint: **~8s**
- Total iteration: **~130s**

**Critical Julia pitfall (resolved):** Both `execute_psf` and `psf_adjoint` previously suffered from closure boxing. Writing `var = nothing` (e.g. `u_m = nothing`, `B_bar = nothing`) after a `@threads` loop in the same function causes Julia to infer the variable as `Union{T, Nothing}`, which forces the entire `@threads` closure to use boxed (heap-allocated, dynamically-dispatched) variable access. This turned every inner-loop operation from ~1ns to ~100ns. Impact: forward modes step 15s→0.2s; adjoint s5+4 226s→1.6s; adjoint s3+2 23s→0.6s. Fix: never reassign captured variables to `nothing` in the same function scope as `@threads`; just call `GC.gc()` and let the GC collect unreferenced arrays.

### Differentiable PSF for Zygote (`psf_intensity`)

**`psf_intensity(plan, t_vals) → Matrix{Float64}`**: returns just the unnormalized PSF intensity (I_raw) as a plain matrix. This is the recommended entry point for Zygote differentiation — the rrule receives/returns a plain matrix cotangent, avoiding fragile NamedTuple extraction.

**Setup and usage:**
```julia
# 1. Load AD packages BEFORE cyffp.jl (so the rrule is auto-defined)
using ChainRulesCore, Zygote

# 2. Load the module
include("cyffp.jl")
using .CyFFP

# 3. Create a plan (once)
plan = prepare_psf(0.3, 6667; lambda_um=0.5, alpha_deg=30.0, NA=0.4)

# 4. Differentiate any loss function involving psf_intensity
grad = Zygote.gradient(t -> begin
    I = psf_intensity(plan, t)
    return -I[cy, cx] / (sum(I) + 1e-10)   # Strehl-like loss
end, t_vals)[1]
```

The rrule is defined automatically inside `cyffp.jl` when ChainRulesCore is detected as already loaded. No separate file needed. Zygote differentiates everything outside `psf_intensity` (loss composition, indexing, arithmetic) while the rrule routes the PSF gradient through `psf_adjoint`.

## Key Design Decisions

- **Scalar formulation**: Each Cartesian component of E satisfies the scalar Helmholtz equation (proven in the tex formulation). This gives ONE Hankel transform per mode at order m, replacing the 4-transform TE/TM approach. The TE/TM combination Ã = A^TE + A^TM was found to be incorrect — it conflates different Bessel structures and produces wrong PSFs.
- **Scalar symmetry**: For y-polarized illumination with tilt in xz-plane, E_y is even under y→-y, giving u_{-m} = u_m. Combined with J_{-m} = (-1)^m J_m, this gives ã_{-m} = (-1)^m ã_m. This halves the FFTLog calls.
- **Threading**: `Threads.@threads` for shared-memory parallelism (no serialization overhead). FFTW plan cache is warmed before threaded regions.
- **FFTLog for Hankel transforms**: Log-space FFT approach, stable for Bessel orders |m| ~ 600+. Filter kernel uses `loggamma` to avoid overflow. Key: |U(q)| = 1 for all q (complex conjugate Gamma pair).
- **All grids are log-spaced**: `r[j+1]/r[j] = const`. Output ρ-grid is the reciprocal of kr-grid. **Critical**: `r_min` must be ≤ `1/k = λ/(2π)` or propagating modes are lost.
- **Graf shift optimizations**: Bessel truncation at `|m| > kr·x₀ + 20`; Miller backward recurrence for J₀..J_{M_max} (~10× faster than independent calls); `@simd` inner loop; `J_{-m} = (-1)^m J_m` halves Bessel evaluations; self-normalizing identity `1 = J₀ + 2Σ J_{2k}` with overflow protection.
- **Normal incidence special case**: When α ≈ 0, x₀ ≈ 0, `J_m(0) = δ_{m,0}`, so the Graf shift becomes the identity.

## Formulation

- `doc/cyFFP_formulation.tex` — Self-contained formulation document: scalar proof, FFTLog, Graf derivation, Neumann shift theorem, cost analysis, adjoint derivation. TE/TM retained in Appendix A for reference.
- `cyFFP0.pdf`, `cyFFP1.pdf`, `cyFFP2.pdf` — Historical derivation PDFs (superseded by tex document).

## Deleted Files (for historical reference)

- `dev/` — Old buggy code with wrong FFTLog kernel, wrong TE/TM combination. Deleted.
- `test_step2_te_tm.jl` — Tested the old compute_TE_TM_coeffs (no longer exists). Deleted.
