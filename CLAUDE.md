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

# Hankel round-trip (prerequisite for doublet pipeline)
julia -t auto test/test_hankel_roundtrip.jl  # L² err ~1e-4, Parseval exact

# Doublet adjoint validation
julia -t 4 test/test_adjoint_doublet.jl         # FD checks, 4 configs, rel err ~1e-9
julia -t auto test/test_doublet_production.jl    # R=1mm production timing + FD check

# Gradient check for φ-parameterized PSF (singlet)
julia -t auto test/test_grad_phi.jl       # 12 trials, Zygote vs FD
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
- **NLopt** (optional) - Optimization (CCSA/MMA) for optimize_psf scripts

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

**Production performance** (2-socket 350-thread machine, α=30°, NA=0.4):

| R (μm) | M_max | Forward | Zygote (fwd+adj) | Ratio |
|--------|-------|---------|-------------------|-------|
| 1000 | 6303 | 4.8s | 7.7s | 1.61× |
| 2000 | 12587 | 14.3s | 23.5s | **1.65×** |

The adjoint adds only **~65%** overhead to the forward. 100 L-BFGS iterations: **~39 minutes**.

**Critical Julia pitfall (resolved):** Both `execute_psf` and `psf_adjoint` previously suffered from closure boxing. Writing `var = nothing` (e.g. `u_m = nothing`, `B_bar = nothing`) after a `@threads` loop in the same function causes Julia to infer the variable as `Union{T, Nothing}`, which forces the entire `@threads` closure to use boxed (heap-allocated, dynamically-dispatched) variable access. This turned every inner-loop operation from ~1ns to ~100ns. Impact: forward modes step 15s→0.2s; adjoint s5+4 226s→1.6s; adjoint s3+2 23s→0.6s. Fix: never reassign captured variables to `nothing` in the same function scope as `@threads`; just call `GC.gc()` and let the GC collect unreferenced arrays.

### Doublet pipeline (`execute_psf_doublet`, `psf_adjoint_doublet`)

For two rotationally symmetric metasurfaces t₁(r), t₂(r) separated by distance d. The pipeline inserts extra per-mode steps between FFTLog and prop+Graf:

1. Mode construction: `u_m = i^m t₁(r) J_m(kₓr)`
2. Forward Hankel → `raw₁`
3. Propagate by d: `raw₁ *= exp(ikz d)` (kr cancellation: skip /kr and *kr)
4. Inverse Hankel → `raw₂`
5. Save `u_mid = raw₂/ρ` for adjoint; compute `g = t₂·raw₂` (ρ=r cancellation: skip /ρ and *r)
6. Forward Hankel → `a_m_2 = raw₃/kr`
7. Propagate by (f−d) + fused Graf shift (same as singlet)
8. Steps 5-8: same as singlet

Steps 1-6 are fused per-mode and threaded. The kr/kr and r/ρ cancellations (exploiting FFTLog grid self-reciprocity) eliminate 4 element-wise loops per mode in the forward and 3 in the adjoint.

**Usage:**
```julia
plan = prepare_psf(pitch_um, N_cells; lambda_um=0.5, alpha_deg=30.0, NA=0.4)
result = execute_psf_doublet(plan, t1_vals, t2_vals; d_um=200.0)
dL_dt1, dL_dt2 = psf_adjoint_doublet(plan, t1_vals, t2_vals, result, dL_dI)
```

The same `PSFPlan` is reused — the only new parameter is `d_um`. The adjoint returns Wirtinger derivatives for both surfaces. FD-verified to ~1e-9 (small scale) and ~1e-5 (production).

**Production performance** (R=1mm, M_max=6303, 768 threads):

| Component | Time |
|-----------|------|
| Forward (execute_psf_doublet) | 4.7s |
| Adjoint (psf_adjoint_doublet) | 6.8s |
| Ratio (fwd+adj)/fwd | 1.86× |

**Wirtinger convention note:** `psf_adjoint` and `psf_adjoint_doublet` return the Wirtinger derivative ∂L/∂t̄. For a real direction δt, the directional derivative is `2 Re(conj(∂L/∂t̄) · δt)`. The `psf_intensity` rrule for Zygote returns `2 × psf_adjoint(...)` to match ChainRules cotangent convention `δL = Re(conj(Δt) · δt)`.

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

### Phase optimization scripts

**Singlet** (`optimize_psf.jl`): Maximizes focusing efficiency η = sum(I_raw·mask)·Δx²/(πR²) within the Airy disk for a single metasurface (R=2mm, α=30°, NA=0.4). Uses NLopt CCSA with manual adjoint gradients through the φ-parameterization t = exp(iφ).

```bash
julia -t auto optimize_psf.jl ideal           # start from ideal lens phase
julia -t auto optimize_psf.jl quadratic       # coma-free quadratic phase
julia -t auto optimize_psf.jl random [seed]   # random init
julia -t auto optimize_psf.jl gradcheck ideal  # FD sanity check
```

**Doublet** (`optimize_psf_doublet.jl`): Joint optimization of (φ₁, φ₂) for two surfaces separated by d (R=1mm, same angle/NA). Optimization variable x = [φ₁; φ₂].

```bash
julia -t auto optimize_psf_doublet.jl 200 ideal uniform        # d=200μm
julia -t auto optimize_psf_doublet.jl 500 quadratic quadratic  # d=500μm
julia -t auto optimize_psf_doublet.jl gradcheck 200 ideal uniform
```

Both scripts save φ data and PSF heatmap PNGs every 10 iterations to `opt_results/` or `opt_doublet_results/`.

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
