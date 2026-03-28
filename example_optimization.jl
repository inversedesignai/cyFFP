"""
    example_optimization.jl
    =======================
    Mock metalens inverse-design optimization pipeline.

    Demonstrates the full workflow:
      1. Surrogate model: design parameters → meta-atom transmissions t_vals
      2. PSF computation: t_vals → 2D intensity on Cartesian grid
      3. Focusing efficiency loss: fraction of power in the Airy disk
      4. Gradient-based optimization via L-BFGS (Optim.jl)

    The surrogate model is a simple sinusoidal phase profile parametrised
    by a vector of "pillar widths" w.  In a real pipeline, this would be
    replaced by a neural-network or lookup-table surrogate trained on
    RCWA/FDTD simulations of individual meta-atoms.

    Dependencies (beyond CyFFP):
        ] add ChainRulesCore Zygote Optim

    Run with:
        julia -t 388 example_optimization.jl   # use physical core count

    NOTE: This file is a template.  Adjust physical parameters, surrogate
    model, and loss function to your actual design problem.
"""

# ═══════════════════════════════════════════════════════════════
# 0. Setup
# ═══════════════════════════════════════════════════════════════

using ChainRulesCore, Zygote       # must be BEFORE include("cyffp.jl")

include("cyffp.jl")
using .CyFFP

using Optim                       # L-BFGS optimizer
using LinearAlgebra: norm


# ═══════════════════════════════════════════════════════════════
# 1. Physical parameters
# ═══════════════════════════════════════════════════════════════

const LAMBDA_UM  = 0.5            # wavelength (μm)
const PITCH_UM   = 0.3            # meta-atom unit cell pitch (μm)
const R_UM       = 30.0           # aperture radius (μm)  — small for demo
const NA         = 0.3            # numerical aperture
const ALPHA_DEG  = 10.0           # oblique incidence angle (degrees)
const N_CELLS    = round(Int, R_UM / PITCH_UM)

# PSF grid
const NXY   = 51                  # Cartesian output grid size
const L_MAX = 8                   # local mode truncation
const N_PSI = 32                  # azimuthal samples

# Derived
const K_VAL    = 2π / LAMBDA_UM
const F_UM     = R_UM * sqrt(1/NA^2 - 1)
const NA_EFF   = R_UM * cosd(ALPHA_DEG) / sqrt((R_UM*cosd(ALPHA_DEG))^2 + F_UM^2)
const RHO_AIRY = 0.61 * LAMBDA_UM / NA_EFF   # Airy radius in μm


# ═══════════════════════════════════════════════════════════════
# 2. Surrogate model: design parameters w → transmissions t
#
# A real surrogate would map meta-atom geometry (e.g. pillar width,
# height, shape) to the complex transmission t = |t| exp(iφ).
#
# Here we use a toy model:
#   - Each cell i has a design parameter w_i ∈ [0, 1]
#   - The transmission is t_i = exp(iφ_i) with |t|=1 (lossless)
#   - The phase is φ_i = 2π w_i  (linear map from parameter to phase)
#
# This is differentiable (∂t/∂w = 2πi t), and Zygote handles it
# automatically since it's just element-wise complex arithmetic.
# ═══════════════════════════════════════════════════════════════

"""
    surrogate(w) -> Vector{ComplexF64}

Map design parameters w ∈ ℝᴺ to complex meta-atom transmissions.
"""
function surrogate(w::AbstractVector{<:Real})
    return ComplexF64[exp(2π * im * w[i]) for i in eachindex(w)]
end


# ═══════════════════════════════════════════════════════════════
# 3. Loss function: relative focusing efficiency
#
# Focusing efficiency = power inside the Airy disk / total power
#                     = Σ I(x,y) [within circle] / Σ I(x,y)
#
# We MINIMISE negative efficiency (= maximise efficiency).
#
# The loss function composes:
#   w → surrogate(w) → t_vals → psf_intensity(plan, t_vals) → efficiency
#
# Zygote differentiates through the entire chain:
#   - surrogate: element-wise exp, Zygote handles natively
#   - psf_intensity: custom rrule (auto-defined when ChainRulesCore loaded)
#   - efficiency: indexing + sum, Zygote handles natively
# ═══════════════════════════════════════════════════════════════

"""
    focusing_efficiency_loss(w, plan, mask) -> Float64

Compute the negative relative focusing efficiency.
Minimising this maximises the fraction of PSF power inside the Airy disk.

Arguments:
- `w`: design parameter vector (length N_cells)
- `plan`: precomputed PSFPlan from prepare_psf
- `mask`: binary matrix (same size as PSF output) marking the Airy disk region
"""
function focusing_efficiency_loss(w, plan, mask)
    t = surrogate(w)
    I = psf_intensity(plan, t)

    total_power = sum(I) + 1e-20          # avoid division by zero
    airy_power  = sum(I .* mask)

    efficiency  = airy_power / total_power

    return -efficiency                    # minimise negative = maximise
end


# ═══════════════════════════════════════════════════════════════
# 4. Precompute the PSF plan and the Airy disk mask
# ═══════════════════════════════════════════════════════════════

println("Precomputing PSF plan...")
const PLAN = prepare_psf(PITCH_UM, N_CELLS;
                          lambda_um = LAMBDA_UM,
                          alpha_deg = ALPHA_DEG,
                          NA        = NA,
                          L_max     = L_MAX,
                          N_psi     = N_PSI,
                          Nxy       = NXY)

# Build the Airy disk mask: 1 inside circle of radius ρ_Airy, 0 outside.
# The PSF grid is uniform Cartesian centred at (0,0).
const HW = PLAN.psf_half_width_um
const X_GRID = collect(range(-HW, HW, length=NXY))
const Y_GRID = collect(range(-HW, HW, length=NXY))

const AIRY_MASK = Float64[
    sqrt(X_GRID[ix]^2 + Y_GRID[iy]^2) <= RHO_AIRY ? 1.0 : 0.0
    for iy in 1:NXY, ix in 1:NXY
]

println("  Airy radius: $(round(RHO_AIRY, digits=3)) μm")
println("  Mask pixels inside Airy disk: $(round(Int, sum(AIRY_MASK)))")


# ═══════════════════════════════════════════════════════════════
# 5. Initial design: ideal lens phase (the starting point)
#
# The ideal normal-incidence lens phase is:
#   φ(r) = -k [√(r² + f²) - f]
#
# We initialise w so that surrogate(w) ≈ ideal lens transmission.
# Since t = exp(2πi w) and ideal t = exp(iφ), we need w = φ/(2π).
# ═══════════════════════════════════════════════════════════════

r_centers = [(i - 0.5) * PITCH_UM for i in 1:N_CELLS]
ideal_phase = [-K_VAL * (sqrt(r^2 + F_UM^2) - F_UM) for r in r_centers]
w0 = mod.(ideal_phase ./ (2π), 1.0)   # wrap to [0, 1)

println("\nInitial design: ideal normal-incidence lens")
t0 = surrogate(w0)
I0 = psf_intensity(PLAN, t0)
eff0 = sum(I0 .* AIRY_MASK) / (sum(I0) + 1e-20)
println("  Initial focusing efficiency: $(round(100*eff0, digits=2))%")


# ═══════════════════════════════════════════════════════════════
# 6. Optimisation via Optim.jl L-BFGS
#
# Optim.jl's L-BFGS expects:
#   - f(x): scalar loss
#   - g!(G, x): in-place gradient
#
# We use Zygote.withgradient to compute both in a single forward+backward
# pass (the rrule ensures psf_intensity is not called twice).
#
# Optim also supports a combined fg! interface that returns both the
# value and fills the gradient — this is the most efficient pattern.
# ═══════════════════════════════════════════════════════════════

"""
    fg!(F, G, w)

Combined objective + gradient for Optim.jl.

If G is not nothing, fills G with the gradient.
If F is not nothing, returns the objective value.

Uses Zygote.withgradient for a single forward+backward pass.
"""
function fg!(F, G, w)
    if G !== nothing
        # Need gradient (and value)
        val, grads = Zygote.withgradient(w) do w_
            focusing_efficiency_loss(w_, PLAN, AIRY_MASK)
        end
        # Zygote returns complex gradient for real w (imaginary part
        # is zero for a real-valued loss of real parameters, but
        # numerical noise can create tiny imaginary parts).
        G .= real.(grads[1])
        if F !== nothing
            return val
        end
    else
        # Only need value (no gradient)
        if F !== nothing
            return focusing_efficiency_loss(w, PLAN, AIRY_MASK)
        end
    end
end


# ═══════════════════════════════════════════════════════════════
# 7. Run the optimisation
# ═══════════════════════════════════════════════════════════════

println("\n" * "="^60)
println("Starting L-BFGS optimisation")
println("="^60)

result = Optim.optimize(
    Optim.only_fg!(fg!),      # combined f + g! interface
    w0,                        # initial parameters
    Optim.LBFGS(),             # algorithm
    Optim.Options(
        iterations   = 20,     # max iterations (increase for real runs)
        show_trace   = true,
        show_every   = 1,
        g_tol        = 1e-6,
        f_tol        = 1e-8,
    ),
)

println("\n" * "="^60)
println("Optimisation complete")
println("="^60)
println(result)

# ═══════════════════════════════════════════════════════════════
# 8. Evaluate the optimised design
# ═══════════════════════════════════════════════════════════════

w_opt = Optim.minimizer(result)
t_opt = surrogate(w_opt)
I_opt = psf_intensity(PLAN, t_opt)
eff_opt = sum(I_opt .* AIRY_MASK) / (sum(I_opt) + 1e-20)

println("\nResults:")
println("  Initial efficiency: $(round(100*eff0, digits=2))%")
println("  Optimised efficiency: $(round(100*eff_opt, digits=2))%")
println("  Improvement: $(round(100*(eff_opt - eff0), digits=2)) pp")
println("  Iterations: $(Optim.iterations(result))")
println("  Converged: $(Optim.converged(result))")

# Optional: save the optimised PSF for plotting
# using Plots
# heatmap(X_GRID, Y_GRID, I_opt ./ maximum(I_opt);
#         xlabel="x (μm)", ylabel="y (μm)", title="Optimised PSF",
#         aspect_ratio=:equal, color=:inferno)
# savefig("optimised_psf.png")
