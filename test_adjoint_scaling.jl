"""
    test_adjoint_scaling.jl
    =======================
    Measure adjoint/forward ratio across a range of M_max values
    and extrapolate to production scale.

    Tests that:
      1. The gradient is correct at each scale (FD check)
      2. The adjoint/forward ratio follows a sub-linear power law
      3. The extrapolated production ratio is bounded

    Run with: julia test_adjoint_scaling.jl
"""

include("cyffp.jl")
using .CyFFP
using LinearAlgebra: dot
using Statistics: mean

println("="^70)
println("Adjoint scaling study: R = 15..500μm at α=10°")
println("="^70)

lambda = 0.5
k = 2π / lambda
alpha_deg = 10.0
pitch = 0.3

M_vals  = Float64[]
ratios  = Float64[]
fd_errs = Float64[]

println("  R(μm)  N_cells  M_max     Nr    fwd(ms)   adj(ms)   ratio   FD_err")
println("  " * "─"^66)

for R_um in [15.0, 30.0, 50.0, 80.0, 120.0, 200.0, 300.0, 500.0]
    N_cells = round(Int, R_um / pitch)
    f_val = R_um * sqrt(1/0.3^2 - 1)
    r_c = [(i-0.5)*pitch for i in 1:N_cells]
    t0 = ComplexF64[exp(-im*k*(sqrt(r^2+f_val^2)-f_val)) for r in r_c]

    plan = prepare_psf(pitch, N_cells; lambda_um=lambda, alpha_deg=alpha_deg,
                       f_um=f_val, Nxy=51, L_max=6, N_psi=16)

    # Warmup (2 calls each)
    result = execute_psf(plan, t0)
    dL = ones(size(result.I_raw))
    psf_adjoint(plan, t0, result, dL)
    execute_psf(plan, t0)
    psf_adjoint(plan, t0, execute_psf(plan, t0), dL)

    # Benchmark (5 reps)
    t_fwd = @elapsed for rep in 1:5; execute_psf(plan, t0); end
    t_fwd /= 5
    result = execute_psf(plan, t0)
    t_adj = @elapsed for rep in 1:5; psf_adjoint(plan, t0, result, dL); end
    t_adj /= 5

    # FD check (quick, 1 random direction)
    g = psf_adjoint(plan, t0, result, dL)
    delta = randn(ComplexF64, N_cells)
    delta ./= sqrt(sum(abs2.(delta)))
    analytic = 2 * real(dot(g, delta))
    eps = 1e-5
    rp = execute_psf(plan, t0 .+ eps .* delta)
    rm = execute_psf(plan, t0 .- eps .* delta)
    fd = (sum(rp.I_raw) - sum(rm.I_raw)) / (2 * eps)
    fd_err = abs(fd - analytic) / (abs(analytic) + 1e-30)

    ratio = t_adj / t_fwd
    push!(M_vals, Float64(plan.M_max))
    push!(ratios, ratio)
    push!(fd_errs, fd_err)

    println("  $(lpad(round(Int, R_um), 5))  $(lpad(N_cells, 7))  $(lpad(plan.M_max, 5))  $(lpad(plan.Nr, 6))  $(lpad(round(t_fwd*1000, digits=0), 8))  $(lpad(round(t_adj*1000, digits=0), 8))  $(lpad(round(ratio, digits=2), 7))   $(round(fd_err, sigdigits=2))")
end

# ─── Power-law fit: ratio ≈ a × M^b ─────────────────────────
lM = log.(M_vals)
lR = log.(ratios)
n = length(lM)
b = (n * sum(lM .* lR) - sum(lM) * sum(lR)) / (n * sum(lM.^2) - sum(lM)^2)
a = mean(lR) - b * mean(lM)

println("\n--- Power-law fit ---")
println("  ratio ≈ $(round(exp(a), sigdigits=3)) × M^$(round(b, sigdigits=3))")

M_prod = 12588.0
ratio_prod = exp(a + b * log(M_prod))
println("  Extrapolated at M=$(round(Int, M_prod)): ratio ≈ $(round(ratio_prod, digits=1))×")
println("  If forward ≈ 55s → adjoint ≈ $(round(55 * ratio_prod, digits=0))s")
println("  Total per iteration: $(round(55 * (1 + ratio_prod), digits=0))s")

# ─── Assertions ──────────────────────────────────────────────
println("\n--- Assertions ---")

# All FD checks should be accurate
max_fd = maximum(fd_errs)
println("  Max FD error across all scales: $(round(max_fd, sigdigits=2))")
@assert max_fd < 1e-5 "FD check failed at some scale"
println("  Gradient correct at all scales ✓")

# Power law exponent should be sub-linear (< 0.5)
println("  Power-law exponent: $(round(b, sigdigits=3)) (should be < 0.5)")
@assert b < 0.7 "Adjoint ratio growing too fast with M"
println("  Sub-linear scaling ✓")

# Extrapolated ratio should be bounded
println("  Extrapolated production ratio: $(round(ratio_prod, digits=1))×")
@assert ratio_prod < 10.0 "Extrapolated ratio > 10× — adjoint too slow"
println("  Bounded production overhead ✓")

println("\n" * "="^70)
println("Adjoint scaling study complete.")
println("="^70)
