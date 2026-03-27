"""
    test_adjoint.jl
    ===============
    Finite-difference verification of psf_adjoint.

    Tests that the analytic gradient matches the numerical gradient
    for several loss functions and random perturbations.

    Run with: julia test_adjoint.jl
"""

include("cyffp.jl")
using .CyFFP
using LinearAlgebra: dot

println("="^60)
println("Adjoint (psf_adjoint) finite-difference verification")
println("="^60)

# ─── Setup: small test lens ──────────────────────────────────
pitch = 0.3
R = 15.0
N_cells = round(Int, R / pitch)
k = 2pi / 0.5
f = R * sqrt(1/0.3^2 - 1)
r_c = [(i-0.5)*pitch for i in 1:N_cells]
t0 = ComplexF64[exp(-im*k*(sqrt(r^2+f^2)-f)) for r in r_c]

plan = prepare_psf(pitch, N_cells; lambda_um=0.5, alpha_deg=10.0, f_um=f,
                   Nxy=51, L_max=6, N_psi=16)


# ─── Test 1: L = sum(I_raw)  (total intensity) ──────────────
println("\n--- Test 1: L = sum(I_raw) ---")

result0 = execute_psf(plan, t0)
L0 = sum(result0.I_raw)
dL_dI = ones(size(result0.I_raw))
dL_dt = psf_adjoint(plan, t0, result0, dL_dI)

delta = randn(ComplexF64, N_cells)
delta ./= sqrt(sum(abs2.(delta)))

analytic = 2 * real(dot(dL_dt, delta))
println("  Analytic directional derivative: $(round(analytic, sigdigits=8))")

best_err = Inf
for eps in [1e-4, 1e-5, 1e-6, 1e-7]
    rp = execute_psf(plan, t0 .+ eps .* delta)
    rm = execute_psf(plan, t0 .- eps .* delta)
    fd = (sum(rp.I_raw) - sum(rm.I_raw)) / (2 * eps)
    err = abs(fd - analytic) / (abs(analytic) + 1e-30)
    global best_err = min(best_err, err)
    println("  eps=$eps: FD=$(round(fd, sigdigits=8))  err=$(round(err, sigdigits=3))")
end
@assert best_err < 0.05 "Gradient check failed for L=sum(I_raw)"
println("  PASSED (best rel err = $(round(best_err, sigdigits=3)))")


# ─── Test 2: L = max(I_raw) approximation (peak intensity) ──
println("\n--- Test 2: L = I_raw at peak pixel ---")

peak_idx = argmax(result0.I_raw)
dL_dI2 = zeros(size(result0.I_raw))
dL_dI2[peak_idx] = 1.0
dL_dt2 = psf_adjoint(plan, t0, result0, dL_dI2)

delta2 = randn(ComplexF64, N_cells)
delta2 ./= sqrt(sum(abs2.(delta2)))
analytic2 = 2 * real(dot(dL_dt2, delta2))

best_err2 = Inf
for eps in [1e-4, 1e-5, 1e-6, 1e-7]
    rp = execute_psf(plan, t0 .+ eps .* delta2)
    rm = execute_psf(plan, t0 .- eps .* delta2)
    fd = (rp.I_raw[peak_idx] - rm.I_raw[peak_idx]) / (2 * eps)
    err = abs(fd - analytic2) / (abs(analytic2) + 1e-30)
    global best_err2 = min(best_err2, err)
    println("  eps=$eps: FD=$(round(fd, sigdigits=8))  err=$(round(err, sigdigits=3))")
end
@assert best_err2 < 0.05 "Gradient check failed for peak pixel"
println("  PASSED (best rel err = $(round(best_err2, sigdigits=3)))")


# ─── Test 3: L = weighted sum (random weights) ──────────────
println("\n--- Test 3: L = sum(W .* I_raw) with random weights ---")

W = randn(size(result0.I_raw))
L3 = sum(W .* result0.I_raw)
dL_dt3 = psf_adjoint(plan, t0, result0, W)

delta3 = randn(ComplexF64, N_cells)
delta3 ./= sqrt(sum(abs2.(delta3)))
analytic3 = 2 * real(dot(dL_dt3, delta3))

best_err3 = Inf
for eps in [1e-5, 1e-6, 1e-7]
    rp = execute_psf(plan, t0 .+ eps .* delta3)
    rm = execute_psf(plan, t0 .- eps .* delta3)
    fd = (sum(W .* rp.I_raw) - sum(W .* rm.I_raw)) / (2 * eps)
    err = abs(fd - analytic3) / (abs(analytic3) + 1e-30)
    global best_err3 = min(best_err3, err)
    println("  eps=$eps: FD=$(round(fd, sigdigits=8))  err=$(round(err, sigdigits=3))")
end
@assert best_err3 < 0.05 "Gradient check failed for weighted sum"
println("  PASSED (best rel err = $(round(best_err3, sigdigits=3)))")


# ─── Test 4: Gradient w.r.t. single cell (sparsity check) ───
println("\n--- Test 4: perturb single cell, check gradient entry ---")

cell_idx = N_cells ÷ 2
dL_dI4 = ones(size(result0.I_raw))
dL_dt4 = psf_adjoint(plan, t0, result0, dL_dI4)

# Perturb only cell cell_idx
delta4 = zeros(ComplexF64, N_cells)
delta4[cell_idx] = 1.0 + 0.5im
analytic4 = 2 * real(dot(dL_dt4, delta4))

best_err4 = Inf
for eps in [1e-5, 1e-6, 1e-7]
    rp = execute_psf(plan, t0 .+ eps .* delta4)
    rm = execute_psf(plan, t0 .- eps .* delta4)
    fd = (sum(rp.I_raw) - sum(rm.I_raw)) / (2 * eps)
    err = abs(fd - analytic4) / (abs(analytic4) + 1e-30)
    global best_err4 = min(best_err4, err)
    println("  eps=$eps: FD=$(round(fd, sigdigits=8))  err=$(round(err, sigdigits=3))")
end
@assert best_err4 < 0.05 "Single-cell gradient check failed"
println("  PASSED (best rel err = $(round(best_err4, sigdigits=3)))")


println("\n" * "="^60)
println("All adjoint tests passed.")
println("="^60)
