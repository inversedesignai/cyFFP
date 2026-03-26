"""
    test_step6.jl
    =============
    Rigorous tests for angular_synthesis (Step 6).

    Tests:
      1. Single mode l=0: u(ρ,ψ) = b_0(ρ) (no ψ dependence)
      2. Single mode l=1: |u|² = |b_1|² (constant in ψ)
      3. Two modes l=±1: gives cos(ψ) pattern
      4. Parseval over ψ: (1/N_ψ) Σ_s |u(ρ,ψ_s)|² = Σ_l |b_l(ρ)|²
      5. Output ψ grid: uniform on [0, 2π)
      6. Manual DFT vs angular_synthesis
      7. Full pipeline Steps 1→6 with ideal oblique lens → Airy disk

    Run with: julia test_step6.jl
"""

include("cyffp.jl")
using .CyFFP
using FFTW
using SpecialFunctions: besselj

println("="^60)
println("Step 6: angular_synthesis — rigorous tests")
println("="^60)


# ─── Test 1: Single mode l=0 → no ψ dependence ──────────────
println("\n--- Test 1: l=0 only → u(ρ,ψ) = b_0(ρ) ---")
Nrho = 64
L_max = 3
N_psi = 16
b1 = zeros(ComplexF64, Nrho, 2L_max+1)
b1[:, L_max+1] .= randn(ComplexF64, Nrho)  # l=0 column

u1, psi1 = angular_synthesis(b1, L_max, N_psi)

# u should be constant in ψ and equal to b_0
max_err_1 = 0.0
for ip in 1:N_psi
    err = maximum(abs.(u1[:, ip] .- b1[:, L_max+1]))
    global max_err_1 = max(max_err_1, err)
end
println("  Max |u(ρ,ψ) - b_0(ρ)|: $(round(max_err_1, sigdigits=3))")
@assert max_err_1 < 1e-12 "l=0 should give ψ-independent output"
println("  PASSED ✓")


# ─── Test 2: Single mode l=1 → |u|² constant in ψ ───────────
println("\n--- Test 2: l=1 only → |u(ρ,ψ)|² constant in ψ ---")
b2 = zeros(ComplexF64, Nrho, 2L_max+1)
b2[:, L_max+2] .= randn(ComplexF64, Nrho)  # l=1 column

u2, _ = angular_synthesis(b2, L_max, N_psi)

max_err_2 = 0.0
for ir in 1:Nrho
    I_row = abs2.(u2[ir, :])
    if maximum(I_row) > 1e-20
        variation = (maximum(I_row) - minimum(I_row)) / maximum(I_row)
        global max_err_2 = max(max_err_2, variation)
    end
end
println("  Max ψ-variation of |u|²: $(round(max_err_2, sigdigits=3))")
@assert max_err_2 < 1e-12 "|u|² should be constant for single l mode"
println("  PASSED ✓")


# ─── Test 3: l=±1 with equal coefficients → cos(ψ) pattern ──
println("\n--- Test 3: b_1 = b_{-1} → u ∝ cos(ψ) ---")
b3 = zeros(ComplexF64, Nrho, 2L_max+1)
profile = randn(Nrho)  # real-valued radial profile
b3[:, L_max]   .= profile  # l = -1
b3[:, L_max+2] .= profile  # l = +1

u3, psi3 = angular_synthesis(b3, L_max, N_psi)

# u(ρ,ψ) = b_1 (e^{iψ} + e^{-iψ}) = 2 b_1 cos(ψ)
max_err_3 = 0.0
for ip in 1:N_psi
    expected = 2.0 .* profile .* cos(psi3[ip])
    err = maximum(abs.(real.(u3[:, ip]) .- expected))
    global max_err_3 = max(max_err_3, err)
end
println("  Max |u - 2b cos(ψ)|: $(round(max_err_3, sigdigits=3))")
@assert max_err_3 < 1e-12 "cos(ψ) pattern wrong"
# Also check imaginary part is zero (since b is real and symmetric)
max_imag = maximum(abs.(imag.(u3)))
println("  Max |imag(u)|: $(round(max_imag, sigdigits=3))")
@assert max_imag < 1e-12 "Should be real for real symmetric coefficients"
println("  PASSED ✓")


# ─── Test 4: Parseval over ψ ─────────────────────────────────
println("\n--- Test 4: Parseval: (1/N_ψ) Σ|u|² = Σ|b_l|² ---")
b4 = randn(ComplexF64, Nrho, 2L_max+1)
u4, _ = angular_synthesis(b4, L_max, 32)

max_err_4 = 0.0
for ir in 1:Nrho
    psi_sum = sum(abs2.(u4[ir, :])) / 32
    mode_sum = sum(abs2.(b4[ir, :]))
    if mode_sum > 1e-20
        err = abs(psi_sum - mode_sum) / mode_sum
        global max_err_4 = max(max_err_4, err)
    end
end
println("  Max Parseval error: $(round(max_err_4, sigdigits=3))")
@assert max_err_4 < 1e-12 "Parseval over ψ violated"
println("  PASSED ✓")


# ─── Test 5: Output ψ grid ──────────────────────────────────
println("\n--- Test 5: ψ grid is uniform on [0, 2π) ---")
_, psi5 = angular_synthesis(zeros(ComplexF64, 2, 2L_max+1), L_max, 32)
@assert length(psi5) == 32
@assert abs(psi5[1]) < 1e-15 "ψ[1] should be 0"
@assert abs(psi5[2] - 2π/32) < 1e-14 "ψ spacing wrong"
@assert psi5[end] < 2π "ψ should not include 2π"
println("  ψ = [0, 2π/N, ..., 2π(N-1)/N] ✓")
println("  PASSED ✓")


# ─── Test 6: Manual DFT vs angular_synthesis ─────────────────
println("\n--- Test 6: Manual DFT matches angular_synthesis ---")
N_psi_6 = 24
b6 = randn(ComplexF64, Nrho, 2L_max+1)
u6, psi6 = angular_synthesis(b6, L_max, N_psi_6)

max_err_6 = 0.0
for ir in 1:min(Nrho, 10)  # spot check
    for (ip, psi_val) in enumerate(psi6)
        manual = zero(ComplexF64)
        for (li, l) in enumerate(-L_max:L_max)
            manual += b6[ir, li] * exp(im * l * psi_val)
        end
        err = abs(u6[ir, ip] - manual)
        global max_err_6 = max(max_err_6, err)
    end
end
println("  Max |IFFT - manual|: $(round(max_err_6, sigdigits=3))")
@assert max_err_6 < 1e-12 "IFFT doesn't match manual DFT"
println("  PASSED ✓")


# ─── Test 7: Full pipeline Steps 1→6 with Airy disk ──────────
println("\n--- Test 7: Full pipeline (Steps 1→6) — Airy disk (R=10λ, α=5°) ---")
lambda_7 = 1.0
k_7      = 2π / lambda_7
R_7      = 10.0
f_7      = R_7 / 0.3 * sqrt(1 - 0.3^2)
alpha_7  = deg2rad(5.0)
x0_7     = f_7 * tan(alpha_7)
M_max_7  = ceil(Int, k_7 * sin(alpha_7) * R_7) + 10
L_max_7  = 8
N_psi_7  = 32

Nr_7     = 1024
r_7      = exp.(range(log(1e-3), log(1e3), length=Nr_7))
dln_7    = log(r_7[2] / r_7[1])
Ntheta_7 = 128
theta_7  = range(0.0, 2π, length=Ntheta_7+1)[1:end-1]

NA_eff = R_7 * cos(alpha_7) / sqrt((R_7*cos(alpha_7))^2 + f_7^2)
rho_airy = 0.61 * lambda_7 / NA_eff
println("  R=$(R_7)λ, f=$(round(f_7, digits=1))λ, α=5°, NA_eff=$(round(NA_eff, digits=3))")
println("  ρ_Airy = $(round(rho_airy, digits=3))λ")

function u_obl(rv, th)
    d = sqrt((rv*cos(th) - x0_7)^2 + rv^2*sin(th)^2 + f_7^2)
    return exp(-im*k_7*(d - f_7)) * (rv <= R_7 ? 1.0 : 0.0)
end

u_field = ComplexF64[u_obl(rv, th) for rv in r_7, th in theta_7]

# Steps 1-4
u_m, _, m_pos = angular_decompose(u_field, zeros(ComplexF64, Nr_7, Ntheta_7), M_max_7)
a_m, kr_7 = compute_scalar_coeffs(u_m, m_pos, collect(r_7))
a_tilde, m_full = propagate_scalar(a_m, m_pos, kr_7, k_7, f_7)
B = graf_shift(a_tilde, m_full, kr_7, x0_7, L_max_7; k=k_7)

# Step 5
b, rho = inverse_hankel(B, L_max_7, kr_7)

# Step 6
u_psf, psi = angular_synthesis(b, L_max_7, N_psi_7)

# PSF intensity
I_psf = abs2.(u_psf)

# 7a: No NaN/Inf
@assert !any(isnan, u_psf) "NaN in u_psf"
@assert !any(isinf, u_psf) "Inf in u_psf"
println("  7a: No NaN/Inf ✓")

# 7b: Find peak — should be near ρ = 0
# ψ-averaged radial profile
I_avg = [sum(I_psf[ir, :]) / N_psi_7 for ir in 1:Nr_7]
peak_ir = argmax(I_avg)
peak_rho = rho[peak_ir]
println("  7b: PSF peak at ρ = $(round(peak_rho, digits=4))λ (expected ≈ 0)")
@assert peak_rho < 0.5 "PSF should peak near ρ=0"
println("  7b: Peak location ✓")

# 7c: Radial profile shape — compare against brute-force at ψ=0
prop_kr = findall(kr_7 .< k_7)
rho_check = [0.01, 0.5, 1.0, 1.5, rho_airy]
println("  7c: FFTLog pipeline vs brute-force at ψ=0:")
max_err_7c = 0.0
for rho_val in rho_check
    # Brute-force
    bf = zero(ComplexF64)
    for (li, l) in enumerate(-L_max_7:L_max_7)
        bl_bf = dln_7 * sum(B[j, li] * besselj(l, kr_7[j] * rho_val) * kr_7[j]^2
                            for j in prop_kr)
        bf += bl_bf
    end

    # FFTLog pipeline: interpolate u_psf at (rho_val, ψ=0)
    lr = log(rho_val); lr0 = log(rho[1])
    idx_f = (lr - lr0) / dln_7 + 1.0
    j0 = clamp(floor(Int, idx_f), 1, Nr_7-1)
    w = idx_f - j0
    fftlog_val = (1-w) * u_psf[j0, 1] + w * u_psf[j0+1, 1]  # ψ=0 is column 1

    err = abs(fftlog_val - bf) / (abs(bf) + 1e-30)
    global max_err_7c = max(max_err_7c, err)
    println("    ρ=$(round(rho_val, digits=2))λ: pipeline=$(round(abs(fftlog_val), sigdigits=4)), BF=$(round(abs(bf), sigdigits=4)), err=$(round(err, sigdigits=3))")
end
@assert max_err_7c < 0.15 "Pipeline doesn't match brute-force"
println("  7c: Pipeline vs brute-force ✓")

# 7d: Airy first zero — find first minimum in ψ-averaged profile
# The ρ grid is log-spaced, so find the minimum by scanning all points
# with ρ > 0.5λ (past the main lobe)
first_min_ir = 0
for j in 2:Nr_7-1
    if rho[j] > 0.5 && I_avg[j] < I_avg[j-1] && I_avg[j] < I_avg[j+1]
        global first_min_ir = j
        break
    end
end
if first_min_ir > 0
    first_zero = rho[first_min_ir]
    airy_err = abs(first_zero - rho_airy) / rho_airy
    min_val = I_avg[first_min_ir] / I_avg[peak_ir]
    println("  7d: First zero at ρ = $(round(first_zero, digits=3))λ (predicted $(round(rho_airy, digits=3))λ, error $(round(100*airy_err, digits=1))%)")
    println("      Min intensity: $(round(min_val, sigdigits=3)) of peak")
    @assert airy_err < 0.15 "Airy zero position wrong"
    @assert min_val < 0.01 "First minimum not deep enough"
    println("  7d: Airy disk ✓")
else
    # Fallback: check that intensity at ρ ≈ ρ_airy is very small relative to peak
    idx_airy = argmin(abs.(rho .- rho_airy))
    ratio = I_avg[idx_airy] / I_avg[peak_ir]
    println("  7d: No local min found, but I(ρ_airy)/I_peak = $(round(ratio, sigdigits=3))")
    @assert ratio < 0.01 "Intensity at Airy zero not small enough"
    println("  7d: Airy intensity check ✓")
end

println("  Test 7 PASSED ✓")


println("\n" * "="^60)
println("All Step 6 tests passed.")
println("="^60)
