"""
    test_step5.jl
    =============
    Rigorous tests for inverse_hankel (Step 5).

    Tests:
      1. Round-trip: forward Hankel (Step 2) → inverse should recover input
      2. Parseval: ∫ |B_l|² kr dkr = ∫ |b_l|² ρ dρ
      3. Single-mode l=0: known Gaussian Hankel pair
      4. Output grid: ρ grid = original r grid (reciprocal of kr)
      5. Normal incidence: B_l = ã_l → b_l should match direct inverse of ã_l
      6. Multiple modes: l = -3..3, verify each independently
      7. End-to-end Steps 2→3→4→5 with realistic lens

    Run with: julia test_step5.jl
"""

include("../cyffp.jl")
using .CyFFP
using SpecialFunctions: besselj
using QuadGK

println("="^60)
println("Step 5: inverse_hankel — rigorous tests")
println("="^60)


# ─── Test 1: Round-trip forward→inverse ─────────────────────
println("\n--- Test 1: Round-trip H_m → H_m^{-1} recovers input ---")
# Create u_0(r) = exp(-r²/2σ²) on a log grid, forward transform,
# then inverse transform to recover it.
Nr = 1024
r = exp.(range(log(1e-3), log(1e3), length=Nr))
dln = log(r[2] / r[1])
kr = exp.(log(1.0 / r[end]) .+ dln .* (0:Nr-1))

max_rt_err = 0.0
for m in [0, 1, 3]
    sigma = 2.0
    u_orig = exp.(-r.^2 ./ (2 * sigma^2))

    # Forward: a_m(kr) = H_m[r u_m](kr)
    g = r .* u_orig
    raw_fwd = fftlog_hankel(g, dln, Float64(m))
    a_m = raw_fwd ./ kr

    # Inverse: b_m(ρ) = ∫ a_m(kr) J_m(kr ρ) kr dkr
    f = kr .* a_m
    raw_inv = fftlog_hankel(f, dln, Float64(m))
    # Output ρ grid = reciprocal of kr grid = original r grid
    rho = exp.(log(1.0 / kr[end]) .+ dln .* (0:Nr-1))
    b_m = raw_inv ./ rho

    # Compare b_m(ρ) with u_orig(r) where signal is significant
    # (avoid boundary regions where FFTLog wraps around)
    core = findall(1e-2 .< r .< 1e1)
    u_core = u_orig[core]
    b_core = real.(b_m[core])
    sig = findall(abs.(u_core) .> 1e-3 * maximum(abs.(u_core)))
    if !isempty(sig)
        err = maximum(abs.(b_core[sig] .- u_core[sig]) ./ (abs.(u_core[sig]) .+ 1e-30))
        global max_rt_err = max(max_rt_err, err)
        println("  m=$m: max round-trip error = $(round(err, sigdigits=3))")
    end
end
println("  Max overall: $(round(max_rt_err, sigdigits=3))")
@assert max_rt_err < 0.05 "Round-trip error too large"
println("  PASSED ✓")


# ─── Test 2: Parseval energy conservation ────────────────────
println("\n--- Test 2: Parseval ∫ |B_l|² kr dkr = ∫ |b_l|² ρ dρ ---")
# Use the inverse_hankel function (not raw fftlog_hankel)
B_test = zeros(ComplexF64, Nr, 3)  # L_max=1, modes l=-1,0,1
L_max_test = 1
for (li, l) in enumerate(-L_max_test:L_max_test)
    B_test[:, li] .= exp.(-kr.^2 ./ 2) .* cis.(kr .* 3.0) .* (kr .< 2π)
end

b_test, rho_test = inverse_hankel(B_test, L_max_test, collect(kr))

max_parseval_err = 0.0
for li in 1:3
    spec_energy = dln * sum(abs2(B_test[j, li]) * kr[j]^2 for j in 1:Nr)
    spat_energy = dln * sum(abs2(b_test[j, li]) * rho_test[j]^2 for j in 1:Nr)
    if spec_energy > 1e-20
        ratio = real(spat_energy) / real(spec_energy)
        err = abs(ratio - 1.0)
        global max_parseval_err = max(max_parseval_err, err)
        l = li - L_max_test - 1
        println("  l=$l: spectral=$(round(real(spec_energy), sigdigits=4)), spatial=$(round(real(spat_energy), sigdigits=4)), ratio=$(round(ratio, sigdigits=8))")
    end
end
@assert max_parseval_err < 1e-10 "Parseval violated"
println("  PASSED ✓")


# ─── Test 3: Known pair — Gaussian Hankel transform ──────────
println("\n--- Test 3: Gaussian H_0 pair: exp(-a²r²/2) ↔ exp(-k²/(2a²))/a² ---")
# H_0[r exp(-a²r²/2)](k) = ∫ r exp(-a²r²/2) J_0(kr) r dr
#                         = (1/a²) exp(-k²/(2a²))
# So if B_0(kr) = (1/a²) exp(-kr²/(2a²)), then
# b_0(ρ) = ∫ B_0(kr) J_0(kr ρ) kr dkr = exp(-a²ρ²/2)

a_val = 1.5
B_gauss = zeros(ComplexF64, Nr, 1)
B_gauss[:, 1] .= (1.0 / a_val^2) .* exp.(-kr.^2 ./ (2 * a_val^2))

b_gauss, rho_gauss = inverse_hankel(B_gauss, 0, collect(kr))
b_expected = exp.(-a_val^2 .* rho_gauss.^2 ./ 2)

# Compare in the core region (avoid boundary)
core = findall(1e-2 .< rho_gauss .< 3.0 / a_val)
sig = findall(b_expected[core] .> 1e-3)
err_gauss = maximum(abs.(real.(b_gauss[core[sig], 1]) .- b_expected[core[sig]]) ./
                     (b_expected[core[sig]] .+ 1e-30))
println("  Max error in core: $(round(err_gauss, sigdigits=3))")
@assert err_gauss < 0.05 "Gaussian pair error too large"
println("  PASSED ✓")


# ─── Test 4: Output grid check ──────────────────────────────
println("\n--- Test 4: ρ grid = reciprocal of kr grid ---")
_, rho_out = inverse_hankel(zeros(ComplexF64, Nr, 1), 0, collect(kr))
# ρ should be the same as the original r grid
rho_expected = exp.(log(1.0 / kr[end]) .+ dln .* (0:Nr-1))
grid_err = maximum(abs.(rho_out .- rho_expected))
println("  Max |ρ - expected|: $(round(grid_err, sigdigits=3))")
@assert grid_err < 1e-14 "ρ grid wrong"
# Also check it matches the original r grid
r_match_err = maximum(abs.(rho_out .- r) ./ r)
println("  Max |ρ/r - 1|: $(round(r_match_err, sigdigits=3))")
@assert r_match_err < 1e-10 "ρ grid doesn't match original r grid"
println("  PASSED ✓")


# ─── Test 5: QuadGK cross-check at selected ρ points ─────────
println("\n--- Test 5: b_l(ρ) vs QuadGK numerical integration ---")
# b_l(ρ) = ∫ B_l(kr) J_l(kr ρ) kr dkr
# Use a localized B_l and integrate directly.

L_max_5 = 2
B5 = zeros(ComplexF64, Nr, 2*L_max_5+1)
for (li, l) in enumerate(-L_max_5:L_max_5)
    B5[:, li] .= exp.(-((kr .- 3.0).^2) ./ 0.5) .+ 0.1im .* exp.(-kr.^2)
end
b5, rho5 = inverse_hankel(B5, L_max_5, collect(kr))

# Interpolator for B_l on the kr grid
function interp_B(B_col, kv)
    lk = log(kv); lk0 = log(kr[1])
    idx_f = (lk - lk0) / dln + 1.0
    j0 = clamp(floor(Int, idx_f), 1, Nr-1)
    w = idx_f - j0
    return (1-w) * B_col[j0] + w * B_col[j0+1]
end

max_err_5 = 0.0
test_rho_idx = [Nr÷4, Nr÷3, Nr÷2]  # pick a few ρ points
for l in [0, 1, -2]
    li = l + L_max_5 + 1
    for irho in test_rho_idx
        rv = rho5[irho]
        ref, _ = quadgk(kv -> interp_B(B5[:, li], kv) * besselj(l, kv * rv) * kv,
                         kr[1], kr[end]; rtol=1e-8)
        cyf = b5[irho, li]
        err = abs(cyf - ref) / (abs(ref) + 1e-30)
        global max_err_5 = max(max_err_5, err)
    end
end
println("  Max error vs QuadGK: $(round(max_err_5, sigdigits=3))")
@assert max_err_5 < 0.15 "QuadGK cross-check failed"
println("  PASSED ✓")


# ─── Test 6: Multiple modes — each l independent ────────────
println("\n--- Test 6: l modes are independent ---")
# Verify that changing B for one l doesn't affect b for another l.
B6a = copy(B5)
b6a, _ = inverse_hankel(B6a, L_max_5, collect(kr))

B6b = copy(B5)
B6b[:, 3] .*= 2.0  # modify l=0 only (column 3 for L_max=2)
b6b, _ = inverse_hankel(B6b, L_max_5, collect(kr))

# l=0 should change
@assert maximum(abs.(b6b[:, 3] .- b6a[:, 3])) > 1e-10 "l=0 didn't change"
# Other modes should not change
for li in [1, 2, 4, 5]
    err = maximum(abs.(b6b[:, li] .- b6a[:, li]))
    @assert err < 1e-14 "l=$(li - L_max_5 - 1) changed when it shouldn't"
end
println("  Modes are independent ✓")
println("  PASSED ✓")


# ─── Test 7: Steps 2→3→4→5 end-to-end ──────────────────────
println("\n--- Test 7: Steps 2→5 with small lens (R=10λ, α=5°) ---")
lambda_7 = 1.0
k_7      = 2π / lambda_7
R_7      = 10.0 * lambda_7
f_7      = R_7 / 0.3 * sqrt(1 - 0.3^2)
alpha_7  = deg2rad(5.0)
x0_7     = f_7 * tan(alpha_7)
M_max_7  = ceil(Int, k_7 * sin(alpha_7) * R_7) + 10
L_max_7  = 8

Nr_7     = 1024
r_7      = exp.(range(log(1e-3), log(1e3), length=Nr_7))
dln_7    = log(r_7[2] / r_7[1])
Ntheta_7 = 128
theta_7  = range(0.0, 2π, length=Ntheta_7+1)[1:end-1]

function u_obl_7(rv, th)
    d = sqrt((rv*cos(th) - x0_7)^2 + rv^2*sin(th)^2 + f_7^2)
    return exp(-im*k_7*(d - f_7)) * (rv <= R_7 ? 1.0 : 0.0)
end

u_field_7 = ComplexF64[u_obl_7(rv, th) for rv in r_7, th in theta_7]
u_m_7, _, m_pos_7 = angular_decompose(u_field_7, zeros(ComplexF64, Nr_7, Ntheta_7), M_max_7)
a_m_7, kr_7 = compute_scalar_coeffs(u_m_7, m_pos_7, collect(r_7))
a_tilde_7, m_full_7 = propagate_scalar(a_m_7, m_pos_7, kr_7, k_7, f_7)
B_7 = graf_shift(a_tilde_7, m_full_7, kr_7, x0_7, L_max_7; k=k_7)

# Step 5
b_7, rho_7 = inverse_hankel(B_7, L_max_7, kr_7)

# 7a: No NaN/Inf
@assert !any(isnan, b_7) "NaN in b"
@assert !any(isinf, b_7) "Inf in b"
println("  7a: No NaN/Inf ✓")

# 7b: Correct dimensions
@assert size(b_7) == (Nr_7, 2*L_max_7+1) "Wrong size: $(size(b_7))"
println("  7b: Size $(size(b_7)) ✓")

# 7c: ρ grid matches r grid
@assert maximum(abs.(rho_7 .- r_7) ./ r_7) < 1e-10 "ρ grid ≠ r grid"
println("  7c: ρ grid = r grid ✓")

# 7d: Parseval through Steps 2-5
spec_E = dln_7 * sum(abs2.(B_7[:, L_max_7+1]) .* kr_7.^2)  # l=0
spat_E = dln_7 * sum(abs2.(b_7[:, L_max_7+1]) .* rho_7.^2)
p_ratio = real(spat_E) / (real(spec_E) + 1e-30)
println("  7d: Parseval l=0: ratio=$(round(p_ratio, sigdigits=8))")
@assert abs(p_ratio - 1.0) < 1e-10 "Parseval failed"
println("  7d: Parseval ✓")

# 7e: Compare b_l PSF against brute-force Riemann sum
# Pick a few ρ points and compute PSF both ways
NA_eff = R_7 * cos(alpha_7) / sqrt((R_7*cos(alpha_7))^2 + f_7^2)
rho_airy = 0.61 * lambda_7 / NA_eff

prop_kr = findall(kr_7 .< k_7)
test_rho_pts = [0.01, 0.5, 1.0, rho_airy * 0.8, rho_airy]

println("  7e: FFTLog inverse vs brute-force Riemann sum:")
max_err_7e = 0.0
for rho_val in test_rho_pts
    # Brute-force: sum over kr
    bf = zero(ComplexF64)
    for (li, l) in enumerate(-L_max_7:L_max_7)
        bl_bf = dln_7 * sum(B_7[j, li] * besselj(l, kr_7[j] * rho_val) * kr_7[j]^2
                            for j in prop_kr)
        bf += bl_bf  # ψ = 0
    end

    # FFTLog: interpolate b_l at rho_val, then synthesize at ψ=0
    fftlog_val = zero(ComplexF64)
    lr = log(rho_val)
    lr0 = log(rho_7[1])
    idx_f = (lr - lr0) / dln_7 + 1.0
    j0 = clamp(floor(Int, idx_f), 1, Nr_7-1)
    w = idx_f - j0
    for (li, l) in enumerate(-L_max_7:L_max_7)
        bl_interp = (1-w) * b_7[j0, li] + w * b_7[j0+1, li]
        fftlog_val += bl_interp * exp(im * l * 0.0)  # ψ = 0
    end

    err = abs(fftlog_val - bf) / (abs(bf) + 1e-30)
    global max_err_7e = max(max_err_7e, err)
    println("    ρ=$(round(rho_val, digits=3))λ: FFTLog=$(round(abs(fftlog_val), sigdigits=4)), BF=$(round(abs(bf), sigdigits=4)), err=$(round(err, sigdigits=3))")
end
@assert max_err_7e < 0.15 "FFTLog inverse doesn't match brute-force"
println("  7e: FFTLog vs brute-force ✓")

println("  Test 7 PASSED ✓")


println("\n" * "="^60)
println("All Step 5 tests passed.")
println("="^60)
