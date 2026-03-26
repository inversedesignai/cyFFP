"""
    test_step2_te_tm.jl
    ====================
    Rigorous tests for compute_TE_TM_coeffs.

    Tests the TE/TM projection formulas:
      A^TE_m = (i/2)[H_{m+1} + H_{m-1}](rE_r) + (1/2)[H_{m-1} - H_{m+1}](rE_θ)
      A^TM_m = -(kz/k)/2 [H_{m-1} - H_{m+1}](rE_r) - i(kz/k)/2 [H_{m-1} + H_{m+1}](rE_θ)

    Dependencies: ] add FFTW SpecialFunctions QuadGK

    Run with: julia test_step2_te_tm.jl
"""

include("cyffp.jl")
using .CyFFP
using SpecialFunctions: besselj
using QuadGK

println("="^60)
println("Step 2 (part 2): compute_TE_TM_coeffs — rigorous tests")
println("="^60)

# ─── Shared grid (padded for FFTLog boundary accuracy) ────────
Nr    = 1024
r_min = 1e-4
r_max = 1e4
r     = exp.(range(log(r_min), log(r_max), length=Nr))
dln   = log(r[2] / r[1])
kr    = exp.(log(1.0 / r[end]) .+ dln .* (0:Nr-1))

# Wavenumber for test
k_test = 2π   # λ = 1

println("Grid: Nr=$Nr, r=[$r_min,$r_max], kr=[$(round(kr[1],sigdigits=3)),$(round(kr[end],sigdigits=3))]")
println("k = $(round(k_test, sigdigits=4))")


# ─── Test 1: Single mode, E_θ = 0 (scalar-like) ──────────────
println("\n--- Test 1: Single mode m=0, E_θ=0, verify A^TE via QuadGK ---")
# Input: E_{0,r}(r) = exp(-r²/2), E_{0,θ} = 0
# A^TE_0 = (i/2)[H_1[r E_r] + H_{-1}[r E_r]]
#        = (i/2)[H_1[r e^{-r²/2}] + H_{-1}[r e^{-r²/2}]]
# Since J_{-1}(x) = -J_1(x), H_{-1}[f] = -H_1[f], so:
# A^TE_0 = (i/2)(H_1 - H_1) = 0  ← TE vanishes for m=0, E_θ=0!

m_pos = [0]
Em_r  = reshape(ComplexF64[exp(-rv^2/2) for rv in r], Nr, 1)
Em_th = zeros(ComplexF64, Nr, 1)

A_TE, A_TM, kr_out = compute_TE_TM_coeffs(Em_r, Em_th, m_pos, collect(r), k_test)

# TE should be zero (or near-zero)
te_max = maximum(abs.(A_TE))
println("  |A^TE_0|_max = $(round(te_max, sigdigits=3))  (expected ~0)")
@assert te_max < 0.01 "A^TE should vanish for m=0, E_θ=0"

# TM should be nonzero: A^TM_0 = -(kz/k)/2 [H_{-1} - H_1](r E_r) = (kz/k) H_1[r E_r]
# Verify at a few kr points via QuadGK
max_err_1 = 0.0
for kval in [0.5, 1.0, 3.0]
    ik = argmin(abs.(kr_out .- kval))
    kz_k = sqrt(max(1.0 - (kval/k_test)^2, 0.0))

    H1_ref, _ = quadgk(rr -> rr*exp(-rr^2/2) * besselj(1, kval*rr), 0, 30; rtol=1e-10)
    Hm1_ref = -H1_ref   # H_{-1}[f] = -H_1[f]
    atm_ref = -(kz_k/2) * (Hm1_ref - H1_ref)   # = -(kz/k)/2 × (-2 H_1) = kz/k × H_1

    err = abs(A_TM[ik, 1] - atm_ref) / (abs(atm_ref) + 1e-30)
    global max_err_1 = max(max_err_1, err)
    println("  kr=$kval: A^TM=$(round(A_TM[ik,1], sigdigits=6)), ref=$(round(atm_ref, sigdigits=6)), err=$(round(err, sigdigits=3))")
end
@assert max_err_1 < 0.05 "A^TM_0 doesn't match QuadGK"
println("  PASSED ✓")


# ─── Test 2: Mode m=1, both E_r and E_θ ──────────────────────
println("\n--- Test 2: Mode m=1, E_r and E_θ nonzero, QuadGK check ---")
# Use E_{1,r}(r) = r exp(-r²/2), E_{1,θ}(r) = i r exp(-r²/2)
m_pos_2 = [0, 1]
Em_r_2  = zeros(ComplexF64, Nr, 2)
Em_th_2 = zeros(ComplexF64, Nr, 2)
Em_r_2[:, 2]  .= collect(r) .* exp.(-collect(r).^2 ./ 2)
Em_th_2[:, 2] .= im .* collect(r) .* exp.(-collect(r).^2 ./ 2)

A_TE_2, A_TM_2, kr_2 = compute_TE_TM_coeffs(Em_r_2, Em_th_2, m_pos_2, collect(r), k_test)

# Verify A^TE_1 at k=1.0 via explicit QuadGK of the 4 Hankel integrals
kval = 1.0
ik = argmin(abs.(kr_2 .- kval))

fr_func(rr)  = rr * (rr * exp(-rr^2/2))     # r × E_{1,r} = r² exp(-r²/2)
fth_func(rr) = rr * (im * rr * exp(-rr^2/2)) # r × E_{1,θ} = i r² exp(-r²/2)

Hp1r, _ = quadgk(rr -> fr_func(rr) * besselj(2, kval*rr), 0, 30; rtol=1e-10)   # H_2
Hm1r, _ = quadgk(rr -> fr_func(rr) * besselj(0, kval*rr), 0, 30; rtol=1e-10)   # H_0
Hm1t, _ = quadgk(rr -> fth_func(rr) * besselj(0, kval*rr), 0, 30; rtol=1e-10)  # H_0[fth]
Hp1t, _ = quadgk(rr -> fth_func(rr) * besselj(2, kval*rr), 0, 30; rtol=1e-10)  # H_2[fth]

ate_ref = (im/2)*(Hp1r + Hm1r) + (1/2)*(Hm1t - Hp1t)
kz_k = sqrt(max(1 - (kval/k_test)^2, 0.0))
atm_ref = -(kz_k/2)*(Hm1r - Hp1r) - (im*kz_k/2)*(Hm1t + Hp1t)

err_te = abs(A_TE_2[ik, 2] - ate_ref) / (abs(ate_ref) + 1e-30)
err_tm = abs(A_TM_2[ik, 2] - atm_ref) / (abs(atm_ref) + 1e-30)
println("  A^TE_1 at k=1: FFTLog=$(round(A_TE_2[ik,2], sigdigits=6)), QuadGK=$(round(ate_ref, sigdigits=6)), err=$(round(err_te, sigdigits=3))")
println("  A^TM_1 at k=1: FFTLog=$(round(A_TM_2[ik,2], sigdigits=6)), QuadGK=$(round(atm_ref, sigdigits=6)), err=$(round(err_tm, sigdigits=3))")
@assert err_te < 0.05 "A^TE_1 doesn't match QuadGK"
@assert err_tm < 0.05 "A^TM_1 doesn't match QuadGK"
println("  PASSED ✓")


# ─── Test 3: Evanescent modes zeroed ──────────────────────────
println("\n--- Test 3: TM coefficient vanishes for evanescent kr > k ---")
# kz/k = 0 for kr > k, so A^TM should be zero there
evan_idx = findall(kr_out .> k_test * 1.1)  # well into evanescent
if !isempty(evan_idx)
    tm_evan = maximum(abs.(A_TM[evan_idx, 1]))
    println("  Max |A^TM| for kr > 1.1k: $(round(tm_evan, sigdigits=3))")
    @assert tm_evan < 1e-10 "TM should vanish in evanescent region"
end
println("  PASSED ✓")


# ─── Test 4: TE is independent of kz ─────────────────────────
println("\n--- Test 4: A^TE does NOT depend on kz (no kz/k prefactor) ---")
# Compute with two different k values; A^TE should be the same
# (the formula for A^TE has no kz dependence).
A_TE_k1, _, _ = compute_TE_TM_coeffs(Em_r_2, Em_th_2, m_pos_2, collect(r), 1.0)
A_TE_k2, _, _ = compute_TE_TM_coeffs(Em_r_2, Em_th_2, m_pos_2, collect(r), 100.0)

# Compare at interior kr points
mid = Nr÷4 : 3*Nr÷4
te_diff = maximum(abs.(A_TE_k1[mid, 2] .- A_TE_k2[mid, 2]))
te_norm = maximum(abs.(A_TE_k1[mid, 2])) + 1e-30
rel_diff = te_diff / te_norm
println("  Relative difference in A^TE for k=1 vs k=100: $(round(rel_diff, sigdigits=3))")
@assert rel_diff < 1e-10 "A^TE should not depend on k"
println("  PASSED ✓")


# ─── Test 5: TM changes with k (via kz/k factor) ─────────────
println("\n--- Test 5: A^TM DOES depend on k (through kz/k) ---")
_, A_TM_k1, _ = compute_TE_TM_coeffs(Em_r, Em_th, m_pos, collect(r), 2.0)
_, A_TM_k2, _ = compute_TE_TM_coeffs(Em_r, Em_th, m_pos, collect(r), 200.0)

# At kr=1.0: kz/k = √(1-1/4)=0.866 for k=2, kz/k = √(1-1/40000)≈1 for k=200
ik = argmin(abs.(kr_out .- 1.0))
ratio = abs(A_TM_k1[ik, 1]) / (abs(A_TM_k2[ik, 1]) + 1e-30)
expected_ratio = sqrt(1 - (1.0/2.0)^2) / sqrt(1 - (1.0/200.0)^2)
println("  |A^TM(k=2)| / |A^TM(k=200)| at kr=1: $(round(ratio, sigdigits=4))  (expected $(round(expected_ratio, sigdigits=4)))")
@assert abs(ratio - expected_ratio) / expected_ratio < 0.05 "kz/k scaling wrong"
println("  PASSED ✓")


# ─── Test 6: E_θ=0 gives same result as full path with zero θ ─
println("\n--- Test 6: E_θ=0 fast path matches full path ---")
Em_r_6  = reshape(ComplexF64[exp(-rv^2) * rv for rv in r], Nr, 1)
Em_th_6 = zeros(ComplexF64, Nr, 1)

A_TE_fast, A_TM_fast, _ = compute_TE_TM_coeffs(Em_r_6, Em_th_6, [0], collect(r), k_test)

# Now add tiny nonzero E_θ to force the full (4-call) path
Em_th_6b = Em_th_6 .+ 1e-30
A_TE_full, A_TM_full, _ = compute_TE_TM_coeffs(Em_r_6, Em_th_6b, [0], collect(r), k_test)

diff_te = maximum(abs.(A_TE_fast .- A_TE_full))
diff_tm = maximum(abs.(A_TM_fast .- A_TM_full))
println("  TE difference: $(round(diff_te, sigdigits=3))")
println("  TM difference: $(round(diff_tm, sigdigits=3))")
@assert diff_te < 1e-10 "E_θ=0 fast path differs from full path (TE)"
@assert diff_tm < 1e-10 "E_θ=0 fast path differs from full path (TM)"
println("  PASSED ✓")


# ─── Test 7: Multiple modes simultaneously ────────────────────
println("\n--- Test 7: Multiple modes m=0,1,2,3 simultaneously ---")
M_max_7 = 3
m_pos_7 = collect(0:M_max_7)
Em_r_7  = zeros(ComplexF64, Nr, M_max_7 + 1)
Em_th_7 = zeros(ComplexF64, Nr, M_max_7 + 1)
for (idx, m) in enumerate(m_pos_7)
    Em_r_7[:, idx] .= collect(r).^m .* exp.(-collect(r).^2)
end

A_TE_7, A_TM_7, kr_7 = compute_TE_TM_coeffs(Em_r_7, Em_th_7, m_pos_7, collect(r), k_test)

# Verify each mode independently
max_err_7 = 0.0
for (idx, m) in enumerate(m_pos_7)
    Em_r_single = reshape(Em_r_7[:, idx], Nr, 1)
    Em_th_single = zeros(ComplexF64, Nr, 1)
    A_TE_s, A_TM_s, _ = compute_TE_TM_coeffs(Em_r_single, Em_th_single, [m], collect(r), k_test)

    mid = Nr÷4 : 3*Nr÷4
    sig = findall(abs.(A_TE_s[mid, 1]) .> 1e-20)
    if !isempty(sig)
        err = maximum(abs.(A_TE_7[mid[sig], idx] .- A_TE_s[mid[sig], 1]) ./
                      (abs.(A_TE_s[mid[sig], 1]) .+ 1e-30))
        global max_err_7 = max(max_err_7, err)
    end
end
println("  Max deviation multi-mode vs single-mode: $(round(max_err_7, sigdigits=3))")
@assert max_err_7 < 1e-10 "Multi-mode computation differs from single-mode"
println("  PASSED ✓")


# ─── Test 8: Output kr grid matches expected ──────────────────
println("\n--- Test 8: Output kr grid ---")
@assert length(kr_out) == Nr "kr grid length mismatch"
@assert kr_out[1] ≈ 1.0 / r[end] "kr[1] should be 1/r_max"
kr_ratio = kr_out[2] / kr_out[1]
r_ratio  = r[2] / r[1]
@assert abs(kr_ratio - r_ratio) / r_ratio < 1e-10 "kr spacing should match r spacing"
println("  kr[1] = $(round(kr_out[1], sigdigits=4)) = 1/r_max ✓")
println("  kr spacing ratio = $(round(kr_ratio, sigdigits=8)) = r spacing ✓")
println("  PASSED ✓")


println("\n" * "="^60)
println("All compute_TE_TM_coeffs tests passed.")
println("="^60)
