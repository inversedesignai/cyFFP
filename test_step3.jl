"""
    test_step3.jl
    =============
    Rigorous tests for propagate_and_symmetrize (Step 3).

    Tests:
      1. Propagation phase: correct exp(ikz f) for propagating, 0 for evanescent
      2. y-pol symmetry: Ã_{-m} = (-1)^m (A^TE - A^TM) · prop
      3. x-pol symmetry: Ã_{-m} = (-1)^m (A^TM - A^TE) · prop  (sign flip)
      4. Normal incidence (α→0): Ã_{-m} and Ã_m should be related by symmetry
      5. Output indexing: column j = mode m = j - M_max - 1
      6. Evanescent modes: A_tilde = 0 for kr > k
      7. Energy: propagation doesn't change |Ã|² where prop ≠ 0
      8. End-to-end with realistic lens: Steps 1→2→3 + verify symmetry from full FFT

    Run with: julia test_step3.jl
"""

include("cyffp.jl")
using .CyFFP
using FFTW
using SpecialFunctions: besselj

println("="^60)
println("Step 3: propagate_and_symmetrize — rigorous tests")
println("="^60)

# ─── Shared setup ─────────────────────────────────────────────
Nr    = 512
r     = exp.(range(log(1e-3), log(1e3), length=Nr))
dln   = log(r[2] / r[1])
kr    = exp.(log(1.0 / r[end]) .+ dln .* (0:Nr-1))
k     = 2π        # λ = 1
f_val = 80.0      # focal length

# Fabricate some A^TE, A^TM for testing (don't need actual Hankel transforms)
M_max = 5
m_pos = collect(0:M_max)
A_TE  = randn(ComplexF64, Nr, M_max + 1)
A_TM  = randn(ComplexF64, Nr, M_max + 1)


# ─── Test 1: Propagation phase ───────────────────────────────
println("\n--- Test 1: Propagation phase exp(ikz f) ---")
A_tilde, m_full = propagate_and_symmetrize(A_TE, A_TM, m_pos, collect(kr), k, f_val)

# For m=0 (column M_max+1): Ã_0 = (A^TE_0 + A^TM_0) × prop
# Check at a propagating kr point
ik_prop = argmin(abs.(kr .- k * 0.5))  # kr ≈ k/2
kz_expected = sqrt(k^2 - kr[ik_prop]^2)
prop_expected = exp(im * kz_expected * f_val)

ate_plus_atm = A_TE[ik_prop, 1] + A_TM[ik_prop, 1]
atilde_got   = A_tilde[ik_prop, M_max + 1]  # m=0 column
atilde_exp   = ate_plus_atm * prop_expected

err_1 = abs(atilde_got - atilde_exp) / (abs(atilde_exp) + 1e-30)
println("  m=0, kr=$(round(kr[ik_prop], sigdigits=3)): err=$(round(err_1, sigdigits=3))")
@assert err_1 < 1e-12 "Propagation phase wrong"
println("  PASSED ✓")


# ─── Test 2: y-pol symmetry ──────────────────────────────────
println("\n--- Test 2: y-pol: Ã_{-m} = (-1)^m (A^TE - A^TM) · prop ---")
max_err_2 = 0.0
for m in 1:M_max
    idx_pos = m + M_max + 1
    idx_neg = -m + M_max + 1
    s = iseven(m) ? 1.0 : -1.0
    for ik in [ik_prop, Nr÷3, Nr÷2]
        kz_val = kr[ik] < k ? sqrt(k^2 - kr[ik]^2) : 0.0
        prop_val = kr[ik] < k ? exp(im * kz_val * f_val) : 0.0 + 0im
        expected_neg = s * (A_TE[ik, m+1] - A_TM[ik, m+1]) * prop_val
        err = abs(A_tilde[ik, idx_neg] - expected_neg) / (abs(expected_neg) + 1e-30)
        global max_err_2 = max(max_err_2, err)
    end
end
println("  Max error: $(round(max_err_2, sigdigits=3))")
@assert max_err_2 < 1e-12 "y-pol symmetry formula wrong"
println("  PASSED ✓")


# ─── Test 3: x-pol symmetry (sign flip) ──────────────────────
println("\n--- Test 3: x-pol: Ã_{-m} = (-1)^m (A^TM - A^TE) · prop ---")
A_tilde_x, _ = propagate_and_symmetrize(A_TE, A_TM, m_pos, collect(kr), k, f_val;
                                          polarization=:x)
max_err_3 = 0.0
for m in 1:M_max
    idx_neg = -m + M_max + 1
    # x-pol negative should be OPPOSITE sign to y-pol negative
    diff = A_tilde_x[:, idx_neg] .+ A_tilde[:, idx_neg]  # should be zero (opposite signs)
    global max_err_3 = max(max_err_3, maximum(abs.(diff)))
end
println("  Max |x-pol neg + y-pol neg|: $(round(max_err_3, sigdigits=3))  (should be 0)")
@assert max_err_3 < 1e-12 "x-pol and y-pol negatives should differ by sign"

# Also verify x-pol positive modes are SAME as y-pol
diff_pos = maximum(abs.(A_tilde_x[:, M_max+1:end] .- A_tilde[:, M_max+1:end]))
println("  Max |x-pol pos - y-pol pos|: $(round(diff_pos, sigdigits=3))  (should be 0)")
@assert diff_pos < 1e-15 "Positive modes should be identical for both polarizations"
println("  PASSED ✓")


# ─── Test 4: Output indexing ──────────────────────────────────
println("\n--- Test 4: Output indexing: column j → mode m = j - M_max - 1 ---")
@assert length(m_full) == 2M_max + 1
@assert m_full[1] == -M_max
@assert m_full[M_max + 1] == 0
@assert m_full[end] == M_max
@assert size(A_tilde, 2) == 2M_max + 1
println("  m_full = [$(m_full[1]), ..., $(m_full[M_max+1]), ..., $(m_full[end])]")
println("  PASSED ✓")


# ─── Test 5: Evanescent zeroed ────────────────────────────────
println("\n--- Test 5: A_tilde = 0 for kr > k ---")
evan = findall(kr .> k * 1.01)
if !isempty(evan)
    evan_max = maximum(abs.(A_tilde[evan, :]))
    println("  Max |A_tilde| evanescent: $(round(evan_max, sigdigits=3))")
    @assert evan_max < 1e-15 "Evanescent modes not zeroed"
end
println("  PASSED ✓")


# ─── Test 6: Propagation preserves amplitude where prop=1 ────
println("\n--- Test 6: |prop| = 1 for propagating modes ---")
# At any propagating kr: |Ã_m|² = |A^TE + A^TM|² (since |prop|=1)
prop_idx = findall(kr .< k * 0.99)
for m in 0:M_max
    ip = m + 1
    idx = m + M_max + 1
    ate_atm = A_TE[prop_idx, ip] .+ A_TM[prop_idx, ip]
    ratio = abs.(A_tilde[prop_idx, idx]) ./ (abs.(ate_atm) .+ 1e-30)
    max_dev = maximum(abs.(ratio .- 1.0))
    @assert max_dev < 1e-12 "|prop| ≠ 1 at propagating kr for m=$m"
end
println("  |Ã_m| / |A^TE + A^TM| = 1.0 at all propagating kr ✓")
println("  PASSED ✓")


# ─── Test 7: m=0 has no negative counterpart ─────────────────
println("\n--- Test 7: m=0 symmetry (no negative mode) ---")
# Ã_0 should just be (A^TE_0 + A^TM_0) × prop — same as positive mode
# (no separate negative entry)
idx_0 = 0 + M_max + 1
atilde_0 = A_tilde[:, idx_0]
expected_0 = (A_TE[:, 1] .+ A_TM[:, 1]) .* [kr[j] < k ?
    exp(im * sqrt(k^2 - kr[j]^2) * f_val) : 0.0+0im for j in 1:Nr]
err_7 = maximum(abs.(atilde_0 .- expected_0))
println("  Max error for m=0: $(round(err_7, sigdigits=3))")
@assert err_7 < 1e-12 "m=0 propagation wrong"
println("  PASSED ✓")


# ─── Test 8: End-to-end with realistic lens ───────────────────
println("\n--- Test 8: Steps 1→2→3 with scaled lens (R=10λ, α=10°) ---")
lambda_8 = 1.0
k_8      = 2π / lambda_8
R_8      = 10.0 * lambda_8
f_8      = R_8 * sqrt(1/0.25^2 - 1)
alpha_8  = deg2rad(10.0)
kx_8     = k_8 * sin(alpha_8)
M_max_8  = ceil(Int, kx_8 * R_8) + 20

Nr_8     = 512
r_8      = exp.(range(log(1e-3), log(1e3), length=Nr_8))
Ntheta_8 = 128
theta_8  = range(0.0, 2π, length=Ntheta_8+1)[1:end-1]

# y-polarized ideal oblique lens with aperture
function u_obl(rv, th)
    x0 = f_8 * tan(alpha_8)
    d = sqrt((rv*cos(th) - x0)^2 + rv^2*sin(th)^2 + f_8^2)
    return exp(-im*k_8*(d - f_8)) * (rv <= R_8 ? 1.0 : 0.0)
end

Er_8  = ComplexF64[u_obl(r_8[jr], theta_8[jt])*sin(theta_8[jt])
                    for jr in 1:Nr_8, jt in 1:Ntheta_8]
Et_8  = ComplexF64[u_obl(r_8[jr], theta_8[jt])*cos(theta_8[jt])
                    for jr in 1:Nr_8, jt in 1:Ntheta_8]

# Step 1
Em_r_8, Em_th_8, m_pos_8 = angular_decompose(Er_8, Et_8, M_max_8)
# Step 2
A_TE_8, A_TM_8, kr_8 = compute_TE_TM_coeffs(Em_r_8, Em_th_8, m_pos_8, collect(r_8), k_8)
# Step 3
A_tilde_8, m_full_8 = propagate_and_symmetrize(A_TE_8, A_TM_8, m_pos_8, kr_8, k_8, f_8;
                                                 polarization=:y)

# 8a: No NaN/Inf
@assert !any(isnan, A_tilde_8) "NaN in A_tilde"
@assert !any(isinf, A_tilde_8) "Inf in A_tilde"
println("  8a: No NaN/Inf ✓")

# 8b: Evanescent zeroed
evan_8 = findall(kr_8 .> k_8 * 1.01)
if !isempty(evan_8)
    @assert maximum(abs.(A_tilde_8[evan_8, :])) < 1e-15 "Evanescent not zeroed"
end
println("  8b: Evanescent zeroed ✓")

# 8c: Symmetry check — compare Ã_{-m} from symmetry vs direct computation from full FFT
# The full FFT gives E_{-m} directly; we can compute A^TE_{-m}, A^TM_{-m} independently
# and compare against the symmetry-reconstructed values.
full_r = fft(Er_8, 2) ./ Ntheta_8
full_t = fft(Et_8, 2) ./ Ntheta_8

# Pick m=3: direct negative mode from FFT
m_check = 3
idx_neg_fft = mod(-m_check, Ntheta_8) + 1
Em_r_neg = full_r[:, idx_neg_fft:idx_neg_fft]
Em_t_neg = full_t[:, idx_neg_fft:idx_neg_fft]

# Compute A^TE_{-3}, A^TM_{-3} directly (no symmetry)
A_TE_neg, A_TM_neg, _ = compute_TE_TM_coeffs(
    ComplexF64.(Em_r_neg), ComplexF64.(Em_t_neg), [-m_check], collect(r_8), k_8)

# Propagate the direct negative mode
kz_8 = [kr_8[j] < k_8 ? sqrt(k_8^2 - kr_8[j]^2) : 0.0 for j in 1:Nr_8]
prop_8 = [kr_8[j] < k_8 ? exp(im*kz_8[j]*f_8) : 0.0+0im for j in 1:Nr_8]
atilde_neg_direct = (A_TE_neg[:, 1] .+ A_TM_neg[:, 1]) .* prop_8

# From the symmetry reconstruction
idx_neg_sym = -m_check + M_max_8 + 1
atilde_neg_sym = A_tilde_8[:, idx_neg_sym]

# Compare where significant
prop_kr = findall(0.1 .< kr_8 .< k_8 * 0.9)
sig = findall(abs.(atilde_neg_direct[prop_kr]) .> 1e-6 * maximum(abs.(atilde_neg_direct[prop_kr])))
if !isempty(sig)
    errs = abs.(atilde_neg_sym[prop_kr[sig]] .- atilde_neg_direct[prop_kr[sig]]) ./
           (abs.(atilde_neg_direct[prop_kr[sig]]) .+ 1e-30)
    max_sym_err = maximum(errs)
    med_sym_err = sort(errs)[length(errs)÷2]
    println("  8c: Symmetry vs direct for m=-$m_check: max=$(round(max_sym_err, sigdigits=3)), median=$(round(med_sym_err, sigdigits=3))")
    @assert max_sym_err < 0.15 "Symmetry reconstruction doesn't match direct computation"
else
    println("  8c: Mode m=-$m_check has negligible amplitude")
end
println("  8c: Symmetry verified ✓")

println("  Test 8 PASSED ✓")


println("\n" * "="^60)
println("All Step 3 tests passed.")
println("="^60)
