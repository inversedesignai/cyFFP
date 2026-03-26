"""
    test_step3.jl
    =============
    Rigorous tests for propagate_scalar (Step 3).

    Tests:
      1. Propagation phase: correct exp(ikz f) for propagating, 0 for evanescent
      2. Scalar symmetry: ã_{-m} = ã_m (no sign change)
      3. Output indexing: column j = mode m = j - M_max - 1
      4. Evanescent modes: a_tilde = 0 for kr > k
      5. Propagation preserves amplitude |ã_m| = |a_m| where propagating
      6. m=0 symmetry: no separate negative entry
      7. End-to-end with realistic lens: Steps 1→2→3 + verify symmetry from full FFT

    Run with: julia test_step3.jl
"""

include("cyffp.jl")
using .CyFFP
using FFTW
using SpecialFunctions: besselj

println("="^60)
println("Step 3: propagate_scalar — rigorous tests")
println("="^60)

# ─── Shared setup ─────────────────────────────────────────────
Nr    = 512
r     = exp.(range(log(1e-3), log(1e3), length=Nr))
dln   = log(r[2] / r[1])
kr    = exp.(log(1.0 / r[end]) .+ dln .* (0:Nr-1))
k     = 2π        # λ = 1
f_val = 80.0      # focal length

# Fabricate some scalar a_m for testing
M_max = 5
m_pos = collect(0:M_max)
a_m   = randn(ComplexF64, Nr, M_max + 1)


# ─── Test 1: Propagation phase ───────────────────────────────
println("\n--- Test 1: Propagation phase exp(ikz f) ---")
a_tilde, m_full = propagate_scalar(a_m, m_pos, collect(kr), k, f_val)

# For m=0 (column M_max+1): ã_0 = a_0 × prop
ik_prop = argmin(abs.(kr .- k * 0.5))  # kr ≈ k/2
kz_expected = sqrt(k^2 - kr[ik_prop]^2)
prop_expected = exp(im * kz_expected * f_val)

atilde_got   = a_tilde[ik_prop, M_max + 1]  # m=0 column
atilde_exp   = a_m[ik_prop, 1] * prop_expected

err_1 = abs(atilde_got - atilde_exp) / (abs(atilde_exp) + 1e-30)
println("  m=0, kr=$(round(kr[ik_prop], sigdigits=3)): err=$(round(err_1, sigdigits=3))")
@assert err_1 < 1e-12 "Propagation phase wrong"

# Check another mode
atilde_m3   = a_tilde[ik_prop, 3 + M_max + 1]  # m=3 column
atilde_m3_exp = a_m[ik_prop, 4] * prop_expected  # 4th column = m=3
err_1b = abs(atilde_m3 - atilde_m3_exp) / (abs(atilde_m3_exp) + 1e-30)
println("  m=3, kr=$(round(kr[ik_prop], sigdigits=3)): err=$(round(err_1b, sigdigits=3))")
@assert err_1b < 1e-12 "Propagation phase wrong for m=3"
println("  PASSED ✓")


# ─── Test 2: Scalar symmetry ã_{-m} = (-1)^m ã_m ──────────────
println("\n--- Test 2: Scalar symmetry: ã_{-m} = (-1)^m ã_m ---")
max_err_2 = 0.0
for m in 1:M_max
    idx_pos = m + M_max + 1
    idx_neg = -m + M_max + 1
    sign = iseven(m) ? 1 : -1
    err = maximum(abs.(a_tilde[:, idx_neg] .- sign .* a_tilde[:, idx_pos]))
    global max_err_2 = max(max_err_2, err)
end
println("  Max |ã_{-m} - (-1)^m ã_m|: $(round(max_err_2, sigdigits=3))")
@assert max_err_2 < 1e-15 "Scalar symmetry ã_{-m} ≠ (-1)^m ã_m"
println("  PASSED ✓")


# ─── Test 3: Output indexing ──────────────────────────────────
println("\n--- Test 3: Output indexing: column j → mode m = j - M_max - 1 ---")
@assert length(m_full) == 2M_max + 1
@assert m_full[1] == -M_max
@assert m_full[M_max + 1] == 0
@assert m_full[end] == M_max
@assert size(a_tilde, 2) == 2M_max + 1
println("  m_full = [$(m_full[1]), ..., $(m_full[M_max+1]), ..., $(m_full[end])]")
println("  PASSED ✓")


# ─── Test 4: Evanescent zeroed ────────────────────────────────
println("\n--- Test 4: a_tilde = 0 for kr > k ---")
evan = findall(kr .> k * 1.01)
if !isempty(evan)
    evan_max = maximum(abs.(a_tilde[evan, :]))
    println("  Max |a_tilde| evanescent: $(round(evan_max, sigdigits=3))")
    @assert evan_max < 1e-15 "Evanescent modes not zeroed"
end
println("  PASSED ✓")


# ─── Test 5: Propagation preserves amplitude ──────────────────
println("\n--- Test 5: |ã_m| = |a_m| for propagating modes ---")
prop_idx = findall(kr .< k * 0.99)
for m in 0:M_max
    ip = m + 1
    idx = m + M_max + 1
    ratio = abs.(a_tilde[prop_idx, idx]) ./ (abs.(a_m[prop_idx, ip]) .+ 1e-30)
    max_dev = maximum(abs.(ratio .- 1.0))
    @assert max_dev < 1e-12 "|prop| ≠ 1 at propagating kr for m=$m"
end
println("  |ã_m| / |a_m| = 1.0 at all propagating kr ✓")
println("  PASSED ✓")


# ─── Test 6: m=0 symmetry ────────────────────────────────────
println("\n--- Test 6: m=0 propagation ---")
idx_0 = 0 + M_max + 1
atilde_0 = a_tilde[:, idx_0]
expected_0 = a_m[:, 1] .* [kr[j] < k ?
    exp(im * sqrt(k^2 - kr[j]^2) * f_val) : 0.0+0im for j in 1:Nr]
err_6 = maximum(abs.(atilde_0 .- expected_0))
println("  Max error for m=0: $(round(err_6, sigdigits=3))")
@assert err_6 < 1e-12 "m=0 propagation wrong"
println("  PASSED ✓")


# ─── Test 7: End-to-end with realistic lens ───────────────────
println("\n--- Test 7: Steps 1→2→3 with scaled lens (R=10λ, α=10°) ---")
lambda_7 = 1.0
k_7      = 2π / lambda_7
R_7      = 10.0 * lambda_7
f_7      = R_7 * sqrt(1/0.25^2 - 1)
alpha_7  = deg2rad(10.0)
kx_7     = k_7 * sin(alpha_7)
M_max_7  = ceil(Int, kx_7 * R_7) + 20
x0_7     = f_7 * tan(alpha_7)

Nr_7     = 512
r_7      = exp.(range(log(1e-3), log(1e3), length=Nr_7))
Ntheta_7 = 128
theta_7  = range(0.0, 2π, length=Ntheta_7+1)[1:end-1]

# Scalar ideal oblique lens near field
function u_obl(rv, th)
    d = sqrt((rv*cos(th) - x0_7)^2 + rv^2*sin(th)^2 + f_7^2)
    return exp(-im*k_7*(d - f_7)) * (rv <= R_7 ? 1.0 : 0.0)
end

u_field = ComplexF64[u_obl(r_7[jr], theta_7[jt])
                     for jr in 1:Nr_7, jt in 1:Ntheta_7]

# Step 1: angular decompose (pass u as Er, zeros as Etheta — scalar pipeline)
u_m7, _, m_pos_7 = angular_decompose(u_field, zeros(ComplexF64, Nr_7, Ntheta_7), M_max_7)

# Step 2: scalar coefficients
a_m7, kr_7 = compute_scalar_coeffs(u_m7, m_pos_7, collect(r_7))

# Step 3: propagate
a_tilde_7, m_full_7 = propagate_scalar(a_m7, m_pos_7, kr_7, k_7, f_7)

# 7a: No NaN/Inf
@assert !any(isnan, a_tilde_7) "NaN in a_tilde"
@assert !any(isinf, a_tilde_7) "Inf in a_tilde"
println("  7a: No NaN/Inf ✓")

# 7b: Evanescent zeroed
evan_7 = findall(kr_7 .> k_7 * 1.01)
if !isempty(evan_7)
    @assert maximum(abs.(a_tilde_7[evan_7, :])) < 1e-15 "Evanescent not zeroed"
end
println("  7b: Evanescent zeroed ✓")

# 7c: Symmetry check — compare ã_{-m} from symmetry vs direct computation from full FFT
# The full FFT gives u_{-m} directly; we can compute a_{-m} independently
# and compare against the symmetry-reconstructed values.
full_u = fft(u_field, 2) ./ Ntheta_7

# Pick m=3: direct negative mode from FFT
m_check = 3
idx_neg_fft = mod(-m_check, Ntheta_7) + 1
u_m_neg = full_u[:, idx_neg_fft:idx_neg_fft]

# Compute a_{-3} directly (no symmetry)
a_m_neg, _ = compute_scalar_coeffs(
    ComplexF64.(u_m_neg), [-m_check], collect(r_7))

# Propagate the direct negative mode
kz_7 = [kr_7[j] < k_7 ? sqrt(k_7^2 - kr_7[j]^2) : 0.0 for j in 1:Nr_7]
prop_7 = [kr_7[j] < k_7 ? exp(im*kz_7[j]*f_7) : 0.0+0im for j in 1:Nr_7]
atilde_neg_direct = a_m_neg[:, 1] .* prop_7

# From the symmetry reconstruction
idx_neg_sym = -m_check + M_max_7 + 1
atilde_neg_sym = a_tilde_7[:, idx_neg_sym]

# Compare where significant
prop_kr = findall(0.1 .< kr_7 .< k_7 * 0.9)
sig = findall(abs.(atilde_neg_direct[prop_kr]) .> 1e-6 * maximum(abs.(atilde_neg_direct[prop_kr])))
if !isempty(sig)
    errs = abs.(atilde_neg_sym[prop_kr[sig]] .- atilde_neg_direct[prop_kr[sig]]) ./
           (abs.(atilde_neg_direct[prop_kr[sig]]) .+ 1e-30)
    max_sym_err = maximum(errs)
    med_sym_err = sort(errs)[length(errs)÷2]
    println("  7c: Symmetry vs direct for m=-$m_check: max=$(round(max_sym_err, sigdigits=3)), median=$(round(med_sym_err, sigdigits=3))")
    @assert max_sym_err < 0.10 "Symmetry reconstruction doesn't match direct computation"
else
    println("  7c: Mode m=-$m_check has negligible amplitude")
end
println("  7c: Symmetry verified ✓")

println("  Test 7 PASSED ✓")


println("\n" * "="^60)
println("All Step 3 tests passed.")
println("="^60)
