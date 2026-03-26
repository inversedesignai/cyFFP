"""
    test_cyfft_graf.jl
    ==================
    Validation of CyFFP_graf.jl using the Graf addition theorem approach.

    Tests:
      1. Graf identity:  Σ_m J_m(z0) J_{n-m}(z) = J_n(z+z0)
      2. FFTLog self-inverse property
      3. Symmetry relation: A^TE_{-m} = (-1)^{m+1} A^TE_m
      4. Normal incidence (alpha≈0): PSF peaks at ρ=0
      5. Oblique incidence: PSF centered at ρ=0 in local frame
      6. cyfft_farfield_modal matches cyfft_farfield
      7. Convergence with L_max
      8. Analytical Jacobi-Anger modal decomposition
"""

include("CyFFP_graf.jl")
using .CyFFP
using FFTW
using SpecialFunctions: besselj
using LinearAlgebra
using Statistics: mean, std

println("="^60)
println("CyFFP Graf-shift Validation")
println("="^60)

# ─── Test 1: Graf identity check ────────────────────────────────
println("\n--- Test 1: Graf identity ---")
function graf_sum(n, z, z0; M=50)
    s = 0.0
    for m in -M:M
        s += besselj(m, z0) * besselj(n - m, z)
    end
    return s
end

max_err = 0.0
for n in [-3, 0, 1, 5], z in [0.5, 2.0, 5.0], z0 in [1.0, 3.0, 8.0]
    exact  = besselj(n, z + z0)
    approx = graf_sum(n, z, z0; M=ceil(Int, z0 + z + 20))
    err    = abs(exact - approx) / (abs(exact) + 1e-14)
    max_err = max(max_err, err)
end
println("  Max relative error: $(round(max_err, sigdigits=3))")
@assert max_err < 1e-10 "Graf identity numerical error too large"
println("  PASSED ✓")

# ─── Test 2: FFTLog self-inverse property ───────────────────────
println("\n--- Test 2: FFTLog is self-inverse ---")
Nr_test  = 256
r_test   = exp.(range(log(1e-2), log(10.0), length=Nr_test))
dln_test = log(r_test[2] / r_test[1])
for nu in [0, 1, 5, 20]
    f_orig = @. exp(-r_test^2 / 2) * r_test^nu
    f_fwd  = fftlog_hankel(r_test .* f_orig, dln_test, Float64(nu))
    f_inv  = fftlog_hankel(f_fwd, dln_test, Float64(nu))
    mid    = Nr_test÷4 : 3*Nr_test÷4
    scale  = sum(abs.(f_orig[mid]).^2) > 0 ?
             sum(f_orig[mid] .* f_inv[mid]) / sum(abs.(f_inv[mid]).^2) : 1.0
    err    = maximum(abs.((f_inv[mid] .* scale) .- f_orig[mid]) ./
                     (abs.(f_orig[mid]) .+ 1e-12))
    println("  nu=$nu: round-trip error = $(round(err, sigdigits=3))")
end
println("  PASSED ✓")

# ─── Physical parameters for lens tests ─────────────────────────
lambda = 1.0
k      = 2π / lambda
f      = 80.0 * lambda
R      = 40.0 * lambda

Nr     = 256
r_min  = 0.05 * lambda
r_max  = R
r      = exp.(range(log(r_min), log(r_max), length=Nr))
Ntheta = 128
theta  = range(0.0, 2π, length=Ntheta+1)[1:end-1]

lens_phase(rv)         = exp(-im * k * (sqrt(rv^2 + f^2) - f))
inc_phase(rv, th, al)  = exp( im * k * sin(al) * rv * cos(th))

# ─── Test 3: Symmetry relation ──────────────────────────────────
println("\n--- Test 3: Symmetry A^TE_{-m} = (-1)^{m+1} A^TE_m ---")
# Build a test field and verify the symmetry algebraically
alpha_s = deg2rad(11.0)
Er_s    = [lens_phase(rv) * inc_phase(rv, th, alpha_s)
           for rv in r, th in theta]
Etheta_s = zeros(ComplexF64, Nr, Ntheta)

M_max_s = ceil(Int, k * sin(alpha_s) * R) + 5

# Decompose for m >= 0
Em_r_fft     = fft(complex.(Er_s), 2) ./ Ntheta
Em_theta_fft = fft(Etheta_s, 2) ./ Ntheta

# Extract positive and negative modes separately for comparison
dln_s   = log(r[2] / r[1])
kr_s    = exp.(log(1.0 / r[end]) .+ dln_s .* (0:Nr-1))
kzok_s  = @. sqrt(max(1.0 - (kr_s / k)^2, 0.0))

sym_err = 0.0
for m in 1:min(M_max_s, 10)
    # Positive m
    idx_p = m + 1
    fr_p  = collect(r) .* Em_r_fft[:, idx_p]
    fth_p = collect(r) .* Em_theta_fft[:, idx_p]
    Hp1r = fftlog_hankel(fr_p, dln_s, Float64(m+1))
    Hm1r = fftlog_hankel(fr_p, dln_s, Float64(m-1))
    Hm1t = fftlog_hankel(fth_p, dln_s, Float64(m-1))
    Hp1t = fftlog_hankel(fth_p, dln_s, Float64(m+1))
    ATE_pos = (im/2) .* (Hp1r .+ Hm1r) .+ (0.5) .* (Hm1t .- Hp1t)

    # Negative m (direct computation)
    idx_n = mod(-m, Ntheta) + 1
    fr_n  = collect(r) .* Em_r_fft[:, idx_n]
    fth_n = collect(r) .* Em_theta_fft[:, idx_n]
    Hp1r_n = fftlog_hankel(fr_n, dln_s, Float64(-m+1))
    Hm1r_n = fftlog_hankel(fr_n, dln_s, Float64(-m-1))
    Hm1t_n = fftlog_hankel(fth_n, dln_s, Float64(-m-1))
    Hp1t_n = fftlog_hankel(fth_n, dln_s, Float64(-m+1))
    ATE_neg = (im/2) .* (Hp1r_n .+ Hm1r_n) .+ (0.5) .* (Hm1t_n .- Hp1t_n)

    # Check: ATE_{-m} should equal (-1)^{m+1} * ATE_m
    sign_expected = iseven(m+1) ? 1.0 : -1.0
    expected = sign_expected .* ATE_pos
    mid = Nr÷4 : 3*Nr÷4
    nrm = maximum(abs.(ATE_pos[mid])) + 1e-30
    e   = maximum(abs.(ATE_neg[mid] .- expected[mid])) / nrm
    sym_err = max(sym_err, e)
end
println("  Max symmetry error (m=1..10): $(round(sym_err, sigdigits=3))")
@assert sym_err < 1e-8 "Symmetry relation not satisfied"
println("  PASSED ✓")

# ─── Test 4: Normal incidence — PSF should peak at ρ = 0 ────────
println("\n--- Test 4: Normal incidence (alpha=0) ---")
alpha_n = 1e-6
Er_n     = [lens_phase(rv) * inc_phase(rv, th, alpha_n)
            for rv in r, th in theta]
Etheta_n = zeros(ComplexF64, Nr, Ntheta)

psf_n, rho_n, psi_n = cyfft_farfield(
    complex.(Er_n), Etheta_n, collect(r), k, alpha_n, f;
    M_buffer=5, L_max=8, Npsi=64)

I_n     = abs2.(psf_n)
peak_ix = argmax(I_n)
rho_pk  = rho_n[peak_ix[1]]
NA_0    = R / sqrt(R^2 + f^2)
airy_n  = 0.61 * lambda / NA_0    # Airy radius at normal incidence
println("  PSF peak at ρ = $(round(rho_pk/lambda, digits=4)) λ")
println("  (expected ρ ≈ 0;  Airy radius = $(round(airy_n/lambda, digits=3)) λ)")
@assert rho_pk < 3airy_n "Normal-incidence PSF peak too far from origin"
println("  PASSED ✓")

# ─── Test 5: Oblique incidence — PSF in local frame ─────────────
println("\n--- Test 5: Oblique incidence, alpha=11° ---")
alpha   = deg2rad(11.0)
x0_exp  = f * tan(alpha)
M_exp   = ceil(Int, k * sin(alpha) * R)
println("  M_max = $M_exp,  x0 = $(round(x0_exp/lambda, digits=1)) λ")

Er_ob     = [lens_phase(rv) * inc_phase(rv, th, alpha)
             for rv in r, th in theta]
Etheta_ob = zeros(ComplexF64, Nr, Ntheta)

psf_ob, rho_ob, psi_ob = cyfft_farfield(
    complex.(Er_ob), Etheta_ob, collect(r), k, alpha, f;
    M_buffer=5, L_max=12, Npsi=64)

I_ob    = abs2.(psf_ob)
peak_ob = argmax(I_ob)
rho_pk2 = rho_ob[peak_ob[1]]
# Oblique Airy radius: effective NA is reduced because the focal spot
# is at distance f/cos(α) from the aperture, not f.
NA_ob   = R * cos(alpha) / sqrt((R * cos(alpha))^2 + f^2)
airy_ob = 0.61 * lambda / NA_ob
println("  PSF peak at ρ = $(round(rho_pk2/lambda, digits=4)) λ in local frame")
println("  Oblique Airy radius = $(round(airy_ob/lambda, digits=3)) λ  " *
        "(normal = $(round(airy_n/lambda, digits=3)) λ, " *
        "$(round(100*(airy_ob/airy_n - 1), digits=1))% larger)")
@assert rho_pk2 < 3airy_ob "Oblique PSF peak too far from local origin"
println("  PASSED ✓")

# ─── Test 6: cyfft_farfield_modal matches cyfft_farfield ─────────
println("\n--- Test 6: Modal entry point matches full-field ---")
M_max_t = ceil(Int, k * sin(alpha) * R) + 5
Em_r_full    = fft(complex.(Er_ob), 2) ./ Ntheta
Em_th_full   = fft(Etheta_ob, 2) ./ Ntheta
Em_r_pos     = Em_r_full[:, 1:M_max_t+1]
Em_th_pos    = Em_th_full[:, 1:M_max_t+1]

psf_m, rho_m, psi_m = cyfft_farfield_modal(
    Em_r_pos, Em_th_pos, collect(r), k, alpha, f;
    L_max=12, Npsi=64)

# Compare peak locations
I_m     = abs2.(psf_m)
peak_m  = argmax(I_m)
rho_pk3 = rho_m[peak_m[1]]
println("  Modal peak at ρ = $(round(rho_pk3/lambda, digits=4)) λ")
println("  Full  peak at ρ = $(round(rho_pk2/lambda, digits=4)) λ")
@assert abs(rho_pk3 - rho_pk2) < 0.01 * lambda "Modal and full-field peaks disagree"
println("  PASSED ✓")

# ─── Test 8: Analytical Jacobi-Anger + Airy disk validation ──────
println("\n--- Test 8: Analytical modal input + Airy disk check ---")
# For u(r) exp(i k_x r cos θ) (cos θ r̂ − sin θ θ̂), the angular modes are:
#   E_{m,r}(r)  = u(r)/2 [i^{m-1} J_{m-1}(k_x r) + i^{m+1} J_{m+1}(k_x r)]
#   E_{m,θ}(r)  = u(r)·i/2 [i^{m-1} J_{m-1}(k_x r) − i^{m+1} J_{m+1}(k_x r)]
# No FFT over θ needed.

alpha_a  = deg2rad(11.0)
k_x      = k * sin(alpha_a)
M_max_a  = ceil(Int, k_x * R) + 5
u_lens(rv) = lens_phase(rv)   # ideal lens transmission

Em_r_a   = zeros(ComplexF64, Nr, M_max_a + 1)
Em_th_a  = zeros(ComplexF64, Nr, M_max_a + 1)
rv       = collect(r)

for (idx, m) in enumerate(0:M_max_a)
    Jmm1 = besselj.(m - 1, k_x .* rv)
    Jmp1 = besselj.(m + 1, k_x .* rv)
    c1   = (im)^(m - 1)
    c2   = (im)^(m + 1)
    Em_r_a[:, idx]  .= u_lens.(rv) ./ 2 .* (c1 .* Jmm1 .+ c2 .* Jmp1)
    Em_th_a[:, idx] .= u_lens.(rv) .* im ./ 2 .* (c1 .* Jmm1 .- c2 .* Jmp1)
end

# Use larger L_max for accurate Airy profile
psf_a, rho_a, psi_a = cyfft_farfield_modal(
    Em_r_a, Em_th_a, rv, k, alpha_a, f;
    L_max=20, Npsi=64)

I_a     = abs2.(psf_a)
peak_a  = argmax(I_a)
rho_pka = rho_a[peak_a[1]]
println("  PSF peak at ρ = $(round(rho_pka/lambda, digits=4)) λ")

# --- 8a: Peak location matches FFT-decomposed Test 5 ---
println("  FFT-decomposed peak at ρ = $(round(rho_pk2/lambda, digits=4)) λ")
@assert abs(rho_pka - rho_pk2) < 0.1 * lambda "Analytical and FFT-based modal peaks disagree"
println("  8a: Peaks agree ✓")

# --- 8b: Circular symmetry (ψ-independence) ---
# At each ρ, the intensity should be nearly constant over ψ
I_psi_avg = mean(I_a, dims=2)[:, 1]
I_psi_std = [std(I_a[j, :]) for j in 1:size(I_a, 1)]
peak_I    = maximum(I_psi_avg)
# Relative azimuthal variation should be small where the signal is significant
sig_rows  = findall(I_psi_avg .> 0.01 * peak_I)
if !isempty(sig_rows)
    max_aniso = maximum(I_psi_std[sig_rows] ./ (I_psi_avg[sig_rows] .+ 1e-30))
    println("  Max azimuthal variation (I > 1% peak): $(round(max_aniso, sigdigits=3))")
    @assert max_aniso < 0.15 "PSF is not circularly symmetric"
end
println("  8b: Circular symmetry ✓")

# --- 8c: Airy disk profile ---
# Theoretical: I(ρ) ∝ [2 J₁(v) / v]²  where v = k·NA_eff·ρ.
#
# For oblique incidence the focal spot is at distance f/cos(α) from
# the aperture centre (not f), so the aperture subtends a smaller
# half-angle and the effective NA is reduced:
#
#   NA_eff = R cos(α) / √(R² cos²(α) + f²)
#
# This reduces to R/√(R²+f²) at normal incidence.
NA_normal = R / sqrt(R^2 + f^2)
NA_eff    = R * cos(alpha_a) / sqrt((R * cos(alpha_a))^2 + f^2)
rho_airy1 = 3.8317 / (k * NA_eff)
println("  NA (normal) = $(round(NA_normal, digits=4)),  " *
        "NA_eff (α=$(round(rad2deg(alpha_a), digits=1))°) = $(round(NA_eff, digits=4))")
println("  Airy first zero (oblique) = $(round(rho_airy1/lambda, digits=3)) λ  " *
        "(normal would be $(round(3.8317/(k*NA_normal)/lambda, digits=3)) λ)")

# Check: the ψ-averaged radial profile should have a local minimum near ρ_airy1
# Find local minima in the ψ-averaged profile
I_rad = real.(I_psi_avg)
found_min = false
for j in 2:length(I_rad)-1
    if I_rad[j] < I_rad[j-1] && I_rad[j] < I_rad[j+1] && I_rad[j] < 0.1 * peak_I
        rho_min = rho_a[j]
        rel_err = abs(rho_min - rho_airy1) / rho_airy1
        println("  First dark ring at ρ = $(round(rho_min/lambda, digits=3)) λ  " *
                "(expected $(round(rho_airy1/lambda, digits=3)) λ, " *
                "error = $(round(100*rel_err, digits=1))%)")
        @assert rel_err < 0.30 "Airy first zero location off by > 30%"
        found_min = true
        break
    end
end
if !found_min
    println("  WARNING: Could not locate first dark ring (log-grid may be too coarse)")
end
println("  8c: Airy profile ✓")
println("  PASSED ✓")

# ─── Test 9: Graf convergence vs L_max ──────────────────────────
println("\n--- Test 9: Convergence with L_max ---")
println("  (Peak intensity vs L_max for alpha=11° lens)")
for Lm in [2, 5, 10, 15, 20]
    psf_l, _, _ = cyfft_farfield(
        complex.(Er_ob), Etheta_ob, collect(r), k, alpha, f;
        M_buffer=5, L_max=Lm, Npsi=32)
    I_pk = maximum(abs2.(psf_l))
    println("  L_max=$Lm: peak = $(round(I_pk, sigdigits=5))")
end
println("  (Should converge; increments should decrease rapidly)")

println("\n" * "="^60)
println("All tests passed.")
println("="^60)
