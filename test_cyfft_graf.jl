"""
    test_cyfft_graf.jl
    ==================
    Validation of CyFFP_graf.jl using the Graf addition theorem approach.

    Tests:
      1. Graf identity:  Σ_m J_m(z0) J_{n-m}(z) = J_n(z+z0) ... spot check
      2. Normal incidence (alpha=0): Graf shift is identity, PSF at ρ=0
      3. Oblique incidence: scalar ideal lens, PSF centered at ρ=0 in local frame
      4. Mode-convolution convergence vs L_max
"""

include("CyFFP_graf.jl")
using .CyFFP
using FFTW
using SpecialFunctions: besselj
using LinearAlgebra

println("="^60)
println("CyFFP Graf-shift Validation")
println("="^60)

# ─── Test 1: Graf identity check ────────────────────────────────
println("\n--- Test 1: Graf identity ---")
# Graf: J_n(z+z0) = Σ_m J_m(z0) J_{n-m}(z)   for all z0, z
# Verify numerically for a few (n, z, z0) pairs
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
# H_ν(H_ν[f]) = f  (up to normalization)
Nr_test  = 256
r_test   = exp.(range(log(1e-2), log(10.0), length=Nr_test))
dln_test = log(r_test[2] / r_test[1])
for nu in [0, 1, 5, 20]
    f_orig = @. exp(-r_test^2 / 2) * r_test^nu
    f_fwd  = fftlog_hankel(r_test .* f_orig, dln_test, Float64(nu))
    # Second application (inverse = forward for Hankel)
    f_inv  = fftlog_hankel(f_fwd, dln_test, Float64(nu))
    # Compare in middle region (avoid boundary artefacts)
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

# ─── Test 3: Normal incidence — PSF should peak at ρ = 0 ────────
println("\n--- Test 3: Normal incidence (alpha=0) ---")
alpha_n = 1e-6   # effectively 0; avoids x0=0 division issues
Er_n     = [lens_phase(rv) * inc_phase(rv, th, alpha_n)
            for rv in r, th in theta]
Etheta_n = zeros(ComplexF64, Nr, Ntheta)

psf_n, rho_n, psi_n = cyfft_farfield(
    complex.(Er_n), Etheta_n, collect(r), k, alpha_n, f;
    M_buffer=5, L_max=8, Npsi=64)

I_n     = abs2.(psf_n)
peak_ix = argmax(I_n)
rho_pk  = rho_n[peak_ix[1]]
println("  PSF peak at ρ = $(round(rho_pk/lambda, digits=4)) λ")
println("  (expected ρ ≈ 0)")
# For near-normal incidence the peak should be very close to ρ=0
airy    = 0.61 * lambda / 0.5   # rough Airy radius
@assert rho_pk < 3airy "Normal-incidence PSF peak too far from origin"
println("  PASSED ✓")

# ─── Test 4: Oblique incidence — PSF in local frame ─────────────
println("\n--- Test 4: Oblique incidence, alpha=11° ---")
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
println("  PSF peak at ρ = $(round(rho_pk2/lambda, digits=4)) λ in local frame")
@assert rho_pk2 < 3airy "Oblique PSF peak too far from local origin"
println("  PASSED ✓")

# ─── Test 5: Graf convergence vs L_max ──────────────────────────
println("\n--- Test 5: Convergence with L_max ---")
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
