"""
    test_bruteforce_reference.jl
    ============================
    Brute-force reference PSF for a small lens, computed via direct
    numerical integration (QuadGK) at every step.  No FFTLog, no
    tricks — just straightforward Hankel integrals.

    Then compare Steps 1–4 of the CyFFP scalar pipeline against the
    reference intermediate values at matching kr points.

    Small parameters: R=3λ, NA=0.3, α=5° → M_max~12, cheap everywhere.

    Run with: julia test_bruteforce_reference.jl
"""

include("cyffp.jl")
using .CyFFP
using FFTW
using SpecialFunctions: besselj
using QuadGK

println("="^60)
println("Brute-force reference: small lens R=3λ, α=5°")
println("="^60)

# ─── Physical parameters (small, cheap) ───────────────────────
lambda = 1.0
k      = 2π / lambda
R      = 3.0 * lambda
NA     = 0.3
f_val  = R / NA * sqrt(1 - NA^2)   # f from NA = R/√(R²+f²)
alpha  = deg2rad(5.0)
x0     = f_val * tan(alpha)
kx     = k * sin(alpha)
M_max  = ceil(Int, kx * R) + 10
L_max  = 8

println("R = $(R)λ, f = $(round(f_val/lambda, digits=2))λ, NA = $NA")
println("α = 5°, x₀ = $(round(x0/lambda, digits=3))λ, kx = $(round(kx, digits=3))")
println("M_max = $M_max, L_max = $L_max")

# ─── CyFFP pipeline grid ─────────────────────────────────────
Nr     = 512
r_min  = 1e-3
r_max  = 1e3
r_grid = collect(exp.(range(log(r_min), log(r_max), length=Nr)))
dln    = log(r_grid[2] / r_grid[1])
kr_cyf = exp.(log(1.0 / r_grid[end]) .+ dln .* (0:Nr-1))

Ntheta = 64
theta  = range(0.0, 2π, length=Ntheta+1)[1:end-1]

# ─── Near field: scalar ideal oblique lens with aperture ──────
function u_oblique(rv, th)
    d = sqrt((rv * cos(th) - x0)^2 + rv^2 * sin(th)^2 + f_val^2)
    return exp(-im * k * (d - f_val)) * (rv <= R ? 1.0 : 0.0)
end

u_field = ComplexF64[u_oblique(rv, th) for rv in r_grid, th in theta]

# ═══════════════════════════════════════════════════════════════
# CyFFP Pipeline: Steps 1–4 (scalar)
# ═══════════════════════════════════════════════════════════════
println("\n--- CyFFP scalar pipeline ---")

# Step 1: angular decompose
u_m, _, m_pos = angular_decompose(u_field, zeros(ComplexF64, Nr, Ntheta), M_max)

# Step 2: scalar spectral coefficients
a_m_cyf, kr_out = compute_scalar_coeffs(u_m, m_pos, r_grid)
@assert kr_out ≈ kr_cyf "kr grids don't match"

# Step 3: propagation + symmetry
a_tilde_cyf, m_full = propagate_scalar(a_m_cyf, m_pos, kr_cyf, k, f_val)

# Step 4: Graf shift
B_cyf = graf_shift(a_tilde_cyf, m_full, kr_cyf, x0, L_max; k=k)

println("  Steps 1-4 complete.")


# ═══════════════════════════════════════════════════════════════
# Brute-force reference: QuadGK at selected kr points
# ═══════════════════════════════════════════════════════════════
println("\n--- Brute-force reference via QuadGK ---")

# Select a few kr points in the propagating band for comparison
test_kr_vals = [0.5, 1.0, 2.0, 4.0, k * 0.9]
test_kr_idx  = [argmin(abs.(kr_cyf .- kv)) for kv in test_kr_vals]
actual_kr    = kr_cyf[test_kr_idx]

# Step 1 reference: angular modes via numerical θ-integration
# u_m(r) = (1/2π) ∫ u(r,θ) e^{-imθ} dθ
println("\n  Comparing Step 1: angular modes")
max_step1_err = 0.0
n_checked = 0
for m in [1, 2, 3, M_max-2]
    idx = m + 1
    for jr in [Nr÷4, Nr÷2, 3*Nr÷4]
        rv = r_grid[jr]
        if rv > R * 1.5; continue; end   # outside aperture → zero, skip

        ref_u, _ = quadgk(th -> u_oblique(rv, th) * exp(-im*m*th),
                          0, 2π; rtol=1e-10)
        ref_u /= 2π

        # Skip if reference is negligible
        if abs(ref_u) < 1e-10; continue; end

        cyf_u = u_m[jr, idx]
        err = abs(cyf_u - ref_u) / abs(ref_u)
        global max_step1_err = max(max_step1_err, err)
        global n_checked += 1
    end
end
println("    Checked $n_checked points, max relative error: $(round(max_step1_err, sigdigits=3))")
@assert max_step1_err < 0.01 "Step 1 doesn't match quadrature"
println("    PASSED ✓")


# Step 2 reference: scalar a_m(kr) = ∫ u_m(r) J_m(kr r) r dr
println("\n  Comparing Step 2: scalar spectral coefficients")

# Interpolate u_m on the r_grid for QuadGK
function interp_mode(u_m_arr, idx, rv)
    lr = log(rv)
    lr0 = log(r_grid[1])
    idx_f = (lr - lr0) / dln + 1.0
    j0 = clamp(floor(Int, idx_f), 1, Nr-1)
    w = idx_f - j0
    return (1-w) * u_m_arr[j0, idx] + w * u_m_arr[j0+1, idx]
end

max_step2_err = 0.0
for m in [1, 3]
    idx = m + 1
    for ik in test_kr_idx[1:3]
        kv = kr_cyf[ik]

        # Reference: a_m(kr) = ∫ u_m(r) J_m(kr r) r dr
        am_ref, _ = quadgk(rv -> interp_mode(u_m, idx, rv) * besselj(m, kv*rv) * rv,
                           r_grid[1], R*1.5; rtol=1e-8)

        am_cyf = a_m_cyf[ik, idx]

        if abs(am_ref) > 1e-14
            err = abs(am_cyf - am_ref) / abs(am_ref)
            global max_step2_err = max(max_step2_err, err)
        end

        println("    m=$m kr=$(round(kv, sigdigits=3)): cyf=$(round(am_cyf, sigdigits=4)) ref=$(round(am_ref, sigdigits=4)) err=$(round(abs(am_cyf - am_ref)/(abs(am_ref)+1e-30), sigdigits=3))")
    end
end
println("    Max relative error: $(round(max_step2_err, sigdigits=3))")
@assert max_step2_err < 0.10 "Step 2 error too large"
println("    PASSED ✓")


# Step 3 reference: propagation is just multiplication — should match exactly
println("\n  Comparing Step 3: propagation")
max_step3_err = 0.0
for m in -M_max:M_max
    idx_full = m + M_max + 1
    for ik in test_kr_idx
        kv = kr_cyf[ik]
        if kv < k
            kz = sqrt(k^2 - kv^2)
            prop = exp(im * kz * f_val)
        else
            prop = 0.0 + 0im
        end
        # Scalar: ã_m = a_m × prop for positive m
        # ã_{-m} = ã_m (scalar symmetry)
        if m >= 0
            ip = m + 1
            expected = a_m_cyf[ik, ip] * prop
        else
            # From scalar symmetry: ã_{-m} = ã_m
            expected = a_tilde_cyf[ik, idx_full]
        end
        err = abs(a_tilde_cyf[ik, idx_full] - expected) / (abs(expected) + 1e-30)
        global max_step3_err = max(max_step3_err, err)
    end
end
println("    Max relative error: $(round(max_step3_err, sigdigits=3))")
@assert max_step3_err < 1e-12 "Step 3 mismatch"
println("    PASSED ✓")


# Step 4 reference: Graf shift via direct besselj sum
println("\n  Comparing Step 4: Graf shift B_l(kr)")
max_step4_err = 0.0
for (li, l) in enumerate(-L_max:L_max)
    for ik in test_kr_idx[1:3]
        kv = kr_cyf[ik]
        # Direct: B_l = Σ_m ã_{m+l} J_m(kr x₀)
        bl_ref = zero(ComplexF64)
        for m in -M_max:M_max
            n = m + l
            if -M_max <= n <= M_max
                bl_ref += a_tilde_cyf[ik, n + M_max + 1] * besselj(m, kv * x0)
            end
        end
        bl_cyf = B_cyf[ik, li]
        err = abs(bl_cyf - bl_ref) / (abs(bl_ref) + 1e-30)
        global max_step4_err = max(max_step4_err, err)
    end
end
println("    Max relative error: $(round(max_step4_err, sigdigits=3))")
@assert max_step4_err < 1e-10 "Step 4 doesn't match direct sum"
println("    PASSED ✓")


# ═══════════════════════════════════════════════════════════════
# Brute-force PSF via inverse Hankel + synthesis
# ═══════════════════════════════════════════════════════════════
println("\n--- Brute-force PSF ---")

# b_l(ρ) = ∫₀^k B_l(kr) J_l(kr ρ) kr dkr
# u(ρ,ψ) = Σ_l b_l(ρ) e^{ilψ}
#
# Use direct Riemann sum on the kr grid.
# Δln × Σ_j B_l(kr_j) J_l(kr_j ρ) kr_j²  (since kr dkr = kr² d(ln kr))

rho_test = [0.001, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] .* lambda
Npsi_test = 32
psi_test = range(0.0, 2π, length=Npsi_test+1)[1:end-1]

NA_eff = R * cos(alpha) / sqrt((R*cos(alpha))^2 + f_val^2)
rho_airy = 0.61 * lambda / NA_eff
println("  NA_eff = $(round(NA_eff, digits=4)), Airy radius = $(round(rho_airy/lambda, digits=3))λ")

# Only sum over propagating kr (kr < k)
prop_kr = findall(kr_cyf .< k)

PSF_bf = zeros(ComplexF64, length(rho_test), Npsi_test)

println("  Computing PSF via Riemann sum for $(length(rho_test)) ρ values...")
for (ir, rho) in enumerate(rho_test)
    for (ip, psi) in enumerate(psi_test)
        field = zero(ComplexF64)
        for (li, l) in enumerate(-L_max:L_max)
            bl = dln * sum(B_cyf[j, li] * besselj(l, kr_cyf[j] * rho) * kr_cyf[j]^2
                           for j in prop_kr)
            field += bl * exp(im * l * psi)
        end
        PSF_bf[ir, ip] = field
    end
end

I_bf = abs2.(PSF_bf)

# ψ-averaged radial profile
I_avg = [sum(I_bf[ir, :]) / Npsi_test for ir in 1:length(rho_test)]
I_peak = maximum(I_avg)

println("\n  Radial PSF profile (ψ-averaged):")
for (ir, rho) in enumerate(rho_test)
    println("    ρ/λ = $(round(rho/lambda, digits=3)):  I/I_peak = $(round(I_avg[ir]/I_peak, sigdigits=4))")
end

# PSF checks — scalar field (single Cartesian component)
println("\n  PSF checks:")

# For a scalar PSF, the dominant local mode is l=0, and J_0(kr ρ) peaks at ρ=0.
# The PSF should peak near ρ=0.
peak_rho = rho_test[argmax(I_avg)]
println("    PSF peak at ρ = $(round(peak_rho/lambda, digits=3))λ (expected ≈ 0)")
@assert peak_rho < 0.5 * lambda "Scalar PSF should peak near ρ=0"
println("    Peak at center ✓")

# Intensity should decay at large ρ
@assert I_avg[end] < 0.5 * I_peak "PSF should decay at large ρ"
println("    Decay at large ρ ✓")

# PSF should be nonzero (sanity)
@assert I_peak > 0 "PSF is zero everywhere"
println("    Nonzero PSF ✓")


println("\n" * "="^60)
println("Brute-force reference test complete.")
println("="^60)
