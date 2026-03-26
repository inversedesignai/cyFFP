"""
    test_scalar_airy.jl
    ====================
    End-to-end validation: scalar pipeline (Steps 1-4) + brute-force
    inverse Hankel (Steps 5-6) should produce an Airy disk.

    Uses a small lens (R=3λ) where brute-force QuadGK is cheap.

    For a linearly polarized field E = u(r,θ) ŷ, the scalar u satisfies
    the scalar wave equation exactly.  The scalar spectral coefficient is
    a_m(kr) = H_m[r u_m](kr), ONE Hankel transform at order m.

    Run with: julia test_scalar_airy.jl
"""

include("cyffp.jl")
using .CyFFP
using FFTW
using SpecialFunctions: besselj
using QuadGK
using Statistics: mean, std

println("="^60)
println("Scalar Airy disk validation: R=3λ, α=5°")
println("="^60)

# ─── Physical parameters ──────────────────────────────────────
lambda = 1.0
k      = 2π / lambda
R      = 3.0 * lambda
NA     = 0.3
f_val  = R / NA * sqrt(1 - NA^2)
alpha  = deg2rad(5.0)
x0     = f_val * tan(alpha)
kx     = k * sin(alpha)
M_max  = ceil(Int, kx * R) + 10
L_max  = 8

println("R = $(R)λ, f = $(round(f_val/lambda, digits=2))λ, NA = $NA")
println("α = 5°, x₀ = $(round(x0/lambda, digits=3))λ")
println("M_max = $M_max, L_max = $L_max")

NA_eff = R * cos(alpha) / sqrt((R*cos(alpha))^2 + f_val^2)
rho_airy = 0.61 * lambda / NA_eff
println("NA_eff = $(round(NA_eff, digits=4)), Airy first zero = $(round(rho_airy/lambda, digits=3))λ")

# ─── Grid ─────────────────────────────────────────────────────
Nr     = 512
r_grid = collect(exp.(range(log(1e-3), log(1e3), length=Nr)))
dln    = log(r_grid[2] / r_grid[1])
Ntheta = 64
theta  = range(0.0, 2π, length=Ntheta+1)[1:end-1]

# ─── Scalar near field: u(r,θ) = ideal oblique lens × aperture ─
function u_lens(rv, th)
    d = sqrt((rv * cos(th) - x0)^2 + rv^2 * sin(th)^2 + f_val^2)
    return exp(-im * k * (d - f_val)) * (rv <= R ? 1.0 : 0.0)
end

u_field = ComplexF64[u_lens(rv, th) for rv in r_grid, th in theta]

# ─── CyFFP scalar pipeline: Steps 1-4 ────────────────────────
println("\n--- CyFFP scalar pipeline ---")

# Step 1: angular decompose (extract scalar modes u_m)
u_m, _, m_pos = angular_decompose(u_field, zeros(ComplexF64, Nr, Ntheta), M_max)
# Note: u_field is passed as the "Er" component; we only use the first output.
# For the scalar case, we decompose u directly.

# Step 2: scalar spectral coefficients
a_m, kr = compute_scalar_coeffs(u_m, m_pos, r_grid)

# Step 3: propagation + symmetry
a_tilde, m_full = propagate_scalar(a_m, m_pos, kr, k, f_val)

# Step 4: Graf shift
B = graf_shift(a_tilde, m_full, kr, x0, L_max; k=k)

println("  Steps 1-4 complete.")
println("  |a_m| at m=0,1,5: $(round(maximum(abs.(a_m[:,1])), sigdigits=3)), $(round(maximum(abs.(a_m[:,2])), sigdigits=3)), $(round(maximum(abs.(a_m[:,6])), sigdigits=3))")


# ─── Step 2 cross-check: a_m(kr) vs QuadGK ───────────────────
println("\n--- Step 2: scalar a_m(kr) vs QuadGK ---")
max_err_s2 = 0.0
for m in [1, 3]
    idx = m + 1
    for kval in [0.5, 1.0, 3.0]
        ik = argmin(abs.(kr .- kval))
        # QuadGK: a_m(kr) = ∫ u_m(r) J_m(kr r) r dr
        function u_m_interp(rv)
            lr = log(rv)
            lr0 = log(r_grid[1])
            idx_f = (lr - lr0) / dln + 1.0
            j0 = clamp(floor(Int, idx_f), 1, Nr-1)
            w = idx_f - j0
            return (1-w) * u_m[j0, idx] + w * u_m[j0+1, idx]
        end
        ref, _ = quadgk(rv -> u_m_interp(rv) * besselj(m, kval*rv) * rv,
                        r_grid[1], R*1.5; rtol=1e-8)
        err = abs(a_m[ik, idx] - ref) / (abs(ref) + 1e-30)
        global max_err_s2 = max(max_err_s2, err)
    end
end
println("  Max error: $(round(max_err_s2, sigdigits=3))")
@assert max_err_s2 < 0.10 "Scalar a_m doesn't match QuadGK"
println("  PASSED ✓")


# ─── Brute-force PSF: inverse Hankel + synthesis via Riemann ──
println("\n--- Brute-force PSF (Riemann sum) ---")

prop_kr = findall(kr .< k)
rho_test = collect(range(0.001, 4.0, length=40)) .* lambda

PSF = zeros(ComplexF64, length(rho_test))  # ψ=0 slice (should be max for scalar)

for (ir, rho) in enumerate(rho_test)
    field = zero(ComplexF64)
    for (li, l) in enumerate(-L_max:L_max)
        bl = dln * sum(B[j, li] * besselj(l, kr[j] * rho) * kr[j]^2 for j in prop_kr)
        field += bl * exp(im * l * 0.0)  # ψ = 0
    end
    PSF[ir] = field
end

I_psf = abs2.(PSF)
I_peak = maximum(I_psf)
I_norm = I_psf ./ I_peak

println("\n  Radial PSF profile:")
for ir in 1:5:length(rho_test)
    println("    ρ/λ = $(round(rho_test[ir]/lambda, digits=3)):  I/I_peak = $(round(I_norm[ir], sigdigits=4))")
end

# ─── Airy disk checks ────────────────────────────────────────
println("\n--- Airy disk checks ---")

# 1. Peak at ρ ≈ 0
peak_idx = argmax(I_psf)
peak_rho = rho_test[peak_idx]
println("  Peak at ρ = $(round(peak_rho/lambda, digits=3))λ (expected ≈ 0)")
@assert peak_rho < 0.5 * lambda "PSF should peak near ρ=0"
println("  Peak location ✓")

# 2. Monotonic decrease from center to first minimum
for ir in 2:min(peak_idx + 5, length(rho_test))
    if rho_test[ir] < rho_airy * 0.9  # before the dark ring
        @assert I_norm[ir] <= I_norm[ir-1] + 0.01 "Not monotonically decreasing"
    end
end
println("  Monotonic decrease ✓")

# 3. First minimum near Airy zero
# Find first local minimum
first_min_rho = 0.0
first_min_val = 1.0
for ir in 2:length(I_norm)-1
    if I_norm[ir] < I_norm[ir-1] && I_norm[ir] < I_norm[ir+1]
        global first_min_rho = rho_test[ir]
        global first_min_val = I_norm[ir]
        break
    end
end
if first_min_rho > 0
    airy_err = abs(first_min_rho - rho_airy) / rho_airy
    println("  First minimum at ρ = $(round(first_min_rho/lambda, digits=3))λ (expected $(round(rho_airy/lambda, digits=3))λ, error $(round(100*airy_err, digits=1))%)")
    println("  Intensity at minimum: $(round(first_min_val, sigdigits=3)) of peak")
    @assert airy_err < 0.3 "First zero too far from Airy prediction"
    @assert first_min_val < 0.1 "First minimum not deep enough"
    println("  Airy dark ring ✓")
else
    println("  WARNING: Could not find first minimum")
end

# 4. Azimuthal structure: at α=5°, the aperture r≤R is NOT circular
# in the local frame (ρ,ψ) because the focal spot is offset by x₀.
# This creates l=±1 modes → ψ-dependent PSF.  CORRECT PHYSICS.
# For circular symmetry validation, test at α=0 below.
field_psi0 = zero(ComplexF64)
field_psi90 = zero(ComplexF64)
rho_check = 1.0 * lambda
for (li, l) in enumerate(-L_max:L_max)
    bl = dln * sum(B[j, li] * besselj(l, kr[j] * rho_check) * kr[j]^2 for j in prop_kr)
    global field_psi0 += bl * exp(im * l * 0.0)
    global field_psi90 += bl * exp(im * l * π/2)
end
I_psi0  = abs2(field_psi0)
I_psi90 = abs2(field_psi90)
println("  Azimuthal ratio I(ψ=0)/I(ψ=π/2) = $(round(I_psi0/(I_psi90+1e-30), sigdigits=3))")
println("  (≠1 because aperture is offset by x₀=$(round(x0/lambda, digits=2))λ — correct physics)")

# 5. Compare with theoretical Airy: I(ρ) = [2J₁(v)/v]² where v = k NA ρ
println("\n  Comparison with theoretical Airy [2J₁(v)/v]²:")
for rho_frac in [0.3, 0.5, 0.7, 1.0]
    rho_v = rho_frac * rho_airy
    ir = argmin(abs.(rho_test .- rho_v))
    v = k * NA_eff * rho_test[ir]
    airy_theory = v > 0.01 ? (2 * besselj(1, v) / v)^2 : 1.0
    println("    ρ/ρ_Airy = $(round(rho_frac, digits=1)): computed=$(round(I_norm[ir], sigdigits=4)), theory=$(round(airy_theory, sigdigits=4))")
end


# ─── Normal incidence (α=0): perfect circular Airy ────────────
println("\n--- Normal incidence validation (α=0) ---")
# At α=0, x₀=0, the Graf shift is the identity, and the PSF
# should be a perfect circularly symmetric Airy disk.
alpha_0  = 1e-6   # effectively zero
x0_0     = f_val * tan(alpha_0)
M_max_0  = 5

u_normal = ComplexF64[let d=sqrt(rv^2+f_val^2)
    exp(-im*k*(d-f_val))*(rv<=R ? 1.0 : 0.0) end for rv in r_grid, th in theta]

u_m0, _, m_pos0 = angular_decompose(u_normal, zeros(ComplexF64, Nr, Ntheta), M_max_0)
a_m0, kr0 = compute_scalar_coeffs(u_m0, m_pos0, r_grid)
a_tilde0, m_full0 = propagate_scalar(a_m0, m_pos0, kr0, k, f_val)
B0 = graf_shift(a_tilde0, m_full0, kr0, x0_0, L_max; k=k)

# Check l=0 dominance
B0_energies = [sum(abs2.(B0[:, li])) for li in 1:2L_max+1]
l0_frac = B0_energies[L_max+1] / sum(B0_energies)
println("  l=0 energy fraction: $(round(100*l0_frac, digits=1))%")
@assert l0_frac > 0.99 "l=0 should dominate at normal incidence"

# Circular symmetry at ρ = 1λ
prop_kr0 = findall(kr0 .< k)
f0 = zero(ComplexF64); f90 = zero(ComplexF64)
for (li, l) in enumerate(-L_max:L_max)
    bl = dln * sum(B0[j, li] * besselj(l, kr0[j] * lambda) * kr0[j]^2 for j in prop_kr0)
    global f0 += bl * exp(im * l * 0.0)
    global f90 += bl * exp(im * l * π/2)
end
aniso_0 = abs(abs2(f0) - abs2(f90)) / (abs2(f0) + 1e-30)
println("  I(ψ=0)/I(ψ=π/2) = $(round(abs2(f0)/(abs2(f90)+1e-30), sigdigits=6))")
@assert aniso_0 < 0.001 "Normal incidence PSF should be circularly symmetric"
println("  Circular symmetry at α=0 ✓")


println("\n" * "="^60)
println("Scalar Airy disk validation complete.")
println("="^60)
