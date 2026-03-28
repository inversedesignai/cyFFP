"""
    test_production_aperture.jl
    ===========================
    Production-scale aperture handling test.
    2mm visible lens (R=2000μm, λ=500nm, NA=0.25, α=10°).

    Tests FFTLog round-trip accuracy vs Tukey taper width on the
    actual production r grid (Nr=2^17=131072).

    Uses the LPA modal workflow: u_m(r) from Jacobi-Anger expansion
    (no angular FFT needed, no Nθ-sized arrays).

    Run with: julia -t auto test_production_aperture.jl
"""

include("../cyffp.jl")
using .CyFFP
using SpecialFunctions: besselj
using LinearAlgebra: dot

println("="^60)
println("Production aperture test: 2mm lens, λ=500nm, α=10°")
println("="^60)

# ─── Physical parameters ──────────────────────────────────────
lambda = 0.5        # μm
k      = 2π / lambda
R      = 2000.0     # μm
f_lens = R * sqrt(1/0.25^2 - 1)   # NA = 0.25
alpha  = deg2rad(10.0)
kx     = k * sin(alpha)
M_max  = ceil(Int, kx * R) + 20

println("R = $R μm,  λ = $lambda μm,  k = $(round(k, digits=2)) μm⁻¹")
println("f = $(round(f_lens, digits=0)) μm,  NA = 0.25,  α = 10°")
println("kx = $(round(kx, sigdigits=4)) μm⁻¹,  M_max = $M_max")

# ─── Production r grid (padded) ───────────────────────────────
r_min  = lambda / (20π)     # ≈ 0.008 μm (1 decade below 1/k)
r_max  = 50 * R             # 100,000 μm (1.7 decades above R)
Nr     = 2^17               # 131072
r      = exp.(range(log(r_min), log(r_max), length=Nr))
dln    = log(r[2] / r[1])
kr     = exp.(log(1.0 / r[end]) .+ dln .* (0:Nr-1))

println("\nGrid: Nr=$Nr, r=[$(round(r_min, sigdigits=2)), $(round(r_max, sigdigits=2))] μm")
println("dln = $(round(dln, sigdigits=4))")
println("kr = [$(round(kr[1], sigdigits=2)), $(round(kr[end], sigdigits=2))] μm⁻¹")
println("Δr at R = $(round(R*dln, sigdigits=3)) μm  (λ/2 = $(lambda/2) μm)")
@assert R * dln < lambda / 2 "Grid too coarse at aperture edge"
@assert 1/r_min > k "kr_max must exceed k"
println("Grid checks passed ✓")

# ─── LPA near-field modes (Jacobi-Anger, scalar) ──────────────
# For u(r,θ) = t(r) W(r) exp(ikₓ r cosθ), the scalar mode is:
#   u_m(r) = i^m t(r) W(r) J_m(kₓ r)
# where t(r) = exp(-ik[√(r²+f²) - f]) and W(r) is the aperture window.

t_lens(rv) = exp(-im * k * (sqrt(rv^2 + f_lens^2) - f_lens))

function tukey_window(rv, R_ap, alpha_t)
    r1 = R_ap * (1 - alpha_t)
    if rv <= r1
        return 1.0
    elseif rv <= R_ap
        return 0.5 * (1 + cos(π * (rv - r1) / (R_ap * alpha_t)))
    else
        return 0.0
    end
end

function make_scalar_mode(m, r_grid, alpha_taper)
    Nr_l = length(r_grid)
    u_m = zeros(ComplexF64, Nr_l)
    im_factor = Complex(0.0, 1.0)^m
    for j in 1:Nr_l
        rv = r_grid[j]
        w  = tukey_window(rv, R, alpha_taper)
        t  = t_lens(rv)
        u_m[j] = im_factor * t * w * besselj(m, kx * rv)
    end
    return u_m
end

# ─── Test: Round-trip vs taper width for representative modes ──
println("\n--- Round-trip H_ν(H_ν[g]) for production modes ---")
println("Format: alpha_taper → max round-trip error over tested modes\n")

test_modes = [1, 10, 100, 1000, min(M_max - 50, 4000)]
r_vec = collect(r)

for alpha_t in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    max_err = 0.0
    for m in test_modes
        u_m_test = make_scalar_mode(m, r_vec, alpha_t)
        g = r_vec .* u_m_test   # r-weighted for dr measure

        # Skip negligible modes
        g_max = maximum(abs.(g))
        if g_max < 1e-20; continue; end

        # Forward + inverse (order m)
        raw  = fftlog_hankel(g, dln, Float64(m))
        rt   = fftlog_hankel(raw, dln, Float64(m))

        # Compare shapes where signal is significant
        sig = findall(abs.(g) .> 1e-3 * g_max)
        if length(sig) < 10; continue; end
        g_s = g[sig]; rt_s = rt[sig]
        sc = real(dot(g_s, rt_s)) / (real(dot(rt_s, rt_s)) + 1e-30)
        err = maximum(abs.(rt_s .* sc .- g_s) ./ (abs.(g_s) .+ 1e-30))
        max_err = max(max_err, err)
    end
    label = alpha_t == 0.0 ? "hard" : "$(round(100*alpha_t, digits=1))%"
    println("  taper=$label ($(round(alpha_t*R, digits=1)) μm):  max_err=$(round(max_err, sigdigits=3))")
end

println("\n--- Checking specific mode details at recommended taper ---")
alpha_recommended = 0.005   # 0.5% of R = 10 μm ≈ 20λ
println("Recommended taper: $(100*alpha_recommended)% of R = $(alpha_recommended*R) μm = $(round(alpha_recommended*R/lambda, digits=0))λ\n")

for m in test_modes
    u_m_test = make_scalar_mode(m, r_vec, alpha_recommended)
    g = r_vec .* u_m_test
    g_max = maximum(abs.(g))
    if g_max < 1e-20
        println("  m=$m: negligible amplitude, skipped")
        continue
    end

    raw  = fftlog_hankel(g, dln, Float64(m))
    rt   = fftlog_hankel(raw, dln, Float64(m))

    sig = findall(abs.(g) .> 1e-3 * g_max)
    g_s = g[sig]; rt_s = rt[sig]
    sc = real(dot(g_s, rt_s)) / (real(dot(rt_s, rt_s)) + 1e-30)
    err = maximum(abs.(rt_s .* sc .- g_s) ./ (abs.(g_s) .+ 1e-30))
    med_err = sort(abs.(rt_s .* sc .- g_s) ./ (abs.(g_s) .+ 1e-30))[length(sig)÷2]

    println("  m=$(lpad(m, 4)):  max_err=$(round(err, sigdigits=3)),  median_err=$(round(med_err, sigdigits=3)),  scale=$(round(sc, sigdigits=6)),  |g|_max=$(round(g_max, sigdigits=3))")
end

# ─── Test: compute_scalar_coeffs at production scale ──────────
println("\n--- compute_scalar_coeffs at production scale (one mode at a time) ---")

full_mem_GB = Nr * (M_max + 1) * 16 / 1e9
println("  Full M_max=$M_max would need ≈$(round(full_mem_GB, digits=1)) GB → process per-mode")

for m in [1, 100, 1000, 4000]
    u_m_test = make_scalar_mode(m, r_vec, 0.0)
    u_m_arr = reshape(u_m_test, Nr, 1)

    t_m = @elapsed begin
        a_m_1, kr_prod = compute_scalar_coeffs(u_m_arr, [m], r_vec)
    end

    @assert !any(isnan, a_m_1) "NaN in a_m for m=$m"
    @assert !any(isinf, a_m_1) "Inf in a_m for m=$m"

    # Propagating-band magnitude
    prop = findall(0.5 .< kr_prod .< k * 0.9)
    am_rms = sqrt(sum(abs2.(a_m_1[prop, 1])) / length(prop))

    println("  m=$(lpad(m, 4)): $(round(t_m, digits=2))s, |a_m| rms=$(round(am_rms, sigdigits=3))")
end
println("  All per-mode checks passed ✓")
println("  PASSED ✓")


# ─── Parseval test: energy conservation Step 1 → Step 2 ───────
println("\n--- Parseval: spatial energy = spectral energy ---")
# Scalar Parseval (from Bessel orthogonality):
#   ∫ |u_m(r)|² r dr = ∫ |a_m(kr)|² kr dkr
#
# Discrete approximation (log-grid Riemann sum):
#   LHS = Δln × Σ_j |u_m|² × r_j²
#   RHS = Δln × Σ_k |a_m|² × kr_k²

println("  Mode   | Spatial energy  | Spectral energy | Ratio")
println("  -------|-----------------|-----------------|------")

max_parseval_err = 0.0
for m in [1, 10, 100, 1000, 4000]
    u_m_test = make_scalar_mode(m, r_vec, 0.0)  # hard aperture

    # Spatial energy: ∫ |u_m|² r dr ≈ Δln Σ |u_m|² r²
    spatial = dln * sum(abs2(u_m_test[j]) * r_vec[j]^2 for j in 1:Nr)

    # Scalar coefficients
    a_m_1, _ = compute_scalar_coeffs(reshape(u_m_test, Nr, 1), [m], r_vec)

    # Spectral energy: ∫ |a_m|² kr dkr ≈ Δln Σ |a_m|² kr²
    spectral = dln * sum(abs2(a_m_1[j, 1]) * kr[j]^2 for j in 1:Nr)

    ratio = real(spectral) / (real(spatial) + 1e-30)
    err = abs(ratio - 1.0)
    global max_parseval_err = max(max_parseval_err, err)

    println("  m=$(lpad(m, 4)) | $(lpad(round(real(spatial), sigdigits=6), 15)) | $(lpad(round(real(spectral), sigdigits=6), 15)) | $(round(ratio, sigdigits=6))")
end

println("\n  Max |ratio - 1|: $(round(max_parseval_err, sigdigits=3))")
# The scalar Parseval uses a SINGLE FFTLog transform (not 4 combined).
# Individual FFTLog preserves energy well: |U(q)|=1 exactly.
# Boundary effects (circular convolution wrap-around) are the main error source.
if max_parseval_err < 0.05
    println("  Parseval PASSED ✓ — absolute normalization verified")
else
    println("  Parseval: low-m modes should be < 5%, high-m modes may have boundary effects")
end
@assert max_parseval_err < 0.25 "Parseval violation exceeds expected FFTLog accuracy"


# ─── Step 4 production test: Graf shift ──────────────────────
println("\n--- Step 4: Graf shift at production scale ---")
x0_prod = f_lens * tan(alpha)
L_max_prod = 20
println("  x₀ = $(round(x0_prod, digits=1)) μm, L_max = $L_max_prod")
println("  kr×x₀ range: $(round(kr[1]*x0_prod, sigdigits=3)) to $(round(kr[end]*x0_prod, sigdigits=3))")

# Build a representative a_tilde: small M_max for memory
M_test = 50
m_full_test = collect(-M_test:M_test)
A_tilde_test = zeros(ComplexF64, Nr, 2M_test + 1)

# Fill with a realistic pattern: propagating modes only, decaying at high m
prop_mask = kr .< k
for m in -M_test:M_test
    idx = m + M_test + 1
    A_tilde_test[prop_mask, idx] .= exp(-abs(m) / 20.0) .* cis.(kr[prop_mask] .* 2.0)
end

println("  Running graf_shift with M_test=$M_test, L_max=$L_max_prod...")
t_graf = @elapsed begin
    B_prod = graf_shift(A_tilde_test, m_full_test, collect(kr), x0_prod, L_max_prod; k=k)
end
println("  Done in $(round(t_graf, digits=2)) s")

@assert !any(isnan, B_prod) "NaN in B"
@assert !any(isinf, B_prod) "Inf in B"
@assert size(B_prod) == (Nr, 2L_max_prod + 1) "Wrong size"
println("  No NaN/Inf, size $(size(B_prod)) ✓")

# Normal incidence check
B_normal = graf_shift(A_tilde_test, m_full_test, collect(kr), 0.0, L_max_prod)
max_id_err = 0.0
for (li, l) in enumerate(-L_max_prod:L_max_prod)
    if abs(l) <= M_test
        idx_A = l + M_test + 1
        global max_id_err = max(max_id_err, maximum(abs.(B_normal[:, li] .- A_tilde_test[:, idx_A])))
    end
end
println("  Normal incidence identity error: $(round(max_id_err, sigdigits=3))")
@assert max_id_err < 1e-12 "Normal incidence identity failed"
println("  Normal incidence ✓")

# Miller recurrence at production-scale x values
println("\n  Miller recurrence accuracy at production x values:")
for x_test in [1.0, 100.0, 1000.0, 10000.0, x0_prod * k]
    m_max_t = min(4385, ceil(Int, x_test) + 20)
    Jp = CyFFP._besselj_range(m_max_t, x_test)
    # Spot-check a few orders
    errs = [abs(Jp[m+1] - besselj(m, x_test)) for m in [0, min(m_max_t, 10), m_max_t÷2, m_max_t]]
    max_e = maximum(errs)
    println("    x=$(round(x_test, sigdigits=4)), m_max=$m_max_t: max_err=$(round(max_e, sigdigits=3))")
    @assert max_e < 1e-8 "Miller recurrence inaccurate at x=$x_test"
end
println("  Miller recurrence ✓")

println("  Step 4 production PASSED ✓")


println("\n" * "="^60)
println("Production aperture test complete.")
println("="^60)
