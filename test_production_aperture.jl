"""
    test_production_aperture.jl
    ===========================
    Production-scale aperture handling test.
    2mm visible lens (R=2000μm, λ=500nm, NA=0.25, α=10°).

    Tests FFTLog round-trip accuracy vs Tukey taper width on the
    actual production r grid (Nr=2^17=131072).

    Uses the LPA modal workflow: E_m(r) from Jacobi-Anger expansion
    (no angular FFT needed, no Nθ-sized arrays).

    Run with: julia -t auto test_production_aperture.jl
"""

include("cyffp.jl")
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
# Padding: r_max = 100R (2 decades above R) for small-kr boundary
#          r_min = λ/(20π) (1 decade below 1/k) for large-kr boundary
# Balance padding vs resolution: Δr(R) = R×dln must be < λ/2.
# With Nr=2^17, r_max/r_min ratio ≤ exp(Nr × λ/(2R)) = exp(131072×1.25e-4) ≈ exp(16.4) ≈ 1.3e7
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

# ─── LPA near-field modes (Jacobi-Anger, y-polarized) ─────────
# E_{m,r}(r) = t(r) × W(r) × [i^{m-1} J_{m-1}(kx r) - i^{m+1} J_{m+1}(kx r)] / (2i)
# E_{m,θ}(r) = t(r) × W(r) × [i^{m-1} J_{m-1}(kx r) + i^{m+1} J_{m+1}(kx r)] / 2
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

function make_mode(m, r_grid, alpha_taper)
    Nr_l = length(r_grid)
    Em_r  = zeros(ComplexF64, Nr_l)
    Em_th = zeros(ComplexF64, Nr_l)
    for j in 1:Nr_l
        rv = r_grid[j]
        w  = tukey_window(rv, R, alpha_taper)
        t  = t_lens(rv)
        Jmm1 = besselj(m-1, kx * rv)
        Jmp1 = besselj(m+1, kx * rv)
        c1 = Complex(0.0, 1.0)^(m-1)
        c2 = Complex(0.0, 1.0)^(m+1)
        Em_r[j]  = t * w * (c1*Jmm1 - c2*Jmp1) / (2im)
        Em_th[j] = t * w * (c1*Jmm1 + c2*Jmp1) / 2
    end
    return Em_r, Em_th
end

# ─── Test: Round-trip vs taper width for representative modes ──
println("\n--- Round-trip H_ν(H_ν[g]) for production modes ---")
println("Format: alpha_taper → max round-trip error over tested modes\n")

test_modes = [1, 10, 100, 1000, min(M_max - 50, 4000)]
r_vec = collect(r)

for alpha_t in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    max_err = 0.0
    for m in test_modes
        Em_r_m, _ = make_mode(m, r_vec, alpha_t)
        g = r_vec .* Em_r_m   # r-weighted for dr measure

        # Skip negligible modes
        g_max = maximum(abs.(g))
        if g_max < 1e-20; continue; end

        # Forward + inverse (order m+1)
        raw  = fftlog_hankel(g, dln, Float64(m + 1))
        rt   = fftlog_hankel(raw, dln, Float64(m + 1))

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
    Em_r_m, _ = make_mode(m, r_vec, alpha_recommended)
    g = r_vec .* Em_r_m
    g_max = maximum(abs.(g))
    if g_max < 1e-20
        println("  m=$m: negligible amplitude, skipped")
        continue
    end

    raw  = fftlog_hankel(g, dln, Float64(m + 1))
    rt   = fftlog_hankel(raw, dln, Float64(m + 1))

    sig = findall(abs.(g) .> 1e-3 * g_max)
    g_s = g[sig]; rt_s = rt[sig]
    sc = real(dot(g_s, rt_s)) / (real(dot(rt_s, rt_s)) + 1e-30)
    err = maximum(abs.(rt_s .* sc .- g_s) ./ (abs.(g_s) .+ 1e-30))
    med_err = sort(abs.(rt_s .* sc .- g_s) ./ (abs.(g_s) .+ 1e-30))[length(sig)÷2]

    println("  m=$(lpad(m, 4)):  max_err=$(round(err, sigdigits=3)),  median_err=$(round(med_err, sigdigits=3)),  scale=$(round(sc, sigdigits=6)),  |g|_max=$(round(g_max, sigdigits=3))")
end

println("\n" * "="^60)
println("Production aperture test complete.")
println("="^60)
