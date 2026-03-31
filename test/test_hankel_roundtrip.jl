# test/test_hankel_roundtrip.jl — Verify forward → inverse Hankel round-trip
#
# Essential prerequisite for the doublet pipeline: after forward Hankel
# (r → kr) and inverse Hankel (kr → ρ), we must recover the original
# function on the same grid.
#
# Run:  julia -t auto test/test_hankel_roundtrip.jl

using Printf
include(joinpath(@__DIR__, "..", "cyffp.jl"))
using .CyFFP
using FFTW
using SpecialFunctions: besselj

# ─── Grid setup (same as production: 2mm lens, 30°, NA=0.4) ───
const lambda_um = 0.5
const R_um      = 2000.0
const r_min     = lambda_um / (20π)
const r_max     = 50 * R_um
const dln_max   = lambda_um / (2 * R_um)
const Nr_min    = ceil(Int, log(r_max / r_min) / dln_max)
const Nr        = 1 << ceil(Int, log2(Nr_min))

r   = exp.(range(log(r_min), log(r_max), length=Nr))
dln = log(r[2] / r[1])
kr  = exp.(log(1.0 / r[end]) .+ dln .* (0:Nr-1))
rho = exp.(log(1.0 / kr[end]) .+ dln .* (0:Nr-1))

println("═══ Hankel round-trip test ═══")
println("Nr = $Nr,  dln = $(round(dln, sigdigits=4))")
println("r range:  [$(r[1]), $(r[end])] μm")
println("kr range: [$(kr[1]), $(kr[end])] μm⁻¹")
println()

# Grid self-reciprocity
grid_err = maximum(abs.(rho .- r) ./ r)
println("Grid self-reciprocity: max |ρ-r|/r = $grid_err")
println()

# ─── Precompute kernels and workspaces ───
m_max_test = 6000
n_idx   = [n <= (Nr-1)÷2 ? n : n - Nr for n in 0:Nr-1]
q       = @. (2π / (Nr * dln)) * n_idx
kernels = CyFFP._precompute_kernels(q, collect(0:m_max_test))

ws    = CyFFP._make_workspace(Nr, dln; flags=FFTW.ESTIMATE)
g_buf = Vector{ComplexF64}(undef, Nr)
out   = Vector{ComplexF64}(undef, Nr)

# ─── Round-trip helper ───
function hankel_roundtrip(u_orig::Vector{ComplexF64}, m::Int)
    m_abs = abs(m)
    neg_sign = (m < 0 && isodd(m_abs)) ? -1 : 1

    # Forward: u(r) → a(kr)
    @inbounds for j in 1:Nr; g_buf[j] = r[j] * u_orig[j]; end
    CyFFP._fftlog_with_kernel!(out, ws, g_buf,
                                view(kernels, :, m_abs + 1), neg_sign)
    a_kr = copy(out)
    @inbounds for j in 1:Nr; a_kr[j] /= kr[j]; end

    # Parseval: ∫|u|² r dr ≈ dln Σ |u|² r²  vs  ∫|a|² kr dkr ≈ dln Σ |a|² kr²
    P_r  = dln * sum(abs2(u_orig[j]) * r[j]^2 for j in 1:Nr)
    P_kr = dln * sum(abs2(a_kr[j]) * kr[j]^2 for j in 1:Nr)

    # Inverse: a(kr) → u_recovered(ρ)
    @inbounds for j in 1:Nr; g_buf[j] = kr[j] * a_kr[j]; end
    CyFFP._fftlog_with_kernel!(out, ws, g_buf,
                                view(kernels, :, m_abs + 1), neg_sign)
    u_recov = copy(out)
    @inbounds for j in 1:Nr; u_recov[j] /= rho[j]; end

    # L² relative error
    l2_num = dln * sum(abs2(u_recov[j] - u_orig[j]) * r[j]^2 for j in 1:Nr)
    l2_rel = sqrt(l2_num / (P_r + 1e-30))

    # Parseval ratio
    parseval = P_kr / (P_r + 1e-30)

    return l2_rel, parseval, P_r
end

# ─── Test 1: LPA-like modes i^m t(r) J_m(kₓr) ───
println("── LPA modes: i^m t(r) J_m(kₓr), ideal lens, R=2mm, α=30° ──")
println()

k     = 2π / lambda_um
kx    = k * sin(deg2rad(30.0))
f_val = R_um * sqrt(1/0.4^2 - 1)

@printf("%-6s  %15s  %15s  %15s\n", "m", "L² rel err", "Parseval", "P_r")
println("─"^58)

for m in [0, 1, 2, 5, 10, 50, 100, 500, 1000, 3000, 6000]
    u_orig = ComplexF64[
        (r[j] <= R_um ?
         (im)^m * exp(-im * k * (sqrt(r[j]^2 + f_val^2) - f_val)) * besselj(m, kx * r[j])
         : 0.0 + 0.0im)
        for j in 1:Nr
    ]
    l2_rel, parseval, P_r = hankel_roundtrip(u_orig, m)
    @printf("%-6d  %15.6e  %15.12f  %15.6e\n", m, l2_rel, parseval, P_r)
end
println("─"^58)
println()

# ─── Test 2: Smooth real functions ───
println("── Smooth real functions ──")
println()

test_fns = [
    ("r²·Gauss(σ=50)",  r -> r^2 * exp(-r^2 / (2*50.0^2))),
    ("r²·Gauss(σ=500)", r -> r^2 * exp(-r^2 / (2*500.0^2))),
    ("Oscillatory",      r -> sin(2π*r/10.0) * exp(-r/500.0)),
    ("Gauss(σ=100)",     r -> exp(-r^2 / (2*100.0^2))),
    ("Top-hat R=2000",   r -> r <= R_um ? 1.0 : 0.0),
]

for m in [0, 5, 500]
    @printf("m = %d:\n", m)
    @printf("  %-20s  %15s  %15s\n", "function", "L² rel err", "Parseval")
    for (fname, func) in test_fns
        u_orig = ComplexF64[func(r[j]) for j in 1:Nr]
        l2_rel, parseval, _ = hankel_roundtrip(u_orig, m)
        @printf("  %-20s  %15.6e  %15.12f\n", fname, l2_rel, parseval)
    end
    println()
end

# ─── Test 3: Propagate-then-roundtrip (simulates doublet mid-step) ───
# Forward Hankel → propagate by d → inverse Hankel → multiply t₂ → forward Hankel → propagate back → inverse Hankel
# Check that the full doublet round-trip preserves the function when t₂=1 and total propagation=0.
println("── Doublet simulation: fwd HT → prop(d) → inv HT → ×1 → fwd HT → prop(-d) → inv HT ──")
println("(t₂ = 1, net propagation = 0, should recover original)")
println()

@printf("%-6s  %15s  %15s\n", "m", "L² rel err", "Parseval")
println("─"^40)

d_test = 500.0  # μm gap
kz = @. sqrt(complex(k^2 - kr.^2))
prop_d   = @. ifelse(kr < k, exp(im * real(kz) * d_test), zero(ComplexF64))
prop_neg = @. ifelse(kr < k, exp(-im * real(kz) * d_test), zero(ComplexF64))

for m in [0, 1, 10, 100, 1000, 6000]
    u_orig = ComplexF64[
        (r[j] <= R_um ?
         (im)^m * exp(-im * k * (sqrt(r[j]^2 + f_val^2) - f_val)) * besselj(m, kx * r[j])
         : 0.0 + 0.0im)
        for j in 1:Nr
    ]

    m_abs = abs(m); neg_sign = 1
    P_orig = dln * sum(abs2(u_orig[j]) * r[j]^2 for j in 1:Nr)

    # Step 1: forward Hankel
    @inbounds for j in 1:Nr; g_buf[j] = r[j] * u_orig[j]; end
    CyFFP._fftlog_with_kernel!(out, ws, g_buf, view(kernels, :, m_abs+1), neg_sign)
    a1 = copy(out)
    @inbounds for j in 1:Nr; a1[j] /= kr[j]; end

    # Step 2: propagate by +d
    a1_prop = a1 .* prop_d

    # Step 3: inverse Hankel → spatial at surface 2
    @inbounds for j in 1:Nr; g_buf[j] = kr[j] * a1_prop[j]; end
    CyFFP._fftlog_with_kernel!(out, ws, g_buf, view(kernels, :, m_abs+1), neg_sign)
    u_mid = copy(out)
    @inbounds for j in 1:Nr; u_mid[j] /= rho[j]; end

    # Step 4: multiply by t₂ = 1 (identity)

    # Step 5: forward Hankel again
    @inbounds for j in 1:Nr; g_buf[j] = r[j] * u_mid[j]; end
    CyFFP._fftlog_with_kernel!(out, ws, g_buf, view(kernels, :, m_abs+1), neg_sign)
    a2 = copy(out)
    @inbounds for j in 1:Nr; a2[j] /= kr[j]; end

    # Step 6: propagate by -d
    a2_prop = a2 .* prop_neg

    # Step 7: inverse Hankel → recover original
    @inbounds for j in 1:Nr; g_buf[j] = kr[j] * a2_prop[j]; end
    CyFFP._fftlog_with_kernel!(out, ws, g_buf, view(kernels, :, m_abs+1), neg_sign)
    u_final = copy(out)
    @inbounds for j in 1:Nr; u_final[j] /= rho[j]; end

    P_final = dln * sum(abs2(u_final[j]) * r[j]^2 for j in 1:Nr)
    l2_num  = dln * sum(abs2(u_final[j] - u_orig[j]) * r[j]^2 for j in 1:Nr)
    l2_rel  = sqrt(l2_num / (P_orig + 1e-30))
    parseval = P_final / (P_orig + 1e-30)

    @printf("%-6d  %15.6e  %15.12f\n", m, l2_rel, parseval)
end
println("─"^40)
println()
println("Done.")
