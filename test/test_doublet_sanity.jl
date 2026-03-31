# Quick sanity check: doublet with t₂=1 and d≈0 should match singlet with same t₁
include(joinpath(@__DIR__, "..", "cyffp.jl"))
using .CyFFP
using Random

pitch = 0.3; R = 1000.0; N = round(Int, R/pitch)
k = 2π/0.5; f = R*sqrt(1/0.4^2-1)
r_c = [(i-0.5)*pitch for i in 1:N]
t_ones = ones(ComplexF64, N)

plan = prepare_psf(pitch, N; lambda_um=0.5, alpha_deg=30.0, NA=0.4, Nxy=101)

# Test with several t₁ profiles
Random.seed!(42)
profiles = [
    ("ideal lens",  ComplexF64[exp(-im*k*(sqrt(r^2+f^2)-f)) for r in r_c]),
    ("quadratic",   ComplexF64[exp(-im*k*r^2/(2f)) for r in r_c]),
    ("random",      exp.(im .* 2π .* rand(N))),
    ("uniform",     ones(ComplexF64, N)),
]

for (label, t1) in profiles
    println("── $label ──")
    res_s  = execute_psf(plan, t1)
    res_d  = execute_psf_doublet(plan, t1, t_ones; d_um=0.001)

    diff = maximum(abs.(res_d.I_raw .- res_s.I_raw)) / (maximum(res_s.I_raw) + 1e-30)
    sum_s = sum(res_s.I_raw)
    sum_d = sum(res_d.I_raw)
    println("  singlet sum(I) = $sum_s")
    println("  doublet sum(I) = $sum_d")
    println("  max |diff| / peak = $diff")
    println()
end
