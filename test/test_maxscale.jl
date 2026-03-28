"""
    test_maxscale.jl
    ================
    Maximum-scale validation on limited-memory machines.
    Tests the standard and Neumann paths at up to ~1580 modes with
    memory-saving (free arrays between pipeline steps).

    Validates both paths agree on the PSF (peak-normalized).

    Tests:
      1. R=500λ,  α=30°, M_max≈1581 (1581 modes, largest M tested)
      2. R=1000λ, α=10°, M_max≈1102 (largest R × angle combination)
      3. R=700λ,  α=20°, M_max≈1515 (intermediate)

    Run with: julia test_maxscale.jl
"""

include("../cyffp.jl")
using .CyFFP
using FFTW
using SpecialFunctions: besselj

println("="^70)
println("Maximum-scale validation: Neumann vs Standard (up to ~1580 modes)")
println("="^70)

lambda = 1.0
k      = 2π
Nr     = 65536
L      = 8

function maxscale_test(R_lam, alpha_deg)
    R     = R_lam * lambda
    f_val = R / 0.3 * sqrt(1 - 0.3^2)
    alpha = deg2rad(Float64(alpha_deg))
    kx    = k * sin(alpha)
    x0    = f_val * tan(alpha)
    M_max = ceil(Int, kx * R) + 10
    m_pos = collect(0:M_max)

    r   = collect(exp.(range(log(1e-4), log(1e6), length=Nr)))
    dln = log(r[2] / r[1])
    kr  = exp.(log(1.0 / r[end]) .+ dln .* (0:Nr-1))

    println("\n─── R=$(R_lam)λ  α=$(alpha_deg)°  M_max=$M_max  Δr(R)=$(round(R*dln, sigdigits=3))λ ───")
    @assert R * dln < 0.5 "Grid too coarse"

    t_lens(rv) = exp(-im * k * (sqrt(rv^2 + f_val^2) - f_val)) * (rv <= R ? 1.0 : 0.0)
    t_r = ComplexF64[t_lens(rv) for rv in r]

    # ── Standard path: per-r FFT → compute_scalar_coeffs → propagate → graf ──
    Ntheta = max(2 * M_max + 1, 4096)
    Ntheta = 1 << ceil(Int, log2(Ntheta))

    u_m = zeros(ComplexF64, Nr, M_max + 1)
    u_row = Vector{ComplexF64}(undef, Ntheta)
    for j in 1:Nr
        rv = r[j]; tv = t_lens(rv)
        abs(tv) < 1e-30 && continue
        for it in 1:Ntheta
            u_row[it] = tv * exp(im * kx * rv * cos(2π * (it-1) / Ntheta))
        end
        fft!(u_row); u_row ./= Ntheta
        for (idx, m) in enumerate(m_pos)
            u_m[j, idx] = u_row[m + 1]
        end
    end

    a_std, _ = compute_scalar_coeffs(u_m, m_pos, r)
    u_m = nothing; GC.gc()   # free modes before propagation

    at_s, mf_s = propagate_scalar(a_std, m_pos, collect(kr), k, f_val)
    a_std = nothing; GC.gc()
    B_s = graf_shift(at_s, mf_s, collect(kr), x0, L; k=k)
    at_s = nothing; GC.gc()

    # ── Neumann path ──
    a_neu, _, _ = neumann_shift_coeffs(t_r, r, k, alpha, M_max)
    at_n, mf_n = propagate_scalar(a_neu, m_pos, collect(kr), k, f_val)
    a_neu = nothing; GC.gc()
    B_n = graf_shift(at_n, mf_n, collect(kr), x0, L; k=k)
    at_n = nothing; GC.gc()

    # ── PSF comparison (brute-force Riemann sum at selected ρ) ──
    prop_kr = findall(kr .< k)
    psf_s = ComplexF64[]
    psf_n = ComplexF64[]
    rho_check = [0.001, 0.3, 0.7, 1.0, 1.5, 2.0, 3.0]

    for rho in rho_check
        bf_s = zero(ComplexF64); bf_n = zero(ComplexF64)
        for (li, l) in enumerate(-L:L)
            bf_s += dln * sum(B_s[j, li] * besselj(l, kr[j] * rho) * kr[j]^2
                              for j in prop_kr)
            bf_n += dln * sum(B_n[j, li] * besselj(l, kr[j] * rho) * kr[j]^2
                              for j in prop_kr)
        end
        push!(psf_s, bf_s)
        push!(psf_n, bf_n)
    end

    peak = maximum(abs.(psf_s))
    max_err = maximum(abs.(psf_s .- psf_n)) / peak

    # Also verify PSF peaks near ρ=0
    peak_rho_idx = argmax(abs2.(psf_s))
    peak_rho = rho_check[peak_rho_idx]

    println("  Peak at ρ=$(round(peak_rho, digits=3))λ")
    println("  Neumann vs standard PSF Δ/peak: $(round(100*max_err, digits=2))%")

    # Note: this is an LPA field (normal lens + oblique tilt), which has
    # coma.  At large angles the PSF peak can shift off-center.
    @assert peak_rho < 3.0 "PSF peak beyond 3λ — unexpected"
    @assert max_err < 0.02 "PSF mismatch > 2%"
    println("  PASSED ✓")

    B_s = nothing; B_n = nothing; GC.gc()
    return (R_lam=R_lam, alpha=alpha_deg, M_max=M_max, psf_err=max_err)
end

results = []

push!(results, maxscale_test(500, 30))
GC.gc()

push!(results, maxscale_test(1000, 10))
GC.gc()

push!(results, maxscale_test(700, 20))
GC.gc()

println("\n" * "="^70)
println("Summary")
println("="^70)
println("  R/λ    α°   M_max   PSF Δ/peak")
println("  ───────────────────────────────")
for r in results
    println("  $(lpad(r.R_lam, 5))  $(lpad(r.alpha, 3))°  $(lpad(r.M_max, 5))   $(round(100*r.psf_err, digits=2))%")
end
println("="^70)
println("All maximum-scale tests passed.")
