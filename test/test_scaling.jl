"""
    test_scaling.jl
    ===============
    Scalar Airy disk validation at progressively larger apertures,
    stress-testing FFTLog accuracy as mode count grows.

    Small apertures (R ≤ 100λ): use full 2D field + angular_decompose (Step 1).
    Large apertures (R ≥ 300λ): extract modes via per-r FFT (skips Step 1,
    tests Steps 2-4 only) to avoid O(Nr × Nθ) memory.

    R = 3λ, 10λ, 30λ, 100λ, 300λ, 1000λ
    M_max ≈ 12, 16, 27, 65, 175, 558

    Run with: julia test_scaling.jl
"""

include("../cyffp.jl")
using .CyFFP
using FFTW
using SpecialFunctions: besselj
using QuadGK

# Try to load Plots; if unavailable, skip plotting
do_plot = try
    using Plots
    true
catch
    @warn "Plots not available — install with ] add Plots to generate PSF figures"
    false
end

println("="^70)
println("Scaling test: Airy disk at R = 3λ .. 1000λ")
println("="^70)

lambda = 1.0
k      = 2π / lambda

results = []

# (R/λ, Nr, mode_source)
# :step1  = full 2D field + angular_decompose (tests Step 1)
# :per_r  = per-r FFT (skips Step 1, large-R memory workaround)
test_cases = [
    (3,     512,    64,   :step1),
    (10,    1024,   128,  :step1),
    (30,    2048,   256,  :step1),
    (100,   4096,   512,  :step1),
    (300,   16384,  0,    :per_r),
    (1000,  65536,  0,    :per_r),
]

for (R_lam, Nr, Ntheta_2d, mode_source) in test_cases
    R      = R_lam * lambda
    NA     = 0.3
    f_val  = R / NA * sqrt(1 - NA^2)
    alpha  = deg2rad(5.0)
    x0     = f_val * tan(alpha)
    kx     = k * sin(alpha)
    M_max  = ceil(Int, kx * R) + 10
    L_max  = 10

    NA_eff = R * cos(alpha) / sqrt((R*cos(alpha))^2 + f_val^2)
    rho_airy = 0.61 * lambda / NA_eff

    # Grid: need r_min ≤ 1/k, r_max ≥ ~50R, dln ≤ λ/(2R)
    r_min = min(1e-3, 0.5/k)
    r_max = max(1e3, 100*R)
    r_grid = collect(exp.(range(log(r_min), log(r_max), length=Nr)))
    dln    = log(r_grid[2] / r_grid[1])
    kr     = exp.(log(1.0 / r_grid[end]) .+ dln .* (0:Nr-1))

    println("\n" * "─"^70)
    println("R = $(R_lam)λ  |  Nr=$Nr  M_max=$M_max  L_max=$L_max  mode_source=$mode_source")
    println("f = $(round(f_val/lambda, digits=1))λ  NA_eff=$(round(NA_eff, digits=4))  ρ_Airy=$(round(rho_airy/lambda, digits=3))λ")
    println("dln=$(round(dln, sigdigits=4))  Δr(R)=$(round(R*dln, sigdigits=3))λ  (need < $(lambda/2))")

    if R * dln > lambda / 2
        println("  WARNING: grid too coarse at aperture edge, skipping")
        push!(results, (R_lam=R_lam, M_max=M_max, s2_err=NaN, parseval=NaN,
                         peak_rho=NaN, airy_err=NaN, min_val=NaN, status="SKIP"))
        continue
    end

    # ─── Ideal oblique lens near field ────────────────────────
    function u_lens(rv, th)
        d = sqrt((rv * cos(th) - x0)^2 + rv^2 * sin(th)^2 + f_val^2)
        return exp(-im * k * (d - f_val)) * (rv <= R ? 1.0 : 0.0)
    end

    m_pos = collect(0:M_max)

    if mode_source == :step1
        # ── Full 2D field → angular_decompose (Step 1) ────────
        theta = range(0.0, 2π, length=Ntheta_2d+1)[1:end-1]
        u_field = ComplexF64[u_lens(rv, th) for rv in r_grid, th in theta]
        u_m, _, m_pos = angular_decompose(u_field, zeros(ComplexF64, Nr, Ntheta_2d), M_max)
        println("  Step 1: angular_decompose with Nθ=$Ntheta_2d")

    elseif mode_source == :per_r
        # ── Per-r FFT (memory-efficient, skips Step 1) ─────────
        Ntheta_pr = max(2*M_max + 1, 4096)
        Ntheta_pr = 1 << ceil(Int, log2(Ntheta_pr))
        theta_pr = range(0.0, 2π, length=Ntheta_pr+1)[1:end-1]

        u_m = zeros(ComplexF64, Nr, M_max + 1)
        u_row = Vector{ComplexF64}(undef, Ntheta_pr)

        for j in 1:Nr
            rv = r_grid[j]
            if rv > R; continue; end
            for (it, th) in enumerate(theta_pr)
                u_row[it] = u_lens(rv, th)
            end
            fft!(u_row)
            u_row ./= Ntheta_pr
            for (idx, m) in enumerate(m_pos)
                u_m[j, idx] = u_row[m + 1]
            end
        end
        println("  Modes via per-r FFT with Nθ=$Ntheta_pr (Step 1 bypassed)")
    end

    # ─── Steps 2-4 (always through the module) ───────────────
    t_pipeline = @elapsed begin
        a_m, kr_grid = compute_scalar_coeffs(u_m, m_pos, r_grid)
        a_tilde, m_full = propagate_scalar(a_m, m_pos, kr_grid, k, f_val)
        B = graf_shift(a_tilde, m_full, kr_grid, x0, L_max; k=k)
    end
    println("  Steps 2-4: $(round(t_pipeline, digits=2))s")

    # ─── Step 2 cross-check: a_m vs QuadGK ──────────────────
    max_err_s2 = 0.0
    for m in [1, min(3, M_max)]
        idx = m + 1
        for kval in [0.5, 1.0, min(3.0, k*0.8)]
            ik = argmin(abs.(kr_grid .- kval))
            function u_m_interp(rv)
                lr = log(rv); lr0 = log(r_grid[1])
                idx_f = (lr - lr0) / dln + 1.0
                j0 = clamp(floor(Int, idx_f), 1, Nr-1)
                w = idx_f - j0
                return (1-w) * u_m[j0, idx] + w * u_m[j0+1, idx]
            end
            ref, _ = quadgk(rv -> u_m_interp(rv) * besselj(m, kval*rv) * rv,
                            r_grid[1], R*1.5; rtol=1e-8)
            err = abs(a_m[ik, idx] - ref) / (abs(ref) + 1e-30)
            max_err_s2 = max(max_err_s2, err)
        end
    end
    println("  Step 2 (a_m vs QuadGK): max err = $(round(max_err_s2, sigdigits=3))")

    # ─── Parseval ────────────────────────────────────────────
    max_parseval = 0.0
    for m in [0, 1, min(M_max÷2, 50), min(M_max-1, 100)]
        idx = m + 1
        if idx > size(u_m, 2); continue; end
        spatial  = dln * sum(abs2(u_m[j, idx]) * r_grid[j]^2 for j in 1:Nr)
        spectral = dln * sum(abs2(a_m[j, idx]) * kr_grid[j]^2 for j in 1:Nr)
        if spatial > 1e-20
            max_parseval = max(max_parseval, abs(spectral/spatial - 1))
        end
    end
    println("  Parseval: max |ratio-1| = $(round(max_parseval, sigdigits=3))")

    # ─── Brute-force PSF (Steps 5-6 via Riemann sum) ─────────
    prop_kr = findall(kr_grid .< k)
    rho_test = collect(range(0.001, 4.0, length=60)) .* lambda
    PSF = zeros(ComplexF64, length(rho_test))

    for (ir, rho) in enumerate(rho_test)
        field = zero(ComplexF64)
        for (li, l) in enumerate(-L_max:L_max)
            bl = dln * sum(B[j, li] * besselj(l, kr_grid[j] * rho) * kr_grid[j]^2
                           for j in prop_kr)
            field += bl  # ψ = 0
        end
        PSF[ir] = field
    end

    I_psf = abs2.(PSF)
    I_peak = maximum(I_psf)
    I_norm = I_psf ./ I_peak

    # Peak at ρ ≈ 0
    peak_idx = argmax(I_psf)
    peak_rho = rho_test[peak_idx]

    # First minimum
    first_min_rho = 0.0
    first_min_val = 1.0
    for ir in 2:length(I_norm)-1
        if I_norm[ir] < I_norm[ir-1] && I_norm[ir] < I_norm[ir+1]
            first_min_rho = rho_test[ir]
            first_min_val = I_norm[ir]
            break
        end
    end

    airy_err = first_min_rho > 0 ? abs(first_min_rho - rho_airy) / rho_airy : NaN
    println("  PSF peak at ρ=$(round(peak_rho/lambda, digits=3))λ")
    if first_min_rho > 0
        println("  First zero: ρ=$(round(first_min_rho/lambda, digits=3))λ (predicted $(round(rho_airy/lambda, digits=3))λ, error $(round(100*airy_err, digits=1))%)")
        println("  Intensity at minimum: $(round(first_min_val, sigdigits=3))")
    else
        println("  WARNING: no first minimum found")
    end

    # Monotonic decrease before first zero
    monotonic = true
    for ir in 2:min(peak_idx + 5, length(rho_test))
        if rho_test[ir] < rho_airy * 0.85
            if I_norm[ir] > I_norm[ir-1] + 0.02
                monotonic = false
                break
            end
        end
    end

    peak_ok = peak_rho < 0.5 * lambda
    airy_ok = first_min_rho > 0 && airy_err < 0.15
    deep_ok = first_min_rho > 0 && first_min_val < 0.01
    all_ok  = peak_ok && airy_ok && deep_ok && monotonic

    status = all_ok ? "PASS" : "FAIL"
    println("  Status: $status  (peak=$(peak_ok ? "✓" : "✗") airy=$(airy_ok ? "✓" : "✗") deep=$(deep_ok ? "✓" : "✗") mono=$(monotonic ? "✓" : "✗"))")

    # ─── Plot PSF vs theoretical Airy ─────────────────────────
    if do_plot
        rho_fine = range(0.001, 4.0, length=200) .* lambda
        # Theoretical Airy: [2J₁(v)/v]² where v = k NA_eff ρ
        airy_theory = map(rho_fine) do rho
            v = k * NA_eff * rho
            v < 1e-10 ? 1.0 : (2 * besselj(1, v) / v)^2
        end

        p = plot(rho_test ./ lambda, I_norm,
                 label="CyFFP (R=$(R_lam)λ, M=$(M_max))",
                 lw=2, marker=:circle, ms=3,
                 xlabel="ρ / λ", ylabel="I / I_peak",
                 title="PSF: R=$(R_lam)λ, α=5°, NA_eff=$(round(NA_eff, digits=3))",
                 ylims=(-0.05, 1.05), legend=:topright, size=(700,450))
        plot!(p, rho_fine ./ lambda, airy_theory,
              label="Airy [2J₁(v)/v]²", lw=2, ls=:dash, color=:red)
        vline!(p, [rho_airy / lambda], label="Airy zero ($(round(rho_airy/lambda, digits=2))λ)",
               ls=:dot, color=:gray)

        fname = "psf_R$(R_lam)lambda.png"
        savefig(p, fname)
        println("  Plot saved: $fname")
    end

    push!(results, (R_lam=R_lam, M_max=M_max, s2_err=max_err_s2, parseval=max_parseval,
                     peak_rho=peak_rho/lambda, airy_err=airy_err, min_val=first_min_val, status=status))

    @assert all_ok "FAILED at R=$(R_lam)λ"
end

# ─── Summary table ───────────────────────────────────────────
println("\n" * "="^70)
println("Summary")
println("="^70)
println("  R/λ   M_max  Steps   Step2_err  Parseval    Peak/λ   Airy_err  Min_I    Status")
println("  ──────────────────────────────────────────────────────────────────────────────")
for (i, r) in enumerate(results)
    src = test_cases[i][4] == :step1 ? "1-4" : "2-4"
    r.status == "SKIP" && continue
    println("  $(lpad(r.R_lam, 4))  $(lpad(r.M_max, 5))  $(lpad(src, 5))  $(lpad(round(r.s2_err, sigdigits=3), 9))  $(lpad(round(r.parseval, sigdigits=3), 9))  $(lpad(round(r.peak_rho, digits=3), 7))  $(lpad(round(100*r.airy_err, digits=1), 6))%  $(lpad(round(r.min_val, sigdigits=3), 7))  $(r.status)")
end
println("="^70)
println("Aperture scaling tests passed.")


# ═══════════════════════════════════════════════════════════════
# Part 2: Oblique angle scaling (R=100λ, α=5°..30°)
#
# At large oblique angles, the ideal-lens PSF is NOT a circular
# Airy disk — the aperture appears elliptical from the off-axis
# focal point.  The sagittal direction (ψ=π/2) retains the Airy
# pattern; the tangential direction (ψ=0) is broadened.
#
# This test validates:
#  - Sagittal first zero matches NA_eff Airy prediction at all angles
#  - Tangential broadening grows with angle (correct physics)
#  - PSF peaks at ρ≈0 at all angles
# ═══════════════════════════════════════════════════════════════

println("\n" * "="^70)
println("Oblique angle scaling: R=100λ, α=5°..30°")
println("="^70)

R_obl  = 100.0 * lambda
f_obl  = R_obl / 0.3 * sqrt(1 - 0.3^2)
Nr_obl = 4096
L_max_obl = 12
N_psi_obl = 64

r_obl = collect(exp.(range(log(1e-3), log(1e4), length=Nr_obl)))
dln_obl = log(r_obl[2] / r_obl[1])
kr_obl = exp.(log(1.0 / r_obl[end]) .+ dln_obl .* (0:Nr_obl-1))

angle_results = []

for alpha_deg in [5, 10, 15, 20, 30]
    alpha_val = deg2rad(Float64(alpha_deg))
    x0_val   = f_obl * tan(alpha_val)
    kx_val   = k * sin(alpha_val)
    M_max_val = ceil(Int, kx_val * R_obl) + 10

    NA_eff_val = R_obl * cos(alpha_val) / sqrt((R_obl * cos(alpha_val))^2 + f_obl^2)
    rho_airy_val = 0.61 * lambda / NA_eff_val

    Ntheta_val = max(2 * M_max_val + 1, 128)
    Ntheta_val = 1 << ceil(Int, log2(Ntheta_val))
    theta_val = range(0.0, 2π, length=Ntheta_val+1)[1:end-1]

    println("\n─── α=$(alpha_deg)°  M_max=$(M_max_val)  x₀/R=$(round(x0_val/R_obl, digits=2))  NA_eff=$(round(NA_eff_val, digits=3)) ───")

    # Ideal oblique lens
    u_field_obl = ComplexF64[let d=sqrt((rv*cos(th)-x0_val)^2+rv^2*sin(th)^2+f_obl^2)
        exp(-im*k*(d-f_obl))*(rv<=R_obl ? 1.0 : 0.0) end for rv in r_obl, th in theta_val]

    # Steps 1-6
    u_m_obl, _, m_pos_obl = angular_decompose(u_field_obl,
        zeros(ComplexF64, Nr_obl, Ntheta_val), M_max_val)
    a_m_obl, kr_g_obl = compute_scalar_coeffs(u_m_obl, m_pos_obl, r_obl)
    a_tilde_obl, m_full_obl = propagate_scalar(a_m_obl, m_pos_obl, kr_g_obl, k, f_obl)
    B_obl = graf_shift(a_tilde_obl, m_full_obl, kr_g_obl, x0_val, L_max_obl; k=k)
    b_obl, rho_obl = inverse_hankel(B_obl, L_max_obl, kr_g_obl)
    u_psf_obl, psi_obl = angular_synthesis(b_obl, L_max_obl, N_psi_obl)

    I_psf_obl = abs2.(u_psf_obl)

    # Extract tangential (ψ=0) and sagittal (ψ=π/2) profiles
    I_tang = I_psf_obl[:, 1]                          # ψ = 0
    I_sag  = I_psf_obl[:, N_psi_obl÷4 + 1]           # ψ = π/2
    I_peak_obl = max(maximum(I_tang), maximum(I_sag))

    # Peak location
    I_avg_obl = [sum(I_psf_obl[ir, :]) / N_psi_obl for ir in 1:Nr_obl]
    peak_rho_obl = rho_obl[argmax(I_avg_obl)]

    # Find first zero in each direction
    function find_zero(I_prof, rho_grid, I_pk)
        I_n = I_prof ./ I_pk
        for ir in 2:length(I_n)-1
            if rho_grid[ir] > 0.5 && I_n[ir] < I_n[ir-1] && I_n[ir] < I_n[ir+1]
                return rho_grid[ir], I_n[ir]
            end
        end
        return NaN, NaN
    end

    zt, mt = find_zero(I_tang, rho_obl, I_peak_obl)
    zs, ms = find_zero(I_sag, rho_obl, I_peak_obl)

    ae_tang = isnan(zt) ? NaN : abs(zt - rho_airy_val) / rho_airy_val
    ae_sag  = isnan(zs) ? NaN : abs(zs - rho_airy_val) / rho_airy_val

    println("  Peak at ρ=$(round(peak_rho_obl, digits=3))λ")
    println("  Sagittal   (ψ=π/2): zero=$(round(zs, digits=3))λ  err=$(round(100*ae_sag, digits=1))%  min=$(round(ms, sigdigits=2))")
    println("  Tangential (ψ=0):   zero=$(round(zt, digits=3))λ  err=$(round(100*ae_tang, digits=1))%  min=$(round(mt, sigdigits=2))")

    # Assertions:
    # - Peak at center
    @assert peak_rho_obl < 0.5 "PSF peak not at center at α=$alpha_deg"

    # - Sagittal matches Airy (the aperture is circular in this direction)
    @assert !isnan(ae_sag) "No sagittal zero found at α=$alpha_deg"
    @assert ae_sag < 0.05 "Sagittal Airy error > 5% at α=$alpha_deg"
    @assert ms < 0.001 "Sagittal minimum not deep enough at α=$alpha_deg"

    # - Tangential broadens with angle (expected physics)
    if !isnan(ae_tang) && alpha_deg >= 15
        @assert ae_tang > ae_sag "Tangential should be broader than sagittal at α=$alpha_deg"
    end

    println("  PASSED ✓")

    push!(angle_results, (alpha=alpha_deg, M_max=M_max_val,
          sag_err=ae_sag, tang_err=ae_tang, peak=peak_rho_obl))
end

# Summary
println("\n" * "="^70)
println("Oblique angle summary (R=100λ)")
println("="^70)
println("  α°    M_max  Sag_err  Tang_err  Peak/λ")
println("  ─────────────────────────────────────────")
for r in angle_results
    println("  $(lpad(r.alpha, 3))  $(lpad(r.M_max, 5))   $(lpad(round(100*r.sag_err, digits=1), 5))%   $(lpad(isnan(r.tang_err) ? "N/A" : "$(round(100*r.tang_err, digits=1))%", 7))   $(round(r.peak, digits=3))")
end
println("="^70)
println("All oblique angle tests passed.")
