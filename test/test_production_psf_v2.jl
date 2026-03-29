"""
    test_production_psf_v2.jl
    =========================
    Production PSF test for the IDEAL OBLIQUE lens, using the same
    internal code path as execute_psf for Steps 2-6.

    The only difference from execute_psf is Step 1 (mode construction):
    - execute_psf: Jacobi-Anger u_m = i^m t(r) J_m(kₓr)  (LPA model)
    - This test: per-r FFT of the ideal oblique phase     (exact)

    Steps 2-6 use the identical code: plan.kernels, fused prop+graf,
    ESTIMATE workspaces, inverse_hankel, angular_synthesis.  This
    ensures the production forward path is validated with a known
    analytical result (Airy disk).

    Run with:  julia -t 388 test/test_production_psf_v2.jl
"""

include("../cyffp.jl")
using .CyFFP
using FFTW
using SpecialFunctions: besselj

const HAS_PLOTS = try; using Plots; true; catch; false; end

function run_production_psf_v2(;
        R_um::Float64       = 2000.0,
        lambda_um::Float64  = 0.5,
        NA::Float64         = 0.4,
        alpha_deg::Float64  = 30.0,
        pitch_um::Float64   = 0.3,
        L_max::Int          = 15,
        N_psi::Int          = 64,
        Nxy::Int            = 301,
        rho_max_lambda::Float64 = 5.0,
        prefix::String      = "prod_v2",
        do_plots::Bool      = true,
    )

    k     = 2π / lambda_um
    N_cells = round(Int, R_um / pitch_um)
    R     = N_cells * pitch_um
    f_val = R * sqrt(1/NA^2 - 1)
    alpha = deg2rad(alpha_deg)
    x0    = f_val * tan(alpha)
    kx    = k * sin(alpha)
    M_max = ceil(Int, kx * R) + 20

    NA_eff_sag = R * cos(alpha) / sqrt((R * cos(alpha))^2 + f_val^2)
    rho_airy   = 0.61 * lambda_um / NA_eff_sag

    println("="^70)
    println("Production PSF v2: ideal oblique lens via execute_psf path")
    println("="^70)
    println("  R        = $R μm ($N_cells cells × $(pitch_um)μm)")
    println("  λ        = $lambda_um μm,  NA = $NA")
    println("  f        = $(round(f_val, digits=1)) μm")
    println("  α        = $(alpha_deg)°,  x₀ = $(round(x0, digits=1)) μm  (x₀/R = $(round(x0/R, digits=2)))")
    println("  M_max    = $M_max,  L_max = $L_max")
    println("  NA_eff(sag) = $(round(NA_eff_sag, digits=4))")
    println("  ρ_Airy(sag) = $(round(rho_airy, digits=4)) μm = $(round(rho_airy/lambda_um, digits=3))λ")

    # ─── Create plan (for grid, kernels, workspaces) ──────────
    # We use prepare_psf to get the same grid and precomputed data
    # as execute_psf, but we'll replace the mode construction.
    psf_hw = rho_max_lambda * lambda_um
    println("\n--- prepare_psf ---")
    t_prep = @elapsed begin
        plan = prepare_psf(pitch_um, N_cells;
                           lambda_um=lambda_um, alpha_deg=alpha_deg, NA=NA,
                           L_max=L_max, N_psi=N_psi, Nxy=Nxy,
                           psf_half_width_um=psf_hw)
    end
    Nr  = plan.Nr
    dln = plan.dln
    kr  = plan.kr
    r   = plan.r_log
    m_pos = plan.m_pos
    println("  Nr = $Nr,  dln = $(round(dln, sigdigits=4))")
    println("  prepare_psf: $(round(t_prep, digits=1))s")

    # ═══════════════════════════════════════════════════════════
    # Step 1: ideal oblique lens modes via per-r FFT
    # (This is the ONLY step that differs from execute_psf)
    # ═══════════════════════════════════════════════════════════
    println("\n--- Step 1: ideal oblique modes via per-r FFT ---")
    Ntheta = max(2M_max + 1, 4096)
    Ntheta = 1 << ceil(Int, log2(Ntheta))

    u_m = zeros(ComplexF64, Nr, M_max + 1)
    u_row = Vector{ComplexF64}(undef, Ntheta)

    t1 = @elapsed begin
        for j in 1:Nr
            rv = r[j]
            rv > R && continue
            for it in 1:Ntheta
                th = 2π * (it - 1) / Ntheta
                d = sqrt((rv * cos(th) - x0)^2 + rv^2 * sin(th)^2 + f_val^2)
                u_row[it] = exp(-im * k * (d - f_val))
            end
            fft!(u_row)
            u_row ./= Ntheta
            for (idx, m) in enumerate(m_pos)
                u_m[j, idx] = u_row[m + 1]
            end
        end
    end
    println("  Modes built: $(Nr) × $(M_max+1), Nθ=$Ntheta  ($(round(t1, digits=1))s)")

    # ═══════════════════════════════════════════════════════════
    # Steps 2-6: SAME code path as execute_psf
    # (plan.kernels, fused prop+graf, ESTIMATE workspaces, etc.)
    # ═══════════════════════════════════════════════════════════

    # ── Step 2: FFTLog with plan.kernels + ESTIMATE workspaces ──
    println("\n--- Step 2: FFTLog (plan.kernels + ESTIMATE) ---")
    a_m = zeros(ComplexF64, Nr, M_max + 1)
    t2 = @elapsed begin
        nt = CyFFP._nworkspaces()
        ws_s2 = [CyFFP._make_workspace(Nr, dln; flags=FFTW.ESTIMATE) for _ in 1:nt]
        g_s2  = [Vector{ComplexF64}(undef, Nr) for _ in 1:nt]

        Threads.@threads for idx in 1:M_max+1
            tid = Threads.threadid()
            ws = ws_s2[tid]; g = g_s2[tid]
            m = m_pos[idx]
            m_abs = abs(m)
            neg_sign = (m < 0 && isodd(m_abs)) ? -1 : 1
            @inbounds for j in 1:Nr; g[j] = r[j] * u_m[j, idx]; end
            CyFFP._fftlog_with_kernel!(view(a_m, :, idx), ws, g,
                                        view(plan.kernels, :, m_abs + 1), neg_sign)
            @inbounds for j in 1:Nr; a_m[j, idx] /= kr[j]; end
        end
    end
    println("  $(M_max+1) modes × Nr=$Nr  ($(round(t2, digits=1))s)")

    # Parseval check
    for m in [0, M_max÷2, M_max]
        idx = m + 1
        sp = dln * sum(abs2(u_m[j, idx]) * r[j]^2 for j in 1:Nr)
        sk = dln * sum(abs2(a_m[j, idx]) * kr[j]^2 for j in 1:Nr)
        ratio = sp > 1e-20 ? sk / sp : NaN
        println("    Parseval m=$m: $(round(ratio, sigdigits=6))")
    end

    # ── Steps 3+4 fused: propagation + Graf shift ──
    println("\n--- Steps 3+4: fused prop+graf ---")
    kz_fwd   = @. sqrt(complex(k^2 - kr.^2))
    prop_fwd = @. ifelse(kr < k, exp(im * real(kz_fwd) * f_val), zero(ComplexF64))
    Nl = 2L_max + 1
    B  = zeros(ComplexF64, Nr, Nl)

    nt_g = CyFFP._nworkspaces()
    jp_g = [Vector{Float64}(undef, M_max + 21) for _ in 1:nt_g]
    jw_g = [Vector{Float64}(undef, 2M_max + 1) for _ in 1:nt_g]

    t34 = @elapsed begin
        Threads.@threads for ikr in 1:Nr
            kr[ikr] > k && continue
            tid = Threads.threadid()
            kr_x0 = kr[ikr] * x0
            m_cut = min(M_max, ceil(Int, abs(kr_x0)) + 20)
            pv = prop_fwd[ikr]

            CyFFP._besselj_range!(jp_g[tid], m_cut, kr_x0)
            Jw = jw_g[tid]
            @inbounds for d in 0:m_cut
                Jw[d + m_cut + 1] = jp_g[tid][d + 1]
                Jw[-d + m_cut + 1] = iseven(d) ? jp_g[tid][d + 1] : -jp_g[tid][d + 1]
            end

            @inbounds for (li, l) in enumerate(-L_max:L_max)
                acc = zero(ComplexF64)
                m_lo = max(-m_cut, -M_max - l)
                m_hi = min(m_cut, M_max - l)
                m_split = max(m_lo, -l)
                @simd for m in m_split:m_hi
                    acc += a_m[ikr, m + l + 1] * pv * Jw[m + m_cut + 1]
                end
                @simd for m in m_lo:min(m_hi, -l - 1)
                    n_abs = -m - l
                    sign_n = 1 - 2 * (n_abs & 1)
                    acc += sign_n * a_m[ikr, n_abs + 1] * pv * Jw[m + m_cut + 1]
                end
                B[ikr, li] = acc
            end
        end
    end
    println("  L_max=$L_max, x₀=$(round(x0, digits=1))μm  ($(round(t34, digits=1))s)")

    # ── Step 5: inverse Hankel ──
    println("\n--- Step 5: inverse Hankel ---")
    t5 = @elapsed begin
        b, rho = inverse_hankel(B, L_max, collect(kr))
    end
    println("  $(Nl) modes  ($(round(t5, digits=1))s)")

    # ── Step 6: angular synthesis ──
    println("\n--- Step 6: angular synthesis ---")
    t6 = @elapsed begin
        u_psf, psi = angular_synthesis(b, L_max, N_psi)
    end
    println("  N_ψ=$N_psi  ($(round(t6, digits=1))s)")

    I_polar = abs2.(u_psf)
    t_total = t1 + t2 + t34 + t5 + t6

    println("\n  Total pipeline: $(round(t_total, digits=1))s")
    println("    Step 1 (modes):     $(round(t1, digits=1))s")
    println("    Step 2 (FFTLog):    $(round(t2, digits=1))s")
    println("    Steps 3+4 (fused):  $(round(t34, digits=1))s")
    println("    Step 5 (inv HT):    $(round(t5, digits=1))s")
    println("    Step 6 (synth):     $(round(t6, digits=1))s")

    # ═══════════════════════════════════════════════════════════
    # PSF analysis (same as test_production_psf)
    # ═══════════════════════════════════════════════════════════
    println("\n" * "="^70)
    println("PSF analysis")
    println("="^70)

    I_avg = [sum(I_polar[ir, :]) / N_psi for ir in 1:Nr]
    I_peak = maximum(I_avg)
    peak_ir = argmax(I_avg)
    peak_rho = rho[peak_ir]
    println("  PSF peak at ρ = $(round(peak_rho/lambda_um, digits=3))λ = $(round(peak_rho, digits=4)) μm")

    # Sagittal profile (ψ = π/2)
    psi_sag_idx = N_psi ÷ 4 + 1
    I_sag = I_polar[:, psi_sag_idx]
    I_sag_peak = maximum(I_sag)

    # Sagittal first zero
    z_sag = NaN; z_sag_val = NaN
    for ir in 2:Nr-1
        if rho[ir] > 0.5 * lambda_um && I_sag[ir] < I_sag[ir-1] && I_sag[ir] < I_sag[ir+1]
            z_sag = rho[ir]; z_sag_val = I_sag[ir] / I_sag_peak; break
        end
    end
    ae_sag = isnan(z_sag) ? NaN : abs(z_sag - rho_airy) / rho_airy

    # Tangential profile (ψ = 0)
    I_tang = I_polar[:, 1]
    z_tang = NaN
    for ir in 2:Nr-1
        if rho[ir] > 0.5 * lambda_um && I_tang[ir] < I_tang[ir-1] && I_tang[ir] < I_tang[ir+1]
            z_tang = rho[ir]; break
        end
    end
    ae_tang = isnan(z_tang) ? NaN : abs(z_tang - rho_airy) / rho_airy

    println("\n  Sagittal (ψ=π/2):")
    println("    First zero = $(round(z_sag/lambda_um, digits=3))λ = $(round(z_sag, digits=4)) μm")
    println("    Predicted   = $(round(rho_airy/lambda_um, digits=3))λ = $(round(rho_airy, digits=4)) μm")
    println("    Error       = $(round(100*ae_sag, digits=1))%")
    println("    Min I/I_peak = $(round(z_sag_val, sigdigits=3))")

    println("\n  Tangential (ψ=0):")
    if !isnan(z_tang)
        println("    First zero = $(round(z_tang/lambda_um, digits=3))λ = $(round(z_tang, digits=4)) μm")
        println("    Error vs sagittal prediction = $(round(100*ae_tang, digits=1))%")
    end
    println("    (tangential broadening expected at α=$(alpha_deg)°)")

    # ─── Assertions ──────────────────────────────────────────
    println("\n--- Assertions ---")
    @assert peak_rho < 0.5 * lambda_um "PSF peak not at center"
    println("  Peak at center ✓")

    @assert !isnan(ae_sag) && ae_sag < 0.10 "Sagittal Airy zero >10%"
    println("  Sagittal Airy zero within $(round(100*ae_sag, digits=1))% ✓")

    @assert z_sag_val < 1e-6 "Sagittal dark ring not deep enough"
    println("  Sagittal dark ring depth $(round(z_sag_val, sigdigits=3)) ✓")

    @assert !any(isnan, I_polar) && !any(isinf, I_polar) "NaN/Inf"
    println("  No NaN/Inf ✓")

    println("\n  All PSF checks PASSED ✓")

    # ─── Plots (matching test_production_psf.jl format) ─────
    if do_plots && HAS_PLOTS
        println("\n--- Generating plots ---")

        rho_plot_max = rho_max_lambda * lambda_um

        # ── 1D radial profiles ──
        rho_fine = range(0.001, rho_max_lambda, length=300) .* lambda_um
        airy_th = map(rho_fine) do rv
            v = k * NA_eff_sag * rv
            v < 1e-10 ? 1.0 : (2 * besselj(1, v) / v)^2
        end

        p1 = plot(rho ./ lambda_um, I_avg ./ I_peak,
                  label="ψ-averaged", lw=2,
                  xlabel="ρ / λ", ylabel="I / I_peak",
                  title="PSF: R=$(R)μm, α=$(alpha_deg)°, NA=$(NA)",
                  xlims=(0, rho_max_lambda), ylims=(-0.05, 1.05),
                  legend=:topright, size=(750, 480))
        plot!(p1, rho ./ lambda_um, I_tang ./ I_peak,
              label="ψ=0° (tangential)", lw=1.5, ls=:dash)
        plot!(p1, rho ./ lambda_um, I_sag ./ I_sag_peak,
              label="ψ=90° (sagittal)", lw=1.5, ls=:dashdot)
        plot!(p1, rho_fine ./ lambda_um, airy_th,
              label="Airy (sagittal NA)", lw=2, color=:red, ls=:dot)
        vline!(p1, [rho_airy / lambda_um],
               label="", ls=:dot, color=:gray, lw=0.5)

        fname1 = "$(prefix)_psf_1d.png"
        savefig(p1, fname1)
        println("  Saved: $fname1")

        # ── 2D heatmap on uniform Cartesian grid ──
        xy_max = rho_plot_max
        x_cart = range(-xy_max, xy_max, length=Nxy) ./ lambda_um
        y_cart = range(-xy_max, xy_max, length=Nxy) ./ lambda_um
        dpsi   = psi[2] - psi[1]

        I_cart = zeros(Nxy, Nxy)
        for (ix, xv_lam) in enumerate(x_cart)
            for (iy, yv_lam) in enumerate(y_cart)
                xv = xv_lam * lambda_um
                yv = yv_lam * lambda_um
                rv = sqrt(xv^2 + yv^2)
                (rv < rho[1] || rv > rho[end]) && continue
                pv = atan(yv, xv); pv < 0 && (pv += 2π)
                lr = log(rv); lr0 = log(rho[1])
                idx_r = (lr - lr0) / dln + 1.0
                j0 = clamp(floor(Int, idx_r), 1, Nr - 1); wr = idx_r - j0
                idx_p = pv / dpsi + 1.0
                p0 = clamp(floor(Int, idx_p), 1, N_psi)
                p1_idx = p0 == N_psi ? 1 : p0 + 1; wp = idx_p - p0
                I_cart[iy, ix] = (
                    (1-wr)*(1-wp)*I_polar[j0, p0]     + wr*(1-wp)*I_polar[j0+1, p0] +
                    (1-wr)*wp    *I_polar[j0, p1_idx]  + wr*wp    *I_polar[j0+1, p1_idx]
                )
            end
        end
        I_cart ./= maximum(I_cart)

        p2 = heatmap(collect(x_cart), collect(y_cart), I_cart;
                     xlabel="x / λ", ylabel="y / λ",
                     color=:inferno, clims=(0, 1),
                     title="PSF: R=$(R)μm, α=$(alpha_deg)°, NA=$(NA)",
                     aspect_ratio=:equal, size=(600, 550))
        th_c = range(0, 2π, length=100)
        rl = rho_airy / lambda_um
        plot!(p2, rl .* cos.(th_c), rl .* sin.(th_c),
              label="Airy zero (sag)", color=:white, ls=:dash, lw=1.5)

        fname2 = "$(prefix)_psf_2d.png"
        savefig(p2, fname2)
        println("  Saved: $fname2")

        # ── Log-scale heatmap (shows sidelobes) ──
        I_cart_log = log10.(clamp.(I_cart, 1e-6, 1.0))

        p3 = heatmap(collect(x_cart), collect(y_cart), I_cart_log;
                     xlabel="x / λ", ylabel="y / λ",
                     color=:inferno, clims=(-4, 0),
                     title="PSF (log₁₀): R=$(R)μm, α=$(alpha_deg)°",
                     aspect_ratio=:equal, size=(600, 550))
        plot!(p3, rl .* cos.(th_c), rl .* sin.(th_c),
              label="Airy zero (sag)", color=:white, ls=:dash, lw=1.5)

        fname3 = "$(prefix)_psf_2d_log.png"
        savefig(p3, fname3)
        println("  Saved: $fname3")
    end

    println("\n" * "="^70)
    println("Production PSF v2 test complete.")
    println("="^70)

    return (R_um=R, lambda_um=lambda_um, NA=NA, alpha_deg=alpha_deg,
            M_max=M_max, Nr=Nr, rho_airy=rho_airy,
            sag_zero=z_sag, sag_err=ae_sag, peak_rho=peak_rho,
            t_prep=t_prep, t_total=t_total)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_production_psf_v2()
end
