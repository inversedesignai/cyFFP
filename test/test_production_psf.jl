"""
    test_production_psf.jl
    ======================
    Production-level PSF test for an ideal oblique lens via the
    standard scalar pipeline (Steps 1-6).

    Default parameters: 2mm lens, λ=500nm, NA=0.4, α=30°.
    All parameters are adjustable via keyword arguments.

    Outputs:
      - 1D radial profiles (ψ-averaged, tangential, sagittal) vs Airy theory
      - 2D PSF heatmap on a uniform Cartesian grid
      - Quantitative Airy disk checks (sagittal zero, peak location, depth)

    Designed to run on a large machine (300+ cores, TB-scale RAM).
    Run with:  julia -t auto test_production_psf.jl

    To customize:  julia -t auto -e '
        include("test_production_psf.jl")
        run_production_psf(R_um=1000.0, alpha_deg=15.0, NA=0.3)
    '
"""

include("../cyffp.jl")
using .CyFFP
using FFTW
using SpecialFunctions: besselj

# Try to load Plots; skip plotting if unavailable
const HAS_PLOTS = try
    using Plots
    true
catch
    @warn "Plots not available — install with ] add Plots. Tests will run but no figures."
    false
end


"""
    run_production_psf(; kwargs...) -> NamedTuple

Run the full 6-step scalar pipeline for an ideal oblique lens and
validate the PSF against the Airy disk prediction.

# Keyword arguments
- `R_um       = 2000.0`  : aperture radius in μm
- `lambda_um  = 0.5`     : wavelength in μm
- `NA         = 0.4`     : numerical aperture (determines focal length)
- `alpha_deg  = 30.0`    : oblique incidence angle in degrees
- `Nr         = 0`       : radial grid size (0 = auto-select power of 2)
- `L_max      = 15`      : local mode truncation for Steps 4-6
- `N_psi      = 64`      : azimuthal output points in Step 6
- `rho_max_lambda = 5.0` : PSF plot extent in units of λ
- `Nxy        = 301`     : Cartesian heatmap grid size (Nxy × Nxy)
- `prefix     = "prod"` : filename prefix for saved plots
- `do_plots   = true`    : whether to generate and save plots
"""
function run_production_psf(;
        R_um::Float64       = 2000.0,
        lambda_um::Float64  = 0.5,
        NA::Float64         = 0.4,
        alpha_deg::Float64  = 30.0,
        Nr::Int             = 0,
        L_max::Int          = 15,
        N_psi::Int          = 64,
        rho_max_lambda::Float64 = 5.0,
        Nxy::Int            = 301,
        prefix::String      = "prod",
        do_plots::Bool      = true,
    )

    k     = 2π / lambda_um
    R     = R_um
    f_val = R * sqrt(1/NA^2 - 1)
    alpha = deg2rad(alpha_deg)
    x0    = f_val * tan(alpha)
    kx    = k * sin(alpha)
    M_max = ceil(Int, kx * R) + 20

    NA_eff_sag  = R * cos(alpha) / sqrt((R * cos(alpha))^2 + f_val^2)
    rho_airy    = 0.61 * lambda_um / NA_eff_sag

    println("="^70)
    println("Production PSF: ideal oblique lens")
    println("="^70)
    println("  R        = $R_um μm")
    println("  λ        = $lambda_um μm")
    println("  NA       = $NA")
    println("  f        = $(round(f_val, digits=1)) μm")
    println("  α        = $(alpha_deg)°")
    println("  x₀       = $(round(x0, digits=1)) μm  (x₀/R = $(round(x0/R, digits=2)))")
    println("  M_max    = $M_max")
    println("  L_max    = $L_max")
    println("  NA_eff(sag) = $(round(NA_eff_sag, digits=4))")
    println("  ρ_Airy(sag) = $(round(rho_airy, digits=4)) μm = $(round(rho_airy/lambda_um, digits=3))λ")

    # ─── Grid setup ──────────────────────────────────────────
    # Constraints: r_min ≤ λ/(2π),  R × dln ≤ λ/2
    r_min = lambda_um / (20π)
    r_max = 50 * R
    if Nr == 0
        # Auto-select: Nr must satisfy dln ≤ λ/(2R)
        dln_max = lambda_um / (2R)
        Nr_min  = ceil(Int, log(r_max / r_min) / dln_max)
        Nr = 1 << ceil(Int, log2(Nr_min))   # round up to power of 2
    end

    r   = collect(exp.(range(log(r_min), log(r_max), length=Nr)))
    dln = log(r[2] / r[1])
    kr  = exp.(log(1.0 / r[end]) .+ dln .* (0:Nr-1))

    println("  Nr       = $Nr")
    println("  dln      = $(round(dln, sigdigits=4))")
    println("  Δr(R)    = $(round(R*dln, sigdigits=3)) μm  (need < $(lambda_um/2) μm)")
    @assert R * dln < lambda_um / 2 "Grid too coarse at aperture edge"
    @assert 1/r[1] > k "kr_max must exceed k"

    # Memory estimate
    mem_am_GB   = Nr * (M_max + 1) * 16 / 1e9
    mem_at_GB   = Nr * (2M_max + 1) * 16 / 1e9
    println("  Memory: a_m ≈ $(round(mem_am_GB, digits=1)) GB, a_tilde ≈ $(round(mem_at_GB, digits=1)) GB")

    # ─── Step 1: build modes via per-r FFT ────────────────────
    # For large M_max, the 2D field array (Nr × Nθ) would be huge.
    # Instead, extract modes one r at a time via FFT over θ.
    println("\n--- Step 1: extracting modes via per-r FFT ---")
    Ntheta = max(2M_max + 1, 4096)
    Ntheta = 1 << ceil(Int, log2(Ntheta))
    m_pos  = collect(0:M_max)

    u_m    = zeros(ComplexF64, Nr, M_max + 1)
    u_row  = Vector{ComplexF64}(undef, Ntheta)

    t1 = @elapsed begin
        for j in 1:Nr
            rv = r[j]
            if rv > R; continue; end
            d0 = f_val   # for phase reference
            for it in 1:Ntheta
                th = 2π * (it - 1) / Ntheta
                d  = sqrt((rv * cos(th) - x0)^2 + rv^2 * sin(th)^2 + f_val^2)
                u_row[it] = exp(-im * k * (d - d0))
            end
            fft!(u_row)
            u_row ./= Ntheta
            for (idx, m) in enumerate(m_pos)
                u_m[j, idx] = u_row[m + 1]
            end
        end
    end
    println("  Modes built: $(Nr) × $(M_max+1), Nθ=$Ntheta  ($(round(t1, digits=1))s)")

    # ─── Step 2: scalar spectral coefficients ─────────────────
    println("\n--- Step 2: compute_scalar_coeffs ---")
    t2 = @elapsed begin
        a_m, kr_grid = compute_scalar_coeffs(u_m, m_pos, r)
    end
    println("  $(M_max+1) modes × Nr=$Nr  ($(round(t2, digits=1))s)")

    # Parseval spot check
    for m in [0, M_max÷2, M_max]
        idx = m + 1
        sp = dln * sum(abs2(u_m[j, idx]) * r[j]^2 for j in 1:Nr)
        sk = dln * sum(abs2(a_m[j, idx]) * kr_grid[j]^2 for j in 1:Nr)
        ratio = sp > 1e-30 ? sk / sp : NaN
        println("    Parseval m=$m: $(round(ratio, sigdigits=8))")
    end

    # Free u_m
    u_m = nothing; GC.gc()

    # ─── Step 3: propagation + symmetry ───────────────────────
    println("\n--- Step 3: propagate_scalar ---")
    t3 = @elapsed begin
        a_tilde, m_full = propagate_scalar(a_m, m_pos, collect(kr_grid), k, f_val)
    end
    println("  $(length(m_full)) modes  ($(round(t3, digits=1))s)")
    a_m = nothing; GC.gc()

    # ─── Step 4: Graf shift ───────────────────────────────────
    println("\n--- Step 4: graf_shift ---")
    t4 = @elapsed begin
        B = graf_shift(a_tilde, m_full, collect(kr_grid), x0, L_max; k=k)
    end
    println("  L_max=$L_max, x₀=$(round(x0, digits=1)) μm  ($(round(t4, digits=1))s)")
    a_tilde = nothing; GC.gc()

    # ─── Step 5: inverse Hankel ───────────────────────────────
    println("\n--- Step 5: inverse_hankel ---")
    t5 = @elapsed begin
        b, rho = inverse_hankel(B, L_max, collect(kr_grid))
    end
    println("  $(2L_max+1) modes  ($(round(t5, digits=1))s)")

    # ─── Step 6: angular synthesis ────────────────────────────
    println("\n--- Step 6: angular_synthesis ---")
    t6 = @elapsed begin
        u_psf, psi = angular_synthesis(b, L_max, N_psi)
    end
    println("  N_ψ=$N_psi  ($(round(t6, digits=1))s)")

    I_psf = abs2.(u_psf)

    t_total = t1 + t2 + t3 + t4 + t5 + t6
    println("\n  Total pipeline: $(round(t_total, digits=1))s")
    println("    Step 1 (modes):     $(round(t1, digits=1))s")
    println("    Step 2 (FFTLog):    $(round(t2, digits=1))s")
    println("    Step 3 (propagate): $(round(t3, digits=1))s")
    println("    Step 4 (Graf):      $(round(t4, digits=1))s")
    println("    Step 5 (inv HT):    $(round(t5, digits=1))s")
    println("    Step 6 (synth):     $(round(t6, digits=1))s")

    # ─── PSF analysis ────────────────────────────────────────
    println("\n" * "="^70)
    println("PSF analysis")
    println("="^70)

    # Radial profiles
    I_tang = I_psf[:, 1]                           # ψ = 0
    I_sag  = I_psf[:, N_psi÷4 + 1]                # ψ = π/2
    I_avg  = [sum(I_psf[ir, :]) / N_psi for ir in 1:Nr]
    I_peak = max(maximum(I_tang), maximum(I_sag), maximum(I_avg))

    # Peak location
    peak_ir  = argmax(I_avg)
    peak_rho = rho[peak_ir]
    println("  PSF peak at ρ = $(round(peak_rho / lambda_um, digits=3))λ = $(round(peak_rho, digits=4)) μm")

    # Find first zero in sagittal and tangential directions
    function find_first_zero(I_prof, rho_grid, I_pk, rho_start)
        I_n = I_prof ./ I_pk
        for ir in 2:length(I_n)-1
            if rho_grid[ir] > rho_start && I_n[ir] < I_n[ir-1] && I_n[ir] < I_n[ir+1]
                return rho_grid[ir], I_n[ir]
            end
        end
        return NaN, NaN
    end

    # Start searching after 0.5 × rho_airy to skip FFTLog boundary ripple
    rho_search_start = 0.5 * rho_airy
    z_sag, m_sag = find_first_zero(I_sag, rho, I_peak, rho_search_start)
    z_tang, m_tang = find_first_zero(I_tang, rho, I_peak, rho_search_start)

    ae_sag  = isnan(z_sag)  ? NaN : abs(z_sag - rho_airy) / rho_airy
    ae_tang = isnan(z_tang) ? NaN : abs(z_tang - rho_airy) / rho_airy

    println("\n  Sagittal (ψ=π/2):")
    println("    First zero = $(round(z_sag / lambda_um, digits=3))λ = $(round(z_sag, digits=4)) μm")
    println("    Predicted   = $(round(rho_airy / lambda_um, digits=3))λ = $(round(rho_airy, digits=4)) μm")
    println("    Error       = $(round(100*ae_sag, digits=1))%")
    println("    Min I/I_peak = $(round(m_sag, sigdigits=3))")

    println("\n  Tangential (ψ=0):")
    println("    First zero = $(round(z_tang / lambda_um, digits=3))λ = $(round(z_tang, digits=4)) μm")
    println("    Error vs sagittal prediction = $(round(100*ae_tang, digits=1))%")
    println("    Min I/I_peak = $(round(m_tang, sigdigits=3))")

    if alpha_deg >= 10
        println("    (tangential broadening expected at α=$(alpha_deg)°)")
    end

    # ─── Assertions ──────────────────────────────────────────
    println("\n--- Assertions ---")

    @assert peak_rho < 2.0 * lambda_um "PSF peak not near center"
    println("  Peak at center ✓")

    @assert !isnan(ae_sag) "No sagittal first zero found"
    @assert ae_sag < 0.05 "Sagittal Airy error > 5%"
    println("  Sagittal Airy zero within $(round(100*ae_sag, digits=1))% ✓")

    @assert !isnan(m_sag) && m_sag < 0.001 "Sagittal minimum not deep enough"
    println("  Sagittal dark ring depth $(round(m_sag, sigdigits=2)) ✓")

    # No NaN/Inf
    @assert !any(isnan, u_psf) "NaN in PSF"
    @assert !any(isinf, u_psf) "Inf in PSF"
    println("  No NaN/Inf ✓")

    println("\n  All PSF checks PASSED ✓")

    # ─── Power concentration check ───────────────────────────
    # Near-field power: ∫|t(r)|² 2π r dr = π R² (since |t|=1)
    P_near = π * R^2

    # Far-field power on polar grid: ∫∫ I(ρ,ψ) ρ dρ dψ
    # I_psf[ir, ip] = |u(ρ,ψ)|², on log-spaced ρ and uniform ψ
    dpsi_val = 2π / N_psi
    P_far = 0.0
    for ip in 1:N_psi, ir in 1:Nr
        P_far += I_psf[ir, ip] * rho[ir]^2 * dln * dpsi_val
    end

    # Power within one Airy radius (focal spot)
    P_airy = 0.0
    for ip in 1:N_psi, ir in 1:Nr
        if rho[ir] <= rho_airy
            P_airy += I_psf[ir, ip] * rho[ir]^2 * dln * dpsi_val
        end
    end

    println("\n--- Power budget ---")
    println("  P_nearfield     = $(round(P_near, sigdigits=6)) μm²  (π R²)")
    println("  P_farfield      = $(round(P_far, sigdigits=6)) μm²  (∫ I ρ dρ dψ)")
    println("  P_far / P_near  = $(round(P_far / P_near, sigdigits=6))  (Parseval check)")
    println("  P_airy          = $(round(P_airy, sigdigits=6)) μm²  (ρ ≤ ρ_Airy)")
    println("  P_airy / P_near = $(round(P_airy / P_near, sigdigits=6))  (concentration efficiency)")
    println("  Airy theory     ≈ 0.84  (fraction within first zero)")

    # ─── Plots ───────────────────────────────────────────────
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
                  title="PSF: R=$(R_um)μm, α=$(alpha_deg)°, NA=$(NA)",
                  xlims=(0, rho_max_lambda), ylims=(-0.05, 1.05),
                  legend=:topright, size=(750, 480))
        plot!(p1, rho ./ lambda_um, I_tang ./ I_peak,
              label="ψ=0° (tangential)", lw=1.5, ls=:dash)
        plot!(p1, rho ./ lambda_um, I_sag ./ I_peak,
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
        x_cart = range(-xy_max, xy_max, length=Nxy) ./ lambda_um  # in λ units
        y_cart = range(-xy_max, xy_max, length=Nxy) ./ lambda_um

        I_cart = zeros(Nxy, Nxy)
        dpsi   = psi[2] - psi[1]

        for (ix, xv_lam) in enumerate(x_cart)
            for (iy, yv_lam) in enumerate(y_cart)
                xv = xv_lam * lambda_um
                yv = yv_lam * lambda_um
                rv = sqrt(xv^2 + yv^2)
                if rv < rho[1] || rv > rho[end]; continue; end

                pv = atan(yv, xv)
                if pv < 0; pv += 2π; end

                # Log-interpolate in ρ
                lr  = log(rv); lr0 = log(rho[1])
                idx_r = (lr - lr0) / dln + 1.0
                j0 = clamp(floor(Int, idx_r), 1, Nr - 1)
                wr = idx_r - j0

                # Linear interpolate in ψ
                idx_p = pv / dpsi + 1.0
                p0 = clamp(floor(Int, idx_p), 1, N_psi)
                p1_idx = p0 == N_psi ? 1 : p0 + 1
                wp = idx_p - p0

                I_cart[iy, ix] = (
                    (1-wr)*(1-wp)*I_psf[j0, p0]     + wr*(1-wp)*I_psf[j0+1, p0] +
                    (1-wr)*wp    *I_psf[j0, p1_idx]  + wr*wp    *I_psf[j0+1, p1_idx]
                )
            end
        end
        I_cart ./= maximum(I_cart)

        p2 = heatmap(collect(x_cart), collect(y_cart), I_cart;
                     xlabel="x / λ", ylabel="y / λ",
                     color=:inferno, clims=(0, 1),
                     title="PSF: R=$(R_um)μm, α=$(alpha_deg)°, NA=$(NA)",
                     aspect_ratio=:equal, size=(600, 550))
        # Airy circle (sagittal prediction)
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
                     title="PSF (log₁₀): R=$(R_um)μm, α=$(alpha_deg)°",
                     aspect_ratio=:equal, size=(600, 550))
        plot!(p3, rl .* cos.(th_c), rl .* sin.(th_c),
              label="Airy zero (sag)", color=:white, ls=:dash, lw=1.5)

        fname3 = "$(prefix)_psf_2d_log.png"
        savefig(p3, fname3)
        println("  Saved: $fname3")
    end

    println("\n" * "="^70)
    println("Production PSF test complete.")
    println("="^70)

    return (
        R_um       = R_um,
        lambda_um  = lambda_um,
        NA         = NA,
        alpha_deg  = alpha_deg,
        M_max      = M_max,
        Nr         = Nr,
        rho_airy   = rho_airy,
        sag_zero   = z_sag,
        tang_zero  = z_tang,
        sag_err    = ae_sag,
        peak_rho   = peak_rho,
        t_total    = t_total,
    )
end


"""
    run_neumann_vs_standard(; kwargs...) -> NamedTuple

Compare the Neumann shift path against the standard path for an LPA
near field (normal-incidence lens + oblique plane wave tilt).

The LPA PSF is NOT an ideal Airy disk — it has coma.  This test only
checks consistency between the two computational paths, not against
an analytical prediction.

# Keyword arguments
- `R_um       = 2000.0`  : aperture radius in μm
- `lambda_um  = 0.5`     : wavelength in μm
- `NA         = 0.4`     : numerical aperture
- `alpha_deg  = 30.0`    : oblique angle in degrees
- `Nr         = 0`       : radial grid size (0 = auto)
- `L_max      = 15`      : local mode truncation
- `N_psi      = 64`      : azimuthal output points
- `N_rho_test = 15`      : number of ρ sample points for PSF comparison
- `rho_max_lambda = 5.0` : PSF comparison extent in λ
- `prefix     = "prod_neumann"` : filename prefix for plots
- `do_plots   = true`    : whether to generate plots
"""
function run_neumann_vs_standard(;
        R_um::Float64       = 2000.0,
        lambda_um::Float64  = 0.5,
        NA::Float64         = 0.4,
        alpha_deg::Float64  = 30.0,
        Nr::Int             = 0,
        L_max::Int          = 15,
        N_psi::Int          = 64,
        N_rho_test::Int     = 15,
        rho_max_lambda::Float64 = 5.0,
        prefix::String      = "prod_neumann",
        do_plots::Bool      = true,
    )

    k     = 2π / lambda_um
    R     = R_um
    f_val = R * sqrt(1/NA^2 - 1)
    alpha = deg2rad(alpha_deg)
    x0    = f_val * tan(alpha)
    kx    = k * sin(alpha)
    M_max = ceil(Int, kx * R) + 20
    m_pos = collect(0:M_max)

    println("="^70)
    println("Neumann vs Standard: LPA field (coma-aberrated)")
    println("="^70)
    println("  R = $R_um μm,  λ = $lambda_um μm,  NA = $NA")
    println("  α = $(alpha_deg)°,  x₀ = $(round(x0, digits=1)) μm")
    println("  M_max = $M_max,  L_max = $L_max")

    # ─── Grid setup (same auto-selection as run_production_psf) ──
    r_min = lambda_um / (20π)
    r_max = 50 * R
    if Nr == 0
        dln_max = lambda_um / (2R)
        Nr_min  = ceil(Int, log(r_max / r_min) / dln_max)
        Nr = 1 << ceil(Int, log2(Nr_min))
    end

    r   = collect(exp.(range(log(r_min), log(r_max), length=Nr)))
    dln = log(r[2] / r[1])
    kr  = exp.(log(1.0 / r[end]) .+ dln .* (0:Nr-1))

    println("  Nr = $Nr,  dln = $(round(dln, sigdigits=4)),  Δr(R) = $(round(R*dln, sigdigits=3)) μm")
    @assert R * dln < lambda_um / 2 "Grid too coarse"

    # ─── LPA near field: t(r) = normal-incidence lens phase ──
    t_lens(rv) = exp(-im * k * (sqrt(rv^2 + f_val^2) - f_val)) * (rv <= R ? 1.0 : 0.0)
    t_r = ComplexF64[t_lens(rv) for rv in r]

    # ═══════════════════════════════════════════════════════════
    # Standard path: per-r FFT → compute_scalar_coeffs
    # ═══════════════════════════════════════════════════════════
    println("\n--- Standard path ---")
    Ntheta = max(2M_max + 1, 4096)
    Ntheta = 1 << ceil(Int, log2(Ntheta))

    t_std = @elapsed begin
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
        u_m = nothing; GC.gc()

        at_std, mf_std = propagate_scalar(a_std, m_pos, collect(kr), k, f_val)
        a_std = nothing; GC.gc()

        B_std = graf_shift(at_std, mf_std, collect(kr), x0, L_max; k=k)
        at_std = nothing; GC.gc()
    end
    println("  Steps 1-4: $(round(t_std, digits=1))s")

    # ═══════════════════════════════════════════════════════════
    # Neumann path: neumann_shift_coeffs
    # ═══════════════════════════════════════════════════════════
    println("\n--- Neumann path ---")
    t_neu = @elapsed begin
        a_neu, _, _ = neumann_shift_coeffs(t_r, r, k, alpha, M_max)

        at_neu, mf_neu = propagate_scalar(a_neu, m_pos, collect(kr), k, f_val)
        a_neu = nothing; GC.gc()

        B_neu = graf_shift(at_neu, mf_neu, collect(kr), x0, L_max; k=k)
        at_neu = nothing; GC.gc()
    end
    println("  Steps A-B + 3-4: $(round(t_neu, digits=1))s")
    println("  Speedup: $(round(t_std / t_neu, digits=1))×")

    # ═══════════════════════════════════════════════════════════
    # PSF comparison via brute-force Riemann sum
    # ═══════════════════════════════════════════════════════════
    println("\n--- PSF comparison (brute-force Riemann sum at $(N_rho_test) ρ points) ---")
    prop_kr = findall(kr .< k)

    rho_test = collect(range(0.001, rho_max_lambda, length=N_rho_test)) .* lambda_um
    psf_std  = zeros(ComplexF64, N_rho_test)
    psf_neu  = zeros(ComplexF64, N_rho_test)

    for (ir, rho_val) in enumerate(rho_test)
        for (li, l) in enumerate(-L_max:L_max)
            bl_s = dln * sum(B_std[j, li] * besselj(l, kr[j] * rho_val) * kr[j]^2
                             for j in prop_kr)
            bl_n = dln * sum(B_neu[j, li] * besselj(l, kr[j] * rho_val) * kr[j]^2
                             for j in prop_kr)
            psf_std[ir] += bl_s
            psf_neu[ir] += bl_n
        end
    end

    I_std = abs2.(psf_std)
    I_neu = abs2.(psf_neu)
    peak_std = maximum(I_std)
    peak_neu = maximum(I_neu)

    # Peak-normalized comparison
    max_psf_err = maximum(abs.(I_std .- I_neu)) / peak_std

    # Energy-weighted RMS
    rms_psf = sqrt(sum(abs2.(psf_std .- psf_neu)) / sum(abs2.(psf_std)))

    println("  Peak intensity: std=$(round(peak_std, sigdigits=4))  neu=$(round(peak_neu, sigdigits=4))")
    println("  Peak ratio:     $(round(peak_neu / peak_std, sigdigits=5))")
    println("  Max |I_std - I_neu| / I_peak: $(round(100*max_psf_err, digits=2))%")
    println("  Field RMS difference: $(round(100*rms_psf, digits=2))%")

    println("\n  Per-point comparison:")
    for ir in 1:N_rho_test
        rho_lam = rho_test[ir] / lambda_um
        err = abs(I_std[ir] - I_neu[ir]) / peak_std
        println("    ρ=$(lpad(round(rho_lam, digits=2), 5))λ:  std=$(lpad(round(I_std[ir]/peak_std, sigdigits=3), 6))  neu=$(lpad(round(I_neu[ir]/peak_std, sigdigits=3), 6))  Δ/peak=$(round(100*err, digits=2))%")
    end

    # ─── Assertions ──────────────────────────────────────────
    # The Neumann path's linear interpolation degrades at large M_max.
    # For M_max ≲ 1600: expect <2% PSF error (validated).
    # For M_max ≳ 10000: expect 5-10% PSF error (known limitation).
    # Thresholds are set accordingly.
    tol_psf  = M_max < 2000 ? 0.03 : 0.15
    tol_rms  = M_max < 2000 ? 0.03 : 0.10
    tol_peak = M_max < 2000 ? 0.03 : 0.15

    println("\n--- Assertions (M_max=$M_max, tolerance=$(round(100*tol_psf, digits=0))%) ---")

    @assert max_psf_err < tol_psf "PSF mismatch $(round(100*max_psf_err,digits=1))% exceeds $(round(100*tol_psf,digits=0))%"
    println("  PSF agreement: $(round(100*max_psf_err, digits=2))% of peak (limit $(round(100*tol_psf,digits=0))%) ✓")

    @assert rms_psf < tol_rms "RMS field difference $(round(100*rms_psf,digits=1))% exceeds $(round(100*tol_rms,digits=0))%"
    println("  RMS field: $(round(100*rms_psf, digits=2))% (limit $(round(100*tol_rms,digits=0))%) ✓")

    @assert abs(peak_neu / peak_std - 1) < tol_peak "Peak ratio $(round(peak_neu/peak_std,sigdigits=3)) outside $(round(100*tol_peak,digits=0))%"
    println("  Peak ratio: $(round(peak_neu / peak_std, sigdigits=4)) (limit ±$(round(100*tol_peak,digits=0))%) ✓")

    if M_max > 2000
        println("\n  NOTE: At M_max=$M_max, the Neumann path's linear interpolation")
        println("  has reduced accuracy (~$(round(100*max_psf_err,digits=0))%). The standard path is")
        println("  recommended for production. See tex §Neumann for details.")
    end

    println("\n  All Neumann consistency checks PASSED ✓")

    # ─── Plots ───────────────────────────────────────────────
    if do_plots && HAS_PLOTS
        println("\n--- Generating comparison plots ---")

        # 1D: both PSFs overlaid
        p1 = plot(rho_test ./ lambda_um, I_std ./ peak_std,
                  label="Standard", lw=2, marker=:circle, ms=3,
                  xlabel="ρ / λ", ylabel="I / I_peak",
                  title="LPA PSF: R=$(R_um)μm, α=$(alpha_deg)°, NA=$(NA)",
                  ylims=(-0.05, 1.05), legend=:topright, size=(750, 480))
        plot!(p1, rho_test ./ lambda_um, I_neu ./ peak_std,
              label="Neumann", lw=2, marker=:diamond, ms=3, ls=:dash)

        fname1 = "$(prefix)_psf_1d.png"
        savefig(p1, fname1)
        println("  Saved: $fname1")

        # 1D: difference
        p2 = plot(rho_test ./ lambda_um,
                  (I_std .- I_neu) ./ peak_std .* 100,
                  label="(Standard - Neumann) / peak",
                  lw=2, marker=:circle, ms=3,
                  xlabel="ρ / λ", ylabel="ΔI / I_peak  (%)",
                  title="PSF difference: R=$(R_um)μm, α=$(alpha_deg)°",
                  legend=:topright, size=(750, 400))
        hline!(p2, [0], color=:gray, ls=:dot, label="")

        fname2 = "$(prefix)_psf_diff.png"
        savefig(p2, fname2)
        println("  Saved: $fname2")
    end

    B_std = nothing; B_neu = nothing; GC.gc()

    println("\n" * "="^70)
    println("Neumann vs Standard comparison complete.")
    println("="^70)

    return (
        R_um       = R_um,
        lambda_um  = lambda_um,
        NA         = NA,
        alpha_deg  = alpha_deg,
        M_max      = M_max,
        Nr         = Nr,
        max_psf_err = max_psf_err,
        rms_psf    = rms_psf,
        peak_ratio = peak_neu / peak_std,
        t_std      = t_std,
        t_neu      = t_neu,
        speedup    = t_std / t_neu,
    )
end


# Default invocation when run as a script
if abspath(PROGRAM_FILE) == @__FILE__
    run_production_psf()
    #GC.gc()
    #run_neumann_vs_standard()
end
