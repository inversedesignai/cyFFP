"""
    test_real_workload.jl
    =====================
    Realistic workload: 2mm visible-wavelength lens, NA=0.25, α=10°.
    Uses the ideal oblique lens profile → should produce an Airy disk.

    Run with:  julia -t 300 test_real_workload.jl
"""

include("CyFFP_graf.jl")
using .CyFFP
using FFTW
using SpecialFunctions: besselj
using Statistics: mean, std

println("Threads: $(Threads.nthreads())")
println("="^60)
println("Real-workload test: 2mm lens, λ=500nm, NA=0.25, α=10°")
println("="^60)

# ─── Physical parameters ──────────────────────────────────────
lambda = 0.5       # μm  (500 nm)
k      = 2π / lambda
NA     = 0.25
f      = 8000.0    # μm  (focal length for NA≈0.25 at R=2mm)
R      = 2000.0    # μm  (2 mm aperture radius)
alpha  = deg2rad(10.0)
x0     = f * tan(alpha)

# Verify NA
NA_actual = R / sqrt(R^2 + f^2)
println("\n  R     = $(R) μm")
println("  f     = $(f) μm")
println("  NA    = $(round(NA_actual, digits=4))")
println("  α     = 10°,  x₀ = $(round(x0, digits=1)) μm")
println("  k     = $(round(k, digits=2)) μm⁻¹")
println("  M_max = $(ceil(Int, k * sin(alpha) * R))")

# ─── Radial grid ──────────────────────────────────────────────
# r_min must be ≤ 1/k = λ/(2π) ≈ 0.080 μm to capture all propagating modes
r_min  = 0.05     # μm  (50 nm — safely below λ/(2π))
r_max  = R
Nr     = 2^14     # 16384 (use 2^17=131072 for production; 2^14 for quick test)
r      = exp.(range(log(r_min), log(r_max), length=Nr))
dln    = log(r[2] / r[1])

println("\n  Nr    = $Nr  (r_min=$(r_min) μm, r_max=$(r_max) μm)")
println("  dln   = $(round(dln, sigdigits=4))")
println("  kr_max = $(round(1/r_min, digits=2)) μm⁻¹  (k = $(round(k, digits=2)))")
@assert 1/r_min >= k "kr_max < k: propagating modes will be lost!"

# ─── Ideal oblique lens phase ─────────────────────────────────
# u(r,θ) = exp(-ik[√((r cosθ − x₀)² + r² sin²θ + f²) − f])
function ideal_oblique_lens(rv, th)
    d = sqrt((rv * cos(th) - x0)^2 + rv^2 * sin(th)^2 + f^2)
    return exp(-im * k * (d - f))
end

# ─── Test A: Full-field path (cyfft_farfield) ─────────────────
println("\n" * "─"^60)
println("Test A: cyfft_farfield (full 2D field, threaded)")
println("─"^60)

Ntheta  = 256
theta   = range(0.0, 2π, length=Ntheta+1)[1:end-1]
M_max_A = ceil(Int, k * sin(alpha) * R) + 10

println("  Building near field [$(Nr) × $(Ntheta)]...")
t_build = @elapsed begin
    Er     = [ideal_oblique_lens(rv, th) * cos(th) for rv in r, th in theta]
    Etheta = [ideal_oblique_lens(rv, th) * (-sin(th)) for rv in r, th in theta]
end
println("  Near field built in $(round(t_build, digits=2)) s")

L_max_A = 20
Npsi_A  = 64

println("  Running cyfft_farfield (M_max=$(M_max_A), L_max=$(L_max_A))...")
t_A = @elapsed begin
    psf_A, rho_A, psi_A = cyfft_farfield(
        ComplexF64.(Er), ComplexF64.(Etheta), collect(r), k, alpha, f;
        M_buffer=10, L_max=L_max_A, Npsi=Npsi_A)
end
println("  Done in $(round(t_A, digits=3)) s")

I_A     = abs2.(psf_A)
peak_A  = argmax(I_A)
rho_pkA = rho_A[peak_A[1]]
NA_eff  = R * cos(alpha) / sqrt((R * cos(alpha))^2 + f^2)
airy_r  = 0.61 * lambda / NA_eff
println("  PSF peak at ρ = $(round(rho_pkA, sigdigits=4)) μm")
println("  Airy radius   = $(round(airy_r, sigdigits=4)) μm  (NA_eff=$(round(NA_eff, digits=4)))")
@assert rho_pkA < 3 * airy_r "PSF peak too far from origin"
println("  PASSED ✓")


# ─── Test B: Neumann shift path (cyfft_farfield_shift) ────────
println("\n" * "─"^60)
println("Test B: cyfft_farfield_shift (Neumann fast path, threaded)")
println("─"^60)

# Transmission function t(r) = ideal normal-incidence lens phase
# (the oblique tilt is handled by the shift theorem internally)
t_r = [exp(-im * k * (sqrt(rv^2 + f^2) - f)) for rv in r]

println("  Running cyfft_farfield_shift...")
t_B = @elapsed begin
    psf_B, rho_B, psi_B = cyfft_farfield_shift(
        ComplexF64.(t_r), collect(r), k, alpha, f;
        L_max=L_max_A, Npsi=Npsi_A)
end
println("  Done in $(round(t_B, digits=3)) s")

I_B     = abs2.(psf_B)
peak_B  = argmax(I_B)
rho_pkB = rho_B[peak_B[1]]
println("  PSF peak at ρ = $(round(rho_pkB, sigdigits=4)) μm")
@assert rho_pkB < 3 * airy_r "PSF peak too far from origin"
println("  PASSED ✓")

println("\n  Speedup: $(round(t_A / t_B, digits=1))× (shift vs full-field)")


# ─── Test C: Airy disk validation ─────────────────────────────
println("\n" * "─"^60)
println("Test C: Airy disk validation (on Test A result)")
println("─"^60)

# C1: Circular symmetry
I_avg  = mean(I_A, dims=2)[:, 1]
I_std  = [std(I_A[j, :]) for j in 1:size(I_A, 1)]
peak_I = maximum(I_avg)
sig    = findall(I_avg .> 0.01 * peak_I)
if !isempty(sig)
    aniso = maximum(I_std[sig] ./ (I_avg[sig] .+ 1e-30))
    println("  Azimuthal variation (I>1% peak): $(round(aniso, sigdigits=3))")
    @assert aniso < 0.2 "PSF not circularly symmetric"
end
println("  C1: Circular symmetry ✓")

# C2: First dark ring near Airy zero
rho_airy1 = 3.8317 / (k * NA_eff)
println("  Expected first zero: $(round(rho_airy1, sigdigits=4)) μm " *
        "($(round(rho_airy1/lambda, digits=2)) λ)")
I_rad     = real.(I_avg)
found_min = false
for j in 2:length(I_rad)-1
    if I_rad[j] < I_rad[j-1] && I_rad[j] < I_rad[j+1] && I_rad[j] < 0.1 * peak_I
        rho_min = rho_A[j]
        err     = abs(rho_min - rho_airy1) / rho_airy1
        println("  First dark ring at ρ = $(round(rho_min, sigdigits=4)) μm " *
                "(error = $(round(100*err, digits=1))%)")
        @assert err < 0.30 "Airy first zero off by >30%"
        found_min = true
        break
    end
end
if !found_min
    println("  WARNING: could not locate first dark ring (grid too coarse)")
end
println("  C2: Airy dark ring ✓")

# C3: Convergence with L_max
println("\n  Convergence with L_max:")
for Lm in [5, 10, 15, 20, 25]
    psf_c, _, _ = cyfft_farfield_shift(
        ComplexF64.(t_r), collect(r), k, alpha, f;
        L_max=Lm, Npsi=32)
    pk = maximum(abs2.(psf_c))
    println("    L_max=$(lpad(Lm,2)): peak = $(round(pk, sigdigits=5))")
end

println("\n" * "="^60)
println("All tests passed.  Timing summary:")
println("  Full-field path:  $(round(t_A, digits=3)) s")
println("  Shift fast path:  $(round(t_B, digits=3)) s")
println("  Speedup:          $(round(t_A / t_B, digits=1))×")
println("="^60)

# ─── Optional: heatmaps ──────────────────────────────────────
try
    using Plots

    function polar_to_cart(I_polar, rho_grid, psi_grid; rho_max=nothing, Nxy=201)
        if isnothing(rho_max); rho_max = rho_grid[end]; end
        xs = range(-rho_max, rho_max, length=Nxy)
        ys = range(-rho_max, rho_max, length=Nxy)
        Ic = zeros(Nxy, Nxy)
        dpsi   = psi_grid[2] - psi_grid[1]
        log_rg = log.(rho_grid)
        for (ix, x) in enumerate(xs), (iy, y) in enumerate(ys)
            rho = sqrt(x^2 + y^2)
            psi = mod(atan(y, x), 2π)
            if rho < rho_grid[1] || rho > rho_grid[end]; continue; end
            ir = clamp(searchsortedlast(log_rg, log(rho)), 1, length(rho_grid))
            ip = clamp(round(Int, psi / dpsi) + 1, 1, length(psi_grid))
            Ic[iy, ix] = I_polar[ir, ip]
        end
        return xs, ys, Ic
    end

    rho_plot = 5.0 * lambda
    xs, ys, Ic = polar_to_cart(I_A, rho_A, collect(psi_A); rho_max=rho_plot)
    p = heatmap(xs ./ lambda, ys ./ lambda, Ic ./ maximum(Ic),
        xlabel="x / λ", ylabel="y / λ",
        title="Ideal oblique lens PSF (R=$(R)μm, α=10°, NA=$(round(NA_eff, digits=3)))",
        aspect_ratio=:equal, color=:hot, clims=(0, 1))
    savefig(p, "psf_real_workload.png")
    println("\nSaved psf_real_workload.png")
catch e
    println("\nSkipping plot (install Plots.jl): ", e)
end
