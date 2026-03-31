# optimize_psf.jl — Phase optimization to maximize focal-spot PSF intensity
#
# Maximizes sum of I_raw within a 10-pixel-radius circle around the geometric
# focal spot of a 2mm lens at 30° oblique incidence, NA=0.4.
#
# Uses NLopt's CCSA (LD_CCSAQ) with adjoint gradients.
# Saves φ data and PSF plot every 10 iterations.
#
# Usage:
#   julia -t auto optimize_psf.jl uniform [val]   # all φ = val (default 0)
#   julia -t auto optimize_psf.jl ideal           # ideal oblique lens phase (default)
#   julia -t auto optimize_psf.jl quadratic       # quadratic phase φ(r) = -(k/2f)r²
#   julia -t auto optimize_psf.jl random [seed]   # random φ ∈ [0,2π], default seed=42
#
# Gradient sanity check (no optimization, just FD vs adjoint):
#   julia -t auto optimize_psf.jl gradcheck <init> [arg]
#   e.g.  julia -t auto optimize_psf.jl gradcheck ideal
#         julia -t auto optimize_psf.jl gradcheck random 99

using Printf
include("cyffp.jl")
using .CyFFP
using LinearAlgebra: dot, norm
using Random
using NLopt
using Plots
using DelimitedFiles

# ─── Problem parameters ───
const pitch_um  = 0.3
const R_um      = 2000.0
const N_cells   = round(Int, R_um / pitch_um)   # 6667
const lambda_um = 0.5
const alpha_deg = 30.0
const NA_val    = 0.4
const Nxy       = 201

println("═══ PSF phase optimization (R=2 mm, α=30°, NA=0.4) ═══")
println("N_cells = $N_cells,  Nxy = $Nxy,  threads = $(Threads.nthreads())")
println(); flush(stdout)

# ─── Prepare plan (once) ───
print("prepare_psf ... ")
t_prep = @elapsed plan = prepare_psf(pitch_um, N_cells;
    lambda_um = lambda_um, alpha_deg = alpha_deg, NA = NA_val,
    L_max = 15, N_psi = 32, Nxy = Nxy)
@printf("done  (%.1f s,  M_max = %d,  Nr = %d)\n\n", t_prep, plan.M_max, plan.Nr); flush(stdout)

# ─── Focal-spot mask: circle of radius ρ_Airy around center ───
const c     = Nxy ÷ 2 + 1
const rho_airy = plan.rho_airy
hw   = plan.psf_half_width_um
dx   = 2 * hw / (Nxy - 1)
const R_pix = round(Int, rho_airy / dx)
const mask  = Float64[(iy - c)^2 + (ix - c)^2 <= R_pix^2 for iy in 1:Nxy, ix in 1:Nxy]
@printf("Mask: circle r≤%d px (ρ_Airy = %.4f μm, dx = %.4f μm), %d pixels\n",
        R_pix, rho_airy, dx, Int(sum(mask)))

# ─── Pixel area and near-field power (for efficiency metric) ───
const dx2    = dx^2
const R_act  = N_cells * pitch_um
const P_near = π * R_act^2    # ∫|t|² 2π r dr = π R² since |t|=1
@printf("Pixel size = %.4f μm,  P_near = πR² = %.4e μm²\n", dx, P_near)

# ─── Plot axes ───
x_ax = collect(range(-hw, hw, length=Nxy))
y_ax = collect(range(-hw, hw, length=Nxy))

# ─── Output directory ───
outdir = "opt_results"
mkpath(outdir)
println("Output directory: $outdir")
println(); flush(stdout)

# ─── NLopt objective: maximize sum(I_raw .* mask) ───
const iter_count = Ref(0)

function nlopt_obj(phi::Vector{Float64}, grad::Vector{Float64})
    iter_count[] += 1
    n = iter_count[]
    snapshot = (n == 1 || n % 10 == 0)

    t   = exp.(im .* phi)
    res = execute_psf(plan, t)
    L   = sum(res.I_raw .* mask) * dx2 / P_near

    if length(grad) > 0
        # dL/dI = (dx²/P_near) * mask
        dL_dI = (dx2 / P_near) .* mask
        dL_dt = psf_adjoint(plan, t, res, dL_dI)
        @inbounds for j in eachindex(grad)
            grad[j] = 2.0 * real(conj(dL_dt[j]) * im * t[j])
        end
    end

    @printf("iter %4d:  η = %.6f\n", n, L); flush(stdout)

    if snapshot
        writedlm(joinpath(outdir, @sprintf("phi_%04d.txt", n)), phi)

        I_peak = maximum(res.I_raw)
        I_norm = I_peak > 0 ? res.I_raw ./ I_peak : res.I_raw

        p = heatmap(x_ax, y_ax, I_norm;
                    aspect_ratio = :equal, c = :inferno, clims = (0, 1),
                    xlabel = "x (μm)", ylabel = "y (μm)",
                    title = @sprintf("iter %d,  η = %.4f", n, L))
        savefig(p, joinpath(outdir, @sprintf("psf_%04d.png", n)))
        println("  → saved phi & PSF snapshot"); flush(stdout)
    end

    return L
end

# ─── NLopt setup ───
opt = Opt(:LD_CCSAQ, N_cells)
opt.max_objective = nlopt_obj
opt.maxeval = 200
opt.ftol_rel = 1e-8

# ─── Initial φ from command line ───
k     = 2π / lambda_um
f_val = R_um * sqrt(1 / NA_val^2 - 1)
r_c   = [(i - 0.5) * pitch_um for i in 1:N_cells]

# ─── Parse mode and init ───
run_gradcheck = length(ARGS) >= 1 && lowercase(ARGS[1]) == "gradcheck"
init_args = run_gradcheck ? ARGS[2:end] : ARGS

init_mode = length(init_args) >= 1 ? lowercase(init_args[1]) : "ideal"

if init_mode == "uniform"
    val = length(init_args) >= 2 ? parse(Float64, init_args[2]) : 0.0
    phi0 = fill(val, N_cells)
    println("Init: uniform (all φ = $val)")
elseif init_mode == "ideal"
    phi0 = Float64[-k * (sqrt(r^2 + f_val^2) - f_val) for r in r_c]
    println("Init: ideal oblique lens phase")
elseif init_mode == "quadratic"
    phi0 = Float64[-(k / (2 * f_val)) * r^2 for r in r_c]
    println("Init: quadratic phase φ(r) = -(k/2f)r²")
elseif init_mode == "random"
    seed = length(init_args) >= 2 ? parse(Int, init_args[2]) : 42
    Random.seed!(seed)
    phi0 = 2π .* rand(N_cells)
    println("Init: random φ ∈ [0,2π], seed=$seed")
else
    error("Unknown init mode '$init_mode'. Use: uniform, ideal, quadratic, or random [seed]")
end
flush(stdout)

if run_gradcheck
    # ─── Gradient sanity check: nlopt_obj gradient vs central FD ───
    println()
    println("═══ Gradient check of nlopt_obj ═══")
    flush(stdout)

    grad = zeros(N_cells)
    L0 = nlopt_obj(copy(phi0), grad)

    d = randn(N_cells)
    d ./= norm(d)
    analytic = dot(grad, d)

    eps_fd = 1e-5
    grad_dummy = Float64[]
    Lp = nlopt_obj(phi0 .+ eps_fd .* d, grad_dummy)
    Lm = nlopt_obj(phi0 .- eps_fd .* d, grad_dummy)
    fd = (Lp - Lm) / (2 * eps_fd)

    rel_err = abs(fd - analytic) / (abs(analytic) + 1e-30)

    println()
    println("  loss     = $L0")
    println("  adjoint  = $analytic")
    println("  FD       = $fd")
    println("  rel err  = $rel_err")
    flush(stdout)
else
    # ─── Run optimization ───
    println("Starting CCSA optimization (maxeval=200) ...")
    println(); flush(stdout)
    (maxf, maxphi, ret) = optimize(opt, phi0)

    println()
    println("Optimization finished: $ret")
    @printf("Final η = %.6f\n", maxf); flush(stdout)

    # ─── Save final result ───
    writedlm(joinpath(outdir, "phi_final.txt"), maxphi)
    I_final = psf_intensity(plan, exp.(im .* maxphi))
    I_peak  = maximum(I_final)
    I_norm  = I_peak > 0 ? I_final ./ I_peak : I_final

    p = heatmap(x_ax, y_ax, I_norm;
                aspect_ratio = :equal, c = :inferno, clims = (0, 1),
                xlabel = "x (μm)", ylabel = "y (μm)",
                title = @sprintf("final (iter %d),  η = %.4f", iter_count[], maxf))
    savefig(p, joinpath(outdir, "psf_final.png"))
    println("Saved: $(outdir)/phi_final.txt, $(outdir)/psf_final.png"); flush(stdout)
end
