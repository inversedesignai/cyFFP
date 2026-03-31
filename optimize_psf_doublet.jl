# optimize_psf_doublet.jl — Doublet phase optimization for oblique metalens
#
# Maximizes focusing efficiency η = sum(I_raw .* mask) × Δx² / (πR²)
# within the Airy disk for a metalens doublet: t₁(r), t₂(r) separated
# by distance d, at R=1mm, 30° oblique incidence, NA=0.4.
#
# Uses NLopt's CCSA (LD_CCSAQ) with adjoint gradients for both surfaces.
# Saves intermediate φ₁, φ₂ data and PSF plots every 10 iterations.
#
# Usage:
#   julia -t auto optimize_psf_doublet.jl [d_um] <init1> [arg1] <init2> [arg2]
#
# Init modes for each surface:
#   uniform [val]   — all φ = val (default 0)
#   ideal           — ideal oblique lens phase (default for surface 1)
#   quadratic       — quadratic phase φ(r) = -(k/2f)r²
#   random [seed]   — random φ ∈ [0,2π]
#
# Examples:
#   julia -t auto optimize_psf_doublet.jl 200 ideal uniform
#   julia -t auto optimize_psf_doublet.jl 500 ideal random 42
#   julia -t auto optimize_psf_doublet.jl 200 quadratic quadratic
#
# Gradient sanity check:
#   julia -t auto optimize_psf_doublet.jl gradcheck [d_um] <init1> [arg1] <init2> [arg2]

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
const R_um      = 1000.0
const N_cells   = round(Int, R_um / pitch_um)
const lambda_um = 0.5
const alpha_deg = 30.0
const NA_val    = 0.4
const Nxy       = 201

println("═══ Doublet PSF optimization (R=1 mm, α=30°, NA=0.4) ═══")
println("N_cells = $N_cells,  Nxy = $Nxy,  threads = $(Threads.nthreads())")
println(); flush(stdout)

# ─── Prepare plan (once) ───
print("prepare_psf ... "); flush(stdout)
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
        R_pix, rho_airy, dx, Int(sum(mask))); flush(stdout)

# ─── Pixel area and near-field power ───
const dx2    = dx^2
const R_act  = N_cells * pitch_um
const P_near = π * R_act^2
@printf("Pixel size = %.4f μm,  P_near = πR² = %.4e μm²\n", dx, P_near); flush(stdout)

# ─── Plot axes ───
x_ax = collect(range(-hw, hw, length=Nxy))
y_ax = collect(range(-hw, hw, length=Nxy))

# ─── Output directory ───
outdir = "opt_doublet_results"
mkpath(outdir)
println("Output directory: $outdir")
println(); flush(stdout)

# ─── Derived quantities for phase profiles ───
k     = 2π / lambda_um
f_val = R_um * sqrt(1 / NA_val^2 - 1)
r_c   = [(i - 0.5) * pitch_um for i in 1:N_cells]

# ─── Parse command line ───
function parse_init(args, default="uniform")
    mode = length(args) >= 1 ? lowercase(args[1]) : default
    if mode == "uniform"
        val = length(args) >= 2 ? parse(Float64, args[2]) : 0.0
        phi = fill(val, N_cells)
        desc = "uniform (φ=$val)"
        consumed = val == 0.0 && length(args) < 2 ? 1 : 2
    elseif mode == "ideal"
        phi = Float64[-k * (sqrt(r^2 + f_val^2) - f_val) for r in r_c]
        desc = "ideal lens"
        consumed = 1
    elseif mode == "quadratic"
        phi = Float64[-(k / (2 * f_val)) * r^2 for r in r_c]
        desc = "quadratic -(k/2f)r²"
        consumed = 1
    elseif mode == "random"
        seed = length(args) >= 2 ? parse(Int, args[2]) : 42
        Random.seed!(seed)
        phi = 2π .* rand(N_cells)
        desc = "random seed=$seed"
        consumed = 2
    else
        error("Unknown init mode '$mode'. Use: uniform, ideal, quadratic, or random [seed]")
    end
    return phi, desc, consumed
end

run_gradcheck = length(ARGS) >= 1 && lowercase(ARGS[1]) == "gradcheck"
args = run_gradcheck ? ARGS[2:end] : ARGS

# First arg (after optional gradcheck): d_um
d_um = length(args) >= 1 ? parse(Float64, args[1]) : 200.0
remaining = args[2:end]

# Parse init for surface 1
phi1, desc1, n1 = parse_init(remaining, "ideal")
remaining = remaining[min(n1+1, length(remaining)+1):end]

# Parse init for surface 2
phi2, desc2, _ = parse_init(remaining, "uniform")

@printf("d = %.1f μm\n", d_um)
println("Surface 1: $desc1")
println("Surface 2: $desc2"); flush(stdout)

# ─── NLopt objective ───
# Optimization variable: x = [φ₁; φ₂] (length 2*N_cells)
const iter_count = Ref(0)
const dL_dI_scaled = (dx2 / P_near) .* mask   # precompute once

function nlopt_obj(x::Vector{Float64}, grad::Vector{Float64})
    iter_count[] += 1
    n = iter_count[]
    snapshot = (n == 1 || n % 10 == 0)

    phi1_v = @view x[1:N_cells]
    phi2_v = @view x[N_cells+1:end]
    t1 = exp.(im .* phi1_v)
    t2 = exp.(im .* phi2_v)

    res = execute_psf_doublet(plan, t1, t2; d_um=d_um)
    L   = sum(res.I_raw .* mask) * dx2 / P_near

    if length(grad) > 0
        dL_dt1, dL_dt2 = psf_adjoint_doublet(plan, t1, t2, res, dL_dI_scaled)
        @inbounds for j in 1:N_cells
            grad[j]           = 2.0 * real(conj(dL_dt1[j]) * im * t1[j])
            grad[j + N_cells] = 2.0 * real(conj(dL_dt2[j]) * im * t2[j])
        end
    end

    @printf("iter %4d:  η = %.6f\n", n, L); flush(stdout)

    if snapshot
        writedlm(joinpath(outdir, @sprintf("phi1_%04d.txt", n)), collect(phi1_v))
        writedlm(joinpath(outdir, @sprintf("phi2_%04d.txt", n)), collect(phi2_v))

        I_peak = maximum(res.I_raw)
        I_norm = I_peak > 0 ? res.I_raw ./ I_peak : res.I_raw

        p = heatmap(x_ax, y_ax, I_norm;
                    aspect_ratio = :equal, c = :inferno, clims = (0, 1),
                    xlabel = "x (μm)", ylabel = "y (μm)",
                    title = @sprintf("iter %d,  η = %.4f", n, L))
        savefig(p, joinpath(outdir, @sprintf("psf_%04d.png", n)))
        println("  → saved snapshot"); flush(stdout)
    end

    return L
end

# ─── NLopt setup ───
opt = Opt(:LD_CCSAQ, 2 * N_cells)
opt.max_objective = nlopt_obj
opt.maxeval = 200
opt.ftol_rel = 1e-8

# ─── Initial x = [φ₁; φ₂] ───
x0 = vcat(phi1, phi2)

if run_gradcheck
    println()
    println("═══ Gradient check ═══"); flush(stdout)

    grad = zeros(2 * N_cells)
    L0 = nlopt_obj(copy(x0), grad)

    d = randn(2 * N_cells)
    d ./= norm(d)
    analytic = dot(grad, d)

    eps_fd = 1e-5
    grad_dummy = Float64[]
    Lp = nlopt_obj(x0 .+ eps_fd .* d, grad_dummy)
    Lm = nlopt_obj(x0 .- eps_fd .* d, grad_dummy)
    fd = (Lp - Lm) / (2 * eps_fd)
    rel_err = abs(fd - analytic) / (abs(analytic) + 1e-30)

    println()
    println("  loss     = $L0")
    println("  adjoint  = $analytic")
    println("  FD       = $fd")
    println("  rel err  = $rel_err"); flush(stdout)
else
    println("Starting CCSA optimization (maxeval=200) ...")
    println(); flush(stdout)
    (maxf, maxx, ret) = optimize(opt, x0)

    println()
    println("Optimization finished: $ret")
    @printf("Final η = %.6f\n", maxf); flush(stdout)

    # ─── Save final ───
    writedlm(joinpath(outdir, "phi1_final.txt"), maxx[1:N_cells])
    writedlm(joinpath(outdir, "phi2_final.txt"), maxx[N_cells+1:end])

    t1_final = exp.(im .* maxx[1:N_cells])
    t2_final = exp.(im .* maxx[N_cells+1:end])
    I_final = execute_psf_doublet(plan, t1_final, t2_final; d_um=d_um).I_raw
    I_peak  = maximum(I_final)
    I_norm  = I_peak > 0 ? I_final ./ I_peak : I_final

    p = heatmap(x_ax, y_ax, I_norm;
                aspect_ratio = :equal, c = :inferno, clims = (0, 1),
                xlabel = "x (μm)", ylabel = "y (μm)",
                title = @sprintf("final (iter %d),  η = %.4f", iter_count[], maxf))
    savefig(p, joinpath(outdir, "psf_final.png"))
    println("Saved to $(outdir)/"); flush(stdout)
end
