# test/test_grad_phi.jl — Gradient check: Zygote vs FD for φ-parameterized PSF
#
# t_vals[i] = exp(im * φ[i]),  loss L(φ) = sum(I_raw)
#
# 12 trials with wildly different φ arrays; reports actual relative errors
# of directional derivatives  dL/dφ · d   (Zygote vs central FD).
#
# Run:  julia -t auto test/test_grad_phi.jl

using Printf
using ChainRulesCore, Zygote   # must load BEFORE cyffp.jl so the rrule is registered
include(joinpath(@__DIR__, "..", "cyffp.jl"))
using .CyFFP
using LinearAlgebra: dot, norm
using Random

# ─── Problem parameters: 2 mm lens, 30° oblique, NA = 0.4 ───
const pitch_um  = 0.3
const R_um      = 2000.0
const N_cells   = round(Int, R_um / pitch_um)   # 6667
const lambda_um = 0.5
const alpha_deg = 30.0
const NA_val    = 0.4
const Nxy       = 201

println("═══ Gradient check: φ-parameterised PSF (R=2 mm, α=30°, NA=0.4) ═══")
println("N_cells = $N_cells,  Nxy = $Nxy,  threads = $(Threads.nthreads())")
println()

# ─── Prepare plan (once) ───
print("prepare_psf ... ")
t_prep = @elapsed plan = prepare_psf(pitch_um, N_cells;
    lambda_um = lambda_um, alpha_deg = alpha_deg, NA = NA_val,
    L_max = 15, N_psi = 32, Nxy = Nxy)
@printf("done  (%.1f s,  M_max = %d,  Nr = %d)\n\n", t_prep, plan.M_max, plan.Nr)

# ─── Ideal lens phase (for one of the trials) ───
k     = 2π / lambda_um
f_val = R_um * sqrt(1 / NA_val^2 - 1)
r_c   = [(i - 0.5) * pitch_um for i in 1:N_cells]
phi_ideal = Float64[-k * (sqrt(r^2 + f_val^2) - f_val) for r in r_c]

@printf("f = %.1f μm,  x₀ = f·tan(α) = %.1f μm\n", f_val, f_val * tand(alpha_deg))
const c  = Nxy ÷ 2 + 1
const R_pix = 10
# Precompute binary mask: 1.0 inside circle, 0.0 outside
const mask = Float64[(iy - c)^2 + (ix - c)^2 <= R_pix^2 for iy in 1:Nxy, ix in 1:Nxy]
println("Loss = sum(I_raw .* mask)  (circle r≤$(R_pix)px around center, $(Int(sum(mask))) pixels)")
println()

# ─── Loss: sum of intensities within circle of 10 pixels around center ───
loss(phi) = sum(psf_intensity(plan, exp.(im .* phi)) .* mask)

# ─── Build 12 wildly different φ arrays ───
phis   = Vector{Vector{Float64}}(undef, 12)
labels = Vector{String}(undef, 12)
Random.seed!(2024)

phis[1]  = 2π .* rand(N_cells);          labels[1]  = "uniform [0,2π] #1"
phis[2]  = 2π .* rand(N_cells);          labels[2]  = "uniform [0,2π] #2"
phis[3]  = 2π .* rand(N_cells);          labels[3]  = "uniform [0,2π] #3"
phis[4]  = zeros(N_cells);               labels[4]  = "all zeros"
phis[5]  = fill(π, N_cells);             labels[5]  = "all π"
phis[6]  = Float64[rand(Bool) ? π : 0.0 for _ in 1:N_cells]
                                          labels[6]  = "binary {0,π}"
phis[7]  = collect(range(0, 6π, length=N_cells))
                                          labels[7]  = "linear 0→6π"
phis[8]  = 3.0 .* randn(N_cells);        labels[8]  = "Gaussian σ=3"
phis[9]  = copy(phi_ideal);              labels[9]  = "ideal lens"
phis[10] = phi_ideal .+ 0.3 .* randn(N_cells)
                                          labels[10] = "ideal + noise σ=0.3"
phis[11] = Float64[2π*sin(50*2π*i/N_cells) for i in 1:N_cells]
                                          labels[11] = "sin(50 cycles)"
phis[12] = 2π .* rand(N_cells);          labels[12] = "uniform [0,2π] #4"

# ─── Run gradient checks ───
const FD_EPS = 1e-5

println("Running 12 trials  (ε = $FD_EPS, each: 1 Zygote grad + 2 FD forwards) ...")
println()

for (i, (phi, lab)) in enumerate(zip(phis, labels))
    println("── Trial $i: $lab ──")

    # ── Zygote gradient (forward + adjoint via rrule) ──
    ta = @elapsed begin
        L0, pb = Zygote.pullback(loss, phi)
        g = pb(1.0)[1]
    end

    # ── random real direction (unit) ──
    d = randn(N_cells)
    d ./= norm(d)
    analytic = dot(g, d)

    # ── central FD ──
    tf = @elapsed begin
        Lp = loss(phi .+ FD_EPS .* d)
        Lm = loss(phi .- FD_EPS .* d)
    end
    fd      = (Lp - Lm) / (2 * FD_EPS)
    rel_err = abs(fd - analytic) / (abs(analytic) + 1e-30)

    println("  loss     = $L0")
    println("  adjoint  = $analytic")
    println("  FD       = $fd")
    println("  rel err  = $rel_err")
    println("  (Zygote $(round(ta, digits=1))s + FD $(round(tf, digits=1))s)")
    println()
end

println("Done.")
