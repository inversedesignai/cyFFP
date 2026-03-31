# test/test_doublet_production.jl — Production-scale doublet timing + FD check
#
# R=1000μm, α=30°, NA=0.4.  Times forward + adjoint, then verifies
# gradients for both t₁ and t₂ via central FD.
#
# Run:  julia -t auto test/test_doublet_production.jl

using Printf
include(joinpath(@__DIR__, "..", "cyffp.jl"))
using .CyFFP
using LinearAlgebra: dot, norm
using Random

const pitch_um  = 0.3
const R_um      = 1000.0
const N_cells   = round(Int, R_um / pitch_um)
const lambda_um = 0.5
const alpha_deg = 30.0
const NA_val    = 0.4
const Nxy       = 101
const d_um      = 200.0   # doublet gap

println("═══ Doublet production test (R=1mm, α=30°, NA=0.4, d=$(d_um)μm) ═══")
println("N_cells = $N_cells,  Nxy = $Nxy,  threads = $(Threads.nthreads())")
println(); flush(stdout)

# ─── Prepare plan ───
print("prepare_psf ... "); flush(stdout)
t_prep = @elapsed plan = prepare_psf(pitch_um, N_cells;
    lambda_um=lambda_um, alpha_deg=alpha_deg, NA=NA_val,
    L_max=15, N_psi=32, Nxy=Nxy)
@printf("done  (%.1f s,  M_max = %d,  Nr = %d)\n\n", t_prep, plan.M_max, plan.Nr)
flush(stdout)

# ─── Transmission profiles ───
k     = 2π / lambda_um
f_val = R_um * sqrt(1 / NA_val^2 - 1)
r_c   = [(i - 0.5) * pitch_um for i in 1:N_cells]

Random.seed!(42)
t1 = ComplexF64[exp(-im * k * (sqrt(r^2 + f_val^2) - f_val)) for r in r_c]  # ideal lens
t2 = exp.(im .* 2π .* rand(N_cells))  # random phase

# ─── Forward timing ───
println("── Forward pass ──"); flush(stdout)
t_fwd = @elapsed res = execute_psf_doublet(plan, t1, t2; d_um=d_um)
@printf("Total forward: %.1f s\n\n", t_fwd); flush(stdout)

# ─── Adjoint timing ───
println("── Adjoint pass ──"); flush(stdout)
dL_dI = ones(size(res.I_raw))
t_adj = @elapsed dL_dt1, dL_dt2 = psf_adjoint_doublet(plan, t1, t2, res, dL_dI)
@printf("Total adjoint: %.1f s\n", t_adj)
@printf("Ratio (fwd+adj)/fwd: %.2f×\n\n", (t_fwd + t_adj) / t_fwd); flush(stdout)

# ─── FD gradient check ───
println("── FD gradient check (ε=1e-5) ──"); flush(stdout)
L0 = sum(res.I_raw)

for (label, dL_dt, perturb_t1, perturb_t2) in [
    ("∂L/∂t₁", dL_dt1, true, false),
    ("∂L/∂t₂", dL_dt2, false, true),
]
    delta = randn(ComplexF64, N_cells)
    delta ./= sqrt(sum(abs2.(delta)))
    analytic = 2 * real(dot(dL_dt, delta))

    eps_fd = 1e-5
    if perturb_t1
        rp = execute_psf_doublet(plan, t1 .+ eps_fd .* delta, t2; d_um=d_um)
        rm = execute_psf_doublet(plan, t1 .- eps_fd .* delta, t2; d_um=d_um)
    else
        rp = execute_psf_doublet(plan, t1, t2 .+ eps_fd .* delta; d_um=d_um)
        rm = execute_psf_doublet(plan, t1, t2 .- eps_fd .* delta; d_um=d_um)
    end
    fd = (sum(rp.I_raw) - sum(rm.I_raw)) / (2 * eps_fd)
    rel_err = abs(fd - analytic) / (abs(analytic) + 1e-30)

    println("  $label:")
    println("    adjoint = $analytic")
    println("    FD      = $fd")
    println("    rel err = $rel_err")
    flush(stdout)
end

println()
println("Done."); flush(stdout)
