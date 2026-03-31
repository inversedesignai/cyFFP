# test/test_adjoint_doublet.jl — FD gradient check for doublet PSF
#
# Checks ∂L/∂t₁ and ∂L/∂t₂ from psf_adjoint_doublet against central FD.
# Loss = sum(I_raw)  (total PSF intensity).
#
# Run:  julia -t auto test/test_adjoint_doublet.jl

using Printf
include(joinpath(@__DIR__, "..", "cyffp.jl"))
using .CyFFP
using LinearAlgebra: dot, norm
using Random

# ─── Small problem for fast FD checks ───
const pitch  = 0.3
const R      = 15.0
const N_cells = round(Int, R / pitch)
const k      = 2π / 0.5
const f      = R * sqrt(1/0.3^2 - 1)
const alpha_deg = 10.0

plan = prepare_psf(pitch, N_cells;
    lambda_um=0.5, alpha_deg=alpha_deg, f_um=f,
    Nxy=51, L_max=6, N_psi=16)

println("═══ Doublet adjoint FD check ═══")
println("N_cells = $N_cells,  M_max = $(plan.M_max),  Nr = $(plan.Nr)")
println()

# ─── Test configurations ───
Random.seed!(123)
r_c = [(i-0.5)*pitch for i in 1:N_cells]

configs = [
    ("ideal + ideal, d=10",
     ComplexF64[exp(-im*k*(sqrt(r^2+f^2)-f)) for r in r_c],
     ComplexF64[exp(-im*k*r^2/(2f)) for r in r_c],
     10.0),
    ("random + random, d=5",
     exp.(im .* 2π .* rand(N_cells)),
     exp.(im .* 2π .* rand(N_cells)),
     5.0),
    ("uniform + random, d=20",
     ones(ComplexF64, N_cells),
     exp.(im .* 2π .* rand(N_cells)),
     20.0),
    ("random + uniform, d=50",
     exp.(im .* 2π .* rand(N_cells)),
     ones(ComplexF64, N_cells),
     50.0),
]

const FD_EPS = 1e-5

for (label, t1, t2, d_um) in configs
    println("── $label ──")

    # Forward + adjoint
    res = execute_psf_doublet(plan, t1, t2; d_um=d_um)
    dL_dI = ones(size(res.I_raw))
    dL_dt1, dL_dt2 = psf_adjoint_doublet(plan, t1, t2, res, dL_dI)
    L0 = sum(res.I_raw)

    # ── Check ∂L/∂t₁ ──
    delta1 = randn(ComplexF64, N_cells)
    delta1 ./= sqrt(sum(abs2.(delta1)))
    analytic1 = 2 * real(dot(dL_dt1, delta1))

    rp1 = execute_psf_doublet(plan, t1 .+ FD_EPS .* delta1, t2; d_um=d_um)
    rm1 = execute_psf_doublet(plan, t1 .- FD_EPS .* delta1, t2; d_um=d_um)
    fd1 = (sum(rp1.I_raw) - sum(rm1.I_raw)) / (2 * FD_EPS)
    err1 = abs(fd1 - analytic1) / (abs(analytic1) + 1e-30)

    println("  ∂L/∂t₁:  adjoint = $analytic1")
    println("           FD      = $fd1")
    println("           rel err = $err1")

    # ── Check ∂L/∂t₂ ──
    delta2 = randn(ComplexF64, N_cells)
    delta2 ./= sqrt(sum(abs2.(delta2)))
    analytic2 = 2 * real(dot(dL_dt2, delta2))

    rp2 = execute_psf_doublet(plan, t1, t2 .+ FD_EPS .* delta2; d_um=d_um)
    rm2 = execute_psf_doublet(plan, t1, t2 .- FD_EPS .* delta2; d_um=d_um)
    fd2 = (sum(rp2.I_raw) - sum(rm2.I_raw)) / (2 * FD_EPS)
    err2 = abs(fd2 - analytic2) / (abs(analytic2) + 1e-30)

    println("  ∂L/∂t₂:  adjoint = $analytic2")
    println("           FD      = $fd2")
    println("           rel err = $err2")
    println()
end

println("Done.")
