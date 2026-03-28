"""
    test_adjoint_production.jl
    ==========================
    Production-scale adjoint correctness and performance via Zygote.

    Tests psf_intensity (the Zygote-differentiable entry point) at
    scales up to R=2000μm (M_max≈12,600) at α=30°.

    For each scale:
      - Zygote.gradient of a Strehl-like loss
      - FD verification of the Zygote gradient
      - Forward and adjoint timing
      - Memory estimate

    Run on the production machine with:
        julia -t 388 test_adjoint_production.jl

    Or with custom parameters:
        julia -t 388 -e '
            include("test_adjoint_production.jl")
            run_adjoint_production(alpha_deg=30.0, NA=0.4)
        '
"""

using ChainRulesCore, Zygote       # must be BEFORE include("cyffp.jl")
include("cyffp.jl")
using .CyFFP
using LinearAlgebra: dot

function run_adjoint_production(;
        alpha_deg::Float64 = 30.0,
        NA::Float64        = 0.4,
        lambda_um::Float64 = 0.5,
        pitch_um::Float64  = 0.3,
        L_max::Int         = 15,
        N_psi::Int         = 32,
        Nxy::Int           = 51,
        fd_eps::Float64    = 1e-5,
    )

    k = 2π / lambda_um

    # Test cases: progressively larger R up to production
    test_cases = [
        (R_um =  100.0, label = "small"),
        (R_um =  300.0, label = "medium"),
        (R_um =  500.0, label = "large"),
        (R_um = 1000.0, label = "xlarge"),
        (R_um = 2000.0, label = "production"),
    ]

    println("="^75)
    println("Adjoint production scaling (Zygote psf_intensity)")
    println("  α = $(alpha_deg)°,  NA = $NA,  λ = $(lambda_um)μm,  pitch = $(pitch_um)μm")
    println("  Threads: $(Threads.nthreads())  (maxthreadid: $(Threads.maxthreadid()))")
    println("="^75)

    results = []

    for tc in test_cases
        R_um    = tc.R_um
        N_cells = round(Int, R_um / pitch_um)
        f_val   = R_um * sqrt(1/NA^2 - 1)

        # Lens transmission
        r_c = [(i - 0.5) * pitch_um for i in 1:N_cells]
        t0  = ComplexF64[exp(-im * k * (sqrt(r^2 + f_val^2) - f_val)) for r in r_c]

        println("\n─── $(tc.label): R=$(R_um)μm, N_cells=$N_cells ───")

        # Prepare plan
        t_prep = @elapsed begin
            plan = prepare_psf(pitch_um, N_cells;
                               lambda_um=lambda_um, alpha_deg=alpha_deg, NA=NA,
                               L_max=L_max, N_psi=N_psi, Nxy=Nxy)
        end
        println("  M_max = $(plan.M_max),  Nr = $(plan.Nr)")
        println("  prepare_psf: $(round(t_prep, digits=1))s")

        mem_GB = (sizeof(plan.Jm) + sizeof(plan.kernels)) / 1e9
        println("  Plan memory: $(round(mem_GB, digits=1)) GB")

        # ── Forward timing: execute_psf path ──
        # Warmup
        I0 = psf_intensity(plan, t0)
        cy, cx = Nxy ÷ 2 + 1, Nxy ÷ 2 + 1  # center pixel

        t_fwd = @elapsed for rep in 1:3
            psf_intensity(plan, t0)
        end
        t_fwd /= 3
        println("  Forward (execute_psf): $(round(t_fwd, digits=1))s")

        # ── Forward timing: compute_scalar_coeffs path (for comparison) ──
        # This is the same code path used in test_production_psf.jl.
        # Build modes via Jacobi-Anger, then call compute_scalar_coeffs directly.
        u_m_direct = zeros(ComplexF64, plan.Nr, plan.M_max + 1)
        t_log_direct = zeros(ComplexF64, plan.Nr)
        @inbounds for j in 1:plan.Nr
            ic = plan.cell_idx[j]
            ic > 0 && (t_log_direct[j] = t0[ic])
        end
        Threads.@threads for idx in 1:plan.M_max+1
            @inbounds for j in 1:plan.Nr
                u_m_direct[j, idx] = plan.im_factors[idx] * t_log_direct[j] * plan.Jm[j, idx]
            end
        end

        compute_scalar_coeffs(u_m_direct, collect(0:plan.M_max), plan.r_log)  # warmup
        t_cs = @elapsed compute_scalar_coeffs(u_m_direct, collect(0:plan.M_max), plan.r_log)
        println("  compute_scalar_coeffs alone: $(round(t_cs, digits=1))s")
        u_m_direct = nothing; GC.gc()

        # ── Zygote gradient timing ──
        # Strehl-like loss: -peak / total
        loss(t) = let I = psf_intensity(plan, t)
            -I[cy, cx] / (sum(I) + 1e-10)
        end

        # Warmup
        g_zy = Zygote.gradient(loss, t0)[1]

        t_adj = @elapsed for rep in 1:3
            Zygote.gradient(loss, t0)[1]
        end
        t_adj /= 3
        # Adjoint time = total Zygote time minus forward time
        # (Zygote runs forward internally via the rrule)
        t_adj_only = t_adj - t_fwd
        ratio = t_adj / t_fwd

        println("  Zygote gradient: $(round(t_adj, digits=1))s  (fwd+adj)")
        println("  Adjoint only:    $(round(t_adj_only, digits=1))s")
        println("  Ratio (total/fwd): $(round(ratio, digits=2))×")

        # ── FD verification ──
        println("  FD check...")
        delta = randn(ComplexF64, N_cells)
        delta ./= sqrt(sum(abs2.(delta)))

        analytic = 2 * real(dot(g_zy, delta))

        Lp = loss(t0 .+ fd_eps .* delta)
        Lm = loss(t0 .- fd_eps .* delta)
        fd = (Lp - Lm) / (2 * fd_eps)

        fd_err = abs(fd - analytic) / (abs(analytic) + 1e-30)
        println("  FD err: $(round(fd_err, sigdigits=3))")

        ok = fd_err < 1e-3
        println("  $(ok ? "PASSED ✓" : "FAILED ✗")")

        push!(results, (
            label    = tc.label,
            R_um     = R_um,
            N_cells  = N_cells,
            M_max    = plan.M_max,
            Nr       = plan.Nr,
            t_prep   = t_prep,
            t_fwd    = t_fwd,
            t_cs     = t_cs,
            t_adj    = t_adj,
            ratio    = ratio,
            fd_err   = fd_err,
            passed   = ok,
        ))

        # Free plan to reclaim memory before next scale
        plan = nothing; GC.gc()
    end

    # ── Summary table ──
    println("\n" * "="^75)
    println("Summary")
    println("="^75)
    println("  Label        R(μm)  N_cells  M_max      Nr   prep(s)  fwd(s)  cs(s)   adj(s)  ratio   FD_err   OK")
    println("  " * "─"^95)
    for r in results
        println("  $(rpad(r.label, 12)) $(lpad(round(Int,r.R_um),6))  $(lpad(r.N_cells,7))  $(lpad(r.M_max,5))  $(lpad(r.Nr,6))  $(lpad(round(r.t_prep,digits=1),7))  $(lpad(round(r.t_fwd,digits=1),6))  $(lpad(round(r.t_cs,digits=1),6))  $(lpad(round(r.t_adj,digits=1),6))  $(lpad(round(r.ratio,digits=2),5))×  $(lpad(round(r.fd_err,sigdigits=2),8))   $(r.passed ? "✓" : "✗")")
    end
    println("="^75)

    n_pass = count(r -> r.passed, results)
    n_total = length(results)
    println("$(n_pass)/$(n_total) passed.")
    if n_pass == n_total
        println("All adjoint production tests PASSED ✓")
    end

    return results
end

# Default invocation when run as a script
if abspath(PROGRAM_FILE) == @__FILE__
    run_adjoint_production()
end
