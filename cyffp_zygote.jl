"""
    cyffp_zygote.jl
    ================
    ChainRulesCore rrule for execute_psf, enabling Zygote differentiation.

    Usage:
        include("cyffp.jl")
        using .CyFFP
        using ChainRulesCore, Zygote
        include("cyffp_zygote.jl")

        plan = prepare_psf(0.3, 6667; lambda_um=0.5, alpha_deg=30.0, NA=0.4)

        # Zygote differentiates through execute_psf automatically:
        grad = Zygote.gradient(t -> begin
            r = execute_psf(plan, t)
            return -r.I_raw[cy, cx]   # maximize peak intensity
        end, t_vals)[1]

        # Composed losses work too:
        grad = Zygote.gradient(t -> begin
            r = execute_psf(plan, t)
            I = r.I_raw
            return -I[cy, cx] / (sum(I) + 1e-10)  # Strehl-like
        end, t_vals)[1]

    The rrule uses psf_adjoint internally (hand-derived reverse-mode,
    no AD).  The adjoint/forward ratio is ~2.4-4× depending on M_max.
"""

using ChainRulesCore

function ChainRulesCore.rrule(::typeof(CyFFP.execute_psf),
                               plan::CyFFP.PSFPlan,
                               t_vals::Vector{ComplexF64})
    result = CyFFP.execute_psf(plan, t_vals)

    function execute_psf_pullback(Δ)
        dL_dI_raw = zeros(Float64, plan.Nxy, plan.Nxy)

        # Handle cotangent of I_raw (the primary differentiable output)
        Δ_I_raw = try; Δ.I_raw; catch; nothing; end
        if Δ_I_raw !== nothing && !isa(Δ_I_raw, ChainRulesCore.AbstractZero)
            dL_dI_raw .+= real.(Δ_I_raw)
        end

        # Also support differentiation through result.I (normalized).
        # I = I_raw / I_peak ⟹ dL/dI_raw ≈ dL/dI / I_peak.
        # (Ignores the derivative of I_peak at the argmax — acceptable
        # for smooth losses that don't depend on exact normalization.)
        Δ_I = try; Δ.I; catch; nothing; end
        if Δ_I !== nothing && !isa(Δ_I, ChainRulesCore.AbstractZero)
            I_peak = maximum(result.I_raw)
            I_peak > 0 && (dL_dI_raw .+= real.(Δ_I) ./ I_peak)
        end

        dL_dt = CyFFP.psf_adjoint(plan, t_vals, result, dL_dI_raw)
        return NoTangent(), NoTangent(), dL_dt
    end

    return result, execute_psf_pullback
end
