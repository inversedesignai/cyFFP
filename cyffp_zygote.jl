"""
    cyffp_zygote.jl
    ================
    ChainRulesCore rrule for psf_intensity, enabling Zygote differentiation.

    Usage:
        include("cyffp.jl")
        using .CyFFP
        using ChainRulesCore, Zygote
        include("cyffp_zygote.jl")

        plan = prepare_psf(0.3, 6667; lambda_um=0.5, alpha_deg=30.0, NA=0.4)

        # Zygote differentiates through psf_intensity automatically:
        grad = Zygote.gradient(t -> begin
            I = psf_intensity(plan, t)
            return -I[cy, cx] / (sum(I) + 1e-10)  # Strehl-like
        end, t_vals)[1]

    The rrule intercepts psf_intensity and uses psf_adjoint internally.
    Zygote differentiates everything outside (loss composition, indexing).

    psf_intensity returns a plain Matrix{Float64} (not a NamedTuple),
    so the cotangent is just a matrix — no fragile field extraction.
"""

using ChainRulesCore

function ChainRulesCore.rrule(::typeof(CyFFP.psf_intensity),
                               plan::CyFFP.PSFPlan,
                               t_vals::Vector{ComplexF64})
    # Forward: run execute_psf (saves u_psf etc. for adjoint)
    result = CyFFP.execute_psf(plan, t_vals)
    I_raw  = result.I_raw

    function psf_intensity_pullback(Δ_I_raw)
        # Δ_I_raw is a plain Matrix{Float64} — no NamedTuple extraction needed
        dL_dI = if Δ_I_raw isa ChainRulesCore.AbstractZero
            zeros(Float64, plan.Nxy, plan.Nxy)
        else
            Float64.(real.(Δ_I_raw))
        end
        dL_dt = CyFFP.psf_adjoint(plan, t_vals, result, dL_dI)
        return NoTangent(), NoTangent(), dL_dt
    end

    return I_raw, psf_intensity_pullback
end
