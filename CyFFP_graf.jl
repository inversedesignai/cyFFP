"""
    CyFFP.jl — Cylindrical Far-Field Propagation
    =============================================
    Near-to-far-field transform via Vector Cylindrical Harmonics.
    Reformulated for large oblique angles:

      1. FFTLog Hankel transforms  — order-agnostic, stable for |m| ~ 600+
      2. Graf's addition theorem   — shift cylindrical basis from optical axis
                                     to focal spot in pure modal space.
                                     No interpolation, no Cartesian regridding.

    Algorithm steps:
      1. Angular FFT  : decompose near field into modes m
      2. Forward HT   : compute A_m(kr) via FFTLog (orders m±1)
      3. Propagate    : Ã_m(kr) = A_m(kr) * exp(i kz f)
      4. Graf shift   : B_l(kr) = Σ_m Ã_{m+l}(kr) J_m(kr x0)   [mode convolution]
      5. Inverse HT   : b_l(ρ)  = H_l^{-1}[kr B_l(kr)](ρ)       [FFTLog again]
      6. Angular IFFT : E(ρ,ψ)  = Σ_l b_l(ρ) exp(il ψ)

    Based on notes by Arvin Keshvari (Supervisor: Dr. Zin Lin).
    Dependencies: FFTW, SpecialFunctions
"""
module CyFFP

using FFTW
using Distributed
using SpecialFunctions: loggamma, besselj

export cyfft_farfield,
       fftlog_hankel,
       angular_decompose,
       compute_TE_TM_coeffs,
       propagate_modes,
       graf_shift,
       graf_shift_all_kr,
       local_hankel_inverse,
       synthesize_local_psf

# ═══════════════════════════════════════════════════════════════
# §1  FFTLog Hankel Transform
#     H_ν[f](k) = ∫₀^∞ J_ν(kr) f(r) r dr
#     via log-variable change → convolution → FFT
#     Filter kernel U(q) = Γ((ν+1+q)/2)/Γ((ν+1-q)/2) * 2^q
#     evaluated via loggamma for stability at any |ν|.
# ═══════════════════════════════════════════════════════════════

"""
    fftlog_hankel(f_r, dln, nu) -> A_kr

Compute the order-ν Hankel transform of `f_r` sampled on a log-spaced grid
    r[j] = r0 * exp(j * dln),  j = 0, 1, ..., N-1.
Returns the transform A on the reciprocal log-grid
    kr[j] = exp(-log(r[end])) * exp(j * dln).

Stable for any real ν, including |ν| ~ 600.
"""
function fftlog_hankel(f_r::AbstractVector, dln::Real, nu::Real)
    N  = length(f_r)
    # FFTW-ordered frequency indices: 0,1,...,N/2-1,-N/2,...,-1
    n_idx = [0:N÷2-1; -N÷2:-1]
    q     = @. (2π / (N * dln)) * n_idx   # continuous Fourier variable

    # Filter kernel U_ν(q) computed in log-space for |ν| >> 1 stability
    U_c = map(q) do qj
        a = complex((nu + 1.0 + qj) / 2)
        b = complex((nu + 1.0 - qj) / 2)
        exp(loggamma(a) - loggamma(b) + qj * log(2.0))
    end

    F = fft(complex.(f_r))
    return real.(ifft(F .* U_c))
end


# ═══════════════════════════════════════════════════════════════
# §2  Step 1 — Angular Decomposition
# ═══════════════════════════════════════════════════════════════

"""
    angular_decompose(Er, Etheta, M_max) -> (Em_r, Em_theta, m_list)

FFT over θ axis (columns) of near-field arrays [Nr × Ntheta].
Returns modal arrays [Nr × (2M_max+1)] for modes m = -M_max:M_max.
"""
function angular_decompose(Er::Matrix{ComplexF64},
                            Etheta::Matrix{ComplexF64},
                            M_max::Int)
    Nr, Ntheta = size(Er)
    @assert size(Etheta) == (Nr, Ntheta)
    @assert Ntheta >= 2M_max + 1 "Ntheta too small for M_max=$M_max"

    Em_r_fft     = fft(Er,     2) ./ Ntheta
    Em_theta_fft = fft(Etheta, 2) ./ Ntheta

    m_list = collect(-M_max:M_max)
    # FFTW index for mode m: mod(m, Ntheta) + 1
    idx = mod.(m_list, Ntheta) .+ 1
    return Em_r_fft[:, idx], Em_theta_fft[:, idx], m_list
end


# ═══════════════════════════════════════════════════════════════
# §3  Step 2 — Forward Hankel Transforms (TE + TM)
# ═══════════════════════════════════════════════════════════════

"""
    compute_TE_TM_coeffs(Em_r, Em_theta, m_list, r, k)
    -> (A_TE, A_TM, kr_grid)

Compute TE and TM expansion coefficients for all angular modes via FFTLog.

For mode m (using Bessel recurrence J_{m±1} = (m/x)J_m ∓ J_m'):
  A_TE_m(kr) = (i/2)[H_{m+1}(r E_r) + H_{m-1}(r E_r)]
             + (1/2)[H_{m-1}(r E_θ) - H_{m+1}(r E_θ)]

  A_TM_m(kr) = -(kz/k)/2 * [H_{m-1}(r E_r) - H_{m+1}(r E_r)]
             - i(kz/k)/2 * [H_{m-1}(r E_θ) + H_{m+1}(r E_θ)]

4 FFTLog calls per mode; kz/k is applied pointwise in kr.
"""
function compute_TE_TM_coeffs(Em_r::Matrix{ComplexF64},
                               Em_theta::Matrix{ComplexF64},
                               m_list::Vector{Int},
                               r::Vector{Float64},
                               k::Float64)
    Nr      = length(r)
    N_modes = length(m_list)
    dln     = log(r[2] / r[1])

    # Output kr grid (reciprocal log-grid of r)
    kr_grid = exp.(log(1.0 / r[end]) .+ dln .* (0:Nr-1))

    # kz/k at each kr (real for propagating modes, 0 for evanescent)
    kz_over_k = @. sqrt(max(1.0 - (kr_grid / k)^2, 0.0))

    results = pmap(enumerate(m_list)) do (idx, m)
        fr  = r .* Em_r[:, idx]      # integrand weight r * E_r
        fth = r .* Em_theta[:, idx]

        Hmp1_r  = fftlog_hankel(fr,  dln, Float64(m + 1))
        Hmm1_r  = fftlog_hankel(fr,  dln, Float64(m - 1))
        Hmm1_th = fftlog_hankel(fth, dln, Float64(m - 1))
        Hmp1_th = fftlog_hankel(fth, dln, Float64(m + 1))

        col_TE = (im/2)  .* (Hmp1_r  .+ Hmm1_r)  .+
                 (1.0/2) .* (Hmm1_th .- Hmp1_th)

        col_TM = -(kz_over_k ./ 2) .* (Hmm1_r  .- Hmp1_r)  .-
                 im .* (kz_over_k ./ 2) .* (Hmm1_th .+ Hmp1_th)

        (col_TE, col_TM)
    end

    A_TE = zeros(ComplexF64, Nr, N_modes)
    A_TM = zeros(ComplexF64, Nr, N_modes)
    for (idx, (col_TE, col_TM)) in enumerate(results)
        A_TE[:, idx] .= col_TE
        A_TM[:, idx] .= col_TM
    end

    return A_TE, A_TM, kr_grid
end


# ═══════════════════════════════════════════════════════════════
# §4  Step 3 — Propagation to Focal Plane
# ═══════════════════════════════════════════════════════════════

"""
    propagate_modes(A_TE, A_TM, kr_grid, k, f) -> A_tilde

Combine TE+TM and apply propagation phase:
    Ã_m(kr) = [A_TE_m(kr) + A_TM_m(kr)] * exp(i kz(kr) f)
Evanescent modes (kr > k) are zeroed.
Returns A_tilde [Nkr × N_modes].
"""
function propagate_modes(A_TE::Matrix{ComplexF64},
                          A_TM::Matrix{ComplexF64},
                          kr_grid::Vector{Float64},
                          k::Float64, f::Float64)
    kz   = @. sqrt(complex(k^2 - kr_grid^2))
    prop = @. ifelse(kr_grid < k, exp(im * real(kz) * f), zero(ComplexF64))
    # prop is [Nkr]; broadcast over modes (columns)
    return (A_TE .+ A_TM) .* prop
end


# ═══════════════════════════════════════════════════════════════
# §5  Step 4 — Graf's Addition Theorem (Modal Basis Shift)
#
#  J_n(kr r) e^{inθ} = Σ_m J_m(kr x0) J_{n-m}(kr ρ) e^{i(n-m)ψ}
#  (convergent for ρ < x0)
#
#  Substituting into the far-field sum and relabeling n = m+l:
#  B_l(kr) = Σ_m Ã_{m+l}(kr) J_m(kr x0)
#
#  This is a discrete convolution in mode index m at each kr.
# ═══════════════════════════════════════════════════════════════

"""
    graf_shift_one_kr(A_tilde_vec, m_list, kr_x0, L_max) -> B [2L_max+1]

Apply the Graf addition theorem for a single kr value.
`A_tilde_vec` is the propagated modal coefficient vector at this kr (length 2M_max+1).
`kr_x0` = kr * x0 where x0 = f*tan(alpha).

The Bessel weight J_m(kr_x0) is negligible for |m| > kr_x0 + buffer,
so the inner sum is naturally truncated.
"""
function graf_shift_one_kr(A_tilde_vec::Vector{ComplexF64},
                            m_list::Vector{Int},
                            kr_x0::Float64,
                            L_max::Int)
    M_max  = maximum(abs.(m_list))
    l_list = -L_max:L_max
    Nl     = 2L_max + 1
    B      = zeros(ComplexF64, Nl)

    # Effective Bessel cutoff: J_m(z) ~ 0 for m > z + buffer
    m_cut  = min(M_max, ceil(Int, abs(kr_x0)) + 20)

    # Pre-compute Bessel weights J_m(kr_x0) for m in -m_cut:m_cut
    # besselj handles negative m via J_{-m}(x) = (-1)^m J_m(x)
    Jw = Vector{Float64}(undef, 2m_cut + 1)
    for (i, m) in enumerate(-m_cut:m_cut)
        Jw[i] = besselj(m, kr_x0)
    end

    # Build A_tilde lookup: m -> coefficient
    A_dict = Dict(m_list[i] => A_tilde_vec[i] for i in eachindex(m_list))

    for (li, l) in enumerate(l_list)
        acc = zero(ComplexF64)
        for (i, m) in enumerate(-m_cut:m_cut)
            n = m + l
            if haskey(A_dict, n)
                acc += A_dict[n] * Jw[i]
            end
        end
        B[li] = acc
    end
    return B
end


"""
    graf_shift_all_kr(A_tilde, m_list, kr_grid, x0, L_max)
    -> B [Nkr × (2L_max+1)]

Apply Graf's addition theorem at every kr point.
B_l(kr) = Σ_m Ã_{m+l}(kr) * J_m(kr * x0)   for |l| ≤ L_max.

For x0 = 0 (normal incidence), J_m(0) = δ_{m,0}, so B_l = Ã_l (identity).
"""
function graf_shift_all_kr(A_tilde::Matrix{ComplexF64},
                            m_list::Vector{Int},
                            kr_grid::Vector{Float64},
                            x0::Float64,
                            L_max::Int)
    Nkr = length(kr_grid)
    Nl  = 2L_max + 1

    rows = pmap(1:Nkr) do ikr
        kr_x0 = kr_grid[ikr] * x0
        graf_shift_one_kr(A_tilde[ikr, :], m_list, kr_x0, L_max)
    end

    B = zeros(ComplexF64, Nkr, Nl)
    for ikr in 1:Nkr
        B[ikr, :] = rows[ikr]
    end
    return B
end


# ═══════════════════════════════════════════════════════════════
# §6  Step 5 — Inverse Hankel Transforms in the Local Basis
#
#  b_l(ρ) = H_l^{-1}[kr B_l(kr)](ρ)
#          = FFTLog_l[ kr * B_l(kr) ]
#
#  Since H_ν is self-inverse, this is another fftlog_hankel call.
#  Now only |l| ≤ L_max ≪ M_max calls are needed.
# ═══════════════════════════════════════════════════════════════

"""
    local_hankel_inverse(B, kr_grid, dln, L_max) -> b [Nkr × (2L_max+1)]

Inverse Hankel transform in the local cylindrical basis.
For each mode l:
    b_l(ρ) = ∫₀^∞ B_l(kr) J_l(kr ρ) kr dkr  =  FFTLog_l[ kr * B_l ]

`kr_grid` is the log-spaced grid from compute_TE_TM_coeffs.
`dln` = log(kr_grid[2]/kr_grid[1]).
Output ρ grid is the reciprocal log-grid of kr_grid.
"""
function local_hankel_inverse(B::Matrix{ComplexF64},
                               kr_grid::Vector{Float64},
                               dln::Float64,
                               L_max::Int)
    Nkr  = size(B, 1)
    Nl   = 2L_max + 1

    cols = pmap(enumerate(-L_max:L_max)) do (li, l)
        # Hankel measure: multiply by kr before transform
        integrand = kr_grid .* B[:, li]
        fftlog_hankel(integrand, dln, Float64(l))
    end

    b = zeros(ComplexF64, Nkr, Nl)
    for (li, col) in enumerate(cols)
        b[:, li] .= col
    end
    return b
end


# ═══════════════════════════════════════════════════════════════
# §7  Step 6 — Angular Synthesis (IFFT over l)
# ═══════════════════════════════════════════════════════════════

"""
    synthesize_local_psf(b, L_max, Npsi) -> (psf [Nr×Npsi], psi_grid)

Reconstruct the PSF in local polar coordinates (ρ, ψ) via IFFT over l:
    E(ρ_j, ψ_s) = Σ_{l=-L_max}^{L_max} b_l(ρ_j) exp(il ψ_s)

ρ_j is the radial output grid (from local_hankel_inverse).
ψ_s = 2π s/Npsi, s = 0,...,Npsi-1.
"""
function synthesize_local_psf(b::Matrix{ComplexF64},
                               L_max::Int,
                               Npsi::Int)
    Nr   = size(b, 1)
    psi  = collect(range(0.0, 2π, length=Npsi+1)[1:end-1])
    psf  = zeros(ComplexF64, Nr, Npsi)

    for j in 1:Nr
        buf = zeros(ComplexF64, Npsi)
        for (li, l) in enumerate(-L_max:L_max)
            jj       = mod(l, Npsi) + 1   # FFTW index for mode l
            buf[jj] += b[j, li]
        end
        psf[j, :] .= ifft(buf) .* Npsi
    end
    return psf, psi
end


# ═══════════════════════════════════════════════════════════════
# §8  Top-level driver
# ═══════════════════════════════════════════════════════════════

"""
    cyfft_farfield(Er, Etheta, r, k, alpha, f;
                   M_buffer=10, L_max=nothing, Npsi=128)
    -> (psf, rho_grid, psi_grid)

Full near-to-far-field transform using the Graf-shift approach.
Returns PSF in local polar coordinates (ρ, ψ) centered at (f tan α, 0).

# Arguments
- `Er`, `Etheta` : complex [Nr × Ntheta] near-field (cylindrical components)
- `r`            : log-spaced radial grid.  Must satisfy r[j+1]/r[j] = const.
- `k`            : wavenumber 2π/λ
- `alpha`        : oblique angle [rad]
- `f`            : focal length
- `M_buffer`     : extra angular modes beyond M_max (default 10)
- `L_max`        : local mode truncation (default: ceil(k*NA*rho_max) + 5)
- `Npsi`         : azimuthal output points (default 128)

# Returns
- `psf`      : complex [Nrho × Npsi] field in local coordinates.
               Take abs2.(psf) for intensity.
- `rho_grid` : radial grid for local PSF (same length as input r)
- `psi_grid` : azimuthal grid [0, 2π)

# Notes
- For normal incidence (alpha ≈ 0), the Graf shift is trivially the identity.
- The PSF patch is valid for ρ < x0 = f tan(alpha).
- The ρ-grid is the FFTLog reciprocal grid of kr_grid.
"""
function cyfft_farfield(Er::Matrix{ComplexF64},
                        Etheta::Matrix{ComplexF64},
                        r::Vector{Float64},
                        k::Float64, alpha::Float64, f::Float64;
                        M_buffer::Int = 10,
                        L_max::Union{Int,Nothing} = nothing,
                        Npsi::Int = 128)

    R     = maximum(r)
    dln   = log(r[2] / r[1])
    x0    = f * tan(alpha)

    # Mode count for near-field decomposition
    M_max = ceil(Int, k * sin(alpha) * R) + M_buffer

    # Local mode count: enough to represent PSF out to a few Airy radii
    # L_max ~ k * NA * rho_max; default rho_max ~ 10 lambda
    if isnothing(L_max)
        NA    = sin(alpha)  # crude NA estimate; user can override
        NA    = max(NA, 0.1)
        L_max = ceil(Int, k * NA * 10.0 / (2π)) + 5
        L_max = max(L_max, 5)
    end

    # ── Step 1: Angular decomposition ──────────────────────────────
    Em_r, Em_theta, m_list = angular_decompose(Er, Etheta, M_max)

    # ── Step 2: Forward Hankel transforms (FFTLog) ─────────────────
    A_TE, A_TM, kr_grid = compute_TE_TM_coeffs(
        Em_r, Em_theta, m_list, r, k)

    # ── Step 3: Propagate to z = f ─────────────────────────────────
    A_tilde = propagate_modes(A_TE, A_TM, kr_grid, k, f)

    # ── Step 4: Graf shift (mode convolution) ──────────────────────
    B = graf_shift_all_kr(A_tilde, m_list, kr_grid, x0, L_max)

    # ── Step 5: Inverse Hankel transforms in local basis ───────────
    b = local_hankel_inverse(B, kr_grid, dln, L_max)

    # Output ρ grid: reciprocal log-grid of kr_grid
    rho_grid = exp.(log(1.0 / kr_grid[end]) .+ dln .* (0:length(kr_grid)-1))

    # ── Step 6: Angular IFFT → local PSF ──────────────────────────
    psf, psi_grid = synthesize_local_psf(b, L_max, Npsi)

    return psf, rho_grid, psi_grid
end

end  # module CyFFP
