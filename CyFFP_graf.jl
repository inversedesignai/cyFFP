"""
    CyFFP.jl — Cylindrical Far-Field Propagation
    =============================================
    Near-to-far-field transform via Vector Cylindrical Harmonics.
    Reformulated for large oblique angles:

      1. FFTLog Hankel transforms  — order-agnostic, stable for |m| ~ 600+
      2. Graf's addition theorem   — shift cylindrical basis from optical axis
                                     to focal spot in pure modal space.
                                     No interpolation, no Cartesian regridding.
      3. Negative-m symmetry       — E_{-m} = σ̂ E_m halves the Hankel work.

    Algorithm steps:
      1. Angular FFT  : decompose near field into modes m ≥ 0 only
      2. Forward HT   : compute A^{TE/TM}_m(kr) via FFTLog (orders m±1)
      3. Propagate +   : Ã_m = (A^TE + A^TM)·prop,
         symmetrise      Ã_{-m} = (-1)^m (A^TM - A^TE)·prop
      4. Graf shift   : B_l(kr) = Σ_m Ã_{m+l}(kr) J_m(kr x0)
      5. Inverse HT   : b_l(ρ)  = H_l^{-1}[kr B_l(kr)](ρ)
      6. Angular IFFT : E(ρ,ψ)  = Σ_l b_l(ρ) exp(il ψ)

    Based on notes by Arvin Keshvari (Supervisor: Dr. Zin Lin).
    Dependencies: FFTW, SpecialFunctions
"""
module CyFFP

using FFTW
using Distributed
using SpecialFunctions: loggamma, besselj

export cyfft_farfield,
       cyfft_farfield_modal,
       fftlog_hankel,
       angular_decompose,
       compute_TE_TM_coeffs,
       propagate_and_symmetrize,
       graf_shift_one_kr,
       graf_shift_all_kr,
       local_hankel_inverse,
       synthesize_local_psf


# ═══════════════════════════════════════════════════════════════
# §1  FFTLog Hankel Transform
#     H_ν[f](k) = ∫₀^∞ J_ν(kr) f(r) dr
#     via log-variable change → convolution → FFT
#     Filter kernel U(q) = Γ((ν+1+q)/2)/Γ((ν+1-q)/2) * 2^q
#     evaluated via loggamma for stability at any |ν|.
# ═══════════════════════════════════════════════════════════════

"""
    fftlog_hankel(f_r, dln, nu) -> A_kr

Compute the order-ν Hankel transform H_ν[f](k) = ∫₀^∞ f(r) J_ν(kr) dr
of `f_r` sampled on a log-spaced grid
    r[j] = r0 * exp(j * dln),  j = 0, 1, ..., N-1.
Returns the transform on the reciprocal log-grid
    kr[j] = exp(-log(r[end])) * exp(j * dln).

Uses the plain dr measure.  Call sites needing the r dr measure
(e.g. inverse Hankel) pre-multiply the integrand by r or kr.

Stable for any real ν, including |ν| ~ 600.
"""
function fftlog_hankel(f_r::AbstractVector, dln::Real, nu::Real)
    N     = length(f_r)
    n_idx = [n <= (N-1)÷2 ? n : n - N for n in 0:N-1]
    q     = @. (2π / (N * dln)) * n_idx

    U_c = map(q) do qj
        a = complex((nu + 1.0 + qj) / 2)
        b = complex((nu + 1.0 - qj) / 2)
        exp(loggamma(a) - loggamma(b) + qj * log(2.0))
    end

    F = fft(complex.(f_r))
    return ifft(F .* U_c)
end


# ═══════════════════════════════════════════════════════════════
# §2  Step 1 — Angular Decomposition  (m ≥ 0 only)
#     Negative modes are reconstructed via symmetry in Step 3.
# ═══════════════════════════════════════════════════════════════

"""
    angular_decompose(Er, Etheta, M_max) -> (Em_r, Em_theta, m_pos)

FFT over θ axis (columns) of near-field arrays [Nr × Ntheta].
Returns modal arrays [Nr × (M_max+1)] for modes m = 0, 1, ..., M_max.
"""
function angular_decompose(Er::Matrix{ComplexF64},
                            Etheta::Matrix{ComplexF64},
                            M_max::Int)
    Nr, Ntheta = size(Er)
    @assert size(Etheta) == (Nr, Ntheta)
    @assert Ntheta >= 2M_max + 1 "Ntheta too small for M_max=$M_max"

    Em_r_fft     = fft(Er,     2) ./ Ntheta
    Em_theta_fft = fft(Etheta, 2) ./ Ntheta

    m_pos = collect(0:M_max)
    idx   = m_pos .+ 1          # FFTW index for m ≥ 0
    return Em_r_fft[:, idx], Em_theta_fft[:, idx], m_pos
end


# ═══════════════════════════════════════════════════════════════
# §3  Step 2 — Forward Hankel Transforms  (m ≥ 0 only)
#
#  A^TE_m(kr) = (i/2)[H_{m+1} + H_{m-1}](r E_r)
#             + (1/2)[H_{m-1} − H_{m+1}](r E_θ)
#  A^TM_m(kr) = −(kz/k)/2 [H_{m-1} − H_{m+1}](r E_r)
#             − i(kz/k)/2 [H_{m-1} + H_{m+1}](r E_θ)
#
#  4 FFTLog calls per mode.  Only m = 0,...,M_max computed;
#  negative-m coefficients follow from symmetry:
#    A^TE_{-m} = (-1)^{m+1} A^TE_m
#    A^TM_{-m} = (-1)^m     A^TM_m
# ═══════════════════════════════════════════════════════════════

"""
    compute_TE_TM_coeffs(Em_r, Em_theta, m_pos, r, k)
    -> (A_TE, A_TM, kr_grid)

Compute TE and TM expansion coefficients for modes m = 0,...,M_max
via FFTLog (4 calls per mode).  Returns matrices [Nr × (M_max+1)].
"""
function compute_TE_TM_coeffs(Em_r::Matrix{ComplexF64},
                               Em_theta::Matrix{ComplexF64},
                               m_pos::Vector{Int},
                               r::Vector{Float64},
                               k::Float64)
    Nr      = length(r)
    N_modes = length(m_pos)
    dln     = log(r[2] / r[1])

    kr_grid   = exp.(log(1.0 / r[end]) .+ dln .* (0:Nr-1))
    kz_over_k = @. sqrt(max(1.0 - (kr_grid / k)^2, 0.0))

    # Detect if E_theta is identically zero → skip 2 FFTLog calls per mode
    etheta_zero = iszero(Em_theta)

    results = pmap(1:N_modes) do idx
        m   = m_pos[idx]
        fr  = r .* Em_r[:, idx]

        Hmp1_r  = fftlog_hankel(fr,  dln, Float64(m + 1))
        Hmm1_r  = fftlog_hankel(fr,  dln, Float64(m - 1))

        if etheta_zero
            col_TE = (im/2) .* (Hmp1_r .+ Hmm1_r)
            col_TM = -(kz_over_k ./ 2) .* (Hmm1_r .- Hmp1_r)
        else
            fth = r .* Em_theta[:, idx]
            Hmm1_th = fftlog_hankel(fth, dln, Float64(m - 1))
            Hmp1_th = fftlog_hankel(fth, dln, Float64(m + 1))

            col_TE = (im/2)  .* (Hmp1_r  .+ Hmm1_r)  .+
                     (1.0/2) .* (Hmm1_th .- Hmp1_th)

            col_TM = -(kz_over_k ./ 2) .* (Hmm1_r  .- Hmp1_r)  .-
                     im .* (kz_over_k ./ 2) .* (Hmm1_th .+ Hmp1_th)
        end

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
# §4  Step 3 — Propagation + Symmetry Reconstruction
#
#  For m ≥ 0:  Ã_m    = (A^TE_m + A^TM_m) · exp(ikz f)
#  For m < 0:  Ã_{-m} = (-1)^m (A^TM_m − A^TE_m) · exp(ikz f)
#
#  Derived from J_{-n}(x) = (-1)^n J_n(x) and the source symmetry
#  E_{-m,r} = E_{m,r},  E_{-m,θ} = −E_{m,θ}   (cyFFP0 Eqs.65–66).
# ═══════════════════════════════════════════════════════════════

"""
    propagate_and_symmetrize(A_TE, A_TM, m_pos, kr_grid, k, f)
    -> (A_tilde, m_full)

Combine TE+TM, apply propagation phase, and reconstruct negative-m
modes via symmetry.  Returns A_tilde [Nkr × (2M_max+1)] for
m = -M_max:M_max, and the corresponding mode list.
"""
function propagate_and_symmetrize(A_TE::Matrix{ComplexF64},
                                   A_TM::Matrix{ComplexF64},
                                   m_pos::Vector{Int},
                                   kr_grid::Vector{Float64},
                                   k::Float64, f::Float64)
    M_max = m_pos[end]
    Nkr   = length(kr_grid)

    kz   = @. sqrt(complex(k^2 - kr_grid^2))
    prop = @. ifelse(kr_grid < k, exp(im * real(kz) * f), zero(ComplexF64))

    N_full  = 2M_max + 1
    A_tilde = zeros(ComplexF64, Nkr, N_full)
    m_full  = collect(-M_max:M_max)

    for (ip, m) in enumerate(m_pos)
        ate = @view A_TE[:, ip]
        atm = @view A_TM[:, ip]

        # Positive mode: Ã_m = (A^TE_m + A^TM_m) · prop
        idx_pos = m + M_max + 1
        A_tilde[:, idx_pos] .= (ate .+ atm) .* prop

        # Negative mode: Ã_{-m} = (-1)^m (A^TM_m − A^TE_m) · prop
        if m > 0
            idx_neg = -m + M_max + 1
            s = iseven(m) ? 1.0 : -1.0
            A_tilde[:, idx_neg] .= s .* (atm .- ate) .* prop
        end
    end

    return A_tilde, m_full
end


# ═══════════════════════════════════════════════════════════════
# §5  Step 4 — Graf's Addition Theorem (Modal Basis Shift)
#
#  B_l(kr) = Σ_m Ã_{m+l}(kr) J_m(kr x0)
#
#  A_tilde_vec is indexed as m = -M_max:M_max, i.e.
#  A_tilde_vec[j] corresponds to mode m = j − M_max − 1.
# ═══════════════════════════════════════════════════════════════

"""
    graf_shift_one_kr(A_tilde_vec, M_max, kr_x0, L_max) -> B [2L_max+1]

Graf addition theorem for a single kr value.
Uses direct indexing (no Dict) for speed.
"""
function graf_shift_one_kr(A_tilde_vec::AbstractVector{ComplexF64},
                            M_max::Int,
                            kr_x0::Float64,
                            L_max::Int)
    Nl = 2L_max + 1
    B  = zeros(ComplexF64, Nl)

    m_cut = min(M_max, ceil(Int, abs(kr_x0)) + 20)

    # Pre-compute Bessel weights as Vector for fast access
    Jw = Vector{Float64}(undef, 2m_cut + 1)
    @inbounds for (i, m) in enumerate(-m_cut:m_cut)
        Jw[i] = besselj(m, kr_x0)
    end

    @inbounds for (li, l) in enumerate(-L_max:L_max)
        acc = zero(ComplexF64)
        m_lo = max(-m_cut, -M_max - l)
        m_hi = min( m_cut,  M_max - l)
        for m in m_lo:m_hi
            acc += A_tilde_vec[m + l + M_max + 1] * Jw[m + m_cut + 1]
        end
        B[li] = acc
    end
    return B
end


"""
    graf_shift_all_kr(A_tilde, m_full, kr_grid, x0, L_max)
    -> B [Nkr × (2L_max+1)]

Apply Graf's addition theorem at every kr point.
For x0 = 0 (normal incidence), J_m(0) = δ_{m,0}, so B_l = Ã_l.
"""
function graf_shift_all_kr(A_tilde::Matrix{ComplexF64},
                            m_full::Vector{Int},
                            kr_grid::Vector{Float64},
                            x0::Float64,
                            L_max::Int)
    M_max = (length(m_full) - 1) ÷ 2
    Nkr   = length(kr_grid)
    Nl    = 2L_max + 1

    rows = pmap(1:Nkr) do ikr
        kr_x0 = kr_grid[ikr] * x0
        graf_shift_one_kr(A_tilde[ikr, :], M_max, kr_x0, L_max)
    end

    B = zeros(ComplexF64, Nkr, Nl)
    @inbounds for ikr in 1:Nkr
        B[ikr, :] = rows[ikr]
    end
    return B
end


# ═══════════════════════════════════════════════════════════════
# §6  Step 5 — Inverse Hankel Transforms in the Local Basis
#
#  b_l(ρ) = H_l^{-1}[kr B_l(kr)](ρ) = FFTLog_l[ kr · B_l(kr) ]
#
#  Only |l| ≤ L_max ≪ M_max calls needed.
# ═══════════════════════════════════════════════════════════════

"""
    local_hankel_inverse(B, kr_grid, dln, L_max) -> b [Nkr × (2L_max+1)]

Inverse Hankel transform in the local cylindrical basis.
"""
function local_hankel_inverse(B::Matrix{ComplexF64},
                               kr_grid::Vector{Float64},
                               dln::Float64,
                               L_max::Int)
    Nkr = size(B, 1)

    cols = pmap(enumerate(-L_max:L_max)) do (li, l)
        integrand = kr_grid .* B[:, li]
        fftlog_hankel(integrand, dln, Float64(l))
    end

    b = zeros(ComplexF64, Nkr, 2L_max + 1)
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

Reconstruct the PSF in local polar coordinates (ρ, ψ) via IFFT over l.
"""
function synthesize_local_psf(b::Matrix{ComplexF64},
                               L_max::Int,
                               Npsi::Int)
    Nr  = size(b, 1)
    psi = collect(range(0.0, 2π, length=Npsi+1)[1:end-1])
    psf = zeros(ComplexF64, Nr, Npsi)

    for j in 1:Nr
        buf = zeros(ComplexF64, Npsi)
        for (li, l) in enumerate(-L_max:L_max)
            jj       = mod(l, Npsi) + 1
            buf[jj] += b[j, li]
        end
        psf[j, :] .= ifft(buf) .* Npsi
    end
    return psf, psi
end


# ═══════════════════════════════════════════════════════════════
# §8  Internal shared pipeline  (Steps 2–6)
# ═══════════════════════════════════════════════════════════════

function _pipeline(Em_r, Em_theta, m_pos, r, k, f, x0, dln, L_max, Npsi)
    # Check that the kr grid reaches the propagating band
    kr_max = 1.0 / r[1]
    if kr_max < k
        @warn "kr_max = 1/r_min = $(round(kr_max, sigdigits=4)) < k = $(round(k, sigdigits=4)). " *
              "Propagating modes with kr > kr_max are lost. " *
              "Decrease r_min to at most 1/k = $(round(1/k, sigdigits=4))."
    end

    # Step 2: Forward Hankel transforms (m ≥ 0 only — half the work)
    A_TE, A_TM, kr_grid = compute_TE_TM_coeffs(Em_r, Em_theta, m_pos, r, k)

    # Step 3: Propagation + symmetry → full m range
    A_tilde, m_full = propagate_and_symmetrize(A_TE, A_TM, m_pos, kr_grid, k, f)

    # Step 4: Graf shift (mode convolution)
    B = graf_shift_all_kr(A_tilde, m_full, kr_grid, x0, L_max)

    # Step 5: Inverse Hankel transforms in local basis
    b = local_hankel_inverse(B, kr_grid, dln, L_max)

    # Output ρ grid: reciprocal log-grid of kr_grid
    rho_grid = exp.(log(1.0 / kr_grid[end]) .+ dln .* (0:length(kr_grid)-1))

    # Step 6: Angular IFFT → local PSF
    psf, psi_grid = synthesize_local_psf(b, L_max, Npsi)

    return psf, rho_grid, psi_grid
end


# ═══════════════════════════════════════════════════════════════
# §9  Top-level driver  (from full 2D field arrays)
# ═══════════════════════════════════════════════════════════════

"""
    cyfft_farfield(Er, Etheta, r, k, alpha, f;
                   M_buffer=10, L_max=nothing, Npsi=128)
    -> (psf, rho_grid, psi_grid)

Full near-to-far-field transform using the Graf-shift approach.
Exploits the E_{-m} = σ̂ E_m symmetry to halve the number of
Hankel transforms (only m ≥ 0 computed).

**Symmetry assumption:** The input field must arise from x-polarized
illumination (φ=0) of an axisymmetric structure, so that
E_{-m,r} = E_{m,r} and E_{-m,θ} = -E_{m,θ} (cyFFP0 Eqs. 65–66).
This holds for ideal lenses, axisymmetric metalenses under LPA, and
any rotationally symmetric scatterer.  For other polarizations or
non-axisymmetric structures, the symmetry does not hold and results
will be incorrect.

# Arguments
- `Er`, `Etheta` : complex [Nr × Ntheta] near-field (cylindrical components)
- `r`            : log-spaced radial grid.  Must satisfy r[j+1]/r[j] = const.
- `k`            : wavenumber 2π/λ
- `alpha`        : oblique angle [rad]
- `f`            : focal length
- `M_buffer`     : extra angular modes beyond M_max (default 10)
- `L_max`        : local mode truncation (default: ceil(k*NA*ρ_max) + 5)
- `Npsi`         : azimuthal output points (default 128)

# Returns
- `psf`      : complex [Nrho × Npsi] field in local coordinates.
               Take abs2.(psf) for intensity.
- `rho_grid` : radial grid for local PSF (same length as input r)
- `psi_grid` : azimuthal grid [0, 2π)
"""
function cyfft_farfield(Er::Matrix{ComplexF64},
                        Etheta::Matrix{ComplexF64},
                        r::Vector{Float64},
                        k::Float64, alpha::Float64, f::Float64;
                        M_buffer::Int = 10,
                        L_max::Union{Int,Nothing} = nothing,
                        Npsi::Int = 128)

    R   = maximum(r)
    dln = log(r[2] / r[1])
    x0  = f * tan(alpha)

    M_max    = ceil(Int, k * sin(alpha) * R) + M_buffer
    lambda   = 2π / k
    rho_max_ = 5.0 * lambda
    if isnothing(L_max)
        NA    = max(sin(alpha), 0.1)
        L_max = ceil(Int, k * NA * rho_max_) + 5
        L_max = max(L_max, 5)
    end

    if x0 > 0 && R > x0
        @warn "Output rho_grid extends beyond x0 = f·tan(α) = $(round(x0, sigdigits=4)). " *
              "Graf's addition theorem converges only for ρ < x0. " *
              "Results at large ρ may be inaccurate."
    end

    # Step 1: Angular decomposition (m ≥ 0 only)
    Em_r, Em_theta, m_pos = angular_decompose(Er, Etheta, M_max)

    return _pipeline(Em_r, Em_theta, m_pos, r, k, f, x0, dln, L_max, Npsi)
end


# ═══════════════════════════════════════════════════════════════
# §10  Top-level driver  (from pre-decomposed modal amplitudes)
#      Input only needs m ≥ 0 — no negative-m simulation required.
# ═══════════════════════════════════════════════════════════════

"""
    cyfft_farfield_modal(Em_r_pos, Em_theta_pos, r, k, alpha, f;
                         L_max=nothing, Npsi=128)
    -> (psf, rho_grid, psi_grid)

Near-to-far-field transform from pre-decomposed modal amplitudes
for m = 0, 1, ..., M_max.  Negative-m modes are reconstructed
internally via the symmetry E_{-m,r} = E_{m,r}, E_{-m,θ} = -E_{m,θ}.

This is the entry point when modes come directly from FDTD/RCWA
simulations — no need to simulate m < 0.

# Arguments
- `Em_r_pos`     : complex [Nr × (M_max+1)], r-component modes m=0,...,M_max
- `Em_theta_pos` : complex [Nr × (M_max+1)], θ-component modes m=0,...,M_max
- `r`            : log-spaced radial grid
- `k, alpha, f`  : wavenumber, oblique angle, focal length
"""
function cyfft_farfield_modal(Em_r_pos::Matrix{ComplexF64},
                               Em_theta_pos::Matrix{ComplexF64},
                               r::Vector{Float64},
                               k::Float64, alpha::Float64, f::Float64;
                               L_max::Union{Int,Nothing} = nothing,
                               Npsi::Int = 128)
    Nr, N_modes = size(Em_r_pos)
    @assert size(Em_theta_pos) == (Nr, N_modes)

    M_max = N_modes - 1
    m_pos = collect(0:M_max)

    R   = maximum(r)
    dln = log(r[2] / r[1])
    x0  = f * tan(alpha)

    lambda   = 2π / k
    rho_max_ = 5.0 * lambda
    if isnothing(L_max)
        NA    = max(sin(alpha), 0.1)
        L_max = ceil(Int, k * NA * rho_max_) + 5
        L_max = max(L_max, 5)
    end

    if x0 > 0 && R > x0
        @warn "Output rho_grid extends beyond x0 = f·tan(α) = $(round(x0, sigdigits=4)). " *
              "Graf's addition theorem converges only for ρ < x0. " *
              "Results at large ρ may be inaccurate."
    end

    return _pipeline(Em_r_pos, Em_theta_pos, m_pos, r, k, f, x0, dln, L_max, Npsi)
end

end  # module CyFFP
