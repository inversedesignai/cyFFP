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
      4. Neumann shift theorem     — for LPA fields t(r)·exp(ikₓr cosθ),
         (scalar fast path)          ALL modal coefficients from ONE low-order
                                     Hankel transform + FFT over shift angle.

    Algorithm steps (standard path):
      1. Angular FFT  : decompose near field into modes m ≥ 0 only
      2. Forward HT   : compute A^{TE/TM}_m(kr) via FFTLog (orders m±1)
      3. Propagate +   : Ã_m = (A^TE + A^TM)·prop,
         symmetrise      Ã_{-m} = (-1)^m (A^TM - A^TE)·prop
      4. Graf shift   : B_l(kr) = Σ_m Ã_{m+l}(kr) J_m(kr x0)
      5. Inverse HT   : b_l(ρ)  = H_l^{-1}[kr B_l(kr)](ρ)
      6. Angular IFFT : E(ρ,ψ)  = Σ_l b_l(ρ) exp(il ψ)

    Threading: uses Threads.@threads (shared memory, no serialization).
    Start Julia with  julia -t N  or set JULIA_NUM_THREADS=N.

    Based on notes by Arvin Keshvari (Supervisor: Dr. Zin Lin).
    Dependencies: FFTW, SpecialFunctions
"""
module CyFFP

using FFTW
using SpecialFunctions: loggamma, besselj

export cyfft_farfield,
       cyfft_farfield_modal,
       cyfft_farfield_shift,
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
#     Filter kernel U(q) = Γ((ν+1+q)/2)/Γ((ν+1-q)/2) * 2^q
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
    idx   = m_pos .+ 1
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
#  Symmetry: A^TE_{-m} = (-1)^{m+1} A^TE_m,
#            A^TM_{-m} = (-1)^m     A^TM_m
# ═══════════════════════════════════════════════════════════════

"""
    compute_TE_TM_coeffs(Em_r, Em_theta, m_pos, r, k)
    -> (A_TE, A_TM, kr_grid)

Compute TE and TM expansion coefficients for modes m = 0,...,M_max
via FFTLog.  Threaded over modes.  Skips E_θ transforms when zero.
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

    etheta_zero = iszero(Em_theta)

    A_TE = zeros(ComplexF64, Nr, N_modes)
    A_TM = zeros(ComplexF64, Nr, N_modes)

    # Warm the FFTW plan cache (thread-safety: first call is single-threaded)
    fftlog_hankel(zeros(ComplexF64, Nr), dln, 0.0)

    Threads.@threads for idx in 1:N_modes
        m  = m_pos[idx]
        fr = r .* Em_r[:, idx]

        Hmp1_r = fftlog_hankel(fr, dln, Float64(m + 1))
        Hmm1_r = fftlog_hankel(fr, dln, Float64(m - 1))

        if etheta_zero
            A_TE[:, idx] .= (im/2) .* (Hmp1_r .+ Hmm1_r)
            A_TM[:, idx] .= -(kz_over_k ./ 2) .* (Hmm1_r .- Hmp1_r)
        else
            fth = r .* Em_theta[:, idx]
            Hmm1_th = fftlog_hankel(fth, dln, Float64(m - 1))
            Hmp1_th = fftlog_hankel(fth, dln, Float64(m + 1))

            A_TE[:, idx] .= (im/2)  .* (Hmp1_r  .+ Hmm1_r)  .+
                            (1.0/2) .* (Hmm1_th .- Hmp1_th)
            A_TM[:, idx] .= -(kz_over_k ./ 2) .* (Hmm1_r  .- Hmp1_r)  .-
                            im .* (kz_over_k ./ 2) .* (Hmm1_th .+ Hmp1_th)
        end
    end

    return A_TE, A_TM, kr_grid
end


# ═══════════════════════════════════════════════════════════════
# §4  Step 3 — Propagation + Symmetry Reconstruction
#
#  Ã_m    = (A^TE_m + A^TM_m) · exp(ikz f)
#  Ã_{-m} = (-1)^m (A^TM_m − A^TE_m) · exp(ikz f)
# ═══════════════════════════════════════════════════════════════

"""
    propagate_and_symmetrize(A_TE, A_TM, m_pos, kr_grid, k, f; polarization=:y)
    -> (A_tilde, m_full)

`polarization` selects the symmetry sign:
- `:y` (s-pol, default): Ã_{-m} = (-1)^m (A^TE - A^TM) · prop
- `:x` (p-pol):          Ã_{-m} = (-1)^m (A^TM - A^TE) · prop
"""
function propagate_and_symmetrize(A_TE::Matrix{ComplexF64},
                                   A_TM::Matrix{ComplexF64},
                                   m_pos::Vector{Int},
                                   kr_grid::Vector{Float64},
                                   k::Float64, f::Float64;
                                   polarization::Symbol = :y)
    @assert polarization in (:x, :y) "polarization must be :x or :y"
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

        idx_pos = m + M_max + 1
        A_tilde[:, idx_pos] .= (ate .+ atm) .* prop

        if m > 0
            idx_neg = -m + M_max + 1
            s = iseven(m) ? 1.0 : -1.0
            # y-pol: Ã_{-m} = (-1)^m (A^TE - A^TM);  x-pol: (-1)^m (A^TM - A^TE)
            if polarization == :y
                A_tilde[:, idx_neg] .= s .* (ate .- atm) .* prop
            else
                A_tilde[:, idx_neg] .= s .* (atm .- ate) .* prop
            end
        end
    end

    return A_tilde, m_full
end


# ═══════════════════════════════════════════════════════════════
# §5  Step 4 — Graf's Addition Theorem
#  B_l(kr) = Σ_m Ã_{m+l}(kr) J_m(kr x0)
# ═══════════════════════════════════════════════════════════════

"""
    _besselj_range(m_max, x) -> Vector{Float64}

Compute J_0(x), J_1(x), ..., J_{m_max}(x) via Miller backward recurrence.
~10× faster than individual besselj calls for large m_max.
"""
function _besselj_range(m_max::Int, x::Float64)
    if m_max < 0; return Float64[]; end
    if abs(x) < 1e-30
        out = zeros(Float64, m_max + 1)
        out[1] = 1.0   # J_0(0) = 1
        return out
    end
    # Miller backward recurrence: start from m_start > m_max, recur downward
    m_start = max(m_max + 30, ceil(Int, abs(x)) + 30)
    jnp1 = 0.0
    jn   = 1.0
    out  = zeros(Float64, m_max + 1)
    for m in m_start:-1:0
        jnm1 = (2(m + 1) / x) * jn - jnp1
        if m <= m_max
            out[m + 1] = jn
        end
        jnp1 = jn
        jn   = jnm1
    end
    # Normalize using J_0(x) from besselj
    scale = besselj(0, x) / out[1]
    out .*= scale
    return out
end

"""
    graf_shift_one_kr(A_tilde_vec, M_max, kr_x0, L_max) -> B [2L_max+1]
"""
function graf_shift_one_kr(A_tilde_vec::AbstractVector{ComplexF64},
                            M_max::Int,
                            kr_x0::Float64,
                            L_max::Int)
    Nl = 2L_max + 1
    B  = zeros(ComplexF64, Nl)

    m_cut = min(M_max, ceil(Int, abs(kr_x0)) + 20)

    # Compute J_0..J_{m_cut} via fast recurrence, then use J_{-m} = (-1)^m J_m
    Jp = _besselj_range(m_cut, kr_x0)
    Jw = Vector{Float64}(undef, 2m_cut + 1)
    @inbounds for m in 0:m_cut
        Jw[m + m_cut + 1] = Jp[m + 1]                         # m ≥ 0
        Jw[-m + m_cut + 1] = iseven(m) ? Jp[m + 1] : -Jp[m + 1]  # m < 0
    end

    @inbounds for (li, l) in enumerate(-L_max:L_max)
        acc = zero(ComplexF64)
        m_lo = max(-m_cut, -M_max - l)
        m_hi = min( m_cut,  M_max - l)
        @simd for m in m_lo:m_hi
            acc += A_tilde_vec[m + l + M_max + 1] * Jw[m + m_cut + 1]
        end
        B[li] = acc
    end
    return B
end

"""
    graf_shift_all_kr(A_tilde, m_full, kr_grid, x0, L_max) -> B
"""
function graf_shift_all_kr(A_tilde::Matrix{ComplexF64},
                            m_full::Vector{Int},
                            kr_grid::Vector{Float64},
                            x0::Float64,
                            L_max::Int)
    M_max = (length(m_full) - 1) ÷ 2
    Nkr   = length(kr_grid)
    Nl    = 2L_max + 1
    B     = zeros(ComplexF64, Nkr, Nl)

    Threads.@threads for ikr in 1:Nkr
        kr_x0 = kr_grid[ikr] * x0
        B[ikr, :] = graf_shift_one_kr(A_tilde[ikr, :], M_max, kr_x0, L_max)
    end
    return B
end


# ═══════════════════════════════════════════════════════════════
# §6  Step 5 — Inverse Hankel Transforms in the Local Basis
# ═══════════════════════════════════════════════════════════════

"""
    local_hankel_inverse(B, kr_grid, dln, L_max) -> b [Nkr × (2L_max+1)]
"""
function local_hankel_inverse(B::Matrix{ComplexF64},
                               kr_grid::Vector{Float64},
                               dln::Float64,
                               L_max::Int)
    Nkr = size(B, 1)
    Nl  = 2L_max + 1
    b   = zeros(ComplexF64, Nkr, Nl)

    Threads.@threads for li in 1:Nl
        l = li - L_max - 1
        integrand = kr_grid .* B[:, li]
        b[:, li] .= fftlog_hankel(integrand, dln, Float64(l))
    end
    return b
end


# ═══════════════════════════════════════════════════════════════
# §7  Step 6 — Angular Synthesis (IFFT over l)
# ═══════════════════════════════════════════════════════════════

"""
    synthesize_local_psf(b, L_max, Npsi) -> (psf, psi_grid)
"""
function synthesize_local_psf(b::Matrix{ComplexF64},
                               L_max::Int,
                               Npsi::Int)
    Nr  = size(b, 1)
    psi = collect(range(0.0, 2π, length=Npsi+1)[1:end-1])
    psf = zeros(ComplexF64, Nr, Npsi)

    Threads.@threads for j in 1:Nr
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

function _pipeline(Em_r, Em_theta, m_pos, r, k, f, x0, dln, L_max, Npsi;
                   polarization::Symbol = :y)
    kr_max = 1.0 / r[1]
    if kr_max < k
        @warn "kr_max = 1/r_min = $(round(kr_max, sigdigits=4)) < k = $(round(k, sigdigits=4)). " *
              "Propagating modes with kr > kr_max are lost. " *
              "Decrease r_min to at most 1/k = $(round(1/k, sigdigits=4))."
    end

    A_TE, A_TM, kr_grid = compute_TE_TM_coeffs(Em_r, Em_theta, m_pos, r, k)
    A_tilde, m_full = propagate_and_symmetrize(A_TE, A_TM, m_pos, kr_grid, k, f;
                                                polarization=polarization)
    B = graf_shift_all_kr(A_tilde, m_full, kr_grid, x0, L_max)
    b = local_hankel_inverse(B, kr_grid, dln, L_max)
    rho_grid = exp.(log(1.0 / kr_grid[end]) .+ dln .* (0:length(kr_grid)-1))
    psf, psi_grid = synthesize_local_psf(b, L_max, Npsi)
    return psf, rho_grid, psi_grid
end


# ═══════════════════════════════════════════════════════════════
# §9  Top-level driver  (from full 2D field arrays)
# ═══════════════════════════════════════════════════════════════

"""
    cyfft_farfield(Er, Etheta, r, k, alpha, f; ...) -> (psf, rho_grid, psi_grid)

Full near-to-far-field transform.  Exploits E_{-m} = σ̂ E_m symmetry.

**Symmetry assumption:** x-polarized illumination (φ=0) of an axisymmetric
structure, so E_{-m,r}=E_{m,r} and E_{-m,θ}=-E_{m,θ} (cyFFP0 Eqs.65–66).
"""
function cyfft_farfield(Er::Matrix{ComplexF64},
                        Etheta::Matrix{ComplexF64},
                        r::Vector{Float64},
                        k::Float64, alpha::Float64, f::Float64;
                        M_buffer::Int = 10,
                        L_max::Union{Int,Nothing} = nothing,
                        Npsi::Int = 128,
                        polarization::Symbol = :y)
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
        @warn "Output rho_grid may extend beyond x0 = f·tan(α). " *
              "Graf's theorem converges only for ρ < x0."
    end

    Em_r, Em_theta, m_pos = angular_decompose(Er, Etheta, M_max)
    return _pipeline(Em_r, Em_theta, m_pos, r, k, f, x0, dln, L_max, Npsi;
                     polarization=polarization)
end


# ═══════════════════════════════════════════════════════════════
# §10  Top-level driver  (from pre-decomposed modal amplitudes)
# ═══════════════════════════════════════════════════════════════

"""
    cyfft_farfield_modal(Em_r_pos, Em_theta_pos, r, k, alpha, f; ...)

From pre-decomposed m ≥ 0 modes.  No negative-m simulation needed.
"""
function cyfft_farfield_modal(Em_r_pos::Matrix{ComplexF64},
                               Em_theta_pos::Matrix{ComplexF64},
                               r::Vector{Float64},
                               k::Float64, alpha::Float64, f::Float64;
                               L_max::Union{Int,Nothing} = nothing,
                               Npsi::Int = 128,
                               polarization::Symbol = :y)
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
        @warn "Output rho_grid may extend beyond x0 = f·tan(α). " *
              "Graf's theorem converges only for ρ < x0."
    end

    return _pipeline(Em_r_pos, Em_theta_pos, m_pos, r, k, f, x0, dln, L_max, Npsi;
                     polarization=polarization)
end


# ═══════════════════════════════════════════════════════════════
# §11  Neumann shift-theorem fast path  (scalar, LPA fields)
#
#  For fields of the form t(r)·exp(ikₓ r cosθ)·x̂, the Neumann
#  addition formula lets us compute ALL M_max scalar modal
#  coefficients from a SINGLE order-0 Hankel transform:
#
#    T₀(k) = H₀[r · t(r)](k)           — one FFTLog call
#
#    a_m(kr) = (-i)^m kr ∫ t(r) Jm(kx r) Jm(kr r) r dr
#
#  Using J_m(ar)J_m(br) = (1/2π) ∫₀²π J₀(c(φ)r) e^{-imφ} dφ
#  with c(φ) = √(a²+b²−2ab cosφ):
#
#    a_m(kr) = (-i)^m kr · (1/2π) ∫₀²π T₀(c(kr,kx,φ)) e^{-imφ} dφ
#
#  The m-sum becomes an FFT over φ at each kr.
#  Total cost: 1 FFTLog + Nkr interpolations + Nkr FFTs.
# ═══════════════════════════════════════════════════════════════

"""
    cyfft_farfield_shift(t_r, r, k, alpha, f; M_max=nothing, L_max=nothing, Npsi=128, Nphi=nothing)
    -> (psf, rho_grid, psi_grid)

**Fast path for LPA / Jacobi-Anger fields.**

Instead of M_max separate FFTLog calls, computes ALL scalar modal
coefficients from a single order-0 Hankel transform T₀ via the
Neumann shift theorem.  Then applies TE/TM splitting, propagation,
symmetry, Graf shift, and synthesis as usual.

# Arguments
- `t_r`   : complex [Nr], radial transmission t(r) (e.g. ideal lens phase)
- `r`     : log-spaced radial grid
- `k`     : wavenumber, `alpha`: oblique angle, `f`: focal length
- `M_max` : mode truncation (default: ceil(k sinα R) + 10)
- `Nphi`  : angular resolution for Neumann integral (default: next power-of-2 ≥ 2M_max+1)
"""
function cyfft_farfield_shift(t_r::Vector{ComplexF64},
                               r::Vector{Float64},
                               k::Float64, alpha::Float64, f::Float64;
                               M_max::Union{Int,Nothing} = nothing,
                               L_max::Union{Int,Nothing} = nothing,
                               Npsi::Int = 128,
                               Nphi::Union{Int,Nothing} = nothing)
    Nr  = length(r)
    R   = maximum(r)
    dln = log(r[2] / r[1])
    x0  = f * tan(alpha)
    kx  = k * sin(alpha)

    if isnothing(M_max)
        M_max = ceil(Int, kx * R) + 10
    end
    if isnothing(Nphi)
        Nphi = max(nextpow(2, 2M_max + 1), 64)
    end

    lambda   = 2π / k
    rho_max_ = 5.0 * lambda
    if isnothing(L_max)
        NA    = max(sin(alpha), 0.1)
        L_max = ceil(Int, k * NA * rho_max_) + 5
        L_max = max(L_max, 5)
    end

    kr_grid   = exp.(log(1.0 / r[end]) .+ dln .* (0:Nr-1))
    log_kr    = log.(kr_grid)

    kr_max = 1.0 / r[1]
    if kr_max < k
        @warn "kr_max = $(round(kr_max, sigdigits=4)) < k = $(round(k, sigdigits=4)). " *
              "Decrease r_min to at most 1/k."
    end

    # ── Step A: Single order-0 Hankel transform ─────────────────
    T0 = fftlog_hankel(r .* t_r, dln, 0.0)  # T₀(kr) = H₀[r·t](kr)

    # ── Step B: Neumann shift → all scalar a_m(kr) via FFT ──────
    # For each kr, evaluate T₀ at c(φ) = √(kr²+kx²−2·kr·kx·cosφ)
    # then FFT over φ to extract modes m = -M_max..M_max.
    phi_grid = range(0.0, 2π, length=Nphi+1)[1:end-1]

    # Propagation phase (applied per-kr)
    kz   = @. sqrt(complex(k^2 - kr_grid^2))
    prop = @. ifelse(kr_grid < k, exp(im * real(kz) * f), zero(ComplexF64))

    N_full  = 2M_max + 1
    A_tilde = zeros(ComplexF64, Nr, N_full)

    Threads.@threads for ikr in 1:Nr
        kr_val = kr_grid[ikr]

        # Evaluate T₀ at shifted arguments c(φ)
        buf = Vector{ComplexF64}(undef, Nphi)
        for (ip, phi) in enumerate(phi_grid)
            c = sqrt(max(kr_val^2 + kx^2 - 2*kr_val*kx*cos(phi), 0.0))
            # Interpolate T₀ on log-spaced kr grid
            if c < kr_grid[1] || c > kr_grid[end]
                buf[ip] = zero(ComplexF64)
            else
                lc = log(c)
                # Linear interpolation in log-kr
                idx_f = (lc - log_kr[1]) / dln + 1.0
                j0 = clamp(floor(Int, idx_f), 1, Nr-1)
                w  = idx_f - j0
                buf[ip] = (1-w) * T0[j0] + w * T0[j0+1]
            end
        end

        # FFT over φ → scalar modes a_m
        # IFFT convention: mode m at FFTW index mod(m, Nphi)+1
        # But we want: a_m ∝ ∫ T₀(c(φ)) e^{-imφ} dφ, which is fft (not ifft)
        am_fft = fft(buf)  # am_fft[mod(m,Nphi)+1] ∝ Σ_φ buf(φ) e^{-2πi m φ_idx/Nphi}

        # Build scalar A_tilde_m = (-i)^m · kr · (2π/Nphi) · am_fft[m] · prop
        # (The 2π/Nphi factor comes from the trapezoidal quadrature of the φ integral,
        #  and the 1/(2π) from the Neumann formula, giving net 1/Nphi)
        # Then split into TE+TM for x-polarized field.
        # For scalar: Ã_m = a_m · prop  (TE+TM combined).
        for m in -M_max:M_max
            jj   = mod(m, Nphi) + 1
            phase = (Complex(-1.0im))^m   # (-i)^m
            a_m  = phase * kr_val * am_fft[jj] / Nphi
            idx  = m + M_max + 1
            A_tilde[ikr, idx] = a_m * prop[ikr]
        end
    end

    m_full = collect(-M_max:M_max)

    # ── Steps 4–6: Graf shift → inverse Hankel → synthesis ──────
    if x0 > 0 && R > x0
        @warn "Output rho_grid may extend beyond x0 = f·tan(α). " *
              "Graf's theorem converges only for ρ < x0."
    end

    B = graf_shift_all_kr(A_tilde, m_full, kr_grid, x0, L_max)
    b = local_hankel_inverse(B, kr_grid, dln, L_max)
    rho_grid = exp.(log(1.0 / kr_grid[end]) .+ dln .* (0:length(kr_grid)-1))
    psf, psi_grid = synthesize_local_psf(b, L_max, Npsi)

    return psf, rho_grid, psi_grid
end

end  # module CyFFP
