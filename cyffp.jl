"""
    CyFFP — Cylindrical Far-Field Propagation
    ==========================================
    Near-to-far-field transform via Vector Cylindrical Harmonics.

    This module is built incrementally; each step of the pipeline
    is added, tested, and validated before proceeding to the next.
"""
module CyFFP

using FFTW
using SpecialFunctions: loggamma

export angular_decompose, fftlog_hankel, compute_TE_TM_coeffs, propagate_and_symmetrize

# ═══════════════════════════════════════════════════════════════
# Step 1 — Angular Decomposition
#
#   E^Near(r, θ) = Σ_m E_m(r) e^{imθ}
#
#   E_{m,r/θ}(r_j) = (1/Nθ) Σ_l E_{r/θ}(r_j, θ_l) e^{-imθ_l}
#
# This is a DFT over the θ axis.  FFTW convention:
#   fft(X, 2)[j, n]  =  Σ_l X[j,l] exp(-2πi (l-1)(n-1)/Nθ)
#
# With θ_l = 2π(l-1)/Nθ, the FFTW output at index n corresponds
# to mode m = n-1 for n ≤ Nθ/2+1, and m = n-1-Nθ for n > Nθ/2+1.
# Equivalently, mode m ≥ 0 is at index m+1.
#
# We extract only m = 0, 1, ..., M_max (positive modes).
# Negative modes are reconstructed later via symmetry.
# ═══════════════════════════════════════════════════════════════

"""
    angular_decompose(Er, Etheta, M_max)
    -> (Em_r, Em_theta, m_pos)

Extract angular modes m = 0, 1, ..., M_max from 2D field arrays
`Er[Nr, Ntheta]` and `Etheta[Nr, Ntheta]` via FFT over the θ axis.

The θ grid is assumed uniform: θ_l = 2π l / Nθ for l = 0, ..., Nθ-1.

Returns:
- `Em_r[Nr, M_max+1]`    : r-component of mode m (column index = m+1)
- `Em_theta[Nr, M_max+1]`: θ-component of mode m
- `m_pos = [0, 1, ..., M_max]`

Requires `Ntheta ≥ 2*M_max + 1` to satisfy the angular Nyquist criterion.
"""
function angular_decompose(Er::Matrix{ComplexF64},
                            Etheta::Matrix{ComplexF64},
                            M_max::Int)
    Nr, Ntheta = size(Er)
    @assert size(Etheta) == (Nr, Ntheta) "Er and Etheta must have the same size"
    @assert M_max >= 0 "M_max must be non-negative"
    @assert Ntheta >= 2M_max + 1 "Ntheta=$Ntheta too small for M_max=$M_max (need ≥ $(2M_max+1))"

    # FFT along θ (dimension 2), normalise by 1/Nθ
    Em_r_full     = fft(Er,     2) ./ Ntheta
    Em_theta_full = fft(Etheta, 2) ./ Ntheta

    # Extract m = 0, 1, ..., M_max.
    # FFTW: mode m ≥ 0 lives at index m + 1.
    m_pos = collect(0:M_max)
    idx   = m_pos .+ 1

    return Em_r_full[:, idx], Em_theta_full[:, idx], m_pos
end


# ═══════════════════════════════════════════════════════════════
# Step 2 (part 1) — FFTLog Hankel Transform
#
#   H_ν[f](k) = ∫₀^∞ f(r) J_ν(k r) dr
#
# Computed via log-variable change r = r₀ eᵗ, k = k₀ eˢ (r₀k₀=1),
# which converts the Hankel integral into a convolution:
#   A(s) = ∫ g(t) K_ν(s+t) dt,   g(t) = r₀ eᵗ f(r₀ eᵗ),  K_ν(τ) = J_ν(eᵗ)
#
# In Fourier space:  Â(q) = ĝ(q) · û(q)
# where the filter kernel (FFTW convention) is:
#   û(q) = 2^{-iq} Γ((ν+1-iq)/2) / Γ((ν+1+iq)/2)
# with q = 2πn/(N·Δln) being real DFT frequency values.
# |û(q)| = 1 for all q (conjugate Gamma pair).
#
# Input f_r must be sampled on a log-spaced grid:
#   r[j] = r₀ · exp(j · Δln),   j = 0, ..., N-1
#
# Output lives on the reciprocal log-grid:
#   kr[j] = (1/r[end]) · exp(j · Δln)
#
# Convention: plain dr measure (no r weight).  Call sites needing
# the r dr measure pre-multiply by r or kr.
#
# Stable for any real ν, including |ν| ~ 600+.
# ═══════════════════════════════════════════════════════════════

"""
    fftlog_hankel(f_r, dln, nu) -> A_kr

Compute the order-ν Hankel transform

    H_ν[f](k) = ∫₀^∞ f(r) J_ν(k r) dr

of `f_r` sampled on a log-spaced grid `r[j] = r₀ exp(j Δln)`.

Returns the transform on the reciprocal log-grid
`kr[j] = (1/r[end]) exp(j Δln)`.

Uses the plain `dr` measure.  For the standard Hankel pair with `r dr`
measure, pre-multiply the input by `r`.

# Arguments
- `f_r`  : input samples (length N, may be complex)
- `dln`  : log-spacing Δln = log(r[2]/r[1])
- `nu`   : Bessel order (any real number)
"""
function fftlog_hankel(f_r::AbstractVector, dln::Real, nu::Real)
    # Handle negative integer orders via J_{-n}(x) = (-1)^n J_n(x).
    # This avoids Gamma-function poles at non-positive integer arguments.
    if nu < 0 && nu == round(nu)
        n = round(Int, -nu)
        sign = iseven(n) ? 1 : -1
        return sign .* fftlog_hankel(f_r, dln, Float64(n))
    end

    N = length(f_r)

    # FFTW-ordered frequency indices: 0,1,...,⌊(N-1)/2⌋, -⌊N/2⌋,...,-1
    n_idx = [n <= (N-1)÷2 ? n : n - N for n in 0:N-1]
    q     = @. (2π / (N * dln)) * n_idx

    # Filter kernel (following mcfit / Hamilton 2000):
    #
    #   û(q) = 2^{iq} Γ((ν+1+iq)/2) / Γ((ν+1-iq)/2)
    #
    # where q = 2πn/(NΔln) are real DFT frequency values.
    #
    # This is the Mellin transform M[t J_ν(t)](iq) which arises when
    # the Hankel kernel K(t) = t J_ν(t) is used with the d(ln r) measure.
    # Since the caller pre-multiplies by r for the dr measure, the
    # effective transform is ∫ f(r) J_ν(kr) dr.
    #
    # Key property: |û(q)| = 1 for all q (conjugate Gamma pair).
    U_c = map(q) do qj
        a = complex((nu + 1.0) / 2, +qj / 2)   # (ν+1+iq)/2
        b = complex((nu + 1.0) / 2, -qj / 2)   # (ν+1-iq)/2
        exp(loggamma(a) - loggamma(b) + im * qj * log(2.0))
    end

    F = fft(complex.(f_r))
    A = ifft(F .* U_c)

    # The output must be reversed: the "natural" FFTLog output grid starts
    # at 1/r_min (huge), and reversing maps it to the useful grid
    # kr[j] = (1/r_max) exp(j Δln), matching the dev convention.
    #
    # IMPORTANT: the mcfit-convention kernel K(t) = t J_ν(t) introduces an
    # extra factor of kr in the output.  The raw output is kr × H_ν[f](kr).
    # The caller must divide by kr to obtain the pure Hankel transform.
    return reverse(A)
end


# ═══════════════════════════════════════════════════════════════
# Step 2 (part 2) — TE and TM Expansion Coefficients
#
# From the tex formulation (§4–§5), the TE/TM coefficients are:
#
#  A^TE_m(kr) = (i/2)[H_{m+1}[rE_r] + H_{m-1}[rE_r]]
#             + (1/2)[H_{m-1}[rE_θ] − H_{m+1}[rE_θ]]
#
#  A^TM_m(kr) = −(kz/k)/2 [H_{m-1}[rE_r] − H_{m+1}[rE_r]]
#             − i(kz/k)/2 [H_{m-1}[rE_θ] + H_{m+1}[rE_θ]]
#
# where H_ν[f](kr) = ∫₀^∞ f(r) J_ν(kr r) dr.
#
# Each mode m requires 4 FFTLog calls (2 if E_θ ≡ 0).
# Only m ≥ 0 modes are computed; negative modes use symmetry later.
# ═══════════════════════════════════════════════════════════════

"""
    compute_TE_TM_coeffs(Em_r, Em_theta, m_pos, r, k)
    -> (A_TE, A_TM, kr_grid)

Compute TE and TM expansion coefficients for modes m = 0,...,M_max.

# Arguments
- `Em_r[Nr, M_max+1]`    : r-component modal amplitudes
- `Em_theta[Nr, M_max+1]`: θ-component modal amplitudes
- `m_pos`                 : mode indices [0, 1, ..., M_max]
- `r`                     : log-spaced radial grid (Vector{Float64})
- `k`                     : wavenumber 2π/λ

# Returns
- `A_TE[Nr, M_max+1]`  : TE coefficients on the kr grid
- `A_TM[Nr, M_max+1]`  : TM coefficients on the kr grid
- `kr_grid[Nr]`         : reciprocal log-grid from FFTLog
"""
function compute_TE_TM_coeffs(Em_r::Matrix{ComplexF64},
                               Em_theta::Matrix{ComplexF64},
                               m_pos::Vector{Int},
                               r::Vector{Float64},
                               k::Float64)
    Nr      = length(r)
    N_modes = length(m_pos)
    dln     = log(r[2] / r[1])

    # Output kr grid (reciprocal log-grid, same as FFTLog output)
    kr_grid = exp.(log(1.0 / r[end]) .+ dln .* (0:Nr-1))

    # kz/k at each kr point: real for propagating, 0 for evanescent
    kz_over_k = @. sqrt(max(1.0 - (kr_grid / k)^2, 0.0))

    # Detect E_theta ≡ 0 → skip 2 FFTLog calls per mode
    etheta_zero = iszero(Em_theta)

    A_TE = zeros(ComplexF64, Nr, N_modes)
    A_TM = zeros(ComplexF64, Nr, N_modes)

    # Warm FFTW plan cache (thread-safety)
    fftlog_hankel(zeros(ComplexF64, Nr), dln, 0.0)

    Threads.@threads for idx in 1:N_modes
        m  = m_pos[idx]
        fr = r .* Em_r[:, idx]      # r × E_{m,r} — integrand for Hankel

        # FFTLog returns kr × H_ν[f](kr).  We collect the raw (kr×H) values
        # and divide by kr once at the end.
        raw_mp1_r = fftlog_hankel(fr, dln, Float64(m + 1))  # kr × H_{m+1}[r E_r]
        raw_mm1_r = fftlog_hankel(fr, dln, Float64(m - 1))  # kr × H_{m-1}[r E_r]

        if etheta_zero
            # A^TE = (i/2)(H_{m+1} + H_{m-1})[r E_r]  (θ-terms vanish)
            # A^TM = -(kz/k)/2 (H_{m-1} - H_{m+1})[r E_r]
            A_TE[:, idx] .= (im/2) .* (raw_mp1_r .+ raw_mm1_r) ./ kr_grid
            A_TM[:, idx] .= -(kz_over_k ./ 2) .* (raw_mm1_r .- raw_mp1_r) ./ kr_grid
        else
            fth = r .* Em_theta[:, idx]  # r × E_{m,θ}

            raw_mm1_th = fftlog_hankel(fth, dln, Float64(m - 1))
            raw_mp1_th = fftlog_hankel(fth, dln, Float64(m + 1))

            A_TE[:, idx] .= ((im/2) .* (raw_mp1_r .+ raw_mm1_r) .+
                             (1.0/2) .* (raw_mm1_th .- raw_mp1_th)) ./ kr_grid

            A_TM[:, idx] .= (-(kz_over_k ./ 2) .* (raw_mm1_r .- raw_mp1_r) .-
                             im .* (kz_over_k ./ 2) .* (raw_mm1_th .+ raw_mp1_th)) ./ kr_grid
        end
    end

    return A_TE, A_TM, kr_grid
end


# ═══════════════════════════════════════════════════════════════
# Step 3 — Propagation + Negative-m Symmetry Reconstruction
#
# Propagation:
#   Ã_m(kr) = [A^TE_m(kr) + A^TM_m(kr)] · exp(ikz f)
#   kz = √(k² - kr²),  zeroed for kr > k (evanescent).
#
# Symmetry (from §6 of the tex, E_{-m} = σ̂ E_m):
#   y-pol: Ã_{-m} = (-1)^m (A^TE_m − A^TM_m) · exp(ikz f)
#   x-pol: Ã_{-m} = (-1)^m (A^TM_m − A^TE_m) · exp(ikz f)
#
# Input:  A_TE, A_TM  [Nkr × (M_max+1)]  for m = 0,...,M_max
# Output: A_tilde     [Nkr × (2M_max+1)]  for m = -M_max,...,M_max
#         m_full = [-M_max, ..., -1, 0, 1, ..., M_max]
#         Column index j corresponds to mode m = j - M_max - 1
# ═══════════════════════════════════════════════════════════════

"""
    propagate_and_symmetrize(A_TE, A_TM, m_pos, kr_grid, k, f; polarization=:y)
    -> (A_tilde, m_full)

Combine TE+TM, apply propagation phase exp(ikz f), and reconstruct
negative-m modes via symmetry.

Returns A_tilde[Nkr, 2M_max+1] for m = -M_max:M_max.
Column j corresponds to mode m = j - M_max - 1.

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

    # Propagation phase: exp(ikz f) for propagating, 0 for evanescent
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

        # Negative mode via symmetry
        if m > 0
            idx_neg = -m + M_max + 1
            s = iseven(m) ? 1.0 : -1.0   # (-1)^m
            if polarization == :y
                A_tilde[:, idx_neg] .= s .* (ate .- atm) .* prop
            else  # :x
                A_tilde[:, idx_neg] .= s .* (atm .- ate) .* prop
            end
        end
    end

    return A_tilde, m_full
end

end  # module CyFFP
