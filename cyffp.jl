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

export angular_decompose, fftlog_hankel

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
# where the filter kernel is:
#   û(q) = Γ((ν+1+q)/2) / Γ((ν+1-q)/2) · 2^q
# with q = 2πn/(N·Δln) being the DFT frequency values.
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
    N = length(f_r)

    # FFTW-ordered frequency indices: 0,1,...,⌊(N-1)/2⌋, -⌊N/2⌋,...,-1
    n_idx = [n <= (N-1)÷2 ? n : n - N for n in 0:N-1]
    q     = @. (2π / (N * dln)) * n_idx

    # Filter kernel: û(q) = Γ((ν+1+q)/2) / Γ((ν+1-q)/2) · 2^q
    # where q = 2πn/(N·Δln) are the DFT frequency values (real).
    # The Gamma arguments are made complex to handle the loggamma
    # branch cut for negative real arguments at large |ν|.
    U_c = map(q) do qj
        a = complex((nu + 1.0 + qj) / 2)
        b = complex((nu + 1.0 - qj) / 2)
        exp(loggamma(a) - loggamma(b) + qj * log(2.0))
    end

    F = fft(complex.(f_r))
    return ifft(F .* U_c)
end

end  # module CyFFP
