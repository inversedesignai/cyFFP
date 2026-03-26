"""
    CyFFP — Cylindrical Far-Field Propagation
    ==========================================
    Scalar near-to-far-field transform via cylindrical harmonics.

    For a linearly polarized field E = u(r,θ) p̂, the Cartesian component
    along p̂ satisfies the scalar wave equation exactly.  The pipeline
    propagates this scalar u through:
      1. Angular FFT → modes u_m(r)
      2. Scalar Hankel transform → spectral coefficients a_m(kr)
      3. Propagation → ã_m(kr) = a_m(kr) e^{ikz f}
      4. Graf shift → local modes B_l(kr)
      5. Inverse Hankel → b_l(ρ)
      6. Angular synthesis → PSF u(ρ,ψ)

    The TE/TM decomposition is NOT used for the scalar PSF — it conflates
    different Bessel structures and gives incorrect results when naively
    combined as Ã = A^TE + A^TM.
"""
module CyFFP

using FFTW
using SpecialFunctions: loggamma, besselj

export angular_decompose, fftlog_hankel,
       compute_scalar_coeffs, propagate_scalar,
       graf_shift

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
# Step 2 — Scalar Spectral Coefficients
#
# For a scalar field u(r,θ) = Σ_m u_m(r) e^{imθ}, the spectral
# coefficient in the cylindrical harmonic expansion is:
#
#   a_m(kr) = ∫₀^∞ u_m(r) J_m(kr r) r dr = H_m[r u_m](kr)
#
# This is a SINGLE Hankel transform at order m — much simpler and
# more correct than the TE/TM approach which uses 4 transforms at
# orders m±1 and conflates different Bessel structures.
#
# For linearly polarized light E = u(r,θ) p̂, the Cartesian component
# along p̂ is exactly the scalar field u, so this gives the correct PSF.
# ═══════════════════════════════════════════════════════════════

"""
    compute_scalar_coeffs(u_m, m_pos, r) -> (a_m, kr_grid)

Compute scalar spectral coefficients a_m(kr) = H_m[r u_m](kr) for
modes m = 0, ..., M_max.  One FFTLog call per mode at order m.

# Arguments
- `u_m[Nr, M_max+1]` : scalar modal amplitudes u_m(r)
- `m_pos`             : mode indices [0, 1, ..., M_max]
- `r`                 : log-spaced radial grid

# Returns
- `a_m[Nr, M_max+1]` : spectral coefficients on the kr grid
- `kr_grid[Nr]`       : reciprocal log-grid from FFTLog
"""
function compute_scalar_coeffs(u_m::Matrix{ComplexF64},
                                m_pos::Vector{Int},
                                r::Vector{Float64})
    Nr      = length(r)
    N_modes = length(m_pos)
    dln     = log(r[2] / r[1])

    kr_grid = exp.(log(1.0 / r[end]) .+ dln .* (0:Nr-1))

    a_m = zeros(ComplexF64, Nr, N_modes)

    # Warm FFTW plan cache (thread-safety)
    fftlog_hankel(zeros(ComplexF64, Nr), dln, 0.0)

    Threads.@threads for idx in 1:N_modes
        m   = m_pos[idx]
        g   = r .* u_m[:, idx]    # r × u_m — Hankel integrand with r dr measure

        # FFTLog returns kr × H_m[g](kr).  Divide by kr to get a_m.
        raw = fftlog_hankel(g, dln, Float64(m))
        a_m[:, idx] .= raw ./ kr_grid
    end

    return a_m, kr_grid
end


# ═══════════════════════════════════════════════════════════════
# Step 3 — Propagation + Symmetry Reconstruction (Scalar)
#
# Propagation:
#   ã_m(kr) = a_m(kr) · exp(ikz f)
#   kz = √(k² - kr²),  zeroed for kr > k (evanescent).
#
# Symmetry: for a field with u(r,-θ) = u(r,θ) (any linearly
# polarized illumination of an axisymmetric structure with oblique
# tilt in the xz-plane), we have u_{-m} = u_m, hence ã_{-m} = ã_m.
# No sign change, no TE/TM distinction.
# ═══════════════════════════════════════════════════════════════

"""
    propagate_scalar(a_m, m_pos, kr_grid, k, f) -> (a_tilde, m_full)

Apply propagation phase and reconstruct negative-m modes via symmetry.

Returns a_tilde[Nkr, 2M_max+1] for m = -M_max:M_max.
For a symmetric scalar field: ã_{-m} = ã_m (no sign change).
"""
function propagate_scalar(a_m::Matrix{ComplexF64},
                           m_pos::Vector{Int},
                           kr_grid::Vector{Float64},
                           k::Float64, f::Float64)
    M_max = m_pos[end]
    Nkr   = length(kr_grid)

    kz   = @. sqrt(complex(k^2 - kr_grid^2))
    prop = @. ifelse(kr_grid < k, exp(im * real(kz) * f), zero(ComplexF64))

    N_full  = 2M_max + 1
    a_tilde = zeros(ComplexF64, Nkr, N_full)
    m_full  = collect(-M_max:M_max)

    for (ip, m) in enumerate(m_pos)
        coeff = @view a_m[:, ip]
        propagated = coeff .* prop

        idx_pos = m + M_max + 1
        a_tilde[:, idx_pos] .= propagated

        # Scalar symmetry: ã_{-m} = ã_m
        if m > 0
            idx_neg = -m + M_max + 1
            a_tilde[:, idx_neg] .= propagated
        end
    end

    return a_tilde, m_full
end


# ═══════════════════════════════════════════════════════════════
# Step 4 — Graf's Addition Theorem (Modal Basis Shift)
#
# Re-expand the far field in a local basis centered at (x₀, 0):
#
#   B_l(kr) = Σ_{m=-M_max}^{M_max} Ã_{m+l}(kr) J_m(kr x₀)
#
# for |l| ≤ L_max.  The weight J_m(kr x₀) decays exponentially
# for |m| > kr x₀, so the inner sum is naturally truncated at
# m_cut = min(M_max, ceil(kr x₀) + buffer).
#
# Normal incidence (x₀→0): J_m(0) = δ_{m,0} ⟹ B_l = Ã_l.
# ═══════════════════════════════════════════════════════════════

"""
    _besselj_range(m_max, x) -> Vector{Float64}

Compute J_0(x), J_1(x), ..., J_{m_max}(x) via Miller backward recurrence
with overflow protection.  Works for ANY x (no fallback to individual
besselj calls).

Uses the self-normalizing identity: 1 = J_0(x) + 2[J_2(x) + J_4(x) + ...]
to avoid needing besselj(0, x) for normalization.

~10× faster than individual besselj calls at all x.
"""
function _besselj_range(m_max::Int, x::Float64)
    if m_max < 0; return Float64[]; end
    if abs(x) < 1e-30
        out = zeros(Float64, m_max + 1)
        out[1] = 1.0   # J_0(0) = 1
        return out
    end

    # Start above both m_max and the turning point |x|.
    m_start = max(m_max, ceil(Int, abs(x))) + max(30, ceil(Int, 15 * sqrt(max(1.0, abs(x)))))

    jnp1     = 0.0
    jn       = 1.0
    out      = zeros(Float64, m_max + 1)
    norm_sum = 0.0   # accumulates J_0 + 2(J_2 + J_4 + ...)

    for m in m_start:-1:0
        jnm1 = (2m / x) * jn - jnp1

        if m <= m_max
            out[m + 1] = jn
        end

        # Accumulate normalization identity: 1 = J_0 + 2Σ_{k≥1} J_{2k}
        if m == 0
            norm_sum += jn
        elseif iseven(m)
            norm_sum += 2 * jn
        end

        jnp1 = jn
        jn   = jnm1

        # Overflow protection: rescale ALL accumulated values by the same
        # factor.  This preserves ratios (the normalization divides out).
        if abs(jn) > 1e200
            jn       *= 1e-200
            jnp1     *= 1e-200
            norm_sum *= 1e-200
            out      .*= 1e-200
        end
    end

    # Normalize: out[j] / norm_sum gives the true J_m(x)
    out ./= norm_sum
    return out
end

"""
    graf_shift(A_tilde, m_full, kr_grid, x0, L_max; k=Inf) -> B[Nkr, 2L_max+1]

Apply Graf's addition theorem at every kr point:
    B_l(kr) = Σ_m Ã_{m+l}(kr) J_m(kr x₀)
for |l| ≤ L_max.

If `k` is provided, evanescent kr points (kr > k) are skipped
(A_tilde is zero there from Step 3, so B = 0 automatically).

# Arguments
- `A_tilde[Nkr, 2M_max+1]`: propagated coefficients (m = -M_max:M_max)
- `m_full`: mode list [-M_max, ..., M_max]
- `kr_grid`: log-spaced kr grid
- `x0`: lateral shift = f tan(α)
- `L_max`: local mode truncation
- `k`: wavenumber (optional; enables evanescent skip)

# Returns
- `B[Nkr, 2L_max+1]`: local modal coefficients, columns = l = -L_max:L_max
"""
function graf_shift(A_tilde::Matrix{ComplexF64},
                     m_full::Vector{Int},
                     kr_grid::Vector{Float64},
                     x0::Float64,
                     L_max::Int;
                     k::Float64 = Inf)
    M_max = (length(m_full) - 1) ÷ 2
    Nkr   = length(kr_grid)
    Nl    = 2L_max + 1
    B     = zeros(ComplexF64, Nkr, Nl)

    # Thread over kr points (each is independent).
    Threads.@threads for ikr in 1:Nkr
        # Skip evanescent: A_tilde = 0 for kr > k (from Step 3)
        if kr_grid[ikr] > k
            continue   # B[ikr, :] stays zero
        end

        kr_x0 = kr_grid[ikr] * x0

        # Bessel truncation: J_m(z) ≈ 0 for |m| > z + buffer
        m_cut = min(M_max, ceil(Int, abs(kr_x0)) + 20)

        # Compute J_0..J_{m_cut} via overflow-safe recurrence
        Jp = _besselj_range(m_cut, kr_x0)

        # Build full Bessel weight vector for m = -m_cut:m_cut
        # J_{-m}(x) = (-1)^m J_m(x)
        Jw = Vector{Float64}(undef, 2m_cut + 1)
        @inbounds for m in 0:m_cut
            Jw[m + m_cut + 1] = Jp[m + 1]                            # m ≥ 0
            Jw[-m + m_cut + 1] = iseven(m) ? Jp[m + 1] : -Jp[m + 1] # m < 0
        end

        # Compute B_l = Σ_m Ã_{m+l} J_m for each l
        @inbounds for (li, l) in enumerate(-L_max:L_max)
            acc = zero(ComplexF64)
            m_lo = max(-m_cut, -M_max - l)
            m_hi = min( m_cut,  M_max - l)
            @simd for m in m_lo:m_hi
                acc += A_tilde[ikr, m + l + M_max + 1] * Jw[m + m_cut + 1]
            end
            B[ikr, li] = acc
        end
    end
    return B
end

end  # module CyFFP
