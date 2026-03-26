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
       compute_scalar_coeffs, neumann_shift_coeffs, propagate_scalar,
       graf_shift, inverse_hankel, angular_synthesis

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

# ─── Pre-allocated FFTLog workspace (for zero-allocation hot loops) ───
#
# Holds per-thread buffers and in-place FFTW plans.  Created once per
# (N, dln) pair, reused across all Bessel orders within that grid.

struct _FFTLogWorkspace
    N::Int
    buf::Vector{ComplexF64}      # work buffer (FFT/IFFT in-place)
    U_c::Vector{ComplexF64}      # kernel buffer (rewritten per order)
    q::Vector{Float64}           # DFT frequency indices (fixed for given N, dln)
    pfft!::Any                   # in-place FFT plan (MEASURE)
    pifft!::Any                  # in-place IFFT plan (MEASURE)
end

function _make_workspace(N::Int, dln::Float64)
    buf  = zeros(ComplexF64, N)
    U_c  = zeros(ComplexF64, N)
    n_idx = [n <= (N-1)÷2 ? n : n - N for n in 0:N-1]
    q    = @. (2π / (N * dln)) * n_idx
    # FFTW.MEASURE benchmarks several FFT algorithms and picks the fastest.
    # Plan creation is slower (~1s) but each FFT execution is 20-50% faster.
    pfft!  = plan_fft!(buf; flags=FFTW.MEASURE)
    pifft! = plan_ifft!(buf; flags=FFTW.MEASURE)
    return _FFTLogWorkspace(N, buf, U_c, q, pfft!, pifft!)
end

"""
    _precompute_kernels(q, m_orders) -> Matrix{ComplexF64}

Precompute FFTLog filter kernels for all Bessel orders using the
recurrence U_{m+2}(q) = U_m(q) × (m+1+iq)/(m+1-iq).

Only orders ≥ 0 are computed.  Negative orders are handled at the
call site via J_{-n} = (-1)^n J_n.

Base cases U_0, U_1 use loggamma.  All subsequent orders use the
recurrence (simple complex multiplication, ~40× faster than loggamma).
"""
function _precompute_kernels(q::Vector{Float64}, m_orders::Vector{Int})
    N = length(q)
    M_max = maximum(m_orders)
    @assert minimum(m_orders) >= 0 "Only non-negative orders supported"

    # Compute base cases via loggamma
    kernels = zeros(ComplexF64, N, M_max + 1)

    for m_base in 0:min(1, M_max)
        nu = Float64(m_base)
        @inbounds for i in 1:N
            qj = q[i]
            a = complex((nu + 1.0) / 2, +qj / 2)
            b = complex((nu + 1.0) / 2, -qj / 2)
            kernels[i, m_base + 1] = exp(loggamma(a) - loggamma(b) + im * qj * log(2.0))
        end
    end

    # Recurrence: U_{m+2}(q) = U_m(q) × (m+1+iq)/(m+1-iq)
    # Derived from Γ(z+1) = zΓ(z) applied to the kernel ratio.
    for m in 2:M_max
        m_prev = m - 2  # same-parity predecessor
        c = Float64(m_prev + 1)
        @inbounds for i in 1:N
            kernels[i, m + 1] = kernels[i, m_prev + 1] * complex(c, q[i]) / complex(c, -q[i])
        end
    end

    return kernels
end

"""
    _fftlog_with_kernel!(out, ws, f_r, kernel, neg_sign)

FFTLog Hankel transform using a pre-computed kernel vector.
Zero allocations.  `neg_sign` is ±1 for the J_{-n} = (-1)^n J_n sign.
"""
function _fftlog_with_kernel!(out::AbstractVector{ComplexF64},
                               ws::_FFTLogWorkspace,
                               f_r::AbstractVector,
                               kernel::AbstractVector{ComplexF64},
                               neg_sign::Int)
    N = ws.N

    # Copy input into work buffer
    @inbounds for i in 1:N
        ws.buf[i] = ComplexF64(f_r[i])
    end

    # In-place FFT
    ws.pfft! * ws.buf

    # Multiply by pre-computed kernel in-place
    @inbounds for i in 1:N
        ws.buf[i] *= kernel[i]
    end

    # In-place IFFT
    ws.pifft! * ws.buf

    # Reverse into output with optional sign
    if neg_sign == 1
        @inbounds for i in 1:N
            out[i] = ws.buf[N - i + 1]
        end
    else
        @inbounds for i in 1:N
            out[i] = -ws.buf[N - i + 1]
        end
    end

    return out
end

"""
    _fftlog_hankel!(out, ws, f_r, nu)

Zero-allocation FFTLog Hankel transform.  Writes result into `out`.
Uses pre-allocated workspace `ws` (from `_make_workspace`).

Same mathematical operation as `fftlog_hankel`, but avoids all
heap allocations in the hot path.
"""
function _fftlog_hankel!(out::AbstractVector{ComplexF64},
                          ws::_FFTLogWorkspace,
                          f_r::AbstractVector,
                          nu::Float64)
    N = ws.N

    # Handle negative integer orders without recursion
    nu_eff = nu
    neg_sign = 1
    if nu < 0 && nu == round(nu)
        n = round(Int, -nu)
        neg_sign = iseven(n) ? 1 : -1
        nu_eff = Float64(n)
    end

    # Compute kernel in-place (only part that depends on ν)
    @inbounds for i in 1:N
        qj = ws.q[i]
        a = complex((nu_eff + 1.0) / 2, +qj / 2)
        b = complex((nu_eff + 1.0) / 2, -qj / 2)
        ws.U_c[i] = exp(loggamma(a) - loggamma(b) + im * qj * log(2.0))
    end

    # Copy input into work buffer
    @inbounds for i in 1:N
        ws.buf[i] = ComplexF64(f_r[i])
    end

    # In-place FFT
    ws.pfft! * ws.buf

    # Multiply by kernel in-place
    @inbounds for i in 1:N
        ws.buf[i] *= ws.U_c[i]
    end

    # In-place IFFT
    ws.pifft! * ws.buf

    # Reverse into output, with optional sign flip
    if neg_sign == 1
        @inbounds for i in 1:N
            out[i] = ws.buf[N - i + 1]
        end
    else
        @inbounds for i in 1:N
            out[i] = -ws.buf[N - i + 1]
        end
    end

    return out
end


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
    # Public API: allocates workspace each call (for backward compatibility).
    # Hot loops should use _fftlog_hankel! with pre-allocated workspaces.

    # Handle negative integer orders via J_{-n}(x) = (-1)^n J_n(x).
    if nu < 0 && nu == round(nu)
        n = round(Int, -nu)
        sign = iseven(n) ? 1 : -1
        return sign .* fftlog_hankel(f_r, dln, Float64(n))
    end

    N = length(f_r)
    n_idx = [n <= (N-1)÷2 ? n : n - N for n in 0:N-1]
    q     = @. (2π / (N * dln)) * n_idx

    U_c = map(q) do qj
        a = complex((nu + 1.0) / 2, +qj / 2)
        b = complex((nu + 1.0) / 2, -qj / 2)
        exp(loggamma(a) - loggamma(b) + im * qj * log(2.0))
    end

    F = fft(complex.(f_r))
    A = ifft(F .* U_c)

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

    # Precompute all FFTLog kernels via recurrence (sequential, ~40× faster
    # than per-call loggamma).  Kernels are for |m| (non-negative); negative
    # orders use J_{-n} = (-1)^n J_n sign flip.
    m_abs_max = maximum(abs, m_pos)
    n_idx = [n <= (Nr-1)÷2 ? n : n - Nr for n in 0:Nr-1]
    q     = @. (2π / (Nr * dln)) * n_idx
    kernels = _precompute_kernels(q, collect(0:m_abs_max))

    # Pre-allocate one workspace per thread (in-place FFTW MEASURE plans).
    nt = Threads.maxthreadid()
    workspaces = [_make_workspace(Nr, dln) for _ in 1:nt]
    g_bufs     = [Vector{ComplexF64}(undef, Nr) for _ in 1:nt]

    Threads.@threads for idx in 1:N_modes
        tid = Threads.threadid()
        ws  = workspaces[tid]
        g   = g_bufs[tid]

        m = m_pos[idx]
        m_abs    = abs(m)
        neg_sign = (m < 0 && isodd(m_abs)) ? -1 : 1

        # g = r × u_m (Hankel integrand with r dr measure) — zero-alloc
        @inbounds for j in 1:Nr
            g[j] = r[j] * u_m[j, idx]
        end

        # FFTLog with pre-computed kernel — zero-alloc, no loggamma
        _fftlog_with_kernel!(view(a_m, :, idx), ws, g,
                             view(kernels, :, m_abs + 1), neg_sign)

        # Divide by kr to get a_m — zero-alloc
        @inbounds for j in 1:Nr
            a_m[j, idx] /= kr_grid[j]
        end
    end

    return a_m, kr_grid
end


# ═══════════════════════════════════════════════════════════════
# Step 2 (Neumann fast path) — Shift Theorem for LPA Fields
#
# For near fields of the form u(r,θ) = t(r) exp(ikₓ r cosθ),
# where t(r) is a radially symmetric transmission, the Neumann
# addition formula converts the per-mode Hankel transforms into:
#
#   a_m(kr) = (i^m / 2π) ∫₀^{2π} T₀(c(φ)) e^{-imφ} dφ
#
# where T₀(q) = H₀[r t(r)](q) is a SINGLE order-0 Hankel transform,
# and c(φ) = √(kr² + kₓ² - 2 kr kₓ cosφ).
#
# Cost: O(Nr log Nr) for T₀, plus O(Nkr · Nφ log Nφ) for the
# per-kr interpolation and FFTs.  This replaces O(M_max · Nr log Nr)
# per-mode FFTLog calls with a single one.
# ═══════════════════════════════════════════════════════════════

"""
    neumann_shift_coeffs(t_r, r, k, alpha, M_max) -> (a_m, kr_grid, m_pos)

Compute scalar spectral coefficients for an LPA near field
`u(r,θ) = t(r) exp(ikₓ r cosθ)` via the Neumann addition formula.

Replaces `compute_scalar_coeffs` when the near field factors as a
radial transmission × oblique plane wave tilt.  Uses a **single**
order-0 FFTLog call instead of M_max individual calls.

# Arguments
- `t_r[Nr]`   : radial transmission t(r) on log-spaced r grid
- `r[Nr]`     : log-spaced radial grid
- `k`         : wavenumber 2π/λ
- `alpha`     : oblique incidence angle (radians)
- `M_max`     : maximum angular mode index

# Returns
- `a_m[Nr, M_max+1]` : spectral coefficients for m = 0, ..., M_max
- `kr_grid[Nr]`       : reciprocal log-grid (same as compute_scalar_coeffs)
- `m_pos[M_max+1]`    : mode indices [0, 1, ..., M_max]
"""
function neumann_shift_coeffs(t_r::AbstractVector,
                               r::Vector{Float64},
                               k::Float64, alpha::Float64,
                               M_max::Int)
    Nr  = length(r)
    dln = log(r[2] / r[1])
    kx  = k * sin(alpha)

    kr_grid = exp.(log(1.0 / r[end]) .+ dln .* (0:Nr-1))
    m_pos   = collect(0:M_max)

    # ─── Step A: single order-0 FFTLog for T₀ ────────────────
    # T₀(q) = ∫₀^∞ r t(r) J₀(qr) dr = H₀[r t(r)](q)
    g    = r .* complex.(t_r)
    raw  = fftlog_hankel(g, dln, 0.0)
    T0   = raw ./ kr_grid   # divide by kr (FFTLog output convention)

    # ─── Step B: per-kr interpolation + FFT over φ ────────────
    N_phi = max(2M_max + 1, 256)
    N_phi = 1 << ceil(Int, log2(N_phi))   # round up to power of 2
    phi   = range(0.0, 2π, length=N_phi+1)[1:end-1]

    a_m = zeros(ComplexF64, Nr, M_max + 1)

    # Precompute i^m factors (cyclic: 1, i, -1, -i)
    im_factors = [Complex(0.0, 1.0)^m for m in m_pos]

    # Log-interpolation setup for T₀
    log_kr_min = log(kr_grid[1])

    # Pre-allocate per-thread FFT buffers and plans
    nt = Threads.maxthreadid()
    phi_bufs   = [Vector{ComplexF64}(undef, N_phi) for _ in 1:nt]
    phi_plans  = [plan_fft!(phi_bufs[t]; flags=FFTW.MEASURE) for t in 1:nt]

    # Precompute cos(φ) values
    cos_phi = [cos(phi[s]) for s in 1:N_phi]

    Threads.@threads for ikr in 1:Nr
        tid = Threads.threadid()
        buf = phi_bufs[tid]
        pf  = phi_plans[tid]

        kv = kr_grid[ikr]

        # Interpolate T₀ at c(φ) = √(kr² + kₓ² - 2 kr kₓ cosφ)
        @inbounds for s in 1:N_phi
            c_sq = kv^2 + kx^2 - 2*kv*kx*cos_phi[s]
            c = sqrt(max(c_sq, 0.0))

            if c < kr_grid[1] || c > kr_grid[end]
                buf[s] = zero(ComplexF64)
            else
                lc    = log(c)
                idx_f = (lc - log_kr_min) / dln + 1.0
                j0    = clamp(floor(Int, idx_f), 1, Nr - 1)
                w     = idx_f - j0
                buf[s] = (1 - w) * T0[j0] + w * T0[j0 + 1]
            end
        end

        # FFT over φ: F[m] = Σ_s buf[s] e^{-2πi ms/Nφ}
        pf * buf

        # Extract modes: a_m(kr) = i^m / Nφ × F[m]
        @inbounds for idx in 1:M_max+1
            m = m_pos[idx]
            a_m[ikr, idx] = im_factors[idx] * buf[m + 1] / N_phi
        end
    end

    return a_m, kr_grid, m_pos
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
# tilt in the xz-plane), we have u_{-m} = u_m.  Combined with
# J_{-m}(x) = (-1)^m J_m(x), this gives ã_{-m} = (-1)^m ã_m.
# ═══════════════════════════════════════════════════════════════

"""
    propagate_scalar(a_m, m_pos, kr_grid, k, f) -> (a_tilde, m_full)

Apply propagation phase and reconstruct negative-m modes via symmetry.

Returns a_tilde[Nkr, 2M_max+1] for m = -M_max:M_max.
For a symmetric scalar field: ã_{-m} = (-1)^m ã_m.
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

        # Scalar symmetry: ã_{-m} = (-1)^m ã_m
        # The (-1)^m comes from J_{-m}(x) = (-1)^m J_m(x) in the
        # cylindrical harmonic basis, combined with u_{-m} = u_m.
        if m > 0
            idx_neg = -m + M_max + 1
            sign = iseven(m) ? 1 : -1
            a_tilde[:, idx_neg] .= sign .* propagated
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


# ═══════════════════════════════════════════════════════════════
# Step 5 — Inverse Hankel Transform
#
# Transform from spectral (kr) to spatial (ρ) domain:
#
#   b_l(ρ) = ∫₀^∞ B_l(kr) J_l(kr ρ) kr dkr
#
# This is a Hankel transform with the kr dkr measure.  Using
# fftlog_hankel (which computes ∫ f(x) J_ν(xy) dx with plain dx
# measure and returns y × H_ν[f](y)):
#
#   1. Pre-multiply: f(kr) = kr × B_l(kr)
#   2. raw = fftlog_hankel(f, dln, l)  →  returns ρ × b_l(ρ)
#   3. b_l = raw / ρ
#
# The output ρ grid is the reciprocal of the kr grid, which equals
# the original r grid from Step 2.
#
# Only 2L_max + 1 FFTLog calls are needed (typically ~27).
# ═══════════════════════════════════════════════════════════════

"""
    inverse_hankel(B, L_max, kr_grid) -> (b, rho_grid)

Compute b_l(ρ) = ∫ B_l(kr) J_l(kr ρ) kr dkr for each local mode
l = -L_max, ..., L_max via FFTLog.

# Arguments
- `B[Nkr, 2L_max+1]` : local spectral coefficients from graf_shift
- `L_max`              : local mode truncation
- `kr_grid[Nkr]`       : log-spaced kr grid

# Returns
- `b[Nrho, 2L_max+1]` : local spatial coefficients on the ρ grid
- `rho_grid[Nrho]`      : output ρ grid (= reciprocal of kr grid)
"""
function inverse_hankel(B::Matrix{ComplexF64},
                         L_max::Int,
                         kr_grid::Vector{Float64})
    Nkr = length(kr_grid)
    Nl  = 2L_max + 1
    @assert size(B) == (Nkr, Nl) "B size mismatch: got $(size(B)), expected ($Nkr, $Nl)"

    dln = log(kr_grid[2] / kr_grid[1])

    # Output ρ grid: reciprocal of kr grid
    rho_grid = exp.(log(1.0 / kr_grid[end]) .+ dln .* (0:Nkr-1))

    b = zeros(ComplexF64, Nkr, Nl)

    # Precompute kernels for orders 0..L_max via recurrence.
    # Negative orders use kernel at |l| with sign (-1)^|l|.
    n_idx = [n <= (Nkr-1)÷2 ? n : n - Nkr for n in 0:Nkr-1]
    q     = @. (2π / (Nkr * dln)) * n_idx
    kernels = _precompute_kernels(q, collect(0:L_max))

    # Pre-allocate one workspace per thread.
    nt = Threads.maxthreadid()
    workspaces = [_make_workspace(Nkr, dln) for _ in 1:nt]
    f_bufs     = [Vector{ComplexF64}(undef, Nkr) for _ in 1:nt]

    Threads.@threads for li in 1:Nl
        tid = Threads.threadid()
        ws  = workspaces[tid]
        f   = f_bufs[tid]
        l   = li - L_max - 1

        # For negative l: use kernel at |l| with sign (-1)^|l|
        l_abs    = abs(l)
        neg_sign = (l < 0 && isodd(l_abs)) ? -1 : 1

        # f = kr × B_l (pre-multiply for kr dkr measure) — zero-alloc
        @inbounds for j in 1:Nkr
            f[j] = kr_grid[j] * B[j, li]
        end

        # FFTLog with pre-computed kernel — zero-alloc
        _fftlog_with_kernel!(view(b, :, li), ws, f,
                             view(kernels, :, l_abs + 1), neg_sign)

        # Divide by ρ to get b_l(ρ) — zero-alloc
        @inbounds for j in 1:Nkr
            b[j, li] /= rho_grid[j]
        end
    end

    return b, rho_grid
end


# ═══════════════════════════════════════════════════════════════
# Step 6 — Angular Synthesis
#
# Reconstruct the 2D PSF from local modes:
#
#   u(ρ_j, ψ_s) = Σ_{l=-L_max}^{L_max} b_l(ρ_j) e^{il ψ_s}
#
# where ψ_s = 2π s / N_ψ,  s = 0, ..., N_ψ - 1.
#
# This is an inverse DFT over the mode index l at each ρ point.
# FFTW convention: ifft(X)[s] = (1/N) Σ_k X[k] e^{+2πi ks/N}
# We want: u[s] = Σ_l b_l e^{+2πi ls/N_ψ} = N_ψ × ifft(b_fft)[s]
#
# Modes are placed in FFTW order:
#   l ≥ 0 → index l + 1
#   l < 0 → index N_ψ + l + 1
# ═══════════════════════════════════════════════════════════════

"""
    angular_synthesis(b, L_max, N_psi) -> (u, psi)

Reconstruct u(ρ,ψ) = Σ_l b_l(ρ) e^{ilψ} via IFFT over the mode
index at each radial point.

# Arguments
- `b[Nrho, 2L_max+1]` : local spatial coefficients from inverse_hankel
- `L_max`               : local mode truncation
- `N_psi`               : number of azimuthal output points (≥ 2L_max+1)

# Returns
- `u[Nrho, N_psi]` : complex field; PSF intensity = |u|²
- `psi[N_psi]`      : azimuthal angles ψ_s = 2π s / N_ψ
"""
function angular_synthesis(b::Matrix{ComplexF64},
                            L_max::Int,
                            N_psi::Int)
    Nrho = size(b, 1)
    Nl   = 2L_max + 1
    @assert size(b, 2) == Nl "b has $(size(b,2)) columns, expected $Nl"
    @assert N_psi >= Nl "N_psi=$N_psi too small for L_max=$L_max (need ≥ $Nl)"

    # Arrange modes in FFTW order
    b_fft = zeros(ComplexF64, Nrho, N_psi)
    for (li, l) in enumerate(-L_max:L_max)
        col = l >= 0 ? l + 1 : N_psi + l + 1
        b_fft[:, col] .= b[:, li]
    end

    # u[s] = Σ_l b_l e^{2πi ls/N_ψ} = N_ψ × ifft(b_fft)[s]
    u = ifft(b_fft, 2) .* N_psi

    psi = collect(range(0.0, 2π, length=N_psi+1)[1:end-1])

    return u, psi
end

end  # module CyFFP
