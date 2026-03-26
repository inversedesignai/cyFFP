"""
    CyFFP — Cylindrical Far-Field Propagation
    ==========================================
    Near-to-far-field transform via Vector Cylindrical Harmonics.

    This module is built incrementally; each step of the pipeline
    is added, tested, and validated before proceeding to the next.
"""
module CyFFP

using FFTW

export angular_decompose

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

end  # module CyFFP
