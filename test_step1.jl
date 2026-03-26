"""
    test_step1.jl
    =============
    Rigorous tests for angular_decompose (Step 1 of CyFFP pipeline).

    Run with:  julia test_step1.jl
"""

include("cyffp.jl")
using .CyFFP
using FFTW
using SpecialFunctions: besselj

println("="^60)
println("Step 1: angular_decompose — rigorous tests")
println("="^60)

# ─── Shared setup ─────────────────────────────────────────────
Nr     = 32
Ntheta = 64
r      = range(0.1, 10.0, length=Nr)
theta  = range(0.0, 2π, length=Ntheta+1)[1:end-1]  # uniform, excludes 2π

# Helper: build 2D field from modal amplitudes
# Helper: check that modes outside a given set are zero
function check_sparsity(Em, m_pos, active_modes::Set{Int}; atol=1e-12)
    for (idx, m) in enumerate(m_pos)
        col_norm = maximum(abs.(Em[:, idx]))
        if m in active_modes
            @assert col_norm > atol "Mode m=$m should be nonzero but is $(col_norm)"
        else
            @assert col_norm < atol "Mode m=$m should be zero but has norm $(col_norm)"
        end
    end
end


# ─── Test 1: Single pure mode ────────────────────────────────
println("\n--- Test 1: Single pure mode E_r = f(r) e^{i·3·θ} ---")
m_test = 3
f_r    = collect(r) .^ 2   # arbitrary radial profile
Er_1   = [f_r[jr] * exp(im * m_test * th) for jr in 1:Nr, th in theta]
Et_1   = zeros(ComplexF64, Nr, Ntheta)

M_max_1 = 10
Em_r_1, Em_t_1, m_pos_1 = angular_decompose(Er_1, Et_1, M_max_1)

# Mode m=3 should recover f(r); all others should be zero
check_sparsity(Em_r_1, m_pos_1, Set([m_test]))
err_1 = maximum(abs.(Em_r_1[:, m_test+1] .- f_r))
println("  Max error in recovered mode: $(round(err_1, sigdigits=3))")
@assert err_1 < 1e-12 "Single-mode recovery failed"
println("  PASSED ✓")


# ─── Test 2: Two modes superposed ─────────────────────────────
println("\n--- Test 2: Two modes m=1 and m=7 ---")
f1 = sin.(collect(r))
f2 = cos.(collect(r))
Er_2 = [f1[jr]*exp(im*1*th) + f2[jr]*exp(im*7*th) for jr in 1:Nr, th in theta]
Et_2 = zeros(ComplexF64, Nr, Ntheta)

Em_r_2, _, m_pos_2 = angular_decompose(Er_2, Et_2, 10)
check_sparsity(Em_r_2, m_pos_2, Set([1, 7]))
err_2a = maximum(abs.(Em_r_2[:, 2] .- f1))   # m=1 at index 2
err_2b = maximum(abs.(Em_r_2[:, 8] .- f2))   # m=7 at index 8
println("  Mode m=1 error: $(round(err_2a, sigdigits=3))")
println("  Mode m=7 error: $(round(err_2b, sigdigits=3))")
@assert err_2a < 1e-12 && err_2b < 1e-12 "Two-mode recovery failed"
println("  PASSED ✓")


# ─── Test 3: x-polarized field ────────────────────────────────
println("\n--- Test 3: x-polarized field u(r)(cosθ r̂ - sinθ θ̂) ---")
# x̂ = cosθ r̂ - sinθ θ̂
# cosθ = (e^{iθ} + e^{-iθ})/2        → m=+1 coeff: 1/2,  m=-1 coeff: 1/2
# -sinθ = -(e^{iθ} - e^{-iθ})/(2i)   → m=+1 coeff: -1/(2i) = i/2
#                                        m=-1 coeff:  1/(2i) = -i/2
# So: E_{+1,r} = u/2,    E_{-1,r} = u/2
#     E_{+1,θ} = iu/2,   E_{-1,θ} = -iu/2
u_r = exp.(-collect(r))
Er_3  = [u_r[jr] * cos(th) for jr in 1:Nr, th in theta]
Et_3  = [-u_r[jr] * sin(th) for jr in 1:Nr, th in theta]

Em_r_3, Em_t_3, m_pos_3 = angular_decompose(ComplexF64.(Er_3), ComplexF64.(Et_3), 5)

# Only m=1 should be nonzero in the m≥0 extraction
# (m=-1 is not extracted since we only get m≥0)
check_sparsity(Em_r_3, m_pos_3, Set([1]))
check_sparsity(Em_t_3, m_pos_3, Set([1]))

err_3r = maximum(abs.(Em_r_3[:, 2] .- u_r ./ 2))
err_3t = maximum(abs.(Em_t_3[:, 2] .- (im .* u_r ./ 2)))
println("  E_{1,r} error: $(round(err_3r, sigdigits=3))  (expected u/2)")
println("  E_{1,θ} error: $(round(err_3t, sigdigits=3))  (expected iu/2)")
@assert err_3r < 1e-12 && err_3t < 1e-12 "x-pol mode amplitudes wrong"
println("  PASSED ✓")


# ─── Test 4: y-polarized field ────────────────────────────────
println("\n--- Test 4: y-polarized field u(r)(sinθ r̂ + cosθ θ̂) ---")
# ŷ = sinθ r̂ + cosθ θ̂
# sinθ = (e^{iθ} - e^{-iθ})/(2i)  →  m=+1 coeff = 1/(2i) = -i/2
#                                      m=-1 coeff = -1/(2i) = i/2
# cosθ = (e^{iθ} + e^{-iθ})/2     →  m=+1 coeff = 1/2
#                                      m=-1 coeff = 1/2
# So: E_{+1,r} = -iu/2, E_{-1,r} = iu/2
#     E_{+1,θ} = u/2,   E_{-1,θ} = u/2
Er_4 = [u_r[jr] * sin(th) for jr in 1:Nr, th in theta]
Et_4 = [u_r[jr] * cos(th) for jr in 1:Nr, th in theta]

Em_r_4, Em_t_4, _ = angular_decompose(ComplexF64.(Er_4), ComplexF64.(Et_4), 5)

check_sparsity(Em_r_4, m_pos_3, Set([1]))
check_sparsity(Em_t_4, m_pos_3, Set([1]))

err_4r = maximum(abs.(Em_r_4[:, 2] .- (-im .* u_r ./ 2)))
err_4t = maximum(abs.(Em_t_4[:, 2] .- (u_r ./ 2)))
println("  E_{1,r} error: $(round(err_4r, sigdigits=3))  (expected -iu/2)")
println("  E_{1,θ} error: $(round(err_4t, sigdigits=3))  (expected u/2)")
@assert err_4r < 1e-12 && err_4t < 1e-12 "y-pol mode amplitudes wrong"
println("  PASSED ✓")


# ─── Test 5: Jacobi-Anger expansion ──────────────────────────
println("\n--- Test 5: Jacobi-Anger: exp(ia cosθ) = Σ i^m J_m(a) e^{imθ} ---")
a_val  = 3.7
M_max_5 = 15
# Scalar field: E_r = exp(i a cosθ), E_θ = 0
Er_5 = [exp(im * a_val * cos(th)) for _ in 1:Nr, th in theta]
Et_5 = zeros(ComplexF64, Nr, Ntheta)

Em_r_5, _, m_pos_5 = angular_decompose(Er_5, Et_5, M_max_5)

# Each mode m should be i^m J_m(a) (constant over r since the field is r-independent)
max_err_5 = 0.0
for (idx, m) in enumerate(m_pos_5)
    expected = (im)^m * besselj(m, a_val)
    err = maximum(abs.(Em_r_5[:, idx] .- expected))
    max_err_5 = max(max_err_5, err)
end
println("  Max mode error over m=0..$M_max_5: $(round(max_err_5, sigdigits=3))")
@assert max_err_5 < 1e-10 "Jacobi-Anger decomposition failed"
println("  PASSED ✓")


# ─── Test 6: Manual DFT vs angular_decompose ──────────────────
println("\n--- Test 6: Manual DFT matches angular_decompose ---")
# Build a random complex field and verify against explicit DFT formula
Er_6 = randn(ComplexF64, Nr, Ntheta)
Et_6 = randn(ComplexF64, Nr, Ntheta)
M_max_6 = 20

Em_r_6, Em_t_6, m_pos_6 = angular_decompose(Er_6, Et_6, M_max_6)

# Manual DFT for each m
max_err_6 = 0.0
for (idx, m) in enumerate(m_pos_6)
    for jr in 1:Nr
        manual_r = sum(Er_6[jr, jt] * exp(-im * m * theta[jt]) for jt in 1:Ntheta) / Ntheta
        manual_t = sum(Et_6[jr, jt] * exp(-im * m * theta[jt]) for jt in 1:Ntheta) / Ntheta
        max_err_6 = max(max_err_6, abs(Em_r_6[jr, idx] - manual_r))
        max_err_6 = max(max_err_6, abs(Em_t_6[jr, idx] - manual_t))
    end
end
println("  Max error vs manual DFT: $(round(max_err_6, sigdigits=3))")
@assert max_err_6 < 1e-10 "FFT does not match manual DFT"
println("  PASSED ✓")


# ─── Test 7: Negative-mode content accessible from FFT ────────
println("\n--- Test 7: Negative modes in full FFT ---")
# Verify that mode m=-3 from the full FFT matches the expected value
# for a field with a pure m=-3 component.
m_neg = -3
Er_7  = [f_r[jr] * exp(im * m_neg * th) for jr in 1:Nr, th in theta]
Et_7  = zeros(ComplexF64, Nr, Ntheta)

# angular_decompose only returns m≥0, so m=-3 should show up as zero
Em_r_7, _, m_pos_7 = angular_decompose(Er_7, Et_7, 10)
check_sparsity(Em_r_7, m_pos_7, Set{Int}())  # nothing in m≥0

# But the full FFT should have it at FFTW index mod(-3, Ntheta)+1
full_fft = fft(Er_7, 2) ./ Ntheta
idx_neg3 = mod(m_neg, Ntheta) + 1
err_7 = maximum(abs.(full_fft[:, idx_neg3] .- f_r))
println("  Mode m=-3 in full FFT error: $(round(err_7, sigdigits=3))")
@assert err_7 < 1e-12 "Negative mode not at expected FFTW index"
println("  PASSED ✓")


# ─── Test 8: M_max = 0 (DC mode only) ────────────────────────
println("\n--- Test 8: M_max = 0 (DC mode only) ---")
# Constant field: E_r = u(r), E_θ = 0
Er_8 = [u_r[jr] + 0im for jr in 1:Nr, th in theta]
Et_8 = zeros(ComplexF64, Nr, Ntheta)

Em_r_8, Em_t_8, m_pos_8 = angular_decompose(Er_8, Et_8, 0)
@assert length(m_pos_8) == 1 && m_pos_8[1] == 0
err_8 = maximum(abs.(Em_r_8[:, 1] .- u_r))
println("  m=0 mode error: $(round(err_8, sigdigits=3))")
@assert err_8 < 1e-12 "DC mode recovery failed"
println("  PASSED ✓")


# ─── Test 9: Parseval's theorem ───────────────────────────────
println("\n--- Test 9: Parseval's theorem ---")
# Σ_m |E_m|² = (1/Nθ) Σ_l |E(θ_l)|²
# (energy in mode space = energy in physical space / Nθ)
Er_9 = randn(ComplexF64, Nr, Ntheta)
Et_9 = zeros(ComplexF64, Nr, Ntheta)

# Get ALL modes (m = 0 to Ntheta-1 to span the full FFT)
M_max_9 = Ntheta ÷ 2   # max extractable without aliasing
Em_r_9, _, m_pos_9 = angular_decompose(Er_9, Et_9, M_max_9)

# Also get negative modes from the full FFT
full_fft_9 = fft(Er_9, 2) ./ Ntheta

# Parseval: Σ_{m=0}^{Nθ-1} |Em[m]|² = (1/Nθ) Σ_l |E(θ_l)|²  per radial point
for jr in 1:Nr
    phys_energy = sum(abs2.(Er_9[jr, :])) / Ntheta
    mode_energy = sum(abs2.(full_fft_9[jr, :]))
    rel_err = abs(phys_energy - mode_energy) / (phys_energy + 1e-30)
    @assert rel_err < 1e-10 "Parseval failed at r[$jr]: rel_err=$rel_err"
end
println("  Parseval's theorem holds at all radial points")
println("  PASSED ✓")


# ─── Test 10: Nyquist assertion ───────────────────────────────
println("\n--- Test 10: Nyquist assertion (should error) ---")
Er_10 = zeros(ComplexF64, Nr, 8)
Et_10 = zeros(ComplexF64, Nr, 8)
caught = false
try
    angular_decompose(Er_10, Et_10, 5)  # needs 2*5+1=11 but only have 8
catch e
    if isa(e, AssertionError)
        caught = true
        println("  Correctly caught: $(e.msg)")
    end
end
@assert caught "Should have thrown AssertionError for insufficient Ntheta"
println("  PASSED ✓")


# ─── Test 11: Oblique plane wave (vector, y-polarized) ────────
println("\n--- Test 11: y-pol oblique plane wave exp(ikx r cosθ) ŷ ---")
kx   = 2.5
M_11 = 12
# E_r = exp(ikx r cosθ) sinθ,  E_θ = exp(ikx r cosθ) cosθ
Er_11 = [exp(im * kx * r[jr] * cos(th)) * sin(th) for jr in 1:Nr, th in theta]
Et_11 = [exp(im * kx * r[jr] * cos(th)) * cos(th) for jr in 1:Nr, th in theta]

Em_r_11, Em_t_11, m_pos_11 = angular_decompose(ComplexF64.(Er_11), ComplexF64.(Et_11), M_11)

# Analytical: from Jacobi-Anger + trig identities,
#   E_{m,r} = [i^{m-1} J_{m-1}(kx r) - i^{m+1} J_{m+1}(kx r)] / (2i)
#   E_{m,θ} = [i^{m-1} J_{m-1}(kx r) + i^{m+1} J_{m+1}(kx r)] / 2
max_err_11 = 0.0
for (idx, m) in enumerate(m_pos_11)
    for jr in 1:Nr
        Jmm1 = besselj(m-1, kx * r[jr])
        Jmp1 = besselj(m+1, kx * r[jr])
        c1 = (im)^(m-1)
        c2 = (im)^(m+1)
        exp_r = (c1 * Jmm1 - c2 * Jmp1) / (2im)
        exp_t = (c1 * Jmm1 + c2 * Jmp1) / 2
        max_err_11 = max(max_err_11, abs(Em_r_11[jr, idx] - exp_r))
        max_err_11 = max(max_err_11, abs(Em_t_11[jr, idx] - exp_t))
    end
end
println("  Max error vs analytical (m=0..$M_11): $(round(max_err_11, sigdigits=3))")
@assert max_err_11 < 1e-10 "y-pol Jacobi-Anger decomposition failed"
println("  PASSED ✓")


# ─── Test 12: Symmetry check E_{-m} = σ̂ E_m ──────────────────
println("\n--- Test 12: y-pol symmetry E_{-m,r} = -E_{m,r}, E_{-m,θ} = +E_{m,θ} ---")
# Using the full FFT, verify the symmetry for the y-pol oblique wave from Test 11
full_r = fft(ComplexF64.(Er_11), 2) ./ Ntheta
full_t = fft(ComplexF64.(Et_11), 2) ./ Ntheta

max_err_12 = 0.0
for m in 1:M_11
    idx_pos = m + 1
    idx_neg = mod(-m, Ntheta) + 1
    # E_{-m,r} should equal -E_{m,r}
    err_r = maximum(abs.(full_r[:, idx_neg] .+ full_r[:, idx_pos]))
    # E_{-m,θ} should equal +E_{m,θ}
    err_t = maximum(abs.(full_t[:, idx_neg] .- full_t[:, idx_pos]))
    max_err_12 = max(max_err_12, err_r, err_t)
end
println("  Max symmetry error (m=1..$M_11): $(round(max_err_12, sigdigits=3))")
@assert max_err_12 < 1e-10 "y-pol symmetry violated"
println("  PASSED ✓")


println("\n" * "="^60)
println("All Step 1 tests passed.")
println("="^60)
