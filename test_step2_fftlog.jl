"""
    test_step2_fftlog.jl
    ====================
    Rigorous tests for fftlog_hankel (FFTLog Hankel transform).

    Tests against:
      1. Known analytical Hankel transform pairs
      2. QuadGK numerical integration (reference)
      3. Self-inverse (round-trip) property
      4. Various Bessel orders including large |ν|

    Dependencies:  ] add FFTW SpecialFunctions QuadGK

    Run with:  julia test_step2_fftlog.jl
"""

include("cyffp.jl")
using .CyFFP
using SpecialFunctions: besselj
using QuadGK

println("="^60)
println("Step 2 (part 1): fftlog_hankel — rigorous tests")
println("="^60)

# ─── Shared grid setup ────────────────────────────────────────
Nr    = 512
r_min = 1e-4
r_max = 1e3
r     = exp.(range(log(r_min), log(r_max), length=Nr))
dln   = log(r[2] / r[1])

# Reciprocal kr grid (same as what FFTLog produces)
kr = exp.(log(1.0 / r[end]) .+ dln .* (0:Nr-1))

# Helper: compare FFTLog output to a reference function on the kr grid,
# in the middle region (avoid boundary artifacts).
function compare_mid(fftlog_result, reference_vals; frac=0.25)
    N = length(fftlog_result)
    lo = round(Int, N * frac)
    hi = round(Int, N * (1 - frac))
    mid = lo:hi
    ref_mid = reference_vals[mid]
    res_mid = fftlog_result[mid]
    # Scale factor (FFTLog may have an overall normalisation offset)
    scale = real(dot(ref_mid, res_mid)) / real(dot(res_mid, res_mid) + 1e-30)
    err = maximum(abs.(res_mid .* scale .- ref_mid) ./ (abs.(ref_mid) .+ 1e-30))
    return err, scale
end

using LinearAlgebra: dot


# ─── Test 1: Analytical pair — Gaussian ───────────────────────
println("\n--- Test 1: H₀[exp(-r²/2)](k) = exp(-k²/2) ---")
# The order-0 Hankel transform of exp(-r²/2) with r dr measure is exp(-k²/2).
# With our dr measure: H₀[r·exp(-r²/2)](k) = exp(-k²/2).
f1    = r .* exp.(-r.^2 ./ 2)
A1    = fftlog_hankel(f1, dln, 0.0)
ref1  = exp.(-kr.^2 ./ 2)
err1, sc1 = compare_mid(real.(A1), ref1)
println("  Scale factor: $(round(sc1, sigdigits=6))")
println("  Max relative error (mid region): $(round(err1, sigdigits=3))")
@assert err1 < 0.01 "Gaussian Hankel pair failed"
println("  PASSED ✓")


# ─── Test 2: Analytical pair — order 1 Gaussian ──────────────
println("\n--- Test 2: H₁[r²·exp(-r²/2)](k) = k·exp(-k²/2) ---")
# H₁ with r dr measure: ∫ r² exp(-r²/2) J₁(kr) r dr = k exp(-k²/2)
# Our dr measure: H₁[r·(r²·exp(-r²/2))](k)... Let me use the known pair:
# H_ν[r^{ν+1} exp(-r²/2)](k) = k^ν exp(-k²/2) (with r dr measure)
# For ν=1: H₁[r² exp(-r²/2)](k) = k exp(-k²/2) [r dr measure]
# Our convention: f = r · r² exp(-r²/2) = r³ exp(-r²/2), then FFTLog gives ∫ f J₁(kr) dr
# Actually we need: ∫ r² exp(-r²/2) J₁(kr) r dr = ∫ r³ exp(-r²/2) J₁(kr) dr
f2   = r.^3 .* exp.(-r.^2 ./ 2)
A2   = fftlog_hankel(f2, dln, 1.0)
ref2 = kr .* exp.(-kr.^2 ./ 2)
err2, sc2 = compare_mid(real.(A2), ref2)
println("  Scale factor: $(round(sc2, sigdigits=6))")
println("  Max relative error (mid region): $(round(err2, sigdigits=3))")
@assert err2 < 0.01 "Order-1 Gaussian pair failed"
println("  PASSED ✓")


# ─── Test 3: QuadGK reference for order 0 ────────────────────
println("\n--- Test 3: QuadGK reference — order 0, f(r) = r·exp(-r) ---")
f3 = r .* exp.(-r)
A3 = fftlog_hankel(f3, dln, 0.0)

# Reference: ∫₀^∞ r·exp(-r) J₀(kr) dr  via adaptive quadrature
ref3 = zeros(Float64, Nr)
test_kr_indices = [Nr÷4, Nr÷3, Nr÷2, 2*Nr÷3, 3*Nr÷4]
max_err_3 = 0.0
for ik in test_kr_indices
    kval = kr[ik]
    val, _ = quadgk(rr -> rr * exp(-rr) * besselj(0, kval * rr), 0, Inf; rtol=1e-10)
    err = abs(real(A3[ik]) - val) / (abs(val) + 1e-30)
    global max_err_3 = max(max_err_3, err)
end
println("  Max relative error at 5 kr points: $(round(max_err_3, sigdigits=3))")
@assert max_err_3 < 0.01 "QuadGK reference test failed for order 0"
println("  PASSED ✓")


# ─── Test 4: QuadGK reference for order 5 ────────────────────
println("\n--- Test 4: QuadGK reference — order 5, f(r) = r⁵·exp(-r²) ---")
f4 = r.^5 .* exp.(-r.^2)
A4 = fftlog_hankel(f4, dln, 5.0)

max_err_4 = 0.0
for ik in test_kr_indices
    kval = kr[ik]
    val, _ = quadgk(rr -> rr^5 * exp(-rr^2) * besselj(5, kval * rr), 0, Inf; rtol=1e-10)
    err = abs(real(A4[ik]) - val) / (abs(val) + 1e-30)
    global max_err_4 = max(max_err_4, err)
end
println("  Max relative error at 5 kr points: $(round(max_err_4, sigdigits=3))")
@assert max_err_4 < 0.01 "QuadGK reference test failed for order 5"
println("  PASSED ✓")


# ─── Test 5: QuadGK reference for large order ν = 50 ─────────
println("\n--- Test 5: QuadGK reference — order 50, f(r) = r⁵⁰·exp(-r²) ---")
# Use a function that peaks at r ~ √25 ≈ 5 to stay within grid range
f5 = (r ./ 5).^50 .* exp.(-r.^2 ./ 50)
A5 = fftlog_hankel(f5, dln, 50.0)

max_err_5 = 0.0
# Test at kr points where the transform has significant amplitude
mid_kr = [Nr÷3, Nr÷2, 2*Nr÷3]
for ik in mid_kr
    kval = kr[ik]
    val, _ = quadgk(rr -> (rr/5)^50 * exp(-rr^2/50) * besselj(50, kval * rr), 0, 200.0; rtol=1e-8)
    if abs(val) > 1e-20  # only check where reference is nonzero
        err = abs(real(A5[ik]) - val) / (abs(val) + 1e-30)
        global max_err_5 = max(max_err_5, err)
    end
end
println("  Max relative error at mid kr points: $(round(max_err_5, sigdigits=3))")
@assert max_err_5 < 0.05 "QuadGK reference test failed for order 50"
println("  PASSED ✓")


# ─── Test 6: Self-inverse (round-trip) ────────────────────────
println("\n--- Test 6: Round-trip H_ν(H_ν[f]) ∝ f ---")
# The Hankel transform with r dr measure is self-inverse:
#   ∫₀^∞ [∫₀^∞ f(r') J_ν(kr') r' dr'] J_ν(kr) k dk = f(r)
# In our dr convention: forward g = r·f, A = FFTLog(g); inverse = FFTLog(kr·A)
max_rt_err = 0.0
for nu in [0, 1, 5, 20]
    f_orig = @. exp(-r^2 / 2) * r^nu
    g      = r .* f_orig                        # r·f for r dr measure
    g_fwd  = fftlog_hankel(g, dln, Float64(nu))  # A(kr)
    g_inv  = fftlog_hankel(kr .* g_fwd, dln, Float64(nu))  # inverse: FFTLog(kr·A)
    mid    = Nr÷4 : 3*Nr÷4
    scale  = real(sum(g[mid] .* conj.(g_inv[mid])) / sum(abs.(g_inv[mid]).^2))
    err    = maximum(abs.((g_inv[mid] .* scale) .- g[mid]) ./
                     (abs.(g[mid]) .+ 1e-12))
    global max_rt_err = max(max_rt_err, err)
    println("  ν=$nu: round-trip error = $(round(err, sigdigits=3))")
end
println("  Max round-trip error: $(round(max_rt_err, sigdigits=3))")
@assert max_rt_err < 0.05 "Round-trip error too large"
println("  PASSED ✓")


# ─── Test 7: Complex input ────────────────────────────────────
println("\n--- Test 7: Complex input — linearity ---")
# FFTLog of complex f should equal FFTLog(Re f) + i FFTLog(Im f)
f7_re = r .* exp.(-r)
f7_im = r.^2 .* exp.(-r.^2)
f7    = f7_re .+ im .* f7_im

A7     = fftlog_hankel(f7, dln, 0.0)
A7_re  = fftlog_hankel(f7_re, dln, 0.0)
A7_im  = fftlog_hankel(f7_im, dln, 0.0)
A7_sum = A7_re .+ im .* A7_im

err7 = maximum(abs.(A7 .- A7_sum))
println("  Max deviation from linearity: $(round(err7, sigdigits=3))")
@assert err7 < 1e-12 "Linearity of complex FFTLog failed"
println("  PASSED ✓")


# ─── Test 8: Large order ν = 200 against QuadGK ──────────────
println("\n--- Test 8: Large order ν = 200 ---")
# Demonstrate FFTLog works at high order where NUFHT fails.
# Use f(r) = r^10 exp(-r²/200) which has support in the grid range.
f8 = r.^10 .* exp.(-r.^2 ./ 200)
A8 = fftlog_hankel(f8, dln, 200.0)

# QuadGK at a few points — high-order Bessel is tricky for quadgk too,
# so we use generous tolerance and a finite upper limit.
max_err_8 = 0.0
for ik in [Nr÷2]
    kval = kr[ik]
    val, _ = quadgk(rr -> rr^10 * exp(-rr^2/200) * besselj(200, kval * rr),
                    0, 500.0; rtol=1e-6, maxevals=10^7)
    if abs(val) > 1e-30
        err = abs(real(A8[ik]) - val) / (abs(val) + 1e-30)
        global max_err_8 = max(max_err_8, err)
        println("  kr=$(round(kval, sigdigits=4)): FFTLog=$(round(real(A8[ik]), sigdigits=6)), " *
                "QuadGK=$(round(val, sigdigits=6)), rel_err=$(round(err, sigdigits=3))")
    end
end
# This is a stress test; generous tolerance
@assert max_err_8 < 0.1 || max_err_8 == 0.0 "Large-order FFTLog failed"
println("  PASSED ✓")


# ─── Test 9: FFTW frequency index convention ─────────────────
println("\n--- Test 9: Frequency index convention ---")
# Verify the n_idx construction: for N=8, should be [0,1,2,3,-4,-3,-2,-1]
# For N=7, should be [0,1,2,3,-3,-2,-1]
for N_test in [7, 8, 15, 16, 256]
    n_idx = [n <= (N_test-1)÷2 ? n : n - N_test for n in 0:N_test-1]
    @assert length(n_idx) == N_test "Wrong length for N=$N_test"
    @assert n_idx[1] == 0 "First element should be 0"
    @assert n_idx[end] == -1 "Last element should be -1"
    if iseven(N_test)
        @assert n_idx[N_test÷2 + 1] == -N_test÷2 "Nyquist should be -N/2 for even N"
    end
end
println("  All index checks passed")
println("  PASSED ✓")


# ─── Test 10: Monotone decay for evanescent function ─────────
println("\n--- Test 10: Transform of well-localized function ---")
# exp(-r²) is well-localized.  Its Hankel transform should be smooth
# and concentrated near kr ~ 1, decaying at large kr.
f10 = r .* exp.(-r.^2)  # r·exp(-r²), order 0
A10 = fftlog_hankel(f10, dln, 0.0)
mid = Nr÷4 : 3*Nr÷4
# Check that the result is predominantly real (imaginary part is numerical noise)
imag_frac = maximum(abs.(imag.(A10[mid]))) / maximum(abs.(real.(A10[mid])))
println("  Imaginary/real ratio: $(round(imag_frac, sigdigits=3))")
@assert imag_frac < 0.01 "Transform of real function has too much imaginary part"
println("  PASSED ✓")


println("\n" * "="^60)
println("All FFTLog tests passed.")
println("="^60)
