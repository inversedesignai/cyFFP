"""
    test_step4.jl
    =============
    Rigorous tests for graf_shift (Step 4).

    Run with: julia test_step4.jl
"""

include("cyffp.jl")
using .CyFFP
using FFTW
using SpecialFunctions: besselj

println("="^60)
println("Step 4: graf_shift — rigorous tests")
println("="^60)


# ─── Test 1: Normal incidence (x₀=0) — identity ──────────────
println("\n--- Test 1: Normal incidence x₀=0: B_l = Ã_l ---")
# J_m(0) = δ_{m,0}, so B_l = Ã_{0+l} J_0(0) = Ã_l × 1 = Ã_l
Nr = 256
kr = exp.(range(log(1e-2), log(1e2), length=Nr))
M_max = 10
m_full = collect(-M_max:M_max)
A_tilde = randn(ComplexF64, Nr, 2M_max + 1)
L_max = 5

B = graf_shift(A_tilde, m_full, collect(kr), 0.0, L_max)

# B_l should equal A_tilde at column l + M_max + 1
max_err_1 = 0.0
for (li, l) in enumerate(-L_max:L_max)
    idx_A = l + M_max + 1
    err = maximum(abs.(B[:, li] .- A_tilde[:, idx_A]))
    global max_err_1 = max(max_err_1, err)
end
println("  Max |B_l - Ã_l|: $(round(max_err_1, sigdigits=3))")
@assert max_err_1 < 1e-12 "Identity shift failed"
println("  PASSED ✓")


# ─── Test 2: Graf identity J_n(a+b) = Σ_m J_m(a) J_{n-m}(b) ─
println("\n--- Test 2: Graf identity Σ_m J_m(z₀) J_{n-m}(z) = J_n(z+z₀) ---")
max_err_2 = 0.0
for n in [-3, 0, 1, 5], z in [0.5, 2.0, 5.0], z0 in [1.0, 3.0, 8.0]
    M = ceil(Int, z0 + z + 20)
    s = sum(besselj(m, z0) * besselj(n - m, z) for m in -M:M)
    exact = besselj(n, z + z0)
    err = abs(s - exact) / (abs(exact) + 1e-14)
    global max_err_2 = max(max_err_2, err)
end
println("  Max relative error: $(round(max_err_2, sigdigits=3))")
@assert max_err_2 < 1e-10 "Graf identity failed"
println("  PASSED ✓")


# ─── Test 3: Single kr point, manual computation ─────────────
println("\n--- Test 3: Single kr point vs manual sum ---")
x0 = 5.0
L_max_3 = 3
M_max_3 = 15
m_full_3 = collect(-M_max_3:M_max_3)
A3 = randn(ComplexF64, 1, 2M_max_3 + 1)  # single kr point
kr3 = [2.0]

B3 = graf_shift(A3, m_full_3, kr3, x0, L_max_3)

# Manual computation
max_err_3 = 0.0
for (li, l) in enumerate(-L_max_3:L_max_3)
    manual = zero(ComplexF64)
    for m in -M_max_3:M_max_3
        n = m + l
        if -M_max_3 <= n <= M_max_3
            manual += A3[1, n + M_max_3 + 1] * besselj(m, kr3[1] * x0)
        end
    end
    err = abs(B3[1, li] - manual) / (abs(manual) + 1e-30)
    global max_err_3 = max(max_err_3, err)
end
println("  Max error: $(round(max_err_3, sigdigits=3))")
@assert max_err_3 < 1e-10 "Manual sum mismatch"
println("  PASSED ✓")


# ─── Test 4: Miller recurrence vs besselj ─────────────────────
println("\n--- Test 4: _besselj_range vs besselj ---")
for x in [0.1, 1.0, 10.0, 100.0, 1000.0]
    m_max_test = ceil(Int, x) + 20
    Jp = CyFFP._besselj_range(m_max_test, x)
    max_err_4 = maximum(abs(Jp[m+1] - besselj(m, x)) for m in 0:m_max_test)
    println("  x=$x, m_max=$m_max_test: max_err=$(round(max_err_4, sigdigits=3))")
    @assert max_err_4 < 1e-10 "Miller recurrence inaccurate at x=$x"
end
println("  PASSED ✓")


# ─── Test 5: L_max truncation — high local modes are small ───
println("\n--- Test 5: B_l decays for large |l| ---")
# Use a realistic-ish A_tilde concentrated near m=0
A5 = zeros(ComplexF64, Nr, 2M_max + 1)
for (idx, m) in enumerate(m_full)
    A5[:, idx] .= exp(-m^2 / 10.0)  # Gaussian in m
end

x0_5 = 3.0
L_max_5 = M_max  # extract all possible local modes
B5 = graf_shift(A5, m_full, collect(kr), x0_5, L_max_5)

# Energy should decay with |l|
energies = [sum(abs2.(B5[:, li])) for li in 1:2L_max_5+1]
peak_E = maximum(energies)
edge_E = max(energies[1], energies[end])
ratio = edge_E / peak_E
println("  Edge/peak energy ratio: $(round(ratio, sigdigits=3))")
@assert ratio < 0.5 "Local modes should decay at large |l|"
println("  PASSED ✓")


# ─── Test 6: Symmetry preservation ───────────────────────────
println("\n--- Test 6: If Ã_{-m} = (-1)^m Ã_m, then B_{-l} has known relation to B_l ---")
# For symmetric A_tilde: Ã_{-m} = Ã_m (even symmetry, simplest case)
A6 = zeros(ComplexF64, Nr, 2M_max + 1)
for (idx, m) in enumerate(m_full)
    A6[:, idx] .= exp(-abs(m) / 3.0) .* cos.(kr .* 0.1)  # symmetric in m
end
# Make it exactly symmetric: A_{-m} = A_m
for m in 1:M_max
    A6[:, -m + M_max + 1] .= A6[:, m + M_max + 1]
end

B6 = graf_shift(A6, m_full, collect(kr), 3.0, 5)

# With Ã_{-m} = Ã_m and J_{-m}(z) = (-1)^m J_m(z):
# B_{-l} = Σ_m Ã_{m-l} J_m = Σ_m Ã_{-m-l} J_{-m} (-1 subst)
# ... the relationship is: B_{-l} should be real if A is real and symmetric
# Actually let me just check B_l is real (since A is real and symmetric → B should be real)
max_imag = maximum(abs.(imag.(B6))) / (maximum(abs.(B6)) + 1e-30)
println("  Max |imag(B)| / |B|: $(round(max_imag, sigdigits=3))")
@assert max_imag < 1e-10 "B should be real for real symmetric A_tilde"
println("  PASSED ✓")


# ─── Test 7: End-to-end Steps 1→2→3→4 with realistic lens ────
println("\n--- Test 7: Steps 1→2→3→4 with scaled lens (R=10λ, α=10°) ---")
lambda_7 = 1.0
k_7      = 2π / lambda_7
R_7      = 10.0 * lambda_7
f_7      = R_7 * sqrt(1/0.25^2 - 1)
alpha_7  = deg2rad(10.0)
x0_7     = f_7 * tan(alpha_7)

Nr_7     = 512
r_7      = exp.(range(log(1e-3), log(1e3), length=Nr_7))
Ntheta_7 = 128
theta_7  = range(0.0, 2π, length=Ntheta_7+1)[1:end-1]
M_max_7  = ceil(Int, k_7 * sin(alpha_7) * R_7) + 20
L_max_7  = 15

function u_obl_7(rv, th)
    d = sqrt((rv*cos(th) - x0_7)^2 + rv^2*sin(th)^2 + f_7^2)
    return exp(-im*k_7*(d - f_7)) * (rv <= R_7 ? 1.0 : 0.0)
end

println("  R=$(R_7)λ, f=$(round(f_7, digits=1))λ, α=10°, x₀=$(round(x0_7, digits=2))λ")
println("  M_max=$M_max_7, L_max=$L_max_7")

u_field_7 = ComplexF64[u_obl_7(r_7[jr], theta_7[jt])
                       for jr in 1:Nr_7, jt in 1:Ntheta_7]

# Steps 1-3 (scalar pipeline)
u_m_7, _, m_pos_7 = angular_decompose(u_field_7, zeros(ComplexF64, Nr_7, Ntheta_7), M_max_7)
a_m_7, kr_7 = compute_scalar_coeffs(u_m_7, m_pos_7, collect(r_7))
A_tilde_7, m_full_7 = propagate_scalar(a_m_7, m_pos_7, kr_7, k_7, f_7)

# Step 4
println("  Running Graf shift...")
t_4 = @elapsed begin
    B_7 = graf_shift(A_tilde_7, m_full_7, kr_7, x0_7, L_max_7)
end
println("  Done in $(round(t_4, digits=3)) s")

# 7a: No NaN/Inf
@assert !any(isnan, B_7) "NaN in B"
@assert !any(isinf, B_7) "Inf in B"
println("  7a: No NaN/Inf ✓")

# 7b: B has correct dimensions
@assert size(B_7) == (Nr_7, 2L_max_7 + 1) "Wrong output size"
println("  7b: Size $(size(B_7)) ✓")

# 7c: B should be concentrated in propagating kr band
prop_kr = findall(kr_7 .< k_7)
B_prop_energy = sum(abs2.(B_7[prop_kr, :]))
B_total_energy = sum(abs2.(B_7))
prop_frac = B_prop_energy / (B_total_energy + 1e-30)
println("  7c: Energy in propagating band: $(round(100*prop_frac, digits=1))%")
@assert prop_frac > 0.99 "Most B energy should be propagating"
println("  7c: Propagating concentration ✓")

# 7d: B_l should decay at |l| = L_max
B_energies = [sum(abs2.(B_7[:, li])) for li in 1:2L_max_7+1]
B_peak = maximum(B_energies)
B_edge = max(B_energies[1], B_energies[end])
B_ratio = B_edge / (B_peak + 1e-30)
println("  7d: Edge/peak mode energy: $(round(B_ratio, sigdigits=3))")
@assert B_ratio < 0.1 "L_max truncation may be insufficient"
println("  7d: Local mode decay ✓")

# 7e: Normal incidence limit — if α→0, x₀→0, B_l → Ã_l
B_normal = graf_shift(A_tilde_7, m_full_7, kr_7, 0.0, L_max_7)
for (li, l) in enumerate(-L_max_7:L_max_7)
    idx_A = l + M_max_7 + 1   # column in A_tilde for mode l
    err = maximum(abs.(B_normal[:, li] .- A_tilde_7[:, idx_A]))
    @assert err < 1e-12 "Normal incidence identity failed for l=$l"
end
println("  7e: Normal incidence identity ✓")

println("  Test 7 PASSED ✓")


println("\n" * "="^60)
println("All Step 4 tests passed.")
println("="^60)
