"""
    profile_adjoint_graf.jl
    =======================
    Diagnostic script for profiling the fused Graf+propagation adjoint
    (s5+4 in psf_adjoint), which is the dominant bottleneck.

    CONTEXT:
    - The fused s5+4 adjoint takes ~226s at R=1000μm (M=6303, Nr=65536)
      on a 350-thread 2-socket machine.
    - The forward graf_shift for the same problem takes only 0.8s.
    - The adjoint does ~2× the FLOPs of the forward, yet is 280× slower.
    - The FLOP estimate suggests ~10ms on 350 threads with SIMD.
    - GC was ruled out (pre-allocated buffers, no improvement).

    THIS SCRIPT profiles the bottleneck step in isolation:
    1. Measures the fused s5+4 with per-component timing
    2. Tests with different thread counts to check scaling
    3. Tests with @threads :static vs default scheduling
    4. Checks if @simd is actually vectorizing
    5. Compares the forward graf_shift timing for reference

    Run with:
        julia -t 350 profile_adjoint_graf.jl
        julia -t 1   profile_adjoint_graf.jl   # single-thread baseline
"""

include("../cyffp.jl")
using .CyFFP

println("="^70)
println("Profiling fused Graf+propagation adjoint (s5+4)")
println("Threads: $(Threads.nthreads())  maxthreadid: $(Threads.maxthreadid())")
println("="^70)

# ─── Setup: R=1000μm, α=30° ──────────────────────────────────
lambda_um = 0.5
pitch = 0.3
R_um = 1000.0
N_cells = round(Int, R_um / pitch)
k = 2π / lambda_um
NA = 0.4
f_val = R_um * sqrt(1/NA^2 - 1)
alpha = deg2rad(30.0)
x0 = f_val * tan(alpha)
kx = k * sin(alpha)

plan = prepare_psf(pitch, N_cells; lambda_um=lambda_um, alpha_deg=30.0, NA=NA,
                   Nxy=31, L_max=8, N_psi=32)
Nr = plan.Nr
M_max = plan.M_max
L_max = plan.L_max
m_pos = plan.m_pos
kr = plan.kr

println("\nR=$(R_um)μm  M_max=$M_max  Nr=$Nr  L_max=$L_max")
println("x₀=$(round(x0, digits=1))μm  k=$(round(k, digits=2))  kr·x₀ range: $(round(kr[1]*x0, sigdigits=3)) to $(round(kr[end]*x0, sigdigits=3))")

# Count propagating kr
n_prop = count(kr .< k)
println("Propagating kr: $n_prop / $Nr")

# ─── Create a realistic B_bar ────────────────────────────────
# Run the forward to get B, then use it as B_bar
t0 = ComplexF64[exp(-im*k*(sqrt(((i-0.5)*pitch)^2+f_val^2)-f_val)) for i in 1:N_cells]
result = execute_psf(plan, t0)
# Use ones as dL_dI to get a realistic B_bar via partial adjoint
# (run Steps 9-8-7-6 of the adjoint)
dL_dI = ones(Float64, plan.Nxy, plan.Nxy)

# Manually run Steps 9-8-7-6 to get B_bar
u_psf = result._u_psf
I_polar = result._I_polar
rho = result._rho
psi = result._psi
dln = plan.dln

hw = plan.psf_half_width_um
x_um = collect(range(-hw, hw, length=plan.Nxy))
y_um = collect(range(-hw, hw, length=plan.Nxy))
dpsi = psi[2] - psi[1]
N_psi = plan.N_psi

I_polar_bar = zeros(Float64, Nr, N_psi)
@inbounds for ix in 1:plan.Nxy, iy in 1:plan.Nxy
    dl = dL_dI[iy, ix]; abs(dl) < 1e-30 && continue
    xv = x_um[ix]; yv = y_um[iy]
    rv = sqrt(xv^2 + yv^2)
    (rv < rho[1] || rv > rho[end]) && continue
    pv = atan(yv, xv); pv < 0 && (pv += 2π)
    lr = log(rv); lr0 = log(rho[1])
    idx_r = (lr - lr0) / dln + 1.0
    j0 = clamp(floor(Int, idx_r), 1, Nr - 1); wr = idx_r - j0
    idx_p = pv / dpsi + 1.0
    p0 = clamp(floor(Int, idx_p), 1, N_psi)
    p1 = p0 == N_psi ? 1 : p0 + 1; wp = idx_p - p0
    I_polar_bar[j0,p0]+=(1-wr)*(1-wp)*dl; I_polar_bar[j0+1,p0]+=wr*(1-wp)*dl
    I_polar_bar[j0,p1]+=(1-wr)*wp*dl; I_polar_bar[j0+1,p1]+=wr*wp*dl
end

u_bar = zeros(ComplexF64, Nr, N_psi)
@inbounds for ip in 1:N_psi, j in 1:Nr
    u_bar[j, ip] = I_polar_bar[j, ip] * u_psf[j, ip]
end

using FFTW
b_bar_fft = fft(u_bar, 2)
b_bar = zeros(ComplexF64, Nr, 2L_max + 1)
for (li, l) in enumerate(-L_max:L_max)
    col = l >= 0 ? l + 1 : N_psi + l + 1
    b_bar[:, li] .= b_bar_fft[:, col]
end

# Step 6 adjoint (inverse Hankel) to get B_bar
n_idx_q = [n <= (Nr-1)÷2 ? n : n - Nr for n in 0:Nr-1]
q_vec = @. (2π / (Nr * dln)) * n_idx_q
kernels_l = CyFFP._precompute_kernels(q_vec, collect(0:L_max))

nt = CyFFP._nworkspaces()
adj_ws = [CyFFP._make_workspace(Nr, dln) for _ in 1:nt]
raw_bufs = [Vector{ComplexF64}(undef, Nr) for _ in 1:nt]
f_bufs = [Vector{ComplexF64}(undef, Nr) for _ in 1:nt]

B_bar = zeros(ComplexF64, Nr, 2L_max + 1)
Threads.@threads for li in 1:2L_max+1
    tid = Threads.threadid()
    ws = adj_ws[tid]; raw_bar = raw_bufs[tid]; f_bar = f_bufs[tid]
    l = li - L_max - 1; l_abs = abs(l)
    neg_sign = (l < 0 && isodd(l_abs)) ? -1 : 1
    @inbounds for j in 1:Nr; raw_bar[j] = b_bar[j, li] / rho[j]; end
    CyFFP._adjoint_fftlog!(f_bar, ws, raw_bar, view(kernels_l, :, l_abs+1), neg_sign)
    @inbounds for j in 1:Nr; B_bar[j, li] = kr[j] * f_bar[j]; end
end

println("\nB_bar ready: $(size(B_bar)), max |B_bar| = $(round(maximum(abs.(B_bar)), sigdigits=3))")

# ─── Profile the fused s5+4 ──────────────────────────────────
println("\n" * "─"^70)
println("PROFILING: fused graf+prop adjoint (s5+4)")
println("─"^70)

kz = @. sqrt(complex(k^2 - kr.^2))
prop_conj = @. ifelse(kr < k, exp(-im * real(kz) * f_val), zero(ComplexF64))

# Pre-allocate (same as psf_adjoint)
pos_bufs = [zeros(ComplexF64, M_max + 1) for _ in 1:nt]
neg_bufs = [zeros(ComplexF64, M_max + 1) for _ in 1:nt]
jw_bufs = [Vector{Float64}(undef, 2(M_max + L_max) + 1) for _ in 1:nt]
jp_bufs = [Vector{Float64}(undef, M_max + L_max + 21) for _ in 1:nt]

function run_fused_s5_4!(a_bar, B_bar, kr, prop_conj, plan, m_pos, M_max, L_max,
                          pos_bufs, neg_bufs, jw_bufs, jp_bufs)
    Nr = length(kr)

    Threads.@threads for ikr in 1:Nr
        kr_val = kr[ikr]
        kr_val > plan.k && continue

        tid = Threads.threadid()
        buf_pos = pos_bufs[tid]; buf_neg = neg_bufs[tid]
        Jw = jw_bufs[tid]; Jp_buf = jp_bufs[tid]

        kr_x0 = kr_val * plan.x0
        n_cut = min(M_max + L_max, ceil(Int, abs(kr_x0)) + 20)
        pc = prop_conj[ikr]

        CyFFP._besselj_range!(Jp_buf, n_cut, kr_x0)
        Jp = view(Jp_buf, 1:n_cut+1)

        @inbounds for d in 0:n_cut
            Jw[d + n_cut + 1] = Jp[d + 1]
            Jw[-d + n_cut + 1] = iseven(d) ? Jp[d + 1] : -Jp[d + 1]
        end

        m_eff = min(M_max, n_cut + L_max)

        @inbounds for m in 0:m_eff
            buf_pos[m + 1] = zero(ComplexF64)
            buf_neg[m + 1] = zero(ComplexF64)
        end

        @inbounds for (li, l) in enumerate(-L_max:L_max)
            bl = B_bar[ikr, li]
            m_lo_p = max(0, l - n_cut); m_hi_p = min(m_eff, l + n_cut)
            @simd for m in m_lo_p:m_hi_p
                buf_pos[m + 1] += bl * Jw[m - l + n_cut + 1]
            end
            m_lo_n = max(0, -l - n_cut); m_hi_n = min(m_eff, -l + n_cut)
            @simd for m in m_lo_n:m_hi_n
                buf_neg[m + 1] += bl * Jw[-m - l + n_cut + 1]
            end
        end

        @inbounds begin
            a_bar[ikr, 1] = pc * buf_pos[1]
            for m in 1:m_eff
                s = iseven(m) ? 1 : -1
                a_bar[ikr, m + 1] = pc * (buf_pos[m + 1] + s * buf_neg[m + 1])
            end
        end
    end
end

# Warmup
a_bar = zeros(ComplexF64, Nr, M_max + 1)
run_fused_s5_4!(a_bar, B_bar, collect(kr), prop_conj, plan, m_pos, M_max, L_max,
                pos_bufs, neg_bufs, jw_bufs, jp_bufs)

# Benchmark
a_bar .= 0
t_fused = @elapsed run_fused_s5_4!(a_bar, B_bar, collect(kr), prop_conj, plan, m_pos,
                                     M_max, L_max, pos_bufs, neg_bufs, jw_bufs, jp_bufs)
println("  Fused s5+4: $(round(t_fused, digits=1))s")

# ─── Profile just the Bessel computation ──────────────────────
println("\n--- Bessel computation alone ---")
t_bessel = @elapsed begin
    Threads.@threads for ikr in 1:Nr
        kr[ikr] > plan.k && continue
        tid = Threads.threadid()
        kr_x0 = kr[ikr] * plan.x0
        n_cut = min(M_max + L_max, ceil(Int, abs(kr_x0)) + 20)
        CyFFP._besselj_range!(jp_bufs[tid], n_cut, kr_x0)
    end
end
println("  Bessel only: $(round(t_bessel, digits=1))s")

# ─── Profile just the scatter loop (no Bessel) ───────────────
println("\n--- Scatter loop alone (reuse precomputed Jw) ---")
# Precompute Jw for one representative kr
kr_mid = kr[Nr÷2]
kr_x0_mid = kr_mid * plan.x0
n_cut_mid = min(M_max + L_max, ceil(Int, abs(kr_x0_mid)) + 20)
CyFFP._besselj_range!(jp_bufs[1], n_cut_mid, kr_x0_mid)
Jw_fixed = Vector{Float64}(undef, 2n_cut_mid + 1)
for d in 0:n_cut_mid
    Jw_fixed[d + n_cut_mid + 1] = jp_bufs[1][d + 1]
    Jw_fixed[-d + n_cut_mid + 1] = iseven(d) ? jp_bufs[1][d + 1] : -jp_bufs[1][d + 1]
end
m_eff_mid = min(M_max, n_cut_mid + L_max)

t_scatter = @elapsed begin
    for rep in 1:10  # repeat to get stable timing
        Threads.@threads for ikr in 1:Nr
            kr[ikr] > plan.k && continue
            tid = Threads.threadid()
            buf_p = pos_bufs[tid]; buf_n = neg_bufs[tid]
            bl_test = B_bar[ikr, L_max + 1]  # one representative l value
            @inbounds @simd for m in 0:m_eff_mid
                buf_p[m + 1] += bl_test * Jw_fixed[m + n_cut_mid + 1]
            end
        end
    end
end
println("  Scatter (10 reps, 1 l value): $(round(t_scatter*1000/10, digits=1))ms per rep")
println("  Estimated full (31 l × 2): $(round(t_scatter*1000/10 * 62, digits=0))ms")

# ─── Forward graf_shift for comparison ────────────────────────
println("\n--- Forward graf_shift ---")
a_tilde_dummy, m_full_dummy = propagate_scalar(
    zeros(ComplexF64, Nr, M_max+1) .+ 1e-10, m_pos, collect(kr), k, f_val)
t_fwd_graf = @elapsed graf_shift(a_tilde_dummy, m_full_dummy, collect(kr), x0, L_max; k=k)
println("  Forward graf_shift: $(round(t_fwd_graf, digits=2))s")

# ─── Single-kr profiling ─────────────────────────────────────
println("\n--- Per-kr breakdown (single thread, propagating kr near k) ---")
ikr_test = argmin(abs.(kr .- k * 0.9))
kr_test = kr[ikr_test]
kr_x0_test = kr_test * plan.x0
n_cut_test = min(M_max + L_max, ceil(Int, abs(kr_x0_test)) + 20)
m_eff_test = min(M_max, n_cut_test + L_max)

println("  kr = $(round(kr_test, sigdigits=4)), kr·x₀ = $(round(kr_x0_test, sigdigits=4))")
println("  n_cut = $n_cut_test, m_eff = $m_eff_test")

t_bess1 = @elapsed for rep in 1:100
    CyFFP._besselj_range!(jp_bufs[1], n_cut_test, kr_x0_test)
end
println("  Bessel per kr: $(round(t_bess1*1000/100, digits=2))ms")

t_jw1 = @elapsed for rep in 1:100
    Jw_t = jw_bufs[1]
    @inbounds for d in 0:n_cut_test
        Jw_t[d + n_cut_test + 1] = jp_bufs[1][d + 1]
        Jw_t[-d + n_cut_test + 1] = iseven(d) ? jp_bufs[1][d + 1] : -jp_bufs[1][d + 1]
    end
end
println("  Jw build per kr: $(round(t_jw1*1000/100, digits=3))ms")

t_zero1 = @elapsed for rep in 1:100
    @inbounds for m in 0:m_eff_test
        pos_bufs[1][m+1] = zero(ComplexF64)
        neg_bufs[1][m+1] = zero(ComplexF64)
    end
end
println("  Zero bufs per kr: $(round(t_zero1*1000/100, digits=3))ms")

t_scat1 = @elapsed for rep in 1:100
    @inbounds for (li, l) in enumerate(-L_max:L_max)
        bl = B_bar[ikr_test, li]
        m_lo_p = max(0, l - n_cut_test); m_hi_p = min(m_eff_test, l + n_cut_test)
        @simd for m in m_lo_p:m_hi_p
            pos_bufs[1][m + 1] += bl * jw_bufs[1][m - l + n_cut_test + 1]
        end
        m_lo_n = max(0, -l - n_cut_test); m_hi_n = min(m_eff_test, -l + n_cut_test)
        @simd for m in m_lo_n:m_hi_n
            neg_bufs[1][m + 1] += bl * jw_bufs[1][-m - l + n_cut_test + 1]
        end
    end
end
println("  Scatter per kr: $(round(t_scat1*1000/100, digits=2))ms")

t_comb1 = @elapsed for rep in 1:100
    @inbounds begin
        a_bar[ikr_test, 1] = prop_conj[ikr_test] * pos_bufs[1][1]
        for m in 1:m_eff_test
            s = iseven(m) ? 1 : -1
            a_bar[ikr_test, m + 1] = prop_conj[ikr_test] * (pos_bufs[1][m+1] + s * neg_bufs[1][m+1])
        end
    end
end
println("  Combine per kr: $(round(t_comb1*1000/100, digits=3))ms")

total_per_kr = (t_bess1 + t_jw1 + t_zero1 + t_scat1 + t_comb1) / 100
println("  Total per kr: $(round(total_per_kr*1000, digits=2))ms")
println("  Estimated total ($(n_prop) prop kr / $(Threads.nthreads()) threads): $(round(n_prop * total_per_kr / Threads.nthreads(), digits=1))s")
println("  Actual fused s5+4: $(round(t_fused, digits=1))s")
println("  Overhead factor: $(round(t_fused / (n_prop * total_per_kr / Threads.nthreads()), digits=1))×")

println("\n" * "="^70)
println("Done. Share this output for diagnosis.")
println("="^70)
