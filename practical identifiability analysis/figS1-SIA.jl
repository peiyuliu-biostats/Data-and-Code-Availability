using ForwardDiff
using LinearAlgebra
using CairoMakie

# ============================================================
# observable moment system outputs at a single time point t
# ============================================================
function obs_moments(t, θ_free, scenario)
    # seasonal forcing parameters (known, fixed)
    ω  = 2π / 365.0
    ū  = 1.0
    ε  = 0.3

    γ = θ_free[1]

    # convolution integrals under harmonic forcing
    I1 = ū / γ +
         ε * (γ * cos(ω * t) + ω * sin(ω * t)) /
         (γ^2 + ω^2)
    I2 = ū / (2γ) +
         ε * (2γ * cos(ω * t) + ω * sin(ω * t)) /
         (4γ^2 + ω^2)

    if scenario == :A
        # free = [γ, B0, C, C2, β0],  σ² fixed
        γ, B0, C, C2, β0 = θ_free
        σ²  = 0.05
    elseif scenario == :B
        # free = [γ, B0, C, C2],  β0 and σ² fixed
        γ, B0, C, C2 = θ_free
        β0  = 0.0
        σ²  = 0.05
    elseif scenario == :C
        # free = [γ, B0, C, C2, σ²],  β0 fixed
        γ, B0, C, C2, σ² = θ_free
        β0  = 0.0
    end

    λ     = exp(β0)
    μ_obs = B0 + C * λ * I1
    V_obs = C2 * λ * I2 + σ²
    return [μ_obs, V_obs]
end

# ============================================================
# build sensitivity jacobian over a given time grid
# ============================================================
function build_jacobian(θ_free, scenario, t_grid)
    n_t   = length(t_grid)
    n_obs = 2 * n_t
    n_par = length(θ_free)
    J     = zeros(n_obs, n_par)
    for (j, t) in enumerate(t_grid)
        f             = p -> obs_moments(t, p, scenario)
        jac_t         = ForwardDiff.jacobian(f, θ_free)
        J[2j-1:2j, :] = jac_t
    end
    return J
end

# ============================================================
# true parameter values
# ============================================================
γ_true  = 0.01
B0_true = 0.5
C_true  = 1.2
C2_true = 2.0
β0_true = 0.0
σ²_true = 0.05

# parameter vectors for each scenario
θ_A = [γ_true, B0_true, C_true, C2_true, β0_true]
θ_B = [γ_true, B0_true, C_true, C2_true]
θ_C = [γ_true, B0_true, C_true, C2_true, σ²_true]

# ============================================================
# time grids: 1 month, 4 months, 12 months, 24 months
# ============================================================
t_1mo  = collect(0.0:30.0:30.0)    # 1 time point
t_4mo  = collect(0.0:30.0:90.0)    # 4 time points
t_12mo = collect(0.0:30.0:330.0)   # 12 time points
t_24mo = collect(0.0:30.0:690.0)   # 24 time points

t_grids = [t_1mo, t_4mo, t_12mo, t_24mo]
labels  = ["1 month", "4 months", "12 months", "24 months"]

# ============================================================
# compute singular values for all scenarios × time grids
# pad sv_B to length 5 with sentinel 1e-15 for alignment
# ============================================================
results_A = Vector{Vector{Float64}}()
results_B = Vector{Vector{Float64}}()
results_C = Vector{Vector{Float64}}()

for tg in t_grids
    sv_a = svd(build_jacobian(θ_A, :A, tg)).S
    sv_b = svd(build_jacobian(θ_B, :B, tg)).S
    sv_c = svd(build_jacobian(θ_C, :C, tg)).S

    # pad sv_b to length 5
    sv_b_pad = length(sv_b) < 5 ? vcat(sv_b, fill(1e-15, 5 - length(sv_b))) : sv_b

    push!(results_A, sv_a)
    push!(results_B, sv_b_pad)
    push!(results_C, sv_c)
end

# ============================================================
# print results
# ============================================================
for (k, lbl) in enumerate(labels)
    println("=== $lbl ===")
    println("  scenario a: ", round.(results_A[k], sigdigits = 3))
    println("  scenario b: ", round.(results_B[k], sigdigits = 3))
    println("  scenario c: ", round.(results_C[k], sigdigits = 3))
end

# ============================================================
# plot: 2×2 panel, one subplot per time grid
# ============================================================
fig = Figure(size = (1000, 800), fontsize = 13)

threshold = 1e-10

for (k, lbl) in enumerate(labels)
    row = (k - 1) ÷ 2 + 1
    col = (k - 1) % 2 + 1

    ax = Axis(fig[row, col],
        title   = lbl,
        xlabel  = "singular value index",
        ylabel  = "magnitude (log10 scale)",
        yscale  = log10,
        xticks  = 1:5,
    )

    # threshold line
    hlines!(ax, [threshold],
        color     = :gray,
        linestyle = :dash,
        linewidth = 1.5,
        label     = "null threshold (1e-10)")

    sv_a = results_A[k]
    sv_b = results_B[k]
    sv_c = results_C[k]

    # pad shorter vectors with sentinel for plotting
    na = length(sv_a)
    nb = length(sv_b)
    nc = length(sv_c)

    scatter!(ax, 1:na, sv_a,
        color      = :tomato,
        marker     = :circle,
        markersize = 16,
        label      = "scenario a: β0 unknown, σ²ε fixed")

    scatter!(ax, 1:nb, sv_b,
        color      = :royalblue,
        marker     = :rect,
        markersize = 14,
        label      = "scenario b: β0 fixed, σ²ε fixed")

    scatter!(ax, 1:nc, sv_c,
        color      = :darkgreen,
        marker     = :utriangle,
        markersize = 14,
        label      = "scenario c: β0 fixed, σ²ε unknown")

    # legend only on first panel
    if k == 1
        axislegend(ax, position = :lb, framevisible = true, labelsize = 11)
    end
end

save_path = joinpath(
    "C:/Users/TopSoarer/Desktop/MSP data-2026/code/Julia",
    "Figure_S1_Identifiability_multiwindow.pdf"
)
save(save_path, fig)
display(fig)