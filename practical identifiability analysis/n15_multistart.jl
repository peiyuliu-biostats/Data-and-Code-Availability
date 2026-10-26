# ==============================================================================
# n15_multistart.jl  (PLOS CB revision, 2026-10) -- diagnostic for the N = 15 fit
#
# In the t_int = 910 re-run (pia-mle-interventionalcs-simdata_rev.jl) the N = 15 MLE
# ended on a box bound: the null-space UQ found no admissible draws and Optim warned
# "Terminated early due to NaN in gradient". This script re-fits ONLY N = 15 from
# several starting values with exactly the same model, likelihood, bounds and
# optimiser, and records for every start:
#   final -loglik, convergence, iterations, run time, NaN in the gradient at the
#   start and at the solution, which parameters sit on a bound, the estimates, and
#   the smallest FIM eigenvalues at the solution.
# If the best solution is in the interior, it also writes the N = 15 PIA tables and
# UQ envelope (suffix _multistart) so that S7a can show N = 15 again.
#
# Model code below is copied verbatim from pia-mle-interventionalcs-simdata_rev.jl.
# Run (cmd):  julia n15_multistart.jl          (about 5-15 minutes)
# Outputs  :  outputs/n15_multistart_results.csv
#             outputs/PIA_MLE_Results_Int_N15_multistart.xlsx      (if interior)
#             outputs/PIA_MLE_UQ_Envelop_Int_N15_multistart.csv    (if interior)
# ==============================================================================
using DataFrames
using XLSX
using Statistics
using LinearAlgebra
using QuadGK
using ForwardDiff
using Optim
using CSV
using SpecialFunctions
using Random

# ==============================================================================
# 0. constants and interventional model formulation (8 parameters)
# ==============================================================================
const omega = 2 * pi / 365.0

# intervention time: [FIX-1] 910.0 days, matching the data-generating value
# T_INT <- 910 in run_sim_intervention_single.R (original script used 730.0)
const t_int = 910.0

# adaptive integration bounds constants
const U_CAP            = 1e6
const DECAY_MULTIPLIER = 50.0

# measurement noise variance fixed to match simulation data-generating value:
# sim sigma_epsilon = 0.15 (sd) => sigma2_eps = 0.15^2 = 0.0225
const SIGMA2_EPS = 0.0225

_extract_val(x) = float(x)
_extract_val(x::ForwardDiff.Dual) = float(ForwardDiff.value(x))

# interrupted time series exposure rate
# ifelse ensures type-stability for forwarddiff dual numbers across the step function
function exposure_rate(s, beta0, beta_sin, beta_cos, beta_int)
    intervention_effect = ifelse(s > t_int, beta_int, zero(beta0))
    return exp(beta0 + beta_sin * sin(omega * s) + beta_cos * cos(omega * s) + intervention_effect)
end

function calc_mu(t, gamma, B0, C_bar, beta0, beta_sin, beta_cos, beta_int)
    g            = max(_extract_val(gamma), 1e-8)
    upper_bound  = min(U_CAP, DECAY_MULTIPLIER / g)
    integrand(u) = exp(-gamma * u) * exposure_rate(t - u, beta0, beta_sin, beta_cos, beta_int)
    val, err     = quadgk(integrand, 0.0, upper_bound; rtol=1e-5)
    return B0 + C_bar * val
end

# observable variance: v_obs(t) = v*(t) + sigma2_eps
function calc_var(t, gamma, C2_bar, beta0, beta_sin, beta_cos, beta_int)
    g            = max(_extract_val(gamma), 1e-8)
    upper_bound  = min(U_CAP, DECAY_MULTIPLIER / g)
    integrand(u) = exp(-2 * gamma * u) * exposure_rate(t - u, beta0, beta_sin, beta_cos, beta_int)
    val, err     = quadgk(integrand, 0.0, upper_bound; rtol=1e-5)
    return C2_bar * val + SIGMA2_EPS
end

# forward model wrapper for uq simulation
function model_output(theta, times)
    gamma, B0, C_bar, C2_bar, beta0, beta_sin, beta_cos, beta_int = theta
    mu_vec  = [calc_mu( t, gamma, B0,    C_bar,  beta0, beta_sin, beta_cos, beta_int) for t in times]
    var_vec = [calc_var(t, gamma, C2_bar, beta0, beta_sin, beta_cos, beta_int)        for t in times]
    return vcat(mu_vec, var_vec)
end

function gamma_logpdf(y, mu, var)
    mu_safe     = max(mu,  1e-8)
    var_safe    = max(var, 1e-8)
    k           = max(mu_safe^2 / var_safe, 1e-8)
    theta_scale = var_safe / mu_safe
    return (k - 1) * log(y) - (y / theta_scale) - k * log(theta_scale) - loggamma(k)
end

function compute_coordinate_metrics(S_matrix)
    n_params = size(S_matrix, 2)
    metrics  = zeros(n_params)
    for i in 1:n_params
        s_i        = S_matrix[:, i]
        A          = S_matrix[:, setdiff(1:n_params, i)]
        svd_A      = svd(A)
        tol        = maximum(size(A)) * eps(eltype(A)) * maximum(svd_A.S)
        r          = count(>(tol), svd_A.S)
        U_r        = svd_A.U[:, 1:r]
        proj_resid = s_i - U_r * (U_r' * s_i)
        metrics[i] = norm(proj_resid, Inf)
    end
    return metrics
end

# unified parameterization mapping for 8 parameters (eta to theta)
# beta_int is a log-fold change multiplier and remains linear in eta space
function eta_to_theta(eta::AbstractVector{T}) where T
    return [
        exp(eta[1]), # gamma
        exp(eta[2]), # B0
        exp(eta[3]), # C_bar
        exp(eta[4]), # C2_bar
        eta[5],      # beta0
        eta[6],      # beta_sin
        eta[7],      # beta_cos
        eta[8]       # beta_int
    ]
end


# ==============================================================================
# 1. data (N = 15) and settings identical to the main script
# ==============================================================================
const PROJ_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
file_path = joinpath(PROJ_ROOT, "4-23-PIA", "Simulation datasets", "interventional cs",
                     "Sim_Data_Rep1_Combined.xlsx")
out_dir   = joinpath(@__DIR__, "outputs")
isdir(out_dir) || mkpath(out_dir)

eta_init  = [log(0.05), log(0.3), log(0.2), log(0.25), log(1/15), 0.5,  0.2, -1.0]
eta_lower = [log(1e-4), log(1e-4), log(1e-4), log(1e-4), -10.0,  -5.0, -5.0, -10.0]
eta_upper = [log(1.0),  log(5.0),  log(5.0),  log(5.0),   5.0,    5.0,  5.0,   5.0]
param_names = ["gamma", "B0", "C_bar", "C2_bar", "beta0", "beta_sin", "beta_cos", "beta_int"]
# data-generating values (run_sim_intervention_single.R)
eta_true  = [log(log(2)/14), log(0.3), log(0.2), log(0.29), log(1/15), 0.5, 0.2, log(0.4)]

const raw_data     = DataFrame(XLSX.readtable(file_path, "N15_Data"))
const unique_times = Float64.(sort(unique(raw_data.time)))
const time_to_idx  = Dict(t => i for (i, t) in enumerate(unique_times))
const indices      = [time_to_idx[Float64(t)] for t in raw_data.time]
const y_vals       = Float64.(max.(raw_data.antibody, 1e-10))
println("N = 15: ", length(y_vals), " observations, ", length(unique_times), " unique times")

function ll_vector(eta)
    theta = eta_to_theta(eta)
    gamma, B0, C_bar, C2_bar, beta0, beta_sin, beta_cos, beta_int = theta
    T_type   = eltype(eta)
    mu_uniq  = zeros(T_type, length(unique_times))
    var_uniq = zeros(T_type, length(unique_times))
    for i in 1:length(unique_times)
        mu_uniq[i]  = calc_mu( unique_times[i], gamma, B0,    C_bar,  beta0, beta_sin, beta_cos, beta_int)
        var_uniq[i] = calc_var(unique_times[i], gamma, C2_bar, beta0, beta_sin, beta_cos, beta_int)
    end
    ll_vec = zeros(T_type, length(y_vals))
    for i in 1:length(y_vals)
        ll_vec[i] = gamma_logpdf(y_vals[i], mu_uniq[indices[i]], var_uniq[indices[i]])
    end
    return ll_vec
end
f_obj(eta) = -sum(ll_vector(eta))
g_grad!(G, eta) = ForwardDiff.gradient!(G, f_obj, eta)

# ==============================================================================
# 2. starting values: original start, true values, and 10 random interior starts
# ==============================================================================
Random.seed!(20261011)
starts = Vector{Tuple{String, Vector{Float64}}}()
push!(starts, ("original_init", copy(eta_init)))
push!(starts, ("true_values",   copy(eta_true)))
for k in 1:10
    e = [log(log(2) / (7 + 113 * rand())),    # half-life 7-120 d
         log(0.05 + 0.75 * rand()),           # B0 0.05-0.8
         log(0.05 + 0.45 * rand()),           # C_bar 0.05-0.5
         0.0,                                 # C2_bar set below (>= C_bar^2)
         log(1/30) + (log(1/7) - log(1/30)) * rand(),
         -1.0 + 2.0 * rand(), -1.0 + 2.0 * rand(),
         -2.0 + 2.0 * rand()]                 # beta_int -2..0
    e[4] = log(exp(2 * e[3]) * (1.5 + 8.5 * rand()))   # E[C^2] = C_bar^2 * (1.5-10)
    push!(starts, ("random_$(k)", e))
end

# ==============================================================================
# 3. fit from every start
# ==============================================================================
tol_b = 1e-3   # "on a bound" = within 1e-3 of it on the eta (log) scale
rows  = DataFrame()
fits  = Dict{String, Vector{Float64}}()
for (lab, e0) in starts
    e0 = clamp.(e0, eta_lower .+ 1e-3, eta_upper .- 1e-3)
    G0 = similar(e0); nan_start = false
    try
        g_grad!(G0, e0); nan_start = any(isnan, G0)
    catch
        nan_start = true
    end
    println("\nstart: $lab  (NaN in gradient at start: $nan_start)")
    t_el = @elapsed begin
        res = try
            optimize(OnceDifferentiable(f_obj, g_grad!, e0), eta_lower, eta_upper, e0, Fminbox(LBFGS()))
        catch err
            println("  optimisation error: ", err); nothing
        end
    end
    if res === nothing
        push!(rows, (start = lab, nll = NaN, converged = false, iterations = 0, time_s = t_el,
                     nan_grad_start = nan_start, nan_grad_solution = true,
                     at_lower = "", at_upper = "",
                     gamma_half_life_d = NaN, B0 = NaN, C_bar = NaN, C2_bar = NaN, beta0 = NaN,
                     beta_sin = NaN, beta_cos = NaN, beta_int = NaN,
                     eig_min1 = NaN, eig_min2 = NaN, eig_min3 = NaN); promote = true)
        continue
    end
    eh = Optim.minimizer(res)
    G  = similar(eh); g_grad!(G, eh)
    lo = param_names[abs.(eh .- eta_lower) .< tol_b]
    up = param_names[abs.(eh .- eta_upper) .< tol_b]
    th = eta_to_theta(eh)
    ev = try
        S = ForwardDiff.jacobian(ll_vector, eh); sort(abs.(eigen(Symmetric(S' * S)).values))
    catch
        fill(NaN, 8)
    end
    fits[lab] = eh
    push!(rows, (start = lab, nll = Optim.minimum(res), converged = Optim.converged(res),
                 iterations = Optim.iterations(res), time_s = t_el,
                 nan_grad_start = nan_start, nan_grad_solution = any(isnan, G),
                 at_lower = join(lo, ";"), at_upper = join(up, ";"),
                 gamma_half_life_d = log(2) / th[1], B0 = th[2], C_bar = th[3], C2_bar = th[4],
                 beta0 = th[5], beta_sin = th[6], beta_cos = th[7], beta_int = th[8],
                 eig_min1 = ev[1], eig_min2 = ev[2], eig_min3 = ev[3]); promote = true)
    println("  nll = ", round(Optim.minimum(res), digits = 4), "  converged = ", Optim.converged(res),
            "  bounds hit: lower[", join(lo, ","), "] upper[", join(up, ","), "]  (", round(t_el, digits = 1), " s)")
end
sort!(rows, :nll)
CSV.write(joinpath(out_dir, "n15_multistart_results.csv"), rows)
println("\n=== results written to outputs/n15_multistart_results.csv ===")
show(stdout, select(rows, :start, :nll, :converged, :at_lower, :at_upper, :gamma_half_life_d, :B0, :beta_int);
     allrows = true, allcols = true); println()

# ==============================================================================
# 4. if the best solution is interior: write PIA tables and UQ envelope for N = 15
# ==============================================================================
best = rows[1, :]
let   # local scope (avoids soft-scope ambiguity for loop counters in a script)
if isfinite(best.nll) && best.at_lower == "" && best.at_upper == ""
    eta_hat  = fits[best.start]
    S_matrix = ForwardDiff.jacobian(ll_vector, eta_hat)
    er = eigen(Symmetric(S_matrix' * S_matrix)); ev = abs.(er.values); si = sortperm(ev, rev = true)
    U  = er.vectors[:, si]; evs = ev[si]
    df_e = DataFrame(Component = ["Eig_1(max)","Eig_2","Eig_3","Eig_4","Eig_5","Eig_6","Eig_7","Eig_8(min)"], Eigenvalue = evs)
    df_c = DataFrame(Parameter = param_names, Coordinate_Metric = compute_coordinate_metrics(S_matrix))
    df_x = DataFrame(SampleSize_N = [15], Xi_Metric = [evs[end] / evs[1]])
    df_h = DataFrame(abs.(U), :auto); rename!(df_h, ["U1","U2","U3","U4","U5","U6","U7","U8"])
    df_h.Parameter = param_names; select!(df_h, :Parameter, Not(:Parameter))
    XLSX.writetable(joinpath(out_dir, "PIA_MLE_Results_Int_N15_multistart.xlsx"),
                    "Eigenvalues" => df_e, "Coordinate_Metric" => df_c, "Xi_Metric" => df_x,
                    "Heatmap_U" => df_h, overwrite = true)
    # UQ: identical procedure to the main script
    null_idx = findall((evs ./ maximum(evs)) .< 1e-8)
    isempty(null_idx) && (null_idx = [8])
    Unull = U[:, null_idx]; draws = []
    attempts = 0
    while length(draws) < 1000 && attempts < 10000
        attempts += 1
        z = randn(length(null_idx)) .* 0.5
        en = eta_hat .+ Unull * z; sf = 1.0
        while !(all(en .>= eta_lower) && all(en .<= eta_upper)) && sf > 1e-4
            sf *= 0.5; en = eta_hat .+ Unull * (z * sf)
        end
        (all(en .>= eta_lower) && all(en .<= eta_upper)) && push!(draws, eta_to_theta(en))
    end
    println("UQ: $(length(draws)) admissible draws")
    if !isempty(draws)
        tt = collect(0:1.0:1095); P = zeros(length(draws), length(tt))
        for (k, th) in enumerate(draws)
            P[k, :] = model_output(th, tt)[1:length(tt)]
        end
        CSV.write(joinpath(out_dir, "PIA_MLE_UQ_Envelop_Int_N15_multistart.csv"),
                  DataFrame(Time_Days = tt,
                            Mean_Prediction = vec(mean(P, dims = 1)),
                            Lower_95_CI = [quantile(P[:, j], 0.025) for j in 1:length(tt)],
                            Upper_95_CI = [quantile(P[:, j], 0.975) for j in 1:length(tt)]))
        println("=== N = 15 PIA tables and UQ envelope written (suffix _multistart) ===")
    end
else
    println("=== best N = 15 solution is on a bound (or failed): no PIA/UQ files written ===")
end
end   # let
