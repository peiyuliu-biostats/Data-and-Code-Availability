# ==============================================================================
# REVISION COPY of 4-23-PIA/pia-mle-interventionalcs-simdata.jl  (PLOS CB revision, 2026-10)
# Changes vs original (see pia-mle-interventionalcs-simdata_rev.diff):
#   [FIX-1] t_int 730 -> 910: the simulated data (run_sim_intervention_single.R,
#           T_INT <- 910) start the intervention on day 910 and are observed on
#           days 909-956; the original PIA assumed day 730.
#   [PATH ] input/output paths are relative to this script's folder; results are
#           written to ./outputs (original results are left untouched).
#   [SEED ] Random.seed! added so the null-space UQ draws are reproducible.
#   [DIAG ] optional block at the end (RUN_TRUEPARAM_DIAGNOSTIC): same FIM /
#           coordinate metric evaluated at the TRUE parameter values for every N
#           (no refit), written to separate files *_TrueParam_*.xlsx. It does not
#           change the main outputs used by Fig 3 / S7.
# Nothing else (model, likelihood, optimiser, bounds, thresholds) is changed.
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
# 1. global setup and data import
# ==============================================================================
# [PATH] project root = two levels above this script (revision_work/01_.../)
const PROJ_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
file_path = joinpath(PROJ_ROOT, "4-23-PIA", "Simulation datasets", "interventional cs",
                     "Sim_Data_Rep1_Combined.xlsx")
out_dir   = joinpath(@__DIR__, "outputs")
isfile(file_path) || error("input not found: $file_path")
println("input : $file_path")
println("output: $out_dir")

if !isdir(out_dir)
    mkpath(out_dir)
end

N_list = [15, 70, 120, 300]

# 8-dimensional optimization in log-transformed eta space
# initial guess for beta_int is -1.0 (intervention reduces exposure)
eta_init  = [log(0.05), log(0.3), log(0.2), log(0.25), log(1/15), 0.5,  0.2, -1.0]
eta_lower = [log(1e-4), log(1e-4), log(1e-4), log(1e-4), -10.0,  -5.0, -5.0, -10.0]
eta_upper = [log(1.0),  log(5.0),  log(5.0),  log(5.0),   5.0,    5.0,  5.0,   5.0]

param_names = ["gamma", "B0", "C_bar", "C2_bar", "beta0", "beta_sin", "beta_cos", "beta_int"]

uq_scenarios = Dict()

println("=== starting individual-level mle-based pia for interventional model ===")
println("    observable variance: v_obs = v* + sigma2_eps, sigma2_eps = $SIGMA2_EPS")

# ==============================================================================
# 2. main loop over sample sizes
# ==============================================================================
for N in N_list
    sheet_name = "N$(N)_Data"
    println("\nloading data: $sheet_name ...")

    raw_data = DataFrame(XLSX.readtable(file_path, sheet_name))

    unique_times = sort(unique(raw_data.time))
    time_to_idx  = Dict(t => i for (i, t) in enumerate(unique_times))
    indices      = [time_to_idx[t] for t in raw_data.time]
    y_vals       = max.(raw_data.antibody, 1e-10)

    # vectorized log-likelihood in eta space
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
            idx       = indices[i]
            ll_vec[i] = gamma_logpdf(y_vals[i], mu_uniq[idx], var_uniq[idx])
        end
        return ll_vec
    end

    # objective: minimize negative log-likelihood in eta space
    f_obj(eta) = -sum(ll_vector(eta))
    function g_grad!(G, eta)
        ForwardDiff.gradient!(G, f_obj, eta)
    end

    println("  -> solving mle for 8 parameters (native eta-space)...")
    d_loss  = OnceDifferentiable(f_obj, g_grad!, eta_init)
    opt_res = optimize(d_loss, eta_lower, eta_upper, eta_init, Fminbox(LBFGS()))
    eta_hat = Optim.minimizer(opt_res)

    # jacobian computed natively in eta space, no post-hoc scaling required
    println("  -> calculating score matrix and fim...")
    S_matrix = ForwardDiff.jacobian(ll_vector, eta_hat)
    FIM      = S_matrix' * S_matrix

    eig_res         = eigen(Symmetric(FIM))
    eig_vals        = abs.(eig_res.values)
    sorted_idx      = sortperm(eig_vals, rev=true)
    eig_vals_sorted = eig_vals[sorted_idx]
    U_matrix_sorted = eig_res.vectors[:, sorted_idx]

    # stash scenario for uq module
    uq_scenarios[N] = (
        eta_hat  = eta_hat,
        U_matrix = U_matrix_sorted,
        eig_vals = eig_vals_sorted
    )

    df_eigen = DataFrame(
        Component  = ["Eig_1(max)", "Eig_2", "Eig_3", "Eig_4", "Eig_5", "Eig_6", "Eig_7", "Eig_8(min)"],
        Eigenvalue = eig_vals_sorted
    )

    xi_metric = eig_vals_sorted[end] / eig_vals_sorted[1]
    df_xi     = DataFrame(SampleSize_N = [N], Xi_Metric = [xi_metric])

    metrics  = compute_coordinate_metrics(S_matrix)
    df_coord = DataFrame(Parameter = param_names, Coordinate_Metric = metrics)

    U_abs   = abs.(U_matrix_sorted)
    df_heat = DataFrame(U_abs, :auto)
    rename!(df_heat, ["U1", "U2", "U3", "U4", "U5", "U6", "U7", "U8"])
    df_heat.Parameter = param_names
    select!(df_heat, :Parameter, Not(:Parameter))

    out_filename = joinpath(out_dir, "PIA_MLE_Results_Int_N$(N).xlsx")
    XLSX.writetable(out_filename,
        "Eigenvalues"       => df_eigen,
        "Coordinate_Metric" => df_coord,
        "Xi_Metric"         => df_xi,
        "Heatmap_U"         => df_heat,
        overwrite = true
    )
    println("  -> successfully exported pia metrics to: $out_filename")
end

println("\n=== grid analysis complete. starting uncertainty quantification (uq) ===")

# ==============================================================================
# 3. uq module: strict null space perturbation (wang-style)
# ==============================================================================
Random.seed!(20261010)   # [SEED]
num_samples              = 1000
perturbation_scale       = 0.5
relative_eigen_threshold = 1e-8

for N_val in N_list
    println("\nrunning uq for scenario: N = $N_val ...")

    if !haskey(uq_scenarios, N_val)
        println("  scenario not found in stashed data. skipping.")
        continue
    end

    stash      = uq_scenarios[N_val]
    lambda_max = maximum(stash.eig_vals)

    # extract directions in the null space (structural unidentifiability)
    null_indices = findall((stash.eig_vals ./ lambda_max) .< relative_eigen_threshold)

    if isempty(null_indices)
        println("  no eigenvalues below relative threshold. defaulting to smallest component (u8).")
        null_indices = [8]
    end

    U_null_space = stash.U_matrix[:, null_indices]
    println("  identified $(length(null_indices)) null basis vector(s).")

    sampled_thetas = []
    attempts       = 0
    max_attempts   = num_samples * 10

    # strict linear additive perturbation in log-parameter space
    while length(sampled_thetas) < num_samples && attempts < max_attempts
        attempts += 1
        z       = randn(length(null_indices)) .* perturbation_scale
        eta_new = stash.eta_hat .+ (U_null_space * z)

        # dynamic step shrinking: preserve null space direction while enforcing bounds
        shrink_factor = 1.0
        while !(all(eta_new .>= eta_lower) && all(eta_new .<= eta_upper)) && shrink_factor > 1e-4
            shrink_factor *= 0.5
            eta_new = stash.eta_hat .+ (U_null_space * (z * shrink_factor))
        end

        if all(eta_new .>= eta_lower) && all(eta_new .<= eta_upper)
            push!(sampled_thetas, eta_to_theta(eta_new))
        end
    end

    println("  generated $(length(sampled_thetas)) valid structurally perturbed parameter sets.")

    if isempty(sampled_thetas)
        println("  [warning] failed to generate any valid samples within boundaries. skipping uq for n = $N_val.")
        continue
    end

    # smooth_times 0-1095 days covers the post-intervention window (t_int = 910, data 909-956)
    smooth_times      = collect(0:1.0:1095)
    prediction_matrix = zeros(length(sampled_thetas), length(smooth_times))

    println("  executing forward simulations for uq envelopes...")
    for (idx, theta) in enumerate(sampled_thetas)
        preds = model_output(theta, smooth_times)
        prediction_matrix[idx, :] = preds[1:length(smooth_times)]
    end

    lower_95    = zeros(length(smooth_times))
    upper_95    = zeros(length(smooth_times))
    mean_pred = zeros(length(smooth_times))

    for j in 1:length(smooth_times)
        sorted_preds   = sort(prediction_matrix[:, j])
        mean_pred[j] = mean(sorted_preds)
        lower_95[j]    = quantile(sorted_preds, 0.025)
        upper_95[j]    = quantile(sorted_preds, 0.975)
    end

    df_uq = DataFrame(
        Time_Days         = smooth_times,
        Mean_Prediction = mean_pred,
        Lower_95_CI       = lower_95,
        Upper_95_CI       = upper_95
    )

    uq_file = joinpath(out_dir, "PIA_MLE_UQ_Envelop_Int_N$(N_val).csv")
    CSV.write(uq_file, df_uq)
    println("  -> uq envelope saved to: $uq_file")
end

println("\n=== full interventional mle-pia and uq execution completed successfully ===")

# ==============================================================================
# 4. [DIAG] optional: FIM and coordinate metric at the TRUE parameter values
#    (no refit; identical eta for every N, so differences across N reflect the
#    data only). Written to separate files; not used by Fig 3 / S7.
# ==============================================================================
const RUN_TRUEPARAM_DIAGNOSTIC = true
if RUN_TRUEPARAM_DIAGNOSTIC
    # data-generating values from run_sim_intervention_single.R:
    #   half-life 14 d, B0 0.3, E[C] 0.2, Var(C) 0.25 (=> E[C^2] = 0.29),
    #   beta0 log(1/15), beta_sin 0.5, beta_cos 0.2, beta_int log(1-0.60)
    eta_true = [log(log(2)/14), log(0.3), log(0.2), log(0.29), log(1/15), 0.5, 0.2, log(0.4)]
    println("\n=== [DIAG] FIM at true parameters ===")
    for N in N_list
        raw_data     = DataFrame(XLSX.readtable(file_path, "N$(N)_Data"))
        unique_times = sort(unique(raw_data.time))
        time_to_idx  = Dict(t => i for (i, t) in enumerate(unique_times))
        indices      = [time_to_idx[t] for t in raw_data.time]
        y_vals       = max.(raw_data.antibody, 1e-10)
        function ll_vec_true(eta)
            theta = eta_to_theta(eta)
            gamma, B0, C_bar, C2_bar, beta0, beta_sin, beta_cos, beta_int = theta
            T_type = eltype(eta)
            mu_u  = [calc_mu( t, gamma, B0, C_bar, beta0, beta_sin, beta_cos, beta_int) for t in unique_times]
            var_u = [calc_var(t, gamma, C2_bar, beta0, beta_sin, beta_cos, beta_int)    for t in unique_times]
            return T_type[gamma_logpdf(y_vals[i], mu_u[indices[i]], var_u[indices[i]]) for i in 1:length(y_vals)]
        end
        S_true   = ForwardDiff.jacobian(ll_vec_true, eta_true)
        FIM_true = S_true' * S_true
        er       = eigen(Symmetric(FIM_true))
        ev       = abs.(er.values); si = sortperm(ev, rev=true)
        df_e     = DataFrame(Component = ["Eig_1(max)","Eig_2","Eig_3","Eig_4","Eig_5","Eig_6","Eig_7","Eig_8(min)"],
                             Eigenvalue = ev[si])
        df_c     = DataFrame(Parameter = param_names, Coordinate_Metric = compute_coordinate_metrics(S_true))
        Uabs     = abs.(er.vectors[:, si])
        df_h     = DataFrame(Uabs, :auto); rename!(df_h, ["U1","U2","U3","U4","U5","U6","U7","U8"])
        df_h.Parameter = param_names; select!(df_h, :Parameter, Not(:Parameter))
        f_out = joinpath(out_dir, "PIA_TrueParam_Results_Int_N$(N).xlsx")
        XLSX.writetable(f_out, "Eigenvalues" => df_e, "Coordinate_Metric" => df_c,
                        "Heatmap_U" => df_h, overwrite = true)
        println("  N = $N -> $f_out")
    end
end
println("\n=== [DIAG] done ===")
