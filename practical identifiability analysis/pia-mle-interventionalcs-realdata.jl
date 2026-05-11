using DataFrames
using XLSX
using Statistics
using LinearAlgebra
using QuadGK
using ForwardDiff
using Optim
using CSV
using SpecialFunctions

# ==============================================================================
# 0. constants and interventional model formulation (8 parameters)
# ==============================================================================
const omega = 2 * pi / 365.0

# critical: set to 365.0 to match the r cleaning script logic
# (baseline was shifted to 365 days to simulate 1 year of pre-intervention dynamics)
const t_int = 365.0

# adaptive integration bounds constants
const U_CAP            = 1e6
const DECAY_MULTIPLIER = 50.0

# measurement noise variance: sigma2_eps = 0 for real data analysis
const SIGMA2_EPS = 0

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

# maps exactly 8 dimensions from log space to physical space
# beta_int is already a log-fold change and remains linear in eta space
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
# 1. global setup and directory management
# ==============================================================================
input_dir = "C:/Users/TopSoarer/Desktop/MSP data-2026/4-23-PIA/Real datasets/Cleaned_PIA_Inputs"
out_dir   = "C:/Users/TopSoarer/Desktop/MSP data-2026/4-23-PIA/Real datasets/Real datasets PIA results/interventional cs"

if !isdir(out_dir)
    mkpath(out_dir)
end

# filter strictly for interventional csv files
file_list = filter(x -> endswith(x, ".csv") && occursin("Interventional", x), readdir(input_dir))

if isempty(file_list)
    error("no interventional csv files found in the specified input directory.")
end

# 8-dimensional optimization initialization and boundaries
eta_init  = [log(0.05), log(0.3), log(0.2), log(0.25), log(1/15), 0.5,  0.2, -1.0]
eta_lower = [log(1e-4), log(1e-4), log(1e-4), log(1e-4), -10.0,  -5.0, -5.0, -10.0]
eta_upper = [log(1.0),  log(5.0),  log(5.0),  log(5.0),   5.0,    5.0,  5.0,   5.0]

param_names = ["gamma", "B0", "C_bar", "C2_bar", "beta0", "beta_sin", "beta_cos", "beta_int"]

# uq configuration
num_uq_samples           = 1000
perturbation_scale       = 0.5
relative_eigen_threshold = 1e-8

println("=== starting individual-level mle-based pia for real interventional datasets ===")
println("    observable variance: v_obs = v* + sigma2_eps, sigma2_eps = $SIGMA2_EPS")

# ==============================================================================
# 2. sequential processing loop
# ==============================================================================
for file_name in file_list
    dataset_name = replace(file_name, ".csv" => "")
    println("\n[processing dataset]: $dataset_name")

    raw_data      = CSV.read(joinpath(input_dir, file_name), DataFrame)
    actual_N      = nrow(raw_data)
    actual_T_days = maximum(raw_data.time)

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

    println("  running optimization (8 parameters)...")
    d_loss  = OnceDifferentiable(f_obj, g_grad!, eta_init)
    opt_res = optimize(d_loss, eta_lower, eta_upper, eta_init, Fminbox(LBFGS()))
    eta_hat = Optim.minimizer(opt_res)

    # jacobian computed natively in eta space, no post-hoc scaling required
    println("  calculating score matrix and fim...")
    S_matrix = ForwardDiff.jacobian(ll_vector, eta_hat)
    FIM      = S_matrix' * S_matrix

    eig_res         = eigen(Symmetric(FIM))
    eig_vals        = abs.(eig_res.values)
    sorted_idx      = sortperm(eig_vals, rev=true)
    eig_vals_sorted = eig_vals[sorted_idx]
    U_matrix_sorted = eig_res.vectors[:, sorted_idx]

    df_eigen = DataFrame(
        Component  = ["Eig_1(max)", "Eig_2", "Eig_3", "Eig_4", "Eig_5", "Eig_6", "Eig_7", "Eig_8(min)"],
        Eigenvalue = eig_vals_sorted
    )

    xi_metric = eig_vals_sorted[end] / eig_vals_sorted[1]
    df_xi     = DataFrame(
        Dataset       = [dataset_name],
        Actual_N      = [actual_N],
        Max_Time_Days = [actual_T_days],
        Xi_Metric     = [xi_metric]
    )

    metrics  = compute_coordinate_metrics(S_matrix)
    df_coord = DataFrame(Parameter = param_names, Coordinate_Metric = metrics)

    U_abs   = abs.(U_matrix_sorted)
    df_heat = DataFrame(U_abs, :auto)
    rename!(df_heat, ["U1", "U2", "U3", "U4", "U5", "U6", "U7", "U8"])
    df_heat.Parameter = param_names
    select!(df_heat, :Parameter, Not(:Parameter))

    out_excel = joinpath(out_dir, "PIA_Results_$(dataset_name).xlsx")
    XLSX.writetable(out_excel,
        "Eigenvalues"       => df_eigen,
        "Coordinate_Metric" => df_coord,
        "Xi_Metric"         => df_xi,
        "Heatmap_U"         => df_heat,
        overwrite = true
    )
    println("  -> exported pia metrics to: $out_excel")

    # ==========================================================================
    # 3. uq module for current dataset (with auto-shrink protection)
    # ==========================================================================
    println("  starting uncertainty quantification (uq) exploration...")

    lambda_max   = maximum(eig_vals_sorted)
    null_indices = findall((eig_vals_sorted ./ lambda_max) .< relative_eigen_threshold)

    if isempty(null_indices)
        println("    no eigenvalues below relative threshold. defaulting to smallest component (u8).")
        null_indices = [8]
    end

    U_null_space = U_matrix_sorted[:, null_indices]
    println("    identified $(length(null_indices)) basis vector(s) in the null space.")

    sampled_thetas = []
    attempts       = 0
    max_attempts   = num_uq_samples * 10

    while length(sampled_thetas) < num_uq_samples && attempts < max_attempts
        attempts += 1
        step    = randn(length(null_indices)) .* perturbation_scale
        eta_new = eta_hat .+ (U_null_space * step)

        # dynamic step shrinking: preserve null space direction while enforcing bounds
        shrink_factor = 1.0
        while !(all(eta_new .>= eta_lower) && all(eta_new .<= eta_upper)) && shrink_factor > 1e-4
            shrink_factor *= 0.5
            eta_new = eta_hat .+ (U_null_space * (step * shrink_factor))
        end

        if all(eta_new .>= eta_lower) && all(eta_new .<= eta_upper)
            push!(sampled_thetas, eta_to_theta(eta_new))
        end
    end

    if isempty(sampled_thetas)
        println("    [warning] failed to generate valid samples within boundaries. skipping uq.")
        continue
    end

    # smooth_times covers the full dataset tracking duration post-intervention
    smooth_times      = collect(0:1.0:actual_T_days)
    prediction_matrix = zeros(length(sampled_thetas), length(smooth_times))

    println("    executing forward simulations for uq envelopes...")
    for (idx, theta) in enumerate(sampled_thetas)
        preds = model_output(theta, smooth_times)
        # store population mean trajectories (first half of preds vector)
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

    out_csv = joinpath(out_dir, "UQ_Envelop_$(dataset_name).csv")
    CSV.write(out_csv, df_uq)
    println("  -> exported uq envelope to: $out_csv")
end

println("\n=== all real interventional datasets processed successfully ===")
