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
# 0. Constants and Mathematical Kernels (Strictly protected)
# ==============================================================================
const omega = 2 * pi / 365.0
const U_CAP = 1e6
const DECAY_MULTIPLIER = 50.0

# measurement noise variance fixed to match simulation data-generating value:
# sim sigma_epsilon = 0.15 (sd) => sigma2_eps = 0.15^2 = 0.0225
const SIGMA2_EPS = 0 

_extract_val(x) = float(x)
_extract_val(x::ForwardDiff.Dual) = float(ForwardDiff.value(x))

function exposure_rate(s, beta0, beta_sin, beta_cos)
    return exp(beta0 + beta_sin * sin(omega * s) + beta_cos * cos(omega * s))
end

function calc_mu(t, gamma, B0, C_bar, beta0, beta_sin, beta_cos)
    g = max(_extract_val(gamma), 1e-8)
    upper_bound = min(U_CAP, DECAY_MULTIPLIER / g)
    integrand(u) = exp(-gamma * u) * exposure_rate(t - u, beta0, beta_sin, beta_cos)
    val, err = quadgk(integrand, 0.0, upper_bound; rtol=1e-5)
    return B0 + C_bar * val
end

# observable variance mapping: V_obs(t) = V*(t) + sigma2_eps
function calc_var(t, gamma, C2_bar, beta0, beta_sin, beta_cos)
    g = max(_extract_val(gamma), 1e-8)
    upper_bound = min(U_CAP, DECAY_MULTIPLIER / g)
    integrand(u) = exp(-2 * gamma * u) * exposure_rate(t - u, beta0, beta_sin, beta_cos)
    val, err = quadgk(integrand, 0.0, upper_bound; rtol=1e-5)
    return C2_bar * val + SIGMA2_EPS
end

function gamma_logpdf(y, mu, var)
    mu_safe  = max(mu,  1e-8)
    var_safe = max(var, 1e-8)
    k = max(mu_safe^2 / var_safe, 1e-8)
    theta_scale = var_safe / mu_safe
    return (k - 1) * log(y) - (y / theta_scale) - k * log(theta_scale) - loggamma(k)
end

function compute_coordinate_metrics(S_matrix)
    n_params = size(S_matrix, 2)
    metrics  = zeros(n_params)
    for i in 1:n_params
        s_i   = S_matrix[:, i]
        A     = S_matrix[:, setdiff(1:n_params, i)]
        svd_A = svd(A)
        tol   = maximum(size(A)) * eps(eltype(A)) * maximum(svd_A.S)
        r     = count(>(tol), svd_A.S)
        U_r   = svd_A.U[:, 1:r]
        proj_resid = s_i - U_r * (U_r' * s_i)
        metrics[i] = norm(proj_resid, Inf)
    end
    return metrics
end

function eta_to_theta(eta::AbstractVector{T}) where T
    return [exp(eta[1]), exp(eta[2]), exp(eta[3]), exp(eta[4]), eta[5], eta[6], eta[7]]
end

# forward model for UQ: uses observable variance V_obs = V* + sigma2_eps
function model_output(theta, times)
    gamma, B0, C_bar, C2_bar, beta0, beta_sin, beta_cos = theta
    mu_vec  = [calc_mu( t, gamma, B0,     C_bar,  beta0, beta_sin, beta_cos) for t in times]
    var_vec = [calc_var(t, gamma, C2_bar, beta0,   beta_sin, beta_cos)       for t in times]
    return vcat(mu_vec, var_vec)
end

# ==============================================================================
# 1. Global Setup & Directory Paths for Real Datasets (Rep CS)
# ==============================================================================
input_dir = "C:/Users/TopSoarer/Desktop/MSP data-2026/4-23-PIA/Real datasets/Cleaned_PIA_Inputs"
out_dir = "C:/Users/TopSoarer/Desktop/MSP data-2026/4-23-PIA/Real datasets/Real datasets PIA results/rep cs"

if !isdir(out_dir)
    mkpath(out_dir)
end

file_list = filter(x -> endswith(x, ".csv") && 
                        occursin("RepCS", x) && 
                        x != "Cleaned_Fustec_2021_RepCS.csv", 
                   readdir(input_dir))
if isempty(file_list)
    error("No Rep CS CSV files found in $input_dir")
end

# η = [log(γ), log(B0), log(C_bar), log(C2_bar), β0, β_sin, β_cos]
# *** 改动1: 单起点改为多起点，覆盖不同gamma量级，避免优化撞边界 ***
eta_inits = [
    [log(0.05),  log(0.3),  log(0.2),  log(0.25), log(1/15), 0.5,  0.2 ],
    [log(0.005), log(0.3),  log(0.2),  log(0.25), log(1/15), 0.5,  0.2 ],
    [log(0.2),   log(0.3),  log(0.2),  log(0.25), log(1/15), 0.5,  0.2 ],
    [log(0.05),  log(0.4),  log(0.3),  log(0.3),  -2.0,      0.3,  0.1 ],
    [log(0.01),  log(0.4),  log(0.15), log(0.2),  -3.0,      0.1,  0.3 ],
    [log(1.0),   log(1e-4), log(0.1),  log(0.03),  0.8,     -0.05,-0.13],
    [log(2.0),   log(1e-4), log(0.1),  log(0.03),  0.5,      0.0,  0.0 ],
]
eta_lower = [log(1e-4), log(1e-4), log(1e-4), log(1e-4), -10.0, -5.0, -5.0]
eta_upper = [log(5.0),  log(5.0),  log(5.0),  log(5.0),   5.0,   5.0,  5.0]
# *** 改动1结束 ***

param_names = ["gamma", "B0", "C_bar", "C2_bar", "beta0", "beta_sin", "beta_cos"]

num_samples              = 1000
perturbation_scale       = 0.5
relative_eigen_threshold = 1e-8

println("=== Starting Strict η-Space MLE-based PIA for Real Datasets (Rep CS) ===")
println("    Observable variance: V_obs = V* + sigma2_eps, sigma2_eps = $SIGMA2_EPS")

# ==============================================================================
# 2. Sequential Processing Loop for Each Dataset
# ==============================================================================
for file_name in file_list
    dataset_name = replace(file_name, ".csv" => "")
    println("\n[Processing Dataset]: $dataset_name ...")

    raw_data       = CSV.read(joinpath(input_dir, file_name), DataFrame)
    actual_N       = nrow(raw_data)
    actual_T_days  = maximum(raw_data.time)

    unique_times = sort(unique(raw_data.time))
    time_to_idx  = Dict(t => i for (i, t) in enumerate(unique_times))
    indices      = [time_to_idx[t] for t in raw_data.time]
    y_vals       = max.(raw_data.antibody, 1e-10)

    function ll_vector(eta)
        theta = eta_to_theta(eta)
        gamma, B0, C_bar, C2_bar, beta0, beta_sin, beta_cos = theta
        T_type   = eltype(eta)
        mu_uniq  = zeros(T_type, length(unique_times))
        var_uniq = zeros(T_type, length(unique_times))
        for i in 1:length(unique_times)
            mu_uniq[i]  = calc_mu( unique_times[i], gamma, B0,     C_bar,  beta0, beta_sin, beta_cos)
            var_uniq[i] = calc_var(unique_times[i], gamma, C2_bar, beta0,   beta_sin, beta_cos)
        end
        ll_vec = zeros(T_type, length(y_vals))
        for i in 1:length(y_vals)
            idx       = indices[i]
            ll_vec[i] = gamma_logpdf(y_vals[i], mu_uniq[idx], var_uniq[idx])
        end
        return ll_vec
    end

    f_obj(eta) = -sum(ll_vector(eta))
    function g_grad!(G, eta)
        ForwardDiff.gradient!(G, f_obj, eta)
    end

    # *** 改动2: 多起点优化，取NLL最小的结果 ***
    println("  Running multi-start optimization (Native η-space)...")
    best_nll = Inf
    eta_hat  = eta_inits[1]

    for (k, e_init) in enumerate(eta_inits)
        if !all(e_init .>= eta_lower) || !all(e_init .<= eta_upper)
            println("    Init $k: skipped (out of bounds)")
            continue
        end
        try
            d_loss_k = OnceDifferentiable(f_obj, g_grad!, e_init)
            opt_k    = optimize(d_loss_k, eta_lower, eta_upper, e_init, Fminbox(LBFGS()))
            nll_k    = Optim.minimum(opt_k)
            gamma_k  = exp(Optim.minimizer(opt_k)[1])
            beta0_k  = Optim.minimizer(opt_k)[5]
            println("    Init $k: NLL = $(round(nll_k, digits=4)),  gamma = $(round(gamma_k, digits=6)),  beta0 = $(round(beta0_k, digits=4))")
            if nll_k < best_nll
                best_nll = nll_k
                eta_hat  = Optim.minimizer(opt_k)
            end
        catch e
            println("    Init $k failed: $e")
        end
    end

    println("  Best NLL = $best_nll")
    theta_hat = eta_to_theta(eta_hat)
    println("  >>> Estimated parameters for $dataset_name:")
    println("      gamma    = ", theta_hat[1], "  (半衰期 = ", round(log(2)/theta_hat[1], digits=2), " 天)")
    println("      B0       = ", theta_hat[2])
    println("      C_bar    = ", theta_hat[3])
    println("      C2_bar   = ", theta_hat[4])
    println("      beta0    = ", theta_hat[5])
    println("      beta_sin = ", theta_hat[6])
    println("      beta_cos = ", theta_hat[7])
    println("      积分上限 upper_bound = ", min(U_CAP, DECAY_MULTIPLIER / theta_hat[1]))
    # *** 改动2结束 ***

    println("  Calculating Score Matrix and FIM...")
    S_matrix = ForwardDiff.jacobian(ll_vector, eta_hat)
    FIM      = S_matrix' * S_matrix

    eig_res         = eigen(Symmetric(FIM))
    eig_vals        = abs.(eig_res.values)
    sorted_idx      = sortperm(eig_vals, rev=true)
    eig_vals_sorted = eig_vals[sorted_idx]
    U_matrix_sorted = eig_res.vectors[:, sorted_idx]

    df_eigen = DataFrame(
        Component  = ["Eig_1(max)", "Eig_2", "Eig_3", "Eig_4", "Eig_5", "Eig_6", "Eig_7(min)"],
        Eigenvalue = eig_vals_sorted
    )
    xi_metric = eig_vals_sorted[end] / eig_vals_sorted[1]
    df_xi     = DataFrame(
        Dataset      = [dataset_name],
        Actual_N     = [actual_N],
        Max_Time_Days = [actual_T_days],
        Xi_Metric    = [xi_metric]
    )

    metrics  = compute_coordinate_metrics(S_matrix)
    df_coord = DataFrame(Parameter = param_names, Coordinate_Metric = metrics)

    U_abs   = abs.(U_matrix_sorted)
    df_heat = DataFrame(U_abs, :auto)
    rename!(df_heat, ["U1", "U2", "U3", "U4", "U5", "U6", "U7"])
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
    println("  -> Exported PIA metrics to: $out_excel")

    # ==============================================================================
    # 3. UQ Module: Strict Null Space Perturbation (Wang-Style)
    # ==============================================================================
    println("  Starting Uncertainty Quantification (UQ) exploration...")

    lambda_max   = maximum(eig_vals_sorted)
    null_indices = findall((eig_vals_sorted ./ lambda_max) .< relative_eigen_threshold)

    if isempty(null_indices)
        println("    No eigenvalues below relative threshold. Defaulting to the smallest component (U7).")
        null_indices = [7]
    end

    U_null_space = U_matrix_sorted[:, null_indices]
    println("    Identified $(length(null_indices)) null basis vector(s).")

    sampled_thetas = []
    attempts       = 0
    max_attempts   = num_samples * 10

    while length(sampled_thetas) < num_samples && attempts < max_attempts
        attempts += 1
        z       = randn(length(null_indices)) .* perturbation_scale
        eta_new = eta_hat .+ (U_null_space * z)
        if all(eta_new .>= eta_lower) && all(eta_new .<= eta_upper)
            push!(sampled_thetas, eta_to_theta(eta_new))
        end
    end

    println("    Generated $(length(sampled_thetas)) valid structurally perturbed parameter sets.")

    smooth_times      = collect(0:1.0:actual_T_days)
    prediction_matrix = zeros(length(sampled_thetas), length(smooth_times))

    println("    Executing forward simulations for UQ envelopes...")
    for (idx, theta) in enumerate(sampled_thetas)
        preds = model_output(theta, smooth_times)
        prediction_matrix[idx, :] = preds[1:length(smooth_times)]
    end

    lower_95    = zeros(length(smooth_times))
    upper_95    = zeros(length(smooth_times))
    median_pred = zeros(length(smooth_times))
    mean_pred   = zeros(length(smooth_times))

    for j in 1:length(smooth_times)
        col            = prediction_matrix[:, j]
        mean_pred[j]   = mean(col)
        lower_95[j]    = quantile(col, 0.025)
        upper_95[j]    = quantile(col, 0.975)
    end

    df_uq = DataFrame(
        Time_Days        = smooth_times,
        Mean_Prediction  = mean_pred,
        Lower_95_CI      = lower_95,
        Upper_95_CI      = upper_95
    )

    out_csv = joinpath(out_dir, "UQ_Envelop_$(dataset_name).csv")
    CSV.write(out_csv, df_uq)
    println("  -> Exported UQ envelope to: $out_csv")
end

println("\n=== All Rep CS Datasets Processed Successfully! ===")
