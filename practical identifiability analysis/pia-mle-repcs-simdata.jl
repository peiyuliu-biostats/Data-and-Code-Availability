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

# measurement noise variance: sigma_epsilon = 0.15 (sd) in R data-generating script
# therefore sigma2_eps = 0.15^2 = 0.0225
const SIGMA2_EPS = 0.0225

# safe dual extraction for AD
_extract_val(x) = float(x)
_extract_val(x::ForwardDiff.Dual) = float(ForwardDiff.value(x))

# exposure rate function
function exposure_rate(s, beta0, beta_sin, beta_cos)
    return exp(beta0 + beta_sin * sin(omega * s) + beta_cos * cos(omega * s))
end

# mean mapping (theoretical μ*)
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

# gamma log-PDF with strict singularity protection
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
        proj_resid  = s_i - U_r * (U_r' * s_i)
        metrics[i]  = norm(proj_resid, Inf)
    end
    return metrics
end

# transforms log-space (eta) back to physical space (theta)
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
# 1. Global Setup & Data Import
# ==============================================================================
file_path = "C:/Users/TopSoarer/Desktop/MSP data-2026/4-23-PIA/Simulation datasets/repeated cs/Sim_Data_Rep1_M24_Combined.xlsx"
out_dir   = "C:/Users/TopSoarer/Desktop/MSP data-2026/4-23-PIA/Simulation datasets/Simulation datasets PIA results/repeated cs/"

if !isdir(out_dir)
    mkpath(out_dir)
end

N_list = [15, 70, 120, 300]
T_list = [6, 12, 18, 24]

# η = [log(γ), log(B0), log(C_bar), log(C2_bar), β0, β_sin, β_cos]
eta_init  = [log(0.05), log(0.3), log(0.2), log(0.25), log(1/15), 0.5, 0.2]
eta_lower = [log(1e-4), log(1e-4), log(1e-4), log(1e-4), -10.0, -5.0, -5.0]
eta_upper = [log(1.0),  log(5.0),  log(5.0),  log(5.0),   5.0,   5.0,  5.0]

param_names  = ["gamma", "B0", "C_bar", "C2_bar", "beta0", "beta_sin", "beta_cos"]
uq_scenarios = Dict()
uq_targets   = [(6, 15), (12, 120), (18, 120), (24, 300)]

println("=== Starting Strict η-Space MLE-based PIA Grid Analysis (RepCS Simulation) ===")
println("    Observable variance: V_obs = V* + sigma2_eps, sigma2_eps = $SIGMA2_EPS")

# ==============================================================================
# 2. Main Nested Loop: Grid (T, N)
# ==============================================================================
for N in N_list
    sheet_name = "Rep1_M24_N$N"
    println("\nLoading data: $sheet_name ...")
    raw_data = DataFrame(XLSX.readtable(file_path, sheet_name))

    for T in T_list
        println("  Processing scenario: T = $T months, N = $N ...")
        sub_data = filter(row -> row.survey_month <= T, raw_data)

        unique_times = sort(unique(sub_data.time))
        time_to_idx  = Dict(t => i for (i, t) in enumerate(unique_times))
        indices      = [time_to_idx[t] for t in sub_data.time]
        y_vals       = max.(sub_data.antibody, 1e-10)

        # vectorized log-likelihood natively accepting η
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

        d_loss  = OnceDifferentiable(f_obj, g_grad!, eta_init)
        opt_res = optimize(d_loss, eta_lower, eta_upper, eta_init, Fminbox(LBFGS()))
        eta_hat = Optim.minimizer(opt_res)

        # jacobian natively computed in η space
        S_matrix = ForwardDiff.jacobian(ll_vector, eta_hat)
        FIM      = S_matrix' * S_matrix

        eig_res        = eigen(Symmetric(FIM))
        eig_vals       = abs.(eig_res.values)
        sorted_idx     = sortperm(eig_vals, rev=true)
        eig_vals_sorted   = eig_vals[sorted_idx]
        U_matrix_sorted   = eig_res.vectors[:, sorted_idx]

        if (T, N) in uq_targets
            uq_scenarios[(T, N)] = (
                eta_hat  = eta_hat,
                U_matrix = U_matrix_sorted,
                eig_vals = eig_vals_sorted
            )
        end

        df_eigen = DataFrame(
            Component  = ["Eig_1(max)", "Eig_2", "Eig_3", "Eig_4", "Eig_5", "Eig_6", "Eig_7(min)"],
            Eigenvalue = eig_vals_sorted
        )
        xi_metric = eig_vals_sorted[end] / eig_vals_sorted[1]
        df_xi     = DataFrame(Duration_T = [T], SampleSize_N = [N], Xi_Metric = [xi_metric])

        metrics  = compute_coordinate_metrics(S_matrix)
        df_coord = DataFrame(Parameter = param_names, Coordinate_Metric = metrics)

        U_abs    = abs.(U_matrix_sorted)
        df_heat  = DataFrame(U_abs, :auto)
        rename!(df_heat, ["U1", "U2", "U3", "U4", "U5", "U6", "U7"])
        df_heat.Parameter = param_names
        select!(df_heat, :Parameter, Not(:Parameter))

        out_filename = joinpath(out_dir, "PIA_MLE_Results_T$(T)_N$(N).xlsx")
        XLSX.writetable(out_filename,
            "Eigenvalues"      => df_eigen,
            "Coordinate_Metric" => df_coord,
            "Xi_Metric"        => df_xi,
            "Heatmap_U"        => df_heat,
            overwrite = true
        )
        println("    Successfully exported: $out_filename")
    end
end

println("\n=== Grid Analysis Complete. Starting Uncertainty Quantification (UQ) ===")

# ==============================================================================
# 3. UQ Module: Strict Null Space Perturbation (Wang-Style)
# ==============================================================================
num_samples              = 1000
perturbation_scale       = 0.5
relative_eigen_threshold = 1e-8

for target in uq_targets
    T_val, N_val = target
    println("\nRunning UQ for Scenario: T = $T_val, N = $N_val ...")

    if !haskey(uq_scenarios, target)
        println("  Scenario not found in stashed data. Skipping.")
        continue
    end

    stash      = uq_scenarios[target]
    lambda_max = maximum(stash.eig_vals)

    null_indices = findall((stash.eig_vals ./ lambda_max) .< relative_eigen_threshold)
    if isempty(null_indices)
        println("  No eigenvalues below relative threshold. Defaulting to the smallest component (U7).")
        null_indices = [7]
    end

    U_null_space = stash.U_matrix[:, null_indices]
    println("  Identified $(length(null_indices)) null basis vector(s).")

    sampled_thetas = []
    attempts       = 0
    max_attempts   = num_samples * 10

    while length(sampled_thetas) < num_samples && attempts < max_attempts
        attempts += 1
        z        = randn(length(null_indices)) .* perturbation_scale
        eta_new  = stash.eta_hat .+ (U_null_space * z)
        if all(eta_new .>= eta_lower) && all(eta_new .<= eta_upper)
            push!(sampled_thetas, eta_to_theta(eta_new))
        end
    end

    println("  Generated $(length(sampled_thetas)) valid structurally perturbed parameter sets.")

    smooth_times      = collect(0:1.0:730)
    prediction_matrix = zeros(length(sampled_thetas), length(smooth_times))

    for (idx, theta) in enumerate(sampled_thetas)
        preds = model_output(theta, smooth_times)
        prediction_matrix[idx, :] = preds[1:length(smooth_times)]
    end

lower_95  = zeros(length(smooth_times))
upper_95  = zeros(length(smooth_times))
median_pred = zeros(length(smooth_times))
mean_pred   = zeros(length(smooth_times))

for j in 1:length(smooth_times)
    col           = prediction_matrix[:, j]
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

    uq_file = joinpath(out_dir, "PIA_MLE_UQ_Envelop_T$(T_val)_N$(N_val).csv")
    CSV.write(uq_file, df_uq)
    println("  UQ envelope saved to: $uq_file")
end

println("\n=== Full MLE-PIA and UQ Execution Completed Successfully! ===")