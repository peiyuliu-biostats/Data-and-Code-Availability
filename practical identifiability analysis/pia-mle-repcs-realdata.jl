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
# 0. constants and mathematical kernels (strictly protected)
# ==============================================================================
const omega            = 2 * pi / 365.0
const U_CAP            = 1e6
const DECAY_MULTIPLIER = 50.0

# measurement noise variance: sigma2_eps = 0 means no additive noise term
const SIGMA2_EPS = 0

_extract_val(x) = float(x)
_extract_val(x::ForwardDiff.Dual) = float(ForwardDiff.value(x))

function exposure_rate(s, beta0, beta_sin, beta_cos)
    return exp(beta0 + beta_sin * sin(omega * s) + beta_cos * cos(omega * s))
end

function calc_mu(t, gamma, B0, C_bar, beta0, beta_sin, beta_cos)
    g            = max(_extract_val(gamma), 1e-8)
    upper_bound  = min(U_CAP, DECAY_MULTIPLIER / g)
    integrand(u) = exp(-gamma * u) * exposure_rate(t - u, beta0, beta_sin, beta_cos)
    val, err     = quadgk(integrand, 0.0, upper_bound; rtol=1e-5)
    return B0 + C_bar * val
end

# observable variance: v_obs(t) = v*(t) + sigma2_eps
function calc_var(t, gamma, C2_bar, beta0, beta_sin, beta_cos)
    g            = max(_extract_val(gamma), 1e-8)
    upper_bound  = min(U_CAP, DECAY_MULTIPLIER / g)
    integrand(u) = exp(-2 * gamma * u) * exposure_rate(t - u, beta0, beta_sin, beta_cos)
    val, err     = quadgk(integrand, 0.0, upper_bound; rtol=1e-5)
    return C2_bar * val + SIGMA2_EPS
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

# transforms log-space eta to physical space theta
function eta_to_theta(eta::AbstractVector{T}) where T
    return [exp(eta[1]), exp(eta[2]), exp(eta[3]), exp(eta[4]), eta[5], eta[6], eta[7]]
end

# forward model for uq using observable variance v_obs = v* + sigma2_eps
function model_output(theta, times)
    gamma, B0, C_bar, C2_bar, beta0, beta_sin, beta_cos = theta
    mu_vec  = [calc_mu( t, gamma, B0,    C_bar,  beta0, beta_sin, beta_cos) for t in times]
    var_vec = [calc_var(t, gamma, C2_bar, beta0, beta_sin, beta_cos)        for t in times]
    return vcat(mu_vec, var_vec)
end

# ==============================================================================
# 1. global setup and directory paths
# ==============================================================================
input_dir = "C:/Users/TopSoarer/Desktop/MSP data-2026/4-23-PIA/Real datasets/Cleaned_PIA_Inputs"
out_dir   = "C:/Users/TopSoarer/Desktop/MSP data-2026/4-23-PIA/Real datasets/Real datasets PIA results/rep cs"

if !isdir(out_dir)
    mkpath(out_dir)
end

file_list = filter(x -> endswith(x, ".csv") &&
                        occursin("RepCS", x) &&
                        x != "Cleaned_Fustec_2021_RepCS.csv",
                   readdir(input_dir))
if isempty(file_list)
    error("no rep cs csv files found in $input_dir")
end

# eta = [log(gamma), log(b0), log(c_bar), log(c2_bar), beta0, beta_sin, beta_cos]
eta_inits = [
    [log(0.05),  log(0.3),  log(0.2),  log(0.25), log(1/15), 0.5,  0.2 ],
    [log(0.005), log(0.3),  log(0.2),  log(0.25), log(1/15), 0.5,  0.2 ],
    [log(0.2),   log(0.3),  log(0.2),  log(0.25), log(1/15), 0.5,  0.2 ],
    [log(0.05),  log(0.4),  log(0.3),  log(0.3),  -2.0,      0.3,  0.1 ],
    [log(0.01),  log(0.4),  log(0.15), log(0.2),  -3.0,      0.1,  0.3 ],
]
eta_lower = [log(1e-3), log(1e-4), log(1e-4), log(1e-4), -6.0, -3.0, -3.0]
eta_upper = [log(0.5),  log(5.0),  log(5.0),  log(5.0),   5.0,  5.0,  5.0]

param_names = ["gamma", "B0", "C_bar", "C2_bar", "beta0", "beta_sin", "beta_cos"]

num_samples              = 1000
perturbation_scale       = 0.5
relative_eigen_threshold = 1e-8

println("=== starting strict eta-space mle-based pia for real datasets (rep cs) ===")
println("    observable variance: v_obs = v* + sigma2_eps, sigma2_eps = $SIGMA2_EPS")

# ==============================================================================
# 2. sequential processing loop
# ==============================================================================
for file_name in file_list
    dataset_name = replace(file_name, ".csv" => "")
    println("\n[processing dataset]: $dataset_name ...")

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
        gamma, B0, C_bar, C2_bar, beta0, beta_sin, beta_cos = theta
        T_type   = eltype(eta)
        mu_uniq  = zeros(T_type, length(unique_times))
        var_uniq = zeros(T_type, length(unique_times))
        for i in 1:length(unique_times)
            mu_uniq[i]  = calc_mu( unique_times[i], gamma, B0,    C_bar,  beta0, beta_sin, beta_cos)
            var_uniq[i] = calc_var(unique_times[i], gamma, C2_bar, beta0, beta_sin, beta_cos)
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

    # multi-start optimization: take the result with lowest nll
    println("  running multi-start optimization (native eta-space)...")
    best_nll = Inf
    eta_hat  = eta_inits[1]

    for (k, e_init) in enumerate(eta_inits)
        if !all(e_init .>= eta_lower) || !all(e_init .<= eta_upper)
            println("    init $k: skipped (out of bounds)")
            continue
        end
        try
            d_loss_k = OnceDifferentiable(f_obj, g_grad!, e_init)
            opt_k    = optimize(d_loss_k, eta_lower, eta_upper, e_init, Fminbox(LBFGS()))
            nll_k    = Optim.minimum(opt_k)
            gamma_k  = exp(Optim.minimizer(opt_k)[1])
            beta0_k  = Optim.minimizer(opt_k)[5]
            println("    init $k: NLL = $(round(nll_k, digits=4)),  gamma = $(round(gamma_k, digits=6)),  beta0 = $(round(beta0_k, digits=4))")
            if nll_k < best_nll
                best_nll = nll_k
                eta_hat  = Optim.minimizer(opt_k)
            end
        catch e
            println("    init $k failed: $e")
        end
    end

    println("  best NLL = $best_nll")
    theta_hat = eta_to_theta(eta_hat)
    println("  >>> estimated parameters for $dataset_name:")
    println("      gamma    = ", theta_hat[1])
    println("      B0       = ", theta_hat[2])
    println("      C_bar    = ", theta_hat[3])
    println("      C2_bar   = ", theta_hat[4])
    println("      beta0    = ", theta_hat[5])
    println("      beta_sin = ", theta_hat[6])
    println("      beta_cos = ", theta_hat[7])
    println("      upper_bound = ", min(U_CAP, DECAY_MULTIPLIER / theta_hat[1]))
    println("      half_life   = ", log(2) / theta_hat[1], " days")

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
        Component  = ["Eig_1(max)", "Eig_2", "Eig_3", "Eig_4", "Eig_5", "Eig_6", "Eig_7(min)"],
        Eigenvalue = eig_vals_sorted
    )
    xi_metric = eig_vals_sorted[end] / eig_vals_sorted[1]
    df_xi = DataFrame(
        Dataset       = [dataset_name],
        Actual_N      = [actual_N],
        Max_Time_Days = [actual_T_days],
        Xi_Metric     = [xi_metric]
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
    println("  -> exported pia metrics to: $out_excel")

    # ==============================================================================
    # 3. uq module: strict null space perturbation (wang-style)
    # ==============================================================================
    println("  starting uncertainty quantification (uq) exploration...")

    lambda_max   = maximum(eig_vals_sorted)
    null_indices = findall((eig_vals_sorted ./ lambda_max) .< relative_eigen_threshold)

    if isempty(null_indices)
        println("    no eigenvalues below relative threshold, defaulting to smallest component (u7)")
        null_indices = [7]
    end

    U_null_space = U_matrix_sorted[:, null_indices]
    println("    identified $(length(null_indices)) null basis vector(s)")

    sampled_thetas = []
    attempts       = 0
    max_attempts   = num_samples * 10

    # strict linear additive perturbation in log-parameter space
    while length(sampled_thetas) < num_samples && attempts < max_attempts
        attempts += 1
        z       = randn(length(null_indices)) .* perturbation_scale
        eta_new = eta_hat .+ (U_null_space * z)
        if all(eta_new .>= eta_lower) && all(eta_new .<= eta_upper)
            push!(sampled_thetas, eta_to_theta(eta_new))
        end
    end

    println("    generated $(length(sampled_thetas)) valid structurally perturbed parameter sets")

    # smooth_times covers the full dataset range for accurate real-data envelopes
    smooth_times      = collect(0:1.0:actual_T_days)
    prediction_matrix = zeros(length(sampled_thetas), length(smooth_times))

    println("    executing forward simulations for uq envelopes...")
    for (idx, theta) in enumerate(sampled_thetas)
        preds = model_output(theta, smooth_times)
        prediction_matrix[idx, :] = preds[1:length(smooth_times)]
    end

    lower_95    = zeros(length(smooth_times))
    upper_95    = zeros(length(smooth_times))
    mean_pred   = zeros(length(smooth_times))
    median_pred = zeros(length(smooth_times))

    for j in 1:length(smooth_times)
        col            = prediction_matrix[:, j]
        mean_pred[j]   = mean(col)
        median_pred[j] = median(col)
        lower_95[j]    = quantile(col, 0.025)
        upper_95[j]    = quantile(col, 0.975)
    end

    df_uq = DataFrame(
        Time_Days         = smooth_times,
        Mean_Prediction   = mean_pred,
        Median_Prediction = median_pred,
        Lower_95_CI       = lower_95,
        Upper_95_CI       = upper_95
    )

    out_csv = joinpath(out_dir, "UQ_Envelop_$(dataset_name).csv")
    CSV.write(out_csv, df_uq)
    println("  -> exported uq envelope to: $out_csv")
end

println("\n=== all rep cs datasets processed successfully ===")
