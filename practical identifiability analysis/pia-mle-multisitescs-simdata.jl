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
# 0. Define constants and base model functions (Multi-Site Formulation - 10 Parameters)
# ==============================================================================
const omega = 2 * pi / 365.0
const U_CAP = 1e6
const DECAY_MULTIPLIER = 50.0

# measurement noise variance: sigma_epsilon = 0.15 (sd) in R data-generating script
# therefore sigma2_eps = 0.15^2 = 0.0225
const SIGMA2_EPS = 0.0225

_extract_val(x) = float(x)
_extract_val(x::ForwardDiff.Dual) = float(ForwardDiff.value(x))

function exposure_rate(s, beta0, beta_sin, beta_cos, beta_u, beta_spc, beta_clim,
                       is_urban, is_aegypti, is_tropical)
    log_rate = beta0 + beta_spc * is_aegypti +
               beta_sin * sin(omega * s) + beta_cos * cos(omega * s) +
               beta_u * is_urban + beta_clim * is_tropical
    return exp(log_rate)
end

function calc_mu(t, gamma, B0, C_bar, beta0, beta_sin, beta_cos,
                 beta_u, beta_spc, beta_clim, is_urban, is_aegypti, is_tropical)
    g = max(_extract_val(gamma), 1e-8)
    upper_bound = min(U_CAP, DECAY_MULTIPLIER / g)
    integrand(u) = exp(-gamma * u) * exposure_rate(
        t - u, beta0, beta_sin, beta_cos, beta_u, beta_spc, beta_clim,
        is_urban, is_aegypti, is_tropical)
    val, err = quadgk(integrand, 0.0, upper_bound; rtol=1e-5)
    return B0 + C_bar * val
end

# observable variance mapping: V_obs(t) = V*(t) + sigma2_eps
function calc_var(t, gamma, C2_bar, beta0, beta_sin, beta_cos,
                  beta_u, beta_spc, beta_clim, is_urban, is_aegypti, is_tropical)
    g = max(_extract_val(gamma), 1e-8)
    upper_bound = min(U_CAP, DECAY_MULTIPLIER / g)
    integrand(u) = exp(-2 * gamma * u) * exposure_rate(
        t - u, beta0, beta_sin, beta_cos, beta_u, beta_spc, beta_clim,
        is_urban, is_aegypti, is_tropical)
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

function approx_gamma_quantiles(mu, var)
    k        = max(mu^2 / max(var, 1e-10), 1e-8)
    z        = 1.95996
    term     = 1.0 - 1.0 / (9.0 * k)
    sd_term  = z / (3.0 * sqrt(k))
    low_q    = mu * max(term - sd_term, 0.0)^3
    high_q   = mu * max(term + sd_term, 0.0)^3
    return low_q, high_q
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
    return [
        exp(eta[1]),  # gamma
        exp(eta[2]),  # B0
        exp(eta[3]),  # C_bar
        exp(eta[4]),  # C2_bar
        eta[5],       # beta0
        eta[6],       # beta_sin
        eta[7],       # beta_cos
        eta[8],       # beta_u
        eta[9],       # beta_spc
        eta[10]       # beta_clim
    ]
end

# ==============================================================================
# 1. Main Function Encapsulation
# ==============================================================================
function main()
    file_path = "C:/Users/TopSoarer/Desktop/MSP data-2026/4-23-PIA/Simulation datasets/multisites cs/Sim_Data_Rep1_Combined.xlsx"
    out_dir   = "C:/Users/TopSoarer/Desktop/MSP data-2026/4-23-PIA/Simulation datasets/Simulation datasets PIA results/multisites cs/"
    if !isdir(out_dir)
        mkpath(out_dir)
    end

    N_list = [15, 70, 120, 300]

    eta_init  = [log(0.05), log(0.3), log(0.2), log(0.25), log(1/15), 0.5, 0.2, 0.0, 0.0, 0.0]
    eta_lower = [log(1e-4), log(1e-4), log(1e-4), log(1e-4), -10.0, -5.0, -5.0, -5.0, -5.0, -5.0]
    eta_upper = [log(1.0),  log(5.0),  log(5.0),  log(5.0),   5.0,   5.0,   5.0,   5.0,   5.0,   5.0]

    param_names = ["gamma", "B0", "C_bar", "C2_bar", "beta0",
                   "beta_sin", "beta_cos", "beta_U", "beta_spc", "beta_clim"]

    uq_scenarios = Dict()

    xf               = XLSX.readxlsx(file_path)
    available_sheets = XLSX.sheetnames(xf)

    println("=== Start Multi-site Model (10D Parameters) Individual MLE-PIA Analysis ===")
    println("    Observable variance: V_obs = V* + sigma2_eps, sigma2_eps = $SIGMA2_EPS")

    for N in N_list
        sheet_name = first(filter(s -> occursin("N$N", s), available_sheets))
        println("\nLoading data: $sheet_name ...")

        raw_data  = DataFrame(XLSX.readtable(file_path, sheet_name))
       raw_data  = filter(row -> row.site_name in ["Site_A", "Site_B", "Site_D"], raw_data)
       # filter simulation outliers: remove observations exceeding biological plausibility threshold
      raw_data  = filter(row -> row.antibody <= 7.0, raw_data)
      println("   After outlier filtering (antibody <= 7.0): $(nrow(raw_data)) observations retained.")

        conds = [(
            Float64(row.time),
            row.is_urban  == 1          ? 1.0 : 0.0,
            row.species   == "Ae. aegypti" ? 1.0 : 0.0,
            row.climate   == "Tropical"  ? 1.0 : 0.0
        ) for row in eachrow(raw_data)]

        unique_conds = unique(conds)
        cond_to_idx  = Dict(c => i for (i, c) in enumerate(unique_conds))
        indices      = [cond_to_idx[c] for c in conds]
        y_vals       = max.(raw_data.antibody, 1e-10)

        function ll_vector(eta)
            theta = eta_to_theta(eta)
            gamma, B0, C_bar, C2_bar, beta0, beta_sin, beta_cos, beta_u, beta_spc, beta_clim = theta
            T_type   = eltype(eta)
            mu_uniq  = zeros(T_type, length(unique_conds))
            var_uniq = zeros(T_type, length(unique_conds))
            for i in 1:length(unique_conds)
                t, u_val, spc_val, clim_val = unique_conds[i]
                mu_uniq[i]  = calc_mu( t, gamma, B0, C_bar, beta0, beta_sin, beta_cos,
                                        beta_u, beta_spc, beta_clim,
                                        u_val, spc_val, clim_val)
                var_uniq[i] = calc_var(t, gamma, C2_bar, beta0, beta_sin, beta_cos,
                                        beta_u, beta_spc, beta_clim,
                                        u_val, spc_val, clim_val)
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

        println("   Solving for optimal 10D parameters based on individual data...")
        d_loss  = OnceDifferentiable(f_obj, g_grad!, eta_init)
        opt_res = optimize(d_loss, eta_lower, eta_upper, eta_init, Fminbox(LBFGS()))
        eta_hat = Optim.minimizer(opt_res)

        println("   Calculating Jacobian and FIM...")
        S_matrix = ForwardDiff.jacobian(ll_vector, eta_hat)
        FIM      = S_matrix' * S_matrix

        eig_res         = eigen(Symmetric(FIM))
        eig_vals        = abs.(eig_res.values)
        sorted_idx      = sortperm(eig_vals, rev=true)
        eig_vals_sorted = eig_vals[sorted_idx]
        U_matrix_sorted = eig_res.vectors[:, sorted_idx]

        uq_scenarios[N] = (
            eta_hat  = eta_hat,
            U_matrix = U_matrix_sorted,
            eig_vals = eig_vals_sorted
        )

        comp_names = ["Eig_1(max)", "Eig_2", "Eig_3", "Eig_4", "Eig_5",
                      "Eig_6", "Eig_7", "Eig_8", "Eig_9", "Eig_10(min)"]
        df_eigen = DataFrame(Component = comp_names, Eigenvalue = eig_vals_sorted)

        xi_metric = eig_vals_sorted[end] / eig_vals_sorted[1]
        df_xi     = DataFrame(SampleSize_N = [N], Xi_Metric = [xi_metric])

        metrics  = compute_coordinate_metrics(S_matrix)
        df_coord = DataFrame(Parameter = param_names, Coordinate_Metric = metrics)

        U_abs   = abs.(U_matrix_sorted)
        df_heat = DataFrame(U_abs, :auto)
        rename!(df_heat, ["U1", "U2", "U3", "U4", "U5", "U6", "U7", "U8", "U9", "U10"])
        df_heat.Parameter = param_names
        select!(df_heat, :Parameter, Not(:Parameter))

        out_filename = joinpath(out_dir, "PIA_Multisite_MLE_Results_N$(N).xlsx")
        XLSX.writetable(out_filename,
            "Eigenvalues"       => df_eigen,
            "Coordinate_Metric" => df_coord,
            "Xi_Metric"         => df_xi,
            "Heatmap_U"         => df_heat,
            overwrite = true
        )
        println("   Results for N=$N exported to: $out_filename")

        # ======================================================================
        # 3. UQ Module
        # ======================================================================
        println("   Starting Uncertainty Quantification (UQ)...")
        num_samples              = 1000
        perturbation_scale       = 0.5
        relative_eigen_threshold = 1e-8

        lambda_max   = maximum(eig_vals_sorted)
        null_indices = findall((eig_vals_sorted ./ lambda_max) .< relative_eigen_threshold)
        if isempty(null_indices)
            null_indices = [10]
        end

        U_null_space = U_matrix_sorted[:, null_indices]

        sampled_thetas = []
        attempts       = 0
        while length(sampled_thetas) < num_samples && attempts < num_samples * 10
            attempts += 1
            z       = randn(length(null_indices)) .* perturbation_scale
            eta_new = eta_hat .+ (U_null_space * z)

            shrink_factor = 1.0
            while !(all(eta_new .>= eta_lower) && all(eta_new .<= eta_upper)) &&
                    shrink_factor > 1e-4
                shrink_factor *= 0.5
                eta_new = eta_hat .+ (U_null_space * (z * shrink_factor))
            end

            if all(eta_new .>= eta_lower) && all(eta_new .<= eta_upper)
                push!(sampled_thetas, eta_to_theta(eta_new))
            end
        end

        if isempty(sampled_thetas)
            println("   Warning: All parameters out of bounds, skipping UQ.")
            continue
        end

        smooth_times = collect(0:1.0:730)

        site_conds = unique([
            (row.site_name,
             row.is_urban  == 1           ? 1.0 : 0.0,
             row.species   == "Ae. aegypti" ? 1.0 : 0.0,
             row.climate   == "Tropical"   ? 1.0 : 0.0)
            for row in eachrow(raw_data)
        ])

        for sc in site_conds
            site_name, u_val, spc_val, clim_val = sc

            prediction_matrix = zeros(length(sampled_thetas), length(smooth_times))
            pi_lower_matrix   = zeros(length(sampled_thetas), length(smooth_times))
            pi_upper_matrix   = zeros(length(sampled_thetas), length(smooth_times))

            for (idx, theta) in enumerate(sampled_thetas)
                gamma, B0, C_bar, C2_bar, beta0, beta_sin, beta_cos,
                    beta_u, beta_spc, beta_clim = theta
                for (j, t) in enumerate(smooth_times)
                    m = calc_mu( t, gamma, B0, C_bar, beta0, beta_sin, beta_cos,
                                  beta_u, beta_spc, beta_clim, u_val, spc_val, clim_val)
                    v = calc_var(t, gamma, C2_bar, beta0, beta_sin, beta_cos,
                                  beta_u, beta_spc, beta_clim, u_val, spc_val, clim_val)
                    prediction_matrix[idx, j] = m
                    q_low, q_high = approx_gamma_quantiles(m, v)
                    pi_lower_matrix[idx, j]   = q_low
                    pi_upper_matrix[idx, j]   = q_high
                end
            end

            df_uq = DataFrame(
                Time_Days         = smooth_times,
                Mean_Prediction = [mean(prediction_matrix[:, j]) for j in 1:length(smooth_times)],
                Lower_95_CI       = [quantile(prediction_matrix[:, j], 0.025) for j in 1:length(smooth_times)],
                Upper_95_CI       = [quantile(prediction_matrix[:, j], 0.975) for j in 1:length(smooth_times)],
                Lower_95_PI       = [median(pi_lower_matrix[:, j]) for j in 1:length(smooth_times)],
                Upper_95_PI       = [median(pi_upper_matrix[:, j]) for j in 1:length(smooth_times)]
            )

            uq_file = joinpath(out_dir,
                "PIA_MLE_UQ_Envelop_Multisite_N$(N)_$(site_name).csv")
            CSV.write(uq_file, df_uq)
        end
        println("   UQ: Successfully generated prediction files for all sites.")
    end

    println("\n=== Multi-site MLE-PIA Joint Model Analysis Completed! ===")
end

main()