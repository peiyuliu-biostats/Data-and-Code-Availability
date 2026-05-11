# ==============================================================================
# stable inference for repeated cross-sectional simulation data
# updated: fixed gamma, b0, sigma_eps2; estimate remaining 5 parameters
# ==============================================================================

using DataFrames
using RData
using CodecZlib
using Distributions
using Optimization
using OptimizationOptimJL
using ForwardDiff
using StatsBase
using Statistics
using Printf
using CSV

# ------------------------------------------------------------------------------
# 1. parse task id from slurm (1 to 40 for 2000 replicates)
# ------------------------------------------------------------------------------
if length(ARGS) == 0
    println("no task id provided. defaulting to 1 for local testing.")
    TASK_ID = 1
else
    TASK_ID = parse(Int, ARGS[1])
end
TASK_STR = @sprintf("%03d", TASK_ID)

# ------------------------------------------------------------------------------
# 2. global configurations & parameters
# ------------------------------------------------------------------------------
const INPUT_DIR  = "/home/peiyu.liu/3-29-repcs-inference/sim_results/"
const OUTPUT_DIR = "/home/peiyu.liu/4-26-inference-sim-repcs/infer-results-sim-repcs/"

mkpath(OUTPUT_DIR)

# --- fixed known parameters (gamma, b0, and sigma_eps2) ---
const TRUE_GAMMA      = log(2) / 21
const TRUE_B0         = 0.3
const TRUE_SIGMA_EPS2 = 0.0225

# --- parameters to estimate (5 parameters) ---
const TRUE_SIGMA_C  = sqrt(log(0.15 / 0.1^2 + 1))
const TRUE_MU_C     = log(0.1) - 0.5 * TRUE_SIGMA_C^2
const TRUE_BETA0    = log(1 / 15)
const TRUE_BETA_SIN = 0.5
const TRUE_BETA_COS = 0.2

const THETA_TRUE  = [TRUE_MU_C, TRUE_SIGMA_C, TRUE_BETA0, TRUE_BETA_SIN, TRUE_BETA_COS]
const PARAM_NAMES = ["mu_c", "sigma_c", "beta0", "beta_sin", "beta_cos"]

# weakly informative prior (offset by 20% for regularization anchor)
const THETA_PRIOR = Float64.(THETA_TRUE .* 1.20)

const REG_LAMBDA  = 0.5
const OMEGA = 2 * π / 365.0

const DS = 1.0
const S_GRID = collect(0.0:DS:1095.0)

# ------------------------------------------------------------------------------
# 3. model logic: observable moments now include sigma_eps2 in variance
# ------------------------------------------------------------------------------
function compute_latent_moments(t_obs, θ)
    # unpack the 5 estimated parameters
    μ_c, σ_c, β0, β_sin, β_cos = θ

    # inject the fixed global parameters
    γ  = TRUE_GAMMA
    B0 = TRUE_B0

    C_mean   = exp(μ_c + σ_c^2 / 2)
    C_sq_bar = exp(2 * μ_c + 2 * σ_c^2)

    exp_gamma_s  = exp.(-γ .* S_GRID)
    exp_2gamma_s = exp.(-2γ .* S_GRID)

    T_type = eltype(θ)
    μ_vec = zeros(T_type, length(t_obs))
    V_vec = zeros(T_type, length(t_obs))

    @inbounds for i in eachindex(t_obs)
        t = t_obs[i]
        lambda_hist = exp.(β0 .+ β_sin .* sin.(OMEGA .* (t .- S_GRID)) .+
                           β_cos .* cos.(OMEGA .* (t .- S_GRID)))

        I1 = sum(exp_gamma_s  .* lambda_hist) * DS
        I2 = sum(exp_2gamma_s .* lambda_hist) * DS

        μ_vec[i] = B0 + C_mean * I1
        # observable variance: process variance + fixed measurement noise
        V_vec[i] = C_sq_bar * I2 + TRUE_SIGMA_EPS2
    end
    return μ_vec, V_vec
end

function gamma_nll(θ, data; use_reg=false)
    # guard: sigma_c (θ[2]) must be strictly positive
    if θ[2] <= 0.0
        return eltype(θ)(1e9) + sum(abs2, θ)
    end

    t_obs = data.time
    Y_obs = data.antibody
    μ_vec, V_vec = compute_latent_moments(t_obs, θ)
    nll = zero(eltype(θ))

    @inbounds for i in eachindex(Y_obs)
        μ = μ_vec[i]
        V = V_vec[i]

        if μ <= 0 || V <= 0 || !isfinite(μ) || !isfinite(V)
            return eltype(θ)(1e9)
        end

        κ        = (μ^2) / V
        θ_scale  = V / μ

        if κ <= 0 || θ_scale <= 0 || !isfinite(κ) || !isfinite(θ_scale)
            return eltype(θ)(1e9)
        end

        d  = Gamma(κ, θ_scale)
        yi = max(Y_obs[i], 1e-8)
        nll -= logpdf(d, yi)
    end

    if use_reg
        nll += REG_LAMBDA * sum((θ .- THETA_PRIOR).^2)
    end

    return nll
end

function fit_model(obs_data, use_reg::Bool)
    θ_init = Float64.(THETA_TRUE .* 1.05)
    optf   = OptimizationFunction((θ, p) -> gamma_nll(θ, p, use_reg=use_reg),
                                  Optimization.AutoForwardDiff())
    prob   = OptimizationProblem(optf, θ_init, obs_data)
    sol    = solve(prob, LBFGS(); maxiters=1000)
    return sol
end

# ------------------------------------------------------------------------------
# 4. universal data extractor
# ------------------------------------------------------------------------------
function extract_named_list(container)
    keys_out = String[]
    dfs_out  = DataFrame[]

    if container isa AbstractDict
        for (k, v) in pairs(container)
            push!(keys_out, string(k))
            push!(dfs_out, v isa DataFrame ? v : DataFrame(v))
        end
        return keys_out, dfs_out
    end

    if container isa AbstractVector
        for f in fieldnames(typeof(container))
            val = getfield(container, f)
            if val isa AbstractDict
                sorted_pairs = sort(collect(val), by = p -> p[2])
                for (k, idx) in sorted_pairs
                    push!(keys_out, string(k))
                    v = container[idx]
                    push!(dfs_out, v isa DataFrame ? v : DataFrame(v))
                end
                return keys_out, dfs_out
            end
        end
    end
    error("cannot extract datasets. container type: $(typeof(container))")
end

function parse_dataset_info(key_str)
    m = match(r"Rep(\d+)_M(\d+)_N(\d+)", key_str)
    if m !== nothing
        return parse(Int, m.captures[1]), parse(Int, m.captures[2]), parse(Int, m.captures[3])
    end
    return 1, 12, 100
end

# ------------------------------------------------------------------------------
# 5. process this specific batch
# ------------------------------------------------------------------------------
println("=== processing file sim_batch_$TASK_STR.rdata ===")

results_df = DataFrame(Rep=Int[], M=Int[], N=Int[], Method=String[], converged=Bool[])
for param in PARAM_NAMES
    results_df[!, param] = Float64[]
end

filepath = joinpath(INPUT_DIR, "sim_batch_$TASK_STR.RData")
if !isfile(filepath)
    error("file not found: $filepath")
end

rdata = load(filepath)
keys_list, dfs_list = extract_named_list(rdata["results_container"])
println("-> found $(length(keys_list)) datasets. starting optimization...")

for i in eachindex(keys_list)
    key_str = String(keys_list[i])
    df      = dfs_list[i] isa DataFrame ? dfs_list[i] : DataFrame(dfs_list[i])

    rep_id, M_val, N_val = parse_dataset_info(key_str)

    if i % 50 == 1
        println("   processing dataset $i / $(length(keys_list)) ...")
    end

    cols_sym = Symbol.(names(df))
    if !(:time in cols_sym) || !(:antibody in cols_sym) continue end
    obs_data = (time     = Vector{Float64}(df[!, :time]),
                antibody = Vector{Float64}(df[!, :antibody]))

    # mle
    try
        sol_mle    = fit_model(obs_data, false)
        conv_flag  = string(sol_mle.retcode) == "Success"
        push!(results_df, [rep_id, M_val, N_val, "MLE", conv_flag, sol_mle.u...])
    catch e
        push!(results_df, [rep_id, M_val, N_val, "MLE", false,
                           fill(NaN, length(PARAM_NAMES))...])
    end

    # regmle
    try
        sol_reg    = fit_model(obs_data, true)
        conv_flag  = string(sol_reg.retcode) == "Success"
        push!(results_df, [rep_id, M_val, N_val, "RegMLE", conv_flag, sol_reg.u...])
    catch e
        push!(results_df, [rep_id, M_val, N_val, "RegMLE", false,
                           fill(NaN, length(PARAM_NAMES))...])
    end
end

# ------------------------------------------------------------------------------
# 6. calculate metrics and save
# ------------------------------------------------------------------------------
clean_results = filter(row -> row.converged == true &&
                              all(!isnan(row[p]) for p in PARAM_NAMES), results_df)

if nrow(clean_results) > 0
    metrics_df = combine(groupby(clean_results, [:M, :N, :Method])) do grp
        dict = Dict{String,Any}(
            "M"      => first(grp.M),
            "N"      => first(grp.N),
            "Method" => first(grp.Method),
            "n_sims" => nrow(grp)
        )
        for (i, param) in enumerate(PARAM_NAMES)
            estimates          = grp[!, param]
            dict[param * "_Bias"] = mean(estimates) - THETA_TRUE[i]
            dict[param * "_MSE"]  = mean((estimates .- THETA_TRUE[i]).^2)
        end
        DataFrame(dict)
    end

    CSV.write(joinpath(OUTPUT_DIR, "raw_results_$TASK_STR.csv"), results_df)
    CSV.write(joinpath(OUTPUT_DIR, "metrics_$TASK_STR.csv"),     metrics_df)
    println("=== batch $TASK_STR complete and saved ===")
else
    println("=== warning: all fits failed or failed to converge in batch $TASK_STR ===")
end
