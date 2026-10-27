# sensitivity runs of the bayesian inference (gamma surrogate likelihood, emcee) for repeated cross-sectional data

import os
import sys
import argparse
import warnings
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, gammainc
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import emcee
import corner

warnings.filterwarnings("ignore")

HERE      = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
DATA_DIR  = os.path.join(PROJ_ROOT, "4-30-real-repcs-infer", "Real datasets", "Cleaned_PIA_Inputs")
OUTPUT_DIR = os.path.join(HERE, "outputs")

# input files
REPCS_FILES = {
    "ElangaNdille_2012": "Cleaned_ElangaNdille_2012_RepCS.csv",
    "Fustec_KhonKaen":   "Cleaned_Fustec_2021_KhonKaen_RepCS.csv",
    "Fustec_RoiEt":      "Cleaned_Fustec_2021_RoiEt_RepCS.csv",
}

# antibody half-lives (days) and baseline levels b0
HALF_LIVES  = [21, 42, 82, 120]
B0_VALUES   = [0.012, 0.044]
SIGMA_EPS2  = 0.0

# main-text setting for each cohort: (half-life, b0)
BASELINE = {"ElangaNdille_2012": (82, 0.012),
            "Fustec_KhonKaen":   (82, 0.012),
            "Fustec_RoiEt":      (82, 0.044)}
# runs: E0 re-run of the main fit; E2a/E2b assay noise sigma_eps; E3a drop the floored value; E3b left-censor <= 0.01; E1 two harmonics
RUNS = {
    "E0":  dict(),
    "E2a": dict(sigma_eps=0.05),
    "E2b": dict(sigma_eps=0.10),
    "E3a": dict(drop_floored=True),
    "E3b": dict(censor_at=0.01),
    "E1":  dict(K=2),
}
# defaults, set in main()
K_HARM     = 1
CENSOR_AT  = None
N_CORES    = os.cpu_count()

# angular frequency (1/day)
OMEGA = 2.0 * np.pi / 365.0

# integration grid (days)
DS      = 1.0
S_GRID  = np.arange(0.0, 1096.0, DS)

# mcmc settings
N_WALKERS   = 32
N_BURNIN    = 1000
N_SAMPLES   = 3000
N_THIN      = 1

PARAM_NAMES = [r"$\log A_1$", r"$\log A_2$",
               r"$\beta_{\sin}$", r"$\beta_{\cos}$"]
PARAM_KEYS  = ["log_a1", "log_a2", "beta_sin", "beta_cos"]

# uniform prior bounds; extra harmonics use (-5, 5)
BOUNDS = [(-15.0, 2.0),
          (-15.0, 2.0),
          ( -5.0, 5.0),
          ( -5.0, 5.0)]

def param_keys(K):
    return PARAM_KEYS + [f"beta_{f}{k}" for k in range(2, K + 1) for f in ("sin", "cos")]

def bounds_for(n_par):
    return BOUNDS + [(-5.0, 5.0)] * (n_par - len(BOUNDS))

# log seasonal factor with len(betas) / 2 harmonics
def log_u(t, betas):
    out = 0.0
    for k in range(len(betas) // 2):
        out = out + (betas[2 * k] * np.sin((k + 1) * OMEGA * t) +
                     betas[2 * k + 1] * np.cos((k + 1) * OMEGA * t))
    return out

# mean and variance of antibody levels at t_obs, computed once per unique time
def compute_moments(t_obs, theta, gamma, b0, sigma_eps2=SIGMA_EPS2):
    t_obs = np.asarray(t_obs, dtype=float)
    tu, inv = np.unique(t_obs, return_inverse=True)
    s_arg = tu[:, None] - S_GRID[None, :]
    u_mat = np.exp(log_u(s_arg, theta[2:]))
    i1 = (u_mat @ np.exp(-gamma * S_GRID)) * DS
    i2 = (u_mat @ np.exp(-2.0 * gamma * S_GRID)) * DS
    mu_u = b0 + np.exp(theta[0]) * i1
    var_u = np.exp(theta[1]) * i2 + sigma_eps2
    return mu_u[inv], var_u[inv]

# loop version, kept for checking
def compute_moments_original(t_obs, log_a1, log_a2, beta_sin, beta_cos, gamma, b0):
    a1 = np.exp(log_a1)
    a2 = np.exp(log_a2)

    exp_g1 = np.exp(-gamma * S_GRID)
    exp_g2 = np.exp(-2.0 * gamma * S_GRID)

    n_t     = len(t_obs)
    mu_vec  = np.empty(n_t)
    var_vec = np.empty(n_t)

    for i, t in enumerate(t_obs):
        s_arg = t - S_GRID
        u_vec = np.exp(
            beta_sin * np.sin(OMEGA * s_arg) +
            beta_cos * np.cos(OMEGA * s_arg)
        )
        i1 = np.dot(exp_g1, u_vec) * DS
        i2 = np.dot(exp_g2, u_vec) * DS

        mu_vec[i]  = b0 + a1 * i1
        var_vec[i] = a2 * i2 + SIGMA_EPS2

    return mu_vec, var_vec

# negative log-likelihood
def gamma_nll(theta, t_obs, y_obs, gamma, b0, opts=None):
    pointwise = gamma_loglik_pointwise(theta, t_obs, y_obs, gamma, b0, opts)
    if pointwise is None:
        return 1e9
    return -np.sum(pointwise)

# pointwise gamma log-likelihood; values <= censor_at are left-censored when set
def gamma_loglik_pointwise(theta, t_obs, y_obs, gamma, b0, opts=None):
    opts = opts or {}
    mu, var = compute_moments(t_obs, theta, gamma, b0, opts.get("sigma_eps2", SIGMA_EPS2))
    if np.any(mu <= 0.0) or np.any(var <= 0.0) or not np.all(np.isfinite(mu)) \
            or not np.all(np.isfinite(var)):
        return None
    kappa = mu * mu / var
    theta_s = var / mu
    yi = np.maximum(y_obs, 1e-8)
    ll = (kappa - 1.0) * np.log(yi) - yi / theta_s - kappa * np.log(theta_s) - gammaln(kappa)
    c = opts.get("censor_at")
    if c is not None:
        cens = y_obs <= c
        if np.any(cens):
            ll = ll.copy()
            ll[cens] = np.log(np.maximum(gammainc(kappa[cens], c / theta_s[cens]), 1e-300))
    return ll

# loop version, kept for checking
def gamma_nll_original(theta, t_obs, y_obs, gamma, b0):
    log_a1, log_a2, beta_sin, beta_cos = theta
    mu_vec, var_vec = compute_moments_original(
        t_obs, log_a1, log_a2, beta_sin, beta_cos, gamma, b0)

    nll = 0.0
    for i in range(len(y_obs)):
        mu  = mu_vec[i]
        var = var_vec[i]
        if mu <= 0.0 or var <= 0.0 or not np.isfinite(mu) or not np.isfinite(var):
            return 1e9
        kappa   = mu * mu / var
        theta_s = var / mu
        yi      = max(y_obs[i], 1e-8)
        nll += (
            - (kappa - 1.0) * np.log(yi)
            + yi / theta_s
            + kappa * np.log(theta_s)
            + gammaln(kappa)
        )
    return nll

# flat prior within BOUNDS
def log_prior(theta):
    for val, (lo, hi) in zip(theta, bounds_for(len(theta))):
        if not (lo < val < hi):
            return -np.inf
    return 0.0

def log_posterior(theta, t_obs, y_obs, gamma, b0, opts=None):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    nll = gamma_nll(theta, t_obs, y_obs, gamma, b0, opts)
    if not np.isfinite(nll):
        return -np.inf
    return lp - nll

# start values from the overall mean and variance
def initialise_theta(df_agg, gamma, b0):
    i1_ref = 1.0 / gamma
    i2_ref = 1.0 / (2.0 * gamma)

    mean_y   = df_agg["mean_y"].mean()
    var_y    = df_agg["var_y"].mean()

    a1_init  = max((mean_y - b0) / i1_ref, 1e-8)
    a2_init  = max(var_y / i2_ref, 1e-8)

    log_a1_0 = np.clip(np.log(a1_init), BOUNDS[0][0] + 0.1, BOUNDS[0][1] - 0.1)
    log_a2_0 = np.clip(np.log(a2_init), BOUNDS[1][0] + 0.1, BOUNDS[1][1] - 0.1)

    return np.array([log_a1_0, log_a2_0, 0.05, 0.05] + [0.0] * (2 * (K_HARM - 1)))

# map estimate (l-bfgs-b, nelder-mead if it fails)
def run_mle(t_obs, y_obs, gamma, b0, theta0, opts=None):
    result = minimize(
        gamma_nll,
        theta0,
        args=(t_obs, y_obs, gamma, b0, opts),
        method="L-BFGS-B",
        bounds=bounds_for(len(theta0)),
        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        result2 = minimize(
            gamma_nll,
            theta0,
            args=(t_obs, y_obs, gamma, b0, opts),
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-6},
        )
        if result2.fun < result.fun:
            result = result2
    return result

# emcee started around the map estimate
def run_mcmc(t_obs, y_obs, gamma, b0, theta_map, opts=None):
    ndim = len(theta_map)

    rng    = np.random.default_rng(seed=42)
    scale  = np.array([0.1, 0.1, 0.05, 0.05] + [0.05] * (ndim - 4))
    pos    = theta_map + scale * rng.standard_normal((N_WALKERS, ndim))

    for j, (lo, hi) in enumerate(bounds_for(ndim)):
        pos[:, j] = np.clip(pos[:, j], lo + 0.01, hi - 0.01)

    np.random.seed(20261012)
    with Pool(N_CORES) as pool:
        sampler = emcee.EnsembleSampler(
            N_WALKERS, ndim, log_posterior,
            args=(t_obs, y_obs, gamma, b0, opts),
            pool=pool
        )

        state = sampler.run_mcmc(pos, N_BURNIN, progress=False)
        sampler.reset()

        sampler.run_mcmc(state, N_SAMPLES, thin_by=N_THIN, progress=False)

        flat_chain = sampler.get_chain(flat=True)

        full_chain = sampler.get_chain()

        try:
            tau = sampler.get_autocorr_time(quiet=True)
        except Exception:
            tau = np.full(ndim, np.nan)

        acceptance = np.mean(sampler.acceptance_fraction)

    return flat_chain, full_chain, tau, acceptance

# shortest interval containing credible_mass of the samples
def hpd_interval(samples, credible_mass=0.95):
    sorted_s = np.sort(samples)
    n        = len(sorted_s)
    n_inside = int(np.floor(credible_mass * n))
    widths   = sorted_s[n_inside:] - sorted_s[:n - n_inside]
    idx      = np.argmin(widths)
    return sorted_s[idx], sorted_s[idx + n_inside]

# peak day and amplitude for each draw; grid search when there is more than one harmonic
def peak_and_amplitude(chain, step=0.25, chunk=4000):
    if chain.shape[1] == 4:
        r = np.sqrt(chain[:, 2]**2 + chain[:, 3]**2)
        phi = (np.arctan2(chain[:, 2], chain[:, 3]) / (2.0 * np.pi) * 365.0) % 365.0
        return phi, r
    g = np.arange(0.0, 365.0, step)
    phi = np.empty(len(chain)); r = np.empty(len(chain))
    for a in range(0, len(chain), chunk):
        b = chain[a:a + chunk, 2:]
        v = sum(b[:, [j]] * f for j, f in enumerate(
            [fn((k + 1) * OMEGA * g)[None, :] for k in range(b.shape[1] // 2) for fn in (np.sin, np.cos)]))
        phi[a:a + chunk] = g[np.argmax(v, axis=1)]
        r[a:a + chunk] = (v.max(axis=1) - v.min(axis=1)) / 2
    return phi, r

# posterior mean, median, sd, hpd and percentile intervals
def posterior_summary(chain):
    summaries = {}
    keys = param_keys(chain.shape[1] // 2 - 1)
    for k, key in enumerate(keys):
        s = chain[:, k]
        lo, hi = hpd_interval(s)
        summaries[key] = {
            "mean":   np.mean(s),
            "median": np.median(s),
            "sd":     np.std(s),
            "hpd_lo": lo,
            "hpd_hi": hi,
            "pct_lo": np.percentile(s, 2.5),
            "pct_hi": np.percentile(s, 97.5),
        }

    summaries["a1"] = {
        "mean":   np.mean(np.exp(chain[:, 0])),
        "median": np.median(np.exp(chain[:, 0])),
        "sd":     np.std(np.exp(chain[:, 0])),
        "hpd_lo": np.exp(hpd_interval(chain[:, 0])[0]),
        "hpd_hi": np.exp(hpd_interval(chain[:, 0])[1]),
    }
    summaries["a2"] = {
        "mean":   np.mean(np.exp(chain[:, 1])),
        "median": np.median(np.exp(chain[:, 1])),
        "sd":     np.std(np.exp(chain[:, 1])),
        "hpd_lo": np.exp(hpd_interval(chain[:, 1])[0]),
        "hpd_hi": np.exp(hpd_interval(chain[:, 1])[1]),
    }

    for key, col in (("a1", 0), ("a2", 1)):
        summaries[key]["pct_lo"] = np.exp(np.percentile(chain[:, col], 2.5))
        summaries[key]["pct_hi"] = np.exp(np.percentile(chain[:, col], 97.5))

    phi_samples, r_samples = peak_and_amplitude(chain)
    lo, hi = hpd_interval(r_samples)
    summaries["seasonal_amplitude_r"] = {
        "mean": np.mean(r_samples), "median": np.median(r_samples),
        "sd":   np.std(r_samples),  "hpd_lo": lo, "hpd_hi": hi,
        "pct_lo": np.percentile(r_samples, 2.5), "pct_hi": np.percentile(r_samples, 97.5),
    }

    lo, hi = hpd_interval(phi_samples)
    summaries["peak_day_phi"] = {
        "mean": np.mean(phi_samples), "median": np.median(phi_samples),
        "sd":   np.std(phi_samples),  "hpd_lo": lo, "hpd_hi": hi,
        "pct_lo": np.percentile(phi_samples, 2.5), "pct_hi": np.percentile(phi_samples, 97.5),
    }

    return summaries

# 95% bands of the mean antibody level and of the relative biting rate (mean 1)
def compute_lambda_bands(chain, t_grid, gamma, b0, n_posterior=2000, opts=None):
    np.random.seed(1)
    idx   = np.random.choice(len(chain), size=n_posterior, replace=False)
    samps = chain[idx]

    mu_mat  = np.zeros((n_posterior, len(t_grid)))
    lam_mat = np.zeros((n_posterior, len(t_grid)))
    s2 = (opts or {}).get("sigma_eps2", SIGMA_EPS2)

    for s_i, theta in enumerate(samps):
        mu_vec, _ = compute_moments(t_grid, theta, gamma, b0, s2)
        mu_mat[s_i] = mu_vec

        lam_t = np.exp(log_u(t_grid, theta[2:]))
        lam_mat[s_i] = lam_t / np.mean(lam_t)

    mu_lo, mu_med, mu_hi = np.percentile(mu_mat, [2.5, 50, 97.5], axis=0)
    lam_lo, lam_med, lam_hi = np.percentile(lam_mat, [2.5, 50, 97.5], axis=0)

    return (mu_lo, mu_med, mu_hi), (lam_lo, lam_med, lam_hi)

# posterior predictive checks of the survey means and variances
def goodness_of_fit(chain, t_obs_unique, y_all, df, gamma, b0, n_posterior=1000, opts=None):
    np.random.seed(2)
    idx   = np.random.choice(len(chain), size=n_posterior, replace=False)
    samps = chain[idx]

    n_t = len(t_obs_unique)

    pred_mu  = np.zeros((n_posterior, n_t))
    pred_var = np.zeros((n_posterior, n_t))
    s2 = (opts or {}).get("sigma_eps2", SIGMA_EPS2)

    for s_i, theta in enumerate(samps):
        mu_vec, var_vec = compute_moments(t_obs_unique, theta, gamma, b0, s2)
        pred_mu[s_i]  = mu_vec
        pred_var[s_i] = var_vec

    obs_means = df.groupby("time")["antibody"].mean().loc[t_obs_unique].values
    obs_vars  = df.groupby("time")["antibody"].var().loc[t_obs_unique].values
    obs_ns    = df.groupby("time")["antibody"].count().loc[t_obs_unique].values

    pred_mu_lo,  pred_mu_med,  pred_mu_hi  = np.percentile(pred_mu,  [2.5, 50, 97.5], axis=0)
    pred_var_lo, pred_var_med, pred_var_hi = np.percentile(pred_var, [2.5, 50, 97.5], axis=0)

    std_resid    = (obs_means - pred_mu_med) / (np.sqrt(pred_var_med / obs_ns) + 1e-12)
    coverage_mu  = np.mean((obs_means >= pred_mu_lo) & (obs_means <= pred_mu_hi))
    coverage_var = np.mean((obs_vars  >= pred_var_lo) & (obs_vars  <= pred_var_hi))

    var_pred_means = np.var(pred_mu_med)
    bayes_r2 = var_pred_means / (
        var_pred_means + np.mean((obs_means - pred_mu_med)**2) + 1e-12)

    mape = np.mean(np.abs((obs_means - pred_mu_med) / (obs_means + 1e-8))) * 100.0

    # posterior predictive p-value of a chi-square-type discrepancy for the survey means
    t_obs_stat  = np.sum(obs_ns * (obs_means - pred_mu_med)**2 / (pred_var_med + 1e-12))
    t_rep_stats = []
    for s_i in range(n_posterior):
        rep_means = np.array([
            np.mean(np.random.gamma(
                pred_mu[s_i, j]**2 / pred_var[s_i, j],
                pred_var[s_i, j] / pred_mu[s_i, j],
                size=obs_ns[j]
            )) for j in range(n_t)
        ])
        t_rep = np.sum(obs_ns * (rep_means - pred_mu[s_i])**2 /
                       (pred_var[s_i] + 1e-12))
        t_rep_stats.append(t_rep)
    ppp = np.mean(np.array(t_rep_stats) >= t_obs_stat)

    # surveys with the smallest and largest standardised residual
    abs_resid  = np.abs(std_resid)
    idx_best   = int(np.argmin(abs_resid))
    idx_worst  = int(np.argmax(abs_resid))
    t_best     = t_obs_unique[idx_best]
    t_worst    = t_obs_unique[idx_worst]

    dist_points = {}
    for label, idx_t, t_val in [("best",  idx_best,  t_best),
                                 ("worst", idx_worst, t_worst)]:
        kappa_samps = pred_mu[:, idx_t]**2 / (pred_var[:, idx_t] + 1e-12)
        scale_samps = pred_var[:, idx_t]   / (pred_mu[:, idx_t]  + 1e-12)
        y_obs_t = df[df["time"] == t_val]["antibody"].values
        dist_points[label] = {
            "t":           t_val,
            "y_obs":       y_obs_t,
            "kappa_samps": kappa_samps,
            "scale_samps": scale_samps,
            "std_resid":   std_resid[idx_t],
        }

    gof = {
        "obs_means":     obs_means,
        "obs_vars":      obs_vars,
        "obs_ns":        obs_ns,
        "pred_mu_lo":    pred_mu_lo,
        "pred_mu_med":   pred_mu_med,
        "pred_mu_hi":    pred_mu_hi,
        "pred_var_lo":   pred_var_lo,
        "pred_var_med":  pred_var_med,
        "pred_var_hi":   pred_var_hi,
        "std_resid":     std_resid,
        "coverage_mu":   coverage_mu,
        "coverage_var":  coverage_var,
        "bayes_r2":      bayes_r2,
        "mape":          mape,
        "ppp":           ppp,
        "dist_points":   dist_points,
    }
    return gof

# figure: fit, seasonal shape, and predicted vs observed distributions
def plot_fit_and_lambda(df, t_grid, mu_bands, lam_bands,
                        gof, tag, out_dir):
    t_obs_unique = np.array(sorted(df["time"].unique()))
    mu_lo, mu_med, mu_hi     = mu_bands
    lam_lo, lam_med, lam_hi  = lam_bands

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(df["time"], df["antibody"],
                s=6, alpha=0.18, color="#aaaaaa", rasterized=True, label="individual")
    for t_pt in t_obs_unique:
        sub = df[df["time"] == t_pt]["antibody"]
        ax0.plot(t_pt, sub.mean(), "o", color="#c0392b", ms=4, zorder=5)

    ax0.fill_between(t_grid, mu_lo, mu_hi,
                     alpha=0.30, color="#2980b9", label="95% ppb (mean)")
    ax0.plot(t_grid, mu_med, color="#2980b9", lw=1.8, label="median prediction")
    ax0.set_xlabel("time (days)")
    ax0.set_ylabel("antibody level (od)")
    ax0.set_title("observed data & posterior predictive band\n"
                  "(red dots = per-timepoint sample mean)", fontsize=10)
    ax0.legend(fontsize=7, loc="upper right")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.fill_between(t_grid, lam_lo, lam_hi,
                     alpha=0.30, color="#27ae60", label="95% ppb")
    ax1.plot(t_grid, lam_med, color="#27ae60", lw=1.8, label="median")
    ax1.axhline(1.0, color="#7f8c8d", lw=0.8, ls="--")
    ax1.set_xlabel("time (days)")
    ax1.set_ylabel(r"relative $\lambda(t)$ (normalised to mean=1)")
    ax1.set_title("seasonal exposure shape", fontsize=10)
    ax1.legend(fontsize=7)
    if abs(lam_med.max() / (lam_med.min() + 1e-9)) > 50:
        ax1.annotate("note: strong seasonality detected",
                     xy=(0.02, 0.96), xycoords="axes fraction",
                     fontsize=7, color="#c0392b")

    for col, label in enumerate(["best", "worst"]):
        ax = fig.add_subplot(gs[1, col])
        dp      = gof["dist_points"][label]
        t_val   = dp["t"]
        y_obs_t = dp["y_obs"]
        kappa_s = dp["kappa_samps"]
        scale_s = dp["scale_samps"]
        resid   = dp["std_resid"]

        from scipy.stats import gaussian_kde, gamma as gamma_dist
        y_range = np.linspace(max(y_obs_t.min() * 0.5, 1e-4),
                              y_obs_t.max() * 1.5, 300)
        kde = gaussian_kde(y_obs_t, bw_method="scott")
        ax.fill_between(y_range, kde(y_range), alpha=0.35,
                        color="#2c3e50", label="empirical kde")
        ax.plot(y_range, kde(y_range), color="#2c3e50", lw=1.4)

        n_draw  = min(200, len(kappa_s))
        draw_idx = np.random.choice(len(kappa_s), size=n_draw, replace=False)
        pdf_mat  = np.zeros((n_draw, len(y_range)))
        for i, si in enumerate(draw_idx):
            k = max(kappa_s[si], 1e-4)
            sc = max(scale_s[si], 1e-8)
            pdf_mat[i] = gamma_dist.pdf(y_range, a=k, scale=sc)

        pdf_lo  = np.percentile(pdf_mat, 2.5,  axis=0)
        pdf_med = np.percentile(pdf_mat, 50.0, axis=0)
        pdf_hi  = np.percentile(pdf_mat, 97.5, axis=0)

        ax.fill_between(y_range, pdf_lo, pdf_hi,
                        alpha=0.30, color="#e74c3c",
                        label="predicted gamma (95% ppb)")
        ax.plot(y_range, pdf_med, color="#e74c3c", lw=1.6,
                label="predicted gamma (median)")

        ax.axvline(y_obs_t.mean(), color="#2c3e50", lw=1.0,
                   ls="--", label=f"obs mean={y_obs_t.mean():.3f}")

        qual = "best fit" if label == "best" else "worst fit"
        ax.set_xlabel("antibody level (od)")
        ax.set_ylabel("density")
        ax.set_title(
            f"distribution gof: t={t_val:.0f}d  ({qual})\n"
            f"std resid={resid:.2f}  n={len(y_obs_t)}",
            fontsize=9)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(tag.replace("_", "  "), fontsize=11, y=1.01)
    out_path = os.path.join(out_dir, f"fit_{tag}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path

# corner plot
def plot_corner(chain, tag, out_dir):
    labels = [r"$\log A_1$", r"$\log A_2$",
              r"$\beta_{\sin}$", r"$\beta_{\cos}$"]
    labels = labels + [r"$\beta_{\sin 2}$", r"$\beta_{\cos 2}$"][:chain.shape[1] - 4]
    fig = corner.corner(
        chain,
        labels=labels,
        quantiles=[0.025, 0.5, 0.975],
        show_titles=True,
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 9},
        hist_kwargs={"density": True},
        color="#2c3e50",
    )
    fig.suptitle(tag.replace("_", "  "), fontsize=10, y=1.01)
    out_path = os.path.join(out_dir, f"corner_{tag}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

# trace plot
def plot_trace(chain_full, tag, out_dir):
    labels = [r"$\log A_1$", r"$\log A_2$",
              r"$\beta_{\sin}$", r"$\beta_{\cos}$"]
    labels = labels + [r"$\beta_{\sin 2}$", r"$\beta_{\cos 2}$"][:chain_full.shape[2] - 4]
    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 8), sharex=True)
    for k in range(len(labels)):
        axes[k].plot(chain_full[:, :, k], alpha=0.3, lw=0.4, color="#2980b9")
        axes[k].set_ylabel(labels[k], fontsize=9)
    axes[-1].set_xlabel("step")
    fig.suptitle(f"mcmc trace: {tag.replace('_','  ')}", fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"trace_{tag}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

# output files
def save_summary_csv(summaries, convergence, tag, out_dir):
    rows = []
    for key, d in summaries.items():
        rows.append({
            "dataset_tag":   tag,
            "parameter":     key,
            "post_mean":     d["mean"],
            "post_median":   d["median"],
            "post_sd":       d["sd"],
            "hpd_2.5":       d["hpd_lo"],
            "hpd_97.5":      d["hpd_hi"],
            "pct_2.5":       d.get("pct_lo", np.nan),
            "pct_97.5":      d.get("pct_hi", np.nan),
        })
    for k, key in enumerate(param_keys(len(convergence["tau"]) // 2 - 1)):
        rows.append({
            "dataset_tag": tag,
            "parameter":   f"tau_{key}",
            "post_mean":   convergence["tau"][k] if convergence["tau"] is not None else np.nan,
            "post_median": np.nan, "post_sd": np.nan,
            "hpd_2.5": np.nan, "hpd_97.5": np.nan,
        })
    rows.append({
        "dataset_tag": tag,
        "parameter":   "acceptance_fraction",
        "post_mean":   convergence["acceptance"],
        "post_median": np.nan, "post_sd": np.nan,
        "hpd_2.5": np.nan, "hpd_97.5": np.nan,
    })
    df_out = pd.DataFrame(rows)
    path   = os.path.join(out_dir, f"summary_{tag}.csv")
    df_out.to_csv(path, index=False)
    return path

def save_prediction_bands_csv(t_grid, mu_bands, lam_bands, tag, out_dir):
    mu_lo,  mu_med,  mu_hi  = mu_bands
    lam_lo, lam_med, lam_hi = lam_bands
    df_out = pd.DataFrame({
        "t":       t_grid,
        "mu_lo":   mu_lo,
        "mu_med":  mu_med,
        "mu_hi":   mu_hi,
        "lam_lo":  lam_lo,
        "lam_med": lam_med,
        "lam_hi":  lam_hi,
    })
    path = os.path.join(out_dir, f"pred_bands_{tag}.csv")
    df_out.to_csv(path, index=False)
    return path

def save_dist_csv(gof, tag, out_dir):
    for label in ["best", "worst"]:
        dp = gof["dist_points"][label]

        pd.DataFrame({"antibody": dp["y_obs"]}).to_csv(
            os.path.join(out_dir, f"obs_{label}_{tag}.csv"), index=False)

        pd.DataFrame({
            "kappa": dp["kappa_samps"],
            "scale": dp["scale_samps"],
        }).to_csv(
            os.path.join(out_dir, f"gamma_samps_{label}_{tag}.csv"), index=False)

def save_chain_csv(flat_chain, tag, out_dir):
    df_out = pd.DataFrame(flat_chain, columns=param_keys(flat_chain.shape[1] // 2 - 1))
    path   = os.path.join(out_dir, f"chain_{tag}.csv")
    df_out.to_csv(path, index=False)
    return path

def save_trace_csv(full_chain, tag, out_dir):
    n_steps, n_walkers, _ = full_chain.shape
    steps   = np.repeat(np.arange(n_steps),   n_walkers)
    walkers = np.tile(np.arange(n_walkers),    n_steps)
    keys    = param_keys(full_chain.shape[2] // 2 - 1)
    params  = full_chain.reshape(-1, len(keys))
    df_out  = pd.DataFrame(params, columns=keys)
    df_out.insert(0, "walker", walkers)
    df_out.insert(0, "step",   steps)
    path = os.path.join(out_dir, f"trace_{tag}.csv")
    df_out.to_csv(path, index=False)
    return path

# psis-loo with pointwise elpd and pareto k (draws thinned)
def save_loo_pointwise(full_chain, t_obs, y_obs, gamma, b0, opts, tag, out_dir, thin=15):
    import arviz as az
    sub = full_chain[::thin]
    n_s, n_w, _ = sub.shape
    ll = np.empty((n_w, n_s, len(y_obs)))
    for i in range(n_s):
        for w in range(n_w):
            ll[w, i] = gamma_loglik_pointwise(sub[i, w], t_obs, y_obs, gamma, b0, opts)
    dt = az.from_dict({"posterior": {"log_a1": sub[:, :, 0].T},
                       "log_likelihood": {"y": ll}})
    r = az.loo(dt, pointwise=True)
    pd.DataFrame({"t": t_obs, "y": y_obs,
                  "elpd_i": np.asarray(r.elpd_i).ravel(),
                  "pareto_k": np.asarray(r.pareto_k).ravel()}).to_csv(
        os.path.join(out_dir, f"loo_pointwise_{tag}.csv"), index=False)
    return {"elpd_loo": float(r.elpd), "elpd_loo_se": float(r.se), "p_loo": float(r.p),
            "n_pareto_k_gt_0.7": int(np.sum(np.asarray(r.pareto_k) > 0.7))}

# map, mcmc, summaries and output files for each (half-life, b0) setting
def run_inference_for_dataset(dataset_name, df, out_dir, settings=None, opts=None,
                              run_id="", save_loo=False):
    settings = settings or list(itertools.product(HALF_LIVES, B0_VALUES))
    print(f"\n{'='*60}")
    print(f"dataset: {dataset_name}  |  n={len(df)}  |  "
          f"timepoints={df['time'].nunique()}")
    print(f"{'='*60}")

    t_obs        = df["time"].values.astype(float)
    y_obs        = df["antibody"].values.astype(float)
    t_obs_unique = np.array(sorted(df["time"].unique()), dtype=float)

    df_agg = (df.groupby("time")["antibody"]
              .agg(mean_y="mean", var_y="var", n="count")
              .reset_index())

    t_grid = np.linspace(t_obs.min(), t_obs.max() + 30, 400)

    all_summaries = []

    for t12, b0 in settings:

        gamma  = np.log(2.0) / t12
        tag    = f"{dataset_name}__t12_{t12}__b0_{str(b0).replace('.','p')}" + (f"__{run_id}" if run_id else "")
        subdir = os.path.join(out_dir, dataset_name)
        os.makedirs(subdir, exist_ok=True)

        print(f"\n  half-life={t12}d  gamma={gamma:.5f}/d  b0={b0}")

        theta0 = initialise_theta(df_agg, gamma, b0)
        print(f"  init theta0: {theta0.round(4)}")

        mle_result = run_mle(t_obs, y_obs, gamma, b0, theta0, opts)
        theta_map  = mle_result.x
        nll_map    = mle_result.fun
        print(f"  mle converged={mle_result.success}  nll={nll_map:.2f}")
        print(f"  map: log_a1={theta_map[0]:.3f}  log_a2={theta_map[1]:.3f}"
              f"  beta_sin={theta_map[2]:.3f}  beta_cos={theta_map[3]:.3f}")

        print(f"  running mcmc ({N_WALKERS} walkers x {N_SAMPLES} steps) ...")
        flat_chain, full_chain, tau, acceptance = run_mcmc(
            t_obs, y_obs, gamma, b0, theta_map, opts)
        print(f"  acceptance={acceptance:.3f}  tau={np.round(tau, 1)}")

        summaries = posterior_summary(flat_chain)

        convergence = {"tau": tau, "acceptance": acceptance}

        mu_bands, lam_bands = compute_lambda_bands(flat_chain, t_grid, gamma, b0, opts=opts)

        gof = goodness_of_fit(flat_chain, t_obs_unique, y_obs, df, gamma, b0, opts=opts)

        plot_fit_and_lambda(df, t_grid, mu_bands, lam_bands, gof, tag, subdir)
        plot_corner(flat_chain, tag, subdir)
        print(f"  plots saved to {subdir}/")

        save_summary_csv(summaries, convergence, tag, subdir)
        save_prediction_bands_csv(t_grid, mu_bands, lam_bands, tag, subdir)
        save_dist_csv(gof, tag, subdir)
        save_chain_csv(flat_chain, tag, subdir)
        save_trace_csv(full_chain, tag, subdir)
        loo_row = save_loo_pointwise(full_chain, t_obs, y_obs, gamma, b0, opts, tag, subdir) if save_loo else {}

        row = {
            "dataset":          dataset_name,
            "half_life_days":   t12,
            "b0":               b0,
            "mle_nll":          nll_map,
            "acceptance":       acceptance,
            "run":              run_id,
            "n_obs":            len(y_obs),
            "max_tau":          np.nanmax(tau),
        }
        row.update(loo_row)
        for key in param_keys(len(theta_map) // 2 - 1) + ["a1", "a2", "seasonal_amplitude_r", "peak_day_phi"]:
            row[f"{key}_mean"]   = summaries[key]["mean"]
            row[f"{key}_hpd_lo"] = summaries[key]["hpd_lo"]
            row[f"{key}_hpd_hi"] = summaries[key]["hpd_hi"]
            row[f"{key}_pct_lo"] = summaries[key]["pct_lo"]
            row[f"{key}_pct_hi"] = summaries[key]["pct_hi"]
        all_summaries.append(row)

    df_combo = pd.DataFrame(all_summaries)
    combo_path = os.path.join(out_dir, f"combined_summary_{dataset_name}{('_' + run_id) if run_id else ''}.csv")
    df_combo.to_csv(combo_path, index=False)
    print(f"\n  combined summary saved: {combo_path}")

    return df_combo

# usage: python repcs_inference_v4_rev.py RUN [--datasets ...] [--cores n] [--all-settings]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", choices=list(RUNS))
    ap.add_argument("--datasets", default=",".join(REPCS_FILES))
    ap.add_argument("--cores", type=int, default=os.cpu_count())
    ap.add_argument("--all-settings", action="store_true",
                    help="all 8 (t1/2, B0) settings instead of the Fig 5/6 setting")
    a = ap.parse_args()
    global K_HARM, N_CORES
    cfg = RUNS[a.run]
    K_HARM = cfg.get("K", 1)
    N_CORES = a.cores
    opts = {"sigma_eps2": cfg.get("sigma_eps", 0.0) ** 2, "censor_at": cfg.get("censor_at")}
    out_dir = os.path.join(OUTPUT_DIR, a.run)
    os.makedirs(out_dir, exist_ok=True)
    print("repcs bayesian inference pipeline -- revision run", a.run, cfg)
    print(f"output directory: {out_dir}\n")

    all_results = []

    for dataset_name in a.datasets.split(","):
        filename = REPCS_FILES[dataset_name]
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(filepath):
            print(f"[warning] file not found: {filepath} -- skipping")
            continue

        df = pd.read_csv(filepath)
        df = df.dropna(subset=["antibody"])
        df = df[df["antibody"] > 0].copy()
        if cfg.get("drop_floored"):
            n0 = len(df)
            df = df[df["antibody"] > 1e-9].copy()
            print(f"  E3a: dropped {n0 - len(df)} value(s) floored to 1e-10")
            if n0 == len(df):
                print("  E3a: nothing to drop for this dataset -- skipped")
                continue
        if opts["censor_at"] is not None:
            print(f"  E3b: {(df['antibody'] <= opts['censor_at']).sum()} value(s) <= {opts['censor_at']} treated as left-censored")

        settings = None if a.all_settings else [BASELINE[dataset_name]]
        df_results = run_inference_for_dataset(dataset_name, df, out_dir, settings=settings,
                                               opts=opts, run_id=a.run,
                                               save_loo=a.run in ("E0", "E1"))
        all_results.append(df_results)

    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        final_path = os.path.join(out_dir, f"all_datasets_summary_{a.run}.csv")
        df_all.to_csv(final_path, index=False)
        print(f"\nfinal combined results: {final_path}")

    print("\ndone.")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
