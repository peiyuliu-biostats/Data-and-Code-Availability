"""
repeated cross-sectional antibody data: bayesian inference via mcmc
reparameterized model with identifiable parameters only
fixed: gamma (4 half-lives), b0 (2 values) -> 8 sensitivity combinations
estimated: log_a1, log_a2, beta_sin, beta_cos
outputs: posterior summaries, lambda(t) prediction bands, corner plots,
         goodness-of-fit diagnostics
"""

import os
import warnings
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import emcee
import corner

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 1. paths and global settings
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR = (
    r"C:\Users\TopSoarer\Desktop\MSP data-2026"
    r"\4-30-real-repcs-infer\Real datasets\Cleaned_PIA_Inputs"
)

OUTPUT_DIR = (
    r"C:\Users\TopSoarer\Desktop\MSP data-2026"
    r"\4-30-real-repcs-infer\results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

REPCS_FILES = {
    "ElangaNdille_2012": "Cleaned_ElangaNdille_2012_RepCS.csv",
    "Fustec_KhonKaen":   "Cleaned_Fustec_2021_KhonKaen_RepCS.csv",
    "Fustec_RoiEt":      "Cleaned_Fustec_2021_RoiEt_RepCS.csv",
}

# fixed parameter sensitivity grid
HALF_LIVES  = [21, 42, 82, 120]          # days
B0_VALUES   = [0.012, 0.044]
SIGMA_EPS2  = 0.0                   # fixed at zero for real data

OMEGA = 2.0 * np.pi / 365.0

# integration grid: 3-year burn-in at 1-day resolution
DS      = 1.0
S_GRID  = np.arange(0.0, 1096.0, DS)   # shape (1095,)

# mcmc settings
N_WALKERS   = 32
N_BURNIN    = 1000
N_SAMPLES   = 3000
N_THIN      = 1

# parameter names for reporting
PARAM_NAMES = [r"$\log A_1$", r"$\log A_2$",
               r"$\beta_{\sin}$", r"$\beta_{\cos}$"]
PARAM_KEYS  = ["log_a1", "log_a2", "beta_sin", "beta_cos"]

# box constraints for l-bfgs-b and prior support
BOUNDS = [(-15.0, 2.0),   # log_a1
          (-15.0, 2.0),   # log_a2
          ( -5.0, 5.0),   # beta_sin  (wide enough for strong seasonality)
          ( -5.0, 5.0)]   # beta_cos


# ──────────────────────────────────────────────────────────────────────────────
# 2. model: observable moments under reparameterization
# ──────────────────────────────────────────────────────────────────────────────

def compute_moments(t_obs, log_a1, log_a2, beta_sin, beta_cos, gamma, b0):
    """
    compute predicted mean and variance at each observation time.

    reparameterization:
        a1 = c_bar * exp(beta0)   (identifiable scale for mean equation)
        a2 = c2_bar * exp(beta0)  (identifiable scale for variance equation)

    observable moments:
        mu(t)  = b0 + a1 * integral_0^inf exp(-gamma*s) * u(t-s) ds
        var(t) = a2 * integral_0^inf exp(-2*gamma*s) * u(t-s) ds

    where u(s) = exp(beta_sin*sin(omega*s) + beta_cos*cos(omega*s))
    integral approximated over s_grid (3-year burn-in window)
    """
    a1 = np.exp(log_a1)
    a2 = np.exp(log_a2)

    exp_g1 = np.exp(-gamma * S_GRID)          # shape (1095,)
    exp_g2 = np.exp(-2.0 * gamma * S_GRID)

    n_t     = len(t_obs)
    mu_vec  = np.empty(n_t)
    var_vec = np.empty(n_t)

    for i, t in enumerate(t_obs):
        # seasonal forcing evaluated at historical times t - s
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


# ──────────────────────────────────────────────────────────────────────────────
# 3. gamma-distribution negative log-likelihood
# ──────────────────────────────────────────────────────────────────────────────

def gamma_nll(theta, t_obs, y_obs, gamma, b0):
    """
    negative log-likelihood under gamma observation model.
    shape kappa = mu^2 / var,  scale theta_s = var / mu.
    returns 1e9 if any moment is invalid.
    """
    log_a1, log_a2, beta_sin, beta_cos = theta
    mu_vec, var_vec = compute_moments(
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


# ──────────────────────────────────────────────────────────────────────────────
# 4. log-posterior for mcmc
# ──────────────────────────────────────────────────────────────────────────────

def log_prior(theta):
    """
    uniform prior within box constraints.
    returns 0 if inside bounds, -inf otherwise.
    """
    for val, (lo, hi) in zip(theta, BOUNDS):
        if not (lo < val < hi):
            return -np.inf
    return 0.0


def log_posterior(theta, t_obs, y_obs, gamma, b0):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    nll = gamma_nll(theta, t_obs, y_obs, gamma, b0)
    if not np.isfinite(nll):
        return -np.inf
    return lp - nll


# ──────────────────────────────────────────────────────────────────────────────
# 5. smart initialisation from data moments
# ──────────────────────────────────────────────────────────────────────────────

def initialise_theta(df_agg, gamma, b0):
    """
    moment-based starting point for optimisation and mcmc.
    flat-lambda approximation: i1_ref ~ 1/gamma, i2_ref ~ 1/(2*gamma)
    """
    i1_ref = 1.0 / gamma
    i2_ref = 1.0 / (2.0 * gamma)

    mean_y   = df_agg["mean_y"].mean()
    var_y    = df_agg["var_y"].mean()

    a1_init  = max((mean_y - b0) / i1_ref, 1e-8)
    a2_init  = max(var_y / i2_ref, 1e-8)

    log_a1_0 = np.clip(np.log(a1_init), BOUNDS[0][0] + 0.1, BOUNDS[0][1] - 0.1)
    log_a2_0 = np.clip(np.log(a2_init), BOUNDS[1][0] + 0.1, BOUNDS[1][1] - 0.1)

    return np.array([log_a1_0, log_a2_0, 0.05, 0.05])


# ──────────────────────────────────────────────────────────────────────────────
# 6. mle: find map estimate to seed mcmc walkers
# ──────────────────────────────────────────────────────────────────────────────

def run_mle(t_obs, y_obs, gamma, b0, theta0):
    """l-bfgs-b optimisation with box constraints."""
    result = minimize(
        gamma_nll,
        theta0,
        args=(t_obs, y_obs, gamma, b0),
        method="L-BFGS-B",
        bounds=BOUNDS,
        options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        # fallback: nelder-mead without bounds (softer convergence)
        result2 = minimize(
            gamma_nll,
            theta0,
            args=(t_obs, y_obs, gamma, b0),
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-6},
        )
        if result2.fun < result.fun:
            result = result2
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 7. mcmc sampling
# ──────────────────────────────────────────────────────────────────────────────

def run_mcmc(t_obs, y_obs, gamma, b0, theta_map):
    """
    ensemble mcmc via emcee.
    walkers initialised in a small gaussian ball around the map estimate.
    returns flat posterior chain after burn-in and thinning.
    uses all available cpu cores via multiprocessing pool.
    """
    ndim = len(theta_map)

    # initial walker positions: small perturbation around map
    rng    = np.random.default_rng(seed=42)
    scale  = np.array([0.1, 0.1, 0.05, 0.05])
    pos    = theta_map + scale * rng.standard_normal((N_WALKERS, ndim))

    # clip walkers to be strictly within bounds
    for j, (lo, hi) in enumerate(BOUNDS):
        pos[:, j] = np.clip(pos[:, j], lo + 0.01, hi - 0.01)

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            N_WALKERS, ndim, log_posterior,
            args=(t_obs, y_obs, gamma, b0),
            pool=pool
        )

        # burn-in
        state = sampler.run_mcmc(pos, N_BURNIN, progress=False)
        sampler.reset()

        # production run
        sampler.run_mcmc(state, N_SAMPLES, thin_by=N_THIN, progress=False)

        # flat chain: shape (n_walkers * n_samples, ndim) — for posterior summaries
        flat_chain = sampler.get_chain(flat=True)

        # full chain: shape (n_samples, n_walkers, ndim) — for trace plots
        full_chain = sampler.get_chain()

        # convergence diagnostics
        try:
            tau = sampler.get_autocorr_time(quiet=True)
        except Exception:
            tau = np.full(ndim, np.nan)

        acceptance = np.mean(sampler.acceptance_fraction)

    return flat_chain, full_chain, tau, acceptance


# ──────────────────────────────────────────────────────────────────────────────
# 8. posterior summaries
# ──────────────────────────────────────────────────────────────────────────────

def hpd_interval(samples, credible_mass=0.95):
    """
    compute highest posterior density (hpd) interval.
    """
    sorted_s = np.sort(samples)
    n        = len(sorted_s)
    n_inside = int(np.floor(credible_mass * n))
    widths   = sorted_s[n_inside:] - sorted_s[:n - n_inside]
    idx      = np.argmin(widths)
    return sorted_s[idx], sorted_s[idx + n_inside]


def posterior_summary(chain):
    """
    returns dict with mean, median, sd, and 95% hpd for each parameter.
    also computes derived quantities: amplitude r and peak day phi.
    """
    summaries = {}
    for k, key in enumerate(PARAM_KEYS):
        s = chain[:, k]
        lo, hi = hpd_interval(s)
        summaries[key] = {
            "mean":   np.mean(s),
            "median": np.median(s),
            "sd":     np.std(s),
            "hpd_lo": lo,
            "hpd_hi": hi,
        }

    # derived: a1, a2
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

    # seasonal amplitude r = sqrt(beta_sin^2 + beta_cos^2)
    r_samples = np.sqrt(chain[:, 2]**2 + chain[:, 3]**2)
    lo, hi = hpd_interval(r_samples)
    summaries["seasonal_amplitude_r"] = {
        "mean": np.mean(r_samples), "median": np.median(r_samples),
        "sd":   np.std(r_samples),  "hpd_lo": lo, "hpd_hi": hi,
    }

    # seasonal peak day phi (day of year, 1-365)
    phi_samples = (np.arctan2(chain[:, 2], chain[:, 3]) / (2.0 * np.pi) * 365.0) % 365.0
    lo, hi = hpd_interval(phi_samples)
    summaries["peak_day_phi"] = {
        "mean": np.mean(phi_samples), "median": np.median(phi_samples),
        "sd":   np.std(phi_samples),  "hpd_lo": lo, "hpd_hi": hi,
    }

    return summaries


# ──────────────────────────────────────────────────────────────────────────────
# 9. lambda(t) prediction bands
# ──────────────────────────────────────────────────────────────────────────────

def compute_lambda_bands(chain, t_grid, gamma, b0, n_posterior=2000):
    """
    draw n_posterior samples from chain, compute mu(t) and lambda_rel(t)
    for each, return pointwise 2.5%, 50%, 97.5% quantiles.

    lambda_rel(t) = a1 * integral(...) / mean(a1 * integral(...))
                  = relative seasonal exposure (dimensionless, peak=1 normalised)
    mu(t) = b0 + a1 * integral(...)  = predicted population mean antibody
    """
    idx   = np.random.choice(len(chain), size=n_posterior, replace=False)
    samps = chain[idx]

    mu_mat  = np.zeros((n_posterior, len(t_grid)))
    lam_mat = np.zeros((n_posterior, len(t_grid)))

    for s_i, theta in enumerate(samps):
        log_a1, log_a2, beta_sin, beta_cos = theta
        mu_vec, _ = compute_moments(
            t_grid, log_a1, log_a2, beta_sin, beta_cos, gamma, b0)
        mu_mat[s_i] = mu_vec

        # relative lambda(t): seasonal shape only, normalised to mean=1
        lam_t = np.exp(beta_sin * np.sin(OMEGA * t_grid) +
                       beta_cos * np.cos(OMEGA * t_grid))
        lam_mat[s_i] = lam_t / np.mean(lam_t)

    mu_lo, mu_med, mu_hi = np.percentile(mu_mat, [2.5, 50, 97.5], axis=0)
    lam_lo, lam_med, lam_hi = np.percentile(lam_mat, [2.5, 50, 97.5], axis=0)

    return (mu_lo, mu_med, mu_hi), (lam_lo, lam_med, lam_hi)


# ──────────────────────────────────────────────────────────────────────────────
# 10. goodness-of-fit diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def goodness_of_fit(chain, t_obs_unique, y_all, df, gamma, b0, n_posterior=1000):
    """
    computes gof metrics and selects two representative timepoints for
    empirical vs predicted distribution comparison:
      - t_best: timepoint with smallest absolute standardised residual
      - t_worst: timepoint with largest absolute standardised residual
    stores per-timepoint posterior predictive gamma parameters for plotting.
    """
    idx   = np.random.choice(len(chain), size=n_posterior, replace=False)
    samps = chain[idx]

    n_t = len(t_obs_unique)

    pred_mu  = np.zeros((n_posterior, n_t))
    pred_var = np.zeros((n_posterior, n_t))

    for s_i, theta in enumerate(samps):
        log_a1, log_a2, beta_sin, beta_cos = theta
        mu_vec, var_vec = compute_moments(
            t_obs_unique, log_a1, log_a2, beta_sin, beta_cos, gamma, b0)
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

    # select two representative timepoints based on standardised residuals:
    # t_best has residual closest to zero (model fits best)
    # t_worst has residual furthest from zero (model fits worst)
    abs_resid  = np.abs(std_resid)
    idx_best   = int(np.argmin(abs_resid))
    idx_worst  = int(np.argmax(abs_resid))
    t_best     = t_obs_unique[idx_best]
    t_worst    = t_obs_unique[idx_worst]

    # store posterior predictive gamma parameters at those two timepoints
    # shape: (n_posterior,) for each timepoint
    dist_points = {}
    for label, idx_t, t_val in [("best",  idx_best,  t_best),
                                 ("worst", idx_worst, t_worst)]:
        kappa_samps = pred_mu[:, idx_t]**2 / (pred_var[:, idx_t] + 1e-12)
        scale_samps = pred_var[:, idx_t]   / (pred_mu[:, idx_t]  + 1e-12)
        # observed raw values at this timepoint
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


# ──────────────────────────────────────────────────────────────────────────────
# 11. plotting functions
# ──────────────────────────────────────────────────────────────────────────────

def plot_fit_and_lambda(df, t_grid, mu_bands, lam_bands,
                        gof, tag, out_dir):
    """
    main results figure: 4-panel layout
      top-left:   observed scatter + posterior predictive mean band
      top-right:  relative lambda(t) seasonal curve with 95% band
      bottom-left:  empirical vs predicted distribution at t_best
      bottom-right: empirical vs predicted distribution at t_worst
    red dots in top-left are per-timepoint observed sample means.
    """
    t_obs_unique = np.array(sorted(df["time"].unique()))
    mu_lo, mu_med, mu_hi     = mu_bands
    lam_lo, lam_med, lam_hi  = lam_bands

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    # ── panel a: individual scatter + posterior predictive mean band ─────────
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

    # ── panel b: relative seasonal lambda(t) ─────────────────────────────────
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

    # ── panels c & d: empirical vs predicted distribution at two timepoints ──
    for col, label in enumerate(["best", "worst"]):
        ax = fig.add_subplot(gs[1, col])
        dp      = gof["dist_points"][label]
        t_val   = dp["t"]
        y_obs_t = dp["y_obs"]
        kappa_s = dp["kappa_samps"]
        scale_s = dp["scale_samps"]
        resid   = dp["std_resid"]

        # empirical density via kde
        from scipy.stats import gaussian_kde, gamma as gamma_dist
        y_range = np.linspace(max(y_obs_t.min() * 0.5, 1e-4),
                              y_obs_t.max() * 1.5, 300)
        kde = gaussian_kde(y_obs_t, bw_method="scott")
        ax.fill_between(y_range, kde(y_range), alpha=0.35,
                        color="#2c3e50", label="empirical kde")
        ax.plot(y_range, kde(y_range), color="#2c3e50", lw=1.4)

        # posterior predictive gamma density band:
        # draw 200 samples, compute gamma pdf for each, take 2.5/50/97.5
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

        # mark observed mean with vertical line
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





def plot_corner(chain, tag, out_dir):
    """
    corner plot for the 4 estimated parameters.
    """
    labels = [r"$\log A_1$", r"$\log A_2$",
              r"$\beta_{\sin}$", r"$\beta_{\cos}$"]
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


def plot_trace(chain_full, tag, out_dir):
    """
    trace plot for the 4 parameters to visually inspect mixing.
    chain_full has shape (n_steps, n_walkers, ndim).
    """
    labels = [r"$\log A_1$", r"$\log A_2$",
              r"$\beta_{\sin}$", r"$\beta_{\cos}$"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for k in range(4):
        axes[k].plot(chain_full[:, :, k], alpha=0.3, lw=0.4, color="#2980b9")
        axes[k].set_ylabel(labels[k], fontsize=9)
    axes[-1].set_xlabel("step")
    fig.suptitle(f"mcmc trace: {tag.replace('_','  ')}", fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"trace_{tag}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 12. save posterior summary to csv
# ──────────────────────────────────────────────────────────────────────────────

def save_summary_csv(summaries, convergence, tag, out_dir):
    """
    write posterior summary table as csv.
    each row is one parameter / derived quantity.
    """
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
        })
    # append convergence info
    for k, key in enumerate(PARAM_KEYS):
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
    """
    save prediction band time series for panel a and panel b.
    columns: t, mu_lo, mu_med, mu_hi, lam_lo, lam_med, lam_hi.
    sufficient for reproducing the continuous prediction bands in R.
    """
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
    """
    save data for distribution comparison panels (panel c and d).
    two files per tag, one for t_best and one for t_worst:
      - obs_<label>_<tag>.csv : raw observed antibody values at that timepoint
      - gamma_samps_<label>_<tag>.csv : posterior gamma kappa/scale samples
    """
    for label in ["best", "worst"]:
        dp = gof["dist_points"][label]

        # raw observed values at this timepoint
        pd.DataFrame({"antibody": dp["y_obs"]}).to_csv(
            os.path.join(out_dir, f"obs_{label}_{tag}.csv"), index=False)

        # posterior gamma parameters: one row per posterior sample
        pd.DataFrame({
            "kappa": dp["kappa_samps"],
            "scale": dp["scale_samps"],
        }).to_csv(
            os.path.join(out_dir, f"gamma_samps_{label}_{tag}.csv"), index=False)


def save_chain_csv(flat_chain, tag, out_dir):
    """
    save full flat posterior chain for corner plots in R.
    columns: log_a1, log_a2, beta_sin, beta_cos.
    rows: n_walkers * n_samples posterior draws.
    """
    df_out = pd.DataFrame(flat_chain, columns=PARAM_KEYS)
    path   = os.path.join(out_dir, f"chain_{tag}.csv")
    df_out.to_csv(path, index=False)
    return path


def save_trace_csv(full_chain, tag, out_dir):
    """
    save unflattened mcmc chain for trace plots in R.
    full_chain shape: (n_samples, n_walkers, n_params).
    output columns: step, walker, log_a1, log_a2, beta_sin, beta_cos.
    one row per (step, walker) combination.
    """
    n_steps, n_walkers, _ = full_chain.shape
    steps   = np.repeat(np.arange(n_steps),   n_walkers)
    walkers = np.tile(np.arange(n_walkers),    n_steps)
    params  = full_chain.reshape(-1, len(PARAM_KEYS))
    df_out  = pd.DataFrame(params, columns=PARAM_KEYS)
    df_out.insert(0, "walker", walkers)
    df_out.insert(0, "step",   steps)
    path = os.path.join(out_dir, f"trace_{tag}.csv")
    df_out.to_csv(path, index=False)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# 13. main inference loop
# ──────────────────────────────────────────────────────────────────────────────

def run_inference_for_dataset(dataset_name, df, out_dir):
    """
    run full inference pipeline for one repcs dataset
    across all 6 (gamma, b0) sensitivity combinations.
    """
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

    for t12, b0 in itertools.product(HALF_LIVES, B0_VALUES):

        gamma  = np.log(2.0) / t12
        tag    = f"{dataset_name}__t12_{t12}__b0_{str(b0).replace('.','p')}"
        subdir = os.path.join(out_dir, dataset_name)
        os.makedirs(subdir, exist_ok=True)

        print(f"\n  half-life={t12}d  gamma={gamma:.5f}/d  b0={b0}")

        # step 1: moment-based initialisation
        theta0 = initialise_theta(df_agg, gamma, b0)
        print(f"  init theta0: {theta0.round(4)}")

        # step 2: mle
        mle_result = run_mle(t_obs, y_obs, gamma, b0, theta0)
        theta_map  = mle_result.x
        nll_map    = mle_result.fun
        print(f"  mle converged={mle_result.success}  nll={nll_map:.2f}")
        print(f"  map: log_a1={theta_map[0]:.3f}  log_a2={theta_map[1]:.3f}"
              f"  beta_sin={theta_map[2]:.3f}  beta_cos={theta_map[3]:.3f}")

        # step 3: mcmc
        print(f"  running mcmc ({N_WALKERS} walkers x {N_SAMPLES} steps) ...")
        flat_chain, full_chain, tau, acceptance = run_mcmc(
            t_obs, y_obs, gamma, b0, theta_map)
        print(f"  acceptance={acceptance:.3f}  tau={np.round(tau, 1)}")

        # step 4: posterior summaries
        summaries = posterior_summary(flat_chain)

        convergence = {"tau": tau, "acceptance": acceptance}

        # step 5: lambda(t) prediction bands
        mu_bands, lam_bands = compute_lambda_bands(flat_chain, t_grid, gamma, b0)

        # step 6: goodness of fit (used only for dist_points in panel c/d)
        gof = goodness_of_fit(flat_chain, t_obs_unique, y_obs, df, gamma, b0)

        # step 7: plots
        plot_fit_and_lambda(df, t_grid, mu_bands, lam_bands, gof, tag, subdir)
        plot_corner(flat_chain, tag, subdir)
        print(f"  plots saved to {subdir}/")

        # step 8: save csvs
        save_summary_csv(summaries, convergence, tag, subdir)
        save_prediction_bands_csv(t_grid, mu_bands, lam_bands, tag, subdir)
        save_dist_csv(gof, tag, subdir)
        save_chain_csv(flat_chain, tag, subdir)
        save_trace_csv(full_chain, tag, subdir)

        # collect for cross-combination summary
        row = {
            "dataset":          dataset_name,
            "half_life_days":   t12,
            "b0":               b0,
            "mle_nll":          nll_map,
            "acceptance":       acceptance,
        }
        for key in PARAM_KEYS + ["a1", "a2", "seasonal_amplitude_r", "peak_day_phi"]:
            row[f"{key}_mean"]   = summaries[key]["mean"]
            row[f"{key}_hpd_lo"] = summaries[key]["hpd_lo"]
            row[f"{key}_hpd_hi"] = summaries[key]["hpd_hi"]
        all_summaries.append(row)

    # save combined summary for this dataset
    df_combo = pd.DataFrame(all_summaries)
    combo_path = os.path.join(out_dir, f"combined_summary_{dataset_name}.csv")
    df_combo.to_csv(combo_path, index=False)
    print(f"\n  combined summary saved: {combo_path}")

    return df_combo


# ──────────────────────────────────────────────────────────────────────────────
# 14. entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("repcs bayesian inference pipeline")
    print(f"output directory: {OUTPUT_DIR}\n")

    all_results = []

    for dataset_name, filename in REPCS_FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(filepath):
            print(f"[warning] file not found: {filepath} -- skipping")
            continue

        df = pd.read_csv(filepath)
        df = df.dropna(subset=["antibody"])
        df = df[df["antibody"] > 0].copy()

        df_results = run_inference_for_dataset(dataset_name, df, OUTPUT_DIR)
        all_results.append(df_results)

    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        final_path = os.path.join(OUTPUT_DIR, "all_datasets_summary.csv")
        df_all.to_csv(final_path, index=False)
        print(f"\nfinal combined results: {final_path}")

    print("\ndone.")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
