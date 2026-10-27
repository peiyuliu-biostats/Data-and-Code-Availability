# survey spacing x antibody half-life: expected se of the peak day, checked by simulation

import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np, pandas as pd
from scipy.special import polygamma
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "04_SI_S5_robustness_sim", "engine"))
from anpp_core import TRUE, J_integrals, circ_diff
from anpp_ext import simulate_obs, GammaFitGen

OUT = os.path.join(HERE, "outputs"); os.makedirs(OUT, exist_ok=True)
# days between surveys, half-lives (days) and study length (days)
INTERVALS = [15, 30, 60, 90, 120]
HALF_LIVES = [21, 42, 82, 120]
STUDY = 720.0
BETAS = np.array([TRUE["beta_sin"], TRUE["beta_cos"]])
CBAR, C2 = TRUE["mean_c"], TRUE["var_c"] + TRUE["mean_c"] ** 2
PHI_TRUE = np.r_[np.log(CBAR) + TRUE["beta0"], np.log(C2) + TRUE["beta0"], BETAS]
B0, S2 = TRUE["B0"], TRUE["sigma_eps"] ** 2

# survey days over the study period
def survey_times(interval):
    return interval * np.arange(1, int(STUDY // interval) + 1, dtype=float)

# mean and variance at times t
def moments(phi, t, gam):
    J1, J2 = J_integrals(t, phi[2], phi[3], gam)
    return B0 + np.exp(phi[0]) * J1, np.exp(phi[1]) * J2 + S2

# expected fisher information for n people per survey (gamma observation model)
def expected_fim(t, n, gam):
    mu, V = moments(PHI_TRUE, t, gam)
    k, th = mu ** 2 / V, V / mu
    Ikt = np.array([[polygamma(1, k), 1 / th], [1 / th, k / th ** 2]]).transpose(2, 0, 1)
    D = np.zeros((len(t), 2, 4)); h = 1e-6
    for i in range(4):
        e = np.eye(4)[i] * h
        mp, Vp = moments(PHI_TRUE + e, t, gam); mm, Vm = moments(PHI_TRUE - e, t, gam)
        dmu, dV = (mp - mm) / (2 * h), (Vp - Vm) / (2 * h)
        D[:, 0, i] = 2 * mu / V * dmu - mu ** 2 / V ** 2 * dV
        D[:, 1, i] = -V / mu ** 2 * dmu + dV / mu
    return np.einsum("j,jai,jab,jbk->ik", n, D, Ikt, D)

# delta-method se of the peak day (days) and of R
def se_phi_R(cov, bs=BETAS[0], bc=BETAS[1]):
    R = np.hypot(bs, bc)
    g_phi = np.array([365 / (2 * np.pi) * bc / R ** 2, -365 / (2 * np.pi) * bs / R ** 2])
    g_R = np.array([bs / R, bc / R])
    c = cov[2:, 2:]
    return np.sqrt(g_phi @ c @ g_phi), np.sqrt(g_R @ c @ g_R)

# part a: 120 people per survey, or 2,880 people in total
def part_a():
    rows = []
    for budget in ("fixed_N_per_round", "fixed_total"):
        for iv in INTERVALS:
            t = survey_times(iv)
            npr = 120.0 if budget == "fixed_N_per_round" else 2880.0 / len(t)
            for th in HALF_LIVES:
                I = expected_fim(t, np.full(len(t), npr), np.log(2) / th)
                cov = np.linalg.inv(I)
                sp, sR = se_phi_R(cov)
                rows.append(dict(budget=budget, interval_days=iv, n_surveys=len(t), N_per_survey=npr,
                                 total_N=npr * len(t), t_half=th, SE_phi_days=sp, SE_R=sR,
                                 SE_R_rel_pct=sR / np.hypot(*BETAS) * 100,
                                 min_eig_scaled=np.linalg.eigvalsh(I / np.max(np.abs(I))).min()))
    return pd.DataFrame(rows)

# part b: (interval, half-life) settings checked by simulation
CELLS_B = [(30, 21), (30, 120), (120, 21), (120, 120)]

# simulate and fit one replicate
def one(job):
    ci, rep = job
    iv, th = CELLS_B[ci]
    rng = np.random.default_rng([20261013, ci, rep])
    tj = survey_times(iv)
    t = np.concatenate([x + rng.uniform(-2, 2, 120) for x in tj])
    p = dict(TRUE)
    kern = {"type": "exponential", "t_half": float(th)}
    y = simulate_obs(rng, t, np.arange(t.size), p, list(BETAS), kern, boost="per_bite")
    F = GammaFitGen(t, y, B0, S2, kern, K=1)
    r = F.fit(PHI_TRUE * 1.05)
    phi = (365 / (2 * np.pi) * np.arctan2(r.x[2], r.x[3])) % 365
    return dict(interval_days=iv, t_half=th, rep=rep, converged=bool(r.success), phi=phi,
                R=np.hypot(r.x[2], r.x[3]))

# expected se of the peak day against survey interval
def figure(A):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    titles = {"fixed_N_per_round": "(a) 120 people per survey", "fixed_total": "(b) 2,880 people in total"}
    for ax, (b, g) in zip(axes, A.groupby("budget", sort=False)):
        for th, gg in g.groupby("t_half"):
            ax.plot(gg.interval_days, gg.SE_phi_days, marker="o", label=f"$t_{{1/2}}$ = {th} d")
        ax.set_xticks(INTERVALS); ax.set_xlabel("Days between surveys"); ax.set_title(titles[b], fontsize=10)
        ax.set_yscale("log"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Expected SE of peak day (days)")
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"figS-A_survey_design.{ext}"), dpi=200)

# usage: python run_S5_2_design.py [S]   (S replicates for part b, default 200)
if __name__ == "__main__":
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    A = part_a()
    A.to_csv(os.path.join(OUT, "design_SE_grid.csv"), index=False)
    figure(A)
    print(A.round(3).to_string())
    t0 = time.time()
    with Pool(2) as P:
        raw = pd.DataFrame(P.map(one, [(c, r) for c in range(len(CELLS_B)) for r in range(S)], chunksize=4))
    raw.to_csv(os.path.join(OUT, "validation_raw.csv"), index=False)
    rows = []
    for (iv, th), g in raw.groupby(["interval_days", "t_half"]):
        nc = int((~g.converged).sum()); g = g[g.converged]
        a = A[(A.budget == "fixed_N_per_round") & (A.interval_days == iv) & (A.t_half == th)].iloc[0]
        e = circ_diff(g.phi.values, (365 / (2 * np.pi) * np.arctan2(*BETAS)) % 365)
        rows.append(dict(interval_days=iv, t_half=th, S=len(g), non_converged=nc,
                         SE_phi_expected_FIM=a.SE_phi_days, SD_phi_empirical=e.std(ddof=1),
                         phi_bias_days=e.mean(),
                         ratio_SD_over_expectedSE_phi=e.std(ddof=1) / a.SE_phi_days,
                         SE_R_expected_FIM=a.SE_R, SD_R_empirical=g.R.std(ddof=1),
                         ratio_SD_over_expectedSE_R=g.R.std(ddof=1) / a.SE_R))
    B = pd.DataFrame(rows)
    B.to_csv(os.path.join(OUT, "validation_partB.csv"), index=False)
    print(B.round(3).to_string())
    print(f"part B {(time.time() - t0) / 60:.1f} min")
