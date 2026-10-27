# summaries of the sensitivity runs: re-run check, mean-only fit, loo, and tables of peak day and amplitude

import os, sys
import numpy as np, pandas as pd
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import repcs_inference_v4_rev as rev

OUT = os.path.join(HERE, "outputs"); os.makedirs(OUT, exist_ok=True)
ORIG = os.path.join(ROOT, "4-30-real-repcs-infer", "results")
DS = list(rev.REPCS_FILES)
LABEL = {"ElangaNdille_2012": "Elanga-Ndille 2012 (Benin)", "Fustec_KhonKaen": "Fustec Khon Kaen",
         "Fustec_RoiEt": "Fustec Roi Et"}

# file tag of a chain
def tag(ds, t12, b0):
    return f"{ds}__t12_{t12}__b0_{str(b0).replace('.', 'p')}"

# chain from the original analysis
def orig_chain(ds, t12, b0):
    return pd.read_csv(os.path.join(ORIG, ds, f"chain_{tag(ds, t12, b0)}.csv")).values

# chain from a sensitivity run (None if not run)
def run_chain(run, ds):
    t12, b0 = rev.BASELINE[ds]
    f = os.path.join(OUT, run, ds, f"chain_{tag(ds, t12, b0)}__{run}.csv")
    return pd.read_csv(f).values if os.path.isfile(f) else None

# mean, 95% percentile interval, median and sd of peak day and R
def summ(chain):
    phi, r = rev.peak_and_amplitude(chain)
    q = lambda v: (v.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5), np.median(v), v.std())
    return dict(zip(["phi_mean", "phi_lo", "phi_hi", "phi_med", "phi_sd"], q(phi)),
                **dict(zip(["R_mean", "R_lo", "R_hi", "R_med", "R_sd"], q(r))))

def load_df(ds):
    df = pd.read_csv(os.path.join(rev.DATA_DIR, rev.REPCS_FILES[ds])).dropna(subset=["antibody"])
    return df[df["antibody"] > 0].copy()

# run E0: re-run vs original chains
def e0_check():
    rows = []
    for ds in DS:
        new = run_chain("E0", ds)
        if new is None:
            continue
        old = orig_chain(ds, *rev.BASELINE[ds])
        for name, f in [("log_a1", lambda c: c[:, 0]), ("log_a2", lambda c: c[:, 1]),
                        ("R", lambda c: rev.peak_and_amplitude(c)[1]),
                        ("peak_day", lambda c: rev.peak_and_amplitude(c)[0])]:
            a, b = f(old), f(new)
            rows.append(dict(dataset=ds, quantity=name, original_mean=a.mean(), E0_mean=b.mean(),
                             original_q025=np.percentile(a, 2.5), E0_q025=np.percentile(b, 2.5),
                             original_q975=np.percentile(a, 97.5), E0_q975=np.percentile(b, 97.5),
                             diff_mean_in_posterior_sd=(b.mean() - a.mean()) / a.std()))
    return pd.DataFrame(rows)

# run E7: weighted least squares on the survey means (no gamma likelihood), delta-method intervals
def e7_wls(ds, t12, b0):
    df = load_df(ds)
    g = df.groupby("time")["antibody"].agg(["mean", "var", "count"]).reset_index()
    t, ybar, w = g["time"].values.astype(float), g["mean"].values, g["count"].values / g["var"].values
    gam = np.log(2) / t12

    def mu(th):
        m, _ = rev.compute_moments(t, np.r_[th[0], 0.0, th[1], th[2]], gam, b0)
        return m

    def res(th):
        return np.sqrt(w) * (ybar - mu(th))

    x0 = rev.initialise_theta(g.rename(columns={"mean": "mean_y", "var": "var_y"}), gam, b0)
    r = least_squares(res, np.r_[x0[0], 0.05, 0.05], method="lm", x_scale="jac")
    J = r.jac
    chi2 = np.sum(r.fun ** 2); dof = len(t) - 3
    cov = np.linalg.inv(J.T @ J) * max(1.0, chi2 / dof)
    bs, bc = r.x[1], r.x[2]; R = np.hypot(bs, bc)
    phi = (365 / (2 * np.pi) * np.arctan2(bs, bc)) % 365
    g_phi = np.array([0, 365 / (2 * np.pi) * bc / R ** 2, -365 / (2 * np.pi) * bs / R ** 2])
    g_R = np.array([0, bs / R, bc / R])
    sp, sR = np.sqrt(g_phi @ cov @ g_phi), np.sqrt(g_R @ cov @ g_R)
    return dict(dataset=ds, half_life_days=t12, b0=b0, phi=phi, phi_lo=phi - 1.96 * sp, phi_hi=phi + 1.96 * sp,
                R=R, R_lo=R - 1.96 * sR, R_hi=R + 1.96 * sR, chi2=chi2, dof=dof, n_rounds=len(t))

# run E1: loo difference, two harmonics vs one
def e1_loo():
    rows = []
    for ds in DS:
        t12, b0 = rev.BASELINE[ds]
        f0 = os.path.join(OUT, "E0", ds, f"loo_pointwise_{tag(ds, t12, b0)}__E0.csv")
        f1 = os.path.join(OUT, "E1", ds, f"loo_pointwise_{tag(ds, t12, b0)}__E1.csv")
        if not (os.path.isfile(f0) and os.path.isfile(f1)):
            continue
        a, b = pd.read_csv(f0), pd.read_csv(f1)
        d = b.elpd_i.values - a.elpd_i.values
        rows.append(dict(dataset=ds, elpd_K1=a.elpd_i.sum(), elpd_K2=b.elpd_i.sum(), elpd_diff_K2_minus_K1=d.sum(),
                         se_diff=np.sqrt(len(d) * d.var()), n_obs=len(d),
                         n_pareto_k_gt_07_K1=int((a.pareto_k > 0.7).sum()),
                         n_pareto_k_gt_07_K2=int((b.pareto_k > 0.7).sum())))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    e0 = e0_check(); e0.to_csv(os.path.join(OUT, "E0_validation.csv"), index=False)
    print("E0 validation\n", e0.round(3).to_string())

    e7 = pd.DataFrame([e7_wls(ds, t, b) for ds in DS for t in rev.HALF_LIVES for b in rev.B0_VALUES])
    # posterior summaries of the original 24 fits
    pct = pd.read_csv(os.path.join(HERE, "..", "02_fix_Fig6_7_8_S11_calendar_labels", "outputs",
                                   "posterior_summary_percentile_24configs.csv"))
    e7 = e7.merge(pct[["dataset", "half_life_days", "b0", "peak_day_phi_mean", "seasonal_amplitude_r_mean"]],
                  on=["dataset", "half_life_days", "b0"])
    e7["phi_WLS_minus_Gamma_d"] = e7.phi - e7.peak_day_phi_mean
    e7.to_csv(os.path.join(OUT, "E7_mean_only_WLS_24settings.csv"), index=False)
    print("E7\n", e7.round(2).to_string())

    loo = e1_loo(); loo.to_csv(os.path.join(OUT, "E1_loo_K2_vs_K1.csv"), index=False)
    print("E1 LOO\n", loo.round(2).to_string())

    # peak day and R for each run at the main-text setting
    rows = []
    for ds in DS:
        t12, b0 = rev.BASELINE[ds]
        base = summ(orig_chain(ds, t12, b0))
        rows.append(dict(dataset=ds, run="Original (Fig 5/6 setting)", **base, shift_phi_d=0.0))
        for run, lab in [("E0", "E0 re-run (check)"), ("E2a", "E2a sigma_eps 0.05"), ("E2b", "E2b sigma_eps 0.10"),
                         ("E3a", "E3a drop floored value"), ("E3b", "E3b left-censor <= 0.01"),
                         ("E1", "E1 two harmonics (K = 2)")]:
            c = run_chain(run, ds)
            if c is None:
                continue
            s = summ(c)
            rows.append(dict(dataset=ds, run=lab, **s, shift_phi_d=s["phi_mean"] - base["phi_mean"]))
        w = e7[(e7.dataset == ds) & (e7.half_life_days == t12) & (e7.b0 == b0)].iloc[0]
        rows.append(dict(dataset=ds, run="E7 mean-only WLS (no Gamma)", phi_mean=w.phi, phi_lo=w.phi_lo,
                         phi_hi=w.phi_hi, R_mean=w.R, R_lo=w.R_lo, R_hi=w.R_hi,
                         shift_phi_d=w.phi - base["phi_mean"]))
    SB = pd.DataFrame(rows); SB.to_csv(os.path.join(OUT, "table_S-B_empirical_sensitivity.csv"), index=False)

    # kinetics-averaged intervals and the roi et - khon kaen peak difference
    rows = []
    for ds in DS:
        mix = np.vstack([orig_chain(ds, t, b) for t in rev.HALF_LIVES for b in rev.B0_VALUES])
        s = summ(mix)
        rows.append(dict(dataset=ds, item="P1 kinetics-averaged (8 settings, equal weight)",
                         phi=s["phi_mean"], phi_lo=s["phi_lo"], phi_hi=s["phi_hi"], R=s["R_mean"], R_lo=s["R_lo"], R_hi=s["R_hi"]))
    rng = np.random.default_rng(11)
    pairs = [(t, b, t, b) for t in rev.HALF_LIVES for b in rev.B0_VALUES] + [(82, 0.012, 82, 0.044)]
    for tk, bk, tr, br in pairs:
        pk = rev.peak_and_amplitude(orig_chain("Fustec_KhonKaen", tk, bk))[0]
        pr = rev.peak_and_amplitude(orig_chain("Fustec_RoiEt", tr, br))[0]
        d = rng.permutation(pr) - rng.permutation(pk)
        lab = (f"P2 Roi Et - Khon Kaen, t1/2 {tk}, B0 {bk}" if bk == br else
               f"P2 Roi Et - Khon Kaen, Fig 5/6 settings (KK B0 {bk}, RE B0 {br}, t1/2 {tk})")
        rows.append(dict(dataset="RE - KK", item=lab, phi=d.mean(), phi_lo=np.percentile(d, 2.5),
                         phi_hi=np.percentile(d, 97.5), P_RE_later=np.mean(d > 0)))
    ck, cr = run_chain("E1", "Fustec_KhonKaen"), run_chain("E1", "Fustec_RoiEt")
    if ck is not None and cr is not None:
        pk, pr = rev.peak_and_amplitude(ck)[0], rev.peak_and_amplitude(cr)[0]
        d = ((rng.permutation(pr) - rng.permutation(pk) + 182.5) % 365) - 182.5
        rows.append(dict(dataset="RE - KK", item="P2 Roi Et - Khon Kaen, Fig 5/6 settings, K = 2 (E1 chains)",
                         phi=d.mean(), phi_lo=np.percentile(d, 2.5), phi_hi=np.percentile(d, 97.5),
                         P_RE_later=np.mean(d > 0)))
    SC = pd.DataFrame(rows); SC.to_csv(os.path.join(OUT, "table_S-C_kinetics_timing.csv"), index=False)

    # seasonal shape with one vs two harmonics
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    g = np.arange(0, 365, 1.0); rs = np.random.default_rng(3)
    for ax, ds in zip(axes, DS):
        for run, col, lab in [("E0", "#2980b9", "K = 1"), ("E1", "#c0392b", "K = 2")]:
            c = run_chain(run, ds)
            if c is None:
                continue
            c = c[rs.choice(len(c), 2000, replace=False)]
            L = np.exp(np.array([rev.log_u(g, th[2:]) for th in c])); L = L / L.mean(axis=1, keepdims=True)
            lo, md_, hi = np.percentile(L, [2.5, 50, 97.5], axis=0)
            ax.fill_between(g, lo, hi, color=col, alpha=0.2); ax.plot(g, md_, color=col, label=lab)
        ax.set_title(LABEL[ds], fontsize=9); ax.set_xlabel("Day of year"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Relative biting rate"); axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"figE1_season_shape_K1_vs_K2.{ext}"), dpi=200)

    # markdown tables
    f = lambda v: "" if pd.isna(v) else f"{v:.1f}"
    md = ["## Table S-B  Empirical sensitivity (Fig 5/6 setting per cohort; mean [95% percentile interval])", "",
          "| Cohort | Run | Peak day | Shift vs original (d) | R |", "|---|---|---|---|---|"]
    for _, r in SB.iterrows():
        md.append(f"| {LABEL[r.dataset]} | {r.run} | {f(r.phi_mean)} [{f(r.phi_lo)}, {f(r.phi_hi)}] | {r.shift_phi_d:+.1f} | "
                  f"{r.R_mean:.3f} [{r.R_lo:.3f}, {r.R_hi:.3f}] |")
    md += ["", "## Table S-C  Kinetics-averaged intervals; Roi Et - Khon Kaen timing", "",
           "| Cohort | Item | Peak day / difference (d) | R | P(RE later) |", "|---|---|---|---|---|"]
    for _, r in SC.iterrows():
        Rtxt = "" if pd.isna(r.get("R", np.nan)) else f"{r.R:.3f} [{r.R_lo:.3f}, {r.R_hi:.3f}]"
        P = "" if pd.isna(r.get("P_RE_later", np.nan)) else f"{r.P_RE_later:.3f}"
        md.append(f"| {LABEL.get(r.dataset, r.dataset)} | {r['item']} | {f(r.phi)} [{f(r.phi_lo)}, {f(r.phi_hi)}] | {Rtxt} | {P} |")
    open(os.path.join(OUT, "tables_S-B_S-C.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
