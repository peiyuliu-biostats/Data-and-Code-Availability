# simulation robustness scenarios; each changes one assumption relative to the baseline S0

import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "engine"))
from anpp_core import TRUE, circ_diff
from anpp_ext import simulate_obs, GammaFitGen, season_peaks, transfer_lag_days

OUT = os.path.join(HERE, "outputs"); os.makedirs(OUT, exist_ok=True)
# baseline seasonal coefficients and antibody kernels
BASE_B = [0.5, 0.2]
R_MAIN = float(np.hypot(*BASE_B))
EXP21 = {"type": "exponential", "t_half": 21.0}
EXP82 = {"type": "exponential", "t_half": 82.0}
BIPH = {"type": "biphasic", "w_fast": 0.7, "t_half_fast": 14.0, "t_half_slow": 180.0}
DELAY = {"type": "delayed_rise", "tau_rise": 10.0, "t_half": 21.0}

# one scenario: data-generating settings and the model(s) fitted
def cell(id, label, comments, design="monthly_indep", betas=BASE_B, kern=EXP21, beta0=np.log(1 / 15),
         exposure=None, fits=None):
    return dict(id=id, label=label, comments=comments, design=design, betas=list(betas), kern=kern,
                beta0=beta0, exposure=exposure,
                fits=fits or [dict(name="K1_" + ("exp%d" % kern["t_half"] if kern["type"] == "exponential" else kern["type"]),
                                   kern=kern, K=1)])

def scaled(R):
    return [b * R / R_MAIN for b in BASE_B]

# scenarios
CELLS = [
    cell("S0", "Baseline (C per bite; moment equations correct)", "R2-11, R1-3"),
    cell("S2a", "Same 300 people in all 24 rounds (cohort), IDs ignored", "R1-1", design="monthly_cohort"),
    cell("S2b", "Fustec-like cohort (70 monthly + 130 of 260 every 4 months, 19 months)", "R1-1", design="fustec_cohort"),
    cell("S2b_indep", "As S2b, new people at every visit", "R1-1", design="fustec_indep"),
    cell("S3a", "Low biting intensity (e^beta0 = 1/60)", "R1-3", beta0=np.log(1 / 60)),
    cell("S3b", "Two exposure groups (30% x2.5, 70% x0.357)", "R1-3, R1-7",
         exposure={"type": "two_point", "p_high": 0.3, "z_high": 2.5, "z_low": 0.357}),
    cell("S7", "Exposure heterogeneity Z ~ Gamma(mean 1, var 15), matched to S1", "R1-7",
         exposure={"type": "gamma", "mean": 1.0, "var": 15.0}),
    cell("S4a", "Biphasic decay (0.7 x t1/2 14 d + 0.3 x t1/2 180 d)", "R1-5", kern=BIPH,
         fits=[dict(name="K1_exp21 (assumed)", kern=EXP21, K=1), dict(name="K1_biphasic (true kernel)", kern=BIPH, K=1)]),
    cell("S4b", "Delayed rise (1 - e^{-s/10}) e^{-gamma s}, t1/2 21 d", "R1-5, R3-11", kern=DELAY,
         fits=[dict(name="K1_exp21 (assumed)", kern=EXP21, K=1), dict(name="K1_delayed (true kernel)", kern=DELAY, K=1)]),
    cell("S5a", "Skewed single-peak season (2nd harmonic bs2 = 0.2)", "R1-6", betas=[0.5, 0.2, 0.2, 0.0],
         fits=[dict(name="K1_exp21", kern=EXP21, K=1), dict(name="K2_exp21", kern=EXP21, K=2)]),
    cell("S5b", "Bimodal season (bs 0.2, bc 0.1, bc2 0.45)", "R1-6", betas=[0.2, 0.1, 0.0, 0.45],
         fits=[dict(name="K1_exp21", kern=EXP21, K=1), dict(name="K2_exp21", kern=EXP21, K=2)]),
    cell("S6_R0.2_t21", "Amplitude R 0.2, t1/2 21", "R3-7", betas=scaled(0.2)),
    cell("S6_R0.8_t21", "Amplitude R 0.8, t1/2 21", "R3-7", betas=scaled(0.8)),
    cell("S6_R0.2_t82", "Amplitude R 0.2, t1/2 82", "R3-7", betas=scaled(0.2), kern=EXP82),
    cell("S6_R0.54_t82", "Amplitude R 0.54, t1/2 82", "R3-7", betas=scaled(R_MAIN), kern=EXP82),
    cell("S6_R0.8_t82", "Amplitude R 0.8, t1/2 82", "R3-7", betas=scaled(0.8), kern=EXP82),
]
CID = {c["id"]: i for i, c in enumerate(CELLS)}
# fixed seed index for each scenario
SEED_ID = {"S0": 0, "S2a": 1, "S2b": 2, "S2b_indep": 3, "S3a": 8, "S3b": 9, "S7": 10, "S4a": 11,
           "S4b": 12, "S5a": 13, "S5b": 14, "S6_R0.2_t21": 15, "S6_R0.8_t21": 16,
           "S6_R0.2_t82": 17, "S6_R0.54_t82": 18, "S6_R0.8_t82": 19}

# survey designs: 24 monthly rounds of 300 (new or same people) or a fustec-like cohort
def design(rng, typ):
    if typ.startswith("monthly"):
        M, N = 24, 300
        t = np.concatenate([30.0 * (j + 1) + rng.uniform(-2, 2, N) for j in range(M)])
        pid = np.tile(np.arange(N), M) if typ == "monthly_cohort" else np.arange(M * N)
        return t, pid
    if typ.startswith("fustec"):
        T, P = [], []
        main_ids = 70 + np.arange(260)
        for j in range(1, 20):
            ids = np.arange(70)
            if j in (1, 5, 9, 13, 17):
                ids = np.r_[ids, rng.choice(main_ids, 130, replace=False)]
            T.append(30.0 * j + rng.uniform(-2, 2, ids.size)); P.append(ids)
        t, pid = np.concatenate(T), np.concatenate(P)
        return t, (pid if typ == "fustec_cohort" else np.arange(t.size))
    raise ValueError(typ)

# true (log A1, log A2, betas) for a fit with K harmonics
def truth_phi(c, K):
    cbar, c2 = TRUE["mean_c"], TRUE["var_c"] + TRUE["mean_c"] ** 2
    b = list(c["betas"]) + [0.0] * max(0, 2 * K - len(c["betas"]))
    return np.r_[np.log(cbar) + c["beta0"], np.log(c2) + c["beta0"], b[:2 * K]]

# simulate one replicate and fit each model
def one(job):
    cid, rep = job
    c = CELLS[cid]
    rng = np.random.default_rng([20261012, SEED_ID[c['id']], rep])
    p = dict(TRUE, beta0=c["beta0"])
    t, pid = design(rng, c["design"])
    y = simulate_obs(rng, t, pid, p, c["betas"], c["kern"], boost="per_bite", exposure=c["exposure"])
    out = []
    for f in c["fits"]:
        t0 = time.time()
        F = GammaFitGen(t, y, p["B0"], p["sigma_eps"] ** 2, f["kern"], K=f["K"])
        x0 = truth_phi(c, f["K"]) * 1.05
        r = F.fit(x0)
        ph, loc, R = season_peaks(r.x[2:])
        d = dict(cell=c["id"], fit=f["name"], rep=rep, n_obs=len(y), converged=bool(r.success), nit=r.nit,
                 nll=r.fun, log_a1=r.x[0], log_a2=r.x[1], peak=ph, R_eff=R, n_local_max=len(loc),
                 local_max=";".join(f"{v:.1f}" for v in loc), secs=time.time() - t0,
                 bs1=np.nan, bc1=np.nan, bs2=np.nan, bc2=np.nan)
        for j, b in enumerate(r.x[2:]):
            d[["bs1", "bc1", "bs2", "bc2"][j]] = b
        out.append(d)
    return out

# true peak day and amplitude, and the peak shift predicted from the kernel lag
def truth_table():
    rows = []
    for c in CELLS:
        ph, loc, R = season_peaks(np.array(c["betas"]))
        fund = season_peaks(np.array(c["betas"][:2]))[0]
        for f in c["fits"]:
            pred = transfer_lag_days(c["kern"]) - transfer_lag_days(f["kern"])
            rows.append(dict(cell=c["id"], fit=f["name"], true_peak=ph, true_local_max=";".join(f"{v:.1f}" for v in loc),
                             true_R_eff=R, true_peak_fundamental=fund, true_log_a1=truth_phi(c, 1)[0],
                             true_log_a2=truth_phi(c, 1)[1], tf_pred_peak_shift_d=pred))
    return pd.DataFrame(rows)

# peak-day bias, rmse and sd (days) and relative bias of R, with mc se
def summarise(raw):
    tt = truth_table().set_index(["cell", "fit"])
    rows = []
    for (cid, fit), g in raw.groupby(["cell", "fit"], sort=False):
        tr = tt.loc[(cid, fit)]
        n_all = len(g); nconv = int(g.converged.sum()); g = g[g.converged]; S = len(g)
        e = circ_diff(g.peak.values, tr.true_peak)
        eR = g.R_eff.values - tr.true_R_eff
        row = dict(cell=cid, fit=fit, label=CELLS[CID[cid]]["label"], comments=CELLS[CID[cid]]["comments"],
                   S=S, non_converged=n_all - nconv,
                   true_peak=tr.true_peak, peak_bias_d=e.mean(), peak_bias_mcse=e.std(ddof=1) / np.sqrt(S),
                   peak_rmse_d=np.sqrt((e ** 2).mean()), peak_sd_d=e.std(ddof=1),
                   true_R=tr.true_R_eff, R_relbias_pct=eR.mean() / tr.true_R_eff * 100,
                   R_relbias_mcse=eR.std(ddof=1) / np.sqrt(S) / tr.true_R_eff * 100,
                   log_a1_bias=(g.log_a1 - tr.true_log_a1).mean(), log_a2_bias=(g.log_a2 - tr.true_log_a2).mean(),
                   tf_pred_peak_shift_d=tr.tf_pred_peak_shift_d if cid in ("S4a", "S4b") else np.nan)
        if cid == "S5a" and fit.startswith("K1"):
            row["peak_bias_vs_fundamental_d"] = circ_diff(g.peak.values, tr.true_peak_fundamental).mean()
        if cid == "S5b":
            tl = [float(v) for v in tr.true_local_max.split(";")]
            def both(s):
                v = [float(x) for x in str(s).split(";") if x not in ("", "nan")]
                return len(v) == 2 and all(min(abs(circ_diff(np.array(v), m))) <= 30 for m in tl)
            row["frac_both_peaks_within_30d"] = np.mean([both(s) for s in g.local_max])
        rows.append(row)
    out = pd.DataFrame(rows)
    out["_o"] = out.cell.map(CID)
    return out.sort_values(["_o"], kind="stable").drop(columns="_o").reset_index(drop=True)

# markdown summary table
def write_md(s):
    f = lambda v, d=1: "" if pd.isna(v) else f"{v:.{d}f}"
    L = ["| cell | fit | S | true peak | peak bias d (MCSE) | RMSE d | SD d | R bias % (MCSE) | extra |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        ex = []
        if not pd.isna(r.get("tf_pred_peak_shift_d", np.nan)) and "assumed" in r.fit:
            ex.append(f"transfer-fn predicted shift {r.tf_pred_peak_shift_d:+.1f} d")
        if not pd.isna(r.get("peak_bias_vs_fundamental_d", np.nan)):
            ex.append(f"bias vs fundamental peak {r.peak_bias_vs_fundamental_d:+.1f} d")
        if not pd.isna(r.get("frac_both_peaks_within_30d", np.nan)):
            ex.append(f"both peaks found (±30 d) {r.frac_both_peaks_within_30d:.2f}")
        L.append(f"| {r.cell} | {r.fit} | {r.S} | {f(r.true_peak)} | {f(r.peak_bias_d)} ({f(r.peak_bias_mcse)}) | "
                 f"{f(r.peak_rmse_d)} | {f(r.peak_sd_d)} | {f(r.R_relbias_pct)} ({f(r.R_relbias_mcse)}) | "
                 f"{'; '.join(ex)} |")
    open(os.path.join(OUT, "S5_1_summary.md"), "w").write(
        "Peak = day of the global maximum of lambda(t); R = (max - min of log lambda)/2 (= sqrt(bs^2+bc^2) for K = 1).\n"
        "Bias, RMSE and SD over the S replicates (peak day in days, circular).\n\n"
        + "\n".join(L) + "\n")
    return "\n".join(L)

# usage: python run_S5_1_scenarios.py [S] [scenario ids]   (S replicates, default 200; resumes from saved fits)
if __name__ == "__main__":
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    cells = [c for c in CELLS if only is None or c["id"] in only]
    tag = "" if (S == 200 and only is None) else f"_pilot_S{S}"
    rawf = os.path.join(OUT, f"S5_1_raw{tag}.csv")
    done = set()
    if os.path.isfile(rawf):
        prev = pd.read_csv(rawf); done = set(zip(prev.cell, prev.rep))
    jobs = [(CID[c["id"]], r) for r in range(S) for c in cells if (c["id"], r) not in done]
    log = open(os.path.join(OUT, f"S5_1_run{tag}.log"), "a")
    print(f"{len(jobs)} jobs, {len(done)} already done", file=log, flush=True)
    t0 = time.time(); n = 0
    with Pool(2) as P:
        for out in P.imap_unordered(one, jobs, chunksize=1):
            df = pd.DataFrame(out)
            df.to_csv(rawf, mode="a", header=not os.path.isfile(rawf), index=False)
            n += 1
            if n % 50 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} jobs  {(time.time() - t0) / 60:.1f} min", file=log, flush=True)
    raw = pd.read_csv(rawf)
    summ = summarise(raw)
    summ.to_csv(os.path.join(OUT, f"S5_1_summary{tag}.csv"), index=False)
    truth_table().to_csv(os.path.join(OUT, "S5_1_truth.csv"), index=False)
    md = write_md(summ)
    if tag:
        os.replace(os.path.join(OUT, "S5_1_summary.md"), os.path.join(OUT, f"S5_1_summary{tag}.md"))
    print(md)
    print(f"elapsed {(time.time() - t0) / 60:.1f} min; fits {len(raw)}; non-converged {int((~raw.converged).sum())}",
          file=log, flush=True)
