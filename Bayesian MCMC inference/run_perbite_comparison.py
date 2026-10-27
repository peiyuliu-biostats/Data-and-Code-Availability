# simulation: boost size drawn per bite vs fixed per person, with the designs of table 1

import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "engine"))
from anpp_core import (TRUE, true_phi, R_phi, circ_diff, lognormal_params,
                       simulate_round, GammaFit, OMEGA)

OUT = os.path.join(HERE, "outputs"); os.makedirs(OUT, exist_ok=True)
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
# true (beta_sin, beta_cos) and (rounds M, people per round N) for each scenario
SCEN = {"Main": (0.5, 0.2), "ScenarioI": (0.2, 0.5), "ScenarioII": (0.5, 0.5)}
CELLS = {"Main": [(24, 300), (36, 540)], "ScenarioI": [(36, 540)], "ScenarioII": [(36, 540)]}
MODES = ["person", "bite"]
GAM = np.log(2) / TRUE["t_half"]; B0 = TRUE["B0"]; S2 = TRUE["sigma_eps"] ** 2

# simulate the largest design once; smaller designs are subsets of it
def master(rng, p, mode, M=36, N=540):
    mu_c, s_c = lognormal_params(p["mean_c"], p["var_c"])
    Cp = rng.lognormal(mu_c, s_c, N)
    T, Y, R, P = [], [], [], []
    for j in range(M):
        t = 30.0 * (j + 1) + rng.uniform(-2, 2, N)
        y = simulate_round(rng, t, p, c_mode=mode, C_person=Cp)
        T.append(t); Y.append(y); R.append(np.full(N, j + 1)); P.append(np.arange(N))
    return np.concatenate(T), np.concatenate(Y), np.concatenate(R), np.concatenate(P)

# gamma surrogate mle started at 1.05 x truth
def fit(t, y, x_true):
    F = GammaFit(t, y, GAM, B0, S2)
    r = F.fit(x_true * 1.05)
    bs, bc = r.x[2], r.x[3]
    return dict(converged=bool(r.success), nit=r.nit, log_a1=r.x[0], log_a2=r.x[1],
                beta_sin=bs, beta_cos=bc)

# one replicate of one scenario and boost mode
def one(job):
    scen, mode, rep = job
    bs, bc = SCEN[scen]
    p = dict(TRUE, beta_sin=bs, beta_cos=bc)
    rng = np.random.default_rng([20261010, list(SCEN).index(scen), MODES.index(mode), rep])
    t, y, rnd, per = master(rng, p, mode)
    x_true = true_phi(p); out = []
    for (M, N) in CELLS[scen]:
        k = (rnd <= M) & (per < N)
        d = fit(t[k], y[k], x_true)
        d.update(scenario=scen, mode=mode, rep=rep, M=M, N=N, true_beta_sin=bs, true_beta_cos=bc)
        out.append(d)
    return out

# relative bias (%) with mc se, mse, and peak-day bias and rmse (days)
def summarise(raw):
    rows = []
    for (scen, mode, M, N), g in raw.groupby(["scenario", "mode", "M", "N"], sort=False):
        g = g[g.converged]
        bs0, bc0 = g.true_beta_sin.iloc[0], g.true_beta_cos.iloc[0]
        x0 = true_phi(dict(TRUE, beta_sin=bs0, beta_cos=bc0))
        R0, ph0 = R_phi(bs0, bc0)
        R, ph = R_phi(g.beta_sin.values, g.beta_cos.values)
        e = dict(log_a1=g.log_a1 - x0[0], log_a2=g.log_a2 - x0[1],
                 beta_sin=g.beta_sin - bs0, beta_cos=g.beta_cos - bc0, R=pd.Series(R - R0))
        den = dict(log_a1=abs(x0[0]), log_a2=abs(x0[1]), beta_sin=abs(bs0), beta_cos=abs(bc0), R=R0)
        S = len(g); row = dict(scenario=scen, mode=mode, M=M, N=N, S=S)
        for k in e:
            v = np.asarray(e[k], float)
            row[f"are_{k}"] = v.mean() / den[k] * 100
            row[f"se_{k}"] = v.std(ddof=0) / np.sqrt(S) / den[k] * 100
            row[f"mse_{k}"] = (v ** 2).mean()
        ep = circ_diff(ph, ph0)
        row.update(bias_phi_d=ep.mean(), se_phi_d=ep.std(ddof=0) / np.sqrt(S),
                   rmse_phi_d=np.sqrt((ep ** 2).mean()))
        rows.append(row)
    return pd.DataFrame(rows)

# original table 1 results, for comparison
def original_table1():
    f = os.path.join(ROOT, "4-26-sim-rep-infer", "infer-results-sim-repcs", "table_are_combined_with_se.csv")
    if not os.path.isfile(f):
        return None
    o = pd.read_csv(f)
    o["scenario"] = np.where(o.Scenario.str.startswith("Main"), "Main",
                     np.where(o.Scenario.str.contains("Scenario I ") | o.Scenario.str.contains(r"Scenario I \("), "ScenarioI", "ScenarioII"))
    o.loc[o.Scenario.str.contains("Scenario II"), "scenario"] = "ScenarioII"
    o.loc[o.Scenario.str.contains("Scenario I ") & ~o.Scenario.str.contains("Scenario II"), "scenario"] = "ScenarioI"
    keep = [(s, m, n) for s, cells in CELLS.items() for (m, n) in cells]
    o = o[[(s, m, n) in keep for s, m, n in zip(o.scenario, o.m, o.n)]].copy()
    o = o.rename(columns={"m": "M", "n": "N", "n_sims": "S"}); o["mode"] = "Table1_original(person)"
    return o[["scenario", "mode", "M", "N", "S", "are_log_a1", "se_log_a1", "are_log_a2", "se_log_a2",
              "are_beta_sin", "se_beta_sin", "are_beta_cos", "se_beta_cos", "are_r", "se_r",
              "bias_phi_d", "se_phi_d", "rmse_phi_d"]].rename(columns={"are_r": "are_R", "se_r": "se_R"})

# usage: python run_perbite_comparison.py [S]   (S replicates, default 200)
if __name__ == "__main__":
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    jobs = [(s, m, r) for s in SCEN for m in MODES for r in range(S)]
    t0 = time.time()
    with Pool(2) as P:
        res = [d for out in P.imap_unordered(one, jobs, chunksize=2) for d in out]
    raw = pd.DataFrame(res).sort_values(["scenario", "mode", "M", "N", "rep"])
    raw.to_csv(os.path.join(OUT, "perbite_raw.csv"), index=False)
    summ = summarise(raw)
    orig = original_table1()
    full = pd.concat([summ, orig], ignore_index=True) if orig is not None else summ
    order = {"Main": 0, "ScenarioI": 1, "ScenarioII": 2}; morder = {"Table1_original(person)": 0, "person": 1, "bite": 2}
    full = full.sort_values(by=["scenario", "M", "mode"], key=lambda c: c.map(order) if c.name == "scenario" else (c.map(morder) if c.name == "mode" else c))
    full.to_csv(os.path.join(OUT, "perbite_summary.csv"), index=False)
    f = lambda a, b: "" if pd.isna(a) else f"{a:.1f}({b:.1f})"
    lines = ["| scenario | M | N | mode | S | logA1 % | logA2 % | beta_sin % | beta_cos % | R % | phi bias d | phi RMSE d |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in full.iterrows():
        lines.append(f"| {r.scenario} | {r.M} | {r.N} | {r['mode']} | {int(r.S)} | {f(r.are_log_a1, r.se_log_a1)} | {f(r.are_log_a2, r.se_log_a2)} | "
                     f"{f(r.are_beta_sin, r.se_beta_sin)} | {f(r.are_beta_cos, r.se_beta_cos)} | {f(r.are_R, r.se_R)} | "
                     f"{f(r.bias_phi_d, r.se_phi_d)} | {r.rmse_phi_d:.1f} |")
    open(os.path.join(OUT, "perbite_summary.md"), "w").write(
        "Relative bias (MC SE) in %, as in Table 1; phi bias/RMSE in days.\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"\nelapsed {(time.time() - t0) / 60:.1f} min, fits={len(raw)}, non-converged={int((~raw.converged).sum())}")
