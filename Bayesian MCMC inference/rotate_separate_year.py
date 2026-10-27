# peak day of the separate-year fits converted from window time to calendar day of year

import os, glob, re
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
SEP  = os.path.join(ROOT, "4-30-real-repcs-infer", "results_separate_yr")
OUT  = os.path.join(HERE, "outputs"); os.makedirs(OUT, exist_ok=True)
W = 2 * np.pi / 365.0
T0 = pd.Timestamp("2017-01-01")

# circular mean and 95% interval of peak days
def circ_summary(phi):
    a = 2 * np.pi * phi / 365
    m = (np.arctan2(np.sin(a).mean(), np.cos(a).mean()) * 365 / (2 * np.pi)) % 365
    d = (phi - m + 182.5) % 365 - 182.5
    return m, (m + np.quantile(d, .025)) % 365, (m + np.quantile(d, .975)) % 365

# one row per window, half-life and b0
rows = []
for win in sorted(glob.glob(os.path.join(SEP, "Fustec_*_year*"))):
    name = os.path.basename(win)
    tm = pd.read_csv(os.path.join(SEP, f"timemap_{name}.csv"))
    offset = tm.time_abs.min() - tm.time.min()
    summ = pd.read_csv(os.path.join(SEP, f"combined_summary_{name}.csv"))
    for f in sorted(glob.glob(os.path.join(win, f"chain_{name}__t12_*__b0_*.csv"))):
        t12, b0 = re.search(r"__t12_(\d+)__b0_(0p\d+)", f).groups()
        t12, b0 = int(t12), float(b0.replace("p", "."))
        ch = pd.read_csv(f)
        bs, bc = ch.beta_sin.values, ch.beta_cos.values
        # peak day on window time, then shifted by the window start
        phi_rel = (np.arctan2(bs, bc) / W) % 365
        phi_cal = (phi_rel + offset) % 365
        m, lo, hi = circ_summary(phi_cal)
        mr, _, _ = circ_summary(phi_rel)
        orig = summ[(summ.half_life_days == t12) & (summ.b0 == b0)].peak_day_phi_mean
        rows.append(dict(window=name, t_half=t12, B0=b0, window_offset_days=offset,
                         window_start=(T0 + pd.Timedelta(days=int(offset))).date(),
                         phi_reported_original=float(orig.iloc[0]) if len(orig) else np.nan,
                         phi_rel_circmean=mr,
                         phi_cal_circmean=m, phi_cal_q025=lo, phi_cal_q975=hi,
                         peak_date_in_window=(T0 + pd.Timedelta(days=float(offset + (m - offset) % 365))).date(),
                         R_mean=np.hypot(bs, bc).mean()))
df = pd.DataFrame(rows).sort_values(["window", "B0", "t_half"])
df.to_csv(os.path.join(OUT, "separate_year_phi_calendar.csv"), index=False)
pd.set_option("display.width", 220)
print(df.round(1).to_string(index=False))
