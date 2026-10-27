# extensions: more harmonics, other antibody kernels, exposure heterogeneity and repeated person ids

import numpy as np
from scipy.special import gammaln, digamma
from scipy.optimize import minimize, minimize_scalar
from anpp_core import OMEGA, S_GRID, lognormal_params

LN2 = np.log(2.0)

# log seasonal factor with len(betas) / 2 harmonics
def log_u(t, betas):
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    for k in range(len(betas) // 2):
        out = out + betas[2 * k] * np.sin((k + 1) * OMEGA * t) + betas[2 * k + 1] * np.cos((k + 1) * OMEGA * t)
    return out

# peak day, local maxima and amplitude of the seasonal factor
def season_peaks(betas, step=0.05):
    betas = np.asarray(betas, float)
    if len(betas) == 2:
        R = np.hypot(*betas)
        ph = (365 / (2 * np.pi) * np.arctan2(betas[0], betas[1])) % 365
        return ph, np.array([ph]), R
    g = np.arange(0.0, 365.0, step)
    v = log_u(g, betas)
    i = int(np.argmax(v))
    f = lambda x: -log_u(np.array([x]), betas)[0]
    r = minimize_scalar(f, bounds=(g[i] - 2 * step, g[i] + 2 * step), method="bounded",
                        options=dict(xatol=1e-7))
    ph = r.x % 365
    vl, vr = np.roll(v, 1), np.roll(v, -1)
    loc = g[(v > vl) & (v >= vr)]
    return ph, loc, (v.max() - v.min()) / 2

# antibody kernel: exponential, biphasic or delayed rise
def kernel(s, spec):
    s = np.asarray(s, float)
    typ = spec.get("type", "exponential")
    if typ == "exponential":
        return np.exp(-LN2 / spec["t_half"] * s)
    if typ == "biphasic":
        w = spec["w_fast"]
        return (w * np.exp(-LN2 / spec["t_half_fast"] * s)
                + (1 - w) * np.exp(-LN2 / spec["t_half_slow"] * s))
    if typ == "delayed_rise":
        return (1 - np.exp(-s / spec["tau_rise"])) * np.exp(-LN2 / spec["t_half"] * s)
    raise ValueError(typ)

# phase lag (days) of the annual component caused by the kernel
def transfer_lag_days(spec, grid=S_GRID):
    h = kernel(grid, spec)
    H = np.sum(h * np.exp(-1j * OMEGA * grid))
    return (-np.angle(H)) / OMEGA

# person-level exposure multiplier (none, two-point or gamma)
def draw_Z(rng, n, spec):
    if spec is None:
        return np.ones(n)
    typ = spec["type"]
    if typ == "two_point":
        return np.where(rng.random(n) < spec["p_high"], spec["z_high"], spec["z_low"])
    if typ == "gamma":
        shape = spec["mean"] ** 2 / spec["var"]
        return rng.gamma(shape, spec["var"] / spec["mean"], n)
    raise ValueError(typ)

# antibody levels for observations (t, person id); repeated ids share bites, multiplier and boost size
def simulate_obs(rng, t, pid, p, betas, kern, boost="per_bite", exposure=None,
                 burn=1095.0):
    t = np.asarray(t, float)
    pid = np.asarray(pid)
    upid, inv = np.unique(pid, return_inverse=True)
    n_p = len(upid)
    mu_c, s_c = lognormal_params(p["mean_c"], p["var_c"])
    Z = draw_Z(rng, n_p, exposure)
    Cp = rng.lognormal(mu_c, s_c, n_p) if boost == "per_person" else None
    gg = np.arange(0, 365, 0.25)
    lu_max = log_u(gg, betas).max() + 0.01
    lam_max = Z * np.exp(p["beta0"] + lu_max)
    t_lo = np.full(n_p, np.inf); t_hi = np.full(n_p, -np.inf)
    np.minimum.at(t_lo, inv, t); np.maximum.at(t_hi, inv, t)
    t_lo = t_lo - burn
    L = t_hi - t_lo
    ncand = rng.poisson(lam_max * L)
    b_p = np.repeat(np.arange(n_p), ncand)
    b_t = t_lo[b_p] + L[b_p] * rng.random(b_p.size)
    keep = rng.random(b_p.size) < np.exp(log_u(b_t, betas) - lu_max)
    b_p, b_t = b_p[keep], b_t[keep]
    # boost size: per bite, per person or constant
    if boost == "per_bite":
        b_c = rng.lognormal(mu_c, s_c, b_p.size)
    elif boost == "per_person":
        b_c = Cp[b_p]
    elif boost == "const":
        b_c = np.full(b_p.size, p["mean_c"])
    else:
        raise ValueError(boost)
    cnt = np.bincount(b_p, minlength=n_p)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    n_obs = len(t)
    A = np.full(n_obs, p["B0"], float)
    per_obs = cnt[inv]
    # sum bite contributions in chunks to limit memory
    chunk = 2_000_000
    o_idx = np.arange(n_obs)
    a = 0
    while a < n_obs:
        cs = np.cumsum(per_obs[a:])
        b = a + max(1, int(np.searchsorted(cs, chunk, side="right")))
        b = min(b, n_obs)
        oo = np.repeat(o_idx[a:b], per_obs[a:b])
        if oo.size:
            first = np.repeat(start[inv[a:b]], per_obs[a:b])
            offs = np.arange(oo.size) - np.repeat(np.cumsum(per_obs[a:b]) - per_obs[a:b], per_obs[a:b])
            bi = first + offs
            s = t[oo] - b_t[bi]
            ok = (s >= 0) & (s <= burn)
            contrib = np.where(ok, b_c[bi] * kernel(np.where(ok, s, 0.0), kern), 0.0)
            A += np.bincount(oo, weights=contrib, minlength=n_obs)
        a = b
    y = A + rng.normal(0, p["sigma_eps"], n_obs)
    return np.maximum(0.001, y)

# gamma surrogate likelihood with K harmonics and a general kernel
class GammaFitGen:

    def __init__(self, t, y, B0, s2eps, kern, K=1, tgrid_step=0.25, grid=S_GRID):
        self.t, self.y = np.asarray(t, float), np.maximum(np.asarray(y, float), 1e-8)
        self.ly = np.log(self.y)
        self.B0, self.s2, self.K = B0, s2eps, K
        h = kernel(grid, kern)
        self.w1, self.w2 = h * 1.0, h * h * 1.0
        self.grid = grid
        tun = np.unique(self.t)
        if len(tun) < 400:
            self.tu, self.map, self.interp = tun, np.searchsorted(tun, self.t), False
        else:
            self.tu = np.arange(self.t.min() - tgrid_step, self.t.max() + 2 * tgrid_step, tgrid_step)
            self.interp = True
        arg = OMEGA * (self.tu[:, None] - grid[None, :])
        self.basis = []
        for k in range(K):
            self.basis += [np.sin((k + 1) * arg), np.cos((k + 1) * arg)]

    def _J(self, betas):
        lu = sum(b * f for b, f in zip(betas, self.basis))
        u = np.exp(lu)
        J1, J2 = u @ self.w1, u @ self.w2
        d1 = np.stack([(u * f) @ self.w1 for f in self.basis], 1)
        d2 = np.stack([(u * f) @ self.w2 for f in self.basis], 1)
        if not self.interp:
            m = self.map
            return J1[m], J2[m], d1[m], d2[m]
        f = lambda v: np.interp(self.t, self.tu, v)
        return (f(J1), f(J2), np.stack([f(d1[:, j]) for j in range(d1.shape[1])], 1),
                np.stack([f(d2[:, j]) for j in range(d2.shape[1])], 1))

    # negative log-likelihood and gradient
    def nll(self, phi, grad=True):
        la1, la2 = phi[0], phi[1]
        J1, J2, d1, d2 = self._J(phi[2:])
        A1, A2 = np.exp(la1), np.exp(la2)
        mu = self.B0 + A1 * J1
        V = A2 * J2 + self.s2
        if np.any(mu <= 0) or np.any(V <= 0) or not np.all(np.isfinite(mu * V)):
            return (1e12, np.zeros(len(phi))) if grad else 1e12
        k = mu * mu / V
        th = V / mu
        ll = (k - 1) * self.ly - self.y / th - k * np.log(th) - gammaln(k)
        f = -np.sum(ll)
        if not grad:
            return f
        dl_dk = self.ly - np.log(th) - digamma(k)
        dl_dth = self.y / th**2 - k / th
        dl_dmu = dl_dk * (2 * mu / V) + dl_dth * (-V / mu**2)
        dl_dV = dl_dk * (-mu**2 / V**2) + dl_dth * (1 / mu)
        g = np.empty(len(phi))
        g[0] = np.sum(dl_dmu * A1 * J1)
        g[1] = np.sum(dl_dV * A2 * J2)
        for j in range(d1.shape[1]):
            g[2 + j] = np.sum(dl_dmu * A1 * d1[:, j] + dl_dV * A2 * d2[:, j])
        return f, -g

    def fit(self, x0):
        return minimize(self.nll, np.asarray(x0, float), jac=True, method="L-BFGS-B",
                        options=dict(maxiter=2000, ftol=1e-13, gtol=1e-8))

    # numerical hessian from the analytic gradient
    def hessian(self, x, h=1e-5):
        n = len(x); H = np.zeros((n, n))
        for i in range(n):
            e = np.eye(n)[i] * h
            H[:, i] = (self.nll(x + e)[1] - self.nll(x - e)[1]) / (2 * h)
        return (H + H.T) / 2
