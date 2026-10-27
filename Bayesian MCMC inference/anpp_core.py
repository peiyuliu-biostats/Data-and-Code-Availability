# simulator and gamma surrogate likelihood for the anpp model

import numpy as np
from scipy.special import gammaln, digamma
from scipy.optimize import minimize

# angular frequency (1/day) and integration grid (days)
OMEGA = 2 * np.pi / 365.0
S_GRID = np.arange(0.0, 1096.0, 1.0)

# lognormal (mu, sigma) giving boost mean mean_c and variance var_c
def lognormal_params(mean_c=0.1, var_c=0.15):
    s2 = np.log(var_c / mean_c**2 + 1.0)
    return np.log(mean_c) - 0.5 * s2, np.sqrt(s2)

# true parameter values used in the simulations
TRUE = dict(mean_c=0.1, var_c=0.15, beta0=np.log(1 / 15), beta_sin=0.5, beta_cos=0.2,
            t_half=21.0, B0=0.3, sigma_eps=0.15)

# identifiable parameters (log A1, log A2, beta_sin, beta_cos) implied by p
def true_phi(p):
    cbar = p["mean_c"]
    c2 = p["var_c"] + cbar**2
    return np.array([np.log(cbar) + p["beta0"], np.log(c2) + p["beta0"], p["beta_sin"], p["beta_cos"]])

# amplitude R and peak day (day of year)
def R_phi(bs, bc):
    R = np.hypot(bs, bc)
    ph = (365 / (2 * np.pi) * np.arctan2(bs, bc)) % 365
    return R, ph

# signed circular difference a - b with period P
def circ_diff(a, b, P=365.0):
    d = (a - b) % P
    return np.where(d > P / 2, d - P, d)

# antibody levels at t_obs; bites by thinning a poisson process over a burn-in window
def simulate_round(rng, t_obs, p, c_mode="person", C_person=None, burn=1095.0,
                   chunk=20000):
    mu_c, s_c = lognormal_params(p["mean_c"], p["var_c"])
    gam = np.log(2) / p["t_half"]
    lam = lambda t: np.exp(p["beta0"] + p["beta_sin"] * np.sin(OMEGA * t) + p["beta_cos"] * np.cos(OMEGA * t))
    lam_max = np.exp(p["beta0"] + np.hypot(p["beta_sin"], p["beta_cos"])) * 1.5
    n = len(t_obs)
    out = np.empty(n)
    # c_mode 'person': one boost per person; 'bite': one boost per bite; otherwise the mean boost
    if c_mode == "person" and C_person is None:
        C_person = rng.lognormal(mu_c, s_c, n)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        tt = t_obs[a:b]
        m = b - a
        ncand = rng.poisson(lam_max * burn, m)
        idx = np.repeat(np.arange(m), ncand)
        tau = tt[idx] - burn * rng.random(idx.size)
        keep = rng.random(idx.size) < lam(tau) / lam_max
        idx, tau = idx[keep], tau[keep]
        dec = np.exp(-gam * (tt[idx] - tau))
        if c_mode == "bite":
            dec = dec * rng.lognormal(mu_c, s_c, dec.size)
            s = np.bincount(idx, weights=dec, minlength=m)
        else:
            s = np.bincount(idx, weights=dec, minlength=m)
            s = s * (C_person[a:b] if c_mode == "person" else p["mean_c"])
        A = p["B0"] + s
        y = A + rng.normal(0, p["sigma_eps"], m)
        out[a:b] = np.maximum(0.001, y)
    return out

# design with M rounds of N people; cohort=True keeps the same boost sizes in every round
def simulate_design(rng, p, M=24, N=300, jitter=2.0, c_mode="person", cohort=True,
                    spacing=30.0, t0=30.0):
    mu_c, s_c = lognormal_params(p["mean_c"], p["var_c"])
    Cp = rng.lognormal(mu_c, s_c, N)
    T, Y, R = [], [], []
    for j in range(M):
        t = t0 + spacing * j + rng.uniform(-jitter, jitter, N)
        Cuse = Cp if cohort else rng.lognormal(mu_c, s_c, N)
        y = simulate_round(rng, t, p, c_mode=c_mode, C_person=Cuse)
        T.append(t); Y.append(y); R.append(np.full(N, j + 1))
    return np.concatenate(T), np.concatenate(Y), np.concatenate(R)

# seasonal integrals for the mean (J1) and variance (J2), optionally with gradients in bs, bc
def J_integrals(t, bs, bc, gam, grid=S_GRID, ds=1.0, grad=False):
    t = np.atleast_1d(t)
    arg = OMEGA * (t[:, None] - grid[None, :])
    sn, cs = np.sin(arg), np.cos(arg)
    u = np.exp(bs * sn + bc * cs)
    w1 = np.exp(-gam * grid) * ds
    w2 = np.exp(-2 * gam * grid) * ds
    J1, J2 = u @ w1, u @ w2
    if not grad:
        return J1, J2
    dJ1 = np.stack([(u * sn) @ w1, (u * cs) @ w1], 1)
    dJ2 = np.stack([(u * sn) @ w2, (u * cs) @ w2], 1)
    return J1, J2, dJ1, dJ2

# mean and variance of antibody levels at times t
def moments(phi, t, gam, B0, s2eps, **kw):
    la1, la2, bs, bc = phi
    J1, J2 = J_integrals(t, bs, bc, gam, **kw)
    return B0 + np.exp(la1) * J1, np.exp(la2) * J2 + s2eps

# gamma surrogate likelihood on (log A1, log A2, beta_sin, beta_cos) with analytic gradient
class GammaFit:

    def __init__(self, t, y, gam, B0, s2eps, tgrid_step=0.25, weights=None, grid=S_GRID):
        self.t, self.y = np.asarray(t, float), np.maximum(np.asarray(y, float), 1e-8)
        self.ly = np.log(self.y)
        self.gam, self.B0, self.s2 = gam, B0, s2eps
        self.w = np.ones_like(self.y) if weights is None else weights
        self.grid = grid
        # integrals on the unique times, or on a fine grid with interpolation when there are many
        tun = np.unique(self.t)
        if len(tun) < 400:
            self.tu = tun
            self.map = np.searchsorted(self.tu, self.t)
            self.interp = False
        else:
            self.tu = np.arange(self.t.min() - tgrid_step, self.t.max() + 2 * tgrid_step, tgrid_step)
            self.interp = True

    def _J(self, bs, bc):
        J1, J2, d1, d2 = J_integrals(self.tu, bs, bc, self.gam, grid=self.grid, grad=True)
        if not self.interp:
            m = self.map
            return J1[m], J2[m], d1[m], d2[m]
        f = lambda v: np.interp(self.t, self.tu, v)
        return (f(J1), f(J2), np.stack([f(d1[:, 0]), f(d1[:, 1])], 1),
                np.stack([f(d2[:, 0]), f(d2[:, 1])], 1))

    # negative log-likelihood and gradient
    def nll(self, phi, grad=True):
        la1, la2, bs, bc = phi
        J1, J2, d1, d2 = self._J(bs, bc)
        A1, A2 = np.exp(la1), np.exp(la2)
        mu = self.B0 + A1 * J1
        V = A2 * J2 + self.s2
        if np.any(mu <= 0) or np.any(V <= 0):
            return (1e12, np.zeros(4)) if grad else 1e12
        k = mu * mu / V
        th = V / mu
        ll = (k - 1) * self.ly - self.y / th - k * np.log(th) - gammaln(k)
        f = -np.sum(self.w * ll)
        if not grad:
            return f
        dl_dk = self.ly - np.log(th) - digamma(k)
        dl_dth = self.y / th**2 - k / th
        dl_dmu = dl_dk * (2 * mu / V) + dl_dth * (-V / mu**2)
        dl_dV = dl_dk * (-mu**2 / V**2) + dl_dth * (1 / mu)
        g = np.empty(4)
        g[0] = np.sum(self.w * dl_dmu * A1 * J1)
        g[1] = np.sum(self.w * dl_dV * A2 * J2)
        g[2] = np.sum(self.w * (dl_dmu * A1 * d1[:, 0] + dl_dV * A2 * d2[:, 0]))
        g[3] = np.sum(self.w * (dl_dmu * A1 * d1[:, 1] + dl_dV * A2 * d2[:, 1]))
        return f, -g

    def fit(self, x0, **opt):
        r = minimize(self.nll, np.asarray(x0, float), jac=True, method="L-BFGS-B",
                     options=dict(maxiter=2000, ftol=1e-13, gtol=1e-8, **opt))
        return r

# parameter value minimising the expected negative log-likelihood (pseudo-true value)
def pseudo_true(t_pts, Ey, Elogy, n_w, gam, B0, s2eps, x0):
    t_pts = np.asarray(t_pts, float)

    def f(phi):
        la1, la2, bs, bc = phi
        mu, V = moments(phi, t_pts, gam, B0, s2eps)
        if np.any(mu <= 0) or np.any(V <= 0):
            return 1e12
        k = mu * mu / V
        th = V / mu
        el = (k - 1) * Elogy - Ey / th - k * np.log(th) - gammaln(k)
        return -np.sum(n_w * el)

    r = minimize(f, x0, method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000, maxfev=40000))
    r = minimize(f, r.x, method="BFGS", options=dict(gtol=1e-9))
    return r

# moments by numerical integration; var_term adds the variance from one boost per person
def continuous_moments(t, p, var_term=True, h=0.02, window=1095.0):
    gam = np.log(2) / p["t_half"]
    s = np.arange(0, window + h / 2, h)
    wts = np.full(s.size, h); wts[0] = wts[-1] = h / 2
    t = np.atleast_1d(t)
    lam = np.exp(p["beta0"] + p["beta_sin"] * np.sin(OMEGA * (t[:, None] - s)) + p["beta_cos"] * np.cos(OMEGA * (t[:, None] - s)))
    m1 = lam @ (np.exp(-gam * s) * wts)
    m2 = lam @ (np.exp(-2 * gam * s) * wts)
    cbar, vc = p["mean_c"], p["var_c"]
    c2 = vc + cbar**2
    mu = p["B0"] + cbar * m1
    V = c2 * m2 + (vc * m1**2 if var_term else 0) + p["sigma_eps"] ** 2
    return mu, V, m1, m2
