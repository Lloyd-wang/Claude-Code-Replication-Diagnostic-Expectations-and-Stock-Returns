"""
Replication of Figure 6 from Bordalo, Gennaioli, La Porta, Shleifer (2019).
"Diagnostic Expectations and Stock Returns" - Journal of Finance.

6-panel simulation figure using the paper's published calibration (Table 3).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── 0. Seed for reproducibility ──────────────────────────────────────────────
np.random.seed(12345)

# ── 1. Fundamental parameters (Table 3, Section VI.B) ────────────────────────

a = 0.97          # persistence of forcing term
b = 0.56          # persistence of earnings
sf = 0.138        # std dev of forcing term shocks
se = 0.083        # std dev of earnings shocks
d = 0.9           # theta (diagnostic parameter)
R = (1.0 + 0.045) ** 0.25   # gross required return per quarter
tscale = 11       # time scale for DE (12 quarters / 3 years)

# Steady-state signal-to-noise ratio
B = sf ** 2 + (1 - a ** 2) * (se ** 2)
su = ((-B + np.sqrt(B ** 2 + 4 * (a * sf * se) ** 2)) / (2 * a ** 2)) ** 0.5
K = (a ** 2 * su ** 2 + sf ** 2) / (a ** 2 * su ** 2 + sf ** 2 + se ** 2)

# Simulation parameters
F = 4000          # number of firms (paper: "4,000 firms")
N = 275           # number of quarters


# ── 2. Functions from BGLS_functions_final.py ─────────────────────────────────

def ts_eps(a, b, sf, se, N):
    """Time series of ln(eps)."""
    f = np.zeros(N)
    x = np.zeros(N)
    eta = np.random.normal(0, sf, N)
    epsilon = np.random.normal(0, se, N)
    f0 = 0.0
    x[0] = 10
    for i in range(1, N):
        f[i] = (1 - a) * f0 + a * f[i - 1] + eta[i]
        x[i] = b * x[i - 1] + f[i] + epsilon[i]
    return x


def ts_RE(a, b, K, x):
    """Rational Expectations of CURRENT forcing term."""
    N = len(x)
    RE = np.zeros(N)
    for i in range(1, N):
        RE[i] = a * RE[i - 1] + K * (x[i] - b * x[i - 1] - a * RE[i - 1])
    return RE


def ts_DE(a, b, K, d, x, RE, tscale):
    """Diagnostic Expectations of CURRENT forcing term."""
    N = len(x)
    DE = np.zeros(N)
    for i in range(tscale, N):
        DE[i] = RE[i] + d * K * (x[i] - b * x[i - 1] - a ** tscale * RE[i - tscale])
    return DE


def ts_LTG(a, b, x, E, h):
    """LTG expectations under E = RE or DE; h is horizon."""
    N = len(x)
    LTG = np.zeros(N)
    for i in range(1, N):
        LTG[i] = -(1.0 - b ** h) * x[i] + a ** h * E[i] * (1.0 - (b / a) ** h) / (1.0 - (b / a))
    return LTG


def ts_avg(Vec, LTG, n):
    """Short-run dynamics around portfolio formation (quarterly)."""
    N = LTG.shape[1]
    avg = np.zeros([25, N - 13])
    for t in range(12, N - 13):
        fhltg = np.percentile(LTG[:, t], 90)
        flltg = np.percentile(LTG[:, t], 10)
        if n == 1:
            for j in range(-12, 13):
                avg[j + 12, t] = Vec[LTG[:, t] > fhltg, t + j].mean()
        if n == 0:
            for j in range(-12, 13):
                avg[j + 12, t] = Vec[LTG[:, t] < flltg, t + j].mean()
    avg = np.mean(avg[:, 12:N - 13], axis=1)
    return avg


def ts_avg_yr(Vec, LTG, n):
    """Short-run dynamics around portfolio formation (annual)."""
    N = LTG.shape[1]
    avg = np.zeros([8, N - 8])
    for t in range(3, N - 5):
        fhltg = np.percentile(LTG[:, t], 90)
        flltg = np.percentile(LTG[:, t], 10)
        if n == 1:
            for j in range(-3, 5):
                avg[j + 3, t - 3] = Vec[LTG[:, t] > fhltg, t + j].mean()
        if n == 0:
            for j in range(-3, 5):
                avg[j + 3, t - 3] = Vec[LTG[:, t] < flltg, t + j].mean()
    avg = np.mean(avg[:, 3:N - 8], axis=1)
    return avg


def ts_price(a, b, EPS, E, R, V):
    """Time series of price for a firm (vectorized inner loop)."""
    N = E.shape[0]
    S = V.shape[0]
    Pr = 190 * np.ones(N)
    s_range = np.arange(1, S)
    bs = b ** s_range
    Q = a ** s_range * (1 - (b / a) ** s_range) / (1 - (b / a))
    Rs = R ** s_range
    half_V = 0.5 * V[1:S]
    for t in range(1, N):
        ExpEPS = np.exp(bs * EPS[t] + Q * E[t] + half_V) / Rs
        Pr[t] = np.sum(ExpEPS)
    return Pr


def decile_returns(LTG, Ret):
    """Year-ahead returns by LTG decile."""
    N = LTG.shape[1]
    dec_returns = np.zeros([N - 1, 10])
    for t in range(N - 1):
        for i in range(10):
            fltg_low = np.percentile(LTG[:, t], i * 10)
            fltg_high = np.percentile(LTG[:, t], (i + 1) * 10)
            ltg = (LTG[:, t] > fltg_low) & (LTG[:, t] < fltg_high)
            dec_returns[t, i] = Ret[ltg, t + 1].mean()
    return np.mean(dec_returns, axis=0)


def FE(LTG, EPS, n, h):
    """Forecast errors of LTG. n=0: LLTG, n=1: HLTG. h is horizon."""
    N = LTG.shape[1]
    fe = np.zeros(N - h)
    for t in range(1, N - h):
        fhltg = np.percentile(LTG[:, t], 90)
        flltg = np.percentile(LTG[:, t], 10)
        if n == 0:
            lltg = LTG[:, t] < flltg
            fe[t] = np.mean(EPS[lltg, t + h] - EPS[lltg, t] - LTG[lltg, t])
        if n == 1:
            hltg = LTG[:, t] > fhltg
            fe[t] = np.mean(EPS[hltg, t + h] - EPS[hltg, t] - LTG[hltg, t])
    fe = np.mean(fe)
    return fe


def distrib(Vec, LTG, n):
    """Distributions of Vec conditional on LTG percentiles."""
    N = LTG.shape[1]
    dist = []
    for i in range(1, N - 16):
        f_hltg = np.percentile(LTG[:, i], 90)
        f_lltg = np.percentile(LTG[:, i], 10)
        if n == 0:   # HLTG
            dist.extend(0.25 * (Vec[LTG[:, i] > f_hltg, i + 16] - Vec[LTG[:, i] > f_hltg, i]))
        if n == 1:   # not HLTG
            dist.extend(0.25 * (Vec[LTG[:, i] < f_hltg, i + 16] - Vec[LTG[:, i] < f_hltg, i]))
        if n == 2:   # LLTG
            dist.extend(0.25 * (Vec[LTG[:, i] < f_lltg, i + 20] - Vec[LTG[:, i] < f_lltg, i]))
        if n == 3:   # not LLTG
            dist.extend(0.25 * (Vec[LTG[:, i] > f_lltg, i + 20] - Vec[LTG[:, i] > f_lltg, i]))
        if n == 4:
            dist.extend(0.25 * (Vec[LTG[:, i] > f_hltg, i]))
    dist = np.exp(dist)
    hist, bin_edges = np.histogram(dist, bins=np.arange(0, 2.1, 0.1))
    return hist, bin_edges


# ── 3. Generate fundamentals and expectations ─────────────────────────────────

print("Simulating fundamentals and expectations for %d firms..." % F)

lnEPS = np.zeros([F, N])
RE_mat = np.zeros([F, N])
DE_mat = np.zeros([F, N])
LTG_mat = np.zeros([F, N])

for j in range(F):
    lnEPS[j, :] = ts_eps(a, b, sf, se, N)
    RE_mat[j, :] = ts_RE(a, b, K, lnEPS[j, :])
    DE_mat[j, :] = ts_DE(a, b, K, d, lnEPS[j, :], RE_mat[j, :], tscale)
    LTG_mat[j, :] = ts_LTG(a, b, lnEPS[j, :], DE_mat[j, :], 16)  # 4-year horizon

# Horizon to discount eps
S = 200
V = np.zeros(S)
for s in range(S):
    V[s] = np.var(lnEPS[:, s])

# Restrict sample to last 200 periods for stability
N = 200
lnEPS = lnEPS[:, -N:]
RE_mat = RE_mat[:, -N:]
DE_mat = DE_mat[:, -N:]
LTG_mat = LTG_mat[:, -N:]

# Annual versions
Nyr = N // 4
LTGyr = np.zeros([F, Nyr])
lnEPSyr = np.zeros([F, Nyr])
for j in range(F):
    LTGyr[j, :] = LTG_mat[j, 0::4]
    lnEPSyr[j, :] = lnEPS[j, 0::4]

EPS_hltg_yr = ts_avg_yr(np.exp(lnEPSyr), LTGyr, 1)
EPS_lltg_yr = ts_avg_yr(np.exp(lnEPSyr), LTGyr, 0)
EPS_hltg_yr = EPS_hltg_yr / EPS_hltg_yr[0]
EPS_lltg_yr = EPS_lltg_yr / EPS_lltg_yr[0]


# ── 4. Compute prices and returns ────────────────────────────────────────────

print("Computing prices and returns...")

PDE = np.zeros([F, N])
RetDE = np.zeros([F, N])
PRE = np.zeros([F, N])
RetRE = np.zeros([F, N])
RetREyr = np.zeros([F, Nyr - 1])
RetDEyr = np.zeros([F, Nyr - 1])

for j in range(F):
    PDE[j, :] = ts_price(a, b, lnEPS[j, :], DE_mat[j, :], R, V)
    PRE[j, :] = ts_price(a, b, lnEPS[j, :], RE_mat[j, :], R, V)
    for t in range(N):
        RetRE[j, t] = (np.exp(lnEPS[j, t]) + PRE[j, t]) / PRE[j, t - 1]
        RetDE[j, t] = (np.exp(lnEPS[j, t]) + PDE[j, t]) / PDE[j, t - 1]
        if (t % 4 == 0) and (t > 0):
            RetREyr[j, t // 4 - 1] = np.prod(RetRE[j, t - 3:t + 1])
            RetDEyr[j, t // 4 - 1] = np.prod(RetDE[j, t - 3:t + 1])

RetRE = RetRE[:, 2:N]
RetDE = RetDE[:, 2:N]

N = N - 2
LTGyr = LTGyr[:, 1:N // 4]

RetRE_hltg_yr = ts_avg_yr(RetREyr, LTGyr, 1)
RetRE_lltg_yr = ts_avg_yr(RetREyr, LTGyr, 0)
RetDE_hltg_yr = ts_avg_yr(RetDEyr, LTGyr, 1)
RetDE_lltg_yr = ts_avg_yr(RetDEyr, LTGyr, 0)


# ── 5. Figures ────────────────────────────────────────────────────────────────

print("Generating Figure 6...")

fig = plt.figure(figsize=(16, 10))

# Panel 1: Annual returns vs LTG decile
LTG_decile_returns = decile_returns(LTGyr, RetDEyr)
ax1 = fig.add_subplot(231)
ax1.plot(LTG_decile_returns, "b")
ax1.set_xticks([0, 3, 6, 9])
ax1.set_xticklabels(["LLTG", "4", "7", "HLTG"])
ax1.set_yticks([1.05, 1.1, 1.15])
ax1.set_yticklabels(["5%", "10%", "15%"])
ax1.set_title("1. Annual returns vs LTG")

# Panel 2: EPS short-run dynamics
EPS_hltg = ts_avg(np.exp(lnEPS), LTG_mat[:, -N:], 1)
EPS_lltg = ts_avg(np.exp(lnEPS), LTG_mat[:, -N:], 0)
EPS_hltg = EPS_hltg / EPS_hltg[0]
EPS_lltg = EPS_lltg / EPS_lltg[0]

ax2 = fig.add_subplot(232)
ax2.plot(EPS_hltg, "r", label="HLTG")
ax2.plot(EPS_lltg, "b", label="LLTG")
ax2.legend(loc="best")
ax2.set_xticks([4, 12, 20])
ax2.set_xticklabels(["-2", "0", "2"])
ax2.set_yticks([1, 3, 5])
ax2.set_yticklabels(["1", "3", "5"])
ax2.set_title("2. Evolution of EPS")

# Panel 3: LTG short-run dynamics
LTG_use = LTG_mat[:, -N:]
LTG_hltg = ts_avg(LTG_use, LTG_use, 1)
LTG_lltg = ts_avg(LTG_use, LTG_use, 0)

ax3 = fig.add_subplot(233)
ax3.plot(np.exp(0.25 * LTG_hltg) - 1, "r", label="HLTG")
ax3.plot(np.exp(0.25 * LTG_lltg) - 1, "b", label="LLTG")
ax3.set_xticks([4, 12, 20])
ax3.set_xticklabels(["-2", "0", "2"])
ax3.set_yticks([-0.2, 0.0, 0.2])
ax3.set_yticklabels(["-20%", "0", "20%"])
ax3.set_title("3. Evolution of LTG")

# Panel 4: Forecast errors
ltg_fe = np.zeros([F, N])
HLTG_FE = np.zeros(6)
LLTG_FE = np.zeros(6)
lnEPS_fe = lnEPS[:, -N:]
LTG_fe = LTG_use

for h in range(1, 6):
    for j in range(F):
        ltg_fe[j, :] = ts_LTG(a, b, lnEPS_fe[j, :], DE_mat[j, -N:], h)
    LLTG_FE[h] = FE(ltg_fe, lnEPS_fe, 0, h)
    HLTG_FE[h] = FE(ltg_fe, lnEPS_fe, 1, h)

LLTG_FE = np.exp(0.25 * LLTG_FE) - 1
HLTG_FE = np.exp(0.25 * HLTG_FE) - 1

ax4 = fig.add_subplot(234)
ax4.plot(HLTG_FE[1:6], "r", label="HLTG")
ax4.plot(LLTG_FE[1:6], "b", label="LLTG")
ax4.legend(loc="best")
ax4.set_xticks([0, 2, 4])
ax4.set_xticklabels(["1", "3", "5"])
ax4.set_yticks([-0.2, 0, 0.2])
ax4.set_yticklabels(["-0.2", "0", "0.2"])
ax4.set_title("4. Forecast errors")

# Panel 5: Return dynamics
ax5 = fig.add_subplot(235)
ax5.plot(RetDE_hltg_yr, "r", label="HLTG")
ax5.plot(RetDE_lltg_yr, "b", label="LLTG")
ax5.legend(loc="best")
ax5.set_yticks([0.8, 1.2, 1.6])
ax5.set_yticklabels(["-20%", "20%", "60%"])
ax5.set_xticks([1, 3, 5])
ax5.set_xticklabels(["-2", "0", "2"])
ax5.set_title("5. Evolution of returns")

# Panel 6: Realized vs expected growth distributions
HLTG_growth, bins1 = distrib(lnEPS_fe, LTG_fe, 0)
nonHLTG_growth, bins2 = distrib(lnEPS_fe, LTG_fe, 1)
HLTG_forecast, bins3 = distrib(LTG_fe, LTG_fe, 4)
HLTG_growth = 1.0 * HLTG_growth / np.sum(HLTG_growth)
nonHLTG_growth = 1.0 * nonHLTG_growth / np.sum(nonHLTG_growth)
HLTG_forecast = 1.0 * HLTG_forecast / np.sum(HLTG_forecast)

ax6 = fig.add_subplot(236)
ax6.plot(bins1[0:20], HLTG_growth, "b", label="HLTG")
ax6.plot(bins2[0:20], nonHLTG_growth, "g", label="nonHLTG")
line2 = np.sum(bins1[0:20] * HLTG_growth)
line3 = np.sum(bins3[0:20] * HLTG_forecast)
lim1 = 1.2 * np.max(HLTG_growth)
ax6.plot([line2, line2], [0.0, lim1], "b--")
ax6.plot([line3, line3], [0.0, lim1], "r", label="LTG")
ax6.set_yticks([])
ax6.set_title("6. Realized vs expected growth")
ax6.legend(loc="best")

plt.subplots_adjust(wspace=0.3, hspace=0.4)

# Save
outpath = "/workspaces/Dividend-growth/BGLS2019_rep/replication/output/Figure6.png"
fig.savefig(outpath, dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: %s" % outpath)
