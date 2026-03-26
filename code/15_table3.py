"""
Table III: SMM Calibration (BGLS 2019)
Following BGLS_calibration_quarterly_final.py

Grid search over (a, b, sf, se, d) to minimize distance between
simulated and empirical moments:
- EPS autocorrelations at lags 1-4: [0.82, 0.75, 0.70, 0.65]
- CG coefficients: CG_3yr = -0.276, CG_1yr = -0.125

Inputs: None (pure simulation)
Outputs: replication/output/Table3.txt
"""
import numpy as np
import pandas as pd

OUT_DIR = 'replication/output'


def simulate_all(a, b, sf, se, N):
    """Simulate DGP and compute Kalman-filtered RE. Returns x, RE, K."""
    # Kalman gain
    B = sf**2 + (1 - a**2) * se**2
    su = ((-B + np.sqrt(B**2 + 4*(a*sf*se)**2)) / (2*a**2))**0.5
    K = (a**2 * su**2 + sf**2) / (a**2 * su**2 + sf**2 + se**2)

    eta = np.random.normal(0, sf, N)
    epsilon = np.random.normal(0, se, N)
    f = np.zeros(N)
    x = np.zeros(N)
    RE = np.zeros(N)
    for i in range(1, N):
        f[i] = a * f[i-1] + eta[i]
        x[i] = b * x[i-1] + f[i] + epsilon[i]
        RE[i] = a * RE[i-1] + K * (x[i] - b * x[i-1] - a * RE[i-1])
    return x, RE, K


def compute_DE_LTG_CG(a, b, K, d, x, RE, s, h):
    """Compute DE, LTG, and CG coefficients for given d and sluggishness s.

    Correct formula: DE_t = (1+theta)*RE_t - theta*a^s*RE_{t-s}
    (representativeness: distort current posterior relative to s-period-ago posterior)
    """
    N = len(x)
    DE = np.zeros(N)
    a_s = a ** s
    for i in range(s, N):
        DE[i] = (1 + d) * RE[i] - d * a_s * RE[i - s]

    LTG = np.zeros(N)
    bh = b ** h
    ah = a ** h
    ba_ratio = (1.0 - (b/a)**h) / (1.0 - b/a)
    for i in range(1, N):
        LTG[i] = -(1.0 - bh) * x[i] + ah * DE[i] * ba_ratio

    LTGyr = LTG[0::4]
    xyr = x[0::4]
    Nyr = len(xyr)

    FE = xyr[8:Nyr] - xyr[4:Nyr-4] - LTGyr[4:Nyr-4]
    FR1 = LTGyr[4:Nyr-4] - LTGyr[3:Nyr-5]
    FR3 = LTGyr[4:Nyr-4] - LTGyr[1:Nyr-7]

    ml = min(len(FE), len(FR1), len(FR3))
    if ml < 10:
        return np.nan, np.nan

    C1 = np.cov(FE[:ml], FR1[:ml])
    C3 = np.cov(FE[:ml], FR3[:ml])
    CG1 = C1[0, 1] / C1[1, 1] if C1[1, 1] != 0 else np.nan
    CG3 = C3[0, 1] / C3[1, 1] if C3[1, 1] != 0 else np.nan
    return CG1, CG3


# Target moments
cg_3yr = -0.276
cg_1yr = -0.125


def distance_corr(rho1, rho2, rho3, rho4, CG3, CG1):
    return np.sqrt(
        (rho1 - 0.82)**2 + (rho2 - 0.75)**2
        + (rho3 - 0.70)**2 + (rho4 - 0.65)**2
        + (CG3 - cg_3yr)**2 + (CG1 - cg_1yr)**2
    )


# Grid: paper finds a=0.97, b=0.56, sf=0.138, se=0.083, theta=0.9, s=11
# Corrected DE formula: DE = (1+theta)*RE - theta*a^s*RE_{t-s}
r_a = np.array([0.95, 0.96, 0.97, 0.98])
r_b = np.arange(0.46, 0.66, 0.02)          # 10 values
r_sf = np.arange(0.08, 0.20, 0.02)         # 6 values
r_se = np.arange(0.04, 0.14, 0.02)         # 5 values
r_d = np.arange(0.70, 1.21, 0.10)          # 6 values: theta
r_s = [4, 8, 11, 12, 16]                   # 5 values: sluggishness

H = 4
h = 4 * H

# Effective combos after b < a filter and reuse across d/s:
n_sim_combos = sum(1 for a in r_a for b in r_b if b < a) * len(r_sf) * len(r_se)
n_ds = len(r_d) * len(r_s)
print(f"Grid: {n_sim_combos} unique simulations x {n_ds} (d,s) combos")

np.random.seed(42)
N_sim = 50000  # Balance stability vs speed

results = []
sim_count = 0

for a in r_a:
    for b_param in r_b:
        if b_param >= a:
            continue
        for sf in r_sf:
            for se in r_se:
                sim_count += 1
                if sim_count % 50 == 0:
                    print(f"  Simulation {sim_count}/{n_sim_combos} ...")

                # Simulate once per (a, b, sf, se)
                x, RE, K = simulate_all(a, b_param, sf, se, N_sim)

                # Autocorrelations don't depend on d or s
                xyr = x[0::4]
                Nyr = len(xyr)
                rhos = []
                for lag in range(1, 5):
                    C = np.cov(xyr[5:Nyr], xyr[5-lag:Nyr-lag])
                    rhos.append(C[0, 1] / C[1, 1] if C[1, 1] != 0 else np.nan)

                if np.any(np.isnan(rhos)):
                    continue

                # Loop over d and s
                for d in r_d:
                    for s_val in r_s:
                        CG1, CG3 = compute_DE_LTG_CG(a, b_param, K, d, x, RE,
                                                       s_val, h)
                        if np.isnan(CG1) or np.isnan(CG3):
                            continue

                        dist = distance_corr(rhos[0], rhos[1], rhos[2], rhos[3],
                                             CG3, CG1)
                        results.append({
                            'a': a, 'b': b_param, 'sf': sf, 'se': se,
                            'd': d, 's': s_val, 'K': K,
                            'rho1': rhos[0], 'rho2': rhos[1],
                            'rho3': rhos[2], 'rho4': rhos[3],
                        'CG3': CG3, 'CG1': CG1, 'distance': dist
                    })

print(f"\nDone: {sim_count} simulations, {len(results)} valid parameter combos.")

# Best fit
df = pd.DataFrame(results).sort_values('distance')
best = df.iloc[0]

print(f"\nBest: a={best['a']:.2f}, b={best['b']:.2f}, "
      f"sf={best['sf']:.3f}, se={best['se']:.3f}, theta={best['d']:.1f}, s={best['s']:.0f}")
print(f"Distance: {best['distance']:.4f}")

# Output
lines = [
    "=" * 60,
    "Table III: SMM Parameter Estimates",
    "BGLS (2019) Replication",
    "=" * 60,
    "",
    "Panel A: Estimated Parameters",
    "-" * 50,
    f"  {'Parameter':<30} {'Estimate':>8}  {'Paper':>8}",
    f"  {'-'*30} {'-'*8}  {'-'*8}",
    f"  {'a (persistence of forcing)':<30} {best['a']:>8.3f}  {'0.970':>8}",
    f"  {'b (mean reversion)':<30} {best['b']:>8.3f}  {'0.560':>8}",
    f"  {'sigma_f (forcing shock)':<30} {best['sf']:>8.3f}  {'0.138':>8}",
    f"  {'sigma_e (earnings shock)':<30} {best['se']:>8.3f}  {'0.083':>8}",
    f"  {'theta (diagnostic param)':<30} {best['d']:>8.3f}  {'0.900':>8}",
    f"  {'s (sluggishness)':<30} {best['s']:>8.0f}  {'11':>8}",
    f"  {'Kalman gain K':<30} {best['K']:>8.4f}",
    "",
    "Panel B: Moment Comparison",
    "-" * 50,
    f"  {'Moment':<25} {'Data':>8}  {'Model':>8}  {'Diff':>8}",
    f"  {'-'*25} {'-'*8}  {'-'*8}  {'-'*8}",
    f"  {'rho(1)':<25} {'0.820':>8}  {best['rho1']:>8.3f}  {best['rho1']-0.82:>+8.3f}",
    f"  {'rho(2)':<25} {'0.750':>8}  {best['rho2']:>8.3f}  {best['rho2']-0.75:>+8.3f}",
    f"  {'rho(3)':<25} {'0.700':>8}  {best['rho3']:>8.3f}  {best['rho3']-0.70:>+8.3f}",
    f"  {'rho(4)':<25} {'0.650':>8}  {best['rho4']:>8.3f}  {best['rho4']-0.65:>+8.3f}",
    f"  {'CG (1yr revision)':<25} {'-0.125':>8}  {best['CG1']:>8.3f}  {best['CG1']-cg_1yr:>+8.3f}",
    f"  {'CG (3yr revision)':<25} {'-0.276':>8}  {best['CG3']:>8.3f}  {best['CG3']-cg_3yr:>+8.3f}",
    "",
    f"  Loss (Euclidean distance): {best['distance']:.6f}",
    "",
    "Panel C: Top 5 Parameter Combinations",
    "-" * 70,
]

header = (f"  {'a':>5} {'b':>5} {'sf':>6} {'se':>6} {'d':>5} {'s':>3}"
          f" | {'rho1':>6} {'rho2':>6} {'rho3':>6} {'rho4':>6}"
          f" {'CG1':>7} {'CG3':>7} | {'dist':>8}")
lines.append(header)
lines.append("  " + "-" * 66)
for _, row in df.head(5).iterrows():
    lines.append(
        f"  {row['a']:5.2f} {row['b']:5.2f} {row['sf']:6.3f} {row['se']:6.3f} {row['d']:5.2f} {row['s']:3.0f}"
        f" | {row['rho1']:6.3f} {row['rho2']:6.3f} {row['rho3']:6.3f} {row['rho4']:6.3f}"
        f" {row['CG1']:7.4f} {row['CG3']:7.4f} | {row['distance']:8.5f}"
    )

lines.extend([
    "",
    f"Grid: {n_sim_combos} simulations x {n_ds} (d,s) combos,",
    f"  N={N_sim} quarterly obs, H={H}yr.",
    "DE formula: (1+theta)*RE_t - theta*a^s*RE_{t-s}.",
    "Kalman gain computed analytically.",
    "=" * 60,
])

table_text = "\n".join(lines)
print("\n" + table_text)

with open(f'{OUT_DIR}/Table3.txt', 'w') as fout:
    fout.write(table_text + "\n")
print(f"\nSaved Table3.txt")
